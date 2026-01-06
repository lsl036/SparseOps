/**
 * @file spgemm_bin.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Implementation of BIN class for SpGEMM load balancing
 * @version 0.1
 * @date 2026
 */

#include "../include/spgemm_bin.h"
#include <cmath>
#include <algorithm>

// Helper function: prefix sum (scan)
template <typename IndexType>
void scan(const IndexType *input, IndexType *output, IndexType n) {
    output[0] = 0;
    for (IndexType i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

template <typename IndexType, typename ValueType>
SpGEMM_BIN<IndexType, ValueType>::SpGEMM_BIN(IndexType nrows, IndexType min_ht_sz)
    : num_rows(nrows), min_ht_size(min_ht_sz), max_bin_id(0), allocated_thread_num(0)
{
    bin_id = new_array<IndexType>(num_rows);
    row_nz = new_array<IndexType>(num_rows);
    rows_offset = nullptr;
    local_hash_table_id = nullptr;
    local_hash_table_val = nullptr;
    hash_table_size = nullptr;
    
    // Initialize arrays
    std::fill_n(bin_id, num_rows, static_cast<IndexType>(0));
    std::fill_n(row_nz, num_rows, static_cast<IndexType>(0));
}

template <typename IndexType, typename ValueType>
SpGEMM_BIN<IndexType, ValueType>::~SpGEMM_BIN()
{
    delete_array(bin_id);
    delete_array(row_nz);
    delete_array(rows_offset);
    
    // Free hash table allocations using allocated_thread_num
    if (local_hash_table_id != nullptr) {
        // Use allocated_thread_num to safely free only allocated memory
        for (int i = 0; i < allocated_thread_num; ++i) {
            if (local_hash_table_id[i] != nullptr) {
                delete_array(local_hash_table_id[i]);
            }
        }
        delete[] local_hash_table_id;
    }
    if (local_hash_table_val != nullptr) {
        // Use allocated_thread_num to safely free only allocated memory
        for (int i = 0; i < allocated_thread_num; ++i) {
            if (local_hash_table_val[i] != nullptr) {
                delete_array(local_hash_table_val[i]);
            }
        }
        delete[] local_hash_table_val;
    }
    if (hash_table_size != nullptr) {
        delete_array(hash_table_size);
    }
}

template <typename IndexType, typename ValueType>
IndexType SpGEMM_BIN<IndexType, ValueType>::get_hash_size(IndexType ncols, IndexType min_ht_sz)
{
    IndexType hash_size = min_ht_sz;
    while (hash_size < ncols) {
        IndexType new_size = hash_size * HASH_SCAL / 100;
        // Ensure hash_size actually increases to avoid infinite loop
        if (new_size <= hash_size) {
            hash_size = hash_size * 2;  // Double the size if scaling doesn't help
        } else {
            hash_size = new_size;
        }
        // Safety check: prevent overflow
        if (hash_size < min_ht_sz) {
            hash_size = ncols * 2;  // Fallback: use 2x ncols
            break;
        }
    }
    return hash_size;
}

template <typename IndexType, typename ValueType>
void SpGEMM_BIN<IndexType, ValueType>::set_max_bin(
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, IndexType c_rows, IndexType c_cols)
{
    // Estimate work for each row: sum of B's row lengths for each column in A's row
    #pragma omp parallel for
    for (IndexType i = 0; i < num_rows; i++) {
        IndexType row_nnz = 0;
        for (IndexType j = arpt[i]; j < arpt[i + 1]; j++) {
            IndexType col = acol[j];
            if (col < c_cols) {
                row_nnz += brpt[col + 1] - brpt[col];
            }
        }
        row_nz[i] = row_nnz;
    }
    
    // Set rows offset for load balancing
    set_rows_offset(c_rows);
    
    // Set bin ID based on estimated work
    set_bin_id(c_rows, c_cols, min_ht_size);
}

template <typename IndexType, typename ValueType>
void SpGEMM_BIN<IndexType, ValueType>::set_rows_offset(IndexType nrows)
{
    int thread_num = Le_get_thread_num();
    
    // Allocate or reallocate rows_offset based on current thread number
    // This is critical: if thread number changed, we must reallocate
    if (rows_offset == nullptr) {
        rows_offset = new_array<IndexType>(thread_num + 1);
    } else {
        // Check if we need to reallocate (we can't know the old size exactly,
        // but we can check if current thread_num would require more space)
        // For safety, always reallocate if thread_num might have changed
        // We'll use a simple heuristic: if thread_num > some threshold, reallocate
        // Actually, the safest approach is to always reallocate when called,
        // since we don't track the old thread_num
        delete_array(rows_offset);
        rows_offset = new_array<IndexType>(thread_num + 1);
    }
    
    // Prefix sum of row_nz
    IndexType *ps_row_nz = new_array<IndexType>(nrows + 1);
    scan(row_nz, ps_row_nz, nrows + 1);
    
    // Calculate average work per thread
    IndexType total_work = ps_row_nz[nrows];
    IndexType average_work = (total_work + thread_num - 1) / thread_num;
    
    // Set rows_offset for each thread
    rows_offset[0] = 0;
    #pragma omp parallel num_threads(thread_num)
    {
        int tid = Le_get_thread_id();
        // Safety check: ensure tid is within valid range
        if (tid < thread_num) {
            IndexType target_work = average_work * (tid + 1);
            
            // Binary search for the row index
            IndexType *pos = std::lower_bound(ps_row_nz, ps_row_nz + nrows + 1, target_work);
            rows_offset[tid + 1] = static_cast<IndexType>(pos - ps_row_nz);
        }
    }
    rows_offset[thread_num] = nrows;
    
    delete_array(ps_row_nz);
}

template <typename IndexType, typename ValueType>
void SpGEMM_BIN<IndexType, ValueType>::set_bin_id(
    IndexType nrows, IndexType ncols, IndexType min_ht_sz)
{
    // Assign bin ID based on estimated work (logarithmic binning)
    // Match reference implementation: while loop instead of log2
    #pragma omp parallel for
    for (IndexType i = 0; i < num_rows; i++) {
        IndexType nz_per_row = row_nz[i];
        if (nz_per_row > ncols) nz_per_row = ncols;
        
        if (nz_per_row == 0) {
            bin_id[i] = 0;
        } else {
            IndexType j = 0;
            while (nz_per_row > (min_ht_sz << j)) {
                j++;
            }
            bin_id[i] = j + 1;
        }
    }
}

template <typename IndexType, typename ValueType>
void SpGEMM_BIN<IndexType, ValueType>::create_local_hash_table(IndexType max_cols)
{
    int thread_num = Le_get_thread_num();
    
    // Free old allocations if they exist (thread number might have changed)
    // Always free and reallocate to ensure thread safety
    if (local_hash_table_id != nullptr) {
        // Free individual hash tables (safe upper bound: 256 threads)
        for (int i = 0; i < 256; ++i) {
            if (local_hash_table_id[i] != nullptr) {
                delete_array(local_hash_table_id[i]);
            }
        }
        delete[] local_hash_table_id;
        local_hash_table_id = nullptr;
    }
    if (local_hash_table_val != nullptr) {
        for (int i = 0; i < 256; ++i) {
            if (local_hash_table_val[i] != nullptr) {
                delete_array(local_hash_table_val[i]);
            }
        }
        delete[] local_hash_table_val;
        local_hash_table_val = nullptr;
    }
    if (hash_table_size != nullptr) {
        delete_array(hash_table_size);
        hash_table_size = nullptr;
    }
    
    // Allocate new arrays with current thread number
    hash_table_size = new_array<IndexType>(thread_num);
    local_hash_table_id = new IndexType*[thread_num];
    local_hash_table_val = new ValueType*[thread_num];
    
    // Initialize pointers to nullptr
    for (int i = 0; i < thread_num; ++i) {
        local_hash_table_id[i] = nullptr;
        local_hash_table_val[i] = nullptr;
    }
    
    #pragma omp parallel num_threads(thread_num)
    {
        int tid = Le_get_thread_id();
        // Safety check: ensure tid is within valid range
        if (tid < thread_num) {
            IndexType ht_size = 0;
            
            // Get max hash table size needed for this thread's rows
            if (rows_offset != nullptr) {
                // Safety check: ensure rows_offset indices are valid
                if (tid + 1 <= thread_num) {
                    for (IndexType j = rows_offset[tid]; j < rows_offset[tid + 1]; ++j) {
                        if (ht_size < row_nz[j]) ht_size = row_nz[j];
                    }
                }
            }
            
            // Align to power of 2 (2^n)
            if (ht_size > 0) {
                if (ht_size > max_cols) ht_size = max_cols;
                IndexType k = min_ht_size;
                while (k < ht_size) {
                    k <<= 1;
                }
                ht_size = k;
            } else {
                ht_size = min_ht_size;
            }
            
            hash_table_size[tid] = ht_size;
            local_hash_table_id[tid] = new_array<IndexType>(ht_size);
            local_hash_table_val[tid] = new_array<ValueType>(ht_size);
            
            // Initialize hash table
            std::fill_n(local_hash_table_id[tid], ht_size, static_cast<IndexType>(-1));
            std::fill_n(local_hash_table_val[tid], ht_size, static_cast<ValueType>(0));
        }
    }
    
    // Update allocated_thread_num after successful allocation
    allocated_thread_num = thread_num;
}

template <typename IndexType, typename ValueType>
double SpGEMM_BIN<IndexType, ValueType>::calculate_size_in_gb()
{
    double size_gb = 0.0;
    int thread_num = Le_get_thread_num();
    
    if (hash_table_size != nullptr) {
        for (int i = 0; i < thread_num; i++) {
            size_gb += hash_table_size[i] * (sizeof(IndexType) + sizeof(ValueType));
        }
    }
    
    size_gb += num_rows * (sizeof(IndexType) * 2); // bin_id + row_nz
    
    return size_gb / (1024.0 * 1024.0 * 1024.0);
}

// Hash table utility functions
template <typename IndexType>
IndexType hash_find_pos(IndexType key, IndexType hash_size,
                        const IndexType *hash_table_id)
{
    IndexType hash = hash_func(key, hash_size);
    IndexType start_hash = hash;
    
    while (hash_table_id[hash] != -1 && hash_table_id[hash] != key) {
        hash = (hash + 1) & (hash_size - 1);  // hash_size is power of 2, use bitwise AND
        if (hash == start_hash) {
            return -1; // Hash table full
        }
    }
    
    return hash;
}

template <typename IndexType>
IndexType hash_insert_symbolic(IndexType key, IndexType hash_size,
                                IndexType *hash_table_id)
{
    IndexType pos = hash_find_pos(key, hash_size, hash_table_id);
    if (pos == -1) {
        return -1; // Hash table full
    }
    
    if (hash_table_id[pos] == -1) {
        hash_table_id[pos] = key;
        return pos;
    }
    
    return pos; // Already exists
}

template <typename IndexType, typename ValueType>
IndexType hash_insert_numeric(IndexType key, ValueType val, IndexType hash_size,
                               IndexType *hash_table_id, ValueType *hash_table_val)
{
    IndexType pos = hash_find_pos(key, hash_size, hash_table_id);
    if (pos == -1) {
        return -1; // Hash table full
    }
    
    if (hash_table_id[pos] == -1) {
        hash_table_id[pos] = key;
        hash_table_val[pos] = val;
        return pos;
    } else if (hash_table_id[pos] == key) {
        hash_table_val[pos] += val; // Accumulate
        return pos;
    }
    
    return pos;
}

// Explicit template instantiations
template class SpGEMM_BIN<int, float>;
template class SpGEMM_BIN<int, double>;
template class SpGEMM_BIN<long long, float>;
template class SpGEMM_BIN<long long, double>;

template int hash_find_pos<int>(int, int, const int*);
template long long hash_find_pos<long long>(long long, long long, const long long*);

template int hash_insert_symbolic<int>(int, int, int*);
template long long hash_insert_symbolic<long long>(long long, long long, long long*);

template int hash_insert_numeric<int, float>(int, float, int, int*, float*);
template int hash_insert_numeric<int, double>(int, double, int, int*, double*);
template long long hash_insert_numeric<long long, float>(long long, float, long long, long long*, float*);
template long long hash_insert_numeric<long long, double>(long long, double, long long, long long*, double*);

