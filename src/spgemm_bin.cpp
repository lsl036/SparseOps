/**
 * @file spgemm_bin.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Implementation of BIN class for SpGEMM load balancing
 * @version 0.1
 * @date 2026
 */

#include "../include/spgemm_bin.h"
#include <cmath>

template <typename IndexType, typename ValueType>
SpGEMM_BIN<IndexType, ValueType>::SpGEMM_BIN(IndexType nrows, IndexType min_ht_sz)
    : num_rows(nrows), min_ht_size(min_ht_sz), max_bin_id(0)
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
    
    if (local_hash_table_id != nullptr) {
        int thread_num = Le_get_thread_num();
        for (int i = 0; i < thread_num; i++) {
            if (local_hash_table_id[i] != nullptr) {
                delete_array(local_hash_table_id[i]);
            }
            if (local_hash_table_val[i] != nullptr) {
                delete_array(local_hash_table_val[i]);
            }
        }
        delete[] local_hash_table_id;
        delete[] local_hash_table_val;
    }
    
    delete_array(hash_table_size);
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
    
    // Find max bin ID based on work distribution
    IndexType max_work = 0;
    for (IndexType i = 0; i < num_rows; i++) {
        if (row_nz[i] > max_work) {
            max_work = row_nz[i];
        }
    }
    
    // Calculate max_bin_id: use logarithmic binning
    if (max_work > 0) {
        max_bin_id = static_cast<IndexType>(std::log2(max_work)) + 1;
    } else {
        max_bin_id = 1;
    }
}

template <typename IndexType, typename ValueType>
void SpGEMM_BIN<IndexType, ValueType>::set_bin_id(
    IndexType nrows, IndexType ncols, IndexType min_ht_sz)
{
    // Assign bin ID based on estimated work (logarithmic binning)
    #pragma omp parallel for
    for (IndexType i = 0; i < num_rows; i++) {
        if (row_nz[i] > 0) {
            IndexType log_work = static_cast<IndexType>(std::log2(row_nz[i]));
            bin_id[i] = log_work;
        } else {
            bin_id[i] = 0;
        }
    }
}

template <typename IndexType, typename ValueType>
void SpGEMM_BIN<IndexType, ValueType>::create_local_hash_table(IndexType max_cols)
{
    int thread_num = Le_get_thread_num();
    
    IndexType hash_size = get_hash_size(max_cols, min_ht_size);
    
    hash_table_size = new_array<IndexType>(thread_num);
    std::fill_n(hash_table_size, thread_num, hash_size);
    
    local_hash_table_id = new IndexType*[thread_num];
    local_hash_table_val = new ValueType*[thread_num];
    
    #pragma omp parallel
    {
        int tid = Le_get_thread_id();
        local_hash_table_id[tid] = new_array<IndexType>(hash_size);
        local_hash_table_val[tid] = new_array<ValueType>(hash_size);
        
        // Initialize hash table
        std::fill_n(local_hash_table_id[tid], hash_size, static_cast<IndexType>(-1));
        std::fill_n(local_hash_table_val[tid], hash_size, static_cast<ValueType>(0));
    }
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
    IndexType pos = hash_func(key, hash_size);
    IndexType start_pos = pos;
    
    while (hash_table_id[pos] != -1 && hash_table_id[pos] != key) {
        pos = (pos + 1) % hash_size;
        if (pos == start_pos) {
            return -1; // Hash table full
        }
    }
    
    return pos;
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

