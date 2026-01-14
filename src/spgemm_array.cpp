/**
 * @file spgemm_array.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Row-wise SpGEMM implementation using sorted array method (HSMU-SpGEMM inspired)
 * @version 0.1
 * @date 2024
 */

#include "../include/spgemm_array.h"
#include "../include/spgemm_utility.h"
#include <cstring>
#include <vector>
#include <algorithm>

// ============================================================================
// Helper Functions
// ============================================================================


/**
 * @brief Binary search to find position for insertion in sorted array
 * @return Position index where key should be inserted, or -1 if key already exists
 */
template <typename IndexType>
IndexType binary_search_pos(const IndexType *arr, IndexType size, IndexType key) {
    if (size == 0) return 0;
    
    IndexType left = 0;
    IndexType right = size;
    
    while (left < right) {
        IndexType mid = left + (right - left) / 2;
        if (arr[mid] < key) {
            left = mid + 1;
        } else if (arr[mid] > key) {
            right = mid;
        } else {
            return -1; // Key already exists
        }
    }
    
    return left; // Position to insert
}

/**
 * @brief Check if key exists in sorted array (symbolic phase)
 * @return true if key exists, false otherwise
 */
template <typename IndexType>
bool binary_search_exists(const IndexType *arr, IndexType size, IndexType key) {
    if (size == 0) return false;
    
    IndexType left = 0;
    IndexType right = size;
    
    while (left < right) {
        IndexType mid = left + (right - left) / 2;
        if (arr[mid] < key) {
            left = mid + 1;
        } else if (arr[mid] > key) {
            right = mid;
        } else {
            return true; // Key exists
        }
    }
    
    return false;
}

/**
 * @brief Insert key into sorted array if not exists (symbolic phase)
 * @return true if inserted, false if already exists
 */
template <typename IndexType>
bool insert_if_not_exists(IndexType *arr, IndexType &size, IndexType capacity, IndexType key) {
    // For small arrays, use linear search (more cache-friendly)
    if (size < 32) {
        for (IndexType i = 0; i < size; i++) {
            if (arr[i] == key) {
                return false; // Already exists
            }
            if (arr[i] > key) {
                // Insert at position i
                if (size >= capacity) return false; // Capacity exceeded
                for (IndexType j = size; j > i; j--) {
                    arr[j] = arr[j - 1];
                }
                arr[i] = key;
                size++;
                return true;
            }
        }
        // Insert at end
        if (size >= capacity) return false;
        arr[size++] = key;
        return true;
    }
    
    // For larger arrays, use binary search
    IndexType pos = binary_search_pos(arr, size, key);
    if (pos == -1) {
        return false; // Already exists
    }
    
    if (size >= capacity) return false; // Capacity exceeded
    
    // Shift elements to make room
    for (IndexType i = size; i > pos; i--) {
        arr[i] = arr[i - 1];
    }
    arr[pos] = key;
    size++;
    return true;
}

/**
 * @brief Insert or accumulate in sorted array (numeric phase)
 * @return true if new element inserted, false if accumulated
 */
template <typename IndexType, typename ValueType>
bool insert_or_accumulate(IndexType *arr_col, ValueType *arr_val, 
                          IndexType &size, IndexType capacity,
                          IndexType key, ValueType val) {
    // For small arrays, use linear search (more cache-friendly)
    if (size < 32) {
        for (IndexType i = 0; i < size; i++) {
            if (arr_col[i] == key) {
                arr_val[i] += val; // Accumulate
                return false;
            }
            if (arr_col[i] > key) {
                // Insert at position i
                if (size >= capacity) return false; // Capacity exceeded
                for (IndexType j = size; j > i; j--) {
                    arr_col[j] = arr_col[j - 1];
                    arr_val[j] = arr_val[j - 1];
                }
                arr_col[i] = key;
                arr_val[i] = val;
                size++;
                return true;
            }
        }
        // Insert at end
        if (size >= capacity) return false;
        arr_col[size] = key;
        arr_val[size] = val;
        size++;
        return true;
    }
    
    // For larger arrays, use binary search
    IndexType pos = binary_search_pos(arr_col, size, key);
    if (pos == -1) {
        // Key exists, find it and accumulate
        // Binary search to find exact position
        IndexType left = 0;
        IndexType right = size;
        while (left < right) {
            IndexType mid = left + (right - left) / 2;
            if (arr_col[mid] < key) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        arr_val[left] += val; // Accumulate
        return false;
    }
    
    // Insert new element at pos
    if (size >= capacity) return false; // Capacity exceeded
    
    // Shift elements to make room
    for (IndexType i = size; i > pos; i--) {
        arr_col[i] = arr_col[i - 1];
        arr_val[i] = arr_val[i - 1];
    }
    arr_col[pos] = key;
    arr_val[pos] = val;
    size++;
    return true;
}

// ============================================================================
// Symbolic Phase Implementations
// ============================================================================

template <typename IndexType, typename ValueType>
void spgemm_array_symbolic_omp_lb(
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_rows, IndexType c_cols,
    IndexType *cpt, IndexType &c_nnz,
    SpGEMM_BIN<IndexType, ValueType> *bin)
{
    // Symbolic phase: count unique columns per row using sorted arrays
    // Note: We don't use bin_id here, only row_nz and rows_offset
    
    int thread_num = Le_get_thread_num();
    
    // Allocate temporary arrays for each thread
    // Each thread needs a buffer to store unique columns for one row at a time
    // We'll use a vector that can grow, but try to estimate max size
    
    #pragma omp parallel num_threads(thread_num)
    {
        int tid = Le_get_thread_id();
        IndexType start_row = bin->rows_offset[tid];
        IndexType end_row = bin->rows_offset[tid + 1];
        
        // Temporary buffer for storing unique columns (per row)
        // Use estimated max size from row_nz (from set_max_bin)
        IndexType max_nz = 0;
        for (IndexType i = start_row; i < end_row; ++i) {
            if (bin->row_nz[i] > max_nz) {
                max_nz = bin->row_nz[i];
            }
        }
        
        // Cap at c_cols
        if (max_nz > c_cols) max_nz = c_cols;
        
        // Allocate temporary array for unique columns
        IndexType *temp_cols = new_array<IndexType>(max_nz);
        
        for (IndexType i = start_row; i < end_row; ++i) {
            IndexType nz = 0;
            IndexType temp_size = 0;
            
            // Collect unique columns for this row
            for (IndexType j = arpt[i]; j < arpt[i + 1]; ++j) {
                IndexType t_acol = acol[j];
                for (IndexType k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                    IndexType key = bcol[k];
                    if (insert_if_not_exists(temp_cols, temp_size, max_nz, key)) {
                        nz++;
                    }
                }
            }
            
            bin->row_nz[i] = nz;
        }
        
        delete_array(temp_cols);
    }
    
    // Set row pointer of matrix C using scan
    scan(bin->row_nz, cpt, c_rows + 1, bin->allocated_thread_num);
    c_nnz = cpt[c_rows];
}

// ============================================================================
// Numeric Phase Implementations
// ============================================================================

template <bool sortOutput, typename IndexType, typename ValueType>
void spgemm_array_numeric_omp_lb(
    const IndexType *arpt, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_rows, IndexType c_cols,
    const IndexType *cpt, IndexType *ccol, ValueType *cval,
    SpGEMM_BIN<IndexType, ValueType> *bin)
{
    // Numeric phase: compute values using sorted arrays
    // Note: We don't use bin_id here, only row_nz and rows_offset
    // Array size for each row = row_nz[i] (exact size, no padding)
    
    int thread_num = Le_get_thread_num();
    
    #pragma omp parallel num_threads(thread_num)
    {
        int tid = Le_get_thread_id();
        IndexType start_row = bin->rows_offset[tid];
        IndexType end_row = bin->rows_offset[tid + 1];
        
        // Find max array size needed for this thread's rows
        IndexType max_array_size = 0;
        for (IndexType i = start_row; i < end_row; ++i) {
            IndexType nz = bin->row_nz[i];
            if (nz > max_array_size) {
                max_array_size = nz;
            }
        }
        
        // Allocate temporary arrays for accumulation (reused for each row)
        IndexType *temp_cols = new_array<IndexType>(max_array_size);
        ValueType *temp_vals = new_array<ValueType>(max_array_size);
        
        for (IndexType i = start_row; i < end_row; ++i) {
            IndexType nz = bin->row_nz[i];
            if (nz == 0) continue;
            
            IndexType offset = cpt[i];
            IndexType array_size = 0;
            
            // Clear arrays (not needed, but for safety)
            // We'll reset array_size to 0 for each row
            
            // Accumulate intermediate products
            for (IndexType j = arpt[i]; j < arpt[i + 1]; ++j) {
                IndexType t_acol = acol[j];
                ValueType t_aval = aval[j];
                
                for (IndexType k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                    ValueType t_val = t_aval * bval[k];
                    IndexType key = bcol[k];
                    
                    // Insert or accumulate in sorted array
                    insert_or_accumulate(temp_cols, temp_vals, array_size, nz, key, t_val);
                }
            }
            
            // Copy sorted array directly to output (already sorted, no extra sort needed)
            // Note: sortOutput is ignored since array is already sorted
            for (IndexType j = 0; j < array_size; ++j) {
                ccol[offset + j] = temp_cols[j];
                cval[offset + j] = temp_vals[j];
            }
        }
        
        delete_array(temp_cols);
        delete_array(temp_vals);
    }
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

#include <cstdint>

// Helper functions
template int64_t binary_search_pos<int64_t>(const int64_t*, int64_t, int64_t);
template bool binary_search_exists<int64_t>(const int64_t*, int64_t, int64_t);
template bool insert_if_not_exists<int64_t>(int64_t*, int64_t&, int64_t, int64_t);
template bool insert_or_accumulate<int64_t, float>(int64_t*, float*, int64_t&, int64_t, int64_t, float);
template bool insert_or_accumulate<int64_t, double>(int64_t*, double*, int64_t&, int64_t, int64_t, double);

// Symbolic phase
template void spgemm_array_symbolic_omp_lb<int64_t, float>(
    const int64_t*, const int64_t*, const int64_t*, const int64_t*, 
    int64_t, int64_t, int64_t*, int64_t&, SpGEMM_BIN<int64_t, float>*);
template void spgemm_array_symbolic_omp_lb<int64_t, double>(
    const int64_t*, const int64_t*, const int64_t*, const int64_t*, 
    int64_t, int64_t, int64_t*, int64_t&, SpGEMM_BIN<int64_t, double>*);

// Numeric phase
template void spgemm_array_numeric_omp_lb<false, int64_t, float>(
    const int64_t*, const int64_t*, const float*, 
    const int64_t*, const int64_t*, const float*, 
    int64_t, int64_t, const int64_t*, int64_t*, float*, 
    SpGEMM_BIN<int64_t, float>*);
template void spgemm_array_numeric_omp_lb<false, int64_t, double>(
    const int64_t*, const int64_t*, const double*, 
    const int64_t*, const int64_t*, const double*, 
    int64_t, int64_t, const int64_t*, int64_t*, double*, 
    SpGEMM_BIN<int64_t, double>*);
template void spgemm_array_numeric_omp_lb<true, int64_t, float>(
    const int64_t*, const int64_t*, const float*, 
    const int64_t*, const int64_t*, const float*, 
    int64_t, int64_t, const int64_t*, int64_t*, float*, 
    SpGEMM_BIN<int64_t, float>*);
template void spgemm_array_numeric_omp_lb<true, int64_t, double>(
    const int64_t*, const int64_t*, const double*, 
    const int64_t*, const int64_t*, const double*, 
    int64_t, int64_t, const int64_t*, int64_t*, double*, 
    SpGEMM_BIN<int64_t, double>*);

