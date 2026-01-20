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
 * @brief Binary search to find position of key in sorted array (numeric phase)
 *        Returns the index if found, or -1 if not found
 *        Used for finding position in pre-sorted Ccol array
 *        Optimized: uses linear search for small arrays (< 32) for better cache performance
 * 
 * @param arr Sorted array
 * @param size Array size
 * @param key Key to search for
 * @return Index of key if found, -1 if not found
 */
template <typename IndexType>
IndexType binary_search_find(const IndexType *arr, IndexType size, IndexType key) {
    if (size == 0) return -1;
    
    // For small arrays, use linear search (more cache-friendly, similar to kernel 2)
    if (size < 32) {
        for (IndexType i = 0; i < size; i++) {
            if (arr[i] == key) {
                return i; // Key found
            }
            if (arr[i] > key) {
                return -1; // Key not found (array is sorted)
            }
        }
        return -1; // Key not found
    }
    
    // For larger arrays, use binary search
    IndexType left = 0;
    IndexType right = size;
    
    while (left < right) {
        IndexType mid = left + (right - left) / 2;
        if (arr[mid] < key) {
            left = mid + 1;
        } else if (arr[mid] > key) {
            right = mid;
        } else {
            return mid; // Key found at position mid
        }
    }
    
    return -1; // Key not found
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

/**
 * @brief Optimized symbolic phase: generate and sort Ccol (HSMU-SpGEMM inspired)
 *        This version generates and sorts column indices during symbolic phase,
 *        allowing numeric phase to use binary search instead of insertion.
 * 
 * Implementation strategy (optimized to reduce traversal, inspired by HSMU-SpGEMM):
 * 1. Single pass: Collect unique columns per row and store in per-row buffers
 * 2. Scan: Compute row offsets (cpt)
 * 3. Write: Copy stored columns to ccol (already sorted from insert_if_not_exists)
 * 
 * Key optimization: Store columns during collection to avoid second traversal
 * Note: insert_if_not_exists keeps array sorted, so we can directly copy after scan
 */
template <typename IndexType, typename ValueType>
void spgemm_array_symbolic_new(
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_rows, IndexType c_cols,
    IndexType *cpt, IndexType *&ccol, IndexType &c_nnz,
    SpGEMM_BIN<IndexType, ValueType> *bin)
{
    // Allocate per-row buffers to store columns (we'll need them after scan)
    // Use std::vector for each row to store columns during collection
    std::vector<std::vector<IndexType>> row_cols(c_rows);
    
    // Phase 1: Collect unique columns and store in per-row buffers (single pass)
    #pragma omp parallel num_threads(bin->allocated_thread_num)
    {
        int tid = Le_get_thread_id();
        IndexType start_row = bin->rows_offset[tid];
        IndexType end_row = bin->rows_offset[tid + 1];
        
        // Find max array size needed for this thread's rows
        IndexType max_nz = 0;
        for (IndexType i = start_row; i < end_row; ++i) {
            if (bin->row_nz[i] > max_nz) {
                max_nz = bin->row_nz[i];
            }
        }
        
        // Cap at c_cols
        if (max_nz > c_cols) max_nz = c_cols;
        
        // Allocate temporary array for unique columns (per row, reused)
        IndexType *temp_cols = new_array<IndexType>(max_nz);
        
        for (IndexType i = start_row; i < end_row; ++i) {
            IndexType temp_size = 0;
            
            // Collect unique columns for this row
            for (IndexType j = arpt[i]; j < arpt[i + 1]; ++j) {
                IndexType t_acol = acol[j];
                for (IndexType k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                    IndexType key = bcol[k];
                    if (insert_if_not_exists(temp_cols, temp_size, max_nz, key)) {
                        // Key inserted, temp_cols remains sorted
                    }
                }
            }
            
            // Update row_nz (number of unique columns for this row)
            bin->row_nz[i] = temp_size;
            
            // Store columns for this row (temp_cols is already sorted from insert_if_not_exists)
            row_cols[i].resize(temp_size);
            for (IndexType j = 0; j < temp_size; ++j) {
                row_cols[i][j] = temp_cols[j];
            }
        }
        
        delete_array(temp_cols);
    }
    
    // Scan: Compute row offsets (cpt)
    scan(bin->row_nz, cpt, c_rows + 1, bin->allocated_thread_num);
    c_nnz = cpt[c_rows];
    
    // Allocate ccol now that we know the exact size
    ccol = new_array<IndexType>(c_nnz);
    
    // Phase 2: Write stored columns to ccol (already sorted, no need to sort again)
    #pragma omp parallel num_threads(bin->allocated_thread_num)
    {
        int tid = Le_get_thread_id();
        IndexType start_row = bin->rows_offset[tid];
        IndexType end_row = bin->rows_offset[tid + 1];
        
        for (IndexType i = start_row; i < end_row; ++i) {
            IndexType row_start = cpt[i];
            IndexType row_nnz = bin->row_nz[i];
            
            // Copy stored columns to output ccol (already sorted from insert_if_not_exists)
            for (IndexType j = 0; j < row_nnz; ++j) {
                ccol[row_start + j] = row_cols[i][j];
            }
        }
    }
}

// ============================================================================
// Numeric Phase Implementations
// ============================================================================

/**
 * @brief Optimized numeric phase: find position and accumulate (HSMU-SpGEMM inspired)
 *        Uses pre-sorted Ccol from symbolic phase, eliminating insertion operations
 */
template <bool sortOutput, typename IndexType, typename ValueType>
void spgemm_array_numeric_new(
    const IndexType *arpt, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_rows, IndexType c_cols,
    const IndexType *cpt, const IndexType *ccol, ValueType *cval,
    SpGEMM_BIN<IndexType, ValueType> *bin)
{
    // Numeric phase: compute values using pre-sorted Ccol
    // Ccol is already sorted from spgemm_array_symbolic_new
    // We only need to find position and accumulate (no insertion)
    
    #pragma omp parallel num_threads(bin->allocated_thread_num)
    {
        int tid = Le_get_thread_id();
        IndexType start_row = bin->rows_offset[tid];
        IndexType end_row = bin->rows_offset[tid + 1];
        
        // Initialize cval to 0 for this thread's rows
        // Use memset for better performance (faster than loop for large arrays)
        for (IndexType i = start_row; i < end_row; ++i) {
            IndexType row_start = cpt[i];
            IndexType row_nnz = cpt[i + 1] - row_start;
            if (row_nnz > 0) {
                std::memset(cval + row_start, 0, row_nnz * sizeof(ValueType));
            }
        }
        
        // Accumulate intermediate products
        for (IndexType i = start_row; i < end_row; ++i) {
            IndexType row_start = cpt[i];
            IndexType row_nnz = cpt[i + 1] - row_start;
            if (row_nnz == 0) continue;
            
            // Get pointer to this row's sorted column indices
            const IndexType *row_ccol = ccol + row_start;
            
            // For each non-zero in row i of A
            for (IndexType j = arpt[i]; j < arpt[i + 1]; ++j) {
                IndexType t_acol = acol[j];
                ValueType t_aval = aval[j];
                
                // For each non-zero in column t_acol of B
                for (IndexType k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                    IndexType target_col = bcol[k];
                    ValueType product = t_aval * bval[k];
                    
                    // Binary search to find position in pre-sorted Ccol
                    IndexType pos = binary_search_find(row_ccol, row_nnz, target_col);
                    
                    // Accumulate if found (should always be found since Ccol was generated from symbolic phase)
                    if (pos != -1) {
                        cval[row_start + pos] += product;
                    }
                    // Note: If pos == -1, it means the column was not in symbolic phase
                    // This should not happen if symbolic and numeric phases are consistent
                }
            }
        }
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

// Template instantiations for optimized symbolic phase
template void spgemm_array_symbolic_new<int64_t, float>(
    const int64_t*, const int64_t*, const int64_t*, const int64_t*,
    int64_t, int64_t, int64_t*, int64_t*&, int64_t&, SpGEMM_BIN<int64_t, float>*);
template void spgemm_array_symbolic_new<int64_t, double>(
    const int64_t*, const int64_t*, const int64_t*, const int64_t*,
    int64_t, int64_t, int64_t*, int64_t*&, int64_t&, SpGEMM_BIN<int64_t, double>*);

// Template instantiations for optimized numeric phase
template void spgemm_array_numeric_new<false, int64_t, float>(
    const int64_t*, const int64_t*, const float*,
    const int64_t*, const int64_t*, const float*,
    int64_t, int64_t, const int64_t*, const int64_t*, float*,
    SpGEMM_BIN<int64_t, float>*);
template void spgemm_array_numeric_new<false, int64_t, double>(
    const int64_t*, const int64_t*, const double*,
    const int64_t*, const int64_t*, const double*,
    int64_t, int64_t, const int64_t*, const int64_t*, double*,
    SpGEMM_BIN<int64_t, double>*);
template void spgemm_array_numeric_new<true, int64_t, float>(
    const int64_t*, const int64_t*, const float*,
    const int64_t*, const int64_t*, const float*,
    int64_t, int64_t, const int64_t*, const int64_t*, float*,
    SpGEMM_BIN<int64_t, float>*);
template void spgemm_array_numeric_new<true, int64_t, double>(
    const int64_t*, const int64_t*, const double*,
    const int64_t*, const int64_t*, const double*,
    int64_t, int64_t, const int64_t*, const int64_t*, double*,
    SpGEMM_BIN<int64_t, double>*);

// Helper function instantiations
template int64_t binary_search_find<int64_t>(const int64_t*, int64_t, int64_t);

