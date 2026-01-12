#ifndef SPGEMM_ARRAY_H
#define SPGEMM_ARRAY_H

#include "sparse_format.h"
#include "spgemm_bin.h"
#include "memopt.h"
#include "thread.h"
#include "general_config.h"
#include <omp.h>
#include <algorithm>

/**
 * @brief Row-wise SpGEMM using sorted array method (HSMU-SpGEMM inspired)
 *        OpenMP with load balancing using BIN
 * 
 * Key differences from hash-based approach:
 * 1. Uses sorted arrays instead of hash tables (no hash collisions)
 * 2. Array size = exact row_nz (no 2^N padding, better memory efficiency)
 * 3. Natural sorting during accumulation (no extra sort step needed)
 * 4. Binary search for O(log n) lookup/insert
 */

/**
 * @brief Symbolic phase: compute structure of C = A * B
 *        OpenMP with load balancing using BIN
 *        cpt should be pre-allocated (c_rows + 1 elements)
 *        This function performs scan internally to compute nnz
 * 
 * Note: For array-based method, symbolic phase is similar to hash-based,
 *       but we can use a simpler approach (set or sorted array) to count unique columns.
 */
template <typename IndexType, typename ValueType>
void spgemm_array_symbolic_omp_lb(
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_rows, IndexType c_cols,
    IndexType *cpt, IndexType &c_nnz,
    SpGEMM_BIN<IndexType, ValueType> *bin);

/**
 * @brief Numeric phase: compute values of C = A * B using sorted arrays
 *        OpenMP with load balancing using BIN
 * 
 * Implementation details:
 * - Each row uses a sorted array of size row_nz[i] (exact size, no padding)
 * - Binary search for O(log n) lookup/insert
 * - Array is kept sorted during accumulation
 * - Direct output (already sorted, no extra sort step needed)
 * 
 * @tparam sortOutput If true, ensures output is sorted (already sorted by default)
 */
template <bool sortOutput, typename IndexType, typename ValueType>
void spgemm_array_numeric_omp_lb(
    const IndexType *arpt, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_rows, IndexType c_cols,
    const IndexType *cpt, IndexType *ccol, ValueType *cval,
    SpGEMM_BIN<IndexType, ValueType> *bin);

/**
 * @brief Helper: Binary search to find position for insertion in sorted array
 * @return Position index, or -1 if key already exists (for numeric phase)
 */
template <typename IndexType>
IndexType binary_search_pos(const IndexType *arr, IndexType size, IndexType key);

/**
 * @brief Helper: Insert or accumulate in sorted array (numeric phase)
 * @param arr_col Column indices array (sorted)
 * @param arr_val Values array
 * @param size Current size of array (will be updated if new element inserted)
 * @param key Column index to insert/accumulate
 * @param val Value to add
 * @return true if new element inserted, false if accumulated
 */
template <typename IndexType, typename ValueType>
bool insert_or_accumulate(IndexType *arr_col, ValueType *arr_val, 
                          IndexType &size, IndexType capacity,
                          IndexType key, ValueType val);

/**
 * @brief Helper: Check if key exists in sorted array (symbolic phase)
 * @return true if key exists, false otherwise
 */
template <typename IndexType>
bool binary_search_exists(const IndexType *arr, IndexType size, IndexType key);

/**
 * @brief Helper: Insert key into sorted array if not exists (symbolic phase)
 * @return true if inserted, false if already exists
 */
template <typename IndexType>
bool insert_if_not_exists(IndexType *arr, IndexType &size, IndexType capacity, IndexType key);

#endif /* SPGEMM_ARRAY_H */

