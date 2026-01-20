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
 * @brief Optimized symbolic phase: generate and sort Ccol (HSMU-SpGEMM inspired)
 *        OpenMP with load balancing using BIN
 *        This version generates and sorts column indices in symbolic phase,
 *        eliminating the need for insertion operations in numeric phase.
 * 
 * Key optimization:
 * - Generates sorted Ccol during symbolic phase
 * - Numeric phase only needs to find position and accumulate (no insertion)
 * - Better performance for dense rows (no O(n) element shifting)
 * 
 * @param cpt Row pointer array (pre-allocated, c_rows + 1 elements)
 * @param ccol Column index array (will be allocated and filled with sorted columns)
 * @param c_nnz Total number of non-zeros (output)
 */
template <typename IndexType, typename ValueType>
void spgemm_array_symbolic_new(
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_rows, IndexType c_cols,
    IndexType *cpt, IndexType *&ccol, IndexType &c_nnz,
    SpGEMM_BIN<IndexType, ValueType> *bin);

/**
 * @brief Optimized numeric phase: find position and accumulate (HSMU-SpGEMM inspired)
 *        OpenMP with load balancing using BIN
 *        This version uses pre-sorted Ccol from symbolic phase,
 *        eliminating the need for insertion operations.
 * 
 * Key optimization:
 * - Ccol is already sorted from symbolic phase
 * - Use binary search to find position in pre-sorted array
 * - Direct accumulation to cval (no insertion, no temporary arrays)
 * - Better performance: O(log n) lookup + O(1) accumulate vs O(n) insertion
 * 
 * @tparam sortOutput Ignored (ccol is already sorted from symbolic phase)
 * @param ccol Pre-sorted column index array (from spgemm_array_symbolic_new)
 * @param cval Values array (will be initialized to 0 and accumulated)
 */
template <bool sortOutput, typename IndexType, typename ValueType>
void spgemm_array_numeric_new(
    const IndexType *arpt, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_rows, IndexType c_cols,
    const IndexType *cpt, const IndexType *ccol, ValueType *cval,
    SpGEMM_BIN<IndexType, ValueType> *bin);

/**
 * @brief SPA-based numeric phase: use dense accumulator for O(1) access (HSMU-SpGEMM inspired)
 *        Uses Sparse Accumulator (SPA) - a dense array of size c_cols per thread
 *        This eliminates the need for binary search, achieving O(1) access instead of O(log n)
 * 
 * Key optimization:
 * - Each thread maintains a dense accumulator array (SPA) of size c_cols
 * - Direct O(1) access via column index: spa_val[bcol[k]] += product
 * - No binary search needed, eliminating lookup overhead
 * - Better cache locality for dense accumulation
 * - Only clear columns that are actually used (from ccol)
 * 
 * Memory trade-off:
 * - Memory: O(c_cols) per thread (can be large for wide matrices)
 * - Performance: O(1) access vs O(log n) binary search
 * - Optimization: Automatically falls back to binary search method if c_cols > 1M
 *   to avoid excessive memory allocation for very wide matrices
 * 
 * @tparam sortOutput Ignored (ccol is already sorted from symbolic phase)
 * @param ccol Pre-sorted column index array (from spgemm_array_symbolic_new)
 * @param cval Values array (will be filled from SPA)
 */
template <bool sortOutput, typename IndexType, typename ValueType>
void spgemm_spa_numeric(
    const IndexType *arpt, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_rows, IndexType c_cols,
    const IndexType *cpt, const IndexType *ccol, ValueType *cval,
    SpGEMM_BIN<IndexType, ValueType> *bin);

/**
 * @brief Helper: Binary search to find position for insertion in sorted array
 * @return Position index, or -1 if key already exists (for numeric phase)
 */
template <typename IndexType>
IndexType binary_search_pos(const IndexType *arr, IndexType size, IndexType key);

/**
 * @brief Helper: Binary search to find position of key in sorted array (numeric phase)
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
IndexType binary_search_find(const IndexType *arr, IndexType size, IndexType key);

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

