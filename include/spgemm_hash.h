#ifndef SPGEMM_HASH_H
#define SPGEMM_HASH_H

#include "sparse_format.h"
#include "spgemm_bin.h"
#include "memopt.h"
#include "thread.h"
#include "general_config.h"
#include <omp.h>
#include <algorithm>

/**
 * @brief Row-wise SpGEMM using hash table method
 *        OpenMP with load balancing using BIN
 */

/**
 * @brief Symbolic phase: compute structure of C = A * B
 *        OpenMP with load balancing using BIN
 *        cpt should be pre-allocated (c_rows + 1 elements)
 *        This function performs scan internally to compute nnz
 */
template <typename IndexType, typename ValueType>
void spgemm_hash_symbolic_omp_lb(
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_rows, IndexType c_cols,
    IndexType *cpt, IndexType &c_nnz,
    SpGEMM_BIN<IndexType, ValueType> *bin);

/**
 * @brief Numeric phase: compute values of C = A * B
 *        OpenMP with load balancing using BIN
 */
template <bool sortOutput, typename IndexType, typename ValueType>
void spgemm_hash_numeric_omp_lb(
    const IndexType *arpt, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_rows, IndexType c_cols,
    const IndexType *cpt, IndexType *ccol, ValueType *cval,
    SpGEMM_BIN<IndexType, ValueType> *bin);

/**
 * @brief Numeric phase with Top-K selection: compute values and keep only top-k per row
 *        Matching reference hash_numeric_topk implementation
 *        OpenMP with load balancing using BIN
 *        Excludes diagonal elements (i == j) from top-k selection
 *        The values stored in C are already similarity scores (intersection sizes for binary pattern)
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param arpt Row pointer array of matrix A
 * @param acol Column index array of matrix A
 * @param aval Values array of matrix A (should be binary pattern, all 1.0)
 * @param brpt Row pointer array of matrix B (AT)
 * @param bcol Column index array of matrix B (AT)
 * @param bval Values array of matrix B (AT, should be binary pattern, all 1.0)
 * @param c_rows Number of rows in output matrix C
 * @param c_cols Number of columns in output matrix C
 * @param cpt Row pointer array of output matrix C (from symbolic phase)
 * @param ccol Output column indices array (pre-allocated, will be filled)
 * @param cval Output values array (pre-allocated, will be filled with similarity scores)
 * @param top_k Number of top similarities to keep per row
 * @param row_nnz Output: number of non-zeros per row after top-k selection (will be filled)
 * @param bin BIN object for load balancing
 */
template <typename IndexType, typename ValueType>
void spgemm_hash_numeric_topk_omp_lb(
    const IndexType *arpt, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_rows, IndexType c_cols,
    const IndexType *cpt, IndexType *ccol, ValueType *cval,
    IndexType top_k, IndexType *row_nnz,
    SpGEMM_BIN<IndexType, ValueType> *bin);

/**
 * @brief Sort columns within each row (for output)
 */
template <typename IndexType, typename ValueType>
void sort_csr_columns(IndexType num_rows,
                      const IndexType *row_offset,
                      IndexType *col_index,
                      ValueType *values);

#endif /* SPGEMM_HASH_H */

