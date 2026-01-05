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
 *        Kernel 1: OpenMP parallel implementation (default)
 *        Kernel 2: OpenMP with load balancing using BIN
 */

/**
 * @brief Symbolic phase: compute structure of C = A * B
 *        OpenMP parallel version
 */
template <typename IndexType, typename ValueType>
void spgemm_hash_symbolic_omp(
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_rows, IndexType c_cols,
    IndexType *&cpt, IndexType *&ccol, IndexType &c_nnz);

/**
 * @brief Symbolic phase: compute structure of C = A * B
 *        OpenMP with load balancing using BIN
 */
template <typename IndexType, typename ValueType>
void spgemm_hash_symbolic_omp_lb(
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_rows, IndexType c_cols,
    IndexType *&cpt, IndexType *&ccol, IndexType &c_nnz,
    SpGEMM_BIN<IndexType, ValueType> *bin);

/**
 * @brief Numeric phase: compute values of C = A * B
 *        OpenMP parallel version
 */
template <typename IndexType, typename ValueType>
void spgemm_hash_numeric_omp(
    const IndexType *arpt, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_rows, IndexType c_cols,
    const IndexType *cpt, const IndexType *ccol, ValueType *cval);

/**
 * @brief Numeric phase: compute values of C = A * B
 *        OpenMP with load balancing using BIN
 */
template <typename IndexType, typename ValueType>
void spgemm_hash_numeric_omp_lb(
    const IndexType *arpt, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_rows, IndexType c_cols,
    const IndexType *cpt, const IndexType *ccol, ValueType *cval,
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

