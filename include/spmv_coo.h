#ifndef SPMV_COO_H
#define SPMV_COO_H

#include "sparse_format.h"

/**
 * @brief Compute y += alpha * A * x + beta * y for a sparse matrix
 *        Matrix Format: COO
 *        Inside call : __spmv_coo_omp_simple() to calculation
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param alpha  scaling factor of A*x
 * @param coo    COO Matrix
 * @param x      vector x
 * @param beta   scaling factor of vector y 
 * @param y      result vector y
 */
template <typename IndexType, typename ValueType>
void LeSpMV_coo(const ValueType alpha, const COO_Matrix<IndexType, ValueType>& coo, const ValueType * x, const ValueType beta, ValueType * y);

/**
 * @brief Ai = coo.row_index
 *        Aj = coo.col_index
 *        Ax = coo.values
 */
template <typename IndexType, typename ValueType>
void __spmv_coo_omp_simple (const IndexType num_rows,
                            const IndexType num_nnzs, 
                            const ValueType alpha, 
                            const IndexType *Ai,
                            const IndexType *Aj,
                            const ValueType *Ax,
                            const ValueType * x, 
                            const ValueType beta, ValueType * y);

template <typename IndexType, typename ValueType>
void __spmv_coo_serial_simple(  const IndexType num_rows,
                                const IndexType num_nnzs, 
                                const ValueType alpha, 
                                const IndexType *Ai,
                                const IndexType *Aj,
                                const ValueType *Ax,
                                const ValueType * x, 
                                const ValueType beta, ValueType * y);

template <typename IndexType, typename ValueType>
void __spmv_coo_omp_lb (    const IndexType num_rows,
                            const IndexType num_nnzs, 
                            const ValueType alpha, 
                            const IndexType *Ai,
                            const IndexType *Aj,
                            const ValueType *Ax,
                            const ValueType * x, 
                            const ValueType beta, ValueType * y);

template <typename IndexType, typename ValueType>
void __spmv_coo_omp_alpha (    const IndexType num_rows,
                                const IndexType num_nnzs, 
                                const ValueType alpha, 
                                const IndexType *Ai,
                                const IndexType *Aj,
                                const ValueType *Ax,
                                const ValueType * x, 
                                const ValueType beta, ValueType * y);

#endif /* SPMV_COO_H */
