#ifndef SPMV_S_ELL_H
#define SPMV_S_ELL_H

#include "sparse_format.h"

template <typename IndexType, typename ValueType>
void __spmv_sell_serial_simple(const IndexType num_rows,
                            const IndexType row_num_perC,
                            const IndexType total_chunk_num,
                            const ValueType alpha,
                            const IndexType *max_row_width,
                            const IndexType * const *col_index,
                            const ValueType * const *values,
                            const ValueType * x, 
                            const ValueType beta, 
                            ValueType * y);

template <typename IndexType, typename ValueType>
void __spmv_sell_omp_simple(const IndexType num_rows,
                            const IndexType row_num_perC,
                            const IndexType total_chunk_num,
                            const ValueType alpha,
                            const IndexType *max_row_width,
                            const IndexType * const *col_index,
                            const ValueType * const *values,
                            const ValueType * x, 
                            const ValueType beta, 
                            ValueType * y);

template <typename IndexType, typename ValueType>
void __spmv_sell_omp_lb_row(const IndexType num_rows,
                            const IndexType row_num_perC,
                            const IndexType total_chunk_num,
                            const IndexType num_nnzs, 
                            const ValueType alpha, 
                            const IndexType *max_row_width,
                            const IndexType * const *col_index,
                            const ValueType * const *values,
                            const ValueType * x, 
                            const ValueType beta, 
                            ValueType * y,
                            IndexType *partition);

/**
 * @brief Compute y += alpha * A * x + beta * y for a sparse matrix
 *        Matrix Format: SELL
 *        Inside call : __spmv_sell_omp_simple() to calculation
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param alpha  scaling factor of A*x
 * @param sell    SELL Matrix
 * @param x      vector x
 * @param beta   scaling factor of vector y 
 * @param y      result vector y
 */
template <typename IndexType, typename ValueType>
void LeSpMV_sell(const ValueType alpha, const S_ELL_Matrix<IndexType, ValueType>& sell, const ValueType * x, const ValueType beta, ValueType * y);

#endif /* SPMV_S_ELL_H */
