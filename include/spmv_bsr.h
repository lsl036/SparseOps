#ifndef SPMV_BSR_H
#define SPMV_BSR_H

#include "sparse_format.h"

template <typename IndexType, typename ValueType>
void __spmv_bsr_serial_simple(  const IndexType num_rows,
                                const IndexType blockDimRow,
                                const IndexType blockDimCol,
                                const IndexType mb,
                                const ValueType alpha,
                                const IndexType *row_ptr,
                                const IndexType *col_index,
                                const ValueType *values,
                                const ValueType *x,
                                const ValueType beta,
                                ValueType *y);

template <typename IndexType, typename ValueType>
void __spmv_bsr_omp_simple( const IndexType num_rows,
                            const IndexType blockDimRow,
                            const IndexType blockDimCol,
                            const IndexType mb,
                            const ValueType alpha,
                            const IndexType *row_ptr,
                            const IndexType *col_index,
                            const ValueType *values,
                            const ValueType *x,
                            const ValueType beta,
                            ValueType *y);

template <typename IndexType, typename ValueType>
void __spmv_bsr_lb_alpha(   const IndexType blockDimRow,
                            const IndexType blockDimCol,
                            const IndexType mb,
                            const IndexType num_rows,
                            const ValueType alpha,
                            const IndexType *row_ptr,
                            const IndexType *col_index,
                            const ValueType *values,
                            const ValueType *x,
                            const ValueType beta,
                            ValueType *y);                            

/**
 * @brief Compute y += alpha * A * x + beta * y for a sparse matrix
 *        Matrix Format: Blocked CSR
 *        Inside call : __spmv_bsr_omp_simple() to calculation
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param alpha 
 * @param bsr
 * @param x 
 * @param beta 
 * @param y 
 */
template <typename IndexType, typename ValueType>
void LeSpMV_bsr(const ValueType alpha, const BSR_Matrix<IndexType, ValueType>& bsr, const ValueType *x, const ValueType beta, ValueType *y);

#endif // SPMV_BSR_H