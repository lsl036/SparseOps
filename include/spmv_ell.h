#ifndef SPMV_ELL_H
#define SPMV_ELL_H

#include "sparse_format.h"

template <typename IndexType, typename ValueType>
void __spmv_ell_serial_simple(  const IndexType num_rows,
                                const IndexType maxNonzeros, 
                                const ValueType alpha, 
                                const IndexType *colIndex,
                                const ValueType *values,
                                const ValueType * x, 
                                const ValueType beta, ValueType * y,
                                const LeadingDimension ld);

template <typename IndexType, typename ValueType>
void __spmv_ell_omp_simple( const IndexType num_rows,
                            const IndexType maxNonzeros, 
                            const ValueType alpha, 
                            const IndexType *colIndex,
                            const ValueType *values,
                            const ValueType * x, 
                            const ValueType beta, ValueType * y,
                            const LeadingDimension ld);

template <typename IndexType, typename ValueType>
void __spmv_ell_omp_lb_row( const IndexType num_rows,
                            const IndexType maxNonzeros, 
                            const IndexType num_nnzs,
                            const ValueType alpha, 
                            const IndexType *colIndex,
                            const ValueType *values,
                            const ValueType * x, 
                            const ValueType beta, ValueType * y,
                            const LeadingDimension ld,
                            IndexType *partition);


/**
 * @brief Compute y += alpha * A * x + beta * y for a sparse matrix
 *        Matrix Format: ELL
 *        Inside call : __spmv_ell_omp_simple() to calculation
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param alpha  scaling factor of A*x
 * @param ell    ELL Matrix
 * @param x      vector x
 * @param beta   scaling factor of vector y 
 * @param y      result vector y
 */
template <typename IndexType, typename ValueType>
void LeSpMV_ell(const ValueType alpha, const ELL_Matrix<IndexType, ValueType>& ell, const ValueType * x, const ValueType beta, ValueType * y);

#endif /* SPMV_ELL_H */
