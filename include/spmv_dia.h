#ifndef SPMV_DIA_H
#define SPMV_DIA_H

#include "sparse_format.h"

/**
 * @brief Compute y += alpha * A * x + beta * y for a sparse matrix
 *        Matrix Format: DIA
 *        Inside call : 
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param alpha  scaling factor of A*x
 * @param dia    DIA Matrix
 * @param x      vector x
 * @param beta   scaling factor of vector y 
 * @param y      result vector y
 */
template <typename IndexType, typename ValueType>
void LeSpMV_dia(const ValueType alpha, const DIA_Matrix<IndexType, ValueType>& dia, const ValueType * x, const ValueType beta, ValueType * y);

template <typename IndexType, typename ValueType>
void __spmv_dia_serial_simple(  const ValueType alpha, 
                                const IndexType num_rows,
                                const IndexType stride,
                                const IndexType complete_ndiags,
                                const long int  * dia_offset,
                                const ValueType * dia_data,
                                const ValueType * x,
                                const ValueType beta, ValueType * y);

template <typename IndexType, typename ValueType>
void __spmv_dia_omp_simple(  const ValueType alpha, 
                                const IndexType num_rows,
                                const IndexType stride,
                                const IndexType complete_ndiags,
                                const long int  * dia_offset,
                                const ValueType * dia_data,
                                const ValueType * x,
                                const ValueType beta, ValueType * y);

template <typename IndexType, typename ValueType>
void __spmv_dia_alpha(  const ValueType alpha, 
                        const IndexType num_rows,
                        const IndexType stride,
                        const IndexType complete_ndiags,
                        const long int  * dia_offset,
                        const ValueType * dia_data,
                        const ValueType * x,
                        const ValueType beta, ValueType * y);                                

#endif /* SPMV_DIA_H */
