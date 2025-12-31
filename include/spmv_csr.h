#ifndef SPMV_CSR_H
#define SPMV_CSR_H

#include "sparse_format.h"

/**
 * @brief Compute y += alpha * A * x + beta * y for a sparse matrix
 *        Matrix Format: CSR
 *        Inside call : __spmv_csr_omp_simple() to calculation
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param alpha  scaling factor of A*x
 * @param csr    CSR Matrix
 * @param x      vector x
 * @param beta   scaling factor of vector y 
 * @param y      result vector y
 */
template <typename IndexType, typename ValueType>
void LeSpMV_csr(const ValueType alpha, const CSR_Matrix<IndexType, ValueType>& csr, const ValueType * x, const ValueType beta, ValueType * y);

/**
 * @brief Ap = csr.row_offest
 *        Aj = csr.col_index
 *        Ax = csr.values
 */
template <typename IndexType, typename ValueType>
void __spmv_csr_omp_simple (const IndexType num_rows, 
                            const ValueType alpha, 
                            const IndexType *Ap,
                            const IndexType *Aj,
                            const ValueType *Ax,
                            const ValueType * x, 
                            const ValueType beta, ValueType * y);

template <typename IndexType, typename ValueType>
void __spmv_csr_serial_simple(  const IndexType num_rows, 
                                const ValueType alpha, 
                                const IndexType *Ap,
                                const IndexType *Aj,
                                const ValueType *Ax,
                                const ValueType * x, 
                                const ValueType beta, ValueType * y);

template <typename IndexType, typename ValueType>
void __spmv_csr_omp_lb (const IndexType num_rows, 
                        const ValueType alpha, 
                        const IndexType *Ap,
                        const IndexType *Aj,
                        const ValueType *Ax,
                        const ValueType * x, 
                        const ValueType beta, ValueType * y,
                        IndexType *partition);

template <typename IndexType, typename ValueType>
inline void  __spmv_csr_perthread(  const ValueType alpha, 
                                    const IndexType *Ap,
                                    const IndexType *Aj,
                                    const ValueType *Ax,
                                    const ValueType * x, 
                                    const ValueType beta, ValueType * y,
                                    const IndexType local_m_s,
                                    const IndexType local_m_e);

#endif /* SPMV_CSR_H */
