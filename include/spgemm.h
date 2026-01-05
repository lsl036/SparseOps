#ifndef SPGEMM_H
#define SPGEMM_H

#include "sparse_format.h"
#include "spgemm_hash.h"
#include "spgemm_bin.h"

/**
 * @brief Compute C = A * B (SpGEMM) using hash table method
 *        Supports multiple kernel implementations:
 *        Kernel 1: OpenMP parallel implementation (default)
 *        Kernel 2: OpenMP with load balancing using BIN
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param A Input matrix A in CSR format
 * @param B Input matrix B in CSR format  
 * @param C Output matrix C in CSR format (will be allocated)
 * @param sort_output Whether to sort output columns (default: true)
 * @param kernel_flag Kernel selection flag (1=omp default, 2=omp_lb)
 */
template <typename IndexType, typename ValueType>
void LeSpGEMM(const CSR_Matrix<IndexType, ValueType> &A,
              const CSR_Matrix<IndexType, ValueType> &B,
              CSR_Matrix<IndexType, ValueType> &C,
              bool sort_output = true,
              int kernel_flag = 1);

/**
 * @brief Calculate FLOPs for SpGEMM operation C = A * B
 *        FLOPs = sum over all rows i: (nnz(A[i,:]) * nnz(B[:,:]))
 *        Actually: sum over all (i,k) in A: nnz(B[k,:])
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param A Input matrix A
 * @param B Input matrix B
 * @return long long int Estimated FLOPs
 */
template <typename IndexType, typename ValueType>
long long int get_spgemm_flop(const CSR_Matrix<IndexType, ValueType> &A,
                               const CSR_Matrix<IndexType, ValueType> &B);

/**
 * @brief Row-wise SpGEMM implementation (current implementation)
 *        This is the base implementation, can be extended with cluster-wise later
 */
template <typename IndexType, typename ValueType>
void LeSpGEMM_rowwise(const CSR_Matrix<IndexType, ValueType> &A,
                      const CSR_Matrix<IndexType, ValueType> &B,
                      CSR_Matrix<IndexType, ValueType> &C,
                      bool sort_output = true,
                      int kernel_flag = 1);

#endif /* SPGEMM_H */

