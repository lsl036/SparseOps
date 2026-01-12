#ifndef SPGEMM_H
#define SPGEMM_H

#include "sparse_format.h"
#include "spgemm_hash.h"
#include "spgemm_array.h"
#include "spgemm_bin.h"

/**
 * @brief Compute C = A * B (SpGEMM)
 *        Supports multiple kernel implementations:
 *        Kernel 1: Hash-based row-wise method (default, OpenMP with load balancing)
 *        Kernel 2: Array-based row-wise method (HSMU-SpGEMM inspired, sorted arrays)
 *        Future: Kernel 3+: Cluster-wise and other methods
 * 
 * @tparam sortOutput Whether to sort output columns (template parameter)
 * @tparam IndexType 
 * @tparam ValueType 
 * @param A Input matrix A in CSR format
 * @param B Input matrix B in CSR format  
 * @param C Output matrix C in CSR format (will be allocated)
 * @param kernel_flag Kernel selection flag (1=hash row-wise, 2=array row-wise)
 */
template <bool sortOutput = true, typename IndexType, typename ValueType>
void LeSpGEMM(const CSR_Matrix<IndexType, ValueType> &A,
              const CSR_Matrix<IndexType, ValueType> &B,
              CSR_Matrix<IndexType, ValueType> &C,
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
 * @brief Hash-based row-wise SpGEMM implementation
 *        Uses hash tables for accumulation (kernel_flag = 1)
 * 
 * @tparam sortOutput Whether to sort output columns (template parameter)
 */
template <bool sortOutput = true, typename IndexType, typename ValueType>
void LeSpGEMM_hash_rowwise(const CSR_Matrix<IndexType, ValueType> &A,
                           const CSR_Matrix<IndexType, ValueType> &B,
                           CSR_Matrix<IndexType, ValueType> &C,
                           int kernel_flag = 1);

/**
 * @brief Array-based row-wise SpGEMM implementation (HSMU-SpGEMM inspired)
 *        Uses sorted arrays for accumulation (kernel_flag = 2)
 *        Benefits: no hash collisions, better memory efficiency, natural sorting
 * 
 * @tparam sortOutput Whether to sort output columns (template parameter, ignored since array is already sorted)
 */
template <bool sortOutput = true, typename IndexType, typename ValueType>
void LeSpGEMM_array_rowwise(const CSR_Matrix<IndexType, ValueType> &A,
                            const CSR_Matrix<IndexType, ValueType> &B,
                            CSR_Matrix<IndexType, ValueType> &C,
                            int kernel_flag = 2);

#endif /* SPGEMM_H */

