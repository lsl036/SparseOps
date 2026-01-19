#ifndef SPGEMM_H
#define SPGEMM_H

#include "sparse_format.h"
#include "spgemm_hash.h"
#include "spgemm_array.h"
#include "spgemm_bin.h"
#include "spgemm_Flength_hash.h"
#include "spgemm_cluster.h"
#include "sparse_conversion.h"

/**
 * @brief Compute C = A * B (SpGEMM) - Row-wise methods only
 *        Supports multiple kernel implementations:
 *        Kernel 1: Hash-based row-wise method (default, OpenMP with load balancing)
 *        Kernel 2: Array-based row-wise method (HSMU-SpGEMM inspired, sorted arrays, original version)
 *        Kernel 3: Optimized array-based row-wise method (HSMU-SpGEMM inspired, pre-sorted Ccol, optimized version)
 *        Note: For cluster-wise methods, use LeSpGEMM_FLength instead
 * 
 * @tparam sortOutput Whether to sort output columns (template parameter)
 * @tparam IndexType 
 * @tparam ValueType 
 * @param A Input matrix A in CSR format
 * @param B Input matrix B in CSR format  
 * @param C Output matrix C in CSR format (will be allocated)
 * @param kernel_flag Kernel selection flag (1=hash row-wise, 2=array row-wise original, 3=array row-wise optimized)
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

/**
 * @brief Optimized array-based row-wise SpGEMM implementation (HSMU-SpGEMM inspired)
 *        Uses pre-sorted Ccol from symbolic phase, eliminating insertion operations
 *        This is the optimized version using spgemm_array_symbolic_new and spgemm_array_numeric_new
 * 
 * Key optimizations:
 * - Symbolic phase generates and sorts Ccol
 * - Numeric phase uses binary search to find position (no insertion)
 * - Better performance for dense rows (no O(n) element shifting)
 * 
 * @tparam sortOutput Whether to sort output columns (ignored, ccol is already sorted)
 */
template <bool sortOutput = true, typename IndexType, typename ValueType>
void LeSpGEMM_array_rowwise_new(const CSR_Matrix<IndexType, ValueType> &A,
                                 const CSR_Matrix<IndexType, ValueType> &B,
                                 CSR_Matrix<IndexType, ValueType> &C);

/**
 * @brief Fixed-length Cluster-wise SpGEMM implementation
 *        Input and output are CSR_FlengthCluster format (no format conversion)
 *        Supports multiple kernels:
 *        Kernel 1: Hash-based cluster-wise method (default)
 *        Kernel 2: Array-based cluster-wise method (future)
 * 
 * @tparam sortOutput Whether to sort output columns (template parameter)
 * @tparam IndexType 
 * @tparam ValueType 
 * @param A_cluster Input matrix A in CSR_FlengthCluster format
 * @param B Input matrix B in CSR_Matrix format
 * @param C_cluster Output matrix C in CSR_FlengthCluster format (will be allocated)
 * @param kernel_flag Kernel selection flag (1=hash-based cluster-wise, default)
 */
template <bool sortOutput = true, typename IndexType, typename ValueType>
void LeSpGEMM_FLength(const CSR_FlengthCluster<IndexType, ValueType> &A_cluster,
                      const CSR_Matrix<IndexType, ValueType> &B,
                      CSR_FlengthCluster<IndexType, ValueType> &C_cluster,
                      int kernel_flag = 1);

/**
 * @brief Hash-based Fixed-length Cluster-wise SpGEMM implementation
 *        Uses hash tables for accumulation at cluster level (kernel_flag = 1)
 *        This is a sub-interface called by LeSpGEMM_FLength
 * 
 * @tparam sortOutput Whether to sort output columns (template parameter)
 * @tparam IndexType 
 * @tparam ValueType 
 * @param A_cluster Input matrix A in CSR_FlengthCluster format
 * @param B Input matrix B in CSR_Matrix format
 * @param C_cluster Output matrix C in CSR_FlengthCluster format (will be allocated)
 */
template <bool sortOutput = true, typename IndexType, typename ValueType>
void LeSpGEMM_hash_FLength(const CSR_FlengthCluster<IndexType, ValueType> &A_cluster,
                           const CSR_Matrix<IndexType, ValueType> &B,
                           CSR_FlengthCluster<IndexType, ValueType> &C_cluster);

/**
 * @brief Array-based Fixed-length Cluster-wise SpGEMM implementation
 *        Uses sorted arrays for accumulation at cluster level (kernel_flag = 2)
 *        This is a sub-interface called by LeSpGEMM_FLength
 * 
 * Key differences from hash-based method:
 * - Uses sorted arrays instead of hash tables (no hash collisions)
 * - Array size = exact cluster_nz (no 2^N padding, better memory efficiency)
 * - Natural sorting during symbolic phase (no extra sort step needed)
 * - Binary search for O(log n) lookup in numeric phase
 * 
 * @tparam sortOutput Ignored (ccolids is already sorted from symbolic phase)
 * @tparam IndexType 
 * @tparam ValueType 
 * @param A_cluster Input matrix A in CSR_FlengthCluster format
 * @param B Input matrix B in CSR_Matrix format
 * @param C_cluster Output matrix C in CSR_FlengthCluster format (will be allocated)
 */
template <bool sortOutput = true, typename IndexType, typename ValueType>
void LeSpGEMM_array_FLength(const CSR_FlengthCluster<IndexType, ValueType> &A_cluster,
                            const CSR_Matrix<IndexType, ValueType> &B,
                            CSR_FlengthCluster<IndexType, ValueType> &C_cluster);

#endif /* SPGEMM_H */

