#ifndef SPGEMM_FLENGTH_HASH_H
#define SPGEMM_FLENGTH_HASH_H

#include "sparse_format.h"
#include "spgemm_cluster.h"
#include "memopt.h"
#include "thread.h"
#include "general_config.h"
#include <omp.h>
#include <algorithm>

/**
 * @brief Fixed-length Cluster-wise SpGEMM using hash table method
 *        OpenMP with load balancing using SpGEMM_BIN_FlengthCluster
 * 
 * Key differences from row-wise hash SpGEMM:
 * - Input matrix A is in CSR_FlengthCluster format (fixed-size clusters)
 * - Uses SpGEMM_BIN_FlengthCluster for cluster-level load balancing
 * - Output matrix C is in CSR_FlengthCluster format (cluster-wise)
 * - Each cluster contains cluster_sz rows, results are computed per cluster
 */

/**
 * @brief Symbolic phase: compute structure of C_cluster = A_cluster * B
 *        OpenMP with load balancing using SpGEMM_BIN_FlengthCluster
 *        Computes nnz per cluster (not per row)
 *        crpt should be pre-allocated (c_clusters + 1 elements)
 *        This function performs scan internally to compute nnzc
 *        Reference: hash_mult_flengthcluster.h::hash_symbolic_cluster
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param arpt A_cluster row pointer array
 * @param acol A_cluster column index array
 * @param brpt B matrix row pointer array
 * @param bcol B matrix column index array
 * @param c_clusters Number of clusters in output matrix C
 * @param c_cols Number of columns in output matrix C
 * @param crpt Output cluster pointer array (pre-allocated, length: c_clusters + 1)
 * @param c_nnzc Output total number of unique column IDs across all clusters
 * @param bin SpGEMM_BIN_FlengthCluster for load balancing
 */
template <typename IndexType, typename ValueType>
void spgemm_Flength_hash_symbolic_omp_lb(
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_clusters, IndexType c_cols,
    IndexType *crpt, IndexType &c_nnzc,
    SpGEMM_BIN_FlengthCluster<IndexType, ValueType> *bin);

/**
 * @brief Numeric phase: compute values of C_cluster = A_cluster * B using hash tables
 *        OpenMP with load balancing using SpGEMM_BIN_FlengthCluster
 *        Output is in CSR_FlengthCluster format (cluster-wise)
 *        Reference: hash_mult_flengthcluster.h::hash_numeric_cluster
 * 
 * @tparam sortOutput If true, ensures output columns are sorted
 * @tparam IndexType 
 * @tparam ValueType 
 * @param arpt A_cluster row pointer array
 * @param acol A_cluster column index array
 * @param aval A_cluster values array
 * @param brpt B matrix row pointer array
 * @param bcol B matrix column index array
 * @param bval B matrix values array
 * @param c_clusters Number of clusters in output matrix C
 * @param c_cols Number of columns in output matrix C
 * @param crpt Cluster pointer array (from symbolic phase)
 * @param ccolids Output column index array (pre-allocated, length: c_nnzc)
 * @param cvalues Output values array (pre-allocated, length: c_nnzc * cluster_sz)
 * @param bin SpGEMM_BIN_FlengthCluster for load balancing
 * @param csr_rows Number of CSR rows (for bounds checking)
 * @param cluster_sz Cluster size (number of rows per cluster)
 * @param eps Epsilon for zero-value filtering (default: 0.000001f)
 */
template <bool sortOutput, typename IndexType, typename ValueType>
void spgemm_Flength_hash_numeric_omp_lb(
    const IndexType *arpt, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_clusters, IndexType c_cols,
    const IndexType *crpt, IndexType *ccolids, ValueType *cvalues,
    SpGEMM_BIN_FlengthCluster<IndexType, ValueType> *bin,
    IndexType csr_rows, IndexType cluster_sz, const ValueType eps = 0.000001f);

#endif /* SPGEMM_FLENGTH_HASH_H */
