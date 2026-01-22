#ifndef SPGEMM_VLENGTH_HASH_H
#define SPGEMM_VLENGTH_HASH_H

#include "sparse_format.h"
#include "spgemm_cluster.h"
#include "memopt.h"
#include "thread.h"
#include "general_config.h"
#include <omp.h>
#include <algorithm>

/**
 * @brief Variable-length Cluster-wise SpGEMM using hash table method
 *        OpenMP with load balancing using SpGEMM_BIN_VlengthCluster
 *        Reference: hash_mult_vlengthcluster.h::HashSpGEMMVLCluster
 * 
 * Key differences from Fixed-length Cluster SpGEMM:
 * - Input matrix A is in CSR_VlengthCluster format (variable-size clusters)
 * - Uses SpGEMM_BIN_VlengthCluster for cluster-level load balancing
 * - Output matrix C is in CSR_VlengthCluster format (cluster-wise)
 * - Each cluster can have different size (cluster_sz[i])
 * - Requires rowptr_val for values indexing (values are not uniformly sized)
 */

/**
 * @brief Symbolic phase: compute structure of C_cluster = A_cluster * B
 *        OpenMP with load balancing using SpGEMM_BIN_VlengthCluster
 *        Computes nnz per cluster (not per row)
 *        crpt and crpt_val should be pre-allocated (c_clusters + 1 elements each)
 *        This function performs weighted scan internally to compute nnzc and nnzv
 *        Reference: hash_mult_vlengthcluster.h::hash_symbolic_vlcluster
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param arpt A_cluster row pointer array
 * @param acol A_cluster column index array
 * @param brpt B matrix row pointer array
 * @param bcol B matrix column index array
 * @param c_clusters Number of clusters in output matrix C
 * @param c_cols Number of columns in output matrix C
 * @param crpt Output cluster pointer array for colids (pre-allocated, length: c_clusters + 1)
 * @param crpt_val Output cluster pointer array for values (pre-allocated, length: c_clusters + 1)
 * @param c_nnzc Output total number of unique column IDs across all clusters
 * @param c_nnzv Output total size of values array: sum((rowptr[i+1] - rowptr[i]) * cluster_sz[i])
 * @param bin SpGEMM_BIN_VlengthCluster for load balancing
 */
template <typename IndexType, typename ValueType>
void spgemm_Vlength_hash_symbolic_omp_lb(
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_clusters,
    IndexType *crpt, IndexType *crpt_val,
    IndexType &c_nnzc, IndexType &c_nnzv,
    SpGEMM_BIN_VlengthCluster<IndexType, ValueType> *bin);

/**
 * @brief Numeric phase: compute values of C_cluster = A_cluster * B using hash tables
 *        OpenMP with load balancing using SpGEMM_BIN_VlengthCluster
 *        Output is in CSR_VlengthCluster format (cluster-wise)
 *        Reference: hash_mult_vlengthcluster.h::hash_numeric_vlcluster_V1
 * 
 * @tparam sortOutput If true, ensures output columns are sorted
 * @tparam IndexType 
 * @tparam ValueType 
 * @param arpt A_cluster row pointer array
 * @param arpt_val A_cluster row pointer array for values
 * @param acol A_cluster column index array
 * @param aval A_cluster values array
 * @param brpt B matrix row pointer array
 * @param bcol B matrix column index array
 * @param bval B matrix values array
 * @param c_clusters Number of clusters in output matrix C
 * @param c_cols Number of columns in output matrix C
 * @param crpt Cluster pointer array for colids (from symbolic phase)
 * @param crpt_val Cluster pointer array for values (from symbolic phase)
 * @param ccolids Output column index array (pre-allocated, length: c_nnzc)
 * @param cvalues Output values array (pre-allocated, length: c_nnzv)
 * @param bin SpGEMM_BIN_VlengthCluster for load balancing
 * @param cnnz Total number of unique column IDs (for bounds checking)
 * @param cluster_sz Array of cluster sizes (length: c_clusters)
 * @param eps Epsilon for zero-value filtering (default: 0.000001f)
 */
template <bool sortOutput, typename IndexType, typename ValueType>
void spgemm_Vlength_hash_numeric_omp_lb(
    const IndexType *arpt, const IndexType *arpt_val, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_clusters, IndexType c_cols,
    const IndexType *crpt, const IndexType *crpt_val,
    IndexType *ccolids, ValueType *cvalues,
    SpGEMM_BIN_VlengthCluster<IndexType, ValueType> *bin,
    IndexType cnnz, const IndexType *cluster_sz, const ValueType eps = 0.000001f);

#endif /* SPGEMM_VLENGTH_HASH_H */
