#ifndef SPGEMM_FLENGTH_ARRAY_H
#define SPGEMM_FLENGTH_ARRAY_H

#include "sparse_format.h"
#include "spgemm_cluster.h"
#include "memopt.h"
#include "thread.h"
#include "general_config.h"
#include <omp.h>
#include <algorithm>

/**
 * @brief Fixed-length Cluster-wise SpGEMM using sorted array method (HSMU-SpGEMM inspired)
 *        OpenMP with load balancing using SpGEMM_BIN_FlengthCluster
 * 
 * Key differences from hash-based cluster SpGEMM:
 * 1. Uses sorted arrays instead of hash tables (no hash collisions)
 * 2. Array size = exact cluster_nz (no 2^N padding, better memory efficiency)
 * 3. Natural sorting during symbolic phase (no extra sort step needed)
 * 4. Binary search for O(log n) lookup in numeric phase
 * 
 * Key differences from row-wise array SpGEMM:
 * 1. Works with clusters instead of individual rows
 * 2. Each cluster contains cluster_sz rows
 * 3. Each column in output has cluster_sz values (one per row in cluster)
 */

/**
 * @brief Optimized symbolic phase: generate and sort Ccolids (HSMU-SpGEMM inspired)
 *        OpenMP with load balancing using SpGEMM_BIN_FlengthCluster
 *        This version generates and sorts column indices during symbolic phase,
 *        allowing numeric phase to use binary search instead of insertion.
 * 
 * Implementation strategy (optimized to reduce traversal, inspired by HSMU-SpGEMM):
 * 1. Single pass: Collect unique columns per cluster and store in per-cluster buffers
 * 2. Scan: Compute cluster offsets (crpt)
 * 3. Write: Copy stored columns to ccolids (already sorted from insert_if_not_exists)
 * 
 * Key optimization: Store columns during collection to avoid second traversal
 * Note: insert_if_not_exists keeps array sorted, so we can directly copy after scan
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param A_cluster Input matrix A in CSR_FlengthCluster format
 * @param brpt B matrix row pointer array
 * @param bcol B matrix column index array
 * @param c_clusters Number of clusters in output matrix C (equals A_cluster.rows)
 * @param c_cols Number of columns in output matrix C (equals B.num_cols)
 * @param crpt Output cluster pointer array (pre-allocated, length: c_clusters + 1)
 * @param ccolids Output column index array (will be allocated and filled with sorted columns)
 * @param c_nnzc Output total number of unique column IDs across all clusters
 * @param bin SpGEMM_BIN_FlengthCluster for load balancing
 */
template <typename IndexType, typename ValueType>
void spgemm_Flength_array_symbolic_new(
    const CSR_FlengthCluster<IndexType, ValueType> &A_cluster,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_clusters, IndexType c_cols,
    IndexType *crpt, IndexType *&ccolids, IndexType &c_nnzc,
    SpGEMM_BIN_FlengthCluster<IndexType, ValueType> *bin);

/**
 * @brief Optimized numeric phase: find position and accumulate (HSMU-SpGEMM inspired)
 *        Uses pre-sorted Ccolids from symbolic phase, eliminating insertion operations
 *        OpenMP with load balancing using SpGEMM_BIN_FlengthCluster
 * 
 * Key optimization:
 * - Ccolids is already sorted from symbolic phase
 * - Use binary search to find position in pre-sorted array
 * - Direct accumulation to cvalues (no insertion, no temporary arrays)
 * - Better performance: O(log n) lookup + O(1) accumulate vs O(n) insertion
 * - Each column has cluster_sz values (one per row in cluster)
 * 
 * @tparam sortOutput Ignored (ccolids is already sorted from symbolic phase)
 * @tparam IndexType 
 * @tparam ValueType 
 * @param A_cluster Input matrix A in CSR_FlengthCluster format
 * @param brpt B matrix row pointer array
 * @param bcol B matrix column index array
 * @param bval B matrix values array
 * @param c_clusters Number of clusters in output matrix C
 * @param c_cols Number of columns in output matrix C
 * @param crpt Cluster pointer array (from symbolic phase)
 * @param ccolids Pre-sorted column index array (from spgemm_Flength_array_symbolic_new)
 * @param cvalues Output values array (pre-allocated, length: c_nnzc * cluster_sz)
 *                Value layout: cvalues[(cluster_start + col_pos) * cluster_sz + row_in_cluster]
 * @param cluster_sz Cluster size (number of rows per cluster)
 * @param bin SpGEMM_BIN_FlengthCluster for load balancing
 */
template <bool sortOutput, typename IndexType, typename ValueType>
void spgemm_Flength_array_numeric_new(
    const CSR_FlengthCluster<IndexType, ValueType> &A_cluster,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_clusters, IndexType c_cols,
    const IndexType *crpt, const IndexType *ccolids, ValueType *cvalues,
    IndexType cluster_sz,
    SpGEMM_BIN_FlengthCluster<IndexType, ValueType> *bin);

#endif /* SPGEMM_FLENGTH_ARRAY_H */
