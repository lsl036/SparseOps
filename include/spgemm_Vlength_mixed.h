#ifndef SPGEMM_VLENGTH_MIXED_H
#define SPGEMM_VLENGTH_MIXED_H

#include "sparse_format.h"
#include "spgemm_cluster.h"
#include "memopt.h"
#include "thread.h"
#include "general_config.h"
#include "plat_config.h"
#include <omp.h>
#include <algorithm>

/**
 * @file spgemm_Vlength_mixed.h
 * @brief Mixed-accumulator Variable-length Cluster-wise SpGEMM.
 *
 * Strategy:
 *   - Symbolic phase: hash-based (reuses existing logic), additionally records
 *     per-cluster min/max column index in C (min_ccol, max_ccol).
 *   - Classification: for each cluster, decide hash vs. dense accumulator
 *     based on B-row L2-residency and C-column density.
 *   - Numeric phase: per-cluster, either
 *       (a) hash accumulator  (sparse / scattered C columns), or
 *       (b) dense offset-based accumulator (compact C columns that fit L2).
 */

// Default tuning knobs (can be overridden at compile time)
#ifndef MIXED_DENSITY_THRESHOLD
#define MIXED_DENSITY_THRESHOLD 0.15
#endif

#ifndef MIXED_L2_FRACTION
#define MIXED_L2_FRACTION 0.75
#endif

/**
 * @brief Symbolic phase for mixed-accumulator VLength SpGEMM.
 *        Same hash-symbolic as the pure-hash kernel, but additionally
 *        records min_ccol[i] and max_ccol[i] per cluster.
 *
 * @param min_ccol  Output array (pre-allocated, length c_clusters) of min C-column per cluster.
 * @param max_ccol  Output array (pre-allocated, length c_clusters) of max C-column per cluster.
 *        (remaining params identical to spgemm_Vlength_hash_symbolic_omp_lb)
 */
template <typename IndexType, typename ValueType>
void spgemm_mixed_symbolic_omp_lb(
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_clusters,
    IndexType *crpt, IndexType *crpt_val,
    IndexType &c_nnzc, IndexType &c_nnzv,
    IndexType *min_ccol, IndexType *max_ccol,
    SpGEMM_BIN_VlengthCluster<IndexType, ValueType> *bin);

/**
 * @brief Classify each cluster as hash (0) or dense (1).
 *
 * Decision rule:
 *   col_range = max_ccol[i] - min_ccol[i] + 1
 *   density   = cluster_nz[i] / col_range
 *   dense_bytes = col_range * cluster_sz[i] * sizeof(ValueType)
 *   B_access_bytes = (arpt[i+1]-arpt[i]) * B_rowdense * (sizeof(IndexType)+sizeof(ValueType))
 *
 *   use_dense  iff  (B_access_bytes + dense_bytes <= L2_budget)
 *                && (density >= MIXED_DENSITY_THRESHOLD)
 *
 * @param acc_flag  Output array (pre-allocated, length c_clusters).
 *                  0 = hash, 1 = dense.
 * @return          Number of clusters classified as dense.
 */
template <typename IndexType, typename ValueType>
IndexType classify_clusters(
    const IndexType *arpt,
    const IndexType *cluster_nz,
    const IndexType *cluster_sz,
    const IndexType *min_ccol, const IndexType *max_ccol,
    IndexType c_clusters,
    double B_rowdense,
    size_t L2_budget,
    double density_threshold,
    char *acc_flag);

/**
 * @brief Numeric phase for mixed-accumulator VLength SpGEMM.
 *        Each cluster uses either hash or dense accumulator according to acc_flag.
 */
template <bool sortOutput, typename IndexType, typename ValueType>
void spgemm_mixed_numeric_omp_lb(
    const IndexType *arpt, const IndexType *arpt_val, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_clusters, IndexType c_cols,
    const IndexType *crpt, const IndexType *crpt_val,
    IndexType *ccolids, ValueType *cvalues,
    SpGEMM_BIN_VlengthCluster<IndexType, ValueType> *bin,
    IndexType cnnz, const IndexType *cluster_sz,
    const IndexType *min_ccol, const IndexType *max_ccol,
    const char *acc_flag,
    const ValueType eps = 1e-12f);

#endif /* SPGEMM_VLENGTH_MIXED_H */
