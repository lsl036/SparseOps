/**
 * @file spgemm_Vlength_mixed.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Mixed-accumulator Variable-length Cluster-wise SpGEMM.
 *        Each cluster is classified as hash or dense based on
 *        B-row L2-residency and C-column density.
 * @version 0.1
 * @date 2026
 */

#include "../include/spgemm_Vlength_mixed.h"
#include "../include/spgemm_Vlength_hash.h"
#include "../include/spgemm_utility.h"
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

// ============================================================================
// Helper: store dense buffer to output (analogous to sort_and_store_table2mat)
// ============================================================================

template <bool sortOutput, typename IndexType, typename ValueType>
static inline void store_dense_to_output(
    const ValueType *dense_val,
    IndexType min_col, IndexType col_range, IndexType csz,
    IndexType *out_colids, ValueType *out_values,
    IndexType expected_nz,
    const ValueType eps = static_cast<ValueType>(1e-12))
{
    IndexType index = 0;
    if (sortOutput) {
        for (IndexType c = 0; c < col_range; ++c) {
            const ValueType *slot = dense_val + static_cast<size_t>(c) * csz;
            bool all_zero = true;
            for (IndexType l = 0; l < csz; ++l) {
                if (std::abs(slot[l]) >= eps) { all_zero = false; break; }
            }
            if (!all_zero) {
                out_colids[index] = min_col + c;
                IndexType val_base = index * csz;
                for (IndexType l = 0; l < csz; ++l)
                    out_values[val_base + l] = slot[l];
                index++;
            }
        }
    } else {
        for (IndexType c = 0; c < col_range; ++c) {
            const ValueType *slot = dense_val + static_cast<size_t>(c) * csz;
            bool all_zero = true;
            for (IndexType l = 0; l < csz; ++l) {
                if (std::abs(slot[l]) >= eps) { all_zero = false; break; }
            }
            if (!all_zero) {
                out_colids[index] = min_col + c;
                IndexType val_base = index * csz;
                for (IndexType l = 0; l < csz; ++l)
                    out_values[val_base + l] = slot[l];
                index++;
            }
        }
    }
}

// ============================================================================
// Symbolic Phase: hash-based + record min/max C-column per cluster
// ============================================================================

template <typename IndexType, typename ValueType>
void spgemm_mixed_symbolic_omp_lb(
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_clusters,
    IndexType *crpt, IndexType *crpt_val,
    IndexType &c_nnzc, IndexType &c_nnzv,
    IndexType *min_ccol, IndexType *max_ccol,
    SpGEMM_BIN_VlengthCluster<IndexType, ValueType> *bin)
{
    #pragma omp parallel num_threads(bin->allocated_thread_num)
    {
        int tid = Le_get_thread_id();
        IndexType start_cluster = bin->clusters_offset[tid];
        IndexType end_cluster = bin->clusters_offset[tid + 1];

        IndexType *check = bin->local_hash_table_id[tid];
        IndexType t_acol, key, hash;

        for (IndexType cluster_id = start_cluster; cluster_id < end_cluster; ++cluster_id) {
            IndexType bid = bin->bin_id_cluster[cluster_id];
            IndexType nz = 0;
            IndexType cmin = std::numeric_limits<IndexType>::max();
            IndexType cmax = 0;

            if (bid > 0) {
                IndexType ht_size = MIN_HT_S << (bid - 1);

                for (IndexType j = 0; j < ht_size; ++j)
                    check[j] = -1;

                IndexType col_start = arpt[cluster_id];
                IndexType col_end = arpt[cluster_id + 1];

                for (IndexType j = col_start; j < col_end; ++j) {
                    t_acol = acol[j];
                    for (IndexType k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        key = bcol[k];
                        hash = (key * HASH_SCAL) & (ht_size - 1);
                        while (1) {
                            if (check[hash] == key) {
                                break;
                            } else if (check[hash] == -1) {
                                check[hash] = key;
                                nz++;
                                if (key < cmin) cmin = key;
                                if (key > cmax) cmax = key;
                                break;
                            } else {
                                hash = (hash + 1) & (ht_size - 1);
                            }
                        }
                    }
                }
            }
            bin->cluster_nz[cluster_id] = nz;
            min_ccol[cluster_id] = (nz > 0) ? cmin : 0;
            max_ccol[cluster_id] = (nz > 0) ? cmax : 0;
        }
    }

    scan<IndexType>(bin->cluster_nz, crpt, crpt_val, bin->cluster_sz,
                    c_clusters + 1, bin->allocated_thread_num);
    c_nnzc = crpt[c_clusters];
    c_nnzv = crpt_val[c_clusters];
}

// ============================================================================
// Classification: decide hash (0) vs. dense (1) per cluster
// ============================================================================

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
    char *acc_flag)
{
    IndexType dense_count = 0;
    const size_t bytes_per_B_entry = sizeof(IndexType) + sizeof(ValueType);

    #pragma omp parallel for reduction(+:dense_count)
    for (IndexType i = 0; i < c_clusters; ++i) {
        acc_flag[i] = 0;
        IndexType nz = cluster_nz[i];
        if (nz == 0) continue;

        IndexType col_range = max_ccol[i] - min_ccol[i] + 1;
        double density = static_cast<double>(nz) / static_cast<double>(col_range);

        IndexType num_acols = arpt[i + 1] - arpt[i];
        size_t B_access_bytes = static_cast<size_t>(
            static_cast<double>(num_acols) * B_rowdense * static_cast<double>(bytes_per_B_entry));
        size_t dense_bytes = static_cast<size_t>(col_range) * static_cast<size_t>(cluster_sz[i]) * sizeof(ValueType);

        if ((B_access_bytes + dense_bytes) <= L2_budget && density >= density_threshold) {
            acc_flag[i] = 1;
            dense_count++;
        }
    }
    return dense_count;
}

// ============================================================================
// Numeric Phase: per-cluster hash or dense accumulator
// ============================================================================

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
    const ValueType eps)
{
    #pragma omp parallel num_threads(bin->allocated_thread_num)
    {
        int tid = Le_get_thread_id();
        IndexType start_cluster = bin->clusters_offset[tid];
        IndexType end_cluster = bin->clusters_offset[tid + 1];

        IndexType *ht_check = bin->local_hash_table_id[tid];
        ValueType *ht_value = bin->local_hash_table_val[tid];
        ValueType *dense_buf = (bin->local_dense_buf && bin->local_dense_buf[tid]) ? bin->local_dense_buf[tid] : nullptr;

        IndexType t_acol;
        ValueType t_aval, t_val;

        for (IndexType cluster_id = start_cluster; cluster_id < end_cluster; ++cluster_id) {
            IndexType bid = bin->bin_id_cluster[cluster_id];
            if (bid == 0) continue;

            IndexType csz = cluster_sz[cluster_id];
            IndexType offset = crpt[cluster_id];
            IndexType cval_offset = crpt_val[cluster_id];
            IndexType aval_offset = arpt_val[cluster_id];
            IndexType col_start = arpt[cluster_id];
            IndexType col_end = arpt[cluster_id + 1];

            if (acc_flag[cluster_id] == 1) {
                /* =========== Dense (offset-based) accumulator =========== */
                IndexType mc = min_ccol[cluster_id];
                IndexType col_range = max_ccol[cluster_id] - mc + 1;
                size_t buf_sz = static_cast<size_t>(col_range) * csz;
                std::memset(dense_buf, 0, buf_sz * sizeof(ValueType));

                for (IndexType j = col_start; j < col_end; ++j) {
                    t_acol = acol[j];
                    IndexType tmp1 = aval_offset + (j - col_start) * csz;

                    for (IndexType k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        IndexType key = bcol[k];
                        IndexType slot = (key - mc) * csz;
                        ValueType bv = bval[k];
                        for (IndexType l = 0; l < csz; ++l) {
                            dense_buf[slot + l] += aval[tmp1 + l] * bv;
                        }
                    }
                }

                store_dense_to_output<sortOutput, IndexType, ValueType>(
                    dense_buf, mc, col_range, csz,
                    ccolids + offset, cvalues + cval_offset,
                    crpt[cluster_id + 1] - offset, eps);

            } else {
                /* =========== Hash accumulator (same as existing) =========== */
                IndexType ht_size = MIN_HT_N << (bid - 1);

                for (IndexType j = 0; j < ht_size; ++j)
                    ht_check[j] = -1;
                std::memset(ht_value, 0, static_cast<size_t>(ht_size) * csz * sizeof(ValueType));

                IndexType tmp1, tmp2;
                for (IndexType j = col_start; j < col_end; ++j) {
                    t_acol = acol[j];
                    tmp1 = aval_offset + (j - col_start) * csz;

                    for (IndexType k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        IndexType key = bcol[k];
                        IndexType hash = (key * HASH_SCAL) & (ht_size - 1);
                        while (1) {
                            tmp2 = hash * csz;
                            if (ht_check[hash] == key) {
                                for (IndexType l = 0; l < csz; ++l) {
                                    ht_value[tmp2 + l] += aval[tmp1 + l] * bval[k];
                                }
                                break;
                            } else if (ht_check[hash] == -1) {
                                ht_check[hash] = key;
                                for (IndexType l = 0; l < csz; ++l) {
                                    ht_value[tmp2 + l] = aval[tmp1 + l] * bval[k];
                                }
                                break;
                            } else {
                                hash = (hash + 1) & (ht_size - 1);
                            }
                        }
                    }
                }

                sort_and_store_table2mat_vlcluster<sortOutput, IndexType, ValueType>(
                    ht_check, ht_value,
                    ccolids + offset, cvalues + cval_offset,
                    (crpt[cluster_id + 1] - offset), ht_size, csz, eps);
            }
        }
    }
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

#include <cstdint>

template void spgemm_mixed_symbolic_omp_lb<int64_t, float>(
    const int64_t*, const int64_t*, const int64_t*, const int64_t*,
    int64_t, int64_t*, int64_t*, int64_t&, int64_t&,
    int64_t*, int64_t*, SpGEMM_BIN_VlengthCluster<int64_t, float>*);
template void spgemm_mixed_symbolic_omp_lb<int64_t, double>(
    const int64_t*, const int64_t*, const int64_t*, const int64_t*,
    int64_t, int64_t*, int64_t*, int64_t&, int64_t&,
    int64_t*, int64_t*, SpGEMM_BIN_VlengthCluster<int64_t, double>*);

template int64_t classify_clusters<int64_t, float>(
    const int64_t*, const int64_t*, const int64_t*,
    const int64_t*, const int64_t*, int64_t,
    double, size_t, double, char*);
template int64_t classify_clusters<int64_t, double>(
    const int64_t*, const int64_t*, const int64_t*,
    const int64_t*, const int64_t*, int64_t,
    double, size_t, double, char*);

template void spgemm_mixed_numeric_omp_lb<false, int64_t, float>(
    const int64_t*, const int64_t*, const int64_t*, const float*,
    const int64_t*, const int64_t*, const float*,
    int64_t, int64_t, const int64_t*, const int64_t*, int64_t*, float*,
    SpGEMM_BIN_VlengthCluster<int64_t, float>*, int64_t, const int64_t*,
    const int64_t*, const int64_t*, const char*, const float);
template void spgemm_mixed_numeric_omp_lb<false, int64_t, double>(
    const int64_t*, const int64_t*, const int64_t*, const double*,
    const int64_t*, const int64_t*, const double*,
    int64_t, int64_t, const int64_t*, const int64_t*, int64_t*, double*,
    SpGEMM_BIN_VlengthCluster<int64_t, double>*, int64_t, const int64_t*,
    const int64_t*, const int64_t*, const char*, const double);
template void spgemm_mixed_numeric_omp_lb<true, int64_t, float>(
    const int64_t*, const int64_t*, const int64_t*, const float*,
    const int64_t*, const int64_t*, const float*,
    int64_t, int64_t, const int64_t*, const int64_t*, int64_t*, float*,
    SpGEMM_BIN_VlengthCluster<int64_t, float>*, int64_t, const int64_t*,
    const int64_t*, const int64_t*, const char*, const float);
template void spgemm_mixed_numeric_omp_lb<true, int64_t, double>(
    const int64_t*, const int64_t*, const int64_t*, const double*,
    const int64_t*, const int64_t*, const double*,
    int64_t, int64_t, const int64_t*, const int64_t*, int64_t*, double*,
    SpGEMM_BIN_VlengthCluster<int64_t, double>*, int64_t, const int64_t*,
    const int64_t*, const int64_t*, const char*, const double);
