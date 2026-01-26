/**
 * @file spgemm_Vlength_hash.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Variable-length Cluster-wise SpGEMM implementation using hash table method
 * @version 0.1
 * @date 2026
 */

#include "../include/spgemm_Vlength_hash.h"
#include "../include/spgemm_utility.h"
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief After calculating on each hash table, sort them in ascending order if necessary, 
 *        and then store them as output matrix in variable-length cluster format
 *        Reference: hash_mult_vlengthcluster.h::sort_and_store_table2mat_vlcluster_V1
 *        Values are stored in variable-length cluster format: 
 *        values[rowptr_val[i] + (j - rowptr[i]) * cluster_sz[i] + l]
 *        Note: Reference implementation filters out entries where all values in the cluster are zero
 */
template <bool sortOutput, typename IndexType, typename ValueType>
inline void sort_and_store_table2mat_vlcluster(IndexType *ht_check, ValueType *ht_value, 
                                               IndexType *colids, ValueType *values, 
                                               IndexType nz, IndexType ht_size, 
                                               IndexType cluster_sz,
                                               const ValueType eps = static_cast<ValueType>(1e-12))
{
    IndexType index = 0;
    IndexType val_idx, ht_idx;
    // Sort elements in ascending order if necessary, and store them as output matrix
    if (sortOutput) {
        // TODO: Implement sorting if needed (currently disabled in reference)
        // For now, just store without sorting (matching reference implementation)
        for (IndexType j = 0; j < ht_size; ++j) {
            if (ht_check[j] != -1) {
                // Check if all values in this cluster are zero (matching reference implementation)
                ht_idx = j * cluster_sz;
                bool all_zero = true;
                for (IndexType l = 0; l < cluster_sz; l++) {
                    if (std::abs(ht_value[ht_idx + l]) >= eps) {
                        all_zero = false;
                        break;
                    }
                }
                
                // Only store if not all zero (matching reference implementation)
                if (!all_zero) {
                    colids[index] = ht_check[j];
                    val_idx = index * cluster_sz;  // Matching reference: val_idx = (index * cluster_sz)
                    for (IndexType l = 0; l < cluster_sz; l++) {
                        values[val_idx + l] = ht_value[ht_idx + l];
                    }
                    index++;
                }
            }
        }
    }
    else {
        // Store the results in variable-length cluster format
        
        for (IndexType j = 0; j < ht_size; ++j) {
            if (ht_check[j] != -1) {
                // Check if all values in this cluster are zero (matching reference implementation)
                // Reference: sort_and_store_table2mat_vlcluster_V1 filters zero entries
                ht_idx = j * cluster_sz;
                // bool all_zero = true;
                // for (IndexType l = 0; l < cluster_sz; l++) {
                //     if (std::abs(ht_value[ht_idx + l]) >= eps) {
                //         all_zero = false;
                //         break;
                //     }
                // }
                
                // Only store if not all zero (matching reference implementation)
                // if (!all_zero) {
                    colids[index] = ht_check[j];
                    val_idx = index * cluster_sz;
                    for (IndexType l = 0; l < cluster_sz; l++) {
                        values[val_idx + l] = ht_value[ht_idx + l];
                    }
                    index++;
                // }
            }
        }
    }
}

// ============================================================================
// Symbolic Phase Implementation
// ============================================================================

template <typename IndexType, typename ValueType>
void spgemm_Vlength_hash_symbolic_omp_lb(
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_clusters,
    IndexType *crpt, IndexType *crpt_val,
    IndexType &c_nnzc, IndexType &c_nnzv,
    SpGEMM_BIN_VlengthCluster<IndexType, ValueType> *bin)
{   
    // Symbolic phase: count unique column IDs per cluster (matching reference hash_symbolic_kernel_vlcluster)
    #pragma omp parallel num_threads(bin->allocated_thread_num)
    {
        int tid = Le_get_thread_id();
        IndexType start_cluster = bin->clusters_offset[tid];
        IndexType end_cluster = bin->clusters_offset[tid + 1];
        
        IndexType *check = bin->local_hash_table_id[tid];
        IndexType t_acol, key, hash;

        // Process each cluster assigned to this thread
        for (IndexType cluster_id = start_cluster; cluster_id < end_cluster; ++cluster_id) {
            IndexType bid = bin->bin_id_cluster[cluster_id];
            IndexType nz = 0;
            
            if (bid > 0) {
                // Calculate hash table size for this cluster based on bin_id
                IndexType ht_size = MIN_HT_S << (bid - 1);
                
                // Initialize hash table
                for (IndexType j = 0; j < ht_size; ++j) {
                    check[j] = -1;
                }
                
                // Get cluster's column range
                IndexType col_start = arpt[cluster_id];
                IndexType col_end = arpt[cluster_id + 1];
                
                // For each column in this cluster, collect unique column IDs from B
                for (IndexType j = col_start; j < col_end; ++j) {
                    t_acol = acol[j];
                    // Multiply with B[t_acol, :]
                    for (IndexType k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        key = bcol[k];
                        hash = (key * HASH_SCAL) & (ht_size - 1);
                        while (1) {  // Loop for hash probing
                            if (check[hash] == key) {  // if the key is already inserted, it's ok
                                break;
                            }
                            else if (check[hash] == -1) {  // if the key has not been inserted yet, then it's added.
                                check[hash] = key;
                                nz++;
                                break;
                            }
                            else {  // linear probing: check next entry
                                hash = (hash + 1) & (ht_size - 1);  // hash = (hash + 1) % ht_size
                            }
                        }
                    }
                }
            }
            // Update cluster_nz by cluster (not by row)
            // Previously it was set to flops of cluster, now it's updated to nnz
            bin->cluster_nz[cluster_id] = nz;
        }
    }
    
    // Set cluster pointer of matrix C using weighted scan (matching reference hash_symbolic_vlcluster)
    // Reference: scan<IT>(bin.cluster_nz, crpt, crpt_val, bin.cluster_sz, nrow + 1)
    // This computes:
    //   crpt[i+1] = crpt[i] + cluster_nz[i] (regular prefix sum)
    //   crpt_val[i+1] = crpt_val[i] + (cluster_nz[i] * cluster_sz[i]) (weighted prefix sum)
    scan<IndexType>(bin->cluster_nz, crpt, crpt_val, bin->cluster_sz, c_clusters + 1, bin->allocated_thread_num);
    c_nnzc = crpt[c_clusters];
    c_nnzv = crpt_val[c_clusters];
}

// ============================================================================
// Numeric Phase Implementation
// ============================================================================

template <bool sortOutput, typename IndexType, typename ValueType>
void spgemm_Vlength_hash_numeric_omp_lb(
    const IndexType *arpt, const IndexType *arpt_val, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_clusters, IndexType c_cols,
    const IndexType *crpt, const IndexType *crpt_val,
    IndexType *ccolids, ValueType *cvalues,
    SpGEMM_BIN_VlengthCluster<IndexType, ValueType> *bin,
    IndexType cnnz, const IndexType *cluster_sz, const ValueType eps)
{
    // Numeric phase (matching reference hash_numeric_vlcluster_V1)
    #pragma omp parallel num_threads(bin->allocated_thread_num)
    {
        int tid = Le_get_thread_id();
        IndexType start_cluster = bin->clusters_offset[tid];
        IndexType end_cluster = bin->clusters_offset[tid + 1];
        
        IndexType *ht_check = bin->local_hash_table_id[tid];
        ValueType *ht_value = bin->local_hash_table_val[tid];
        
        IndexType t_acol;
        ValueType t_aval, t_val;

        // Process each cluster assigned to this thread
        for (IndexType cluster_id = start_cluster; cluster_id < end_cluster; ++cluster_id) {
            IndexType bid = bin->bin_id_cluster[cluster_id];
            
            if (bid > 0) {
                IndexType csz = cluster_sz[cluster_id];
                IndexType offset = crpt[cluster_id];
                IndexType cval_offset = crpt_val[cluster_id];
                IndexType aval_offset = arpt_val[cluster_id];
                IndexType ht_size = MIN_HT_N << (bid - 1);
                
                // Initialize hash table
                for (IndexType j = 0; j < ht_size; ++j) {
                    ht_check[j] = -1;
                }
                // Initialize hash table values (variable-length cluster format: ht_value[hash * cluster_sz[i] + l])
                std::memset(ht_value, 0, ht_size * csz * sizeof(ValueType));
                
                // Get cluster's column range
                IndexType col_start = arpt[cluster_id];
                IndexType col_end = arpt[cluster_id + 1];
                
                // Pre-compute temporary variables for better performance (matching reference)
                IndexType tmp1, tmp2;
                
                // For each column in this cluster
                for (IndexType j = col_start; j < col_end; ++j) {
                    t_acol = acol[j];
                    // Pre-compute base index for A_cluster.values: rowptr_val[i] + (j - rowptr[i]) * cluster_sz[i]
                    tmp1 = aval_offset + ((j - col_start) * csz);
                    
                    // Multiply with B[t_acol, :]
                    for (IndexType k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        IndexType key = bcol[k];
                        IndexType hash = (key * HASH_SCAL) & (ht_size - 1);
                        while (1) {  // Loop for hash probing
                            tmp2 = hash * csz;
                            if (ht_check[hash] == key) {  // key is already inserted
                                // Loop over all rows in the cluster
                                // Matching reference: tmp2 = (hash * csz); then tmp2++ in loop
                                for (IndexType l = 0; l < csz; l++) {
                                    // Value from A: values[rowptr_val[i] + (j - rowptr[i]) * cluster_sz[i] + l]
                                    t_aval = aval[tmp1 + l];
                                    t_val = t_aval * bval[k];
                                    // 不进行 阈值判断，只在转化到 csr 格式时判断阈值
                                    ht_value[tmp2] += t_val;
                                    tmp2++;
                                }
                                break;
                            }
                            else if (ht_check[hash] == -1) {  // insert new entry
                                ht_check[hash] = key;
                                // Loop over all rows in the cluster
                                // Matching reference: tmp2 = (hash * csz); then tmp2++ in loop
                                for (IndexType l = 0; l < csz; l++) {
                                    // Value from A: values[rowptr_val[i] + (j - rowptr[i]) * cluster_sz[i] + l]
                                    t_aval = aval[tmp1 + l];
                                    t_val = t_aval * bval[k];
                                    ht_value[tmp2] = t_val;
                                    tmp2++;
                                }
                                break;
                            }
                            else {
                                hash = (hash + 1) & (ht_size - 1);  // (hash + 1) % ht_size
                            }
                        }
                    }
                }
                
                // copy results from ht to the C_csr
                sort_and_store_table2mat_vlcluster<sortOutput, IndexType, ValueType>(
                    ht_check, ht_value,
                    ccolids + offset, cvalues + cval_offset,
                    (crpt[cluster_id + 1] - offset), ht_size, csz, eps);
            }
        }
    }
}

// Explicit template instantiations (only int64_t for IndexType)
#include <cstdint>

template void spgemm_Vlength_hash_symbolic_omp_lb<int64_t, float>(
    const int64_t*, const int64_t*, const int64_t*, const int64_t*, 
    int64_t, int64_t*, int64_t*, int64_t&, int64_t&, 
    SpGEMM_BIN_VlengthCluster<int64_t, float>*);
template void spgemm_Vlength_hash_symbolic_omp_lb<int64_t, double>(
    const int64_t*, const int64_t*, const int64_t*, const int64_t*, 
    int64_t, int64_t*, int64_t*, int64_t&, int64_t&, 
    SpGEMM_BIN_VlengthCluster<int64_t, double>*);

template void spgemm_Vlength_hash_numeric_omp_lb<false, int64_t, float>(
    const int64_t*, const int64_t*, const int64_t*, const float*,
    const int64_t*, const int64_t*, const float*, 
    int64_t, int64_t, const int64_t*, const int64_t*, int64_t*, float*, 
    SpGEMM_BIN_VlengthCluster<int64_t, float>*, int64_t, const int64_t*, const float);
template void spgemm_Vlength_hash_numeric_omp_lb<false, int64_t, double>(
    const int64_t*, const int64_t*, const int64_t*, const double*,
    const int64_t*, const int64_t*, const double*, 
    int64_t, int64_t, const int64_t*, const int64_t*, int64_t*, double*, 
    SpGEMM_BIN_VlengthCluster<int64_t, double>*, int64_t, const int64_t*, const double);
template void spgemm_Vlength_hash_numeric_omp_lb<true, int64_t, float>(
    const int64_t*, const int64_t*, const int64_t*, const float*,
    const int64_t*, const int64_t*, const float*, 
    int64_t, int64_t, const int64_t*, const int64_t*, int64_t*, float*, 
    SpGEMM_BIN_VlengthCluster<int64_t, float>*, int64_t, const int64_t*, const float);
template void spgemm_Vlength_hash_numeric_omp_lb<true, int64_t, double>(
    const int64_t*, const int64_t*, const int64_t*, const double*,
    const int64_t*, const int64_t*, const double*, 
    int64_t, int64_t, const int64_t*, const int64_t*, int64_t*, double*, 
    SpGEMM_BIN_VlengthCluster<int64_t, double>*, int64_t, const int64_t*, const double);
