/**
 * @file spgemm_Flength_hash.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Fixed-length Cluster-wise SpGEMM implementation using hash table method
 * @version 0.1
 * @date 2026
 */

#include "../include/spgemm_Flength_hash.h"
#include "../include/spgemm_utility.h"
#include <cstring>
#include <vector>
#include <algorithm>

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Sort function for pairs (matching reference implementation)
 */
template <typename IndexType, typename ValueType>
bool sort_less(const std::pair<IndexType, ValueType> &left, const std::pair<IndexType, ValueType> &right)
{
    return left.first < right.first;
}

/**
 * @brief After calculating on each hash table, sort them in ascending order if necessary, 
 *        and then store them as output matrix in cluster format (matching reference sort_and_store_table2mat_cluster)
 *        Values are stored in cluster format: values[(col_idx * cluster_sz) + row_in_cluster]
 */
template <bool sortOutput, typename IndexType, typename ValueType>
inline void sort_and_store_table2mat_cluster(IndexType *ht_check, ValueType *ht_value, 
                                             IndexType *colids, ValueType *values, 
                                             IndexType nz, IndexType ht_size, IndexType offset,
                                             IndexType cluster_sz)
{
    IndexType index = 0;
    
    // Sort elements in ascending order if necessary, and store them as output matrix
    if (sortOutput) {
        std::vector<std::pair<IndexType, IndexType>> p_vec(nz);  // <col-id, position-in-hashtable>
        for (IndexType j = 0; j < ht_size; ++j) { // accumulate non-zero entry from hash table
            if (ht_check[j] != -1) {
                p_vec[index++] = std::make_pair(ht_check[j], j);
            }
        }
        std::sort(p_vec.begin(), p_vec.end(), [](const std::pair<IndexType, IndexType> &a, const std::pair<IndexType, IndexType> &b) {
            return a.first < b.first;
        }); // sort only non-zero elements
        
        // Store the results in cluster format
        for (IndexType j = 0; j < index; ++j) {
            colids[j] = p_vec[j].first;
            IndexType val_idx = j * cluster_sz;
            IndexType ht_idx = p_vec[j].second * cluster_sz;
            for (IndexType l = 0; l < cluster_sz; l++) {
                values[val_idx + l] = ht_value[ht_idx + l];
            }
        }
    }
    else {
        // Store the results in cluster format
        for (IndexType j = 0; j < ht_size; ++j) {
            if (ht_check[j] != -1) {
                colids[index] = ht_check[j];
                IndexType val_idx = index * cluster_sz;
                IndexType ht_idx = j * cluster_sz;
                for (IndexType l = 0; l < cluster_sz; l++) {
                    values[val_idx + l] = ht_value[ht_idx + l];
                }
                index++;
            }
        }
    }
}

// ============================================================================
// Symbolic Phase Implementation
// ============================================================================

template <typename IndexType, typename ValueType>
void spgemm_Flength_hash_symbolic_omp_lb(
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_clusters, IndexType c_cols,
    IndexType *crpt, IndexType &c_nnzc,
    SpGEMM_BIN_FlengthCluster<IndexType, ValueType> *bin)
{   
    // Symbolic phase: count unique column IDs per cluster (matching reference hash_symbolic_kernel_cluster)
    #pragma omp parallel num_threads(bin->allocated_thread_num)
    {
        int tid = Le_get_thread_id();
        IndexType start_cluster = bin->clusters_offset[tid];
        IndexType end_cluster = bin->clusters_offset[tid + 1];
        
        IndexType *check = bin->local_hash_table_id[tid];
        IndexType t_acol, key, hash;

        // Process each cluster assigned to this thread
        for (IndexType cluster_id = start_cluster; cluster_id < end_cluster; ++cluster_id) {
            IndexType bid = bin->bin_id[cluster_id];
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
                // Note: Reference code doesn't check bounds here, assuming valid input
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
            bin->cluster_nz[cluster_id] = nz;
        }
    }
    
    // Set cluster pointer of matrix C using scan (matching reference hash_symbolic_cluster)
    scan(bin->cluster_nz, crpt, c_clusters + 1, bin->allocated_thread_num);
    c_nnzc = crpt[c_clusters];
}

// ============================================================================
// Numeric Phase Implementation
// ============================================================================

template <bool sortOutput, typename IndexType, typename ValueType>
void spgemm_Flength_hash_numeric_omp_lb(
    const IndexType *arpt, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_clusters, IndexType c_cols,
    const IndexType *crpt, IndexType *ccolids, ValueType *cvalues,
    SpGEMM_BIN_FlengthCluster<IndexType, ValueType> *bin,
    IndexType csr_rows, IndexType cluster_sz, const ValueType eps)
{
    // Numeric phase (matching reference hash_numeric_cluster)
    #pragma omp parallel num_threads(bin->allocated_thread_num)
    {
        int tid = Le_get_thread_id();
        IndexType start_cluster = bin->clusters_offset[tid];
        IndexType end_cluster = bin->clusters_offset[tid + 1];
        
        IndexType *ht_check = bin->local_hash_table_id[tid];
        ValueType *ht_value = bin->local_hash_table_val[tid];
        
        // Process each cluster assigned to this thread
        for (IndexType cluster_id = start_cluster; cluster_id < end_cluster; ++cluster_id) {
            IndexType bid = bin->bin_id[cluster_id];
            
            if (bid > 0) {
                IndexType offset = crpt[cluster_id];
                IndexType ht_size = MIN_HT_N << (bid - 1);
                
                // Initialize hash table
                for (IndexType j = 0; j < ht_size; ++j) {
                    ht_check[j] = -1;
                }
                // Initialize hash table values (cluster format: ht_value[hash * cluster_sz + row_in_cluster])
                std::memset(ht_value, 0, ht_size * cluster_sz * sizeof(ValueType));
                
                // Get cluster's column range
                IndexType col_start = arpt[cluster_id];
                IndexType col_end = arpt[cluster_id + 1];
                
                // Declare variables outside inner loops (matching reference implementation for better compiler optimization)
                IndexType t_acol;
                ValueType t_aval, t_val;
                ValueType t_bval;  // Cache bval[k] to avoid repeated memory access
                
                // For each column in this cluster
                // reuse col-index of A, then rows of matrix B have cache locality
                for (IndexType j = col_start; j < col_end; ++j) {
                    t_acol = acol[j];
                    // Pre-compute base index for A_cluster.values to reduce repeated calculations
                    IndexType a_val_base = j * cluster_sz;
                    // Multiply with B[t_acol, :]
                    for (IndexType k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        IndexType key = bcol[k];
                        t_bval = bval[k];  // Cache bval[k] once per iteration
                        IndexType hash = (key * HASH_SCAL) & (ht_size - 1);
                        while (1) {  // Loop for hash probing
                            if (ht_check[hash] == key) {  // key is already inserted
                                // Loop over all rows in the cluster
                                // Pre-compute base index for ht_value to reduce repeated calculations
                                IndexType ht_val_base = hash * cluster_sz;
                                for (IndexType l = 0; l < cluster_sz; l++) {
                                    t_aval = aval[a_val_base + l];
                                    // Zero-value check: avoid flop when (A.value[] == 0.0) (matching reference implementation)
                                    if (std::abs(t_aval) >= eps) {
                                        t_val = t_aval * t_bval;
                                        ht_value[ht_val_base + l] += t_val;
                                    }
                                }
                                break;
                            }
                            else if (ht_check[hash] == -1) {  // insert new entry
                                ht_check[hash] = key;
                                // Loop over all rows in the cluster
                                // Pre-compute base index for ht_value to reduce repeated calculations
                                IndexType ht_val_base = hash * cluster_sz;
                                for (IndexType l = 0; l < cluster_sz; l++) {
                                    t_aval = aval[a_val_base + l];
                                    t_val = t_aval * t_bval;
                                    // ht_value will be automatically initialized by 0.0 if t_val is zero
                                    ht_value[ht_val_base + l] = t_val;
                                }
                                break;
                            }
                            else {
                                hash = (hash + 1) & (ht_size - 1);  // (hash + 1) % ht_size
                            }
                        }
                    }
                }
                
                // Extract from hash table and store to output (cluster format)
                IndexType nz = crpt[cluster_id + 1] - offset;
                sort_and_store_table2mat_cluster<sortOutput, IndexType, ValueType>(
                    ht_check, ht_value,
                    ccolids + offset, cvalues + (offset * cluster_sz),
                    nz, ht_size, offset, cluster_sz);
            }
        }
    }
}

// Explicit template instantiations (only int64_t for IndexType)
#include <cstdint>

template void spgemm_Flength_hash_symbolic_omp_lb<int64_t, float>(
    const int64_t*, const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t, int64_t*, int64_t&, SpGEMM_BIN_FlengthCluster<int64_t, float>*);
template void spgemm_Flength_hash_symbolic_omp_lb<int64_t, double>(
    const int64_t*, const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t, int64_t*, int64_t&, SpGEMM_BIN_FlengthCluster<int64_t, double>*);

template void spgemm_Flength_hash_numeric_omp_lb<false, int64_t, float>(
    const int64_t*, const int64_t*, const float*, const int64_t*, const int64_t*, const float*, int64_t, int64_t, const int64_t*, int64_t*, float*, SpGEMM_BIN_FlengthCluster<int64_t, float>*, int64_t, int64_t, const float);
template void spgemm_Flength_hash_numeric_omp_lb<false, int64_t, double>(
    const int64_t*, const int64_t*, const double*, const int64_t*, const int64_t*, const double*, int64_t, int64_t, const int64_t*, int64_t*, double*, SpGEMM_BIN_FlengthCluster<int64_t, double>*, int64_t, int64_t, const double);
template void spgemm_Flength_hash_numeric_omp_lb<true, int64_t, float>(
    const int64_t*, const int64_t*, const float*, const int64_t*, const int64_t*, const float*, int64_t, int64_t, const int64_t*, int64_t*, float*, SpGEMM_BIN_FlengthCluster<int64_t, float>*, int64_t, int64_t, const float);
template void spgemm_Flength_hash_numeric_omp_lb<true, int64_t, double>(
    const int64_t*, const int64_t*, const double*, const int64_t*, const int64_t*, const double*, int64_t, int64_t, const int64_t*, int64_t*, double*, SpGEMM_BIN_FlengthCluster<int64_t, double>*, int64_t, int64_t, const double);
