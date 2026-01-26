/**
 * @file spgemm_Vlength_array.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Variable-length Cluster-wise SpGEMM implementation using sorted array method
 *        OpenMP with load balancing using SpGEMM_BIN_VlengthCluster
 * @version 0.1
 * @date 2026
 */

#include "../include/spgemm_Vlength_array.h"
#include "../include/spgemm_array.h"  // For binary_search_find and insert_if_not_exists
#include "../include/spgemm_utility.h"
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

// ============================================================================
// Symbolic Phase Implementation
// ============================================================================

/**
 * @brief Optimized symbolic phase: generate and sort Ccolids (HSMU-SpGEMM inspired)
 *        OpenMP with load balancing using SpGEMM_BIN_VlengthCluster
 *        This version generates and sorts column indices during symbolic phase,
 *        allowing numeric phase to use binary search instead of insertion.
 */
template <typename IndexType, typename ValueType>
void spgemm_Vlength_array_symbolic_new(
    const CSR_VlengthCluster<IndexType, ValueType> &A_cluster,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_clusters, IndexType c_cols,
    IndexType *crpt, IndexType *crpt_val,
    IndexType *&ccolids, IndexType &c_nnzc, IndexType &c_nnzv,
    SpGEMM_BIN_VlengthCluster<IndexType, ValueType> *bin)
{
    // Allocate per-cluster buffers to store columns (we'll need them after scan)
    // Use std::vector for each cluster to store columns during collection
    std::vector<std::vector<IndexType>> cluster_cols(c_clusters);
    
    // Phase 1: Collect unique columns and store in per-cluster buffers (single pass)
    #pragma omp parallel num_threads(bin->allocated_thread_num)
    {
        int tid = Le_get_thread_id();
        IndexType start_cluster = bin->clusters_offset[tid];
        IndexType end_cluster = bin->clusters_offset[tid + 1];
        
        // Find max array size needed for this thread's clusters
        IndexType max_nz = 0;
        for (IndexType i = start_cluster; i < end_cluster; ++i) {
            if (bin->cluster_nz[i] > max_nz) {
                max_nz = bin->cluster_nz[i];
            }
        }
        
        // Cap at c_cols
        if (max_nz > c_cols) max_nz = c_cols;
        
        // Allocate temporary array for unique columns (per cluster, reused)
        IndexType *temp_cols = new_array<IndexType>(max_nz);
        
        for (IndexType cluster_id = start_cluster; cluster_id < end_cluster; ++cluster_id) {
            IndexType temp_size = 0;
            
            // Get cluster's column range in A_cluster
            IndexType col_start = A_cluster.rowptr[cluster_id];
            IndexType col_end = A_cluster.rowptr[cluster_id + 1];
            
            // Collect unique columns for this cluster
            // For each column in this cluster, traverse B[t_acol, :] to collect unique column IDs
            for (IndexType j = col_start; j < col_end; ++j) {
                IndexType t_acol = A_cluster.colids[j];
                // Traverse B[t_acol, :] to collect column indices
                for (IndexType k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                    IndexType key = bcol[k];
                    if (insert_if_not_exists(temp_cols, temp_size, max_nz, key)) {
                        // Key inserted, temp_cols remains sorted
                    }
                }
            }
            
            // Update cluster_nz (number of unique columns for this cluster)
            bin->cluster_nz[cluster_id] = temp_size;
            
            // Store columns for this cluster (temp_cols is already sorted from insert_if_not_exists)
            cluster_cols[cluster_id].resize(temp_size);
            for (IndexType j = 0; j < temp_size; ++j) {
                cluster_cols[cluster_id][j] = temp_cols[j];
            }
        }
        
        delete_array(temp_cols);
    }
    
    // Weighted scan: Compute cluster offsets (crpt) and value offsets (crpt_val)
    // crpt: regular prefix sum for column indices
    // crpt_val: weighted prefix sum considering cluster_sz[i] for each cluster
    // Matching reference: scan<IT>(bin.cluster_nz, crpt, crpt_val, bin.cluster_sz, nrow + 1)
    scan<IndexType>(bin->cluster_nz, crpt, crpt_val, bin->cluster_sz, c_clusters + 1, bin->allocated_thread_num);
    c_nnzc = crpt[c_clusters];
    c_nnzv = crpt_val[c_clusters];
    
    // Allocate ccolids now that we know the exact size
    ccolids = new_array<IndexType>(c_nnzc);
    
    // Phase 2: Write stored columns to ccolids (already sorted, no need to sort again)
    #pragma omp parallel num_threads(bin->allocated_thread_num)
    {
        int tid = Le_get_thread_id();
        IndexType start_cluster = bin->clusters_offset[tid];
        IndexType end_cluster = bin->clusters_offset[tid + 1];
        
        for (IndexType cluster_id = start_cluster; cluster_id < end_cluster; ++cluster_id) {
            IndexType cluster_start = crpt[cluster_id];
            IndexType cluster_nnz = bin->cluster_nz[cluster_id];
            
            // Copy stored columns to output ccolids (already sorted from insert_if_not_exists)
            for (IndexType j = 0; j < cluster_nnz; ++j) {
                ccolids[cluster_start + j] = cluster_cols[cluster_id][j];
            }
        }
    }
}

// ============================================================================
// Numeric Phase Implementation
// ============================================================================

/**
 * @brief Optimized numeric phase: find position and accumulate (HSMU-SpGEMM inspired)
 *        Uses pre-sorted Ccolids from symbolic phase, eliminating insertion operations
 */
template <bool sortOutput, typename IndexType, typename ValueType>
void spgemm_Vlength_array_numeric_new(
    const CSR_VlengthCluster<IndexType, ValueType> &A_cluster,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_clusters, IndexType c_cols,
    const IndexType *crpt, const IndexType *crpt_val,
    const IndexType *ccolids, ValueType *cvalues,
    const IndexType *cluster_sz,
    SpGEMM_BIN_VlengthCluster<IndexType, ValueType> *bin,
    const ValueType eps)
{
    // Numeric phase: compute values using pre-sorted Ccolids
    // Ccolids is already sorted from spgemm_Vlength_array_symbolic_new
    // We only need to find position and accumulate (no insertion)
    
    #pragma omp parallel num_threads(bin->allocated_thread_num)
    {
        int tid = Le_get_thread_id();
        IndexType start_cluster = bin->clusters_offset[tid];
        IndexType end_cluster = bin->clusters_offset[tid + 1];
        
        // Initialize cvalues to 0 for this thread's clusters
        // Use memset for better performance (faster than loop for large arrays)
        for (IndexType cluster_id = start_cluster; cluster_id < end_cluster; ++cluster_id) {
            IndexType cluster_start = crpt[cluster_id];
            IndexType cluster_nnz = crpt[cluster_id + 1] - cluster_start;
            IndexType csz = cluster_sz[cluster_id];
            if (cluster_nnz > 0) {
                // Initialize all values for this cluster (cluster_nnz * csz values)
                // Values start at rowptr_val[cluster_id]
                IndexType val_start = crpt_val[cluster_id];
                std::memset(cvalues + val_start, 0, cluster_nnz * csz * sizeof(ValueType));
            }
        }
        
        // Accumulate intermediate products
        // Declare variables outside inner loops for better compiler optimization
        IndexType t_acol;
        ValueType t_aval, product;
        ValueType t_bval;  // Cache bval[k] to avoid repeated memory access
        
        for (IndexType cluster_id = start_cluster; cluster_id < end_cluster; ++cluster_id) {
            IndexType cluster_start = crpt[cluster_id];
            IndexType cluster_nnz = crpt[cluster_id + 1] - cluster_start;
            if (cluster_nnz == 0) continue;
            
            IndexType csz = cluster_sz[cluster_id];
            
            // Get pointer to this cluster's sorted column indices
            const IndexType *cluster_ccolids = ccolids + cluster_start;
            
            // Get cluster's column range in A_cluster
            IndexType a_col_start = A_cluster.rowptr[cluster_id];
            IndexType a_col_end = A_cluster.rowptr[cluster_id + 1];
            
            // Get value offset for this cluster
            IndexType val_offset = crpt_val[cluster_id];
            
            // Get A_cluster value offset for this cluster
            IndexType a_val_offset = A_cluster.rowptr_val[cluster_id];
            
            // Pre-compute CSR row offset for this cluster (cumulative sum of previous cluster sizes)
            IndexType csr_row_offset = 0;
            for (IndexType i = 0; i < cluster_id; ++i) {
                csr_row_offset += cluster_sz[i];
            }
            
            // Process each row within the cluster
            for (IndexType row_in_cluster = 0; row_in_cluster < csz; ++row_in_cluster) {
                // Calculate the actual CSR row number
                IndexType csr_row = csr_row_offset + row_in_cluster;
                
                // Skip if this row is beyond the original matrix rows (last cluster may be incomplete)
                if (csr_row >= A_cluster.csr_rows) break;
                
                // Traverse all columns in this cluster
                // Reuse col-index of A, then rows of matrix B have cache locality
                for (IndexType j = a_col_start; j < a_col_end; ++j) {
                    t_acol = A_cluster.colids[j];
                    
                    // Pre-compute base index for A_cluster.values
                    // Values for column j in cluster i: values[rowptr_val[i] + (j - rowptr[i]) * cluster_sz[i] + row_in_cluster]
                    IndexType a_val_idx = a_val_offset + ((j - a_col_start) * csz) + row_in_cluster;
                    
                    // Get the value for this row in this column
                    t_aval = A_cluster.values[a_val_idx];
                    
                    // Skip zero values (sparse matrix optimization)
                    // if (std::abs(t_aval) < eps) continue;
                    
                    // Traverse B[t_acol, :] to accumulate products
                    for (IndexType k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        IndexType target_col = bcol[k];
                        t_bval = bval[k];  // Cache bval[k] once per iteration
                        product = t_aval * t_bval;
                        
                        // Binary search to find position in pre-sorted cluster_ccolids
                        IndexType pos = binary_search_find(cluster_ccolids, cluster_nnz, target_col);
                        
                        // Accumulate if found (should always be found since Ccolids was generated from symbolic phase)
                        if (pos != -1) {
                            // Value position for variable-length cluster:
                            // cvalues[rowptr_val[i] + (col_pos - rowptr[i]) * cluster_sz[i] + row_in_cluster]
                            // where col_pos = cluster_start + pos, rowptr[i] = cluster_start
                            // Simplified: val_idx = val_offset + (pos * csz) + row_in_cluster
                            IndexType val_idx = val_offset + (pos * csz) + row_in_cluster;
                            cvalues[val_idx] += product;
                        }
                        // Note: If pos == -1, it means the column was not in symbolic phase
                        // This should not happen if symbolic and numeric phases are consistent
                    }
                }
            }
        }
    }
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

#include <cstdint>

// Symbolic phase
template void spgemm_Vlength_array_symbolic_new<int64_t, float>(
    const CSR_VlengthCluster<int64_t, float> &A_cluster,
    const int64_t *brpt, const int64_t *bcol,
    int64_t c_clusters, int64_t c_cols,
    int64_t *crpt, int64_t *crpt_val,
    int64_t *&ccolids, int64_t &c_nnzc, int64_t &c_nnzv,
    SpGEMM_BIN_VlengthCluster<int64_t, float> *bin);

template void spgemm_Vlength_array_symbolic_new<int64_t, double>(
    const CSR_VlengthCluster<int64_t, double> &A_cluster,
    const int64_t *brpt, const int64_t *bcol,
    int64_t c_clusters, int64_t c_cols,
    int64_t *crpt, int64_t *crpt_val,
    int64_t *&ccolids, int64_t &c_nnzc, int64_t &c_nnzv,
    SpGEMM_BIN_VlengthCluster<int64_t, double> *bin);

// Numeric phase
template void spgemm_Vlength_array_numeric_new<true, int64_t, float>(
    const CSR_VlengthCluster<int64_t, float> &A_cluster,
    const int64_t *brpt, const int64_t *bcol, const float *bval,
    int64_t c_clusters, int64_t c_cols,
    const int64_t *crpt, const int64_t *crpt_val,
    const int64_t *ccolids, float *cvalues,
    const int64_t *cluster_sz,
    SpGEMM_BIN_VlengthCluster<int64_t, float> *bin,
    const float eps);

template void spgemm_Vlength_array_numeric_new<false, int64_t, float>(
    const CSR_VlengthCluster<int64_t, float> &A_cluster,
    const int64_t *brpt, const int64_t *bcol, const float *bval,
    int64_t c_clusters, int64_t c_cols,
    const int64_t *crpt, const int64_t *crpt_val,
    const int64_t *ccolids, float *cvalues,
    const int64_t *cluster_sz,
    SpGEMM_BIN_VlengthCluster<int64_t, float> *bin,
    const float eps);

template void spgemm_Vlength_array_numeric_new<true, int64_t, double>(
    const CSR_VlengthCluster<int64_t, double> &A_cluster,
    const int64_t *brpt, const int64_t *bcol, const double *bval,
    int64_t c_clusters, int64_t c_cols,
    const int64_t *crpt, const int64_t *crpt_val,
    const int64_t *ccolids, double *cvalues,
    const int64_t *cluster_sz,
    SpGEMM_BIN_VlengthCluster<int64_t, double> *bin,
    const double eps);

template void spgemm_Vlength_array_numeric_new<false, int64_t, double>(
    const CSR_VlengthCluster<int64_t, double> &A_cluster,
    const int64_t *brpt, const int64_t *bcol, const double *bval,
    int64_t c_clusters, int64_t c_cols,
    const int64_t *crpt, const int64_t *crpt_val,
    const int64_t *ccolids, double *cvalues,
    const int64_t *cluster_sz,
    SpGEMM_BIN_VlengthCluster<int64_t, double> *bin,
    const double eps);
