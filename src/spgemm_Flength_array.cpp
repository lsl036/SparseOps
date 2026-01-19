/**
 * @file spgemm_Flength_array.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Fixed-length Cluster-wise SpGEMM implementation using sorted array method (HSMU-SpGEMM inspired)
 * @version 0.1
 * @date 2026
 */

#include "../include/spgemm_Flength_array.h"
#include "../include/spgemm_utility.h"
#include "../include/spgemm_array.h"  // For binary_search_find and insert_if_not_exists
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

// ============================================================================
// Symbolic Phase Implementation
// ============================================================================

/**
 * @brief Optimized symbolic phase: generate and sort Ccolids (HSMU-SpGEMM inspired)
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
 */
template <typename IndexType, typename ValueType>
void spgemm_Flength_array_symbolic_new(
    const CSR_FlengthCluster<IndexType, ValueType> &A_cluster,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_clusters, IndexType c_cols,
    IndexType *crpt, IndexType *&ccolids, IndexType &c_nnzc,
    SpGEMM_BIN_FlengthCluster<IndexType, ValueType> *bin)
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
    
    // Scan: Compute cluster offsets (crpt)
    scan(bin->cluster_nz, crpt, c_clusters + 1, bin->allocated_thread_num);
    c_nnzc = crpt[c_clusters];
    
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
void spgemm_Flength_array_numeric_new(
    const CSR_FlengthCluster<IndexType, ValueType> &A_cluster,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_clusters, IndexType c_cols,
    const IndexType *crpt, const IndexType *ccolids, ValueType *cvalues,
    IndexType cluster_sz,
    SpGEMM_BIN_FlengthCluster<IndexType, ValueType> *bin)
{
    // Numeric phase: compute values using pre-sorted Ccolids
    // Ccolids is already sorted from spgemm_Flength_array_symbolic_new
    // We only need to find position and accumulate (no insertion)
    
    // Small epsilon for zero-value check
    const ValueType eps = static_cast<ValueType>(1e-6);
    
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
            if (cluster_nnz > 0) {
                // Initialize all values for this cluster (cluster_nnz * cluster_sz values)
                std::memset(cvalues + cluster_start * cluster_sz, 0, 
                           cluster_nnz * cluster_sz * sizeof(ValueType));
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
            
            // Get pointer to this cluster's sorted column indices
            const IndexType *cluster_ccolids = ccolids + cluster_start;
            
            // Get cluster's column range in A_cluster
            IndexType a_col_start = A_cluster.rowptr[cluster_id];
            IndexType a_col_end = A_cluster.rowptr[cluster_id + 1];
            
            // Pre-compute base index for cvalues to reduce repeated calculations
            IndexType cval_base = cluster_start * cluster_sz;
            
            // Process each row within the cluster
            for (IndexType row_in_cluster = 0; row_in_cluster < cluster_sz; ++row_in_cluster) {
                // Calculate the actual CSR row number
                IndexType csr_row = cluster_id * cluster_sz + row_in_cluster;
                
                // Skip if this row is beyond the original matrix rows (last cluster may be incomplete)
                if (csr_row >= A_cluster.csr_rows) break;
                
                // Traverse all columns in this cluster
                // Reuse col-index of A, then rows of matrix B have cache locality
                for (IndexType j = a_col_start; j < a_col_end; ++j) {
                    t_acol = A_cluster.colids[j];
                    
                    // Pre-compute base index for A_cluster.values to reduce repeated calculations
                    IndexType a_val_base = j * cluster_sz;
                    
                    // Get the value for this row in this column
                    t_aval = A_cluster.values[a_val_base + row_in_cluster];
                    
                    // Skip zero values (sparse matrix optimization)
                    if (std::abs(t_aval) < eps) continue;
                    
                    // Traverse B[t_acol, :] to accumulate products
                    for (IndexType k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        IndexType target_col = bcol[k];
                        t_bval = bval[k];  // Cache bval[k] once per iteration
                        product = t_aval * t_bval;
                        
                        // Binary search to find position in pre-sorted cluster_ccolids
                        IndexType pos = binary_search_find(cluster_ccolids, cluster_nnz, target_col);
                        
                        // Accumulate if found (should always be found since Ccolids was generated from symbolic phase)
                        if (pos != -1) {
                            // Value position: cval_base + pos * cluster_sz + row_in_cluster
                            IndexType val_idx = cval_base + pos * cluster_sz + row_in_cluster;
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
template void spgemm_Flength_array_symbolic_new<int64_t, float>(
    const CSR_FlengthCluster<int64_t, float> &A_cluster,
    const int64_t *brpt, const int64_t *bcol,
    int64_t c_clusters, int64_t c_cols,
    int64_t *crpt, int64_t *&ccolids, int64_t &c_nnzc,
    SpGEMM_BIN_FlengthCluster<int64_t, float> *bin);

template void spgemm_Flength_array_symbolic_new<int64_t, double>(
    const CSR_FlengthCluster<int64_t, double> &A_cluster,
    const int64_t *brpt, const int64_t *bcol,
    int64_t c_clusters, int64_t c_cols,
    int64_t *crpt, int64_t *&ccolids, int64_t &c_nnzc,
    SpGEMM_BIN_FlengthCluster<int64_t, double> *bin);

// Numeric phase
template void spgemm_Flength_array_numeric_new<true, int64_t, float>(
    const CSR_FlengthCluster<int64_t, float> &A_cluster,
    const int64_t *brpt, const int64_t *bcol, const float *bval,
    int64_t c_clusters, int64_t c_cols,
    const int64_t *crpt, const int64_t *ccolids, float *cvalues,
    int64_t cluster_sz,
    SpGEMM_BIN_FlengthCluster<int64_t, float> *bin);

template void spgemm_Flength_array_numeric_new<false, int64_t, float>(
    const CSR_FlengthCluster<int64_t, float> &A_cluster,
    const int64_t *brpt, const int64_t *bcol, const float *bval,
    int64_t c_clusters, int64_t c_cols,
    const int64_t *crpt, const int64_t *ccolids, float *cvalues,
    int64_t cluster_sz,
    SpGEMM_BIN_FlengthCluster<int64_t, float> *bin);

template void spgemm_Flength_array_numeric_new<true, int64_t, double>(
    const CSR_FlengthCluster<int64_t, double> &A_cluster,
    const int64_t *brpt, const int64_t *bcol, const double *bval,
    int64_t c_clusters, int64_t c_cols,
    const int64_t *crpt, const int64_t *ccolids, double *cvalues,
    int64_t cluster_sz,
    SpGEMM_BIN_FlengthCluster<int64_t, double> *bin);

template void spgemm_Flength_array_numeric_new<false, int64_t, double>(
    const CSR_FlengthCluster<int64_t, double> &A_cluster,
    const int64_t *brpt, const int64_t *bcol, const double *bval,
    int64_t c_clusters, int64_t c_cols,
    const int64_t *crpt, const int64_t *ccolids, double *cvalues,
    int64_t cluster_sz,
    SpGEMM_BIN_FlengthCluster<int64_t, double> *bin);
