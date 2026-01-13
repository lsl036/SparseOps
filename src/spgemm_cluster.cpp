/**
 * @file spgemm_cluster.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Implementation of BIN class for Fixed-length Cluster SpGEMM load balancing
 * @version 0.1
 * @date 2026
 */

#include "../include/spgemm_cluster.h"
#include <cmath>
#include <algorithm>

// Helper function: prefix sum (scan)
template <typename IndexType>
inline void scan_spgemm(const IndexType *input, IndexType *output, IndexType n) {
    output[0] = 0;
    for (IndexType i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

// ============================================================================
// Fixed-length Cluster BIN Structure
// ============================================================================
template <typename IndexType, typename ValueType>
SpGEMM_BIN_FlengthCluster<IndexType, ValueType>::SpGEMM_BIN_FlengthCluster(
    IndexType nclusters, IndexType cluster_size, IndexType min_ht_sz)
    : num_clusters(nclusters), cluster_sz(cluster_size), min_ht_size(min_ht_sz), 
      max_bin_id(0), allocated_thread_num(0)
{
    int thread_num = Le_get_thread_num();
    
    bin_id = new_array<IndexType>(num_clusters);
    cluster_nz = new_array<IndexType>(num_clusters);
    clusters_offset = new_array<IndexType>(thread_num + 1);
    
    // Allocate pointer arrays for hash tables (matching reference implementation)
    local_hash_table_id = new IndexType*[thread_num];
    local_hash_table_val = new ValueType*[thread_num];
    hash_table_size = new_array<IndexType>(thread_num);
    
    // Initialize pointers to nullptr (matching reference implementation)
    for (int i = 0; i < thread_num; ++i) {
        local_hash_table_id[i] = nullptr;
        local_hash_table_val[i] = nullptr;
    }
    
    // Initialize arrays
    std::fill_n(bin_id, num_clusters, static_cast<IndexType>(0));
    std::fill_n(cluster_nz, num_clusters, static_cast<IndexType>(0));
    
    allocated_thread_num = thread_num;
}

template <typename IndexType, typename ValueType>
SpGEMM_BIN_FlengthCluster<IndexType, ValueType>::~SpGEMM_BIN_FlengthCluster()
{
    delete_array(bin_id);
    delete_array(cluster_nz);
    delete_array(clusters_offset);
    
    // Free hash table allocations using parallel (matching reference implementation)
    if (local_hash_table_id != nullptr) {
        #pragma omp parallel
        {
            int tid = Le_get_thread_id();
            if (tid < allocated_thread_num && local_hash_table_id[tid] != nullptr) {
                delete_array(local_hash_table_id[tid]);
                delete_array(local_hash_table_val[tid]);
            }
        }
        delete[] local_hash_table_id;
        delete[] local_hash_table_val;
    }
    if (hash_table_size != nullptr) {
        delete_array(hash_table_size);
    }
}

template <typename IndexType, typename ValueType>
IndexType SpGEMM_BIN_FlengthCluster<IndexType, ValueType>::get_hash_size(
    IndexType ncols, IndexType min_ht_sz)
{
    IndexType hash_size = min_ht_sz;
    while (hash_size < ncols) {
        IndexType new_size = hash_size * HASH_SCAL / 100;
        // Ensure hash_size actually increases to avoid infinite loop
        if (new_size <= hash_size) {
            hash_size = hash_size * 2;  // Double the size if scaling doesn't help
        } else {
            hash_size = new_size;
        }
        // Safety check: prevent overflow
        if (hash_size < min_ht_sz) {
            hash_size = ncols * 2;  // Fallback: use 2x ncols
            break;
        }
    }
    return hash_size;
}

template <typename IndexType, typename ValueType>
void SpGEMM_BIN_FlengthCluster<IndexType, ValueType>::set_max_bin(
    const CSR_FlengthCluster<IndexType, ValueType> &A_cluster,
    const CSR_Matrix<IndexType, ValueType> &B,
    IndexType c_cols)
{
    // Estimate work for each cluster: sum of B's row lengths for each column in A_cluster's cluster
    // For cluster i, work = sum over all columns j in cluster i: nnz(B[j, :])
    #pragma omp parallel for
    for (IndexType i = 0; i < num_clusters; i++) {
        IndexType cluster_work = 0;
        IndexType col_start = A_cluster.rowptr[i];
        IndexType col_end = A_cluster.rowptr[i + 1];
        
        // Sum up the work for all columns in this cluster
        for (IndexType j = col_start; j < col_end; j++) {
            IndexType col = A_cluster.colids[j];
            if (col < c_cols && col < B.num_rows) {
                cluster_work += B.row_offset[col + 1] - B.row_offset[col];
            }
        }
        cluster_nz[i] = cluster_work;
    }
    
    // Set clusters offset for load balancing
    set_clusters_offset(num_clusters);
    
    // Set bin ID based on estimated work
    set_bin_id(num_clusters, c_cols, min_ht_size);
}

template <typename IndexType, typename ValueType>
void SpGEMM_BIN_FlengthCluster<IndexType, ValueType>::set_clusters_offset(IndexType nclusters)
{
    int thread_num = Le_get_thread_num();
    
    // clusters_offset is already allocated in constructor (matching reference implementation)
    
    // Prefix sum of cluster_nz
    IndexType *ps_cluster_nz = new_array<IndexType>(nclusters + 1);
    scan_spgemm(cluster_nz, ps_cluster_nz, nclusters + 1);
    
    // Calculate average work per thread
    IndexType total_work = ps_cluster_nz[nclusters];
    IndexType average_work = (total_work + thread_num - 1) / thread_num;
    
    // Set clusters_offset for each thread (matching reference implementation)
    clusters_offset[0] = 0;
    #pragma omp parallel
    {
        int tid = Le_get_thread_id();

        long long int end_itr = (std::lower_bound(ps_cluster_nz, ps_cluster_nz + nclusters + 1, average_work * (tid + 1))) - ps_cluster_nz;
        clusters_offset[tid + 1] = end_itr;
    }
    clusters_offset[thread_num] = nclusters;
    
    delete_array(ps_cluster_nz);
}

template <typename IndexType, typename ValueType>
void SpGEMM_BIN_FlengthCluster<IndexType, ValueType>::set_bin_id(
    IndexType nclusters, IndexType ncols, IndexType min_ht_sz)
{
    // Assign bin ID based on estimated work (logarithmic binning)
    // Match reference implementation: while loop instead of log2
    #pragma omp parallel for
    for (IndexType i = 0; i < num_clusters; i++) {
        IndexType nz_per_cluster = cluster_nz[i];
        if (nz_per_cluster > ncols) nz_per_cluster = ncols;
        
        if (nz_per_cluster == 0) {
            bin_id[i] = 0;
        } else {
            IndexType j = 0;
            while (nz_per_cluster > (min_ht_sz << j)) {
                j++;
            }
            bin_id[i] = j + 1;
        }
    }
}

template <typename IndexType, typename ValueType>
void SpGEMM_BIN_FlengthCluster<IndexType, ValueType>::create_local_hash_table(IndexType max_cols)
{
    // Allocate hash tables for each thread (matching reference implementation)
    // Note: Pointer arrays are already allocated in constructor
    #pragma omp parallel
    {
        int tid = Le_get_thread_id();
        IndexType ht_size = 0;
        
        // Get max hash table size needed for this thread's clusters
        for (IndexType j = clusters_offset[tid]; j < clusters_offset[tid + 1]; ++j) {
            if (ht_size < cluster_nz[j]) ht_size = cluster_nz[j];
        }
        
        // Align to power of 2 (2^n) - matching reference implementation
        if (ht_size > 0) {
            if (ht_size > max_cols) ht_size = max_cols;
            IndexType k = min_ht_size;
            while (k < ht_size) {
                k <<= 1;
            }
            ht_size = k;
        }
        
        // Free old allocation if exists (matching reference implementation: always reallocate)
        if (local_hash_table_id[tid] != nullptr) {
            delete_array(local_hash_table_id[tid]);
            delete_array(local_hash_table_val[tid]);
        }
        
        // Allocate new hash table (matching reference implementation)
        // Note: For cluster format, each hash table entry stores cluster_sz values
        // (one for each row in the cluster)
        local_hash_table_id[tid] = new_array<IndexType>(ht_size);
        local_hash_table_val[tid] = new_array<ValueType>((size_t)ht_size * cluster_sz);
        hash_table_size[tid] = ht_size;
    }
}

template <typename IndexType, typename ValueType>
double SpGEMM_BIN_FlengthCluster<IndexType, ValueType>::calculate_size_in_gb()
{
    double size_gb = 0.0;
    int thread_num = Le_get_thread_num();
    
    if (hash_table_size != nullptr) {
        for (int i = 0; i < thread_num; i++) {
            // Note: local_hash_table_val size is ht_size * cluster_sz (cluster format)
            size_gb += hash_table_size[i] * sizeof(IndexType) + 
                       hash_table_size[i] * cluster_sz * sizeof(ValueType);
        }
    }
    
    size_gb += num_clusters * (sizeof(IndexType) * 2); // bin_id + cluster_nz
    
    return size_gb / (1024.0 * 1024.0 * 1024.0);
}

// Explicit template instantiations (only int64_t for IndexType)
#include <cstdint>

template class SpGEMM_BIN_FlengthCluster<int64_t, float>;
template class SpGEMM_BIN_FlengthCluster<int64_t, double>;
