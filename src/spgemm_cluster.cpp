/**
 * @file spgemm_cluster.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Implementation of BIN class for Fixed-length Cluster SpGEMM load balancing
 * @version 0.1
 * @date 2026
 */

#include "../include/spgemm_cluster.h"
#include "../include/spgemm_utility.h"
#include <cmath>
#include <algorithm>

// ============================================================================
// Fixed-length Cluster BIN Structure
// ============================================================================
template <typename IndexType, typename ValueType>
SpGEMM_BIN_FlengthCluster<IndexType, ValueType>::SpGEMM_BIN_FlengthCluster(
    IndexType nclusters, IndexType cluster_size, IndexType min_ht_sz)
    : num_clusters(nclusters), cluster_sz(cluster_size), min_ht_size(min_ht_sz), 
      max_bin_id(0), allocated_thread_num(Le_get_thread_num()), total_intprod(0), total_size(0)
{
    cluster_nz = new_array<IndexType>(num_clusters);
    clusters_offset = new_array<IndexType>(allocated_thread_num + 1);
    bin_id = new_array<char>(num_clusters);

    // Allocate pointer arrays for hash tables (matching reference implementation)
    local_hash_table_id  = new IndexType*[allocated_thread_num];
    local_hash_table_val = new ValueType*[allocated_thread_num];
    
    // Initialize pointers to nullptr (matching reference implementation)
    for (int i = 0; i < allocated_thread_num; ++i) {
        local_hash_table_id[i] = nullptr;
    }
    
    // Calculate initial total_size (matching reference implementation)
    total_size += (num_clusters * sizeof(IndexType));             // cluster_nz
    total_size += ((allocated_thread_num + 1) * sizeof(IndexType)); // clusters_offset
    total_size += (num_clusters * sizeof(char));                  // bin_id (char for memory efficiency)
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
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, IndexType cols)
{
    // Reset total_intprod (matching reference set_intprod_num)
    total_intprod = 0;
    
    // Estimate work for each cluster: sum of B's row lengths for each column in A_cluster's cluster
    // For cluster i, work = sum over all columns j in cluster i: nnz(B[j, :])
    // Also calculate total_intprod = sum(cluster_nz[i] * cluster_sz) for load balancing
    #pragma omp parallel
    {
        int64_t each_int_prod = 0;
        #pragma omp for
        for (IndexType i = 0; i < num_clusters; i++) {
            IndexType cluster_work = 0;
            IndexType col_start = arpt[i];
            IndexType col_end = arpt[i + 1];
            
            // Sum up the work for all columns in this cluster
            for (IndexType j = col_start; j < col_end; j++) {
                IndexType col = acol[j];
                cluster_work += (brpt[col + 1] - brpt[col]);
            }
            cluster_nz[i] = cluster_work;
            each_int_prod += (static_cast<int64_t>(cluster_work) * cluster_sz);
        }
        #pragma omp atomic
        total_intprod += each_int_prod;
    }
    
    // Set clusters offset for load balancing
    set_clusters_offset();
    
    // Set bin ID based on estimated work
    set_bin_id(cols, min_ht_size);
}

template <typename IndexType, typename ValueType>
void SpGEMM_BIN_FlengthCluster<IndexType, ValueType>::set_clusters_offset()
{
    int thread_num = Le_get_thread_num();
    
    // clusters_offset is already allocated in constructor (matching reference implementation)
    
    // Prefix sum of cluster_nz with multiplier cluster_sz (matching reference scan function)
    // This computes prefix sum of (cluster_nz[i] * cluster_sz) for load balancing
    IndexType *ps_cluster_nz = new_array<IndexType>(num_clusters + 1);
    scan(cluster_nz, ps_cluster_nz, cluster_sz, num_clusters + 1, allocated_thread_num);
    
    // Calculate average work per thread using total_intprod (matching reference implementation)
    // total_intprod = sum(cluster_nz[i] * cluster_sz) for all clusters
    int64_t average_intprod = (total_intprod + thread_num - 1) / thread_num;
    
    // Set clusters_offset for each thread (matching reference implementation)
    clusters_offset[0] = 0;
    #pragma omp parallel
    {
        int tid = Le_get_thread_id();

        long long int end_itr = (std::lower_bound(ps_cluster_nz, ps_cluster_nz + num_clusters + 1, 
                                                   average_intprod * (tid + 1))) - ps_cluster_nz;
        clusters_offset[tid + 1] = end_itr;
    }
    clusters_offset[thread_num] = num_clusters;
    
    delete_array(ps_cluster_nz);
}

template <typename IndexType, typename ValueType>
void SpGEMM_BIN_FlengthCluster<IndexType, ValueType>::set_bin_id(
    IndexType ncols, IndexType min_ht_sz)
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

        local_hash_table_id[tid] = new_array<IndexType>(ht_size);
        local_hash_table_val[tid] = new_array<ValueType>(ht_size * cluster_sz);
        
        // Update total_size
        // #pragma omp atomic
        {
            total_size += (ht_size * sizeof(IndexType));                   // local_hash_table_id
            total_size += ((ht_size * cluster_sz) * sizeof(ValueType));   // local_hash_table_val
        }
    }
}

template <typename IndexType, typename ValueType>
size_t SpGEMM_BIN_FlengthCluster<IndexType, ValueType>::calculate_size()
{
    // Return total_size (matching reference implementation)
    return total_size;
}

template <typename IndexType, typename ValueType>
double SpGEMM_BIN_FlengthCluster<IndexType, ValueType>::calculate_size_in_gb()
{
    size_t size_bytes = 0;

    // cluster_nz
    size_bytes += num_clusters * sizeof(IndexType);

    // clusters_offset
    size_bytes += (allocated_thread_num + 1) * sizeof(IndexType);

    // bin_id
    size_bytes += num_clusters * sizeof(char);

    // local_hash_table_id pointers
    size_bytes += allocated_thread_num * sizeof(IndexType *);

    // local_hash_table_val pointers
    size_bytes += allocated_thread_num * sizeof(ValueType *);

    // Add per-thread allocations (if they exist)
    for (IndexType i = 0; i < allocated_thread_num; ++i) {
        if (local_hash_table_id && local_hash_table_id[i]) {
            size_bytes += min_ht_size * sizeof(IndexType);  // assuming same size across threads
        }
        if (local_hash_table_val && local_hash_table_val[i]) {
            size_bytes += min_ht_size * sizeof(ValueType);  // assuming same size across threads
        }
    }

    return static_cast<double>(size_bytes);
}

// ============================================================================
// Variable-length Cluster BIN Structure
// ============================================================================
template <typename IndexType, typename ValueType>
SpGEMM_BIN_VlengthCluster<IndexType, ValueType>::SpGEMM_BIN_VlengthCluster()
    : num_clusters(0), cluster_sz(nullptr), min_ht_size(0), max_bin_id(0),
      allocated_thread_num(Le_get_thread_num()), total_intprod(0), total_size(0),
      bin_id_cluster(nullptr), cluster_nz(nullptr), clusters_offset(nullptr),
      local_hash_table_id(nullptr), local_hash_table_val(nullptr)
{
}

template <typename IndexType, typename ValueType>
SpGEMM_BIN_VlengthCluster<IndexType, ValueType>::SpGEMM_BIN_VlengthCluster(
    IndexType nclusters, IndexType min_ht_sz, const IndexType *cluster_size)
    : num_clusters(nclusters), min_ht_size(min_ht_sz), max_bin_id(0),
      allocated_thread_num(Le_get_thread_num()), total_intprod(0), total_size(0)
{
    // Allocate and copy cluster_sz array (matching reference implementation)
    cluster_sz = new_array<IndexType>(num_clusters);
    std::memcpy(cluster_sz, cluster_size, sizeof(IndexType) * num_clusters);
    
    cluster_nz = new_array<IndexType>(num_clusters);
    clusters_offset = new_array<IndexType>(allocated_thread_num + 1);
    bin_id_cluster = new_array<char>(num_clusters);

    // Allocate pointer arrays for hash tables (matching reference implementation)
    local_hash_table_id  = new IndexType*[allocated_thread_num];
    local_hash_table_val = new ValueType*[allocated_thread_num];
    
    // Initialize pointers to nullptr (matching reference implementation)
    for (int i = 0; i < allocated_thread_num; ++i) {
        local_hash_table_id[i] = nullptr;
    }
    
    // Calculate initial total_size (matching reference implementation)
    // total_size += (num_clusters * sizeof(IndexType));             // cluster_sz
    // total_size += (num_clusters * sizeof(IndexType));             // cluster_nz
    // total_size += ((allocated_thread_num + 1) * sizeof(IndexType)); // clusters_offset
    // total_size += (num_clusters * sizeof(char));                  // bin_id_cluster
}

template <typename IndexType, typename ValueType>
SpGEMM_BIN_VlengthCluster<IndexType, ValueType>::~SpGEMM_BIN_VlengthCluster()
{
    delete_array(cluster_sz);
    delete_array(bin_id_cluster);
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
}

template <typename IndexType, typename ValueType>
IndexType SpGEMM_BIN_VlengthCluster<IndexType, ValueType>::get_hash_size(
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
void SpGEMM_BIN_VlengthCluster<IndexType, ValueType>::set_max_bin(
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, IndexType cols)
{
    // Reset total_intprod (matching reference set_intprod_num)
    total_intprod = 0;
    
    // Estimate work for each cluster: sum of B's row lengths for each column in A_cluster's cluster
    // For cluster i, work = sum over all columns j in cluster i: nnz(B[j, :])
    // Also calculate total_intprod = sum(cluster_nz[i] * cluster_sz[i]) for load balancing
    // Matching reference: set_intprod_num(arpt, acol, brpt)
    #pragma omp parallel
    {
        int64_t each_int_prod = 0;
        #pragma omp for
        for (IndexType i = 0; i < num_clusters; i++) { // loop over clusters
            IndexType nz_per_cluster = 0;
            for (IndexType j = arpt[i]; j < arpt[i + 1]; ++j) {
                nz_per_cluster += ((brpt[acol[j] + 1] - brpt[acol[j]]));
            }
            cluster_nz[i] = nz_per_cluster;  // cluster_nz is populated by flops
            each_int_prod += ((int64_t) nz_per_cluster * (int64_t) cluster_sz[i]);
        }
        #pragma omp atomic
        total_intprod += each_int_prod;
    }
    
    // Set clusters offset for load balancing
    set_clusters_offset();
    
    // Set bin ID based on estimated work
    set_bin_id(cols, min_ht_size);
}

template <typename IndexType, typename ValueType>
void SpGEMM_BIN_VlengthCluster<IndexType, ValueType>::set_clusters_offset()
{
    
    // clusters_offset is already allocated in constructor (matching reference implementation)
    
    // Prefix sum of cluster_nz with multiplier cluster_sz array (matching reference scan function)
    // This computes prefix sum of (cluster_nz[i] * cluster_sz[i]) for load balancing
    // Reference: scan<int64_t, IT>(cluster_nz, ps_cluster_nz, cluster_sz, num_clusters + 1)
    int64_t *ps_cluster_nz = new_array<int64_t>(num_clusters + 1);
    scan<int64_t, IndexType>(cluster_nz, ps_cluster_nz, cluster_sz, num_clusters + 1, allocated_thread_num);
    
    // Calculate average work per thread using total_intprod (matching reference implementation)
    // total_intprod = sum(cluster_nz[i] * cluster_sz[i]) for all clusters
    int64_t average_intprod = (total_intprod + allocated_thread_num - 1) / allocated_thread_num;
    
    // Set clusters_offset for each thread (matching reference implementation)
    clusters_offset[0] = 0;
    #pragma omp parallel
    {
        int tid = Le_get_thread_id();
        long long int end_itr = (std::lower_bound(ps_cluster_nz, ps_cluster_nz + num_clusters + 1, 
                                                   average_intprod * (tid + 1))) - ps_cluster_nz;
        clusters_offset[tid + 1] = end_itr;
    }
    clusters_offset[allocated_thread_num] = num_clusters;
    
    delete_array(ps_cluster_nz);
}

template <typename IndexType, typename ValueType>
void SpGEMM_BIN_VlengthCluster<IndexType, ValueType>::set_bin_id(
    IndexType cols, IndexType min)
{
    // Assign bin ID based on estimated work (logarithmic binning)
    // Match reference implementation: set_bin_id(cols, min)
    #pragma omp parallel for
    for (IndexType i = 0; i < num_clusters; i++) {
        IndexType nz_per_cluster = cluster_nz[i];
        if (nz_per_cluster > cols) nz_per_cluster = cols;
        
        if (nz_per_cluster == 0) {
            bin_id_cluster[i] = 0;
        } else {
            IndexType j = 0;
            while (nz_per_cluster > (min << j)) {
                j++;
            }
            bin_id_cluster[i] = j + 1;
        }
    }
}

template <typename IndexType, typename ValueType>
void SpGEMM_BIN_VlengthCluster<IndexType, ValueType>::set_min_bin(IndexType cols)
{
    // Reset bin_id based on current cluster_nz (matching reference implementation)
    set_bin_id(cols, min_ht_size);
}

template <typename IndexType, typename ValueType>
void SpGEMM_BIN_VlengthCluster<IndexType, ValueType>::create_local_hash_table(IndexType max_cols)
{
    // Allocate hash tables for each thread (matching reference implementation)
    // Note: Pointer arrays are already allocated in constructor
    #pragma omp parallel
    {
        int tid = Le_get_thread_id();
        IndexType ht_size = 0;                     // max cluster_nz among all the clusters
        IndexType max_cluster_sz = 0;              // max cluster_sz among all the clusters
        IndexType ht_val_size = 0;
        
        // Safety check: ensure tid is within valid range and clusters_offset is valid
        // if (tid < allocated_thread_num && clusters_offset != nullptr && tid + 1 <= allocated_thread_num) {
            // Get max size of hash table and max_cluster_sz for this thread's clusters
            // Matching reference: create_local_hash_table(cols)
            IndexType start_cluster = clusters_offset[tid];
            IndexType end_cluster = clusters_offset[tid + 1];
            
            // Safety check: ensure valid cluster range
            // if (start_cluster <= end_cluster && end_cluster <= num_clusters) {
            for (IndexType j = start_cluster; j < end_cluster; ++j) {
                    if (ht_size < cluster_nz[j]) ht_size = cluster_nz[j];
                    if (max_cluster_sz < cluster_sz[j]) max_cluster_sz = cluster_sz[j];
            }
            // }
        // }
        
        // Align to power of 2 (2^n) - matching reference implementation
        // Note: Reference code does NOT set ht_size = min_ht_size when ht_size == 0
        // It directly uses ht_size = 0 and allocates with size 0
        if (ht_size > 0) {
            if (ht_size > max_cols) ht_size = max_cols;
            IndexType k = min_ht_size;
            while (k < ht_size) {
                k <<= 1;
            }
            ht_size = k;
        }
        
        // Calculate ht_val_size = ht_size * max_cluster_sz (matching reference)
        // Note: Variable-length cluster needs max_cluster_sz per thread, not fixed cluster_sz
        ht_val_size = ht_size * max_cluster_sz;
        
        // Allocate hash tables (matching reference: my_malloc even with size 0)
        // Only allocate if tid is within valid range
        local_hash_table_id[tid] = new_array<IndexType>(ht_size);
        local_hash_table_val[tid] = new_array<ValueType>(ht_val_size);
        
        
        // Update total_size (thread-safe update, matching reference implementation)
        // #pragma omp atomic
        // total_size += (ht_size * sizeof(IndexType));                   // local_hash_table_id
        // #pragma omp atomic
        // total_size += (ht_val_size * sizeof(ValueType));               // local_hash_table_val
    }
}

template <typename IndexType, typename ValueType>
size_t SpGEMM_BIN_VlengthCluster<IndexType, ValueType>::calculate_size()
{
    // Return total_size (matching reference implementation)
    return total_size;
}

template <typename IndexType, typename ValueType>
double SpGEMM_BIN_VlengthCluster<IndexType, ValueType>::calculate_size_in_gb()
{
    size_t size_bytes = 0;

    // cluster_sz
    if (cluster_sz) size_bytes += num_clusters * sizeof(IndexType);
    
    // cluster_nz
    if (cluster_nz) size_bytes += num_clusters * sizeof(IndexType);

    // clusters_offset
    if (clusters_offset) size_bytes += (allocated_thread_num + 1) * sizeof(IndexType);

    // bin_id_cluster
    if (bin_id_cluster) size_bytes += num_clusters * sizeof(char);

    // local_hash_table_id pointers
    if (local_hash_table_id) size_bytes += allocated_thread_num * sizeof(IndexType *);

    // local_hash_table_val pointers
    if (local_hash_table_val) size_bytes += allocated_thread_num * sizeof(ValueType *);

    // Add per-thread allocations (if they exist)
    // Note: Reference implementation has a bug (uses min_ht_size instead of actual size)
    // We'll use a conservative estimate here
    for (IndexType i = 0; i < allocated_thread_num; ++i) {
        if (local_hash_table_id && local_hash_table_id[i]) {
            size_bytes += min_ht_size * sizeof(IndexType);  // conservative estimate
        }
        if (local_hash_table_val && local_hash_table_val[i]) {
            size_bytes += min_ht_size * sizeof(ValueType);  // conservative estimate
        }
    }

    return static_cast<double>(size_bytes);
}

// Explicit template instantiations (only int64_t for IndexType)
#include <cstdint>

template class SpGEMM_BIN_FlengthCluster<int64_t, float>;
template class SpGEMM_BIN_FlengthCluster<int64_t, double>;

template class SpGEMM_BIN_VlengthCluster<int64_t, float>;
template class SpGEMM_BIN_VlengthCluster<int64_t, double>;
