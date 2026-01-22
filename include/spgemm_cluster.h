#ifndef SPGEMM_CLUSTER_H
#define SPGEMM_CLUSTER_H

#include "sparse_format.h"
#include "memopt.h"
#include "thread.h"
#include "general_config.h"
#include <omp.h>
#include <cstring>
#include <algorithm>

/**
 * @file spgemm_cluster.h
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Cluster CSR formats for Cluster-wise SpGEMM (Fixed-length and Variable-length)
 * @version 0.1
 * @date 2026
 * 
 * @brief This file defines two cluster formats:
 *        1. Fixed-length Cluster: Groups CSR rows into clusters of fixed size (cluster_sz)
 *        2. Variable-length Cluster: Groups CSR rows into clusters of variable size (cluster_sz[i])
 *        
 *        Both formats improve cache locality and reduce thread synchronization overhead.
 *        Each cluster stores unique column IDs with values for each row in the cluster.
 */

/**
 * @brief Fixed-length Cluster CSR Matrix Format
 *        Groups multiple CSR rows into clusters of fixed size (cluster_sz)
 *        Each cluster stores unique column IDs, with cluster_sz values per column
 *        Value layout: values[j * cluster_sz + k], where j is column index, k is row index within cluster
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 */
template <typename IndexType, typename ValueType>
struct CSR_FlengthCluster : public Matrix_Features<IndexType>
{
    typedef IndexType index_type;
    typedef ValueType value_type;

    IndexType csr_rows;      // Original number of CSR rows (before clustering)
    IndexType rows;           // Number of clusters (csr_rows / cluster_sz, rounded up)
    IndexType cols;           // Number of columns
    IndexType cluster_sz;     // Number of rows per cluster (fixed size)
    IndexType nnzc;           // Total number of unique column IDs across all clusters, to get B rows
    
    IndexType *rowptr;        // Cluster pointer array (length: rows + 1)
                              // rowptr[i] points to the start of cluster i's column IDs in colids
    IndexType *colids;        // Column IDs for all clusters (length: nnzc)
                              // Column IDs for cluster i are in colids[rowptr[i]..rowptr[i+1]-1]
    ValueType *values;        // Values array (length: nnzc * cluster_sz)
                              // Values for column j in cluster i: values[(rowptr[i] + j) * cluster_sz + k]
                              // where k is the row index within the cluster (0 <= k < cluster_sz)
};

/**
 * @brief Delete memory allocated for CSR_FlengthCluster
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param cluster CSR_FlengthCluster matrix to delete
 */
template <typename IndexType, typename ValueType>
void delete_cluster_matrix(CSR_FlengthCluster<IndexType, ValueType> &cluster)
{
    delete_array(cluster.rowptr);
    delete_array(cluster.colids);
    delete_array(cluster.values);
    
    // Reset fields
    cluster.csr_rows = 0;
    cluster.rows = 0;
    cluster.cols = 0;
    cluster.cluster_sz = 0;
    cluster.nnzc = 0;
}

/**
 * @brief Delete CSR_FlengthCluster matrix (alias for delete_cluster_matrix)
 *        Follows the naming convention used in sparse_format.h
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param cluster CSR_FlengthCluster matrix to delete
 */
template <typename IndexType, typename ValueType>
void delete_host_matrix(CSR_FlengthCluster<IndexType, ValueType> &cluster)
{
    delete_cluster_matrix(cluster);
}

/**
 * @brief Variable-length Cluster CSR Matrix Format
 *        Groups multiple CSR rows into clusters of variable size (cluster_sz[i])
 *        Each cluster stores unique column IDs, with cluster_sz[i] values per column
 *        Value layout: values[rowptr_val[i] + (j - rowptr[i]) * cluster_sz[i] + k]
 *        where i is cluster ID, j is column index within cluster, k is row index within cluster
 * 
 * Key differences from CSR_FlengthCluster:
 * - cluster_sz is an array (each cluster can have different size)
 * - rowptr_val is needed to index into values array (values are not uniformly sized)
 * - nnzv = sum((rowptr[i+1] - rowptr[i]) * cluster_sz[i]) for all clusters
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 */
template <typename IndexType, typename ValueType>
struct CSR_VlengthCluster : public Matrix_Features<IndexType>
{
    typedef IndexType index_type;
    typedef ValueType value_type;

    IndexType csr_rows;      // Original number of CSR rows (before clustering)
    IndexType rows;          // Number of clusters
    IndexType cols;          // Number of columns
    IndexType nnzc;          // Total number of unique column IDs across all clusters
    IndexType nnzv;          // Total size of values array: sum((rowptr[i+1] - rowptr[i]) * cluster_sz[i])
    IndexType max_cluster_sz; // Maximum cluster size (useful for SPA and hash table sizing)
    
    IndexType *cluster_sz;   // Number of rows per cluster (array, length: rows)
                             // cluster_sz[i] = number of rows in cluster i
    IndexType *rowptr;       // Cluster pointer array for colids (length: rows + 1)
                             // rowptr[i] points to the start of cluster i's column IDs in colids
    IndexType *colids;       // Column IDs for all clusters (length: nnzc)
                             // Column IDs for cluster i are in colids[rowptr[i]..rowptr[i+1]-1]
    IndexType *rowptr_val;   // Cluster pointer array for values (length: rows + 1)
                             // rowptr_val[i] points to the start of cluster i's values in values array
    ValueType *values;       // Values array (length: nnzv)
                             // Values for column j in cluster i:
                             //   values[rowptr_val[i] + (j - rowptr[i]) * cluster_sz[i] + k]
                             // where k is the row index within the cluster (0 <= k < cluster_sz[i])
};

/**
 * @brief Delete memory allocated for CSR_VlengthCluster
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param cluster CSR_VlengthCluster matrix to delete
 */
template <typename IndexType, typename ValueType>
void delete_vlength_cluster_matrix(CSR_VlengthCluster<IndexType, ValueType> &cluster)
{
    delete_array(cluster.cluster_sz);
    delete_array(cluster.rowptr);
    delete_array(cluster.colids);
    delete_array(cluster.rowptr_val);
    delete_array(cluster.values);
    
    // Reset fields
    cluster.csr_rows = 0;
    cluster.rows = 0;
    cluster.cols = 0;
    cluster.nnzc = 0;
    cluster.nnzv = 0;
    cluster.max_cluster_sz = 0;
}

/**
 * @brief Delete CSR_VlengthCluster matrix (alias for delete_vlength_cluster_matrix)
 *        Follows the naming convention used in sparse_format.h
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param cluster CSR_VlengthCluster matrix to delete
 */
template <typename IndexType, typename ValueType>
void delete_host_matrix(CSR_VlengthCluster<IndexType, ValueType> &cluster)
{
    delete_vlength_cluster_matrix(cluster);
}

/**
 * @brief BIN class for load balancing in Fixed-length Cluster SpGEMM
 *        Used to partition clusters into bins based on computational complexity
 *        This class manages cluster-level load balancing (not row-level)
 * 
 * Key differences from SpGEMM_BIN:
 * - Works with clusters instead of individual rows
 * - Each cluster has fixed size (cluster_sz rows)
 * - bin_id, cluster_nz, and clusters_offset are cluster-level arrays
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 */
template <typename IndexType, typename ValueType>
class SpGEMM_BIN_FlengthCluster {
public:
    IndexType num_clusters;      // Number of clusters (not rows)
    IndexType cluster_sz;        // Fixed size of each cluster (number of rows per cluster)
    IndexType max_bin_id;
    IndexType min_ht_size;
    
    char *bin_id;                   // bin ID for each cluster (matching reference implementation: char for memory efficiency)
    IndexType *cluster_nz;          // number of nonzeros per cluster (estimated)
    IndexType *clusters_offset;      // cluster offset for each thread (for load balancing)
    
    // Thread-local hash tables (for cluster-wise accumulation)
    IndexType **local_hash_table_id;  // [thread_id][hash_size]
    ValueType **local_hash_table_val; // [thread_id][hash_size]
    int allocated_thread_num;        // number of threads for which memory was allocated
    int64_t total_intprod;           // total intermediate products (for load balancing)
    size_t total_size;               // total memory size in bytes (matching reference implementation)
    
    /**
     * @brief Constructor
     * @param nclusters Number of clusters
     * @param cluster_size Fixed size of each cluster (number of rows per cluster)
     * @param min_ht_sz Minimum hash table size
     */
    SpGEMM_BIN_FlengthCluster(IndexType nclusters, IndexType cluster_size, IndexType min_ht_sz);
    
    /**
     * @brief Destructor
     */
    ~SpGEMM_BIN_FlengthCluster();
    
    /**
     * @brief Get hash table size for a given number of columns
     */
    static IndexType get_hash_size(IndexType ncols, IndexType min_ht_sz);
    
    /**
     * @brief Set max bin ID based on cluster matrix structure
     *        Estimates work per cluster based on A and B matrices
     *        Matching reference implementation: set_max_bin(arpt, acol, brpt, cols)
     */
    void set_max_bin(const IndexType *arpt, const IndexType *acol,
                     const IndexType *brpt, IndexType cols);
    
    /**
     * @brief Set clusters offset for load balancing
     *        Partitions clusters among threads based on estimated work
     */
    void set_clusters_offset();
    
    /**
     * @brief Assign bin ID to each cluster based on estimated work
     *        Work is estimated as cluster_nz[i] * avg_nnz_per_col_in_B
     */
    void set_bin_id(IndexType ncols, IndexType min_ht_sz);
    
    /**
     * @brief Create thread-local hash tables
     * @param max_cols Maximum number of columns expected per thread
     */
    void create_local_hash_table(IndexType max_cols);
    
    /**
     * @brief Calculate memory size in bytes (matching reference implementation)
     */
    size_t calculate_size();
    
    /**
     * @brief Calculate memory size in bytes (matching reference implementation)
     *        Note: Despite the name, returns bytes, not GB (matching reference)
     */
    double calculate_size_in_gb();
};

/**
 * @brief BIN class for load balancing in Variable-length Cluster SpGEMM
 *        Similar to SpGEMM_BIN_FlengthCluster, but supports variable cluster sizes
 *        Reference: BIN_VlengthCluster.h
 * 
 * Key differences from Fixed-length:
 * - cluster_sz is an array (each cluster can have different size)
 * - set_clusters_offset uses weighted scan with cluster_sz array
 * - create_local_hash_table finds max_cluster_sz per thread
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 */
template <typename IndexType, typename ValueType>
class SpGEMM_BIN_VlengthCluster {
public:
    IndexType num_clusters;      // Number of clusters (not rows)
    IndexType *cluster_sz;       // Array: size of each cluster (number of rows per cluster)
    IndexType max_bin_id;
    IndexType min_ht_size;
    
    char *bin_id_cluster;          // bin ID for each cluster (matching reference: char for memory efficiency)
    IndexType *cluster_nz;         // number of nonzeros per cluster (estimated)
    IndexType *clusters_offset;     // cluster offset for each thread (for load balancing)
    
    // Thread-local hash tables (for cluster-wise accumulation)
    IndexType **local_hash_table_id;  // [thread_id][hash_size]
    ValueType **local_hash_table_val; // [thread_id][hash_size * max_cluster_sz]
    int allocated_thread_num;          // number of threads for which memory was allocated
    int64_t total_intprod;           // total intermediate products (for load balancing)
    size_t total_size;               // total memory size in bytes (matching reference implementation)
    
    /**
     * @brief Default constructor
     */
    SpGEMM_BIN_VlengthCluster();
    
    /**
     * @brief Constructor
     * @param nclusters Number of clusters
     * @param min_ht_sz Minimum hash table size
     * @param cluster_size Array of cluster sizes (length: nclusters)
     */
    SpGEMM_BIN_VlengthCluster(IndexType nclusters, IndexType min_ht_sz, const IndexType *cluster_size);
    
    /**
     * @brief Destructor
     */
    ~SpGEMM_BIN_VlengthCluster();
    
    /**
     * @brief Get hash table size for a given number of columns
     */
    static IndexType get_hash_size(IndexType ncols, IndexType min_ht_sz);
    
    /**
     * @brief Set max bin ID based on cluster matrix structure
     *        Estimates work per cluster based on A and B matrices
     *        Matching reference implementation: set_max_bin(arpt, acol, brpt, cols)
     */
    void set_max_bin(const IndexType *arpt, const IndexType *acol,
                     const IndexType *brpt, IndexType cols);
    
    /**
     * @brief Set min bin (reset bin_id based on current cluster_nz)
     *        Matching reference implementation: set_min_bin(cols)
     */
    void set_min_bin(IndexType cols);
    
    /**
     * @brief Set clusters offset for load balancing
     *        Partitions clusters among threads based on estimated work
     *        Uses weighted scan with cluster_sz array
     */
    void set_clusters_offset();
    
    /**
     * @brief Assign bin ID to each cluster based on estimated work
     *        Matching reference implementation: set_bin_id(cols, min)
     */
    void set_bin_id(IndexType cols, IndexType min);
    
    /**
     * @brief Create thread-local hash tables
     *        Finds max_cluster_sz per thread and allocates accordingly
     * @param max_cols Maximum number of columns expected per thread
     */
    void create_local_hash_table(IndexType max_cols);
    
    /**
     * @brief Calculate memory size in bytes (matching reference implementation)
     */
    size_t calculate_size();
    
    /**
     * @brief Calculate memory size in bytes (matching reference implementation)
     *        Note: Despite the name, returns bytes, not GB (matching reference)
     */
    double calculate_size_in_gb();
};

#endif /* SPGEMM_CLUSTER_H */
