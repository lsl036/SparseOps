#ifndef SPGEMM_BIN_H
#define SPGEMM_BIN_H

#include "sparse_format.h"
#include "memopt.h"
#include "thread.h"
#include "general_config.h"
#include <omp.h>
#include <cstring>
#include <algorithm>

/**
 * @brief BIN class for load balancing in SpGEMM
 *        Used to partition rows into bins based on computational complexity
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 */
template <typename IndexType, typename ValueType>
class SpGEMM_BIN {
public:
    IndexType num_rows;
    IndexType max_bin_id;
    IndexType min_ht_size;
    
    IndexType *bin_id;              // bin ID for each row
    IndexType *row_nz;              // number of nonzeros per row (estimated)
    IndexType *rows_offset;          // row offset for each thread
    
    // Thread-local hash tables
    IndexType **local_hash_table_id;  // [thread_id][hash_size]
    ValueType **local_hash_table_val; // [thread_id][hash_size]
    IndexType *hash_table_size;      // hash table size for each thread
    
    SpGEMM_BIN(IndexType nrows, IndexType min_ht_sz);
    ~SpGEMM_BIN();
    
    /**
     * @brief Get hash table size for a given number of columns
     */
     static IndexType get_hash_size(IndexType ncols, IndexType min_ht_sz);
     
    /**
     * @brief Set max bin ID based on matrix structure
     */
    void set_max_bin(const IndexType *arpt, const IndexType *acol, 
                     const IndexType *brpt, IndexType c_rows, IndexType c_cols);
    
    /**
     * @brief Assign bin ID to each row based on estimated work
     */
    void set_bin_id(IndexType nrows, IndexType ncols, IndexType min_ht_sz);
    
    /**
     * @brief Create thread-local hash tables
     */
    void create_local_hash_table(IndexType max_cols);
    
    /**
     * @brief Calculate memory size in GB
     */
    double calculate_size_in_gb();
};

/**
 * @brief Hash function for hash table
 */
template <typename IndexType>
inline IndexType hash_func(IndexType key, IndexType hash_size) {
    return key % hash_size;
}

/**
 * @brief Find position in hash table (linear probing)
 */
template <typename IndexType>
IndexType hash_find_pos(IndexType key, IndexType hash_size, 
                        const IndexType *hash_table_id);

/**
 * @brief Insert into hash table (symbolic phase)
 */
template <typename IndexType>
IndexType hash_insert_symbolic(IndexType key, IndexType hash_size,
                                IndexType *hash_table_id);

/**
 * @brief Insert into hash table (numeric phase)
 */
template <typename IndexType, typename ValueType>
IndexType hash_insert_numeric(IndexType key, ValueType val, IndexType hash_size,
                               IndexType *hash_table_id, ValueType *hash_table_val);

#endif /* SPGEMM_BIN_H */

