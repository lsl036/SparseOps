/**
 * @file spgemm.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief SpGEMM main interface implementation
 * @version 0.1
 * @date 2024
 */

#include "../include/spgemm.h"
#include "../include/spgemm_array.h"
#include "../include/spgemm_Flength_hash.h"
#include "../include/spgemm_Flength_array.h"
#include "../include/spgemm_Vlength_hash.h"
#include "../include/spgemm_Vlength_array.h"
#include "../include/sparse_conversion.h"
#include "../include/sparse_operation.h"
#include <cassert>
#include <cstdint>
#include <cstring>

template <typename IndexType, typename ValueType>
long long int get_spgemm_flop(const CSR_Matrix<IndexType, ValueType> &A,
                               const CSR_Matrix<IndexType, ValueType> &B)
{
    long long int flops = 0;
    
    for (IndexType i = 0; i < A.num_rows; i++) {
        for (IndexType j = A.row_offset[i]; j < A.row_offset[i + 1]; j++) {
            IndexType col_a = A.col_index[j];
            if (col_a < B.num_rows) {
                IndexType row_nnz = B.row_offset[col_a + 1] - B.row_offset[col_a];
                flops += row_nnz;
            }
        }
    }
    
    // Total number of floating-point operations including addition and multiplication in SpGEMM
    return (flops * 2);
}

template <bool sortOutput, typename IndexType, typename ValueType>
void LeSpGEMM_hash_rowwise(const CSR_Matrix<IndexType, ValueType> &A,
                           const CSR_Matrix<IndexType, ValueType> &B,
                           CSR_Matrix<IndexType, ValueType> &C,
                           int kernel_flag)
{
    // Sanity checks
    assert(A.num_cols == B.num_rows);
    
    // Initialize output matrix
    C.num_rows = A.num_rows;
    C.num_cols = B.num_cols;
    C.num_nnzs = 0;
    C.kernel_flag = kernel_flag;
    C.tag = 0;
    C.partition = nullptr;
    
    // Adapt field names: CSR_Matrix uses row_offset/col_index/values
    // Internal functions expect arpt/acol/aval, brpt/bcol/bval
    const IndexType *arpt = A.row_offset;
    const IndexType *acol = A.col_index;
    const ValueType *aval = A.values;
    
    const IndexType *brpt = B.row_offset;
    const IndexType *bcol = B.col_index;
    const ValueType *bval = B.values;
    
    // Create BIN for load balancing (matching reference RowSpGEMM)
    SpGEMM_BIN<IndexType, ValueType> *bin = new SpGEMM_BIN<IndexType, ValueType>(A.num_rows, MIN_HT_S);
    
    // Set max bin (calls set_intprod_num, set_rows_offset, set_bin_id)
    bin->set_max_bin(arpt, acol, brpt, C.num_rows, C.num_cols);
    
    // Create hash table (thread local)
    bin->create_local_hash_table(C.num_cols);
    
    // Allocate row pointer (matching reference RowSpGEMM: c.rowptr = my_malloc<IT>(c.rows + 1))
    IndexType *cpt = new_array<IndexType>(C.num_rows + 1);
    IndexType c_nnz = 0;
    
    // Symbolic Phase (matching reference hash_symbolic)
    spgemm_hash_symbolic_omp_lb<IndexType, ValueType>(arpt, acol, brpt, bcol,
                                  C.num_rows, C.num_cols,
                                  cpt, c_nnz, bin);
    
    // Re-adjust bin_id after symbolic phase (to reduce hashtable size)
    bin->set_bin_id(C.num_rows, C.num_cols, bin->min_ht_size);
    
    C.num_nnzs = c_nnz;
    C.row_offset = cpt;
    
    // Allocate column indices and values (will be filled in numeric phase)
    C.col_index = new_array<IndexType>(c_nnz);
    C.values = new_array<ValueType>(c_nnz);
    
    // Numeric Phase (sorting is handled inside numeric phase based on sortOutput template parameter)
    spgemm_hash_numeric_omp_lb<sortOutput, IndexType, ValueType>(arpt, acol, aval,
                                     brpt, bcol, bval,
                                     C.num_rows, C.num_cols,
                                     cpt, C.col_index, C.values, bin);
    
    // Cleanup
    delete bin;
}

/**
 * @brief Optimized array-based row-wise SpGEMM implementation
 *        Uses pre-sorted Ccol from symbolic phase, eliminating insertion operations
 */
template <bool sortOutput, typename IndexType, typename ValueType>
void LeSpGEMM_array_rowwise_new(const CSR_Matrix<IndexType, ValueType> &A,
                                 const CSR_Matrix<IndexType, ValueType> &B,
                                 CSR_Matrix<IndexType, ValueType> &C)
{
    // Sanity checks
    assert(A.num_cols == B.num_rows);
    
    // Initialize output matrix
    C.num_rows = A.num_rows;
    C.num_cols = B.num_cols;
    C.num_nnzs = 0;
    C.kernel_flag = 2; // Array-based method
    C.tag = 0;
    C.partition = nullptr;
    
    // Adapt field names: CSR_Matrix uses row_offset/col_index/values
    // Internal functions expect arpt/acol/aval, brpt/bcol/bval
    const IndexType *arpt = A.row_offset;
    const IndexType *acol = A.col_index;
    const ValueType *aval = A.values;
    
    const IndexType *brpt = B.row_offset;
    const IndexType *bcol = B.col_index;
    const ValueType *bval = B.values;
    
    // Create BIN for load balancing
    SpGEMM_BIN<IndexType, ValueType> *bin = new SpGEMM_BIN<IndexType, ValueType>(A.num_rows, MIN_HT_S);
    
    // Set max bin (calls set_intprod_num, set_rows_offset, set_bin_id)
    bin->set_max_bin(arpt, acol, brpt, C.num_rows, C.num_cols);
    
    // Allocate row pointer
    IndexType *cpt = new_array<IndexType>(C.num_rows + 1);
    IndexType c_nnz = 0;
    IndexType *ccol = nullptr; // Will be allocated inside spgemm_array_symbolic_new
    
    // Symbolic Phase: generate and sort Ccol (optimized version)
    // Note: spgemm_array_symbolic_new will allocate ccol internally after scan
    spgemm_array_symbolic_new<IndexType, ValueType>(arpt, acol, brpt, bcol,
                                                     C.num_rows, C.num_cols,
                                                     cpt, ccol, c_nnz, bin);
    
    C.num_nnzs = c_nnz;
    C.row_offset = cpt;
    C.col_index = ccol;
    
    // Allocate values array (will be filled in numeric phase)
    C.values = new_array<ValueType>(c_nnz);
    
    // Numeric Phase: find position and accumulate (optimized version)
    // Note: sortOutput is ignored since ccol is already sorted from symbolic phase
    spgemm_array_numeric_new<sortOutput, IndexType, ValueType>(arpt, acol, aval,
                                                               brpt, bcol, bval,
                                                               C.num_rows, C.num_cols,
                                                               cpt, ccol, C.values, bin);
    
    // Cleanup
    delete bin;
}

/**
 * @brief SPA-based array row-wise SpGEMM implementation
 *        Uses Sparse Accumulator (SPA) for O(1) access in numeric phase
 */
template <bool sortOutput, typename IndexType, typename ValueType>
void LeSpGEMM_spa_rowwise(const CSR_Matrix<IndexType, ValueType> &A,
                          const CSR_Matrix<IndexType, ValueType> &B,
                          CSR_Matrix<IndexType, ValueType> &C)
{
    // Sanity checks
    assert(A.num_cols == B.num_rows);
    
    // Initialize output matrix
    C.num_rows = A.num_rows;
    C.num_cols = B.num_cols;
    C.num_nnzs = 0;
    C.kernel_flag = 3; // SPA-based array method
    C.tag = 0;
    C.partition = nullptr;
    
    // Adapt field names: CSR_Matrix uses row_offset/col_index/values
    // Internal functions expect arpt/acol/aval, brpt/bcol/bval
    const IndexType *arpt = A.row_offset;
    const IndexType *acol = A.col_index;
    const ValueType *aval = A.values;
    
    const IndexType *brpt = B.row_offset;
    const IndexType *bcol = B.col_index;
    const ValueType *bval = B.values;
    
    // Create BIN for load balancing
    SpGEMM_BIN<IndexType, ValueType> *bin = new SpGEMM_BIN<IndexType, ValueType>(A.num_rows, MIN_HT_S);
    
    // Set max bin (calls set_intprod_num, set_rows_offset, set_bin_id)
    bin->set_max_bin(arpt, acol, brpt, C.num_rows, C.num_cols);
    
    // Allocate row pointer
    IndexType *cpt = new_array<IndexType>(C.num_rows + 1);
    IndexType c_nnz = 0;
    IndexType *ccol = nullptr; // Will be allocated inside spgemm_array_symbolic_new
    
    // Symbolic Phase: generate and sort Ccol (same as array_rowwise_new)
    spgemm_array_symbolic_new<IndexType, ValueType>(arpt, acol, brpt, bcol,
                                                     C.num_rows, C.num_cols,
                                                     cpt, ccol, c_nnz, bin);
    
    C.num_nnzs = c_nnz;
    C.row_offset = cpt;
    C.col_index = ccol;
    
    // Allocate values array (will be filled in numeric phase)
    C.values = new_array<ValueType>(c_nnz);
    
    // Numeric Phase: use SPA for O(1) direct access (no binary search)
    spgemm_spa_numeric<sortOutput, IndexType, ValueType>(arpt, acol, aval,
                                     brpt, bcol, bval,
                                     C.num_rows, C.num_cols,
                                     cpt, ccol, C.values, bin);
    
    // Cleanup
    delete bin;
}

template <bool sortOutput, typename IndexType, typename ValueType>
void LeSpGEMM_hash_FLength(const CSR_FlengthCluster<IndexType, ValueType> &A_cluster,
                           const CSR_Matrix<IndexType, ValueType> &B,
                           CSR_FlengthCluster<IndexType, ValueType> &C_cluster)
{
    // Create BIN for cluster-level load balancing
    SpGEMM_BIN_FlengthCluster<IndexType, ValueType> *bin = 
        new SpGEMM_BIN_FlengthCluster<IndexType, ValueType>(A_cluster.rows, A_cluster.cluster_sz, MIN_HT_S);
    
    // Sanity checks
    assert(A_cluster.cols == B.num_rows);
    
    // Initialize output cluster matrix
    C_cluster.csr_rows = A_cluster.csr_rows;
    C_cluster.rows = A_cluster.rows;
    C_cluster.cols = B.num_cols;
    C_cluster.cluster_sz = A_cluster.cluster_sz;
    
    // Adapt field names for B matrix
    const IndexType *brpt = B.row_offset;
    const IndexType *bcol = B.col_index;
    const ValueType *bval = B.values;
    
    // Set max bin (calls set_intprod_num, set_clusters_offset, set_bin_id)
    // Matching reference implementation: set_max_bin(a.rowptr, a.colids, b.rowptr, c.cols)
    bin->set_max_bin(A_cluster.rowptr, A_cluster.colids, brpt, C_cluster.cols);
    
    // Create hash table (thread local)
    bin->create_local_hash_table(C_cluster.cols);
    
    // Allocate cluster pointer (for output CSR_FlengthCluster matrix C)
    C_cluster.rowptr = new_array<IndexType>(C_cluster.rows + 1);
    
    // Symbolic Phase (matching reference HashSpGEMMCluster)
    spgemm_Flength_hash_symbolic_omp_lb<IndexType, ValueType>(
        A_cluster.rowptr, A_cluster.colids, brpt, bcol,
        C_cluster.rows, C_cluster.cols,
        C_cluster.rowptr, C_cluster.nnzc, bin);
    
    // Re-adjust bin_id after symbolic phase (to reduce hashtable size)
    bin->set_bin_id(C_cluster.cols, bin->min_ht_size);
    
    // Allocate column indices and values (will be filled in numeric phase)
    C_cluster.colids = new_array<IndexType>(C_cluster.nnzc);
    C_cluster.values = new_array<ValueType>(C_cluster.nnzc * C_cluster.cluster_sz);
    
    // Numeric Phase (sorting is handled inside numeric phase based on sortOutput template parameter)
    spgemm_Flength_hash_numeric_omp_lb<sortOutput, IndexType, ValueType>(
        A_cluster.rowptr, A_cluster.colids, A_cluster.values,
        brpt, bcol, bval,
        C_cluster.rows, C_cluster.cols,
        C_cluster.rowptr, C_cluster.colids, C_cluster.values, bin, 
        C_cluster.csr_rows, C_cluster.cluster_sz);
    
    // Set Matrix_Features fields
    // num_rows should be the actual matrix rows (csr_rows), not the number of clusters (rows)
    C_cluster.num_rows = C_cluster.csr_rows;
    C_cluster.num_cols = C_cluster.cols;
    
    // Cleanup
    delete bin;
}

template <bool sortOutput, typename IndexType, typename ValueType>
void LeSpGEMM_array_FLength(const CSR_FlengthCluster<IndexType, ValueType> &A_cluster,
                            const CSR_Matrix<IndexType, ValueType> &B,
                            CSR_FlengthCluster<IndexType, ValueType> &C_cluster)
{
    // Create BIN for cluster-level load balancing
    SpGEMM_BIN_FlengthCluster<IndexType, ValueType> *bin = 
        new SpGEMM_BIN_FlengthCluster<IndexType, ValueType>(A_cluster.rows, A_cluster.cluster_sz, MIN_HT_S);
    
    // Sanity checks
    assert(A_cluster.cols == B.num_rows);
    
    // Initialize output cluster matrix
    C_cluster.csr_rows = A_cluster.csr_rows;
    C_cluster.rows = A_cluster.rows;
    C_cluster.cols = B.num_cols;
    C_cluster.cluster_sz = A_cluster.cluster_sz;
    
    // Adapt field names for B matrix
    const IndexType *brpt = B.row_offset;
    const IndexType *bcol = B.col_index;
    const ValueType *bval = B.values;
    
    // Set max bin (calls set_intprod_num, set_clusters_offset, set_bin_id)
    // Matching reference implementation: set_max_bin(a.rowptr, a.colids, b.rowptr, c.cols)
    bin->set_max_bin(A_cluster.rowptr, A_cluster.colids, brpt, C_cluster.cols);
    
    // Note: Array-based method does NOT need create_local_hash_table
    // Each cluster will use sorted arrays instead of hash tables
    
    // Allocate cluster pointer (for output CSR_FlengthCluster matrix C)
    C_cluster.rowptr = new_array<IndexType>(C_cluster.rows + 1);
    
    // Symbolic Phase: generate and sort Ccolids (optimized version)
    // Note: spgemm_Flength_array_symbolic_new will allocate ccolids internally after scan
    spgemm_Flength_array_symbolic_new<IndexType, ValueType>(
        A_cluster, brpt, bcol,
        C_cluster.rows, C_cluster.cols,
        C_cluster.rowptr, C_cluster.colids, C_cluster.nnzc, bin);
    
    // Allocate values array (will be filled in numeric phase)
    // Each column needs cluster_sz values (one per row in cluster)
    C_cluster.values = new_array<ValueType>(C_cluster.nnzc * C_cluster.cluster_sz);
    
    // Numeric Phase: find position and accumulate (optimized version)
    // Note: sortOutput is ignored since ccolids is already sorted from symbolic phase
    spgemm_Flength_array_numeric_new<sortOutput, IndexType, ValueType>(
        A_cluster, brpt, bcol, bval,
        C_cluster.rows, C_cluster.cols,
        C_cluster.rowptr, C_cluster.colids, C_cluster.values,
        C_cluster.cluster_sz, bin);
    
    // Set Matrix_Features fields
    // num_rows should be the actual matrix rows (csr_rows), not the number of clusters (rows)
    C_cluster.num_rows = C_cluster.csr_rows;
    C_cluster.num_cols = C_cluster.cols;
    
    // Cleanup
    delete bin;
}

template <bool sortOutput, typename IndexType, typename ValueType>
void LeSpGEMM_FLength(const CSR_FlengthCluster<IndexType, ValueType> &A_cluster,
                      const CSR_Matrix<IndexType, ValueType> &B,
                      CSR_FlengthCluster<IndexType, ValueType> &C_cluster,
                      int kernel_flag)
{
    // Select implementation based on kernel_flag
    // kernel_flag = 1: Hash-based cluster-wise method (default)
    // kernel_flag = 2: Array-based cluster-wise method (HSMU-SpGEMM inspired, sorted arrays)
    if (kernel_flag == 1) {
        LeSpGEMM_hash_FLength<sortOutput, IndexType, ValueType>(A_cluster, B, C_cluster);
    } else if (kernel_flag == 2) {
        // Array-based cluster-wise method (HSMU-SpGEMM inspired)
        LeSpGEMM_array_FLength<sortOutput, IndexType, ValueType>(A_cluster, B, C_cluster);
    } else {
        // Default to hash-based cluster-wise method
        LeSpGEMM_hash_FLength<sortOutput, IndexType, ValueType>(A_cluster, B, C_cluster);
    }
}

// ============================================================================
// Variable-length Cluster-wise SpGEMM Implementation
// ============================================================================

template <bool sortOutput, typename IndexType, typename ValueType>
void LeSpGEMM_hash_VLength(const CSR_VlengthCluster<IndexType, ValueType> &A_cluster,
                           const CSR_Matrix<IndexType, ValueType> &B,
                           CSR_VlengthCluster<IndexType, ValueType> &C_cluster)
{
    // Create BIN for cluster-level load balancing
    // Matching reference: BIN_VlengthCluster<IT, NT> bin(a.rows, MIN_HT_S, a.cluster_sz);
    SpGEMM_BIN_VlengthCluster<IndexType, ValueType> *bin = 
        new SpGEMM_BIN_VlengthCluster<IndexType, ValueType>(A_cluster.rows, MIN_HT_S, A_cluster.cluster_sz);
    
    // Sanity checks
    assert(A_cluster.cols == B.num_rows);
    
    // Initialize output cluster matrix (matching reference HashSpGEMMVLCluster)
    C_cluster.csr_rows = A_cluster.csr_rows;
    C_cluster.rows = A_cluster.rows;
    C_cluster.cols = B.num_cols;
    
    // Allocate and copy cluster_sz array (matching reference)
    C_cluster.cluster_sz = new_array<IndexType>(A_cluster.rows);
    std::memcpy(C_cluster.cluster_sz, A_cluster.cluster_sz, sizeof(IndexType) * A_cluster.rows);
    
    // Adapt field names for B matrix
    const IndexType *brpt = B.row_offset;
    const IndexType *bcol = B.col_index;
    const ValueType *bval = B.values;
    
    // Set max bin (calls set_intprod_num, set_clusters_offset, set_bin_id)
    // Matching reference implementation: bin.set_max_bin(a.rowptr, a.colids, b.rowptr, c.cols)
    bin->set_max_bin(A_cluster.rowptr, A_cluster.colids, brpt, C_cluster.cols);
    
    // Create hash table (thread local)
    // Matching reference: bin.create_local_hash_table(c.cols)
    bin->create_local_hash_table(C_cluster.cols);
    
    // Allocate cluster pointers (for output CSR_VlengthCluster matrix C)
    // Matching reference: c.rowptr = my_malloc<IT>(c.rows + 1); c.rowptr_val = my_malloc<IT>(c.rows + 1);
    C_cluster.rowptr = new_array<IndexType>(C_cluster.rows + 1);
    C_cluster.rowptr_val = new_array<IndexType>(C_cluster.rows + 1);
    
    // Symbolic Phase (matching reference HashSpGEMMVLCluster::hash_symbolic_vlcluster)
    spgemm_Vlength_hash_symbolic_omp_lb<IndexType, ValueType>(
        A_cluster.rowptr, A_cluster.colids, brpt, bcol,
        C_cluster.rows,
        C_cluster.rowptr, C_cluster.rowptr_val,
        C_cluster.nnzc, C_cluster.nnzv, bin);
    
    // Re-adjust bin_id after symbolic phase (to reduce hashtable size)
    // Matching reference: bin.set_bin_id(c.cols, bin.min_ht_size)
    bin->set_bin_id(C_cluster.cols, bin->min_ht_size);
    
    // Allocate column indices and values (will be filled in numeric phase)
    // Matching reference: c.colids = my_malloc<IT>(c.nnzc); c.values = my_malloc<NT>(c.nnzv);
    C_cluster.colids = new_array<IndexType>(C_cluster.nnzc);
    C_cluster.values = new_array<ValueType>(C_cluster.nnzv);
    
    // Numeric Phase (sorting is handled inside numeric phase based on sortOutput template parameter)
    // Matching reference: hash_numeric_vlcluster_V1<sortOutput>(...)
    spgemm_Vlength_hash_numeric_omp_lb<sortOutput, IndexType, ValueType>(
        A_cluster.rowptr, A_cluster.rowptr_val, A_cluster.colids, A_cluster.values,
        brpt, bcol, bval,
        C_cluster.rows, C_cluster.cols,
        C_cluster.rowptr, C_cluster.rowptr_val,
        C_cluster.colids, C_cluster.values, bin, C_cluster.nnzc, C_cluster.cluster_sz);
    
    // Set Matrix_Features fields
    C_cluster.num_rows = C_cluster.csr_rows;
    C_cluster.num_cols = C_cluster.cols;
    // C_cluster.num_nnzs = C_cluster.nnzv;  // Total number of stored values
    
    // Calculate max_cluster_sz (for compatibility)
    C_cluster.max_cluster_sz = 0;
    for (IndexType i = 0; i < C_cluster.rows; i++) {
        if (C_cluster.cluster_sz[i] > C_cluster.max_cluster_sz) {
            C_cluster.max_cluster_sz = C_cluster.cluster_sz[i];
        }
    }
    
    // Cleanup
    delete bin;
}

template <bool sortOutput, typename IndexType, typename ValueType>
void LeSpGEMM_array_VLength(const CSR_VlengthCluster<IndexType, ValueType> &A_cluster,
                             const CSR_Matrix<IndexType, ValueType> &B,
                             CSR_VlengthCluster<IndexType, ValueType> &C_cluster)
{
    // Create BIN for cluster-level load balancing
    // Matching reference: BIN_VlengthCluster<IT, NT> bin(a.rows, MIN_HT_S, a.cluster_sz);
    SpGEMM_BIN_VlengthCluster<IndexType, ValueType> *bin = 
        new SpGEMM_BIN_VlengthCluster<IndexType, ValueType>(A_cluster.rows, MIN_HT_S, A_cluster.cluster_sz);
    
    // Sanity checks
    assert(A_cluster.cols == B.num_rows);
    
    // Initialize output cluster matrix
    C_cluster.csr_rows = A_cluster.csr_rows;
    C_cluster.rows = A_cluster.rows;
    C_cluster.cols = B.num_cols;
    
    // Allocate and copy cluster_sz array (matching reference)
    C_cluster.cluster_sz = new_array<IndexType>(A_cluster.rows);
    std::memcpy(C_cluster.cluster_sz, A_cluster.cluster_sz, sizeof(IndexType) * A_cluster.rows);
    
    // Adapt field names for B matrix
    const IndexType *brpt = B.row_offset;
    const IndexType *bcol = B.col_index;
    const ValueType *bval = B.values;
    
    // Set max bin (calls set_intprod_num, set_clusters_offset, set_bin_id)
    // Matching reference implementation: bin.set_max_bin(a.rowptr, a.colids, b.rowptr, c.cols)
    bin->set_max_bin(A_cluster.rowptr, A_cluster.colids, brpt, C_cluster.cols);
    
    // Note: Array-based method does NOT need create_local_hash_table
    // Each cluster will use sorted arrays instead of hash tables
    
    // Allocate cluster pointers (for output CSR_VlengthCluster matrix C)
    // Matching reference: c.rowptr = my_malloc<IT>(c.rows + 1); c.rowptr_val = my_malloc<IT>(c.rows + 1);
    C_cluster.rowptr = new_array<IndexType>(C_cluster.rows + 1);
    C_cluster.rowptr_val = new_array<IndexType>(C_cluster.rows + 1);
    
    // Symbolic Phase: generate and sort Ccolids (optimized version)
    // Note: spgemm_Vlength_array_symbolic_new will allocate ccolids internally after scan
    spgemm_Vlength_array_symbolic_new<IndexType, ValueType>(
        A_cluster, brpt, bcol,
        C_cluster.rows, C_cluster.cols,
        C_cluster.rowptr, C_cluster.rowptr_val,
        C_cluster.colids, C_cluster.nnzc, C_cluster.nnzv, bin);
    
    // Allocate values array (will be filled in numeric phase)
    // Length: c_nnzv (computed from weighted scan in symbolic phase)
    C_cluster.values = new_array<ValueType>(C_cluster.nnzv);
    
    // Numeric Phase: find position and accumulate (optimized version)
    // Note: sortOutput is ignored since ccolids is already sorted from symbolic phase
    spgemm_Vlength_array_numeric_new<sortOutput, IndexType, ValueType>(
        A_cluster, brpt, bcol, bval,
        C_cluster.rows, C_cluster.cols,
        C_cluster.rowptr, C_cluster.rowptr_val,
        C_cluster.colids, C_cluster.values,
        C_cluster.cluster_sz, bin);
    
    // Set Matrix_Features fields
    C_cluster.num_rows = C_cluster.csr_rows;
    C_cluster.num_cols = C_cluster.cols;
    // C_cluster.num_nnzs = C_cluster.nnzv;  // Total number of stored values
    
    // Calculate max_cluster_sz (for compatibility)
    C_cluster.max_cluster_sz = 0;
    for (IndexType i = 0; i < C_cluster.rows; i++) {
        if (C_cluster.cluster_sz[i] > C_cluster.max_cluster_sz) {
            C_cluster.max_cluster_sz = C_cluster.cluster_sz[i];
        }
    }
    
    // Cleanup
    delete bin;
}

template <bool sortOutput, typename IndexType, typename ValueType>
void LeSpGEMM_VLength(const CSR_VlengthCluster<IndexType, ValueType> &A_cluster,
                      const CSR_Matrix<IndexType, ValueType> &B,
                      CSR_VlengthCluster<IndexType, ValueType> &C_cluster,
                      int kernel_flag)
{
    // Select implementation based on kernel_flag
    // kernel_flag = 1: Hash-based cluster-wise method (default)
    // kernel_flag = 2: Array-based cluster-wise method (HSMU-SpGEMM inspired, sorted arrays)
    if (kernel_flag == 1) {
        LeSpGEMM_hash_VLength<sortOutput, IndexType, ValueType>(A_cluster, B, C_cluster);
    } else if (kernel_flag == 2) {
        // Array-based cluster-wise method (HSMU-SpGEMM inspired)
        LeSpGEMM_array_VLength<sortOutput, IndexType, ValueType>(A_cluster, B, C_cluster);
    } else {
        // Default to hash-based cluster-wise method
        LeSpGEMM_hash_VLength<sortOutput, IndexType, ValueType>(A_cluster, B, C_cluster);
    }
}

template <bool sortOutput, typename IndexType, typename ValueType>
void LeSpGEMM(const CSR_Matrix<IndexType, ValueType> &A,
              const CSR_Matrix<IndexType, ValueType> &B,
              CSR_Matrix<IndexType, ValueType> &C,
              int kernel_flag)
{
    // Select implementation based on kernel_flag
    // kernel_flag = 1: Hash-based row-wise method (default)
    // kernel_flag = 2: Array-based row-wise method (HSMU-SpGEMM inspired, optimized version with pre-sorted Ccol, binary search)
    // kernel_flag = 3: SPA-based array row-wise method (HSMU-SpGEMM inspired, Sparse Accumulator for O(1) access)
    // Note: For cluster-wise methods, use LeSpGEMM_FLength instead
    
    if (kernel_flag == 1) {
        LeSpGEMM_hash_rowwise<sortOutput, IndexType, ValueType>(A, B, C, kernel_flag);
    } else if (kernel_flag == 2) {
        // Use optimized array-based method (HSMU-SpGEMM inspired, pre-sorted Ccol in symbolic phase, and compute values using binary search)
        LeSpGEMM_array_rowwise_new<sortOutput, IndexType, ValueType>(A, B, C);
    } else if (kernel_flag == 3) {
        // Use SPA-based array method (HSMU-SpGEMM inspired, Sparse Accumulator for O(1) direct access, fastest)
        LeSpGEMM_spa_rowwise<sortOutput, IndexType, ValueType>(A, B, C);
    } else {
        // Default to hash-based row-wise method
        LeSpGEMM_hash_rowwise<sortOutput, IndexType, ValueType>(A, B, C, kernel_flag);
    }
}

// Explicit template instantiations for SpGEMM (only int64_t for IndexType)
template long long int get_spgemm_flop<int64_t, float>(
    const CSR_Matrix<int64_t, float>&, const CSR_Matrix<int64_t, float>&);
template long long int get_spgemm_flop<int64_t, double>(
    const CSR_Matrix<int64_t, double>&, const CSR_Matrix<int64_t, double>&);

// LeSpGEMM instantiations (sortOutput = true and false)
template void LeSpGEMM<true, int64_t, float>(
    const CSR_Matrix<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_Matrix<int64_t, float>&, int);
template void LeSpGEMM<false, int64_t, float>(
    const CSR_Matrix<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_Matrix<int64_t, float>&, int);
template void LeSpGEMM<true, int64_t, double>(
    const CSR_Matrix<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_Matrix<int64_t, double>&, int);
template void LeSpGEMM<false, int64_t, double>(
    const CSR_Matrix<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_Matrix<int64_t, double>&, int);

// LeSpGEMM_hash_rowwise instantiations (sortOutput = true and false)
template void LeSpGEMM_hash_rowwise<true, int64_t, float>(
    const CSR_Matrix<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_Matrix<int64_t, float>&, int);
template void LeSpGEMM_hash_rowwise<false, int64_t, float>(
    const CSR_Matrix<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_Matrix<int64_t, float>&, int);
template void LeSpGEMM_hash_rowwise<true, int64_t, double>(
    const CSR_Matrix<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_Matrix<int64_t, double>&, int);
template void LeSpGEMM_hash_rowwise<false, int64_t, double>(
    const CSR_Matrix<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_Matrix<int64_t, double>&, int);

// LeSpGEMM_array_rowwise_new instantiations (sortOutput = true and false)
template void LeSpGEMM_array_rowwise_new<true, int64_t, float>(
    const CSR_Matrix<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_Matrix<int64_t, float>&);
template void LeSpGEMM_array_rowwise_new<false, int64_t, float>(
    const CSR_Matrix<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_Matrix<int64_t, float>&);
template void LeSpGEMM_array_rowwise_new<true, int64_t, double>(
    const CSR_Matrix<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_Matrix<int64_t, double>&);
template void LeSpGEMM_array_rowwise_new<false, int64_t, double>(
    const CSR_Matrix<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_Matrix<int64_t, double>&);

// LeSpGEMM_spa_rowwise instantiations (sortOutput = true and false)
template void LeSpGEMM_spa_rowwise<true, int64_t, float>(
    const CSR_Matrix<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_Matrix<int64_t, float>&);
template void LeSpGEMM_spa_rowwise<false, int64_t, float>(
    const CSR_Matrix<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_Matrix<int64_t, float>&);
template void LeSpGEMM_spa_rowwise<true, int64_t, double>(
    const CSR_Matrix<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_Matrix<int64_t, double>&);
template void LeSpGEMM_spa_rowwise<false, int64_t, double>(
    const CSR_Matrix<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_Matrix<int64_t, double>&);

// Note: LeSpGEMM_array_rowwise_new and LeSpGEMM_spa_rowwise are implicitly instantiated through LeSpGEMM
// No explicit instantiation needed to avoid duplication

// LeSpGEMM_hash_FLength instantiations (sortOutput = true and false)
template void LeSpGEMM_hash_FLength<true, int64_t, float>(
    const CSR_FlengthCluster<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_FlengthCluster<int64_t, float>&);
template void LeSpGEMM_hash_FLength<false, int64_t, float>(
    const CSR_FlengthCluster<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_FlengthCluster<int64_t, float>&);
template void LeSpGEMM_hash_FLength<true, int64_t, double>(
    const CSR_FlengthCluster<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_FlengthCluster<int64_t, double>&);
template void LeSpGEMM_hash_FLength<false, int64_t, double>(
    const CSR_FlengthCluster<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_FlengthCluster<int64_t, double>&);

// LeSpGEMM_array_FLength instantiations (sortOutput = true and false)
template void LeSpGEMM_array_FLength<true, int64_t, float>(
    const CSR_FlengthCluster<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_FlengthCluster<int64_t, float>&);
template void LeSpGEMM_array_FLength<false, int64_t, float>(
    const CSR_FlengthCluster<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_FlengthCluster<int64_t, float>&);
template void LeSpGEMM_array_FLength<true, int64_t, double>(
    const CSR_FlengthCluster<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_FlengthCluster<int64_t, double>&);
template void LeSpGEMM_array_FLength<false, int64_t, double>(
    const CSR_FlengthCluster<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_FlengthCluster<int64_t, double>&);

// LeSpGEMM_FLength instantiations (sortOutput = true and false)
template void LeSpGEMM_FLength<true, int64_t, float>(
    const CSR_FlengthCluster<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_FlengthCluster<int64_t, float>&, int);
template void LeSpGEMM_FLength<false, int64_t, float>(
    const CSR_FlengthCluster<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_FlengthCluster<int64_t, float>&, int);
template void LeSpGEMM_FLength<true, int64_t, double>(
    const CSR_FlengthCluster<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_FlengthCluster<int64_t, double>&, int);
template void LeSpGEMM_FLength<false, int64_t, double>(
    const CSR_FlengthCluster<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_FlengthCluster<int64_t, double>&, int);

// LeSpGEMM_hash_VLength instantiations (sortOutput = true and false)
template void LeSpGEMM_hash_VLength<true, int64_t, float>(
    const CSR_VlengthCluster<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_VlengthCluster<int64_t, float>&);
template void LeSpGEMM_hash_VLength<false, int64_t, float>(
    const CSR_VlengthCluster<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_VlengthCluster<int64_t, float>&);
template void LeSpGEMM_hash_VLength<true, int64_t, double>(
    const CSR_VlengthCluster<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_VlengthCluster<int64_t, double>&);
template void LeSpGEMM_hash_VLength<false, int64_t, double>(
    const CSR_VlengthCluster<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_VlengthCluster<int64_t, double>&);

// LeSpGEMM_array_VLength instantiations (sortOutput = true and false)
template void LeSpGEMM_array_VLength<true, int64_t, float>(
    const CSR_VlengthCluster<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_VlengthCluster<int64_t, float>&);
template void LeSpGEMM_array_VLength<false, int64_t, float>(
    const CSR_VlengthCluster<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_VlengthCluster<int64_t, float>&);
template void LeSpGEMM_array_VLength<true, int64_t, double>(
    const CSR_VlengthCluster<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_VlengthCluster<int64_t, double>&);
template void LeSpGEMM_array_VLength<false, int64_t, double>(
    const CSR_VlengthCluster<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_VlengthCluster<int64_t, double>&);

// LeSpGEMM_VLength instantiations (sortOutput = true and false)
template void LeSpGEMM_VLength<true, int64_t, float>(
    const CSR_VlengthCluster<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_VlengthCluster<int64_t, float>&, int);
template void LeSpGEMM_VLength<false, int64_t, float>(
    const CSR_VlengthCluster<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_VlengthCluster<int64_t, float>&, int);
template void LeSpGEMM_VLength<true, int64_t, double>(
    const CSR_VlengthCluster<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_VlengthCluster<int64_t, double>&, int);
template void LeSpGEMM_VLength<false, int64_t, double>(
    const CSR_VlengthCluster<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_VlengthCluster<int64_t, double>&, int);

