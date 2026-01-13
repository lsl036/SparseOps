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
#include "../include/sparse_conversion.h"
#include "../include/sparse_operation.h"
#include <cassert>
#include <cstdint>

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

template <bool sortOutput, typename IndexType, typename ValueType>
void LeSpGEMM_array_rowwise(const CSR_Matrix<IndexType, ValueType> &A,
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
    // Note: Array-based method only uses row_nz and rows_offset, ignores bin_id
    SpGEMM_BIN<IndexType, ValueType> *bin = new SpGEMM_BIN<IndexType, ValueType>(A.num_rows, MIN_HT_S);
    
    // Set max bin (calls set_intprod_num, set_rows_offset, set_bin_id)
    // Note: bin_id is set but not used in array-based method
    bin->set_max_bin(arpt, acol, brpt, C.num_rows, C.num_cols);
    
    // Note: Array-based method does NOT need create_local_hash_table
    // Each row will allocate its own array of size row_nz[i] (exact size, no padding)
    
    // Allocate row pointer (matching reference RowSpGEMM: c.rowptr = my_malloc<IT>(c.rows + 1))
    IndexType *cpt = new_array<IndexType>(C.num_rows + 1);
    IndexType c_nnz = 0;
    
    // Symbolic Phase: count unique columns per row using sorted arrays
    spgemm_array_symbolic_omp_lb<IndexType, ValueType>(arpt, acol, brpt, bcol,
                                    C.num_rows, C.num_cols,
                                    cpt, c_nnz, bin);
    
    // Note: Array-based method does NOT need set_bin_id adjustment
    // because it doesn't use bin_id for array sizing
    
    C.num_nnzs = c_nnz;
    C.row_offset = cpt;
    
    // Allocate column indices and values (will be filled in numeric phase)
    C.col_index = new_array<IndexType>(c_nnz);
    C.values = new_array<ValueType>(c_nnz);
    
    // Numeric Phase: compute values using sorted arrays
    // Note: sortOutput template parameter is passed through (array is already sorted)
    spgemm_array_numeric_omp_lb<sortOutput, IndexType, ValueType>(arpt, acol, aval,
                                      brpt, bcol, bval,
                                      C.num_rows, C.num_cols,
                                      cpt, C.col_index, C.values, bin);
    
    // Cleanup
    delete bin;
}

template <bool sortOutput, typename IndexType, typename ValueType>
void LeSpGEMM_hash_FLength(const CSR_FlengthCluster<IndexType, ValueType> &A_cluster,
                           const CSR_Matrix<IndexType, ValueType> &B,
                           CSR_FlengthCluster<IndexType, ValueType> &C_cluster)
{
    // Sanity checks
    assert(A_cluster.cols == B.num_rows);
    
    // Initialize output cluster matrix
    C_cluster.csr_rows = A_cluster.csr_rows;
    C_cluster.rows = A_cluster.rows;
    C_cluster.cols = B.num_cols;
    C_cluster.cluster_sz = A_cluster.cluster_sz;
    C_cluster.nnzc = 0;
    C_cluster.tag = 0;
    
    // Adapt field names for B matrix
    const IndexType *brpt = B.row_offset;
    const IndexType *bcol = B.col_index;
    const ValueType *bval = B.values;
    
    // Create BIN for cluster-level load balancing
    SpGEMM_BIN_FlengthCluster<IndexType, ValueType> *bin = 
        new SpGEMM_BIN_FlengthCluster<IndexType, ValueType>(A_cluster.rows, A_cluster.cluster_sz, MIN_HT_S);
    
    // Set max bin (calls set_intprod_num, set_clusters_offset, set_bin_id)
    bin->set_max_bin(A_cluster, B, C_cluster.cols);
    
    // Create hash table (thread local)
    bin->create_local_hash_table(C_cluster.cols);
    
    // Allocate cluster pointer (for output CSR_FlengthCluster matrix C)
    IndexType *crpt = new_array<IndexType>(C_cluster.rows + 1);
    IndexType c_nnzc = 0;
    
    // Symbolic Phase (matching reference HashSpGEMMCluster)
    spgemm_Flength_hash_symbolic_omp_lb<IndexType, ValueType>(
        A_cluster, brpt, bcol,
        C_cluster.rows, C_cluster.cols,
        crpt, c_nnzc, bin);
    
    // Re-adjust bin_id after symbolic phase (to reduce hashtable size)
    bin->set_bin_id(C_cluster.rows, C_cluster.cols, bin->min_ht_size);
    
    C_cluster.nnzc = c_nnzc;
    C_cluster.rowptr = crpt;
    
    // Allocate column indices and values (will be filled in numeric phase)
    C_cluster.colids = new_array<IndexType>(c_nnzc);
    C_cluster.values = new_array<ValueType>((size_t)c_nnzc * C_cluster.cluster_sz);
    
    // Initialize values array to zero
    std::fill_n(C_cluster.values, (size_t)c_nnzc * C_cluster.cluster_sz, static_cast<ValueType>(0));
    
    // Numeric Phase (sorting is handled inside numeric phase based on sortOutput template parameter)
    spgemm_Flength_hash_numeric_omp_lb<sortOutput, IndexType, ValueType>(
        A_cluster, brpt, bcol, bval,
        C_cluster.rows, C_cluster.cols,
        crpt, C_cluster.colids, C_cluster.values, bin, C_cluster.cluster_sz);
    
    // Set Matrix_Features fields
    C_cluster.num_rows = C_cluster.rows;
    C_cluster.num_cols = C_cluster.cols;
    C_cluster.num_nnzs = C_cluster.nnzc * C_cluster.cluster_sz;  // Total number of stored values
    
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
    // kernel_flag = 2: Array-based cluster-wise method (future)
    
    if (kernel_flag == 1) {
        LeSpGEMM_hash_FLength<sortOutput, IndexType, ValueType>(A_cluster, B, C_cluster);
    } else if (kernel_flag == 2) {
        // Future: Array-based cluster-wise method
        // LeSpGEMM_array_FLength<sortOutput, IndexType, ValueType>(A_cluster, B, C_cluster);
        // For now, default to hash-based
        LeSpGEMM_hash_FLength<sortOutput, IndexType, ValueType>(A_cluster, B, C_cluster);
    } else {
        // Default to hash-based cluster-wise method
        LeSpGEMM_hash_FLength<sortOutput, IndexType, ValueType>(A_cluster, B, C_cluster);
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
    // kernel_flag = 2: Array-based row-wise method (HSMU-SpGEMM inspired)
    // Note: For cluster-wise methods, use LeSpGEMM_FLength instead
    
    if (kernel_flag == 1) {
        LeSpGEMM_hash_rowwise<sortOutput, IndexType, ValueType>(A, B, C, kernel_flag);
    } else if (kernel_flag == 2) {
        LeSpGEMM_array_rowwise<sortOutput, IndexType, ValueType>(A, B, C, kernel_flag);
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

// LeSpGEMM_array_rowwise instantiations (sortOutput = true and false)
template void LeSpGEMM_array_rowwise<true, int64_t, float>(
    const CSR_Matrix<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_Matrix<int64_t, float>&, int);
template void LeSpGEMM_array_rowwise<false, int64_t, float>(
    const CSR_Matrix<int64_t, float>&, const CSR_Matrix<int64_t, float>&,
    CSR_Matrix<int64_t, float>&, int);
template void LeSpGEMM_array_rowwise<true, int64_t, double>(
    const CSR_Matrix<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_Matrix<int64_t, double>&, int);
template void LeSpGEMM_array_rowwise<false, int64_t, double>(
    const CSR_Matrix<int64_t, double>&, const CSR_Matrix<int64_t, double>&,
    CSR_Matrix<int64_t, double>&, int);

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

