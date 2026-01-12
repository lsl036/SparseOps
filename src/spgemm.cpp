/**
 * @file spgemm.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief SpGEMM main interface implementation
 * @version 0.1
 * @date 2024
 */

#include "../include/spgemm.h"
#include "../include/spgemm_array.h"
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
void LeSpGEMM(const CSR_Matrix<IndexType, ValueType> &A,
              const CSR_Matrix<IndexType, ValueType> &B,
              CSR_Matrix<IndexType, ValueType> &C,
              int kernel_flag)
{
    // Select implementation based on kernel_flag
    // kernel_flag = 1: Hash-based row-wise method (default)
    // kernel_flag = 2: Array-based row-wise method (HSMU-SpGEMM inspired)
    // Future: kernel_flag = 3: Cluster-wise method, etc.
    
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

