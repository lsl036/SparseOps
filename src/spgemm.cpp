/**
 * @file spgemm.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief SpGEMM main interface implementation
 * @version 0.1
 * @date 2024
 */

#include "../include/spgemm.h"
#include "../include/sparse_operation.h"
#include <cassert>

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
    
    return flops;
}

template <typename IndexType, typename ValueType>
void LeSpGEMM_rowwise(const CSR_Matrix<IndexType, ValueType> &A,
                      const CSR_Matrix<IndexType, ValueType> &B,
                      CSR_Matrix<IndexType, ValueType> &C,
                      bool sort_output,
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
    
    IndexType *cpt = nullptr;
    IndexType *ccol = nullptr;
    ValueType *cval = nullptr;
    IndexType c_nnz = 0;
    
    // SpGEMM_BIN for load balancing (only used in kernel_flag == 2)
    SpGEMM_BIN<IndexType, ValueType> *bin = nullptr;
    
    if (kernel_flag == 2) {
        // Create BIN for load balancing
        bin = new SpGEMM_BIN<IndexType, ValueType>(A.num_rows, MIN_HT_S);
        bin->set_max_bin(arpt, acol, brpt, C.num_rows, C.num_cols);
        bin->set_bin_id(A.num_rows, C.num_cols, MIN_HT_S);
    }
    
    // Symbolic phase: compute structure of C
    if (kernel_flag == 1) {
        spgemm_hash_symbolic_omp<IndexType, ValueType>(arpt, acol, brpt, bcol,
                                  C.num_rows, C.num_cols,
                                  cpt, ccol, c_nnz);
    } else if (kernel_flag == 2) {
        spgemm_hash_symbolic_omp_lb<IndexType, ValueType>(arpt, acol, brpt, bcol,
                                      C.num_rows, C.num_cols,
                                      cpt, ccol, c_nnz, bin);
    } else {
        // Default: OpenMP (kernel_flag == 1)
        spgemm_hash_symbolic_omp<IndexType, ValueType>(arpt, acol, brpt, bcol,
                                  C.num_rows, C.num_cols,
                                  cpt, ccol, c_nnz);
    }
    
    C.num_nnzs = c_nnz;
    C.row_offset = cpt;
    C.col_index = ccol;
    C.values = new_array<ValueType>(c_nnz);
    
    // Numeric phase: compute values of C
    if (kernel_flag == 1) {
        spgemm_hash_numeric_omp<IndexType, ValueType>(arpt, acol, aval,
                                 brpt, bcol, bval,
                                 C.num_rows, C.num_cols,
                                 cpt, ccol, C.values);
    } else if (kernel_flag == 2) {
        spgemm_hash_numeric_omp_lb<IndexType, ValueType>(arpt, acol, aval,
                                    brpt, bcol, bval,
                                    C.num_rows, C.num_cols,
                                    cpt, ccol, C.values, bin);
    } else {
        // Default: OpenMP (kernel_flag == 1)
        spgemm_hash_numeric_omp<IndexType, ValueType>(arpt, acol, aval,
                                 brpt, bcol, bval,
                                 C.num_rows, C.num_cols,
                                 cpt, ccol, C.values);
    }
    
    // Sort columns if requested
    if (sort_output) {
        sort_csr_columns(C.num_rows, C.row_offset, C.col_index, C.values);
    }
    
    // Cleanup
    if (bin != nullptr) {
        delete bin;
    }
}

template <typename IndexType, typename ValueType>
void LeSpGEMM(const CSR_Matrix<IndexType, ValueType> &A,
              const CSR_Matrix<IndexType, ValueType> &B,
              CSR_Matrix<IndexType, ValueType> &C,
              bool sort_output,
              int kernel_flag)
{
    // Currently only row-wise implementation is available
    // Future: can add cluster-wise implementation here
    // if (use_cluster_wise) {
    //     LeSpGEMM_clusterwise(A, B, C, sort_output, kernel_flag);
    // } else {
        LeSpGEMM_rowwise(A, B, C, sort_output, kernel_flag);
    // }
}

// Explicit template instantiations
template long long int get_spgemm_flop<int, float>(
    const CSR_Matrix<int, float>&, const CSR_Matrix<int, float>&);
template long long int get_spgemm_flop<int, double>(
    const CSR_Matrix<int, double>&, const CSR_Matrix<int, double>&);
template long long int get_spgemm_flop<long long, float>(
    const CSR_Matrix<long long, float>&, const CSR_Matrix<long long, float>&);
template long long int get_spgemm_flop<long long, double>(
    const CSR_Matrix<long long, double>&, const CSR_Matrix<long long, double>&);

template void LeSpGEMM<int, float>(
    const CSR_Matrix<int, float>&, const CSR_Matrix<int, float>&,
    CSR_Matrix<int, float>&, bool, int);
template void LeSpGEMM<int, double>(
    const CSR_Matrix<int, double>&, const CSR_Matrix<int, double>&,
    CSR_Matrix<int, double>&, bool, int);
template void LeSpGEMM<long long, float>(
    const CSR_Matrix<long long, float>&, const CSR_Matrix<long long, float>&,
    CSR_Matrix<long long, float>&, bool, int);
template void LeSpGEMM<long long, double>(
    const CSR_Matrix<long long, double>&, const CSR_Matrix<long long, double>&,
    CSR_Matrix<long long, double>&, bool, int);

template void LeSpGEMM_rowwise<int, float>(
    const CSR_Matrix<int, float>&, const CSR_Matrix<int, float>&,
    CSR_Matrix<int, float>&, bool, int);
template void LeSpGEMM_rowwise<int, double>(
    const CSR_Matrix<int, double>&, const CSR_Matrix<int, double>&,
    CSR_Matrix<int, double>&, bool, int);
template void LeSpGEMM_rowwise<long long, float>(
    const CSR_Matrix<long long, float>&, const CSR_Matrix<long long, float>&,
    CSR_Matrix<long long, float>&, bool, int);
template void LeSpGEMM_rowwise<long long, double>(
    const CSR_Matrix<long long, double>&, const CSR_Matrix<long long, double>&,
    CSR_Matrix<long long, double>&, bool, int);

