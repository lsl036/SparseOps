/**
 * @file baseline_mkl_spgemm.cpp
 * @brief SpGEMM baseline using Intel oneAPI MKL (C API: mkl_sparse_spmm).
 *        Reads two matrices A, B from .mtx files, computes C = A * B.
 *
 * Usage:
 *   ./baseline_mkl_spgemm A.mtx B.mtx
 *   ./baseline_mkl_spgemm A.mtx B.mtx [--iterations N]
 *
 * Requires: ENABLE_MKL=ON, link with MKL.
 */

#ifdef ENABLE_MKL
#include <mkl.h>
#include <mkl_spblas.h>
#endif

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "../include/SpOps.h"
#include "../include/cmdline.h"
#include "../include/sparse_io.h"
#include "../include/timer.h"

static void usage(int argc, char **argv) {
    std::cout << "Usage:\n";
    std::cout << "\t" << argv[0] << " A.mtx B.mtx [--iterations N]\n";
    std::cout << "\tCompute C = A * B using Intel oneAPI MKL (mkl_sparse_spmm).\n";
    std::cout << "\t--iterations  Number of SpGEMM runs for timing (default 10).\n";
}

#ifdef ENABLE_MKL
static int run_mkl_spgemm(int argc, char **argv) {
    char *pathA = nullptr;
    char *pathB = nullptr;
    int n = 0;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            if (n == 0) {
                pathA = argv[i];
                n++;
            } else if (n == 1) {
                pathB = argv[i];
                n++;
                break;
            }
        }
    }
    if (!pathA || !pathB) {
        std::cerr << "Error: need two matrix files A.mtx and B.mtx.\n";
        usage(argc, argv);
        return 2;
    }

    int iterations = 10;
    char *p = get_argval(argc, argv, "iterations");
    if (p) iterations = std::max(1, atoi(p));

    using IndexType = int64_t;
    std::cout << "Reading A: " << pathA << std::endl;
    CSR_Matrix<IndexType, double> A = read_csr_matrix<IndexType, double>(pathA, true);
    std::cout << "  rows=" << A.num_rows << " cols=" << A.num_cols << " nnz=" << A.num_nnzs << std::endl;

    std::cout << "Reading B: " << pathB << std::endl;
    CSR_Matrix<IndexType, double> B = read_csr_matrix<IndexType, double>(pathB, true);
    std::cout << "  rows=" << B.num_rows << " cols=" << B.num_cols << " nnz=" << B.num_nnzs << std::endl;

    if (A.num_cols != B.num_rows) {
        std::cerr << "Error: A cols (" << A.num_cols << ") != B rows (" << B.num_rows << "). Cannot compute A*B.\n";
        delete_csr_matrix(A);
        delete_csr_matrix(B);
        return 1;
    }

    sparse_matrix_t hA = nullptr;
    sparse_matrix_t hB = nullptr;
    sparse_matrix_t hC = nullptr;

    // Copy to MKL_INT arrays for MKL (library uses int64_t for get_spgemm_flop)
    std::vector<MKL_INT> a_row(A.num_rows + 1), a_col(A.num_nnzs);
    std::vector<double> a_val(A.num_nnzs);
    for (IndexType i = 0; i <= A.num_rows; i++) a_row[i] = static_cast<MKL_INT>(A.row_offset[i]);
    for (IndexType i = 0; i < A.num_nnzs; i++) {
        a_col[i] = static_cast<MKL_INT>(A.col_index[i]);
        a_val[i] = A.values[i];
    }
    std::vector<MKL_INT> b_row(B.num_rows + 1), b_col(B.num_nnzs);
    std::vector<double> b_val(B.num_nnzs);
    for (IndexType i = 0; i <= B.num_rows; i++) b_row[i] = static_cast<MKL_INT>(B.row_offset[i]);
    for (IndexType i = 0; i < B.num_nnzs; i++) {
        b_col[i] = static_cast<MKL_INT>(B.col_index[i]);
        b_val[i] = B.values[i];
    }

    sparse_status_t st = mkl_sparse_d_create_csr(
        &hA,
        SPARSE_INDEX_BASE_ZERO,
        static_cast<MKL_INT>(A.num_rows),
        static_cast<MKL_INT>(A.num_cols),
        a_row.data(),
        a_row.data() + 1,
        a_col.data(),
        a_val.data());
    if (st != SPARSE_STATUS_SUCCESS) {
        std::cerr << "mkl_sparse_d_create_csr A failed: " << st << std::endl;
        delete_csr_matrix(A);
        delete_csr_matrix(B);
        return 1;
    }

    st = mkl_sparse_d_create_csr(
        &hB,
        SPARSE_INDEX_BASE_ZERO,
        static_cast<MKL_INT>(B.num_rows),
        static_cast<MKL_INT>(B.num_cols),
        b_row.data(),
        b_row.data() + 1,
        b_col.data(),
        b_val.data());
    if (st != SPARSE_STATUS_SUCCESS) {
        std::cerr << "mkl_sparse_d_create_csr B failed: " << st << std::endl;
        mkl_sparse_destroy(hA);
        delete_csr_matrix(A);
        delete_csr_matrix(B);
        return 1;
    }

    // C := op(A) * B, op(A) = A
    st = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, hA, hB, &hC);
    if (st != SPARSE_STATUS_SUCCESS) {
        std::cerr << "mkl_sparse_spmm failed: " << st << std::endl;
        mkl_sparse_destroy(hA);
        mkl_sparse_destroy(hB);
        delete_csr_matrix(A);
        delete_csr_matrix(B);
        return 1;
    }

    // Optional: get C dimensions and nnz for reporting
    sparse_index_base_t c_indexing;
    MKL_INT c_rows, c_cols;
    MKL_INT *c_rows_start = nullptr, *c_rows_end = nullptr, *c_col_indx = nullptr;
    double *c_values = nullptr;
    st = mkl_sparse_d_export_csr(hC, &c_indexing, &c_rows, &c_cols, &c_rows_start, &c_rows_end, &c_col_indx, &c_values);
    MKL_INT c_nnz = (st == SPARSE_STATUS_SUCCESS && c_rows > 0) ? (c_rows_end[c_rows - 1] - (c_indexing == SPARSE_INDEX_BASE_ZERO ? 0 : 1)) : 0;
    if (st == SPARSE_STATUS_SUCCESS) {
        std::cout << "C = A*B: rows=" << c_rows << " cols=" << c_cols << " nnz=" << c_nnz << std::endl;
    }

    // Warmup
    mkl_sparse_destroy(hC);
    hC = nullptr;
    st = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, hA, hB, &hC);
    if (st != SPARSE_STATUS_SUCCESS) {
        std::cerr << "mkl_sparse_spmm (warmup) failed: " << st << std::endl;
        mkl_sparse_destroy(hA);
        mkl_sparse_destroy(hB);
        delete_csr_matrix(A);
        delete_csr_matrix(B);
        return 1;
    }
    mkl_sparse_destroy(hC);
    hC = nullptr;

    // Timed runs
    timer t;
    for (int i = 0; i < iterations; i++) {
        st = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, hA, hB, &hC);
        if (st != SPARSE_STATUS_SUCCESS) {
            std::cerr << "mkl_sparse_spmm iteration " << i << " failed: " << st << std::endl;
            break;
        }
        mkl_sparse_destroy(hC);
        hC = nullptr;
    }
    double total_ms = t.milliseconds_elapsed();
    double avg_ms = total_ms / iterations;

    long long flops = get_spgemm_flop(A, B);
    double gflops = (avg_ms > 0) ? (flops / 1e9) / (avg_ms / 1000.0) : 0.0;

    std::cout << "MKL SpGEMM: " << iterations << " iterations, average " << avg_ms << " ms, "
              << gflops << " GFLOPS" << std::endl;

    mkl_sparse_destroy(hA);
    mkl_sparse_destroy(hB);
    delete_csr_matrix(A);
    delete_csr_matrix(B);
    return 0;
}
#else
static int run_mkl_spgemm(int argc, char **argv) {
    (void)argc;
    (void)argv;
    std::cerr << "Error: MKL support is not enabled. Configure with -DENABLE_MKL=ON and link MKL.\n";
    usage(argc, argv);
    return 1;
}
#endif

int main(int argc, char **argv) {
    if (get_arg(argc, argv, "help") != nullptr) {
        usage(argc, argv);
        return EXIT_SUCCESS;
    }
    return run_mkl_spgemm(argc, argv);
}
