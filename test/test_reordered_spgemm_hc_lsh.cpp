/**
 * @file test_reordered_spgemm_hc_lsh.cpp
 * @brief Reordered A + fixed-size row clustering -> VlengthCluster -> LeSpGEMM_VLength.
 *        Input: matA, matB, reordered_A (file with permutation vector, size = A.num_rows).
 *        Flow: read permutation -> reorder A -> partition into clusters by cluster_size -> csr_to_vlength_cluster -> LeSpGEMM_VLength.
 */

#include "../include/SpOps.h"
#include "../include/spgemm.h"
#include "../include/timer.h"
#include "../include/cmdline.h"
#include "../include/sparse_io.h"
#include "../include/sparse_conversion.h"
#include "../include/spgemm_utility.h"
#include "../include/mmio.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

static void usage(int argc, char **argv) {
    std::cout << "Usage:\n";
    std::cout << "\t" << argv[0] << " <A.mtx> <B.mtx> <reordered_A> [options]\n";
    std::cout << "\t  reordered_A: permutation vector file, one index per line, size must equal A.num_rows.\n";
    std::cout << "\t  New row i = original row reordered_A[i].\n";
    std::cout << "\tOptions:\n";
    std::cout << "\t  --precision   = 32|64 (default 64)\n";
    std::cout << "\t  --threads    = number of OMP threads\n";
    std::cout << "\t  --iterations = iterations for performance (default 10)\n";
    std::cout << "\t  --kernel     = 1 (Hash VLength), 2 (Array VLength), or 3 (Mixed, default 1)\n";
    std::cout << "\t  --l2_fraction= L2 budget fraction for kernel=3 (default: MIXED_L2_FRACTION)\n";
    std::cout << "\t  --cluster_size = rows per cluster (default 8)\n";
    std::cout << "\t  --sort       = 0|1 (sort output columns, default 0)\n";
}

template <typename IndexType, typename ValueType>
void run_reordered_spgemm_hc_lsh(const char *matA_path, const char *matB_path, const char *reordered_A_path,
                                  int iterations, int kernel_flag, bool sortOutput,
                                  IndexType cluster_size, double l2_fraction)
{
    cout << "========================================" << endl;
    cout << "Reordered A -> Vlength-Cluster SpGEMM" << endl;
    cout << "========================================" << endl;

    cout << "Reading A: " << matA_path << endl;
    CSR_Matrix<IndexType, ValueType> A_csr = read_csr_matrix<IndexType, ValueType>(matA_path);
    cout << "A: " << A_csr.num_rows << " x " << A_csr.num_cols << ", nnz: " << A_csr.num_nnzs << endl;

    cout << "Reading B: " << matB_path << endl;
    CSR_Matrix<IndexType, ValueType> B = read_csr_matrix<IndexType, ValueType>(matB_path);
    cout << "B: " << B.num_rows << " x " << B.num_cols << ", nnz: " << B.num_nnzs << endl;

    if (A_csr.num_cols != B.num_rows) {
        cerr << "Error: A.cols != B.rows" << endl;
        delete_host_matrix(A_csr);
        delete_host_matrix(B);
        return;
    }

    cout << "Reading reordered_A (permutation): " << reordered_A_path << endl;
    std::vector<IndexType> permutation = read_row_permutation<IndexType>(reordered_A_path, A_csr.num_rows);
    if (static_cast<IndexType>(permutation.size()) != A_csr.num_rows) {
        cerr << "Error: reordered_A size (" << permutation.size() << ") != A.num_rows (" << A_csr.num_rows << ")" << endl;
        delete_host_matrix(A_csr);
        delete_host_matrix(B);
        return;
    }

    long long int flops = get_spgemm_flop(A_csr, B);
    cout << "FLOPs: " << flops << endl;

    cout << "Reordering A by permutation..." << endl;
    CSR_Matrix<IndexType, ValueType> A_reordered = csr_reorder_rows<IndexType, ValueType>(A_csr, permutation);
    delete_host_matrix(A_csr);

    IndexType n = A_reordered.num_rows;
    std::vector<IndexType> offsets;
    offsets.push_back(0);
    for (IndexType i = cluster_size; i < n; i += cluster_size)
        offsets.push_back(i);
    offsets.push_back(n);

    cout << "Clusters: " << (offsets.size() - 1) << " (cluster_size=" << cluster_size << ")" << endl;
    cout << "Converting A to CSR_VlengthCluster..." << endl;
    CSR_VlengthCluster<IndexType, ValueType> A_cluster = csr_to_vlength_cluster<IndexType, ValueType>(A_reordered, offsets);
    delete_host_matrix(A_reordered);
    cout << "A_cluster: " << A_cluster.rows << " clusters, nnzc=" << A_cluster.nnzc << ", nnzv=" << A_cluster.nnzv << endl;

    cout << "Kernel: " << kernel_flag << endl;

    CSR_VlengthCluster<IndexType, ValueType> C_cluster;
    if (kernel_flag == 3) {
        C_cluster.acc_flag = new_array<char>(A_cluster.rows);
        C_cluster.min_ccol = new_array<IndexType>(A_cluster.rows);
        C_cluster.max_ccol = new_array<IndexType>(A_cluster.rows);
    }
    if (sortOutput)
        LeSpGEMM_VLength<true, IndexType, ValueType>(A_cluster, B, C_cluster, kernel_flag, l2_fraction);
    else
        LeSpGEMM_VLength<false, IndexType, ValueType>(A_cluster, B, C_cluster, kernel_flag, l2_fraction);
    if (kernel_flag == 3) {
        cout << "[mixed_acc] dense clusters: " << C_cluster.dense_cluster_count
             << " / " << C_cluster.rows
             << " (" << (100.0 * C_cluster.dense_cluster_count / C_cluster.rows) << "%)" << endl;
    }
    delete_vlength_cluster_matrix(C_cluster);

    double total_ms = 0.0;
    for (int i = 0; i < iterations; i++) {
#ifdef _OPENMP
        double t0 = omp_get_wtime();
#else
        double t0 = static_cast<double>(clock()) / CLOCKS_PER_SEC;
#endif
        if (sortOutput)
            LeSpGEMM_VLength<true, IndexType, ValueType>(A_cluster, B, C_cluster, kernel_flag, l2_fraction);
        else
            LeSpGEMM_VLength<false, IndexType, ValueType>(A_cluster, B, C_cluster, kernel_flag, l2_fraction);
#ifdef _OPENMP
        double t1 = omp_get_wtime();
#else
        double t1 = static_cast<double>(clock()) / CLOCKS_PER_SEC;
#endif
        total_ms += (t1 - t0) * 1000.0;
        if (i < iterations - 1) delete_vlength_cluster_matrix(C_cluster);
    }
    double avg_ms = total_ms / iterations;
    cout << "Average time (LeSpGEMM_VLength): " << avg_ms << " ms" << endl;
    cout << "Average GFLOPS: " << (flops / 1e9) / (avg_ms / 1000.0) << endl;

    delete_vlength_cluster_matrix(C_cluster);
    delete_vlength_cluster_matrix(A_cluster);
    delete_host_matrix(B);
    if (kernel_flag == 3) {
        delete_array(C_cluster.acc_flag);
        delete_array(C_cluster.min_ccol);
        delete_array(C_cluster.max_ccol);
    }
    cout << "Done." << endl;
}

template <typename IndexType, typename ValueType>
void run_test(int argc, char **argv) {
    char *matA = nullptr, *matB = nullptr, *reordered_A = nullptr;
    int n = 0;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            if (n == 0) { matA = argv[i]; n++; }
            else if (n == 1) { matB = argv[i]; n++; }
            else if (n == 2) { reordered_A = argv[i]; n++; break; }
        }
    }
    if (!matA || !matB || !reordered_A) {
        printf("Error: need A.mtx B.mtx reordered_A\n");
        usage(argc, argv);
        return;
    }

    int iterations = 10;
    char *p = get_argval(argc, argv, "iterations");
    if (p) iterations = std::max(1, atoi(p));

    int kernel_flag = 1;
    p = get_argval(argc, argv, "kernel");
    if (p) {
        kernel_flag = atoi(p);
        if (kernel_flag != 1 && kernel_flag != 2 && kernel_flag != 3) kernel_flag = 1;
    }

    IndexType cluster_size = 8;
    p = get_argval(argc, argv, "cluster_size");
    if (p) cluster_size = static_cast<IndexType>(atoi(p));

    bool sortOutput = false;
    p = get_argval(argc, argv, "sort");
    if (p) sortOutput = (atoi(p) != 0);

    double l2_fraction = -1.0;
    p = get_argval(argc, argv, "l2_fraction");
    if (p) l2_fraction = atof(p);

    cout << "A: " << matA << ", B: " << matB << ", reordered_A: " << reordered_A << endl;
    cout << "kernel: " << kernel_flag << ", cluster_size: " << cluster_size << ", iterations: " << iterations << endl;
    cout << "threads: " << Le_get_thread_num() << endl;

    run_reordered_spgemm_hc_lsh<IndexType, ValueType>(
        matA, matB, reordered_A, iterations, kernel_flag, sortOutput, cluster_size, l2_fraction);
}

int main(int argc, char **argv) {
    if (get_arg(argc, argv, "help") != nullptr || argc < 4) {
        usage(argc, argv);
        return EXIT_SUCCESS;
    }

#ifdef CPU_SOCKET
    Le_set_thread_num(CPU_SOCKET * CPU_CORES_PER_SOC);
#else
    Le_set_thread_num(Le_get_core_num());
#endif
    char *p = get_argval(argc, argv, "threads");
    if (p) Le_set_thread_num(atoi(p));

    int precision = 64;
    p = get_argval(argc, argv, "precision");
    if (p) precision = atoi(p);
    printf("Precision: %d-bit, threads: %d\n\n", precision, Le_get_thread_num());

    if (precision == 32)
        run_test<int64_t, float>(argc, argv);
    else
        run_test<int64_t, double>(argc, argv);
    return EXIT_SUCCESS;
}
