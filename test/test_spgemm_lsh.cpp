/**
 * @file test_spgemm_lsh.cpp
 * @brief Read A.mtx, B.mtx and LSH clustering info (.perm + .offsets) from lsh_order dir,
 *        reorder A -> csr_to_vlength_cluster -> LeSpGEMM_VLength kernel=3 only.
 */

#include "../include/SpOps.h"
#include "../include/spgemm.h"
#include "../include/timer.h"
#include "../include/cmdline.h"
#include "../include/sparse_io.h"
#include "../include/sparse_conversion.h"
#include "../include/spgemm_utility.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

#ifndef LSH_ORDER_OUT_DIR
#define LSH_ORDER_OUT_DIR "/data2/linshengle_data/SpGEMM-Reordering/lsh_order"
#endif

/** Read offsets file: one IndexType per line. */
template <typename IndexType>
std::vector<IndexType> read_offsets_file(const char *filename) {
    std::vector<IndexType> out;
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Error: cannot open offsets file: " << filename << std::endl;
        throw std::runtime_error("Cannot open offsets file");
    }
    IndexType x;
    while (f >> x)
        out.push_back(x);
    return out;
}

static void usage(int argc, char **argv) {
    std::cout << "Usage:\n";
    std::cout << "\t" << argv[0] << " <A.mtx> <B.mtx> [options]\n";
    std::cout << "\t  Reads permutation and offsets from lsh_order/<mtx_name>.perm and .offsets (mtx_name from A path).\n";
    std::cout << "\t  Runs LeSpGEMM_VLength kernel=3 only.\n";
    std::cout << "\tOptions:\n";
    std::cout << "\t  --lsh-order-dir = directory for .perm/.offsets (default: " LSH_ORDER_OUT_DIR ")\n";
    std::cout << "\t  --precision     = 32|64 (default 64)\n";
    std::cout << "\t  --threads      = OMP threads\n";
    std::cout << "\t  --iterations   = iterations for timing (default 10)\n";
    std::cout << "\t  --l2_fraction  = L2 fraction for kernel=3 (default: MIXED_L2_FRACTION)\n";
    std::cout << "\t  --sort         = 0|1 (sort output columns, default 0)\n";
}

template <typename IndexType, typename ValueType>
void run_spgemm_lsh(const char *matA_path, const char *matB_path, const std::string &lsh_order_dir,
                    int iterations, bool sortOutput, double l2_fraction)
{
    const int kernel_flag = 3;

    cout << "========================================" << endl;
    cout << "LSH-order SpGEMM (LeSpGEMM_VLength kernel=3)" << endl;
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

    std::string mtx_name = extractFileNameWithoutExtension(std::string(matA_path));
    std::string perm_path = lsh_order_dir + "/" + mtx_name + ".perm";
    std::string off_path = lsh_order_dir + "/" + mtx_name + ".offsets";

    cout << "Reading permutation: " << perm_path << endl;
    std::vector<IndexType> permutation = read_row_permutation<IndexType>(perm_path.c_str(), A_csr.num_rows);
    if (static_cast<IndexType>(permutation.size()) != A_csr.num_rows) {
        cerr << "Error: permutation size (" << permutation.size() << ") != A.num_rows (" << A_csr.num_rows << ")" << endl;
        delete_host_matrix(A_csr);
        delete_host_matrix(B);
        return;
    }

    cout << "Reading offsets: " << off_path << endl;
    std::vector<IndexType> offsets = read_offsets_file<IndexType>(off_path.c_str());
    if (offsets.empty() || offsets.front() != 0 || offsets.back() != A_csr.num_rows) {
        cerr << "Error: offsets must be non-empty, start with 0 and end with A.num_rows. Got "
             << offsets.size() << " entries, first=" << (offsets.empty() ? -1 : (int64_t)offsets.front())
             << " last=" << (offsets.empty() ? -1 : (int64_t)offsets.back()) << endl;
        delete_host_matrix(A_csr);
        delete_host_matrix(B);
        return;
    }

    long long int flops = get_spgemm_flop(A_csr, B);
    cout << "FLOPs: " << flops << endl;

    cout << "Converting A to CSR_VlengthCluster..." << endl;
    anonymouslib_timer timer;
    timer.start();
    CSR_Matrix<IndexType, ValueType> A_reordered = csr_reorder_rows<IndexType, ValueType>(A_csr, permutation);
    CSR_VlengthCluster<IndexType, ValueType> A_cluster = csr_to_vlength_cluster<IndexType, ValueType>(A_reordered, offsets);
    double t_convert_ms = timer.stop();
    cout << "Format Conversion time: " << t_convert_ms << " ms" << endl;
    
    cout << "Clusters: " << (offsets.size() - 1) << endl;
    // cout << "Converting A to CSR_VlengthCluster..." << endl;
    
    delete_host_matrix(A_csr);
    delete_host_matrix(A_reordered);
    cout << "A_cluster: " << A_cluster.rows << " clusters, nnzc=" << A_cluster.nnzc << ", nnzv=" << A_cluster.nnzv << endl;

    CSR_VlengthCluster<IndexType, ValueType> C_cluster;
    C_cluster.acc_flag = new_array<char>(A_cluster.rows);
    C_cluster.min_ccol = new_array<IndexType>(A_cluster.rows);
    C_cluster.max_ccol = new_array<IndexType>(A_cluster.rows);

    if (sortOutput)
        LeSpGEMM_VLength<true, IndexType, ValueType>(A_cluster, B, C_cluster, kernel_flag, l2_fraction);
    else
        LeSpGEMM_VLength<false, IndexType, ValueType>(A_cluster, B, C_cluster, kernel_flag, l2_fraction);
    cout << "[mixed_acc] dense clusters: " << C_cluster.dense_cluster_count
         << " / " << C_cluster.rows
         << " (" << (100.0 * C_cluster.dense_cluster_count / C_cluster.rows) << "%)" << endl;
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
    cout << "Average time (LeSpGEMM_VLength kernel=3): " << avg_ms << " ms" << endl;
    cout << "Average GFLOPS: " << (flops / 1e9) / (avg_ms / 1000.0) << endl;

    delete_vlength_cluster_matrix(C_cluster);
    delete_array(C_cluster.acc_flag);
    delete_array(C_cluster.min_ccol);
    delete_array(C_cluster.max_ccol);
    delete_vlength_cluster_matrix(A_cluster);
    delete_host_matrix(B);
    cout << "Done." << endl;
}

int main(int argc, char **argv) {
    if (get_arg(argc, argv, "help") != nullptr || argc < 3) {
        usage(argc, argv);
        return EXIT_SUCCESS;
    }

    char *matA = nullptr, *matB = nullptr;
    int n = 0;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            if (n == 0) { matA = argv[i]; n++; }
            else if (n == 1) { matB = argv[i]; n++; break; }
        }
    }
    if (!matA || !matB) {
        std::cerr << "Error: need A.mtx and B.mtx\n";
        usage(argc, argv);
        return EXIT_FAILURE;
    }

#ifdef CPU_SOCKET
    Le_set_thread_num(CPU_SOCKET * CPU_CORES_PER_SOC);
#else
    Le_set_thread_num(Le_get_core_num());
#endif
    char *p = get_argval(argc, argv, "threads");
    if (p) Le_set_thread_num(atoi(p));

    std::string lsh_order_dir = LSH_ORDER_OUT_DIR;
    p = get_argval(argc, argv, "lsh-order-dir");
    if (p) lsh_order_dir = p;

    int iterations = 10;
    p = get_argval(argc, argv, "iterations");
    if (p) iterations = std::max(1, atoi(p));

    bool sortOutput = false;
    p = get_argval(argc, argv, "sort");
    if (p) sortOutput = (atoi(p) != 0);

    double l2_fraction = -1.0;
    p = get_argval(argc, argv, "l2_fraction");
    if (p) l2_fraction = atof(p);

    int precision = 64;
    p = get_argval(argc, argv, "precision");
    if (p) precision = atoi(p);

    printf("Precision: %d-bit, threads: %d, lsh_order_dir: %s, kernel=3\n\n",
           precision, Le_get_thread_num(), lsh_order_dir.c_str());

    if (precision == 32)
        run_spgemm_lsh<int64_t, float>(matA, matB, lsh_order_dir, iterations, sortOutput, l2_fraction);
    else
        run_spgemm_lsh<int64_t, double>(matA, matB, lsh_order_dir, iterations, sortOutput, l2_fraction);
    return EXIT_SUCCESS;
}
