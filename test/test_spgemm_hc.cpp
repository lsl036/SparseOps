/**
 * @file test_spgemm_hc.cpp
 * @brief Hierarchical Cluster SpGEMM test (reference: HierarchicalClusterSpGEMM.cpp).
 *        Input: A.mtx, B.mtx, candidate_pairs (must be from A*AT similarity analysis).
 *        Flow: close_pairs -> hierarchical_clustering_v0 -> permutation + offsets ->
 *              reorder A -> CSR_VlengthCluster -> LeSpGEMM_VLength -> C.
 *        correctness: restore C to original row order and write MTX.
 *        performance: time LeSpGEMM_VLength only.
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
#include <cstdint>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <ctime>

using namespace std;

/** Read close_pairs from file: plain "u v [sim]" per line (0-based) or MTX (row col value). */
template <typename IndexType, typename ValueType>
std::map<std::pair<IndexType, IndexType>, ValueType> read_close_pairs(const char *filepath)
{
    std::map<std::pair<IndexType, IndexType>, ValueType> close_pairs;
    std::ifstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot open file: " << filepath << std::endl;
        return close_pairs;
    }
    std::string first_line;
    std::getline(f, first_line);
    if (first_line.empty()) {
        f.close();
        return close_pairs;
    }
    auto is_comment = [](const std::string &s) { return s.empty() || s[0] == '%'; };

    if (first_line.find("%%MatrixMarket") != std::string::npos) {
        std::string line = first_line;
        while (is_comment(line) && std::getline(f, line)) {}
        std::istringstream dim_stream(line);
        IndexType rows, cols, nnz;
        if (dim_stream >> rows >> cols >> nnz) {}
        bool first_data = true;
        bool zero_based = false;
        while (std::getline(f, line)) {
            if (is_comment(line)) continue;
            std::istringstream iss(line);
            IndexType u, v;
            ValueType sim;
            if (!(iss >> u >> v)) continue;
            if (!(iss >> sim)) sim = static_cast<ValueType>(1.0);
            if (first_data) {
                zero_based = (u == 0 || v == 0);
                first_data = false;
            }
            if (!zero_based) { u--; v--; }
            if (u < 0 || v < 0) continue;
            close_pairs[std::make_pair(u, v)] = sim;
        }
    } else {
        f.close();
        f.open(filepath);
        std::string line = first_line;
        do {
            if (is_comment(line)) continue;
            std::istringstream iss(line);
            IndexType u, v;
            ValueType sim;
            if (!(iss >> u >> v)) continue;
            if (!(iss >> sim)) sim = static_cast<ValueType>(1.0);
            if (u < 0 || v < 0) continue;
            close_pairs[std::make_pair(u, v)] = sim;
        } while (std::getline(f, line));
    }
    f.close();
    return close_pairs;
}

void usage(int argc, char **argv)
{
    std::cout << "Usage:\n";
    std::cout << "\t" << argv[0] << " <A.mtx> <B.mtx> <candidate_pairs.mtx> [options]\n";
    std::cout << "\t  candidate_pairs must be from A*AT similarity analysis (same row count as A).\n";
    std::cout << "\tOptions:\n";
    std::cout << "\t  --precision  = 32|64 (default 64)\n";
    std::cout << "\t  --threads    = number of OMP threads\n";
    std::cout << "\t  --test_type  = correctness|performance (default performance)\n";
    std::cout << "\t  --iterations = iterations for performance (default 10)\n";
    std::cout << "\t  --kernel     = 1 (Hash VLength) or 2 (Array VLength, default 1)\n";
    std::cout << "\t  --cluster_size = max cluster size (default 8)\n";
    std::cout << "\t  --sort       = 0|1 (sort output columns, default 0)\n";
}

template <typename IndexType, typename ValueType>
void test_spgemm_hc_correctness(const char *matA_path, const char *matB_path, const char *candidate_path,
                                 int kernel_flag, bool sortOutput, IndexType cluster_size)
{
    cout << "========================================" << endl;
    cout << "Hierarchical Cluster SpGEMM Correctness" << endl;
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

    cout << "Reading candidate pairs: " << candidate_path << endl;
    std::map<std::pair<IndexType, IndexType>, ValueType> close_pairs = read_close_pairs<IndexType, ValueType>(candidate_path);
    cout << "Loaded " << close_pairs.size() << " pairs" << endl;
    if (close_pairs.empty()) {
        cerr << "Error: No candidate pairs." << endl;
        delete_host_matrix(A_csr);
        delete_host_matrix(B);
        return;
    }

    cout << "Hierarchical clustering (cluster_size=" << cluster_size << ")..." << endl;
    std::map<IndexType, std::vector<IndexType>> reordered_dict = hierarchical_clustering_v0<IndexType, ValueType>(
        A_csr.row_offset, A_csr.col_index, A_csr.num_rows, close_pairs, cluster_size);
    cout << "Clusters: " << reordered_dict.size() << endl;

    std::vector<IndexType> permutation, offsets;
    reordered_dict_to_permutation_and_offsets<IndexType>(reordered_dict, A_csr.num_rows, permutation, offsets);
    if (static_cast<IndexType>(permutation.size()) != A_csr.num_rows) {
        cerr << "Error: permutation size != A.num_rows" << endl;
        delete_host_matrix(A_csr);
        delete_host_matrix(B);
        return;
    }

    cout << "Reordering A by cluster..." << endl;
    CSR_Matrix<IndexType, ValueType> A_reordered = csr_reorder_rows<IndexType, ValueType>(A_csr, permutation);
    cout << "Converting A to CSR_VlengthCluster..." << endl;
    CSR_VlengthCluster<IndexType, ValueType> A_cluster = csr_to_vlength_cluster<IndexType, ValueType>(A_reordered, offsets);
    cout << "A_cluster: " << A_cluster.rows << " clusters, nnzc=" << A_cluster.nnzc << ", nnzv=" << A_cluster.nnzv << endl;

    cout << "\n--- Kernel " << kernel_flag << " ---" << endl;
    CSR_VlengthCluster<IndexType, ValueType> C_cluster;
    anonymouslib_timer timer;
    timer.start();
    if (sortOutput)
        LeSpGEMM_VLength<true, IndexType, ValueType>(A_cluster, B, C_cluster, kernel_flag);
    else
        LeSpGEMM_VLength<false, IndexType, ValueType>(A_cluster, B, C_cluster, kernel_flag);
    double time_ms = timer.stop();
    cout << "C_cluster: " << C_cluster.rows << " clusters, nnzc=" << C_cluster.nnzc << endl;
    cout << "Time: " << time_ms << " ms" << endl;
    long long int flops = get_spgemm_flop(A_csr, B);
    cout << "Performance: " << (flops / 1e9) / (time_ms / 1000.0) << " GFLOPS" << endl;

    cout << "Converting C_cluster to CSR (cluster order)..." << endl;
    CSR_Matrix<IndexType, ValueType> C_csr = vlength_cluster2csr<IndexType, ValueType>(C_cluster);
    cout << "C_csr: " << C_csr.num_rows << " x " << C_csr.num_cols << ", nnz: " << C_csr.num_nnzs << endl;

    cout << "Restoring C to original row order..." << endl;
    std::vector<IndexType> inv_perm(A_csr.num_rows);
    for (IndexType k = 0; k < A_csr.num_rows; k++)
        inv_perm[permutation[k]] = k;
    CSR_Matrix<IndexType, ValueType> C_original = csr_reorder_rows<IndexType, ValueType>(C_csr, inv_perm);

    std::string base = extractFileNameWithoutExtension(std::string(matA_path));
    std::string suffix = (kernel_flag == 1) ? "hashvlengthcluster_hc" : "arrayvlengthcluster_hc";
    std::string out_path = base + "_SpOps_" + suffix + ".mtx";

    COO_Matrix<IndexType, ValueType> coo = csr_to_coo(C_original);
    int *I = new int[coo.num_nnzs];
    int *J = new int[coo.num_nnzs];
    double *V = new double[coo.num_nnzs];
    for (IndexType i = 0; i < coo.num_nnzs; i++) {
        I[i] = static_cast<int>(coo.row_index[i] + 1);
        J[i] = static_cast<int>(coo.col_index[i] + 1);
        V[i] = static_cast<double>(coo.values[i]);
    }
    MM_typecode matcode;
    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_coordinate(&matcode);
    mm_set_real(&matcode);
    mm_set_general(&matcode);
    mm_write_mtx_crd(const_cast<char*>(out_path.c_str()),
                     static_cast<int>(C_original.num_rows), static_cast<int>(C_original.num_cols),
                     static_cast<int>(C_original.num_nnzs), I, J, V, matcode);
    delete[] I;
    delete[] J;
    delete[] V;

    delete_host_matrix(C_original);
    delete_host_matrix(C_csr);
    delete_host_matrix(coo);
    delete_vlength_cluster_matrix(C_cluster);
    delete_vlength_cluster_matrix(A_cluster);
    delete_host_matrix(A_reordered);
    delete_host_matrix(A_csr);
    delete_host_matrix(B);
    cout << "Result (original row order) written to: " << out_path << endl;
    cout << "Correctness test done." << endl;
}

template <typename IndexType, typename ValueType>
void test_spgemm_hc_performance(const char *matA_path, const char *matB_path, const char *candidate_path,
                                 int iterations, int kernel_flag, bool sortOutput, IndexType cluster_size)
{
    cout << "========================================" << endl;
    cout << "Hierarchical Cluster SpGEMM Performance" << endl;
    cout << "========================================" << endl;

    CSR_Matrix<IndexType, ValueType> A_csr = read_csr_matrix<IndexType, ValueType>(matA_path);
    CSR_Matrix<IndexType, ValueType> B = read_csr_matrix<IndexType, ValueType>(matB_path);
    if (A_csr.num_cols != B.num_rows) {
        cerr << "Error: A.cols != B.rows" << endl;
        delete_host_matrix(A_csr);
        delete_host_matrix(B);
        return;
    }

    std::map<std::pair<IndexType, IndexType>, ValueType> close_pairs = read_close_pairs<IndexType, ValueType>(candidate_path);
    if (close_pairs.empty()) {
        cerr << "Error: No candidate pairs." << endl;
        delete_host_matrix(A_csr);
        delete_host_matrix(B);
        return;
    }

    std::map<IndexType, std::vector<IndexType>> reordered_dict = hierarchical_clustering_v0<IndexType, ValueType>(
        A_csr.row_offset, A_csr.col_index, A_csr.num_rows, close_pairs, cluster_size);
    std::vector<IndexType> permutation, offsets;
    reordered_dict_to_permutation_and_offsets<IndexType>(reordered_dict, A_csr.num_rows, permutation, offsets);

    CSR_Matrix<IndexType, ValueType> A_reordered = csr_reorder_rows<IndexType, ValueType>(A_csr, permutation);
    CSR_VlengthCluster<IndexType, ValueType> A_cluster = csr_to_vlength_cluster<IndexType, ValueType>(A_reordered, offsets);
    delete_host_matrix(A_reordered);

    long long int flops = get_spgemm_flop(A_csr, B);
    cout << "FLOPs: " << flops << endl;
    cout << "Kernel: " << kernel_flag << endl;

    CSR_VlengthCluster<IndexType, ValueType> C_cluster;
    if (sortOutput)
        LeSpGEMM_VLength<true, IndexType, ValueType>(A_cluster, B, C_cluster, kernel_flag);
    else
        LeSpGEMM_VLength<false, IndexType, ValueType>(A_cluster, B, C_cluster, kernel_flag);
    delete_vlength_cluster_matrix(C_cluster);

    double total_ms = 0.0;
    for (int i = 0; i < iterations; i++) {
#ifdef _OPENMP
        double t0 = omp_get_wtime();
#else
        double t0 = static_cast<double>(clock()) / CLOCKS_PER_SEC;
#endif
        if (sortOutput)
            LeSpGEMM_VLength<true, IndexType, ValueType>(A_cluster, B, C_cluster, kernel_flag);
        else
            LeSpGEMM_VLength<false, IndexType, ValueType>(A_cluster, B, C_cluster, kernel_flag);
#ifdef _OPENMP
        double t1 = omp_get_wtime();
#else
        double t1 = static_cast<double>(clock()) / CLOCKS_PER_SEC;
#endif
        total_ms += (t1 - t0) * 1000.0;
        if (i < iterations - 1) delete_vlength_cluster_matrix(C_cluster);
    }
    double avg_ms = total_ms / iterations;
    cout << "Average time: " << avg_ms << " ms" << endl;
    cout << "Average GFLOPS: " << (flops / 1e9) / (avg_ms / 1000.0) << endl;

    delete_vlength_cluster_matrix(C_cluster);
    delete_vlength_cluster_matrix(A_cluster);
    delete_host_matrix(A_csr);
    delete_host_matrix(B);
}

template <typename IndexType, typename ValueType>
void run_spgemm_hc_test(int argc, char **argv)
{
    char *matA = NULL, *matB = NULL, *candidate = NULL;
    int n = 0;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            if (n == 0) { matA = argv[i]; n++; }
            else if (n == 1) { matB = argv[i]; n++; }
            else if (n == 2) { candidate = argv[i]; n++; break; }
        }
    }
    if (!matA || !matB || !candidate) {
        printf("Error: need A.mtx B.mtx candidate_pairs.mtx\n");
        usage(argc, argv);
        return;
    }

    const char *test_type = "performance";
    char *p = get_argval(argc, argv, "test_type");
    if (p) test_type = p;

    int iterations = 10;
    p = get_argval(argc, argv, "iterations");
    if (p) iterations = atoi(p);

    int kernel_flag = 1;
    p = get_argval(argc, argv, "kernel");
    if (p) {
        kernel_flag = atoi(p);
        if (kernel_flag != 1 && kernel_flag != 2) kernel_flag = 1;
    }

    IndexType cluster_size = 8;
    p = get_argval(argc, argv, "cluster_size");
    if (p) cluster_size = static_cast<IndexType>(atoi(p));

    bool sortOutput = false;
    p = get_argval(argc, argv, "sort");
    if (p) sortOutput = (atoi(p) != 0);

    cout << "A: " << matA << ", B: " << matB << ", candidate_pairs: " << candidate << endl;
    cout << "test_type: " << test_type << ", kernel: " << kernel_flag << ", cluster_size: " << cluster_size << endl;
    cout << "threads: " << Le_get_thread_num() << endl;

    if (strcmp(test_type, "performance") == 0) {
        test_spgemm_hc_performance<IndexType, ValueType>(
            matA, matB, candidate, iterations, kernel_flag, sortOutput, cluster_size);
    } else {
        test_spgemm_hc_correctness<IndexType, ValueType>(
            matA, matB, candidate, kernel_flag, sortOutput, cluster_size);
    }
}

int main(int argc, char *argv[])
{
    if (get_arg(argc, argv, "help") != NULL || argc < 4) {
        usage(argc, argv);
        return EXIT_SUCCESS;
    }

    int precision = 64;
    char *p = get_argval(argc, argv, "precision");
    if (p) precision = atoi(p);

#ifdef CPU_SOCKET
    Le_set_thread_num(CPU_SOCKET * CPU_CORES_PER_SOC);
#else
    Le_set_thread_num(Le_get_core_num());
#endif
    p = get_argval(argc, argv, "threads");
    if (p) Le_set_thread_num(atoi(p));

    printf("Precision: %d-bit, threads: %d\n\n", precision, Le_get_thread_num());

    if (precision == 32)
        run_spgemm_hc_test<int64_t, float>(argc, argv);
    else if (precision == 64)
        run_spgemm_hc_test<int64_t, double>(argc, argv);
    else {
        usage(argc, argv);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
