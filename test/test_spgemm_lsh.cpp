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
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

#ifndef LSH_ORDER_OUT_DIR
#define LSH_ORDER_OUT_DIR "/data/linshengle_data/SpGEMM-Reordering/lsh_order"
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

/**
 * @brief Fast path for permutation + CSR_Vlength conversion.
 *        This is intentionally local to this benchmark test so other tests
 *        keep using the generic conversion path.
 */
template <typename IndexType, typename ValueType>
CSR_VlengthCluster<IndexType, ValueType> csr_perm_to_vlength_cluster_fast(
    const CSR_Matrix<IndexType, ValueType> &csr,
    const std::vector<IndexType> &row_permutation,
    const std::vector<IndexType> &offsets)
{
    if (offsets.size() < 2) {
        throw std::invalid_argument("offsets size must be >= 2");
    }
    if (offsets.front() != 0 || offsets.back() != csr.num_rows) {
        throw std::invalid_argument("offsets must start at 0 and end at csr.num_rows");
    }
    if (row_permutation.size() < static_cast<size_t>(csr.num_rows)) {
        throw std::invalid_argument("row_permutation size must be >= csr.num_rows");
    }

    CSR_VlengthCluster<IndexType, ValueType> cluster;
    cluster.csr_rows = csr.num_rows;
    cluster.cols = csr.num_cols;
    cluster.rows = static_cast<IndexType>(offsets.size() - 1);
    cluster.nnzc = 0;
    cluster.nnzv = 0;
    cluster.max_cluster_sz = 0;
    cluster.acc_flag = nullptr;
    cluster.min_ccol = nullptr;
    cluster.max_ccol = nullptr;
    cluster.dense_cluster_count = 0;

    cluster.rowptr = new_array<IndexType>(cluster.rows + 1);
    CHECK_ALLOC(cluster.rowptr);
    cluster.rowptr_val = new_array<IndexType>(cluster.rows + 1);
    CHECK_ALLOC(cluster.rowptr_val);
    cluster.cluster_sz = new_array<IndexType>(cluster.rows);
    CHECK_ALLOC(cluster.cluster_sz);

    std::vector<IndexType> work(cluster.rows, static_cast<IndexType>(0));
    std::vector<IndexType> work_val(cluster.rows, static_cast<IndexType>(0));

    // Pass-1: count unique columns per cluster without creating an intermediate reordered CSR.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (IndexType cid = 0; cid < cluster.rows; ++cid) {
        const IndexType start_row = offsets[cid];
        const IndexType end_row = offsets[cid + 1];
        const IndexType csz = end_row - start_row;
        cluster.cluster_sz[cid] = csz;

        std::unordered_map<IndexType, IndexType> col2slot;
        col2slot.reserve(static_cast<size_t>(std::max<IndexType>(csz * 16, 64)));

        for (IndexType new_row = start_row; new_row < end_row; ++new_row) {
            const IndexType original_row = row_permutation[new_row];
            if (original_row < 0 || original_row >= csr.num_rows) continue;

            for (IndexType jj = csr.row_offset[original_row]; jj < csr.row_offset[original_row + 1]; ++jj) {
                const IndexType acol = csr.col_index[jj];
                col2slot.emplace(acol, static_cast<IndexType>(0));
            }
        }

        const IndexType unique_cols = static_cast<IndexType>(col2slot.size());
        work[cid] = unique_cols;
        work_val[cid] = unique_cols * csz;
    }

    // Prefix sums for rowptr / rowptr_val and totals.
    cluster.rowptr[0] = 0;
    cluster.rowptr_val[0] = 0;
    for (IndexType cid = 0; cid < cluster.rows; ++cid) {
        cluster.nnzc += work[cid];
        cluster.nnzv += work_val[cid];
        cluster.max_cluster_sz = std::max(cluster.max_cluster_sz, cluster.cluster_sz[cid]);
        cluster.rowptr[cid + 1] = cluster.rowptr[cid] + work[cid];
        cluster.rowptr_val[cid + 1] = cluster.rowptr_val[cid] + work_val[cid];
    }

    cluster.colids = (cluster.nnzc > 0) ? new_array<IndexType>(cluster.nnzc) : nullptr;
    if (cluster.nnzc > 0) CHECK_ALLOC(cluster.colids);
    cluster.values = (cluster.nnzv > 0) ? new_array<ValueType>(cluster.nnzv) : nullptr;
    if (cluster.nnzv > 0) CHECK_ALLOC(cluster.values);
    if (cluster.nnzv > 0) {
        std::fill_n(cluster.values, cluster.nnzv, static_cast<ValueType>(0));
    }

    // Pass-2: fill colids + values directly in final storage.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (IndexType cid = 0; cid < cluster.rows; ++cid) {
        const IndexType start_row = offsets[cid];
        const IndexType end_row = offsets[cid + 1];
        const IndexType csz = cluster.cluster_sz[cid];
        const IndexType base_col = cluster.rowptr[cid];
        const IndexType base_val = cluster.rowptr_val[cid];

        std::unordered_map<IndexType, IndexType> col2slot;
        col2slot.reserve(static_cast<size_t>(std::max<IndexType>(work[cid] * 2, 64)));
        IndexType next_slot = 0;

        for (IndexType new_row = start_row; new_row < end_row; ++new_row) {
            const IndexType original_row = row_permutation[new_row];
            if (original_row < 0 || original_row >= csr.num_rows) continue;
            const IndexType row_in_cluster = new_row - start_row;

            for (IndexType jj = csr.row_offset[original_row]; jj < csr.row_offset[original_row + 1]; ++jj) {
                const IndexType acol = csr.col_index[jj];
                const ValueType aval = csr.values[jj];

                auto it = col2slot.find(acol);
                IndexType slot;
                if (it == col2slot.end()) {
                    slot = next_slot++;
                    col2slot.emplace(acol, slot);
                    cluster.colids[base_col + slot] = acol;
                } else {
                    slot = it->second;
                }
                cluster.values[base_val + (slot * csz) + row_in_cluster] = aval;
            }
        }
    }

    cluster.num_rows = cluster.csr_rows;
    cluster.num_cols = cluster.cols;
    cluster.num_nnzs = cluster.nnzv;
    return cluster;
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
    CSR_VlengthCluster<IndexType, ValueType> A_cluster =
        csr_perm_to_vlength_cluster_fast<IndexType, ValueType>(A_csr, permutation, offsets);
    double t_convert_ms = timer.stop();
    cout << "Format Conversion time: " << t_convert_ms << " ms" << endl;
    
    cout << "Clusters: " << (offsets.size() - 1) << endl;
    // cout << "Converting A to CSR_VlengthCluster..." << endl;
    
    delete_host_matrix(A_csr);
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
