/**
 * @file record_permutation_lsh.cpp
 * @brief Run LSH + hierarchical clustering (same as test_spgemm_hc_lsh), then write
 *        permutation to mtx_name.perm and offsets to mtx_name.offsets under
 *        /data2/linshengle_data/SpGEMM-Reordering/lsh_order/.
 *
 * Usage: ./record_permutation_lsh <A.mtx> [options]
 *        Options: --k, --bands, --seed, --hc_v, --cluster_size, --out-dir
 */

#include "../include/SpOps.h"
#include "../include/cmdline.h"
#include "../include/sparse_io.h"
#include "../include/sparse_conversion.h"
#include "../include/spgemm_utility.h"
#include "../include/spgemm_MinHashLSH.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <map>
#include <vector>

using namespace std;

#ifndef LSH_ORDER_OUT_DIR
#define LSH_ORDER_OUT_DIR "/data/linshengle_data/SpGEMM-Reordering/lsh_order"
#endif

static void create_directories_if_not_exists(const std::string &path) {
    std::string cmd = "mkdir -p \"" + path + "\"";
    if (std::system(cmd.c_str()) != 0) {
        std::cerr << "Warning: mkdir -p " << path << " failed" << std::endl;
    }
}

static void usage(int argc, char **argv) {
    std::cout << "Usage:\n";
    std::cout << "\t" << argv[0] << " <A.mtx> [options]\n";
    std::cout << "\t  Runs LSH + HC, writes permutation and offsets to out_dir/mtx_name.perm and .offsets\n";
    std::cout << "\tOptions:\n";
    std::cout << "\t  --out-dir     = output directory (default: " LSH_ORDER_OUT_DIR ")\n";
    std::cout << "\t  --k           = MinHash signature length (default 64)\n";
    std::cout << "\t  --bands       = LSH bands, k % bands == 0 (default 16)\n";
    std::cout << "\t  --seed        = random seed (default 7)\n";
    std::cout << "\t  --hc_v        = 0|1|2 (default 0)\n";
    std::cout << "\t  --cluster_size= max cluster size (default 8)\n";
    std::cout << "\t  --precision   = 32|64 (default 64)\n";
}

template <typename IndexType, typename ValueType>
void run_record(int argc, char **argv) {
    char *matA = nullptr;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            matA = argv[i];
            break;
        }
    }
    if (!matA) {
        std::cerr << "Error: need A.mtx\n";
        usage(argc, argv);
        return;
    }

    std::string out_dir = LSH_ORDER_OUT_DIR;
    char *p = get_argval(argc, argv, "out-dir");
    if (p) out_dir = p;

    IndexType cluster_size = 8;
    p = get_argval(argc, argv, "cluster_size");
    if (p) cluster_size = static_cast<IndexType>(atoi(p));

    int k = 64, num_bands = 16;
    p = get_argval(argc, argv, "k");
    if (p) k = atoi(p);
    p = get_argval(argc, argv, "bands");
    if (p) num_bands = atoi(p);

    uint64_t seed = 7;
    p = get_argval(argc, argv, "seed");
    if (p) seed = static_cast<uint64_t>(atoll(p));

    int hc_v = 0;
    p = get_argval(argc, argv, "hc_v");
    if (p) {
        hc_v = atoi(p);
        if (hc_v != 0 && hc_v != 1 && hc_v != 2) hc_v = 0;
    }

    std::cout << "Reading A: " << matA << std::endl;
    CSR_Matrix<IndexType, ValueType> A_csr = read_csr_matrix<IndexType, ValueType>(matA);
    std::cout << "A: " << A_csr.num_rows << " x " << A_csr.num_cols << ", nnz: " << A_csr.num_nnzs << std::endl;

    if (k <= 0 || num_bands <= 0 || k % num_bands != 0) {
        std::cerr << "Error: k % bands == 0 required. k=" << k << " bands=" << num_bands << std::endl;
        delete_host_matrix(A_csr);
        return;
    }

    std::cout << "LSH + HC (k=" << k << ", bands=" << num_bands << ", seed=" << seed << ", hc_v=" << hc_v << ")..." << std::endl;
    std::vector<IndexType> permutation, offsets;
    if (hc_v == 0) {
        std::map<std::pair<IndexType, IndexType>, ValueType> close_pairs = lsh_candidate_pairs<IndexType, ValueType>(
            A_csr.row_offset, A_csr.col_index, A_csr.num_rows, k, num_bands, seed);
        std::map<IndexType, std::vector<IndexType>> reordered_dict = hierarchical_clustering_v0<IndexType, ValueType>(
            A_csr.row_offset, A_csr.col_index, A_csr.num_rows, close_pairs, cluster_size);
        reordered_dict_to_permutation_and_offsets(reordered_dict, A_csr.num_rows, permutation, offsets);
    } else if (hc_v == 2) {
        std::vector<CandidatePair<IndexType, ValueType>> pairs = lsh_candidate_pairs_vector<IndexType, ValueType>(
            A_csr.row_offset, A_csr.col_index, A_csr.num_rows, k, num_bands, seed, false); // 退化回 LSH 的原始实现
        std::map<std::pair<IndexType, IndexType>, ValueType> close_pairs = candidate_pairs_vector_to_map<IndexType, ValueType>(pairs);
        std::map<IndexType, std::vector<IndexType>> reordered_dict = hierarchical_clustering_v0<IndexType, ValueType>(
            A_csr.row_offset, A_csr.col_index, A_csr.num_rows, close_pairs, cluster_size);
        reordered_dict_to_permutation_and_offsets(reordered_dict, A_csr.num_rows, permutation, offsets);
    } else {
        std::vector<CandidatePair<IndexType, ValueType>> pairs = lsh_candidate_pairs_vector<IndexType, ValueType>(
            A_csr.row_offset, A_csr.col_index, A_csr.num_rows, k, num_bands, seed);
        hierarchical_clustering_v1<IndexType, ValueType>(
            A_csr.row_offset, A_csr.col_index, A_csr.num_rows, pairs, cluster_size, permutation, offsets);
    }
    delete_host_matrix(A_csr);

    std::string mtx_name = extractFileNameWithoutExtension(std::string(matA));
    create_directories_if_not_exists(out_dir);

    std::string perm_path = out_dir + "/" + mtx_name + ".perm";
    std::string off_path = out_dir + "/" + mtx_name + ".offsets";
    {
        std::ofstream f(perm_path);
        if (!f) {
            std::cerr << "Error: cannot write " << perm_path << std::endl;
            return;
        }
        for (IndexType x : permutation)
            f << x << "\n";
    }
    {
        std::ofstream f(off_path);
        if (!f) {
            std::cerr << "Error: cannot write " << off_path << std::endl;
            return;
        }
        for (IndexType x : offsets)
            f << x << "\n";
    }
    std::cout << "Wrote " << perm_path << " (" << permutation.size() << " entries)" << std::endl;
    std::cout << "Wrote " << off_path << " (" << offsets.size() << " entries)" << std::endl;
}

int main(int argc, char **argv) {
    if (get_arg(argc, argv, "help") != nullptr || argc < 2) {
        usage(argc, argv);
        return EXIT_SUCCESS;
    }

    int precision = 64;
    char *p = get_argval(argc, argv, "precision");
    if (p) precision = atoi(p);

    if (precision == 32)
        run_record<int64_t, float>(argc, argv);
    else
        run_record<int64_t, double>(argc, argv);
    return EXIT_SUCCESS;
}
