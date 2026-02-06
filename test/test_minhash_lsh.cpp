/**
 * @file test_minhash_lsh.cpp
 * @brief Test MinHash signatures and LSH candidate pairs.
 *        Optionally compare with exact Jaccard and write pairs (est. Jaccard) for downstream.
 */

#include "../include/SpOps.h"
#include "../include/sparse_io.h"
#include "../include/spgemm_MinHashLSH.h"
#include "../include/spgemm_utility.h"
#include "../include/timer.h"
#include "../include/cmdline.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>

using namespace std;

static void usage(int argc, char **argv) {
    std::cout << "Usage:\n";
    std::cout << "\t" << argv[0] << " <A.mtx> [options]\n";
    std::cout << "\tOptions:\n";
    std::cout << "\t  --k       MinHash signature length (default: 128)\n";
    std::cout << "\t  --bands   Number of LSH bands, k % bands == 0 (default: 16)\n";
    std::cout << "\t  --seed    Random seed (default: 42)\n";
    std::cout << "\t  --output  Write candidate pairs to file (0-based: i j est_jaccard)\n";
    std::cout << "\t  --exact   For each pair, compute and print exact Jaccard (for verification)\n";
    std::cout << "\t  --precision 32|64 (default: 64)\n";
}

template <typename IndexType, typename ValueType>
void run_test(int argc, char **argv, const char *matA_path) {
    int k = 128;
    int num_bands = 16;
    uint64_t seed = 42;
    const char *output_path = get_argval(argc, argv, "output");
    bool with_exact = (get_argval(argc, argv, "exact") != nullptr);

    char *k_str = get_argval(argc, argv, "k");
    if (k_str) k = atoi(k_str);
    char *b_str = get_argval(argc, argv, "bands");
    if (b_str) num_bands = atoi(b_str);
    char *seed_str = get_argval(argc, argv, "seed");
    if (seed_str) seed = static_cast<uint64_t>(atoll(seed_str));

    if (k <= 0 || num_bands <= 0 || k % num_bands != 0) {
        cerr << "Error: k must be positive and divisible by bands. k=" << k << " bands=" << num_bands << endl;
        return;
    }

    cout << "Reading A: " << matA_path << endl;
    CSR_Matrix<IndexType, ValueType> A = read_csr_matrix<IndexType, ValueType>(matA_path);
    cout << "A: " << A.num_rows << " x " << A.num_cols << ", nnz: " << A.num_nnzs << endl;

    anonymouslib_timer timer;
    timer.start();
    auto pairs = lsh_candidate_pairs<IndexType, ValueType>(
        A.row_offset, A.col_index, A.num_rows, k, num_bands, seed);
    double t_ms = timer.stop();
    cout << "LSH candidate pairs: " << pairs.size() << " (time: " << t_ms << " ms)" << endl;

    if (pairs.empty()) {
        cout << "No candidate pairs; try smaller bands or larger k." << endl;
        delete_host_matrix(A);
        return;
    }

    int sample = 0;
    const int max_sample = 5;
    cout << "\nSample pairs (i, j) -> est_jaccard";
    if (with_exact) cout << " [exact_jaccard]";
    cout << ":\n";
    for (const auto &kv : pairs) {
        if (sample >= max_sample) break;
        IndexType i = kv.first.first, j = kv.first.second;
        ValueType est = kv.second;
        cout << "  (" << i << ", " << j << ") -> " << est;
        if (with_exact) {
            double exact = jaccard_similarity<IndexType, ValueType>(
                A.row_offset, A.col_index, i, j);
            cout << " [exact: " << exact << ", err: " << (static_cast<double>(est) - exact) << "]";
        }
        cout << endl;
        sample++;
    }

    if (output_path) {
        ofstream f(output_path);
        if (!f) {
            cerr << "Error: cannot open output " << output_path << endl;
            delete_host_matrix(A);
            return;
        }
        for (const auto &kv : pairs)
            f << kv.first.first << " " << kv.first.second << " " << kv.second << "\n";
        f.close();
        cout << "Wrote " << pairs.size() << " pairs to " << output_path
             << " (value = MinHash-est. Jaccard; optional to overwrite with exact later)." << endl;
    }

    delete_host_matrix(A);
}

int main(int argc, char **argv) {
    char *matA_path = nullptr;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            matA_path = argv[i];
            break;
        }
    }
    if (!matA_path) {
        usage(argc, argv);
        return 1;
    }

    bool use_float = false;
    char *prec = get_argval(argc, argv, "precision");
    if (prec && strcmp(prec, "32") == 0) use_float = true;

    if (use_float)
        run_test<int64_t, float>(argc, argv, matA_path);
    else
        run_test<int64_t, double>(argc, argv, matA_path);
    return 0;
}
