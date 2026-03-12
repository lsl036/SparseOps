#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <random>
#include <iomanip>
#include "../include/plat_config.h"
// // 模拟硬件参数 - 假设 1MB L2
// #define CPU_L2CACHE_SIZE (1024 * 1024)
const int RANGE = CPU_L2CACHE_SIZE / CPU_CORES_PER_SOC/ CPU_SOCKET/8; 
const int REPEAT = 100;
const int CSZ = 8; // 模拟向量化宽度 (例如 AVX-512 double)

struct HashEntry {
    int key;
    double val;
};

// 仿真 Hash 累加器
double test_hash_accumulator(int total_nnz, int range, int csz) {
    int table_size = range * 2;
    std::vector<HashEntry> table(table_size, {-1, 0.0});
    std::vector<int> indices(total_nnz);
    for(int i=0; i<total_nnz; ++i) indices[i] = rand() % range;

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < REPEAT; ++r) {
        for (int i = 0; i < total_nnz; ++i) {
            int k = indices[i];
            unsigned int h = (static_cast<unsigned int>(k) * 2654435761U) % table_size;
            while (table[h].key != -1 && table[h].key != k) {
                h = (h + 1) % table_size;
            }
            table[h].key = k;
            for(int l=0; l<csz; ++l) table[h].val += 1.1;
        }
        // 重置 Hash 表：整表清零（按 indices 查找会在有重复 key 时死循环，因已清除的 slot 为 -1 永远 != k）
        for (int i = 0; i < table_size; ++i) {
            table[i].key = -1;
            table[i].val = 0.0;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// 仿真 Dense 累加器
double test_dense_accumulator_simulated(int nnz_per_col, int num_cols, int range, int csz) {
    std::vector<double> dense_buf(range * csz, 0.0);
    std::vector<double> a_vals(num_cols * csz, 1.1);
    std::vector<int> b_cols(nnz_per_col, 0); 
    for(int i=0; i<nnz_per_col; ++i) b_cols[i] = (rand() % range);

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < REPEAT; ++r) {
        // 1. 清零
        std::memset(dense_buf.data(), 0, range * csz * sizeof(double));

        // 2. 仿真你的广播-向量化逻辑
        for (int j = 0; j < num_cols; ++j) {
            double* a_ptr = &a_vals[j * csz];
            for (int k = 0; k < nnz_per_col; ++k) {
                int slot_idx = b_cols[k] * csz;
                double bv = 1.23;
                #pragma omp simd
                for (int l = 0; l < csz; ++l) {
                    dense_buf[slot_idx + l] += a_ptr[l] * bv;
                }
            }
        }

        // 3. 仿真写回 (只遍历有数据的部分，或者为了公平只做简单检查)
        // 注意：原代码这里的全量遍历是耗时大户，我们做个限时模拟
        for (int k = 0; k < nnz_per_col; ++k) {
            int idx = b_cols[k] * csz;
            if (dense_buf[idx] > 1e-9) dense_buf[idx] = 0;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

int main() {
    std::cout << "Simulating Per-Core L2 Range: " << RANGE << " elements" << std::endl;
    std::cout << "Total Dense Buffer Size: " << (RANGE * CSZ * 8) / 1024 << " KB" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Density | Hash Time(s) | Dense Time(s) | Faster One" << std::endl;

    const int NUM_COLS_A = 8; 

    for (double density = 0.02; density <= 0.52; density += 0.02) {
        int total_nnz = static_cast<int>(RANGE * density);
        int nnz_per_col = total_nnz / NUM_COLS_A;

        double t_hash = test_hash_accumulator(total_nnz, RANGE, CSZ);
        double t_dense = test_dense_accumulator_simulated(nnz_per_col, NUM_COLS_A, RANGE, CSZ);

        std::string winner = (t_hash < t_dense) ? "Hash" : "Dense";
        std::cout << std::setw(7) << density << " | " 
                  << std::setw(12) << t_hash << " | " 
                  << std::setw(12) << t_dense << " | " << winner << std::endl;
    }
    return 0;
}