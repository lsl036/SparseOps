#ifndef SPGEMM_UTILITY_H
#define SPGEMM_UTILITY_H

/**
 * @file spgemm_utility.h
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Utility functions for SpGEMM operations (prefix sum/scan functions)
 * @version 0.1
 * @date 2026
 * 
 * @brief This file provides parallel prefix sum (scan) functions for SpGEMM operations.
 *        Reference implementation: utility.h from clusterwise-spgemm-main
 */

#include "memopt.h"
#include "thread.h"
#include <omp.h>

// ============================================================================
// Sequential Scan Functions (for small arrays)
// ============================================================================

/**
 * @brief Sequential prefix sum (scan) - basic version
 * @tparam T Index type
 * @param input Input array
 * @param output Output array (prefix sum)
 * @param n Array size
 */
template <typename T>
inline void seq_scan(const T *input, T *output, T n) {
    output[0] = 0;
    for (T i = 0; i < n - 1; ++i) {
        output[i + 1] = output[i] + input[i];
    }
}

/**
 * @brief Sequential prefix sum (scan) with fixed multiplier
 * @tparam T Output type
 * @tparam I Input type
 * @param input Input array
 * @param output Output array (prefix sum)
 * @param multiplier Fixed multiplier for each element
 * @param n Array size
 */
template <typename T, typename I>
inline void seq_scan_mult(const I *input, T *output, const I multiplier, I n) {
    output[0] = 0;
    for (I i = 0; i < n - 1; ++i) {
        output[i + 1] = output[i] + (input[i] * multiplier);
    }
}

// ============================================================================
// Parallel Scan Functions (for large arrays)
// ============================================================================

/**
 * @brief Parallel prefix sum (scan) - basic version
 *        Reference: utility.h line 246-288 (scan with T* input)
 * @tparam T Index type
 * @param input Input array
 * @param output Output array (prefix sum)
 * @param n Array size
 * @param tnum Number of threads for parallel execution (default: 56)
 * 
 * @note Uses sequential version for small arrays (N < 131072)
 *       Uses parallel version for large arrays
 */
template <typename T>
void scan(const T *input, T *output, T n, int tnum = 56) {
    // Use sequential version for small arrays (matching reference: N < (1 << 17))
    if (n < (1 << 17)) {
        seq_scan(input, output, n);
        return;
    }
    
    // Parallel version for large arrays
    T each_n = n / tnum;
    T *partial_sum = new_array<T>(tnum);
    
    #pragma omp parallel num_threads(tnum)
    {
        int tid = omp_get_thread_num();
        T start = each_n * tid;
        T end = (tid < tnum - 1) ? start + each_n : n;
        
        output[start] = 0;
        for (T i = start; i < end - 1; ++i) {
            output[i + 1] = output[i] + input[i];
        }
        partial_sum[tid] = output[end - 1] + input[end - 1];
        
        #pragma omp barrier
        
        T offset = 0;
        for (int ii = 0; ii < tid; ++ii) {
            offset += partial_sum[ii];
        }
        for (T i = start; i < end; ++i) {
            output[i] += offset;
        }
    }
    
    delete_array(partial_sum);
}

/**
 * @brief Parallel prefix sum (scan) with fixed multiplier
 *        Reference: utility.h line 427-469 (scan with in_weight parameter)
 * @tparam T Output type
 * @tparam I Input type
 * @param input Input array
 * @param output Output array (prefix sum)
 * @param multiplier Fixed multiplier for each element
 * @param n Array size
 * @param tnum Number of threads for parallel execution (default: 56)
 * 
 * @note Uses sequential version for small arrays (N < 131072)
 *       Uses parallel version for large arrays
 */
template <typename T, typename I>
void scan(const I *input, T *output, const I multiplier, I n, int tnum = 56) {
    // Use sequential version for small arrays (matching reference: N < (1 << 17))
    if (n < (1 << 17)) {
        seq_scan_mult(input, output, multiplier, n);
        return;
    }
    
    // Parallel version for large arrays
    I each_n = n / tnum;
    T *partial_sum = new_array<T>(tnum);
    
    #pragma omp parallel num_threads(tnum)
    {
        int tid = omp_get_thread_num();
        I start = each_n * tid;
        I end = (tid < tnum - 1) ? start + each_n : n;
        
        output[start] = 0;
        for (I i = start; i < end - 1; ++i) {
            output[i + 1] = output[i] + (input[i] * multiplier);
        }
        partial_sum[tid] = output[end - 1] + (input[end - 1] * multiplier);
        
        #pragma omp barrier
        
        T offset = 0;
        for (int ii = 0; ii < tid; ++ii) {
            offset += partial_sum[ii];
        }
        for (I i = start; i < end; ++i) {
            output[i] += offset;
        }
    }
    
    delete_array(partial_sum);
}

#endif /* SPGEMM_UTILITY_H */
