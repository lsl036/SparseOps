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
#include <set>
#include <vector>
#include <algorithm>

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

/**
 * @brief Sequential prefix sum (scan) with two outputs: regular and weighted
 *        Computes: out[i+1] = out[i] + in[i] and out1[i+1] = out1[i] + (in[i] * in_weight[i])
 *        Reference: utility.h seq_scan_V1
 * @tparam T Type
 * @param input Input array
 * @param output Output array (regular prefix sum)
 * @param output1 Output array (weighted prefix sum)
 * @param weights Weight array (one weight per input element)
 * @param n Array size
 */
template <typename T>
inline void seq_scan_dual(const T *input, T *output, T *output1, const T *weights, T n) {
    output[0] = 0;
    output1[0] = 0;
    for (T i = 0; i < n - 1; ++i) {
        output[i + 1] = output[i] + input[i];
        output1[i + 1] = output1[i] + (input[i] * weights[i]);
    }
}

/**
 * @brief Sequential prefix sum (scan) with array weights
 *        Computes weighted prefix sum: out[i+1] = out[i] + (in[i] * in_weight[i])
 *        Reference: utility.h seq_scan_V2
 * @tparam T Output type
 * @tparam I Input type
 * @param input Input array
 * @param output Output array (prefix sum)
 * @param weights Weight array (one weight per input element)
 * @param n Array size
 */
template <typename T, typename I>
inline void seq_scan_weighted(const I *input, T *output, const I *weights, I n) {
    output[0] = 0;
    for (I i = 0; i < n - 1; ++i) {
        output[i + 1] = output[i] + (input[i] * weights[i]);
    }
}

/**
 * @brief Parallel prefix sum (scan) with two outputs: regular and weighted
 *        Computes: out[i+1] = out[i] + in[i] and out1[i+1] = out1[i] + (in[i] * in_weight[i])
 *        Reference: utility.h line 337-379 (scan with two outputs)
 * @tparam T Type
 * @param input Input array
 * @param output Output array (regular prefix sum)
 * @param output1 Output array (weighted prefix sum)
 * @param weights Weight array (one weight per input element)
 * @param n Array size
 * @param tnum Number of threads for parallel execution (default: 56)
 * 
 * @note Uses sequential version for small arrays (N < 131072)
 *       Uses parallel version for large arrays
 */
template <typename T>
void scan(const T *input, T *output, T *output1, const T *weights, T n, int tnum = 56) {
    // Use sequential version for small arrays (matching reference: N < (1 << 17))
    if (n < (1 << 17)) {
        seq_scan_dual(input, output, output1, weights, n);
        return;
    }
    
    // Parallel version for large arrays
    T each_n = n / tnum;
    T *partial_sum = new_array<T>(tnum);
    T *partial_sum1 = new_array<T>(tnum);
    
    #pragma omp parallel num_threads(tnum)
    {
        int tid = omp_get_thread_num();
        T start = each_n * tid;
        T end = (tid < tnum - 1) ? start + each_n : n;
        
        output[start] = 0;
        output1[start] = 0;
        for (T i = start; i < end - 1; ++i) {
            output[i + 1] = output[i] + input[i];
            output1[i + 1] = output1[i] + (input[i] * weights[i]);
        }
        partial_sum[tid] = output[end - 1] + input[end - 1];
        partial_sum1[tid] = output1[end - 1] + (input[end - 1] * weights[end - 1]);
        
        #pragma omp barrier
        
        T offset = 0;
        T offset1 = 0;
        for (int ii = 0; ii < tid; ++ii) {
            offset += partial_sum[ii];
            offset1 += partial_sum1[ii];
        }
        for (T i = start; i < end; ++i) {
            output[i] += offset;
            output1[i] += offset1;
        }
    }
    
    delete_array(partial_sum);
    delete_array(partial_sum1);
}

/**
 * @brief Parallel prefix sum (scan) with array weights
 *        Computes weighted prefix sum: out[i+1] = out[i] + (in[i] * in_weight[i])
 *        Reference: utility.h line 383-424 (scan with in_weight array)
 * @tparam T Output type
 * @tparam I Input type
 * @param input Input array
 * @param output Output array (prefix sum)
 * @param weights Weight array (one weight per input element)
 * @param n Array size
 * @param tnum Number of threads for parallel execution (default: 56)
 * 
 * @note Uses sequential version for small arrays (N < 131072)
 *       Uses parallel version for large arrays
 */
template <typename T, typename I>
void scan(const I *input, T *output, const I *weights, I n, int tnum = 56) {
    // Use sequential version for small arrays (matching reference: N < (1 << 17))
    if (n < (1 << 17)) {
        seq_scan_weighted(input, output, weights, n);
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
            output[i + 1] = output[i] + (input[i] * weights[i]);
        }
        partial_sum[tid] = output[end - 1] + (input[end - 1] * weights[end - 1]);
        
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

/**
 * @brief Sequential prefix sum that modifies input array in-place (CumulativeSum)
 *        Reference: utility.h line 62-74
 *        Modifies arr in-place: arr[i] becomes prefix sum up to i-1, returns total sum
 * @tparam T Index type
 * @param arr Input/output array (modified in-place)
 * @param size Array size
 * @return T Total sum of all elements
 */
template <typename T>
inline T CumulativeSum(T *arr, T size) {
    T prev;
    T tempnz = 0;
    for (T i = 0; i < size; ++i) {
        prev = arr[i];
        arr[i] = tempnz;
        tempnz += prev;
    }
    return tempnz;  // return sum
}

// ============================================================================
// Jaccard Similarity Functions (for variable-length cluster generation)
// ============================================================================

/**
 * @brief Calculate Jaccard similarity score between two rows in a CSR matrix
 *        Reference: CSR.h line 766-776
 *        Jaccard similarity = |intersection| / |union|
 * @tparam IndexType Index type
 * @tparam ValueType Value type (not used, but kept for template consistency)
 * @param rowptr Row pointer array (CSR format)
 * @param colids Column index array (CSR format)
 * @param row_a First row index
 * @param row_b Second row index
 * @return double Jaccard similarity score (0.0 to 1.0)
 */
template <typename IndexType, typename ValueType>
inline double jaccard_similarity(const IndexType *rowptr, const IndexType *colids,
                                  IndexType row_a, IndexType row_b)
{
    // Count common elements (intersection)
    IndexType c = 0;
    std::set<IndexType> sb;
    
    // Insert all column indices of row_a into set
    for (IndexType j = rowptr[row_a]; j < rowptr[row_a + 1]; ++j) {
        sb.insert(colids[j]);
    }
    
    // Count how many column indices of row_b are in the set (intersection)
    for (IndexType j = rowptr[row_b]; j < rowptr[row_b + 1]; ++j) {
        if (sb.find(colids[j]) != sb.end()) {
            c++;
        }
    }
    
    // Calculate union size: |A| + |B| - |A ∩ B|
    IndexType u = (rowptr[row_a + 1] - rowptr[row_a]) + 
                  (rowptr[row_b + 1] - rowptr[row_b]) - c;
    
    // Return Jaccard similarity: intersection / union
    return (u == 0) ? 0.0 : (1.0 * c / u);
}

/**
 * @brief Generate offsets for variable-length cluster based on Jaccard similarity
 *        Reference: User-provided clustering algorithm
 *        Clusters consecutive rows with similarity >= similarity_threshold
 * @tparam IndexType Index type
 * @param rowptr Row pointer array (CSR format)
 * @param colids Column index array (CSR format)
 * @param num_rows Number of rows in the matrix
 * @param similarity_th Similarity threshold (default: 0.5)
 * @param max_cluster_size Maximum cluster size (-1 for unlimited, default: -1)
 * @param eps Epsilon for floating-point comparison (default: 1e-6)
 * @return std::vector<IndexType> Vector of cluster offsets (including 0 and num_rows)
 */
template <typename IndexType>
inline std::vector<IndexType> generate_offsets_jaccard(
    const IndexType *rowptr, const IndexType *colids,
    IndexType num_rows, double similarity_th = 0.5, 
    IndexType max_cluster_size = -1, double eps = 1e-6)
{
    std::vector<IndexType> offsets;
    IndexType curr_off = 0;
    offsets.push_back(curr_off);
    
    IndexType real_max_cluster_size = 0;
    
    while (curr_off < num_rows) {
        IndexType next_off = curr_off + 1;
        
        // Try to extend the cluster by adding similar rows
        while (next_off < num_rows) {
            double sim_score = jaccard_similarity<IndexType, double>(
                rowptr, colids, curr_off, next_off);
            
            // Break if similarity is too low
            if (sim_score < similarity_th - eps) {
                break;
            }
            
            // Break if max cluster size is reached
            if (max_cluster_size != -1 && (next_off - curr_off) == max_cluster_size) {
                break;
            }
            
            next_off += 1;
        }
        
        // Update real_max_cluster_size
        real_max_cluster_size = std::max(real_max_cluster_size, (next_off - curr_off));
        
        // Add the offset
        offsets.push_back(next_off);
        curr_off = next_off;
    }
    
    return offsets;
}

#endif /* SPGEMM_UTILITY_H */
