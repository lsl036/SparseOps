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
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <map>
#include <queue>
#include <utility>
#include <cstdint>
#include <functional>

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

template <typename IndexType>
inline bool csr_rows_are_strictly_sorted_unique(
    const IndexType *rowptr, const IndexType *colids, IndexType num_rows)
{
    for (IndexType r = 0; r < num_rows; ++r) {
        for (IndexType j = rowptr[r] + 1; j < rowptr[r + 1]; ++j) {
            if (colids[j - 1] >= colids[j]) return false;
        }
    }
    return true;
}

template <typename IndexType, typename ValueType>
inline double jaccard_similarity_sorted_unique(
    const IndexType *rowptr, const IndexType *colids, IndexType row_a, IndexType row_b)
{
    IndexType ia = rowptr[row_a];
    IndexType ib = rowptr[row_b];
    const IndexType enda = rowptr[row_a + 1];
    const IndexType endb = rowptr[row_b + 1];
    IndexType common = 0;

    while (ia < enda && ib < endb) {
        const IndexType ca = colids[ia];
        const IndexType cb = colids[ib];
        if (ca == cb) {
            ++common;
            ++ia;
            ++ib;
        } else if (ca < cb) {
            ++ia;
        } else {
            ++ib;
        }
    }

    const IndexType len_a = enda - rowptr[row_a];
    const IndexType len_b = endb - rowptr[row_b];
    const IndexType uni = len_a + len_b - common;
    return (uni == 0) ? 0.0 : (1.0 * common / uni);
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

// ============================================================================
// Hierarchical Clustering (reference: HierarchicalClusterSpGEMM.cpp)
// ============================================================================

template <typename IndexType>
struct PairHash {
    static inline uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    size_t operator()(const std::pair<IndexType, IndexType> &p) const {
        uint64_t a = splitmix64(static_cast<uint64_t>(p.first));
        uint64_t b = splitmix64(static_cast<uint64_t>(p.second));
        return static_cast<size_t>(a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2)));
    }
};

/**
 * @brief Hierarchical clustering v0: Union-Find + priority queue by similarity from close_pairs.
 *        Returns map<root, vector<row indices in that cluster>>.
 * @param rowptr A's row pointer (CSR)
 * @param colids A's column indices (CSR)
 * @param num_rows A's number of rows
 * @param close_pairs (i,j) -> similarity; may be extended with on-the-fly Jaccard for root pairs
 * @param cluster_size Maximum cluster size; when a root's size >= cluster_size, it is marked invalid
 */
template <typename IndexType, typename ValueType>
inline std::map<IndexType, std::vector<IndexType>> hierarchical_clustering_v0(
    const IndexType *rowptr, const IndexType *colids, IndexType num_rows,
    std::map<std::pair<IndexType, IndexType>, ValueType> &close_pairs,
    IndexType cluster_size)
{
    using item_t = std::pair<ValueType, std::pair<IndexType, IndexType>>;
    /* Same comparator as v1: max by score, tie-break by (i,j) for deterministic merge order. */
    auto cmp = [](const item_t &a, const item_t &b) {
        return a.first < b.first || (a.first == b.first && a.second < b.second);
    };
    std::priority_queue<item_t, std::vector<item_t>, decltype(cmp)> sims(cmp);

    for (const auto &p : close_pairs) {
        IndexType ii = p.first.first, jj = p.first.second;
        if (ii > jj) std::swap(ii, jj);
        sims.push(std::make_pair(p.second, std::make_pair(ii, jj)));
    }

    std::vector<IndexType> clusters(num_rows);
    std::vector<IndexType> sz(num_rows);
    std::vector<int> valid(num_rows, 1);
    for (IndexType i = 0; i < num_rows; i++) {
        clusters[i] = i;
        sz[i] = 1;
    }
    int nclusters = static_cast<int>(num_rows);

    while (!sims.empty() && nclusters != 0) {
        item_t s = sims.top();
        sims.pop();
        IndexType i = s.second.first;
        IndexType j = s.second.second;
        if (i >= num_rows || j >= num_rows) continue;

        if (clusters[i] == i && clusters[j] == j) {  // i 和 j 都是簇的代表（根节点）
            if (!valid[i] || !valid[j]) continue;
            nclusters--;
            if (sz[i] < sz[j]) {
                clusters[i] = j;        // i 所在的簇指向 j，j 成为新的父节点/根
                sz[j] += sz[i];
                if (sz[j] >= cluster_size) {
                    valid[j] = 0;
                    nclusters--;
                }
            } else {
                clusters[j] = i;        // j 所在的簇指向 i，i 成为新的父节点/根
                sz[i] += sz[j];
                if (sz[i] >= cluster_size) {
                    valid[i] = 0;
                    nclusters--;
                }
            }
        } else {
            while (i != clusters[i]) {
                clusters[i] = clusters[clusters[i]];
                i = clusters[i];
            }
            while (j != clusters[j]) {
                clusters[j] = clusters[clusters[j]];
                j = clusters[j];
            }
            if (!valid[i] || !valid[j]) continue;
            if (i != j) {   // 当两个行已经属于不同簇时，计算它们代表行的精确 Jaccard 相似度，并加入优先队列
                IndexType pi = (i < j) ? i : j, pj = (i < j) ? j : i;
                auto p = std::make_pair(pi, pj);
                if (close_pairs.find(p) == close_pairs.end()) {
                    ValueType s_val = static_cast<ValueType>(jaccard_similarity<IndexType, ValueType>(rowptr, colids, pi, pj));
                    sims.push(std::make_pair(s_val, p));
                    close_pairs[p] = s_val;
                }
            }
        }
    }

    std::map<IndexType, std::vector<IndexType>> reordered_dict;
    for (IndexType i = 0; i < num_rows; i++) {
        IndexType j = i;
        while (j != clusters[j]) j = clusters[j];
        reordered_dict[j].push_back(i);
    }
    return reordered_dict;
}

/**
 * @brief Hierarchical clustering v1: no map; uses set for seen pairs and priority queue.
 *        - Full path compression in union-find.
 *        - Priority queue by score; when merging, may discover new root pairs and push with exact Jaccard (no map).
 *        - Output permutation and offsets directly (flat arrays).
 *        PairLike must have .i, .j, .score (e.g. CandidatePair<IndexType, ValueType>).
 */
template <typename IndexType, typename ValueType, typename PairLike>
inline void hierarchical_clustering_v1(
    const IndexType *rowptr, const IndexType *colids, IndexType num_rows,
    const std::vector<PairLike> &pairs,
    IndexType cluster_size,
    std::vector<IndexType> &permutation_out,
    std::vector<IndexType> &offsets_out)
{
    permutation_out.clear();
    offsets_out.clear();

    std::vector<IndexType> clusters(num_rows);
    std::vector<IndexType> sz(num_rows);
    std::vector<int> valid(num_rows, 1);
    for (IndexType i = 0; i < num_rows; i++) {
        clusters[i] = i;
        sz[i] = 1;
    }

    /* Find with full path compression: return root and set all on path to root. */
    auto find = [&clusters](IndexType x) -> IndexType {
        IndexType root = x;
        while (clusters[root] != root) root = clusters[root];
        while (clusters[x] != root) {
            IndexType next = clusters[x];
            clusters[x] = root;
            x = next;
        }
        return root;
    };

    using item_t = std::pair<ValueType, std::pair<IndexType, IndexType>>;
    /* Same comparator as v0: max by score, tie-break by (i,j) for same merge order. */
    auto cmp = [](const item_t &a, const item_t &b) {
        return a.first < b.first || (a.first == b.first && a.second < b.second);
    };
    std::priority_queue<item_t, std::vector<item_t>, decltype(cmp)> sims(cmp);
    std::set<std::pair<IndexType, IndexType>> seen;

    for (const auto &p : pairs) {
        IndexType ii = p.i, jj = p.j;
        if (ii > jj) std::swap(ii, jj);
        if (ii == jj || ii >= num_rows || jj >= num_rows) continue;
        auto key = std::make_pair(ii, jj);
        if (seen.find(key) != seen.end()) continue;
        seen.insert(key);
        sims.push(std::make_pair(static_cast<ValueType>(p.score), key));
    }

    int nclusters = static_cast<int>(num_rows);
    while (!sims.empty() && nclusters > 0) {
        item_t top = sims.top();
        sims.pop();
        IndexType i = top.second.first;
        IndexType j = top.second.second;
        if (i >= num_rows || j >= num_rows) continue;
        IndexType ri = find(i);
        IndexType rj = find(j);
        if (ri == rj) continue;
        if (!valid[ri] || !valid[rj]) continue;

        if (clusters[i] == i && clusters[j] == j) {
            /* Both are roots: merge. */
            nclusters--;
            if (sz[ri] < sz[rj]) {
                clusters[ri] = rj;
                sz[rj] += sz[ri];
                if (sz[rj] >= cluster_size) {
                    valid[rj] = 0;
                    nclusters--;
                }
            } else {
                clusters[rj] = ri;
                sz[ri] += sz[rj];
                if (sz[ri] >= cluster_size) {
                    valid[ri] = 0;
                    nclusters--;
                }
            }
        } else {
            /* Discovered (ri, rj) when at least one of i,j was already merged; add if not seen. */
            IndexType pi = (ri < rj) ? ri : rj;
            IndexType pj = (ri < rj) ? rj : ri;
            auto key = std::make_pair(pi, pj);
            if (seen.find(key) == seen.end()) {
                seen.insert(key);
                ValueType s_val = static_cast<ValueType>(jaccard_similarity<IndexType, ValueType>(rowptr, colids, pi, pj));
                sims.push(std::make_pair(s_val, key));
            }
        }
    }

    /* Build permutation and offsets from union-find result (no map). */
    std::vector<IndexType> root(num_rows);
    for (IndexType i = 0; i < num_rows; i++)
        root[i] = find(i);

    std::vector<IndexType> count(num_rows, 0);
    for (IndexType i = 0; i < num_rows; i++)
        count[root[i]]++;

    std::vector<IndexType> sorted_roots;
    sorted_roots.reserve(static_cast<size_t>(num_rows));
    for (IndexType r = 0; r < num_rows; r++)
        if (count[r] > 0) sorted_roots.push_back(r);

    std::vector<IndexType> cluster_id(num_rows, static_cast<IndexType>(-1));
    for (size_t c = 0; c < sorted_roots.size(); c++)
        cluster_id[sorted_roots[c]] = static_cast<IndexType>(c);

    offsets_out.resize(sorted_roots.size() + 1);
    offsets_out[0] = 0;
    for (size_t c = 0; c < sorted_roots.size(); c++)
        offsets_out[c + 1] = offsets_out[c] + count[sorted_roots[c]];

    permutation_out.resize(static_cast<size_t>(num_rows));
    std::vector<IndexType> curr(sorted_roots.size());
    for (size_t c = 0; c < sorted_roots.size(); c++)
        curr[c] = offsets_out[c];

    for (IndexType i = 0; i < num_rows; i++) {
        IndexType r = root[i];
        IndexType c = cluster_id[r];
        permutation_out[static_cast<size_t>(curr[c]++)] = i;
    }
}

/**
 * @brief Fast v0-style hierarchical clustering from compact candidate pairs.
 *        Keeps the same priority-queue merge model as v0, but avoids std::map:
 *        - input pairs are already sorted/deduplicated by vector LSH;
 *        - initial pairs are checked by binary search, and only discovered root pairs use unordered_set;
 *        - permutation and offsets are built directly.
 */
template <typename IndexType, typename ValueType, typename PairLike>
inline void hierarchical_clustering_v0_fast(
    const IndexType *rowptr, const IndexType *colids, IndexType num_rows,
    const std::vector<PairLike> &pairs,
    IndexType cluster_size,
    std::vector<IndexType> &permutation_out,
    std::vector<IndexType> &offsets_out)
{
    permutation_out.clear();
    offsets_out.clear();

    std::vector<IndexType> clusters(num_rows);
    std::vector<IndexType> sz(num_rows);
    std::vector<int> valid(num_rows, 1);
    for (IndexType i = 0; i < num_rows; i++) {
        clusters[i] = i;
        sz[i] = 1;
    }

    auto find = [&clusters](IndexType x) -> IndexType {
        IndexType root = x;
        while (clusters[root] != root) root = clusters[root];
        while (clusters[x] != root) {
            IndexType next = clusters[x];
            clusters[x] = root;
            x = next;
        }
        return root;
    };

    using pair_t = std::pair<IndexType, IndexType>;
    using item_t = std::pair<ValueType, pair_t>;
    auto cmp = [](const item_t &a, const item_t &b) {
        return a.first < b.first || (a.first == b.first && a.second < b.second);
    };
    std::map<ValueType, std::vector<pair_t>, std::greater<ValueType>> initial_buckets;

    for (const auto &p : pairs) {
        IndexType ii = p.i, jj = p.j;
        if (ii > jj) std::swap(ii, jj);
        if (ii == jj || ii >= num_rows || jj >= num_rows) continue;
        pair_t key = std::make_pair(ii, jj);
        initial_buckets[static_cast<ValueType>(p.score)].push_back(key);
    }
    std::priority_queue<item_t, std::vector<item_t>, decltype(cmp)> dynamic_sims(cmp);

    auto initial_has_pair = [&pairs](const pair_t &key) -> bool {
        auto it = std::lower_bound(
            pairs.begin(), pairs.end(), key,
            [](const PairLike &lhs, const pair_t &rhs) {
                if (lhs.i != rhs.first) return lhs.i < rhs.first;
                return lhs.j < rhs.second;
            });
        return it != pairs.end() && it->i == key.first && it->j == key.second;
    };

    std::unordered_set<pair_t, PairHash<IndexType>> discovered;
    discovered.reserve(pairs.size() / 8 + 1024);
    const bool use_sorted_jaccard =
        csr_rows_are_strictly_sorted_unique<IndexType>(rowptr, colids, num_rows);
    auto exact_jaccard = [&](IndexType a, IndexType b) -> ValueType {
        if (use_sorted_jaccard) {
            return static_cast<ValueType>(
                jaccard_similarity_sorted_unique<IndexType, ValueType>(rowptr, colids, a, b));
        }
        return static_cast<ValueType>(jaccard_similarity<IndexType, ValueType>(rowptr, colids, a, b));
    };

    auto bucket_it = initial_buckets.begin();
    auto advance_initial_bucket = [&]() {
        while (bucket_it != initial_buckets.end() && bucket_it->second.empty()) ++bucket_it;
    };
    auto has_initial_pair = [&]() -> bool {
        advance_initial_bucket();
        return bucket_it != initial_buckets.end();
    };
    auto peek_initial_pair = [&]() -> item_t {
        return std::make_pair(bucket_it->first, bucket_it->second.back());
    };
    auto pop_initial_pair = [&]() -> item_t {
        item_t item = peek_initial_pair();
        bucket_it->second.pop_back();
        return item;
    };

    int nclusters = static_cast<int>(num_rows);
    while ((has_initial_pair() || !dynamic_sims.empty()) && nclusters > 0) {
        item_t top;
        if (dynamic_sims.empty()) {
            top = pop_initial_pair();
        } else if (!has_initial_pair()) {
            top = dynamic_sims.top();
            dynamic_sims.pop();
        } else {
            item_t initial_top = peek_initial_pair();
            if (cmp(dynamic_sims.top(), initial_top)) {
                top = pop_initial_pair();
            } else {
                top = dynamic_sims.top();
                dynamic_sims.pop();
            }
        }

        IndexType i = top.second.first;
        IndexType j = top.second.second;
        if (i >= num_rows || j >= num_rows) continue;

        if (clusters[i] == i && clusters[j] == j) {
            if (!valid[i] || !valid[j]) continue;
            nclusters--;
            if (sz[i] < sz[j]) {
                clusters[i] = j;
                sz[j] += sz[i];
                if (sz[j] >= cluster_size) {
                    valid[j] = 0;
                    nclusters--;
                }
            } else {
                clusters[j] = i;
                sz[i] += sz[j];
                if (sz[i] >= cluster_size) {
                    valid[i] = 0;
                    nclusters--;
                }
            }
        } else {
            IndexType ri = find(i);
            IndexType rj = find(j);
            if (!valid[ri] || !valid[rj]) continue;
            if (ri != rj) {
                IndexType pi = (ri < rj) ? ri : rj;
                IndexType pj = (ri < rj) ? rj : ri;
                pair_t key = std::make_pair(pi, pj);
                if (!initial_has_pair(key) && discovered.insert(key).second) {
                    ValueType s_val = exact_jaccard(pi, pj);
                    dynamic_sims.push(std::make_pair(s_val, key));
                }
            }
        }
    }

    std::vector<IndexType> root(num_rows);
    for (IndexType i = 0; i < num_rows; i++)
        root[i] = find(i);

    std::vector<IndexType> count(num_rows, 0);
    for (IndexType i = 0; i < num_rows; i++)
        count[root[i]]++;

    std::vector<IndexType> sorted_roots;
    sorted_roots.reserve(static_cast<size_t>(num_rows));
    for (IndexType r = 0; r < num_rows; r++)
        if (count[r] > 0) sorted_roots.push_back(r);

    std::vector<IndexType> cluster_id(num_rows, static_cast<IndexType>(-1));
    for (size_t c = 0; c < sorted_roots.size(); c++)
        cluster_id[sorted_roots[c]] = static_cast<IndexType>(c);

    offsets_out.resize(sorted_roots.size() + 1);
    offsets_out[0] = 0;
    for (size_t c = 0; c < sorted_roots.size(); c++)
        offsets_out[c + 1] = offsets_out[c] + count[sorted_roots[c]];

    permutation_out.resize(static_cast<size_t>(num_rows));
    std::vector<IndexType> curr(sorted_roots.size());
    for (size_t c = 0; c < sorted_roots.size(); c++)
        curr[c] = offsets_out[c];

    for (IndexType i = 0; i < num_rows; i++) {
        IndexType r = root[i];
        IndexType c = cluster_id[r];
        permutation_out[static_cast<size_t>(curr[c]++)] = i;
    }
}

/**
 * @brief Convert reordered_dict (map<root, vector<row indices>>) to permutation and offsets
 *        permutation[new_row] = original row index; offsets = [0, len0, len0+len1, ...].
 */
template <typename IndexType>
inline void reordered_dict_to_permutation_and_offsets(
    const std::map<IndexType, std::vector<IndexType>> &reordered_dict,
    IndexType num_rows,
    std::vector<IndexType> &permutation_out,
    std::vector<IndexType> &offsets_out)
{
    permutation_out.clear();
    offsets_out.clear();
    offsets_out.push_back(0);
    for (const auto &kv : reordered_dict) {
        const std::vector<IndexType> &rows = kv.second;
        for (IndexType r : rows) {
            permutation_out.push_back(r);
        }
        offsets_out.push_back(static_cast<IndexType>(permutation_out.size()));
    }
    (void)num_rows;
}

#endif /* SPGEMM_UTILITY_H */
