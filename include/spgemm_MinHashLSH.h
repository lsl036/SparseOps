#ifndef SPGEMM_MINHASH_LSH_H
#define SPGEMM_MINHASH_LSH_H

/**
 * @file spgemm_MinHashLSH.h
 * @brief MinHash and LSH preprocessing for SpGEMM (candidate pair generation).
 *        Phase 1: MinHash signatures; Phase 2+: LSH banding etc. to be added.
 *
 * MinHash: each row of A is a set of column indices; we compute k independent
 * MinHash values per row. For two rows, the fraction of dimensions where
 * signatures match estimates Jaccard similarity.
 */

#include <vector>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <map>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX512F__)
#include <immintrin.h>
#endif
#if defined(__GNUC__) && !defined(__clang__) && defined(__has_include)
#if __has_include(<parallel/algorithm>)
#include <parallel/algorithm>
#define SPGEMM_LSH_USE_PARALLEL_SORT 1
#endif
#endif

// ============================================================================
// MinHash 签名 (Phase 1)
// ============================================================================

namespace minhash_internal {

/** LCG step for generating hash parameters from seed. */
inline uint64_t lcg_next(uint64_t &state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state;
}

/** Return h_i(x) = a_i * x + b_i (overflow = mod 2^64). a_i must be odd. */
inline uint64_t hash_apply(uint64_t a, uint64_t b, uint64_t x) {
    return a * x + b;
}

} // namespace minhash_internal

/**
 * @brief Compute MinHash signatures for each row of a CSR matrix.
 *
 * Each row is treated as a set of column indices. For k hash functions,
 * we store the minimum hash value over that row's columns in each dimension.
 * Same column appearing multiple times in a row is effectively one element
 * (min over repeated hashes is unchanged).
 *
 * @tparam IndexType  Row/column index type (e.g. int, long).
 * @param rowptr      CSR row pointer; row i has columns in [rowptr[i], rowptr[i+1]).
 * @param col_index   Column indices (length = rowptr[num_rows]).
 * @param num_rows    Number of rows.
 * @param k           Signature length (number of MinHash dimensions).
 * @param seed        Random seed for hash family (default 12345).
 * @return            signatures[row][dim] = dim-th MinHash value for that row.
 *                    Empty rows get UINT64_MAX in every dimension.
 */
template <typename IndexType>
std::vector<std::vector<uint64_t>> minhash_signatures(
    const IndexType *rowptr,
    const IndexType *col_index,
    IndexType num_rows,
    int k,
    uint64_t seed = 12345)
{
    using namespace minhash_internal;
    std::vector<std::vector<uint64_t>> signatures(
        static_cast<size_t>(num_rows),
        std::vector<uint64_t>(static_cast<size_t>(k), std::numeric_limits<uint64_t>::max()));

    if (k <= 0) return signatures;

    // Build k hash coefficients (a_i odd, b_i arbitrary).
    std::vector<uint64_t> a(k), b(k);
    uint64_t state = seed;
    for (int i = 0; i < k; ++i) {
        lcg_next(state);
        a[i] = (state >> 1) * 2 + 1;  // odd
        lcg_next(state);
        b[i] = state;
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (IndexType r = 0; r < num_rows; ++r) {
        IndexType start = rowptr[r];
        IndexType end   = rowptr[r + 1];
        std::vector<uint64_t> &sig = signatures[static_cast<size_t>(r)];

        for (IndexType j = start; j < end; ++j) {
            uint64_t x = static_cast<uint64_t>(col_index[j]);
            for (int d = 0; d < k; ++d) {
                uint64_t h = hash_apply(a[d], b[d], x);
                if (h < sig[static_cast<size_t>(d)])
                    sig[static_cast<size_t>(d)] = h;
            }
        }
    }

    return signatures;
}

/**
 * @brief Compute MinHash signatures into a single contiguous buffer (row-major: row i at sigs[i*k..(i+1)*k-1]).
 *        Faster and more cache-friendly when followed by LSH on the same buffer.
 */
template <typename IndexType>
std::vector<uint64_t> minhash_signatures_flat(
    const IndexType *rowptr,
    const IndexType *col_index,
    IndexType num_rows,
    int k,
    uint64_t seed = 12345)
{
    using namespace minhash_internal;
    const size_t n = static_cast<size_t>(num_rows) * static_cast<size_t>(k);
    std::vector<uint64_t> sigs(n, std::numeric_limits<uint64_t>::max());
    if (k <= 0) return sigs;

    std::vector<uint64_t> a(static_cast<size_t>(k)), b(static_cast<size_t>(k));
    uint64_t state = seed;
    for (int i = 0; i < k; ++i) {
        lcg_next(state);
        a[static_cast<size_t>(i)] = (state >> 1) * 2 + 1;
        lcg_next(state);
        b[static_cast<size_t>(i)] = state;
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (IndexType r = 0; r < num_rows; ++r) {
        IndexType start = rowptr[r];
        IndexType end   = rowptr[r + 1];
        uint64_t *sig   = sigs.data() + static_cast<size_t>(r) * static_cast<size_t>(k);

        for (IndexType j = start; j < end; ++j) {
            uint64_t x = static_cast<uint64_t>(col_index[j]);
            int d = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
            __m512i vx = _mm512_set1_epi64(static_cast<int64_t>(x));
            for (; d + 8 <= k; d += 8) {
                __m512i va = _mm512_loadu_epi64(a.data() + d);
                __m512i vb = _mm512_loadu_epi64(b.data() + d);
                __m512i vh = _mm512_add_epi64(_mm512_mullo_epi64(va, vx), vb);
                __m512i vsig = _mm512_loadu_epi64(sig + d);
                __m512i vmin = _mm512_min_epu64(vsig, vh);
                _mm512_storeu_epi64(sig + d, vmin);
            }
#endif
            for (; d + 4 <= k; d += 4) {
                uint64_t h0 = hash_apply(a[static_cast<size_t>(d)], b[static_cast<size_t>(d)], x);
                uint64_t h1 = hash_apply(a[static_cast<size_t>(d+1)], b[static_cast<size_t>(d+1)], x);
                uint64_t h2 = hash_apply(a[static_cast<size_t>(d+2)], b[static_cast<size_t>(d+2)], x);
                uint64_t h3 = hash_apply(a[static_cast<size_t>(d+3)], b[static_cast<size_t>(d+3)], x);
                if (h0 < sig[d])   sig[d]   = h0;
                if (h1 < sig[d+1]) sig[d+1] = h1;
                if (h2 < sig[d+2]) sig[d+2] = h2;
                if (h3 < sig[d+3]) sig[d+3] = h3;
            }
            for (; d < k; ++d) {
                uint64_t h = hash_apply(a[static_cast<size_t>(d)], b[static_cast<size_t>(d)], x);
                if (h < sig[d]) sig[d] = h;
            }
        }
    }
    return sigs;
}

/** Pointer version for flat layout; k = number of dimensions. */
inline double minhash_estimated_jaccard(const uint64_t *sig_a, const uint64_t *sig_b, int k) {
    if (k <= 0) return 0.0;
    int match = 0;
#if defined(__AVX512F__)
    // std::cout << "using AVX512 for minhash_estimated_jaccard" << std::endl;
    int d = 0;
    for (; d + 8 <= k; d += 8) {
        __m512i va = _mm512_loadu_epi64(sig_a + d);
        __m512i vb = _mm512_loadu_epi64(sig_b + d);
        __mmask8 m = _mm512_cmpeq_epi64_mask(va, vb);
        match += __builtin_popcount(static_cast<unsigned int>(m));
    }
    for (; d < k; ++d)
        if (sig_a[d] == sig_b[d]) ++match;
#else
    #pragma omp simd
    // std::cout << "using SIMD for minhash_estimated_jaccard" << std::endl;
    for (int d = 0; d < k; ++d)
        if (sig_a[d] == sig_b[d]) ++match;
#endif
    return static_cast<double>(match) / static_cast<double>(k);
}

/**
 * @brief Estimate Jaccard similarity from two MinHash signature vectors.
 *
 * Estimate = (number of dimensions where sig_a[d] == sig_b[d]) / k.
 * If either row had an empty set, signatures contain UINT64_MAX; dimensions
 * where both are UINT64_MAX are counted as match (both empty).
 *
 * @param sig_a  MinHash signature of row a (length k).
 * @param sig_b  MinHash signature of row b (length k).
 * @return       Estimated Jaccard in [0, 1].
 */
inline double minhash_estimated_jaccard(
    const std::vector<uint64_t> &sig_a,
    const std::vector<uint64_t> &sig_b)
{
    if (sig_a.size() != sig_b.size() || sig_a.empty())
        return 0.0;
    return minhash_estimated_jaccard(sig_a.data(), sig_b.data(), static_cast<int>(sig_a.size()));
}

// ============================================================================
// LSH 候选对 (Phase 2)
// ============================================================================

/** Compact candidate pair for vector-based API: i, j (row indices, i < j), score (e.g. MinHash-est. Jaccard). */
template <typename IndexType, typename ValueType>
struct CandidatePair {
    IndexType i;
    IndexType j;
    ValueType score;
};

namespace lsh_internal {

/** If bucket size exceeds this, use fixed-window sampling instead of full O(n^2) pairs. */
constexpr size_t k_bucket_size_limit = 256;
/** When bucket is over limit, each row pairs with at most this many following rows. */
constexpr int k_window_pairs = 16;

/** Hash a band of r MinHash values to a bucket id. */
inline uint64_t band_hash(const uint64_t *band_sig, int r) {
    const uint64_t mul = 0x9e3779b97f4a7c15ULL;
    uint64_t h = 0;
    for (int i = 0; i < r; ++i)
        h = h * mul + band_sig[i];
    return h;
}

/** Parallel sort when available (GCC parallel mode), else std::sort. */
template <typename It>
inline void parallel_sort_pairs(It first, It last) {
#ifdef SPGEMM_LSH_USE_PARALLEL_SORT
    __gnu_parallel::sort(first, last);
#else
    std::sort(first, last);
#endif
}

/** LSH on flat signatures [row-major: row i at sigs + i*k], with parallel band loop and parallel Jaccard fill. */
template <typename IndexType, typename ValueType>
std::map<std::pair<IndexType, IndexType>, ValueType> lsh_candidate_pairs_from_flat(
    const uint64_t *sigs, size_t num_rows, int k, int num_bands)
{
    using pair_t = std::pair<IndexType, IndexType>;
    std::map<pair_t, ValueType> out;
    if (!sigs || num_rows == 0 || num_bands <= 0 || k <= 0) return out;
    const int r = k / num_bands;
    if (r <= 0 || k != r * num_bands) return out;

    std::vector<pair_t> all_pairs;
#ifdef _OPENMP
    #pragma omp parallel
    {
        std::vector<pair_t> my_pairs;
        #pragma omp for schedule(guided)
        for (int band = 0; band < num_bands; ++band) {
            std::unordered_map<uint64_t, std::vector<IndexType>> buckets;
            const int offset = band * r;

            for (size_t row = 0; row < num_rows; ++row) {
                const uint64_t *row_sig = sigs + row * static_cast<size_t>(k);
                uint64_t bh = band_hash(row_sig + offset, r);
                buckets[bh].push_back(static_cast<IndexType>(row));
            }

            for (const auto &kv : buckets) {
                const std::vector<IndexType> &rows = kv.second;
                if (rows.size() <= k_bucket_size_limit) {
                    for (size_t a = 0; a < rows.size(); ++a) {
                        for (size_t b = a + 1; b < rows.size(); ++b) {
                            IndexType i = rows[a], j = rows[b];
                            if (i > j) std::swap(i, j);
                            my_pairs.push_back(std::make_pair(i, j));
                        }
                    }
                } else {
                    for (size_t a = 0; a < rows.size(); ++a) {
                        size_t b_end = a + 1 + static_cast<size_t>(k_window_pairs);
                        if (b_end > rows.size()) b_end = rows.size();
                        for (size_t b = a + 1; b < b_end; ++b) {
                            IndexType i = rows[a], j = rows[b];
                            if (i > j) std::swap(i, j);
                            my_pairs.push_back(std::make_pair(i, j));
                        }
                    }
                }
            }
        }
        #pragma omp critical
        all_pairs.insert(all_pairs.end(), my_pairs.begin(), my_pairs.end());
    }
#else
    for (int band = 0; band < num_bands; ++band) {
        std::unordered_map<uint64_t, std::vector<IndexType>> buckets;
        const int offset = band * r;
        for (size_t row = 0; row < num_rows; ++row) {
            const uint64_t *row_sig = sigs + row * static_cast<size_t>(k);
            uint64_t bh = band_hash(row_sig + offset, r);
            buckets[bh].push_back(static_cast<IndexType>(row));
        }
        for (const auto &kv : buckets) {
            const std::vector<IndexType> &rows = kv.second;
            if (rows.size() <= k_bucket_size_limit) {
                for (size_t a = 0; a < rows.size(); ++a) {
                    for (size_t b = a + 1; b < rows.size(); ++b) {
                        IndexType i = rows[a], j = rows[b];
                        if (i > j) std::swap(i, j);
                        all_pairs.push_back(std::make_pair(i, j));
                    }
                }
            } else {
                for (size_t a = 0; a < rows.size(); ++a) {
                    size_t b_end = a + 1 + static_cast<size_t>(k_window_pairs);
                    if (b_end > rows.size()) b_end = rows.size();
                    for (size_t b = a + 1; b < b_end; ++b) {
                        IndexType i = rows[a], j = rows[b];
                        if (i > j) std::swap(i, j);
                        all_pairs.push_back(std::make_pair(i, j));
                    }
                }
            }
        }
    }
#endif

    parallel_sort_pairs(all_pairs.begin(), all_pairs.end());
    all_pairs.erase(std::unique(all_pairs.begin(), all_pairs.end()), all_pairs.end());

    const size_t np = all_pairs.size();
    std::vector<ValueType> est_vals(np);
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (size_t i = 0; i < np; ++i) {
        IndexType ii = all_pairs[i].first, jj = all_pairs[i].second;
        est_vals[i] = static_cast<ValueType>(minhash_estimated_jaccard(
            sigs + static_cast<size_t>(ii) * static_cast<size_t>(k),
            sigs + static_cast<size_t>(jj) * static_cast<size_t>(k), k));
    }
    for (size_t i = 0; i < np; ++i)
        out[all_pairs[i]] = est_vals[i];

    return out;
}

/** Vector-based LSH on flat signatures: returns compact array, uses parallel sort, no map. */
template <typename IndexType, typename ValueType>
std::vector<CandidatePair<IndexType, ValueType>> lsh_candidate_pairs_from_flat_vector(
    const uint64_t *sigs, size_t num_rows, int k, int num_bands)
{
    using pair_t = std::pair<IndexType, IndexType>;
    std::vector<CandidatePair<IndexType, ValueType>> result;
    if (!sigs || num_rows == 0 || num_bands <= 0 || k <= 0) return result;
    const int r = k / num_bands;
    if (r <= 0 || k != r * num_bands) return result;

    std::vector<pair_t> all_pairs;
#ifdef _OPENMP
    #pragma omp parallel
    {
        std::vector<pair_t> my_pairs;
        #pragma omp for schedule(guided)
        for (int band = 0; band < num_bands; ++band) {
            std::unordered_map<uint64_t, std::vector<IndexType>> buckets;
            const int offset = band * r;
            for (size_t row = 0; row < num_rows; ++row) {
                const uint64_t *row_sig = sigs + row * static_cast<size_t>(k);
                uint64_t bh = band_hash(row_sig + offset, r);
                buckets[bh].push_back(static_cast<IndexType>(row));
            }
            for (const auto &kv : buckets) {
                const std::vector<IndexType> &rows = kv.second;
                if (rows.size() <= k_bucket_size_limit) {
                    for (size_t a = 0; a < rows.size(); ++a) {
                        for (size_t b = a + 1; b < rows.size(); ++b) {
                            IndexType i = rows[a], j = rows[b];
                            if (i > j) std::swap(i, j);
                            my_pairs.push_back(std::make_pair(i, j));
                        }
                    }
                } else {
                    for (size_t a = 0; a < rows.size(); ++a) {
                        size_t b_end = a + 1 + static_cast<size_t>(k_window_pairs);
                        if (b_end > rows.size()) b_end = rows.size();
                        for (size_t b = a + 1; b < b_end; ++b) {
                            IndexType i = rows[a], j = rows[b];
                            if (i > j) std::swap(i, j);
                            my_pairs.push_back(std::make_pair(i, j));
                        }
                    }
                }
            }
        }
        #pragma omp critical
        all_pairs.insert(all_pairs.end(), my_pairs.begin(), my_pairs.end());
    }
#else
    for (int band = 0; band < num_bands; ++band) {
        std::unordered_map<uint64_t, std::vector<IndexType>> buckets;
        const int offset = band * r;
        for (size_t row = 0; row < num_rows; ++row) {
            const uint64_t *row_sig = sigs + row * static_cast<size_t>(k);
            uint64_t bh = band_hash(row_sig + offset, r);
            buckets[bh].push_back(static_cast<IndexType>(row));
        }
        for (const auto &kv : buckets) {
            const std::vector<IndexType> &rows = kv.second;
            if (rows.size() <= k_bucket_size_limit) {
                for (size_t a = 0; a < rows.size(); ++a) {
                    for (size_t b = a + 1; b < rows.size(); ++b) {
                        IndexType i = rows[a], j = rows[b];
                        if (i > j) std::swap(i, j);
                        all_pairs.push_back(std::make_pair(i, j));
                    }
                }
            } else {
                for (size_t a = 0; a < rows.size(); ++a) {
                    size_t b_end = a + 1 + static_cast<size_t>(k_window_pairs);
                    if (b_end > rows.size()) b_end = rows.size();
                    for (size_t b = a + 1; b < b_end; ++b) {
                        IndexType i = rows[a], j = rows[b];
                        if (i > j) std::swap(i, j);
                        all_pairs.push_back(std::make_pair(i, j));
                    }
                }
            }
        }
    }
#endif

    parallel_sort_pairs(all_pairs.begin(), all_pairs.end());
    all_pairs.erase(std::unique(all_pairs.begin(), all_pairs.end()), all_pairs.end());

    const size_t np = all_pairs.size();
    result.resize(np);
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (size_t idx = 0; idx < np; ++idx) {
        IndexType ii = all_pairs[idx].first, jj = all_pairs[idx].second;
        result[idx].i = ii;
        result[idx].j = jj;
        result[idx].score = static_cast<ValueType>(minhash_estimated_jaccard(
            sigs + static_cast<size_t>(ii) * static_cast<size_t>(k),
            sigs + static_cast<size_t>(jj) * static_cast<size_t>(k), k));
    }
    return result;
}

} // namespace lsh_internal

/**
 * @brief Generate candidate pairs via LSH banding on MinHash signatures.
 *
 * Divides k dimensions into num_bands bands of r = k/num_bands rows each.
 * Two rows are a candidate pair if they share at least one bucket (same band,
 * same band-hash). For each candidate pair, the value stored is the
 * MinHash-estimated Jaccard (matching ratio of the two signatures), so that
 * callers can optionally overwrite with exact Jaccard later.
 *
 * @tparam IndexType  Row index type.
 * @tparam ValueType  Similarity value type (e.g. double); stored as estimated Jaccard.
 * @param signatures  MinHash signatures from minhash_signatures (num_rows x k).
 * @param num_bands   Number of bands (must satisfy k % num_bands == 0).
 * @return            Map (i,j) -> estimated Jaccard; only pairs with i < j; duplicates merged.
 */
template <typename IndexType, typename ValueType>
std::map<std::pair<IndexType, IndexType>, ValueType> lsh_candidate_pairs(
    const std::vector<std::vector<uint64_t>> &signatures,
    int num_bands)
{
    using namespace lsh_internal;
    using pair_t = std::pair<IndexType, IndexType>;
    std::map<pair_t, ValueType> out;
    if (signatures.empty() || num_bands <= 0) return out;

    const size_t num_rows = signatures.size();
    const int k = static_cast<int>(signatures[0].size());
    const int r = k / num_bands;
    if (r <= 0 || k != r * num_bands) return out;

    std::vector<pair_t> all_pairs;
// #ifdef _OPENMP
    #pragma omp parallel
    {
        std::vector<pair_t> my_pairs;
        #pragma omp for schedule(guided)
        for (int band = 0; band < num_bands; ++band) {
            std::unordered_map<uint64_t, std::vector<IndexType>> buckets;
            const int offset = band * r;

            for (size_t row = 0; row < num_rows; ++row) {
                const std::vector<uint64_t> &sig = signatures[row];
                if (sig.size() < static_cast<size_t>(offset + r)) continue;
                uint64_t bh = band_hash(sig.data() + offset, r);
                buckets[bh].push_back(static_cast<IndexType>(row));
            }

            for (const auto &kv : buckets) {
                const std::vector<IndexType> &rows = kv.second;
                if (rows.size() <= k_bucket_size_limit) {
                    for (size_t a = 0; a < rows.size(); ++a) {
                        for (size_t b = a + 1; b < rows.size(); ++b) {
                            IndexType i = rows[a], j = rows[b];
                            if (i > j) std::swap(i, j);
                            my_pairs.push_back(std::make_pair(i, j));
                        }
                    }
                } else {
                    for (size_t a = 0; a < rows.size(); ++a) {
                        size_t b_end = a + 1 + static_cast<size_t>(k_window_pairs);
                        if (b_end > rows.size()) b_end = rows.size();
                        for (size_t b = a + 1; b < b_end; ++b) {
                            IndexType i = rows[a], j = rows[b];
                            if (i > j) std::swap(i, j);
                            my_pairs.push_back(std::make_pair(i, j));
                        }
                    }
                }
            }
        }
        #pragma omp critical
        all_pairs.insert(all_pairs.end(), my_pairs.begin(), my_pairs.end());
    }
// #else
//     for (int band = 0; band < num_bands; ++band) {
//         std::unordered_map<uint64_t, std::vector<IndexType>> buckets;
//         const int offset = band * r;

//         for (size_t row = 0; row < num_rows; ++row) {
//             const std::vector<uint64_t> &sig = signatures[row];
//             if (sig.size() < static_cast<size_t>(offset + r)) continue;
//             uint64_t bh = band_hash(sig.data() + offset, r);
//             buckets[bh].push_back(static_cast<IndexType>(row));
//         }

//         for (const auto &kv : buckets) {
//             const std::vector<IndexType> &rows = kv.second;
//             if (rows.size() <= k_bucket_size_limit) {
//                 for (size_t a = 0; a < rows.size(); ++a) {
//                     for (size_t b = a + 1; b < rows.size(); ++b) {
//                         IndexType i = rows[a], j = rows[b];
//                         if (i > j) std::swap(i, j);
//                         all_pairs.push_back(std::make_pair(i, j));
//                     }
//                 }
//             } else {
//                 for (size_t a = 0; a < rows.size(); ++a) {
//                     size_t b_end = a + 1 + static_cast<size_t>(k_window_pairs);
//                     if (b_end > rows.size()) b_end = rows.size();
//                     for (size_t b = a + 1; b < b_end; ++b) {
//                         IndexType i = rows[a], j = rows[b];
//                         if (i > j) std::swap(i, j);
//                         all_pairs.push_back(std::make_pair(i, j));
//                     }
//                 }
//             }
//         }
//     }
// #endif

    // Deduplicate: same (i,j) can appear in multiple bands
    lsh_internal::parallel_sort_pairs(all_pairs.begin(), all_pairs.end());
    all_pairs.erase(std::unique(all_pairs.begin(), all_pairs.end()), all_pairs.end());

    const size_t np = all_pairs.size();
    std::vector<ValueType> est_vals(np);
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (size_t i = 0; i < np; ++i) {
        const pair_t &p = all_pairs[i];
        est_vals[i] = static_cast<ValueType>(minhash_estimated_jaccard(
            signatures[static_cast<size_t>(p.first)], signatures[static_cast<size_t>(p.second)]));
    }
    for (size_t i = 0; i < np; ++i)
        out[all_pairs[i]] = est_vals[i];

    return out;
}

/**
 * @brief Compute MinHash signatures and run LSH to generate candidate pairs.
 *
 * Convenience overload: builds signatures from CSR then calls lsh_candidate_pairs.
 * Each candidate pair is stored with value = MinHash-estimated Jaccard.
 *
 * @param rowptr      CSR row pointer.
 * @param col_index   Column indices.
 * @param num_rows    Number of rows.
 * @param k           MinHash signature length (must be divisible by num_bands).
 * @param num_bands   Number of LSH bands.
 * @param seed        Random seed for MinHash (default 12345).
 */
template <typename IndexType, typename ValueType>
std::map<std::pair<IndexType, IndexType>, ValueType> lsh_candidate_pairs(
    const IndexType *rowptr,
    const IndexType *col_index,
    IndexType num_rows,
    int k,
    int num_bands,
    uint64_t seed = 12345)
{
    std::vector<uint64_t> sigs = minhash_signatures_flat<IndexType>(
        rowptr, col_index, num_rows, k, seed);
    return lsh_internal::lsh_candidate_pairs_from_flat<IndexType, ValueType>(
        sigs.data(), static_cast<size_t>(num_rows), k, num_bands);
}

// ============================================================================
// Vector-based API (sequential storage, parallel sort, no map)
// ============================================================================

/**
 * @brief LSH candidate pairs from CSR: returns vector of CandidatePair (i, j, score).
 *        Uses parallel sort when available; no std::map for better cache/memory.
 */
template <typename IndexType, typename ValueType>
std::vector<CandidatePair<IndexType, ValueType>> lsh_candidate_pairs_vector(
    const IndexType *rowptr,
    const IndexType *col_index,
    IndexType num_rows,
    int k,
    int num_bands,
    uint64_t seed = 12345)
{
    std::vector<uint64_t> sigs = minhash_signatures_flat<IndexType>(
        rowptr, col_index, num_rows, k, seed);
    return lsh_internal::lsh_candidate_pairs_from_flat_vector<IndexType, ValueType>(
        sigs.data(), static_cast<size_t>(num_rows), k, num_bands);
}

/**
 * @brief Convert vector of CandidatePair to map (i,j) -> score for APIs that need it (e.g. hierarchical_clustering_v0).
 */
template <typename IndexType, typename ValueType>
std::map<std::pair<IndexType, IndexType>, ValueType> candidate_pairs_vector_to_map(
    const std::vector<CandidatePair<IndexType, ValueType>> &pairs)
{
    std::map<std::pair<IndexType, IndexType>, ValueType> out;
    for (const auto &p : pairs)
        out[std::make_pair(p.i, p.j)] = p.score;
    return out;
}

#endif /* SPGEMM_MINHASH_LSH_H */
