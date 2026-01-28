/**
 * @file spgemm_hash.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Row-wise SpGEMM implementation using hash table method
 * @version 0.1
 * @date 2024
 */

#include "../include/spgemm_hash.h"
#include "../include/spgemm_utility.h"
#include <cstring>
#include <vector>
#include <algorithm>

// ============================================================================
// Symbolic Phase Implementations
// ============================================================================

template <typename IndexType, typename ValueType>
void spgemm_hash_symbolic_omp_lb(
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_rows, IndexType c_cols,
    IndexType *cpt, IndexType &c_nnz,
    SpGEMM_BIN<IndexType, ValueType> *bin)
{
    // Use BIN's hash tables
    if (bin->local_hash_table_id == nullptr) {
        bin->create_local_hash_table(c_cols);
    }
    
    // Symbolic phase: count nonzeros per row (matching reference hash_symbolic_kernel)
    int thread_num = Le_get_thread_num();
    #pragma omp parallel num_threads(thread_num)
    {
        int tid = Le_get_thread_id();
        IndexType start_row = bin->rows_offset[tid];
        IndexType end_row = bin->rows_offset[tid + 1];
        
        IndexType *check = bin->local_hash_table_id[tid];
        
        for (IndexType i = start_row; i < end_row; ++i) {
            IndexType nz = 0;
            IndexType bid = bin->bin_id[i];
            
            if (bid > 0) {
                // Calculate hash table size for this row based on bin_id (matching reference)
                IndexType ht_size = MIN_HT_S << (bid - 1);
                
                // Clear hash table efficiently using memset (faster than loop)
                std::memset(check, -1, ht_size * sizeof(IndexType));
                
                for (IndexType j = arpt[i]; j < arpt[i + 1]; ++j) {
                    IndexType t_acol = acol[j];
                    for (IndexType k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        IndexType key = bcol[k];
                        IndexType hash = (key * HASH_SCAL) & (ht_size - 1);
                        while (1) {  // Loop for hash probing
                            if (check[hash] == key) {  // if the key is already inserted, it's ok
                                break;
                            }
                            else if (check[hash] == -1) {  // if the key has not been inserted yet, then it's added.
                                check[hash] = key;
                                nz++;
                                break;
                            }
                            else {  // linear probing: check next entry
                                hash = (hash + 1) & (ht_size - 1);  // hash = (hash + 1) % ht_size
                            }
                        }
                    }
                }
            }
            bin->row_nz[i] = nz;
        }
    }
    
    // Set row pointer of matrix C using scan (matching reference hash_symbolic)
    scan(bin->row_nz, cpt, c_rows + 1, bin->allocated_thread_num);
    c_nnz = cpt[c_rows];
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Sort function for pairs (ascending order, matching reference implementation)
 */
template <typename IndexType, typename ValueType>
bool sort_less(const std::pair<IndexType, ValueType> &left, const std::pair<IndexType, ValueType> &right)
{
    return left.first < right.first;
}

/**
 * @brief Sort function for pairs (descending order, matching reference sort_large)
 *        When similarity values are equal, sort by column index in ascending order for stability
 */
template <typename IndexType, typename ValueType>
bool sort_large(const std::pair<ValueType, IndexType> &left, const std::pair<ValueType, IndexType> &right)
{
    // Primary sort: by similarity value (descending)
    if (left.first != right.first) {
        return left.first > right.first;
    }
    // Secondary sort: by column index (ascending) for stability when similarities are equal
    return left.second < right.second;
}

/**
 * @brief After calculating on each hash table, sort them in ascending order if necessary, 
 *        and then store them as output matrix (matching reference sort_and_store_table2mat)
 */
template <bool sortOutput, typename IndexType, typename ValueType>
inline void sort_and_store_table2mat(IndexType *ht_check, ValueType *ht_value, 
                                     IndexType *colids, ValueType *values, 
                                     IndexType nz, IndexType ht_size, IndexType offset)
{
    IndexType index = 0;
    
    // Sort elements in ascending order if necessary, and store them as output matrix
    if (sortOutput) {
        std::vector<std::pair<IndexType, ValueType>> p_vec(nz);
        for (IndexType j = 0; j < ht_size; ++j) { // accumulate non-zero entry from hash table
            if (ht_check[j] != -1) {
                p_vec[index++] = std::make_pair(ht_check[j], ht_value[j]);
            }
        }
        std::sort(p_vec.begin(), p_vec.end(), sort_less<IndexType, ValueType>); // sort only non-zero elements
        for (IndexType j = 0; j < index; ++j) { // store the results
            colids[j] = p_vec[j].first;
            values[j] = p_vec[j].second;
        }
    }
    else {
        for (IndexType j = 0; j < ht_size; ++j) {
            if (ht_check[j] != -1) {
                colids[index] = ht_check[j];
                values[index] = ht_value[j];
                index++;
            }
        }
    }
}

/**
 * @brief Sort and store top-k Jaccard similarities from hash table
 *        Matching reference sort_and_store_table2mat_topK_jaccard
 *        Converts intersection sizes to Jaccard similarity: jacc = intersection / union
 *        Jaccard similarity = |A[i] ∩ A[j]| / |A[i] ∪ A[j]|
 *        Excludes diagonal elements (self-similarity) as they are always 1.0 and not needed for clustering
 * 
 * @tparam sortOutput Whether to sort output (always true for top-k)
 * @tparam IndexType 
 * @tparam ValueType 
 * @param ht_check Hash table check array (column indices)
 * @param ht_value Hash table value array (intersection sizes)
 * @param colids Output column indices array
 * @param values Output values array (will contain Jaccard similarities)
 * @param nz Number of non-zeros in hash table (before filtering)
 * @param ht_size Hash table size
 * @param top_k Number of top similarities to keep
 * @param arpt Row pointer array of matrix A (for computing union size)
 * @param a_row_id Current row index in matrix A
 * @return IndexType Number of elements actually stored (after top-k selection, excluding diagonal)
 */
template <bool sortOutput, typename IndexType, typename ValueType>
inline IndexType sort_and_store_table2mat_topk_jaccard(
    IndexType *ht_check, ValueType *ht_value,
    IndexType *colids, ValueType *values,
    IndexType nz, IndexType ht_size, IndexType top_k,
    const IndexType *arpt, IndexType a_row_id)
{
    IndexType index = 0;
    
    if (sortOutput) {
        // Collect all non-zero entries and convert to Jaccard similarity
        std::vector<std::pair<ValueType, IndexType>> p_vec;  // <jaccard_similarity, col_index>
        p_vec.reserve(nz);
        
        for (IndexType j = 0; j < ht_size; ++j) {
            if (ht_check[j] != -1) {
                // Exclude diagonal elements (self-similarity is always 1.0 and not needed for clustering)
                if (ht_check[j] == a_row_id) {
                    continue;
                }
                
                // Compute union size: |A[i]| + |A[j]| - |A[i] ∩ A[j]|
                IndexType u = (arpt[a_row_id + 1] - arpt[a_row_id]) + 
                              (arpt[ht_check[j] + 1] - arpt[ht_check[j]]) - 
                              static_cast<IndexType>(ht_value[j]);
                
                // Compute Jaccard similarity: intersection / union
                ValueType jacc = (u == 0) ? static_cast<ValueType>(0.0) : 
                                 static_cast<ValueType>(1.0) * ht_value[j] / static_cast<ValueType>(u);
                
                p_vec.push_back(std::make_pair(jacc, ht_check[j]));
            }
        }
        
        // Sort by Jaccard similarity in descending order (largest first, matching reference sort_large)
        std::sort(p_vec.begin(), p_vec.end(), sort_large<IndexType, ValueType>);
        
        // Keep top-k (or all if fewer than k)
        index = std::min(static_cast<IndexType>(p_vec.size()), top_k);
        
        // Store the results
        for (IndexType j = 0; j < index; ++j) {
            colids[j] = p_vec[j].second;
            values[j] = p_vec[j].first;
        }
    }
    else {
        // Non-sorted version (should not be used for top-k, but included for completeness)
        for (IndexType j = 0; j < ht_size; ++j) {
            if (ht_check[j] != -1) {
                // Exclude diagonal elements (self-similarity is always 1.0 and not needed for clustering)
                if (ht_check[j] == a_row_id) {
                    continue;
                }
                
                // Compute union size
                IndexType u = (arpt[a_row_id + 1] - arpt[a_row_id]) + 
                              (arpt[ht_check[j] + 1] - arpt[ht_check[j]]) - 
                              static_cast<IndexType>(ht_value[j]);
                
                // Compute Jaccard similarity
                ValueType jacc = (u == 0) ? static_cast<ValueType>(0.0) : 
                                 static_cast<ValueType>(1.0) * ht_value[j] / static_cast<ValueType>(u);
                
                colids[index] = ht_check[j];
                values[index] = jacc;
                index++;
            }
        }
    }
    
    return index;
}

// ============================================================================
// Numeric Phase Implementations
// ============================================================================

template <bool sortOutput, typename IndexType, typename ValueType>
void spgemm_hash_numeric_omp_lb(
    const IndexType *arpt, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_rows, IndexType c_cols,
    const IndexType *cpt, IndexType *ccol, ValueType *cval,
    SpGEMM_BIN<IndexType, ValueType> *bin)
{
    // Use BIN's hash tables
    // if (bin->local_hash_table_id == nullptr) {
    //     bin->create_local_hash_table(c_cols);
    // }
    
    // Numeric phase (matching reference implementation)
    int thread_num = Le_get_thread_num();
    #pragma omp parallel num_threads(thread_num)
    {
        int tid = Le_get_thread_id();
        IndexType start_row = bin->rows_offset[tid];
        IndexType end_row = bin->rows_offset[tid + 1];
        
        IndexType *ht_check = bin->local_hash_table_id[tid];
        ValueType *ht_value = bin->local_hash_table_val[tid];
        
        for (IndexType i = start_row; i < end_row; ++i) {
            IndexType bid = bin->bin_id[i];
            if (bid > 0) {
                // Calculate hash table size for this row based on bin_id (matching reference)
                IndexType ht_size = MIN_HT_N << (bid - 1);
                IndexType offset = cpt[i];
                
                // Use memset for faster initialization
                std::memset(ht_check, -1, ht_size * sizeof(IndexType));
                // std::memset(ht_value, 0, ht_size * sizeof(ValueType));
                
                // union of col-ids of cluster
                for (IndexType j = arpt[i]; j < arpt[i + 1]; ++j) {  // A.cols
                    IndexType t_acol = acol[j];
                    ValueType t_aval = aval[j];
                    // B[i]
                    for (IndexType k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        ValueType t_val = t_aval * bval[k];
                        IndexType key = bcol[k];
                        IndexType hash = (key * HASH_SCAL) & (ht_size - 1);
                        while (1) {  // Loop for hash probing
                            if (ht_check[hash] == key) {  // key is already inserted
                                ht_value[hash] += t_val;
                                break;
                            }
                            else if (ht_check[hash] == -1) {  // insert new entry
                                ht_check[hash] = key;
                                ht_value[hash] = t_val;
                                break;
                            }
                            else {
                                hash = (hash + 1) & (ht_size - 1);  // (hash + 1) % ht_size
                            }
                        }
                    }
                }
                
                // Extract from hash table and store to output (matching reference sort_and_store_table2mat)
                IndexType nz = cpt[i + 1] - offset;
                sort_and_store_table2mat<sortOutput, IndexType, ValueType>(
                    ht_check, ht_value,
                    ccol + offset, cval + offset,
                    nz, ht_size, offset);
            }
        }
    }
}

/**
 * @brief Numeric phase with Top-K selection: compute values and keep only top-k per row
 *        Matching reference hash_numeric_topk implementation
 *        Converts intersection sizes to Jaccard similarities
 *        The values stored in C are Jaccard similarity scores (0.0 to 1.0)
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param arpt Row pointer array of matrix A
 * @param acol Column index array of matrix A
 * @param aval Values array of matrix A (should be binary pattern, all 1.0)
 * @param brpt Row pointer array of matrix B (AT)
 * @param bcol Column index array of matrix B (AT)
 * @param bval Values array of matrix B (AT, should be binary pattern, all 1.0)
 * @param c_rows Number of rows in output matrix C
 * @param c_cols Number of columns in output matrix C
 * @param cpt Row pointer array of output matrix C (from symbolic phase)
 * @param ccol Output column indices array (pre-allocated, will be filled)
 * @param cval Output values array (pre-allocated, will be filled with Jaccard similarities)
 * @param top_k Number of top similarities to keep per row
 * @param row_nnz Output: number of non-zeros per row after top-k selection (will be filled)
 * @param bin BIN object for load balancing
 */
template <typename IndexType, typename ValueType>
void spgemm_hash_numeric_topk_omp_lb(
    const IndexType *arpt, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_rows, IndexType c_cols,
    const IndexType *cpt, IndexType *ccol, ValueType *cval,
    IndexType top_k, IndexType *row_nnz,
    SpGEMM_BIN<IndexType, ValueType> *bin)
{
    // Numeric phase with Top-K selection (matching reference hash_numeric_topk)
    int thread_num = Le_get_thread_num();
    #pragma omp parallel num_threads(thread_num)
    {
        int tid = Le_get_thread_id();
        IndexType start_row = bin->rows_offset[tid];
        IndexType end_row = bin->rows_offset[tid + 1];
        
        IndexType *ht_check = bin->local_hash_table_id[tid];
        ValueType *ht_value = bin->local_hash_table_val[tid];
        
        for (IndexType i = start_row; i < end_row; ++i) {
            IndexType bid = bin->bin_id[i];
            if (bid > 0) {
                // Calculate hash table size for this row based on bin_id (matching reference)
                IndexType ht_size = MIN_HT_N << (bid - 1);
                IndexType offset = cpt[i];
                IndexType nz = cpt[i + 1] - offset;
                
                // Clear hash table
                std::memset(ht_check, -1, ht_size * sizeof(IndexType));
                
                // Accumulate products in hash table
                // For binary pattern A * AT, this computes intersection sizes
                for (IndexType j = arpt[i]; j < arpt[i + 1]; ++j) {
                    IndexType t_acol = acol[j];
                    ValueType t_aval = aval[j];
                    
                    for (IndexType k = brpt[t_acol]; k < brpt[t_acol + 1]; ++k) {
                        ValueType t_val = t_aval * bval[k];
                        IndexType key = bcol[k];
                        IndexType hash = (key * HASH_SCAL) & (ht_size - 1);
                        
                        while (1) {
                            if (ht_check[hash] == key) {
                                ht_value[hash] += t_val;
                                break;
                            }
                            else if (ht_check[hash] == -1) {
                                ht_check[hash] = key;
                                ht_value[hash] = t_val;
                                break;
                            }
                            else {
                                hash = (hash + 1) & (ht_size - 1);
                            }
                        }
                    }
                }
                
                // Extract from hash table, convert to Jaccard similarity, and store top-k
                // Matching reference sort_and_store_table2mat_topK_jaccard
                row_nnz[i] = sort_and_store_table2mat_topk_jaccard<true, IndexType, ValueType>(
                    ht_check, ht_value,
                    ccol + offset, cval + offset,
                    nz, ht_size, top_k,
                    arpt, i);
            } else {
                row_nnz[i] = 0;
            }
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

template <typename IndexType, typename ValueType>
void sort_csr_columns(IndexType num_rows,
                      const IndexType *row_offset,
                      IndexType *col_index,
                      ValueType *values)
{
    #pragma omp parallel for
    for (IndexType i = 0; i < num_rows; i++) {
        IndexType start = row_offset[i];
        IndexType end = row_offset[i + 1];
        
        // Create pairs for sorting
        std::vector<std::pair<IndexType, ValueType>> pairs;
        for (IndexType j = start; j < end; j++) {
            pairs.push_back(std::make_pair(col_index[j], values[j]));
        }
        
        // Sort by column index
        std::sort(pairs.begin(), pairs.end(),
                  [](const std::pair<IndexType, ValueType> &a,
                     const std::pair<IndexType, ValueType> &b) {
                      return a.first < b.first;
                  });
        
        // Write back
        for (size_t j = 0; j < pairs.size(); j++) {
            col_index[start + j] = pairs[j].first;
            values[start + j] = pairs[j].second;
        }
    }
}

// Explicit template instantiations (only int64_t for IndexType)
#include <cstdint>

template void spgemm_hash_symbolic_omp_lb<int64_t, float>(
    const int64_t*, const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t, int64_t*, int64_t&, SpGEMM_BIN<int64_t, float>*);
template void spgemm_hash_symbolic_omp_lb<int64_t, double>(
    const int64_t*, const int64_t*, const int64_t*, const int64_t*, int64_t, int64_t, int64_t*, int64_t&, SpGEMM_BIN<int64_t, double>*);

template void spgemm_hash_numeric_omp_lb<false, int64_t, float>(
    const int64_t*, const int64_t*, const float*, const int64_t*, const int64_t*, const float*, int64_t, int64_t, const int64_t*, int64_t*, float*, SpGEMM_BIN<int64_t, float>*);
template void spgemm_hash_numeric_omp_lb<false, int64_t, double>(
    const int64_t*, const int64_t*, const double*, const int64_t*, const int64_t*, const double*, int64_t, int64_t, const int64_t*, int64_t*, double*, SpGEMM_BIN<int64_t, double>*);
template void spgemm_hash_numeric_omp_lb<true, int64_t, float>(
    const int64_t*, const int64_t*, const float*, const int64_t*, const int64_t*, const float*, int64_t, int64_t, const int64_t*, int64_t*, float*, SpGEMM_BIN<int64_t, float>*);
template void spgemm_hash_numeric_omp_lb<true, int64_t, double>(
    const int64_t*, const int64_t*, const double*, const int64_t*, const int64_t*, const double*, int64_t, int64_t, const int64_t*, int64_t*, double*, SpGEMM_BIN<int64_t, double>*);

template void spgemm_hash_numeric_topk_omp_lb<int64_t, float>(
    const int64_t*, const int64_t*, const float*, const int64_t*, const int64_t*, const float*, 
    int64_t, int64_t, const int64_t*, int64_t*, float*, int64_t, int64_t*, SpGEMM_BIN<int64_t, float>*);
template void spgemm_hash_numeric_topk_omp_lb<int64_t, double>(
    const int64_t*, const int64_t*, const double*, const int64_t*, const int64_t*, const double*, 
    int64_t, int64_t, const int64_t*, int64_t*, double*, int64_t, int64_t*, SpGEMM_BIN<int64_t, double>*);

template void sort_csr_columns<int64_t, float>(int64_t, const int64_t*, int64_t*, float*);
template void sort_csr_columns<int64_t, double>(int64_t, const int64_t*, int64_t*, double*);

