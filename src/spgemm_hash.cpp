/**
 * @file spgemm_hash.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Row-wise SpGEMM implementation using hash table method
 * @version 0.1
 * @date 2024
 */

#include "../include/spgemm_hash.h"
#include <cstring>
#include <vector>
#include <algorithm>

// ============================================================================
// Symbolic Phase Implementations
// ============================================================================

// Helper function: prefix sum (scan) - same as in spgemm_bin.cpp
template <typename IndexType>
inline void scan_spgemm(const IndexType *input, IndexType *output, IndexType n) {
    output[0] = 0;
    for (IndexType i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

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
    scan_spgemm(bin->row_nz, cpt, c_rows + 1);
    c_nnz = cpt[c_rows];
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Sort function for pairs (matching reference implementation)
 */
template <typename IndexType, typename ValueType>
bool sort_less(const std::pair<IndexType, ValueType> &left, const std::pair<IndexType, ValueType> &right)
{
    return left.first < right.first;
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

template void sort_csr_columns<int64_t, float>(int64_t, const int64_t*, int64_t*, float*);
template void sort_csr_columns<int64_t, double>(int64_t, const int64_t*, int64_t*, double*);

