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
void scan_spgemm(const IndexType *input, IndexType *output, IndexType n) {
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
    #pragma omp parallel
    {
        int tid = Le_get_thread_id();
        IndexType start_row = bin->rows_offset[tid];
        IndexType end_row = bin->rows_offset[tid + 1];
        
        IndexType *check = bin->local_hash_table_id[tid];
        
        // Get hash table size for this thread (shared across all rows)
        IndexType ht_size = bin->hash_table_size[tid];
        
        for (IndexType i = start_row; i < end_row; ++i) {
            IndexType nz = 0;
            IndexType bid = bin->bin_id[i];
            
            if (bid > 0) {
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
// Numeric Phase Implementations
// ============================================================================

template <typename IndexType, typename ValueType>
void spgemm_hash_numeric_omp_lb(
    const IndexType *arpt, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_rows, IndexType c_cols,
    const IndexType *cpt, IndexType *ccol, ValueType *cval,
    SpGEMM_BIN<IndexType, ValueType> *bin)
{
    // Use BIN's hash tables
    if (bin->local_hash_table_id == nullptr) {
        bin->create_local_hash_table(c_cols);
    }
    
    // Numeric phase (matching reference implementation)
    #pragma omp parallel
    {
        int tid = Le_get_thread_id();
        IndexType start_row = bin->rows_offset[tid];
        IndexType end_row = bin->rows_offset[tid + 1];
        
        IndexType *ht_check = bin->local_hash_table_id[tid];
        ValueType *ht_value = bin->local_hash_table_val[tid];
        IndexType ht_size = bin->hash_table_size[tid];
        
        for (IndexType i = start_row; i < end_row; ++i) {
            IndexType bid = bin->bin_id[i];
            if (bid > 0) {
                IndexType offset = cpt[i];
                // Use memset for faster initialization
                std::memset(ht_check, -1, ht_size * sizeof(IndexType));
                std::memset(ht_value, 0, ht_size * sizeof(ValueType));
                
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
                IndexType index = 0;
                
                // Extract non-zero entries from hash table
                for (IndexType j = 0; j < ht_size; ++j) {
                    if (ht_check[j] != -1) {
                        ccol[offset + index] = ht_check[j];
                        cval[offset + index] = ht_value[j];
                        index++;
                    }
                }
                
                // Sort columns if needed (for consistent output)
                // Note: Reference code uses sortOutput template parameter, we sort by default
                // Use in-place sort for better performance (avoid creating pairs)
                if (nz > 1) {
                    // Create index array for sorting (more efficient than pairs)
                    std::vector<IndexType> indices(nz);
                    for (IndexType j = 0; j < nz; ++j) {
                        indices[j] = j;
                    }
                    // Sort indices based on column values
                    std::sort(indices.begin(), indices.end(),
                              [&ccol, offset](IndexType a, IndexType b) {
                                  return ccol[offset + a] < ccol[offset + b];
                              });
                    // Reorder columns and values using indices
                    std::vector<IndexType> temp_col(nz);
                    std::vector<ValueType> temp_val(nz);
                    for (IndexType j = 0; j < nz; ++j) {
                        temp_col[j] = ccol[offset + indices[j]];
                        temp_val[j] = cval[offset + indices[j]];
                    }
                    for (IndexType j = 0; j < nz; ++j) {
                        ccol[offset + j] = temp_col[j];
                        cval[offset + j] = temp_val[j];
                    }
                }
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

// Explicit template instantiations
template void spgemm_hash_symbolic_omp_lb<int, float>(
    const int*, const int*, const int*, const int*, int, int, int*, int&, SpGEMM_BIN<int, float>*);
template void spgemm_hash_symbolic_omp_lb<int, double>(
    const int*, const int*, const int*, const int*, int, int, int*, int&, SpGEMM_BIN<int, double>*);
template void spgemm_hash_symbolic_omp_lb<long long, float>(
    const long long*, const long long*, const long long*, const long long*, long long, long long, long long*, long long&, SpGEMM_BIN<long long, float>*);
template void spgemm_hash_symbolic_omp_lb<long long, double>(
    const long long*, const long long*, const long long*, const long long*, long long, long long, long long*, long long&, SpGEMM_BIN<long long, double>*);

template void spgemm_hash_numeric_omp_lb<int, float>(
    const int*, const int*, const float*, const int*, const int*, const float*, int, int, const int*, int*, float*, SpGEMM_BIN<int, float>*);
template void spgemm_hash_numeric_omp_lb<int, double>(
    const int*, const int*, const double*, const int*, const int*, const double*, int, int, const int*, int*, double*, SpGEMM_BIN<int, double>*);
template void spgemm_hash_numeric_omp_lb<long long, float>(
    const long long*, const long long*, const float*, const long long*, const long long*, const float*, long long, long long, const long long*, long long*, float*, SpGEMM_BIN<long long, float>*);
template void spgemm_hash_numeric_omp_lb<long long, double>(
    const long long*, const long long*, const double*, const long long*, const long long*, const double*, long long, long long, const long long*, long long*, double*, SpGEMM_BIN<long long, double>*);

template void sort_csr_columns<int, float>(int, const int*, int*, float*);
template void sort_csr_columns<int, double>(int, const int*, int*, double*);
template void sort_csr_columns<long long, float>(long long, const long long*, long long*, float*);
template void sort_csr_columns<long long, double>(long long, const long long*, long long*, double*);

