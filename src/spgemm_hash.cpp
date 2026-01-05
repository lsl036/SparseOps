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

template <typename IndexType, typename ValueType>
void spgemm_hash_symbolic_omp(
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_rows, IndexType c_cols,
    IndexType *&cpt, IndexType *&ccol, IndexType &c_nnz)
{
    int thread_num = Le_get_thread_num();
    cpt = new_array<IndexType>(c_rows + 1);
    cpt[0] = 0;
    
    IndexType hash_size = SpGEMM_BIN<IndexType, ValueType>::get_hash_size(c_cols, MIN_HT_S);
    
    // Thread-local hash tables
    IndexType **local_hash_table_id = new IndexType*[thread_num];
    #pragma omp parallel
    {
        int tid = Le_get_thread_id();
        local_hash_table_id[tid] = new_array<IndexType>(hash_size);
    }
    
    // First pass: count nonzeros per row
    #pragma omp parallel for
    for (IndexType i = 0; i < c_rows; i++) {
        int tid = Le_get_thread_id();
        IndexType *hash_table_id = local_hash_table_id[tid];
        std::fill_n(hash_table_id, hash_size, static_cast<IndexType>(-1));
        
        IndexType row_nnz = 0;
        
        for (IndexType j = arpt[i]; j < arpt[i + 1]; j++) {
            IndexType col_a = acol[j];
            if (col_a >= c_cols) continue;
            
            for (IndexType k = brpt[col_a]; k < brpt[col_a + 1]; k++) {
                IndexType col_b = bcol[k];
                if (col_b >= c_cols) continue;
                
                IndexType pos = hash_insert_symbolic(col_b, hash_size, hash_table_id);
                if (pos != -1 && hash_table_id[pos] == col_b) {
                    row_nnz++;
                }
            }
        }
        
        cpt[i + 1] = row_nnz;
    }
    
    // Prefix sum
    for (IndexType i = 1; i <= c_rows; i++) {
        cpt[i] += cpt[i - 1];
    }
    
    c_nnz = cpt[c_rows];
    ccol = new_array<IndexType>(c_nnz);
    
        // Second pass: fill column indices (recompute hash table and extract unique columns)
    #pragma omp parallel for
    for (IndexType i = 0; i < c_rows; i++) {
        int tid = Le_get_thread_id();
        IndexType *hash_table_id = local_hash_table_id[tid];
        std::fill_n(hash_table_id, hash_size, static_cast<IndexType>(-1));
        
        // Build hash table for this row
        for (IndexType j = arpt[i]; j < arpt[i + 1]; j++) {
            IndexType col_a = acol[j];
            if (col_a >= c_cols) continue;
            
            for (IndexType k = brpt[col_a]; k < brpt[col_a + 1]; k++) {
                IndexType col_b = bcol[k];
                if (col_b >= c_cols) continue;
                
                hash_insert_symbolic(col_b, hash_size, hash_table_id);
            }
        }
        
        // Extract unique columns from hash table
        IndexType ccol_start = cpt[i];
        IndexType ccol_idx = ccol_start;
        for (IndexType h = 0; h < hash_size && ccol_idx < cpt[i + 1]; h++) {
            if (hash_table_id[h] != -1) {
                ccol[ccol_idx++] = hash_table_id[h];
            }
        }
    }
    
    // Cleanup
    #pragma omp parallel
    {
        int tid = Le_get_thread_id();
        delete_array(local_hash_table_id[tid]);
    }
    delete[] local_hash_table_id;
}

template <typename IndexType, typename ValueType>
void spgemm_hash_symbolic_omp_lb(
    const IndexType *arpt, const IndexType *acol,
    const IndexType *brpt, const IndexType *bcol,
    IndexType c_rows, IndexType c_cols,
    IndexType *&cpt, IndexType *&ccol, IndexType &c_nnz,
    SpGEMM_BIN<IndexType, ValueType> *bin)
{
    // Similar to omp version but use BIN for load balancing
    int thread_num = Le_get_thread_num();
    cpt = new_array<IndexType>(c_rows + 1);
    cpt[0] = 0;
    
    IndexType hash_size = SpGEMM_BIN<IndexType, ValueType>::get_hash_size(c_cols, MIN_HT_S);
    
    // Use BIN's hash tables
    if (bin->local_hash_table_id == nullptr) {
        bin->create_local_hash_table(c_cols);
    }
    
    // First pass: count nonzeros per row
    #pragma omp parallel
    {
        int tid = Le_get_thread_id();
        IndexType *hash_table_id = bin->local_hash_table_id[tid];
        
        #pragma omp for
        for (IndexType i = 0; i < c_rows; i++) {
            std::fill_n(hash_table_id, hash_size, static_cast<IndexType>(-1));
            
            IndexType row_nnz = 0;
            
            for (IndexType j = arpt[i]; j < arpt[i + 1]; j++) {
                IndexType col_a = acol[j];
                if (col_a >= c_cols) continue;
                
                for (IndexType k = brpt[col_a]; k < brpt[col_a + 1]; k++) {
                    IndexType col_b = bcol[k];
                    if (col_b >= c_cols) continue;
                    
                    IndexType pos = hash_insert_symbolic(col_b, hash_size, hash_table_id);
                    if (pos != -1 && hash_table_id[pos] == col_b) {
                        row_nnz++;
                    }
                }
            }
            
            cpt[i + 1] = row_nnz;
        }
    }
    
    // Prefix sum
    for (IndexType i = 1; i <= c_rows; i++) {
        cpt[i] += cpt[i - 1];
    }
    
    c_nnz = cpt[c_rows];
    ccol = new_array<IndexType>(c_nnz);
    
    // Second pass: fill column indices (recompute hash table and extract unique columns)
    #pragma omp parallel
    {
        int tid = Le_get_thread_id();
        IndexType *hash_table_id = bin->local_hash_table_id[tid];
        
        #pragma omp for
        for (IndexType i = 0; i < c_rows; i++) {
            std::fill_n(hash_table_id, hash_size, static_cast<IndexType>(-1));
            
            // Build hash table for this row
            for (IndexType j = arpt[i]; j < arpt[i + 1]; j++) {
                IndexType col_a = acol[j];
                if (col_a >= c_cols) continue;
                
                for (IndexType k = brpt[col_a]; k < brpt[col_a + 1]; k++) {
                    IndexType col_b = bcol[k];
                    if (col_b >= c_cols) continue;
                    
                    hash_insert_symbolic(col_b, hash_size, hash_table_id);
                }
            }
            
            // Extract unique columns from hash table
            IndexType ccol_start = cpt[i];
            IndexType ccol_idx = ccol_start;
            for (IndexType h = 0; h < hash_size && ccol_idx < cpt[i + 1]; h++) {
                if (hash_table_id[h] != -1) {
                    ccol[ccol_idx++] = hash_table_id[h];
                }
            }
        }
    }
}

// ============================================================================
// Numeric Phase Implementations
// ============================================================================

template <typename IndexType, typename ValueType>
void spgemm_hash_numeric_omp(
    const IndexType *arpt, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_rows, IndexType c_cols,
    const IndexType *cpt, const IndexType *ccol, ValueType *cval)
{
    int thread_num = Le_get_thread_num();
    IndexType hash_size = SpGEMM_BIN<IndexType, ValueType>::get_hash_size(c_cols, MIN_HT_N);
    
    // Thread-local hash tables
    IndexType **local_hash_table_id = new IndexType*[thread_num];
    ValueType **local_hash_table_val = new ValueType*[thread_num];
    
    #pragma omp parallel
    {
        int tid = Le_get_thread_id();
        local_hash_table_id[tid] = new_array<IndexType>(hash_size);
        local_hash_table_val[tid] = new_array<ValueType>(hash_size);
    }
    
    // Initialize output values
    std::fill_n(cval, cpt[c_rows], static_cast<ValueType>(0));
    
    #pragma omp parallel for
    for (IndexType i = 0; i < c_rows; i++) {
        int tid = Le_get_thread_id();
        IndexType *hash_table_id = local_hash_table_id[tid];
        ValueType *hash_table_val = local_hash_table_val[tid];
        
        std::fill_n(hash_table_id, hash_size, static_cast<IndexType>(-1));
        std::fill_n(hash_table_val, hash_size, static_cast<ValueType>(0));
        
        for (IndexType j = arpt[i]; j < arpt[i + 1]; j++) {
            IndexType col_a = acol[j];
            ValueType val_a = aval[j];
            if (col_a >= c_cols) continue;
            
            for (IndexType k = brpt[col_a]; k < brpt[col_a + 1]; k++) {
                IndexType col_b = bcol[k];
                ValueType val_b = bval[k];
                if (col_b >= c_cols) continue;
                
                ValueType prod = val_a * val_b;
                hash_insert_numeric(col_b, prod, hash_size, hash_table_id, hash_table_val);
            }
        }
        
        // Copy from hash table to output
        for (IndexType j = cpt[i]; j < cpt[i + 1]; j++) {
            IndexType col = ccol[j];
            IndexType pos = hash_find_pos(col, hash_size, hash_table_id);
            if (pos != -1 && hash_table_id[pos] == col) {
                cval[j] = hash_table_val[pos];
            }
        }
    }
    
    // Cleanup
    #pragma omp parallel
    {
        int tid = Le_get_thread_id();
        delete_array(local_hash_table_id[tid]);
        delete_array(local_hash_table_val[tid]);
    }
    delete[] local_hash_table_id;
    delete[] local_hash_table_val;
}

template <typename IndexType, typename ValueType>
void spgemm_hash_numeric_omp_lb(
    const IndexType *arpt, const IndexType *acol, const ValueType *aval,
    const IndexType *brpt, const IndexType *bcol, const ValueType *bval,
    IndexType c_rows, IndexType c_cols,
    const IndexType *cpt, const IndexType *ccol, ValueType *cval,
    SpGEMM_BIN<IndexType, ValueType> *bin)
{
    IndexType hash_size = SpGEMM_BIN<IndexType, ValueType>::get_hash_size(c_cols, MIN_HT_N);
    
    // Use BIN's hash tables
    if (bin->local_hash_table_id == nullptr) {
        bin->create_local_hash_table(c_cols);
    }
    
    // Initialize output values
    std::fill_n(cval, cpt[c_rows], static_cast<ValueType>(0));
    
    #pragma omp parallel
    {
        int tid = Le_get_thread_id();
        IndexType *hash_table_id = bin->local_hash_table_id[tid];
        ValueType *hash_table_val = bin->local_hash_table_val[tid];
        
        #pragma omp for
        for (IndexType i = 0; i < c_rows; i++) {
            std::fill_n(hash_table_id, hash_size, static_cast<IndexType>(-1));
            std::fill_n(hash_table_val, hash_size, static_cast<ValueType>(0));
            
            for (IndexType j = arpt[i]; j < arpt[i + 1]; j++) {
                IndexType col_a = acol[j];
                ValueType val_a = aval[j];
                if (col_a >= c_cols) continue;
                
                for (IndexType k = brpt[col_a]; k < brpt[col_a + 1]; k++) {
                    IndexType col_b = bcol[k];
                    ValueType val_b = bval[k];
                    if (col_b >= c_cols) continue;
                    
                    ValueType prod = val_a * val_b;
                    hash_insert_numeric(col_b, prod, hash_size, hash_table_id, hash_table_val);
                }
            }
            
            // Copy from hash table to output
            for (IndexType j = cpt[i]; j < cpt[i + 1]; j++) {
                IndexType col = ccol[j];
                IndexType pos = hash_find_pos(col, hash_size, hash_table_id);
                if (pos != -1 && hash_table_id[pos] == col) {
                    cval[j] = hash_table_val[pos];
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
template void spgemm_hash_symbolic_omp<int, float>(
    const int*, const int*, const int*, const int*, int, int, int*&, int*&, int&);
template void spgemm_hash_symbolic_omp<int, double>(
    const int*, const int*, const int*, const int*, int, int, int*&, int*&, int&);
template void spgemm_hash_symbolic_omp<long long, float>(
    const long long*, const long long*, const long long*, const long long*, long long, long long, long long*&, long long*&, long long&);
template void spgemm_hash_symbolic_omp<long long, double>(
    const long long*, const long long*, const long long*, const long long*, long long, long long, long long*&, long long*&, long long&);

template void spgemm_hash_symbolic_omp_lb<int, float>(
    const int*, const int*, const int*, const int*, int, int, int*&, int*&, int&, SpGEMM_BIN<int, float>*);
template void spgemm_hash_symbolic_omp_lb<int, double>(
    const int*, const int*, const int*, const int*, int, int, int*&, int*&, int&, SpGEMM_BIN<int, double>*);
template void spgemm_hash_symbolic_omp_lb<long long, float>(
    const long long*, const long long*, const long long*, const long long*, long long, long long, long long*&, long long*&, long long&, SpGEMM_BIN<long long, float>*);
template void spgemm_hash_symbolic_omp_lb<long long, double>(
    const long long*, const long long*, const long long*, const long long*, long long, long long, long long*&, long long*&, long long&, SpGEMM_BIN<long long, double>*);

template void spgemm_hash_numeric_omp<int, float>(
    const int*, const int*, const float*, const int*, const int*, const float*, int, int, const int*, const int*, float*);
template void spgemm_hash_numeric_omp<int, double>(
    const int*, const int*, const double*, const int*, const int*, const double*, int, int, const int*, const int*, double*);
template void spgemm_hash_numeric_omp<long long, float>(
    const long long*, const long long*, const float*, const long long*, const long long*, const float*, long long, long long, const long long*, const long long*, float*);
template void spgemm_hash_numeric_omp<long long, double>(
    const long long*, const long long*, const double*, const long long*, const long long*, const double*, long long, long long, const long long*, const long long*, double*);

template void spgemm_hash_numeric_omp_lb<int, float>(
    const int*, const int*, const float*, const int*, const int*, const float*, int, int, const int*, const int*, float*, SpGEMM_BIN<int, float>*);
template void spgemm_hash_numeric_omp_lb<int, double>(
    const int*, const int*, const double*, const int*, const int*, const double*, int, int, const int*, const int*, double*, SpGEMM_BIN<int, double>*);
template void spgemm_hash_numeric_omp_lb<long long, float>(
    const long long*, const long long*, const float*, const long long*, const long long*, const float*, long long, long long, const long long*, const long long*, float*, SpGEMM_BIN<long long, float>*);
template void spgemm_hash_numeric_omp_lb<long long, double>(
    const long long*, const long long*, const double*, const long long*, const long long*, const double*, long long, long long, const long long*, const long long*, double*, SpGEMM_BIN<long long, double>*);

template void sort_csr_columns<int, float>(int, const int*, int*, float*);
template void sort_csr_columns<int, double>(int, const int*, int*, double*);
template void sort_csr_columns<long long, float>(long long, const long long*, long long*, float*);
template void sort_csr_columns<long long, double>(long long, const long long*, long long*, double*);

