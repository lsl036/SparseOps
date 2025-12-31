/**
 * @file spmv_sell_c_sigma.cpp
 * @author your name (you@domain.com)
 * @brief Simple implementation of SpMV in Sliced SELL-c-sigma format.
 *         Using OpenMP automatic parallism and vectorization.
 * @version 0.1
 * @date 2024-01-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include"../include/LeSpMV.h"

#include"../include/thread.h"

template <typename IndexType, typename ValueType>
void __spmv_sell_cs_serial_simple( const IndexType * Reorder,
                                   const IndexType num_rows,
                                   const IndexType chunk_rowNum,
                                   const IndexType total_chunk_num,
                                   const ValueType alpha,
                                   const IndexType *max_row_width,
                                   const IndexType * const *col_index,
                                   const ValueType * const *values,
                                   const ValueType * x, 
                                   const ValueType beta, 
                                   ValueType * y)
{
    for ( size_t chunkID = 0; chunkID < total_chunk_num; ++chunkID)
    {
        size_t chunk_width = max_row_width[chunkID];
        size_t chunk_start_row = chunkID * chunk_rowNum;

        for (size_t row = 0; row < chunk_rowNum; row++)
        {
            size_t global_row = chunk_start_row + row;
            if ( global_row >= num_rows) break; // 越界检查
            
            size_t sumPos = Reorder[global_row];
            ValueType sum    = 0;

            for (size_t i = 0; i < chunk_width; i++)
            {
                size_t col_index_pos = row * chunk_width + i;
                size_t col = col_index[chunkID][col_index_pos];

                if (col >= 0) { // 检查是否为填充的空位
                    sum += values[chunkID][col_index_pos] * x[col];
                }
            }

            y[sumPos] = alpha * sum + beta * y[sumPos];
        }   
    }
}

template <typename IndexType, typename ValueType>
void __spmv_sell_cs_omp_simple( const IndexType * Reorder,
                                const IndexType num_rows,
                                const IndexType chunk_rowNum,
                                const IndexType total_chunk_num,
                                const ValueType alpha,
                                const IndexType *max_row_width,
                                const IndexType * const *col_index,
                                const ValueType * const *values,
                                const ValueType * x, 
                                const ValueType beta, 
                                ValueType * y)
{
    const IndexType thread_num = Le_get_thread_num();

    #pragma omp parallel for num_threads(thread_num)
    for ( size_t chunkID = 0; chunkID < total_chunk_num; ++chunkID)
    {
        size_t chunk_width = max_row_width[chunkID];
        size_t chunk_start_row = chunkID * chunk_rowNum;

        for (size_t row = 0; row < chunk_rowNum; row++)
        {
            size_t global_row = chunk_start_row + row;
            if ( global_row >= num_rows) break; // 越界检查
            
            size_t sumPos = Reorder[global_row];
            ValueType sum    = 0;

            #pragma omp simd reduction(+:sum)
            for (size_t i = 0; i < chunk_width; i++)
            {
                size_t col_index_pos = row * chunk_width + i;
                size_t col = col_index[chunkID][col_index_pos];

                if (col >= 0) { // 检查是否为填充的空位
                    sum += values[chunkID][col_index_pos] * x[col];
                }
            }

            y[sumPos] = alpha * sum + beta * y[sumPos];
        }   
    }
}

template <typename IndexType, typename ValueType>
inline void __spmv_sell_cs_perthread(const IndexType * Reorder,
                                    const ValueType alpha, 
                                    const IndexType * const *col_index,
                                    const ValueType * const *values,
                                    const ValueType * x, 
                                    const ValueType beta, 
                                    ValueType * y, 
                                    const IndexType chunk_lrs, 
                                    const IndexType chunk_lre, 
                                    const IndexType num_rows, 
                                    const IndexType *max_row_width, 
                                    const IndexType chunk_size)
{
    for (size_t chunkID = chunk_lrs; chunkID < chunk_lre; chunkID++)
    {
        size_t chunk_width = max_row_width[chunkID];
        size_t chunk_start_row = chunkID * chunk_size;

        for (size_t row = 0; row < chunk_size; ++row)
        {
            size_t global_row = chunk_start_row + row;
            if (global_row >= num_rows) break; // 越界检查

            size_t sumPos = Reorder[global_row];
            ValueType sum = 0.0;
            if (beta)
            {
                y[sumPos] = beta * y[sumPos];
            }

            #pragma omp simd reduction(+:sum)
            for (size_t i = 0; i < chunk_width; ++i) 
            {
                size_t col_index_pos = row * chunk_width + i;
                size_t col = col_index[chunkID][col_index_pos];

                if (col >= 0) { // 检查是否为填充的空位
                    sum += values[chunkID][col_index_pos] * x[col];
                }
            }
            // Scale the sum by alpha and add to the y vector scaled by beta
            if(alpha == 1)
            {
                y[sumPos] += sum;
            }
            else {
                // y[sumPos] = alpha * sum + beta * y[sumPos];
                y[sumPos] += alpha * sum;
            }
        }
    }
}

template <typename IndexType, typename ValueType>
void __spmv_sell_cs_omp_lb_row( const IndexType * Reorder,
                                const IndexType num_rows,
                                const IndexType row_num_perC,
                                const IndexType total_chunk_num,
                                const IndexType num_nnzs, 
                                const ValueType alpha, 
                                const IndexType *max_row_width,
                                const IndexType * const *col_index,
                                const ValueType * const *values,
                                const ValueType * x, 
                                const ValueType beta, 
                                ValueType * y,
                                IndexType *partition)
{
    const IndexType thread_num = Le_get_thread_num();
    
    if(partition == nullptr)
    {
        partition = new_array<IndexType>(thread_num + 1);
        balanced_partition_row_by_nnz_sell(col_index, num_nnzs, row_num_perC, total_chunk_num, max_row_width, thread_num, partition);
        
    }
    #pragma omp parallel num_threads(thread_num)
    {
        IndexType tid = Le_get_thread_id();
        IndexType local_chunk_start = partition[tid];
        IndexType local_chunk_end   = partition[tid + 1];
        __spmv_sell_cs_perthread(Reorder, alpha, col_index, values, x, beta, y, local_chunk_start, local_chunk_end, num_rows, max_row_width, row_num_perC);
    }
}

template <typename IndexType, typename ValueType>
void LeSpMV_sell_c_sigma(const ValueType alpha, const SELL_C_Sigma_Matrix<IndexType, ValueType>& sell_c_sigma, const ValueType *x, const ValueType beta, ValueType *y)
{
    if (0 == sell_c_sigma.kernel_flag)
    {
        __spmv_sell_cs_serial_simple(sell_c_sigma.reorder, sell_c_sigma.num_rows, sell_c_sigma.chunkWidth_C, sell_c_sigma.validchunkNum, alpha, sell_c_sigma.chunk_len, sell_c_sigma.col_index, sell_c_sigma.values, x, beta, y);
    }
    else if (1 == sell_c_sigma.kernel_flag)
    {
        __spmv_sell_cs_omp_simple(sell_c_sigma.reorder, sell_c_sigma.num_rows, sell_c_sigma.chunkWidth_C, sell_c_sigma.validchunkNum, alpha, sell_c_sigma.chunk_len, sell_c_sigma.col_index, sell_c_sigma.values, x, beta, y);
    }
    else if (2 == sell_c_sigma.kernel_flag)
    {
        // call the load balanced by nnz of chunks in omp
        // just consider RowMajor
        __spmv_sell_cs_omp_lb_row( sell_c_sigma.reorder, sell_c_sigma.num_rows, sell_c_sigma.chunkWidth_C, sell_c_sigma.validchunkNum, sell_c_sigma.num_nnzs, alpha, sell_c_sigma.chunk_len, sell_c_sigma.col_index, sell_c_sigma.values, x, beta, y, sell_c_sigma.partition);

    }
    else{
        //DEFAULT: omp simple implementation
        __spmv_sell_cs_omp_simple(sell_c_sigma.reorder, sell_c_sigma.num_rows, sell_c_sigma.chunkWidth_C, sell_c_sigma.validchunkNum, alpha, sell_c_sigma.chunk_len, sell_c_sigma.col_index, sell_c_sigma.values, x, beta, y);
    }
}

template void LeSpMV_sell_c_sigma<int, float>(const float alpha, const SELL_C_Sigma_Matrix<int, float>& sell, const float * x, const float beta, float * y);

template void LeSpMV_sell_c_sigma<int, double>(const double alpha, const SELL_C_Sigma_Matrix<int, double>& sell, const double * x, const double beta, double * y);

template void LeSpMV_sell_c_sigma<long long, float>(const float alpha, const SELL_C_Sigma_Matrix<long long, float>& sell, const float * x, const float beta, float * y);

template void LeSpMV_sell_c_sigma<long long, double>(const double alpha, const SELL_C_Sigma_Matrix<long long, double>& sell, const double * x, const double beta, double * y);