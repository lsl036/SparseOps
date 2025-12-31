/**
 * @file spmv_s_ell.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief  Simple implementation of SpMV in Sliced SELL format.
 *         Using OpenMP automatic parallism and vectorization.
 * @version 0.1
 * @date 2023-12-11
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include"../include/LeSpMV.h"

#include"../include/thread.h"

template <typename IndexType, typename ValueType>
void __spmv_sell_serial_simple( const IndexType num_rows,
                                const IndexType row_num_perC,
                                const IndexType total_chunk_num,
                                const ValueType alpha,
                                const IndexType *max_row_width,
                                const IndexType * const *col_index,
                                const ValueType * const *values,
                                const ValueType * x, 
                                const ValueType beta, 
                                ValueType * y)
{
    for (size_t chunk = 0; chunk < total_chunk_num; ++chunk)
    {
        size_t chunk_width = max_row_width[chunk];
        size_t chunk_start_row = chunk * row_num_perC;

        for (size_t row = 0; row < row_num_perC; ++row)
        {
            size_t global_row = chunk_start_row + row;
            if (global_row >= num_rows) break; // 越界检查
            
            ValueType sum = 0;
            // #pragma omp simd reduction(+:sum)
            for (size_t i = 0; i < chunk_width; ++i) 
            {
                size_t col_index_pos = row * chunk_width + i;
                size_t col = col_index[chunk][col_index_pos];

                if (col >= 0) { // 检查是否为填充的空位
                    sum += values[chunk][col_index_pos] * x[col];
                }
            }

            y[global_row] = alpha * sum + beta * y[global_row];
        }
    }

}


/**
 * @brief 
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param row_num_perC      一个 chunk 内的 行数目
 * @param total_chunk_num 
 * @param alpha 
 * @param col_index 
 * @param values 
 * @param x 
 * @param beta 
 * @param y 
 */
template <typename IndexType, typename ValueType>
void __spmv_sell_omp_simple(const IndexType num_rows,
                            const IndexType row_num_perC,
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
    // const IndexType chunk_size = std::max(1, total_chunk_num/ thread_num);

    // const int colindex_align_bytes = ALIGNMENT_NUM * sizeof(IndexType);
    // const int values_align_bytes   = ALIGNMENT_NUM * sizeof(ValueType);

    //  Only spmv for row major SELL
    // #pragma omp parallel for num_threads(thread_num) schedule(SCHEDULE_STRATEGY)
    #pragma omp parallel for num_threads(thread_num)
    for (size_t chunk = 0; chunk < total_chunk_num; ++chunk)
    {
        size_t chunk_width = max_row_width[chunk];
        size_t chunk_start_row = chunk * row_num_perC;

        for (size_t row = 0; row < row_num_perC; ++row)
        {
            ValueType sum = 0;
            size_t global_row = chunk_start_row + row;
            if (global_row >= num_rows) break; // 越界检查

            #pragma omp simd reduction(+:sum)
            for (size_t i = 0; i < chunk_width; ++i) 
            {
                size_t col_index_pos = row * chunk_width + i;
                size_t col = col_index[chunk][col_index_pos];

                if (col >= 0) { // 检查是否为填充的空位
                    sum += values[chunk][col_index_pos] * x[col];
                }
            }

            y[global_row] = alpha * sum + beta * y[global_row];
        }
    }
}

template <typename IndexType, typename ValueType>
inline void __spmv_sell_perthread(  const ValueType alpha, 
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
            ValueType sum = 0.0;
            size_t global_row = chunk_start_row + row;
            if (global_row >= num_rows) break; // 越界检查

            if (beta)
            {
                y[global_row] = beta * y[global_row];
            }

            // #pragma omp simd reduction(+:sum)
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
                y[global_row] += sum;
            }
            else {
                // y[global_row] = alpha * sum + beta * y[global_row];
                y[global_row] += alpha * sum;
            }
        }
    }
}

template <typename IndexType, typename ValueType>
void __spmv_sell_omp_lb_row(const IndexType num_rows,
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
        __spmv_sell_perthread(alpha, col_index, values, x, beta, y, local_chunk_start, local_chunk_end, num_rows, max_row_width, row_num_perC);
    }
}

template <typename IndexType, typename ValueType>
void LeSpMV_sell(const ValueType alpha, const S_ELL_Matrix<IndexType, ValueType>& sell, const ValueType * x, const ValueType beta, ValueType * y){
    if (0 == sell.kernel_flag)
    {
        __spmv_sell_serial_simple(sell.num_rows, sell.sliceWidth, sell.chunk_num, alpha, sell.row_width, sell.col_index, sell.values, x, beta, y);

    }
    else if(1 == sell.kernel_flag)
    {
        __spmv_sell_omp_simple(sell.num_rows, sell.sliceWidth, sell.chunk_num, alpha, sell.row_width, sell.col_index, sell.values, x, beta, y);

    }
    else if(2 == sell.kernel_flag)
    {
        // call the load balanced by nnz of chunks in omp
        // just consider RowMajor
        __spmv_sell_omp_lb_row( sell.num_rows, sell.sliceWidth, sell.chunk_num, sell.num_nnzs, alpha, sell.row_width, sell.col_index, sell.values, x, beta, y, sell.partition);
    }
    else{
        //DEFAULT: omp simple implementation
        __spmv_sell_omp_simple(sell.num_rows, sell.sliceWidth, sell.chunk_num, alpha, sell.row_width, sell.col_index, sell.values, x, beta, y);
    }
}

template void LeSpMV_sell<int, float>(const float alpha, const S_ELL_Matrix<int, float>& sell, const float * x, const float beta, float * y);

template void LeSpMV_sell<int, double>(const double alpha, const S_ELL_Matrix<int, double>& sell, const double * x, const double beta, double * y);

template void LeSpMV_sell<long long, float>(const float alpha, const S_ELL_Matrix<long long, float>& sell, const float * x, const float beta, float * y);

template void LeSpMV_sell<long long, double>(const double alpha, const S_ELL_Matrix<long long, double>& sell, const double * x, const double beta, double * y);