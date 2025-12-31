/**
 * @file spmv_bsr.cpp
 * @author your name (you@domain.com)
 * @brief Simple implementation of SpMV in Sliced SELL-c-R format.
 *         Using OpenMP automatic parallism and vectorization.
 * @version 0.1
 * @date 2024-02-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include"../include/LeSpMV.h"

#include"../include/thread.h"

template <typename IndexType, typename ValueType>
void __spmv_bsr_serial_simple(  const IndexType num_rows,
                                const IndexType blockDimRow,
                                const IndexType blockDimCol,
                                const IndexType mb,
                                const ValueType alpha,
                                const IndexType *row_ptr,
                                const IndexType *col_index,
                                const ValueType *values,
                                const ValueType *x,
                                const ValueType beta,
                                ValueType *y)
{
    for (size_t i = 0; i < mb; i++)
    {
        size_t start = row_ptr[i];
        size_t end   = row_ptr[i+1];

        std::vector<ValueType> tmp(blockDimRow,0);

        for (size_t j = start; j < end; j++)
        {
            // 获取当前块的列索引
            size_t block_col = col_index[j];

            // 执行块与向量的乘法
            for (size_t br = 0; br < blockDimRow; ++br) {
                for (size_t bc = 0; bc < blockDimCol; ++bc) {
                    // 计算输入向量x 的索引
                    size_t x_index = block_col * blockDimCol + bc;
                    // 累加结果
                    // tmp[br] += alpha * values[j * blockDimRow * blockDimCol + br * blockDimCol + bc] * x[x_index];
                    tmp[br] += values[j * blockDimRow * blockDimCol + br * blockDimCol + bc] * x[x_index];
                }
            }
        }

        for (size_t br = 0; br < blockDimRow; br++)
        {
            // 计算输出向量的索引
            size_t y_index = i * blockDimRow + br;
            if (y_index < num_rows)
            {
                // 更新 y
                // y[y_index] = tmp[br] + beta * y[y_index];
                y[y_index] = alpha * tmp[br] + beta * y[y_index];
            }
        }
    }
}

template <typename IndexType, typename ValueType>
void __spmv_bsr_omp_simple( const IndexType num_rows,
                            const IndexType blockDimRow,
                            const IndexType blockDimCol,
                            const IndexType mb,
                            const ValueType alpha,
                            const IndexType *row_ptr,
                            const IndexType *col_index,
                            const ValueType *values,
                            const ValueType *x,
                            const ValueType beta,
                            ValueType *y)
{
    const size_t thread_num = Le_get_thread_num();

    #pragma omp parallel for num_threads(thread_num)
    for (size_t i = 0; i < mb; i++)
    {
        size_t start = row_ptr[i];
        size_t end   = row_ptr[i+1];

        std::vector<ValueType> tmp(blockDimRow,0);

        for (size_t j = start; j < end; j++)
        {
            // 获取当前块的列索引
            size_t block_col = col_index[j];

            // 执行块与向量的乘法
            for (size_t br = 0; br < blockDimRow; ++br) {
                #pragma omp simd
                for (size_t bc = 0; bc < blockDimCol; ++bc) {
                    // 计算输入向量x 的索引
                    size_t x_index = block_col * blockDimCol + bc;
                    // 累加结果
                    tmp[br] += values[j * blockDimRow * blockDimCol + br * blockDimCol + bc] * x[x_index];
                }
            }
        }
        // 更新 y
        for (size_t br = 0; br < blockDimRow; br++)
        {
            // 计算输出向量的索引
            size_t y_index = i * blockDimRow + br;
            if (y_index < num_rows)
            {
                y[y_index] = alpha * tmp[br] + beta * y[y_index];
            }
        }
    }
}

template <typename IndexType, typename ValueType>
inline void __spmv_bsr_perthread(   const ValueType alpha,
                                    const IndexType blockDimRow,
                                    const IndexType blockDimCol,
                                    const IndexType mb,
                                    const IndexType num_rows,
                                    const IndexType *row_ptr,
                                    const IndexType *col_index,
                                    const ValueType *values,
                                    const ValueType *x,
                                    const ValueType beta,
                                    ValueType *y,
                                    const IndexType lrs,
                                    const IndexType lre)
{
    // 这里 lrs ~ lre 代表要计算的 row_block 数目
    size_t task_rows = (lre - lrs) * blockDimRow;

    // For matC, block_layout is defaulted as row_major
    std::vector<ValueType> tmp(task_rows,0);

    // Only support Rowmajor layout of BSR format
    for (size_t i = lrs, j = 0; i < lre; ++i, ++j)
    {
        for(size_t ai = row_ptr[i]; ai < row_ptr[i+1]; ++ai)
        {
             // 执行块与向量的乘法
            for (size_t br = 0; br < blockDimRow; ++br) {
                #pragma omp simd
                for (size_t bc = 0; bc < blockDimCol; ++bc) {
                    // 累加结果
                    tmp[ j*blockDimRow + br] += values[ai*blockDimRow*blockDimCol + br*blockDimCol + bc] * x[col_index[ai]*blockDimCol + bc];
                }
            }
        }
    }

    if ( alpha == 1 && beta ==0)
    {
        for (size_t m = lrs * blockDimRow, m_t = 0; m < lre * blockDimRow; m++, m_t++)
        {
            if (m < num_rows)
            y[m] = tmp[m_t];
        }
    }
    else if (beta == 0)
    {
        for (size_t m = lrs * blockDimRow, m_t = 0; m < lre * blockDimRow; m++, m_t++)
        {
            if (m < num_rows)
            y[m] = alpha * tmp[m_t];
        }
    }
    else {
        // m: row_idx;  m_t : row_block_idx
        for (size_t m = lrs * blockDimRow, m_t = 0; m < lre * blockDimRow; m++, m_t++)
        {
            if (m < num_rows)
            y[m] = alpha * tmp[m_t] + beta * y[m];
        }
    }
}

template <typename IndexType, typename ValueType>
void __spmv_bsr_lb_alpha(   const IndexType blockDimRow,
                            const IndexType blockDimCol,
                            const IndexType mb,
                            const IndexType num_rows,
                            const ValueType alpha,
                            const IndexType *row_ptr,
                            const IndexType *col_index,
                            const ValueType *values,
                            const ValueType *x,
                            const ValueType beta,
                            ValueType *y,
                            IndexType* partition)
{
    const IndexType thread_num = Le_get_thread_num();

    if(partition == nullptr)
    {
        partition = new_array<IndexType>(thread_num + 1);
        balanced_partition_row_by_nnz(row_ptr, mb, thread_num, partition);
    }

    #pragma omp parallel num_threads(thread_num)
    {
        IndexType tid = Le_get_thread_id();
        IndexType local_m_start = partition[tid];
        IndexType local_m_end   = partition[tid + 1];
        __spmv_bsr_perthread(alpha, blockDimRow, blockDimCol, mb, num_rows, row_ptr, col_index, values, x, beta, y, local_m_start, local_m_end);
    }
}                            

template <typename IndexType, typename ValueType>
void LeSpMV_bsr(const ValueType alpha, const BSR_Matrix<IndexType, ValueType>& bsr, const ValueType *x, const ValueType beta, ValueType *y)
{
    if ( 0 == bsr.kernel_flag)
    {
        __spmv_bsr_serial_simple(bsr.num_rows, bsr.blockDim_r, bsr.blockDim_c, bsr.mb, alpha, bsr.row_ptr, bsr.block_colindex, bsr.block_data, x, beta, y);
    }
    else if (1 == bsr.kernel_flag)
    {
        __spmv_bsr_omp_simple(bsr.num_rows, bsr.blockDim_r, bsr.blockDim_c, bsr.mb, alpha, bsr.row_ptr, bsr.block_colindex, bsr.block_data, x, beta, y);
    }
    else if(2 == bsr.kernel_flag)
    {
        __spmv_bsr_lb_alpha(bsr.blockDim_r, bsr.blockDim_c, bsr.mb, bsr.num_rows, alpha, bsr.row_ptr, bsr.block_colindex, bsr.block_data, x, beta, y, bsr.partition);
    }
    else
    {
        __spmv_bsr_omp_simple(bsr.num_rows, bsr.blockDim_r, bsr.blockDim_c, bsr.mb, alpha, bsr.row_ptr, bsr.block_colindex, bsr.block_data, x, beta, y);
    }
}

template void LeSpMV_bsr<int, float>(const float alpha, const BSR_Matrix<int, float>& bsr, const float *x, const float beta, float *y);

template void LeSpMV_bsr<int, double>(const double alpha, const BSR_Matrix<int, double>& bsr, const double *x, const double beta, double *y);

template void LeSpMV_bsr<long long, float>(const float alpha, const BSR_Matrix<long long, float>& bsr, const float *x, const float beta, float *y);

template void LeSpMV_bsr<long long, double>(const double alpha, const BSR_Matrix<long long, double>& bsr, const double *x, const double beta, double *y);