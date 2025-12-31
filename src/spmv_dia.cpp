/**
 * @file spmv_dia.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Simple implementation of SpMV in DIA format.
 *         Using OpenMP automatic parallism and vectorization.
 * @version 0.1
 * @date 2023-11-23
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include"../include/LeSpMV.h"

template <typename IndexType, typename ValueType>
void __spmv_dia_serial_simple(  const ValueType alpha, 
                                const IndexType num_rows,
                                const IndexType stride,
                                const IndexType complete_ndiags,
                                const long int  * dia_offset,
                                const ValueType * dia_data,
                                const ValueType * x,
                                const ValueType beta, ValueType * y)
{
    for (size_t i = 0; i < num_rows; ++i) {
        y[i] = beta * y[i];
    }

    // 按照对角线来找数进行计算
    for (size_t d = 0; d < complete_ndiags; ++d) {
        IndexType offset = dia_offset[d]; // 获取当前对角线的偏移量
        IndexType start = std::max((IndexType)0, -offset); // 如果是负的，说明行要从 -offset 开始算
        IndexType end = std::min(num_rows, num_rows - offset); // 这个则是对应对角线最后一个元素实际的最大行号 end是多少， 如果是正的，说明对角线在上方，行号到不了最后一行

        // 遍历当前对角线上的非零元素， 按行号 i ，也是这条对角线往后的位移
        for (IndexType i = start; i < end; ++i) {
            IndexType j = i + offset; // 计算实际的列索引
            ValueType val = dia_data[d * stride + i];
            // #pragma omp atomic
            y[i] += alpha * val * x[j];
        }
    }
}

template <typename IndexType, typename ValueType>
void __spmv_dia_omp_simple(  const ValueType alpha, 
                                const IndexType num_rows,
                                const IndexType stride,
                                const IndexType complete_ndiags,
                                const long int  * dia_offset,
                                const ValueType * dia_data,
                                const ValueType * x,
                                const ValueType beta, ValueType * y)
{
    const IndexType thread_num = Le_get_thread_num();
    if (beta){
        #pragma omp parallel for num_threads(thread_num)
        for (size_t i = 0; i < num_rows; ++i) {
            y[i] *= beta;
        }
    }

    if ( 1 == alpha)
    {
        // alpha 不乘
        #pragma omp parallel for num_threads(thread_num)
        for (size_t d = 0; d < complete_ndiags; ++d) {
            IndexType offset = dia_offset[d]; // 获取当前偏移量
            IndexType start = std::max((IndexType)0, -offset);
            IndexType end = std::min(num_rows, num_rows - offset);

            // 遍历当前对角线上的非零元素
            for (IndexType i = start; i < end; ++i) {
                IndexType j = i + offset; // 计算实际的列索引
                ValueType val = dia_data[d * stride + i];
                #pragma omp atomic
                y[i] += val * x[j];
            }
        }
    }
    else {
        // 然后并行计算矩阵向量乘法
        #pragma omp parallel for num_threads(thread_num)
        for (size_t d = 0; d < complete_ndiags; ++d) {
            IndexType offset = dia_offset[d]; // 获取当前偏移量
            IndexType start = std::max((IndexType)0, -offset);
            IndexType end = std::min(num_rows, num_rows - offset);

            // 遍历当前对角线上的非零元素
            for (IndexType i = start; i < end; ++i) {
                IndexType j = i + offset; // 计算实际的列索引
                ValueType val = alpha * dia_data[d * stride + i];
                #pragma omp atomic
                y[i] += val * x[j];
            }
        }
    }
}

template <typename IndexType, typename ValueType>
void __spmv_dia_alpha(  const ValueType alpha, 
                        const IndexType num_rows,
                        const IndexType num_cols,
                        const IndexType stride,
                        const IndexType complete_ndiags,
                        const long int  * dia_offset,
                        const ValueType * dia_data,
                        const ValueType * x,
                        const ValueType beta, ValueType * y)
{
    const IndexType thread_num = Le_get_thread_num();
    ValueType **tmp = (ValueType**)malloc(sizeof(ValueType*) * thread_num);
    for (int i = 0; i < thread_num; i++)
    {
        tmp[i] = (ValueType*)malloc(sizeof(ValueType) * num_rows);
        memset(tmp[i], 0 , sizeof(ValueType) * num_rows);
    }

    if ( 1 == alpha)
    {
        #pragma omp parallel for num_threads(thread_num)
        for (size_t i = 0; i < complete_ndiags; ++i)
        {
            const IndexType threadId = Le_get_thread_id();
            const IndexType dis = dia_offset[i];
            const IndexType row_start = std::max((IndexType)0, -dis);
            const IndexType col_start = std::max((IndexType)0, dis);
            const IndexType nnz = (num_rows - row_start)<(num_cols - col_start)?(num_rows - row_start):(num_cols - col_start);
            const IndexType start = i * stride;
            for (size_t j = 0; j < nnz; ++j)
            {
                // ValueType v = alpha * dia_data[start + row_start + j];
                tmp[threadId][row_start + j] += dia_data[start + row_start + j] * x[col_start + j];
            }
        }
    }
    else{
        #pragma omp parallel for num_threads(thread_num)
        for (size_t i = 0; i < complete_ndiags; ++i)
        {
            const IndexType threadId = Le_get_thread_id();
            const IndexType dis = dia_offset[i];
            const IndexType row_start = std::max((IndexType)0, -dis);
            const IndexType col_start = std::max((IndexType)0, dis);
            const IndexType nnz = (num_rows - row_start)<(num_cols - col_start)?(num_rows - row_start):(num_cols - col_start);
            const IndexType start = i * stride;
            for (size_t j = 0; j < nnz; ++j)
            {
                ValueType v = alpha * dia_data[start + row_start + j];
                tmp[threadId][row_start + j] += v * x[col_start + j];
            }
        }
    }

    #pragma omp parallel for num_threads(thread_num)
    for(size_t i = 0; i < num_rows; ++i)
	{
        y[i] *= beta;
		for(size_t j = 0; j < thread_num; ++j)
	 	{
			// alpha_add(y[i], y[i], tmp[j][i]);
            y[i] += tmp[j][i];
	 	}
	}

    #pragma omp parallel for num_threads(thread_num)
    for(int i = 0; i < thread_num; ++i)
	{
		free(tmp[i]);
	}

	free(tmp);

}

template <typename IndexType, typename ValueType>
void LeSpMV_dia(const ValueType alpha, const DIA_Matrix<IndexType, ValueType>& dia, const ValueType * x, const ValueType beta, ValueType * y)
{
    if ( 0 == dia.kernel_flag)
    {
        // call the simple serial implementation of DIA SpMV
        __spmv_dia_serial_simple(alpha, dia.num_rows, dia.stride, dia.complete_ndiags, dia.diag_offsets, dia.diag_data, x, beta, y);
    }
    else if( 1 == dia.kernel_flag)
    {
        // call the simple OMP implementation of DIA SpMV.
        __spmv_dia_omp_simple(alpha, dia.num_rows, dia.stride, dia.complete_ndiags, dia.diag_offsets, dia.diag_data, x, beta, y);
    }
    else if (2 == dia.kernel_flag)
    {
        __spmv_dia_alpha(alpha, dia.num_rows, dia.num_cols, dia.stride, dia.complete_ndiags, dia.diag_offsets, dia.diag_data, x, beta, y);
    }
    else // default
    {
        __spmv_dia_omp_simple(alpha, dia.num_rows, dia.stride, dia.complete_ndiags, dia.diag_offsets, dia.diag_data, x, beta, y);
    }
}

template void LeSpMV_dia<int, float>(const float, const DIA_Matrix<int, float>&, const float*, const float, float*);

template void LeSpMV_dia<int, double>(const double, const DIA_Matrix<int, double>&, const double*, const double, double*);

template void LeSpMV_dia<long long, float>(const float, const DIA_Matrix<long long, float>&, const float*, const float, float*);

template void LeSpMV_dia<long long, double>(const double, const DIA_Matrix<long long, double>&, const double*, const double, double*);