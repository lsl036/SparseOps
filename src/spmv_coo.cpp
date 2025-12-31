/**
 * @file spmv_coo.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief  Simple implementation of SpMV in COO format.
 *         Using OpenMP automatic parallism and vectorization.
 * @version 0.1
 * @date 2023-11-15
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include"../include/LeSpMV.h"

#include <thread.h>

// 串行
template <typename IndexType, typename ValueType>
void __spmv_coo_serial_simple(  const IndexType num_rows,
                                const IndexType num_nnzs, 
                                const ValueType alpha, 
                                const IndexType *Ai,
                                const IndexType *Aj,
                                const ValueType *Ax,
                                const ValueType * x, 
                                const ValueType beta, ValueType * y)
{
    for (IndexType i = 0; i < num_rows; ++i) {
        y[i] *= beta;
    }
    for (IndexType i = 0; i < num_nnzs; ++i) {
        y[Ai[i]] += alpha * Ax[i] * x[Aj[i]];
    }
}

//直接 omp 并行
template <typename IndexType, typename ValueType>
void __spmv_coo_omp_simple (const IndexType num_rows,
                            const IndexType num_nnzs, 
                            const ValueType alpha, 
                            const IndexType *Ai,
                            const IndexType *Aj,
                            const ValueType *Ax,
                            const ValueType * x, 
                            const ValueType beta, ValueType * y)
{
    const IndexType thread_num = Le_get_thread_num();
    if (beta){
        // #pragma omp parallel for schedule(static) num_threads(thread_num)
        #pragma omp parallel for num_threads(thread_num)
        for (IndexType i = 0; i < num_rows; ++i) {
            y[i] *= beta;
        }
    }

    if ( 1 == alpha)
    {
        // omp automatic parallel but not using SIMD
        // #pragma omp parallel for schedule(static) num_threads(thread_num)
        #pragma omp parallel for num_threads(thread_num)
        for (IndexType i = 0; i < num_nnzs; ++i) {
            // OpenMP的reduction机制确保线程安全地更新y向量
            #pragma omp atomic
            y[Ai[i]] += Ax[i] * x[Aj[i]];
        }
    }
    else{
        // omp automatic parallel but not using SIMD
        // #pragma omp parallel for schedule(static) num_threads(thread_num)
        #pragma omp parallel for num_threads(thread_num)
        for (IndexType i = 0; i < num_nnzs; ++i) {
            // OpenMP的reduction机制确保线程安全地更新y向量
            #pragma omp atomic
            y[Ai[i]] += alpha * Ax[i] * x[Aj[i]];
        }
    }
}

template <typename IndexType, typename ValueType>
void __spmv_coo_omp_lb (    const IndexType num_rows,
                            const IndexType num_nnzs, 
                            const ValueType alpha, 
                            const IndexType *Ai,
                            const IndexType *Aj,
                            const ValueType *Ax,
                            const ValueType * x, 
                            const ValueType beta, ValueType * y)
{
    // omp is enough for nnz load balanced
}

//openmp load balanced in alphasparse
template <typename IndexType, typename ValueType>
void __spmv_coo_omp_alpha (    const IndexType num_rows,
                            const IndexType num_nnzs, 
                            const ValueType alpha, 
                            const IndexType *Ai,
                            const IndexType *Aj,
                            const ValueType *Ax,
                            const ValueType * x, 
                            const ValueType beta, ValueType * y)
{
    // or CPU_SOCKET * CPU_CORES_PER_SOC * CPU_HYPER_THREAD
    const IndexType thread_num = Le_get_thread_num();
    ValueType **tmp = (ValueType **)malloc(thread_num*sizeof(ValueType *));

    #pragma omp parallel for num_threads(thread_num)
    for (IndexType i = 0; i < thread_num; i++)
    {
        tmp[i] = (ValueType *)malloc(num_rows * sizeof(ValueType));
        memset(tmp[i], 0 , num_rows * sizeof(ValueType));
    }

// 计算 alpha * A * x
    if ( 1 == alpha){
        #pragma omp parallel for num_threads(thread_num)
        for (IndexType i = 0; i < num_nnzs; i++)
        {
            const IndexType threadId = Le_get_thread_id();
            const IndexType rowId = Ai[i];
            const IndexType colId = Aj[i];
            ValueType v;
            // alpha_mul(v, A->values[i], x[c]);
            v = Ax[i] * x[colId];
            // alpha_madde(tmp[threadId][rowId], alpha, v);
            // #pragma omp atomic
            tmp[threadId][rowId] += v;
        }
    }
    else{
        #pragma omp parallel for num_threads(thread_num)
        for (IndexType i = 0; i < num_nnzs; i++)
        {
            const IndexType threadId = Le_get_thread_id();
            const IndexType rowId = Ai[i];
            const IndexType colId = Aj[i];
            ValueType v;
            // alpha_mul(v, A->values[i], x[c]);
            v = Ax[i] * x[colId];
            // alpha_madde(tmp[threadId][rowId], alpha, v);
            // #pragma omp atomic
            tmp[threadId][rowId] += alpha*v;
        }
    }

// 计算beta
    #pragma omp parallel for num_threads(thread_num)
    for (IndexType i = 0; i < num_rows; ++i)
	{
        y[i] = beta * y[i];
        for (IndexType j = 0; j < thread_num; ++j)
		{
            // alpha_add(y[i], y[i], tmp[j][i]);
            y[i] = y[i] + tmp[j][i];
        }
    }
}


template <typename IndexType, typename ValueType>
void LeSpMV_coo(const ValueType alpha, const COO_Matrix<IndexType, ValueType>& coo, const ValueType * x, const ValueType beta, ValueType * y)
{
    if (0 == coo.kernel_flag){
        __spmv_coo_serial_simple(coo.num_rows, coo.num_nnzs, alpha, coo.row_index, coo.col_index, coo.values, x, beta, y);
    }
    else if (1 == coo.kernel_flag){
        __spmv_coo_omp_simple(coo.num_rows, coo.num_nnzs, alpha, coo.row_index, coo.col_index, coo.values, x, beta, y);
    }
    else if (2 == coo.kernel_flag){
    
        __spmv_coo_omp_lb(coo.num_rows, coo.num_nnzs, alpha, coo.row_index, coo.col_index, coo.values, x, beta, y);
    }
    else if (3 == coo.kernel_flag){
    
        __spmv_coo_omp_alpha(coo.num_rows, coo.num_nnzs, alpha, coo.row_index, coo.col_index, coo.values, x, beta, y);
    }
    else
    {
        // DEFAULT: omp simple implementation
        __spmv_coo_omp_simple(coo.num_rows, coo.num_nnzs, alpha, coo.row_index, coo.col_index, coo.values, x, beta, y);
    }
}

template void LeSpMV_coo<int, float>(const float, const COO_Matrix<int, float>&, const float* , const float, float*);

template void LeSpMV_coo<int, double>(const double, const COO_Matrix<int, double>&, const double* , const double, double*);

template void LeSpMV_coo<long long, float>(const float, const COO_Matrix<long long, float>&, const float* , const float, float*);

template void LeSpMV_coo<long long, double>(const double, const COO_Matrix<long long, double>&, const double* , const double, double*);