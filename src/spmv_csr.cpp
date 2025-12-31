/**
 * @file spmv_csr.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief  Simple implementation of SpMV in CSR format.
 *         Using OpenMP automatic parallism and vectorization.
 * @version 0.1
 * @date 2023-11-17
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include"../include/LeSpMV.h"

#include"../include/thread.h"

/**
 * @brief Inline routine for each thread that compute rows SpMV for 
 *        lrs-th row to lre-th end.  
 *        Implemented by omp SIMD reduction
 * @tparam IndexType 
 * @tparam ValueType 
 * @param alpha 
 * @param Ap 
 * @param Aj 
 * @param Ax 
 * @param x 
 * @param beta 
 * @param y 
 * @param lrs 
 * @param lre 
 */
template <typename IndexType, typename ValueType>
inline void  __spmv_csr_perthread(  const ValueType alpha, 
                                    const IndexType *Ap,
                                    const IndexType *Aj,
                                    const ValueType *Ax,
                                    const ValueType * x, 
                                    const ValueType beta, ValueType * y,
                                    const IndexType lrs,
                                    const IndexType lre)
{
    for (IndexType row = lrs; row < lre; row++)
    {
        IndexType pks = Ap[row];
        IndexType pke = Ap[row+1];
        
        ValueType sum = 0;
        // 对每行中的非零元素执行乘法和累加, omp 自动化 SIMD
        #pragma omp simd
        for (IndexType col_id = pks; col_id < pke; ++col_id) {
            sum += Ax[col_id] * x[Aj[col_id]];
        }
        // 更新y向量
        if ( alpha == 1 && beta ==0){
            y[row] = sum;
        }
        else{
            y[row] = alpha * sum + beta * y[row];
        }
    }
}

template <typename IndexType, typename ValueType>
void __spmv_csr_serial_simple(  const IndexType num_rows, 
                                const ValueType alpha, 
                                const IndexType *Ap,
                                const IndexType *Aj,
                                const ValueType *Ax,
                                const ValueType * x, 
                                const ValueType beta, ValueType * y)
{
    for (IndexType row = 0; row < num_rows; ++row) 
    {
        ValueType sum = 0;
        const IndexType row_start = Ap[row];
        const IndexType row_end   = Ap[row+1];

        for (IndexType jj = row_start; jj < row_end; ++jj) {
            sum += Ax[jj] * x[Aj[jj]];
        }
        // 更新y向量
        if ( alpha == 1 && beta ==0){
            y[row] = sum;
        }
        else{
            y[row] = alpha * sum + beta * y[row];
        }
    }
}


template <typename IndexType, typename ValueType>
void __spmv_csr_omp_simple (const IndexType num_rows, 
                            const ValueType alpha, 
                            const IndexType *Ap,
                            const IndexType *Aj,
                            const ValueType *Ax,
                            const ValueType * x, 
                            const ValueType beta, ValueType * y)
{
    const IndexType thread_num = Le_get_thread_num();
    // 并行处理每一行
    // #pragma omp parallel for num_threads(thread_num) schedule(SCHEDULE_STRATEGY)
    #pragma omp parallel for num_threads(thread_num)
    for (IndexType row = 0; row < num_rows; ++row) 
    {
        ValueType sum = 0;
        const IndexType row_start = Ap[row];
        const IndexType row_end   = Ap[row+1];

        // 对每行中的非零元素执行乘法和累加, omp 自动化 SIMD
        #pragma omp simd reduction(+:sum)
        for (IndexType jj = row_start; jj < row_end; ++jj) {
            sum += Ax[jj] * x[Aj[jj]];
        }

        // 更新y向量
        y[row] = alpha * sum + beta * y[row];
    } 
}

/**
 * @brief Load balance by nnz of rows in CSR method by alphasparse
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param num_rows 
 * @param alpha 
 * @param Ap 
 * @param Aj 
 * @param Ax 
 * @param x 
 * @param beta 
 * @param y 
 */
template <typename IndexType, typename ValueType>
void __spmv_csr_omp_lb (const IndexType num_rows, 
                        const ValueType alpha, 
                        const IndexType *Ap,
                        const IndexType *Aj,
                        const ValueType *Ax,
                        const ValueType * x, 
                        const ValueType beta, ValueType * y,
                        IndexType* partition)
{
    const IndexType thread_num = Le_get_thread_num();
    // IndexType partition[thread_num + 1];
    if(partition == nullptr)
    {
        partition = new_array<IndexType>(thread_num + 1);
        balanced_partition_row_by_nnz(Ap, num_rows, thread_num, partition);
    }

    #pragma omp parallel num_threads(thread_num)
    {
        IndexType tid = Le_get_thread_id();
        IndexType local_m_start = partition[tid];
        IndexType local_m_end   = partition[tid + 1];
        __spmv_csr_perthread(alpha, Ap, Aj, Ax, x, beta, y, local_m_start, local_m_end);
    }
}

template <typename IndexType, typename ValueType>
void LeSpMV_csr(const ValueType alpha, const CSR_Matrix<IndexType, ValueType>& csr, const ValueType * x, const ValueType beta, ValueType * y)
{
    if (0 == csr.kernel_flag)
    {
        // call the simple serial implementation of CSR SpMV
        __spmv_csr_serial_simple(csr.num_rows, alpha, csr.row_offset, csr.col_index, csr.values, x, beta, y);
    }
    else if (1 == csr.kernel_flag){
        // call the simple OMP implementation of CSR SpMV.
        __spmv_csr_omp_simple(csr.num_rows, alpha, csr.row_offset, csr.col_index, csr.values, x, beta, y);
    }
    else if (2 == csr.kernel_flag)
    {
        // Call the load balanced by nnzs of rows of CSR SpMV
        __spmv_csr_omp_lb(csr.num_rows, alpha, csr.row_offset, csr.col_index, csr.values, x, beta, y, csr.partition);
    }
    else{
        // DEFAULT: omp simple implementation
        __spmv_csr_omp_simple(csr.num_rows, alpha, csr.row_offset, csr.col_index, csr.values, x, beta, y);
    }
}

template void LeSpMV_csr<int, float>(const float, const CSR_Matrix<int, float>&, const float* , const float, float*);

template void LeSpMV_csr<int, double>(const double, const CSR_Matrix<int, double>&, const double* , const double, double*);

template void LeSpMV_csr<long long, float>(const float, const CSR_Matrix<long long, float>&, const float* , const float, float*);

template void LeSpMV_csr<long long, double>(const double, const CSR_Matrix<long long, double>&, const double* , const double, double*);
