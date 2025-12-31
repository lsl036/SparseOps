/**
 * @file spmv_ell.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief  Simple implementation of SpMV in ELL format.
 *         Using OpenMP automatic parallism and vectorization.
 * @version 0.1
 * @date 2023-11-17
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include"../include/LeSpMV.h"

#include"../include/thread.h"
template <typename IndexType, typename ValueType>
void __spmv_ell_serial_simple(  const IndexType num_rows,
                                const IndexType maxNonzeros, 
                                const ValueType alpha, 
                                const IndexType *colIndex,
                                const ValueType *values,
                                const ValueType * x, 
                                const ValueType beta, ValueType * y,
                                const LeadingDimension ld)
{
// COLMAJOR ELL:
    if(ColMajor == ld)
    {
        // #pragma omp parallel for num_threads(thread_num) schedule(SCHEDULE_STRATEGY)
        for (size_t rowId = 0; rowId < num_rows; ++rowId)
        {
            ValueType sum = 0.0;
            // #pragma omp simd reduction(+:sum)
            for (size_t item = 0; item < maxNonzeros; item++)
            {
                IndexType colID = colIndex[rowId + item * num_rows];
                ValueType val = values[rowId + item * num_rows];   // 放外面可能是为SIMD
                if (colID >= 0)  // colID != -1
                {
                    sum += val * x[colID];
                }
            }
            y[rowId] = alpha * sum + beta * y[rowId];
        }
    }
// ROWMAJOR ELL:
    else if(RowMajor == ld)
    {
        // #pragma omp parallel for num_threads(thread_num) schedule(SCHEDULE_STRATEGY)
        for (size_t rowId = 0; rowId < num_rows; ++rowId)
        {
            ValueType sum = 0.0;
            size_t rowOff = rowId * maxNonzeros;
            // #pragma omp simd reduction(+:sum)
            for (size_t item = 0; item < maxNonzeros; item++)
            {
                IndexType colID = colIndex[rowOff + item];
                ValueType val = values[rowOff + item]; // 放外面可能是为SIMD
                if (colID >= 0)  // colID != -1
                {
                    sum += val * x[colID];
                }
            }
            y[rowId] = alpha * sum + beta * y[rowId];
        }
    }
}


template <typename IndexType, typename ValueType>
void __spmv_ell_omp_simple( const IndexType num_rows,
                            const IndexType maxNonzeros, 
                            const ValueType alpha, 
                            const IndexType *colIndex,
                            const ValueType *values,
                            const ValueType * x, 
                            const ValueType beta, ValueType * y,
                            const LeadingDimension ld)
{
    const IndexType thread_num = Le_get_thread_num();

// COLMAJOR ELL:
    if(ColMajor == ld)
    {
        // #pragma omp parallel for num_threads(thread_num) schedule(SCHEDULE_STRATEGY)
        #pragma omp parallel for num_threads(thread_num)
        for (size_t rowId = 0; rowId < num_rows; ++rowId)
        {
            ValueType sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t item = 0; item < maxNonzeros; item++)
            {
                IndexType colID = colIndex[rowId + item * num_rows];
                ValueType val = values[rowId + item * num_rows]; // 放外面可能是为了SIMD
                if (colID >= 0)  // colID != -1
                {
                    sum += val * x[colID];
                }
            }
            y[rowId] = alpha * sum + beta * y[rowId];
        }
    }
// ROWMAJOR ELL:
    else if(RowMajor == ld)
    {
        // #pragma omp parallel for num_threads(thread_num) schedule(SCHEDULE_STRATEGY)
        #pragma omp parallel for num_threads(thread_num)
        for (size_t rowId = 0; rowId < num_rows; ++rowId)
        {
            ValueType sum = 0.0;
            size_t rowOff = rowId * maxNonzeros;
            #pragma omp simd reduction(+:sum)
            for (size_t item = 0; item < maxNonzeros; item++)
            {
                IndexType colID = colIndex[rowOff + item];
                ValueType val = values[rowOff + item]; // 放外面可能是为了SIMD
                if (colID >= 0)  // colID != -1
                {
                    sum += val * x[colID];
                }
            }
            y[rowId] = alpha * sum + beta * y[rowId];
        }
    }
}

template <typename IndexType, typename ValueType>
inline void  __spmv_ell_perthread(  const ValueType alpha, 
                                    const IndexType *colIndex,
                                    const ValueType *values,
                                    const ValueType * x, 
                                    const ValueType beta, ValueType * y,
                                    const IndexType lrs,
                                    const IndexType lre,
                                    const IndexType num_rows,
                                    const IndexType maxNonzeros )
{
    for (size_t row = lrs; row < lre; row++)
    {
        ValueType sum = 0.0;
        // Iterate over all possible non-zeros in the row
        for (size_t j = 0; j < maxNonzeros; ++j) {
            IndexType col = colIndex[row*maxNonzeros + j];
            if (col >= 0) { // Assuming -1 is used to indicate padding
                sum += values[row*maxNonzeros + j] * x[col];
            }
            else
                break;
        }
        // Scale the sum by alpha and add to the y vector scaled by beta
        y[row] = alpha * sum + beta * y[row];

    }
}

template <typename IndexType, typename ValueType>
void __spmv_ell_omp_lb_row( const IndexType num_rows,
                            const IndexType maxNonzeros,
                            const IndexType num_nnzs, 
                            const ValueType alpha, 
                            const IndexType *colIndex,
                            const ValueType *values,
                            const ValueType * x, 
                            const ValueType beta, ValueType * y,
                            const LeadingDimension ld,
                            IndexType *partition)
{
    const IndexType thread_num = Le_get_thread_num();
    // IndexType partition[thread_num + 1];  // index 0 ~ thread_num

    if(RowMajor == ld)
    {
        if(partition == nullptr)
        {
            partition = new_array<IndexType>(thread_num + 1);
            balanced_partition_row_by_nnz_ell(colIndex, num_nnzs, 
                                          num_rows, maxNonzeros, 
                                          thread_num, partition);
        }
        #pragma omp parallel num_threads(thread_num)
        {
            IndexType tid = Le_get_thread_id();
            IndexType local_m_start = partition[tid];
            IndexType local_m_end   = partition[tid + 1];
            __spmv_ell_perthread(alpha, colIndex, values, x, beta, y, local_m_start, local_m_end, num_rows, maxNonzeros);
        }
    }
    else
    {
        //  暂时使用 omp_simple 的 ColMajor实现方法
        // #pragma omp parallel for num_threads(thread_num) schedule(SCHEDULE_STRATEGY)
        #pragma omp parallel for num_threads(thread_num)
        for (size_t rowId = 0; rowId < num_rows; ++rowId)
        {
            ValueType sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t item = 0; item < maxNonzeros; item++)
            {
                IndexType colID = colIndex[rowId + item * num_rows];
                ValueType val = values[rowId + item * num_rows]; // 放外面可能是为了SIMD
                if (colID >= 0)  // colID != -1
                {
                    sum += val * x[colID];
                }
            }
            y[rowId] = alpha * sum + beta * y[rowId];
        }
    }
}

/**
 * @brief Compute y += alpha * A * x + beta * y for a sparse matrix
 *        Matrix Format: ELL
 *        Inside call : __spmv_ell_omp_simple() to calculation
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param alpha  scaling factor of A*x
 * @param ell    ELL Matrix
 * @param x      vector x
 * @param beta   scaling factor of vector y 
 * @param y      result vector y
 */
template <typename IndexType, typename ValueType>
void LeSpMV_ell(const ValueType alpha, const ELL_Matrix<IndexType, ValueType>& ell, const ValueType * x, const ValueType beta, ValueType * y)
{
    if (0 == ell.kernel_flag)
    {
        // call the simple serial implementation of ELL SpMV
        __spmv_ell_serial_simple(ell.num_rows, ell.max_row_width, alpha, ell.col_index, ell.values, x, beta, y, ell.ld);
    }
    else if(1 == ell.kernel_flag)
    {
        // call the simple OMP implementation of ELL SpMV
        __spmv_ell_omp_simple(ell.num_rows, ell.max_row_width, alpha, ell.col_index, ell.values, x, beta, y, ell.ld);
    }
    else if(2 == ell.kernel_flag)
    {
        // call the load balanced by nnz of each row in omp
        // Now just consider RowMajor
        __spmv_ell_omp_lb_row(ell.num_rows, ell.max_row_width, ell.num_nnzs, alpha, ell.col_index, ell.values, x, beta, y, ell.ld, ell.partition);
    }
    else{
        // DEFAULT: omp simple implementation
        __spmv_ell_omp_simple(ell.num_rows, ell.max_row_width, alpha, ell.col_index, ell.values, x, beta, y, ell.ld);
    }
}

template void LeSpMV_ell<int, float>(const float, const ELL_Matrix<int, float>&, const float*, const float, float*);

template void LeSpMV_ell<int, double>(const double, const ELL_Matrix<int, double>&, const double*, const double, double*);

template void LeSpMV_ell<long long, float>(const float, const ELL_Matrix<long long, float>&, const float*, const float, float*);

template void LeSpMV_ell<long long, double>(const double, const ELL_Matrix<long long, double>&, const double*, const double, double*);