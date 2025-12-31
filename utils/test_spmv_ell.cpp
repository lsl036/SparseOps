/**
 * @file test_spmv_ell.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Test routine for spmv_ell.cpp
 * @version 0.1
 * @date 2023-11-17
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include"../include/LeSpMV.h"
#include<iostream>


/**
 * @brief Input CSR format for reference. Inside we make an ELL format
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param csr 
 * @param kernel_tag 
 * @return int 
 */
template <typename IndexType, typename ValueType>
double test_ell_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, LeadingDimension ld, int schedule_mod,  double &convert_time)
{
    double msec_per_iteration;
    std::cout << "=====  Testing ELL Kernels  =====" << std::endl;

    ELL_Matrix<IndexType,ValueType> ell;

    // formats convert overhead
    timer t;

    ell = csr_to_ell(csr_ref, ld);
    
    double msec_convert = (double) t.milliseconds_elapsed();
    // double sec_convert = msec_convert / 1000.0;
    convert_time = msec_convert;

    // 测试这个routine 要我们测的 kernel_tag
    ell.kernel_flag = kernel_tag;

    if(0 == ell.kernel_flag){
        std::cout << "\n===  Compared ELL serial with csr default  ===" << std::endl;
        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         ell, LeSpMV_ell<IndexType, ValueType>,
                         "ell_serial_simple");

        std::cout << "\n===  Performance of ELL serial simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(ell,LeSpMV_ell<IndexType, ValueType>,"ell_serial_simple");
    }
    else if (1 == ell.kernel_flag)
    {
        std::cout << "\n===  Compared ELL omp with csr default  ===" << std::endl;
        
        // 设置 omp 调度策略
        const IndexType thread_num = Le_get_thread_num();
        
        IndexType chunk_size = OMP_ROWS_SIZE;
        if (ld == RowMajor)
            chunk_size = std::max((IndexType)chunk_size, ell.num_rows/thread_num);
        else
            chunk_size = std::max((IndexType)chunk_size, ell.num_cols/thread_num);

        set_omp_schedule(schedule_mod, chunk_size);

        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         ell, LeSpMV_ell<IndexType, ValueType>,
                         "ell_omp_simple");

        std::cout << "\n===  Performance of ELL omp simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(ell, LeSpMV_ell<IndexType, ValueType>,"ell_omp_simple");     
    }
    else if (2 == ell.kernel_flag)
    {
        std::cout << "\n===  Compared ELL Load-Balance with csr default  ===" << std::endl;

        // Pre- partition by numer of nnz per row balanced
        const IndexType thread_num = Le_get_thread_num();
        ell.partition = new_array<IndexType>(thread_num + 1);

        // balanced_partition_row_by_nnz_ell(ell.col_index, ell.num_nnzs, ell.num_rows, ell.max_row_width, thread_num, ell.partition);
        balanced_partition_row_by_nnz_ell_n2(ell.col_index, ell.num_nnzs, ell.num_rows, ell.max_row_width, thread_num, ell.partition);

        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         ell, LeSpMV_ell<IndexType, ValueType>,
                         "ell_omp_ld");

        std::cout << "\n===  Performance of ELL omp Load-Balance  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(ell,LeSpMV_ell<IndexType, ValueType>,"ell_omp_ld");
    }

    delete_ell_matrix(ell);
    return msec_per_iteration;
}

template double test_ell_matrix_kernels<int,float>(const CSR_Matrix<int,float> &csr_ref, int kernel_tag, LeadingDimension ld, int schedule_mod, double &convert_time);

template double test_ell_matrix_kernels<int,double>(const CSR_Matrix<int,double> &csr_ref, int kernel_tag, LeadingDimension ld, int schedule_mod, double &convert_time);

template double test_ell_matrix_kernels<long long,float>(const CSR_Matrix<long long,float> &csr_ref, int kernel_tag, LeadingDimension ld, int schedule_mod, double &convert_time);

template double test_ell_matrix_kernels<long long,double>(const CSR_Matrix<long long,double> &csr_ref, int kernel_tag, LeadingDimension ld, int schedule_mod, double &convert_time);