/**
 * @file test_spmv_sell.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief  Test routine for spmv_s_ell.cpp
 * @version 0.1
 * @date 2023-12-12
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include"../include/LeSpMV.h"
#include<iostream>

template <typename IndexType, typename ValueType>
double test_s_ell_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, int schedule_mod, double &convert_time)
{
    double msec_per_iteration;
    std::cout << "=====  Testing S_ELL Kernels  =====" << std::endl;

    S_ELL_Matrix<IndexType,ValueType> sell;

    FILE* save_features = fopen(MAT_FEATURES,"w");
    IndexType alignment = (SIMD_WIDTH/8/sizeof(ValueType));
    IndexType chunk_width = CHUNK_SIZE;

    // formats convert overhead
    timer t;
    sell = csr_to_sell(csr_ref, save_features, chunk_width, alignment);
    double msec_convert = (double) t.milliseconds_elapsed();
    // double sec_convert = msec_convert / 1000.0;
    convert_time = msec_convert;
    
    fclose(save_features);

    // 测试这个routine 要我们测的 kernel_tag
    sell.kernel_flag = kernel_tag;

    if(0 == sell.kernel_flag){
        std::cout << "\n===  Compared S_ELL serial with csr default  ===" << std::endl;
        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         sell, LeSpMV_sell<IndexType, ValueType>,
                         "sell_serial_simple");

        std::cout << "\n===  Performance of S_ELL serial simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(sell, LeSpMV_sell<IndexType, ValueType>,"sell_serial_simple");
    }
    else if (1 == sell.kernel_flag)
    {
        std::cout << "\n===  Compared S_ELL omp with csr default  ===" << std::endl;
        // 设置 omp 调度策略
        const IndexType thread_num = Le_get_thread_num();
        const IndexType chunk_size = std::max((IndexType)1, sell.chunk_num/ thread_num);
        set_omp_schedule(schedule_mod, chunk_size);
        
        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         sell, LeSpMV_sell<IndexType, ValueType>,
                         "sell_omp_simple");

        std::cout << "\n===  Performance of S_ELL omp simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(sell, LeSpMV_sell<IndexType, ValueType>,"sell_omp_simple");     
    }
    else if (2 == sell.kernel_flag)
    {
        std::cout << "\n===  Compared S_ELL Load-Balance with csr default  ===" << std::endl;

        // Pre- partition by numer of nnz per row balanced
        const IndexType thread_num = Le_get_thread_num();
        sell.partition = new_array<IndexType>(thread_num + 1);

        balanced_partition_row_by_nnz_sell(sell.col_index, sell.num_nnzs, sell.sliceWidth, sell.chunk_num, sell.row_width, thread_num, sell.partition);

        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         sell, LeSpMV_sell<IndexType, ValueType>,
                         "sell_omp_ld");

        std::cout << "\n===  Performance of SELL omp Load-Balance  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(sell,LeSpMV_sell<IndexType, ValueType>,"sell_omp_ld");
    }

    delete_s_ell_matrix(sell);
    return msec_per_iteration;
}

template double test_s_ell_matrix_kernels<int,float>(const CSR_Matrix<int,float> &csr_ref, int kernel_tag, int sche, double &convert_time);

template double test_s_ell_matrix_kernels<int,double>(const CSR_Matrix<int,double> &csr_ref, int kernel_tag, int sche, double &convert_time);

template double test_s_ell_matrix_kernels<long long,float>(const CSR_Matrix<long long,float> &csr_ref, int kernel_tag, int sche, double &convert_time);

template double test_s_ell_matrix_kernels<long long,double>(const CSR_Matrix<long long,double> &csr_ref, int kernel_tag, int sche, double &convert_time);