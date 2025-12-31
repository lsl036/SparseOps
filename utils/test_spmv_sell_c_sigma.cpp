/**
 * @file test_spmv_sell_c_sigma.cpp
 * @author your name (you@domain.com)
 * @brief Test routine for spmv_sell_c_sigma.cpp
 * @version 0.1
 * @date 2024-01-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include"../include/LeSpMV.h"
#include<iostream>

template <typename IndexType, typename ValueType>
double test_sell_c_sigma_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, int schedule_mod, double &convert_time)
{
    double msec_per_iteration;
    std::cout << "=====  Testing SELL-c-sigma Kernels  =====" << std::endl;

    SELL_C_Sigma_Matrix<IndexType,ValueType> sell_c_sigma;

    FILE* save_features = fopen(MAT_FEATURES,"w");

    IndexType slicewidth = SELL_SIGMA;
    IndexType chunkwidth = CHUNK_SIZE;
    IndexType alignment  = SIMD_WIDTH/8/sizeof(ValueType);

    // formats convert overhead
    timer t;
    sell_c_sigma = csr_to_sell_c_sigma(csr_ref, save_features, slicewidth, chunkwidth, alignment);
    double msec_convert = (double) t.milliseconds_elapsed();
    // double sec_convert = msec_convert / 1000.0;
    convert_time = msec_convert;

    fclose(save_features);

    // 测试这个routine 要我们测的 kernel_tag
    sell_c_sigma.kernel_flag = kernel_tag;

    if ( 0 == sell_c_sigma.kernel_flag){
        std::cout << "\n===  Compared SELL-c-sigma serial with csr default  ===" << std::endl;
        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         sell_c_sigma, LeSpMV_sell_c_sigma<IndexType, ValueType>,
                         "sell_c_sigma_serial_simple");

        std::cout << "\n===  Performance of SELL-c-sigma serial simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(sell_c_sigma, LeSpMV_sell_c_sigma<IndexType, ValueType>,"sell_c_sigma_serial_simple");

    }
    else if ( 1 == sell_c_sigma.kernel_flag)
    {
        std::cout << "\n===  Compared SELL-c-sigma omp with csr default  ===" << std::endl;
        // 设置 omp 调度策略
        const IndexType thread_num = Le_get_thread_num();
        const IndexType chunk_size = std::max((IndexType)1, sell_c_sigma.validchunkNum/thread_num);
        set_omp_schedule(schedule_mod, chunk_size);

        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         sell_c_sigma, LeSpMV_sell_c_sigma<IndexType, ValueType>,
                         "sell_c_sigma_omp_simple");

        std::cout << "\n===  Performance of SELL-c-sigma omp simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(sell_c_sigma, LeSpMV_sell_c_sigma<IndexType, ValueType>,"sell_c_sigma_omp_simple");
        
    }
    else if ( 2 == sell_c_sigma.kernel_flag)
    {
        // just the same as omp simple now
        std::cout << "\n===  Compared SELL-c-sigma Load-Balance with csr default  ===" << std::endl;

        // Pre- partition by numer of nnz per row balanced
        const IndexType thread_num = Le_get_thread_num();
        sell_c_sigma.partition = new_array<IndexType>(thread_num + 1);

        balanced_partition_row_by_nnz_sell(sell_c_sigma.col_index, sell_c_sigma.num_nnzs, sell_c_sigma.chunkWidth_C, sell_c_sigma.validchunkNum, sell_c_sigma.chunk_len, thread_num, sell_c_sigma.partition);

        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         sell_c_sigma, LeSpMV_sell_c_sigma<IndexType, ValueType>,
                         "sell_c_sigma_omp_ld");

        std::cout << "\n===  Performance of SELL-c-sigma Load-Balance  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(sell_c_sigma, LeSpMV_sell_c_sigma<IndexType, ValueType>,"sell_c_sigma_omp_ld");
    }

    delete_host_matrix(sell_c_sigma);
    return msec_per_iteration;
}

template double test_sell_c_sigma_matrix_kernels<int,float>(const CSR_Matrix<int,float> &csr_ref, int kernel_tag, int sche, double &convert_time);

template double test_sell_c_sigma_matrix_kernels<int,double>(const CSR_Matrix<int,double> &csr_ref, int kernel_tag, int sche, double &convert_time);

template double test_sell_c_sigma_matrix_kernels<long long,float>(const CSR_Matrix<long long,float> &csr_ref, int kernel_tag, int sche, double &convert_time);

template double test_sell_c_sigma_matrix_kernels<long long,double>(const CSR_Matrix<long long,double> &csr_ref, int kernel_tag, int sche, double &convert_time);