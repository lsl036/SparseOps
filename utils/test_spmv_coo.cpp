/**
 * @file test_spmv_coo.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief  Test routine for spmv_coo.cpp
 * @version 0.1
 * @date 2023-11-15
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include"../include/LeSpMV.h"
#include<iostream>

template <typename IndexType, typename ValueType>
double test_coo_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, int schedule_mod, double &convert_time)
{
    double msec_per_iteration;
    std::cout << "=====  Testing COO Kernels  =====" << std::endl;

    // coo_test 和 CSR的默认实现对比一下
    COO_Matrix<IndexType,ValueType> coo_test;
    
    // formats convert overhead
    timer t;

    coo_test = csr_to_coo(csr_ref);

    double msec_convert = (double) t.milliseconds_elapsed();
    // double sec_convert = msec_convert / 1000.0;
    convert_time = msec_convert;

    // 测试这个routine 要我们测的 kernel_tag
    coo_test.kernel_flag = kernel_tag;

    if( 0 == kernel_tag){
        std::cout << "\n===  Compared coo serial with csr default  ===" << std::endl;

        // test correctness
        test_spmv_kernel(csr_ref,  LeSpMV_csr<IndexType, ValueType>,
                         coo_test, LeSpMV_coo<IndexType, ValueType>,
                         "coo_serial_simple");

        std::cout << "\n===  Performance of COO serial simple ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(coo_test,LeSpMV_coo<IndexType, ValueType>,"coo_serial_simple");
    }
    else if(1 == kernel_tag){
        std::cout << "\n===  Compared coo omp with csr default  ===" << std::endl;

        // 设置 omp 调度策略
        const IndexType thread_num = Le_get_thread_num();
        
        // IndexType chunk_size = OMP_ROWS_SIZE;
        IndexType chunk_size = 512;
        chunk_size = std::max(chunk_size, coo_test.num_nnzs/thread_num);

        set_omp_schedule(schedule_mod, chunk_size);

        // test correctness
        test_spmv_kernel(csr_ref,  LeSpMV_csr<IndexType, ValueType>,
                         coo_test, LeSpMV_coo<IndexType, ValueType>,
                         "coo_omp_simple");

        std::cout << "\n===  Performance of COO omp simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(coo_test,LeSpMV_coo<IndexType, ValueType>,"coo_omp_simple");
    }
    else if(2 == kernel_tag){
        std::cout << "\n===  Compared coo alpha implementation with csr default ===" << std::endl;

        // test correctness
        test_spmv_kernel(csr_ref,  LeSpMV_csr<IndexType, ValueType>,
                         coo_test, LeSpMV_coo<IndexType, ValueType>,
                         "coo_omp_lb");

        std::cout << "\n===  Performance of coo alpha implementation  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host( coo_test, LeSpMV_coo<IndexType, ValueType>,"coo_omp_lb");
    }

    // *gflops = coo.gflops;
    // delete_coo_matrix(coo);
    delete_host_matrix(coo_test);
    // return 0;
    return msec_per_iteration;
}

template double test_coo_matrix_kernels<int,float>(const CSR_Matrix<int,float> &csr_ref, int kernel_tag, int schedule_mod, double &convert_time);

template double test_coo_matrix_kernels<int,double>(const CSR_Matrix<int,double> &csr_ref, int kernel_tag, int schedule_mod, double &convert_time);

template double test_coo_matrix_kernels<long long,float>(const CSR_Matrix<long long,float> &csr_ref, int kernel_tag, int schedule_mod, double &convert_time);

template double test_coo_matrix_kernels<long long,double>(const CSR_Matrix<long long,double> &csr_ref, int kernel_tag, int schedule_mod, double &convert_time);