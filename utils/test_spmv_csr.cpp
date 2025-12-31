/**
 * @file test_spmv_csr.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Test routine for spmv_csr.cpp
 * @version 0.1
 * @date 2023-11-16
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include"../include/LeSpMV.h"
#include<iostream>

template <typename IndexType, typename ValueType>
double test_csr_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, int schedule_mod)
{
    double msec_per_iteration;
    std::cout << "=====  Testing CSR Kernels  =====" << std::endl;

    // csr_test 测试所有kernel， csr_ref 为 omp simple实现
    CSR_Matrix<IndexType,ValueType> csr_test;
    csr_test.num_rows = csr_ref.num_rows;
    csr_test.num_cols = csr_ref.num_cols;
    csr_test.num_nnzs = csr_ref.num_nnzs;
    csr_test.row_offset = copy_array(csr_ref.row_offset, csr_ref.num_rows+1);
    csr_test.col_index  = copy_array(csr_ref.col_index, csr_test.num_nnzs);
    csr_test.values     = copy_array(csr_ref.values, csr_test.num_nnzs);
    // csr_test.partition  = copy_array(csr_ref.partition, Le_get_thread_num()+1);
    csr_test.kernel_flag = 0;

    // 测试这个routine 要我们测的 kernel_tag
    csr_test.kernel_flag = kernel_tag;

    if(0 == kernel_tag){
        std::cout << "\n===  Compared csr serial with csr default  ===" << std::endl;
        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         csr_test, LeSpMV_csr<IndexType, ValueType>,
                         "csr_serial_simple");

        std::cout << "\n===  Performance of CSR serial simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(csr_test, LeSpMV_csr<IndexType, ValueType>, "csr_serial_simple");
    }
    else if(1 == kernel_tag){
        std::cout << "\n===  Compared csr omp with csr default  ===" << std::endl;

        // 设置 omp 调度策略
        const IndexType thread_num = Le_get_thread_num();
        
        IndexType chunk_size = OMP_ROWS_SIZE;
        chunk_size = std::max(chunk_size, csr_test.num_rows/thread_num);

        set_omp_schedule(schedule_mod, chunk_size);

        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         csr_test, LeSpMV_csr<IndexType, ValueType>,
                         "csr_omp_simple");

        std::cout << "\n===  Performance of CSR omp simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(csr_test, LeSpMV_csr<IndexType, ValueType>, "csr_omp_simple");
    }
    else if(2 == kernel_tag){
        std::cout << "\n===  Compared csr_lb_nnz with csr default  ===" << std::endl;

        // Pre- partition by numer of nnz per row balanced
        const IndexType thread_num = Le_get_thread_num();
        csr_test.partition = new_array<IndexType>(thread_num + 1);

        balanced_partition_row_by_nnz(csr_test.row_offset, csr_test.num_rows, thread_num, csr_test.partition);

        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         csr_test, LeSpMV_csr<IndexType, ValueType>,
                         "csr_omp_lb_nnz");

        std::cout << "\n===  Performance of CSR_lb_nnz  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(csr_test, LeSpMV_csr<IndexType, ValueType>, "csr_omp_lb_nnz");
    
    }

    delete_csr_matrix(csr_test);
    return msec_per_iteration;
}

template double test_csr_matrix_kernels<int,float>(const CSR_Matrix<int,float> &csr_ref, int kernel_tag, int schedule_mod);

template double test_csr_matrix_kernels<int,double>(const CSR_Matrix<int,double> &csr_ref, int kernel_tag, int schedule_mod);

template double test_csr_matrix_kernels<long long,float>(const CSR_Matrix<long long,float> &csr_ref, int kernel_tag, int schedule_mod);

template double test_csr_matrix_kernels<long long,double>(const CSR_Matrix<long long,double> &csr_ref, int kernel_tag, int schedule_mod);