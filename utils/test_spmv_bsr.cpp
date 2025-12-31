/**
 * @file test_spmv_bsr.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Test routine for spmv_bsr.cpp
 * @version 0.1
 * @date 2024-02-21
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include"../include/LeSpMV.h"
#include<iostream>

template <typename IndexType, typename ValueType>
double test_bsr_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, int schedule_mod, double &convert_time)
{
    double msec_per_iteration;
    std::cout << "=====  Testing BSR Kernels  =====" << std::endl;

    BSR_Matrix<IndexType,ValueType> bsr;

    IndexType alignment = (SIMD_WIDTH/8/sizeof(ValueType));
    IndexType bsr_rowdim = BSR_BlockDimRow;
    IndexType bsr_coldim = 1*alignment;
    // formats convert overhead
    timer t;

    bsr = csr_to_bsr(csr_ref, bsr_rowdim, bsr_coldim);

    double msec_convert = (double) t.milliseconds_elapsed();
    // double sec_convert = msec_convert / 1000.0;
    convert_time = msec_convert;
    
    // 测试这个routine 要我们测的 kernel_tag
    bsr.kernel_flag = kernel_tag;

    if(0 == bsr.kernel_flag){
        std::cout << "\n===  Compared BSR serial with csr default  ===" << std::endl;
        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         bsr, LeSpMV_bsr<IndexType, ValueType>,
                         "bsr_serial_simple");

        std::cout << "\n===  Performance of BSR serial simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(bsr, LeSpMV_bsr<IndexType, ValueType>,"bsr_serial_simple");
    }
    else if (1 == bsr.kernel_flag)
    {
        std::cout << "\n===  Compared BSR omp with csr default  ===" << std::endl;

        // 设置 omp 调度策略
        const IndexType thread_num = Le_get_thread_num();
        
        // IndexType chunk_size = OMP_ROWS_SIZE;
        IndexType chunk_size = 1; // BSR 调度的最小单位是一个block，行block数目为: mb
        chunk_size = std::max(chunk_size, bsr.mb/thread_num);

        set_omp_schedule(schedule_mod, chunk_size);

        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         bsr, LeSpMV_bsr<IndexType, ValueType>,
                         "bsr_omp_simple");

        std::cout << "\n===  Performance of BSR omp simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(bsr, LeSpMV_bsr<IndexType, ValueType>,"bsr_omp_simple");
    }
    else if(2 == kernel_tag){
        std::cout << "\n===  Compared bsr_lb_nnz with csr default  ===" << std::endl;

        // Pre- partition by numer of nnz per row balanced
        const IndexType thread_num = Le_get_thread_num();
        bsr.partition = new_array<IndexType>(thread_num + 1);

        balanced_partition_row_by_nnz(bsr.row_ptr, bsr.mb, thread_num, bsr.partition);

        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         bsr, LeSpMV_bsr<IndexType, ValueType>,
                         "bsr_omp_lb_nnz");

        std::cout << "\n===  Performance of BSR_lb_nnz  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(bsr, LeSpMV_bsr<IndexType, ValueType>, "bsr_omp_lb_nnz");
    }

    delete_host_matrix(bsr);
    return msec_per_iteration;
}

template double test_bsr_matrix_kernels<int,float>(const CSR_Matrix<int,float> &csr_ref, int kernel_tag, int sche,  double &convert_time);

template double test_bsr_matrix_kernels<int,double>(const CSR_Matrix<int,double> &csr_ref, int kernel_tag, int sche,  double &convert_time);

template double test_bsr_matrix_kernels<long long,float>(const CSR_Matrix<long long,float> &csr_ref, int kernel_tag, int sche,  double &convert_time);

template double test_bsr_matrix_kernels<long long,double>(const CSR_Matrix<long long,double> &csr_ref, int kernel_tag, int sche,  double &convert_time);