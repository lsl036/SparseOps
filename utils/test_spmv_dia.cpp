/**
 * @file test_spmv_dia.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief   Test routine for spmv_dia.cpp
 * @version 0.1
 * @date 2023-11-23
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include"../include/LeSpMV.h"
#include<iostream>

template <typename IndexType, typename ValueType>
double test_dia_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, int schedule_mod, double &convert_time)
{
    double msec_per_iteration;
    std::cout << "=====  Testing DIA Kernels  =====" << std::endl;

    DIA_Matrix<IndexType,ValueType> dia;

    IndexType max_diags = MAX_DIAG_NUM;
    IndexType alignment = (SIMD_WIDTH/8/sizeof(ValueType));
    FILE* save_features = fopen(MAT_FEATURES,"w");
    
    // formats convert overhead
    timer t;
    
    dia = csr_to_dia(csr_ref, max_diags, save_features, alignment);
    
    double msec_convert = (double) t.milliseconds_elapsed();
    // double sec_convert = msec_convert / 1000.0;
    convert_time = msec_convert;

    fclose(save_features);
    // 测试这个routine 要我们测的 kernel_tag
    dia.kernel_flag = kernel_tag;

    if(0 == dia.kernel_flag){
        std::cout << "\n===  Compared DIA serial with csr default  ===" << std::endl;
        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         dia, LeSpMV_dia<IndexType, ValueType>,
                         "dia_serial_simple");

        std::cout << "\n===  Performance of DIA serial simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(dia, LeSpMV_dia<IndexType, ValueType>,"dia_serial_simple");
    }
    else if (1 == dia.kernel_flag)
    {
        std::cout << "\n===  Compared DIA omp with csr default  ===" << std::endl;

        // 设置 omp 调度策略
        const IndexType thread_num = Le_get_thread_num();
        
        // IndexType chunk_size = OMP_ROWS_SIZE;
        const IndexType chunk_size = std::max((IndexType)1, dia.complete_ndiags/thread_num); // 对角线数目 除以线程数

        set_omp_schedule(schedule_mod, chunk_size);

        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         dia, LeSpMV_dia<IndexType, ValueType>,
                         "dia_omp_simple");

        std::cout << "\n===  Performance of DIA omp simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(dia, LeSpMV_dia<IndexType, ValueType>,"dia_omp_simple");
          
    }
    else if (2 == dia.kernel_flag)
    {
        std::cout << "\n===  Compared DIA alpha with csr default  ===" << std::endl;

        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                         dia, LeSpMV_dia<IndexType, ValueType>,
                         "dia_alphasparse");

        std::cout << "\n===  Performance of DIA omp simple  ===" << std::endl;
        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(dia, LeSpMV_dia<IndexType, ValueType>,"dia_alphasparse");

    }

    delete_dia_matrix(dia);
    return msec_per_iteration;
}

template double test_dia_matrix_kernels<int,float>(const CSR_Matrix<int,float> &csr_ref, int kernel_tag, int schedule_mod, double &convert_time);

template double test_dia_matrix_kernels<int,double>(const CSR_Matrix<int,double> &csr_ref, int kernel_tag, int schedule_mod, double &convert_time);

template double test_dia_matrix_kernels<long long,float>(const CSR_Matrix<long long,float> &csr_ref, int kernel_tag, int schedule_mod, double &convert_time);

template double test_dia_matrix_kernels<long long,double>(const CSR_Matrix<long long,double> &csr_ref, int kernel_tag, int schedule_mod, double &convert_time);