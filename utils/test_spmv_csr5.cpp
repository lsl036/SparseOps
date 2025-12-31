/**
 * @file test_spmv_csr5.cpp
 * @author your name (you@domain.com)
 * @brief Test routine for spmv_csrs.cpp
 * @version 0.1
 * @date 2024-01-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include"../include/LeSpMV.h"
#include<iostream>


// CSR5 只有一种实现方式，且 schedule mod 也不需要人为指定
template <typename IndexType, typename UIndexType, typename ValueType>
double test_csr5_matrix_kernels(const CSR_Matrix<IndexType, ValueType> &csr_ref, int kernel_tag, int schedule_mod)
{
    double msec_per_iteration;
    std::cout << "=====  Testing CSR5 Kernels  =====" << std::endl;

    CSR5_Matrix<IndexType, UIndexType, ValueType> csr5;
    
    FILE* save_features = fopen(MAT_FEATURES,"w");
    // printf("11111111\n");
    csr5 = csr_to_csr5<IndexType, UIndexType, ValueType>(csr_ref, save_features);
    // printf("22222222\n");
    fclose(save_features);

    // 只有一种kernel 不需要分类啦
    // csr5.kernel_flag = kernel_tag;

    {
        std::cout << "\n===  Compared csr5 with csr default  ===" << std::endl;
        // test correctness
        test_spmv_kernel(csr_ref, LeSpMV_csr<IndexType, ValueType>,
                        csr5, LeSpMV_csr5<IndexType, UIndexType, ValueType>,
                        "csr5_AVX512");

        std::cout << "\n===  Performance of csr5 (AVX512)  ===" << std::endl;

        // count performance of Gflops and Gbytes
        msec_per_iteration = benchmark_spmv_on_host(csr5, LeSpMV_csr5<IndexType, UIndexType, ValueType>, "csr5_AVX512");
    }
    
    delete_csr5_matrix(csr5);
    // return 0;
    return msec_per_iteration;
}

// template int test_csr5_matrix_kernels<int, uint32_t, float>(const CSR_Matrix<int,float> &csr_ref, int kernel_tag, int schedule_mod);

// AVX512 只 制作了 double 精度的计算
template double test_csr5_matrix_kernels<int, uint32_t, double>(const CSR_Matrix<int,double> &csr_ref, int kernel_tag, int schedule_mod);