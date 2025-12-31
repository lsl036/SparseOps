/**
 * @file test_csr5_format.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-12-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include<iostream>
#include<cstdio>
#include"../include/LeSpMV.h"
#include"../include/cmdline.h"

template <typename IndexType, typename UIndexType, typename ValueType>
void test_martixfile(int argc, char** argv)
{
    char * mm_filename = NULL;
    for(int i = 1; i < argc; i++){
        if(argv[i][0] != '-'){
            mm_filename = argv[i];
            break;
        }
    }

    if(mm_filename == NULL)
    {
        printf("You need to input a matrix file!\n");
        return;
    }

    // CSR5_Matrix <IndexType, UIndexType, ValueType> csr5;
    
    // csr5 = read_csr5_matrix<IndexType, UIndexType, ValueType>(mm_filename);

    // printf("Using %d-by-%d matrix with %d nonzero values\n", csr5.num_rows, csr5.num_cols, csr5.num_nnzs);

    // reference CSR kernel for csr5 test
    CSR_Matrix<IndexType, ValueType> csr;
    csr = read_csr_matrix<IndexType, ValueType> (mm_filename);

    CSR5_Matrix <IndexType, UIndexType, ValueType> csr5;
    FILE* save_features = fopen(MAT_FEATURES,"w");
    
    csr5 = csr_to_csr5<IndexType, UIndexType, ValueType>(csr, save_features);
    
    fclose(save_features);

    // printf("chunk_num   = %d\n", s_ell_matrix.chunk_num);
    // printf("chunk_width = %d\n", s_ell_matrix.sliceWidth);
    printf("csr5 sigma  = %d\n", csr5.sigma);
    printf("csr5 omega  = %d\n", csr5.omega);
    printf("\n");
    printf("csr5 bit_y_offset    = %d\n", csr5.bit_y_offset);
    printf("csr5 bit_seg_offset  = %d\n", csr5.bit_scansum_offset);
    printf("\n");
    printf("csr5 packets  = %d\n", csr5.num_packets);
    printf("csr5 _p  = %d\n", csr5._p);

    delete_host_matrix(csr5);

}

int main(int argc, char** argv)
{
    test_martixfile<int, uint32_t, double>(argc, argv);
    // test_martixfile<int, uint64_t, float>(argc, argv);

    return 0;
}