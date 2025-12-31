/**
 * @file test_sell_c_sigma_format.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-01-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include<iostream>
#include<cstdio>
#include"../include/LeSpMV.h"
#include"../include/cmdline.h"

template <typename IndexType, typename ValueType>
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

    SELL_C_Sigma_Matrix <int, float> s_ell_c_sigma;
    
    IndexType slicewidth = SELL_SIGMA;
    IndexType chunkwidth = CHUNK_SIZE;
    IndexType alignment  = SIMD_WIDTH/8/sizeof(ValueType);
    
    s_ell_c_sigma = read_sell_c_sigma_matrix<IndexType, ValueType> (mm_filename, slicewidth, chunkwidth, alignment);

    printf("Using %d-by-%d matrix with %d nonzero values\n", s_ell_c_sigma.num_rows, s_ell_c_sigma.num_cols, s_ell_c_sigma.num_nnzs);

    printf("SliceNum   = %d\n", s_ell_c_sigma.sliceNum);
    printf("ChunkNum   = %d\n", s_ell_c_sigma.chunkNum);
    printf("ValidChunkNum   = %d\n", s_ell_c_sigma.validchunkNum);

    printf("sigma      = %d\n", s_ell_c_sigma.sliceWidth_Sigma);
    printf("chunk      = %d\n", s_ell_c_sigma.chunkWidth_C);
    printf("c_per_slice      = %d\n", s_ell_c_sigma.chunk_num_per_slice);
    printf("alignment        = %d\n", s_ell_c_sigma.alignment);

    printf("Reorder        = \n");
    for (IndexType i = 0; i < s_ell_c_sigma.sliceWidth_Sigma; i++)
    {
        if ( i < s_ell_c_sigma.num_rows)
        {
        printf("%d ", s_ell_c_sigma.reorder[i]);
        }
    }
    printf("\n");

    delete_host_matrix(s_ell_c_sigma);
}

int main(int argc, char** argv)
{
    test_martixfile<int,float>(argc, argv);

    return 0;
}