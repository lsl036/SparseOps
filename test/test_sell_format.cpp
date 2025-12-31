/**
 * @file test_sell_format.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2023-12-08
 * 
 * @copyright Copyright (c) 2023
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

    S_ELL_Matrix <int, float> s_ell_matrix;
    // IndexType max_diags = MAX_DIAG_NUM;
    IndexType alignment = (SIMD_WIDTH/8/sizeof(ValueType));
    IndexType chunk_width = CHUNK_SIZE;
    
    s_ell_matrix = read_sell_matrix<int, float>(mm_filename, chunk_width, alignment);

    printf("Using %d-by-%d matrix with %d nonzero values\n", s_ell_matrix.num_rows, s_ell_matrix.num_cols, s_ell_matrix.num_nnzs);

    printf("chunk_num   = %d\n", s_ell_matrix.chunk_num);
    printf("chunk_width = %d\n", s_ell_matrix.sliceWidth);

    delete_host_matrix(s_ell_matrix);

}

int main(int argc, char** argv)
{
    test_martixfile<int,float>(argc, argv);

    return 0;
}