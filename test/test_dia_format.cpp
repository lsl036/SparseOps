/**
 * @file test_dia_format.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2023-11-23
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

    DIA_Matrix <int, float> dia_matrix;
    IndexType max_diags = MAX_DIAG_NUM;
    IndexType alignment = (SIMD_WIDTH/8/sizeof(ValueType));
    dia_matrix = read_dia_matrix<int, float>(mm_filename, max_diags, alignment);

    printf("Using %d-by-%d matrix with %d nonzero values\n", dia_matrix.num_rows, dia_matrix.num_cols, dia_matrix.num_nnzs);

    printf("dia stride = %d\n", dia_matrix.stride);
    printf("dia complete_ndiags = %d\n", dia_matrix.complete_ndiags);

    delete_host_matrix(dia_matrix);

}

int main(int argc, char** argv)
{
    test_martixfile<int,float>(argc, argv);

    return 0;
}