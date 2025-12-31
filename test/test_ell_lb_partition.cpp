/**
 * @file test_ell_lb_partition.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief   Test and dubug balanced_partition_row_by_nnz_ell function
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

int main(int argc, char** argv)
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
        return -1;
    }

    ELL_Matrix <int, float> ell_matrix;

    ell_matrix = read_ell_matrix<int, float>(mm_filename);

    printf("Using %d-by-%d matrix with %d nonzero values\n", ell_matrix.num_rows, ell_matrix.num_cols, ell_matrix.num_nnzs); 

    // 不用超线程，只计算真实CORE
    Le_set_thread_num(CPU_SOCKET * CPU_CORES_PER_SOC);

    char * threads_str = get_argval(argc, argv, "threads");
    if(threads_str != NULL)
        Le_set_thread_num(atoi(threads_str));
    
    const int thread_num= Le_get_thread_num();
    int ave = ell_matrix.num_nnzs / thread_num;

    printf("\nUsing 32-bit floating point precision, threads = %d, AVE = %d\n\n", thread_num, ave);

    
    int partition[thread_num + 1];  // index 0 ~ thread_num

    balanced_partition_row_by_nnz_ell(ell_matrix.col_index, ell_matrix.num_nnzs, ell_matrix.num_rows, ell_matrix.max_row_width, thread_num, partition);

    printf("Thread Partition ELL: \n");
    for (size_t i = 0; i < thread_num + 1; i++)
    {
        printf(" %d", partition[i]);
    }

    printf("\n\n");

    delete_ell_matrix(ell_matrix);

    CSR_Matrix<int, float> csr;
    csr = read_csr_matrix<int, float> (mm_filename);

    int partition_csr[thread_num + 1];

    balanced_partition_row_by_nnz(csr.row_offset, csr.num_rows, thread_num, partition_csr);

    printf("Thread Partition CSR: \n");
    for (size_t i = 0; i < thread_num + 1; i++)
    {
        printf(" %d", partition_csr[i]);
    }

    printf("\n");
    delete_csr_matrix(csr);

    return 0;
}
