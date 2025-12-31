/**
 * @file test_bsr_format.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-02-21
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

    // reference CSR kernel for bsr format test
    CSR_Matrix<IndexType, ValueType> csr;
    csr = read_csr_matrix<IndexType, ValueType> (mm_filename);

    BSR_Matrix<IndexType, ValueType> bsr;
    
    bsr = csr_to_bsr<IndexType, ValueType>(csr, BSR_BlockDimRow, SIMD_WIDTH/8/sizeof(ValueType));

    std::cout << " block Row = " << bsr.blockDim_r << std::endl;
    std::cout << " block Col = " << bsr.blockDim_c << std::endl;
    std::cout << " mb = " << bsr.mb << std::endl;
    std::cout << " nb = " << bsr.nb << std::endl;
    std::cout << " nnz block = " << bsr.nnzb << std::endl;

    printf("BSR Row_ptr: \n");
    for (size_t i = 0; i < bsr.mb + 1; i++)
    {
        printf("%d ", bsr.row_ptr[i]);
    }
    printf("\n");
    

    delete_host_matrix(bsr);

}

int main(int argc, char** argv)
{
    test_martixfile<int, float>(argc, argv);
    // test_martixfile<int, double>(argc, argv);

    return 0;
}