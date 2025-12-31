/**
 * @file test_features.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-01-23
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include<iostream>
#include<cstdio>
#include"../include/LeSpMV.h"
#include"../include/cmdline.h"

void usage(int argc, char** argv)
{
    std::cout << "Usage:\n";
    std::cout << "\t" << argv[0] << " with following parameters:\n";
    std::cout << "\t" << " my_matrix.mtx\n";
    std::cout << "\t" << " --matID     = m_num, giving the matrix ID number in dataset (default 0).\n";
    std::cout << "\t" << " --Index     = 0 (int) or 1 (long long) (default 1).\n";
    std::cout << "\t" << " --precision = 64(or 32), for counting features (default 64).\n";
    std::cout << "\t" << " --threads   = t_num, define the number of omp threads.\n";
    std::cout << "Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"; 
}

template <typename IndexType, typename ValueType>
void test_features(int argc, char** argv)
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
        printf("You need to input a matrix file! see '--help' for more details\n");
        return;
    }

    int matID = 0;
    char * matID_str = get_argval(argc, argv, "matID");
    if(matID_str != NULL)
    {
        matID = atoi(matID_str);
    }

    MTX<IndexType, ValueType> mtx(matID);
    mtx.MtxLoad(mm_filename);

    timer t;
    mtx.CalculateFeatures();
    
    // if (mtx.getRowNum() >= mtx.getTileSize() && mtx.getColNum() >= mtx.getTileSize()){
    mtx.CalculateTilesExtraFeatures(mm_filename);

    double msec_fea_overhead = (double) t.milliseconds_elapsed();

    FILE *save_overhead = fopen(MAT_FEATURES, "a");
    if ( save_overhead == nullptr)
    {
        std::cout << "Unable to open perf-saved file: "<< MAT_FEATURES << std::endl;
        return ;
    }
    fprintf(save_overhead, "%d %s %5.4f \n", matID, mtx.getMatName().c_str(), msec_fea_overhead);

    mtx.FeaturesPrint();
    mtx.ExtraFeaturesPrint();

    // }
    

    // mtx.FeaturesWrite(MAT_FEATURES);
}

int main(int argc, char** argv) {
    if (get_arg(argc, argv, "help") != NULL){
        usage(argc, argv);
        return EXIT_SUCCESS;
    }

    // int precision = 64;
    int precision = 32;
    char * precision_str = get_argval(argc, argv, "precision");
    if(precision_str != NULL)
        precision = atoi(precision_str);
    
    // int Index = 1;
    int Index = 0;
    char * Index_str = get_argval(argc, argv, "Index");
    if(Index_str != NULL)
        Index = atoi(Index_str);

    // 包括超线程
    Le_set_thread_num(CPU_SOCKET * CPU_CORES_PER_SOC * CPU_HYPER_THREAD);
    char * threads_str = get_argval(argc, argv, "threads");
    if(threads_str != NULL)
        Le_set_thread_num(atoi(threads_str));

    // if(precision ==  32){
    //     test_features<int, float>(argc, argv);
    // }
    // else if(precision == 64){
    //     test_features<int, double>(argc, argv);
    // }
    if (Index == 0 && precision ==  32){
        test_features<int, float>(argc, argv);
    }
    else if (Index == 0 && precision == 64){
        test_features<int, double>(argc, argv);
    }
    else if (Index == 1 && precision ==  32){
        test_features<long long, float>(argc, argv);
    }
    else if (Index == 1 && precision == 64){
        test_features<long long, double>(argc, argv);
    }
    else{
        usage(argc, argv);
        return EXIT_FAILURE;
    }

    return 0;
}