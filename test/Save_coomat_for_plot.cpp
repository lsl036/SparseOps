#include<iostream>
#include<cstdio>
#include"../include/LeSpMV.h"
#include"../include/cmdline.h"

void usage(int argc, char** argv)
{
    std::cout << "Usage:\n";
    std::cout << "\t" << argv[0] << " with following parameters:\n";
    std::cout << "\t" << " my_matrix.mtx\n";
    std::cout << "\t" << " --threads   = t_num, define the number of omp threads.\n";
    std::cout << "Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"; 
}

template <typename IndexType, typename ValueType>
void SaveMat(int argc, char** argv)
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

    std::string matrixName = extractFileNameWithoutExtension(mm_filename);

    std::string MatFile = matrixName + ".mat";

    COO_Matrix<IndexType, ValueType> coo;

    coo = read_coo_matrix<IndexType, ValueType> (mm_filename);

    FILE *saveMat = fopen(MatFile.c_str(), "a");
    if (saveMat == nullptr)
    {
        std::cout << "Unable to open perf-saved file: "<< MatFile << std::endl;
        return ;
    }
    
    fprintf( saveMat, "%d %d %d\n\n", coo.num_rows, coo.num_cols, coo.num_nnzs);

    for (int i = 0; i < coo.num_nnzs; i++)
    {
        fprintf( saveMat, "%d %d %lf\n", coo.row_index[i], coo.col_index[i], coo.values[i]);
    }
    
    fclose(saveMat);
    delete_host_matrix(coo);

}

int main(int argc, char** argv) {
    if (get_arg(argc, argv, "help") != NULL){
        usage(argc, argv);
        return EXIT_SUCCESS;
    }

    // 包括超线程
    Le_set_thread_num(CPU_SOCKET * CPU_CORES_PER_SOC * CPU_HYPER_THREAD);
    char * threads_str = get_argval(argc, argv, "threads");
    if(threads_str != NULL)
        Le_set_thread_num(atoi(threads_str));
    
    SaveMat<int, double>(argc, argv);

}