#ifdef ENABLE_MKL
#include "mkl.h"
#endif
#include<iostream>
#include<cstdio>
#include"../include/SpOps.h"
#include"../include/cmdline.h"

void usage(int argc, char** argv)
{
    std::cout << "Usage:\n";
    std::cout << "\t" << argv[0] << " with following parameters:\n";
    std::cout << "\t" << " my_matrix.mtx\n";
    std::cout << "\t" << " --matID     = m_num, giving the matrix ID number in dataset (default 0).\n";
    std::cout << "\t" << " --Index     = 0 (int)\n";
    std::cout << "\t" << " --precision = 64 for mkl_dcsrmv\n";
    std::cout << "\t" << " --sche      = chosing the schedule strategy\n";
    std::cout << "\t" << "               0: static | 1: static, CHUNK_SIZE | 2: dynamic | 3: guided\n";
    std::cout << "\t" << " --threads   = define the num of omp threads\n";
    std::cout << "Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"; 
}

#ifdef ENABLE_MKL
void mklcsr_baseline(int argc, char **argv)
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

    int matID = 0;
    char * matID_str = get_argval(argc, argv, "matID");
    if(matID_str != NULL)
    {
        matID = atoi(matID_str);
    }

    CSR_Matrix<MKL_INT, double> csr;
    csr = read_csr_matrix<MKL_INT, double> (mm_filename);

    double * x_host = new_array<double>(csr.num_cols);
    double * y_host = new_array<double>(csr.num_rows);
    for(MKL_INT i = 0; i < csr.num_cols; i++)
        x_host[i] = rand() / (RAND_MAX + 1.0); 
    std::fill(y_host, y_host + csr.num_rows, 0);

    // 调用 MKL 的 SpMV 函数
    char transa = 'N'; // 不转置
    double alpha = 1.0;
    double beta = 0.0;

    timer time_one_iteration;
    // warmup
    mkl_dcsrmv(&transa, &csr.num_rows, &csr.num_cols, &alpha, "G**C", csr.values, csr.col_index, csr.row_offset, csr.row_offset + 1, x_host, &beta, y_host);
    double estimated_time = time_one_iteration.milliseconds_elapsed();

    // determine # of seconds dynamically
    int num_iterations;
    num_iterations = MAX_ITER;
    if (estimated_time < 20) // less than 20 ms, so it can tolerate 20s for each SpMV
        num_iterations = MAX_ITER;
    else
        num_iterations = std::min(MAX_ITER, std::max(MIN_ITER, (int) (TIME_LIMIT*1000 / estimated_time)) ); 
    printf("\tPerforming %d iterations\n", num_iterations);

    // time several SpMV iterations
    timer t;
    for(int i = 0; i < num_iterations; i++)
       mkl_dcsrmv(&transa, &csr.num_rows, &csr.num_cols, &alpha, "G**C", csr.values, csr.col_index, csr.row_offset, csr.row_offset + 1, x_host, &beta, y_host); // alpha = 1, beta = 0;

    double msec_per_iteration = t.milliseconds_elapsed() / (double) num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;

    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) csr.num_nnzs / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_spmv(csr) / sec_per_iteration) / 1e9;
    
    csr.gflops = GFLOPs;
    csr.gbytes = GBYTEs;
    csr.time = msec_per_iteration;

    const char * location = "cpu" ;
    printf("\tbenchmarking %-20s [%s]: %8.4f ms ( %5.4f GFLOP/s %5.4f GB/s)\n", \
            "mkl_dcsr_spmv", location, msec_per_iteration, GFLOPs, GBYTEs);

    // 保存测试性能结果
    FILE *save_perf = fopen(MAT_PERFORMANCE, "a");
    if ( save_perf == nullptr)
    {
        std::cout << "Unable to open perf-saved file: "<< MAT_PERFORMANCE << std::endl;
        return ;
    }
    fprintf(save_perf, "%d %s MKL_dcsr %8.4f %5.4f \n", matID, matrixName.c_str(),  msec_per_iteration, GFLOPs);

    fclose(save_perf);
    delete_csr_matrix(csr);
}
#else
void mklcsr_baseline(int argc, char **argv)
{
    std::cout << "Error: MKL support is not enabled. Please configure with -DENABLE_MKL=ON" << std::endl;
    usage(argc, argv);
}
#endif

int main(int argc, char** argv)
{
    if (get_arg(argc, argv, "help") != NULL){
        usage(argc, argv);
        return EXIT_SUCCESS;
    }

    int precision = 64;
    char * precision_str = get_argval(argc, argv, "precision");
    if(precision_str != NULL)
        precision = atoi(precision_str);

    int Index = 0;
    char * Index_str = get_argval(argc, argv, "Index");
    if(Index_str != NULL)
        Index = atoi(Index_str);

    printf("\nUsing %d-bit floating point precision, %d-bit Index, MKL BASELINE\n\n", precision, (Index+1)*32);


    if (Index == 0 && precision == 64){
        mklcsr_baseline(argc,argv);
    }
    else{
        usage(argc, argv);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}