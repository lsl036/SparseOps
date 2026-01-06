/**
 * @file test_spgemm.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Test program for SpGEMM correctness and performance
 * @version 0.1
 * @date 2026
 */

#include "../include/SpOps.h"
#include "../include/spgemm.h"
#include "../include/timer.h"
#include "../include/cmdline.h"
#include "../include/sparse_io.h"
#include "../include/mmio.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <ctime>

using namespace std;

void usage(int argc, char** argv)
{
    std::cout << "Usage:\n";
    std::cout << "\t" << argv[0] << " <matrix_A.mtx> <matrix_B.mtx> [options]\n";
    std::cout << "\t" << "Options:\n";
    std::cout << "\t" << " --precision = 32 (float) or 64 (double:default)\n";
    std::cout << "\t" << " --threads   = define the num of omp threads (default: all cores)\n";
    std::cout << "\t" << " --test_type = correctness (default) or performance\n";
    std::cout << "\t" << " --iterations= number of iterations for performance test (default: 10)\n";
    std::cout << "\t" << " --kernel    = 1 (OpenMP:default) or 2 (OpenMP with load balancing)\n";
    std::cout << "\t" << " --sort      = 0 (no sort:default) or 1 (sort output columns)\n";
    std::cout << "Note: Matrix files must be real-valued sparse matrices in the MatrixMarket file format.\n";
}

template <typename IndexType, typename ValueType>
void test_spgemm_correctness(const char *matA_path, const char *matB_path, int kernel_flag = 1, bool sortOutput = false)
{
    cout << "========================================" << endl;
    cout << "Testing SpGEMM Correctness" << endl;
    cout << "========================================" << endl;
    
    // Read matrices
    cout << "Reading matrix A from: " << matA_path << endl;
    CSR_Matrix<IndexType, ValueType> A = read_csr_matrix<IndexType, ValueType>(matA_path);
    cout << "A: " << A.num_rows << " x " << A.num_cols << ", nnz: " << A.num_nnzs << endl;
    
    cout << "Reading matrix B from: " << matB_path << endl;
    CSR_Matrix<IndexType, ValueType> B = read_csr_matrix<IndexType, ValueType>(matB_path);
    cout << "B: " << B.num_rows << " x " << B.num_cols << ", nnz: " << B.num_nnzs << endl;
    
    if (A.num_cols != B.num_rows) {
        cerr << "Error: A.num_cols (" << A.num_cols << ") != B.num_rows (" << B.num_rows << ")" << endl;
        delete_host_matrix(A);
        delete_host_matrix(B);
        return;
    }
    
    cout << "\n--- Testing Kernel " << kernel_flag << " ---" << endl;
    
    CSR_Matrix<IndexType, ValueType> C;
    anonymouslib_timer timer;
    
    timer.start();
    LeSpGEMM(A, B, C, sortOutput, kernel_flag);
    double time = timer.stop();
    
    cout << "C: " << C.num_rows << " x " << C.num_cols << ", nnz: " << C.num_nnzs << endl;
    cout << "Time: " << time << " ms" << endl;
    
    // Calculate performance
    long long int flops = get_spgemm_flop(A, B);
    double gflops = (flops / 1e9) / (time / 1000.0);
    cout << "Performance: " << gflops << " GFLOPS" << endl;
    
    // Verify basic properties
    bool valid = true;
    if (C.num_rows != A.num_rows) {
        cerr << "Error: C.num_rows != A.num_rows" << endl;
        valid = false;
    }
    if (C.num_cols != B.num_cols) {
        cerr << "Error: C.num_cols != B.num_cols" << endl;
        valid = false;
    }
    if (C.num_nnzs < 0) {
        cerr << "Error: C.num_nnzs < 0" << endl;
        valid = false;
    }
    
    if (valid) {
        cout << "Basic validation: PASSED" << endl;
    } else {
        cout << "Basic validation: FAILED" << endl;
    }
    
    // Write matrix C to MTX file
    std::string matA_name = extractFileNameWithoutExtension(matA_path);
    std::string output_filename = matA_name + "_SpOps.mtx";
    
    // Convert CSR to COO for writing
    COO_Matrix<IndexType, ValueType> coo_C = csr_to_coo(C);
    
    // Prepare arrays for mm_write_mtx_crd
    // Note: MTX format uses 1-based indexing, and mm_write_mtx_crd expects int and double
    int *I = new int[coo_C.num_nnzs];
    int *J = new int[coo_C.num_nnzs];
    double *V = new double[coo_C.num_nnzs];
    
    for (IndexType i = 0; i < coo_C.num_nnzs; i++) {
        I[i] = static_cast<int>(coo_C.row_index[i] + 1);  // Convert to 1-based and int
        J[i] = static_cast<int>(coo_C.col_index[i] + 1);  // Convert to 1-based and int
        V[i] = static_cast<double>(coo_C.values[i]);       // Convert to double
    }
    
    // Create MM_typecode for real coordinate sparse matrix
    MM_typecode matcode;
    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_coordinate(&matcode);
    mm_set_real(&matcode);
    mm_set_general(&matcode);
    
    // Write to MTX file
    int ret = mm_write_mtx_crd(const_cast<char*>(output_filename.c_str()), 
                                static_cast<int>(C.num_rows), 
                                static_cast<int>(C.num_cols), 
                                static_cast<int>(C.num_nnzs),
                                I, J, V, matcode);
    
    if (ret == 0) {
        cout << "Matrix C written to: " << output_filename << endl;
    } else {
        cerr << "Error writing matrix C to file: " << output_filename << endl;
    }
    
    delete[] I;
    delete[] J;
    delete[] V;
    delete_host_matrix(coo_C);
    
    delete_host_matrix(C);
    
    delete_host_matrix(A);
    delete_host_matrix(B);
}

template <typename IndexType, typename ValueType>
void test_spgemm_performance(const char *matA_path, const char *matB_path, int iterations = 10, int kernel_flag = 1, bool sortOutput = false)
{
    cout << "========================================" << endl;
    cout << "Testing SpGEMM Performance" << endl;
    cout << "========================================" << endl;
    
    // Read matrices
    CSR_Matrix<IndexType, ValueType> A = read_csr_matrix<IndexType, ValueType>(matA_path);
    CSR_Matrix<IndexType, ValueType> B = read_csr_matrix<IndexType, ValueType>(matB_path);
    
    if (A.num_cols != B.num_rows) {
        cerr << "Error: Dimension mismatch" << endl;
        delete_host_matrix(A);
        delete_host_matrix(B);
        return;
    }
    
    long long int flops = get_spgemm_flop(A, B);
    cout << "Estimated FLOPs: " << flops << endl;
    
    cout << "\n--- Kernel " << kernel_flag << " Performance ---" << endl;
    
    CSR_Matrix<IndexType, ValueType> C;
    
    // Warmup (first execution is excluded from evaluation)
    LeSpGEMM(A, B, C, sortOutput, kernel_flag);
    delete_host_matrix(C);
    
    // Benchmark (matching reference RowSpGEMM.cpp timing)
    double total_time = 0.0;
    for (int i = 0; i < iterations; i++) {
        double start, end, msec;
        #ifdef _OPENMP
        start = omp_get_wtime();
        #else
        start = (double)clock() / CLOCKS_PER_SEC;
        #endif
        
        LeSpGEMM(A, B, C, sortOutput, kernel_flag);
        
        #ifdef _OPENMP
        end = omp_get_wtime();
        #else
        end = (double)clock() / CLOCKS_PER_SEC;
        #endif
        
        msec = (end - start) * 1000.0;
        total_time += msec;
        
        // Clear C after timing (except for last iteration)
        if (i < iterations - 1) {
            delete_host_matrix(C);
        }
    }
    double avg_time = total_time / iterations;
    
    cout << "Average time: " << avg_time << " ms" << endl;
    double gflops = (flops / 1e9) / (avg_time / 1000.0);
    cout << "Average performance: " << gflops << " GFLOPS" << endl;
    cout << "C nnz: " << C.num_nnzs << endl;
    
    delete_host_matrix(C);
    
    delete_host_matrix(A);
    delete_host_matrix(B);
}

template <typename IndexType, typename ValueType>
void run_spgemm_test(int argc, char **argv)
{
    char *matA_path = NULL;
    char *matB_path = NULL;
    
    // Extract matrix file paths (non-option arguments)
    int file_count = 0;
    for(int i = 1; i < argc; i++){
        if(argv[i][0] != '-'){
            if(file_count == 0) {
                matA_path = argv[i];
                file_count++;
            } else if(file_count == 1) {
                matB_path = argv[i];
                file_count++;
                break;
            }
        }
    }
    
    if(matA_path == NULL || matB_path == NULL) {
        printf("Error: You need to provide two matrix files!\n");
        usage(argc, argv);
        return;
    }
    
    // Parse test_type
    const char *test_type = "performance";
    char *test_type_str = get_argval(argc, argv, "test_type");
    if(test_type_str != NULL) {
        test_type = test_type_str;
    }
    
    // Parse iterations
    int iterations = 10;
    char *iterations_str = get_argval(argc, argv, "iterations");
    if(iterations_str != NULL) {
        iterations = atoi(iterations_str);
    }
    
    // Parse kernel_flag
    int kernel_flag = 1;
    char *kernel_str = get_argval(argc, argv, "kernel");
    if(kernel_str != NULL) {
        kernel_flag = atoi(kernel_str);
        if(kernel_flag != 1 && kernel_flag != 2) {
            printf("Error: kernel must be 1 or 2\n");
            return;
        }
    }
    
    // Parse sort_output
    bool sortOutput = false;
    char *sort_str = get_argval(argc, argv, "sort");
    if(sort_str != NULL) {
        int sort_val = atoi(sort_str);
        sortOutput = (sort_val != 0);
    }
    
    cout << "SpGEMM Test Program" << endl;
    cout << "Matrix A: " << matA_path << endl;
    cout << "Matrix B: " << matB_path << endl;
    cout << "Test type: " << test_type << endl;
    cout << "Kernel: " << kernel_flag << endl;
    cout << "Sort output: " << (sortOutput ? "yes" : "no") << endl;
    cout << "Threads: " << Le_get_thread_num() << endl;
    cout << endl;
    
    if (strcmp(test_type, "performance") == 0) {
        test_spgemm_performance<IndexType, ValueType>(matA_path, matB_path, iterations, kernel_flag, sortOutput);
    } else {
        test_spgemm_correctness<IndexType, ValueType>(matA_path, matB_path, kernel_flag, sortOutput);
    }
}

int main(int argc, char *argv[])
{
    if (get_arg(argc, argv, "help") != NULL || argc < 3){
        usage(argc, argv);
        return EXIT_SUCCESS;
    }
    
    // Parse precision (default: 64 for double)
    int precision = 64;
    char * precision_str = get_argval(argc, argv, "precision");
    if(precision_str != NULL)
        precision = atoi(precision_str);
    
    // 不用超线程，默认设置为 CPU_SOCKET * CPU_CORES_PER_SOC
    #ifdef CPU_SOCKET
    Le_set_thread_num(CPU_SOCKET * CPU_CORES_PER_SOC);
    #else
    Le_set_thread_num(Le_get_core_num());
    #endif
    
    // Parse threads
    char * threads_str = get_argval(argc, argv, "threads");
    if(threads_str != NULL)
        Le_set_thread_num(atoi(threads_str));
    
    // SpGEMM only supports int64_t for IndexType (fixed)
    printf("\nUsing %d-bit floating point precision, 64-bit Index (int64_t), threads = %d\n\n", 
           precision, Le_get_thread_num());
    
    // Call appropriate template instantiation (only int64_t for IndexType)
    if (precision == 32){
        run_spgemm_test<int64_t, float>(argc, argv);
    }
    else if (precision == 64){
        run_spgemm_test<int64_t, double>(argc, argv);
    }
    else{
        usage(argc, argv);
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

