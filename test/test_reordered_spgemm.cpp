/**
 * @file test_reordered_spgemm.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Test program for reordered SpGEMM: C = P(A) * A
 *        Where P(A) is the reordered version of matrix A according to permutation file
 * @version 0.1
 * @date 2026
 */

#include "../include/SpOps.h"
#include "../include/spgemm.h"
#include "../include/timer.h"
#include "../include/cmdline.h"
#include "../include/sparse_io.h"
#include "../include/sparse_conversion.h"
#include "../include/mmio.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <ctime>

using namespace std;

void usage(int argc, char** argv)
{
    std::cout << "Usage:\n";
    std::cout << "\t" << argv[0] << " <matrix_A.mtx> <permutation.reorder> [options]\n";
    std::cout << "\t" << "Options:\n";
    std::cout << "\t" << " --precision = 32 (float) or 64 (double:default)\n";
    std::cout << "\t" << " --threads   = define the num of omp threads (default: all cores)\n";
    std::cout << "\t" << " --test_type = correctness or performance (default)\n";
    std::cout << "\t" << " --iterations= number of iterations for performance test (default: 10)\n";
    std::cout << "\t" << " --kernel    = 1 (Hash-based row-wise:default), 2 (Array-based row-wise optimized), or 3 (SPA-based array row-wise)\n";
    std::cout << "\t" << " --sort      = 0 (no sort:default) or 1 (sort output columns)\n";
    std::cout << "Note: Matrix files must be real-valued sparse matrices in the MatrixMarket file format.\n";
    std::cout << "Note: Permutation file contains one row index per line (0-based or 1-based, auto-detected).\n";
    std::cout << "Note: This program computes C = P(A) * A, where P(A) is the reordered matrix A.\n";
}

/**
 * @brief Test reordered SpGEMM correctness
 *        Computes C = P(A) * A, where P(A) is the reordered version of A
 */
template <typename IndexType, typename ValueType>
void test_reordered_spgemm_correctness(const char *matA_path, const char *permutation_path, int kernel_flag = 1, bool sortOutput = false)
{
    cout << "========================================" << endl;
    cout << "Testing Reordered SpGEMM Correctness" << endl;
    cout << "========================================" << endl;
    
    // Read matrix A
    cout << "Reading matrix A from: " << matA_path << endl;
    CSR_Matrix<IndexType, ValueType> A_original = read_csr_matrix<IndexType, ValueType>(matA_path);
    cout << "A_original: " << A_original.num_rows << " x " << A_original.num_cols << ", nnz: " << A_original.num_nnzs << endl;
    
    // Read permutation file
    cout << "Reading permutation from: " << permutation_path << endl;
    std::vector<IndexType> row_permutation = read_row_permutation<IndexType>(permutation_path, A_original.num_rows);
    cout << "Permutation size: " << row_permutation.size() << endl;
    
    // Validate permutation size
    if (static_cast<IndexType>(row_permutation.size()) != A_original.num_rows) {
        cerr << "Error: Permutation size (" << row_permutation.size() 
             << ") != A.num_rows (" << A_original.num_rows << ")" << endl;
        delete_host_matrix(A_original);
        return;
    }
    
    // Reorder matrix A: P(A) = reordered A
    // Reference implementation approach: row_permutation[i] = original row index for new row i
    cout << "Reordering matrix A..." << endl;
    CSR_Matrix<IndexType, ValueType> A_reordered = csr_reorder_rows(A_original, row_permutation);
    cout << "A_reordered: " << A_reordered.num_rows << " x " << A_reordered.num_cols << ", nnz: " << A_reordered.num_nnzs << endl;
    
    // Verify dimensions match for matrix multiplication: C = P(A) * A
    if (A_reordered.num_cols != A_original.num_rows) {
        cerr << "Error: A_reordered.num_cols (" << A_reordered.num_cols 
             << ") != A_original.num_rows (" << A_original.num_rows << ")" << endl;
        delete_host_matrix(A_original);
        delete_host_matrix(A_reordered);
        return;
    }
    
    cout << "\n--- Testing Kernel " << kernel_flag << " (C = P(A) * A) ---" << endl;
    if (kernel_flag == 1) {
        cout << "Kernel: Hash-based row-wise SpGEMM (OpenMP with load balancing)" << endl;
    } else if (kernel_flag == 2) {
        cout << "Kernel: Array-based row-wise SpGEMM (HSMU-SpGEMM inspired, optimized version)" << endl;
    } else if (kernel_flag == 3) {
        cout << "Kernel: SPA-based array row-wise SpGEMM (HSMU-SpGEMM inspired, Sparse Accumulator)" << endl;
    } else {
        cout << "Kernel: Unknown kernel flag, defaulting to Hash-based row-wise" << endl;
    }
    
    CSR_Matrix<IndexType, ValueType> C;
    anonymouslib_timer timer;
    
    timer.start();
    if (sortOutput) {
        LeSpGEMM<true, IndexType, ValueType>(A_reordered, A_original, C, kernel_flag);
    } else {
        LeSpGEMM<false, IndexType, ValueType>(A_reordered, A_original, C, kernel_flag);
    }
    double time = timer.stop();
    
    cout << "C: " << C.num_rows << " x " << C.num_cols << ", nnz: " << C.num_nnzs << endl;
    cout << "Time: " << time << " ms" << endl;
    
    // Calculate performance
    long long int flops = get_spgemm_flop(A_reordered, A_original);
    double gflops = (flops / 1e9) / (time / 1000.0);
    cout << "Estimated FLOPs: " << flops << endl;
    cout << "Performance: " << gflops << " GFLOPS" << endl;
    
    // Verify basic properties
    bool valid = true;
    if (C.num_rows != A_reordered.num_rows) {
        cerr << "Error: C.num_rows != A_reordered.num_rows" << endl;
        valid = false;
    }
    if (C.num_cols != A_original.num_cols) {
        cerr << "Error: C.num_cols != A_original.num_cols" << endl;
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
    std::string suffix_str;
    if (kernel_flag == 1) {
        suffix_str = "hashrowwise";
    } else if (kernel_flag == 2) {
        suffix_str = "arrayrowwise";
    } else if (kernel_flag == 3) {
        suffix_str = "sparowwise";
    } else {
        suffix_str = "hashrowwise";  // default
    }
    
    std::string output_filename = matA_name + "_SpOps_reordered_" + suffix_str + ".mtx";
    
    // Convert CSR to COO for writing
    COO_Matrix<IndexType, ValueType> coo_C = csr_to_coo(C);
    
    // Prepare arrays for mm_write_mtx_crd
    int *I = new int[coo_C.num_nnzs];
    int *J = new int[coo_C.num_nnzs];
    double *V = new double[coo_C.num_nnzs];
    
    for (IndexType i = 0; i < coo_C.num_nnzs; i++) {
        I[i] = static_cast<int>(coo_C.row_index[i] + 1);
        J[i] = static_cast<int>(coo_C.col_index[i] + 1);
        V[i] = static_cast<double>(coo_C.values[i]);
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
    delete_host_matrix(A_reordered);
    delete_host_matrix(A_original);
}

/**
 * @brief Test reordered SpGEMM performance
 *        Computes C = P(A) * A, where P(A) is the reordered version of A
 */
template <typename IndexType, typename ValueType>
void test_reordered_spgemm_performance(const char *matA_path, const char *permutation_path, int iterations = 10, int kernel_flag = 1, bool sortOutput = false)
{
    cout << "========================================" << endl;
    cout << "Testing Reordered SpGEMM Performance" << endl;
    cout << "========================================" << endl;
    
    // Read matrix A
    CSR_Matrix<IndexType, ValueType> A_original = read_csr_matrix<IndexType, ValueType>(matA_path);
    
    // Read permutation file
    std::vector<IndexType> row_permutation = read_row_permutation<IndexType>(permutation_path, A_original.num_rows);
    
    // Validate permutation size
    if (static_cast<IndexType>(row_permutation.size()) != A_original.num_rows) {
        cerr << "Error: Permutation size mismatch" << endl;
        delete_host_matrix(A_original);
        return;
    }
    
    // Reorder matrix A: P(A) = reordered A
    CSR_Matrix<IndexType, ValueType> A_reordered = csr_reorder_rows(A_original, row_permutation);
    
    if (A_reordered.num_cols != A_original.num_rows) {
        cerr << "Error: Dimension mismatch for matrix multiplication" << endl;
        delete_host_matrix(A_original);
        delete_host_matrix(A_reordered);
        return;
    }
    
    long long int flops = get_spgemm_flop(A_reordered, A_original);
    cout << "Estimated FLOPs: " << flops << endl;
    
    cout << "\n--- Kernel " << kernel_flag << " Performance (C = P(A) * A) ---" << endl;
    if (kernel_flag == 1) {
        cout << "Kernel: Hash-based row-wise SpGEMM (OpenMP with load balancing)" << endl;
    } else if (kernel_flag == 2) {
        cout << "Kernel: Array-based row-wise SpGEMM (HSMU-SpGEMM inspired, optimized version)" << endl;
    } else if (kernel_flag == 3) {
        cout << "Kernel: SPA-based array row-wise SpGEMM (HSMU-SpGEMM inspired, Sparse Accumulator)" << endl;
    } else {
        cout << "Kernel: Unknown kernel flag, defaulting to Hash-based row-wise" << endl;
    }
    
    CSR_Matrix<IndexType, ValueType> C;
    
    // Warmup (first execution is excluded from evaluation)
    if (sortOutput) {
        LeSpGEMM<true, IndexType, ValueType>(A_reordered, A_original, C, kernel_flag);
    } else {
        LeSpGEMM<false, IndexType, ValueType>(A_reordered, A_original, C, kernel_flag);
    }
    delete_host_matrix(C);
    
    // Benchmark
    double total_time = 0.0;
    for (int i = 0; i < iterations; i++) {
        double start, end, msec;
        #ifdef _OPENMP
        start = omp_get_wtime();
        #else
        start = (double)clock() / CLOCKS_PER_SEC;
        #endif
        
        if (sortOutput) {
            LeSpGEMM<true, IndexType, ValueType>(A_reordered, A_original, C, kernel_flag);
        } else {
            LeSpGEMM<false, IndexType, ValueType>(A_reordered, A_original, C, kernel_flag);
        }
        
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
    delete_host_matrix(A_reordered);
    delete_host_matrix(A_original);
}

template <typename IndexType, typename ValueType>
void run_reordered_spgemm_test(int argc, char **argv)
{
    char *matA_path = NULL;
    char *permutation_path = NULL;
    
    // Extract file paths (non-option arguments)
    int file_count = 0;
    for(int i = 1; i < argc; i++){
        if(argv[i][0] != '-'){
            if(file_count == 0) {
                matA_path = argv[i];
                file_count++;
            } else if(file_count == 1) {
                permutation_path = argv[i];
                file_count++;
                break;
            }
        }
    }
    
    if(matA_path == NULL || permutation_path == NULL) {
        printf("Error: You need to provide matrix A file and permutation file!\n");
        printf("Usage: %s <matrix_A.mtx> <permutation.reorder> [options]\n", argv[0]);
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
        if(kernel_flag != 1 && kernel_flag != 2 && kernel_flag != 3) {
            printf("Error: kernel must be 1, 2, or 3\n");
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
    
    cout << "Reordered SpGEMM Test Program" << endl;
    cout << "Matrix A: " << matA_path << endl;
    cout << "Permutation file: " << permutation_path << endl;
    cout << "Test type: " << test_type << endl;
    cout << "Kernel: " << kernel_flag;
    if (kernel_flag == 1) {
        cout << " (Hash-based row-wise)" << endl;
    } else if (kernel_flag == 2) {
        cout << " (Array-based row-wise, optimized version)" << endl;
    } else if (kernel_flag == 3) {
        cout << " (SPA-based array row-wise)" << endl;
    } else {
        cout << " (Unknown, defaulting to Hash-based row-wise)" << endl;
    }
    cout << "Sort output: " << (sortOutput ? "yes" : "no") << endl;
    cout << "Threads: " << Le_get_thread_num() << endl;
    cout << "Computing: C = P(A) * A" << endl;
    cout << endl;
    
    if (strcmp(test_type, "performance") == 0) {
        test_reordered_spgemm_performance<IndexType, ValueType>(matA_path, permutation_path, iterations, kernel_flag, sortOutput);
    } else {
        test_reordered_spgemm_correctness<IndexType, ValueType>(matA_path, permutation_path, kernel_flag, sortOutput);
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
    printf("\nUsing %d-bit floating point precision, 64-bit Index (int64_t), threads = %d\n", 
           precision, Le_get_thread_num());
    printf("Mode: Reordered SpGEMM (C = P(A) * A)\n");
    printf("\n");
    
    // Call appropriate template instantiation (only int64_t for IndexType)
    if (precision == 32){
        run_reordered_spgemm_test<int64_t, float>(argc, argv);
    }
    else if (precision == 64){
        run_reordered_spgemm_test<int64_t, double>(argc, argv);
    }
    else{
        usage(argc, argv);
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
