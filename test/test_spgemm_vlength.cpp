/**
 * @file test_spgemm_vlength.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Test program for Variable-length Cluster SpGEMM correctness and performance
 * @version 0.1
 * @date 2026
 */

#include "../include/SpOps.h"
#include "../include/spgemm.h"
#include "../include/timer.h"
#include "../include/cmdline.h"
#include "../include/sparse_io.h"
#include "../include/sparse_conversion.h"
#include "../include/spgemm_utility.h"
#include "../include/mmio.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <cmath>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <ctime>

using namespace std;

/**
 * @brief Generate offsets for variable-length cluster based on average cluster size
 *        Simple strategy: divide rows into clusters of approximately avg_cluster_sz size
 *        Last cluster may be smaller
 */
template <typename IndexType>
std::vector<IndexType> generate_offsets_simple(IndexType num_rows, IndexType avg_cluster_sz)
{
    std::vector<IndexType> offsets;
    offsets.push_back(0);
    
    IndexType curr_off = 0;
    while (curr_off < num_rows) {
        IndexType next_off = std::min(curr_off + avg_cluster_sz, num_rows);
        offsets.push_back(next_off);
        curr_off = next_off;
    }
    
    return offsets;
}

void usage(int argc, char** argv)
{
    std::cout << "Usage:\n";
    std::cout << "\t" << argv[0] << " <matrix_A.mtx> <matrix_B.mtx> [options]\n";
    std::cout << "\t" << "Options:\n";
    std::cout << "\t" << " --precision = 32 (float) or 64 (double:default)\n";
    std::cout << "\t" << " --threads   = define the num of omp threads (default: all cores)\n";
    std::cout << "\t" << " --test_type = correctness or performance (default)\n";
    std::cout << "\t" << " --iterations= number of iterations for performance test (default: 10)\n";
    std::cout << "\t" << " --kernel    = 1 (Hash-based variable-length cluster-wise:default)\n";
    std::cout << "\t" << " --avg_cluster_sz = average cluster size (default: 8, used for simple method)\n";
    std::cout << "\t" << " --similarity_th = similarity threshold for Jaccard-based clustering (default: 0.3)\n";
    std::cout << "\t" << " --max_cluster_size = maximum cluster size (-1 for unlimited, default: 8)\n";
    std::cout << "\t" << " --cluster_method = simple or jaccard (default: jaccard)\n";
    std::cout << "\t" << " --sort      = 0 (no sort:default) or 1 (sort output columns)\n";
    std::cout << "Note: Matrix files must be real-valued sparse matrices in the MatrixMarket file format.\n";
}

template <typename IndexType, typename ValueType>
void test_spgemm_vlength_correctness(const char *matA_path, const char *matB_path, 
                                     int kernel_flag = 1, bool sortOutput = false, 
                                     IndexType avg_cluster_sz = 8,
                                     const char *cluster_method = "jaccard",
                                     double similarity_th = 0.5,
                                     IndexType max_cluster_size = -1)
{
    cout << "========================================" << endl;
    cout << "Testing Variable-length Cluster SpGEMM Correctness" << endl;
    cout << "========================================" << endl;
    
    // Read matrices
    cout << "Reading matrix A from: " << matA_path << endl;
    CSR_Matrix<IndexType, ValueType> A_csr = read_csr_matrix<IndexType, ValueType>(matA_path);
    cout << "A: " << A_csr.num_rows << " x " << A_csr.num_cols << ", nnz: " << A_csr.num_nnzs << endl;
    
    cout << "Reading matrix B from: " << matB_path << endl;
    CSR_Matrix<IndexType, ValueType> B = read_csr_matrix<IndexType, ValueType>(matB_path);
    cout << "B: " << B.num_rows << " x " << B.num_cols << ", nnz: " << B.num_nnzs << endl;
    
    if (A_csr.num_cols != B.num_rows) {
        cerr << "Error: A.num_cols (" << A_csr.num_cols << ") != B.num_rows (" << B.num_rows << ")" << endl;
        delete_host_matrix(A_csr);
        delete_host_matrix(B);
        return;
    }
    
    // Generate offsets for variable-length cluster
    std::vector<IndexType> offsets;
    
    if (strcmp(cluster_method, "jaccard") == 0) {
        cout << "Generating offsets using Jaccard similarity (similarity_th = " << similarity_th;
        if (max_cluster_size != -1) {
            cout << ", max_cluster_size = " << max_cluster_size;
        }
        cout << ")..." << endl;
        offsets = generate_offsets_jaccard<IndexType>(
            A_csr.row_offset, A_csr.col_index, A_csr.num_rows, 
            similarity_th, max_cluster_size);
        cout << "Generated " << (offsets.size() - 1) << " clusters using Jaccard similarity method" << endl;
    } else {
        cout << "Generating offsets for variable-length cluster (avg_cluster_sz = " << avg_cluster_sz << ")..." << endl;
        offsets = generate_offsets_simple(A_csr.num_rows, avg_cluster_sz);
        cout << "Generated " << (offsets.size() - 1) << " clusters using simple method" << endl;
    }
    
    // Convert A to CSR_VlengthCluster format
    cout << "Converting A to CSR_VlengthCluster format..." << endl;
    CSR_VlengthCluster<IndexType, ValueType> A_cluster = csr_to_vlength_cluster<IndexType, ValueType>(A_csr, offsets);
    cout << "A_cluster: " << A_cluster.rows << " clusters, " << A_cluster.nnzc << " unique columns" << endl;
    cout << "A_cluster: max_cluster_sz = " << A_cluster.max_cluster_sz << ", nnzv = " << A_cluster.nnzv << endl;
    
    cout << "\n--- Testing Kernel " << kernel_flag << " ---" << endl;
    if (kernel_flag == 1) {
        cout << "Kernel: Hash-based variable-length cluster-wise SpGEMM (OpenMP with load balancing)" << endl;
    } else {
        cout << "Kernel: Unknown kernel flag, defaulting to Hash-based variable-length cluster-wise" << endl;
    }
    
    CSR_VlengthCluster<IndexType, ValueType> C_cluster;
    anonymouslib_timer timer;
    
    timer.start();
    if (sortOutput) {
        LeSpGEMM_VLength<true, IndexType, ValueType>(A_cluster, B, C_cluster, kernel_flag);
    } else {
        LeSpGEMM_VLength<false, IndexType, ValueType>(A_cluster, B, C_cluster, kernel_flag);
    }
    double time = timer.stop();
    
    cout << "C_cluster: " << C_cluster.rows << " clusters, " << C_cluster.nnzc << " unique columns" << endl;
    cout << "C_cluster (CSR equivalent): " << C_cluster.csr_rows << " x " << C_cluster.cols << endl;
    cout << "C_cluster: max_cluster_sz = " << C_cluster.max_cluster_sz << ", nnzv = " << C_cluster.nnzv << endl;
    cout << "Time: " << time << " ms" << endl;
    
    // Calculate performance
    long long int flops = get_spgemm_flop(A_csr, B);
    double gflops = (flops / 1e9) / (time / 1000.0);
    cout << "Performance: " << gflops << " GFLOPS" << endl;
    
    // Verify basic properties
    bool valid = true;
    if (C_cluster.csr_rows != A_csr.num_rows) {
        cerr << "Error: C_cluster.csr_rows != A.num_rows" << endl;
        valid = false;
    }
    if (C_cluster.cols != B.num_cols) {
        cerr << "Error: C_cluster.cols != B.num_cols" << endl;
        valid = false;
    }
    if (C_cluster.num_nnzs < 0) {
        cerr << "Error: C_cluster.num_nnzs < 0" << endl;
        valid = false;
    }
    
    if (valid) {
        cout << "Basic validation: PASSED" << endl;
    } else {
        cout << "Basic validation: FAILED" << endl;
    }
    
    // Convert C_cluster to CSR format for writing
    cout << "\nConverting C_cluster to CSR format..." << endl;
    CSR_Matrix<IndexType, ValueType> C_csr = vlength_cluster2csr<IndexType, ValueType>(C_cluster);
    cout << "C_csr: " << C_csr.num_rows << " x " << C_csr.num_cols << ", nnz: " << C_csr.num_nnzs << endl;
    
    // Write matrix C to MTX file
    std::string matA_name = extractFileNameWithoutExtension(matA_path);
    std::string suffix_str;
    if (kernel_flag == 1) {
        suffix_str = "hashvlengthcluster";
    } else {
        suffix_str = "hashvlengthcluster";  // default
    }
    
    std::string output_filename = matA_name + "_SpOps_" + suffix_str + ".mtx";
    
    // Convert CSR to COO for writing
    COO_Matrix<IndexType, ValueType> coo_C = csr_to_coo(C_csr);
    
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
    
    cout << "Writing result to: " << output_filename << endl;
    mm_write_mtx_crd(const_cast<char*>(output_filename.c_str()), 
                     static_cast<int>(C_csr.num_rows), 
                     static_cast<int>(C_csr.num_cols), 
                     static_cast<int>(C_csr.num_nnzs),
                     I, J, V, matcode);
    
    delete[] I;
    delete[] J;
    delete[] V;
    
    // Cleanup
    delete_host_matrix(C_csr);
    delete_host_matrix(coo_C);
    delete_vlength_cluster_matrix(C_cluster);
    delete_vlength_cluster_matrix(A_cluster);
    delete_host_matrix(A_csr);
    delete_host_matrix(B);
    
    cout << "\nTest completed successfully!" << endl;
}

template <typename IndexType, typename ValueType>
void test_spgemm_vlength_performance(const char *matA_path, const char *matB_path,
                                     int iterations = 10, int kernel_flag = 1, bool sortOutput = false,
                                     IndexType avg_cluster_sz = 8,
                                     const char *cluster_method = "jaccard",
                                     double similarity_th = 0.5,
                                     IndexType max_cluster_size = -1)
{
    cout << "========================================" << endl;
    cout << "Testing Variable-length Cluster SpGEMM Performance" << endl;
    cout << "========================================" << endl;
    
    // Read matrices
    CSR_Matrix<IndexType, ValueType> A_csr = read_csr_matrix<IndexType, ValueType>(matA_path);
    CSR_Matrix<IndexType, ValueType> B = read_csr_matrix<IndexType, ValueType>(matB_path);
    
    if (A_csr.num_cols != B.num_rows) {
        cerr << "Error: Dimension mismatch" << endl;
        delete_host_matrix(A_csr);
        delete_host_matrix(B);
        return;
    }
    
    // Generate offsets for variable-length cluster
    std::vector<IndexType> offsets;
    if (strcmp(cluster_method, "jaccard") == 0) {
        offsets = generate_offsets_jaccard<IndexType>(
            A_csr.row_offset, A_csr.col_index, A_csr.num_rows, 
            similarity_th, max_cluster_size);
    } else {
        offsets = generate_offsets_simple(A_csr.num_rows, avg_cluster_sz);
    }
    
    // Convert A to CSR_VlengthCluster format
    CSR_VlengthCluster<IndexType, ValueType> A_cluster = csr_to_vlength_cluster<IndexType, ValueType>(A_csr, offsets);
    
    long long int flops = get_spgemm_flop(A_csr, B);
    cout << "Estimated FLOPs: " << flops << endl;
    
    cout << "\n--- Kernel " << kernel_flag << " Performance ---" << endl;
    if (kernel_flag == 1) {
        cout << "Kernel: Hash-based variable-length cluster-wise SpGEMM (OpenMP with load balancing)" << endl;
    } else {
        cout << "Kernel: Unknown kernel flag, defaulting to Hash-based variable-length cluster-wise" << endl;
    }
    
    CSR_VlengthCluster<IndexType, ValueType> C_cluster;
    
    // Warmup (first execution is excluded from evaluation)
    if (sortOutput) {
        LeSpGEMM_VLength<true, IndexType, ValueType>(A_cluster, B, C_cluster, kernel_flag);
    } else {
        LeSpGEMM_VLength<false, IndexType, ValueType>(A_cluster, B, C_cluster, kernel_flag);
    }
    delete_vlength_cluster_matrix(C_cluster);
    
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
            LeSpGEMM_VLength<true, IndexType, ValueType>(A_cluster, B, C_cluster, kernel_flag);
        } else {
            LeSpGEMM_VLength<false, IndexType, ValueType>(A_cluster, B, C_cluster, kernel_flag);
        }
        
        #ifdef _OPENMP
        end = omp_get_wtime();
        #else
        end = (double)clock() / CLOCKS_PER_SEC;
        #endif
        
        msec = (end - start) * 1000.0;
        total_time += msec;
        
        // Clear C_cluster after timing (except for last iteration)
        if (i < iterations - 1) {
            delete_vlength_cluster_matrix(C_cluster);
        }
    }
    double avg_time = total_time / iterations;
    
    cout << "Average time: " << avg_time << " ms" << endl;
    double gflops = (flops / 1e9) / (avg_time / 1000.0);
    cout << "Average performance: " << gflops << " GFLOPS" << endl;
    
    delete_vlength_cluster_matrix(C_cluster);
    delete_vlength_cluster_matrix(A_cluster);
    delete_host_matrix(A_csr);
    delete_host_matrix(B);
}

template <typename IndexType, typename ValueType>
void run_spgemm_vlength_test(int argc, char **argv)
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
        if(kernel_flag != 1) {
            printf("Error: kernel must be 1 (Hash-based variable-length cluster-wise)\n");
            return;
        }
    }
    
    // Parse avg_cluster_sz
    IndexType avg_cluster_sz = 8;
    char *avg_cluster_sz_str = get_argval(argc, argv, "avg_cluster_sz");
    if(avg_cluster_sz_str != NULL) {
        avg_cluster_sz = static_cast<IndexType>(atoi(avg_cluster_sz_str));
    }
    
    // Parse cluster_method
    const char *cluster_method = "jaccard";
    char *cluster_method_str = get_argval(argc, argv, "cluster_method");
    if(cluster_method_str != NULL) {
        cluster_method = cluster_method_str;
        if(strcmp(cluster_method, "simple") != 0 && strcmp(cluster_method, "jaccard") != 0) {
            printf("Error: cluster_method must be 'simple' or 'jaccard'\n");
            return;
        }
    }
    
    // Parse similarity_th (for Jaccard method)
    double similarity_th = 0.3;
    char *similarity_th_str = get_argval(argc, argv, "similarity_th");
    if(similarity_th_str != NULL) {
        similarity_th = atof(similarity_th_str);
    }
    
    // Parse max_cluster_size (for Jaccard method)
    IndexType max_cluster_size = 8;
    char *max_cluster_size_str = get_argval(argc, argv, "max_cluster_size");
    if(max_cluster_size_str != NULL) {
        max_cluster_size = static_cast<IndexType>(atoi(max_cluster_size_str));
    }
    
    // Parse sort_output
    bool sortOutput = false;
    char *sort_str = get_argval(argc, argv, "sort");
    if(sort_str != NULL) {
        int sort_val = atoi(sort_str);
        sortOutput = (sort_val != 0);
    }
    
    cout << "Variable-length Cluster SpGEMM Test Program" << endl;
    cout << "Matrix A: " << matA_path << endl;
    cout << "Matrix B: " << matB_path << endl;
    cout << "Test type: " << test_type << endl;
    cout << "Kernel: " << kernel_flag;
    if (kernel_flag == 1) {
        cout << " (Hash-based variable-length cluster-wise)" << endl;
    } else {
        cout << " (Unknown, defaulting to Hash-based variable-length cluster-wise)" << endl;
    }
    cout << "Cluster method: " << cluster_method << endl;
    if (strcmp(cluster_method, "simple") == 0) {
        cout << "Average cluster size: " << avg_cluster_sz << endl;
    } else {
        cout << "Similarity threshold: " << similarity_th << endl;
        if (max_cluster_size != -1) {
            cout << "Max cluster size: " << max_cluster_size << endl;
        }
    }
    cout << "Sort output: " << (sortOutput ? "yes" : "no") << endl;
    cout << "Threads: " << Le_get_thread_num() << endl;
    cout << endl;
    
    if (strcmp(test_type, "performance") == 0) {
        test_spgemm_vlength_performance<IndexType, ValueType>(
            matA_path, matB_path, iterations, kernel_flag, sortOutput, 
            avg_cluster_sz, cluster_method, similarity_th, max_cluster_size);
    } else {
        test_spgemm_vlength_correctness<IndexType, ValueType>(
            matA_path, matB_path, kernel_flag, sortOutput, 
            avg_cluster_sz, cluster_method, similarity_th, max_cluster_size);
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
        run_spgemm_vlength_test<int64_t, float>(argc, argv);
    }
    else if (precision == 64){
        run_spgemm_vlength_test<int64_t, double>(argc, argv);
    }
    else{
        usage(argc, argv);
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
