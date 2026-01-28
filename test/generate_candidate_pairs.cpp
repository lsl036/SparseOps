/**
 * @file generate_candidate_pairs.cpp
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Generate candidate pairs for hierarchical clustering using HashSpGEMMTopK
 *        Computes A * AT with binary pattern and keeps top-k Jaccard similarities per row
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
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <ctime>

using namespace std;

// Helper function to create directory if it doesn't exist
bool ensure_directory_exists(const std::string& dir_path) {
    struct stat info;
    if (stat(dir_path.c_str(), &info) != 0) {
        // Directory doesn't exist, try to create it
        #ifdef _WIN32
            if (_mkdir(dir_path.c_str()) != 0) {
                return false;
            }
        #else
            if (mkdir(dir_path.c_str(), 0755) != 0) {
                return false;
            }
        #endif
    } else if (!(info.st_mode & S_IFDIR)) {
        // Path exists but is not a directory
        return false;
    }
    return true;
}

// Helper function to create filename suffix with top-k
std::string create_filename_suffix(int64_t top_k) {
    return "_candidate_pairs_topk" + std::to_string(top_k) + ".mtx";
}

void usage(int argc, char** argv)
{
    std::cout << "Usage:\n";
    std::cout << "\t" << argv[0] << " <matrix_A.mtx> [options]\n";
    std::cout << "\t" << "Options:\n";
    std::cout << "\t" << " --precision = 32 (float) or 64 (double:default)\n";
    std::cout << "\t" << " --threads   = define the num of omp threads (default: all cores)\n";
    std::cout << "\t" << " --topk      = number of top similarities to keep per row (default: 7)\n";
    std::cout << "\t" << " --output    = output file path for candidate pairs matrix (default: <CLOSE_PAIR_DATA_PATH>/<matrix_A>_candidate_pairs_topk<k>.mtx)\n";
    std::cout << "Note: Matrix file must be a sparse matrix in the MatrixMarket file format.\n";
    std::cout << "      The output matrix C contains top-k Jaccard similarities per row (0.0 to 1.0).\n";
    std::cout << "      Output directory is determined by CLOSE_PAIR_DATA_PATH environment variable.\n";
    std::cout << "      If CLOSE_PAIR_DATA_PATH is not set, default is /data/lsl/SparseOps/script\n";
}

template <typename IndexType, typename ValueType>
void generate_candidate_pairs(const char *matA_path, const char *output_path, IndexType top_k)
{
    cout << "========================================" << endl;
    cout << "Generate Candidate Pairs for Hierarchical Clustering" << endl;
    cout << "========================================" << endl;
    
    // Read matrix A
    cout << "Reading matrix A from: " << matA_path << endl;
    CSR_Matrix<IndexType, ValueType> A = read_csr_matrix<IndexType, ValueType>(matA_path);
    cout << "A: " << A.num_rows << " x " << A.num_cols << ", nnz: " << A.num_nnzs << endl;
    
    // Compute candidate pairs using HashSpGEMMTopK
    cout << "\nComputing candidate pairs (A * AT with binary pattern, top-k = " << top_k << ")..." << endl;
    CSR_Matrix<IndexType, ValueType> C;
    anonymouslib_timer timer;
    
    timer.start();
    HashSpGEMMTopK<IndexType, ValueType>(A, C, top_k);
    double time = timer.stop();
    
    cout << "C: " << C.num_rows << " x " << C.num_cols << ", nnz: " << C.num_nnzs << endl;
    cout << "Time: " << time << " ms" << endl;
    
    // Verify basic properties
    bool valid = true;
    if (C.num_rows != A.num_rows) {
        cerr << "Error: C.num_rows (" << C.num_rows << ") != A.num_rows (" << A.num_rows << ")" << endl;
        valid = false;
    }
    if (C.num_cols != A.num_rows) {
        cerr << "Error: C.num_cols (" << C.num_cols << ") != A.num_rows (" << A.num_rows << ")" << endl;
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
        delete_host_matrix(A);
        delete_host_matrix(C);
        return;
    }
    
    // Write matrix C to MTX file
    cout << "\nWriting candidate pairs matrix to: " << output_path << endl;
    
    // Convert CSR to COO for writing
    COO_Matrix<IndexType, ValueType> coo_C = csr_to_coo(C);
    
    // Prepare arrays for writing (0-based indexing, matching reference GenerateCandidatePairs.cpp)
    int *I = new int[coo_C.num_nnzs];
    int *J = new int[coo_C.num_nnzs];
    double *V = new double[coo_C.num_nnzs];
    
    for (IndexType i = 0; i < coo_C.num_nnzs; i++) {
        I[i] = static_cast<int>(coo_C.row_index[i]);  // 0-based indexing (matching reference)
        J[i] = static_cast<int>(coo_C.col_index[i]);  // 0-based indexing (matching reference)
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
    int ret = mm_write_mtx_crd(const_cast<char*>(output_path), 
                                static_cast<int>(C.num_rows), 
                                static_cast<int>(C.num_cols), 
                                static_cast<int>(C.num_nnzs),
                                I, J, V, matcode);
    
    if (ret == 0) {
        cout << "Candidate pairs matrix written successfully to: " << output_path << endl;
        cout << "Matrix contains top-" << top_k << " Jaccard similarities per row (0.0 to 1.0)" << endl;
    } else {
        cerr << "Error writing candidate pairs matrix to file: " << output_path << endl;
    }
    
    delete[] I;
    delete[] J;
    delete[] V;
    delete_host_matrix(coo_C);
    delete_host_matrix(C);
    delete_host_matrix(A);
    
    cout << "\nCandidate pairs generation completed successfully!" << endl;
}

template <typename IndexType, typename ValueType>
void run_generate_candidate_pairs(int argc, char **argv)
{
    char *matA_path = NULL;
    
    // Extract matrix file path (non-option arguments)
    int file_count = 0;
    for(int i = 1; i < argc; i++){
        if(argv[i][0] != '-'){
            if(file_count == 0) {
                matA_path = argv[i];
                file_count++;
                break;
            }
        }
    }
    
    if(matA_path == NULL) {
        printf("Error: You need to provide a matrix file!\n");
        usage(argc, argv);
        return;
    }
    
    // Parse top_k
    IndexType top_k = 7;  // Default value
    char *topk_str = get_argval(argc, argv, "topk");
    if(topk_str != NULL) {
        top_k = static_cast<IndexType>(atoi(topk_str));
        if(top_k <= 0) {
            printf("Error: topk must be a positive integer\n");
            return;
        }
    }
    
    // Get output directory from environment variable or use default
    const char* output_dir_env = std::getenv("CLOSE_PAIR_DATA_PATH");
    std::string output_dir;
    if (output_dir_env != NULL && strlen(output_dir_env) > 0) {
        output_dir = output_dir_env;
    } else {
        // Default directory
        output_dir = "/data/lsl/SparseOps/script";
    }
    
    // Ensure output directory exists
    if (!ensure_directory_exists(output_dir)) {
        cerr << "Warning: Cannot create or access output directory: " << output_dir << endl;
        cerr << "Will use current directory instead." << endl;
        output_dir = ".";
    }
    
    // Parse output path
    std::string output_path;
    char *output_str = get_argval(argc, argv, "output");
    if(output_str != NULL) {
        output_path = output_str;
    } else {
        // Generate default output path: <output_dir>/<matrix_A>_candidate_pairs_topk<k>.mtx
        std::string matA_name = extractFileNameWithoutExtension(matA_path);
        std::string filename_suffix = create_filename_suffix(static_cast<int64_t>(top_k));
        output_path = output_dir + "/" + matA_name + filename_suffix;
    }
    
    cout << "Generate Candidate Pairs Program" << endl;
    cout << "Matrix A: " << matA_path << endl;
    cout << "Top-K: " << top_k << endl;
    cout << "Output directory: " << output_dir << endl;
    cout << "Output file: " << output_path << endl;
    cout << "Threads: " << Le_get_thread_num() << endl;
    cout << endl;
    
    generate_candidate_pairs<IndexType, ValueType>(matA_path, output_path.c_str(), top_k);
}

int main(int argc, char *argv[])
{
    if (get_arg(argc, argv, "help") != NULL || argc < 2){
        usage(argc, argv);
        return EXIT_SUCCESS;
    }
    
    // Parse precision (default: 64 for double)
    int precision = 64;
    char * precision_str = get_argval(argc, argv, "precision");
    if(precision_str != NULL)
        precision = atoi(precision_str);
    
    // Set thread number (default: all cores)
    #ifdef CPU_SOCKET
    Le_set_thread_num(CPU_SOCKET * CPU_CORES_PER_SOC);
    #else
    Le_set_thread_num(Le_get_core_num());
    #endif
    
    // Parse threads
    char * threads_str = get_argval(argc, argv, "threads");
    if(threads_str != NULL)
        Le_set_thread_num(atoi(threads_str));
    
    // HashSpGEMMTopK only supports int64_t for IndexType (fixed)
    printf("\nUsing %d-bit floating point precision, 64-bit Index (int64_t), threads = %d\n\n", 
           precision, Le_get_thread_num());
    
    // Call appropriate template instantiation (only int64_t for IndexType)
    if (precision == 32){
        run_generate_candidate_pairs<int64_t, float>(argc, argv);
    }
    else if (precision == 64){
        run_generate_candidate_pairs<int64_t, double>(argc, argv);
    }
    else{
        usage(argc, argv);
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
