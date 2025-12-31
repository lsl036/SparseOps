#if 0
//  this part should copy to /opt/intel/oneapi/mkl/latest/examples/dpcpp/sparse_blas/source/sparse_gemv.cpp, and using intel cmake to compile
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <vector>

#include "mkl.h"
#include "oneapi/mkl.hpp"
#include <CL/sycl.hpp>

#include<iostream>
#include<cstdio>
#include"../LeSpMV/include/LeSpMV.h"
#include"../LeSpMV/include/cmdline.h"

#include "../common/common_for_examples.hpp"
#include "common_for_sparse_examples.hpp"

template <typename fp, typename intType>
int run_sparse_matrix_vector_multiply_example(int argc, char **argv, const sycl::device &dev)
{
    // Initialize data for Sparse Matrix-Vector Multiply
    oneapi::mkl::transpose transA = oneapi::mkl::transpose::nontrans;

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
        return 2;
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

    intType nrows = csr.num_rows;
    intType ncols = csr.num_cols;

    // Input matrix in CSR format
    std::vector<intType, mkl_allocator<intType, 64>> ia;
    std::vector<intType, mkl_allocator<intType, 64>> ja;
    std::vector<fp, mkl_allocator<fp, 64>> a;

    ia.resize(csr.num_rows + 1);
    ja.resize(csr.num_nnzs);
    a.resize(csr.num_nnzs);
    for (size_t i = 0; i < csr.num_rows + 1; i++)
    {
        ia[i] = csr.row_offset[i];
    }
    for (size_t i = 0; i < csr.num_nnzs; i++)
    {
        ja[i] = csr.col_index[i];
        a[i]  = csr.values[i];
    }
    

    // Vectors x and y
    std::vector<fp, mkl_allocator<fp, 64>> x;
    std::vector<fp, mkl_allocator<fp, 64>> y;
    std::vector<fp, mkl_allocator<fp, 64>> z;
    x.resize(csr.num_cols);
    y.resize(csr.num_rows);
    z.resize(csr.num_rows);

    for(size_t i = 0; i < csr.num_cols; i++)
        x[i] = rand() / (RAND_MAX + 1.0); 
    // std::fill(y, y + csr.num_rows, 0);
    // std::fill(z, z + csr.num_rows, 0);
    for (intType i = 0; i < csr.num_rows; i++) {
        // x[i] = set_fp_value(fp(1.0), fp(0.0));
        y[i] = set_fp_value(fp(0.0), fp(0.0));
        z[i] = set_fp_value(fp(0.0), fp(0.0));
    }

    fp alpha = set_fp_value(fp(1.0), fp(0.0));
    fp beta  = set_fp_value(fp(0.0), fp(0.0));

    // Catch asynchronous exceptions
    auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e) {
                std::cout << "Caught asynchronous SYCL "
                             "exception during sparse::gemv:\n"
                          << e.what() << std::endl;
            }
        }
    };

    //
    // Execute Matrix Multiply
    //

    std::cout << "\n\t\tsparse::gemv parameters:\n";
    std::cout << "\t\t\ttransA = "
              << (transA == oneapi::mkl::transpose::nontrans ?
                          "nontrans" :
                          (transA == oneapi::mkl::transpose::trans ? "trans" : "conjtrans"))
              << std::endl;
    std::cout << "\t\t\tnrows = " << nrows << std::endl;
    std::cout << "\t\t\talpha = " << alpha << ", beta = " << beta << std::endl;

    // create execution queue and buffers of matrix data
    sycl::queue main_queue(dev, exception_handler);

    sycl::buffer<intType, 1> ia_buffer(ia.data(), (nrows + 1));
    sycl::buffer<intType, 1> ja_buffer(ja.data(), (ia[nrows]));
    sycl::buffer<fp, 1> a_buffer(a.data(), (ia[nrows]));
    sycl::buffer<fp, 1> x_buffer(x.data(), x.size());
    sycl::buffer<fp, 1> y_buffer(y.data(), y.size());

    // create and initialize handle for a Sparse Matrix in CSR format
    oneapi::mkl::sparse::matrix_handle_t handle = nullptr;

    double msec_per_iteration = 0.0, sec_per_iteration = 0.0;
    try {
        oneapi::mkl::sparse::init_matrix_handle(&handle);

        oneapi::mkl::sparse::set_csr_data(main_queue, handle, nrows, ncols, oneapi::mkl::index_base::zero,
                                          ia_buffer, ja_buffer, a_buffer);
        timer t;
        int num_iterations = 20;

        for(int i = 0; i < num_iterations; i++)
        {
            oneapi::mkl::sparse::gemv(main_queue, oneapi::mkl::transpose::nontrans, alpha, handle,
                                  x_buffer, beta, y_buffer);
            
            main_queue.wait();
        }

        msec_per_iteration = t.milliseconds_elapsed() / (double) num_iterations;
        sec_per_iteration = msec_per_iteration / 1000.0;
        oneapi::mkl::sparse::release_matrix_handle(main_queue, &handle);
    }
    catch (sycl::exception const &e) {
        std::cout << "\t\tCaught synchronous SYCL exception:\n" << e.what() << std::endl;

        main_queue.wait();
        oneapi::mkl::sparse::release_matrix_handle(main_queue, &handle);
        return 1;
    }
    catch (std::exception const &e) {
        std::cout << "\t\tCaught std exception:\n" << e.what() << std::endl;

        main_queue.wait();
        oneapi::mkl::sparse::release_matrix_handle(main_queue, &handle);
        return 1;
    }

    

    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) csr.num_nnzs / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_spmv(csr) / sec_per_iteration) / 1e9;
    
    csr.gflops = GFLOPs;
    csr.gbytes = GBYTEs;
    csr.time = msec_per_iteration;

    const char * location = "cpu" ;
    printf("\tbenchmarking %-20s [%s]: %8.4f ms ( %5.4f GFLOP/s %5.4f GB/s)\n", \
            "sycl_onapi_spmv", location, msec_per_iteration, GFLOPs, GBYTEs);

    // 保存测试性能结果
    FILE *save_perf = fopen(MAT_PERFORMANCE, "a");
    if ( save_perf == nullptr)
    {
        std::cout << "Unable to open perf-saved file: "<< MAT_PERFORMANCE << std::endl;
        return 3;
    }
    fprintf(save_perf, "%d %s oneAPI_SpMV %8.4f %5.4f \n", matID, matrixName.c_str(),  msec_per_iteration, GFLOPs);

    fclose(save_perf);
    delete_csr_matrix(csr);
    return 0;
}


int main(int argc, char **argv)
{
    std::list<my_sycl_device_types> list_of_devices;
    set_list_of_devices(list_of_devices);

    int status = 0;
    for (auto it = list_of_devices.begin(); it != list_of_devices.end(); ++it) {

        sycl::device my_dev;
        bool my_dev_is_found = false;
        get_sycl_device(my_dev, my_dev_is_found, *it);

        if (my_dev_is_found) {
            std::cout << "Running tests on " << sycl_device_names[*it] << ".\n";

            // std::cout << "\tRunning with single precision real data type:" << std::endl;
            // status = run_sparse_matrix_vector_multiply_example<float, std::int32_t>(my_dev);
            // if(status != 0) return status;

            if (my_dev.get_info<sycl::info::device::double_fp_config>().size() != 0) {
                std::cout << "\tRunning with double precision real data type:" << std::endl;
                status = run_sparse_matrix_vector_multiply_example<double, std::int32_t>(argc, argv, my_dev);
                if(status != 0) return status;
            }
        }
        else {
#ifdef FAIL_ON_MISSING_DEVICES
            std::cout << "No " << sycl_device_names[*it]
                      << " devices found; Fail on missing devices "
                         "is enabled.\n";
            return 1;
#else
            std::cout << "No " << sycl_device_names[*it] << " devices found; skipping "
                      << sycl_device_names[*it] << " tests.\n";
#endif
        }
    }
    mkl_free_buffers();
    return EXIT_SUCCESS;
}
#endif

int main(int argc, char const *argv[])
{
    return 0;
}
