/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

// Functions to test two SpMV kernels

#include <cassert>
#include <algorithm>
#include <limits>
#include <cmath>
#include "../include/LeSpMV.h"

#if 0

template <typename T>
T maximum_relative_error(const T * A, const T * B, const size_t N)
{
    T max_error = 0;

    T eps = std::sqrt( std::numeric_limits<T>::epsilon() );

    for(size_t i = 0; i < N; i++)
    {
        const T a = A[i];
        const T b = B[i];
        const T error = std::abs(a - b);
        if (error != 0){
            max_error = std::max(max_error, error/(std::abs(a) + std::abs(b) + eps) );
        }
    }
 
    return max_error;
}


/**
 * @brief Implementation details of test_spmv_kernel function
 * 
 * @tparam SparseMatrix1 
 * @tparam SpMV1 
 * @tparam SparseMatrix2 
 * @tparam SpMV2 
 * @param sm1_host 
 * @param spmv1 
 * @param sm2_host 
 * @param spmv2 
 */
template <typename SparseMatrix1, typename SpMV1,
          typename SparseMatrix2, typename SpMV2>
void compare_spmv_kernels(const SparseMatrix1 & sm1_host, SpMV1 spmv1,
                          const SparseMatrix2 & sm2_host, SpMV2 spmv2)
{
    // sanity checking
    assert(sm1_host.num_rows == sm2_host.num_rows);
    assert(sm1_host.num_cols == sm2_host.num_cols);
    assert(sm1_host.num_nnzs == sm2_host.num_nnzs);

    typedef typename SparseMatrix1::index_type IndexType;
    typedef typename SparseMatrix2::value_type ValueType;

    ValueType alpha = 1.0;
    ValueType beta  = 0.7;

    const IndexType num_rows = sm1_host.num_rows;
    const IndexType num_cols = sm1_host.num_cols;
  
    // initialize host vectors
    ValueType * x_host = new_array<ValueType>(num_cols);
    ValueType * y_host = new_array<ValueType>(num_rows);
    
    for(IndexType i = 0; i < num_cols; i++)
        x_host[i] = rand() / (RAND_MAX + 1.0); 

    for(IndexType i = 0; i < num_rows; i++)
        y_host[i] = rand() / (RAND_MAX + 1.0);


    // create vectors in appropriate locations
    ValueType * x_host1 = x_host;
    ValueType * y_host1 = y_host;
    ValueType * x_host2 = copy_array(x_host, num_cols);
    ValueType * y_host2 = copy_array(y_host, num_rows);
   
    // compute y = alpha * A*x + beta * y
    spmv1(alpha, sm1_host, x_host1, beta, y_host1);
    spmv2(alpha, sm2_host, x_host2, beta, y_host2);
   
    ValueType max_error = maximum_relative_error(y_host1, y_host2, num_rows);
    printf(" [max error %9f]", max_error);
    
    if ( max_error > 5 * std::sqrt( std::numeric_limits<ValueType>::epsilon() ) )
        printf(" POSSIBLE FAILURE");
               
    delete_array(x_host1);
    delete_array(y_host1);
    delete_array(x_host2);
    delete_array(y_host2);
}


/**
 * @brief A test routine to compare two implementations of SpMV
 * 
 * @tparam SparseMatrix1     must be listed in sparse_format.h
 * @tparam SpMV1             as a reference 
 * @tparam SparseMatrix2     must be listed in sparse_format.h
 * @tparam SpMV2             as a testing spmv implementation
 * @param sm1_host           Sparse matrix format of routine 1
 * @param spmv1              SPMV routine 1
 * @param sm2_host           Sparse matrix format of routine 2
 * @param spmv2              SPMV routine 2
 * @param method_name        Print the name of routine 2 for testing
 */
template <typename SparseMatrix1, typename SpMV1,
          typename SparseMatrix2, typename SpMV2>
void test_spmv_kernel(const SparseMatrix1 & sm1_host, SpMV1 spmv1,
                      const SparseMatrix2 & sm2_host, SpMV2 spmv2, 
                      const char * method_name)
{
    printf("\ttesting %-26s", method_name);
        printf("[cpu]:");
    
    // 里面自定义了alpha 和 beta
    compare_spmv_kernels( sm1_host, spmv1, sm2_host, spmv2);

    printf("\n");
}

#endif