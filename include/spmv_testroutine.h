#ifndef SPMV_TESTROUTINE_H
#define SPMV_TESTROUTINE_H

#include"sparse_format.h"
#include<cstdio>

template <typename IndexType, typename ValueType>
double test_coo_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, int schedule_mod, double &convert_time);

template <typename IndexType, typename ValueType>
double test_csr_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, int schedule_mod);

template <typename IndexType, typename ValueType>
double test_bsr_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, int schedule_mod, double &convert_time);

template <typename IndexType, typename UIndexType, typename ValueType>
double test_csr5_matrix_kernels(const CSR_Matrix<IndexType, ValueType> &csr_ref, int kernel_tag, int schedule_mod);

/**
 * @brief Input CSR format for reference. Inside we make an ELL matrix
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param csr 
 * @param kernel_tag 
 * @return double time in ms 
 */
template <typename IndexType, typename ValueType>
double test_ell_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, LeadingDimension ld, int schedule_mod,  double &convert_time);

template <typename IndexType, typename ValueType>
double test_dia_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, int schedule_mod, double &convert_time);

/**
 * @brief Input CSR format for reference. Inside we make an S_ELL matrix for testing
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param csr_ref 
 * @param kernel_tag 
 * @return int 
 */
template <typename IndexType, typename ValueType>
double test_s_ell_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, int schedule_mod, double &convert_time);

template <typename IndexType, typename ValueType>
double test_sell_c_sigma_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, int schedule_mod, double &convert_time);

template <typename IndexType, typename ValueType>
double test_sell_c_R_matrix_kernels(const CSR_Matrix<IndexType,ValueType> &csr_ref, int kernel_tag, int schedule_mod);

#endif /* SPMV_TESTROUTINE_H */
