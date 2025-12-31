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
#ifndef SPARSE_IO_H
#define SPARSE_IO_H
// #include <stdio.h>
// #include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "mmio.h"
#include "plat_config.h"
#include "general_config.h"
#include "sparse_format.h"
#include "sparse_conversion.h"

std::string extractFileNameWithoutExtension(const std::string& filePath);

/**
 * @brief Read sparse matrix in COO format from ".mtx" format file.
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param mm_filename : The sparse matrix file, must in mtx format.
 * @return COO_Matrix<IndexType,ValueType> 
 */
template <class IndexType,class ValueType>
COO_Matrix<IndexType,ValueType> read_coo_matrix(const char * mm_filename);

/**
 * @brief Read sparse matrix in CSR format from ".mtx" format file.
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param mm_filename The sparse matrix file, must in mtx format.
 * @param compact     Judge whether sum duplicates together in CSR or not
 * @return CSR_Matrix<IndexType, ValueType> 
 */
template <class IndexType, class ValueType>
CSR_Matrix<IndexType, ValueType> read_csr_matrix(const char * mm_filename, bool compact = false);

/**
 * @brief Read sparse matrix in BSR format from ".mtx" format file.
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param mm_filename The sparse matrix file, must in mtx format.
 * @param blockDimRow Row nums of each block
 * @param blockDimCol Col nums of each block
 * @return BSR_Matrix<IndexType, ValueType> 
*/
template <class IndexType, class ValueType>
BSR_Matrix<IndexType, ValueType> read_bsr_matrix(const char * mm_filename, const IndexType blockDimRow = BSR_BlockDimRow, const IndexType blockDimCol = (SIMD_WIDTH/8/sizeof(ValueType)));

/**
 * @brief Read sparse matrix in CSR5 format from ".mtx" format file.
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param mm_filename 
 * @return CSR_Matrix<IndexType, ValueType> 
 */
template <class IndexType, class UIndexType, class ValueType>
CSR5_Matrix<IndexType, UIndexType, ValueType> read_csr5_matrix(const char * mm_filename);

/**
 * @brief Read sparse matrix in ELL format from ".mtx" format file.
 *        Convert from COO format.
 * @tparam IndexType 
 * @tparam ValueType 
 * @param mm_filename The sparse matrix file, must in mtx format.
 * @param ld          ELL matrix prefered leading dimension
 * @return ELL_Matrix<IndexType, ValueType> 
 */
template <class IndexType, class ValueType>
ELL_Matrix<IndexType, ValueType> read_ell_matrix(const char * mm_filename, LeadingDimension ld = RowMajor);

/**
 * @brief Read sparse matrix in DIA format from ".mtx" format file.
 *        Convert from CSR format.
 * @tparam IndexType 
 * @tparam ValueType 
 * @param mm_filename   The sparse matrix file, must in mtx format.
 * @param max_diags     The maximum number of diags, suppose 2048 is enough for sp matrix
 * @param alignment     aligment for DIA, generally use ALIGNMENT_NUM
 * @return DIA_Matrix<IndexType, ValueType> 
 */
template <class IndexType, class ValueType>
DIA_Matrix<IndexType, ValueType> read_dia_matrix(const char * mm_filename, const IndexType max_diags, const IndexType alignment = (SIMD_WIDTH/8/sizeof(ValueType)));

/**
 * @brief Read sparse matrix in Sliced_ELL format from ".mtx" format file.
 *        Convert from CSR format.
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param mm_filename 
 * @param chunkwidth 
 * @param alignment 
 * @return S_ELL_Matrix<IndexType, ValueType> 
 */
template <class IndexType, class ValueType>
S_ELL_Matrix<IndexType, ValueType> read_sell_matrix(const char * mm_filename, const int chunkwidth, const IndexType alignment);

template <class IndexType, class ValueType>
SELL_C_Sigma_Matrix<IndexType, ValueType> read_sell_c_sigma_matrix(const char * mm_filename, const int slicewidth, const int chunkwidth, const IndexType alignment);

template <class IndexType, class ValueType>
SELL_C_R_Matrix<IndexType, ValueType> read_sell_c_R_matrix(const char * mm_filename, const int chunkwidth, const IndexType alignment);

#endif /* SPARSE_IO_H */
