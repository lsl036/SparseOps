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

/**
 * @file sparse_format.h
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief Definition of Sparse Matrix Format.
 * @version 0.1
 * @date 2023-11-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#ifndef SPARSE_FORMAT_H
#define SPARSE_FORMAT_H

#include"memopt.h"
#include<vector>

/* Leading-dimension */
typedef enum
{
    RowMajor = 0, /* C-style */
    ColMajor = 1  /* Fortran-style */
} LeadingDimension;

/**
 * @brief General sparse matrix infos
 *        Basic features: rows, cols, nnzs, and sparsity
 *        Skew  features:
 *        Locality feas :
 *      
 *        Performance : time, GFLOPS
 * @tparam IndexType 
 */
template <typename IndexType>
struct Matrix_Features
{
    typedef IndexType index_type;

    // basic features
    IndexType num_rows;
    IndexType num_cols;
    IndexType num_nnzs;

    double sparsity;

    // partition by number of rows_nnz balance
    IndexType *partition = nullptr;  // size: thread_num + 1 

    // Performance Statistics
    double time, gflops, gbytes;
    int tag;
    int kernel_flag;
};

/**
 * @brief COOrdinate Matrix format (Triplet format)
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 */
template <typename IndexType, typename ValueType>
struct COO_Matrix : public Matrix_Features<IndexType>
{
    typedef IndexType index_type;
    typedef ValueType value_type;

    IndexType *row_index;
    IndexType *col_index;
    ValueType *values;
};
// template struct COO_Matrix<int, float>;
// template struct COO_Matrix<int, double>;

/**
 * @brief Compressed Sparse Row Matrix Format
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 */
template <typename IndexType, typename ValueType>
struct CSR_Matrix : public Matrix_Features<IndexType>
{
    typedef IndexType index_type;
    typedef ValueType value_type;

    IndexType *row_offset;
    IndexType *col_index;
    ValueType *values;
};

/**
 * @brief Blocked Compressed Sparse Row Matrix Format
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 */
template <typename IndexType, typename ValueType>
struct BSR_Matrix : public Matrix_Features<IndexType>
{
    typedef IndexType index_type;
    typedef ValueType value_type;

    IndexType blockDim_r;       // row dims per block
    IndexType blockDim_c;       // col dims per block, which should utilize vectorization length.
    IndexType blockNNZ;  // blockDimRow * blockDimCol

    IndexType mb;         // number of row blocks in matrix
    IndexType nb;         // number of col blocks in matrix
    IndexType nnzb;       // number of none zero block

    IndexType* row_ptr;
    IndexType* block_colindex;
    ValueType* block_data;      // store the nnzs of blocks in row major
};


/**
 * @brief CSR5 Matrix Format from Weifeng Liu
 *  < Liu, Weifeng, and Brian Vinter. "CSR5: An efficient storage format for cross-platform sparse matrix-vector 
 *  multiplication." Proceedings of the 29th ACM on International Conference on Supercomputing. 2015. >
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 */
template <typename IndexType, typename UIndexType, typename ValueType>
struct CSR5_Matrix : public CSR_Matrix<IndexType, ValueType>
{
    // column major for col_index tile and values tile.
    //  TILE size: SIGMA x OMEGA 

    IndexType sigma;  // Typically, sigma = SIMD_WIDTH / sizeof(ValueType)
    IndexType omega;  // Tunable parameter, number of rows within a "sigma" segment

    IndexType bit_y_offset;         //  y_offset 要多少bits   max: log(omega)
    IndexType bit_scansum_offset;   //  seg_offset 要多少bits max: log(omega)
    IndexType num_packets;          // tile中的一列元素需要多少个单位 tile_desc 的bit来存, 通常32bit 的一个 tile_desc 足够存放 y_offset + seg_offset + big_flag
    IndexType num_offsets;
    IndexType _p;                   // 依据 nnz 划分的 tiles 数目
    IndexType tail_partition_start; // tiles 外余下的 nnz 起始

    // 指示每个tile的第一个元素在 row_offset 中所在的行索引， 
    // 如果 tile_ptr 中的 tile_id 为负数，说明这个tile中包含了空行
    UIndexType *tile_ptr;                // length = _p + 1
    UIndexType *tile_desc;               // opt: CSR5 tile descriptor CPU case

    IndexType *tile_desc_offset_ptr;    // opt: CSR5 tile descriptor offset pointer CPU case
    IndexType *tile_desc_offset;        // opt: CSR5 tile descriptor offset CPU case
    ValueType *calibrator;              // opt: CSR5 calibrator CPU case
};

/**
 * @brief DIA format struct of sparse matrix
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 */
template <typename IndexType, typename ValueType>
struct DIA_Matrix : public Matrix_Features<IndexType>
{
    typedef IndexType index_type;
    typedef ValueType value_type;

    IndexType stride;  // diag 存储按照 alignment 对齐后，访问下一条对角线的stride
    IndexType complete_ndiags;  // 存共有 complete_ndiags 条对角线

    long int  * diag_offsets;  //diagonal offsets (must be a signed type)
    ValueType * diag_data;     //nonzero values stored in a (dia.complete_ndiags * dia.stride) matrix 
};

/**
 * @brief ELLPACK Sparse Matrix Format (row_major default)
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 */
template <typename IndexType, typename ValueType>
struct ELL_Matrix : public Matrix_Features<IndexType>
{
    typedef IndexType index_type;
    typedef ValueType value_type;

    LeadingDimension ld;

    IndexType max_row_width; // 单行最大长度
    IndexType min_row_width; // 单行最小长度

    // memory require: num_rows * max_row_width
    IndexType *col_index; // 先默认 row_major 存储
    ValueType *values;
};

/**
 * @brief Sliced ELLPACK Sparse Matrix Format
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 */
template <typename IndexType, typename ValueType>
struct S_ELL_Matrix : public Matrix_Features<IndexType>
{
    typedef IndexType index_type;
    typedef ValueType value_type;
    
    IndexType sliceWidth;         // 代表一个 chunk 内的总行数， 默认设置为c=CHUNK_SIZE
    IndexType chunk_num;          // chunk 的 总数目，按行切分后每个块内实现ELL
    IndexType alignment;          // 每个chunk 内对齐的长度

    // 每个chunk的 最大/小 长度
    // std::vector<IndexType> max_row_width; // length = chunk_num, 每个 width必须是 alignment 的整数倍
    // std::vector<IndexType> min_row_width; // length = chunk_num, 每个 width必须是 alignment 的整数倍
    // 好像只需要记录每一个chunk 的行width就足够了，因为sell里面每个chunk内部都是一样宽的
    // std::vector<IndexType> row_width;       // length = chunk_num, 每个 width必须是 alignment 的整数倍
    IndexType * row_width;

    // 默认按照行优先存储
    // std::vector<std::vector<IndexType>> col_index; // col_index[chunk_num][c * row_width[chunk_id]]
    // std::vector<std::vector<ValueType>> values; // values[chunk_num][c * row_width[chunk_id]]
    // TODO: 将两个数组 col_index 和 values 改成 二维指针进行内存对齐
    IndexType ** col_index;
    ValueType ** values;
};

/**
 * @brief SELL-c-sigma Sparse Matrix Format
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 */
template <typename IndexType, typename ValueType>
struct SELL_C_Sigma_Matrix : public Matrix_Features<IndexType>
{
    typedef IndexType index_type;
    typedef ValueType value_type;
    
    IndexType sliceWidth_Sigma;     // 重排序的分段宽度 Height of each slice (number of rows grouped together for reordering)
    IndexType chunkWidth_C;         // The width of a chunk (number of sigma segments), inside a chunk, rows have the same number of columns

    IndexType sliceNum;             // Total number of slices
    IndexType chunkNum;             // Total number of chunks
    IndexType validchunkNum;        // valid number of chunks (contains nnz is valid)
    IndexType chunk_num_per_slice;  // Number of chunks per slice
    IndexType alignment;            // Memory alignment for SIMD

    IndexType *reorder;             // Reordering of rows within each slice (length = num_rows)
    IndexType *chunk_len;           // Number of elements in each chunk (length = validchunkNum)

    // 默认按照行优先存储
    IndexType ** col_index;         // Column indices for non-zero values, per chunk
    ValueType ** values;            // Non-zero values, per chunk

    // Extra
    // IndexType *chunkLengths;        // Actual number of non-zero elements in each row within the chunk
    // IndexType *slice_ptr;           // Points to the beginning of each slice in the values array (length = sliceNum + 1)
    // IndexType *sliceOffsets;        // Starting point of each slice in the chunk_len array  (length = sliceNum + 1)
    // IndexType *sliceDesc;           // Descriptor for each slice, could include max NNZ per row in the slice
};


template <typename IndexType, typename ValueType>
struct SELL_C_R_Matrix : public Matrix_Features<IndexType>
{
    typedef IndexType index_type;
    typedef ValueType value_type;
    
    // IndexType sliceWidth_Sigma;     // 重排序的分段宽度 Height of each slice 现在固定为 rows 数目
    IndexType chunkWidth_C;         // The width of a chunk (number of sigma segments), inside a chunk, rows have the same number of columns

    // IndexType sliceNum;             // slice 就是一片完整的矩阵
    // IndexType chunkNum;             // Total number of chunks
    IndexType validchunkNum;        // valid number of chunks 现在和chunkNum相等
    // IndexType chunk_num_per_slice;  // Number of chunks per slice 不再需要
    IndexType alignment;            // Memory alignment for SIMD

    IndexType *reorder;             // Reordering of rows within each slice (length = num_rows)
    IndexType *chunk_len;           // Number of elements in each chunk (length = validchunkNum)

    // 默认按照行优先存储
    IndexType ** col_index;         // Column indices for non-zero values, per chunk
    ValueType ** values;            // Non-zero values, per chunk
};

////////////////////////////////////////////////////////////////////////////////
// Delete the memory usage of different Matrix struct
////////////////////////////////////////////////////////////////////////////////
template <typename IndexType, typename ValueType>
void delete_coo_matrix(COO_Matrix<IndexType,ValueType>& coo){
    delete_array(coo.row_index);
    delete_array(coo.col_index);
    delete_array(coo.values);
}
// template void delete_coo_matrix<int,float>(COO_Matrix<int,float>& coo);
// template void delete_coo_matrix<int,double>(COO_Matrix<int,double>& coo);

template <typename IndexType, typename ValueType>
void delete_csr_matrix(CSR_Matrix<IndexType,ValueType>& csr){
    delete_array(csr.row_offset);
    delete_array(csr.col_index);
    delete_array(csr.values);
}

template <typename IndexType, typename ValueType>
void delete_bsr_matrix(BSR_Matrix<IndexType,ValueType>& bsr){
    bsr.blockDim_c = 0;
    bsr.blockDim_r = 0;
    delete_array(bsr.row_ptr);
    delete_array(bsr.block_colindex);
    delete_array(bsr.block_data);
}

template <typename IndexType, typename UIndexType, typename ValueType>
void delete_csr5_matrix(CSR5_Matrix<IndexType,UIndexType,ValueType>& csr5){
    delete_array(csr5.row_offset);
    delete_array(csr5.col_index);
    delete_array(csr5.values);
    delete_array(csr5.tile_ptr);
    delete_array(csr5.tile_desc);
    delete_array(csr5.tile_desc_offset_ptr);
    delete_array(csr5.tile_desc_offset);
    delete_array(csr5.calibrator);
}

template <typename IndexType, typename ValueType>
void delete_dia_matrix(DIA_Matrix<IndexType,ValueType>& dia){
    delete_array(dia.diag_offsets);
    delete_array(dia.diag_data);
}

template <typename IndexType, typename ValueType>
void delete_ell_matrix(ELL_Matrix<IndexType,ValueType>& ell){
    ell.max_row_width = 0;
    ell.min_row_width = 0;
    
    delete_array(ell.col_index);
    delete_array(ell.values);
}

/**
 * @brief clear the SELL matrix by vector destructor.
 *        New format writting by C++11 standard and delete the memory by    
 *        clear().
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param s_ell 
 */
template <typename IndexType, typename ValueType>
void delete_s_ell_matrix(S_ELL_Matrix<IndexType,ValueType>& s_ell){
    s_ell.alignment = 0;

    // s_ell.row_width.clear();
    // s_ell.col_index.clear();
    // s_ell.values.clear(); 

    // for new struct
    delete_array(s_ell.row_width);
    for (IndexType chunk = 0; chunk < s_ell.chunk_num; chunk++)
    {
        delete_array(s_ell.col_index[chunk]);
        delete_array(s_ell.values[chunk]);
    }
    delete[] s_ell.col_index;
    delete[] s_ell.values;
    s_ell.chunk_num = 0;
}

template <typename IndexType, typename ValueType>
void delete_s_ell_c_sigma_matrix(SELL_C_Sigma_Matrix<IndexType,ValueType>& s_ell_c_sigma){
    s_ell_c_sigma.alignment = 0;
    
    s_ell_c_sigma.chunk_num_per_slice = 0;

    delete_array(s_ell_c_sigma.reorder);
    delete_array(s_ell_c_sigma.chunk_len);

    for (IndexType chunkID = 0; chunkID < s_ell_c_sigma.validchunkNum; chunkID++)
    {
        delete_array(s_ell_c_sigma.col_index[chunkID]);
        delete_array(s_ell_c_sigma.values[chunkID]);
    }
    delete[] s_ell_c_sigma.col_index;
    delete[] s_ell_c_sigma.values;
    s_ell_c_sigma.sliceNum  = 0;
    s_ell_c_sigma.chunkNum  = 0;
    s_ell_c_sigma.validchunkNum = 0;
    // delete_array(s_ell_c_sigma.slice_ptr);
    // delete_array(s_ell_c_sigma.sliceOffsets);
}

template <typename IndexType, typename ValueType>
void delete_s_ell_c_R_matrix(SELL_C_R_Matrix<IndexType,ValueType>& s_ell_c_R){
    s_ell_c_R.alignment = 0;

    delete_array(s_ell_c_R.reorder);
    delete_array(s_ell_c_R.chunk_len);

    for (IndexType chunkID = 0; chunkID < s_ell_c_R.validchunkNum; chunkID++)
    {
        delete_array(s_ell_c_R.col_index[chunkID]);
        delete_array(s_ell_c_R.values[chunkID]);
    }
    delete[] s_ell_c_R.col_index;
    delete[] s_ell_c_R.values;
    s_ell_c_R.validchunkNum  = 0;
}
////////////////////////////////////////////////////////////////////////////////
// Delete Matrix struct
////////////////////////////////////////////////////////////////////////////////

template <typename IndexType, typename ValueType>
void delete_host_matrix(COO_Matrix<IndexType,ValueType>& coo){ delete_coo_matrix(coo); }

template <typename IndexType, typename ValueType>
void delete_host_matrix(CSR_Matrix<IndexType,ValueType>& csr){ delete_csr_matrix(csr); }

template <typename IndexType, typename ValueType>
void delete_host_matrix(BSR_Matrix<IndexType,ValueType>& bsr){ delete_bsr_matrix(bsr); }

template <typename IndexType, typename UIndexType, typename ValueType>
void delete_host_matrix(CSR5_Matrix<IndexType,UIndexType,ValueType>& csr5){ delete_csr5_matrix(csr5); }

template <typename IndexType, typename ValueType>
void delete_host_matrix(DIA_Matrix<IndexType,ValueType>& dia){ delete_dia_matrix(dia); }

template <typename IndexType, typename ValueType>
void delete_host_matrix(ELL_Matrix<IndexType,ValueType>& ell){ delete_ell_matrix(ell); }

template <typename IndexType, typename ValueType>
void delete_host_matrix(S_ELL_Matrix<IndexType,ValueType>& s_ell){ delete_s_ell_matrix(s_ell); }

template <typename IndexType, typename ValueType>
void delete_host_matrix(SELL_C_Sigma_Matrix<IndexType,ValueType>& s_ell_c_sigma){ delete_s_ell_c_sigma_matrix(s_ell_c_sigma); }

template <typename IndexType, typename ValueType>
void delete_host_matrix(SELL_C_R_Matrix<IndexType,ValueType>& s_ell_c_R){ delete_s_ell_c_R_matrix(s_ell_c_R); }

#endif /* SPARSE_FORMAT_H */
