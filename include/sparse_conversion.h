#ifndef SPARSE_CONVERSION_H
#define SPARSE_CONVERSION_H

#include"sparse_format.h"
#include"sparse_operation.h"
#include"general_config.h"
#include"plat_config.h"
#include"thread.h"
#include"memopt.h"
#include"timer.h"
#include"csr5_utils.h"
#include <vector>
#include <algorithm>
#include <stdexcept>

// 宏，用于传递当前的函数名、文件名和行号
#define CHECK_ALLOC(ptr) checkAlloc((ptr), __FUNCTION__, __FILE__, __LINE__)

// 用于检查指针是否为 nullptr
inline void checkAlloc(const void* ptr, const char* func, const char* file, int line) {
    if (ptr == nullptr) {
        std::cerr << "Memory allocation failed in " << func
                  << " (" << file << ":" << line << ")" << std::endl;
        // 抛出异常或进行其他错误处理
        throw std::bad_alloc();
    }
}

template <class IndexType, class ValueType>
CSR_Matrix<IndexType, ValueType> coo_to_csr( const COO_Matrix<IndexType, ValueType> &coo, bool compact = false)
{
    CSR_Matrix<IndexType, ValueType> csr;

    csr.num_rows = coo.num_rows;
    csr.num_cols = coo.num_cols;
    csr.num_nnzs = coo.num_nnzs;

    csr.tag = 0;
    csr.row_offset = new_array<IndexType> (csr.num_rows + 1);
    CHECK_ALLOC(csr.row_offset);
    csr.col_index  = new_array<IndexType> (csr.num_nnzs);
    CHECK_ALLOC(csr.col_index);
    csr.values     = new_array<ValueType> (csr.num_nnzs);
    CHECK_ALLOC(csr.values);
    //========== Rowoffset calculation ==========
    for (IndexType i = 0; i < csr.num_rows; i++){
        csr.row_offset[i] = 0;
    }

    // Get each row's nnzs
    for (IndexType i = 0; i < csr.num_nnzs; i++){
        csr.row_offset[ coo.row_index[i] ]++;
    }

    //  sum to get row_offset
    for(IndexType i = 0, cumsum = 0; i < csr.num_rows; i++){
        IndexType temp = csr.row_offset[i];
        csr.row_offset[i] = cumsum;
        cumsum += temp;
    }
    csr.row_offset[csr.num_rows] = csr.num_nnzs;

    // ========== write col_index and values ==========
    for (IndexType i = 0; i < csr.num_nnzs; i++){
        IndexType rowIndex  = coo.row_index[i];
        IndexType destIndex = csr.row_offset[rowIndex];

        csr.col_index[destIndex] = coo.col_index[i];
        csr.values[destIndex]    = coo.values[i];
    
        csr.row_offset[rowIndex]++;  // row_offset move behind
    }

    // Restore the row_offset
    for(IndexType i = 0, last = 0; i <= csr.num_rows; i++){
        IndexType temp = csr.row_offset[i];
        csr.row_offset[i] = last;
        last = temp;
    }

    // ========== Compact Situation ==========
    if(compact) {
        //sum duplicates together 是累加！！
        sum_csr_duplicates(csr.num_rows, csr.num_cols, 
                           csr.row_offset, csr.col_index, csr.values);

        csr.num_nnzs = csr.row_offset[csr.num_rows];
    }
    
    return csr;
}

template <class IndexType, class ValueType>
ELL_Matrix<IndexType, ValueType> coo_to_ell( const COO_Matrix<IndexType, ValueType> &coo, const LeadingDimension ld = RowMajor)
{
    ELL_Matrix<IndexType,ValueType> ell;

    ell.num_rows = coo.num_rows;
    ell.num_cols = coo.num_cols;
    ell.num_nnzs = coo.num_nnzs;

    ell.tag = 0;
    ell.ld = RowMajor;
    std::vector<IndexType> rowCounts (ell.num_rows, 0);

    for (IndexType i = 0; i < coo.num_nnzs; i++){
        rowCounts[coo.row_index[i]]++;
    }

    ell.max_row_width = *std::max_element(rowCounts.begin(), rowCounts.end());
    // 分配矩阵空间
    ell.col_index = new_array<IndexType> ((size_t) ell.num_rows * ell.max_row_width);
    CHECK_ALLOC(ell.col_index);
    ell.values    = new_array<ValueType> ((size_t) ell.num_rows * ell.max_row_width);
    CHECK_ALLOC(ell.values);
    // 初始化ELL格式的数组
    std::fill_n(ell.col_index, ell.num_rows*ell.max_row_width, static_cast<IndexType> (-1)); // 使用 -1 作为填充值，因为它不是有效的列索引
    std::fill_n(ell.values, ell.num_rows*ell.max_row_width, static_cast<ValueType> (0)); // 零填充values

    // 用COO数据填充ELL数组
    std::vector<IndexType> currentPos(ell.num_rows, 0); // 跟踪每行当前填充位置
    if (ColMajor == ell.ld){
        for (size_t i = 0; i < ell.num_nnzs; ++i) {
            size_t row = coo.row_index[i];
            size_t pos = row + (size_t) currentPos[row] * ell.num_rows;
            ell.values[pos] = coo.values[i];
            ell.col_index[pos] = coo.col_index[i];
            currentPos[row]++;
        }
    }
    else if (RowMajor == ell.ld){
        for (size_t i = 0; i < ell.num_nnzs; ++i) {
            size_t row = coo.row_index[i];
            size_t pos = (size_t) row * ell.max_row_width + currentPos[row];
            ell.values[pos] = coo.values[i];
            ell.col_index[pos] = coo.col_index[i];
            currentPos[row]++;
        }
    }

    return ell;
}

template <class IndexType, class ValueType>
COO_Matrix<IndexType, ValueType> csr_to_coo( const CSR_Matrix<IndexType, ValueType> &csr)
{
    COO_Matrix<IndexType, ValueType> coo;

    coo.num_rows = csr.num_rows;
    coo.num_cols = csr.num_cols;
    coo.num_nnzs = csr.num_nnzs;

    coo.row_index  = new_array<IndexType> (coo.num_nnzs);
    CHECK_ALLOC(coo.row_index);
    coo.col_index  = new_array<IndexType> (coo.num_nnzs);
    CHECK_ALLOC(coo.col_index);
    coo.values     = new_array<ValueType> (coo.num_nnzs);
    CHECK_ALLOC(coo.values);
    // 转换，按 row index 递增顺序来存 COO
    for (IndexType row = 0; row < coo.num_rows; ++row) {
        for (IndexType i = csr.row_offset[row]; i < csr.row_offset[row + 1]; ++i) {
            coo.row_index[i] = row;             // 行索引
            coo.col_index[i] = csr.col_index[i];    // 列索引
            coo.values[i] = csr.values[i];          // 非零值
        }
    }
    return coo;
}

/**
 * @brief Create the ELL format matrix from CSR format in row-major
 *        This routine do not delete the CSR_Matrix handle
 * 
 *         x  0  x  0  x   line: 0                      csr.row_offset = [0, 3, 5, 8, 12, 15] 
 *         x  x  0  0  0         1               csr.col = [0,2,4, 0,1, 0,3,4, 1,2,3,4, 1,2,4]
 *   A =   x  0  0  x  x         2              max_nnz_per_row = 4, row_num = 5
 *         0  x  x  x  x         3         ell.col_index = [0,0,0,1,1,  2,1,3,2,2,  4,-,4,3,4,  -,-,-,4,-]   col-major
 *         0  x  x  0  x         4         ell.col_index = [0,2,4,-,  0,1,-,-,  0,3,4,-,  1,2,3,4,  1,2,4,-] row-major
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param csr 
 * @return ELL_Matrix<IndexType, ValueType> 
 */
// 可以在特征分析中加入 引入多少非零元
template <class IndexType, class ValueType>
ELL_Matrix<IndexType, ValueType> csr_to_ell(const CSR_Matrix<IndexType, ValueType> &csr, const LeadingDimension ld = RowMajor)
{
    ELL_Matrix<IndexType,ValueType> ell;

    ell.num_rows = csr.num_rows;
    ell.num_cols = csr.num_cols;
    ell.num_nnzs = csr.num_nnzs;

    ell.tag = 0;
    ell.ld = ld;
    IndexType max_nnz_per_row = 0;
    //计算每行的非零元数量并找到最大值
    for (size_t i = 0; i < csr.num_rows; i++)
    {
        max_nnz_per_row = std::max( max_nnz_per_row, csr.row_offset[i+1] - csr.row_offset[i]);
    }
    ell.max_row_width = max_nnz_per_row;

    // 分配矩阵空间
    ell.col_index = new_array<IndexType> ((size_t) ell.num_rows * ell.max_row_width);
    CHECK_ALLOC(ell.col_index);
    ell.values    = new_array<ValueType> ((size_t) ell.num_rows * ell.max_row_width);
    CHECK_ALLOC(ell.values);

    // 初始化ELL格式的数组
    std::fill_n(ell.col_index, ell.num_rows*ell.max_row_width, static_cast<IndexType> (-1)); // 使用 -1 作为填充值，因为它不是有效的列索引
    std::fill_n(ell.values, ell.num_rows*ell.max_row_width, static_cast<ValueType> (0)); // 零填充values

    if (ColMajor == ld)
    {
        // 给ELL格式的两个数组进行赋值, col-major
        #pragma omp parallel for
        for (size_t rowId = 0; rowId < ell.num_rows; ++rowId)
        {
            size_t ellIndex = rowId;
            // 遍历 CSR row_ptr中 这行的所有非零元素
            for (size_t csrIndex = csr.row_offset[rowId]; csrIndex < csr.row_offset[rowId+1]; ++csrIndex)
            {
                ell.col_index[ellIndex] = csr.col_index[csrIndex];
                ell.values[ellIndex]    =    csr.values[csrIndex];
                ellIndex += ell.num_rows;
            }
        }
    }
    else if (RowMajor == ld)
    {
        // 给ELL格式的两个数组进行赋值, row-major
        #pragma omp parallel for
        for (size_t rowId = 0; rowId < ell.num_rows; ++rowId)
        {
            size_t ellIndex = rowId * ell.max_row_width;
            // 遍历 CSR row_ptr中 这行的所有非零元素
            for (size_t csrIndex = csr.row_offset[rowId]; csrIndex < csr.row_offset[rowId+1]; ++csrIndex)
            {
                ell.col_index[ellIndex] = csr.col_index[csrIndex];
                ell.values[ellIndex]    =    csr.values[csrIndex];
                ellIndex ++;
            }   
        }
    }
    return ell;
}

/**
 * @brief CSR to S_ELL format conversion
 *        Alignment in AVX512: float should be 4 bytes * 16 = 64 bytes. 
 *                             double should be 8 bytes * 8 = 64 bytes. 
 * @tparam IndexType 
 * @tparam ValueType 
 */
template <class IndexType, class ValueType>
S_ELL_Matrix<IndexType, ValueType> csr_to_sell(const CSR_Matrix<IndexType, ValueType> &csr, FILE *fp_feature, const int chunkwidth = CHUNK_SIZE,  const IndexType alignment = (SIMD_WIDTH/8/sizeof(ValueType)))
{
    S_ELL_Matrix<IndexType, ValueType> sell;

    sell.num_rows = csr.num_rows;
    sell.num_cols = csr.num_cols;
    sell.num_nnzs = csr.num_nnzs;

    sell.tag = 0;

    sell.sliceWidth = chunkwidth;
    sell.alignment  = alignment;

    // 确定需要多少个slice/chunk
    sell.chunk_num = (csr.num_rows + sell.sliceWidth - 1) / sell.sliceWidth; // 分块数向上取整

    // sell.row_width.resize (sell.chunk_num, 0);
    sell.row_width = new_array<IndexType>(sell.chunk_num);
    CHECK_ALLOC(sell.row_width);
    memset(sell.row_width, 0 , sell.chunk_num * sizeof(IndexType));

    // #pragma omp parallel for
    for (IndexType row = 0; row < csr.num_rows; ++row) {
        IndexType chunk_id = row / sell.sliceWidth;
        IndexType row_nnz = csr.row_offset[row + 1] - csr.row_offset[row];
        sell.row_width[chunk_id] = std::max(sell.row_width[chunk_id], row_nnz);
    }
    // 对每个chunk的最大行宽度进行对齐
    // for (IndexType& width : sell.row_width) {
    //     width = ((width + sell.alignment - 1) / sell.alignment) * sell.alignment;
    // }
    #pragma omp parallel for
    for (IndexType i = 0; i < sell.chunk_num; i++)
    {
        sell.row_width[i] = ((sell.row_width[i] + sell.alignment - 1) / sell.alignment) * sell.alignment;
    }
    
    // 为每个chunk的行指针数组分配内存
    sell.col_index = new IndexType*[sell.chunk_num];
    sell.values = new ValueType*[sell.chunk_num];
    for (IndexType chunk = 0; chunk < sell.chunk_num; ++chunk) {
        size_t elem_nums = sell.row_width[chunk] * sell.sliceWidth;

        sell.col_index[chunk] = new_array<IndexType> (elem_nums);
        CHECK_ALLOC(sell.col_index[chunk]);
        // 初始化col_index中的每个元素为-1
        std::fill_n(sell.col_index[chunk], elem_nums, static_cast<IndexType>(-1));

        sell.values[chunk] = new_array<ValueType> (elem_nums);
        CHECK_ALLOC(sell.values[chunk]);
        // 初始化values中的每个元素为0
        std::fill_n(sell.values[chunk], elem_nums, ValueType(0));
    }


    //转换 CSR 到 S-ELL
    #pragma omp parallel for
    for (IndexType row = 0; row < csr.num_rows; ++row)
    {
        IndexType chunk_id         = row / sell.sliceWidth;    // 所属的 chunk 号
        IndexType row_within_chunk = row % sell.sliceWidth; // chunk 内部的行号 0 ~ sliceWidth-1
        IndexType row_start        = csr.row_offset[row];
        IndexType row_end          = csr.row_offset[row+1];

        for (IndexType idx = row_start; idx < row_end; idx++)
        {
            IndexType col = csr.col_index[idx];
            ValueType val = csr.values[idx];

            IndexType pos = row_within_chunk * sell.row_width[chunk_id] + (idx - row_start);
            sell.col_index[chunk_id][pos] = col;
            sell.values[chunk_id][pos] = val;
        } 
    }
    
    return sell;
}

template <class IndexType, class ValueType>
SELL_C_Sigma_Matrix<IndexType, ValueType> csr_to_sell_c_sigma(const CSR_Matrix<IndexType, ValueType> &csr, FILE *fp_feature, const int slicewidth = SELL_SIGMA, const int chunkwidth = CHUNK_SIZE,  const IndexType alignment = (SIMD_WIDTH/8/sizeof(ValueType)))
{
    SELL_C_Sigma_Matrix<IndexType, ValueType> sell_c_sigma;

    sell_c_sigma.num_rows = csr.num_rows;
    sell_c_sigma.num_cols = csr.num_cols;
    sell_c_sigma.num_nnzs = csr.num_nnzs;

    sell_c_sigma.tag = 0;

    sell_c_sigma.sliceWidth_Sigma    = slicewidth;
    sell_c_sigma.chunkWidth_C        = chunkwidth;
    sell_c_sigma.alignment           = alignment;

    // 确定需要多少个slice
    sell_c_sigma.sliceNum = (sell_c_sigma.num_rows + sell_c_sigma.sliceWidth_Sigma -1) / sell_c_sigma.sliceWidth_Sigma;

    sell_c_sigma.validchunkNum = (sell_c_sigma.num_rows + sell_c_sigma.chunkWidth_C -1) / sell_c_sigma.chunkWidth_C;
    
    // chunk 无法整除 slice
    if (slicewidth % chunkwidth)
    {
        printf(" Sigma must be multiple divided by c\n");
        exit(-1);
    }
    else{
        sell_c_sigma.chunk_num_per_slice = slicewidth/ chunkwidth;
    }
    sell_c_sigma.chunkNum = sell_c_sigma.sliceNum * sell_c_sigma.chunk_num_per_slice;
    

    sell_c_sigma.reorder = new_array<IndexType>(sell_c_sigma.num_rows);
    CHECK_ALLOC(sell_c_sigma.reorder);
    /*-----------------------------------------------*/
    //  Step1. 确定重排序数组 
    /*-----------------------------------------------*/
    // Create a vector to hold the number of non-zeros per row and the original row index
    std::vector<std::pair<IndexType, IndexType>> nnz_count(csr.num_rows);

    // Count the non-zeros for each row
    #pragma omp parallel for
    for (IndexType i = 0; i < csr.num_rows; ++i) {
        nnz_count[i].first = csr.row_offset[i + 1] - csr.row_offset[i]; // Number of non-zeros
        nnz_count[i].second = i; // Original row index
    }

    #pragma omp parallel for
    for (IndexType slice = 0; slice < sell_c_sigma.sliceNum; slice++)
    {
        // Calculate the start and end row of the current slice
        IndexType start_row = slice * sell_c_sigma.sliceWidth_Sigma;
        IndexType end_row = std::min(start_row + sell_c_sigma.sliceWidth_Sigma, csr.num_rows);

        // Sort the rows in the slice by non-zero count
        std::sort(nnz_count.begin() + start_row, nnz_count.begin() + end_row,
            std::greater<std::pair<IndexType, IndexType>>());

        // Fill the sell_c_sigma.reorder array with the new order of the rows
        for (IndexType i = start_row; i < end_row; ++i) {
            // reorder[i] 保存放置在 重序后第 i 行的 原始行号是多少
            // SpMV时, 累加到y的原始行 y[reorder[i]] += A[i][column_id]*x[column_id]
            sell_c_sigma.reorder[i] = nnz_count[i].second;
        }
    }

    /*-----------------------------------------------*/
    //  Step2. 确定 chunk_len数组，计算每个chunk的列数
    /*-----------------------------------------------*/
    sell_c_sigma.chunk_len = new_array<IndexType> (sell_c_sigma.validchunkNum);
    CHECK_ALLOC(sell_c_sigma.chunk_len);
    // Initialize chunk_len to zeros
    std::fill_n(sell_c_sigma.chunk_len, sell_c_sigma.validchunkNum, 0);

    // Iterate through each row, now using the reorder mapping
    // #pragma omp parallel for
    for (IndexType row = 0; row < csr.num_rows; ++row){
        // get the real_rowID in Reorder array
        IndexType real_rowID = sell_c_sigma.reorder[row];

        IndexType nnzs_in_row = csr.row_offset[real_rowID + 1] - csr.row_offset[real_rowID];

        // Determine the sliceID and chunkID of the reordered row
        // IndexType sliceID = row / sell_c_sigma.sliceWidth_Sigma;
        // IndexType chunkID = (row % sell_c_sigma.sliceWidth_Sigma) / sell_c_sigma.chunkWidth_C;
        IndexType chunkID = row / sell_c_sigma.chunkWidth_C;

        // Update the chunk length
        // sell_c_sigma.chunk_len[sliceID * sell_c_sigma.chunk_num_per_slice + chunkID] = std::max( sell_c_sigma.chunk_len[sliceID * sell_c_sigma.chunk_num_per_slice + chunkID], nnzs_in_row );
        sell_c_sigma.chunk_len[chunkID] = std::max(sell_c_sigma.chunk_len[chunkID], nnzs_in_row);
    }
    
    // alignment for chunk_len
    #pragma omp parallel for
    for (IndexType i = 0; i < sell_c_sigma.validchunkNum; i++)
    {
        sell_c_sigma.chunk_len[i] = ((sell_c_sigma.chunk_len[i] + sell_c_sigma.alignment - 1)/ sell_c_sigma.alignment) * sell_c_sigma.alignment;
    }

    /*-----------------------------------------------*/
    //  Step3. 确定 col_index 和  values. 在计算时可以只看chunk了
    /*-----------------------------------------------*/
    sell_c_sigma.col_index = new IndexType*[sell_c_sigma.validchunkNum];
    sell_c_sigma.values    = new ValueType*[sell_c_sigma.validchunkNum];
    for (IndexType chunk = 0; chunk < sell_c_sigma.validchunkNum; chunk++)
    {
        size_t elem_nums = sell_c_sigma.chunk_len[chunk] * sell_c_sigma.chunkWidth_C;
        sell_c_sigma.col_index[chunk] = new_array<IndexType> (elem_nums);
        CHECK_ALLOC(sell_c_sigma.col_index[chunk]);
        // 初始化 col_index 中每个元素为 -1
        std::fill_n(sell_c_sigma.col_index[chunk], elem_nums, static_cast<IndexType>(-1));

        sell_c_sigma.values[chunk] = new_array<ValueType> (elem_nums);
        CHECK_ALLOC(sell_c_sigma.values[chunk]);
        // 初始化 values 中每个元素为 0 
        std::fill_n(sell_c_sigma.values[chunk], elem_nums, ValueType(0));
    }

    //转换 CSR 到 S-ELL-c-sigma
    #pragma omp parallel for
    for (IndexType row = 0; row < csr.num_rows; row++)
    {
        // get the row index with Reorder array
        IndexType real_rowID = sell_c_sigma.reorder[row];

        IndexType chunk_id = row / sell_c_sigma.chunkWidth_C; // 所属的 chunk 号
        IndexType row_within_chunk = row % sell_c_sigma.chunkWidth_C; // chunk 内部的行号 0 ~ C-1

        IndexType row_start        = csr.row_offset[real_rowID];
        IndexType row_end          = csr.row_offset[real_rowID + 1];

        for (IndexType idx = row_start; idx < row_end; idx++)
        {
            IndexType col = csr.col_index[idx];
            ValueType val = csr.values[idx];

            IndexType pos = row_within_chunk * sell_c_sigma.chunk_len[chunk_id] + (idx - row_start);
            sell_c_sigma.col_index[chunk_id][pos] = col;
            sell_c_sigma.values[chunk_id][pos]    = val;
        }
    }

    return sell_c_sigma;
}

// sell_c_sigma 的简化版， 重排序对完整的矩阵来做
template <class IndexType, class ValueType>
SELL_C_R_Matrix<IndexType, ValueType> csr_to_sell_c_R(const CSR_Matrix<IndexType, ValueType> &csr, FILE *fp_feature, const int chunkwidth = CHUNK_SIZE,  const IndexType alignment = (SIMD_WIDTH/8/sizeof(ValueType)))
{
    SELL_C_R_Matrix<IndexType, ValueType> sell_c_R;

    sell_c_R.num_rows = csr.num_rows;
    sell_c_R.num_cols = csr.num_cols;
    sell_c_R.num_nnzs = csr.num_nnzs;

    sell_c_R.tag = 0;

    sell_c_R.chunkWidth_C        = chunkwidth;
    sell_c_R.alignment           = alignment;

    sell_c_R.validchunkNum = (sell_c_R.num_rows + sell_c_R.chunkWidth_C -1) / sell_c_R.chunkWidth_C;

    sell_c_R.reorder = new_array<IndexType>(sell_c_R.num_rows);
    CHECK_ALLOC(sell_c_R.reorder);
    /*-----------------------------------------------*/
    //  Step1. 确定重排序数组 
    /*-----------------------------------------------*/
    // Create a vector to hold the number of non-zeros per row and the original row index
    std::vector<std::pair<IndexType, IndexType>> nnz_count(csr.num_rows);

    // Count the non-zeros for each row
    #pragma omp parallel for
    for (IndexType i = 0; i < csr.num_rows; ++i) {
        nnz_count[i].first = csr.row_offset[i + 1] - csr.row_offset[i]; // Number of non-zeros
        nnz_count[i].second = i; // Original row index
    }

    // Sort all rows of matrix by non-zero count
    std::sort(nnz_count.begin(), nnz_count.end(),
        std::greater<std::pair<IndexType, IndexType>>());

    // Fill the sell_c_sigma.reorder array with the new order of the rows
    #pragma omp parallel for
    for (IndexType i = 0; i < csr.num_rows; ++i) {
        // reorder[i] 保存放置在 重序后第 i 行的 原始行号是多少
        // SpMV时, 累加到y的原始行 y[reorder[i]] += A[i][column_id]*x[column_id]
        sell_c_R.reorder[i] = nnz_count[i].second;
    }

    /*-----------------------------------------------*/
    //  Step2. 确定 chunk_len数组，计算每个chunk的列数
    /*-----------------------------------------------*/
    sell_c_R.chunk_len = new_array<IndexType> (sell_c_R.validchunkNum);
    CHECK_ALLOC(sell_c_R.chunk_len);
    // Initialize chunk_len to zeros
    std::fill_n(sell_c_R.chunk_len, sell_c_R.validchunkNum, 0);

    // Iterate through each row, now using the reorder mapping
    // #pragma omp parallel for
    for (IndexType row = 0; row < csr.num_rows; ++row){
        // get the real_rowID in Reorder array
        IndexType real_rowID = sell_c_R.reorder[row];

        IndexType nnzs_in_row = csr.row_offset[real_rowID + 1] - csr.row_offset[real_rowID];

        IndexType chunkID = row / sell_c_R.chunkWidth_C;

        // Update the chunk length
        sell_c_R.chunk_len[chunkID] = std::max(sell_c_R.chunk_len[chunkID], nnzs_in_row);
    }
    
    // alignment for chunk_len
    #pragma omp parallel for
    for (IndexType i = 0; i < sell_c_R.validchunkNum; i++)
    {
        sell_c_R.chunk_len[i] = ((sell_c_R.chunk_len[i] + sell_c_R.alignment - 1)/ sell_c_R.alignment) * sell_c_R.alignment;
    }

    /*-----------------------------------------------*/
    //  Step3. 确定 col_index 和  values. 在计算时可以只看chunk了
    /*-----------------------------------------------*/
    sell_c_R.col_index = new IndexType*[sell_c_R.validchunkNum];
    sell_c_R.values    = new ValueType*[sell_c_R.validchunkNum];
    for (IndexType chunk = 0; chunk < sell_c_R.validchunkNum; chunk++)
    {
        size_t elem_nums = sell_c_R.chunk_len[chunk] * sell_c_R.chunkWidth_C;
        sell_c_R.col_index[chunk] = new_array<IndexType> (elem_nums);
        CHECK_ALLOC(sell_c_R.col_index[chunk]);
        // 初始化 col_index 中每个元素为 -1
        std::fill_n(sell_c_R.col_index[chunk], elem_nums, static_cast<IndexType>(-1));

        sell_c_R.values[chunk] = new_array<ValueType> (elem_nums);
        CHECK_ALLOC(sell_c_R.values[chunk]);
        // 初始化 values 中每个元素为 0 
        std::fill_n(sell_c_R.values[chunk], elem_nums, ValueType(0));
    }

    //转换 CSR 到 S-ELL-c-R
    #pragma omp parallel for
    for (IndexType row = 0; row < csr.num_rows; row++)
    {
        // get the row index with Reorder array
        IndexType real_rowID = sell_c_R.reorder[row];

        IndexType chunk_id = row / sell_c_R.chunkWidth_C; // 所属的 chunk 号
        IndexType row_within_chunk = row % sell_c_R.chunkWidth_C; // chunk 内部的行号 0 ~ C-1

        IndexType row_start        = csr.row_offset[real_rowID];
        IndexType row_end          = csr.row_offset[real_rowID + 1];

        for (IndexType idx = row_start; idx < row_end; idx++)
        {
            IndexType col = csr.col_index[idx];
            ValueType val = csr.values[idx];

            IndexType pos = row_within_chunk * sell_c_R.chunk_len[chunk_id] + (idx - row_start);
            sell_c_R.col_index[chunk_id][pos] = col;
            sell_c_R.values[chunk_id][pos]    = val;
        }
    }

    return sell_c_R;
}

/**
 * @brief CSR format to DIA format
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param csr 
 * @param max_diags 
 * @param fp_feature 
 * @param alignment 
 * @return DIA_Matrix<IndexType, ValueType> 
 */
template <class IndexType, class ValueType>
DIA_Matrix<IndexType, ValueType> csr_to_dia(const CSR_Matrix<IndexType, ValueType> &csr, const IndexType max_diags, FILE *fp_feature, const IndexType alignment = (SIMD_WIDTH/8/sizeof(ValueType)))
{
    DIA_Matrix<IndexType, ValueType> dia;

    dia.num_rows     = csr.num_rows;
    dia.num_cols     = csr.num_cols;
    dia.num_nnzs     = csr.num_nnzs;
    dia.diag_offsets = nullptr;
    dia.diag_data    = nullptr;
    dia.tag          = 0;

    // compute number of occupied diagonals and enumerate them
    IndexType complete_ndiags = 0;
    const IndexType unmarked = (IndexType) -1;

    IndexType* diag_map = new_array<IndexType> (dia.num_rows + dia.num_cols);
    CHECK_ALLOC(diag_map);
    std::fill(diag_map, diag_map + dia.num_rows + dia.num_cols, unmarked);

    // IndexType* diag_map_2 = new_array<IndexType> (dia.num_rows + dia.num_cols);
    // std::fill(diag_map_2, diag_map_2 + dia.num_rows + dia.num_cols, 0);

    for (size_t i = 0; i < dia.num_rows; i++)
    {
        //  遍历 csr 的第 i 行元素
        for (size_t jj = csr.row_offset[i]; jj < csr.row_offset[i+1]; jj++)
        {
            size_t j = csr.col_index[jj]; // j : 元素的列序号
            size_t map_index = (csr.num_rows - i) + j; //offset shifted by + num_rows

            if( diag_map[map_index] == unmarked)
            {
                diag_map[map_index] = complete_ndiags;
                complete_ndiags ++;
            }
            // diag_map_2[map_index] ++;
        }
    }

    // size_t j_ndiags = 0;
    // double ratio;
    // IndexType NTdiags = 0;
    // double* array_ndiags = new_array<double>(10);
    // std::fill(array_ndiags, array_ndiags + 10, 0.0);

    // for(size_t i = 0; i < dia.num_rows + dia.num_cols; ++i){
    //     //  此条对角线非空
    //     if( diag_map_2[i] != 0 )
    //     {
    //         j_ndiags ++;
    //         ratio = (double) diag_map_2[i] / csr.num_rows;

    //         if (ratio < 0.1 )
    //             array_ndiags[0] ++;
    //         else if (ratio < 0.2 )
    //             array_ndiags[1] ++;
    //         else if (ratio < 0.3 )
    //             array_ndiags[2] ++;
    //         else if (ratio < 0.4 )
    //             array_ndiags[3] ++;
    //         else if (ratio < 0.5 )
    //             array_ndiags[4] ++;
    //         else if (ratio < 0.6 )
    //             array_ndiags[5] ++;
    //         else if (ratio < 0.7 )
    //             array_ndiags[6] ++;
    //         else if (ratio < 0.8 )
    //             array_ndiags[7] ++;
    //         else if (ratio < 0.9 )
    //             array_ndiags[8] ++;
    //         else if (ratio <= 1.0 )
    //             array_ndiags[9] ++;

    //         if (ratio >= NTRATIO )
    //             NTdiags ++;
    //     }
    // }
    // assert( j_ndiags == complete_ndiags);
    // delete_array (diag_map_2);
// #ifdef COLLECT_FEATURES
//         fprintf(fp_feature, "Ndiags : %d\n", complete_ndiags );
// #endif

//     for ( int i=0; i<10; i++)
//     {
//         array_ndiags[i] /= complete_ndiags;
// // 对角线稠密范围
// #ifdef COLLECT_FEATURES
//           if ( i == 0 )
//             fprintf(fp_feature, "Num_diags ER in ( %d %%, %d %% ) : %lf \n", i*10, (i+1)*10, array_ndiags[i] );
//           else if ( i == 9 )
//             fprintf(fp_feature, "Num_diags ER in [ %d %%, %d %% ] : %lf \n", i*10, (i+1)*10, array_ndiags[i] );
//           else
//             fprintf(fp_feature, "Num_diags ER in [ %d %%, %d %% ) : %lf \n", i*10, (i+1)*10, array_ndiags[i] );
// #endif
//     }
    
// #ifdef COLLECT_FEATURES
//         // 达到 NT 比例的对角线占比
//         double NTdiags_ratio = (double) NTdiags/ complete_ndiags;
//         // DIA 格式下的稠密度
//         double ER_DIA = (double) dia.num_nnzs / (complete_ndiags * dia.num_rows);
//         fprintf(fp_feature, "NTdiags_ratio : %lf  ( TH is 0.6 )\n", NTdiags_ratio );
//         fprintf(fp_feature, "ER_DIA : %lf\n", ER_DIA );
// #endif
    // delete_array(array_ndiags);
    dia.complete_ndiags = complete_ndiags;

    if(complete_ndiags > max_diags)
    {
        if constexpr(std::is_same<IndexType, int>::value) {
            printf("\tNumber of diagonals (%d) excedes limit (%d)\n", dia.complete_ndiags, max_diags);
        }
        else if constexpr(std::is_same<IndexType, long long>::value) {
            printf("\tNumber of diagonals (%lld) excedes limit (%lld)\n", dia.complete_ndiags, max_diags);
        }
        // dia.num_rows     = 0;
        // dia.num_cols     = 0;
        // dia.num_nnzs     = 0;
        dia.stride       = 0; 
        dia.gflops	= 0;
        delete_array(diag_map);
        exit(1);
        // return dia;
    }

    // length of each diagonal in memory, 按照 alignment 对齐
    dia.stride = alignment * ((dia.num_rows + alignment - 1)/ alignment);

    dia.diag_offsets = new_array<long int>  ((size_t) dia.complete_ndiags);
    CHECK_ALLOC(dia.diag_offsets);
    dia.diag_data    = new_array<ValueType> ((size_t) dia.complete_ndiags * dia.stride);
    CHECK_ALLOC(dia.diag_data);

    std::fill(dia.diag_data, dia.diag_data + (size_t) dia.complete_ndiags * dia.stride, ValueType(0));

    for(size_t n = 0; n < dia.num_rows + dia.num_cols; n++)
        if(diag_map[n] != unmarked) // 算出offset
            dia.diag_offsets[diag_map[n]] = (long int) n - (long int) dia.num_rows;
        
    for (size_t i = 0; i < csr.num_rows; i++)
    {
        for(size_t jj = csr.row_offset[i]; jj < csr.row_offset[i+1]; jj++){
            size_t j = csr.col_index[jj];
            size_t map_index = (csr.num_rows - i) + j; //offset shifted by + num_rows
            size_t diag = diag_map[map_index];
            dia.diag_data[diag*dia.stride + i] = csr.values[jj];
        }
    }
    
    delete_array(diag_map);

    return dia;

}

template <class IndexType, class ValueType>
BSR_Matrix<IndexType, ValueType> csr_to_bsr(const CSR_Matrix<IndexType, ValueType> &csr, const IndexType blockDimRow = BSR_BlockDimRow, IndexType blockDimCol = (SIMD_WIDTH/8/sizeof(ValueType)))
{
    BSR_Matrix<IndexType, ValueType> bsr;
    bsr.num_rows = csr.num_rows;
    bsr.num_cols = csr.num_cols;
    bsr.num_nnzs = csr.num_nnzs;

    bsr.blockDim_r = blockDimRow;
    bsr.blockDim_c = blockDimCol;
    bsr.blockNNZ = blockDimRow * blockDimCol;

    bsr.mb = (bsr.num_rows + blockDimRow - 1)/ blockDimRow;
    bsr.nb = (bsr.num_cols + blockDimCol - 1)/ blockDimCol;

    // ** Sepcial Case: quick return if blockDim == 1
    if (blockDimRow == 1 && blockDimCol == 1)
    {
        // malloc the row_ptr
        bsr.row_ptr = new_array<IndexType> (bsr.mb + 1);
        CHECK_ALLOC(bsr.row_ptr);
        memset(bsr.row_ptr, 0, (bsr.mb + 1) * sizeof(IndexType));

        #pragma omp parallel for schedule(dynamic, 1024)
        for(IndexType i = 0; i < csr.num_rows + 1; i++)
        {
            bsr.row_ptr[i] = csr.row_offset[i];
        }

        bsr.nnzb = bsr.row_ptr[bsr.mb] - bsr.row_ptr[0];

        // malloc the colindex
        bsr.block_colindex = new_array<IndexType> (bsr.nnzb);
        CHECK_ALLOC(bsr.block_colindex);
        memset(bsr.block_colindex, 0, bsr.nnzb * sizeof(IndexType));
        // malloc the data
        bsr.block_data = new_array<ValueType> ((size_t) bsr.nnzb * blockDimRow * blockDimCol);
        CHECK_ALLOC(bsr.block_data);
        memset(bsr.block_data, 0, ((size_t) bsr.nnzb * blockDimRow * blockDimCol) * sizeof(ValueType));

        #pragma omp parallel for schedule(dynamic, 1024)
        for(IndexType i = 0; i < csr.num_nnzs; i++)
        {
            bsr.block_colindex[i] = csr.col_index[i];
            bsr.block_data[i] = csr.values[i];
        }
        return bsr;
    }

    // ** General Case:
    // determine number of non-zero block columns for each block row of the bsr matrix
    bsr.row_ptr = new_array<IndexType> (bsr.mb + 1);
    CHECK_ALLOC(bsr.row_ptr);
    memset(bsr.row_ptr, 0, (bsr.mb + 1) * sizeof(IndexType));

    bsr.row_ptr[0] = 0;
    #pragma omp parallel for schedule(dynamic, 1024)
    for(IndexType i = 0; i < bsr.mb; i++)
    {
        IndexType start = csr.row_offset[i * blockDimRow];
        IndexType end   = csr.row_offset[std::min(csr.num_rows, blockDimRow * i + blockDimRow)];

        std::vector<IndexType> temp(bsr.nb, 0);
        for (IndexType j = start; j < end; j++)  // 一个block块内的rowID
        {
            IndexType blockCol = csr.col_index[j] / blockDimCol; // 计算元素所属的列block号
            temp[blockCol] = 1;  // 标记，这一个列块有nnz
        }

        IndexType sum = 0;
        for (IndexType j = 0; j < temp.size(); j++)
        {
            sum += temp[j]; // i 行 block 共有 sum 个 block要存
        }
        bsr.row_ptr[i+1] = sum;
    }

    for (IndexType i = 0; i < bsr.mb; i++)
    {
        bsr.row_ptr[i+1] += bsr.row_ptr[i];
    }

    bsr.nnzb = bsr.row_ptr[bsr.mb] - bsr.row_ptr[0];

    // find bsr col indices array
    // malloc the colindex
    bsr.block_colindex = new_array<IndexType> (bsr.nnzb);
    CHECK_ALLOC(bsr.block_colindex);
    memset(bsr.block_colindex, 0, bsr.nnzb * sizeof(IndexType));
    // malloc the data
    bsr.block_data = new_array<ValueType> ((size_t) bsr.nnzb * blockDimRow * blockDimCol);
    CHECK_ALLOC(bsr.block_data);
    memset(bsr.block_data, 0, ((size_t) bsr.nnzb * blockDimRow * blockDimCol) * sizeof(ValueType));

    IndexType colIndex = 0;
    
    // #pragma omp parallel for 
    for (IndexType i = 0; i < bsr.mb; i++)
    {
        IndexType start = csr.row_offset[i*blockDimRow];
        IndexType end   = csr.row_offset[std::min(csr.num_rows,(i+1)*blockDimRow)];
        
        std::vector<IndexType> temp(bsr.nb, 0);

        for (IndexType j = start; j < end; j++)
        {
            IndexType blockCol = csr.col_index[j] / blockDimCol;
            temp[blockCol] = 1;  // 标记，这一个列块有nnz
        }

        for (IndexType j = 0; j < bsr.nb; j++)
        {
            if( temp[j] == 1)
            {
                bsr.block_colindex[colIndex] = j;
                colIndex++;
            }
        }
    }

    // get bsr block values array.
    #pragma omp parallel for
    for (IndexType i = 0; i < bsr.num_rows; i++)
    {
        IndexType blockRow = i / blockDimRow;
        
        IndexType start = csr.row_offset[i];
        IndexType end   = csr.row_offset[i+1];

        for (IndexType j = start; j < end; j++) // 遍历第 i 行的非零元
        {
            IndexType blockCol = csr.col_index[j] / blockDimCol;

            colIndex = -1;

            for (IndexType k = bsr.row_ptr[blockRow]; k < bsr.row_ptr[blockRow+1]; k++)
            {
                if( bsr.block_colindex[k] == blockCol)
                {
                    colIndex = k - bsr.row_ptr[blockRow];
                    break;
                }
            }
            
            assert(colIndex != -1);

            IndexType blockIndex = 0;

            // row major
            blockIndex = csr.col_index[j] % blockDimCol + (i % blockDimRow) * blockDimCol;    // 块内index位置

            size_t index = (size_t) bsr.row_ptr[blockRow] * blockDimRow * blockDimCol + colIndex * blockDimRow * blockDimCol + blockIndex;

            bsr.block_data[index] = csr.values[j];
        }
    }

    return bsr;
}

/**
 * @brief CSR format to CSR5 format. Only deal with double precision from Weifeng Liu.
 * 
 * @tparam IndexType 
 * @tparam UIndexType 
 * @tparam ValueType 
 * @param csr 
 * @param fp_feature 
 * @return CSR5_Matrix<IndexType, UIndexType, ValueType> 
 */
template <class IndexType, typename UIndexType, class ValueType>
CSR5_Matrix<IndexType, UIndexType, ValueType> csr_to_csr5(const CSR_Matrix<IndexType, ValueType> &csr, FILE *fp_feature)
{
    int err = 0;
    double malloc_time = 0, tile_ptr_time = 0, tile_desc_time = 0, transpose_time = 0;
    anonymouslib_timer malloc_timer, tile_ptr_timer, tile_desc_timer, transpose_timer;

    CSR5_Matrix<IndexType, UIndexType, ValueType> csr5;

    csr5.num_rows = csr.num_rows;
    csr5.num_cols = csr.num_cols;
    csr5.num_nnzs = csr.num_nnzs;

    // original CSR format array
    csr5.row_offset = copy_array(csr.row_offset, csr.num_rows + 1);
    csr5.col_index  = copy_array(csr.col_index , csr.num_nnzs);
    csr5.values     = copy_array(csr.values    , csr.num_nnzs);

    csr5.tile_ptr  = NULL;
    csr5.tile_desc = NULL;
    csr5.tile_desc_offset_ptr = NULL;
    csr5.tile_desc_offset     = NULL;
    csr5.calibrator           = NULL;

    // store sigma and omega (tiles row and column number)
    csr5.omega = SIMD_WIDTH / 8 / sizeof(ValueType);
    csr5.sigma = CSR5_SIGMA;  // fixed in paper   12 or 16
/*
    //  heuristic in ALSPARSE
    int r = 4;
    int s = 32;
    int t = 256;
    int u = 6;
    IndexType csr_nnz_per_row = csr5.num_nnzs / csr5.num_rows;
    if (csr_nnz_per_row <= r)
        csr5.sigma = r;
    else if (csr_nnz_per_row > r && csr_nnz_per_row <= s)
        csr5.sigma = csr_nnz_per_row;
    else if (csr_nnz_per_row <= t && csr_nnz_per_row > s)
        csr5.sigma = s;
    else // csr_nnz_per_row > t
        csr5.sigma = u;
*/
    // compute how many bits required for `y_offset' and `seg_offset'
    IndexType base = 2;
    csr5.bit_y_offset = 1;
    while (base < csr5.sigma * csr5.omega) { // log (sigma * omega)
        base *= 2; ++csr5.bit_y_offset;
    }

    base = 2;
    csr5.bit_scansum_offset = 1;
    while (base < csr5.omega){               // log (omega)
        base *= 2; ++csr5.bit_scansum_offset;
    }
    
    if( csr5.bit_y_offset + csr5.bit_scansum_offset > sizeof(UIndexType) * 8 - 1) //the 1st bit of bit-flag should be in the first packet
    {
        printf("error: UNSUPPORTED CSR5 OMEGA for bit saving\n");
        exit(-1);
    }

    // y_offset + seg_offset + sigma (for a column)
    int bit_all = csr5.bit_y_offset + csr5.bit_scansum_offset + csr5.sigma;
    csr5.num_packets = ceil((double)bit_all / (double)(sizeof(UIndexType) * 8));
    
    // calculate the number of partitions
    csr5._p = ceil( (double) csr5.num_nnzs/ (double) (csr5.omega * csr5.sigma));

    malloc_timer.start();
    // malloc the newly added arrays for CSR5
    csr5.tile_ptr = (UIndexType *) memalign(CACHE_LINE, (uint64_t) (csr5._p + 1) * sizeof(UIndexType));
    if (csr5.tile_ptr == NULL){
        printf("error: UNABLE TO ASIGN MEMORY IN CSR5 tile_ptr \n");
        exit(-2);
    }
    for(IndexType i = 0; i < csr5._p + 1; i++) {
        csr5.tile_ptr[i] = 0;
    }

    csr5.tile_desc = (UIndexType *) memalign(CACHE_LINE, (uint64_t)( csr5._p * csr5.omega * csr5.num_packets) * sizeof(UIndexType));
    if (csr5.tile_desc == NULL){
        printf("error: UNABLE TO ASIGN MEMORY IN CSR5 tile_desc \n");
        exit(-2);
    }
    memset(csr5.tile_desc, 0, csr5._p * csr5.omega * csr5.num_packets * sizeof(UIndexType));

    int thread_num = Le_get_thread_num();
    csr5.calibrator = (ValueType *) memalign(CACHE_LINE, (uint64_t)(thread_num * CACHE_LINE));
    if (csr5.tile_desc == NULL){
        printf("error: UNABLE TO ASIGN MEMORY IN CSR5 calibrator \n");
        exit(-2);
    }
    memset(csr5.calibrator, 0, thread_num * CACHE_LINE);

    csr5.tile_desc_offset_ptr = (IndexType *) memalign(CACHE_LINE, (uint64_t) (csr5._p + 1) * sizeof(IndexType));
    if (csr5.tile_desc_offset_ptr == NULL){
        printf("error: UNABLE TO ASIGN MEMORY IN CSR5 tile_desc_offset_ptr \n");
        exit(-2);
    }
    memset(csr5.tile_desc_offset_ptr, 0, (csr5._p + 1) * sizeof(IndexType));
    malloc_time += malloc_timer.stop();

    // convert csr data to csr5 data (3 steps)
    // step 1. generate partition pointer
    tile_ptr_timer.start();
    // step 1.1 binary search row pointer
    #pragma omp parallel for num_threads(thread_num)
    for (IndexType global_id = 0; global_id < csr5._p; global_id++)
    {
        // compute tile boundaries by tile of size sigma * omega
        IndexType boundary = global_id * csr5.sigma * csr5.omega;

        // clamp tile boundaries to [0, nnz]
        boundary = boundary > csr5.num_nnzs ? csr5.num_nnzs : boundary;

        // binary search
        IndexType start = 0, stop = csr5.num_rows, median;
        IndexType key_median;
        while (stop >= start){
            median = (stop + start)/2;
            key_median = csr5.row_offset[median];
            if (boundary >= key_median)
                start = median + 1;
            else
                stop  = median - 1;
        }
        csr5.tile_ptr[global_id] = start - 1;
    }

    // step 1.2 check empty rows
    #pragma omp parallel for num_threads(thread_num)
    for (IndexType group_id = 0; group_id < csr5._p; group_id++)
    {
        int dirty = 0;

        UIndexType start = csr5.tile_ptr[group_id];
        UIndexType stop  = csr5.tile_ptr[group_id+1];
        // 把符号位刷掉 成0
        start = (start << 1) >> 1;
        stop  = (stop << 1) >> 1;

        if (start == stop)
            continue;
        
        // 找空行 dirty 置1
        for (UIndexType row_idx = start; row_idx <= stop; row_idx++) {
            if (csr5.row_offset[row_idx] == csr5.row_offset[row_idx+1]) {
                dirty = 1;
                break;
            }
        }

        if (dirty) {
            start |= sizeof(UIndexType) == 4
                                ? 0x80000000 : 0x8000000000000000;
            csr5.tile_ptr[group_id] = start;
        }
    }

    tile_ptr_time += tile_ptr_timer.stop();

    csr5.tail_partition_start = (csr5.tile_ptr[csr5._p-1] << 1) >> 1;

    tile_desc_timer.start();
    // step 2. generate partition descriptor
    csr5.num_offsets = 0;
    IndexType bit_all_offset = csr5.bit_y_offset + csr5.bit_scansum_offset;

    // step 2.1 generate_tile_descriptor_s1_kernel
    #pragma omp parallel for num_threads(thread_num)
    for (IndexType par_id = 0; par_id < csr5._p-1; par_id++)
    {
        // 去符号
        const IndexType row_start = csr5.tile_ptr[par_id]   & 0x7FFFFFFF;

        const IndexType row_stop  = csr5.tile_ptr[par_id+1] & 0x7FFFFFFF;

        for (IndexType rid = row_start; rid <= row_stop; rid++)
        {
            IndexType ptr = csr5.row_offset[rid];
            IndexType pid = ptr / (csr5.omega * csr5.sigma);

            // 从pid开始分析这一个tile的信息
            if (pid == par_id)
            {
                int lx = (ptr/ csr5.sigma) % csr5.omega; //行号

                const int glid = ptr % csr5.sigma + bit_all_offset;
                const int ly   = glid / 32; // 列号
                const int llid = glid % 32;

                const UIndexType val = 0x1 << (31 - llid);

                const int location = pid * csr5.omega
                    * csr5.num_packets
                    + ly * csr5.omega + lx;
                csr5.tile_desc[location] |= val;
            }
        }
    }

    // step 2.2 generate_tile_descriptor_s2_kernel
    int *s_segn_scan_all = (int *) memalign(CACHE_LINE, (uint64_t) (2 * csr5.omega * thread_num) * sizeof(int));

    int *s_present_all   = (int *) memalign(CACHE_LINE, (uint64_t) (2 * csr5.omega * thread_num) * sizeof(int));

    for (int i = 0; i < thread_num; i++)
        s_present_all[i*2*csr5.omega + csr5.omega]=1;
    
    #pragma omp parallel for num_threads(thread_num)
    for(int par_id = 0; par_id < csr5._p-1 ; par_id++)
    {
        int tid = Le_get_thread_id();
        int *s_segn_scan = &s_segn_scan_all[tid * 2 * csr5.omega];
        int *s_present   = &s_present_all[tid * 2 * csr5.omega];

        memset(s_segn_scan, 0, (csr5.omega + 1) * sizeof(int));
        memset(s_present, 0, csr5.omega * sizeof(int));

        bool with_empty_rows = (csr5.tile_ptr[par_id] >> 31) & 0x1;
        IndexType row_start       = csr5.tile_ptr[par_id]     & 0x7FFFFFFF;
        const IndexType row_stop  = csr5.tile_ptr[par_id + 1] & 0x7FFFFFFF;

        if (row_start == row_stop)
            continue;
        
        #pragma omp simd
        for (int lane_id = 0; lane_id < csr5.omega; lane_id++)
        {
            int start = 0, stop = 0, segn = 0;
            bool present = 0;
            UIndexType bitflag = 0;

            present |= !lane_id;

            // extract the first bit-flag packet
            int ly = 0;
            UIndexType first_packet = csr5.tile_desc[par_id * csr5.omega * csr5.num_packets + lane_id];
            bitflag = (first_packet << bit_all_offset) | ( (UIndexType) present << 31);

            start = !((bitflag >> 31) & 0x1);
            present |= (bitflag >> 31) & 0x1;

            for (int i = 1; i < csr5.sigma; i++)
            {
                if ((!ly && i == 32 - bit_all_offset) || (ly && (i - (32 - bit_all_offset)) % 32 == 0))
                {
                    ly++;
                    bitflag = csr5.tile_desc[par_id * csr5.omega * csr5.num_packets + ly * csr5.omega + lane_id];
                }
                const int norm_i = !ly ? i : i - (32 - bit_all_offset);
                stop += (bitflag >> (31 - norm_i % 32) ) & 0x1;
                present |= (bitflag >> (31 - norm_i % 32)) & 0x1;
            }

            // compute y_offset for all partitions
            segn = stop - start + present;
            segn = segn > 0 ? segn : 0;

            s_segn_scan[lane_id] = segn;

            // compute scansum_offset
            s_present[lane_id] = present;
        }

        // scan_single<int>(s_segn_scan, ALPHA_CSR5_OMEGA + 1);
        int old_val, new_val;
        old_val = s_segn_scan[0];
        s_segn_scan[0] = 0;
        for (int i = 1; i < csr5.omega + 1; i++)
        {
            new_val = s_segn_scan[i];
            s_segn_scan[i] = old_val + s_segn_scan[i-1];
            old_val = new_val;
        }

        if (with_empty_rows) {
            csr5.tile_desc_offset_ptr[par_id]   = s_segn_scan[csr5.omega];
            csr5.tile_desc_offset_ptr[csr5._p] += s_segn_scan[csr5.omega];
        }

        #pragma omp simd
        for (int lane_id= 0; lane_id< csr5.omega; lane_id++)
        {
            int y_offset = s_segn_scan[lane_id];
            int scansum_offset = 0;
            int next1 = lane_id + 1;
            if (s_present[lane_id])
            {
                while (!s_present[next1] && next1 < csr5.omega)
                {
                    scansum_offset++;
                    next1++;
                }
            }

            UIndexType first_packet = csr5.tile_desc[par_id * csr5.omega * csr5.num_packets + lane_id];

            y_offset = lane_id ? y_offset - 1: 0;

            first_packet |= y_offset << (32-csr5.bit_y_offset);
            first_packet |= scansum_offset << (32-bit_all_offset);

            csr5.tile_desc[par_id * csr5.omega
                * csr5.num_packets + lane_id] = first_packet;
        }
    }
    free(s_segn_scan_all);
    free(s_present_all);

    if (csr5.tile_desc_offset_ptr[csr5._p])
    {
        // scan_single<int>(csr5.tile_desc_offset_ptr, csr5._p+1);
        int old_val, new_val;
        old_val = csr5.tile_desc_offset_ptr[0];
        csr5.tile_desc_offset_ptr[0] = 0;
        for (int i = 1; i < csr5._p + 1; i++)
        {
            new_val = csr5.tile_desc_offset_ptr[i];
            csr5.tile_desc_offset_ptr[i] = old_val + csr5.tile_desc_offset_ptr[i-1];
            old_val = new_val;
        }
    }

    csr5.num_offsets = csr5.tile_desc_offset_ptr[csr5._p];
    tile_desc_time += tile_desc_timer.stop();

    if (csr5.num_offsets) {
        csr5.tile_desc_offset = (IndexType *) memalign(CACHE_LINE, (uint64_t)(csr5.num_offsets) * sizeof(IndexType));

        // generate_tile_descriptor_offset
        const int bit_bitflag = 32 - bit_all_offset;

        #pragma omp parallel for num_threads(thread_num)
        for (int par_id = 0; par_id < csr5._p-1; par_id++)
        {
            // 检查空行，非空则不需要offset
            bool with_empty_rows = (csr5.tile_ptr[par_id] >> 31)&0x1;
            if (!with_empty_rows)
                continue;
            
            IndexType row_start        = csr5.tile_ptr[par_id]   & 0x7FFFFFFF;
            const IndexType row_stop   = csr5.tile_ptr[par_id+1] & 0x7FFFFFFF;

            int offset_pointer = csr5.tile_desc_offset_ptr[par_id];

            #pragma omp simd
            for (int lane_id = 0; lane_id < csr5.omega; lane_id++)
            {
                bool local_bit;

                // extract the first bit-flag packet
                int ly = 0;
                UIndexType descriptor = csr5.tile_desc[par_id * csr5.omega * csr5.num_packets + lane_id];
                int y_offset = descriptor >> (32 - csr5.bit_y_offset);

                descriptor = descriptor << bit_all_offset;
                descriptor = lane_id ? descriptor : descriptor | 0x80000000;

                local_bit = (descriptor >> 31) & 0x1;

                if (local_bit && lane_id)
                {
                    const IndexType idx = par_id * csr5.omega * csr5.sigma + lane_id * csr5.sigma;
                    // const IndexType y_index = binary_search_right_boundary_kernel<IndexType>(&d_row_pointer[row_start+1], idx, row_stop - row_start) - 1;
                    IndexType start = 0;
                    IndexType stop = row_stop - row_start - 1;
                    IndexType median, key_median;
                    while (stop >= start)
                    {
                        median = (stop + start) / 2;
                        key_median = csr5.row_offset[row_start+1+median];
                        if (idx >= key_median)
                            start = median + 1;
                        else
                            stop  = median - 1;
                    }

                    const IndexType y_index = start - 1;
                    csr5.tile_desc_offset[offset_pointer + y_offset] = y_index;

                    y_offset++;
                }

                for (int i = 1; i < csr5.sigma; i++)
                {
                    if ((!ly && i == bit_bitflag) || (ly && !(31 & (i - bit_bitflag))))
                    {
                        ly++;
                        descriptor = csr5.tile_desc[par_id
                            * csr5.omega
                            * csr5.num_packets
                            + ly * csr5.omega + lane_id];
                    }
                    const int norm_i = 31 & (!ly
                                            ? i : i - bit_bitflag);

                    local_bit = (descriptor >> (31 - norm_i))&0x1;

                    if (local_bit)
                    {
                        const IndexType idx = par_id * csr5.omega * csr5.sigma + lane_id * csr5.sigma + i;
                        // const IndexType y_index = binary_search_right_boundary_kernel<iT>(&d_row_pointer[row_start+1], idx, row_stop - row_start) - 1;
                        IndexType start = 0;
                        IndexType stop = row_stop-row_start-1;
                        IndexType median, key_median;
                        while (stop >= start) {
                            median = (stop + start) / 2;
                            key_median=csr5.row_offset[row_start+1+median];
                            if (idx >= key_median)
                                start = median + 1;
                            else
                                stop = median - 1;
                        }
                        const IndexType y_index = start-1;
                        csr5.tile_desc_offset[offset_pointer + y_offset] = y_index;
                        
                        y_offset++;
                    }
                }
            }
        }
    }

    // step 3. transpose column_index and value arrays
    transpose_timer.start();
    err = aosoa_transpose(csr5.sigma, csr5.omega, csr5.num_nnzs, csr5.tile_ptr, csr5.col_index, csr5.values, true);
    if (err != 0)
    {
        printf("Error: aosoa_transpose error %d \n", err);
        exit(err);
    }
    transpose_time += transpose_timer.stop();

    printf("CSR->CSR5 malloc time = %f ms\n", malloc_time);
    printf("CSR->CSR5 tile_ptr time = %f ms\n", tile_ptr_time);
    printf("CSR->CSR5 tile_desc time = %f ms\n", tile_desc_time);
    printf("CSR->CSR5 transpose time = %f ms\n", transpose_time);

    return csr5;
}

#endif /* SPARSE_CONVERSION_H */
