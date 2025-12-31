/**
 * @file sparse_io.cpp for sparse matrix I/O
 *       COO 和 CSR 作为基本类型，直接使用 API 进行读取
 *       ELL 等后续格式应当考虑使用 CSR 转换过去（能记录overhead）
 * @author Shengle Lin (lsl036@hnu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2023-11-16
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include"../include/sparse_io.h"
#include"../include/thread.h"
#include"../include/sparse_partition.h"
#include <cassert>

/**
 * @brief Read sparse matrix in COO format from ".mtx" format file.
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param mm_filename : The sparse matrix file, must in mtx format.
 * @return COO_Matrix<IndexType,ValueType> 
 */
template <class IndexType,class ValueType>
COO_Matrix<IndexType,ValueType> read_coo_matrix(const char * mm_filename)
{
    COO_Matrix<IndexType, ValueType> coo;

    FILE *fid;

    MM_typecode matcode;

    fid = fopen(mm_filename, "r");

    if(fid == NULL){
        std::cout << "Unable to open file: "<< mm_filename << std::endl;
        exit(1);
    }

    if (mm_read_banner(fid, &matcode) != 0){
        std::cout << "Could not process Matrix Market banner." << std::endl;
        exit(1);
    }

    if(!mm_is_valid(matcode)){
        std::cout << "Invalid Matrix" << std::endl;
        exit(1);
    }

    if (!((mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode)) && mm_is_coordinate(matcode) && mm_is_sparse(matcode) ) ){
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        printf("Only sparse real-valued or pattern coordinate matrices are supported\n");
        exit(1);
    }

    // 处理第一行， 获得行数，列数，非零元数目
    int num_rows, num_cols, num_nnzs;
    if (mm_read_mtx_crd_size(fid, &num_rows, &num_cols, &num_nnzs) != 0)
    {
        std::cout << "The line of rows, cols, nnzs is in wrong format" << std::endl;
        exit(1);
    }

    coo.num_rows = (IndexType) num_rows;
    coo.num_cols = (IndexType) num_cols;
    coo.num_nnzs = (IndexType) num_nnzs;

    coo.row_index = new_array<IndexType>(coo.num_nnzs);
    CHECK_ALLOC(coo.row_index);
    coo.col_index = new_array<IndexType>(coo.num_nnzs);
    CHECK_ALLOC(coo.col_index);
    coo.values    = new_array<ValueType>(coo.num_nnzs);
    CHECK_ALLOC(coo.values);

    std::cout << "- Reading sparse matrix from file: "<< mm_filename << std::endl;
    fflush(stdout);

    if(mm_is_pattern(matcode)){
        for (IndexType i = 0; i < coo.num_nnzs; i++)
        {
            if constexpr(std::is_same<IndexType, int>::value) {
                assert(fscanf(fid,"%d %d\n", &(coo.row_index[i]), &(coo.col_index[i])) == 2);
            } else if constexpr(std::is_same<IndexType, long long>::value) {
                assert(fscanf(fid,"%lld %lld\n", &(coo.row_index[i]), &(coo.col_index[i])) == 2);
            }
            // adjust from 1-based to 0-based indexing
            --coo.row_index[i];
            --coo.col_index[i];
            coo.values[i] = 1.0;
        }
    }else if (mm_is_real(matcode) || mm_is_integer(matcode)){
        for( IndexType i = 0; i < coo.num_nnzs; i++ ){
            IndexType row_id, col_id;
            double V; // read in double and convert to ValueType

            if constexpr(std::is_same<IndexType, int>::value) {
                assert(fscanf(fid, "%d %d %lf\n", &row_id, &col_id, &V) == 3);
            } else if constexpr(std::is_same<IndexType, long long>::value) {
                assert(fscanf(fid, "%lld %lld %lf\n", &row_id, &col_id, &V) == 3);
            }

            coo.row_index[i] = (IndexType) row_id - 1;
            coo.col_index[i] = (IndexType) col_id - 1;
            coo.values[i]    = (ValueType) V;
        }
    }else{
        std::cout << "Unsupported data type" << std::endl;
        exit(1);
    }

    fclose(fid);
    std::cout << "- Finish Reading data from " << mm_filename << std::endl;

    // 处理对称情况 duplicate off diagonal entries
    if( mm_is_symmetric(matcode) ){
        IndexType off_diagonals = 0;
        for (IndexType i = 0; i < coo.num_nnzs; i++){
            if(coo.row_index[i] != coo.col_index[i])
                off_diagonals++;
        }
        // realNNZ = 2*off_diagonals + (coo.num_nonzeros - off_diagonals)
        IndexType true_nnz = off_diagonals + coo.num_nnzs;

        IndexType* new_rowindex = new_array<IndexType>(true_nnz);
        CHECK_ALLOC(new_rowindex);
        IndexType* new_colindex = new_array<IndexType>(true_nnz);
        CHECK_ALLOC(new_colindex);
        ValueType* new_V        = new_array<ValueType>(true_nnz);
        CHECK_ALLOC(new_V);

        IndexType ptr = 0;
        for (IndexType i = 0; i < coo.num_nnzs; i++)
        {
            if(coo.row_index[i] != coo.col_index[i]){
                new_rowindex[ptr] = coo.row_index[i];
                new_colindex[ptr] = coo.col_index[i];
                new_V[ptr]        = coo.values[i];
                ptr++;
                new_colindex[ptr] = coo.row_index[i];
                new_rowindex[ptr] = coo.col_index[i];
                new_V[ptr]        = coo.values[i];
                ptr++;
            }else {
                new_rowindex[ptr] = coo.row_index[i];
                new_colindex[ptr] = coo.col_index[i];
                new_V[ptr]        = coo.values[i];
                ptr++;
            }
        }
        delete_array(coo.row_index);
        delete_array(coo.col_index);
        delete_array(coo.values);
        coo.row_index = new_rowindex;
        coo.col_index = new_colindex;
        coo.values    = new_V;
        coo.num_nnzs = true_nnz;
    }
    coo.kernel_flag = KERNEL_FLAG;
    return coo;
}

template COO_Matrix<int, float> read_coo_matrix<int, float>(const char * mm_filename);
template COO_Matrix<int, double> read_coo_matrix<int, double>(const char * mm_filename);
template COO_Matrix<long long, float> read_coo_matrix<long long, float>(const char * mm_filename);
template COO_Matrix<long long, double> read_coo_matrix<long long, double>(const char * mm_filename);

/**
 * @brief Read sparse matrix in CSR format from ".mtx" format file.
 *        Convert from COO format.
 * @tparam IndexType 
 * @tparam ValueType 
 * @param mm_filename The sparse matrix file, must in mtx format.
 * @param compact     Judge whether sum duplicates together in CSR or not
 * @return CSR_Matrix<IndexType, ValueType> 
 */
template <class IndexType, class ValueType>
CSR_Matrix<IndexType, ValueType> read_csr_matrix(const char * mm_filename, bool compact)
{
    // 先按COO的标准读进来
    COO_Matrix<IndexType, ValueType> coo = read_coo_matrix<IndexType, ValueType>(mm_filename);

    CSR_Matrix<IndexType, ValueType> csr;
    if(0 == coo.num_rows){
        csr.num_rows = 0;
        csr.num_cols = 0;
        csr.num_nnzs = 0;
        csr.row_offset = NULL;
        csr.col_index  = NULL;
        csr.values     = NULL;
        return csr;
    }

    csr = coo_to_csr(coo, compact);

    // std::cout << "- Finish CSR convertion -" << std::endl;
    delete_host_matrix(coo);

    csr.kernel_flag = KERNEL_FLAG;

    return csr;
}

template CSR_Matrix<int, float> read_csr_matrix<int, float>(const char * mm_filename, bool);
template CSR_Matrix<int, double> read_csr_matrix<int, double>(const char * mm_filename, bool);
template CSR_Matrix<long long, float> read_csr_matrix<long long, float>(const char * mm_filename, bool);
template CSR_Matrix<long long, double> read_csr_matrix<long long, double>(const char * mm_filename, bool);


template <class IndexType, class ValueType>
BSR_Matrix<IndexType, ValueType> read_bsr_matrix(const char * mm_filename, const IndexType blockDimRow, const IndexType blockDimCol)
{
    CSR_Matrix<IndexType, ValueType> csr;
    csr = read_csr_matrix<IndexType, ValueType>(mm_filename);

    BSR_Matrix<IndexType,ValueType> bsr;

    bsr = csr_to_bsr<IndexType, ValueType>(csr, blockDimRow, blockDimCol);

    std::cout << "- Finish BSR convertion -" << std::endl;

    delete_csr_matrix(csr);
    
    return bsr;
}
template BSR_Matrix<int, float> read_bsr_matrix<int, float>(const char * mm_filename, const int, const int);
template BSR_Matrix<int, double> read_bsr_matrix<int, double>(const char * mm_filename, const int, const int);
template BSR_Matrix<long long, float> read_bsr_matrix<long long, float>(const char * mm_filename, const long long, const long long);
template BSR_Matrix<long long, double> read_bsr_matrix<long long, double>(const char * mm_filename, const long long, const long long);

/**
 * @brief Read a sparse matrix in CSR5 format. Transform from CSR format.
 * 
 * @tparam IndexType 
 * @tparam UIndexType 
 * @tparam ValueType 
 * @param mm_filename 
 * @return CSR5_Matrix<IndexType, UIndexType, ValueType> 
 */
template <class IndexType, class UIndexType, class ValueType>
CSR5_Matrix<IndexType, UIndexType, ValueType> read_csr5_matrix(const char * mm_filename)
{
    CSR_Matrix<IndexType, ValueType> csr;
    csr = read_csr_matrix<IndexType, ValueType>(mm_filename);

    CSR5_Matrix<IndexType, UIndexType, ValueType> csr5;

    FILE* save_features = fopen(MAT_FEATURES,"w");

    csr5 = csr_to_csr5<IndexType, UIndexType, ValueType>(csr, save_features);

    fclose(save_features);
    delete_csr_matrix(csr);

    return csr5;
}
template CSR5_Matrix<int, uint32_t, float> read_csr5_matrix<int, uint32_t, float> (const char *);
template CSR5_Matrix<int, uint32_t, double> read_csr5_matrix<int, uint32_t, double> (const char *);


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
ELL_Matrix<IndexType, ValueType> read_ell_matrix(const char * mm_filename, LeadingDimension ld)
{
    // 先按COO的标准读进来
    COO_Matrix<IndexType, ValueType> coo = read_coo_matrix<IndexType, ValueType>(mm_filename);

    ELL_Matrix<IndexType, ValueType> ell;
    if(0 == coo.num_rows){
        ell.num_rows = 0;
        ell.num_cols = 0;
        ell.num_nnzs = 0;
        ell.max_row_width = 0;

        ell.col_index = NULL;
        ell.values  = NULL;
        return ell;
    }

    // LeadingDimension ld = RowMajor;
    // ell = csr_to_ell(csr, ld);
    ell = coo_to_ell(coo, ld);

    delete_host_matrix(coo);

    ell.kernel_flag = KERNEL_FLAG;

    return ell;
}

template ELL_Matrix<int, float> read_ell_matrix<int, float>(const char * mm_filename, LeadingDimension);
template ELL_Matrix<int, double> read_ell_matrix<int, double>(const char * mm_filename, LeadingDimension);

/**
 * @brief  Read a sparse matrix in DIA format
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param mm_filename 
 * @param max_diags 
 * @param alignment 
 * @return DIA_Matrix<IndexType, ValueType> 
 */
template <class IndexType, class ValueType>
DIA_Matrix<IndexType, ValueType> read_dia_matrix(const char * mm_filename, const IndexType max_diags, const IndexType alignment)
{
    CSR_Matrix<IndexType, ValueType> csr;
    csr = read_csr_matrix<IndexType, ValueType>(mm_filename);

    DIA_Matrix<IndexType, ValueType> dia;

    FILE* save_features = fopen(MAT_FEATURES,"w");
    dia = csr_to_dia(csr, max_diags, save_features, alignment);

    fclose(save_features);
    delete_csr_matrix(csr);

    return dia;
}

template DIA_Matrix<int, float> read_dia_matrix<int, float>(const char * mm_filename, const int max_diags, const int alignment);
template DIA_Matrix<int, double> read_dia_matrix<int, double>(const char * mm_filename, const int max_diags, const int alignment);
template DIA_Matrix<long long, float> read_dia_matrix<long long, float>(const char * mm_filename, const long long max_diags, const long long alignment);
template DIA_Matrix<long long, double> read_dia_matrix<long long, double>(const char * mm_filename, const long long max_diags, const long long alignment);

/**
 * @brief Read a sparse matrix in S-ELL format
 * 
 * @tparam IndexType 
 * @tparam ValueType 
 * @param mm_filename 
 * @param max_diags 
 * @param alignment 
 * @return S_ELL_Matrix<IndexType, ValueType> 
 */
template <class IndexType, class ValueType>
S_ELL_Matrix<IndexType, ValueType> read_sell_matrix(const char * mm_filename, const int chunkwidth, const IndexType alignment)
{
    CSR_Matrix<IndexType, ValueType> csr;
    csr = read_csr_matrix<IndexType, ValueType>(mm_filename);

    S_ELL_Matrix<IndexType, ValueType> s_ell;

    FILE* save_features = fopen(MAT_FEATURES,"w");
    s_ell = csr_to_sell(csr, save_features, chunkwidth, alignment);

    fclose(save_features);
    delete_csr_matrix(csr);

    return s_ell;

}

template S_ELL_Matrix<int, float> read_sell_matrix<int, float>(const char * mm_filename, const int chunkwidth, const int alignment);
template S_ELL_Matrix<int, double> read_sell_matrix<int, double>(const char * mm_filename, const int chunkwidth, const int alignment);
template S_ELL_Matrix<long long, float> read_sell_matrix<long long, float>(const char * mm_filename, const int chunkwidth, const long long alignment);
template S_ELL_Matrix<long long, double> read_sell_matrix<long long, double>(const char * mm_filename, const int chunkwidth, const long long alignment);

template <class IndexType, class ValueType>
SELL_C_Sigma_Matrix<IndexType, ValueType> read_sell_c_sigma_matrix(const char * mm_filename, const int slicewidth, const int chunkwidth, const IndexType alignment)
{
    CSR_Matrix<IndexType, ValueType> csr;
    csr = read_csr_matrix<IndexType, ValueType> (mm_filename);

    SELL_C_Sigma_Matrix<IndexType, ValueType> sell_c_sigma;

    FILE* save_features = fopen(MAT_FEATURES,"w");
    sell_c_sigma = csr_to_sell_c_sigma(csr, save_features, slicewidth, chunkwidth, alignment);

    fclose(save_features);
    delete_csr_matrix(csr);

    return sell_c_sigma;
}

template SELL_C_Sigma_Matrix<int, float> read_sell_c_sigma_matrix<int, float>(const char * mm_filename, const int slicewidth, const int chunkwidth, const int alignment);
template SELL_C_Sigma_Matrix<int, double> read_sell_c_sigma_matrix<int, double>(const char * mm_filename, const int slicewidth, const int chunkwidth, const int alignment);
template SELL_C_Sigma_Matrix<long long, float> read_sell_c_sigma_matrix<long long, float>(const char * mm_filename, const int slicewidth, const int chunkwidth, const long long alignment);
template SELL_C_Sigma_Matrix<long long, double> read_sell_c_sigma_matrix<long long, double>(const char * mm_filename, const int slicewidth, const int chunkwidth, const long long alignment);


template <class IndexType, class ValueType>
SELL_C_R_Matrix<IndexType, ValueType> read_sell_c_R_matrix(const char * mm_filename, const int chunkwidth, const IndexType alignment)
{
    CSR_Matrix<IndexType, ValueType> csr;
    csr = read_csr_matrix<IndexType, ValueType> (mm_filename);

    SELL_C_R_Matrix<IndexType, ValueType> mtx;

    FILE* save_features = fopen(MAT_FEATURES,"w");
    mtx = csr_to_sell_c_R(csr, save_features, chunkwidth, alignment);

    fclose(save_features);
    delete_csr_matrix(csr);

    return mtx;
}

template SELL_C_R_Matrix<int,float> read_sell_c_R_matrix<int,float>(const char * mm_filename, const int chunkwidth, const int alignment);
template SELL_C_R_Matrix<int,double> read_sell_c_R_matrix<int,double>(const char * mm_filename, const int chunkwidth, const int alignment);
template SELL_C_R_Matrix<long long,float> read_sell_c_R_matrix<long long,float>(const char * mm_filename, const int chunkwidth, const long long alignment);
template SELL_C_R_Matrix<long long,double> read_sell_c_R_matrix<long long,double>(const char * mm_filename, const int chunkwidth, const long long alignment);
