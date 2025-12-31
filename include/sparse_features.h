#ifndef SPARSE_FEATURES_H
#define SPARSE_FEATURES_H
#include<iostream>
#include<vector>
#include<string>
#include <map>
#include <unordered_map>

#include"general_config.h"
#include"sparse_format.h"
#include"timer.h"

template <typename IndexType, typename ValueType>
class MTX{
    public:
        MTX(IndexType matID = 0) {
            matrixID_ = matID;
            std::cout << "== MTX Features Extraction ==\n" << std::endl;
        }
        IndexType getRowNum(){
            return num_rows;
        }
        IndexType getColNum(){
            return num_cols;
        }
        IndexType getTileSize(){
            return t_num_blocks;
        }
        std::string getMatName(){
            return matrixName;
        }
        bool MtxLoad(const char* mat_path);
        bool FeaturesWrite(const char* file_path);
        bool ConvertToCSR(CSR_Matrix<IndexType, ValueType> &csr);
        void Stringsplit(const std::string& s, const char split, std::vector<std::string>& res);
        bool CalculateFeatures();
        bool CalculateTilesFeatures();
        bool CalculateTilesExtraFeatures(const char* mat_path);
        bool PrintImage(std::string& outputpath);
        double MtxLoad_time_= 0.0;
        double CalculateFeatures_time_= 0.0;
        double ConvertToCSR_time_= 0.0;

        void FeaturesPrint()
        {
            std::cout << "[" << matrixID_ << "]  " << matrixName << "counting by "<< 8*sizeof(IndexType)<< "-bit Index, " << 8*sizeof(ValueType)<< "-bit Value" << std::endl;
            std::cout << "is_symmetric " << is_symmetric_ << std::endl;
            std::cout << "pattern_symm " << pattern_symm_*100 << "%"<< std::endl;
            std::cout << "value_symm   " << value_symm_*100 << "%"<< std::endl<< std::endl;

            std::cout<< "num_rows   = "<< num_rows << std::endl;
            std::cout<< "num_cols   = "<< num_cols << std::endl;
            std::cout<< "nnz_mtx    = " << nnz_mtx_ << std::endl;
            std::cout<< "real_nnz   = "<< num_nnzs << std::endl;
            std::cout<< "nnz_ratio  = " << nnz_ratio_*100 << "%"<< std::endl;
            std::cout<< "num_of_diags  = " << complete_ndiags << std::endl;
            std::cout<< "nnz_closed2diag_ratio = " << diag_close_ratio_*100 << "%"<< std::endl << std::endl;
            // std::cout<< "ave_distance_= " << distance_per_row_ << std::endl<< std::endl;

            std::cout<< "nnz_lower    : " << nnz_lower_ << std::endl;
            std::cout<< "nnz_upper    : " << nnz_upper_ << std::endl;
            std::cout<< "nnz_diagonal : " << nnz_diagonal_ << std::endl<< std::endl;
            
            // row statistic features
            std::cout<< "NoneZero_Row_Ratio      : " << nz_row_ratio_*100 << "%"<< std::endl;
            std::cout<< "min_nnz_each_row        : " << min_nnz_each_row_ << std::endl;
            std::cout<< "max_nnz_each_row        : " << max_nnz_each_row_ << std::endl;
            std::cout<< "ave_nnz_each_row        : " << ave_nnz_each_row_ << std::endl;
            std::cout<< "var_nnz_each_row        : " << var_nnz_each_row_ << std::endl;
            std::cout<< "standard_dev_row        : " << standard_dev_row_ << std::endl;
            std::cout<< "P-ratio_row             : " << P_ratio_row_ << std::endl;
            std::cout<< "Gini_coeff_row          : " << Gini_row_ << std::endl<< std::endl;

            // col statistic features
            std::cout<< "NoneZero_Col_Ratio      : " << nz_col_ratio_*100 << "%"<< std::endl;
            std::cout<< "min_nnz_each_col        : " << min_nnz_each_col_ << std::endl;
            std::cout<< "max_nnz_each_col        : " << max_nnz_each_col_ << std::endl;
            std::cout<< "ave_nnz_each_col        : " << ave_nnz_each_col_ << std::endl;
            std::cout<< "var_nnz_each_col        : " << var_nnz_each_col_ << std::endl;
            std::cout<< "standard_dev_col        : " << standard_dev_col_ << std::endl;
            std::cout<< "P-ratio_col             : " << P_ratio_col_ << std::endl;
            std::cout<< "Gini_coeff_col          : " << Gini_col_ << std::endl<< std::endl;

            std::cout<< "diagonal_dominant_ratio : " << diagonal_dominant_ratio_*100 << "%"<< std::endl<< std::endl;

            std::cout<< "max_value_offdiag  = " << max_value_offdiag_ << std::endl;
            std::cout<< "max_value_diagonal = " << max_value_diagonal_ << std::endl;
            std::cout<< "row_variability    = " << row_variability_ << std::endl;
            std::cout<< "col_variability    = " << col_variability_ << std::endl << std::endl;

            // Tile features
            std::cout<< "==========   Tile Features   ==========" << std::endl;
            std::cout<< "Number of Tiles  : " << t_num_blocks << std::endl;
            std::cout<< "Tile inner size  : " << t_num_RB << " * " << t_num_CB << std::endl;
            std::cout<< "t_ave_nnz_tiles     : " << t_ave_nnz_all_tiles << std::endl;
            std::cout<< "t_ave_nnz_RB        : " << t_ave_nnz_RB << std::endl;
            std::cout<< "t_ave_nnz_CB        : " << t_ave_nnz_CB << std::endl<< std::endl;

            std::cout<< "t_var_nnz_tiles     : " << t_var_nnz_all_tiles << std::endl;
            std::cout<< "t_var_nnz_RB        : " << t_var_nnz_RB << std::endl;
            std::cout<< "t_var_nnz_CB        : " << t_var_nnz_CB << std::endl<< std::endl;

            std::cout<< "t_stand_dev_tiles     : " << t_standard_dev_all_tiles << std::endl;
            std::cout<< "t_stand_dev_RB        : " << t_standard_dev_RB << std::endl;
            std::cout<< "t_stand_dev_CB        : " << t_standard_dev_CB << std::endl<< std::endl;

            std::cout<< "min_nnz_each_tiles       : " << t_min_nnz_all_tiles_ << std::endl;
            std::cout<< "max_nnz_each_tiles       : " << t_max_nnz_all_tiles_ << std::endl;
            std::cout<< "min_nnz_each_RB          : " << t_min_nnz_each_RB_ << std::endl;
            std::cout<< "max_nnz_each_RB          : " << t_max_nnz_each_RB_ << std::endl;
            std::cout<< "min_nnz_each_CB          : " << t_min_nnz_each_CB_ << std::endl;
            std::cout<< "max_nnz_each_CB          : " << t_max_nnz_each_CB_ << std::endl<< std::endl;

            std::cout<< "Gini_nnz_tiles     : " << t_Gini_all_tiles_ << std::endl;
            std::cout<< "Gini_nnz_RB        : " << t_Gini_RB_ << std::endl;
            std::cout<< "Gini_nnz_CB        : " << t_Gini_CB_ << std::endl<< std::endl;

            std::cout<< "P-ratio_tiles      : " << t_P_ratio_all_tiles_ << std::endl;
            std::cout<< "P-ratio_RB         : " << t_P_ratio_RB_ << std::endl;
            std::cout<< "P-ratio_CB         : " << t_P_ratio_CB_ << std::endl<< std::endl;

            std::cout<< "NE_ratio_tiles_     : " << t_nz_ratio_tiles_ << std::endl;
            std::cout<< "NE_ratio_RB         : " << t_nz_ratio_RB_ << std::endl;
            std::cout<< "NE_ratio_CB         : " << t_nz_ratio_CB_ << std::endl<< std::endl;

        }

        void ExtraFeaturesPrint(){
            // Tile extra features
            std::cout<< "==========   Tile Extra Features   ==========" << std::endl;
            std::cout<< "uniq_R         = " << uniqR << std::endl;
            std::cout<< "uniq_C         = " << uniqC << std::endl<< std::endl;

            std::cout<< "GrX_uniqR      = " << GrX_uniqR << std::endl;
            std::cout<< "GrX_uniqC      = " << GrX_uniqC << std::endl<< std::endl;

            std::cout<< "potReuseR      = " << potReuseR << std::endl;
            std::cout<< "potReuseC      = " << potReuseC << std::endl<< std::endl;

            std::cout<< "GrX_potReuseR  = " << GrX_potReuseR << std::endl;
            std::cout<< "GrX_potReuseC  = " << GrX_potReuseC << std::endl<< std::endl;

            std::cout<< "ave_distance_= " << distance_per_row_ << std::endl<< std::endl;
        }

    private:
        bool is_symmetric_ = false;
        std::string matrixName;
        IndexType matrixID_ = 0;

        IndexType num_rows = 0;
        IndexType num_cols = 0;
        IndexType num_nnzs = 0;         // 真实的 nnz 数目
        IndexType nnz_mtx_ = 0;         // mtx 文件里显示的 nnz数目
        IndexType nnz_lower_ = 0;       // 下三角 非零元的数目
        IndexType nnz_upper_ = 0;       // 上三角 非零元的数目
        IndexType nnz_diagonal_ = 0;    // 对角线上的 nnz 数目
        IndexType complete_ndiags = 0;  // 在DIA格式中的对角线数目
        IndexType min_nnz_each_row_ = 100000000;    // 各行中最少的 nnz 数目
        IndexType max_nnz_each_row_ = 0;    // 各行中最大的 nnz 数目
        IndexType min_nnz_each_col_ = 100000000;    // 各列中最大的 nnz 数目
        IndexType max_nnz_each_col_ = 0;    // 各列中最大的 nnz 数目

    // Structure features
        ValueType pattern_symm_ = 0.0;      // 模式对称比例
        ValueType value_symm_   = 0.0;      // 数值对称比例
        ValueType nnz_ratio_ = 0.0;         // 稠密度， = 1 - 稀疏度
        ValueType diag_close_ratio_ = 0.0;  // 靠近对角线nnz所占比例
        ValueType distance_per_row_ = 0.0;  // 每行内非零元之间距离求和的平均值

        // nnzs skew statistic features
        ValueType nz_row_ratio_ = 0.0;
        ValueType ave_nnz_each_row_ = 0.0;  // 每行平均的 nnz 数目
        ValueType var_nnz_each_row_ = 0.0;  // 每行 nnz 的 方差
        ValueType standard_dev_row_ = 0.0;  // 每行 nnz 的 标准差
        ValueType Gini_row_ = 0.0;          // [0, 1] -> [balanced ~ imbalanced]
        ValueType P_ratio_row_ = 0.0;       // p fraction of rows have (1-p) fraction of nnzs in the matrix   [0, 0.5] -> [imbalanced ~ balanced]

        ValueType nz_col_ratio_ = 0.0;
        ValueType ave_nnz_each_col_ = 0.0;  // 每列平均的 nnz 数目
        ValueType var_nnz_each_col_ = 0.0;  // 每列 nnz 的 方差
        ValueType standard_dev_col_ = 0.0;  // 每列 nnz 的 标准差
        ValueType Gini_col_ = 0.0;
        ValueType P_ratio_col_ = 0.0;       // p fraction of cols have (1-p) fraction of nnzs in the matrix [0, 0.5] -> [imbalanced ~ balanced]

        ValueType diagonal_dominant_ratio_ = 0.0;  // 对角占优比率

    // Values features
        ValueType max_value_offdiag_  = -std::numeric_limits<ValueType>::max();
        ValueType max_value_diagonal_ = -std::numeric_limits<ValueType>::max();

        // 对于digital matrix， 这两个值不会变动，结果为 -1
        ValueType row_variability_ = -1.0;
        ValueType col_variability_ = -1.0;

    // Intermediate variables
        std::vector<IndexType> nnz_by_row_;     // 保存每行的 nnz 数目
        std::vector<IndexType> nnz_by_col_;     // 保存每列的 nnz 数目

        // 每行中 最大值、 最小值的 log10（value）
        std::vector<ValueType> max_each_row_;
        std::vector<ValueType> min_each_row_;
        // 每列中 最大值、 最小值的 log10（value）
        std::vector<ValueType> max_each_col_; 
        std::vector<ValueType> min_each_col_;
        std::vector<ValueType> Diag_Dom; // 计算每行的对角占优值， diag - other_row_sum

        std::vector<std::string> symm_pair_;
        std::unordered_map<std::string,std::string> m_;
        std::vector<std::vector<int> > image_;
        std::unordered_map<IndexType, IndexType> diag_offset_;

    /*

    RB means consider row block including whole cols:
    (Likewise, CB means consider col block including whole rows)
    Here,  t_num_rows = 2.
        ________________________
          x  x  x  x  x  x  x  x |
          x  x  x  x  x  x  x  x |  ---> RB
        ------------------------   
          x  x  x  x  x  x  x  x
          x  x  x  x  x  x  x  x
        ------------------------
          x  x  x  x  x  x  x  x
          x  x  x  x  x  x  x  x
        ________________________
    */
    // Tiles features 默认 2048*2048 tiles
        IndexType t_num_blocks = MAT_TILE_SIZE;
        IndexType t_num_RB = -1;        // tiles 内的行数目 (小)
        IndexType t_num_CB = -1;        // tiles 内的列数目 (小)
        IndexType t_mod_RB = -1;        // 前 t_mod_RB 行块的 t_num_RB+1
        IndexType t_mod_CB = -1;        // 前 t_mod_CB 列块的 t_num_CB+1

        // ave_nnz
        ValueType t_ave_nnz_all_tiles = -1.0;
        ValueType t_ave_nnz_RB = -1.0;   // row block
        ValueType t_ave_nnz_CB = -1.0;   // col block

        // var_nnz
        ValueType t_var_nnz_all_tiles = -1.0;
        ValueType t_var_nnz_RB = -1.0;
        ValueType t_var_nnz_CB = -1.0;

        // standard diviation
        ValueType t_standard_dev_all_tiles = -1.0;
        ValueType t_standard_dev_RB = -1.0;
        ValueType t_standard_dev_CB = -1.0;
    
    // Intermediate variables
        // Rowmajor 存 tiles 分块   (rowID/t_num_RB) * t_num_blocks + (colID/t_num_CB)
        // 这3个vector 是均衡的 block 统计 各块 nnz
        std::vector<IndexType> nnz_by_Tiles_;  // 保存每个tiles的 nnz 数目
        // RB 和 CB 的分块
        std::vector<IndexType> nnz_by_RB_;     // 保存每个行块的 nnz 数目
        std::vector<IndexType> nnz_by_CB_;     // 保存每个列块的 nnz 数目

        std::vector<std::vector<IndexType>> Rows_cnt;
        std::vector<std::vector<IndexType>> Cols_cnt;

        std::vector<IndexType> max_rownnz_per_tile_;
        std::vector<ValueType> ave_rownnz_per_tile_;
        std::vector<ValueType> std_rownnz_per_tile_;


        std::vector<IndexType> max_rownnz_per_RB_;
        std::vector<IndexType> max_colnnz_per_CB_;
        std::vector<ValueType> ave_rownnz_per_RB_;
        std::vector<ValueType> ave_colnnz_per_CB_;
        std::vector<ValueType> std_rownnz_per_RB_;
        std::vector<ValueType> std_colnnz_per_CB_;

        // min and max nnz for each 
        IndexType t_min_nnz_all_tiles_ = 100000000;
        IndexType t_max_nnz_all_tiles_ = 0;
        IndexType t_min_nnz_each_RB_   = 100000000;
        IndexType t_max_nnz_each_RB_   = 0;
        IndexType t_min_nnz_each_CB_   = 100000000;
        IndexType t_max_nnz_each_CB_   = 0;

        // Gini index  [0, 1] -> [balanced ~ imbalanced]
        ValueType t_Gini_all_tiles_ = -1.0;
        ValueType t_Gini_RB_ = -1.0;          
        ValueType t_Gini_CB_ = -1.0;

        // p-ratio [0, 0.5] -> [imbalanced ~ balanced]
        ValueType t_P_ratio_all_tiles_ = -1.0;
        ValueType t_P_ratio_RB_ = -1.0;
        ValueType t_P_ratio_CB_ = -1.0;

        // none zero ratio
        ValueType t_nz_ratio_tiles_ = -1.0;
        ValueType t_nz_ratio_RB_ = -1.0;
        ValueType t_nz_ratio_CB_ = -1.0;

    // Extra information
        // uniq
        std::vector<IndexType> uniq_RB;  // 记录每个tiles的非零行数
        std::vector<IndexType> uniq_CB;  // 记录每个tiles的非零列数
        ValueType uniqR = -1.0;           // sum divide nnz
        ValueType uniqC = -1.0;           // sum divide nnz

        // GrX_uniq  ; for cacheline evaluate
        IndexType GrX = CACHE_LINE / sizeof(ValueType);
        std::vector<IndexType> GrX_uniqRB;
        std::vector<IndexType> GrX_uniqCB;
        ValueType GrX_uniqR = -1.0;       // sum divide nnz
        ValueType GrX_uniqC = -1.0;       // sum divide nnz

        // porReuse ; for data reuse in the LLC
        // 记录这一 行/列 中 有几个 tiles中的 行/列 非零
        // std::vector<IndexType> potReuseRB;  // size: num_rows
        // std::vector<IndexType> potReuseCB;  // size: num_cols
        // Note**: tile内的非零行数目，和一行中非零的tile数目 SUM是一样的,
        // 因此可以不需要额外的空间来统计信息
        ValueType potReuseR = -1.0;      // sum divide num of rows
        ValueType potReuseC = -1.0;      // sum divide num of cols

        // GrX_porReuse ; for data reuse in the LLC with more coarse granularity
        std::vector<IndexType> GrX_potReuseRB;  // size: num_GrXrows
        std::vector<IndexType> GrX_potReuseCB;  // size: num_GrXcols
        ValueType GrX_potReuseR = -1.0;      // sum divide num of GrXrows
        ValueType GrX_potReuseC = -1.0;     // sum divide num of GrXcols

};

#endif /* SPARSE_FEATURES_H */
