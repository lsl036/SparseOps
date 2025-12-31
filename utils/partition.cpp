#include"../include/sparse_partition.h"
#include<vector>
#include"../include/memopt.h"

/**          acc_sum_arr=[3, 5, 8, 12, 15]   rows-1 = 4
 * @brief if row_ptr = [0, 3, 5, 8, 12, 15], row_num = 5, num_threads=5
 *        Input:  row_ptr,  0,  rows-1 = 4,  ave=3 * i (1~num_threads)
 *                  t       l,          r ,    value
 *         x  0  x  0  x   line: 0
 *         x  x  0  0  0         1
 *   A =   x  0  0  x  x         2
 *         0  x  x  x  x         3
 *         0  x  x  0  x         4
 *   when value = 3 * 1:   m = (0+4)/2 = 2 --> a[2] = 8 > 3 --> r = 2
 *                         m = (0+2)/2 = 1 --> a[1] = 5 > 3 --> r = 1
 *                         m = (0+1)/2 = 0 --> a[0] = 3 ==3 --> r = 0 
 *                  return 0
 * 
 *  (! r>l) --> return l = 0;   partition[1] = 0?
 *      partition = [0, 1, 2, 3, 4, 5]
 * @tparam IndexType 
 * @param t          partition array, typically for row_ptr in CSR
 * @param l          left start index
 * @param r          right end index
 * @param value 
 * @return int 
 */
template <typename IndexType>
IndexType lower_bound_int(const IndexType *t, IndexType l, IndexType r, IndexType value)
{
    while (r > l)
    {
        // bio-selection method
        IndexType m = (l + r) / 2;
        if (t[m] < value)
        {
            l = m + 1;
        }
        else
        {
            r = m;
        }
    }
    return l;
}
template int lower_bound_int<int> (const int*, int, int, int);
template long long lower_bound_int<long long> (const long long*, long long, long long, long long);

/**
 * @brief Get a balanced partition of rows by nnzs 
 *        Maybe just used for CSR format SpMV
 * @tparam IndexType 
 * @param row_ptr       CSR format row_ptr[0: rows], size: rows+1
 * @param rows          number of rows
 * @param num_threads   threads number
 * @param partition     partition results
 */
template <typename IndexType>
void balanced_partition_row_by_nnz(const IndexType *row_ptr, IndexType rows, IndexType num_threads, IndexType *partition)
{
    const IndexType* acc_sum_arr = &(row_ptr[1]);
    // IndexType nnz = row_ptr[rows];
    IndexType nnz = acc_sum_arr[rows - 1];
    IndexType ave = nnz / num_threads;
    if (0 == ave)
        ave = 1;
    partition[0] = 0;

    #pragma omp parallel for num_threads(num_threads)
    for (IndexType i = 1; i < num_threads; i++)
    {
        partition[i] = lower_bound_int(acc_sum_arr, (IndexType)0, rows-1, (ave*i));
    }
    partition[num_threads] = rows;
}

template void balanced_partition_row_by_nnz<int>(int const*, int, int, int*);
template void balanced_partition_row_by_nnz<long long>(long long const*, long long, long long, long long*);

/**
 * @brief Get a balanced partition of rows by nnzs 
 *        foe ELL RowMajor format
 * @tparam IndexType 
 * @param col_index     ELL format colIndex matrix whose size num_rows*MaxWidth
 * @param num_rows      number of rows
 * @param max_width     max width of all rows. It's also named MaxWidth
 * @param num_threads   threads number
 * @param partition     partition results array
 */
template <typename IndexType>
void balanced_partition_row_by_nnz_ell(const IndexType *col_index, const IndexType num_nnzs, IndexType num_rows, const IndexType max_width, IndexType num_threads, IndexType *partition)
{
    //    这个实现有点拉， 因为划分会堆积在前面的线程上，导致后面的线程经常空转
    // 平均一行要计算的非零元数目
    IndexType ave = num_nnzs / num_threads;
    if (0 == ave)
        ave = 1;

    partition[0] = 0;

    for (IndexType i = 1; i <= num_threads; ++i) {
        partition[i] = num_rows; // 默认为最后一行
    }

    // 当前处理的非零元素数量
    size_t current_nnz = 0;

    for (size_t i = 0, thread_id = 1; i < num_rows && thread_id < num_threads; ++i) {

        // 计算当前行的非零元素数量
        size_t nnz_this_row = 0;
        for (size_t j = 0; j < max_width; ++j) {
            if( col_index[j + i * max_width] >= 0 )
                ++ nnz_this_row;
            else
                break;
        }

        current_nnz += nnz_this_row;

        if( current_nnz >= ave && thread_id < num_threads)
        {
            partition[thread_id] = i + 1;
            current_nnz = 0;
            ++thread_id;
        }
    }

    partition[num_threads] = num_rows;

}

template void balanced_partition_row_by_nnz_ell(const int*, const int, int, const int, int, int*);
template void balanced_partition_row_by_nnz_ell(const long long*, const long long, long long, const long long, long long, long long*);

template <typename IndexType>
void balanced_partition_row_by_nnz_ell_n2(const IndexType *col_index, const IndexType num_nnzs, IndexType num_rows, const IndexType max_width, IndexType num_threads, IndexType *partition) {
    // 初始化每个线程的分区指针
    partition[0] = 0;
    for (IndexType i = 1; i <= num_threads; ++i) {
        partition[i] = num_rows;
    }

    // 计算每行的非零元素数并累积总和
    std::vector<IndexType> nnz_cumulative(num_rows + 1, 0);
    for (IndexType i = 0; i < num_rows; ++i) {
        IndexType nnz_this_row = 0;
        for (IndexType j = 0; j < max_width; ++j) {
            if (col_index[j + i * max_width] >= 0)
                ++nnz_this_row;
            else
                break;
        }
        nnz_cumulative[i + 1] = nnz_cumulative[i] + nnz_this_row;
    }

    // 使用二分搜索为每个线程找到平衡点
    IndexType target_nnz_per_thread = num_nnzs / num_threads;
    for (IndexType thread_id = 1; thread_id < num_threads; ++thread_id) {
        IndexType target_nnz = thread_id * target_nnz_per_thread;
        auto it = std::lower_bound(nnz_cumulative.begin(), nnz_cumulative.end(), target_nnz);
        partition[thread_id] = std::distance(nnz_cumulative.begin(), it);
    }
}
template void balanced_partition_row_by_nnz_ell_n2(const int*, const int, int, const int, int, int*);
template void balanced_partition_row_by_nnz_ell_n2(const long long*, const long long, long long, const long long, long long, long long*);

template <typename IndexType>
void balanced_partition_row_by_nnz_sell(const IndexType * const *col_index, const IndexType num_nnzs, IndexType chunk_size, IndexType chunk_num, const IndexType *row_width, IndexType num_threads, IndexType *partition)
{
    // 初始化每个线程的分区指针
    partition[0] = 0;
    for (IndexType i = 1; i <= num_threads; ++i) {
        partition[i] = chunk_num;
    }

    // 计算每 CHUNK 的非零元素数并累积总和
    std::vector<IndexType> nnz_cumulative(chunk_num + 1, 0);
    for (IndexType chunkID = 0; chunkID < chunk_num; ++chunkID) {
        IndexType nnz_this_chunk = 0;
        IndexType max_width = row_width[chunkID];
        for (IndexType rowID = 0; rowID < chunk_size; rowID++)
        {
            for (IndexType j = 0; j < max_width; ++j) {
                if (col_index[chunkID][j + rowID * max_width] >= 0)
                    ++nnz_this_chunk;
                else
                    break;
            }
        } 
        nnz_cumulative[chunkID + 1] = nnz_cumulative[chunkID] + nnz_this_chunk;
    }

    // 使用二分搜索为每个线程找到平衡点
    IndexType target_nnz_per_thread = num_nnzs / num_threads;
    for (IndexType thread_id = 1; thread_id < num_threads; ++thread_id) {
        IndexType target_nnz = thread_id * target_nnz_per_thread;
        auto it = std::lower_bound(nnz_cumulative.begin(), nnz_cumulative.end(), target_nnz);
        partition[thread_id] = std::distance(nnz_cumulative.begin(), it);
    }
}

template void balanced_partition_row_by_nnz_sell(const int * const *, const int , int , int , const int *, int , int *);
template void balanced_partition_row_by_nnz_sell(const long long * const *, const long long , long long , long long , const long long *, long long , long long *);