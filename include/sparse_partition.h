#ifndef SPARSE_PARTITION_H
#define SPARSE_PARTITION_H
#include <math.h>
#include <limits.h>
#include <stdint.h>

template <typename IndexType>
IndexType lower_bound_int(const IndexType *t, IndexType l, IndexType r, IndexType value);

/**
 * @brief Get a balanced partition of rows by nnzs 
 *        Maybe just used for CSR format SpMV
 * @tparam IndexType 
 * @param row_ptr       CSR format row_ptr[0: rows], size: rows+1
 * @param rows          number of rows
 * @param num_threads   threads number
 * @param partition     partition results array
 */
template <typename IndexType>
void balanced_partition_row_by_nnz(const IndexType *row_ptr, IndexType rows, IndexType num_threads, IndexType *partition);

/**
 * @brief Get a balanced partition of rows by nnzs 
 *        foe ELL RowMajor format
 * @tparam IndexType 
 * @param col_index     ELL format colIndex matrix whose size num_rows*MaxWidth
 * @param num_nnzs      numer of nnzs of whole matrix
 * @param num_rows      number of rows
 * @param max_width     max width of all rows. It's also named MaxWidth
 * @param num_threads   threads number
 * @param partition     partition results array
 */
template <typename IndexType>
void balanced_partition_row_by_nnz_ell(const IndexType *col_index, const IndexType num_nnzs, IndexType num_rows, const IndexType max_width, IndexType num_threads, IndexType *partition);

template <typename IndexType>
void balanced_partition_row_by_nnz_ell_n2(const IndexType *col_index, const IndexType num_nnzs, IndexType num_rows, const IndexType max_width, IndexType num_threads, IndexType *partition);

template <typename IndexType>
void balanced_partition_row_by_nnz_sell(const IndexType * const *col_index, const IndexType num_nnzs, IndexType chunk_size, IndexType chunk_num, const IndexType *row_width, IndexType num_threads, IndexType *partition);

#endif /* SPARSE_PARTITION_H */
