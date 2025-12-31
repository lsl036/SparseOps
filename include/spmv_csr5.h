#ifndef SPMV_CSR5_H
#define SPMV_CSR5_H

#include "sparse_format.h"

template <typename IndexType, typename UIndexType, typename ValueType>
void LeSpMV_csr5(const ValueType alpha, const CSR5_Matrix<IndexType, UIndexType, ValueType>& csr5, const ValueType * x, const ValueType beta, ValueType * y);

template<typename iT, typename uiT, typename vT>
void spmv_csr5_compute_kernel(const iT           *d_column_index,
                              const vT           *d_value,
                              const iT           *d_row_pointer,
                              const vT           *d_x,
                              const uiT          *d_partition_pointer,
                              const uiT          *d_partition_descriptor,
                              const iT           *d_partition_descriptor_offset_pointer,
                              const iT           *d_partition_descriptor_offset,
                              vT                 *d_calibrator,
                              vT                 *d_y,
                              const iT            p,
                              const int           num_packet,
                              const int           bit_y_offset,
                              const int           bit_scansum_offset,
                              const vT            alpha,
                              const vT            beta,
                              const int           c_sigma,
                              const int           c_omega);

template<typename iT, typename uiT, typename vT>
void spmv_csr5_calibrate_kernel(const uiT *d_partition_pointer,
                                vT        *d_calibrator,
                                vT        *d_y,
                                const iT   p);

template<typename iT, typename uiT, typename vT>
void spmv_csr5_tail_partition_kernel(const iT           *d_row_pointer,
                                     const iT           *d_column_index,
                                     const vT           *d_value,
                                     const vT           *d_x,
                                     vT                 *d_y,
                                     const iT            tail_partition_start,
                                     const iT            p,
                                     const iT            m,
                                     const int           sigma,
                                     const int           omega,
                                     const vT            alpha,
                                     const vT            beta);
#endif /* SPMV_CSR5_H */
