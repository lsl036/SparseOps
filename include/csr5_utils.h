#ifndef CSR5_UTILS_H
#define CSR5_UTILS_H
#include"thread.h"
#include"memopt.h"
#include"plat_config.h"
#include"general_config.h"

#include<math.h>

template<typename T, typename uiT>
void aosoa_transpose_kernel_smem(T         *d_data,
                                 const uiT *d_partition_pointer,
                                 const int  nnz,
                                 const int  sigma,
                                 const int  omega,
                                 const bool R2C) // R2C==true means CSR->CSR5, otherwise CSR5->CSR
{
    int num_p = ceil((double)nnz / (double)(omega * sigma)) - 1;
    int num_thread = Le_get_thread_num();

    T *s_data_all = (T *)memalign(CACHE_LINE, (uint64_t) sigma * omega * sizeof(T) * num_thread);

    #pragma omp parallel for
    for (int par_id = 0; par_id < num_p; par_id++)
    {
        int tid = Le_get_thread_id();
        T *s_data = &s_data_all[sigma * omega * tid];

        // if this is fast track partition, do not transpose it
        if (d_partition_pointer[par_id] == d_partition_pointer[par_id + 1])
            continue;
        
        // load global data to shared mem
        int idx_y, idx_x;

        #pragma omp simd
        for (int idx = 0; idx < omega * sigma; idx++)
        {
            if (R2C)
            {
                idx_y = idx % sigma;
                idx_x = idx / sigma;
            }
            else
            {
                idx_x = idx % omega;
                idx_y = idx / omega;
            }

            s_data[idx_y * omega + idx_x] = d_data[par_id * omega * sigma + idx];
        }

        // store transposed shared mem data to global
        #pragma omp simd
        for (int idx = 0; idx < omega * sigma; idx++)
        {
            if (R2C)
            {
                idx_x = idx % omega;
                idx_y = idx / omega;
            }
            else
            {
                idx_y = idx % sigma;
                idx_x = idx / sigma;
            }

            d_data[par_id * omega * sigma + idx] = s_data[idx_y * omega + idx_x];
        }
    }

    free(s_data_all);
}

template<typename IndexType, typename UIndexType, typename ValueType>
int aosoa_transpose(const int           sigma,
                    const int           omega,
                    const int           nnz,
                    const UIndexType    *partition_pointer,
                    IndexType           *column_index,
                    ValueType           *value,
                    bool                R2C)
{
    aosoa_transpose_kernel_smem<IndexType, UIndexType>(column_index, partition_pointer, nnz, sigma, omega, R2C);
    aosoa_transpose_kernel_smem<ValueType, UIndexType>(value,        partition_pointer, nnz, sigma, omega, R2C);
    return 0;
}

#endif /* CSR5_UTILS_H */
