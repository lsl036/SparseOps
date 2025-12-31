/**
 * @file spmv_csr5.cpp
 * @author your name (you@domain.com)
 * @brief   Using weifeng Liu code
 *          Implement on AVX521, must required.
 * @version 0.1
 * @date 2023-12-24
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include"../include/LeSpMV.h"

#include"../include/thread.h"

#include"immintrin.h"
/*
    https://stackoverflow.com/questions/34428061/inline-assembly-of-reduce-operation-for-xeon-phi
    for KNC intrincs to AVX512
*/

// inclusive prefix-sum scan
inline __m512d hscan_avx512(__m512d scan512d, __m512d zero512d)
{
    // register __m512d t0, t1;
    __m512d t0, t1;

    t0 = _mm512_permutex_pd(scan512d, 0xB1); //_mm512_swizzle_pd(scan512d, _MM_SWIZ_REG_CDAB);
    t1 = _mm512_permutex_pd(t0, 0x4E); //_mm512_swizzle_pd(t0, _MM_SWIZ_REG_BADC);
    t0 = _mm512_mask_blend_pd(0xAA, t1, t0);

    t1 = _mm512_mask_blend_pd(0x0F, zero512d, t0);
    // t1 = _mm512_castsi512_pd(_mm512_permute4f128_epi32(_mm512_castpd_si512(t1), _MM_PERM_BADC));
    t1 = _mm512_castsi512_pd(_mm512_shuffle_i32x4(_mm512_castpd_si512(t1), _mm512_castpd_si512(t1), _MM_PERM_BADC));

    scan512d = _mm512_add_pd(scan512d, _mm512_mask_blend_pd(0x11, t0, t1));
    
    t0 = _mm512_permutex_pd(scan512d, 0x4E); //_mm512_swizzle_pd(scan512d, _MM_SWIZ_REG_BADC);
    
    t1 = _mm512_mask_blend_pd(0x0F, zero512d, t0);
    // t1 = _mm512_castsi512_pd(_mm512_permute4f128_epi32(_mm512_castpd_si512(t1), _MM_PERM_BADC));
    t1 = _mm512_castsi512_pd(_mm512_shuffle_i32x4(_mm512_castpd_si512(t1), _mm512_castpd_si512(t1), _MM_PERM_BADC));

    
    scan512d = _mm512_add_pd(scan512d, _mm512_mask_blend_pd(0x33, t0, t1));
    
    t1 = _mm512_mask_blend_pd(0x0F, zero512d, scan512d);
    // t1 = _mm512_castsi512_pd(_mm512_permute4f128_epi32(_mm512_castpd_si512(t1), _MM_PERM_BADC));
    t1 = _mm512_castsi512_pd(_mm512_shuffle_i32x4(_mm512_castpd_si512(t1), _mm512_castpd_si512(t1), _MM_PERM_BADC));

    scan512d = _mm512_add_pd(scan512d, t1);

    return scan512d;
}

template<typename iT, typename vT>
void partition_fast_track(const vT           *d_value_partition,
                                 const vT           *d_x,
                                 const iT           *d_column_index_partition,
                                 vT                 *d_calibrator,
                                 vT                 *d_y,
                                 const iT            row_start,
                                 const iT            par_id,
                                 const int           tid,
                                 const iT            start_row_start,
                                 const vT            alpha,
                                 const vT            beta ,
                                 const int           sigma,
                                 const int           omega,
                                 const int           stride_vT,
                                 const bool          direct)
{

    __m512d sum512d = _mm512_setzero_pd();
    __m512d value512d, x512d;
    __m512i column_index512i;
    
    #pragma unroll(CSR5_SIGMA)
    for (int i = 0; i < CSR5_SIGMA; i++)
    {
        value512d = _mm512_load_pd(&d_value_partition[i * D_CSR5_OMEGA]);
        // column_index512i = (i % 2) ?
        //             _mm512_permute4f128_epi32(column_index512i, _MM_PERM_BADC) :
        //             _mm512_load_epi32(&d_column_index_partition[i * omega]);
        column_index512i = (i % 2) ?
                    _mm512_shuffle_i32x4(column_index512i, column_index512i, _MM_PERM_BADC) :
                    _mm512_load_epi32(&d_column_index_partition[i * D_CSR5_OMEGA]);
        x512d = _mm512_i32logather_pd(column_index512i, d_x, 8);
        sum512d = _mm512_fmadd_pd(value512d, x512d, sum512d); // csr5.value * x = sum
    }

    vT sum = _mm512_reduce_add_pd(sum512d);
    sum = sum * alpha;

    if (row_start == start_row_start && !direct)
        d_calibrator[tid * stride_vT] += sum;
    else
    {
        // if(direct)
        //     d_y[row_start] = sum;
        // else
        //     d_y[row_start] += sum;
        if(direct)
            d_y[row_start] = beta * d_y[row_start] + sum;
        else
            d_y[row_start] += sum;
    }
}


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
                              const int           c_omega)
{
    const int num_thread = Le_get_thread_num();
    const int chunk = ceil((double)(p-1) / (double)num_thread);

    const __m512d c_zero512d        = _mm512_setzero_pd();
    const __m512i c_one512i         = _mm512_set1_epi32(1);

    const int stride_vT = CACHE_LINE / sizeof(vT);
    const int num_thread_active = ceil((p-1.0)/chunk);

    #pragma omp parallel
    {
        int tid = Le_get_thread_id();
        iT start_row_start = tid < num_thread_active ? d_partition_pointer[tid * chunk] & 0x7FFFFFFF : 0;

        __m512d value512d;
        __m512d x512d;
        __m512i column_index512i;
        __m512d alpha512_d = _mm512_set1_pd(alpha);

        __m512d sum512d = c_zero512d;
        __m512d tmp_sum512d = c_zero512d;
        __m512d first_sum512d = c_zero512d;
        __m512d last_sum512d = c_zero512d;

        __m512i scansum_offset512i;
        __m512i y_offset512i;
        __m512i y_idx512i;
        __m512i start512i;
        __m512i stop512i;
        __m512i descriptor512i;

        __mmask16 local_bit16;
        __mmask16 direct16;

        #pragma omp for schedule(static, chunk)
        for (int par_id = 0; par_id < p - 1; par_id++)
        {
            const vT *d_value_partition = &d_value[par_id * D_CSR5_OMEGA * c_sigma];
            const int *d_column_index_partition = &d_column_index[par_id * D_CSR5_OMEGA * c_sigma];

            uiT row_start     = d_partition_pointer[par_id];
            const iT row_stop = d_partition_pointer[par_id + 1] & 0x7FFFFFFF;

            if (row_start == row_stop) // fast track through reduction
            {
                // check whether the the partition contains the first element of row "row_start"
                // => we are the first writing data to d_y[row_start]
                bool fast_direct = (d_partition_descriptor[par_id * D_CSR5_OMEGA * num_packet] >>
                                                    (31 - (bit_y_offset + bit_scansum_offset)) & 0x1);
                partition_fast_track<iT, vT>
                        (d_value_partition, d_x, d_column_index_partition,
                         d_calibrator, d_y, row_start, par_id,
                         tid, start_row_start, alpha, beta, c_sigma, D_CSR5_OMEGA, stride_vT, fast_direct);
            }
            else // normal track for all the other partitions
            {
                const bool empty_rows = (row_start >> 31) & 0x1;
                row_start &= 0x7FFFFFFF;

                vT *d_y_local = &d_y[row_start+1];
                const int offset_pointer = empty_rows ? d_partition_descriptor_offset_pointer[par_id] : 0;

                __mmask8 storemask8;

                first_sum512d = c_zero512d;
                stop512i = _mm512_castpd_si512(first_sum512d);
#if CSR5_SIGMA > 20
                const uiT *d_partition_descriptor_partition = &d_partition_descriptor[par_id * D_CSR5_OMEGA * num_packet];
                descriptor512i = _mm512_mask_i32gather_epi32(stop512i, 0xFF, _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0), 
                                                             d_partition_descriptor_partition, 4);

#else
                if(par_id % 2)
                {
                    descriptor512i = _mm512_load_epi32(&d_partition_descriptor[(par_id-1) * D_CSR5_OMEGA * num_packet]);
                    // descriptor512i = _mm512_permute4f128_epi32(descriptor512i, _MM_PERM_BADC);
                    descriptor512i = _mm512_shuffle_i32x4(descriptor512i, descriptor512i, _MM_PERM_BADC);
                }
                else
                    descriptor512i = _mm512_load_epi32(&d_partition_descriptor[par_id * D_CSR5_OMEGA * num_packet]);
#endif

                y_offset512i = _mm512_srli_epi32(descriptor512i, 32 - bit_y_offset);
                scansum_offset512i = _mm512_slli_epi32(descriptor512i, bit_y_offset);
                scansum_offset512i = _mm512_srli_epi32(scansum_offset512i, 32 - bit_scansum_offset);

                descriptor512i = _mm512_slli_epi32(descriptor512i, bit_y_offset + bit_scansum_offset);

                local_bit16 = _mm512_cmp_epi32_mask(_mm512_srli_epi32(descriptor512i, 31), c_one512i, _MM_CMPINT_EQ);
                
                // remember if the first element of this partition is the first element of a new row
                bool first_direct = false;
                if(local_bit16 & 0x1)
                    first_direct = true;
                    
                // remember if the first element of the first partition of the current thread is the first element of a new row
                bool first_all_direct = false;
                if(par_id == tid * chunk)
                    first_all_direct = first_direct;
                    
                local_bit16 |= 0x1;

                start512i = _mm512_mask_blend_epi32(local_bit16, c_one512i, _mm512_setzero_epi32());
                direct16 = _mm512_kand(local_bit16, 0xFE);

                value512d = _mm512_load_pd(d_value_partition);

                column_index512i = _mm512_load_epi32(d_column_index_partition);
                x512d = _mm512_i32logather_pd(column_index512i, d_x, 8);
                x512d = _mm512_mul_pd(x512d, alpha512_d);   // x = alpha * x

                sum512d = _mm512_mul_pd(value512d, x512d);
                // sum512d = _mm512_mul_pd(sum512d, alpha512_d);  // * alpha

                // step 1. thread-level seg sum
#if CSR5_SIGMA > 20
                int ly = 0;
#endif
                #pragma unroll(CSR5_SIGMA-1)
                for (int i = 1; i < CSR5_SIGMA; i++)
                {
                    // column_index512i = (i % 2) ?
                    //             _mm512_permute4f128_epi32(column_index512i, _MM_PERM_BADC) :
                    //             _mm512_load_epi32(&d_column_index_partition[i * c_omega]);
                    column_index512i = (i % 2) ?
                                _mm512_shuffle_i32x4(column_index512i, column_index512i, _MM_PERM_BADC) :
                                _mm512_load_epi32(&d_column_index_partition[i * D_CSR5_OMEGA]);

#if CSR5_SIGMA > 20
                    int norm_i = i - (32 - bit_y_offset - bit_scansum_offset);

                    if (!(ly || norm_i) || (ly && !(norm_i % 32)))
                    {
                        ly++;
                        descriptor512i = _mm512_mask_i32gather_epi32(stop512i, 0xFF, _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0), 
                                                                     &d_partition_descriptor_partition[ly * D_CSR5_OMEGA], 4);
                    }
                    norm_i = !ly ? i : norm_i;
                    norm_i = 31 - norm_i % 32;

                    local_bit16 = _mm512_cmp_epi32_mask(_mm512_and_epi32(_mm512_srli_epi32(descriptor512i, norm_i), c_one512i), c_one512i, _MM_CMPINT_EQ);
#else
                    local_bit16 = _mm512_cmp_epi32_mask(_mm512_and_epi32(_mm512_srli_epi32(descriptor512i, 31-i), c_one512i), c_one512i, _MM_CMPINT_EQ);
#endif

                    if (local_bit16 & 0xFF)
                    {

                        //// mask scatter
                        storemask8 = _mm512_kand(direct16, local_bit16) & 0xFF;
                        if (storemask8)
                        {
                            y_idx512i = empty_rows ? 
                                    _mm512_mask_i32gather_epi32(y_offset512i, storemask8, y_offset512i, &d_partition_descriptor_offset[offset_pointer], 4) : 
                                    y_offset512i;
                            //  应该是这里需要改成 beta * y + sum, 暂时没改
                            _mm512_mask_i32loscatter_pd(d_y_local, storemask8, y_idx512i, sum512d, 8);
                            y_offset512i = _mm512_mask_add_epi32(y_offset512i, storemask8, y_offset512i, c_one512i);
                        }

                        storemask8 = _mm512_kandn(direct16, local_bit16) & 0xFF;
                        first_sum512d = _mm512_mask_blend_pd(storemask8, first_sum512d, sum512d);

                        storemask8 = local_bit16 & 0xFF;
                        sum512d = _mm512_mask_blend_pd(storemask8, sum512d, c_zero512d);

                        direct16 = _mm512_kor(local_bit16, direct16);
                        stop512i = _mm512_mask_add_epi32(stop512i, direct16, stop512i, c_one512i);
                    }

                    value512d = _mm512_load_pd(&d_value_partition[i * D_CSR5_OMEGA]);
                    x512d = _mm512_i32logather_pd(column_index512i, d_x, 8);
                    x512d = _mm512_mul_pd(x512d, alpha512_d);   // x = alpha * x
                    sum512d = _mm512_fmadd_pd(value512d, x512d, sum512d);

                }

                storemask8 = direct16 & 0xFF;
                first_sum512d = _mm512_mask_blend_pd(storemask8, sum512d, first_sum512d);

                last_sum512d = sum512d;

                storemask8 = _mm512_cmp_epi32_mask(start512i, c_one512i, _MM_CMPINT_EQ) & 0xFF;
                sum512d = _mm512_mask_blend_pd(storemask8, c_zero512d, first_sum512d);

                sum512d = _mm512_castsi512_pd(_mm512_permutevar_epi32(_mm512_set_epi32(1,0,15,14,13,12,11,10,9,8,7,6,5,4,3,2), _mm512_castpd_si512(sum512d)));
                sum512d = _mm512_mask_blend_pd(0x80, sum512d, c_zero512d);

                tmp_sum512d = sum512d;
                sum512d = hscan_avx512(sum512d, c_zero512d);

                scansum_offset512i = _mm512_add_epi32(scansum_offset512i, _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0));
                scansum_offset512i = _mm512_permutevar_epi32(_mm512_set_epi32(7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0), scansum_offset512i);
                scansum_offset512i = _mm512_add_epi32(scansum_offset512i, scansum_offset512i);
                scansum_offset512i = _mm512_add_epi32(scansum_offset512i, _mm512_set_epi32(1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0));

                sum512d = _mm512_sub_pd(_mm512_castsi512_pd(_mm512_permutevar_epi32(scansum_offset512i, _mm512_castpd_si512(sum512d))), sum512d);
                sum512d = _mm512_add_pd(sum512d, tmp_sum512d);

                storemask8 = _mm512_cmp_epi32_mask(start512i, stop512i, _MM_CMPINT_LE) & 0xFF;
                last_sum512d = _mm512_add_pd(last_sum512d, _mm512_mask_blend_pd(storemask8, c_zero512d, sum512d));

                // mask scatter
                storemask8 = direct16 & 0xFF;
                if (storemask8)
                {
                    y_idx512i = empty_rows ? 
                                _mm512_mask_i32gather_epi32(y_offset512i, direct16, y_offset512i, &d_partition_descriptor_offset[offset_pointer], 4) : 
                                y_offset512i;
                    _mm512_mask_i32loscatter_pd(d_y_local, storemask8, y_idx512i, last_sum512d, 8);
                }

                sum512d = _mm512_mask_blend_pd(storemask8, last_sum512d, first_sum512d);
                sum512d = _mm512_mask_blend_pd(0x1, c_zero512d, sum512d);
                vT sum = _mm512_mask_reduce_add_pd(0x1, sum512d);

                if (row_start == start_row_start && !first_all_direct)
                    d_calibrator[tid * stride_vT] += sum;
                else
                {
                    // if(first_direct)
                    //     d_y[row_start] = sum;
                    // else
                    //     d_y[row_start] += sum;
                    if(first_direct)
                        d_y[row_start] = beta * d_y[row_start] + sum;
                    else
                        d_y[row_start] += sum;
                }

            }
        }
    }
}

template<typename iT, typename uiT, typename vT>
void spmv_csr5_calibrate_kernel(const uiT *d_partition_pointer,
                                vT        *d_calibrator,
                                vT        *d_y,
                                const iT   p)
{
    const int num_thread = Le_get_thread_num();
    const int chunk = ceil((double)(p-1) / (double)num_thread);
    const int stride_vT = CACHE_LINE / sizeof(vT);
    // calculate the number of maximal active threads (for a static loop scheduling with size chunk)
    int num_thread_active = ceil((p-1.0)/chunk);
    int num_cali = num_thread_active < num_thread ? num_thread_active : num_thread;

    for (int i = 0; i < num_cali; i++)
    {
        d_y[(d_partition_pointer[i * chunk] << 1) >> 1] += d_calibrator[i * stride_vT];
    }
}

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
                                     const vT            beta)
{
    const iT index_first_element_tail = (p - 1) * D_CSR5_OMEGA * sigma;
    
    for (iT row_id = tail_partition_start; row_id < m; row_id++)
    {
        const iT idx_start = row_id == tail_partition_start ? (p - 1) * D_CSR5_OMEGA * sigma : d_row_pointer[row_id];
        const iT idx_stop  = d_row_pointer[row_id + 1];

        vT sum = 0;
        for (iT idx = idx_start; idx < idx_stop; idx++)
        {
            // sum += d_value[idx] * d_x[d_column_index[idx]];
            sum += d_value[idx] * d_x[d_column_index[idx]] * alpha;  // * alpha;
        }

        if(row_id == tail_partition_start && d_row_pointer[row_id] != index_first_element_tail)
        {
            d_y[row_id] = d_y[row_id] + sum;
        }
        else
        {
            // d_y[row_id] = sum;
            d_y[row_id] = d_y[row_id] * beta + sum;
        }
    }
}                            

/**
 * @brief CSR spmv from Liu wei feng's code. Need to test correctness 12.24.2023.
 * 
 * @tparam IndexType 
 * @tparam UIndexType 
 * @tparam ValueType 
 * @param alpha 
 * @param csr5 
 * @param x 
 * @param beta 
 * @param y 
 */
template <typename IndexType, typename UIndexType, typename ValueType>
void LeSpMV_csr5(const ValueType alpha, const CSR5_Matrix<IndexType, UIndexType, ValueType>& csr5, const ValueType * x, const ValueType beta, ValueType * y)
{
    spmv_csr5_compute_kernel
            <IndexType, UIndexType, ValueType>
            (csr5.col_index, csr5.values, csr5.row_offset, x,
             csr5.tile_ptr, csr5.tile_desc,
             csr5.tile_desc_offset_ptr, csr5.tile_desc_offset,
             csr5.calibrator, y, csr5._p,
             csr5.num_packets, csr5.bit_y_offset, csr5.bit_scansum_offset, alpha, beta, csr5.sigma, csr5.omega);

    spmv_csr5_calibrate_kernel
            <IndexType, UIndexType, ValueType>
            (csr5.tile_ptr, csr5.calibrator, y, csr5._p);

    spmv_csr5_tail_partition_kernel
            <IndexType, UIndexType, ValueType>
            (csr5.row_offset, csr5.col_index, csr5.values, x, y,
             csr5.tail_partition_start, csr5._p, csr5.num_rows, csr5.sigma, csr5.omega, alpha, beta);
}

template void LeSpMV_csr5<int, uint32_t, double>(const double, const CSR5_Matrix<int, uint32_t, double>&, const double* , const double, double*);