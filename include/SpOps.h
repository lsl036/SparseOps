#ifndef SPOPS_H
#define SPOPS_H

// general config
#include"general_config.h"
#include"plat_config.h"
#include"memopt.h"
#include"mmio.h"
#include"thread.h"
#ifdef ENABLE_CSR5
#include"csr5_utils.h"
#endif


// sparse utils
#include"sparse_io.h"
#include"sparse_format.h"
#include"sparse_operation.h"
#include"sparse_partition.h"
#include"sparse_conversion.h"
#include"spmv_benchmark.h"
#include"spmv_testroutine.h"
#include"sparse_features.h"

// SpMV algorithms
#include"spmv_csr.h"
#include"spmv_bsr.h"
#ifdef ENABLE_CSR5
#include"spmv_csr5.h"
#endif
#include"spmv_coo.h"
#include"spmv_ell.h"
#include"spmv_dia.h"
#include"spmv_s_ell.h"
#include"spmv_sell_c_sigma.h"
#include"spmv_sell_c_R.h"

// SpGEMM algorithms
#include"spgemm.h"

#endif /* SPOPS_H */

