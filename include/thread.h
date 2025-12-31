#ifndef THREAD_H
#define THREAD_H
/*
 * @brief header for multithread utils
 */ 
#ifdef _OPENMP
#include <omp.h>
#endif

int Le_get_core_num();

void Le_set_thread_num(const int thread_num);

// get the avaliable number of threads setting by --threads=x
// if not setting, using omp_get_num_procs() to obtain
int Le_get_thread_num();

// get thread own ID index
int Le_get_thread_id();

void set_omp_schedule(int sche_mode, int chunk_size);

#endif /* THREAD_H */
