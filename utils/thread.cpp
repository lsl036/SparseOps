
#include"../include/thread.h"
#include<stdio.h>

int _thread_num;

int Le_get_core_num()
{
#ifdef _OPENMP
    return omp_get_num_procs();
#else
    return 1;
#endif
}

void Le_set_thread_num(const int thread_num)
{
#ifdef _OPENMP
    _thread_num = thread_num;
#else
    _thread_num = 1;
#endif
}

int Le_get_thread_num()
{
#ifdef _OPENMP
    return _thread_num == 0 ? Le_get_core_num() : _thread_num;
#else
    return 1;
#endif
}
int Le_get_thread_id()
{
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

void set_omp_schedule(int sche_mode, int chunk_size=0) {
#ifdef _OPENMP
    switch (sche_mode) {
        case 0:
        // OpenMP will divides iterations into chunks that are approximately equal in size and it distributes chunks to threads 
        // in order(Notice that is why static method different from others).
            omp_set_schedule(omp_sched_static, 0); // 使用默认的chunk size
            printf("===  OMP Static schedule strategy  === \n");
            break;
        case 1:
        // If you specify chunk-size variable, the iterations will be divide into iter_size / chunk_size chunks.
            omp_set_schedule(omp_sched_static, chunk_size);
            printf("===  OMP StaticConst schedule strategy  === \n");
            break;
        case 2:
        // OpenMP will still split task into iter_size/chunk_size chunks, but distribute trunks to threads dynamically without any specific order.
            omp_set_schedule(omp_sched_dynamic, chunk_size);
            printf("===  OMP Dynamic schedule strategy  === \n");
            break;
        case 3:
        // OpenMP will still split task into iter_size/chunk_size chunks, but distribute trunks to threads dynamically without any specific order.
            omp_set_schedule(omp_sched_guided, chunk_size);
            printf("===  OMP Guided schedule strategy  === \n");
            break;
        default:
            omp_set_schedule(omp_sched_static, 0); // 默认 Static
            printf("===  OMP Static schedule strategy  === \n");
            break;
    }
#endif
}