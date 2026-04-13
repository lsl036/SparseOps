#ifndef CONFIG_H
#define CONFIG_H

#define CPU_FREQUENCY e6
#define CPU_MAX_FREQUENCY 3720.7029e6
#define CPU_SOCKET 2
#define CPU_CORES_PER_SOC 64
#define CPU_HYPER_THREAD 2
#define NUMA_REGIONS 8

// Cache size in Bytes
#define CPU_L3CACHE_SIZE 536870912
#define CPU_L2CACHE_SIZE 67108864
#define CPU_L1DCACHE_SIZE 4194304
#define CPU_L1IACHE_SIZE 4194304
#define CACHE_LINE 64 // bytes

// Main Memory size in Giga Bytes
#define MAIN_MEM_SIZE 503

// SIMD width for platform
#define SIMD_WIDTH 256
#define S_CSR5_OMEGA 8
#define D_CSR5_OMEGA 4

#define S_ALIGNMENT 8
#define D_ALIGNMENT 4

#endif // CONFIG_H
