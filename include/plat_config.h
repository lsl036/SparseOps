#ifndef CONFIG_H
#define CONFIG_H

#define CPU_FREQUENCY 1000.183e6
#define CPU_MAX_FREQUENCY 3200.0000e6
#define CPU_SOCKET 2
#define CPU_CORES_PER_SOC 14
#define CPU_HYPER_THREAD 2
#define NUMA_REGIONS 2

// Cache size in Bytes
#define CPU_L3CACHE_SIZE 403701760
#define CPU_L2CACHE_SIZE 29360128
#define CPU_L1DCACHE_SIZE 917504
#define CPU_L1IACHE_SIZE 917504
#define CACHE_LINE 64 // bytes

// Main Memory size in Giga Bytes
#define MAIN_MEM_SIZE 251

// SIMD width for platform
#define SIMD_WIDTH 512
#define S_CSR5_OMEGA 16
#define D_CSR5_OMEGA 8

#define S_ALIGNMENT 16
#define D_ALIGNMENT 8

#endif // CONFIG_H
