/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef MEMOPT_H
#define MEMOPT_H

#include <stdlib.h>
#include <cstring>
#include <cstdint>
#include <memory.h>
#include <malloc.h>
#include <stdio.h>
#include "plat_config.h"
////////////////////////////////////////////////////////////////////
// allocate and free data between host and device
////////////////////////////////////////////////////////////////////

/**
 * @brief Initialize a new arrary with T precision
 * 
 * @tparam T : float/double, size_t, int, eta.
 * @param N : the array's length
 * @return T* 
 */
template <typename T>
T* new_array(const size_t N){
    //dispatch on location
    // return (T*) malloc(N * sizeof(T));
    
    // aliment memory allocation
    return (T*) memalign(CACHE_LINE, (uint64_t) N * sizeof(T));
}

template <typename T>
void delete_array(T* p){
    free(p);
}

////////////////////////////////////////////////////////////////////
// transfer data between host and device
////////////////////////////////////////////////////////////////////

template <typename T>
void memcpy_array(T* dst, const T* src, const size_t length){
    memcpy(dst, src, sizeof(T) * length);
}

/**
 * @brief  Create a new array and use src to initialize it
 * 
 * @tparam T 
 * @param src 
 * @param N length of new array
 * @return T* 
 */
template <typename T>
T * copy_array(const T * src, const size_t N)
{
    T * dst = new_array<T>(N);
    memcpy_array<T>(dst, src, N);
    return dst;
}

#endif /* MEMOPT_H */
