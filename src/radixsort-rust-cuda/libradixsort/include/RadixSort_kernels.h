/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

//----------------------------------------------------------------------------
// Scans each warp in parallel ("warp-scan"), one element per thread.
// uses 2 numElements of shared memory per thread (64 = elements per warp)
//----------------------------------------------------------------------------
#ifndef _RADIXSORT_KERNELS_H_
#define _RADIXSORT_KERNELS_H_

#include "RadixSort.h"

__global__ void radixSortBlocksKeysK(
   const unsigned int*__restrict__ keysIn,
         unsigned int*__restrict__ keysOut,
   const unsigned int nbits,
   const unsigned int startbit);

__global__ void findRadixOffsetsK(
    const unsigned int*__restrict__ keys,
          unsigned int*__restrict__ counters,
          unsigned int*__restrict__ blockOffsets,
    const unsigned int startbit,
    const unsigned int totalBlocks);

__global__ void reorderDataKeysOnlyK(
          unsigned int*__restrict__ outKeys,
    const unsigned int*__restrict__ keys,
          unsigned int*__restrict__ blockOffsets,
    const unsigned int*__restrict__ offsets,
    const unsigned int startbit,
    const unsigned int numElements,
    const unsigned int totalBlocks);

#endif