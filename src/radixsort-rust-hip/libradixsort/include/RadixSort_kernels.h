#ifndef _RADIXSORT_KERNELSHIP_H_
#define _RADIXSORT_KERNELSHIP_H_

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