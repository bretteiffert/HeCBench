#include "Scan.h"

#ifndef _SCAN_KERNELS_H_
#define _SCAN_KERNELS_H_

unsigned int iSnapUp(const unsigned int dividend, const unsigned int divisor);

__global__  void scanExclusiveLocal1K(
            unsigned int*__restrict__ d_Dst,
      const unsigned int*__restrict__ d_Src,
      const unsigned int size);

//Exclusive scan of top elements of bottom-level scans (4 * THREADBLOCK_SIZE)
__global__ void scanExclusiveLocal2K(
            unsigned int*__restrict__ d_Buf,
            unsigned int*__restrict__ d_Dst,
      const unsigned int*__restrict__ d_Src,
      const unsigned int N,
      const unsigned int arrayLength);

//Final step of large-array scan: combine basic inclusive scan with exclusive scan of top elements of input arrays
__global__ void uniformUpdateK(
      unsigned int*__restrict__ d_Data,
      unsigned int*__restrict__ d_Buf);

#endif