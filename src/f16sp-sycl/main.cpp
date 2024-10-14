/**
 * Copyright 2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>
#include <sycl/sycl.hpp>

#define NUM_OF_BLOCKS (1024 * 16)
#define NUM_OF_THREADS 128

inline
void reduceInShared_native(sycl::half2 *const v, sycl::nd_item<1> &item)
{
  int lid = item.get_local_id(0);
  #pragma unroll
  for (int i = NUM_OF_THREADS/2; i >= 1; i = i / 2) {
    if(lid<i) v[lid] = v[lid] + v[lid+i];
    item.barrier(sycl::access::fence_space::local_space);
  }
}

void scalarProductKernel_native(const sycl::half2 *a,
                                const sycl::half2 *b,
                                float *results, 
                                      sycl::half2 *shArray,
                                const size_t size,
                                sycl::nd_item<1> &item)
{
  int lid = item.get_local_id(0);
  int gid = item.get_group(0); 

  const int stride = item.get_group_range(0) * item.get_local_range(0);

  sycl::half2 value(0.f, 0.f);
  shArray[lid] = value;

  for (int i = item.get_global_id(0); i < size; i += stride)
  {
    value = a[i] * b[i] + value;
  }

  shArray[lid] = value;
  item.barrier(sycl::access::fence_space::local_space);
  reduceInShared_native(shArray, item);

  if (lid == 0)
  {
    sycl::half2 result = shArray[0];
    float f_result = (float)result.y() + (float)result.x();
    auto ao = sycl::atomic_ref<float, sycl::memory_order::relaxed, \
                     sycl::memory_scope::device,\
                     sycl::access::address_space::global_space>(results[0]);
    ao.fetch_add(f_result);
  }
}

void generateInput(sycl::half2 *a, size_t size)
{
  for (size_t i = 0; i < size; ++i)
  {
    sycl::half2 temp;
    temp.x() = -1;
    temp.y() = -1;
    a[i] = temp;
  }
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const size_t size = NUM_OF_BLOCKS*NUM_OF_THREADS;
  const size_t size_bytes = size * sizeof(sycl::half2);
  const size_t result_bytes = sizeof(float);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  sycl::half2 *a = (sycl::half2 *) malloc (size_bytes);
  sycl::half2 *b = (sycl::half2 *) malloc (size_bytes);
  float r;

  float *d_r = sycl::malloc_device<float>(NUM_OF_BLOCKS, q);

  srand(123); 
  generateInput(a, size);
  sycl::half2 *d_a = sycl::malloc_device<sycl::half2>(size, q);
  q.memcpy(d_a, a, size_bytes);

  generateInput(b, size);
  sycl::half2 *d_b = sycl::malloc_device<sycl::half2>(size, q);
  q.memcpy(d_b, b, size_bytes);

  float result_ref = 0.f;
  for (size_t i = 0; i < size; i++)
  {
    result_ref += (float)a[i].x() * (float)b[i].x() +
                  (float)a[i].y() * (float)b[i].y();
  }
  //printf("Result reference: %f\n", result_ref);

  sycl::range<1> gws (NUM_OF_BLOCKS * NUM_OF_THREADS);
  sycl::range<1> lws (NUM_OF_THREADS);

  // warmup
  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<sycl::half2> shArray(sycl::range<1>(NUM_OF_THREADS), cgh);
      cgh.parallel_for<class warm_sp2>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        scalarProductKernel_native(
          d_a,
          d_b,
          d_r,
          shArray.get_pointer(),
          size, item);
      });
    });
  }
  q.wait();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.memset(d_r, 0, result_bytes);
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<sycl::half2> shArray(sycl::range<1>(NUM_OF_THREADS), cgh);
      cgh.parallel_for<class sp2>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        scalarProductKernel_native(
          d_a,
          d_b,
          d_r,
          shArray.get_pointer(),
          size, item);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);

  q.memcpy(&r, d_r, result_bytes).wait();

  printf("Result (native operators)\t: %f \n", r);

  bool ok = fabsf(r - result_ref) < 0.00001f;
  printf("fp16ScalarProduct %s\n", ok ?  "PASS" : "FAIL");

  sycl::free(d_a, q);
  sycl::free(d_b, q);
  sycl::free(d_r, q);
  free(a);
  free(b);

  return EXIT_SUCCESS;
}
