#ifndef FILTER_H
#define FILTER_H

#include <sycl/sycl.hpp>

SYCL_EXTERNAL void
Short_to_Float(const short *imgNDVI, const unsigned char *imgQA, int n_X,
               int n_Y, int n_B, int n_Years, float *__restrict__ img_NDVI,
               float *__restrict__ img_QA, const sycl::nd_item<2> &item);

SYCL_EXTERNAL void Generate_NDVI_reference(
    float cosyear, int win_NDVI, const float *img_NDVI, const float *img_QA,
    int n_X, int n_Y, int n_B, int n_Years, float *__restrict__ reference_data,
    float *__restrict__ d_res_3, int *__restrict__ d_res_vec_res1,
    const sycl::nd_item<2> &item);

SYCL_EXTERNAL void Compute_d_res(const float *img_NDVI, const float *img_QA,
                                 const float *reference_data, int StartY,
                                 int TotalY, int Buffer_Up, int Buffer_Dn,
                                 int n_X, int n_Y, int n_B, int n_Years,
                                 int win, float *d_res,
                                 const sycl::nd_item<2> &item);

SYCL_EXTERNAL void
STSG_filter(const float *img_NDVI, const float *img_QA,
            const float *reference_data, int StartY, int TotalY, int Buffer_Up,
            int Buffer_Dn, int n_X, int n_Y, int n_B, int n_Years, int win,
            float sampcorr, int snow_address, float *__restrict__ vector_out,
            float *__restrict__ d_vector_in, float *__restrict__ d_res,
            float *__restrict__ d_res_3, int *__restrict__ d_index,
            const sycl::nd_item<2> &item);

#endif // !FILTER_H
