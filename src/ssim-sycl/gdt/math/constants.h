// ======================================================================== //
// Copyright 2018 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <sycl/sycl.hpp>
#include "gdt/math/vec.h"
#ifdef DPCT_COMPATIBILITY_TEMP
#endif
#include <limits>

// #ifndef M_PI
// #define M_PI 3.141593f
// #endif
#include <cmath>

namespace gdt {

  constexpr struct ZeroTy
  {
    __both__ operator          double   ( ) const { return 0; }
    __both__ operator          float    ( ) const { return 0; }
    __both__ operator          long long( ) const { return 0; }
    __both__ operator unsigned long long( ) const { return 0; }
    __both__ operator          long     ( ) const { return 0; }
    __both__ operator unsigned long     ( ) const { return 0; }
    __both__ operator          int      ( ) const { return 0; }
    __both__ operator unsigned int      ( ) const { return 0; }
    __both__ operator          short    ( ) const { return 0; }
    __both__ operator unsigned short    ( ) const { return 0; }
    __both__ operator          char     ( ) const { return 0; }
    __both__ operator unsigned char     ( ) const { return 0; }
  } __gdt_device zero MAYBE_UNUSED;

  constexpr struct OneTy
  {
    __both__ operator          double   ( ) const { return 1; }
    __both__ operator          float    ( ) const { return 1; }
    __both__ operator          long long( ) const { return 1; }
    __both__ operator unsigned long long( ) const { return 1; }
    __both__ operator          long     ( ) const { return 1; }
    __both__ operator unsigned long     ( ) const { return 1; }
    __both__ operator          int      ( ) const { return 1; }
    __both__ operator unsigned int      ( ) const { return 1; }
    __both__ operator          short    ( ) const { return 1; }
    __both__ operator unsigned short    ( ) const { return 1; }
    __both__ operator          char     ( ) const { return 1; }
    __both__ operator unsigned char     ( ) const { return 1; }
  } __gdt_device one MAYBE_UNUSED;

  inline __both__ float infty() 
  {

    return sycl::bit_cast<float, int>(0x7f800000U);
  }

  constexpr struct NegInfTy
  {
#ifdef SYCL_LANGUAGE_VERSION
    operator double() const {
      return -sycl::bit_cast<double, long long>(0x7ff0000000000000ULL);
    }
    operator float() const { return -sycl::bit_cast<float, int>(0x7f800000U); }
#else
    __both__ operator          double   ( ) const { return -std::numeric_limits<double>::infinity(); }
    __both__ operator          float    ( ) const { return -std::numeric_limits<float>::infinity(); }
    __both__ operator          long long( ) const { return std::numeric_limits<long long>::min(); }
    __both__ operator unsigned long long( ) const { return std::numeric_limits<unsigned long long>::min(); }
    __both__ operator          long     ( ) const { return std::numeric_limits<long>::min(); }
    __both__ operator unsigned long     ( ) const { return std::numeric_limits<unsigned long>::min(); }
    __both__ operator          int      ( ) const { return std::numeric_limits<int>::min(); }
    __both__ operator unsigned int      ( ) const { return std::numeric_limits<unsigned int>::min(); }
    __both__ operator          short    ( ) const { return std::numeric_limits<short>::min(); }
    __both__ operator unsigned short    ( ) const { return std::numeric_limits<unsigned short>::min(); }
    __both__ operator          char     ( ) const { return std::numeric_limits<char>::min(); }
    __both__ operator unsigned char     ( ) const { return std::numeric_limits<unsigned char>::min(); }
#endif
  } neg_inf MAYBE_UNUSED;

  constexpr struct PosInfTy
  {
#ifdef SYCL_LANGUAGE_VERSION
    operator double() const {
      return sycl::bit_cast<double, long long>(0x7ff0000000000000ULL);
    }
    operator float() const { return sycl::bit_cast<float, int>(0x7f800000U); }
#else
    __both__ operator          double   ( ) const { return std::numeric_limits<double>::infinity(); }
    __both__ operator          float    ( ) const { return std::numeric_limits<float>::infinity(); }
    __both__ operator          long long( ) const { return std::numeric_limits<long long>::max(); }
    __both__ operator unsigned long long( ) const { return std::numeric_limits<unsigned long long>::max(); }
    __both__ operator          long     ( ) const { return std::numeric_limits<long>::max(); }
    __both__ operator unsigned long     ( ) const { return std::numeric_limits<unsigned long>::max(); }
    __both__ operator          int      ( ) const { return std::numeric_limits<int>::max(); }
    __both__ operator unsigned int      ( ) const { return std::numeric_limits<unsigned int>::max(); }
    __both__ operator          short    ( ) const { return std::numeric_limits<short>::max(); }
    __both__ operator unsigned short    ( ) const { return std::numeric_limits<unsigned short>::max(); }
    __both__ operator          char     ( ) const { return std::numeric_limits<char>::max(); }
    __both__ operator unsigned char     ( ) const { return std::numeric_limits<unsigned char>::max(); }
#endif
  } inf MAYBE_UNUSED, pos_inf MAYBE_UNUSED;

  constexpr struct NaNTy
  {
#ifdef SYCL_LANGUAGE_VERSION
    operator double() const { return sycl::bit_cast<float, int>(0x7fffffffU); }
    operator float() const {
      return sycl::bit_cast<double, long long>(0xfff8000000000000ULL);
    }
#else
    __both__ operator double( ) const { return std::numeric_limits<double>::quiet_NaN(); }
    __both__ operator float ( ) const { return std::numeric_limits<float>::quiet_NaN(); }
#endif
  } nan MAYBE_UNUSED;

  constexpr struct UlpTy
  {
#ifdef SYCL_LANGUAGE_VERSION
    // todo
#else
    __both__ operator double( ) const { return std::numeric_limits<double>::epsilon(); }
    __both__ operator float ( ) const { return std::numeric_limits<float>::epsilon(); }
#endif
  } ulp MAYBE_UNUSED;

  template<bool is_integer>
  struct limits_traits;

  template<> struct limits_traits<true> {
    template<typename T> static inline __both__ T value_limits_lower(T) { return std::numeric_limits<T>::min(); }
    template<typename T> static inline __both__ T value_limits_upper(T) { return std::numeric_limits<T>::max(); }
  };
  template<> struct limits_traits<false> {
    template<typename T> static inline __both__ T value_limits_lower(T) { return (T)NegInfTy(); }//{ return -std::numeric_limits<T>::infinity(); }
    template<typename T> static inline __both__ T value_limits_upper(T) { return (T)PosInfTy(); }//{ return +std::numeric_limits<T>::infinity();  }
  };
  
  /*! lower value of a completely *empty* range [+inf..-inf] */
  template<typename T> inline __both__ T empty_bounds_lower()
  {
    return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_upper(T());
  }
  
  /*! upper value of a completely *empty* range [+inf..-inf] */
  template<typename T> inline __both__ T empty_bounds_upper()
  {
    return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_lower(T());
  }

  /*! lower value of a completely *empty* range [+inf..-inf] */
  template<typename T> inline __both__ T empty_range_lower()
  {
    return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_upper(T());
  }
  
  /*! upper value of a completely *empty* range [+inf..-inf] */
  template<typename T> inline __both__ T empty_range_upper()
  {
    return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_lower(T());
  }

  /*! lower value of a completely open range [-inf..+inf] */
  template<typename T> inline __both__ T open_range_lower()
  {
    return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_lower(T());
  }

  /*! upper value of a completely open range [-inf..+inf] */
  template<typename T> inline __both__ T open_range_upper()
  {
    return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_upper(T());
  }
  
} // ::gdt
