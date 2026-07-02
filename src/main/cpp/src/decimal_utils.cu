/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "decimal_utils.hpp"
#include "jni_utils.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/fixed_point/detail/floating_conversion.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/cmath>
#include <cuda/std/functional>
#include <thrust/tabulate.h>

#include <cmath>
#include <cstddef>

namespace {

// Holds the 64-bit chunks of a 256-bit value
struct chunked256 {
  inline chunked256() = default;

  // sign-extend a 128-bit value into a chunked 256-bit value
  inline __device__ explicit chunked256(__int128_t x)
  {
    chunks[0]            = static_cast<uint64_t>(x);
    __int128_t x_shifted = x >> 64;
    chunks[1]            = static_cast<uint64_t>(x_shifted);
    chunks[2]            = static_cast<uint64_t>(x_shifted >> 64);
    chunks[3]            = chunks[2];
  }

  inline __device__ explicit chunked256(uint64_t a, uint64_t b, uint64_t c, uint64_t d)
  {
    chunks[0] = d;
    chunks[1] = c;
    chunks[2] = b;
    chunks[3] = a;
  }

  inline __device__ uint64_t operator[](int i) const { return chunks[i]; }
  inline __device__ uint64_t& operator[](int i) { return chunks[i]; }
  inline __device__ int64_t sign() const { return static_cast<int64_t>(chunks[3]) >> 63; }

  inline __device__ void add(int a) { add(chunked256(static_cast<__int128_t>(a))); }

  inline __device__ void add(chunked256 const& a)
  {
    __uint128_t carry_and_sum = 0;
    for (int i = 0; i < 4; ++i) {
      carry_and_sum += static_cast<__uint128_t>(chunks[i]) + a.chunks[i];
      chunks[i] = static_cast<uint64_t>(carry_and_sum);
      carry_and_sum >>= 64;
    }
  }

  inline __device__ void negate()
  {
    for (int i = 0; i < 4; i++) {
      chunks[i] = ~chunks[i];
    }
    add(1);
  }

  inline __device__ bool lt_unsigned(chunked256 const& other) const
  {
    for (int i = 3; i >= 0; i--) {
      if (chunks[i] < other.chunks[i]) {
        return true;
      } else if (chunks[i] > other.chunks[i]) {
        return false;
      }
    }
    return false;
  }

  inline __device__ bool gte_unsigned(chunked256 const& other) const { return !lt_unsigned(other); }

  inline __device__ __int128_t as_128_bits() const
  {
    return (static_cast<__int128_t>(chunks[1]) << 64) | chunks[0];
  }

  inline __device__ uint64_t as_64_bits() const { return chunks[0]; }

 private:
  uint64_t chunks[4];
};

struct divmod256 {
  chunked256 quotient;
  __int128_t remainder;
};

// Perform a 256-bit multiply in 64-bit chunks
__device__ chunked256 multiply(chunked256 const& a, chunked256 const& b)
{
  chunked256 r;
  __uint128_t mul;
  uint64_t carry = 0;
  for (int a_idx = 0; a_idx < 4; ++a_idx) {
    mul      = static_cast<__uint128_t>(a[a_idx]) * b[0] + carry;
    r[a_idx] = static_cast<uint64_t>(mul);
    carry    = static_cast<uint64_t>(mul >> 64);
  }
  for (int b_idx = 1; b_idx < 4; ++b_idx) {
    carry = 0;
    for (int a_idx = 0; a_idx < 4 - b_idx; ++a_idx) {
      int r_idx = a_idx + b_idx;
      mul       = static_cast<__uint128_t>(a[a_idx]) * b[b_idx] + r[r_idx] + carry;
      r[r_idx]  = static_cast<uint64_t>(mul);
      carry     = static_cast<uint64_t>(mul >> 64);
    }
  }
  return r;
}

__device__ divmod256 divide_unsigned(chunked256 const& n, __int128_t const& d)
{
  // TODO: FIXME this is long division, and so it is likely very slow...
  chunked256 q(0);
  __uint128_t r = 0;

  for (int i = 255; i >= 0; i--) {
    int block = i / 64;
    int bit   = i % 64;
    int read  = (int)((n[block] >> bit) & 0x01);
    r         = r << 1;
    r         = r | read;

    if (r >= d) {
      r               = r - d;
      int64_t bit_set = 1L << bit;
      q[block]        = q[block] | bit_set;
    }
  }
  return divmod256{q, static_cast<__int128_t>(r)};
}

__device__ divmod256 divide(chunked256 const& n, __int128_t const& d)
{
  // We assume that d is not 0. This is because we do the zero check,
  // if needed before calling divide so we can set an overflow properly.
  bool const is_n_neg = n.sign() < 0;
  bool const is_d_neg = d < 0;
  // When computing the absolute value we don't need to worry about overflowing
  // beause we are dealing with decimal numbers that should not go to
  // the maximum value that can be held by d or n
  chunked256 abs_n = n;
  if (is_n_neg) { abs_n.negate(); }

  __int128_t abs_d = is_d_neg ? -d : d;
  divmod256 result = divide_unsigned(abs_n, abs_d);

  if (is_d_neg != is_n_neg) { result.quotient.negate(); }

  if (is_n_neg) { result.remainder = -result.remainder; }

  return result;
}

__device__ chunked256 round_from_remainder(chunked256 const& q,
                                           __int128_t const& r,
                                           chunked256 const& n,
                                           __int128_t const& d)
{
  // We are going to round if the abs value of the remainder is >= half of the divisor
  // but if we divide the divisor in half, we can lose data so instead we are going to
  // multiply the remainder by 2
  __int128_t const double_remainder = r << 1;

  // But this too can lose data if multiplying by 2 pushes off the top bit, it is a
  // little more complicated than that because of negative numbers. That is okay
  // because if we lose information when multiplying, then we know that the number
  // is in a range that would have us round because the divisor has to fit within
  // an __int128_t.

  bool const need_inc =
    ((double_remainder >> 1) != r) ||                                 // if we lost info or
    (double_remainder < 0 ? -double_remainder : double_remainder) >=  // abs remainder is >=
      (d < 0 ? -d : d);                                               // abs divisor

  // To know which way to round, more specifically when the quotient is 0
  // we need to know what the sign of the quotient would have been. In this
  // case that happens if only one of the inputs was negative (xor)
  bool const is_n_neg   = n.sign() < 0;
  bool const is_d_neg   = d < 0;
  bool const round_down = is_n_neg != is_d_neg;

  int const round_inc = (need_inc ? (round_down ? -1 : 1) : 0);
  chunked256 ret      = q;
  ret.add(round_inc);
  return ret;
}

/**
 * Divide n by d and do half up rounding based off of the remainder returned.
 */
__device__ chunked256 divide_and_round(chunked256 const& n, __int128_t const& d)
{
  divmod256 div_result = divide(n, d);

  return round_from_remainder(div_result.quotient, div_result.remainder, n, d);
}

/**
 * Divide n by d and return the quotient. This is essentially what `DOWN` rounding does
 * in Java
 */
__device__ chunked256 integer_divide(chunked256 const& n, __int128_t const& d)
{
  divmod256 div_result = divide(n, d);
  // drop the remainder and only return the quotient
  return div_result.quotient;
}

inline __device__ chunked256 pow_ten(int exp)
{
  // Note that the body of this was generated using the following scala script
  /*
  import java.math.BigInteger
  import java.lang.Long.toHexString

  val lmax = new BigInteger(Long.MaxValue.toString())
  val mask = lmax.or(lmax.shiftLeft(64))

  def printAsInt128s(v: BigInteger): Unit = {
    val len = v.bitLength();
    System.out.print(s"0x${toHexString(v.shiftRight(192).longValue())}, ")
    System.out.print(s"0x${toHexString(v.shiftRight(128).longValue())}, ")
    System.out.print(s"0x${toHexString(v.shiftRight(64).longValue())}, ")
    System.out.print(s"0x${toHexString(v.longValue)}")
  }

  (0 until 77).foreach { exp =>
    val ret = BigInteger.TEN.pow(exp);
    System.out.println(s"    case $exp:")
    System.out.println(s"      //$ret")
    System.out.print("      return chunked256(")
    printAsInt128s(ret)
    System.out.println(");")
  }
  */
  switch (exp) {
    case 0:
      // 1
      return chunked256(0x0, 0x0, 0x0, 0x1);
    case 1:
      // 10
      return chunked256(0x0, 0x0, 0x0, 0xa);
    case 2:
      // 100
      return chunked256(0x0, 0x0, 0x0, 0x64);
    case 3:
      // 1000
      return chunked256(0x0, 0x0, 0x0, 0x3e8);
    case 4:
      // 10000
      return chunked256(0x0, 0x0, 0x0, 0x2710);
    case 5:
      // 100000
      return chunked256(0x0, 0x0, 0x0, 0x186a0);
    case 6:
      // 1000000
      return chunked256(0x0, 0x0, 0x0, 0xf4240);
    case 7:
      // 10000000
      return chunked256(0x0, 0x0, 0x0, 0x989680);
    case 8:
      // 100000000
      return chunked256(0x0, 0x0, 0x0, 0x5f5e100);
    case 9:
      // 1000000000
      return chunked256(0x0, 0x0, 0x0, 0x3b9aca00);
    case 10:
      // 10000000000
      return chunked256(0x0, 0x0, 0x0, 0x2540be400);
    case 11:
      // 100000000000
      return chunked256(0x0, 0x0, 0x0, 0x174876e800);
    case 12:
      // 1000000000000
      return chunked256(0x0, 0x0, 0x0, 0xe8d4a51000);
    case 13:
      // 10000000000000
      return chunked256(0x0, 0x0, 0x0, 0x9184e72a000);
    case 14:
      // 100000000000000
      return chunked256(0x0, 0x0, 0x0, 0x5af3107a4000);
    case 15:
      // 1000000000000000
      return chunked256(0x0, 0x0, 0x0, 0x38d7ea4c68000);
    case 16:
      // 10000000000000000
      return chunked256(0x0, 0x0, 0x0, 0x2386f26fc10000);
    case 17:
      // 100000000000000000
      return chunked256(0x0, 0x0, 0x0, 0x16345785d8a0000);
    case 18:
      // 1000000000000000000
      return chunked256(0x0, 0x0, 0x0, 0xde0b6b3a7640000);
    case 19:
      // 10000000000000000000
      return chunked256(0x0, 0x0, 0x0, 0x8ac7230489e80000);
    case 20:
      // 100000000000000000000
      return chunked256(0x0, 0x0, 0x5, 0x6bc75e2d63100000);
    case 21:
      // 1000000000000000000000
      return chunked256(0x0, 0x0, 0x36, 0x35c9adc5dea00000);
    case 22:
      // 10000000000000000000000
      return chunked256(0x0, 0x0, 0x21e, 0x19e0c9bab2400000);
    case 23:
      // 100000000000000000000000
      return chunked256(0x0, 0x0, 0x152d, 0x2c7e14af6800000);
    case 24:
      // 1000000000000000000000000
      return chunked256(0x0, 0x0, 0xd3c2, 0x1bcecceda1000000);
    case 25:
      // 10000000000000000000000000
      return chunked256(0x0, 0x0, 0x84595, 0x161401484a000000);
    case 26:
      // 100000000000000000000000000
      return chunked256(0x0, 0x0, 0x52b7d2, 0xdcc80cd2e4000000);
    case 27:
      // 1000000000000000000000000000
      return chunked256(0x0, 0x0, 0x33b2e3c, 0x9fd0803ce8000000);
    case 28:
      // 10000000000000000000000000000
      return chunked256(0x0, 0x0, 0x204fce5e, 0x3e25026110000000);
    case 29:
      // 100000000000000000000000000000
      return chunked256(0x0, 0x0, 0x1431e0fae, 0x6d7217caa0000000);
    case 30:
      // 1000000000000000000000000000000
      return chunked256(0x0, 0x0, 0xc9f2c9cd0, 0x4674edea40000000);
    case 31:
      // 10000000000000000000000000000000
      return chunked256(0x0, 0x0, 0x7e37be2022, 0xc0914b2680000000);
    case 32:
      // 100000000000000000000000000000000
      return chunked256(0x0, 0x0, 0x4ee2d6d415b, 0x85acef8100000000);
    case 33:
      // 1000000000000000000000000000000000
      return chunked256(0x0, 0x0, 0x314dc6448d93, 0x38c15b0a00000000);
    case 34:
      // 10000000000000000000000000000000000
      return chunked256(0x0, 0x0, 0x1ed09bead87c0, 0x378d8e6400000000);
    case 35:
      // 100000000000000000000000000000000000
      return chunked256(0x0, 0x0, 0x13426172c74d82, 0x2b878fe800000000);
    case 36:
      // 1000000000000000000000000000000000000
      return chunked256(0x0, 0x0, 0xc097ce7bc90715, 0xb34b9f1000000000);
    case 37:
      // 10000000000000000000000000000000000000
      return chunked256(0x0, 0x0, 0x785ee10d5da46d9, 0xf436a000000000);
    case 38:
      // 100000000000000000000000000000000000000
      return chunked256(0x0, 0x0, 0x4b3b4ca85a86c47a, 0x98a224000000000);
    case 39:
      // 1000000000000000000000000000000000000000
      return chunked256(0x0, 0x2, 0xf050fe938943acc4, 0x5f65568000000000);
    case 40:
      // 10000000000000000000000000000000000000000
      return chunked256(0x0, 0x1d, 0x6329f1c35ca4bfab, 0xb9f5610000000000);
    case 41:
      // 100000000000000000000000000000000000000000
      return chunked256(0x0, 0x125, 0xdfa371a19e6f7cb5, 0x4395ca0000000000);
    case 42:
      // 1000000000000000000000000000000000000000000
      return chunked256(0x0, 0xb7a, 0xbc627050305adf14, 0xa3d9e40000000000);
    case 43:
      // 10000000000000000000000000000000000000000000
      return chunked256(0x0, 0x72cb, 0x5bd86321e38cb6ce, 0x6682e80000000000);
    case 44:
      // 100000000000000000000000000000000000000000000
      return chunked256(0x0, 0x47bf1, 0x9673df52e37f2410, 0x11d100000000000);
    case 45:
      // 1000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x2cd76f, 0xe086b93ce2f768a0, 0xb22a00000000000);
    case 46:
      // 10000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x1c06a5e, 0xc5433c60ddaa1640, 0x6f5a400000000000);
    case 47:
      // 100000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x118427b3, 0xb4a05bc8a8a4de84, 0x5986800000000000);
    case 48:
      // 1000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0xaf298d05, 0xe4395d69670b12b, 0x7f41000000000000);
    case 49:
      // 10000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x6d79f8232, 0x8ea3da61e066ebb2, 0xf88a000000000000);
    case 50:
      // 100000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x446c3b15f9, 0x926687d2c40534fd, 0xb564000000000000);
    case 51:
      // 1000000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x2ac3a4edbbf, 0xb8014e3ba83411e9, 0x15e8000000000000);
    case 52:
      // 10000000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x1aba4714957d, 0x300d0e549208b31a, 0xdb10000000000000);
    case 53:
      // 100000000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x10b46c6cdd6e3, 0xe0828f4db456ff0c, 0x8ea0000000000000);
    case 54:
      // 1000000000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0xa70c3c40a64e6, 0xc51999090b65f67d, 0x9240000000000000);
    case 55:
      // 10000000000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x6867a5a867f103, 0xb2fffa5a71fba0e7, 0xb680000000000000);
    case 56:
      // 100000000000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x4140c78940f6a24, 0xfdffc78873d4490d, 0x2100000000000000);
    case 57:
      // 1000000000000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x28c87cb5c89a2571, 0xebfdcb54864ada83, 0x4a00000000000000);
    case 58:
      // 10000000000000000000000000000000000000000000000000000000000
      return chunked256(0x1, 0x97d4df19d6057673, 0x37e9f14d3eec8920, 0xe400000000000000);
    case 59:
      // 100000000000000000000000000000000000000000000000000000000000
      return chunked256(0xf, 0xee50b7025c36a080, 0x2f236d04753d5b48, 0xe800000000000000);
    case 60:
      // 1000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x9f, 0x4f2726179a224501, 0xd762422c946590d9, 0x1000000000000000);
    case 61:
      // 10000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x639, 0x17877cec0556b212, 0x69d695bdcbf7a87a, 0xa000000000000000);
    case 62:
      // 100000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x3e3a, 0xeb4ae1383562f4b8, 0x2261d969f7ac94ca, 0x4000000000000000);
    case 63:
      // 1000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x26e4d, 0x30eccc3215dd8f31, 0x57d27e23acbdcfe6, 0x8000000000000000);
    case 64:
      // 10000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x184f03, 0xe93ff9f4daa797ed, 0x6e38ed64bf6a1f01, 0x0);
    case 65:
      // 100000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0xf31627, 0x1c7fc3908a8bef46, 0x4e3945ef7a25360a, 0x0);
    case 66:
      // 1000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x97edd87, 0x1cfda3a5697758bf, 0xe3cbb5ac5741c64, 0x0);
    case 67:
      // 10000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x5ef4a747, 0x21e864761ea97776, 0x8e5f518bb6891be8, 0x0);
    case 68:
      // 100000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x3b58e88c7, 0x5313ec9d329eaaa1, 0x8fb92f75215b1710, 0x0);
    case 69:
      // 1000000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x25179157c9, 0x3ec73e23fa32aa4f, 0x9d3bda934d8ee6a0, 0x0);
    case 70:
      // 10000000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x172ebad6ddc, 0x73c86d67c5faa71c, 0x245689c107950240, 0x0);
    case 71:
      // 100000000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0xe7d34c64a9c, 0x85d4460dbbca8719, 0x6b61618a4bd21680, 0x0);
    case 72:
      // 1000000000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x90e40fbeea1d, 0x3a4abc8955e946fe, 0x31cdcf66f634e100, 0x0);
    case 73:
      // 10000000000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x5a8e89d752524, 0x46eb5d5d5b1cc5ed, 0xf20a1a059e10ca00, 0x0);
    case 74:
      // 100000000000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x3899162693736a, 0xc531a5a58f1fbb4b, 0x746504382ca7e400, 0x0);
    case 75:
      // 1000000000000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x235fadd81c2822b, 0xb3f07877973d50f2, 0x8bf22a31be8ee800, 0x0);
    case 76:
      // 10000000000000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x161bcca7119915b5, 0x764b4abe8652979, 0x7775a5f171951000, 0x0);
    default:
      // This is not a supported value...
      CUDF_UNREACHABLE("exponent exceeds supported value");
  }
}

// check that the divide is going to do the right thing
void check_scale_divisor(int source_scale, int target_scale)
{
  int exponent = target_scale - source_scale;
  CUDF_EXPECTS(exponent <= cuda::std::numeric_limits<__int128_t>::digits10, "divisor too big");
}

inline __device__ int precision10(chunked256 value)
{
  if (value.sign() < 0) {
    // we want to do this on positive numbers
    value.negate();
  }
  // TODO this is a horrible way to do this. We should at least
  // be able to approximate the log10 using the leading zeros similar to
  // http://graphics.stanford.edu/~seander/bithacks.html and then start
  // the search around the guess.
  for (int i = 0; i <= 76; i++) {
    chunked256 tmp = pow_ten(i);
    if (tmp.gte_unsigned(value)) { return i; }
  }
  return -1;
}

__device__ bool is_greater_than_decimal_38(chunked256 a)
{
  auto const max_number_for_precision = pow_ten(38);
  if (a.sign() != 0) { a.negate(); }
  return a.gte_unsigned(max_number_for_precision);
}

__device__ chunked256 set_scale_and_round(chunked256 data, int old_scale, int new_scale)
{
  if (old_scale != new_scale) {
    if (new_scale < old_scale) {
      int const raise       = old_scale - new_scale;
      auto const multiplier = pow_ten(raise);
      data                  = multiply(data, multiplier);
    } else {
      int const drop = new_scale - old_scale;
      // `drop` should not exceed 38 thus `10^drop` should fit in __int128_t.
      auto const divisor = pow_ten(drop).as_128_bits();
      data               = divide_and_round(data, divisor);
    }
  }
  return data;
}

// Functor to add two DECIMAL128 columns with rounding and overflow detection.
struct dec128_add_sub {
  dec128_add_sub(bool* overflows,
                 cudf::mutable_column_view const& result_view,
                 cudf::column_view const& a_col,
                 cudf::column_view const& b_col)
    : overflows(overflows),
      a_data(a_col.data<__int128_t>()),
      b_data(b_col.data<__int128_t>()),
      result_data(result_view.data<__int128_t>()),
      a_scale(a_col.type().scale()),
      b_scale(b_col.type().scale()),
      result_scale(result_view.type().scale())
  {
  }

  __device__ void add(chunked256& a, chunked256& b) const { do_add_sub(a, b, false); }

  __device__ void sub(chunked256& a, chunked256& b) const { do_add_sub(a, b, true); }

 private:
  __device__ void do_add_sub(chunked256& a, chunked256& b, bool sub) const
  {
    int intermediate_scale = min(a_scale, b_scale);
    if (a_scale != intermediate_scale) { a = set_scale_and_round(a, a_scale, intermediate_scale); }
    if (b_scale != intermediate_scale) { b = set_scale_and_round(b, b_scale, intermediate_scale); }
    if (sub) {
      // Get 2's complement
      b.negate();
    }
    a.add(b);

    if (result_scale != intermediate_scale) {
      a = set_scale_and_round(a, intermediate_scale, result_scale);
    }
  }

 protected:
  // output column for overflow detected
  bool* const overflows;

  // input data
  __int128_t const* const a_data;
  __int128_t const* const b_data;
  __int128_t* const result_data;
  int const a_scale;
  int const b_scale;
  int const result_scale;
};

// Functor to add two DECIMAL128 columns with rounding and overflow detection.
struct dec128_add : public dec128_add_sub {
  dec128_add(bool* overflows,
             cudf::mutable_column_view const& sum_view,
             cudf::column_view const& a_col,
             cudf::column_view const& b_col)
    : dec128_add_sub(overflows, sum_view, a_col, b_col)
  {
  }

  __device__ void operator()(cudf::size_type const i) const
  {
    chunked256 a(a_data[i]);
    chunked256 b(b_data[i]);

    chunked256& sum = a;
    add(a, b);

    overflows[i]   = is_greater_than_decimal_38(sum);
    result_data[i] = sum.as_128_bits();
  }
};

// Functor to sub two DECIMAL128 columns with rounding and overflow detection.
struct dec128_sub : public dec128_add_sub {
  dec128_sub(bool* overflows,
             cudf::mutable_column_view const& sub_view,
             cudf::column_view const& a_col,
             cudf::column_view const& b_col)
    : dec128_add_sub(overflows, sub_view, a_col, b_col)
  {
  }

  __device__ void operator()(cudf::size_type const i) const
  {
    chunked256 a(a_data[i]);
    chunked256 b(b_data[i]);

    chunked256& res = a;
    sub(a, b);

    overflows[i]   = is_greater_than_decimal_38(res);
    result_data[i] = res.as_128_bits();
  }
};

// Functor to multiply two DECIMAL128 columns with rounding and overflow detection.
struct dec128_multiplier {
  dec128_multiplier(bool* overflows,
                    cudf::mutable_column_view const& product_view,
                    cudf::column_view const& a_col,
                    cudf::column_view const& b_col,
                    bool const cast_interim_result)
    : overflows(overflows),
      a_data(a_col.data<__int128_t>()),
      b_data(b_col.data<__int128_t>()),
      product_data(product_view.data<__int128_t>()),
      a_scale(a_col.type().scale()),
      b_scale(b_col.type().scale()),
      prod_scale(product_view.type().scale()),
      cast_interim_result(cast_interim_result)
  {
  }

  __device__ void operator()(cudf::size_type const i) const
  {
    chunked256 const a(a_data[i]);
    chunked256 const b(b_data[i]);

    chunked256 product = multiply(a, b);

    int const mult_scale = [&]() {
      // According to https://issues.apache.org/jira/browse/SPARK-40129
      // and https://issues.apache.org/jira/browse/SPARK-45786, Spark has a bug in
      // versions 3.2.4, 3.3.3, 3.4.1, 3.5.0 and 4.0.0 The bug is fixed for later versions but to
      // match the legacy behavior we need to first round the result to a precision of 38 then we
      // need to round the result to the final scale that we care about.
      if (cast_interim_result) {
        auto const first_div_precision = precision10(product) - 38;
        if (first_div_precision > 0) {
          auto const first_div_scale_divisor = pow_ten(first_div_precision).as_128_bits();
          product                            = divide_and_round(product, first_div_scale_divisor);

          // a_scale and b_scale are negative. first_div_precision is not
          return a_scale + b_scale + first_div_precision;
        }
      }
      return a_scale + b_scale;
    }();

    int exponent = prod_scale - mult_scale;
    if (exponent < 0) {
      // we need to multiply, but only if this will not overflow.
      int new_precision = precision10(product);
      if (new_precision - exponent > 38) {
        // this would overflow...
        overflows[i] = true;
        return;
      }
      auto const scale_mult = pow_ten(-exponent);
      product               = multiply(product, scale_mult);
    } else {
      // `exponent` should not exceed 38 thus `10^exponent` should fit in __int128_t.
      auto const scale_divisor = pow_ten(exponent).as_128_bits();

      // scale and round to target scale
      if (scale_divisor != 1) { product = divide_and_round(product, scale_divisor); }
    }

    overflows[i]    = is_greater_than_decimal_38(product);
    product_data[i] = product.as_128_bits();
  }

 private:
  // output column for overflow detected
  bool* const overflows;
  bool const cast_interim_result;

  // input data for multiply
  __int128_t const* const a_data;
  __int128_t const* const b_data;
  __int128_t* const product_data;
  int const a_scale;
  int const b_scale;
  int const prod_scale;
};

/**
 * Functor to divide two DECIMAL128 columns with rounding and overflow detection.
 * This functor should be used for a 128-bit regular division or a 64-bit integer division only
 * i.e. dec128_divider<__int128_t, false> and dec128_divider<uint64_t, true>. Any other combination
 * will result in data truncation
 */
template <typename T, bool is_int_div>
struct dec128_divider {
  static_assert((sizeof(T) == sizeof(uint64_t) && is_int_div) ||
                (sizeof(T) == sizeof(__int128_t) && !is_int_div));
  dec128_divider(bool* overflows,
                 cudf::mutable_column_view const& quotient_view,
                 cudf::column_view const& a_col,
                 cudf::column_view const& b_col)
    : overflows(overflows),
      a_data(a_col.data<__int128_t>()),
      b_data(b_col.data<__int128_t>()),
      quotient_data(quotient_view.data<T>()),
      a_scale(a_col.type().scale()),
      b_scale(b_col.type().scale()),
      quot_scale(quotient_view.type().scale())
  {
  }

  __device__ void operator()(cudf::size_type const i) const
  {
    chunked256 n(a_data[i]);
    __int128_t const d(b_data[i]);

    // Divide by zero, not sure if we care or not, but...
    if (d == 0) {
      overflows[i]     = true;
      quotient_data[i] = 0;
      return;
    }

    // The output scale of a divide is a_scale - b_scale. But we want an output scale of
    // quot_scale, so we need to multiply a by a set amount before we can do the divide.

    int n_shift_exp = quot_scale - (a_scale - b_scale);

    if (n_shift_exp > 0) {
      // In this case we have to divide twice to get the answer we want.
      // The first divide is a regular divide
      divmod256 const first_div_result = divide(n, d);

      // Ignore the remainder because we don't need it.
      auto const scale_divisor = pow_ten(n_shift_exp).as_128_bits();

      // The second divide gets the result into the scale that we care about and does the rounding.
      chunked256 result;
      if constexpr (is_int_div) {
        result           = integer_divide(first_div_result.quotient, scale_divisor);
        quotient_data[i] = result.as_64_bits();
      } else {
        result           = divide_and_round(first_div_result.quotient, scale_divisor);
        quotient_data[i] = result.as_128_bits();
      }
      overflows[i] = is_greater_than_decimal_38(result);
    } else if (n_shift_exp < -38) {
      // We need to do a multiply before we can divide, but the multiply might
      // overflow so we do a multiply then a divide and shift the result and
      // remainder over by the amount left to multiply. It is kind of like long
      // division, but base 10.

      // First multiply by 10^38 and divide to get a remainder
      n = multiply(n, chunked256(pow_ten(38)));

      auto const first_div_result = divide(n, d);
      chunked256 const first_div_r(first_div_result.remainder);

      // now we have to multiply each of these by how much is left
      int const remaining_exp = (-n_shift_exp) - 38;
      auto const scale_mult   = pow_ten(remaining_exp);
      auto result             = multiply(first_div_result.quotient, scale_mult);
      auto const scaled_div_r = multiply(first_div_r, scale_mult);

      // Now do a second divide on what is left
      auto const second_div_result = divide(scaled_div_r, d);
      result.add(second_div_result.quotient);

      if constexpr (is_int_div) {
        overflows[i]     = is_greater_than_decimal_38(result);
        quotient_data[i] = result.as_64_bits();
      } else {
        // and finally round
        result = round_from_remainder(result, second_div_result.remainder, scaled_div_r, d);
        quotient_data[i] = result.as_128_bits();
      }
      overflows[i] = is_greater_than_decimal_38(result);
    } else {
      // Regular multiply followed by a divide
      if (n_shift_exp < 0) { n = multiply(n, pow_ten(-n_shift_exp)); }
      chunked256 result;
      if constexpr (is_int_div) {
        result           = integer_divide(n, d);
        quotient_data[i] = result.as_64_bits();
      } else {
        result           = divide_and_round(n, d);
        quotient_data[i] = result.as_128_bits();
      }
      overflows[i] = is_greater_than_decimal_38(result);
    }
  }

 private:
  // output column for overflow detected
  bool* const overflows;
  // input data for multiply
  __int128_t const* const a_data;
  __int128_t const* const b_data;
  T* const quotient_data;
  int const a_scale;
  int const b_scale;
  int const quot_scale;
};

struct dec128_remainder {
  dec128_remainder(bool* overflows,
                   cudf::mutable_column_view const& remainder_view,
                   cudf::column_view const& a_col,
                   cudf::column_view const& b_col)
    : overflows(overflows),
      a_data(a_col.data<__int128_t>()),
      b_data(b_col.data<__int128_t>()),
      remainder_data(remainder_view.data<__int128_t>()),
      a_scale(a_col.type().scale()),
      b_scale(b_col.type().scale()),
      rem_scale(remainder_view.type().scale())
  {
  }

  __device__ void operator()(cudf::size_type const i) const
  {
    chunked256 n(a_data[i]);
    __int128_t const d(b_data[i]);

    // Divide by zero, not sure if we care or not, but...
    if (d == 0) {
      overflows[i]      = true;
      remainder_data[i] = 0;
      return;
    }

    // This implementation of remainder uses the JAVA definition of remainder
    // that Spark relies on. It's *not* the most efficient way of calculating
    // remainder, but we use this to be consistent with CPU Spark.

    // The algorithm is:
    // a % b = a - (a // b) * b
    // Basically we substract the integral_divide result times the divisor from
    // the dividend

    bool const is_n_neg = n.sign() < 0;
    bool const is_d_neg = d < 0;

    __int128_t result;
    // The output scale of remainder is technically the scale of the divisor (b_scale)
    // But since we want an output scale of rem_scale, we have to do the following:
    // First, we have to shift the divisor to the desired rem_scale
    const int d_shift_exp = rem_scale - b_scale;
    // Then, we have to shift the dividend to compute integer divide
    // We use the formula from dec128_divider
    // Start with: quot_scale - (a_scale - b_scale)
    // Then substitute 0 for quot_scale (integer divide), and rem_scale for b_scale
    // (since we updated the divisor scale)
    // 0 - (a_scale - rem_scale)
    // rem_scale - a_scale
    int n_shift_exp  = rem_scale - a_scale;
    __int128_t abs_d = is_d_neg ? -d : d;
    // Unlike in divide, where we can scale the dividend to get the right result
    // remainder relies on the scale on the divisor, so we might have to shift the
    // divisor itself.
    if (d_shift_exp > 0) {
      // We need to shift the the scale of the divisor to rem_scale, but
      // we actual need to round because of how precision is to be handled,
      // since the new scale is smaller than the old scale
      auto const scale_divisor = pow_ten(d_shift_exp).as_128_bits();
      abs_d                    = divide_and_round(chunked256(abs_d), scale_divisor).as_128_bits();
    } else {
      // otherwise we are multiplying the bottom by a power of 10, which divides the numerator
      // by the same power of ten, so we accomodate that in our original n-shift like
      // divide did before
      n_shift_exp -= d_shift_exp;
    }
    // For remainder, we should do the computation using positive numbers only, and then
    // switch the sign based on [n] *only*.
    chunked256 abs_n = n;
    if (is_n_neg) { abs_n.negate(); }
    chunked256 int_div_result;
    if (n_shift_exp > 0) {
      divmod256 const first_div_result = divide(abs_n, abs_d);

      // Ignore the remainder because we don't need it.
      auto const scale_divisor = pow_ten(n_shift_exp).as_128_bits();

      // The second divide gets the result into the scale that we care about and does the rounding.
      int_div_result = integer_divide(first_div_result.quotient, scale_divisor);
    } else {
      if (n_shift_exp < 0) { abs_n = multiply(abs_n, pow_ten(-n_shift_exp)); }
      int_div_result = integer_divide(abs_n, abs_d);
    }
    // Multiply the integer divide result by abs(divisor)
    chunked256 less_n = multiply(int_div_result, chunked256(abs_d));

    if (d_shift_exp < 0) {
      // scale less_n up to equal it to same scale since we were technically scaling up
      // the divisor earlier (even though we only shifted n)
      less_n = multiply(less_n, pow_ten(-d_shift_exp));
    }
    // Subtract our integer divide result from n by adding the negated
    less_n.negate();
    abs_n.add(less_n);
    // This should almost never overflow, but we check anyways
    overflows[i] = is_greater_than_decimal_38(abs_n);
    result       = abs_n.as_128_bits();
    // Change the sign of the result based on n
    if (is_n_neg) { result = -result; }
    remainder_data[i] = result;
  }

 private:
  // output column for overflow detected
  bool* const overflows;
  // input data for multiply
  __int128_t const* const a_data;
  __int128_t const* const b_data;
  __int128_t* const remainder_data;
  int const a_scale;
  int const b_scale;
  int const rem_scale;
};

}  // anonymous namespace

namespace cudf::jni {

std::unique_ptr<cudf::table> multiply_decimal128(cudf::column_view const& a,
                                                 cudf::column_view const& b,
                                                 int32_t product_scale,
                                                 bool const cast_interim_result,
                                                 rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(a.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  CUDF_EXPECTS(b.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  auto const num_rows = a.size();
  CUDF_EXPECTS(num_rows == b.size(), "inputs have mismatched row counts");
  auto [result_null_mask, result_null_count] =
    cudf::bitmask_and(cudf::table_view{{a, b}}, stream, rmm::mr::get_current_device_resource_ref());
  std::vector<std::unique_ptr<cudf::column>> columns;
  // copy the null mask here, as it will be used again later
  columns.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                                  num_rows,
                                                  rmm::device_buffer(result_null_mask, stream),
                                                  result_null_count,
                                                  stream));
  columns.push_back(
    cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::DECIMAL128, product_scale},
                                  num_rows,
                                  std::move(result_null_mask),
                                  result_null_count,
                                  stream));
  auto overflows_view = columns[0]->mutable_view();
  auto product_view   = columns[1]->mutable_view();
  check_scale_divisor(a.type().scale() + b.type().scale(), product_scale);
  thrust::for_each(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(num_rows),
    dec128_multiplier(overflows_view.begin<bool>(), product_view, a, b, cast_interim_result));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> divide_decimal128(cudf::column_view const& a,
                                               cudf::column_view const& b,
                                               int32_t quotient_scale,
                                               rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(a.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  CUDF_EXPECTS(b.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  auto const num_rows = a.size();
  CUDF_EXPECTS(num_rows == b.size(), "inputs have mismatched row counts");
  auto [result_null_mask, result_null_count] =
    cudf::bitmask_and(cudf::table_view{{a, b}}, stream, rmm::mr::get_current_device_resource_ref());
  std::vector<std::unique_ptr<cudf::column>> columns;
  // copy the null mask here, as it will be used again later
  columns.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                                  num_rows,
                                                  rmm::device_buffer(result_null_mask, stream),
                                                  result_null_count,
                                                  stream));
  columns.push_back(
    cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::DECIMAL128, quotient_scale},
                                  num_rows,
                                  std::move(result_null_mask),
                                  result_null_count,
                                  stream));
  auto overflows_view = columns[0]->mutable_view();
  auto quotient_view  = columns[1]->mutable_view();
  thrust::for_each(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(num_rows),
    dec128_divider<__int128_t, false>(overflows_view.begin<bool>(), quotient_view, a, b));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> integer_divide_decimal128(cudf::column_view const& a,
                                                       cudf::column_view const& b,
                                                       int32_t quotient_scale,
                                                       rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(a.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  CUDF_EXPECTS(b.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  auto const num_rows = a.size();
  CUDF_EXPECTS(num_rows == b.size(), "inputs have mismatched row counts");
  auto [result_null_mask, result_null_count] =
    cudf::bitmask_and(cudf::table_view{{a, b}}, stream, rmm::mr::get_current_device_resource_ref());
  std::vector<std::unique_ptr<cudf::column>> columns;
  // copy the null mask here, as it will be used again later
  columns.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                                  num_rows,
                                                  rmm::device_buffer(result_null_mask, stream),
                                                  result_null_count,
                                                  stream));
  columns.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT64},
                                                  num_rows,
                                                  std::move(result_null_mask),
                                                  result_null_count,
                                                  stream));
  auto overflows_view = columns[0]->mutable_view();
  auto quotient_view  = columns[1]->mutable_view();
  thrust::for_each(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(num_rows),
    dec128_divider<uint64_t, true>(overflows_view.begin<bool>(), quotient_view, a, b));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> remainder_decimal128(cudf::column_view const& a,
                                                  cudf::column_view const& b,
                                                  int32_t remainder_scale,
                                                  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(a.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  CUDF_EXPECTS(b.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  auto const num_rows = a.size();
  CUDF_EXPECTS(num_rows == b.size(), "inputs have mismatched row counts");
  auto [result_null_mask, result_null_count] =
    cudf::bitmask_and(cudf::table_view{{a, b}}, stream, rmm::mr::get_current_device_resource_ref());
  std::vector<std::unique_ptr<cudf::column>> columns;
  // copy the null mask here, as it will be used again later
  columns.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                                  num_rows,
                                                  rmm::device_buffer(result_null_mask, stream),
                                                  result_null_count,
                                                  stream));
  columns.push_back(
    cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::DECIMAL128, remainder_scale},
                                  num_rows,
                                  std::move(result_null_mask),
                                  result_null_count,
                                  stream));
  auto overflows_view = columns[0]->mutable_view();
  auto remainder_view = columns[1]->mutable_view();
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator<cudf::size_type>(0),
                   thrust::make_counting_iterator<cudf::size_type>(num_rows),
                   dec128_remainder(overflows_view.begin<bool>(), remainder_view, a, b));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> add_decimal128(cudf::column_view const& a,
                                            cudf::column_view const& b,
                                            int32_t target_scale,
                                            rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(a.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  CUDF_EXPECTS(b.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  auto const num_rows = a.size();
  CUDF_EXPECTS(num_rows == b.size(), "inputs have mismatched row counts");
  auto [result_null_mask, result_null_count] =
    cudf::bitmask_and(cudf::table_view{{a, b}}, stream, rmm::mr::get_current_device_resource_ref());
  std::vector<std::unique_ptr<cudf::column>> columns;
  // copy the null mask here, as it will be used again later
  columns.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                                  num_rows,
                                                  rmm::device_buffer(result_null_mask, stream),
                                                  result_null_count,
                                                  stream));
  columns.push_back(
    cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::DECIMAL128, target_scale},
                                  num_rows,
                                  std::move(result_null_mask),
                                  result_null_count,
                                  stream));
  auto overflows_view = columns[0]->mutable_view();
  auto sum_view       = columns[1]->mutable_view();
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(num_rows),
                   dec128_add(overflows_view.begin<bool>(), sum_view, a, b));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> sub_decimal128(cudf::column_view const& a,
                                            cudf::column_view const& b,
                                            int32_t target_scale,
                                            rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(a.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  CUDF_EXPECTS(b.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  auto const num_rows = a.size();
  CUDF_EXPECTS(num_rows == b.size(), "inputs have mismatched row counts");
  auto [result_null_mask, result_null_count] =
    cudf::bitmask_and(cudf::table_view{{a, b}}, stream, rmm::mr::get_current_device_resource_ref());
  std::vector<std::unique_ptr<cudf::column>> columns;
  // copy the null mask here, as it will be used again later
  columns.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                                  num_rows,
                                                  rmm::device_buffer(result_null_mask, stream),
                                                  result_null_count,
                                                  stream));
  columns.push_back(
    cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::DECIMAL128, target_scale},
                                  num_rows,
                                  std::move(result_null_mask),
                                  result_null_count,
                                  stream));
  auto overflows_view = columns[0]->mutable_view();
  auto sub_view       = columns[1]->mutable_view();
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(num_rows),
                   dec128_sub(overflows_view.begin<bool>(), sub_view, a, b));
  return std::make_unique<cudf::table>(std::move(columns));
}

namespace {

using namespace numeric;
using namespace numeric::detail;

/**
 * @brief Perform floating-point to integer decimal conversion, matching Spark behavior.
 *
 * The desired decimal value is computed as (returned_value * 10^{-pow10}).
 *
 * The rounding and precision decisions made here are chosen to match Apache Spark.
 * Spark wants to perform the conversion as double to have the most precision.
 * However, the behavior is still slightly different if the original type was float.
 *
 * @tparam FloatType The type of floating-point value we are converting from
 * @tparam IntType The type of integer we are converting to, to store the decimal value
 *
 * @param input The floating point value to convert
 * @param pow10 The power of 10 to scale the floating-point value by
 * @return Integer representation of the floating-point value, rounding after scaled
 */
template <typename FloatType,
          typename IntType,
          CUDF_ENABLE_IF(cuda::std::is_floating_point_v<FloatType>)>
__device__ inline IntType scaled_round(FloatType input, int32_t pow10)
{
  // Extract components of the (double-ized) floating point number
  using converter        = floating_converter<double>;
  auto const integer_rep = converter::bit_cast_to_integer(static_cast<double>(input));
  if (converter::is_zero(integer_rep)) { return 0; }

  // Note that the significand here is an unsigned integer with sizeof(double)
  auto const is_negative                  = converter::get_is_negative(integer_rep);
  auto const [significand, floating_pow2] = converter::get_significand_and_pow2(integer_rep);

  auto const unsigned_floating      = (input < 0) ? -input : input;
  auto const rounding_wont_overflow = [&] {
    auto const scale_factor = static_cast<double>(
      multiply_power10<IntType>(cuda::std::make_unsigned_t<IntType>{1}, -pow10));
    return 10.0 * static_cast<double>(unsigned_floating) * scale_factor <
           static_cast<double>(cuda::std::numeric_limits<IntType>::max());
  }();

  // Spark often wants to round the last decimal place, so we'll perform the conversion
  // with one lower power of 10 so that we can (optionally) round at the end.
  // Note that we can't round this way if we've requested the minimum power.
  bool const can_round = cuda::std::is_same_v<IntType, __int128_t> ? rounding_wont_overflow : true;
  auto const shifting_pow10 = can_round ? pow10 - 1 : pow10;

  // Sometimes add half a bit to correct for compiler rounding to nearest floating-point value.
  // See comments in add_half_if_truncates(), with differences detailed below.
  // Even if we don't add the bit, shift bits to line up with what the shifting algorithm is
  // expecting.
  bool const is_whole_number     = cuda::std::floor(input) == input;
  auto const [base2_value, pow2] = [is_whole_number](auto significand, auto floating_pow2) {
    if constexpr (cuda::std::is_same_v<FloatType, double>) {
      // Add the 1/2 bit regardless of truncation, but still not for whole numbers.
      auto const base2_value =
        (significand << 1) + static_cast<decltype(significand)>(!is_whole_number);
      return cuda::std::make_pair(base2_value, floating_pow2 - 1);
    } else {
      // Input was float: never add 1/2 bit.
      // Why? Because we converted to double, and the 1/2 bit beyond float is WAY too large compared
      // to double's precision. And the 1/2 bit beyond double is not due to user input.
      return cuda::std::make_pair(significand << 1, floating_pow2 - 1);
    }
  }(significand, floating_pow2);

  // Main algorithm: Apply the powers of 2 and 10 (except for the last power-of-10).
  // Use larger intermediate type for conversion to avoid overflow for last power-of-10.
  using intermediate_type =
    cuda::std::conditional_t<cuda::std::is_same_v<IntType, std::int32_t>, std::int64_t, __int128_t>;
  cuda::std::make_unsigned_t<intermediate_type> magnitude =
    [&, base2_value = base2_value, pow2 = pow2] {
      if constexpr (cuda::std::is_same_v<IntType, std::int32_t>) {
        return rounding_wont_overflow ? convert_floating_to_integral_shifting<IntType, double>(
                                          base2_value, shifting_pow10, pow2)
                                      : convert_floating_to_integral_shifting<std::int64_t, double>(
                                          base2_value, shifting_pow10, pow2);
      } else {
        return convert_floating_to_integral_shifting<__int128_t, double>(
          base2_value, shifting_pow10, pow2);
      }
    }();

  // Spark wants to floor the last digits of the output, clearing data that was beyond the
  // precision that was available in double.

  // How many digits do we need to floor?
  // From the decimal digit corresponding to pow2 (just past double precision) to the end (pow10).
  int const floor_pow10 = [&](int pow2_bit) {
    // The conversion from pow2 to pow10 is log10(2), which is ~ 90/299 (close enough for ints)
    // But Spark chooses the rougher 3/10 ratio instead of 90/299.
    if constexpr (cuda::std::is_same_v<FloatType, float>) {
      return (3 * pow2_bit - 10 * pow10) / 10;
    } else {
      // Spark rounds up the power-of-10 to floor for DOUBLES >= 2^63 (and yes, this is the exact
      // cutoff).
      bool const round_up = unsigned_floating > cuda::std::numeric_limits<std::int64_t>::max();
      return (3 * pow2_bit - 10 * pow10 + 9 * round_up) / 10;
    }
  }(pow2);

  // Floor end digits
  if (can_round) {
    if (floor_pow10 < 0) {
      // Truncated: The scale factor cut off the extra, imprecise bits.
      // To round to the final decimal place, add 5 to one past the last decimal place.
      magnitude += 5U;
      magnitude /= 10U;  // Apply the last power of 10
    } else {
      // We are keeping decimal digits with data beyond the precision of double.
      // We want to truncate these digits, but sometimes we want to round first.
      // We will round if and only if we didn't already add a half-bit earlier.
      if constexpr (cuda::std::is_same_v<FloatType, double>) {
        // For doubles, only round the extra digits of whole numbers.
        // If it was not a whole number, we already added 1/2 a bit at higher precision than this
        // earlier.
        if (is_whole_number) {
          magnitude += multiply_power10<IntType>(decltype(magnitude)(5), floor_pow10);
        }
      } else {
        // Input was float: we didn't add a half-bit earlier, so round at the edge of precision
        // here.
        magnitude += multiply_power10<IntType>(decltype(magnitude)(5), floor_pow10);
      }

      // +1: Divide the last power-of-10 that we postponed earlier to do rounding.
      auto const truncated = divide_power10<IntType>(magnitude, floor_pow10 + 1);
      magnitude            = multiply_power10<IntType>(truncated, floor_pow10);
    }
  } else if (floor_pow10 > 0) {
    auto const truncated = divide_power10<IntType>(magnitude, floor_pow10);
    magnitude            = multiply_power10<IntType>(truncated, floor_pow10);
  }

  // Reapply the sign and return.
  // NOTE: Cast can overflow!
  auto const signed_magnitude = static_cast<IntType>(magnitude);
  return is_negative ? -signed_magnitude : signed_magnitude;
}

template <typename FloatType, typename DecimalRepType>
struct floating_point_to_decimal_fn {
  cudf::column_device_view input;
  bool* validity;
  cudf::size_type* failure_row_id;
  int32_t decimal_places;
  DecimalRepType exclusive_bound;

  __device__ DecimalRepType operator()(cudf::size_type idx) const
  {
    auto const x = input.element<FloatType>(idx);

    if (input.is_null(idx) || !cuda::std::isfinite(x)) {
      validity[idx] = false;
      return DecimalRepType{0};
    }

    auto const scaled_rounded = scaled_round<FloatType, DecimalRepType>(x, -decimal_places);
    auto const is_out_of_bound =
      -exclusive_bound >= scaled_rounded || scaled_rounded >= exclusive_bound;
    if (is_out_of_bound) { *failure_row_id = idx; }
    validity[idx] = !is_out_of_bound;

    return is_out_of_bound ? DecimalRepType{0} : scaled_rounded;
  }
};

struct floating_point_to_decimal_dispatcher {
  template <typename FloatType, typename DecimalType>
  static constexpr bool supported_types()
  {
    return (std::is_same_v<FloatType, float> ||    //
            std::is_same_v<FloatType, double>) &&  //
           (std::is_same_v<DecimalType, numeric::decimal32> ||
            std::is_same_v<DecimalType, numeric::decimal64> ||
            std::is_same_v<DecimalType, numeric::decimal128>);
  }

  template <typename FloatType,
            typename DecimalType,
            typename... Args,
            CUDF_ENABLE_IF(not supported_types<FloatType, DecimalType>())>
  void operator()(Args...) const
  {
    CUDF_FAIL("Unsupported types for floating_point_to_decimal_fn", cudf::data_type_error);
  }

  template <typename FloatType,
            typename DecimalType,
            CUDF_ENABLE_IF(supported_types<FloatType, DecimalType>())>
  void operator()(cudf::column_view const& input,
                  cudf::mutable_column_view const& output,
                  bool* validity,
                  cudf::size_type* failure_row_id,
                  int32_t decimal_places,
                  int32_t precision,
                  rmm::cuda_stream_view stream) const
  {
    using DecimalRepType = cudf::device_storage_type_t<DecimalType>;

    auto const d_input_ptr     = cudf::column_device_view::create(input, stream);
    auto const exclusive_bound = static_cast<DecimalRepType>(
      multiply_power10<DecimalRepType>(cuda::std::make_unsigned_t<DecimalRepType>{1}, precision));

    thrust::tabulate(rmm::exec_policy_nosync(stream),
                     output.begin<DecimalRepType>(),
                     output.end<DecimalRepType>(),
                     floating_point_to_decimal_fn<FloatType, DecimalRepType>{
                       *d_input_ptr, validity, failure_row_id, decimal_places, exclusive_bound});
  }
};

}  // namespace

std::pair<std::unique_ptr<cudf::column>, cudf::size_type> floating_point_to_decimal(
  cudf::column_view const& input,
  cudf::data_type output_type,
  int32_t precision,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto output = cudf::make_fixed_point_column(
    output_type, input.size(), cudf::mask_state::UNALLOCATED, stream, mr);

  auto const decimal_places = -output_type.scale();
  auto const default_mr     = rmm::mr::get_current_device_resource_ref();

  rmm::device_uvector<bool> validity(input.size(), stream, default_mr);
  rmm::device_scalar<cudf::size_type> failure_row_id(-1, stream, default_mr);

  cudf::double_type_dispatcher(input.type(),
                               output_type,
                               floating_point_to_decimal_dispatcher{},
                               input,
                               output->mutable_view(),
                               validity.begin(),
                               failure_row_id.data(),
                               decimal_places,
                               precision,
                               stream);

  auto [null_mask, null_count] =
    cudf::bools_to_mask(cudf::device_span<bool const>(validity), stream, mr);
  if (null_count > 0) { output->set_null_mask(std::move(*null_mask.release()), null_count); }

  return {std::move(output), failure_row_id.value(stream)};
}

}  // namespace cudf::jni
