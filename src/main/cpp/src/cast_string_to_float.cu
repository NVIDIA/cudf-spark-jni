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

#include "cast_string.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/detail/convert/string_to_float.cuh>
#include <cudf/utilities/bit.hpp>

#include <rmm/resource_ref.hpp>

#include <cub/warp/warp_reduce.cuh>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

using namespace cudf;

namespace spark_rapids_jni {

namespace detail {

__device__ __inline__ bool is_digit(char c) { return c >= '0' && c <= '9'; }

// (1) Scale `digits * 10^q` exactly into a 128-bit value.
//
// Returns `quotient128 * 2^binary_scale` exactly equal to `digits * 10^q`, plus
// the q<0 truncation remainder that the rounding step will fold into the
// sticky bit. Three sub-cases:
//   q < 0:  quotient128 = (digits << 64) / 10^|q|, binary_scale = -64.
//   q > 0:  quotient128 = digits * 10^q (exact in 128 bits), binary_scale = 0.
//   q == 0: quotient128 = digits, binary_scale = 0.
//
// Caller precondition: `digits > 2^53 AND |q| <= 19`. Under those bounds 10^|q|
// fits in uint64_t and `digits * 10^q` fits in 128 bits.
struct scaled_uint128 {
  __uint128_t quotient128;
  uint64_t division_rem;
  int binary_scale;
};

__device__ __inline__ scaled_uint128 scale_digits_times_pow10(uint64_t digits, int q)
{
  assert(digits > (1ULL << 53));
  int const abs_q = cuda::std::abs(q);
  assert(abs_q <= 19);

  // 10^abs_q for abs_q in [0, 19], exact as uint64_t.
  static constexpr uint64_t k_pow10[20] = {1ULL,
                                           10ULL,
                                           100ULL,
                                           1'000ULL,
                                           10'000ULL,
                                           100'000ULL,
                                           1'000'000ULL,
                                           10'000'000ULL,
                                           100'000'000ULL,
                                           1'000'000'000ULL,
                                           10'000'000'000ULL,
                                           100'000'000'000ULL,
                                           1'000'000'000'000ULL,
                                           10'000'000'000'000ULL,
                                           100'000'000'000'000ULL,
                                           1'000'000'000'000'000ULL,
                                           10'000'000'000'000'000ULL,
                                           100'000'000'000'000'000ULL,
                                           1'000'000'000'000'000'000ULL,
                                           10'000'000'000'000'000'000ULL};
  uint64_t const pow10_abs_q            = k_pow10[abs_q];

  if (q < 0) {
    __uint128_t const numerator = static_cast<__uint128_t>(digits) << 64;
    __uint128_t const quotient  = numerator / pow10_abs_q;
    // Equivalent to `numerator % pow10_abs_q` but saves one 128-bit division
    // by reusing `quotient`.
    uint64_t const rem =
      static_cast<uint64_t>(numerator - quotient * static_cast<__uint128_t>(pow10_abs_q));
    return {quotient, rem, -64};
  }
  if (q > 0) {
    __uint128_t const quotient =
      static_cast<__uint128_t>(digits) * static_cast<__uint128_t>(pow10_abs_q);
    return {quotient, 0, 0};
  }
  return {static_cast<__uint128_t>(digits), 0, 0};
}

// (2) Locate the MSB of a non-zero 128-bit value.
//
// `__clzll(x)` is the CUDA count-leading-zeros intrinsic on a 64-bit unsigned
// integer, so `63 - __clzll(x)` is the index of the most-significant set bit.
// For 128-bit inputs we split into hi/lo limbs and only inspect the upper when
// it is non-zero; otherwise the MSB lives in the lower 64 bits.
//
// Caller precondition guarantees `value != 0` (see callers of
// `scale_digits_times_pow10`: every branch produces a value at least `digits`,
// which is > 2^53).
__device__ __inline__ int locate_msb_uint128(__uint128_t value)
{
  uint64_t const hi = static_cast<uint64_t>(value >> 64);
  if (hi != 0) { return 127 - __clzll(hi); }
  uint64_t const lo = static_cast<uint64_t>(value);
  assert(lo != 0);
  return 63 - __clzll(lo);
}

// (3) Round the top 53 bits of a 128-bit integer to nearest-ties-to-even.
//
// `msb_pos` is the bit index of the leading 1 (so the 53-bit mantissa is at
// bits [msb_pos-52 .. msb_pos]). The remaining low bits feed the round and
// sticky decisions; the q<0 truncation remainder threads in as an additional
// sticky input. On overflow back into bit 53 the function renormalizes by
// shifting right and incrementing the unbiased exponent.
struct rounded_mantissa {
  uint64_t mantissa;
  int unbiased_exp;
};

__device__ __inline__ rounded_mantissa round_top_53_bits(__uint128_t quotient128,
                                                         uint64_t division_rem,
                                                         int msb_pos,
                                                         int binary_scale)
{
  int const shift = msb_pos - 52;

  // Caller precondition: digits > 2^53 implies msb_pos(quotient128) > 52 in
  // every (q < 0, q == 0, q > 0) scaling branch, so `shift > 0` always holds
  // and the `shift <= 0` left-shift branch is unreachable.
  assert(shift > 0);

  // Shift in [1, 127]. All bit ops use __uint128_t to avoid undefined 64-bit
  // shifts at the shift == 64 boundary.
  uint64_t const mantissa    = static_cast<uint64_t>(quotient128 >> shift);
  bool const round_bit       = (static_cast<uint64_t>(quotient128 >> (shift - 1)) & 1ULL) != 0;
  __uint128_t const low_mask = (static_cast<__uint128_t>(1) << (shift - 1)) - 1;
  bool const sticky          = ((quotient128 & low_mask) != 0) || (division_rem != 0);

  uint64_t rounded = mantissa;
  if (round_bit && (sticky || (mantissa & 1ULL))) { rounded += 1; }

  int unbiased_exp = msb_pos + binary_scale;
  if (rounded >= (1ULL << 53)) {
    rounded >>= 1;
    unbiased_exp += 1;
  }
  return {rounded, unbiased_exp};
}

// (4) Assemble the IEEE 754 double bit pattern.
//
// Combines the normalized 53-bit mantissa, the unbiased exponent, and the sign
// into the standard 1+11+52 layout, then bit-casts to double.
__device__ __inline__ double assemble_ieee754_double(uint64_t mantissa, int unbiased_exp, int sign)
{
  int const biased_exp = unbiased_exp + 1023;

  // Subnormal range (biased_exp <= 0) and overflow-to-infinity (biased_exp
  // >= 0x7FF) are both unreachable: msb_pos lies in [53, 127] under the
  // caller's precondition, binary_scale lies in {0, -64}, so unbiased_exp is
  // in [-11, 127] and biased_exp is in [1012, 1150], well inside the normal
  // range [1, 0x7FE]. The default path handles subnormals and overflows.
  assert(biased_exp > 0 && biased_exp < 0x7FF);

  uint64_t const mant_bits = mantissa & ((1ULL << 52) - 1);
  uint64_t bits            = (static_cast<uint64_t>(biased_exp) << 52) | mant_bits;
  if (sign < 0) { bits |= (1ULL << 63); }
  return cuda::std::bit_cast<double>(bits);
}

/**
 * @brief Correctly-rounded conversion of `digits * 10^q` to a double for the
 *        case where `digits` exceeds the safe-cast range of double (2^53) and
 *        the existing `static_cast<double>(digits) * exp10(q)` path therefore
 *        loses up to 1 ULP. See NVIDIA/spark-rapids#10773.
 *
 *        Caller precondition: `digits > 2^53 AND |q| <= 19`. The caller (the
 *        high-precision-path gate inside `string_to_float::operator()`) guards
 *        on both bounds before invoking, so they hold inside. The 19 cap matches
 *        the largest power of ten that fits in `uint64_t` (10^19), which the
 *        128-bit arithmetic below relies on. Subnormal range, overflow to
 *        infinity, and degenerate inputs (`digits == 0`, `q_lo == q_hi == 0`)
 *        are all unreachable under that precondition and are asserted inside
 *        the sub-helpers.
 *
 *        The four named steps each live in their own helper:
 *          (1) `scale_digits_times_pow10`   — 128-bit scaling
 *          (2) `locate_msb_uint128`         — MSB location
 *          (3) `round_top_53_bits`          — round-to-nearest-ties-to-even
 *          (4) `assemble_ieee754_double`    — final bit assembly
 *        All four are `__device__ __inline__` and operate on integers only, so
 *        nvcc folds them back into one straight-line sequence in the calling
 *        kernel with no FP-semantics drift across the function boundaries.
 */
__device__ __inline__ double correctly_rounded_uint64_times_pow10(uint64_t digits, int q, int sign)
{
  auto const scaled = scale_digits_times_pow10(digits, q);
  int const msb_pos = locate_msb_uint128(scaled.quotient128);
  auto const rounded =
    round_top_53_bits(scaled.quotient128, scaled.division_rem, msb_pos, scaled.binary_scale);
  return assemble_ieee754_double(rounded.mantissa, rounded.unbiased_exp, sign);
}

/**
 * @brief Default `digits * 10^exp_ten` conversion via
 *        `static_cast<double>(digits) * exp10(exp_ten)`, including subnormal
 *        and overflow handling. Used for float outputs, and for double outputs
 *        outside the high-precision helper's window (`digits <= 2^53` or
 *        `|exp_ten| > 19`).
 *
 *        Returns T-typed result with the input sign applied. The caller is
 *        responsible for the `_warp_lane == 0` guard around the call site.
 */
template <typename T>
__device__ __inline__ T default_double_path(uint64_t digits, int exp_ten, int sign)
{
  double digitsf = sign >= 0 ? static_cast<double>(digits) : -static_cast<double>(digits);

  if (exp_ten > cuda::std::numeric_limits<double>::max_exponent10) {
    return sign >= 0 ? cuda::std::numeric_limits<T>::infinity()
                     : -cuda::std::numeric_limits<T>::infinity();
  }

  // make sure we don't produce a subnormal number.
  // - a normal number is one where the leading digit of the floating point rep is not zero.
  //      eg:   0.0123  represented as  1.23e-2
  //
  // - a denormalized number is one where the leading digit of the floating point rep is zero.
  //      eg:   0.0123 represented as   0.123e-1
  //
  // - a subnormal number is a denormalized number where if you tried to normalize it, the
  // exponent
  //   required would be smaller then the smallest representable exponent.
  //
  // https://en.wikipedia.org/wiki/Denormal_number
  //
  auto const subnormal_shift = cuda::std::numeric_limits<double>::min_exponent10 - exp_ten;
  if (subnormal_shift > 0) {
    // Handle subnormal values. Ensure that both base and exponent are
    // normal values before computing their product.
    int const num_digits = static_cast<int>(log10(static_cast<double>(digits))) + 1;
    digitsf              = digitsf / exp10(static_cast<double>(num_digits - 1 + subnormal_shift));
    exp_ten += num_digits - 1;  // adjust exponent
    auto const exponent = exp10(static_cast<double>(exp_ten + subnormal_shift));
    return static_cast<T>(digitsf * exponent);
  }
  double const exponent = exp10(static_cast<double>(cuda::std::abs(exp_ten)));
  double const result   = exp_ten < 0 ? digitsf / exponent : digitsf * exponent;
  return static_cast<T>(result);
}

/**
 * @brief Identify if a character is whitespace or C0 control code.
 *
 * @param chr character to test
 * @return true if character is a whitespace character
 */
__host__ __device__ constexpr bool is_whitespace(char const chr)
{
  // Whitespace characters include:
  // - Space (0x20, ' ')
  // - Form feed (0x0c, '\f')
  // - Line feed (0x0a, '\n')
  // - Carriage return (0x0d, '\r')
  // - Horizontal tab (0x09, '\t')
  // - Vertical tab (0x0b, '\v')
  auto const c = static_cast<unsigned char>(chr);
  return c <= 0x001F || c == ' ';
}

template <typename T, size_type block_size>
class string_to_float {
 public:
  __device__ string_to_float(T* out,
                             bitmask_type* validity,
                             int32_t* ansi_except,
                             size_type* valid_count,
                             const char* const chars,
                             size_type const* offsets,
                             uint64_t const* const ipow,
                             bitmask_type const* incoming_null_mask,
                             size_type const num_rows)
    : _out(out),
      _validity(validity),
      _ansi_except(ansi_except),
      _valid_count(valid_count),
      _chars(chars),
      _warp_id((threadIdx.x + (blockDim.x * blockIdx.x)) / 32),
      _row(_warp_id),
      _warp_lane((threadIdx.x + (blockDim.x * blockIdx.x)) % 32),
      _row_start(offsets[_row]),
      _len(offsets[_row + 1] - _row_start),
      _ipow(ipow),
      _incoming_null_mask(incoming_null_mask),
      _num_rows(num_rows)
  {
  }

  __device__ void operator()()
  {
    _bstart = 0;              // start position of the current batch
    _blen   = min(32, _len);  // length of the batch
    _bpos   = 0;              // current position within the current batch of chars for the warp
    _c      = _warp_lane < _blen ? _chars[_row_start + _warp_lane] : 0;

    if (_incoming_null_mask != nullptr && !bit_is_set(_incoming_null_mask, _row)) {
      _valid = false;
      compute_validity(_valid, _except);
      return;
    }

    remove_leading_whitespace();

    // check for + or -
    int sign = check_for_sign();

    // check for leading nan
    if (check_for_nan(sign)) {
      _out[_row] = NAN;
      compute_validity(_valid, _except);
      return;
    }

    // check for inf / infinity
    if (check_for_inf()) {
      if (_warp_lane == 0) {
        _out[_row] = sign >= 0 ? cuda::std::numeric_limits<T>::infinity()
                               : -cuda::std::numeric_limits<T>::infinity();
      }
      compute_validity(_valid, _except);
      return;
    }

    // parse the remainder as floating point.
    auto const [digits, exp_base] = parse_digits();
    if (!_valid) {
      compute_validity(_valid, _except);
      return;
    }

    // parse any manual exponent
    auto const manual_exp = parse_manual_exp();
    if (!_valid) {
      compute_validity(_valid, _except);
      return;
    }

    // 0 / -0.
    if (digits == 0) {
      remove_leading_whitespace();
      if (_bpos < _blen) {
        _valid  = false;
        _except = true;
      }

      if (_warp_lane == 0) {
        _out[_row] = sign >= 0 ? static_cast<double>(0) : -static_cast<double>(0);
      }
      compute_validity(_valid, _except);
      return;
    }

    check_trailing_bytes();
    if (!_valid) {
      compute_validity(_valid, _except);
      return;
    }

    // construct the final float value
    if (_warp_lane == 0) {
      int const exp_ten = exp_base + manual_exp;

      // Two output paths:
      //  - High-precision helper for the `digits > 2^53 AND |q| <= 19` window
      //    (T == double only), where the default `static_cast<double>(digits)
      //    * exp10(exp_ten)` path loses up to 1 ULP (NVIDIA/spark-rapids#10773).
      //  - Default path for everything else: float outputs, or doubles outside
      //    the helper window (small digits or large |q|).
      bool const helper_eligible = cuda::std::is_same_v<T, double> && (digits > (1ULL << 53)) &&
                                   (cuda::std::abs(exp_ten) <= 19);
      if (helper_eligible) {
        _out[_row] = static_cast<T>(correctly_rounded_uint64_times_pow10(digits, exp_ten, sign));
      } else {
        _out[_row] = default_double_path<T>(digits, exp_ten, sign);
      }
    }
    compute_validity(_valid, _except);
  }

  static constexpr int max_safe_digits = 19;

 private:
  // shuffle down to remove whitespace
  __device__ void remove_leading_whitespace()
  {
    do {
      // skip any leading whitespace
      //
      auto const chars_left = _blen - _bpos;
      auto const non_whitespace_mask =
        __ballot_sync(0xffffffff, _warp_lane < chars_left && !is_whitespace(_c));
      auto const first_non_whitespace = __ffs(non_whitespace_mask) - 1;

      if (first_non_whitespace > 0) {
        _bpos += first_non_whitespace;
        _c = __shfl_down_sync(0xffffffff, _c, first_non_whitespace);
      } else if (non_whitespace_mask == 0) {
        //  all whitespace
        _bpos += chars_left;
      }

      if (_bpos == _blen) {
        _bstart += _blen;
        // nothing left to read?
        if (_bstart == _len) { break; }
        // read the next batch
        _bpos = 0;
        _blen = min(32, _len - _bstart);
        _c    = _warp_lane < _blen ? _chars[_row_start + _bstart + _warp_lane] : 0;
      } else {
        break;
      }
    } while (1);
  }

  // returns true if we encountered 'nan'
  // potentially changes:  valid/except
  __device__ bool check_for_nan(int const& sign)
  {
    auto const nan_mask = __ballot_sync(0xffffffff,
                                        (_warp_lane == 0 && (_c == 'N' || _c == 'n')) ||
                                          (_warp_lane == 1 && (_c == 'A' || _c == 'a')) ||
                                          (_warp_lane == 2 && (_c == 'N' || _c == 'n')));
    if (nan_mask == 0x7) {
      // if we start with 'nan', then even if we have other garbage character(excluding
      // whitespaces), this is a null row. but for e.g. : "nan   " cases. spark will treat the as
      // "nan", when the trailing characters are whitespaces, it is still a valid string. if we're
      // in ansi mode and this is not -precisely- nan, report that so that we can throw an exception
      // later.

      // move forward the current position by 3
      _bpos += 3;
      _c = __shfl_down_sync(0xffffffff, _c, 3);

      // remove the trailing whitespaces, if there exits
      remove_leading_whitespace();

      // if we're at the end and there is no sign, because Spark treats '-nan' and '+nan' as null.
      if (_bpos == _len && sign == 0) { return true; }
      // if we reach out here, it means that we have other garbage character.
      _valid  = false;
      _except = true;
    }
    return false;
  }

  // The `sign` variables is initialized to 0, indicating no sign.
  // If a sign is detected, it sets `sign` to 1, indicating `+` sign.
  // If `-` is then detected, it sets `sign` to -1.
  // returns 1, 0, -1 to indicate signs.
  __device__ int check_for_sign()
  {
    auto const sign_mask = __ballot_sync(0xffffffff, _warp_lane == 0 && (_c == '+' || _c == '-'));
    int sign             = 0;
    if (sign_mask) {
      sign = 1;
      // NOTE: warp lane 0 is the only thread that ever reads `sign`, so technically it would be
      // valid to just check if(c == '-'), but that would leave other threads with an incorrect
      // value. if this code ever changes, that could lead to hard-to-find bugs.
      if (__ballot_sync(0xffffffff, _warp_lane == 0 && _c == '-')) { sign = -1; }
      _bpos++;
      _c = __shfl_down_sync(0xffffffff, _c, 1);
    }
    return sign;
  }

  // returns true if we encountered an inf
  // potentially changes:  valid
  __device__ bool check_for_inf()
  {
    // check for inf or infinity
    auto const inf_mask = __ballot_sync(0xffffffff,
                                        (_warp_lane == 0 && (_c == 'I' || _c == 'i')) ||
                                          (_warp_lane == 1 && (_c == 'N' || _c == 'n')) ||
                                          (_warp_lane == 2 && (_c == 'F' || _c == 'f')));
    if (inf_mask == 0x7) {
      _bpos += 3;
      _c = __shfl_down_sync(0xffffffff, _c, 3);

      // if we're at the end
      if (_bpos == _len) { return true; }

      // see if we have the whole word
      auto const infinity_mask = __ballot_sync(0xffffffff,
                                               (_warp_lane == 0 && (_c == 'I' || _c == 'i')) ||
                                                 (_warp_lane == 1 && (_c == 'N' || _c == 'n')) ||
                                                 (_warp_lane == 2 && (_c == 'I' || _c == 'i')) ||
                                                 (_warp_lane == 3 && (_c == 'T' || _c == 't')) ||
                                                 (_warp_lane == 4 && (_c == 'Y' || _c == 'y')));
      if (infinity_mask == 0x1f) {
        _bpos += 5;
        // if we're at the end
        if (_bpos == _len) { return true; }
        _c = __shfl_down_sync(0xffffffff, _c, 5);
      }

      // remove the remaining whitespace if exists
      remove_leading_whitespace();

      // if we're at the end
      if (_bpos == _len) { return true; }

      // if we reach here for any reason, it means we have "inf" or "infinity" at the start of the
      // string but also have additional characters, making this whole thing bogus/null
      _valid = false;

      return true;
    }
    return false;
  }

  // parse the actual digits.  returns 64 bit digit holding value and exponent
  __device__ cuda::std::pair<uint64_t, int> parse_digits()
  {
    typedef cub::WarpReduce<uint64_t> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage;

    // what we will need to compute the exponent
    uint64_t digits      = 0;
    int real_digits      = 0;  // total # of digits we've got stored in 'digits'
    int truncated_digits = 0;  // total # of digits we've had to truncate off
    // the # of total digits is (real_digits + truncated_digits)
    bool decimal    = false;  // whether or not we have a decimal
    int decimal_pos = 0;      // absolute decimal pos

    // have we seen a valid digit yet?
    bool seen_valid_digit = false;
    do {
      int num_chars = _blen - _bpos;

      // if our current sum is 0 and we don't have a decimal, strip leading
      // zeros.  handling cases such as
      // 0000001
      if (!decimal && digits == 0) {
        auto const zero_mask = __ballot_sync(0xffffffff, _warp_lane < num_chars && _c != '0');
        // zero_mask is 0 if all digits are 0's and we need to strip those as well.
        auto const nz_pos = zero_mask == 0 ? num_chars : __ffs(zero_mask) - 1;
        if (nz_pos > 0) {
          num_chars -= nz_pos;
          _bpos += nz_pos;
          _c               = __shfl_down_sync(0xffffffff, _c, nz_pos);
          seen_valid_digit = true;
        }
      }

      // handle a decimal point
      auto const decimal_mask = __ballot_sync(0xffffffff, _warp_lane < num_chars && _c == '.');
      if (decimal_mask) {
        // if we have more than one decimal, this is an invalid value
        if (decimal || __popc(decimal_mask) > 1) {
          _valid  = false;
          _except = true;
          return {0, 0};
        }
        auto const dpos = __ffs(decimal_mask) - 1;  // 0th bit is reported as 1 by __ffs
        decimal_pos     = (dpos + real_digits);
        decimal         = true;

        // strip the decimal char out
        if (_warp_lane >= dpos) { _c = __shfl_down_sync(~((1 << dpos) - 1), _c, 1); }
        num_chars--;
      }

      // handle any chars that are not actually digits
      //
      auto const non_digit_mask =
        __ballot_sync(0xffffffff, _warp_lane < num_chars && !is_digit(_c));
      auto const first_non_digit = __ffs(non_digit_mask);

      // first non-digit after location 1 means there is something valid here, note ffs is 0 with no
      // set bits, so 1 is the 0th character is not a digit. first non-digit of 0 means all digits,
      // and that means we have seen a valid digit as well.
      seen_valid_digit |= (num_chars > 0 && first_non_digit != 1);

      num_chars = min(num_chars, first_non_digit > 0 ? first_non_digit - 1 : num_chars);

      if (decimal_pos > 0 && decimal_pos > num_chars + real_digits) {
        _valid  = false;
        _except = true;
        return {0, 0};
      }

      if (num_chars == 0 && _blen == _len) {
        if (!seen_valid_digit) {
          _valid  = false;
          _except = true;
        }
        return {0, 0};
      }

      // we may have to start truncating because we'd go past the 64 bit limit by adding the new
      // digits
      //
      // max uint64_t is 20 digits, so any 19 digit number is valid.
      // 2^64:  18,446,744,073,709,551,616
      //         9,999,999,999,999,999,999
      //
      // if the 20th digit would push us past that limit, we have to start truncating.
      // max_holding:  1,844,674,407,370,955,160
      // so     1,844,674,407,370,955,160 + 9    -> 18,446,744,073,709,551,609  -> ok
      //        1,844,674,407,370,955,160 + 1X   -> 18,446,744,073,709,551,61X  -> potentially rolls
      //        past the limit
      //
      constexpr uint64_t max_holding = (cuda::std::numeric_limits<uint64_t>::max() - 9) / 10;
      // if we're already past the max_holding, just truncate.
      // eg:    9,999,999,999,999,999,999
      if (digits > max_holding) {
        truncated_digits += num_chars;
      } else {
        // add as many digits to the running sum as we can.
        int const safe_count = min(max_safe_digits - real_digits, num_chars);

        // our local digit
        uint64_t const digit = _warp_lane < safe_count ? static_cast<uint64_t>(_c - '0') *
                                                           _ipow[(safe_count - _warp_lane) - 1]
                                                       : 0;

        if (safe_count > 0) {
          // only lane 0 will have the real value so we need to shfl it to the rest of the threads.
          digits = (digits * _ipow[safe_count]) +
                   __shfl_sync(0xffffffff, WarpReduce(temp_storage).Sum(digit, safe_count), 0);
          real_digits += safe_count;
        }

        // if we have more digits
        if (safe_count < num_chars) {
          // we're already past max_holding so we have to start truncating
          if (digits > max_holding) {
            truncated_digits += num_chars - safe_count;
          }
          // we may be able to add one more digit.
          else {
            auto const last_digit =
              static_cast<uint64_t>(__shfl_sync(0xffffffff, _c, safe_count) - '0');
            if ((digits * 10) + last_digit <= max_holding) {
              // we can add this final digit
              digits = (digits * 10) + last_digit;
              truncated_digits += num_chars - (safe_count - 1);
            }
            // everything else gets truncated
            else {
              truncated_digits += num_chars - safe_count;
            }
          }
        }
      }
      _bpos += num_chars + (decimal_mask > 0);

      // read the next batch of chars.
      if (_bpos == _blen) {
        _bstart += _blen;
        // nothing left to read?
        if (_bstart == _len) { break; }
        // read the next batch
        _bpos = 0;
        _blen = min(32, _len - _bstart);
        _c    = _warp_lane < _blen ? _chars[_row_start + _bstart + _warp_lane] : 0;
      } else {
        _c = __shfl_down_sync(0xffffffff, _c, num_chars);

        // if we encountered a non-digit, we're done
        if (first_non_digit) { break; }
      }
    } while (1);

    // 0 / -0.
    if (digits == 0) { return {0, 0}; }

    // the total amount of actual digits
    auto const total_digits = real_digits + truncated_digits;

    // exponent
    // any truncated digits are effectively just trailing zeros
    int exp_ten = (truncated_digits
                   // if we've got a decimal, shift left by it's position
                   - (decimal ? (total_digits - decimal_pos) : 0));
    return {digits, exp_ten};
  }

  // parse manually specified exponent.
  // potentially changes: valid
  __device__ int parse_manual_exp()
  {
    typedef cub::WarpReduce<uint64_t> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage;

    // if we still have chars left, the only thing legal now is a manual exponent.
    // eg:  E-10
    //
    int manual_exp = 0;
    if (_bpos < _blen) {
      // read some trailing chars.

      auto const exp_mask =
        __ballot_sync(0xffffffff, (_warp_lane == 0 && (_c == 'E' || _c == 'e')));
      if (!exp_mask) { return 0; }
      auto const exp_sign_mask =
        __ballot_sync(0xffffffff, (_warp_lane == 1 && (_c == '-' || _c == '+')));
      auto const exp_sign =
        exp_sign_mask ? __ballot_sync(0xffffffff, _warp_lane == 1 && _c == '-') ? -1 : 1 : 1;
      auto const chars_to_skip = exp_sign_mask ? 2 : 1;
      _c                       = __shfl_down_sync(0xffffffff, _c, chars_to_skip);
      _bpos += chars_to_skip;

      // the largest valid exponent for a double is 4 digits (3 for floats).
      int const num_chars = min(4, _blen - _bpos);

      // handle any chars that are not actually digits
      //
      auto const non_digit_mask =
        __ballot_sync(0xffffffff, _warp_lane < num_chars && !is_digit(_c));
      auto const first_non_digit = __ffs(non_digit_mask);

      int const num_digits = first_non_digit > 0 ? first_non_digit - 1 : num_chars;

      if (num_digits == 0) {
        _valid  = false;
        _except = true;
        return 0;
      }

      uint64_t const digit = _warp_lane < num_digits ? static_cast<uint64_t>(_c - '0') *
                                                         _ipow[(num_digits - _warp_lane) - 1]
                                                     : 0;
      manual_exp           = WarpReduce(temp_storage).Sum(digit, num_digits) * exp_sign;
      _c                   = __shfl_down_sync(0xffffffff, _c, num_digits);
      _bpos += num_digits;
    }

    return manual_exp;
  }

  __device__ void check_trailing_bytes()
  {
    if (_blen - _bpos > 0) {
      // strip trailing f if it exists
      // f is a valid character at the end of a float string
      auto const f_mask = __ballot_sync(
        0xffffffff, (_warp_lane == 0 && (_c == 'F' || _c == 'f' || _c == 'd' || _c == 'D')));
      if (f_mask > 0) {
        _c = __shfl_down_sync(0xffffffff, _c, 1);
        _bpos++;
      }
    }

    // nothing trailing
    if (_blen - _bpos == 0) { return; }

    // strip any whitespace
    remove_leading_whitespace();

    // invalid characters in string
    if (_blen - _bpos > 0) {
      _valid  = false;
      _except = true;
    }
  }

  // sets validity bits, updates outgoing validity count for the block and potentially sets the
  // outgoing ansi_except field
  __device__ void compute_validity(bool const valid, bool const except = false)
  {
    if (threadIdx.x == 0 && _ansi_except && except) { atomicMax(_ansi_except, _num_rows - _row); }

    // 0th thread in each warp updates the validity
    size_type const row_id = _warp_id;
    if (threadIdx.x % 32 == 0 && valid) {
      // uses atomics
      cudf::set_bit(_validity, row_id);
      atomicAdd(_valid_count, 1);
    }
  }

  T* _out;
  bitmask_type* _validity;
  int32_t* _ansi_except;
  size_type* _valid_count;
  const char* const _chars;
  size_type const _warp_id;
  size_type const _row;
  size_type const _warp_lane;
  size_type const _row_start;
  size_type const _len;
  size_type const _num_rows;
  uint64_t const* const _ipow;
  bitmask_type const* _incoming_null_mask;

  // shared/modified by the various parsing functions
  size_type _bstart;  // batch start within the entire string
  size_type _bpos;    // position with current batch
  size_type _blen;    // batch length;
  char _c;            // current character
  bool _valid  = true;
  bool _except = false;
};

template <typename T, size_type block_size>
CUDF_KERNEL void string_to_float_kernel(T* out,
                                        bitmask_type* validity,
                                        int32_t* ansi_except,
                                        size_type* valid_count,
                                        const char* const chars,
                                        size_type const* offsets,
                                        bitmask_type const* incoming_null_mask,
                                        size_type const num_rows)
{
  size_type const tid = threadIdx.x + (blockDim.x * blockIdx.x);
  size_type const row = tid / 32;
  if (row >= num_rows) { return; }

  // one more than max safe digits to ensure that we can reference
  // max_safe_digits into the array.
  __shared__ uint64_t ipow[string_to_float<T, block_size>::max_safe_digits + 1];
  if (threadIdx.x == 0) {
    ipow[0]  = 1;
    ipow[1]  = 10;
    ipow[2]  = 100;
    ipow[3]  = 1000;
    ipow[4]  = 10000;
    ipow[5]  = 100000;
    ipow[6]  = 1000000;
    ipow[7]  = 10000000;
    ipow[8]  = 100000000;
    ipow[9]  = 1000000000;
    ipow[10] = 10000000000;
    ipow[11] = 100000000000;
    ipow[12] = 1000000000000;
    ipow[13] = 10000000000000;
    ipow[14] = 100000000000000;
    ipow[15] = 1000000000000000;
    ipow[16] = 10000000000000000;
    ipow[17] = 100000000000000000;
    ipow[18] = 1000000000000000000;
    ipow[19] = 10000000000000000000;
  }
  __syncthreads();

  // convert
  string_to_float<T, block_size>{
    out, validity, ansi_except, valid_count, chars, offsets, ipow, incoming_null_mask, num_rows}();
}

}  // namespace detail

/**
 * @brief Convert a string column into an float column.
 *
 * @param dtype Type of column to return.
 * @param string_col Incoming string column to convert to integers.
 * @param ansi_mode If true, strict conversion and throws on error.
 *                  If false, null invalid entries.
 * @param stream Stream on which to operate.
 * @param mr Memory resource for returned column
 * @return std::unique_ptr<column> Integer column that was created from string_col.
 */
std::unique_ptr<column> string_to_float(data_type dtype,
                                        strings_column_view const& string_col,
                                        bool ansi_mode,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(dtype == data_type{type_id::FLOAT32} || dtype == data_type{type_id::FLOAT64},
               "invalid float data type");
  if (string_col.size() == 0) { return cudf::make_empty_column(dtype); }

  auto out = cudf::make_numeric_column(dtype, string_col.size(), mask_state::ALL_NULL, stream, mr);

  using ScalarType = cudf::scalar_type_t<size_type>;
  auto valid_count = cudf::make_numeric_scalar(cudf::data_type(cudf::type_to_id<size_type>()));
  auto ansi_count  = cudf::make_numeric_scalar(cudf::data_type(cudf::type_to_id<size_type>()));
  static_cast<ScalarType*>(valid_count.get())->set_value(0, stream);
  if (ansi_mode) { static_cast<ScalarType*>(ansi_count.get())->set_value(-1, stream); }

  constexpr auto warps_per_block = 8;
  constexpr auto rows_per_block  = warps_per_block;
  auto const num_blocks = cudf::util::div_rounding_up_safe(string_col.size(), rows_per_block);
  auto const num_rows   = string_col.size();

  if (dtype == data_type{type_id::FLOAT32}) {
    detail::string_to_float_kernel<float, warps_per_block * 32>
      <<<num_blocks, warps_per_block * 32>>>(
        out->mutable_view().begin<float>(),
        out->mutable_view().null_mask(),
        ansi_mode ? static_cast<ScalarType*>(ansi_count.get())->data() : nullptr,
        static_cast<ScalarType*>(valid_count.get())->data(),
        string_col.chars_begin(stream),
        string_col.offsets().begin<size_type>(),
        string_col.null_mask(),
        num_rows);
  } else {
    detail::string_to_float_kernel<double, warps_per_block * 32>
      <<<num_blocks, warps_per_block * 32>>>(
        out->mutable_view().begin<double>(),
        out->mutable_view().null_mask(),
        ansi_mode ? static_cast<ScalarType*>(ansi_count.get())->data() : nullptr,
        static_cast<ScalarType*>(valid_count.get())->data(),
        string_col.chars_begin(stream),
        string_col.offsets().begin<size_type>(),
        string_col.null_mask(),
        num_rows);
  }

  out->set_null_count(num_rows - static_cast<ScalarType*>(valid_count.get())->value(stream));

  if (ansi_mode) {
    auto const val = static_cast<ScalarType*>(ansi_count.get())->value(stream);
    if (val >= 0) {
      auto const error_row = num_rows - val;
      size_type string_bounds[2];
      cudaMemcpyAsync(&string_bounds,
                      &string_col.offsets().data<size_type>()[error_row],
                      sizeof(size_type) * 2,
                      cudaMemcpyDeviceToHost,
                      stream.value());
      stream.synchronize();

      std::string dest;
      dest.resize(string_bounds[1] - string_bounds[0]);

      cudaMemcpyAsync(dest.data(),
                      &string_col.chars_begin(stream)[string_bounds[0]],
                      string_bounds[1] - string_bounds[0],
                      cudaMemcpyDeviceToHost,
                      stream.value());
      stream.synchronize();

      throw cast_error(error_row, dest);
    }
  }

  return out;
}

}  // namespace spark_rapids_jni
