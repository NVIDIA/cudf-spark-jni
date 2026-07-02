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
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/null_mask.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cub/warp/warp_reduce.cuh>
#include <cuda/std/algorithm>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>

using namespace cudf;

namespace spark_rapids_jni {

namespace detail {

constexpr auto NUM_THREADS{256};

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

template <typename T, std::enable_if_t<cuda::std::is_signed_v<T>>* = nullptr>
T __device__ generic_abs(T value)
{
  return numeric::detail::abs(value);
}

template <typename T, std::enable_if_t<not cuda::std::is_signed_v<T>>* = nullptr>
constexpr T __device__ generic_abs(T value)
{
  return value;
}

/**
 * @brief Determine if overflow will occur when multiplying a value by 10.
 *
 * @tparam T type of the values
 * @param val value to test
 * @param adding adding or subtracting values
 * @return true if overflow will occur by multiplying val by 10
 */
template <typename T>
bool __device__ will_overflow(T const val, bool adding)
{
  if constexpr (cuda::std::is_signed_v<T>) {
    if (!adding) {
      auto constexpr minval = cuda::std::numeric_limits<T>::min() / 10;
      return val < minval;
    }
  }

  auto constexpr maxval = cuda::std::numeric_limits<T>::max() / 10;
  return val > maxval;
}

/**
 * @brief Determine if overflow will occur when adding or subtracting values.
 *
 * @tparam T type of the values
 * @param lhs left hand side of the operation
 * @param rhs right hand side of the operation
 * @param adding true if adding, false if subtracting
 * @return true if overflow will occur from the operation
 */
template <typename T>
bool __device__ will_overflow(T const lhs, T const rhs, bool adding)
{
  if constexpr (cuda::std::is_signed_v<T>) {
    if (!adding) {
      auto const minval = cuda::std::numeric_limits<T>::min() + rhs;
      return lhs < minval;
    }
  }

  auto const maxval = cuda::std::numeric_limits<T>::max() - rhs;
  return lhs > maxval;
}

/**
 * @brief process a single digit of input
 *
 * @tparam T type of the input data
 * @param first_value true if this is the first value added
 * @param current_val current computed value
 * @param new_digit digit to append to the computed value
 * @param adding true if adding, false if subtracting
 * @return true if success, false if overflow
 */
template <typename T>
cuda::std::pair<bool, T> __device__
process_value(bool first_value, T current_val, T const new_digit, bool adding)
{
  if (!first_value) {
    if (will_overflow(current_val, adding)) { return {false, current_val}; }

    current_val *= 10;
  }

  if (will_overflow(current_val, new_digit, adding)) { return {false, current_val}; }

  if (adding) {
    current_val += new_digit;
  } else {
    current_val -= new_digit;
  }

  return {true, current_val};
}

/**
 * @brief kernel to cast an array of strings to an array of integers
 *
 * @tparam T type of the integer array
 * @param out integer array being written
 * @param validity validity bit array for output data
 * @param chars incoming character array
 * @param offsets incoming offsets array into the character array
 * @param incoming_null_mask incoming null mask for offsets array
 * @param num_rows total number of elements in the integer array
 * @param ansi_mode true if ansi mode is required, which is more strict and throws
 */
template <typename T>
CUDF_KERNEL void string_to_integer_kernel(T* out,
                                          bitmask_type* validity,
                                          const char* const chars,
                                          size_type const* offsets,
                                          bitmask_type const* incoming_null_mask,
                                          size_type num_rows,
                                          bool ansi_mode,
                                          bool strip)
{
  auto const group = cooperative_groups::this_thread_block();
  auto const warp  = cooperative_groups::tiled_partition<cudf::detail::warp_size>(group);

  // each thread takes a row and marches it and builds the integer for that row
  auto const row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) { return; }
  auto const active      = cooperative_groups::coalesced_threads();
  auto const row_start   = offsets[row];
  auto const len         = offsets[row + 1] - row_start;
  bool const valid_entry = incoming_null_mask == nullptr || bit_is_set(incoming_null_mask, row);
  bool valid             = valid_entry && len > 0;
  T thread_val           = 0;
  int i                  = 0;
  T sign                 = 1;
  constexpr bool is_signed_type = cuda::std::is_signed_v<T>;

  if (valid) {
    if (strip) {
      // skip leading whitespace
      while (i < len && is_whitespace(chars[i + row_start])) {
        i++;
      }
    }

    // check for leading +-
    if constexpr (is_signed_type) {
      if (i < len && (chars[i + row_start] == '+' || chars[i + row_start] == '-')) {
        if (chars[i + row_start] == '-') { sign = -1; }
        i++;
      }
    }

    // if there is no data left, this is invalid
    if (i == len) { valid = false; }

    bool truncating          = false;
    bool trailing_whitespace = false;
    for (int c = i; c < len; ++c) {
      auto const chr = chars[c + row_start];
      // only whitespace is allowed after we find trailing whitespace
      if (trailing_whitespace && !is_whitespace(chr)) {
        valid = false;
        break;
      } else if (!truncating && chr == '.' && !ansi_mode) {
        // Values are truncated after a decimal point. However, invalid characters AFTER this
        // decimal point will still invalidate this entry.
        truncating = true;
      } else {
        if (chr > '9' || chr < '0') {
          if (is_whitespace(chr) && c != i && strip) {
            trailing_whitespace = true;
          } else {
            // invalid character in string!
            valid = false;
            break;
          }
        }
      }

      if (!truncating && !trailing_whitespace) {
        T const new_digit             = chr - '0';
        auto const [success, new_val] = process_value(c == i, thread_val, new_digit, sign > 0);
        if (!success) {
          valid = false;
          break;
        }
        thread_val = new_val;
      }
    }

    out[row] = thread_val;
  }

  auto const validity_int32 = warp.ballot(static_cast<int>(valid));
  if (warp.thread_rank() == 0) {
    validity[warp.meta_group_rank() + blockIdx.x * warp.meta_group_size()] = validity_int32;
  }
}

template <typename T>
__device__ cuda::std::optional<cuda::std::tuple<bool, int, int>> validate_and_exponent(
  const char* chars, const int len, bool strip)
{
  T exponent_val         = 0;
  int i                  = 0;
  bool positive          = true;
  bool exponent_positive = true;
  int decimal_location   = -1;

  // first pass is validation and figuring out decimal location taking into account possible
  // scientific notation.

  enum processing_state {
    ST_DIGITS = 0,           // digits of decimal value
    ST_EXPONENT,             // digits of exponent
    ST_DECIMAL_POINT,        // reading a decimal point
    ST_EXPONENT_OR_SIGN,     // reading exponent or sign value
    ST_EXPONENT_SIGN,        // reading an exponent sign value
    ST_VALIDATING_ONLY,      // validating string, but no longer processing values
    ST_TRAILING_WHITESPACE,  // skipping over trailing whitespace
    ST_INVALID,              // invalid data
  };

  auto validate_char = [&decimal_location, &exponent_positive, &strip](
                         processing_state const state, const char chr, int chr_idx) {
    switch (state) {
      case ST_TRAILING_WHITESPACE:
        if (!is_whitespace(chr)) { return ST_INVALID; }
        break;
      case ST_DECIMAL_POINT:
      case ST_DIGITS:
        if (chr > '9' || chr < '0') {
          if (chr == '.' && decimal_location == -1) {
            decimal_location = chr_idx;
            return ST_DECIMAL_POINT;
          } else if (chr == 'e' || chr == 'E') {
            return ST_EXPONENT_OR_SIGN;
          } else if (strip && is_whitespace(chr) && chr_idx != 0) {
            return ST_TRAILING_WHITESPACE;
          } else {
            // invalid character
            return ST_INVALID;
          }
        }
        return ST_DIGITS;
      case ST_EXPONENT_OR_SIGN:
        // we just skipped the e value, next is either a sign or a digit
        if (chr == '+') {
          return ST_EXPONENT_SIGN;
        } else if (chr == '-') {
          exponent_positive = false;
          return ST_EXPONENT_SIGN;
        } else if (strip && is_whitespace(chr) && chr_idx != 0) {
          return ST_TRAILING_WHITESPACE;
        } else if (chr > '9' || chr < '0') {
          return ST_INVALID;
        } else {
          return ST_EXPONENT;
        }
        break;
      case ST_EXPONENT_SIGN:
      case ST_EXPONENT:
        if (chr > '9' || chr < '0') {
          return ST_INVALID;
        } else {
          return ST_EXPONENT;
        }
        break;
    }
    return state;
  };

  if (len == 0) { return cuda::std::nullopt; }

  processing_state state = ST_DIGITS;

  if (strip) {
    // skip leading whitespace
    while (i < len && is_whitespace(chars[i])) {
      i++;
    }
  }

  // check for leading +-
  if (chars[i] == '-') {
    positive = false;
    i++;
  } else if (chars[i] == '+') {
    i++;
  }

  // if there is no data left, this is invalid
  if (i == len) { return cuda::std::nullopt; }

  auto const first_digit = i;
  int last_digit         = len;
  for (int c = i; c < len; ++c) {
    auto const chr        = chars[c];
    auto const char_num   = c - i;
    auto const last_state = state;
    state                 = validate_char(state, chr, char_num);

    if (state == ST_INVALID) { return cuda::std::nullopt; }

    if (last_state == ST_DIGITS && state != ST_DIGITS && state != ST_DECIMAL_POINT) {
      // past digits, save location
      last_digit = c;
    }

    if (state == ST_EXPONENT) {
      T const new_digit = chr - '0';
      auto const [success, new_val] =
        process_value(exponent_val == 0, exponent_val, new_digit, exponent_positive);
      if (!success) { return cuda::std::nullopt; }
      exponent_val = new_val;
    }
  }

  // decimal location moves to end of digits if no decimal found
  if (decimal_location < 0) { decimal_location = last_digit - first_digit; }

  // adjust decimal location based on exponent
  decimal_location += exponent_val;

  return cuda::std::tuple{positive, decimal_location, first_digit};
}

/**
 * @brief kernel to cast an array of strings to an array of decimal values
 *
 * @tparam T underlying data type for the decimal array
 * @param out array of decimal values
 * @param validity validity for output array
 * @param chars incoming character array
 * @param offsets incoming offsets array into the character array
 * @param incoming_null_mask incoming null mask for offsets array
 * @param num_rows total number of elements in the integer array
 * @param scale scale of desired decimals
 * @param precision precision of desired decimals
 * @param ansi_mode true if ansi mode is required, which is more strict and throws
 */
template <typename T>
CUDF_KERNEL void string_to_decimal_kernel(T* out,
                                          bitmask_type* validity,
                                          const char* const chars,
                                          size_type const* offsets,
                                          bitmask_type const* incoming_null_mask,
                                          size_type num_rows,
                                          int32_t scale,
                                          int32_t precision,
                                          bool strip)
{
  auto const group = cooperative_groups::this_thread_block();
  auto const warp  = cooperative_groups::tiled_partition<cudf::detail::warp_size>(group);

  // each thread takes a row and marches it and builds the decimal for that row
  auto const row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) { return; }
  auto const active      = cooperative_groups::coalesced_threads();
  auto const row_start   = offsets[row];
  auto const len         = offsets[row + 1] - row_start;
  bool const valid_entry = incoming_null_mask == nullptr || bit_is_set(incoming_null_mask, row);
  bool valid             = valid_entry && len > 0;

  // first_digit is distance into the string array for the first digit to process. This skips +, -,
  // whitespace, etc. decimal_location is the index into the string where the decimal point should
  // live relative to first_digit. Note this isn't always where the decimal point is in the string.
  // If it is the index of a digit, that digit is after the decimal point.

  // turn into std::count_if
  auto count_significant_digits = [](const char* str, int len, int num_digits) {
    int count        = 0;
    int digits_found = 0;
    for (int i = 0; i < len && digits_found < num_digits; ++i) {
      if (str[i] == 'e' || str[i] == 'E') {
        break;
      } else if (str[i] != '.') {
        digits_found++;
        if (count != 0 || str[i] != '0') {
          count++;
          continue;
        }
      }
    }

    return count;
  };

  auto const validated =
    valid ? validate_and_exponent<T>(&chars[row_start], len, strip) : cuda::std::nullopt;
  valid = validated.has_value();

  if (valid) {
    bool positive;
    int decimal_location;
    int first_digit;
    cuda::std::tie(positive, decimal_location, first_digit) = *validated;

    auto const max_digits_before_decimal                   = precision + scale;
    auto const significant_digits_before_decimal_in_string = count_significant_digits(
      &chars[row_start + first_digit], len - first_digit, decimal_location);

    // last digit we can process is scale units before or after the decimal
    // depending on the scale sign. Note that rounding still needs to occur after that digit.
    auto const last_digit = decimal_location - scale;

    // number of precise digits we have encountered
    int num_precise_digits = 0;
    // number of digits we have encountered, even leading 0's
    int total_digits     = 0;
    T thread_val         = 0;
    bool found_sig_digit = false;
    int rounding_digits  = 0;

    if (last_digit >= 0) {
      // march string starting at first_digit and build value
      for (int i = first_digit; i < len && valid; ++i) {
        auto const chr = chars[row_start + i];
        if (chr == '.') {
          continue;
        } else if (chr > '9' || chr < '0') {
          // finished processing
          break;
        }

        T const new_digit = chr - '0';
        if (num_precise_digits + 1 > precision || total_digits + 1 > last_digit) {
          // more digits than required, but we need to round
          if (new_digit >= 5) {
            auto const orig_val = thread_val;
            if (will_overflow(thread_val, static_cast<T>(1), positive)) {
              valid = false;
              break;
            } else if (positive) {
              thread_val++;
            } else {
              thread_val--;
            }
            // we need to know if the first digit overflowed and added a new digit
            // this can only happen if the first digit is lower now than before
            // rounding added a digit. There may be a faster route, but it has to work
            // with __int128_t as well.
            auto count_digits = [](T val) {
              int count = 0;
              while (val != 0) {
                count++;
                val /= 10;
              }
              return count;
            };

            auto before_digits = count_digits(orig_val);
            auto after_digits  = count_digits(thread_val);

            // if original value is 0, we can round to 1 without adding a digit, but
            // count_digits will detect the change.
            if (orig_val != 0 && count_digits(thread_val) > count_digits(orig_val)) {
              // more digits now than before rounding
              total_digits++;
              num_precise_digits++;
              decimal_location++;
              rounding_digits++;
            }
          }
          break;
        }

        total_digits++;
        if (found_sig_digit || total_digits > decimal_location || new_digit != 0) {
          found_sig_digit = true;
          num_precise_digits++;
        }

        auto const [success, new_val] =
          process_value(i == first_digit, thread_val, new_digit, positive);
        if (!success) {
          valid = false;
          break;
        }
        thread_val = new_val;
      }
    }

    auto const significant_preceding_zeros = decimal_location < 0 ? -decimal_location : 0;
    auto const zeros_to_decimal            = cuda::std::max(
      0, scale > 0 ? decimal_location - total_digits - scale : decimal_location - total_digits);
    auto const significant_digits_before_decimal =
      significant_digits_before_decimal_in_string + zeros_to_decimal + rounding_digits;

    // too many digits required to store decimal
    auto const leading_zeros = total_digits - num_precise_digits;
    if (max_digits_before_decimal < decimal_location - leading_zeros) { valid = false; }

    // at this point we have the precise digits we need, but we might need trailing zeros on this
    // value both before and after the decimal

    // add zero pad until we hit the decimal location
    // decimal(6,-2)
    // string: 123456
    // thread_value: 1235
    // result -> 123500
    for (int i = 0; i < zeros_to_decimal; ++i) {
      if (will_overflow(thread_val, positive)) {
        valid = false;
        break;
      }
      thread_val *= 10;
      num_precise_digits++;
    }

    // add zero pad to get to scale
    // decimal(6,5)
    // string: 0.012
    // thread_value: 12
    // result -> 1200
    auto const digits_after_decimal =
      num_precise_digits - significant_digits_before_decimal + significant_preceding_zeros;
    auto const digits_needed_after_decimal =
      min(precision - significant_digits_before_decimal, -scale);

    for (int i = digits_after_decimal; i < digits_needed_after_decimal; ++i) {
      if (will_overflow(thread_val, positive)) {
        valid = false;
        break;
      }
      thread_val *= 10;
    }

    if (valid) { out[row] = thread_val; }
  }

  auto const validity_int32 = warp.ballot(static_cast<int>(valid));
  if (warp.thread_rank() == 0) {
    validity[warp.meta_group_rank() + blockIdx.x * warp.meta_group_size()] = validity_int32;
  }
}

struct row_valid_fn {
  bool __device__ operator()(size_type const row)
  {
    return not bit_is_set(null_mask, row) &&
           (string_col_null_mask == nullptr || bit_is_set(string_col_null_mask, row));
  }

  bitmask_type const* null_mask;
  bitmask_type const* string_col_null_mask;
};

/**
 * @brief validates a column has no ansi errors, throws if error
 *
 * @param col column to verify
 * @param source_col original string column used to create col
 * @param stream stream on which to operate
 */
void validate_ansi_column(column_view const& col,
                          strings_column_view const& source_col,
                          rmm::cuda_stream_view stream)
{
  auto const num_nulls      = col.null_count();
  auto const incoming_nulls = source_col.null_count();
  auto const num_errors     = num_nulls - incoming_nulls;
  if (num_errors > 0) {
    auto const first_error = thrust::find_if(rmm::exec_policy(stream),
                                             thrust::make_counting_iterator(0),
                                             thrust::make_counting_iterator(col.size()),
                                             row_valid_fn{col.null_mask(), source_col.null_mask()});

    size_type string_bounds[2];
    cudaMemcpyAsync(&string_bounds,
                    &source_col.offsets().data<size_type>()[*first_error],
                    sizeof(size_type) * 2,
                    cudaMemcpyDeviceToHost,
                    stream.value());
    stream.synchronize();

    std::string dest;
    dest.resize(string_bounds[1] - string_bounds[0]);

    cudaMemcpyAsync(dest.data(),
                    &source_col.chars_begin(stream)[string_bounds[0]],
                    string_bounds[1] - string_bounds[0],
                    cudaMemcpyDeviceToHost,
                    stream.value());
    stream.synchronize();

    throw cast_error(*first_error, dest);
  }
}

struct string_to_integer_impl {
  /**
   * @brief create column of type T from string columns
   *
   * @tparam T type of column to create
   * @param string_col string column of incoming data
   * @param ansi_mode strict ansi mode checking of incoming data, can throw
   * @param stream stream on which to operate
   * @param mr memory resource to use for allocations
   * @return std::unique_ptr<column> column of type T created from strings
   */
  template <typename T, typename std::enable_if_t<is_numeric<T>()>* = nullptr>
  std::unique_ptr<column> operator()(strings_column_view const& string_col,
                                     bool ansi_mode,
                                     bool strip,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    if (string_col.size() == 0) {
      return std::make_unique<column>(
        data_type{type_to_id<T>()}, 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);
    }

    rmm::device_uvector<T> data(string_col.size(), stream, mr);
    auto const num_words = bitmask_allocation_size_bytes(string_col.size()) / sizeof(bitmask_type);
    rmm::device_uvector<bitmask_type> null_mask(num_words, stream, mr);

    dim3 const blocks(util::div_rounding_up_unsafe(string_col.size(), detail::NUM_THREADS));
    dim3 const threads{detail::NUM_THREADS};

    detail::string_to_integer_kernel<<<blocks, threads, 0, stream.value()>>>(
      data.data(),
      null_mask.data(),
      string_col.chars_begin(stream),
      string_col.offsets().data<size_type>(),
      string_col.null_mask(),
      string_col.size(),
      ansi_mode,
      strip);

    auto null_count = cudf::null_count(null_mask.data(), 0, string_col.size(), stream);

    auto col = std::make_unique<column>(data_type{type_to_id<T>()},
                                        string_col.size(),
                                        data.release(),
                                        null_mask.release(),
                                        null_count);

    if (ansi_mode) { validate_ansi_column(col->view(), string_col, stream); }

    return col;
  }

  /**
   * @brief stub function for invalid types
   */
  template <typename T, typename std::enable_if_t<!is_numeric<T>()>* = nullptr>
  std::unique_ptr<column> operator()(strings_column_view const& string_col,
                                     bool ansi_mode,
                                     bool strip,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    CUDF_FAIL("Invalid integer column type");
  }
};

struct string_to_decimal_impl {
  /**
   * @brief create deimal column of type T from strings column
   *
   * @tparam T decimal type requested
   * @param dtype data type of result column, includes scale
   * @param precision precision of incoming string data
   * @param string_col strings to convert to decimal
   * @param ansi_mode strict ansi mode checking of incoming data, can throw
   * @param strip remove leading and trailing whitespace.
   * @param stream stream on which to operate
   * @param mr memory resource to use for allocations
   * @return std::unique_ptr<column> decimal column created from strings
   */
  template <typename T, typename std::enable_if_t<is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(data_type dtype,
                                     int32_t precision,
                                     strings_column_view const& string_col,
                                     bool ansi_mode,
                                     bool strip,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    using Type = device_storage_type_t<T>;

    rmm::device_uvector<Type> data(string_col.size(), stream, mr);
    auto const num_words = bitmask_allocation_size_bytes(string_col.size()) / sizeof(bitmask_type);
    rmm::device_uvector<bitmask_type> null_mask(num_words, stream, mr);

    dim3 const blocks(util::div_rounding_up_unsafe(string_col.size(), detail::NUM_THREADS));
    dim3 const threads{detail::NUM_THREADS};

    detail::string_to_decimal_kernel<<<blocks, threads, 0, stream.value()>>>(
      data.data(),
      null_mask.data(),
      string_col.chars_begin(stream),
      string_col.offsets().data<size_type>(),
      string_col.null_mask(),
      string_col.size(),
      dtype.scale(),
      precision,
      strip);

    auto null_count = cudf::null_count(null_mask.data(), 0, string_col.size(), stream);

    auto col = std::make_unique<column>(
      dtype, string_col.size(), data.release(), null_mask.release(), null_count);

    if (ansi_mode) { validate_ansi_column(col->view(), string_col, stream); }

    return col;
  }

  /**
   * @brief stub function for invalid types
   */
  template <typename T, typename std::enable_if_t<!is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(data_type dtype,
                                     int32_t precision,
                                     strings_column_view const& string_col,
                                     bool ansi_mode,
                                     bool strip,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    CUDF_FAIL("Invalid decimal column type");
  }
};

}  // namespace detail

/**
 * @brief Convert a string column into an integer column.
 *
 * @param dtype Type of column to return.
 * @param string_col Incoming string column to convert to integers.
 * @param ansi_mode If true, strict conversion and throws on erorr.
 *                  If false, null invalid entries.
 * @param strip if true leading and trailing white space is ignored.
 * @param stream Stream on which to operate.
 * @param mr Memory resource for returned column
 * @return std::unique_ptr<column> Integer column that was created from string_col.
 */
std::unique_ptr<column> string_to_integer(data_type dtype,
                                          strings_column_view const& string_col,
                                          bool ansi_mode,
                                          bool strip,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  return type_dispatcher(
    dtype, detail::string_to_integer_impl{}, string_col, ansi_mode, strip, stream, mr);
}

/**
 * @brief Convert a string column into an decimal column.
 *
 * @param precision precision of input data
 * @param scale scale of input data
 * @param string_col Incoming string column to convert to decimals.
 * @param ansi_mode If true, strict conversion and throws on erorr.
 *                  If false, null invalid entries.
 * @param strip if true leading and trailing white space is ignored.
 * @param stream Stream on which to operate.
 * @param mr Memory resource for returned column
 * @return std::unique_ptr<column> Decimal column that was created from string_col.
 */
std::unique_ptr<column> string_to_decimal(int32_t precision,
                                          int32_t scale,
                                          strings_column_view const& string_col,
                                          bool ansi_mode,
                                          bool strip,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  data_type dtype = [precision, scale]() {
    if (precision <= cuda::std::numeric_limits<int32_t>::digits10)
      return data_type(type_id::DECIMAL32, scale);
    else if (precision <= cuda::std::numeric_limits<int64_t>::digits10)
      return data_type(type_id::DECIMAL64, scale);
    else if (precision <= cuda::std::numeric_limits<__int128_t>::digits10)
      return data_type(type_id::DECIMAL128, scale);
    else
      CUDF_FAIL("Unable to support decimal with precision " + std::to_string(precision));
  }();

  if (string_col.size() == 0) {
    return std::make_unique<column>(dtype, 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);
  }

  return type_dispatcher(dtype,
                         detail::string_to_decimal_impl{},
                         dtype,
                         precision,
                         string_col,
                         ansi_mode,
                         strip,
                         stream,
                         mr);
}

}  // namespace spark_rapids_jni
