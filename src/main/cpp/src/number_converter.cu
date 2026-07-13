/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

#include "number_converter.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <cuda/std/cmath>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <thrust/count.h>
#include <thrust/for_each.h>

/**
 * This file is ported from Spark 3.5.0 org.apache.spark.sql.catalyst.util.NumberConverter
 */
namespace spark_rapids_jni {

namespace {

// The range of bases is [2, 36], which is from Spark.
constexpr int MIN_BASE = 2;
// The max base is 36, which means the max char is 'Z' or 'z', e.g.:
// 0-9, A-Z, a-z are all valid digits in base 36.
// 0-9, A-Y, a-y are all valid digits in base 35.
// ...
// 0-9, A-F, a-f are all valid digits in base 16.
// ...
// 0-1 are all valid digits in base 2.
constexpr int MAX_BASE = 36;

enum class ansi_mode {
  OFF = 0,
  ON  = 1,
};

CUDF_HOST_DEVICE bool is_invalid_base_range(int from_base, int to_base)
{
  return from_base < MIN_BASE || from_base > MAX_BASE || cuda::std::abs(to_base) < MIN_BASE ||
         cuda::std::abs(to_base) > MAX_BASE;
}

/**
 * @brief Trims space characters (ASCII 32)
 * @return The first non-space index and last non-space index pair
 */
__device__ cuda::std::pair<int, int> trim(char const* ptr, int len)
{
  int first = 0;
  int last  = len - 1;
  while (first < len && ptr[first] == ' ') {
    ++first;
  }
  while (last > first && ptr[last] == ' ') {
    --last;
  }
  return {first, last};
}

/**
 * @brief Convert byte value to char, input byte_value MUST be valid byte values for bases [2, 36].
 * E.g.:
 *   0 => '0'
 *   ...
 *   9 => '9'
 *   10 => 'A'
 *   ...
 *   15 => 'F'
 *   ...
 *   35 => 'Z'
 */
__device__ char byte_to_char(int byte_value)
{
  if (byte_value >= 0 && byte_value <= 9) {
    return static_cast<char>('0' + (byte_value - 0));
  } else if (byte_value >= 10 && byte_value <= 35) {
    return static_cast<char>('A' + (byte_value - 10));
  } else {
    cudf_assert(false);
    return 0;
  }
}

/**
 * @brief Convert char to byte value, it's a reverse of byte_to_char; If char is invalid
 * considering 'base', return -1.
 * E.g.: base = 36
 *   '0' => 0
 *   ...
 *   '9' => 9
 *   'A' or 'a' => 10
 *   ...
 *   'F' or 'f' => 15
 *   ...
 *   'Z' or 'z' => 35
 * E.g.: base = 8
 *   '8' => -1 // invalid char considering base
 */
__device__ int char_to_byte(char c, int base)
{
  if (c >= '0' && c <= '9' && c - '0' < base) {
    return c - '0';
  } else if (c >= 'A' && c <= 'Z' && c - 'A' + 10 < base) {
    return c - 'A' + 10;
  } else if (c >= 'a' && c <= 'z' && c - 'a' + 10 < base) {
    return c - 'a' + 10;
  } else {
    return -1;
  }
}

/**
 * @brief The types of result of convert function
 */
enum class result_type : int32_t { SUCCESS, OVERFLOW, NULL_VALUE };

/**
 * @brief Convert numbers in string representation between different number bases. It's a two
 phases
 * function, first phase calculates the length of the converted string, second phase converts the
 * string to the target base. It's the fist phase when `out` is nullptr and it's the second phase
 * when `out` is not nullptr. It first trims space characters (ASCII 32) from both sides. If
 * toBase > 0 the result is unsigned, otherwise it is signed.
 *
 * This logic is borrowed from org.apache.spark.sql.catalyst.util.NumberConverter
 *
 * @return result_type and length pair
 *
 */
__device__ cuda::std::pair<result_type, int> convert(
  char const* ptr, int len, int from_base, int to_base, char* out, int out_len, ansi_mode ansi_type)
{
  // trim spaces
  auto [first, last] = trim(ptr, len);
  if (last - first < 0) {
    // return null if the trimmed string is empty
    return {result_type::NULL_VALUE, 0};
  }

  // handle sign
  bool negative = false;
  if (ptr[first] == '-') {
    negative = true;
    ++first;
  }

  // convert to long value
  int64_t v  = 0;
  auto bound = static_cast<unsigned long>(-1L - from_base) / from_base;
  for (int char_idx = first; char_idx <= last; ++char_idx) {
    int b = char_to_byte(ptr[char_idx], from_base);
    if (b < 0) {
      // meet invalid char, ignore the suffix starting there
      break;
    }

    if (v < 0) {
      // if v < 0, which mean its sign(first) bit is 1, so v * base will cause
      // overflow since base is greater than 2 and v is considered as unsigned long
      if (ansi_type == ansi_mode::ON) {
        // overflow for ansi mode, which means throw exception
        return {result_type::OVERFLOW, 0};
      } else {
        // overflow for non-ansi mode, use -1
        v = -1L;
        break;
      }
    }

    // check if v is greater or equal than bound
    // if v is greater than bound, v * base + base may cause overflow.
    if (v >= bound) {
      if (static_cast<unsigned long>(-1L - b) / from_base < v) {
        // if v > bound, which mean its sign(first) bit is 1, so v * base will cause
        // overflow since base is greater than 2 and v is considered as unsigned long
        if (ansi_type == ansi_mode::ON) {
          // overflow for ansi mode, which means throw exception
          return {result_type::OVERFLOW, 0};
        } else {
          // overflow for non-ansi mode, use -1
          v = -1L;
          break;
        }
      }
    }

    v = v * from_base + b;
  }

  if (negative && to_base > 0) {
    if (v < 0) {
      v = -1;
    } else {
      v = -v;
    }
  }
  if (to_base < 0 && v < 0) {
    v        = -v;
    negative = true;
  }

  // write out string representation
  auto to_base_abs = cuda::std::abs(to_base);
  int out_idx      = out_len - 1;
  uint64_t uv      = static_cast<uint64_t>(v);

  if (uv == 0) {
    if (out != nullptr) { out[out_idx] = '0'; }
    --out_idx;
  } else {
    while (uv != 0) {
      auto remainder = uv % to_base_abs;
      if (out != nullptr) { out[out_idx] = byte_to_char(remainder); }
      --out_idx;
      uv /= to_base_abs;
    }
  }

  // write out sign
  if (negative && to_base < 0) {
    if (out != nullptr) { out[out_idx] = '-'; }
    --out_idx;
  }
  return {result_type::SUCCESS, out_len - 1 - out_idx};
}

struct str_iter {
  cudf::column_device_view d_strings;

  __device__ bool is_null(cudf::size_type idx) const { return d_strings.is_null(idx); }

  __device__ cudf::string_view element(cudf::size_type idx) const
  { return d_strings.element<cudf::string_view>(idx); }
};

struct const_str {
  cudf::string_view str;

  __device__ bool is_null(cudf::size_type idx) const { return false; }

  __device__ cudf::string_view element(cudf::size_type idx) const { return str; }
};

struct base_iter {
  cudf::column_device_view d_bases;

  __device__ bool is_null(cudf::size_type idx) const { return d_bases.is_null(idx); }

  __device__ int get(cudf::size_type idx) const { return d_bases.element<int>(idx); }
};

struct const_base {
  int base;

  CUDF_HOST_DEVICE bool is_null(cudf::size_type idx) const { return false; }

  CUDF_HOST_DEVICE int get(cudf::size_type) const { return base; }
};

template <typename STR_ITERATOR, typename FROM_BASE_ITERATOR, typename TO_BASE_ITERATOR>
struct convert_fn {
  static constexpr bool IS_CONST_BASES = cuda::std::is_same_v<const_base, FROM_BASE_ITERATOR> &&
                                         cuda::std::is_same_v<const_base, TO_BASE_ITERATOR>;

  STR_ITERATOR input;
  FROM_BASE_ITERATOR from_base_iter;
  TO_BASE_ITERATOR to_base_iter;
  bool* out_mask;

  // For the first phase: calculate the d_sizes and out_mask of the converted strings
  cudf::size_type* d_sizes{};

  // if d_chars is nullptr, only compute d_sizes and out_mask
  // if d_chars is not nullptr, convert the strings to the target base
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(int idx)
  {
    if (!d_chars) {
      // first phase, set null/length

      // check input null
      if (input.is_null(idx)) {
        // if base is invalid, set null and zero length
        d_sizes[idx]  = 0;
        out_mask[idx] = false;
        return;
      }

      // check base range
      if (!IS_CONST_BASES) {
        if (from_base_iter.is_null(idx) || to_base_iter.is_null(idx)) {
          // if base is invalid, set null and zero length
          d_sizes[idx]  = 0;
          out_mask[idx] = false;
          return;
        }

        int from_base = from_base_iter.get(idx);
        int to_base   = to_base_iter.get(idx);

        // if not const bases, check bases for each row
        if (is_invalid_base_range(from_base, to_base)) {
          // if base is invalid, return all nulls
          // first phase
          d_sizes[idx]  = 0;
          out_mask[idx] = false;
          return;
        }
      }

      auto str      = input.element(idx);
      int from_base = from_base_iter.get(idx);
      int to_base   = to_base_iter.get(idx);
      // first phase, set null/length
      auto [ret_type, len] = convert(str.data(),
                                     str.length(),
                                     from_base,
                                     to_base,
                                     /*compute len*/ nullptr,
                                     /*dummy value*/ -1,
                                     ansi_mode::OFF);
      d_sizes[idx]         = len;
      out_mask[idx]        = (ret_type != result_type::NULL_VALUE);
    } else {
      // second phase, convert the string
      int len = d_offsets[idx + 1] - d_offsets[idx];
      if (len > 0) {
        auto str      = input.element(idx);
        int from_base = from_base_iter.get(idx);
        int to_base   = to_base_iter.get(idx);
        convert(str.data(),
                str.length(),
                from_base,
                to_base,
                /* convert */ d_chars + d_offsets[idx],
                /* len */ len,
                ansi_mode::OFF);
      }
    }
  }
};

template <typename STR_ITERATOR, typename FROM_BASE_ITERATOR, typename TO_BASE_ITERATOR>
std::unique_ptr<cudf::column> convert_impl(cudf::size_type num_rows,
                                           STR_ITERATOR input,
                                           FROM_BASE_ITERATOR from_base,
                                           TO_BASE_ITERATOR to_base,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  static constexpr bool IS_CONST_BASES = cuda::std::is_same_v<const_base, FROM_BASE_ITERATOR> &&
                                         cuda::std::is_same_v<const_base, TO_BASE_ITERATOR>;
  if (num_rows == 0) { return cudf::make_empty_column(cudf::type_id::STRING); }

  // check base is in range: [2, 36]
  if (IS_CONST_BASES) {
    // if const bases, check bases only once
    if (is_invalid_base_range(from_base.get(0), to_base.get(0))) {
      cudf::string_scalar null_s("", false);
      // if base is invalid, return all nulls
      return cudf::make_column_from_scalar(null_s, num_rows, stream, mr);
    }
  }

  auto out_mask =
    rmm::device_uvector<bool>(num_rows, stream, cudf::get_current_device_resource_ref());

  auto [offsets, chars] = cudf::strings::detail::make_strings_children(
    convert_fn<STR_ITERATOR, FROM_BASE_ITERATOR, TO_BASE_ITERATOR>{
      input, from_base, to_base, out_mask.data()},
    num_rows,
    stream,
    mr);

  // make null mask and null count
  auto [null_mask, null_count] =
    cudf::bools_to_mask(cudf::device_span<bool const>(out_mask), stream, mr);

  return cudf::make_strings_column(
    num_rows,
    std::move(offsets),
    chars.release(),
    null_count,
    null_count ? std::move(*null_mask.release()) : rmm::device_buffer{});
}

/**
 * @brief Check if the convert function will cause overflow
 * @return true if overflow, false otherwise
 */
__device__ bool is_convert_overflow(char const* ptr, int len, int from_base, int to_base)
{
  auto pair = convert(ptr, len, from_base, to_base, nullptr, -1, ansi_mode::ON);
  return pair.first == result_type::OVERFLOW;
}

template <typename STR_ITERATOR, typename FROM_BASE_ITERATOR, typename TO_BASE_ITERATOR>
struct is_overflow_fn {
  static constexpr bool IS_CONST_BASES = cuda::std::is_same_v<const_base, FROM_BASE_ITERATOR> &&
                                         cuda::std::is_same_v<const_base, TO_BASE_ITERATOR>;
  STR_ITERATOR input;
  FROM_BASE_ITERATOR from_base_iter;
  TO_BASE_ITERATOR to_base_iter;

  __device__ bool operator()(int idx)
  {
    if (from_base_iter.is_null(idx) || to_base_iter.is_null(idx)) { return false; }

    int from_base = from_base_iter.get(idx);
    int to_base   = to_base_iter.get(idx);

    if (!IS_CONST_BASES) {
      // if not const bases, check bases for each row
      if (is_invalid_base_range(from_base, to_base)) {
        // if base is invalid, return all nulls thus no overflow
        return false;
      }
    }

    if (input.is_null(idx)) {
      return false;
    } else {
      auto str = input.element(idx);
      return is_convert_overflow(str.data(), str.length(), from_base, to_base);
    }
  }
};

template <typename STR_ITERATOR, typename FROM_BASE_ITERATOR, typename TO_BASE_ITERATOR>
bool is_convert_overflow_impl(cudf::size_type num_rows,
                              STR_ITERATOR input,
                              FROM_BASE_ITERATOR from_base,
                              TO_BASE_ITERATOR to_base,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  static constexpr bool IS_CONST_BASES = cuda::std::is_same_v<const_base, FROM_BASE_ITERATOR> &&
                                         cuda::std::is_same_v<const_base, TO_BASE_ITERATOR>;
  if (IS_CONST_BASES) {
    // const bases, check bases only once
    if (is_invalid_base_range(from_base.get(0), to_base.get(0))) {
      // if base is invalid, return all nulls thus no overflow
      return false;
    }
  }
  auto num_overflow = thrust::count_if(
    rmm::exec_policy_nosync(stream),
    thrust::counting_iterator<cudf::size_type>(0),
    thrust::counting_iterator<cudf::size_type>(num_rows),
    is_overflow_fn<STR_ITERATOR, FROM_BASE_ITERATOR, TO_BASE_ITERATOR>{input, from_base, to_base});
  return num_overflow > 0;
}

bool is_cv(convert_number_t const& t) { return std::holds_alternative<cudf::column_view>(t); }

void check_types(convert_number_t const& input,
                 convert_number_t const& from_base,
                 convert_number_t const& to_base,
                 rmm::cuda_stream_view stream)
{
  // check input type
  if (is_cv(input)) {
    auto const input_cv = std::get<cudf::column_view>(input);
    CUDF_EXPECTS(input_cv.type().id() == cudf::type_id::STRING,
                 "Input column must be of type STRING");
  } else {
    auto const input_scalar = std::get<cudf::string_scalar>(input);
    CUDF_EXPECTS(input_scalar.type().id() == cudf::type_id::STRING,
                 "Input scalar must be of type STRING");
    CUDF_EXPECTS(input_scalar.is_valid(stream), "Input scalar must be valid");
  }

  // check from_base type
  if (is_cv(from_base)) {
    auto const from_base_cv = std::get<cudf::column_view>(from_base);
    CUDF_EXPECTS(from_base_cv.type().id() == cudf::type_id::INT32,
                 "From base column must be of type INT32");
  }

  // check to_base type
  if (is_cv(to_base)) {
    auto const to_base_cv = std::get<cudf::column_view>(to_base);
    CUDF_EXPECTS(to_base_cv.type().id() == cudf::type_id::INT32,
                 "To base column must be of type INT32");
  }
}

}  // anonymous namespace

std::unique_ptr<cudf::column> convert(convert_number_t const& input,
                                      convert_number_t const& from_base,
                                      convert_number_t const& to_base,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  check_types(input, from_base, to_base, stream);

  if (is_cv(input)) {
    auto const input_cv = std::get<cudf::column_view>(input);
    auto const d_strs   = cudf::column_device_view::create(input_cv, stream);

    if (is_cv(from_base)) {
      auto const from_base_cv = std::get<cudf::column_view>(from_base);
      auto const d_from_bases = cudf::column_device_view::create(from_base_cv, stream);

      if (is_cv(to_base)) {
        auto const to_base_cv = std::get<cudf::column_view>(to_base);
        auto const d_to_bases = cudf::column_device_view::create(to_base_cv, stream);
        // input is string cv, from base is cv, to base is cv
        return convert_impl<str_iter, base_iter, base_iter>(input_cv.size(),
                                                            str_iter{*d_strs},
                                                            base_iter{*d_from_bases},
                                                            base_iter{*d_to_bases},
                                                            stream,
                                                            mr);
      } else {
        auto const to_base_scalar = std::get<int>(to_base);
        // input is string cv, from base is cv, to base is int scalar
        return convert_impl<str_iter, base_iter, const_base>(input_cv.size(),
                                                             str_iter{*d_strs},
                                                             base_iter{*d_from_bases},
                                                             const_base{to_base_scalar},
                                                             stream,
                                                             mr);
      }
    } else {
      auto const from_base_scalar = std::get<int>(from_base);
      if (is_cv(to_base)) {
        auto const to_base_cv = std::get<cudf::column_view>(to_base);
        auto const d_to_bases = cudf::column_device_view::create(to_base_cv, stream);
        // input is string cv, from base is int scalar, to base is cv
        return convert_impl<str_iter, const_base, base_iter>(input_cv.size(),
                                                             str_iter{*d_strs},
                                                             const_base{from_base_scalar},
                                                             base_iter{*d_to_bases},
                                                             stream,
                                                             mr);
      } else {
        auto const& to_base_scalar = std::get<int>(to_base);
        // input is string cv, from base is int scalar, to base is int scalar
        return convert_impl<str_iter, const_base, const_base>(input_cv.size(),
                                                              str_iter{*d_strs},
                                                              const_base{from_base_scalar},
                                                              const_base{to_base_scalar},
                                                              stream,
                                                              mr);
      }
    }
  } else {
    auto const& input_scalar = std::get<cudf::string_scalar>(input);
    auto str_scalar          = input_scalar.value(stream);

    if (is_cv(from_base)) {
      auto const from_base_cv = std::get<cudf::column_view>(from_base);
      auto const d_from_bases = cudf::column_device_view::create(from_base_cv, stream);

      if (is_cv(to_base)) {
        auto const to_base_cv = std::get<cudf::column_view>(to_base);
        auto const d_to_bases = cudf::column_device_view::create(to_base_cv, stream);
        // input is string scalar, from base is cv, to base is cv
        return convert_impl<const_str, base_iter, base_iter>(from_base_cv.size(),
                                                             const_str{str_scalar},
                                                             base_iter{*d_from_bases},
                                                             base_iter{*d_to_bases},
                                                             stream,
                                                             mr);
      } else {
        auto const to_base_scalar = std::get<int>(to_base);
        // input is string scalar, from base is cv, to base is int scalar
        return convert_impl<const_str, base_iter, const_base>(from_base_cv.size(),
                                                              const_str{str_scalar},
                                                              base_iter{*d_from_bases},
                                                              const_base{to_base_scalar},
                                                              stream,
                                                              mr);
      }
    } else {
      auto const from_base_scalar = std::get<int>(from_base);
      if (is_cv(to_base)) {
        auto const to_base_cv = std::get<cudf::column_view>(to_base);
        auto const d_to_bases = cudf::column_device_view::create(to_base_cv, stream);
        // input is string scalar, from base is int scalar, to base is cv
        return convert_impl<const_str, const_base, base_iter>(to_base_cv.size(),
                                                              const_str{str_scalar},
                                                              const_base{from_base_scalar},
                                                              base_iter{*d_to_bases},
                                                              stream,
                                                              mr);
      } else {
        // MUST not be here
        CUDF_FAIL("Input is string scalar, from base is int scalar, to base is int scalar");
      }
    }
  }
}

bool is_convert_overflow(convert_number_t const& input,
                         convert_number_t const& from_base,
                         convert_number_t const& to_base,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
{
  check_types(input, from_base, to_base, stream);

  if (is_cv(input)) {
    auto const input_cv = std::get<cudf::column_view>(input);
    auto const d_strs   = cudf::column_device_view::create(input_cv, stream);

    if (is_cv(from_base)) {
      auto const from_base_cv = std::get<cudf::column_view>(from_base);
      auto const d_from_bases = cudf::column_device_view::create(from_base_cv, stream);

      if (is_cv(to_base)) {
        auto const to_base_cv = std::get<cudf::column_view>(to_base);
        auto const d_to_bases = cudf::column_device_view::create(to_base_cv, stream);
        // input is string cv, from base is cv, to base is cv
        return is_convert_overflow_impl<str_iter, base_iter, base_iter>(input_cv.size(),
                                                                        str_iter{*d_strs},
                                                                        base_iter{*d_from_bases},
                                                                        base_iter{*d_to_bases},
                                                                        stream,
                                                                        mr);
      } else {
        auto const& to_base_scalar = std::get<int>(to_base);
        // input is string cv, from base is cv, to base is int scalar
        return is_convert_overflow_impl<str_iter, base_iter, const_base>(input_cv.size(),
                                                                         str_iter{*d_strs},
                                                                         base_iter{*d_from_bases},
                                                                         const_base{to_base_scalar},
                                                                         stream,
                                                                         mr);
      }
    } else {
      auto const& from_base_scalar = std::get<int>(from_base);
      if (is_cv(to_base)) {
        auto const to_base_cv = std::get<cudf::column_view>(to_base);
        auto const d_to_bases = cudf::column_device_view::create(to_base_cv, stream);
        // input is string cv, from base is int scalar, to base is cv
        return is_convert_overflow_impl<str_iter, const_base, base_iter>(
          input_cv.size(),
          str_iter{*d_strs},
          const_base{from_base_scalar},
          base_iter{*d_to_bases},
          stream,
          mr);
      } else {
        auto const to_base_scalar = std::get<int>(to_base);
        // input is string cv, from base is int scalar, to base is int scalar
        return is_convert_overflow_impl<str_iter, const_base, const_base>(
          input_cv.size(),
          str_iter{*d_strs},
          const_base{from_base_scalar},
          const_base{to_base_scalar},
          stream,
          mr);
      }
    }
  } else {
    auto const& input_scalar = std::get<cudf::string_scalar>(input);
    auto str_scalar          = input_scalar.value(stream);

    if (is_cv(from_base)) {
      auto const from_base_cv = std::get<cudf::column_view>(from_base);
      auto const d_from_bases = cudf::column_device_view::create(from_base_cv, stream);

      if (is_cv(to_base)) {
        auto const to_base_cv = std::get<cudf::column_view>(to_base);
        auto const d_to_bases = cudf::column_device_view::create(to_base_cv, stream);
        // input is string scalar, from base is cv, to base is cv
        return is_convert_overflow_impl<const_str, base_iter, base_iter>(from_base_cv.size(),
                                                                         const_str{str_scalar},
                                                                         base_iter{*d_from_bases},
                                                                         base_iter{*d_to_bases},
                                                                         stream,
                                                                         mr);
      } else {
        auto const to_base_scalar = std::get<int>(to_base);
        // input is string scalar, from base is cv, to base is int scalar
        return is_convert_overflow_impl<const_str, base_iter, const_base>(
          from_base_cv.size(),
          const_str{str_scalar},
          base_iter{*d_from_bases},
          const_base{to_base_scalar},
          stream,
          mr);
      }
    } else {
      auto const from_base_scalar = std::get<int>(from_base);
      if (is_cv(to_base)) {
        auto const to_base_cv = std::get<cudf::column_view>(to_base);
        auto const d_to_bases = cudf::column_device_view::create(to_base_cv, stream);
        // input is string scalar, from base is int scalar, to base is cv
        return is_convert_overflow_impl<const_str, const_base, base_iter>(
          to_base_cv.size(),
          const_str{str_scalar},
          const_base{from_base_scalar},
          base_iter{*d_to_bases},
          stream,
          mr);
      } else {
        // MUST not be here
        CUDF_FAIL("Input is string scalar, from base is int scalar, to base is int scalar");
      }
    }
  }
}

}  // namespace spark_rapids_jni
