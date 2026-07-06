/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.
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
#include "nvtx_ranges.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/convert/int_to_string.cuh>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/algorithm>
#include <cuda/std/climits>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/generate.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

using namespace cudf;

namespace spark_rapids_jni {

namespace detail {
namespace {

template <typename DecimalType>
struct decimal_to_non_ansi_string_fn {
  column_device_view d_decimals;
  cudf::size_type* d_sizes;
  char* d_chars;
  cudf::detail::input_offsetalator d_offsets;

  /**
   * @brief Calculates the size of the string required to convert the element, in base-10 format.
   *
   * @note This code does not properly handle a max negative decimal value and will overflow. This
   * isn't an issue here because Spark will not use the full range of values and will never cause
   * this issue.
   *
   * Output format is [-]integer.fraction
   */
  __device__ int32_t compute_output_size(DecimalType value)
  {
    auto const scale = d_decimals.type().scale();

    auto const abs_value         = numeric::detail::abs(value);
    auto const abs_value_digits  = strings::detail::count_digits(abs_value);
    auto const adjusted_exponent = scale + (abs_value_digits - 1);

    if (scale == 0) {
      return static_cast<int32_t>(value < 0) +  // sign if negative
             abs_value_digits;                  // integer
    } else if (scale < 0 && adjusted_exponent >= -6) {
      auto const exp_ten   = numeric::detail::exp10<DecimalType>(-scale);
      auto const fraction  = strings::detail::count_digits(abs_value % exp_ten);
      auto const num_zeros = cuda::std::max(0, (-scale - fraction));
      return static_cast<int32_t>(value < 0) +                     // sign if negative
             strings::detail::count_digits(abs_value / exp_ten) +  // integer
             1 +                                                   // decimal point
             num_zeros +                                           // zero padding
             fraction;                                             // size of fraction
    } else {
      // positive scale or adjusted exponent < -6 means scientific notation
      auto const extra_digits = abs_value_digits > 1 ? 3 : 2;
      return static_cast<int32_t>(value < 0) +  // sign if negative
             abs_value_digits +                 // number of digits
             extra_digits +                     // decimal point if exists, E, +/-
             strings::detail::count_digits(
               numeric::detail::abs(adjusted_exponent));  // exponent portion
    }
  }

  /**
   * @brief Converts a decimal element into a string.
   *
   * @note This code does not properly handle a max negative decimal value and will overflow. This
   * isn't an issue here because Spark will not use the full range of values and will never cause
   * this issue.
   *
   * @note Code follows the Java method of decimal to string outlined here:
   * https://docs.oracle.com/javase/8/docs/api/java/math/BigDecimal.html#toString--
   *
   * The value is converted into base-10 digits [0-9]
   * plus the decimal point and a negative sign prefix.
   */
  __device__ void decimal_to_non_ansi_string(size_type idx)
  {
    auto const value = d_decimals.element<DecimalType>(idx);
    auto const scale = d_decimals.type().scale();
    char* d_buffer   = d_chars + d_offsets[idx];

    auto const abs_value         = numeric::detail::abs(value);
    auto const abs_value_digits  = strings::detail::count_digits(abs_value);
    auto const adjusted_exponent = scale + (abs_value_digits - 1);

    if (value < 0) *d_buffer++ = '-';  // add sign

    if (scale <= 0 && adjusted_exponent >= -6) {
      auto const exp_ten = numeric::detail::exp10<DecimalType>(-scale);
      auto const num_zeros =
        cuda::std::max(0, (-scale - strings::detail::count_digits(abs_value % exp_ten)));
      d_buffer +=
        strings::detail::integer_to_string(abs_value / exp_ten, d_buffer);  // add the integer part
      if (scale != 0) {
        *d_buffer++ = '.';  // add decimal point

        thrust::generate_n(thrust::seq, d_buffer, num_zeros, []() { return '0'; });  // add zeros
        d_buffer += num_zeros;

        strings::detail::integer_to_string(abs_value % exp_ten, d_buffer);  // add the fraction part
      }
    } else {
      // positive scale or adjusted exponent < -6 means scientific notation
      if (abs_value_digits > 1) {
        auto const digits_after_decimal = abs_value_digits - 1;
        auto const exp_ten              = numeric::detail::exp10<DecimalType>(digits_after_decimal);
        auto const num_zeros            = cuda::std::max(
          0, (digits_after_decimal - strings::detail::count_digits(abs_value % exp_ten)));
        d_buffer +=
          strings::detail::integer_to_string(abs_value / exp_ten, d_buffer);  // add integer part
        *d_buffer++ = '.';                                                    // add decimal point

        thrust::generate_n(thrust::seq, d_buffer, num_zeros, []() { return '0'; });  // add zeros
        d_buffer += num_zeros;

        d_buffer +=
          strings::detail::integer_to_string(abs_value % exp_ten, d_buffer);  // add fraction part
      } else {
        d_buffer += strings::detail::integer_to_string(abs_value, d_buffer);  // add single digit
      }
      *d_buffer++ = 'E';
      if (adjusted_exponent >= 0) *d_buffer++ = '+';  // minus sign added by integer to string
      strings::detail::integer_to_string(adjusted_exponent, d_buffer);
    }
  }

  __device__ void operator()(size_type idx)
  {
    if (d_decimals.is_null(idx)) {
      if (d_chars == nullptr) { d_sizes[idx] = 0; }
      return;
    }
    if (d_chars != nullptr) {
      decimal_to_non_ansi_string(idx);
    } else {
      d_sizes[idx] = compute_output_size(d_decimals.element<DecimalType>(idx));
    }
  }
};

/**
 * @brief The dispatcher functor for converting fixed-point values into strings.
 */
struct dispatch_decimal_to_non_ansi_string_fn {
  template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    using DecimalType = device_storage_type_t<T>;  // underlying value type

    auto const d_column = column_device_view::create(input, stream);

    auto [offsets, chars] = strings::detail::make_strings_children(
      decimal_to_non_ansi_string_fn<DecimalType>{*d_column}, input.size(), stream, mr);

    return make_strings_column(input.size(),
                               std::move(offsets),
                               chars.release(),
                               input.null_count(),
                               cudf::copy_bitmask(input, stream, mr));
  }

  template <typename T, std::enable_if_t<not cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const&,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref) const
  { CUDF_FAIL("Values for decimal_to_non_ansi_string function must be a decimal type."); }
};

}  // namespace

std::unique_ptr<column> decimal_to_non_ansi_string(column_view const& input,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return make_empty_column(type_id::STRING);
  return type_dispatcher(input.type(), dispatch_decimal_to_non_ansi_string_fn{}, input, stream, mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> decimal_to_non_ansi_string(column_view const& input,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return detail::decimal_to_non_ansi_string(input, stream, mr);
}

}  // namespace spark_rapids_jni
