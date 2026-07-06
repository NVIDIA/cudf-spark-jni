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

#include "exception_with_row_index.hpp"
#include "nvtx_ranges.hpp"
#include "round_float.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/type_traits>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace spark_rapids_jni {

inline float __device__ generic_round(float f) { return roundf(f); }
inline double __device__ generic_round(double d) { return ::round(d); }

inline float __device__ generic_round_half_even(float f) { return rintf(f); }
inline double __device__ generic_round_half_even(double d) { return rint(d); }

inline __device__ float generic_modf(float a, float* b) { return modff(a, b); }
inline __device__ double generic_modf(double a, double* b) { return modf(a, b); }

template <typename T>
struct half_up_zero {
  T n;  // unused in the decimal_places = 0 case
  __device__ T operator()(T e) { return generic_round(e); }
};

template <typename T>
struct half_up_positive {
  T n;
  __device__ T operator()(T e)
  {
    T integer_part;
    T const fractional_part = generic_modf(e, &integer_part);
    return integer_part + generic_round(fractional_part * n) / n;
  }
};

template <typename T>
struct half_up_negative {
  T n;
  __device__ T operator()(T e) { return generic_round(e / n) * n; }
};

template <typename T>
struct half_even_zero {
  T n;  // unused in the decimal_places = 0 case
  __device__ T operator()(T e) { return generic_round_half_even(e); }
};

template <typename T>
struct half_even_positive {
  T n;
  __device__ T operator()(T e)
  {
    T integer_part;
    T const fractional_part = generic_modf(e, &integer_part);
    return integer_part + generic_round_half_even(fractional_part * n) / n;
  }
};

template <typename T>
struct half_even_negative {
  T n;
  __device__ T operator()(T e) { return generic_round_half_even(e / n) * n; }
};

template <typename T, template <typename> typename RoundFunctor>
std::unique_ptr<cudf::column> round_with(cudf::column_view const& input,
                                         int32_t decimal_places,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
  requires(std::is_floating_point_v<T>)
{
  using Functor = RoundFunctor<T>;

  auto result = cudf::make_fixed_width_column(input.type(),
                                              input.size(),
                                              cudf::copy_bitmask(input, stream, mr),
                                              input.null_count(),
                                              stream,
                                              mr);

  auto out_view = result->mutable_view();
  T const n     = std::pow(10, std::abs(decimal_places));

  thrust::transform(
    rmm::exec_policy(stream), input.begin<T>(), input.end<T>(), out_view.begin<T>(), Functor{n});

  result->set_null_count(input.null_count());

  return result;
}

struct round_type_dispatcher {
  template <typename T, typename... Args>
  std::unique_ptr<cudf::column> operator()(Args&&...)
    requires(!std::is_floating_point_v<T>)
  { CUDF_FAIL("Type not supported for spark_rapids_jni::round"); }

  template <typename T>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           int32_t decimal_places,
                                           cudf::rounding_method method,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
    requires(std::is_floating_point_v<T>)
  {
    // clang-format off
    switch (method) {
      case cudf::rounding_method::HALF_UP:
        if      (decimal_places == 0) return round_with<T, half_up_zero      >(input, decimal_places, stream, mr);
        else if (decimal_places >  0) return round_with<T, half_up_positive  >(input, decimal_places, stream, mr);
        else                          return round_with<T, half_up_negative  >(input, decimal_places, stream, mr);
      case cudf::rounding_method::HALF_EVEN:
        if      (decimal_places == 0) return round_with<T, half_even_zero    >(input, decimal_places, stream, mr);
        else if (decimal_places >  0) return round_with<T, half_even_positive>(input, decimal_places, stream, mr);
        else                          return round_with<T, half_even_negative>(input, decimal_places, stream, mr);
      default: CUDF_FAIL("Undefined rounding method");
    }
    // clang-format on
  }
};

std::unique_ptr<cudf::column> round(cudf::column_view const& input,
                                    int32_t decimal_places,
                                    cudf::rounding_method method,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  CUDF_EXPECTS(cudf::is_numeric(input.type()) || cudf::is_fixed_point(input.type()),
               "Only integral/floating point/fixed point currently supported.");

  if (!cudf::is_floating_point(input.type())) {
    return cudf::round_decimal(input, decimal_places, method, stream, mr);
  }
  if (input.is_empty()) { return cudf::empty_like(input); }

  return cudf::type_dispatcher(
    input.type(), round_type_dispatcher{}, input, decimal_places, method, stream, mr);
}

namespace {

/**
 * @brief Compute non-overflow range for integral type rounding.
 *
 * Returns [min_safe, max_safe] where values will NOT overflow when rounded.
 *
 * Examples:
 *   - round(127, -2) for ByteType = 100 (OK)
 *   - round(125, -1) for ByteType = 130, which overflows (throws in ANSI mode, wraps in non-ANSI)
 *
 * Statistics: All values safe when scale ≤ threshold:
 * - int8:  scale ≤ -2   (e.g., rounding to hundreds or more)
 * - int16: scale ≤ -4   (e.g., rounding to ten-thousands or more)
 * - int32: scale ≤ -8   (e.g., rounding to hundred-millions or more)
 * - int64: scale ≤ -20  (e.g., rounding to 10^20 or more)
 */
template <typename T>
std::pair<int64_t, int64_t> compute_non_overflow_range(int32_t decimal_places,
                                                       cudf::rounding_method method)
{
  int64_t const max_val = static_cast<int64_t>(std::numeric_limits<T>::max());
  int64_t const min_val = static_cast<int64_t>(std::numeric_limits<T>::min());

  // Early return: for large negative scales, all values round to 0 (safe)
  constexpr size_t type_bytes  = sizeof(T);
  constexpr int32_t safe_scale = (type_bytes == 1)   ? -2
                                 : (type_bytes == 2) ? -4
                                 : (type_bytes == 4) ? -8
                                                     : -20;
  if (decimal_places <= safe_scale) return {min_val, max_val};

  // Check rounding method
  bool half_up = (method == cudf::rounding_method::HALF_UP);
  if (!half_up && method != cudf::rounding_method::HALF_EVEN) {
    CUDF_FAIL("Unsupported rounding method");
  }

  // Handle scale -19 for int64 (divisor would overflow int64)
  // From CSV analysis: scale -19 has special values for int64 only
  if (decimal_places == -19 && sizeof(T) == 8) {
    // int64, scale -19: half = 5*10^18
    return half_up ? std::make_pair(-4999999999999999999LL, 4999999999999999999LL)
                   : std::make_pair(-5000000000000000000LL, 5000000000000000000LL);
  }

  // Compute divisor = 10^(-decimal_places)
  int64_t divisor = 1;
  for (int32_t i = 0; i < -decimal_places; ++i) {
    divisor *= 10;
  }

  int64_t half     = divisor / 2;
  int64_t max_quot = max_val / divisor;
  int64_t min_quot = min_val / divisor;
  int64_t base_pos = max_quot * divisor;
  int64_t base_neg = min_quot * divisor;

  // HALF_EVEN adjustment: extend safe range by 1 when quotient is even
  int32_t even_pos = (!half_up && max_quot % 2 == 0) ? 1 : 0;
  int32_t even_neg = (!half_up && (-min_quot) % 2 == 0) ? 1 : 0;

  // Compute safe range boundaries
  int64_t max_safe =
    (base_pos <= max_val - half - even_pos) ? base_pos + half - 1 + even_pos : max_val;
  int64_t min_safe =
    (base_neg >= min_val + half - 1 + even_neg) ? base_neg - half + 1 - even_neg : min_val;

  return {min_safe, max_safe};
}

/**
 * @brief Find the first row that would overflow when rounded.
 *
 * Returns -1 if no overflow, or the row index of the first overflow.
 * - decimal_places >= 0: No overflow possible (returns identity)
 * - decimal_places < 0: Compute safe range and check values against it
 */
template <typename T>
cudf::size_type find_first_overflow_for_integral_type(cudf::column_view const& input,
                                                      int32_t decimal_places,
                                                      cudf::rounding_method method,
                                                      rmm::cuda_stream_view stream)
{
  static_assert(std::is_integral_v<T>, "T must be an integral type");

  // decimal_places >= 0: no overflow possible (returns identity)
  if (decimal_places >= 0) { return -1; }

  // Compute safe range and check values against it
  auto [min_safe, max_safe] = compute_non_overflow_range<T>(decimal_places, method);

  auto const input_cdv = cudf::column_device_view::create(input, stream);
  auto overflow_iter   = thrust::make_transform_iterator(
    thrust::make_counting_iterator<cudf::size_type>(0),
    [input_view = *input_cdv, min_safe, max_safe] __device__(cudf::size_type idx) -> bool {
      if (input_view.is_null(idx)) { return false; }
      auto val = static_cast<int64_t>(input_view.element<T>(idx));
      return val > max_safe || val < min_safe;
    });

  auto it = thrust::find(
    rmm::exec_policy_nosync(stream), overflow_iter, overflow_iter + input.size(), true);
  return (it != overflow_iter + input.size()) ? cuda::std::distance(overflow_iter, it) : -1;
}

/**
 * @brief Find first overflow row based on type.
 *
 * Uses unified logic for all integral types (int8, int16, int32, int64).
 */
struct find_overflow_dispatcher {
  template <typename T>
  cudf::size_type operator()(cudf::column_view const& input,
                             int32_t decimal_places,
                             cudf::rounding_method method,
                             rmm::cuda_stream_view stream) const
  {
    if constexpr (std::is_integral_v<T>) {
      return find_first_overflow_for_integral_type<T>(input, decimal_places, method, stream);
    } else {
      CUDF_FAIL("Unsupported type for overflow checking");
    }
  }
};

}  // anonymous namespace

std::unique_ptr<cudf::column> round(cudf::column_view const& input,
                                    int32_t decimal_places,
                                    cudf::rounding_method method,
                                    bool is_ansi_mode,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  if (input.is_empty()) { return cudf::empty_like(input); }

  // Check if this is an integral type that needs overflow handling
  bool const is_integral =
    input.type().id() == cudf::type_id::INT8 || input.type().id() == cudf::type_id::INT16 ||
    input.type().id() == cudf::type_id::INT32 || input.type().id() == cudf::type_id::INT64;

  // For non-integral types, positive decimal_places, or non-ANSI mode, use original round function
  // (integral types with scale >= 0 return themselves, no overflow possible)
  // (non-ANSI mode allows overflow)
  if (!is_integral || decimal_places >= 0 || !is_ansi_mode) {
    return spark_rapids_jni::round(input, decimal_places, method, stream, mr);
  }

  // ANSI mode: Check for overflow before throwing exception
  // First, check if overflow would occur
  auto const overflow_row_index = cudf::type_dispatcher(
    input.type(), find_overflow_dispatcher{}, input, decimal_places, method, stream);

  if (overflow_row_index >= 0) {
    // Overflow detected at row index, throw exception
    throw exception_with_row_index(overflow_row_index);
  }

  // No overflow, perform normal rounding
  return spark_rapids_jni::round(input, decimal_places, method, stream, mr);
}

}  // namespace spark_rapids_jni
