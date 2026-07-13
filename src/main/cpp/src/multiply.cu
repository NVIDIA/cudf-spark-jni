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

#include "exception_with_row_index_utilities.hpp"
#include "multiply.hpp"
#include "utilities.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/transform.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace spark_rapids_jni {

namespace {

void check_multiply_inputs(cudf::data_type left_type,
                           cudf::data_type right_type,
                           cudf::size_type left_size,
                           cudf::size_type right_size,
                           bool is_ansi_mode,
                           bool is_try_mode)
{
  CUDF_EXPECTS(left_type == right_type, "Input columns must have the same data type");
  CUDF_EXPECTS(is_basic_spark_numeric(left_type), "Unsupported data type for multiplication.");
  CUDF_EXPECTS(left_size == right_size, "Input columns must have the same size");
  CUDF_EXPECTS(!(is_ansi_mode && is_try_mode),
               "Cannot enable both ANSI mode and TRY mode at the same time");
}

/**
 * @brief Multiply two int8_t or int16_t values with overflow check.
 * Set `valid` to false if overflow occurs.
 */
template <typename T,
          CUDF_ENABLE_IF(cuda::std::is_same_v<T, int8_t> || cuda::std::is_same_v<T, int16_t>)>
__device__ void multiply(T x, T y, bool check_overflow, bool* valid, T* result)
{
  // when int8_t * int8_t or int16_t * int16_t, the result is promoted to int type.
  int32_t r = x * y;
  if (check_overflow &&
      (r < cuda::std::numeric_limits<T>::min() || r > cuda::std::numeric_limits<T>::max())) {
    *valid = false;  // overflow in ansi mode
  } else {
    *valid  = true;
    *result = static_cast<T>(r);
  }
}

/**
 * @brief Multiply two int32_t values with overflow check.
 * Set `valid` to false if overflow occurs.
 */
template <typename T, CUDF_ENABLE_IF(cuda::std::is_same_v<T, int32_t>)>
__device__ void multiply(T x, T y, bool check_overflow, bool* valid, T* result)
{
  if (check_overflow) {
    int64_t r = static_cast<int64_t>(x) * static_cast<int64_t>(y);
    if (auto r_cast = static_cast<int>(r); static_cast<long>(r_cast) == r) {
      *valid  = true;
      *result = r_cast;
    } else {
      *valid = false;  // overflow in ansi mode
    }
  } else {
    *valid  = true;
    *result = x * y;
  }
}

__device__ bool is_multiply_overflow(int64_t a, int64_t b)
{
  constexpr auto int64_min = cuda::std::numeric_limits<int64_t>::min();
  constexpr auto int64_max = cuda::std::numeric_limits<int64_t>::max();

  if (a > 0 && b > 0 && a > int64_max / b) return true;  // Positive overflow
  if (a < 0 && b < 0 && a < int64_max / b) return true;  // Negative overflow
  if (a > 0 && b < 0 && b < int64_min / a) return true;  // Mixed sign overflow
  if (a < 0 && b > 0 && a < int64_min / b) return true;  // Mixed sign overflow

  return false;
}

/**
 * @brief Multiply two int64_t values with overflow check.
 * Set `valid` to false if overflow occurs.
 */
template <typename T, CUDF_ENABLE_IF(cuda::std::is_same_v<T, int64_t>)>
__device__ void multiply(T x, T y, bool check_overflow, bool* valid, T* result)
{
  if (check_overflow && is_multiply_overflow(x, y)) {
    *valid = false;  // overflow
    return;
  }

  *valid  = true;
  *result = x * y;
}

/**
 * @brief Multiply two float values.
 * `check_overflow` is ignored for float multiplication.
 */
template <typename T,
          CUDF_ENABLE_IF(cuda::std::is_same_v<T, float> || cuda::std::is_same_v<T, double>)>
__device__ void multiply(T x, T y, bool check_overflow, bool* valid, T* result)
{
  *valid  = true;
  *result = x * y;
}

/**
 * @brief Functor to perform multiplication on two accessors,
 * accessor is type of `column_accessor` or `numeric_scalar_accessor`
 */
template <typename T, typename LEFT_ACCESSOR, typename RIGHT_ACCESSOR>
struct multiply_fn {
  LEFT_ACCESSOR left_accessor;
  RIGHT_ACCESSOR right_accessor;
  bool check_overflow;
  T* results;
  bool* validity;

  __device__ void operator()(int row_idx) const
  {
    auto const [left_value, left_valid]   = left_accessor[row_idx];
    auto const [right_value, right_valid] = right_accessor[row_idx];
    if (!left_valid || !right_valid) {
      validity[row_idx] = false;
      return;
    }

    multiply<T>(left_value, right_value, check_overflow, &validity[row_idx], &results[row_idx]);
  }
};

/**
 * @brief Functor for multiplication when both inputs are valid
 * and no overflow check is needed.
 */
template <typename T, typename LEFT_ACCESSOR, typename RIGHT_ACCESSOR>
struct multiply_no_validity_fn {
  LEFT_ACCESSOR left_accessor;
  RIGHT_ACCESSOR right_accessor;
  T* results;

  __device__ void operator()(int row_idx) const
  {
    [[maybe_unused]] auto const [left_value, left_valid]   = left_accessor[row_idx];
    [[maybe_unused]] auto const [right_value, right_valid] = right_accessor[row_idx];
    results[row_idx]                                       = left_value * right_value;
  }
};

template <typename T, typename LEFT_ACCESSOR, typename RIGHT_ACCESSOR>
std::unique_ptr<cudf::column> multiply_impl(cudf::data_type type,
                                            cudf::size_type num_rows,
                                            LEFT_ACCESSOR left_accessor,
                                            RIGHT_ACCESSOR right_accessor,
                                            bool check_overflow,
                                            bool both_inputs_valid,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  auto result =
    cudf::make_numeric_column(type, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);

  // Scenario A: Both inputs are valid and no overflow check is needed
  // No need to allocate validity vector, result has no nulls
  if (both_inputs_valid && !check_overflow) {
    thrust::for_each_n(rmm::exec_policy_nosync(stream),
                       thrust::make_counting_iterator(0),
                       num_rows,
                       multiply_no_validity_fn<T, LEFT_ACCESSOR, RIGHT_ACCESSOR>{
                         left_accessor, right_accessor, result->mutable_view().begin<T>()});
    return result;
  }

  // Scenario B and other cases: Need validity vector
  // For Scenario B (both_inputs_valid && check_overflow),
  // we still need validity vector to track overflow
  auto validity =
    rmm::device_uvector<bool>(num_rows, stream, cudf::get_current_device_resource_ref());

  // execute the multiplication
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(0),
    num_rows,
    multiply_fn<T, LEFT_ACCESSOR, RIGHT_ACCESSOR>{left_accessor,
                                                  right_accessor,
                                                  check_overflow,
                                                  result->mutable_view().begin<T>(),
                                                  validity.data()});

  // collect null mask and set
  auto [null_mask, null_count] =
    cudf::bools_to_mask(cudf::device_span<bool const>(validity), stream, mr);
  if (null_count > 0) { result->set_null_mask(std::move(*null_mask.release()), null_count); }

  return result;
}

/**
 * @brief Operator for multiply(cv, cv).
 */
struct dispatch_multiply {
  // left operand can be either a column_view or a scalar
  cudf::column_view const* left_cv;
  cudf::scalar const* left_scalar;

  // right operand can be either a column_view or a scalar
  cudf::column_view const* right_cv;
  cudf::scalar const* right_scalar;

  template <typename T, CUDF_ENABLE_IF(is_basic_spark_numeric<T>())>
  std::unique_ptr<cudf::column> operator()(cudf::data_type type,
                                           cudf::size_type num_rows,
                                           bool check_overflow,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const
  {
    if (left_cv != nullptr && right_cv != nullptr) {
      auto const left_cdv  = cudf::column_device_view::create(*left_cv, stream);
      auto const right_cdv = cudf::column_device_view::create(*right_cv, stream);
      if (left_cv->has_nulls()) {
        auto const left_accessor = cudf::detail::make_pair_iterator<T, true>(*left_cdv);
        if (right_cv->has_nulls()) {
          auto const right_accessor = cudf::detail::make_pair_iterator<T, true>(*right_cdv);
          return multiply_impl<T, decltype(left_accessor), decltype(right_accessor)>(
            type, num_rows, left_accessor, right_accessor, check_overflow, false, stream, mr);
        } else {
          auto const right_accessor = cudf::detail::make_pair_iterator<T, false>(*right_cdv);
          return multiply_impl<T, decltype(left_accessor), decltype(right_accessor)>(
            type, num_rows, left_accessor, right_accessor, check_overflow, false, stream, mr);
        }
      } else {
        auto const left_accessor = cudf::detail::make_pair_iterator<T, false>(*left_cdv);
        if (right_cv->has_nulls()) {
          auto const right_accessor = cudf::detail::make_pair_iterator<T, true>(*right_cdv);
          return multiply_impl<T, decltype(left_accessor), decltype(right_accessor)>(
            type, num_rows, left_accessor, right_accessor, check_overflow, false, stream, mr);
        } else {
          auto const right_accessor = cudf::detail::make_pair_iterator<T, false>(*right_cdv);
          return multiply_impl<T, decltype(left_accessor), decltype(right_accessor)>(
            type, num_rows, left_accessor, right_accessor, check_overflow, true, stream, mr);
        }
      }
    } else if (left_cv != nullptr && right_scalar != nullptr) {
      auto const right_accessor = cudf::detail::make_pair_iterator<T>(*right_scalar);
      auto const left_cdv       = cudf::column_device_view::create(*left_cv, stream);
      bool const both_valid     = !left_cv->has_nulls() && right_scalar->is_valid();
      if (left_cv->has_nulls()) {
        auto const left_accessor = cudf::detail::make_pair_iterator<T, true>(*left_cdv);
        return multiply_impl<T, decltype(left_accessor), decltype(right_accessor)>(
          type, num_rows, left_accessor, right_accessor, check_overflow, false, stream, mr);
      } else {
        auto const left_accessor = cudf::detail::make_pair_iterator<T, false>(*left_cdv);
        return multiply_impl<T, decltype(left_accessor), decltype(right_accessor)>(
          type, num_rows, left_accessor, right_accessor, check_overflow, both_valid, stream, mr);
      }
    } else if (left_scalar != nullptr && right_cv != nullptr) {
      auto const left_accessor = cudf::detail::make_pair_iterator<T>(*left_scalar);
      auto const right_cdv     = cudf::column_device_view::create(*right_cv, stream);
      bool const both_valid    = left_scalar->is_valid() && !right_cv->has_nulls();
      if (right_cv->has_nulls()) {
        auto const right_accessor = cudf::detail::make_pair_iterator<T, true>(*right_cdv);
        return multiply_impl<T, decltype(left_accessor), decltype(right_accessor)>(
          type, num_rows, left_accessor, right_accessor, check_overflow, false, stream, mr);
      } else {
        auto const right_accessor = cudf::detail::make_pair_iterator<T, false>(*right_cdv);
        return multiply_impl<T, decltype(left_accessor), decltype(right_accessor)>(
          type, num_rows, left_accessor, right_accessor, check_overflow, both_valid, stream, mr);
      }
    } else {
      CUDF_FAIL("Unsupported combination of inputs for multiplication.");
    }
  }

  template <typename T, CUDF_ENABLE_IF(!is_basic_spark_numeric<T>())>
  std::unique_ptr<cudf::column> operator()(cudf::data_type type,
                                           cudf::size_type num_rows,
                                           bool check_overflow,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const
  { CUDF_FAIL("Unsupported type when multiply."); }
};

}  // anonymous namespace

std::unique_ptr<cudf::column> multiply(cudf::column_view const& left_cv,
                                       cudf::column_view const& right_cv,
                                       bool is_ansi_mode,
                                       bool is_try_mode,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  check_multiply_inputs(
    left_cv.type(), right_cv.type(), left_cv.size(), right_cv.size(), is_ansi_mode, is_try_mode);
  auto result = cudf::type_dispatcher(left_cv.type(),
                                      dispatch_multiply{&left_cv, nullptr, &right_cv, nullptr},
                                      left_cv.type(),
                                      left_cv.size(),
                                      is_ansi_mode || is_try_mode,
                                      stream,
                                      mr);
  // check for overflow if ANSI mode is enabled
  if (is_ansi_mode) {
    // throw an error if any row has an overflow if ANSI mode is enabled
    spark_rapids_jni::throw_row_error_if_any(left_cv, right_cv, result->view(), stream);
  }
  return result;
}

std::unique_ptr<cudf::column> multiply(cudf::column_view const& left_cv,
                                       cudf::scalar const& right_scalar,
                                       bool is_ansi_mode,
                                       bool is_try_mode,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  check_multiply_inputs(
    left_cv.type(), right_scalar.type(), left_cv.size(), left_cv.size(), is_ansi_mode, is_try_mode);
  auto result = cudf::type_dispatcher(left_cv.type(),
                                      dispatch_multiply{&left_cv, nullptr, nullptr, &right_scalar},
                                      left_cv.type(),
                                      left_cv.size(),
                                      is_ansi_mode || is_try_mode,
                                      stream,
                                      mr);
  // check for overflow if ANSI mode is enabled
  if (is_ansi_mode) {
    // throw an error if any row has an overflow if ANSI mode is enabled
    spark_rapids_jni::throw_row_error_if_any(left_cv, right_scalar, result->view(), stream);
  }
  return result;
}

std::unique_ptr<cudf::column> multiply(cudf::scalar const& left_scalar,
                                       cudf::column_view const& right_cv,
                                       bool is_ansi_mode,
                                       bool is_try_mode,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  check_multiply_inputs(left_scalar.type(),
                        right_cv.type(),
                        right_cv.size(),
                        right_cv.size(),
                        is_ansi_mode,
                        is_try_mode);
  auto result = cudf::type_dispatcher(left_scalar.type(),
                                      dispatch_multiply{nullptr, &left_scalar, &right_cv, nullptr},
                                      right_cv.type(),
                                      right_cv.size(),
                                      is_ansi_mode || is_try_mode,
                                      stream,
                                      mr);
  // check for overflow if ANSI mode is enabled
  if (is_ansi_mode) {
    // throw an error if any row has an overflow if ANSI mode is enabled
    spark_rapids_jni::throw_row_error_if_any(left_scalar, right_cv, result->view(), stream);
  }
  return result;
}

}  // namespace spark_rapids_jni
