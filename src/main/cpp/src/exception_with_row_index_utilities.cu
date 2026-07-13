/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
#include "exception_with_row_index_utilities.hpp"

#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>

namespace spark_rapids_jni {

namespace {

__device__ bool is_not_null(cudf::bitmask_type const* mask, cudf::size_type idx)
{
  // If the bitmask type is cudf::mask_state::UNALLOCATED, it means all rows are valid(not null).
  return mask == nullptr || cudf::bit_is_set(mask, idx);
}

__device__ bool is_null(cudf::bitmask_type const* mask, cudf::size_type idx)
{ return !is_not_null(mask, idx); }

/**
 * @brief Gets if a row is invalid: input is non-null but the result is null.
 * E.g.: Overflow results in a null result for non-null input for some unary operations.
 */
struct row_invalid_unary_fn {
  cudf::bitmask_type const* input_mask;
  cudf::bitmask_type const* result_mask;

  __device__ bool operator()(cudf::size_type idx) const
  { return is_not_null(input_mask, idx) && is_null(result_mask, idx); }
};

/**
 * @brief Gets if a row is invalid: inputs are non-null but the result is null.
 * E.g.: Overflow results in a null result for non-null inputs for some binary operations.
 */
struct row_invalid_binary_fn {
  cudf::bitmask_type const* input_mask1;
  cudf::bitmask_type const* input_mask2;
  cudf::bitmask_type const* result_mask;

  __device__ bool operator()(cudf::size_type idx) const
  {
    return is_not_null(input_mask1, idx) && is_not_null(input_mask2, idx) &&
           is_null(result_mask, idx);
  }
};

/**
 * @brief Gets if a row is invalid: inputs are non-null but the result is null.
 * E.g.: Overflow results in a null result for non-null inputs for some ternary operations.
 */
struct row_invalid_ternary_fn {
  cudf::bitmask_type const* input_mask1;
  cudf::bitmask_type const* input_mask2;
  cudf::bitmask_type const* input_mask3;
  cudf::bitmask_type const* result_mask;

  __device__ bool operator()(cudf::size_type idx) const
  {
    return is_not_null(input_mask1, idx) && is_not_null(input_mask2, idx) &&
           is_not_null(input_mask3, idx) && is_null(result_mask, idx);
  }
};

}  // anonymous namespace

void throw_row_error_if_any(cudf::column_view const& input,
                            cudf::column_view const& result,
                            rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(input.size() == result.size(),
               "The row counts of the input and result columns must match.");
  if (result.size() == 0) {
    // No rows to check, so no errors.
    return;
  }
  if (result.null_count() > input.null_count()) {
    auto const itr_begin = thrust::make_counting_iterator(0);
    auto const itr_end   = thrust::make_counting_iterator(result.size());
    auto const first_row_idx_with_error =
      thrust::find_if(rmm::exec_policy(stream),
                      itr_begin,
                      itr_end,
                      row_invalid_unary_fn{input.null_mask(), result.null_mask()});
    auto const has_error = first_row_idx_with_error != itr_end;
    if (has_error) { throw spark_rapids_jni::exception_with_row_index(*first_row_idx_with_error); }
  }
}

void throw_row_error_if_any(cudf::column_view const& input1,
                            cudf::column_view const& input2,
                            cudf::column_view const& result,
                            rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS((input1.size() == input2.size() && input2.size() == result.size()),
               "The row counts of the input and result columns must match.");
  if (result.size() == 0) {
    // No rows to check, so no errors.
    return;
  }
  auto const itr_begin                = thrust::make_counting_iterator(0);
  auto const itr_end                  = thrust::make_counting_iterator(result.size());
  auto const first_row_idx_with_error = thrust::find_if(
    rmm::exec_policy(stream),
    itr_begin,
    itr_end,
    row_invalid_binary_fn{input1.null_mask(), input2.null_mask(), result.null_mask()});
  auto const has_error = first_row_idx_with_error != itr_end;
  if (has_error) { throw spark_rapids_jni::exception_with_row_index(*first_row_idx_with_error); }
}

void throw_row_error_if_any(cudf::column_view const& input1,
                            cudf::column_view const& input2,
                            cudf::column_view const& input3,
                            cudf::column_view const& result,
                            rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS((input1.size() == input2.size() && input2.size() == input3.size() &&
                input3.size() == result.size()),
               "The row counts of the input and result columns must match.");
  if (result.size() == 0) {
    // No rows to check, so no errors.
    return;
  }
  auto const itr_begin                = thrust::make_counting_iterator(0);
  auto const itr_end                  = thrust::make_counting_iterator(result.size());
  auto const first_row_idx_with_error = thrust::find_if(
    rmm::exec_policy(stream),
    itr_begin,
    itr_end,
    row_invalid_ternary_fn{
      input1.null_mask(), input2.null_mask(), input3.null_mask(), result.null_mask()});
  auto const has_error = first_row_idx_with_error != itr_end;
  if (has_error) { throw spark_rapids_jni::exception_with_row_index(*first_row_idx_with_error); }
}

void throw_row_error_if_any(cudf::column_view const& input1,
                            cudf::scalar const& input2,
                            cudf::column_view const& result,
                            rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(input1.size() == result.size(),
               "The row counts of the input and result columns must match.");

  if (input2.is_valid(stream)) {
    // scalar is not null, only need to check the `input1` and `result` columns.
    throw_row_error_if_any(input1, result, stream);
  } else {
    // scalar is null
    CUDF_EXPECTS(result.null_count() == result.size(),
                 "If the scalar is null, the result must be all null.");
  }
}

void throw_row_error_if_any(cudf::scalar const& input1,
                            cudf::column_view const& input2,
                            cudf::column_view const& result,
                            rmm::cuda_stream_view stream)
{ throw_row_error_if_any(input2, input1, result, stream); }
}  // namespace spark_rapids_jni
