/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

#include "nvtx_ranges.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace spark_rapids_jni {

namespace {

struct literal_range_pattern_fn {
  __device__ bool operator()(
    cudf::string_view d_string, cudf::string_view d_prefix, int range_len, int start, int end) const
  {
    int const n = d_string.length(), m = d_prefix.length();
    for (int i = 0; i <= n - m - range_len; i++) {
      bool match = true;
      for (int j = 0; j < m; j++) {
        if (d_string[i + j] != d_prefix[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        for (int j = 0; j < range_len; j++) {
          auto code_point = cudf::strings::detail::utf8_to_codepoint(d_string[i + m + j]);
          if (code_point < start || code_point > end) {
            match = false;
            break;
          }
        }
        if (match) { return true; }
      }
    }
    return false;
  }
};

std::unique_ptr<cudf::column> find_literal_range_pattern(cudf::strings_column_view const& strings,
                                                         cudf::string_scalar const& prefix,
                                                         int const range_len,
                                                         int const start,
                                                         int const end,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  auto const strings_count = strings.size();
  if (strings_count == 0) { return cudf::make_empty_column(cudf::type_id::BOOL8); }

  CUDF_EXPECTS(prefix.is_valid(stream), "Parameter prefix must be valid.");

  auto const d_prefix       = cudf::string_view(prefix.data(), prefix.size());
  auto const strings_column = cudf::column_device_view::create(strings.parent(), stream);
  auto const d_strings      = *strings_column;

  auto results         = make_numeric_column(cudf::data_type{cudf::type_id::BOOL8},
                                             strings_count,
                                             cudf::copy_bitmask(strings.parent(), stream, mr),
                                             strings.null_count(),
                                             stream,
                                             mr);
  auto const d_results = results->mutable_view().data<bool>();
  // set the bool values by evaluating the passed function
  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(strings_count),
    d_results,
    [d_strings, d_prefix, range_len, start, end, check_fn = literal_range_pattern_fn{}] __device__(
      cudf::size_type idx) {
      if (!d_strings.is_null(idx)) {
        return check_fn(d_strings.element<cudf::string_view>(idx), d_prefix, range_len, start, end);
      }
      return false;
    });
  results->set_null_count(strings.null_count());
  return results;
}

}  // namespace

/**
 * @brief Check if input string contains regex pattern `literal[start-end]{len,}`, which means
 * a literal string followed by a range of characters in the range of start to end, with at least
 * len characters.
 *
 * @param strings Column of strings to check for literal.
 * @param literal UTF-8 encoded string to check in strings column.
 * @param len Minimum number of characters to check after the literal.
 * @param start Minimum UTF-8 codepoint value to check for in the range.
 * @param end Maximum UTF-8 codepoint value to check for in the range.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
std::unique_ptr<cudf::column> literal_range_pattern(cudf::strings_column_view const& input,
                                                    cudf::string_scalar const& prefix,
                                                    int const range_len,
                                                    int const start,
                                                    int const end,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return find_literal_range_pattern(input, prefix, range_len, start, end, stream, mr);
}

}  // namespace spark_rapids_jni
