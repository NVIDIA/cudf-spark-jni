/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
#include "like.hpp"
#include "nvtx_ranges.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>

namespace spark_rapids_jni {
namespace detail {
namespace {

struct invalid_like_pattern_fn {
  cudf::column_device_view const input;
  cudf::column_device_view const patterns;
  cudf::string_view const escape;

  __device__ bool operator()(cudf::size_type row) const
  {
    if (input.is_null(row)) { return false; }

    auto const pattern           = patterns.element<cudf::string_view>(row);
    auto const escape_code_point = *escape.begin();
    auto it                      = pattern.begin();
    auto const end               = pattern.end();
    while (it != end) {
      if (*it == escape_code_point) {
        ++it;
        if (it == end) { return true; }
        auto const escaped = *it;
        if (escaped != escape_code_point && escaped != '_' && escaped != '%') { return true; }
      }
      ++it;
    }
    return false;
  }
};

void validate_patterns(cudf::strings_column_view const& input,
                       cudf::strings_column_view const& patterns,
                       cudf::string_scalar const& escape_character,
                       cuda::stream_ref stream)
{
  if (input.is_empty()) { return; }

  CUDF_EXPECTS(input.size() == patterns.size(), "Number of patterns must match the input size");
  CUDF_EXPECTS(!patterns.has_nulls(), "Parameter patterns must not contain nulls");
  CUDF_EXPECTS(escape_character.is_valid(stream), "Escape character must be valid");
  auto const escape = escape_character.value(stream);
  CUDF_EXPECTS(escape.size_bytes() == 1, "Escape character must contain exactly one ASCII byte");

  auto const d_input = cudf::column_device_view::create(
    input.parent(), stream, cudf::get_current_device_resource_ref());
  auto const d_patterns = cudf::column_device_view::create(
    patterns.parent(), stream, cudf::get_current_device_resource_ref());
  auto const begin = thrust::make_counting_iterator<cudf::size_type>(0);
  auto const end   = begin + input.size();
  auto const invalid =
    thrust::find_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    begin,
                    end,
                    invalid_like_pattern_fn{*d_input, *d_patterns, escape});
  if (invalid != end) { throw spark_rapids_jni::exception_with_row_index(*invalid); }
}

}  // namespace
}  // namespace detail

std::unique_ptr<cudf::column> like(cudf::strings_column_view const& input,
                                   cudf::strings_column_view const& patterns,
                                   cudf::string_scalar const& escape_character,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  detail::validate_patterns(input, patterns, escape_character, stream);
  return cudf::strings::like(input, patterns, escape_character, stream, mr);
}

}  // namespace spark_rapids_jni
