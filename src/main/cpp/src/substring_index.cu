/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.
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
#include "substring_index.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

using namespace cudf;

namespace spark_rapids_jni {

namespace detail {

namespace {

/**
 * @brief Compute slice indices for each string.
 *
 * When slice_strings is invoked with a delimiter string and a delimiter count, we need to
 * compute the start and end indices of the substring. This function accomplishes that.
 */
template <typename DelimiterItrT>
void compute_substring_indices(column_device_view const& d_column,
                               DelimiterItrT const delim_itr,
                               size_type delimiter_count,
                               size_type* start_char_pos,
                               size_type* end_char_pos,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref)
{
  auto strings_count = d_column.size();

  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    strings_count,
    [delim_itr, delimiter_count, start_char_pos, end_char_pos, d_column] __device__(size_type idx) {
      auto const& delim_val_pair = delim_itr[idx];
      auto const& delim_val      = delim_val_pair.first;  // Don't use it yet

      // If the column value for this row is null, result is null.
      // If the delimiter count is 0, result is empty string.
      // If the global delimiter or the row specific delimiter is invalid or if it is empty, row
      // value is empty.
      if (d_column.is_null(idx) || !delim_val_pair.second || delim_val.empty()) return;
      auto const& col_val = d_column.element<string_view>(idx);

      // If the column value for the row is empty, the row value is empty.
      if (!col_val.empty()) {
        auto const col_val_len   = col_val.length();
        auto const delimiter_len = delim_val.length();

        auto nsearches           = (delimiter_count < 0) ? -delimiter_count : delimiter_count;
        bool const left_to_right = (delimiter_count > 0);

        size_type start_pos = start_char_pos[idx];
        size_type end_pos   = col_val_len;
        size_type char_pos  = -1;

        end_char_pos[idx] = col_val_len;

        for (auto i = 0; i < nsearches; ++i) {
          char_pos = left_to_right ? col_val.find(delim_val, start_pos)
                                   : col_val.rfind(delim_val, 0, end_pos);
          if (char_pos == string_view::npos) return;
          if (left_to_right)
            start_pos = char_pos + delimiter_len;
          else
            end_pos = char_pos;
        }
        if (left_to_right)
          end_char_pos[idx] = char_pos;
        else
          start_char_pos[idx] = end_pos + delimiter_len;
      }
    });
}

}  // namespace

template <typename DelimiterItrT>
std::unique_ptr<column> substring_index(strings_column_view const& strings,
                                        DelimiterItrT const delimiter_itr,
                                        size_type count,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  auto strings_count = strings.size();
  // If there aren't any rows, return an empty strings column
  if (strings_count == 0) { return make_empty_column(type_id::STRING); }

  // Compute the substring indices first
  auto start_chars_pos_vec = make_column_from_scalar(numeric_scalar<size_type>(0, true, stream),
                                                     strings_count,
                                                     stream,
                                                     rmm::mr::get_current_device_resource_ref());
  auto stop_chars_pos_vec  = make_column_from_scalar(numeric_scalar<size_type>(0, true, stream),
                                                     strings_count,
                                                     stream,
                                                     rmm::mr::get_current_device_resource_ref());

  auto start_char_pos = start_chars_pos_vec->mutable_view().data<size_type>();
  auto end_char_pos   = stop_chars_pos_vec->mutable_view().data<size_type>();

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;

  // If delimiter count is 0, the output column will contain empty strings
  if (count != 0) {
    // Compute the substring indices first
    compute_substring_indices(
      d_column, delimiter_itr, count, start_char_pos, end_char_pos, stream, mr);
  }

  return cudf::strings::slice_strings(
    strings, start_chars_pos_vec->view(), stop_chars_pos_vec->view(), stream, mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> substring_index(strings_column_view const& strings,
                                        string_scalar const& delimiter,
                                        size_type count,
                                        rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return detail::substring_index(strings,
                                 cudf::detail::make_pair_iterator<string_view>(delimiter),
                                 count,
                                 cudf::get_default_stream(),
                                 mr);
}
}  // namespace spark_rapids_jni
