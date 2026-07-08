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
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/detail/combine.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_histogram.cuh>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>

namespace spark_rapids_jni {

namespace detail {

namespace {

__host__ __device__ constexpr bool not_whitespace(cudf::char_utf8 ch)
{
  return ch != ' ' && ch != '\r' && ch != '\n' && ch != '\t';
}

__host__ __device__ constexpr bool can_be_delimiter(char c)
{
  // The character list below is from `json_reader_options.set_delimiter`.
  switch (c) {
    case '{':
    case '[':
    case '}':
    case ']':
    case ',':
    case ':':
    case '"':
    case '\0':
    case '\'':
    case '\\':
    case ' ':
    case '\t':
    case '\r': return false;
    default: return true;
  }
}

}  // namespace

std::tuple<std::unique_ptr<rmm::device_buffer>, char, std::unique_ptr<cudf::column>> concat_json(
  cudf::strings_column_view const& input,
  bool nullify_invalid_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) {
    return {std::make_unique<rmm::device_buffer>(0, stream, mr),
            '\n',
            std::make_unique<cudf::column>(
              rmm::device_uvector<bool>{0, stream, mr}, rmm::device_buffer{}, 0)};
  }

  auto const d_input_ptr = cudf::column_device_view::create(input.parent(), stream);
  auto const default_mr  = rmm::mr::get_current_device_resource_ref();

  // Check if the input rows are null, empty (containing only whitespaces), and invalid JSON.
  // This will be used for masking out the null/empty/invalid input rows when doing string
  // concatenation.
  rmm::device_uvector<bool> is_valid_input(input.size(), stream, default_mr);

  // Check if the input rows are null, empty (containing only whitespaces), and may also check
  // for invalid JSON strings.
  // This will be returned to the caller to create null mask for the final output.
  rmm::device_uvector<bool> should_be_nullified(input.size(), stream, mr);

  thrust::for_each(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(0L),
    thrust::make_counting_iterator(input.size() * static_cast<int64_t>(cudf::detail::warp_size)),
    [nullify_invalid_rows,
     input  = *d_input_ptr,
     output = thrust::make_zip_iterator(is_valid_input.begin(),
                                        should_be_nullified.begin())] __device__(int64_t tidx) {
      // Execute one warp per row to minimize thread divergence.
      if ((tidx % cudf::detail::warp_size) != 0) { return; }
      auto const idx = tidx / cudf::detail::warp_size;

      if (input.is_null(idx)) {
        output[idx] = cuda::std::make_tuple(false, true);
        return;
      }

      auto const d_str = input.element<cudf::string_view>(idx);
      auto const size  = d_str.size_bytes();
      int i            = 0;
      char ch;

      // Skip the very first whitespace characters.
      for (; i < size; ++i) {
        ch = d_str[i];
        if (not_whitespace(ch)) { break; }
      }

      auto const not_eol = i < size;

      // If the current row is not null or empty, it should start with `{`. Otherwise, we need to
      // replace it by a null. This is necessary for libcudf's JSON reader to work.
      // Note that if we want to support ARRAY schema, we need to check for `[` instead.
      auto constexpr start_character = '{';
      if (not_eol && ch != start_character) {
        output[idx] = cuda::std::make_tuple(false, nullify_invalid_rows);
        return;
      }

      output[idx] = cuda::std::make_tuple(not_eol, !not_eol);
    });

  auto constexpr num_levels  = 256;
  auto constexpr lower_level = std::numeric_limits<char>::min();
  auto constexpr upper_level = std::numeric_limits<char>::max();
  auto const num_chars       = input.chars_size(stream);

  rmm::device_uvector<uint32_t> histogram(num_levels, stream, default_mr);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream), histogram.begin(), histogram.end(), 0);

  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven(nullptr,
                                      temp_storage_bytes,
                                      input.chars_begin(stream),
                                      histogram.begin(),
                                      num_levels,
                                      lower_level,
                                      upper_level,
                                      num_chars,
                                      stream.value());
  rmm::device_buffer d_temp(temp_storage_bytes, stream);
  cub::DeviceHistogram::HistogramEven(d_temp.data(),
                                      temp_storage_bytes,
                                      input.chars_begin(stream),
                                      histogram.begin(),
                                      num_levels,
                                      lower_level,
                                      upper_level,
                                      num_chars,
                                      stream.value());

  auto const it             = thrust::make_counting_iterator(0);
  auto const zero_level_idx = -lower_level;  // the bin storing count for character `\0`
  auto const zero_level_it  = it + zero_level_idx;
  auto const end            = it + num_levels;

  auto const first_zero_count_pos =
    thrust::find_if(rmm::exec_policy_nosync(stream),
                    zero_level_it,  // ignore the negative characters
                    end,
                    [zero_level_idx, counts = histogram.begin()] __device__(auto idx) -> bool {
                      auto const count = counts[idx];
                      if (count > 0) { return false; }
                      auto const first_non_existing_char = static_cast<char>(idx - zero_level_idx);
                      return can_be_delimiter(first_non_existing_char);
                    });

  // This should never happen since the input should never cover the entire char range.
  if (first_zero_count_pos == end) {
    throw std::logic_error(
      "Cannot find any character suitable as delimiter during joining json strings.");
  }
  auto const delimiter =
    static_cast<char>(cuda::std::distance(zero_level_it, first_zero_count_pos));

  auto [null_mask, null_count] =
    cudf::bools_to_mask(cudf::device_span<bool const>(is_valid_input), stream, default_mr);
  // If the null count doesn't change, just use the input column for concatenation.
  auto const input_applied_null =
    null_count == input.null_count()
      ? cudf::column_view{}
      : cudf::column_view{cudf::data_type{cudf::type_id::STRING},
                          input.size(),
                          input.chars_begin(stream),
                          reinterpret_cast<cudf::bitmask_type const*>(null_mask->data()),
                          null_count,
                          input.offset(),
                          std::vector<cudf::column_view>{input.offsets()}};

  auto concat_strings = cudf::strings::join_strings(
    null_count == input.null_count() ? input : cudf::strings_column_view{input_applied_null},
    cudf::string_scalar(std::string(1, delimiter), true, stream, default_mr),
    cudf::string_scalar("{}", true, stream, default_mr),
    stream,
    mr);

  return {std::move(concat_strings->release().data),
          delimiter,
          std::make_unique<cudf::column>(std::move(should_be_nullified), rmm::device_buffer{}, 0)};
}

}  // namespace detail

std::tuple<std::unique_ptr<rmm::device_buffer>, char, std::unique_ptr<cudf::column>> concat_json(
  cudf::strings_column_view const& input,
  bool nullify_invalid_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return detail::concat_json(input, nullify_invalid_rows, stream, mr);
}

}  // namespace spark_rapids_jni
