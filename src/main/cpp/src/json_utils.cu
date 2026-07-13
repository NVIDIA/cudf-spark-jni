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
#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>

#include <cstdint>

namespace spark_rapids_jni {

namespace detail {

namespace {

__host__ __device__ constexpr bool not_whitespace(cudf::char_utf8 ch)
{ return ch != ' ' && ch != '\r' && ch != '\n' && ch != '\t'; }

__host__ __device__ constexpr bool can_be_delimiter(std::uint8_t c)
{
  // Matches `json_reader_options::set_delimiter`, with NUL additionally excluded because
  // embedded NUL bytes must not be treated as row boundaries.
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

// Delimiter preference, in the order `delimiter_candidate` enumerates it:
//   1. '\n'                         - the natural JSON-lines delimiter
//   2. printable ASCII 0x21..0x7e   - safe, never confused with unquoted control chars
//   3. DEL 0x7f
//   4. C0 controls 0x01..0x1f       - last-resort fallback (see note in `delimiter_candidate`)
// NUL (0x00) is never a candidate; the other excluded bytes are filtered by `can_be_delimiter`.
constexpr std::uint8_t first_printable = 0x21;  // '!'
constexpr std::uint8_t last_printable  = 0x7e;  // '~'
constexpr std::uint8_t del_byte        = 0x7f;
constexpr std::uint8_t first_c0        = 0x01;
constexpr std::uint8_t last_c0         = 0x1f;

constexpr int num_printable         = last_printable - first_printable + 1;
constexpr int num_c0                = last_c0 - first_c0 + 1;
constexpr int first_printable_index = 1;  // index 0 is '\n'
constexpr int del_index             = first_printable_index + num_printable;
constexpr int first_c0_index        = del_index + 1;

// One index per preference slot: '\n' + printables + DEL + C0 fallback (0x01..0x1f).
constexpr int num_delimiter_candidates = first_c0_index + num_c0;

__host__ __device__ constexpr std::uint8_t delimiter_candidate(int candidate_index)
{
  if (candidate_index == 0) { return '\n'; }
  if (candidate_index < del_index) {
    return static_cast<std::uint8_t>(first_printable + candidate_index - first_printable_index);
  }
  if (candidate_index == del_index) { return del_byte; }

  // C0 bytes (0x01..0x1f) are a last-resort fallback, reached only if '\n', every eligible
  // printable, and DEL are all already present in the input. A C0 delimiter can still be
  // mishandled by the reader when `allow_unquoted_control` is set (the very interaction this
  // ordering avoids), so this branch is intentionally the least preferred. NUL (0x00) is not
  // reachable here; tab (0x09) and CR (0x0d) fall in this range but are rejected by
  // `can_be_delimiter`, while LF (0x0a) is already covered at index 0 (its count is nonzero
  // whenever this branch is reached, so the duplicate is harmless).
  return static_cast<std::uint8_t>(first_c0 + candidate_index - first_c0_index);
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

  // CUB expects one more level than bins. Use byte samples and integer bounds so each bin maps
  // exactly to one ASCII byte, including DEL (0x7f). High-bit bytes are intentionally ignored.
  auto constexpr num_bins    = 128;
  auto constexpr num_levels  = num_bins + 1;
  auto constexpr lower_level = 0;
  auto constexpr upper_level = 128;
  auto const num_chars       = input.chars_size(stream);

  rmm::device_uvector<uint32_t> histogram(num_bins, stream, default_mr);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream), histogram.begin(), histogram.end(), 0);

  auto const byte_samples   = reinterpret_cast<std::uint8_t const*>(input.chars_begin(stream));
  size_t temp_storage_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceHistogram::HistogramEven(nullptr,
                                                    temp_storage_bytes,
                                                    byte_samples,
                                                    histogram.begin(),
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    num_chars,
                                                    stream.value()));
  {
    rmm::device_buffer d_temp(temp_storage_bytes, stream);
    CUDF_CUDA_TRY(cub::DeviceHistogram::HistogramEven(d_temp.data(),
                                                      temp_storage_bytes,
                                                      byte_samples,
                                                      histogram.begin(),
                                                      num_levels,
                                                      lower_level,
                                                      upper_level,
                                                      num_chars,
                                                      stream.value()));
  }

  auto const candidates_begin         = thrust::make_counting_iterator(0);
  auto const candidates_end           = candidates_begin + num_delimiter_candidates;
  auto const find_available_delimiter = [&] {
    return thrust::find_if(rmm::exec_policy_nosync(stream),
                           candidates_begin,
                           candidates_end,
                           [counts = histogram.begin()] __device__(auto candidate_index) -> bool {
                             auto const candidate = delimiter_candidate(candidate_index);
                             return can_be_delimiter(candidate) && counts[candidate] == 0;
                           });
  };

  auto first_available = find_available_delimiter();
  if (first_available == candidates_end) {
    // Invalid rows are replaced with "{}" before parsing, so their original bytes do not need
    // to reserve a delimiter. Retry with a row-aware histogram only on this rare exhausted path.
    thrust::fill(rmm::exec_policy_nosync(stream), histogram.begin(), histogram.end(), 0);
    thrust::for_each(
      rmm::exec_policy_nosync(stream),
      thrust::make_counting_iterator(0L),
      thrust::make_counting_iterator(input.size() * static_cast<int64_t>(cudf::detail::warp_size)),
      [input      = *d_input_ptr,
       valid_rows = is_valid_input.data(),
       counts     = histogram.data()] __device__(int64_t tidx) {
        auto const lane = static_cast<int32_t>(tidx % cudf::detail::warp_size);
        auto const row  = tidx / cudf::detail::warp_size;
        if (!valid_rows[row]) { return; }

        auto const d_str = input.element<cudf::string_view>(row);
        auto const bytes = reinterpret_cast<std::uint8_t const*>(d_str.data());
        for (auto byte_index = lane; byte_index < d_str.size_bytes();
             byte_index += cudf::detail::warp_size) {
          auto const byte = bytes[byte_index];
          if (byte < num_bins) { atomicAdd(&counts[byte], 1U); }
        }
      });
    first_available = find_available_delimiter();
  }

  if (first_available == candidates_end) {
    throw std::logic_error(
      "Cannot find an unused cuDF-supported ASCII delimiter while joining JSON strings.");
  }
  auto const delimiter =
    static_cast<char>(delimiter_candidate(cuda::std::distance(candidates_begin, first_available)));

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
