/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "datetime_utils.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/std/bit>
#include <cuda/std/limits>
#include <cuda/stream>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstdint>
#include <memory>
#include <utility>

namespace spark_rapids_jni::date_parse {

// Spark's older UTF8String.trimAll uses Character.isWhitespace; patched versions also
// use Character.isISOControl. Both operate on signed bytes, so only ASCII bytes trim.
__device__ inline bool is_whitespace(int c, bool trim_control_characters)
{
  return trim_control_characters ? (c >= 0 && (c <= 32 || c == 127))
                                 : ((c >= 9 && c <= 13) || (c >= 28 && c <= 32));
}

// Cleaning is a view over the original input: CSV fallback removes each original GMT
// occurrence, as cleanLegacyTimestampStr does, without materializing a second strings column.
template <bool CleanGMT = false>
struct cursor {
  unsigned char const* data;
  int pos;
  int end;

  __device__ int peek()
  {
    if constexpr (CleanGMT) {
      while (end - pos >= 3 && data[pos] == 'G' && data[pos + 1] == 'M' && data[pos + 2] == 'T') {
        pos += 3;
      }
    }
    return pos < end ? data[pos] : -1;
  }

  __device__ bool eof() { return peek() < 0; }

  __device__ bool consume(unsigned char c)
  {
    if (peek() != c) { return false; }
    ++pos;
    return true;
  }

  // Stop at max_digits rather than consuming a following packed field. Return -1 on
  // overflow; years with up to 19 digits may still fit when they contain leading zeroes.
  __device__ int read_digits(int max_digits, int& value)
  {
    value      = 0;
    int digits = 0;
    while (digits < max_digits) {
      int const digit = peek() - '0';
      if (digit < 0 || digit > 9) { break; }
      if (value > (cuda::std::numeric_limits<int>::max() - digit) / 10) { return -1; }
      value = value * 10 + digit;
      ++digits;
      ++pos;
    }
    return digits;
  }

  __device__ void skip_whitespace(bool trim_control_characters)
  {
    while (is_whitespace(peek(), trim_control_characters)) {
      ++pos;
    }
  }

  __device__ bool remaining_is_whitespace(bool trim_control_characters) const
  {
    auto tail = *this;
    tail.skip_whitespace(trim_control_characters);
    return tail.eof();
  }
};

__device__ inline bool to_days(int year, int month, int day, cudf::timestamp_D& result)
{
  if (!date_time_utils::is_valid_date_for_date(year, month, day)) { return false; }
  auto const days = date_time_utils::to_epoch_day(year, month, day);
  if (days < cuda::std::numeric_limits<int32_t>::min() ||
      days > cuda::std::numeric_limits<int32_t>::max()) {
    return false;
  }
  result = cudf::timestamp_D{cudf::duration_D{static_cast<int32_t>(days)}};
  return true;
}

// Shared CAST grammar for toDate, DefaultDateFormatter and CSV parsing fallback.
// Partial dates consume the whole trimmed input; a complete date may have a space/T tail.
template <bool CleanGMT>
__device__ bool parse_cast(cursor<CleanGMT> input,
                           bool trim_control_characters,
                           cudf::timestamp_D& result)
{
  input.skip_whitespace(trim_control_characters);
  bool const negative = input.consume('-');
  if (!negative) { input.consume('+'); }

  date_segments date;
  if (input.read_digits(7, date.year) < 4) { return false; }
  if (negative) { date.year = -date.year; }
  if (input.remaining_is_whitespace(trim_control_characters)) {
    return to_days(date.year, date.month, date.day, result);
  }
  if (!input.consume('-') || input.read_digits(2, date.month) < 1) { return false; }
  if (input.remaining_is_whitespace(trim_control_characters)) {
    return to_days(date.year, date.month, date.day, result);
  }
  if (!input.consume('-') || input.read_digits(2, date.day) < 1) { return false; }
  if (input.peek() != ' ' && input.peek() != 'T' &&
      !input.remaining_is_whitespace(trim_control_characters)) {
    return false;
  }
  return to_days(date.year, date.month, date.day, result);
}

struct cast_parser {
  __device__ bool operator()(cudf::string_view input, cudf::timestamp_D& result) const
  {
    auto const* p = cuda::std::bit_cast<unsigned char const*>(input.data());
    // Preserve the existing CAST API's ISO control character trimming, including
    // its deviation on older Spark runtimes. CSV passes the runtime trim policy;
    // updating CAST requires a separate toDate API and caller change.
    return parse_cast(cursor<>{p, 0, input.size_bytes()}, true, result);
  }
};

// Common null handling and row dispatch for CAST and CSV date dialects.
template <typename Parser>
struct parse_strings_fn {
  cudf::column_device_view strings;
  Parser parser;
  bool* validity;
  cudf::timestamp_D* output;

  __device__ void operator()(cudf::size_type idx) const
  {
    validity[idx] = false;
    if (!strings.is_null(idx)) {
      validity[idx] = parser(strings.element<cudf::string_view>(idx), output[idx]);
    }
  }
};

template <typename Parser>
[[nodiscard]] std::unique_ptr<cudf::column> parse_dates(cudf::strings_column_view const& input,
                                                        Parser parser,
                                                        cuda::stream_ref stream,
                                                        rmm::device_async_resource_ref mr)
{
  auto const dtype = cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
  auto const size  = input.size();
  if (size == 0) { return cudf::make_empty_column(dtype); }
  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto strings       = cudf::column_device_view::create(input.parent(), stream, temp_mr);
  auto result = cudf::make_timestamp_column(dtype, size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto validity = rmm::device_uvector<bool>(size, stream, temp_mr);
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, temp_mr),
    thrust::make_counting_iterator(0),
    size,
    parse_strings_fn<Parser>{
      *strings, parser, validity.data(), result->mutable_view().begin<cudf::timestamp_D>()});
  auto [mask, null_count] =
    cudf::bools_to_mask(cudf::device_span<bool const>{validity}, stream, mr);
  if (null_count) { result->set_null_mask(std::move(*mask), null_count); }
  return result;
}

}  // namespace spark_rapids_jni::date_parse
