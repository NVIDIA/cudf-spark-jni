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

#include "cast_string_to_timestamp_common.hpp"
#include "datetime_utils.cuh"
#include "timezones.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/functional>
#include <thrust/binary_search.h>

using column                   = cudf::column;
using column_device_view       = cudf::column_device_view;
using column_view              = cudf::column_view;
using lists_column_device_view = cudf::lists_column_device_view;
using size_type                = cudf::size_type;
using struct_view              = cudf::struct_view;
using table_view               = cudf::table_view;

namespace {

/**
 * Functor to convert timestamps between UTC and a specific timezone.
 */
template <typename timestamp_type>
struct convert_timestamp_tz_functor {
  using duration_type = typename timestamp_type::duration;

  // Fixed offset transitions in the timezone table
  // Column type is LIST<STRUCT<utcInstant: int64, tzInstant: int64, utcOffset: int32>>.
  lists_column_device_view const fixed_transitions;

  // DST rules in the timezone table
  // column type is LIST<INT>, if it's DST, 12 integers defines two rules
  lists_column_device_view const dst_rules;

  // the index of the specified zone id in the transitions table
  size_type const tz_index;

  // whether we are converting to UTC or converting to the timezone
  bool const to_utc;

  /**
   * @brief Convert the timestamp value to either UTC or a specified timezone
   * @param timestamp input timestamp
   *
   */
  __device__ timestamp_type operator()(timestamp_type const& timestamp) const
  {
    return spark_rapids_jni::convert_timestamp(
      timestamp, fixed_transitions, dst_rules, tz_index, to_utc);
  }
};

template <typename timestamp_type>
auto convert_timestamp_tz(column_view const& input,
                          table_view const& transitions,
                          size_type tz_index,
                          bool to_utc,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  // get the fixed transitions
  auto const ft_cdv_ptr        = column_device_view::create(transitions.column(0), stream);
  auto const fixed_transitions = lists_column_device_view{*ft_cdv_ptr};

  // get the DST rules
  auto const dst_cdv_ptr = cudf::column_device_view::create(transitions.column(1), stream);
  auto const dst_rules   = cudf::lists_column_device_view{*dst_cdv_ptr};

  auto results = cudf::make_timestamp_column(input.type(),
                                             input.size(),
                                             cudf::copy_bitmask(input, stream, mr),
                                             input.null_count(),
                                             stream,
                                             mr);

  thrust::transform(
    rmm::exec_policy(stream),
    input.begin<timestamp_type>(),
    input.end<timestamp_type>(),
    results->mutable_view().begin<timestamp_type>(),
    convert_timestamp_tz_functor<timestamp_type>{fixed_transitions, dst_rules, tz_index, to_utc});

  return results;
}

/**
 * Functor to convert timestamps between UTC and a specific timezone.
 * This is used for casting string(with timezone) to timestamp.
 * This functor can handle multiple timezones for each row.
 */
struct convert_with_timezones_fn {
  // inputs
  int64_t const* input_seconds;
  int32_t const* input_microseconds;
  uint8_t const* invalid;
  uint8_t const* tz_type;
  int32_t const* tz_offset;
  // Fixed offset transitions in the timezone table
  // Column type is LIST<STRUCT<utcInstant: int64, tzInstant: int64, utcOffset: int32>>.
  lists_column_device_view const fixed_transitions;
  // DST rules in the timezone table
  // column type is LIST<INT>, if it's DST, 12 integers defines two rules
  lists_column_device_view const dst_rules;
  int32_t const* tz_indices;

  // outputs
  cudf::timestamp_us* output;
  bool* output_mask;

  /**
   * @brief Convert the timestamp from UTC to a specified timezone
   * @param row_idx row index of the input column
   *
   */
  __device__ void operator()(cudf::size_type row_idx) const
  {
    // 1. check if the input is invalid
    if (invalid[row_idx]) {
      output[row_idx]      = cudf::timestamp_us{cudf::duration_us{0L}};
      output_mask[row_idx] = false;
      return;
    }

    // 2. convert seconds part first
    int64_t epoch_seconds = input_seconds[row_idx];
    int64_t converted_seconds;
    if (static_cast<spark_rapids_jni::TZ_TYPE>(tz_type[row_idx]) ==
        spark_rapids_jni::TZ_TYPE::FIXED_TZ) {
      // Fixed offset, offset is in seconds, add the offset
      // E.g: A valid offset +01:02:03, it's not in the transition table
      // We need to handle it here.
      converted_seconds = epoch_seconds - tz_offset[row_idx];
    } else {
      // not fixed offset, use the fixed_transitions and dst_rules to convert
      cudf::timestamp_s converted_ts = spark_rapids_jni::convert_timestamp<cudf::timestamp_s>(
        cudf::timestamp_s{cudf::duration_s{epoch_seconds}},
        fixed_transitions,
        dst_rules,
        tz_indices[row_idx],
        true);
      converted_seconds = converted_ts.time_since_epoch().count();
    }

    // 3. Adding the microseconds part, this may cause overflow
    int64_t result;
    if (spark_rapids_jni::overflow_checker::get_timestamp_overflow(
          converted_seconds, input_microseconds[row_idx], result)) {
      // overflowed
      output[row_idx]      = cudf::timestamp_us{cudf::duration_us{0L}};
      output_mask[row_idx] = false;
    } else {
      // not overflowed
      output[row_idx]      = cudf::timestamp_us{cudf::duration_us{result}};
      output_mask[row_idx] = true;
    }
  }
};

std::unique_ptr<column> convert_to_utc_with_multiple_timezones(
  column_view const& input_seconds,
  column_view const& input_microseconds,
  column_view const& invalid,
  column_view const& tz_type,
  column_view const& tz_offset,
  table_view const& transitions,
  column_view const tz_indices,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input_seconds.type().id() == cudf::type_id::INT64,
               "seconds column must be of type INT64");
  CUDF_EXPECTS(input_microseconds.type().id() == cudf::type_id::INT32,
               "microseconds column must be of type INT32");

  // get the fixed transitions
  auto const ft_cdv_ptr        = column_device_view::create(transitions.column(0), stream);
  auto const fixed_transitions = lists_column_device_view{*ft_cdv_ptr};

  // get DST rules
  auto const dst_cdv_ptr = cudf::column_device_view::create(transitions.column(1), stream);
  auto const dst_rules   = cudf::lists_column_device_view{*dst_cdv_ptr};

  auto result = cudf::make_timestamp_column(cudf::data_type{cudf::type_to_id<cudf::timestamp_us>()},
                                            input_seconds.size(),
                                            rmm::device_buffer{},
                                            0,
                                            stream,
                                            mr);
  auto null_mask = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                                 input_seconds.size(),
                                                 cudf::mask_state::UNALLOCATED,
                                                 stream,
                                                 mr);

  thrust::for_each_n(rmm::exec_policy_nosync(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     input_seconds.size(),
                     convert_with_timezones_fn{input_seconds.begin<int64_t>(),
                                               input_microseconds.begin<int32_t>(),
                                               invalid.begin<uint8_t>(),
                                               tz_type.begin<uint8_t>(),
                                               tz_offset.begin<int32_t>(),
                                               fixed_transitions,
                                               dst_rules,
                                               tz_indices.begin<int32_t>(),
                                               result->mutable_view().begin<cudf::timestamp_us>(),
                                               null_mask->mutable_view().begin<bool>()});

  auto [output_bitmask, null_count] = cudf::bools_to_mask(null_mask->view(), stream, mr);
  if (null_count) { result->set_null_mask(std::move(*output_bitmask.release()), null_count); }

  return result;
}

// =================== ORC timezones begin ===================
// ORC timezone uses java.util.TimeZone rules, which is different from java.time.ZoneId rules.

/**
 * @brief Get the transition index for the given time `time_ms` using binary search.
 * Find the first transition that is greater or equal to `time_ms`,
 * and return the corresponding offset.
 *
 * @param begin the beginning of the transition array.
 * @param end the end of the transition array.
 * @param time_ms the input time in milliseconds to find the transition index for.
 * @param offset_begin the beginning of the offset array.
 * @param offset_end the end of the offset array.
 *
 * @return the offset.
 */
__device__ static int32_t get_transition_index(int64_t const* begin,
                                               int64_t const* end,
                                               int64_t time_ms,
                                               int32_t const* offset_begin,
                                               int32_t const* offset_end,
                                               int32_t raw_offset)
{
  if (begin == end) {
    // fixed offset timezone, no transitions
    return raw_offset;
  }

  auto const iter = thrust::upper_bound(thrust::seq, begin, end, time_ms);
  if (iter == end) {
    // after the transition table, returns the raw offset
    return raw_offset;
  }

  int32_t index = static_cast<int32_t>(cuda::std::distance(begin, iter));
  if (*iter == time_ms) {
    // find exact match, return the offset at that index
    return offset_begin[index];
  }

  if (index == 0) {
    // prior to the transition table, returns the raw offset
    return raw_offset;
  }

  // return the offset at the previous index
  return offset_begin[index - 1];
}

/**
 * @brief Find the relative offset when moving between timezones at a particular point in time.
 * This is for ORC timezone support.
 *
 * This function implements `org.apache.orc.impl.SerializationUtils.convertBetweenTimezones`
 * Refer to link: https://github.com/apache/orc/blob/rel/release-1.9.1/java/core/src/
 * java/org/apache/orc/impl/SerializationUtils.java#L1440
 *
 * If the input `trans_begin` == `trans_begin` == nullptr, it means the timezone is fixed offset,
 * then use the raw_offset directly.
 */
__device__ static cudf::timestamp_us convert_timestamp_between_timezones(
  cudf::timestamp_us ts,
  int64_t const* writer_trans_begin,
  int64_t const* writer_trans_end,
  int32_t const* writer_offsets_begin,
  int32_t const* writer_offsets_end,
  int32_t writer_raw_offset,
  int64_t const* reader_trans_begin,
  int64_t const* reader_trans_end,
  int32_t const* reader_offsets_begin,
  int32_t const* reader_offsets_end,
  int32_t reader_raw_offset)
{
  constexpr int64_t MICROS_PER_MILLI = 1000L;

  // Floor-divide µs to ms. `duration_cast` truncates toward zero, so a negative timestamp with a
  // non-zero sub-millisecond component would round up by one ms; at a DST gap whose recorded
  // instant is exactly that ms, get_transition_index would then return the post-transition offset
  // (1-hour off-by-one). Same shape of bug as the µs→s path in
  // datetime_utils.cuh::convert_timestamp. GpuTimeZoneDBTest.convertOrcTimezonesOnCPU uses
  // Math.floorDiv/floorMod to match.
  int64_t const epoch_millis = spark_rapids_jni::integer_utils::floor_div(
    static_cast<int64_t>(ts.time_since_epoch().count()), MICROS_PER_MILLI);

  int32_t writer_offset_millis = get_transition_index(writer_trans_begin,
                                                      writer_trans_end,
                                                      epoch_millis,
                                                      writer_offsets_begin,
                                                      writer_offsets_end,
                                                      writer_raw_offset);

  int32_t reader_offset_millis = get_transition_index(reader_trans_begin,
                                                      reader_trans_end,
                                                      epoch_millis,
                                                      reader_offsets_begin,
                                                      reader_offsets_end,
                                                      reader_raw_offset);

  int64_t adjusted_milliseconds    = epoch_millis + (writer_offset_millis - reader_offset_millis);
  int const reader_adjusted_offset = get_transition_index(reader_trans_begin,
                                                          reader_trans_end,
                                                          adjusted_milliseconds,
                                                          reader_offsets_begin,
                                                          reader_offsets_end,
                                                          reader_raw_offset);

  int32_t final_offset_millis = writer_offset_millis - reader_adjusted_offset;
  int64_t const epoch_us      = static_cast<int64_t>(
    cuda::std::chrono::duration_cast<cudf::duration_us>(ts.time_since_epoch()).count());
  int64_t final_result = epoch_us + static_cast<int64_t>(final_offset_millis) * MICROS_PER_MILLI;
  return cudf::timestamp_us{cudf::duration_us{final_result}};
}

struct convert_timezones_functor {
  // writer timezone info
  int64_t const* writer_trans_begin;
  int64_t const* writer_trans_end;
  int32_t const* writer_offsets_begin;
  int32_t const* writer_offsets_end;
  int32_t writer_raw_offset;

  // reader timezone info
  int64_t const* reader_trans_begin;
  int64_t const* reader_trans_end;
  int32_t const* reader_offsets_begin;
  int32_t const* reader_offsets_end;
  int32_t reader_raw_offset;

  __device__ cudf::timestamp_us operator()(cudf::timestamp_us const& timestamp) const
  {
    return convert_timestamp_between_timezones(timestamp,
                                               writer_trans_begin,
                                               writer_trans_end,
                                               writer_offsets_begin,
                                               writer_offsets_end,
                                               writer_raw_offset,
                                               reader_trans_begin,
                                               reader_trans_end,
                                               reader_offsets_begin,
                                               reader_offsets_end,
                                               reader_raw_offset);
  }
};

std::unique_ptr<column> convert_timezones(cudf::column_view const& input,
                                          cudf::table_view const* writer_tz_info_table,
                                          cudf::size_type writer_raw_offset,
                                          cudf::table_view const* reader_tz_info_table,
                                          cudf::size_type reader_raw_offset,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  // Input is from Spark, so it should be TIMESTAMP_MICROSECONDS type
  CUDF_EXPECTS(input.type().id() == cudf::type_id::TIMESTAMP_MICROSECONDS,
               "Input column must be of type TIMESTAMP_MICROSECONDS");
  auto results = cudf::make_timestamp_column(input.type(),
                                             input.size(),
                                             cudf::copy_bitmask(input, stream, mr),
                                             input.null_count(),
                                             stream,
                                             mr);

  int64_t const* writer_trans_begin = [&]() {
    if (writer_tz_info_table != nullptr) {
      return writer_tz_info_table->column(0).begin<int64_t>();
    } else {
      // fixed transition, has no transitions
      return static_cast<int64_t const*>(0);
    }
  }();

  int64_t const* writer_trans_end = [&]() {
    if (writer_tz_info_table != nullptr) {
      return writer_tz_info_table->column(0).end<int64_t>();
    } else {
      // fixed transition, has no transitions
      return static_cast<int64_t const*>(0);
    }
  }();

  int32_t const* writer_offsets_begin = [&]() {
    if (writer_tz_info_table != nullptr) {
      return writer_tz_info_table->column(1).begin<int32_t>();
    } else {
      // fixed transition, has no transitions
      return static_cast<int32_t const*>(0);
    }
  }();

  int32_t const* writer_offsets_end = [&]() {
    if (writer_tz_info_table != nullptr) {
      return writer_tz_info_table->column(1).end<int32_t>();
    } else {
      // fixed transition, has no transitions
      return static_cast<int32_t const*>(0);
    }
  }();

  int64_t const* reader_trans_begin = [&]() {
    if (reader_tz_info_table != nullptr) {
      return reader_tz_info_table->column(0).begin<int64_t>();
    } else {
      // fixed transition, has no transitions
      return static_cast<int64_t const*>(0);
    }
  }();

  int64_t const* reader_trans_end = [&]() {
    if (reader_tz_info_table != nullptr) {
      return reader_tz_info_table->column(0).end<int64_t>();
    } else {
      // fixed transition, has no transitions
      return static_cast<int64_t const*>(0);
    }
  }();

  int32_t const* reader_offsets_begin = [&]() {
    if (reader_tz_info_table != nullptr) {
      return reader_tz_info_table->column(1).begin<int32_t>();
    } else {
      // fixed transition, has no transitions
      return static_cast<int32_t const*>(0);
    }
  }();

  int32_t const* reader_offsets_end = [&]() {
    if (reader_tz_info_table != nullptr) {
      return reader_tz_info_table->column(1).end<int32_t>();
    } else {
      // fixed transition, has no transitions
      return static_cast<int32_t const*>(0);
    }
  }();

  thrust::transform(rmm::exec_policy_nosync(stream),
                    input.begin<cudf::timestamp_us>(),
                    input.end<cudf::timestamp_us>(),
                    results->mutable_view().begin<cudf::timestamp_us>(),
                    convert_timezones_functor{writer_trans_begin,
                                              writer_trans_end,
                                              writer_offsets_begin,
                                              writer_offsets_end,
                                              writer_raw_offset,
                                              reader_trans_begin,
                                              reader_trans_end,
                                              reader_offsets_begin,
                                              reader_offsets_end,
                                              reader_raw_offset});

  return results;
}

// =================== ORC timezones end ===================

}  // namespace

namespace spark_rapids_jni {

std::unique_ptr<column> convert_timestamp(column_view const& input,
                                          table_view const& transitions,
                                          size_type tz_index,
                                          bool to_utc,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  auto const type = input.type().id();

  switch (type) {
    case cudf::type_id::TIMESTAMP_SECONDS:
      return convert_timestamp_tz<cudf::timestamp_s>(
        input, transitions, tz_index, to_utc, stream, mr);
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return convert_timestamp_tz<cudf::timestamp_ms>(
        input, transitions, tz_index, to_utc, stream, mr);
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return convert_timestamp_tz<cudf::timestamp_us>(
        input, transitions, tz_index, to_utc, stream, mr);
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      // Nanoseconds supported for users who need sub-microsecond precision
      // (e.g., when storing the resultant timestamp as string rather than
      // Spark TimestampType).
      return convert_timestamp_tz<cudf::timestamp_ns>(
        input, transitions, tz_index, to_utc, stream, mr);
    default: CUDF_FAIL("Unsupported timestamp unit for timezone conversion");
  }
}

std::unique_ptr<column> convert_timestamp_to_utc(column_view const& input,
                                                 table_view const& transitions,
                                                 size_type tz_index,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{ return convert_timestamp(input, transitions, tz_index, true, stream, mr); }

std::unique_ptr<column> convert_utc_timestamp_to_timezone(column_view const& input,
                                                          table_view const& transitions,
                                                          size_type tz_index,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr)
{ return convert_timestamp(input, transitions, tz_index, false, stream, mr); }

std::unique_ptr<column> convert_timestamp_to_utc(column_view const& input_seconds,
                                                 column_view const& input_microseconds,
                                                 column_view const& invalid,
                                                 column_view const& tz_type,
                                                 column_view const& tz_offset,
                                                 table_view const& transitions,
                                                 column_view const tz_indices,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  return convert_to_utc_with_multiple_timezones(input_seconds,
                                                input_microseconds,
                                                invalid,
                                                tz_type,
                                                tz_offset,
                                                transitions,
                                                tz_indices,
                                                stream,
                                                mr);
}

std::unique_ptr<cudf::column> convert_orc_writer_reader_timezones(
  cudf::column_view const& input,
  cudf::table_view const* writer_tz_info_table,
  cudf::size_type writer_raw_offset,
  cudf::table_view const* reader_tz_info_table,
  cudf::size_type reader_raw_offset,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return convert_timezones(input,
                           writer_tz_info_table,
                           writer_raw_offset,
                           reader_tz_info_table,
                           reader_raw_offset,
                           stream,
                           mr);
}

}  // namespace spark_rapids_jni
