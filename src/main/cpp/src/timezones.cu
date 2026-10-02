/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "integer_utils.cuh"
#include "nvtx_ranges.hpp"
#include "timezones.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/launch>
#include <cuda/std/bit>
#include <cuda/std/chrono>
#include <cuda/std/functional>
#include <cuda/stream>
#include <thrust/binary_search.h>

#include <limits>

using column                   = cudf::column;
using column_device_view       = cudf::column_device_view;
using column_view              = cudf::column_view;
using lists_column_device_view = cudf::lists_column_device_view;
using size_type                = cudf::size_type;
using struct_view              = cudf::struct_view;
using table_view               = cudf::table_view;
using orc_timestamp_kind       = spark_rapids_jni::orc_timestamp_kind;

namespace {

constexpr int64_t MICROS_PER_MILLI  = 1'000;
constexpr int64_t MICROS_PER_SECOND = 1'000'000;
constexpr int64_t MICROS_PER_DAY    = 86'400 * MICROS_PER_SECOND;

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
                          cuda::stream_ref stream,
                          rmm::device_async_resource_ref mr)
{
  // get the fixed transitions
  auto const ft_cdv_ptr = column_device_view::create(
    transitions.column(0), stream, cudf::get_current_device_resource_ref());
  auto const fixed_transitions = lists_column_device_view{*ft_cdv_ptr};

  // get the DST rules
  auto const dst_cdv_ptr = cudf::column_device_view::create(
    transitions.column(1), stream, cudf::get_current_device_resource_ref());
  auto const dst_rules = cudf::lists_column_device_view{*dst_cdv_ptr};

  auto results = cudf::make_timestamp_column(input.type(),
                                             input.size(),
                                             cudf::copy_bitmask(input, stream, mr),
                                             input.null_count(),
                                             stream,
                                             mr);

  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
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
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input_seconds.type().id() == cudf::type_id::INT64,
               "seconds column must be of type INT64");
  CUDF_EXPECTS(input_microseconds.type().id() == cudf::type_id::INT32,
               "microseconds column must be of type INT32");

  // get the fixed transitions
  auto const ft_cdv_ptr = column_device_view::create(
    transitions.column(0), stream, cudf::get_current_device_resource_ref());
  auto const fixed_transitions = lists_column_device_view{*ft_cdv_ptr};

  // get DST rules
  auto const dst_cdv_ptr = cudf::column_device_view::create(
    transitions.column(1), stream, cudf::get_current_device_resource_ref());
  auto const dst_rules = cudf::lists_column_device_view{*dst_cdv_ptr};

  auto result =
    cudf::make_timestamp_column(cudf::data_type{cudf::type_to_id<cudf::timestamp_us>()},
                                input_seconds.size(),
                                cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                0,
                                stream,
                                mr);
  auto null_mask = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                                 input_seconds.size(),
                                                 cudf::mask_state::UNALLOCATED,
                                                 stream,
                                                 cudf::get_current_device_resource_ref());

  thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
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

// ---- Calendar helpers for DST computation on GPU ----
// Consistent with java.util.SimpleTimeZone rule semantics and normalized calendar behavior;
// tests cover all rule modes.

constexpr int32_t MS_PER_SECOND = 1000;
constexpr int32_t MS_PER_MINUTE = 60 * MS_PER_SECOND;
constexpr int32_t MS_PER_HOUR   = 60 * MS_PER_MINUTE;
constexpr int64_t MS_PER_DAY    = 24LL * MS_PER_HOUR;

// DST rule mode constants representing the same four rule categories as SimpleTimeZone
enum dst_rule_mode : int32_t {
  DOM_MODE          = 0,
  DOW_IN_MONTH_MODE = 1,
  DOW_GE_DOM_MODE   = 2,
  DOW_LE_DOM_MODE   = 3
};

// Time mode constants
enum dst_time_mode : int32_t { WALL_TIME = 0, STANDARD_TIME = 1, UTC_TIME = 2 };

struct rule_side {
  int32_t month;
  int32_t day;
  int32_t dow;
  int32_t time;
  int32_t time_mode;
  int32_t mode;
};

/**
 * @brief Day-of-week (1=Sun..7=Sat) for the given epoch day.
 */
__device__ static int32_t day_of_week_1_sun(int64_t epoch_days)
{
  int64_t raw = (epoch_days + 4) % 7;
  if (raw < 0) { raw += 7; }
  return static_cast<int32_t>(raw) + 1;
}

__host__ __device__ constexpr size_t align_up(size_t value, size_t alignment)
{
  return (value + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Compute the day-of-month when a DST rule triggers for the given year and month.
 *
 * Handles the four rule categories supported by SimpleTimeZone:
 * - DOM_MODE: exact day of month
 * - DOW_IN_MONTH_MODE: nth occurrence of dayOfWeek (negative = from end)
 * - DOW_GE_DOM_MODE: first dayOfWeek on or after the given day
 * - DOW_LE_DOM_MODE: last dayOfWeek on or before the given day
 */
__device__ static int32_t compute_rule_day(
  int32_t rule_mode, int32_t rule_day, int32_t rule_dow, int32_t year, int32_t month)
{
  int32_t month_len = spark_rapids_jni::date_time_utils::days_in_month(year, month + 1);

  // Compute day-of-week of the 1st of the month
  int64_t first_of_month_epoch_days =
    spark_rapids_jni::date_time_utils::to_epoch_day(year, month + 1, 1);
  int32_t first_dow = day_of_week_1_sun(first_of_month_epoch_days);

  switch (rule_mode) {
    case DOM_MODE: return rule_day;

    case DOW_IN_MONTH_MODE: {
      if (rule_day > 0) {
        // nth occurrence: 1st=first week, etc.
        int32_t diff = rule_dow - first_dow;
        if (diff < 0) diff += 7;
        return 1 + diff + (rule_day - 1) * 7;
      } else {
        // negative: from end of month. -1 = last occurrence.
        int32_t last_dow = ((first_dow - 1 + (month_len - 1)) % 7) + 1;
        int32_t diff     = last_dow - rule_dow;
        if (diff < 0) diff += 7;
        return month_len - diff + (rule_day + 1) * 7;
      }
    }

    case DOW_GE_DOM_MODE: {
      // First rule_dow on or after rule_day
      int64_t target_epoch = first_of_month_epoch_days + (rule_day - 1);
      int32_t target_dow   = day_of_week_1_sun(target_epoch);
      int32_t diff         = rule_dow - target_dow;
      if (diff < 0) diff += 7;
      return rule_day + diff;
    }

    case DOW_LE_DOM_MODE: {
      // Last rule_dow on or before rule_day
      int64_t target_epoch = first_of_month_epoch_days + (rule_day - 1);
      int32_t target_dow   = day_of_week_1_sun(target_epoch);
      int32_t diff         = target_dow - rule_dow;
      if (diff < 0) diff += 7;
      return rule_day - diff;
    }

    default: return rule_day;
  }
}

/**
 * @brief Compute the UTC millis of a DST transition for a given year.
 *
 * @param year The calendar year.
 * @param side Rule parameters for one side of the DST interval.
 * @param raw_offset_ms The timezone raw offset in ms.
 * @param dst_savings_ms The DST savings in ms (needed for WALL_TIME conversion).
 * @param is_start_rule True for DST start (to determine WALL_TIME adjustment).
 */
__device__ static int64_t compute_transition_utc_ms(int32_t year,
                                                    rule_side const& side,
                                                    int32_t raw_offset_ms,
                                                    int32_t dst_savings_ms,
                                                    bool is_start_rule)
{
  int32_t actual_day = compute_rule_day(side.mode, side.day, side.dow, year, side.month);
  // Rule computation may normalize into an adjacent month. Anchor on the valid first day so
  // to_epoch_day never receives a non-positive or otherwise out-of-range day-of-month.
  int64_t epoch_days =
    spark_rapids_jni::date_time_utils::to_epoch_day(year, side.month + 1, 1) + actual_day - 1;
  int64_t utc_ms = epoch_days * MS_PER_DAY + side.time;

  // Convert from the specified time mode to UTC
  switch (side.time_mode) {
    case WALL_TIME:
      utc_ms -= raw_offset_ms;
      // Wall time during DST-end means DST is still active, subtract savings.
      // Wall time during DST-start means DST is not yet active.
      if (!is_start_rule) { utc_ms -= dst_savings_ms; }
      break;
    case STANDARD_TIME: utc_ms -= raw_offset_ms; break;
    case UTC_TIME: break;
  }

  return utc_ms;
}

// Extract the local calendar year from epoch millis.
__device__ static int32_t millis_to_year(int64_t epoch_ms)
{
  auto const epoch_day = spark_rapids_jni::integer_utils::floor_div(epoch_ms, MS_PER_DAY);
  int year;
  int month;
  int day;
  spark_rapids_jni::date_time_utils::to_date(static_cast<int32_t>(epoch_day), year, month, day);
  return year;
}

/**
 * @brief Compute the total UTC offset (raw + DST) for a UTC timestamp using DST rules.
 *
 * Computes an offset consistent with java.util.SimpleTimeZone rule semantics on GPU.
 * It computes the DST start and end transitions for the year containing the
 * timestamp, then checks if the timestamp falls within the DST window.
 *
 * Handles both Northern Hemisphere (start < end) and Southern Hemisphere
 * (start > end, i.e., DST spans year boundary).
 */
__device__ static int32_t compute_dst_offset(int64_t utc_ms,
                                             int32_t raw_offset_ms,
                                             spark_rapids_jni::dst_rule const& rule)
{
  if (!rule.has_dst) { return raw_offset_ms; }

  // Use raw offset only to avoid circular DST-year computation, consistent with
  // SimpleTimeZone's documented offset semantics.
  int32_t year = millis_to_year(utc_ms + raw_offset_ms);

  // Compute DST-on and DST-off transitions in UTC for this year
  rule_side const start_rule{rule.start_month,
                             rule.start_day,
                             rule.start_dow,
                             rule.start_time,
                             rule.start_time_mode,
                             rule.start_mode};
  rule_side const end_rule{
    rule.end_month, rule.end_day, rule.end_dow, rule.end_time, rule.end_time_mode, rule.end_mode};
  int64_t dst_start =
    compute_transition_utc_ms(year, start_rule, raw_offset_ms, rule.dst_savings, true);
  int64_t dst_end =
    compute_transition_utc_ms(year, end_rule, raw_offset_ms, rule.dst_savings, false);

  bool in_dst;
  if (dst_start < dst_end) {
    // Northern Hemisphere: DST is [start, end)
    in_dst = (utc_ms >= dst_start && utc_ms < dst_end);
  } else {
    // Southern Hemisphere: DST is [start, year_end) ∪ [year_start, end)
    in_dst = (utc_ms >= dst_start || utc_ms < dst_end);
  }

  return in_dst ? raw_offset_ms + rule.dst_savings : raw_offset_ms;
}

/**
 * @brief Get the offset for a UTC time using the transition table + DST rule fallback.
 *
 * For timestamps within the transition table range, uses binary search.
 * For timestamps beyond the table, uses DST rule computation.
 * For timestamps before the first recorded transition, falls back to the
 * historical initial offset to match java.util.TimeZone behavior.
 */
struct tz_side_info {
  int64_t const* trans_begin;
  int64_t const* trans_end;
  int32_t const* offsets_begin;
  int32_t initial_offset;
  int32_t raw_offset;
  spark_rapids_jni::dst_rule dst;
  bool is_fixed;
};

struct orc_base_offset_info {
  int64_t us;
  bool is_second_aligned;
};

[[nodiscard]] orc_base_offset_info make_orc_base_offset_info(int64_t us)
{
  return orc_base_offset_info{us, (us % MICROS_PER_SECOND) == 0};
}

struct orc_tz_side_kernel_args {
  int64_t const* __restrict__ trans;
  int32_t const* __restrict__ offsets;
  int32_t trans_count;
  int32_t initial_offset;
  int32_t raw_offset;
  spark_rapids_jni::dst_rule dst;
  bool is_fixed;
};

[[nodiscard]] orc_tz_side_kernel_args make_orc_tz_side_kernel_args(
  spark_rapids_jni::orc_tz_side const& side)
{
  auto const* table = side.tz_info_table;
  auto const count  = table ? table->column(0).size() : 0;
  return {.trans          = table ? table->column(0).begin<int64_t>() : nullptr,
          .offsets        = table ? table->column(1).begin<int32_t>() : nullptr,
          .trans_count    = count,
          .initial_offset = side.initial_offset,
          .raw_offset     = side.raw_offset,
          .dst            = side.dst,
          .is_fixed       = count == 0 && !side.dst.has_dst};
}

__device__ static int32_t get_transition_index(int64_t time_ms, tz_side_info const& side)
{
  if (side.trans_begin == side.trans_end) {
    // No transition table. Use DST rule if available, else fixed offset.
    return compute_dst_offset(time_ms, side.raw_offset, side.dst);
  }

  // upper_bound returns the first element strictly greater than time_ms, so
  // *iter > time_ms and the index we want is iter - 1.
  auto const iter = thrust::upper_bound(thrust::seq, side.trans_begin, side.trans_end, time_ms);
  if (iter == side.trans_end) {
    // Beyond the transition table -- use DST rule for future dates
    return compute_dst_offset(time_ms, side.raw_offset, side.dst);
  }

  int32_t index = static_cast<int32_t>(cuda::std::distance(side.trans_begin, iter));
  if (index == 0) {
    // Before the first recorded transition, java.util.TimeZone uses the
    // historical offset in effect before that transition, not the future rule.
    return side.initial_offset;
  }

  return side.offsets_begin[index - 1];
}

/**
 * @brief Apply the ORC 2015 writer base offset while preserving Apache's negative timestamp borrow.
 *
 * ORC stores a timestamp as seconds relative to 2015-01-01 in the writer timezone plus a
 * non-negative nanos field. For a pre-epoch timestamp with a fractional part, the Apache writer
 * truncates the seconds toward zero. On read, Apache reconstructs the writer-specific 2015 epoch,
 * then borrows one second when the resulting seconds are negative and the encoded nanos contribute
 * at least one millisecond.
 *
 * With `ignoreTimezoneInStripeFooter=true`, cuDF instead uses the UTC 2015 epoch when deciding
 * whether to borrow. The later writer 2015 base-offset adjustment can move the timestamp across
 * the Unix epoch, making cuDF's earlier borrow decision incorrect. This function first reconstructs
 * the value before cuDF's borrow, applies the writer-specific 2015 base offset, and then makes the
 * borrow decision in the same frame as Apache. It can therefore both add a missing borrow and undo
 * one that cuDF applied before the sign changed.
 *
 * For example, the Asia/Shanghai reproducer from nvidia/cudf#21993 has:
 *
 * @code{.pseudo}
 *   decoded_us     =  21'087'883'873  // cuDF used the UTC 2015 epoch; no borrow
 *   writer_2015_year_base_offset_us =  28'800'000'000  // Shanghai is UTC+08:00 at the ORC epoch
 *   unborrowed_us  =  -7'712'116'127  // negative after using the writer-specific epoch
 *   result_us      =  -7'713'116'127  // Apache-compatible one-second borrow
 * @endcode
 *
 * The one-millisecond threshold matches cuDF's ORC decoder and the Apache writer's nanos encoding.
 */
__device__ static int64_t apply_orc_base_offset(int64_t decoded_us,
                                                orc_base_offset_info base_offset)
{
  if (base_offset.us == 0) { return decoded_us; }

  // ORC timezone base offsets are second-aligned. For an arbitrary microsecond offset, the
  // original nanos field cannot be reconstructed reliably, so retain the plain offset behavior.
  if (!base_offset.is_second_aligned) { return decoded_us - base_offset.us; }

  auto const plain_adjusted_us = decoded_us - base_offset.us;
  if (decoded_us >= 0 && plain_adjusted_us >= 0) { return plain_adjusted_us; }
  if (decoded_us < 0 && plain_adjusted_us < -MICROS_PER_SECOND) { return plain_adjusted_us; }

  auto const fractional_us =
    spark_rapids_jni::integer_utils::floor_mod(decoded_us, MICROS_PER_SECOND);

  bool const has_borrowable_fraction = fractional_us >= MICROS_PER_MILLI;
  bool const cudf_applied_borrow     = decoded_us < 0 && has_borrowable_fraction;

  auto const unborrowed_us = decoded_us + (cudf_applied_borrow ? MICROS_PER_SECOND : int64_t{0});
  auto const adjusted_unborrowed_us = unborrowed_us - base_offset.us;
  bool const apache_applies_borrow  = adjusted_unborrowed_us < 0 && has_borrowable_fraction;

  return adjusted_unborrowed_us - (apache_applies_borrow ? MICROS_PER_SECOND : int64_t{0});
}

/**
 * @brief Convert a timestamp between ORC writer and reader timezones.
 *
 * Matches Apache ORC's read order: first reconstruct the timestamp using the writer timezone's
 * 2015 base instant and apply Apache's negative nanos borrow, then run the equivalent of
 * org.apache.orc.impl.SerializationUtils.convertBetweenTimezones. The first step is required for
 * fixed, transition-table, and DST writers because ORC's 2015 base offset can include historical
 * or DST-specific offset state.
 *
 * Optimized for common cases:
 * - Fixed-offset reader (e.g. UTC): skip all reader lookups, use constant offset.
 * - Fixed-offset writer: skip writer lookups.
 */
__device__ static cudf::timestamp_us convert_timestamp_between_timezones(
  cudf::timestamp_us ts,
  orc_base_offset_info writer_2015_year_base_offset,
  tz_side_info const& writer,
  tz_side_info const& reader,
  bool writer_reader_rules_differ)
{
  int64_t const decoded_us = static_cast<int64_t>(
    cuda::std::chrono::duration_cast<cudf::duration_us>(ts.time_since_epoch()).count());
  int64_t const adjusted_us = apply_orc_base_offset(decoded_us, writer_2015_year_base_offset);
  if (!writer_reader_rules_differ) { return cudf::timestamp_us{cudf::duration_us{adjusted_us}}; }

  // Floor-divide to get epoch millis (handles negative timestamps correctly)
  int64_t const epoch_millis =
    spark_rapids_jni::integer_utils::floor_div(adjusted_us, MICROS_PER_MILLI);

  int32_t writer_offset_millis =
    writer.is_fixed ? writer.raw_offset : get_transition_index(epoch_millis, writer);
  int32_t reader_offset_millis =
    reader.is_fixed ? reader.raw_offset : get_transition_index(epoch_millis, reader);

  int64_t adjusted_milliseconds = epoch_millis + (writer_offset_millis - reader_offset_millis);

  int32_t reader_adjusted_offset =
    reader.is_fixed ? reader.raw_offset : get_transition_index(adjusted_milliseconds, reader);

  int32_t final_offset_millis = writer_offset_millis - reader_adjusted_offset;
  int64_t final_result = adjusted_us + static_cast<int64_t>(final_offset_millis) * MICROS_PER_MILLI;
  return cudf::timestamp_us{cudf::duration_us{final_result}};
}

// Max transition entries per timezone that can be loaded into shared memory.
// Each entry = 8 bytes (transition) + 4 bytes (offset) = 12 bytes.
// With writer + reader at max: 2 * 512 * 12 = 12KB, well within 48KB limit.
constexpr int32_t MAX_SMEM_TRANSITIONS  = 512;
constexpr int32_t CONVERT_TZ_BLOCK_SIZE = 256;

[[nodiscard]] size_t add_staged_transition_smem_bytes(size_t smem_bytes, int32_t trans_count)
{
  if (trans_count <= 0 || trans_count > MAX_SMEM_TRANSITIONS) { return smem_bytes; }
  return align_up(smem_bytes, alignof(int64_t)) +
         static_cast<size_t>(trans_count) * (sizeof(int64_t) + sizeof(int32_t));
}

void validate_timezone_table(cudf::table_view const* table)
{
  if (table == nullptr) { return; }

  CUDF_EXPECTS(table->num_columns() == 2, "Timezone table must have exactly 2 columns");
  CUDF_EXPECTS(table->column(0).type().id() == cudf::type_id::INT64 &&
                 table->column(1).type().id() == cudf::type_id::INT32,
               "Timezone table columns must be INT64 transitions and INT32 offsets");
  CUDF_EXPECTS(table->column(0).size() == table->column(1).size(),
               "Transition and offset columns must have the same size");
}

// Stage one side's transition table into shared memory (if it fits) and resolve the
// begin/end/offsets pointers to use (shared or global), advancing `ptr` past this side's
// storage. Null transition tables are only reachable for fixed-offset timezones.
__device__ static void stage_side_transitions(int64_t const* __restrict__ g_trans,
                                              int32_t const* __restrict__ g_offsets,
                                              int32_t trans_count,
                                              char*& ptr,
                                              int64_t const*& begin,
                                              int64_t const*& end,
                                              int32_t const*& offsets_begin)
{
  bool const fits    = trans_count <= MAX_SMEM_TRANSITIONS;
  int64_t* s_trans   = nullptr;
  int32_t* s_offsets = nullptr;

  if (fits && trans_count > 0) {
    ptr =
      cuda::std::bit_cast<char*>(align_up(cuda::std::bit_cast<uintptr_t>(ptr), alignof(int64_t)));
    s_trans = reinterpret_cast<int64_t*>(ptr);
    ptr += trans_count * sizeof(int64_t);
    s_offsets = reinterpret_cast<int32_t*>(ptr);
    ptr += trans_count * sizeof(int32_t);
  }
  for (int32_t i = threadIdx.x; i < trans_count && fits; i += blockDim.x) {
    s_trans[i]   = g_trans[i];
    s_offsets[i] = g_offsets[i];
  }

  begin         = g_trans ? (fits ? s_trans : g_trans) : nullptr;
  end           = begin ? begin + trans_count : nullptr;
  offsets_begin = g_trans ? (fits ? s_offsets : g_offsets) : nullptr;
}

CUDF_KERNEL void __launch_bounds__(CONVERT_TZ_BLOCK_SIZE)
  convert_timezones_kernel(cudf::timestamp_us const* __restrict__ input,
                           cudf::bitmask_type const* __restrict__ null_mask,
                           cudf::timestamp_us* __restrict__ output,
                           cudf::size_type num_rows,
                           cudf::size_type input_offset,
                           orc_base_offset_info writer_2015_year_base_offset,
                           orc_tz_side_kernel_args writer_args,
                           orc_tz_side_kernel_args reader_args,
                           bool writer_reader_rules_differ)
{
  // Shared memory layout: writer transitions, writer offsets, reader transitions, reader offsets
  extern __shared__ char smem[];

  char* ptr               = smem;
  int64_t const *wt_begin = nullptr, *wt_end = nullptr, *rt_begin = nullptr, *rt_end = nullptr;
  int32_t const *wo_begin = nullptr, *ro_begin = nullptr;
  if (writer_reader_rules_differ) {
    stage_side_transitions(writer_args.trans,
                           writer_args.offsets,
                           writer_args.trans_count,
                           ptr,
                           wt_begin,
                           wt_end,
                           wo_begin);
    stage_side_transitions(reader_args.trans,
                           reader_args.offsets,
                           reader_args.trans_count,
                           ptr,
                           rt_begin,
                           rt_end,
                           ro_begin);
    __syncthreads();
  }

  cudf::size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_rows) {
    if (null_mask && !cudf::bit_is_set(null_mask, idx + input_offset)) { return; }
    tz_side_info const writer{wt_begin,
                              wt_end,
                              wo_begin,
                              writer_args.initial_offset,
                              writer_args.raw_offset,
                              writer_args.dst,
                              writer_args.is_fixed};
    tz_side_info const reader{rt_begin,
                              rt_end,
                              ro_begin,
                              reader_args.initial_offset,
                              reader_args.raw_offset,
                              reader_args.dst,
                              reader_args.is_fixed};
    output[idx] = convert_timestamp_between_timezones(
      input[idx], writer_2015_year_base_offset, writer, reader, writer_reader_rules_differ);
  }
}

std::unique_ptr<column> convert_timezones(cudf::column_view const& input,
                                          int64_t writer_2015_year_base_offset_us,
                                          spark_rapids_jni::orc_tz_side writer,
                                          spark_rapids_jni::orc_tz_side reader,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr,
                                          bool writer_reader_rules_differ)
{
  SRJ_FUNC_RANGE();

  CUDF_EXPECTS(input.type().id() == cudf::type_id::TIMESTAMP_MICROSECONDS,
               "Input column must be of type TIMESTAMP_MICROSECONDS");
  validate_timezone_table(writer.tz_info_table);
  validate_timezone_table(reader.tz_info_table);

  auto results = cudf::make_timestamp_column(input.type(),
                                             input.size(),
                                             cudf::copy_bitmask(input, stream, mr),
                                             input.null_count(),
                                             stream,
                                             mr);

  if (input.size() == 0) { return results; }

  auto const writer_args = make_orc_tz_side_kernel_args(writer);
  auto const reader_args = make_orc_tz_side_kernel_args(reader);

  size_t smem_bytes = 0;
  if (writer_reader_rules_differ) {
    smem_bytes = add_staged_transition_smem_bytes(smem_bytes, writer_args.trans_count);
    smem_bytes = add_staged_transition_smem_bytes(smem_bytes, reader_args.trans_count);
  }

  int32_t num_blocks = cudf::util::div_rounding_up_safe(input.size(), CONVERT_TZ_BLOCK_SIZE);
  auto const writer_2015_year_base_offset =
    make_orc_base_offset_info(writer_2015_year_base_offset_us);

  auto const launch_config = cuda::make_config(cuda::grid_dims(num_blocks),
                                               cuda::block_dims<CONVERT_TZ_BLOCK_SIZE>(),
                                               cuda::dynamic_shared_memory<char[]>(smem_bytes));
  cuda::launch(stream.get(),
               launch_config,
               convert_timezones_kernel,
               input.begin<cudf::timestamp_us>(),
               input.null_mask(),
               results->mutable_view().begin<cudf::timestamp_us>(),
               input.size(),
               input.offset(),
               writer_2015_year_base_offset,
               writer_args,
               reader_args,
               writer_reader_rules_differ);
  CUDF_CHECK_CUDA(stream.get());

  return results;
}

__device__ static int64_t wrapping_subtract(int64_t lhs, int64_t rhs)
{
  return static_cast<int64_t>(static_cast<uint64_t>(lhs) - static_cast<uint64_t>(rhs));
}

__device__ static int64_t wrapping_add(int64_t lhs, int64_t rhs)
{
  return static_cast<int64_t>(static_cast<uint64_t>(lhs) + static_cast<uint64_t>(rhs));
}

template <typename T>
__device__ T convert_orc_from_utc_value(T value, tz_side_info const& reader)
{
  constexpr auto ticks_per_milli = T::duration::period::den / 1'000;
  auto const ticks               = value.time_since_epoch().count();
  auto offset_ms                 = reader.raw_offset;
  if (!reader.is_fixed) {
    auto const value_ms = spark_rapids_jni::integer_utils::floor_div(ticks, ticks_per_milli);
    auto const offset_lookup_ms = wrapping_subtract(value_ms, reader.raw_offset);
    offset_ms                   = get_transition_index(offset_lookup_ms, reader);
  }
  auto const result = wrapping_subtract(ticks, static_cast<int64_t>(offset_ms) * ticks_per_milli);
  return T{typename T::duration{result}};
}

/**
 * Convert an instant to the local wall-clock fields exposed by java.util.TimeZone.
 *
 * Spark's legacy timestamp rebase first extracts local fields from a java.sql.Timestamp with a
 * java.util.Calendar, then binds those fields with java.time.ZoneRules. Keeping this as a separate
 * step is important: treating the instant itself as a local timestamp selects the wrong side of
 * historical transitions such as America/New_York's 1883 LMT-to-EST change.
 */
__device__ static cudf::timestamp_us convert_orc_instant_to_local_value(cudf::timestamp_us value,
                                                                        tz_side_info const& reader)
{
  auto const value_us = value.time_since_epoch().count();
  auto const value_ms = spark_rapids_jni::integer_utils::floor_div(value_us, MICROS_PER_MILLI);
  auto const offset_ms =
    reader.is_fixed ? reader.raw_offset : get_transition_index(value_ms, reader);
  auto const result_us = wrapping_add(value_us, static_cast<int64_t>(offset_ms) * MICROS_PER_MILLI);
  return cudf::timestamp_us{cudf::duration_us{result_us}};
}

__device__ static cudf::timestamp_us convert_historical_local_to_spark(
  cudf::timestamp_us local_timestamp,
  lists_column_device_view const& java_time_fixed_transitions,
  lists_column_device_view const& java_time_dst_rules,
  cudf::size_type java_time_tz_index)
{
  return spark_rapids_jni::convert_timestamp<cudf::timestamp_us, true>(
    local_timestamp, java_time_fixed_transitions, java_time_dst_rules, java_time_tz_index, true);
}

/**
 * Return the later offset when local_timestamp is inside a fixed java.time overlap.
 *
 * The timezone table stores an overlap transition at its upper local boundary with the offset
 * after the transition. The preceding row contains the offset before the transition, so the
 * ambiguous interval is [transition - (offset_before - offset_after), transition).
 */
__device__ static bool get_later_java_time_overlap_offset(
  cudf::timestamp_us local_timestamp,
  lists_column_device_view const& java_time_fixed_transitions,
  cudf::size_type java_time_tz_index,
  int32_t& later_offset_seconds)
{
  auto const tz_transitions =
    cudf::list_device_view{java_time_fixed_transitions, java_time_tz_index};
  auto const list_size = tz_transitions.size();
  if (list_size < 2) { return false; }

  auto const tz_instants      = java_time_fixed_transitions.child().child(1);
  auto const utc_offsets      = java_time_fixed_transitions.child().child(2);
  auto const transition_times = cudf::device_span<int64_t const>(
    tz_instants.data<int64_t>() + tz_transitions.element_offset(0), static_cast<size_t>(list_size));
  auto const local_seconds = spark_rapids_jni::integer_utils::floor_div(
    local_timestamp.time_since_epoch().count(), MICROS_PER_SECOND);
  auto const next_transition = thrust::upper_bound(
    thrust::seq, transition_times.begin(), transition_times.end(), local_seconds);
  auto const next_index =
    static_cast<cudf::size_type>(cuda::std::distance(transition_times.begin(), next_transition));
  if (next_index == list_size) { return false; }

  auto const offset_before =
    utc_offsets.element<int32_t>(tz_transitions.element_offset(next_index - 1));
  auto const offset_after = utc_offsets.element<int32_t>(tz_transitions.element_offset(next_index));
  if (offset_after >= offset_before) { return false; }

  auto const overlap_start = *next_transition - (offset_before - offset_after);
  if (local_seconds < overlap_start) { return false; }

  later_offset_seconds = offset_after;
  return true;
}

/**
 * Match Spark's overlap selection in RebaseDateTime.rebaseJulianToGregorianMicros.
 *
 * Spark advances the original java.util.Calendar by one local day. If the legacy timezone state
 * remains unchanged, the original instant belongs to the later side of the java.time overlap.
 * ORC's convertFromUtc operation maps the next local day back to an instant so the offset lookup
 * follows the same wall-clock-day behavior across DST changes.
 */
__device__ static bool legacy_offset_is_stable_for_next_local_day(cudf::timestamp_us orc_timestamp,
                                                                  cudf::timestamp_us reader_local,
                                                                  tz_side_info const& reader)
{
  auto const orc_timestamp_ms = spark_rapids_jni::integer_utils::floor_div(
    orc_timestamp.time_since_epoch().count(), MICROS_PER_MILLI);
  auto const current_offset =
    reader.is_fixed ? reader.raw_offset : get_transition_index(orc_timestamp_ms, reader);

  auto const next_local_us = wrapping_add(reader_local.time_since_epoch().count(), MICROS_PER_DAY);
  auto const next_local    = cudf::timestamp_us{cudf::duration_us{next_local_us}};
  auto const next_instant  = convert_orc_from_utc_value(next_local, reader);
  auto const next_instant_ms = spark_rapids_jni::integer_utils::floor_div(
    next_instant.time_since_epoch().count(), MICROS_PER_MILLI);
  auto const next_offset =
    reader.is_fixed ? reader.raw_offset : get_transition_index(next_instant_ms, reader);
  return current_offset == next_offset;
}

__device__ static cudf::timestamp_us convert_historical_orc_instant_to_spark(
  cudf::timestamp_us orc_timestamp,
  tz_side_info const& reader,
  lists_column_device_view const& java_time_fixed_transitions,
  lists_column_device_view const& java_time_dst_rules,
  cudf::size_type java_time_tz_index)
{
  auto const reader_local = convert_orc_instant_to_local_value(orc_timestamp, reader);
  int32_t later_offset_seconds;
  if (get_later_java_time_overlap_offset(
        reader_local, java_time_fixed_transitions, java_time_tz_index, later_offset_seconds) &&
      legacy_offset_is_stable_for_next_local_day(orc_timestamp, reader_local, reader)) {
    auto const later_offset_us = static_cast<int64_t>(later_offset_seconds) * MICROS_PER_SECOND;
    auto const result_us =
      wrapping_subtract(reader_local.time_since_epoch().count(), later_offset_us);
    return cudf::timestamp_us{cudf::duration_us{result_us}};
  }
  return convert_historical_local_to_spark(
    reader_local, java_time_fixed_transitions, java_time_dst_rules, java_time_tz_index);
}

template <typename T>
CUDF_KERNEL void __launch_bounds__(CONVERT_TZ_BLOCK_SIZE)
  convert_orc_from_utc_kernel(T const* __restrict__ input,
                              cudf::bitmask_type const* __restrict__ null_mask,
                              T* __restrict__ output,
                              cudf::size_type num_rows,
                              cudf::size_type input_offset,
                              orc_tz_side_kernel_args reader_args)
{
  extern __shared__ char smem[];
  char* ptr               = smem;
  int64_t const *rt_begin = nullptr, *rt_end = nullptr;
  int32_t const* ro_begin = nullptr;
  if (!reader_args.is_fixed) {
    stage_side_transitions(reader_args.trans,
                           reader_args.offsets,
                           reader_args.trans_count,
                           ptr,
                           rt_begin,
                           rt_end,
                           ro_begin);
    __syncthreads();
  }

  cudf::size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_rows) {
    if (null_mask && !cudf::bit_is_set(null_mask, idx + input_offset)) { return; }
    tz_side_info const reader{rt_begin,
                              rt_end,
                              ro_begin,
                              reader_args.initial_offset,
                              reader_args.raw_offset,
                              reader_args.dst,
                              reader_args.is_fixed};
    output[idx] = convert_orc_from_utc_value(input[idx], reader);
  }
}

template <typename T>
std::unique_ptr<column> convert_orc_from_utc_typed(cudf::column_view const& input,
                                                   spark_rapids_jni::orc_tz_side reader,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  auto results = cudf::make_fixed_width_column(input.type(),
                                               input.size(),
                                               cudf::copy_bitmask(input, stream, mr),
                                               input.null_count(),
                                               stream,
                                               mr);
  if (input.size() == 0) { return results; }

  auto const reader_args = make_orc_tz_side_kernel_args(reader);
  auto const smem_bytes  = add_staged_transition_smem_bytes(size_t{0}, reader_args.trans_count);

  int32_t num_blocks       = cudf::util::div_rounding_up_safe(input.size(), CONVERT_TZ_BLOCK_SIZE);
  auto const launch_config = cuda::make_config(cuda::grid_dims(num_blocks),
                                               cuda::block_dims<CONVERT_TZ_BLOCK_SIZE>(),
                                               cuda::dynamic_shared_memory<char[]>(smem_bytes));
  cuda::launch(stream.get(),
               launch_config,
               convert_orc_from_utc_kernel<T>,
               input.begin<T>(),
               input.null_mask(),
               results->mutable_view().begin<T>(),
               input.size(),
               input.offset(),
               reader_args);
  CUDF_CHECK_CUDA(stream.get());
  return results;
}

void validate_java_time_table(cudf::table_view const* table, cudf::size_type tz_index)
{
  CUDF_EXPECTS(table != nullptr, "java.time timezone table is required for historical rebasing");
  CUDF_EXPECTS(table->num_columns() == 2, "java.time timezone table must have exactly 2 columns");
  CUDF_EXPECTS(table->column(0).type().id() == cudf::type_id::LIST &&
                 table->column(1).type().id() == cudf::type_id::LIST,
               "java.time timezone table columns must be LIST columns");
  CUDF_EXPECTS(tz_index >= 0 && tz_index < table->num_rows(),
               "java.time timezone index is out of range");
}

template <orc_timestamp_kind input_kind>
CUDF_KERNEL void __launch_bounds__(CONVERT_TZ_BLOCK_SIZE)
  convert_orc_to_spark_kernel(cudf::timestamp_us const* __restrict__ input,
                              cudf::bitmask_type const* __restrict__ null_mask,
                              cudf::timestamp_us* __restrict__ output,
                              cudf::size_type num_rows,
                              cudf::size_type input_offset,
                              orc_base_offset_info writer_2015_year_base_offset,
                              orc_tz_side_kernel_args writer_args,
                              orc_tz_side_kernel_args reader_args,
                              bool writer_reader_rules_differ,
                              lists_column_device_view java_time_fixed_transitions,
                              lists_column_device_view java_time_dst_rules,
                              cudf::size_type java_time_tz_index,
                              int64_t reader_historical_difference_end_utc_us,
                              int64_t reader_historical_difference_end_local_us)
{
  extern __shared__ char smem[];

  char* ptr               = smem;
  int64_t const *wt_begin = nullptr, *wt_end = nullptr, *rt_begin = nullptr, *rt_end = nullptr;
  int32_t const *wo_begin = nullptr, *ro_begin = nullptr;
  if constexpr (input_kind == orc_timestamp_kind::PHYSICAL) {
    if (writer_reader_rules_differ) {
      stage_side_transitions(writer_args.trans,
                             writer_args.offsets,
                             writer_args.trans_count,
                             ptr,
                             wt_begin,
                             wt_end,
                             wo_begin);
    }
  }
  if (!reader_args.is_fixed) {
    stage_side_transitions(reader_args.trans,
                           reader_args.offsets,
                           reader_args.trans_count,
                           ptr,
                           rt_begin,
                           rt_end,
                           ro_begin);
  }
  // Staging advances ptr only when it writes shared memory. This condition is block-uniform.
  if (ptr != smem) { __syncthreads(); }

  auto const idx = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= num_rows) { return; }
  if (null_mask && !cudf::bit_is_set(null_mask, idx + input_offset)) { return; }

  tz_side_info const writer{wt_begin,
                            wt_end,
                            wo_begin,
                            writer_args.initial_offset,
                            writer_args.raw_offset,
                            writer_args.dst,
                            writer_args.is_fixed};
  tz_side_info const reader{rt_begin,
                            rt_end,
                            ro_begin,
                            reader_args.initial_offset,
                            reader_args.raw_offset,
                            reader_args.dst,
                            reader_args.is_fixed};

  if constexpr (input_kind != orc_timestamp_kind::LOCAL) {
    auto orc_timestamp = input[idx];
    if constexpr (input_kind == orc_timestamp_kind::PHYSICAL) {
      orc_timestamp = convert_timestamp_between_timezones(
        orc_timestamp, writer_2015_year_base_offset, writer, reader, writer_reader_rules_differ);
    }
    if (orc_timestamp.time_since_epoch().count() < reader_historical_difference_end_utc_us) {
      output[idx] = convert_historical_orc_instant_to_spark(orc_timestamp,
                                                            reader,
                                                            java_time_fixed_transitions,
                                                            java_time_dst_rules,
                                                            java_time_tz_index);
    } else {
      output[idx] = orc_timestamp;
    }
  } else {
    auto const local_timestamp = input[idx];
    // Spark rebases the java.sql.Timestamp instant produced by ORC, not the original local value.
    // Preserve ORC's offset choice so an ambiguous local time keeps the same overlap occurrence.
    auto const orc_timestamp = convert_orc_from_utc_value(local_timestamp, reader);
    output[idx] =
      local_timestamp.time_since_epoch().count() < reader_historical_difference_end_local_us
        ? convert_historical_orc_instant_to_spark(orc_timestamp,
                                                  reader,
                                                  java_time_fixed_transitions,
                                                  java_time_dst_rules,
                                                  java_time_tz_index)
        : orc_timestamp;
  }
}

template <orc_timestamp_kind input_kind>
std::unique_ptr<column> convert_orc_to_spark_typed(
  cudf::column_view const& input,
  int64_t writer_2015_year_base_offset_us,
  spark_rapids_jni::orc_tz_side writer,
  spark_rapids_jni::orc_tz_side reader,
  cudf::table_view const& java_time_info,
  cudf::size_type java_time_tz_index,
  int64_t reader_historical_difference_end_utc_us,
  int64_t reader_historical_difference_end_local_us,
  bool writer_reader_rules_differ,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto results = cudf::make_timestamp_column(input.type(),
                                             input.size(),
                                             cudf::copy_bitmask(input, stream, mr),
                                             input.null_count(),
                                             stream,
                                             mr);
  if (input.size() == 0 || input.null_count() == input.size()) { return results; }

  auto const java_time_fixed_cdv = column_device_view::create(
    java_time_info.column(0), stream, cudf::get_current_device_resource_ref());
  auto const java_time_dst_cdv = column_device_view::create(
    java_time_info.column(1), stream, cudf::get_current_device_resource_ref());
  auto const java_time_fixed_transitions = lists_column_device_view{*java_time_fixed_cdv};
  auto const java_time_dst_rules         = lists_column_device_view{*java_time_dst_cdv};

  auto const writer_args = make_orc_tz_side_kernel_args(writer);
  auto const reader_args = make_orc_tz_side_kernel_args(reader);

  size_t smem_bytes = 0;
  if constexpr (input_kind == orc_timestamp_kind::PHYSICAL) {
    if (writer_reader_rules_differ) {
      smem_bytes = add_staged_transition_smem_bytes(smem_bytes, writer_args.trans_count);
    }
  }
  if (!reader_args.is_fixed) {
    smem_bytes = add_staged_transition_smem_bytes(smem_bytes, reader_args.trans_count);
  }

  auto const num_blocks    = cudf::util::div_rounding_up_safe(input.size(), CONVERT_TZ_BLOCK_SIZE);
  auto const launch_config = cuda::make_config(cuda::grid_dims(num_blocks),
                                               cuda::block_dims<CONVERT_TZ_BLOCK_SIZE>(),
                                               cuda::dynamic_shared_memory<char[]>(smem_bytes));
  cuda::launch(stream.get(),
               launch_config,
               convert_orc_to_spark_kernel<input_kind>,
               input.begin<cudf::timestamp_us>(),
               input.null_mask(),
               results->mutable_view().begin<cudf::timestamp_us>(),
               input.size(),
               input.offset(),
               make_orc_base_offset_info(writer_2015_year_base_offset_us),
               writer_args,
               reader_args,
               writer_reader_rules_differ,
               java_time_fixed_transitions,
               java_time_dst_rules,
               java_time_tz_index,
               reader_historical_difference_end_utc_us,
               reader_historical_difference_end_local_us);
  CUDF_CHECK_CUDA(stream.get());
  return results;
}

// =================== ORC timezones end ===================

}  // namespace

namespace spark_rapids_jni {

std::unique_ptr<column> convert_timestamp(column_view const& input,
                                          table_view const& transitions,
                                          size_type tz_index,
                                          bool to_utc,
                                          cuda::stream_ref stream,
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
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  return convert_timestamp(input, transitions, tz_index, true, stream, mr);
}

std::unique_ptr<column> convert_utc_timestamp_to_timezone(column_view const& input,
                                                          table_view const& transitions,
                                                          size_type tz_index,
                                                          cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr)
{
  return convert_timestamp(input, transitions, tz_index, false, stream, mr);
}

std::unique_ptr<column> convert_timestamp_to_utc(column_view const& input_seconds,
                                                 column_view const& input_microseconds,
                                                 column_view const& invalid,
                                                 column_view const& tz_type,
                                                 column_view const& tz_offset,
                                                 table_view const& transitions,
                                                 column_view const tz_indices,
                                                 cuda::stream_ref stream,
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
  int64_t writer_2015_year_base_offset_us,
  orc_tz_side writer,
  orc_tz_side reader,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  bool writer_reader_rules_differ)
{
  return convert_timezones(
    input, writer_2015_year_base_offset_us, writer, reader, stream, mr, writer_reader_rules_differ);
}

std::unique_ptr<cudf::column> convert_orc_from_utc(cudf::column_view const& input,
                                                   orc_tz_side reader,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  validate_timezone_table(reader.tz_info_table);
  switch (input.type().id()) {
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return convert_orc_from_utc_typed<cudf::timestamp_ms>(input, reader, stream, mr);
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return convert_orc_from_utc_typed<cudf::timestamp_us>(input, reader, stream, mr);
    default:
      CUDF_FAIL(
        "ORC convertFromUtc input must be TIMESTAMP_MILLISECONDS or TIMESTAMP_MICROSECONDS");
  }
}

std::unique_ptr<cudf::column> convert_orc_to_spark(cudf::column_view const& input,
                                                   int64_t writer_2015_year_base_offset_us,
                                                   orc_tz_side writer,
                                                   orc_tz_side reader,
                                                   cudf::table_view const* java_time_info,
                                                   cudf::size_type java_time_tz_index,
                                                   orc_to_spark_options options,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::TIMESTAMP_MICROSECONDS,
               "ORC-to-Spark input must be TIMESTAMP_MICROSECONDS");
  validate_timezone_table(writer.tz_info_table);
  validate_timezone_table(reader.tz_info_table);

  auto dispatch = [&]<orc_timestamp_kind kind>() {
    auto const historical_difference_end_us = kind == orc_timestamp_kind::LOCAL
                                                ? options.reader_historical_difference_end_local_us
                                                : options.reader_historical_difference_end_utc_us;
    if (historical_difference_end_us == std::numeric_limits<int64_t>::min()) {
      if constexpr (kind == orc_timestamp_kind::PHYSICAL) {
        return convert_timezones(input,
                                 writer_2015_year_base_offset_us,
                                 writer,
                                 reader,
                                 stream,
                                 mr,
                                 options.writer_reader_rules_differ);
      } else if constexpr (kind == orc_timestamp_kind::LOCAL) {
        return convert_orc_from_utc_typed<cudf::timestamp_us>(input, reader, stream, mr);
      } else {
        return std::make_unique<cudf::column>(input, stream, mr);
      }
    }

    validate_java_time_table(java_time_info, java_time_tz_index);
    return convert_orc_to_spark_typed<kind>(input,
                                            writer_2015_year_base_offset_us,
                                            writer,
                                            reader,
                                            *java_time_info,
                                            java_time_tz_index,
                                            options.reader_historical_difference_end_utc_us,
                                            options.reader_historical_difference_end_local_us,
                                            options.writer_reader_rules_differ,
                                            stream,
                                            mr);
  };
  switch (options.input_kind) {
    case orc_timestamp_kind::PHYSICAL:
      return dispatch.template operator()<orc_timestamp_kind::PHYSICAL>();
    case orc_timestamp_kind::LOCAL:
      return dispatch.template operator()<orc_timestamp_kind::LOCAL>();
    case orc_timestamp_kind::INSTANT:
      return dispatch.template operator()<orc_timestamp_kind::INSTANT>();
    default: CUDF_FAIL("Invalid ORC timestamp input kind");
  }
}

std::unique_ptr<cudf::column> convert_orc_writer_reader_timezones(
  cudf::column_view const& input,
  cudf::table_view const* writer_tz_info_table,
  int32_t writer_raw_offset,
  int64_t writer_2015_year_base_offset_us,
  cudf::table_view const* reader_tz_info_table,
  int32_t reader_raw_offset,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  // Java passes the exact ORC 2015 writer base offset, so this path does not infer it from
  // raw_offset. Passing initial_offset == raw_offset preserves the previous before-first-transition
  // behavior for raw-offset-only callers; full DST fallback uses the orc_tz_side overload.
  return convert_orc_writer_reader_timezones(
    input,
    writer_2015_year_base_offset_us,
    spark_rapids_jni::orc_tz_side{writer_tz_info_table,
                                  /*initial_offset=*/writer_raw_offset,
                                  writer_raw_offset},
    spark_rapids_jni::orc_tz_side{reader_tz_info_table,
                                  /*initial_offset=*/reader_raw_offset,
                                  reader_raw_offset},
    stream,
    mr);
}

}  // namespace spark_rapids_jni
