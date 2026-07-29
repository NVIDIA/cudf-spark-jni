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

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace spark_rapids_jni {

/**
 * @brief Convert input column timestamps in current timezone to UTC
 *
 * The transition rules are in enclosed in a table, and the index corresponding to the
 * current timezone is given.
 *
 * This method is the inverse of convert_utc_timestamp_to_timezone.
 *
 * @param input the column of input timestamps in the current timezone
 * @param timezone_info the timezone info table for all timezones,
 * first column is fixed-transitions, second column is dst rules
 * @param tz_index the index of the row in `timezone_info` corresponding to the current timezone
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned timestamp column's memory
 */
std::unique_ptr<cudf::column> convert_timestamp_to_utc(
  cudf::column_view const& input,
  cudf::table_view const& timezone_info,
  cudf::size_type const tz_index,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Convert input column timestamps in UTC to specified timezone
 *
 * The timezone info is in enclosed in a table, and the index corresponding to the
 * specific timezone is given.
 *
 * This method is the inverse of convert_timestamp_to_utc.
 *
 * @param input the column of input timestamps in UTC
 * @param timezone_info the timezone info table for all timezones,
 * first column is fixed-transitions, second column is dst rules
 * @param tz_index the index of the row in `timezone_info` corresponding to the specific timezone
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned timestamp column's memory
 */
std::unique_ptr<cudf::column> convert_utc_timestamp_to_timezone(
  cudf::column_view const& input,
  cudf::table_view const& timezone_info,
  cudf::size_type const tz_index,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Convert timestamps in multiple timezones to UTC.
 * This is used for casting string(with timezone) to timestamp.
 * Note: The input timestamps are split into seconds and microseconds columns to handle special
 * cases: before conversion the timestamp is overflow, but after conversion it is valid.
 *
 * @param input_seconds the seconds column for the input timestamps
 * @param input_microseconds the microseconds column for the input timestamps
 * @param invalid is the timestamp invalid
 * @param tz_type timezone type: fixed offset or other type
 * @param tz_offset timezone offsets, only apply to fixed offset timezone
 * @param timezone_info the timezone info table for all timezones,
 * first column is fixed-transitions, second column is dst rules
 * @param tz_indices the timezone index to transitions, if tz_type is not fixed offset,
 * use this column
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned timestamp column's memory
 *
 * @return a column of timestamps in microseconds
 */
std::unique_ptr<cudf::column> convert_timestamp_to_utc(
  cudf::column_view const& input_seconds,
  cudf::column_view const& input_microseconds,
  cudf::column_view const& invalid,
  cudf::column_view const& tz_type,
  cudf::column_view const& tz_offset,
  cudf::table_view const& timezone_info,
  cudf::column_view const tz_indices,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Recurring DST rule parameters derived from JVM timezone APIs.
 *
 * Used by the GPU kernel to compute timezone offsets for timestamps beyond
 * the historical transition table, using SimpleTimeZone-compatible rule semantics on GPU.
 *
 * Mode values for start_mode/end_mode:
 *   0 = DOM_MODE: exact day of month
 *   1 = DOW_IN_MONTH_MODE: nth dayOfWeek in month
 *   2 = DOW_GE_DOM_MODE: first dayOfWeek on or after day
 *   3 = DOW_LE_DOM_MODE: last dayOfWeek on or before day
 *
 * Time mode values for start_time_mode/end_time_mode:
 *   0 = WALL_TIME, 1 = STANDARD_TIME, 2 = UTC_TIME
 */
struct dst_rule {
  int32_t has_dst{};    // 0 means no DST, just use raw_offset
  int32_t dst_savings;  // in milliseconds (typically 3600000)
  int32_t start_month;  // 0-based (Jan=0..Dec=11)
  int32_t start_day;    // day-of-month or occurrence, depends on start_mode
  int32_t start_dow;    // day-of-week 1=Sun..7=Sat, 0 for DOM_MODE
  int32_t start_time;   // ms within day
  int32_t start_time_mode;
  int32_t start_mode;  // 0=DOM, 1=DOW_IN_MONTH, 2=DOW_GE_DOM, 3=DOW_LE_DOM
  int32_t end_month;
  int32_t end_day;
  int32_t end_dow;
  int32_t end_time;
  int32_t end_time_mode;
  int32_t end_mode;
};

struct orc_tz_side {
  cudf::table_view const* tz_info_table;  // nullptr for fixed-offset TZ
  int32_t initial_offset;                 // historical offset before the first transition
  int32_t raw_offset;                     // standard/raw offset (ms) used for DST fallback
  dst_rule dst{};
};

/**
 * @brief Convert between ORC writer timezone and reader timezone.
 *
 * Uses historical transition table for dates within the table range, and
 * recurring DST rules derived from JVM timezone APIs for dates beyond the table.
 *
 * @param input The input timestamp column in microseconds.
 * @param writer_2015_year_base_offset_us Writer timezone offset at ORC's 2015-01-01 base instant.
 *        ORC applies this base-timestamp adjustment, including any DST savings in effect at that
 *        instant, before running SerializationUtils.convertBetweenTimezones. The native path
 *        applies the same adjustment first so the negative-timestamp nanos borrow is decided in
 *        the same frame as Apache ORC. Pass 0 for no adjustment.
 * @param writer writer timezone transition data, offsets, and DST rule.
 * @param reader reader timezone transition data, offsets, and DST rule.
 * @param stream CUDA stream.
 * @param mr Device memory resource.
 * @return timestamps rebased between writer and reader timezones.
 */
[[nodiscard]] std::unique_ptr<cudf::column> convert_orc_writer_reader_timezones(
  cudf::column_view const& input,
  int64_t writer_2015_year_base_offset_us,
  orc_tz_side writer,
  orc_tz_side reader,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief JNI-compatible raw-offset overload of the ORC conversion.
 *
 * Java computes and passes the writer timezone offset at ORC's 2015-01-01 base instant separately,
 * because that value is not necessarily the raw offset: for DST zones it can include DST savings,
 * and for historical transition zones it can come from the transition table. This overload uses
 * that exact value for the base-timestamp/borrow adjustment, then forwards the transition tables
 * and raw offsets to the shared conversion kernel. Callers that need recurring DST fallback beyond
 * a transition table should use the full `orc_tz_side` overload.
 *
 * @param input The input timestamp column in microseconds.
 * @param writer_tz_info_table transition/offset table, nullptr for fixed-offset TZ.
 * @param writer_raw_offset the raw offset in milliseconds.
 * @param writer_2015_year_base_offset_us writer timezone offset at ORC's 2015-01-01 base instant,
 *        in microseconds.
 * @param reader_tz_info_table transition/offset table, nullptr for fixed-offset TZ.
 * @param reader_raw_offset the raw offset in milliseconds.
 * @param stream CUDA stream.
 * @param mr Device memory resource.
 * @return timestamps rebased between writer and reader timezones.
 */
[[nodiscard]] std::unique_ptr<cudf::column> convert_orc_writer_reader_timezones(
  cudf::column_view const& input,
  cudf::table_view const* writer_tz_info_table,
  int32_t writer_raw_offset,
  int64_t writer_2015_year_base_offset_us,
  cudf::table_view const* reader_tz_info_table,
  int32_t reader_raw_offset,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
