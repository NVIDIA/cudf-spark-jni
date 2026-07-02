/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

#include "cast_string.hpp"
#include "cast_string_to_timestamp_common.hpp"
#include "datetime_utils.cuh"
#include "version.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/functional>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <chrono>
#include <vector>

/**
 * This file is ported from Spark version 3.5.0 SparkDateTimeUtils.scala
 */
namespace spark_rapids_jni {

namespace {

enum class result_type : uint8_t {
  // Parse success
  SUCCESS = 0,

  // Parse failed, has invalid format
  // Throw exception when it's Ansi mode, or return null otherwise.
  INVALID = 1
};

/**
 * If the timestamp string only contains the time part, TS_TYPE is JUST_TIME.
 * E.g.:
 *   For timestamp string T18:01:01 : TS_TYPE = JUST_TIME
 *   For timestamp string 2020-01-01T12:00:00 : TS_TYPE = NOT_JUST_TIME
 */
enum class TS_TYPE : uint8_t { NOT_JUST_TIME = 0, JUST_TIME = 1 };

/**
 * Represents a timezone.
 */
struct time_zone {
  TZ_TYPE type;

  // for fixed timezone
  int fixed_offset;

  // for java.time.ZoneId.SHORT_IDS and Region-based zone IDs
  // the tz offset to the input string
  int tz_pos_in_string;

  // for java.time.ZoneId.SHORT_IDS and Region-based zone IDs
  // the tz end offset to the input string
  int tz_end_pos_in_string;

  __device__ int tz_len() const { return tz_end_pos_in_string - tz_pos_in_string; }

  __device__ time_zone()
    : type(TZ_TYPE::NOT_SPECIFIED), fixed_offset(0), tz_pos_in_string(0), tz_end_pos_in_string(0)
  {
  }
  __device__ time_zone(TZ_TYPE t, int offset, int tz_pos, int tz_end_pos)
    : type(t), fixed_offset(offset), tz_pos_in_string(tz_pos), tz_end_pos_in_string(tz_end_pos)
  {
  }
};

__device__ time_zone make_fixed_tz(int offset)
{
  return time_zone(TZ_TYPE::FIXED_TZ, offset, 0, 0);
}

__device__ time_zone make_invalid_tz() { return time_zone(TZ_TYPE::INVALID_TZ, 0, 0, 0); }

__device__ time_zone make_other_tz(int tz_pos, int tz_end_pos)
{
  return time_zone(TZ_TYPE::OTHER_TZ, 0, tz_pos, tz_end_pos);
}

/**
 * Is white space, consistent with Spark UTF8String.trimAll for char input
 */
__device__ bool is_whitespace(unsigned char const c)
{
  // Keep consistent with Java: Character.isWhitespace(c) ||
  // Character.isISOControl(c),
  // 0-31 is control characters, 32 is space, 127 is delete
  return c <= 32 || c == 127;
}

__device__ bool is_valid_tz(time_zone const& tz) { return tz.type != TZ_TYPE::INVALID_TZ; }

/**
 * Parse a string to an integer.
 * @param ptr The input string
 * @param[out] pos the pointer to the string when parsing
 * @param end_pos the end pointer of the string
 * @param[out] v the parsed value
 * @param min_digits the minimum digits to parse, if not enough, return false
 * @param max_digits the maximum digits to parse, if exceeds, return false
 * @return true if the input string is valid, false otherwise
 */
__device__ bool parse_int(unsigned char const* const ptr,
                          int& pos,
                          int const end_pos,
                          int& v,
                          int min_digits,
                          int max_digits)
{
  v          = 0;
  int digits = 0;
  while (pos < end_pos) {
    int const parsed_value = static_cast<int32_t>(ptr[pos]) - '0';
    if (parsed_value >= 0 && parsed_value <= 9) {
      if (++digits > max_digits) { return false; }
      v = v * 10 + parsed_value;
    } else {
      break;
    }
    pos++;
  }

  return digits >= min_digits;
}

__device__ bool eof(int pos, int end_pos) { return end_pos - pos <= 0; }

__device__ bool parse_char(unsigned char const* const ptr, int& pos, unsigned char const c)
{
  return ptr[pos++] == c;
}

__device__ bool try_parse_char(unsigned char const* const ptr, int& pos, unsigned char const c)
{
  if (ptr[pos] == c) {
    ++pos;
    return true;
  }
  return false;
}

/**
 * Try to parse `max_digits` consecutive digits into a int value.
 * Exits on the first non-digit character or already handled `max_digits` chars.
 * @param ptr The input string
 * @param[out] pos the pointer to the string when parsing
 * @param end_pos the end pointer of the string
 * @param[out] v the parsed value
 * @param max_digits the maximum digits to parse
 * @returns the number of digits parsed.
 */
__device__ int parse_digits(
  unsigned char const* const ptr, int& pos, int const end_pos, int& v, int const max_digits)
{
  v          = 0;
  int digits = 0;
  while (pos < end_pos) {
    int const parsed_value = static_cast<int32_t>(ptr[pos]) - '0';
    if (parsed_value >= 0 && parsed_value <= 9) {
      v = v * 10 + parsed_value;
      ++pos;
      if (++digits == max_digits) { break; }
    } else {
      // meets non-digit
      break;
    }
  }

  return digits;
}

/**
 * Parse timezone from sign part to end of the string.
 * E.g.: +01:02:03, the '+' is parsed, parse the following: 01:02:03
 * Formats:
 * 1. with no colon: [+-]h, [+-]hh, [+-]hhmm, or [+-]hhmmss
 * 2. with colon:
 *   1): Spark 320: [+-]h[h]:mm, [+-]h[h]:mm:ss
 *   2): Spark 321 and 321+: [+-]h[h]:m[m], [+-]h[h]:mm:ss
 */
__device__ time_zone parse_tz_from_sign(
  unsigned char const* const ptr, int& pos, int const end_pos, int const sign, bool is_spark_320)
{
  int hour   = 0;
  int minute = 0;
  int second = 0;

  int h_digits = 0;
  int m_digits = 0;
  int s_digits = 0;

  // parse hour
  h_digits = parse_digits(ptr, pos, end_pos, hour, /*max_digits*/ 2);
  if (h_digits == 0) {
    // parse hour failed, [+-] with no digits following
    return make_invalid_tz();
  } else {
    // parse hour is OK
    if (!eof(pos, end_pos)) {
      // has more after hour
      if (try_parse_char(ptr, pos, ':')) {
        // without colon path, already parsed [+-]h[h]:
        // parse minute
        m_digits = parse_digits(ptr, pos, end_pos, minute, /*max_digits*/ 2);
        if (m_digits == 0 || (is_spark_320 && m_digits == 1)) {
          // [+-]h[h]: without digits following,
          // or Spark 320 does not support [+-]h[h]:m
          return make_invalid_tz();
        } else {
          // [+-]h[h]:m[m]
          if (!eof(pos, end_pos)) {
            if (!(try_parse_char(ptr, pos, ':') &&
                  // parse second
                  (s_digits = parse_digits(ptr,
                                           pos,
                                           end_pos,
                                           second,
                                           /*max_digits*/ 2) == 2) &&
                  eof(pos, end_pos))) {
              // not: [+-]h[h]:m[m]:ss
              return make_invalid_tz();
            }
          }
        }
      } else {
        // without colon path, already parsed [+-]h[h]
        // should be: [+-]hhmm or [+-]hhmmss
        if (h_digits != 2) { return make_invalid_tz(); }

        // parse minute
        m_digits = parse_digits(ptr, pos, end_pos, minute, /*max_digits*/ 2);

        // parse second
        s_digits = parse_digits(ptr, pos, end_pos, second, /*max_digits*/ 2);

        if (!(m_digits == 2 || m_digits == 0) || !(s_digits == 2 || s_digits == 0)) {
          // not: [+-]hhmm or [+-]hhmmss
          return make_invalid_tz();
        }

        if (!eof(pos, end_pos)) { return make_invalid_tz(); }
      }
    } else {
      // got pattern: [+-]h or [+-]hh, it's valid.
    }
  }

  // check hour, minute and second; max offset is 18:00:00
  if (hour > 18 || minute > 59 || second > 59) { return make_invalid_tz(); }
  int num_seconds = hour * 3600 + minute * 60 + second;
  if (num_seconds > 18 * 3600) { return make_invalid_tz(); }

  if (s_digits > 0 && m_digits != 2) {
    // Special invalid case: [+-]h[h]:m:ss
    return make_invalid_tz();
  }

  return make_fixed_tz(sign * num_seconds);
}

__device__ bool try_parse_sign(unsigned char const* const ptr, int& pos, int& sign_value)
{
  unsigned char const sign_c = ptr[pos];
  if (sign_c == '+' || sign_c == '-') {
    ++pos;
    sign_value = (sign_c == '+') ? 1 : -1;
    return true;
  }
  return false;
}

/**
 * Parse timezone starts with U
 * e.g.: UT+08:00, U is parsed, parse the following: T+08:00
 */
__device__ time_zone try_parse_UT_tz(unsigned char const* const ptr,
                                     int& pos,
                                     int const end_pos,
                                     bool is_spark_320)
{
  // pos_backup points to the char 'U'
  int pos_backup = pos - 1;

  if (eof(pos, end_pos)) {
    // U, invalid
    return make_invalid_tz();
  }

  if (try_parse_char(ptr, pos, 'T')) {
    if (eof(pos, end_pos)) {
      // UT, invalid
      return make_fixed_tz(0);
    }

    if (try_parse_char(ptr, pos, 'C')) {
      if (eof(pos, end_pos)) {
        // UTC
        return make_fixed_tz(0);
      }

      // start with UTC, should be UTC[+-]
      int sign_value;
      if (try_parse_sign(ptr, pos, sign_value)) {
        return parse_tz_from_sign(ptr, pos, end_pos, sign_value, is_spark_320);
      } else {
        // e.g.: UTCx, maybe a valid timezone
        return make_other_tz(pos_backup, end_pos);
      }
    } else {
      // start with UT, followed by non 'C', should be UT[+-]
      int sign_value;
      if (try_parse_sign(ptr, pos, sign_value)) {
        return parse_tz_from_sign(ptr, pos, end_pos, sign_value, is_spark_320);
      } else {
        // e.g.: UTx, maybe a valid timezone
        return make_other_tz(pos_backup, end_pos);
      }
    }
  }

  // start with U, followed by non 'T', maybe: US/Pacific
  return make_other_tz(pos_backup, end_pos);
}

/**
 * Parse timezone starts with G, G is parsed, parse the following: MT
 */
__device__ time_zone try_parse_GMT_tz(unsigned char const* const ptr,
                                      int& pos,
                                      int const end_pos,
                                      bool is_spark_320)
{
  // pos_backup points to the char 'G'
  int pos_backup = pos - 1;

  int len = end_pos - pos;
  if (len >= 2 && ptr[pos] == 'M' && ptr[pos + 1] == 'T') {
    if (len == 2) {
      // GMT
      return make_fixed_tz(0);
    }

    // GMT[+-]... or GMT0
    pos += 2;
    int sign_value;
    if (try_parse_sign(ptr, pos, sign_value)) {
      return parse_tz_from_sign(ptr, pos, end_pos, sign_value, is_spark_320);
    } else if (try_parse_char(ptr, pos, '0') && eof(pos, end_pos)) {
      // special case: GMT0
      return make_fixed_tz(0);
    } else {
      // e.g.: GMTx, maybe a valid timezone
      return make_other_tz(pos_backup, end_pos);
    }
  }

  // maybe a valid TZ: GB
  return make_other_tz(pos_backup, end_pos);
}

/**
 * Parse a string to a timezone.
 * For Spark, timezone should have one of the forms:
 *   - Z - Zulu timezone UTC+0
 *   - [+-]h[h]
 *   - [+-]h[h]:m[m]
 *   - [+-]h[h]:mm:ss
 *  - [+-]hhmm
 *   - [+-]hhmmss
 *   - An id with one of the prefixes UTC+, UTC-, GMT+, GMT-, UT+ or UT-,
 *     and a suffix in the formats:
 *     - h[h]
 *     - h[h]:m[m]
 *     - h[h]:mm:ss
 *     - hhmm
 *     - hhmmss
 *   - A short id, see java.time.ZoneId.SHORT_IDS
 *   - Region-based zone IDs in the form `area/city`, such as `Europe/Paris`
 *
 * Note: max offset for fixed tz is 18 hours.
 */
__device__ time_zone parse_tz(unsigned char const* const ptr,
                              int& pos,
                              int const end_pos,
                              bool is_spark_320)
{
  // empty string
  if (eof(pos, end_pos)) { return make_invalid_tz(); }

  // Z
  if (end_pos - pos == 1 && ptr[pos] == 'Z') { return make_fixed_tz(0); }

  int pos_backup = pos;

  // check first char
  unsigned char const first_char = ptr[pos++];
  if ('U' == first_char) {
    return try_parse_UT_tz(ptr, pos, end_pos, is_spark_320);
  } else if ('G' == first_char) {
    return try_parse_GMT_tz(ptr, pos, end_pos, is_spark_320);
  } else if ('+' == first_char || '-' == first_char) {
    int sign_value = (first_char == '+') ? 1 : -1;
    return parse_tz_from_sign(ptr, pos, end_pos, sign_value, is_spark_320);
  } else {
    return make_other_tz(pos_backup, end_pos);
  }
}

/**
 * Parse from timezone part to end of the string.
 * First trim the string from left, the right has been trimmed.
 */
__device__ time_zone parse_from_tz(unsigned char const* const ptr,
                                   int& pos,
                                   int const pos_end,
                                   bool is_spark_320)
{
  // trim left
  while (pos < pos_end && is_whitespace(ptr[pos])) {
    ++pos;
  }
  return parse_tz(ptr, pos, pos_end, is_spark_320);
}

/**
 * Parse separator between date and time: 'T' or ' '
 * e.g.:
 * 2020-01-01T12:00:00
 * 2020-01-01 12:00:00
 */
__device__ bool parse_date_time_separator(unsigned char const* const ptr, int& pos)
{
  unsigned char const c = ptr[pos];
  if (c == ' ' || c == 'T') {
    ++pos;
    return true;
  }
  return false;
}

/**
 * cuda::std::chrono::year_month_day does not check the validity of the
 * date/time. Eg.: 2020-02-30 is valid for cuda::std::chrono::year_month_day.
 */
__device__ bool is_valid_timestamp(
  int year, int month, int day, int hour, int minute, int second, int microseconds)
{
  return date_time_utils::is_valid_date_for_timestamp(year, month, day) &&
         date_time_utils::is_valid_time(hour, minute, second, microseconds);
}

/**
 * @brief convert a timestamp to seconds and microseconds
 *
 */
__device__ int64_t to_epoch_seconds(int year, int month, int day, int hour, int minute, int second)
{
  // if it's a just time timestamp string, then the `days` is 0, because
  // default date is 1970-01-01
  int64_t const days = date_time_utils::to_epoch_day(year, month, day);

  // get seconds
  return (days * 24L * 3600L) + (hour * 3600L) + (minute * 60L) + second;
}

__device__ bool is_valid_digits(int segment, int digits)
{
  // A Long is able to represent a timestamp within [+-]200 thousand years
  const int maxDigitsYear = 6;

  // Check the validity of the digits based on the segment
  return (segment == 6) || (segment == 0 && digits >= 4 && digits <= maxDigitsYear) ||
         (segment == 7 && digits <= 2) ||
         (segment != 0 && segment != 6 && segment != 7 && digits > 0 && digits <= 2);
}

enum segment_index { YEAR = 0, MONTH, DAY, HOUR, MINUTE, SECOND, MICROSECOND };

/**
 * Parse a string with timezone
 */
__device__ result_type parse_timestamp_string(bool is_spark_320,
                                              bool is_spark_400_or_later_or_db_14_3_or_later,
                                              unsigned char const* const ptr,
                                              unsigned char const* ptr_end,
                                              time_zone& tz,
                                              int64_t& seconds,
                                              int32_t& microseconds,
                                              TS_TYPE& just_time)
{
  int pos     = 0;
  int end_pos = ptr_end - ptr;

  // trim left
  while (pos < end_pos && is_whitespace(ptr[pos])) {
    ++pos;
  }

  // trim right
  while (pos < end_pos && is_whitespace(ptr[end_pos - 1])) {
    --end_pos;
  }

  if (eof(pos, end_pos)) { return result_type::INVALID; }

  int bytes_length           = end_pos - pos;
  int segments[]             = {1970, 1, 1, 0, 0, 0, 0, 0, 0};
  int segments_len           = 9;
  int i                      = 0;
  int current_segment_value  = 0;
  int current_segment_digits = 0;
  int j                      = 0;
  int digits_milli           = 0;

  // It's safe to delete when Spark320 suport is removed.
  // indicates if it has sign tz for spark320
  bool has_sign_tz_for_spark320 = false;
  // sign tz for spark320, it's 1 for '+', -1 for '-'
  int tz_sign_for_spark320;

  cuda::std::optional<int> year_sign;
  if ('-' == ptr[pos + j] || '+' == ptr[pos + j]) {
    if ('-' == ptr[pos + j]) {
      year_sign = -1;
    } else {
      year_sign = 1;
    }
    ++j;
  }

  // Spark400+ and DB14.3+: do not support pattern: spaces + Thh:mm:ss
  // Refer to https://github.com/NVIDIA/spark-rapids-jni/issues/3401
  // Refer to https://issues.apache.org/jira/browse/SPARK-52351
  // Check if Spark is Spark400+ or DB14.3+ and has left spaces
  bool match_issue_52351 = is_spark_400_or_later_or_db_14_3_or_later && pos > 0;

  while (j < bytes_length) {
    unsigned char const b  = ptr[pos + j];
    int const parsed_value = static_cast<int32_t>(b) - '0';
    if (parsed_value < 0 || parsed_value > 9) {
      if (0 == j && 'T' == b && !match_issue_52351) {
        just_time = TS_TYPE::JUST_TIME;
        i += 3;
      } else if (i < 2) {
        if (b == '-') {
          if (!is_valid_digits(i, current_segment_digits)) { return result_type::INVALID; }
          segments[i]            = current_segment_value;
          current_segment_value  = 0;
          current_segment_digits = 0;
          i += 1;
        } else if (0 == i && ':' == b && !year_sign.has_value()) {
          just_time = TS_TYPE::JUST_TIME;
          if (!is_valid_digits(3, current_segment_digits)) { return result_type::INVALID; }
          segments[3]            = current_segment_value;
          current_segment_value  = 0;
          current_segment_digits = 0;
          i                      = 4;
        } else {
          return result_type::INVALID;
        }
      } else if (2 == i) {
        if (' ' == b || 'T' == b) {
          if (!is_valid_digits(i, current_segment_digits)) { return result_type::INVALID; }
          segments[i]            = current_segment_value;
          current_segment_value  = 0;
          current_segment_digits = 0;
          i += 1;
        } else {
          return result_type::INVALID;
        }
      } else if (3 == i || 4 == i) {
        if (':' == b) {
          if (!is_valid_digits(i, current_segment_digits)) { return result_type::INVALID; }
          segments[i]            = current_segment_value;
          current_segment_value  = 0;
          current_segment_digits = 0;
          i += 1;
        } else {
          return result_type::INVALID;
        }
      } else if (5 == i || 6 == i) {
        if (is_spark_320 && ('-' == b || '+' == b)) {
          // It's safe to delete when Spark320 suport is removed.
          if (!is_valid_digits(i, current_segment_digits)) { return result_type::INVALID; }
          segments[i]            = current_segment_value;
          current_segment_value  = 0;
          current_segment_digits = 0;
          i += 1;
          has_sign_tz_for_spark320 = true;
          tz_sign_for_spark320     = (b == '+');

        } else if ('.' == b && 5 == i) {
          if (!is_valid_digits(i, current_segment_digits)) { return result_type::INVALID; }
          segments[i]            = current_segment_value;
          current_segment_value  = 0;
          current_segment_digits = 0;
          i += 1;
        } else {
          if (!is_valid_digits(i, current_segment_digits)) { return result_type::INVALID; }
          segments[i]            = current_segment_value;
          current_segment_value  = 0;
          current_segment_digits = 0;
          i += 1;

          // parse timezone
          int tz_pos = pos + j;
          tz         = parse_from_tz(ptr, tz_pos, end_pos, is_spark_320);
          if (tz.type == TZ_TYPE::INVALID_TZ) { return result_type::INVALID; }

          j = bytes_length - 1;
        }
        if (i == 6 && '.' != b) { i += 1; }
      } else {
        if (i < segments_len && (':' == b || ' ' == b)) {
          if (!is_valid_digits(i, current_segment_digits)) { return result_type::INVALID; }
          segments[i]            = current_segment_value;
          current_segment_value  = 0;
          current_segment_digits = 0;
          i += 1;
        } else {
          return result_type::INVALID;
        }
      }
    } else {
      if (6 == i) { digits_milli += 1; }
      // We will truncate the nanosecond part if there are more than 6 digits,
      // which results in loss of precision
      if (6 != i || current_segment_digits < 6) {
        current_segment_value = current_segment_value * 10 + parsed_value;
      }
      current_segment_digits += 1;
    }
    j += 1;
  }

  if (!is_valid_digits(i, current_segment_digits)) { return result_type::INVALID; }
  segments[i] = current_segment_value;

  while (digits_milli < 6) {
    segments[6] *= 10;
    digits_milli += 1;
  }

  // It's safe to delete when Spark320 suport is removed.
  if (has_sign_tz_for_spark320) {
    // for spark320, the sign is not included in the tz offset
    int hour_for_320   = segments[7];
    int minute_for_320 = segments[8];
    if (hour_for_320 > 18 || minute_for_320 > 59 ||
        hour_for_320 * 3600 + minute_for_320 * 60 > 18 * 3600) {
      // invalid timezone, e.g. +123:00; max offset is 18:00:00
      return result_type::INVALID;
    } else {
      tz = make_fixed_tz(tz_sign_for_spark320 * (segments[7] * 3600 + segments[8] * 60));
    }
  }

  segments[0] *= year_sign.value_or(1);
  // above is ported from Spark.

  if (!is_valid_timestamp(segments[segment_index::YEAR],
                          segments[segment_index::MONTH],
                          segments[segment_index::DAY],
                          segments[segment_index::HOUR],
                          segments[segment_index::MINUTE],
                          segments[segment_index::SECOND],
                          segments[segment_index::MICROSECOND]) ||
      !is_valid_tz(tz)) {
    return result_type::INVALID;
  }

  seconds      = to_epoch_seconds(segments[segment_index::YEAR],
                             segments[segment_index::MONTH],
                             segments[segment_index::DAY],
                             segments[segment_index::HOUR],
                             segments[segment_index::MINUTE],
                             segments[segment_index::SECOND]);
  microseconds = segments[segment_index::MICROSECOND];
  return result_type::SUCCESS;
}

/**
 * Parse a string with timezone to a timestamp.
 */
struct parse_timestamp_string_fn {
  // version info
  bool is_spark_320;
  bool is_spark_400_or_later_or_db_14_3_or_later;

  // inputs
  cudf::column_device_view d_strings;
  cudf::size_type default_tz_index;
  int64_t default_epoch_day;
  // current seconds since epoch, used to calculate the date in just time string
  int64_t current_seconds_since_epoch;
  // STRUCT<tz_name: string, index_to_tz_info_table: int>
  cudf::column_device_view tz_name_to_index_map;
  // Fixed offset transitions in the timezone table
  // Column type is LIST<STRUCT<utcInstant: int64, tzInstant: int64, utcOffset: int32>>.
  cudf::detail::lists_column_device_view fixed_transitions;
  // DST rules in the timezone table
  // column type is LIST<INT>, if it's DST, 12 integers defines two rules
  cudf::detail::lists_column_device_view dst_rules;

  // outputs
  // parsed result types: not supported, invalid, success
  uint8_t* result_types;
  // parsed timestamp in UTC timezone
  int64_t* ts_seconds;
  int32_t* ts_microseconds;
  // parsed timezone info
  uint8_t* tz_types;
  int32_t* tz_fixed_offsets;
  int32_t* tz_indices;

  __device__ void operator()(cudf::size_type const idx) const
  {
    // No need to check null for the `str` element, because get element on
    // null will return empty string and then result in invalid
    auto const str = d_strings.element<cudf::string_view>(idx);

    unsigned char const* str_ptr     = reinterpret_cast<unsigned char const*>(str.data());
    unsigned char const* str_end_ptr = str_ptr + str.size_bytes();

    time_zone tz;
    int64_t seconds      = 0;
    int32_t microseconds = 0;
    TS_TYPE just_time    = TS_TYPE::NOT_JUST_TIME;

    // parse the timestamp string
    auto result_type = parse_timestamp_string(is_spark_320,
                                              is_spark_400_or_later_or_db_14_3_or_later,
                                              str_ptr,
                                              str_end_ptr,
                                              tz,
                                              seconds,
                                              microseconds,
                                              just_time);

    // set result column
    result_types[idx]     = static_cast<uint8_t>(result_type);
    ts_seconds[idx]       = seconds;
    ts_microseconds[idx]  = microseconds;
    tz_types[idx]         = static_cast<uint8_t>(tz.type);
    tz_fixed_offsets[idx] = tz.fixed_offset;
    tz_indices[idx]       = -1;

    if (result_type != result_type::SUCCESS) {
      // already set result_type::INVALID
      return;
    }

    // check the timezone, and get the timezone index
    if (tz.type == TZ_TYPE::NOT_SPECIFIED) {
      // use the default timezone index
      tz_types[idx]   = static_cast<uint8_t>(TZ_TYPE::OTHER_TZ);
      tz_indices[idx] = default_tz_index;
      if (just_time == TS_TYPE::JUST_TIME) {
        // use the default epoch days when the timezone is not specified
        // the `default_epoch_day` is from Java code:
        // LocalDate.now(default_time_zone).toEpochDay()
        ts_seconds[idx] = seconds + (default_epoch_day * 24L * 3600L);
      }
    } else if (tz.type == TZ_TYPE::FIXED_TZ) {
      if (just_time == TS_TYPE::JUST_TIME) {
        // Step 1: Get the current date in the timezone
        // in order to get the correct local date, rebase from utc timezone to
        // local timezone e.g.: current UTC time is 2025-01-01T23:00:00, tz
        // offsets is +01:00, then current date is: 2025-01-01T23:00:00 + 01:00
        // = 2025-01-02T00:00:00
        auto rebased = current_seconds_since_epoch + tz.fixed_offset;
        // This is to get the seconds for the date part with discarding the time
        // part
        auto rebased_days_of_local_date = rebased / (24L * 3600L);

        // Step 2: add date part to the seconds
        ts_seconds[idx] = seconds + (rebased_days_of_local_date * 24L * 3600L);
      }
    } else if (tz.type == TZ_TYPE::OTHER_TZ) {
      /**
       * If tz type is OTHER_TZ, binary search in the `tz_name_to_index_map` to get the
       * timezone index. If not found, set the result type to invalid.
       * If the string is just time, use the current date in the timezone.
       */
      auto const tz_col                  = tz_name_to_index_map.child(0);
      auto const index_in_transition_col = tz_name_to_index_map.child(1);

      auto const tzs_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        [tz_col] __device__(int idx) { return tz_col.element<cudf::string_view>(idx); });
      auto const tzs_end   = tzs_begin + tz_col.size();
      auto const target_tz = cudf::string_view(
        reinterpret_cast<char const*>(str_ptr + tz.tz_pos_in_string), tz.tz_len());

      auto const it = thrust::lower_bound(
        thrust::seq, tzs_begin, tzs_end, target_tz, thrust::less<cudf::string_view>());
      if (it != tzs_end && *it == target_tz) {
        // found tz
        auto const tz_idx_in_table =
          static_cast<cudf::size_type>(cuda::std::distance(tzs_begin, it));
        // update tz index
        tz_indices[idx] = index_in_transition_col.element<int32_t>(tz_idx_in_table);

        if (just_time == TS_TYPE::JUST_TIME) {
          // get current date in the timezone, equivalent to Java code:
          // LocalDate.now(zoneId) E.g.:
          //   LocalDate.now("America/Los_Angeles") = 2025-05-21,
          //   at the same time:
          //   LocalDate.now("Asia/Shanghai")       = 2025-05-22

          // Step 1: rebase `current_seconds_since_epoch` from utc timezone to
          // local timezone to get the current date
          auto rebased_seconds = spark_rapids_jni::convert_timestamp<cudf::timestamp_s>(
            cudf::timestamp_s{cudf::duration_s{current_seconds_since_epoch}},
            fixed_transitions,
            dst_rules,
            tz_indices[idx],
            /* to_utc */ false);

          auto const rebased = static_cast<int64_t>(
            cuda::std::chrono::duration_cast<cudf::duration_s>(rebased_seconds.time_since_epoch())
              .count());
          int64_t rebased_days_of_local_date = rebased / (24L * 3600L);

          // Step 2: add date part to the seconds
          ts_seconds[idx] = seconds + (rebased_days_of_local_date * 24L * 3600L);
        }
      } else {
        // not found tz, update result_type to invalid
        result_types[idx] = static_cast<uint8_t>(result_type::INVALID);
        tz_indices[idx]   = -1;
      }
    } else if (tz.type == TZ_TYPE::INVALID_TZ) {
      cudf_assert(result_type == result_type::INVALID);
    } else {
      // should not happen
      cudf_assert(false);
    }
  }
};

/**
 * Parse strings to an intermediate struct column with 6 sub-columns.
 */
std::unique_ptr<cudf::column> parse_ts_strings(cudf::strings_column_view const& input,
                                               cudf::size_type default_tz_index,
                                               int64_t default_epoch_day,
                                               cudf::column_view const& tz_name_to_index_map,
                                               cudf::table_view const& tz_info_table,
                                               bool is_spark_320,
                                               bool is_spark_400_or_later_or_db_14_3_or_later,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  auto const num_rows            = input.size();
  auto const d_input             = cudf::column_device_view::create(input.parent(), stream);
  auto const d_name_to_index_map = cudf::column_device_view::create(tz_name_to_index_map, stream);

  // the follow saves parsed result
  auto parsed_result_type_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::UINT8}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  auto parsed_utc_seconds_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT64}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  auto parsed_utc_microseconds_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  auto parsed_tz_type_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::UINT8}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  // if tz type is fixed, use this column to store offsets
  auto parsed_tz_fixed_offset_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  // if tz type is other, use this column to store index to transition table
  auto parsed_tz_index_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);

  // Get current seconds since epoch, used to calculate the date in just time
  // string
  auto duration = std::chrono::system_clock::now().time_since_epoch();
  int64_t current_seconds_since_epoch =
    std::chrono::duration_cast<std::chrono::seconds>(duration).count();

  // get the fixed transitions
  auto const ft_cdv_ptr        = cudf::column_device_view::create(tz_info_table.column(0), stream);
  auto const fixed_transitions = cudf::detail::lists_column_device_view{*ft_cdv_ptr};

  // get the DST rules
  auto const dst_cdv_ptr = cudf::column_device_view::create(tz_info_table.column(1), stream);
  auto const dst_rules   = cudf::detail::lists_column_device_view{*dst_cdv_ptr};

  thrust::for_each_n(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(0),
    num_rows,
    parse_timestamp_string_fn{is_spark_320,
                              is_spark_400_or_later_or_db_14_3_or_later,
                              *d_input,
                              default_tz_index,
                              default_epoch_day,
                              current_seconds_since_epoch,
                              *d_name_to_index_map,
                              fixed_transitions,
                              dst_rules,
                              parsed_result_type_col->mutable_view().begin<uint8_t>(),
                              parsed_utc_seconds_col->mutable_view().begin<int64_t>(),
                              parsed_utc_microseconds_col->mutable_view().begin<int32_t>(),
                              parsed_tz_type_col->mutable_view().begin<uint8_t>(),
                              parsed_tz_fixed_offset_col->mutable_view().begin<int32_t>(),
                              parsed_tz_index_col->mutable_view().begin<int32_t>()});

  std::vector<std::unique_ptr<cudf::column>> output_columns;
  output_columns.emplace_back(std::move(parsed_result_type_col));
  output_columns.emplace_back(std::move(parsed_utc_seconds_col));
  output_columns.emplace_back(std::move(parsed_utc_microseconds_col));
  output_columns.emplace_back(std::move(parsed_tz_type_col));
  output_columns.emplace_back(std::move(parsed_tz_fixed_offset_col));
  output_columns.emplace_back(std::move(parsed_tz_index_col));

  return make_structs_column(num_rows,
                             std::move(output_columns),
                             /* null_count */ 0,
                             rmm::device_buffer(),
                             stream,
                             mr);
}

/**
 * Parse date string to year, month, day.
 * Note: Spark date supports max 7 digits year, Spark timestamp supports max 6
 * digits year.
 */
__device__ bool parse_date(unsigned char const* const ptr,
                           unsigned char const* const ptr_end,
                           spark_rapids_jni::date_segments& d)
{
  int pos     = 0;
  int end_pos = ptr_end - ptr;

  // trim left
  while (pos < end_pos && is_whitespace(ptr[pos])) {
    ++pos;
  }

  // trim right
  while (pos < end_pos && is_whitespace(ptr[end_pos - 1])) {
    --end_pos;
  }

  if (eof(pos, end_pos)) { return false; }

  // parse sign
  bool negative_year_sign = false;
  char const sign_c       = ptr[pos];
  if ('-' == sign_c || '+' == sign_c) {
    ++pos;
    if ('-' == sign_c) { negative_year_sign = true; }
  }

  // parse year: yyyy[y][y][y]
  if (!parse_int(ptr,
                 pos,
                 end_pos,
                 d.year,
                 /*min_digits*/ 4,
                 /*max_digits*/ 7)) {
    return false;
  }
  if (negative_year_sign) { d.year = -d.year; }

  if (eof(pos, end_pos)) {
    // only has: yyyy[y][y][y], return early
    return true;
  }

  // parse month: -[m]m
  if (!parse_char(ptr, pos, '-') || !parse_int(ptr,
                                               pos,
                                               end_pos,
                                               d.month,
                                               /*min_digits*/ 1,
                                               /*max_digits*/ 2)) {
    return false;
  }

  if (eof(pos, end_pos)) {
    // only has: yyyy[y][y]-[m]m, return early
    return true;
  }

  // parse day: -[d]d
  if (!parse_char(ptr, pos, '-') || !parse_int(ptr,
                                               pos,
                                               end_pos,
                                               d.day,
                                               /*min_digits*/ 1,
                                               /*max_digits*/ 2)) {
    return false;
  }

  if (eof(pos, end_pos)) {
    // no tailing characters, e.g.: "2025-01-01"
    return true;
  }

  // parse date time separator
  // ignore the tailing chars after ' ' or 'T', e.g.: the following are valid:
  // "2025-01-01T", "2025-01-01Txxxx", and "2025-01-01 xxx"
  return parse_date_time_separator(ptr, pos);
}

struct parse_string_to_date_fn {
  // input strings
  cudf::column_device_view d_strings;

  // output columns
  bool* validity;
  cudf::timestamp_D* output;

  __device__ void operator()(cudf::size_type const idx) const
  {
    // check null
    if (d_strings.is_null(idx)) {
      validity[idx] = false;
      return;
    }

    auto const str                   = d_strings.element<cudf::string_view>(idx);
    unsigned char const* str_ptr     = reinterpret_cast<unsigned char const*>(str.data());
    unsigned char const* str_end_ptr = str_ptr + str.size_bytes();

    // parse the date string to segments
    spark_rapids_jni::date_segments date_segments;
    auto result_success = parse_date(str_ptr, str_end_ptr, date_segments);

    // check parsed result
    if (!result_success || !date_segments.is_valid_date()) {
      validity[idx] = false;
      return;
    }

    // calculate the epoch day and check 'days' fits in int32_t
    int64_t days  = date_segments.to_epoch_day();
    output[idx]   = cudf::timestamp_D{cudf::duration_D{static_cast<int32_t>(days)}};
    validity[idx] = days >= cuda::std::numeric_limits<int32_t>::min() &&
                    days <= cuda::std::numeric_limits<int32_t>::max();
  }
};

/**
 * Parse strings to dates.
 */
std::unique_ptr<cudf::column> parse_to_date(cudf::strings_column_view const& input,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  auto const num_rows = input.size();
  if (num_rows == 0) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_to_id<cudf::timestamp_D>()});
  }

  auto const d_input = cudf::column_device_view::create(input.parent(), stream);
  auto result = cudf::make_timestamp_column(cudf::data_type{cudf::type_to_id<cudf::timestamp_D>()},
                                            input.size(),
                                            rmm::device_buffer{},
                                            0,
                                            stream,
                                            mr);
  auto validity =
    rmm::device_uvector<bool>(num_rows, stream, cudf::get_current_device_resource_ref());

  thrust::for_each_n(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(0),
    num_rows,
    parse_string_to_date_fn{
      *d_input, validity.begin(), result->mutable_view().begin<cudf::timestamp_D>()});

  auto [output_bitmask, null_count] =
    cudf::bools_to_mask(cudf::device_span<bool const>(validity), stream, mr);
  if (null_count) { result->set_null_mask(std::move(*output_bitmask.release()), null_count); }

  return result;
}

}  // anonymous namespace

std::unique_ptr<cudf::column> parse_timestamp_strings(
  cudf::strings_column_view const& input,
  cudf::size_type default_tz_index,
  int64_t default_epoch_day,
  cudf::column_view const& tz_name_to_index_map,
  cudf::table_view const& tz_info_table,
  spark_rapids_jni::spark_system const& spark_system,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  bool is_spark_320 = spark_system.is_vanilla_320();
  bool is_spark_400_or_later_or_db_14_3_or_later =
    spark_system.is_vanilla_400_or_later() || spark_system.is_databricks_14_3_or_later();

  return parse_ts_strings(input,
                          default_tz_index,
                          default_epoch_day,
                          tz_name_to_index_map,
                          tz_info_table,
                          is_spark_320,
                          is_spark_400_or_later_or_db_14_3_or_later,
                          stream,
                          mr);
}

std::unique_ptr<cudf::column> parse_strings_to_date(cudf::strings_column_view const& input,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  return parse_to_date(input, stream, mr);
}

}  // namespace spark_rapids_jni
