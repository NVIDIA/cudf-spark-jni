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

#include "cudf_jni_apis.hpp"
#include "timezones.hpp"

#include <cstdint>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_GpuTimeZoneDB_convertTimestampColumnToUTC(
  JNIEnv* env, jclass, jlong input_handle, jlong timezone_info_handle, jint tz_index)
{
  JNI_NULL_CHECK(env, input_handle, "column is null", 0);
  JNI_NULL_CHECK(env, timezone_info_handle, "column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input         = reinterpret_cast<cudf::column_view const*>(input_handle);
    auto const timezone_info = reinterpret_cast<cudf::table_view const*>(timezone_info_handle);
    auto const index         = static_cast<cudf::size_type>(tz_index);
    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::convert_timestamp_to_utc(*input, *timezone_info, index).release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_GpuTimeZoneDB_convertUTCTimestampColumnToTimeZone(
  JNIEnv* env, jclass, jlong input_handle, jlong timezone_info_handle, jint tz_index)
{
  JNI_NULL_CHECK(env, input_handle, "column is null", 0);
  JNI_NULL_CHECK(env, timezone_info_handle, "column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input         = reinterpret_cast<cudf::column_view const*>(input_handle);
    auto const timezone_info = reinterpret_cast<cudf::table_view const*>(timezone_info_handle);
    auto const index         = static_cast<cudf::size_type>(tz_index);
    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::convert_utc_timestamp_to_timezone(*input, *timezone_info, index).release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_GpuTimeZoneDB_convertTimestampColumnToUTCWithTzCv(
  JNIEnv* env,
  jclass,
  jlong input_seconds_handle,
  jlong input_microseconds_handle,
  jlong invalid_handle,
  jlong tz_type_handle,
  jlong tz_offset_handle,
  jlong timezone_info_handle,
  jlong tz_indices_handle)
{
  JNI_NULL_CHECK(env, input_seconds_handle, "seconds column is null", 0);
  JNI_NULL_CHECK(env, input_microseconds_handle, "microseconds column is null", 0);
  JNI_NULL_CHECK(env, invalid_handle, "invalid column is null", 0);
  JNI_NULL_CHECK(env, tz_type_handle, "tz type column is null", 0);
  JNI_NULL_CHECK(env, tz_offset_handle, "tz offset column is null", 0);
  JNI_NULL_CHECK(env, timezone_info_handle, "timezone info table is null", 0);
  JNI_NULL_CHECK(env, tz_indices_handle, "tz indices column is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input_seconds = reinterpret_cast<cudf::column_view const*>(input_seconds_handle);
    auto const input_microseconds =
      reinterpret_cast<cudf::column_view const*>(input_microseconds_handle);
    auto const invalid       = reinterpret_cast<cudf::column_view const*>(invalid_handle);
    auto const tz_type       = reinterpret_cast<cudf::column_view const*>(tz_type_handle);
    auto const tz_offset     = reinterpret_cast<cudf::column_view const*>(tz_offset_handle);
    auto const timezone_info = reinterpret_cast<cudf::table_view const*>(timezone_info_handle);
    auto const tz_indices    = reinterpret_cast<cudf::column_view const*>(tz_indices_handle);

    return cudf::jni::ptr_as_jlong(spark_rapids_jni::convert_timestamp_to_utc(*input_seconds,
                                                                              *input_microseconds,
                                                                              *invalid,
                                                                              *tz_type,
                                                                              *tz_offset,
                                                                              *timezone_info,
                                                                              *tz_indices)
                                     .release());
  }
  JNI_CATCH(env, 0);
}

static spark_rapids_jni::dst_rule parse_dst_rule(JNIEnv* env, jintArray java_rule)
{
  spark_rapids_jni::dst_rule rule{};
  if (java_rule == nullptr) { return rule; }

  cudf::jni::native_jintArray const values(env, java_rule);
  constexpr int32_t expected_rule_size = 13;
  JNI_ARG_CHECK(
    env, values.size() == expected_rule_size, "ORC DST rule array must contain 13 integers", rule);

  // Keep this field order synchronized with GpuTimeZoneDB.dstRuleToArray.
  rule.has_dst         = true;
  rule.dst_savings     = values[0];
  rule.start_month     = values[1];
  rule.start_day       = values[2];
  rule.start_dow       = values[3];
  rule.start_time      = values[4];
  rule.start_time_mode = values[5];
  rule.start_mode      = values[6];
  rule.end_month       = values[7];
  rule.end_day         = values[8];
  rule.end_dow         = values[9];
  rule.end_time        = values[10];
  rule.end_time_mode   = values[11];
  rule.end_mode        = values[12];
  return rule;
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_GpuTimeZoneDB_convertOrcTimezonesWithRules(
  JNIEnv* env,
  jclass,
  jlong input_handle,
  jlong writer_tz_offset_at_orc_2015_base_us,
  jlong writer_tz_info_table,
  jint writer_tz_initial_offset,
  jint writer_tz_raw_offset,
  jintArray writer_dst_rule,
  jlong reader_tz_info_table,
  jint reader_tz_initial_offset,
  jint reader_tz_raw_offset,
  jintArray reader_dst_rule)
{
  JNI_NULL_CHECK(env, input_handle, "input column is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input              = reinterpret_cast<cudf::column_view const*>(input_handle);
    auto const writer_tz_info_tab = reinterpret_cast<cudf::table_view const*>(writer_tz_info_table);
    auto const reader_tz_info_tab = reinterpret_cast<cudf::table_view const*>(reader_tz_info_table);
    auto const writer_dst         = parse_dst_rule(env, writer_dst_rule);
    cudf::jni::check_java_exception(env);
    auto const reader_dst = parse_dst_rule(env, reader_dst_rule);
    cudf::jni::check_java_exception(env);

    auto const writer = spark_rapids_jni::orc_tz_side{
      writer_tz_info_tab, writer_tz_initial_offset, writer_tz_raw_offset, writer_dst};
    auto const reader = spark_rapids_jni::orc_tz_side{
      reader_tz_info_tab, reader_tz_initial_offset, reader_tz_raw_offset, reader_dst};
    return cudf::jni::release_as_jlong(spark_rapids_jni::convert_orc_writer_reader_timezones(
      *input, static_cast<int64_t>(writer_tz_offset_at_orc_2015_base_us), writer, reader));
  }
  JNI_CATCH(env, 0);
}
}
