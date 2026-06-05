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

#include "timezones.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/wrappers/timestamps.hpp>

#include <limits>

auto constexpr int64_min = std::numeric_limits<int64_t>::min();

using int32_col = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64_col = cudf::test::fixed_width_column_wrapper<int64_t>;

using seconds_col =
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>;

using millis_col =
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_s::rep>;

using micros_col =
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_us, cudf::timestamp_s::rep>;

class TimeZoneTest : public cudf::test::BaseFixture {
 protected:
  void SetUp() override { transitions = make_transitions_table(); }
  std::unique_ptr<cudf::table> transitions;

 private:
  std::unique_ptr<cudf::table> make_transitions_table()
  {
    auto instants_from_utc_col = int64_col({int64_min,
                                            int64_min,
                                            -1585904400L,
                                            -933667200L,
                                            -922093200L,
                                            -908870400L,
                                            -888829200L,
                                            -650019600L,
                                            515527200L,
                                            558464400L,
                                            684867600L});
    auto instants_to_utc_col   = int64_col({int64_min,
                                            int64_min,
                                            -1585904400L,
                                            -933634800L,
                                            -922064400L,
                                            -908838000L,
                                            -888796800L,
                                            -649990800L,
                                            515559600L,
                                            558493200L,
                                            684896400L});
    auto utc_offsets_col =
      int32_col({18000, 29143, 28800, 32400, 28800, 32400, 28800, 28800, 32400, 28800, 28800});
    auto struct_column = cudf::test::structs_column_wrapper{
      {instants_from_utc_col, instants_to_utc_col, utc_offsets_col},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
    auto offsets       = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 11};
    auto list_nullmask = std::vector<bool>(1, 1);
    auto [null_mask, null_count] =
      cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
    auto list_column = cudf::make_lists_column(
      2, offsets.release(), struct_column.release(), null_count, std::move(null_mask));
    auto columns = std::vector<std::unique_ptr<cudf::column>>{};
    columns.push_back(std::move(list_column));

    // make empty DST list<int> column, it means all timezones are non-DST
    auto dst_child   = int32_col({});
    auto dst_offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 0};
    auto dst_col     = cudf::make_lists_column(
      2, dst_offsets.release(), dst_child.release(), 0, rmm::device_buffer{});
    columns.push_back(std::move(dst_col));

    return std::make_unique<cudf::table>(std::move(columns));
  }
};

TEST_F(TimeZoneTest, ConvertToUTCSeconds)
{
  auto const ts_col = seconds_col{
    -1262260800L,
    -908838000L,
    -908840700L,
    -888800400L,
    -888799500L,
    -888796800L,
    0L,
    1699566167L,
    568036800L,
  };
  // check the converted to utc version
  auto const expected = seconds_col{-1262289600L,
                                    -908870400L,
                                    -908869500L,
                                    -888832800L,
                                    -888831900L,
                                    -888825600L,
                                    -28800L,
                                    1699537367L,
                                    568008000L};
  auto const actual =
    spark_rapids_jni::convert_timestamp_to_utc(ts_col,
                                               *transitions,
                                               1,
                                               cudf::get_default_stream(),
                                               rmm::mr::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TEST_F(TimeZoneTest, ConvertToUTCMilliseconds)
{
  auto const ts_col = millis_col{
    -1262260800000L,
    -908838000000L,
    -908840700000L,
    -888800400000L,
    -888799500000L,
    -888796800000L,
    0L,
    1699571634312L,
    568036800000L,
  };
  // check the converted to utc version
  auto const expected = millis_col{-1262289600000L,
                                   -908870400000L,
                                   -908869500000L,
                                   -888832800000L,
                                   -888831900000L,
                                   -888825600000L,
                                   -28800000L,
                                   1699542834312L,
                                   568008000000L};
  auto const actual =
    spark_rapids_jni::convert_timestamp_to_utc(ts_col,
                                               *transitions,
                                               1,
                                               cudf::get_default_stream(),
                                               rmm::mr::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TEST_F(TimeZoneTest, ConvertToUTCMicroseconds)
{
  auto const ts_col = micros_col{
    -1262260800000000L,
    -908838000000000L,
    -908840700000000L,
    -888800400000000L,
    -888799500000000L,
    -888796800000000L,
    0L,
    1699571634312000L,
    568036800000000L,
  };
  // check the converted to utc version
  auto const expected = micros_col{-1262289600000000L,
                                   -908870400000000L,
                                   -908869500000000L,
                                   -888832800000000L,
                                   -888831900000000L,
                                   -888825600000000L,
                                   -28800000000L,
                                   1699542834312000L,
                                   568008000000000L};
  auto const actual =
    spark_rapids_jni::convert_timestamp_to_utc(ts_col,
                                               *transitions,
                                               1,
                                               cudf::get_default_stream(),
                                               rmm::mr::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TEST_F(TimeZoneTest, ConvertFromUTCSeconds)
{
  auto const ts_col = seconds_col{-1262289600L,
                                  -908870400L,
                                  -908869500L,
                                  -888832800L,
                                  -888831900L,
                                  -888825600L,
                                  0L,
                                  1699537367L,
                                  568008000L};
  // check the converted to utc version
  auto const expected = seconds_col{
    -1262260800L,
    -908838000L,
    -908837100L,
    -888800400L,
    -888799500L,
    -888796800L,
    28800L,
    1699566167L,
    568036800L,
  };
  auto const actual =
    spark_rapids_jni::convert_utc_timestamp_to_timezone(ts_col,
                                                        *transitions,
                                                        1,
                                                        cudf::get_default_stream(),
                                                        rmm::mr::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TEST_F(TimeZoneTest, ConvertFromUTCMilliseconds)
{
  auto const ts_col = millis_col{-1262289600000L,
                                 -908870400000L,
                                 -908869500000L,
                                 -888832800000L,
                                 -888831900000L,
                                 -888825600000L,
                                 0L,
                                 1699542834312L,
                                 568008000000L};
  // check the converted to timezone version
  auto const expected = millis_col{
    -1262260800000L,
    -908838000000L,
    -908837100000L,
    -888800400000L,
    -888799500000L,
    -888796800000L,
    28800000L,
    1699571634312L,
    568036800000L,
  };
  auto const actual =
    spark_rapids_jni::convert_utc_timestamp_to_timezone(ts_col,
                                                        *transitions,
                                                        1,
                                                        cudf::get_default_stream(),
                                                        rmm::mr::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TEST_F(TimeZoneTest, ConvertFromUTCMicroseconds)
{
  auto const ts_col = micros_col{-1262289600000000L,
                                 -908870400000000L,
                                 -908869500000000L,
                                 -888832800000000L,
                                 -888831900000000L,
                                 -888825600000000L,
                                 0L,
                                 1699542834312000L,
                                 568008000000000L};
  // check the converted to timezone version
  auto const expected = micros_col{
    -1262260800000000L,
    -908838000000000L,
    -908837100000000L,
    -888800400000000L,
    -888799500000000L,
    -888796800000000L,
    28800000000L,
    1699571634312000L,
    568036800000000L,
  };
  auto const actual =
    spark_rapids_jni::convert_utc_timestamp_to_timezone(ts_col,
                                                        *transitions,
                                                        1,
                                                        cudf::get_default_stream(),
                                                        rmm::mr::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

// Regression for the negative-microsecond floor-division bug: when a negative timestamp lies in
// the last sub-second window before a gap transition (here -908870400s, offset 28800 → 32400),
// truncation toward zero would snap to the transition itself and pick the post-transition offset.
TEST_F(TimeZoneTest, ConvertFromUTCMicrosecondsSubSecondBeforeGap)
{
  auto const ts_col   = micros_col{-908870400000001L, -908870400000000L};
  auto const expected = micros_col{-908841600000001L, -908838000000000L};
  auto const actual =
    spark_rapids_jni::convert_utc_timestamp_to_timezone(ts_col,
                                                        *transitions,
                                                        1,
                                                        cudf::get_default_stream(),
                                                        rmm::mr::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

// Sibling regression for the ORC path (convert_orc_writer_reader_timezones). The two-pass
// `convertBetweenTimezones` algorithm self-corrects on natural DST tables (the second lookup at
// adjusted_ms lands in the same row as the first, so floor and truncate give the same final
// answer). To lock down the floor semantics end-to-end this test uses a contrived 3-transition
// reader table that forces the adjusted_ms lookups in the floor vs truncate paths to land in
// different rows.
//
// Reader transitions (ms): [-1800000, 0, 1000000000], offsets (ms): [0, 3600000, 0],
// raw_offset = 7_200_000. Writer is nullptr (fixed UTC, offset 0). For input µs = -1:
//   * floor: epoch_ms = -1, reader_offset = 0, adjusted_ms = -1, reader_adjusted = 0,
//            final_diff = 0, result = -1.
//   * truncate (pre-fix): epoch_ms = 0, reader_offset = 3_600_000, adjusted_ms = -3_600_000,
//            reader_adjusted = raw_offset = 7_200_000, final_diff = -7_200_000,
//            result = -1 - 7_200_000_000 = -7200000001.
// The fix flips the result back to -1.
TEST_F(TimeZoneTest, ConvertOrcTimezonesSubMillisBeforeGap)
{
  auto reader_trans   = int64_col({-1800000L, 0L, 1000000000L});
  auto reader_offsets = int32_col({0, 3600000, 0});
  auto reader_tv      = cudf::table_view({reader_trans, reader_offsets});

  auto const ts_col   = micros_col{-1L};
  auto const expected = micros_col{-1L};
  auto const actual   = spark_rapids_jni::convert_orc_writer_reader_timezones(
    ts_col,
    nullptr,
    0,
    &reader_tv,
    7200000,
    cudf::get_default_stream(),
    rmm::mr::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

namespace {
// US-style DST rule: DST on the 2nd Sunday of March at 02:00 standard, off the
// 1st Sunday of November at 01:00 standard, +1h savings. Mirrors the encoding
// produced by OrcDstRuleExtractor (0-based month; dayOfWeek 1=Sun..7=Sat;
// mode 2 = DOW_GE_DOM; timeMode 1 = STANDARD).
spark_rapids_jni::dst_rule make_us_dst_rule()
{
  spark_rapids_jni::dst_rule rule{};
  rule.has_dst         = true;
  rule.dst_savings     = 3600000;
  rule.start_month     = 2;        // March
  rule.start_day       = 8;        // "first Sunday on or after the 8th" == 2nd Sunday
  rule.start_dow       = 1;        // Sunday
  rule.start_time      = 7200000;  // 02:00
  rule.start_time_mode = 1;        // STANDARD
  rule.start_mode      = 2;        // DOW_GE_DOM
  rule.end_month       = 10;       // November
  rule.end_day         = 1;        // "first Sunday on or after the 1st"
  rule.end_dow         = 1;
  rule.end_time        = 3600000;  // 01:00
  rule.end_time_mode   = 1;
  rule.end_mode        = 2;
  return rule;
}
}  // namespace

// DST path with no transition table: every instant resolves through the DST
// rule (compute_dst_offset). Writer is fixed UTC (offset 0), reader carries the
// US rule on a raw offset of 0, so a winter instant (standard) is unchanged and
// a summer instant (DST +1h) is shifted back one hour. 2030-01-15 is before the
// 2030 DST window (starts Mar 10) and 2030-07-15 is inside it (ends Nov 3).
TEST_F(TimeZoneTest, ConvertOrcTimezonesReaderDstBeyondTable)
{
  auto const reader_dst = make_us_dst_rule();
  spark_rapids_jni::dst_rule no_dst{};
  no_dst.has_dst = false;

  auto const input    = micros_col{1894665600000000L, 1910304000000000L};
  auto const expected = micros_col{1894665600000000L, 1910300400000000L};
  auto const actual   = spark_rapids_jni::convert_orc_writer_reader_timezones(
    input,
    /*base_offset_us=*/int64_t{0},
    /*writer_tz_info_table=*/nullptr,
    /*writer_initial_offset=*/0,
    /*writer_raw_offset=*/0,
    no_dst,
    /*reader_tz_info_table=*/nullptr,
    /*reader_initial_offset=*/0,
    /*reader_raw_offset=*/0,
    reader_dst,
    cudf::get_default_stream(),
    rmm::mr::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

// Converting a DST zone to itself must be the identity for every instant
// (writer and reader offsets cancel), exercising the DST math on both the
// standard (winter) and daylight (summer) branches.
TEST_F(TimeZoneTest, ConvertOrcTimezonesSameDstZoneIsIdentity)
{
  auto const rule = make_us_dst_rule();

  auto const input    = micros_col{1894665600000000L, 1910304000000000L};
  auto const expected = micros_col{1894665600000000L, 1910304000000000L};
  auto const actual   = spark_rapids_jni::convert_orc_writer_reader_timezones(
    input,
    /*base_offset_us=*/int64_t{0},
    /*writer_tz_info_table=*/nullptr,
    /*writer_initial_offset=*/0,
    /*writer_raw_offset=*/0,
    rule,
    /*reader_tz_info_table=*/nullptr,
    /*reader_initial_offset=*/0,
    /*reader_raw_offset=*/0,
    rule,
    cudf::get_default_stream(),
    rmm::mr::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}
