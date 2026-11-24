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

#include "timezones.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

auto constexpr int64_min = std::numeric_limits<int64_t>::min();

using int32_col = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64_col = cudf::test::fixed_width_column_wrapper<int64_t>;

using seconds_col =
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>;

using millis_col =
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_s::rep>;

using micros_col =
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_us, cudf::timestamp_s::rep>;

// Global environment for temporary files used by the ORC end-to-end test.
auto const timezones_temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

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
    auto dst_col =
      cudf::make_lists_column(2,
                              dst_offsets.release(),
                              dst_child.release(),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
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
    /*writer_2015_year_base_offset_us=*/0,
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
  rule.has_dst         = 1;
  rule.dst_savings     = 3'600'000;
  rule.start_month     = 2;          // March
  rule.start_day       = 8;          // "first Sunday on or after the 8th" == 2nd Sunday
  rule.start_dow       = 1;          // Sunday
  rule.start_time      = 7'200'000;  // 02:00
  rule.start_time_mode = 1;          // STANDARD
  rule.start_mode      = 2;          // DOW_GE_DOM
  rule.end_month       = 10;         // November
  rule.end_day         = 1;          // "first Sunday on or after the 1st"
  rule.end_dow         = 1;
  rule.end_time        = 3'600'000;  // 01:00
  rule.end_time_mode   = 1;
  rule.end_mode        = 2;
  return rule;
}

[[nodiscard]] spark_rapids_jni::dst_rule make_dst_rule(int32_t start_mode,
                                                       int32_t start_day,
                                                       int32_t start_dow,
                                                       int32_t end_mode,
                                                       int32_t end_day,
                                                       int32_t end_dow,
                                                       int32_t start_month = 2,
                                                       int32_t end_month   = 9)
{
  spark_rapids_jni::dst_rule rule{};
  rule.has_dst         = 1;
  rule.dst_savings     = 3'600'000;
  rule.start_month     = start_month;
  rule.start_day       = start_day;
  rule.start_dow       = start_dow;
  rule.start_time      = 0;
  rule.start_time_mode = 2;  // UTC
  rule.start_mode      = start_mode;
  rule.end_month       = end_month;
  rule.end_day         = end_day;
  rule.end_dow         = end_dow;
  rule.end_time        = 0;
  rule.end_time_mode   = 2;
  rule.end_mode        = end_mode;
  return rule;
}

[[nodiscard]] std::unique_ptr<cudf::column> convert_utc_to_dst_reader(
  cudf::column_view const& input,
  spark_rapids_jni::dst_rule reader_dst,
  int64_t writer_2015_year_base_offset_us = 0)
{
  spark_rapids_jni::dst_rule no_dst{};
  no_dst.has_dst = 0;
  return spark_rapids_jni::convert_orc_writer_reader_timezones(
    input,
    writer_2015_year_base_offset_us,
    spark_rapids_jni::orc_tz_side{/*tz_info_table=*/nullptr, 0, 0, no_dst},
    spark_rapids_jni::orc_tz_side{/*tz_info_table=*/nullptr, 0, 0, reader_dst},
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());
}
}  // namespace

TEST_F(TimeZoneTest, ConvertOrcTimezonesAppliesWriter2015BaseOffset)
{
  spark_rapids_jni::dst_rule no_dst{};
  no_dst.has_dst = 0;

  auto const input    = micros_col{3'600'000'000L, 7'200'000'000L};
  auto const expected = micros_col{0L, 3'600'000'000L};
  auto const actual   = convert_utc_to_dst_reader(
    input, no_dst, /*writer_2015_year_base_offset_us=*/int64_t{3'600'000'000});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);

  auto writer_trans   = int64_col({0L, 1'000'000'000L});
  auto writer_offsets = int32_col({3'600'000, 0});
  auto writer_tv      = cudf::table_view({writer_trans, writer_offsets});

  auto const transition_input    = micros_col{1'000L};
  auto const transition_expected = micros_col{-1'000L};
  auto const transition_actual   = spark_rapids_jni::convert_orc_writer_reader_timezones(
    transition_input,
    /*writer_2015_year_base_offset_us=*/int64_t{2'000},
    spark_rapids_jni::orc_tz_side{&writer_tv, 0, 0, no_dst},
    spark_rapids_jni::orc_tz_side{nullptr, 0, 0, no_dst},
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(transition_expected, *transition_actual);
}

TEST_F(TimeZoneTest, ConvertOrcTimezonesAppliesWriterBaseOffsetToPreEpochTimestamp)
{
  spark_rapids_jni::dst_rule no_dst{};
  no_dst.has_dst = 0;

  // Decoder folds the negative nanos borrow in; the Shanghai base offset shifts the decode back
  // to the instant Apache ORC reconstructs.
  auto const input    = micros_col{21'086'883'873L};
  auto const expected = micros_col{-7'713'116'127L};
  auto const actual   = spark_rapids_jni::convert_orc_writer_reader_timezones(
    input,
    /*writer_2015_year_base_offset_us=*/int64_t{28'800'000'000},
    spark_rapids_jni::orc_tz_side{nullptr, 28'800'000, 28'800'000, no_dst},
    spark_rapids_jni::orc_tz_side{nullptr, 28'800'000, 28'800'000, no_dst},
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TEST_F(TimeZoneTest, ConvertOrcTimezonesPreEpochTimestampColumnShapes)
{
  spark_rapids_jni::dst_rule no_dst{};
  no_dst.has_dst     = 0;
  auto const convert = [&](cudf::column_view const& input) {
    return spark_rapids_jni::convert_orc_writer_reader_timezones(
      input,
      /*writer_2015_year_base_offset_us=*/int64_t{28'800'000'000},
      spark_rapids_jni::orc_tz_side{nullptr, 28'800'000, 28'800'000, no_dst},
      spark_rapids_jni::orc_tz_side{nullptr, 28'800'000, 28'800'000, no_dst},
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());
  };

  {
    auto const input    = micros_col{};
    auto const expected = micros_col{};
    auto const actual   = convert(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
  }

  {
    auto const input    = micros_col{{21'086'883'873L, 0L, 8'001'000L}, {true, false, true}};
    auto const expected = micros_col{{-7'713'116'127L, 0L, -28'791'999'000L}, {true, false, true}};
    auto const actual   = convert(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
  }

  {
    auto const source   = micros_col{{0L, 21'086'883'873L, 8'001'000L}, {false, true, true}};
    auto const input    = cudf::slice(source, {1, 3})[0];
    auto const expected = micros_col{{-7'713'116'127L, -28'791'999'000L}, {true, true}};
    auto const actual   = convert(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
  }
}

TEST_F(TimeZoneTest, ConvertOrcTimezonesPreEpochTimestampWithDstRule)
{
  auto const writer_dst = make_dst_rule(/*start_mode=*/2,
                                        /*start_day=*/1,
                                        /*start_dow=*/1,
                                        /*end_mode=*/2,
                                        /*end_day=*/1,
                                        /*end_dow=*/1,
                                        /*start_month=*/9,
                                        /*end_month=*/3);
  spark_rapids_jni::dst_rule no_dst{};
  no_dst.has_dst = 0;

  // Java passes TimeZone.getOffset at ORC's 2015-01-01 base instant: the southern-hemisphere
  // writer (raw UTC+10, DST +1h) is inside its DST window in January, so the helper passes
  // UTC+11. The decoder already folded in the borrow; the shift covers the DST writer offset.
  auto const input    = micros_col{31'886'883'873L};
  auto const expected = micros_col{31'886'883'873L};
  auto const actual   = spark_rapids_jni::convert_orc_writer_reader_timezones(
    input,
    /*writer_2015_year_base_offset_us=*/int64_t{39'600'000'000},
    spark_rapids_jni::orc_tz_side{nullptr, 36'000'000, 36'000'000, writer_dst},
    spark_rapids_jni::orc_tz_side{nullptr, 0, 0, no_dst},
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TEST_F(TimeZoneTest, ConvertOrcTimezonesTableOverloadAppliesWriter2015BaseOffset)
{
  auto const input    = micros_col{21'086'883'873L};
  auto const expected = micros_col{-7'713'116'127L};
  auto const actual =
    spark_rapids_jni::convert_orc_writer_reader_timezones(input,
                                                          /*writer_tz_info_table=*/nullptr,
                                                          /*writer_raw_offset=*/28'800'000,
                                                          /*writer_2015_year_base_offset_us=*/
                                                          int64_t{28'800'000'000},
                                                          /*reader_tz_info_table=*/nullptr,
                                                          /*reader_raw_offset=*/28'800'000,
                                                          cudf::get_default_stream(),
                                                          cudf::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TEST_F(TimeZoneTest, ConvertOrcTimezonesTableOverloadAcceptsWriterTransitionsWithBaseOffset)
{
  auto const input       = micros_col{1'000L};
  auto const expected    = micros_col{-1'000L};
  auto const transitions = int64_col({0L, 1'000'000'000L});
  auto const offsets     = int32_col({3'600'000, 0});
  auto const writer_tv   = cudf::table_view({transitions, offsets});

  auto const actual =
    spark_rapids_jni::convert_orc_writer_reader_timezones(input,
                                                          /*writer_tz_info_table=*/&writer_tv,
                                                          /*writer_raw_offset=*/0,
                                                          /*writer_2015_year_base_offset_us=*/2'000,
                                                          /*reader_tz_info_table=*/nullptr,
                                                          /*reader_raw_offset=*/0,
                                                          cudf::get_default_stream(),
                                                          cudf::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TEST_F(TimeZoneTest, ConvertOrcTimezonesAppliesWriter2015BaseOffsetAfterCudfBorrow)
{
  spark_rapids_jni::dst_rule no_dst{};
  no_dst.has_dst = 0;

  struct test_case {
    int64_t decoded_us;
    int64_t writer_2015_year_base_offset_us;
    int64_t expected_us;
  };
  // Borrow already folded in by the decoder: the base offset is a plain shift. The last case is
  // the negative-base window (decoded < 0, shifted value in [0, 1 s), borrowable fraction).
  auto const cases = std::array<test_case, 5>{{
    {-7'713'116'127L, 0L, -7'713'116'127L},
    {-116'127L, -1'000'000L, 883'873L},
    {9'000'999L, 10'000'000L, -999'001L},
    {8'001'000L, 10'000'000L, -1'999'000L},
    {-17'999'000'001L, -18'000'000'000L, 999'999L},
  }};

  for (auto const& test_case_data : cases) {
    auto const input    = micros_col{test_case_data.decoded_us};
    auto const expected = micros_col{test_case_data.expected_us};
    auto const actual   = spark_rapids_jni::convert_orc_writer_reader_timezones(
      input,
      test_case_data.writer_2015_year_base_offset_us,
      spark_rapids_jni::orc_tz_side{nullptr, 0, 0, no_dst},
      spark_rapids_jni::orc_tz_side{nullptr, 0, 0, no_dst},
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
  }
}

TEST_F(TimeZoneTest, ConvertOrcTimezonesReconstructsBorrowForUnresolvedWriterTimezone)
{
  spark_rapids_jni::dst_rule no_dst{};
  no_dst.has_dst = 0;

  struct test_case {
    int64_t decoded_us;
    int64_t writer_2015_year_base_offset_us;
    int64_t expected_us;
  };
  // UTC-fallback decode: the borrow is NOT folded in. Case 1 is the GMT+08:00-shaped analog of
  // the Shanghai reproducer (plain shift would be 1s late); case 2 lands in [0, 1 s).
  auto const cases = std::array<test_case, 2>{{
    {21'087'883'873L, 28'800'000'000L, -7'713'116'127L},
    {-1'116'127L, -1'000'000L, 883'873L},
  }};

  for (auto const& test_case_data : cases) {
    auto const input    = micros_col{test_case_data.decoded_us};
    auto const expected = micros_col{test_case_data.expected_us};
    auto const actual   = spark_rapids_jni::convert_orc_writer_reader_timezones(
      input,
      test_case_data.writer_2015_year_base_offset_us,
      spark_rapids_jni::orc_tz_side{nullptr, 0, 0, no_dst},
      spark_rapids_jni::orc_tz_side{nullptr, 0, 0, no_dst},
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref(),
      /*writer_reader_rules_differ=*/false,
      /*writer_borrow_applied=*/false);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
  }

  // The same inputs through the default (borrow already applied) path keep the plain-shift
  // results, guarding against a flag leak between the two modes.
  for (auto const& test_case_data : cases) {
    auto const input = micros_col{test_case_data.decoded_us};
    auto const expected =
      micros_col{test_case_data.decoded_us - test_case_data.writer_2015_year_base_offset_us};
    auto const actual = spark_rapids_jni::convert_orc_writer_reader_timezones(
      input,
      test_case_data.writer_2015_year_base_offset_us,
      spark_rapids_jni::orc_tz_side{nullptr, 0, 0, no_dst},
      spark_rapids_jni::orc_tz_side{nullptr, 0, 0, no_dst},
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
  }
}

// End-to-end: real ORC round trip with the decoder in the loop, so a submodule borrow-frame
// change fails here instead of silently matching the hand-wired inputs above.
TEST_F(TimeZoneTest, ConvertOrcTimezonesEndToEndDecoderBorrowFrame)
{
  // Pre-epoch fractional UTC instant of a UTC+08:00 writer: exercises the borrow and the
  // Unix-epoch crossing on the shift.
  auto const instant_us = -7'713'116'127L;

  micros_col const instant_col{instant_us};
  auto const table_view_in = cudf::table_view({instant_col});

  auto const filename =
    timezones_temp_env->get_temp_filepath("ConvertOrcTimezonesEndToEndDecoderBorrow.orc");
  {
    auto writer_options =
      cudf::io::orc_writer_options::builder(cudf::io::sink_info(filename), table_view_in);
    writer_options.writer_timezone("Asia/Shanghai");
    cudf::io::write_orc(writer_options.build());
  }

  auto read_options = cudf::io::orc_reader_options::builder(cudf::io::source_info(filename))
                        .timestamp_type(cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS})
                        .ignore_timezone_in_stripe_footer(true)
                        .build();
  auto const read_result = cudf::io::read_orc(read_options);
  auto const& decoded    = read_result.tbl->get_column(0);

  // The decode must equal the written instant plus the writer epoch offset (UTC+08:00 = 28.8e9 us).
  EXPECT_EQ(decoded.size(), 1);
  int64_t decoded_us = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(&decoded_us,
                                decoded.view().begin<cudf::timestamp_us>(),
                                sizeof(int64_t),
                                cudaMemcpyDefault,
                                cudf::get_default_stream().get()));
  CUDF_CUDA_TRY(cudaStreamSynchronize(cudf::get_default_stream().get()));
  EXPECT_EQ(decoded_us, instant_us + 28'800'000'000L);

  // The conversion must shift straight back to the instant Apache ORC reconstructs.
  spark_rapids_jni::dst_rule no_dst{};
  no_dst.has_dst      = 0;
  auto const expected = micros_col{instant_us};
  auto const actual   = spark_rapids_jni::convert_orc_writer_reader_timezones(
    decoded.view(),
    /*writer_2015_year_base_offset_us=*/int64_t{28'800'000'000},
    spark_rapids_jni::orc_tz_side{nullptr, 28'800'000, 28'800'000, no_dst},
    spark_rapids_jni::orc_tz_side{nullptr, 28'800'000, 28'800'000, no_dst},
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TEST_F(TimeZoneTest, ConvertOrcTimezonesRejectsInvalidTables)
{
  auto const input = micros_col{0L};
  spark_rapids_jni::dst_rule no_dst{};

  auto const transitions = int64_col({0L});
  auto const offsets     = int32_col({0});
  auto const one_column  = cudf::table_view({transitions});
  auto const wrong_types = cudf::table_view({offsets, offsets});

  EXPECT_THROW(static_cast<void>(spark_rapids_jni::convert_orc_writer_reader_timezones(
                 input,
                 /*writer_2015_year_base_offset_us=*/0,
                 spark_rapids_jni::orc_tz_side{&one_column, 0, 0, no_dst},
                 spark_rapids_jni::orc_tz_side{nullptr, 0, 0, no_dst})),
               cudf::logic_error);
  EXPECT_THROW(static_cast<void>(spark_rapids_jni::convert_orc_writer_reader_timezones(
                 input,
                 /*writer_2015_year_base_offset_us=*/0,
                 spark_rapids_jni::orc_tz_side{nullptr, 0, 0, no_dst},
                 spark_rapids_jni::orc_tz_side{&wrong_types, 0, 0, no_dst})),
               cudf::logic_error);
}

// DST path with no transition table: every instant resolves through the DST
// rule (compute_dst_offset). Writer is fixed UTC (offset 0), reader carries the
// US rule on a raw offset of 0, so a winter instant (standard) is unchanged and
// a summer instant (DST +1h) is shifted back one hour. 2030-01-15 is before the
// 2030 DST window (starts Mar 10) and 2030-07-15 is inside it (ends Nov 3).
TEST_F(TimeZoneTest, ConvertOrcTimezonesReaderDstBeyondTable)
{
  auto const reader_dst = make_us_dst_rule();
  spark_rapids_jni::dst_rule no_dst{};
  no_dst.has_dst = 0;

  auto const input    = micros_col{1894665600000000L, 1910304000000000L};
  auto const expected = micros_col{1894665600000000L, 1910300400000000L};
  auto const actual   = spark_rapids_jni::convert_orc_writer_reader_timezones(
    input,
    /*writer_2015_year_base_offset_us=*/int64_t{0},
    spark_rapids_jni::orc_tz_side{nullptr, 0, 0, no_dst},
    spark_rapids_jni::orc_tz_side{nullptr, 0, 0, reader_dst},
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TEST_F(TimeZoneTest, ConvertOrcTimezonesUsesInitialOffsetBeforeFirstTransition)
{
  auto reader_trans   = int64_col({1'000'000'000'000'000L});
  auto reader_offsets = int32_col({7'200'000});
  auto reader_tv      = cudf::table_view({reader_trans, reader_offsets});
  auto const rule     = make_us_dst_rule();

  // July 15, 2030 is inside the recurring DST window, but it is before the first table entry.
  // The historical initial offset must win over the fallback DST rule.
  auto const input    = micros_col{1'910'304'000'000'000L};
  auto const expected = micros_col{1'910'304'000'000'000L};
  auto const actual   = spark_rapids_jni::convert_orc_writer_reader_timezones(
    input,
    /*writer_2015_year_base_offset_us=*/0,
    spark_rapids_jni::orc_tz_side{nullptr, 0, 0},
    spark_rapids_jni::orc_tz_side{&reader_tv, 0, 0, rule},
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TEST_F(TimeZoneTest, ConvertOrcTimezonesDstRuleModes)
{
  // Mode 0 = DOM: exact March 15 through exact October 15.
  {
    auto const reader_dst = make_dst_rule(
      /*start_mode=*/0, /*start_day=*/15, /*start_dow=*/0, /*end_mode=*/0, 15, 0);
    auto const input =
      micros_col{1'899'720'000'000'000L, 1'899'806'400'000'000L, 1'918'296'000'000'000L};
    auto const expected =
      micros_col{1'899'720'000'000'000L, 1'899'802'800'000'000L, 1'918'296'000'000'000L};
    auto const actual = convert_utc_to_dst_reader(input, reader_dst);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
  }

  // Mode 1 = DOW_IN_MONTH with a positive occurrence: 2nd Sunday in March
  // through 1st Sunday in October.
  {
    auto const reader_dst = make_dst_rule(
      /*start_mode=*/1, /*start_day=*/2, /*start_dow=*/1, /*end_mode=*/1, 1, 1);
    auto const input =
      micros_col{1'899'288'000'000'000L, 1'899'374'400'000'000L, 1'917'518'400'000'000L};
    auto const expected =
      micros_col{1'899'288'000'000'000L, 1'899'370'800'000'000L, 1'917'518'400'000'000L};
    auto const actual = convert_utc_to_dst_reader(input, reader_dst);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
  }

  // Mode 1 also supports negative occurrences: last Sunday in March through
  // last Sunday in October.
  {
    auto const reader_dst = make_dst_rule(
      /*start_mode=*/1, /*start_day=*/-1, /*start_dow=*/1, /*end_mode=*/1, -1, 1);
    auto const input =
      micros_col{1'901'102'400'000'000L, 1'901'188'800'000'000L, 1'919'332'800'000'000L};
    auto const expected =
      micros_col{1'901'102'400'000'000L, 1'901'185'200'000'000L, 1'919'332'800'000'000L};
    auto const actual = convert_utc_to_dst_reader(input, reader_dst);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
  }

  // Mode 1 can normalize positive overflow into the following month.
  {
    auto const reader_dst = make_dst_rule(
      /*start_mode=*/1, /*start_day=*/5, /*start_dow=*/1, /*end_mode=*/0, 15, 0, 1);
    auto const input    = micros_col{1'898'510'400'000'000L, 1'898'730'000'000'000L};
    auto const expected = micros_col{1'898'510'400'000'000L, 1'898'726'400'000'000L};
    auto const actual   = convert_utc_to_dst_reader(input, reader_dst);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
  }

  // Mode 1 can normalize negative underflow into the previous month.
  {
    auto const reader_dst = make_dst_rule(
      /*start_mode=*/1, /*start_day=*/-5, /*start_dow=*/1, /*end_mode=*/0, 15, 0, 1);
    auto const input    = micros_col{1'895'659'200'000'000L, 1'895'706'000'000'000L};
    auto const expected = micros_col{1'895'659'200'000'000L, 1'895'702'400'000'000L};
    auto const actual   = convert_utc_to_dst_reader(input, reader_dst);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
  }

  // Mode 3 = DOW_LE_DOM: Sunday on or before March 20 through Sunday on or
  // before October 20.
  {
    auto const reader_dst = make_dst_rule(
      /*start_mode=*/3, /*start_day=*/20, /*start_dow=*/1, /*end_mode=*/3, 20, 1);
    auto const input =
      micros_col{1'899'892'800'000'000L, 1'899'979'200'000'000L, 1'918'728'000'000'000L};
    auto const expected =
      micros_col{1'899'892'800'000'000L, 1'899'975'600'000'000L, 1'918'728'000'000'000L};
    auto const actual = convert_utc_to_dst_reader(input, reader_dst);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
  }

  // Mode 3 can normalize an anchor beyond month length into the following month.
  {
    auto const reader_dst = make_dst_rule(
      /*start_mode=*/3, /*start_day=*/31, /*start_dow=*/7, /*end_mode=*/0, 15, 0, 1);
    auto const input    = micros_col{1'898'510'400'000'000L, 1'898'643'600'000'000L};
    auto const expected = micros_col{1'898'510'400'000'000L, 1'898'640'000'000'000L};
    auto const actual   = convert_utc_to_dst_reader(input, reader_dst);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
  }

  // Mode 3 can normalize a computed non-positive day into the previous month.
  {
    auto const reader_dst = make_dst_rule(
      /*start_mode=*/3, /*start_day=*/1, /*start_dow=*/1, /*end_mode=*/0, 15, 0);
    auto const input    = micros_col{1'898'078'400'000'000L, 1'898'164'800'000'000L};
    auto const expected = micros_col{1'898'078'400'000'000L, 1'898'161'200'000'000L};
    auto const actual   = convert_utc_to_dst_reader(input, reader_dst);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
  }

  // Mode 2 can normalize a computed day into the following month.
  {
    auto const reader_dst = make_dst_rule(
      /*start_mode=*/2, /*start_day=*/31, /*start_dow=*/2, /*end_mode=*/0, 15, 0);
    auto const input    = micros_col{1'901'188'800'000'000L, 1'901'235'600'000'000L};
    auto const expected = micros_col{1'901'188'800'000'000L, 1'901'232'000'000'000L};
    auto const actual   = convert_utc_to_dst_reader(input, reader_dst);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
  }
}

TEST_F(TimeZoneTest, ConvertOrcTimezonesWallTimeTransitions)
{
  auto reader_dst = make_dst_rule(
    /*start_mode=*/2, /*start_day=*/8, /*start_dow=*/1, /*end_mode=*/2, 1, 1);
  reader_dst.start_time_mode = 0;          // WALL_TIME
  reader_dst.end_time_mode   = 0;          // WALL_TIME
  reader_dst.start_time      = 7'200'000;  // 02:00 wall
  reader_dst.end_time        = 3'600'000;  // 01:00 wall

  // In 2030, DST starts at 02:00 UTC on March 10. The ORC conversion's second offset lookup
  // self-corrects within the resulting one-hour gap, so the observable shift starts at 03:00.
  // DST ends at 00:00 UTC on October 6 because 01:00 wall time still includes the one-hour saving.
  auto const input = micros_col{
    1'899'341'999'999'999L, 1'899'342'000'000'000L, 1'917'475'199'999'999L, 1'917'475'200'000'000L};
  auto const expected = micros_col{
    1'899'341'999'999'999L, 1'899'338'400'000'000L, 1'917'471'599'999'999L, 1'917'475'200'000'000L};
  auto const actual = convert_utc_to_dst_reader(input, reader_dst);

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
    /*writer_2015_year_base_offset_us=*/int64_t{0},
    spark_rapids_jni::orc_tz_side{nullptr, 0, 0, rule},
    spark_rapids_jni::orc_tz_side{nullptr, 0, 0, rule},
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}
