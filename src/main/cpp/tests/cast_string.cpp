/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cast_string.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace cudf;

template <typename T>
struct StringToIntegerTests : public test::BaseFixture {};

struct StringToDecimalTests : public test::BaseFixture {};

struct ParseTimestampWithFormatTests : public test::BaseFixture {};

struct HexToBytesTests : public test::BaseFixture {};

struct ParseCsvDateWithFormatTests : public test::BaseFixture {};

template <typename T>
struct StringToFloatTests : public test::BaseFixture {};

TYPED_TEST_SUITE(StringToIntegerTests, cudf::test::IntegralTypesNotBool);
TYPED_TEST_SUITE(StringToFloatTests, cudf::test::FloatingPointTypes);

TEST_F(HexToBytesTests, SparkUnhexSemantics)
{
  std::vector<std::string> values{"4142",
                                  "aF",
                                  "F",
                                  "ABC",
                                  "",
                                  "GG",
                                  "0xFF",
                                  " F",
                                  "00ff",
                                  "7F80",
                                  "00",
                                  "bad",
                                  "001122334455667788",
                                  "ignored"};
  std::vector<bool> valid(values.size(), true);
  valid.back()     = false;
  auto const input = test::strings_column_wrapper(values.begin(), values.end(), valid.begin());
  auto const actual =
    spark_rapids_jni::hex_to_bytes(cudf::strings_column_view{input}, cudf::get_default_stream());
  auto const expected =
    test::lists_column_wrapper<uint8_t>({{0x41, 0x42},
                                         {0xAF},
                                         {0x0F},
                                         {0x0A, 0xBC},
                                         {},
                                         {},
                                         {},
                                         {},
                                         {0x00, 0xFF},
                                         {0x7F, 0x80},
                                         {0x00},
                                         {0x0B, 0xAD},
                                         {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88},
                                         {}},
                                        test::iterators::nulls_at({5, 6, 7, 13}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view(), expected);
}

TEST_F(HexToBytesTests, SlicedInput)
{
  std::vector<std::string> values{"outside", "AB", "bad!", "ignored", "FF", "outside"};
  std::vector<bool> valid{true, true, true, false, true, true};
  auto const strings = test::strings_column_wrapper(values.begin(), values.end(), valid.begin());
  auto const sliced  = cudf::slice(strings, {1, 5})[0];
  auto const actual =
    spark_rapids_jni::hex_to_bytes(cudf::strings_column_view{sliced}, cudf::get_default_stream());
  auto const expected = test::lists_column_wrapper<uint8_t>({{0xAB}, {}, {}, {0xFF}},
                                                            test::iterators::nulls_at({1, 2}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view(), expected);
}

TEST_F(HexToBytesTests, EmptyInput)
{
  auto const input = test::strings_column_wrapper{};
  auto const actual =
    spark_rapids_jni::hex_to_bytes(cudf::strings_column_view{input}, cudf::get_default_stream());
  auto const expected = test::lists_column_wrapper<uint8_t>{};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view(), expected);
}

TEST_F(HexToBytesTests, LongRows)
{
  std::string valid(131'073, 'A');
  std::string invalid = valid;
  invalid.back()      = 'G';
  std::vector<std::string> values{valid, invalid, "ignored"};
  std::vector<bool> input_validity{true, true, false};
  auto const input =
    test::strings_column_wrapper(values.begin(), values.end(), input_validity.begin());
  auto const actual =
    spark_rapids_jni::hex_to_bytes(cudf::strings_column_view{input}, cudf::get_default_stream());

  std::vector<uint8_t> expected_bytes(65'537, 0xAA);
  expected_bytes.front() = 0x0A;
  auto const expected_child =
    test::fixed_width_column_wrapper<uint8_t>(expected_bytes.begin(), expected_bytes.end());
  auto const expected_offsets =
    test::fixed_width_column_wrapper<cudf::size_type>{0, 65'537, 65'537, 65'537};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view().child(0), expected_offsets);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view().child(1), expected_child);
  EXPECT_EQ(actual->view().null_count(), 2);
}

TEST_F(HexToBytesTests, SlicedLongRow)
{
  std::vector<std::string> values{"outside", std::string(131'072, 'F'), "outside"};
  auto const input  = test::strings_column_wrapper(values.begin(), values.end());
  auto const sliced = cudf::slice(input, {1, 2})[0];
  auto const actual =
    spark_rapids_jni::hex_to_bytes(cudf::strings_column_view{sliced}, cudf::get_default_stream());

  std::vector<uint8_t> expected_bytes(65'536, 0xFF);
  auto const expected_child =
    test::fixed_width_column_wrapper<uint8_t>(expected_bytes.begin(), expected_bytes.end());
  auto const expected_offsets = test::fixed_width_column_wrapper<cudf::size_type>{0, 65'536};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view().child(0), expected_offsets);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view().child(1), expected_child);
  EXPECT_EQ(actual->view().null_count(), 0);
}

TEST_F(HexToBytesTests, ManyShortRows)
{
  std::vector<std::string> values(8193, "aF");
  auto const input = test::strings_column_wrapper(values.begin(), values.end());
  auto const actual =
    spark_rapids_jni::hex_to_bytes(cudf::strings_column_view{input}, cudf::get_default_stream());

  std::vector<uint8_t> expected_bytes(values.size(), 0xAF);
  auto const expected_child =
    test::fixed_width_column_wrapper<uint8_t>(expected_bytes.begin(), expected_bytes.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view().child(1), expected_child);
  EXPECT_EQ(actual->view().null_count(), 0);
}

TEST_F(HexToBytesTests, ManyRowsWithLongValues)
{
  std::vector<std::string> values(8193, "aF");
  values[0]        = std::string(131'073, 'A');
  values[1]        = values[0];
  values[1].back() = 'G';
  std::vector<bool> validity(values.size(), true);
  validity[2]      = false;
  auto const input = test::strings_column_wrapper(values.begin(), values.end(), validity.begin());
  auto const actual =
    spark_rapids_jni::hex_to_bytes(cudf::strings_column_view{input}, cudf::get_default_stream());

  std::vector<cudf::size_type> expected_offsets(values.size() + 1);
  for (std::size_t row = 0; row < values.size(); ++row) {
    expected_offsets[row + 1] = expected_offsets[row] + (row == 0 ? 65'537 : row < 3 ? 0 : 1);
  }
  auto expected_offsets_column = test::fixed_width_column_wrapper<cudf::size_type>(
    expected_offsets.begin(), expected_offsets.end());
  std::vector<uint8_t> expected_bytes(expected_offsets.back(), 0xAF);
  std::fill_n(expected_bytes.begin(), 65'537, 0xAA);
  expected_bytes.front() = 0x0A;
  auto expected_child =
    test::fixed_width_column_wrapper<uint8_t>(expected_bytes.begin(), expected_bytes.end());
  auto const valid_iter = test::iterators::nulls_at({1, 2});
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(valid_iter, valid_iter + values.size());
  auto const expected = cudf::make_lists_column(values.size(),
                                                expected_offsets_column.release(),
                                                expected_child.release(),
                                                null_count,
                                                std::move(null_mask));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view(), expected->view());
}

TEST_F(HexToBytesTests, ManyRowsWithSparseLongValues)
{
  std::vector<std::string> values(20'001, "aF");
  values[0]        = std::string(131'073, 'A');
  values[1]        = values[0];
  values[1].back() = 'G';
  std::vector<bool> validity(values.size(), true);
  validity[2]      = false;
  auto const input = test::strings_column_wrapper(values.begin(), values.end(), validity.begin());
  auto const actual =
    spark_rapids_jni::hex_to_bytes(cudf::strings_column_view{input}, cudf::get_default_stream());

  std::vector<cudf::size_type> expected_offsets(values.size() + 1);
  for (std::size_t row = 0; row < values.size(); ++row) {
    expected_offsets[row + 1] = expected_offsets[row] + (row == 0 ? 65'537 : row < 3 ? 0 : 1);
  }
  auto expected_offsets_column = test::fixed_width_column_wrapper<cudf::size_type>(
    expected_offsets.begin(), expected_offsets.end());
  std::vector<uint8_t> expected_bytes(expected_offsets.back(), 0xAF);
  std::fill_n(expected_bytes.begin(), 65'537, 0xAA);
  expected_bytes.front() = 0x0A;
  auto expected_child =
    test::fixed_width_column_wrapper<uint8_t>(expected_bytes.begin(), expected_bytes.end());
  auto const valid_iter = test::iterators::nulls_at({1, 2});
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(valid_iter, valid_iter + values.size());
  auto const expected = cudf::make_lists_column(values.size(),
                                                expected_offsets_column.release(),
                                                expected_child.release(),
                                                null_count,
                                                std::move(null_mask));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view(), expected->view());
}

TEST_F(HexToBytesTests, MixedLengthsAcrossChunks)
{
  // Rows of every length below `max_length` start and end at all positions within the decoder's
  // character chunks. Invalid characters appear at the first, middle, and last positions. A batch
  // of rows up to 32 characters is decoded one thread per row, and one with longer rows by chunks.
  for (int const max_length : {33, 100}) {
    SCOPED_TRACE(max_length);
    std::vector<std::string> values;
    for (int row = 0; row < 3'000; ++row) {
      auto const length = row % max_length;
      std::string value(length, ' ');
      for (int i = 0; i < length; ++i) {
        value[i] = "0123456789abcdefABCDEF"[(row + i) % 22];
      }
      if (length > 0 && row % 7 == 0) { value[(row / 7 % 3) * (length - 1) / 2] = 'g'; }
      values.push_back(value);
    }
    auto input = test::strings_column_wrapper(values.begin(), values.end()).release();
    // Null rows keep their characters, which must be ignored.
    auto const null_iter = test::iterators::nulls_at({5, 1'234, 2'999});
    auto [input_mask, input_null_count] =
      cudf::test::detail::make_null_mask(null_iter, null_iter + values.size());
    input->set_null_mask(std::move(input_mask), input_null_count);
    auto const sliced = cudf::slice(input->view(), {3, 3'000})[0];
    auto const actual =
      spark_rapids_jni::hex_to_bytes(cudf::strings_column_view{sliced}, cudf::get_default_stream());

    auto const decode = [](char value) {
      return static_cast<uint8_t>(value <= '9' ? value - '0' : (value | 0x20) - 'a' + 10);
    };
    std::vector<cudf::size_type> expected_offsets{0};
    std::vector<uint8_t> expected_bytes;
    std::vector<bool> expected_valid;
    for (std::size_t row = 3; row < values.size(); ++row) {
      auto const& value = values[row];
      auto const valid  = null_iter[row] && value.find('g') == std::string::npos;
      if (valid) {
        std::size_t i = 0;
        if (value.size() % 2 != 0) { expected_bytes.push_back(decode(value[i++])); }
        for (; i < value.size(); i += 2) {
          expected_bytes.push_back((decode(value[i]) << 4) | decode(value[i + 1]));
        }
      }
      expected_valid.push_back(valid);
      expected_offsets.push_back(expected_bytes.size());
    }
    auto expected_offsets_column = test::fixed_width_column_wrapper<cudf::size_type>(
      expected_offsets.begin(), expected_offsets.end());
    auto expected_child =
      test::fixed_width_column_wrapper<uint8_t>(expected_bytes.begin(), expected_bytes.end());
    auto [null_mask, null_count] =
      cudf::test::detail::make_null_mask(expected_valid.begin(), expected_valid.end());
    auto const expected = cudf::make_lists_column(expected_valid.size(),
                                                  expected_offsets_column.release(),
                                                  expected_child.release(),
                                                  null_count,
                                                  std::move(null_mask));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view(), expected->view());
  }
}

TEST_F(HexToBytesTests, SlicedSparseLongRowsAtThreshold)
{
  std::vector<std::string> values(20'003, "aF");
  values[10'002]        = std::string(312, 'A');
  values[20'001]        = std::string(312, 'A');
  values[20'001].back() = 'G';
  std::vector<bool> validity(values.size(), true);
  validity[10'003]  = false;
  auto const input  = test::strings_column_wrapper(values.begin(), values.end(), validity.begin());
  auto const sliced = cudf::slice(input, {1, 20'002})[0];
  auto const actual =
    spark_rapids_jni::hex_to_bytes(cudf::strings_column_view{sliced}, cudf::get_default_stream());

  std::vector<cudf::size_type> expected_offsets(20'002);
  for (std::size_t row = 0; row < 20'001; ++row) {
    auto const row_size       = row == 10'001 ? 156 : row == 10'002 || row == 20'000 ? 0 : 1;
    expected_offsets[row + 1] = expected_offsets[row] + row_size;
  }
  auto expected_offsets_column = test::fixed_width_column_wrapper<cudf::size_type>(
    expected_offsets.begin(), expected_offsets.end());

  std::vector<uint8_t> expected_bytes(expected_offsets.back(), 0xAF);
  std::fill_n(expected_bytes.begin() + expected_offsets[10'001], 156, 0xAA);
  auto expected_child =
    test::fixed_width_column_wrapper<uint8_t>(expected_bytes.begin(), expected_bytes.end());
  auto const valid_iter = test::iterators::nulls_at({10'002, 20'000});
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(valid_iter, valid_iter + 20'001);
  auto const expected = cudf::make_lists_column(20'001,
                                                expected_offsets_column.release(),
                                                expected_child.release(),
                                                null_count,
                                                std::move(null_mask));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view(), expected->view());
}

TEST_F(HexToBytesTests, CompactsRowsAfterInvalidRows)
{
  // Rows found invalid while decoding are dropped from the output, so that the bytes of later
  // rows move by amounts that are not word multiples, and long rows span many copy chunks.
  std::vector<std::string> values;
  for (int row = 0; row < 60; ++row) {
    auto const length = row % 3 == 0 ? 9'001 + row : row % 3 == 1 ? 1 + row : 5'000 + row;
    std::string value(length, ' ');
    for (int i = 0; i < length; ++i) {
      value[i] = "0123456789abcdef"[(row * 7 + i) % 16];
    }
    if (row % 4 == 1) { value[length / 2] = 'x'; }
    values.push_back(value);
  }
  auto const input = test::strings_column_wrapper(values.begin(), values.end());
  auto const actual =
    spark_rapids_jni::hex_to_bytes(cudf::strings_column_view{input}, cudf::get_default_stream());

  auto const decode = [](char value) {
    return static_cast<uint8_t>(value <= '9' ? value - '0' : value - 'a' + 10);
  };
  std::vector<cudf::size_type> expected_offsets{0};
  std::vector<uint8_t> expected_bytes;
  std::vector<bool> expected_valid;
  for (auto const& value : values) {
    auto const valid = value.find('x') == std::string::npos;
    if (valid) {
      std::size_t i = 0;
      if (value.size() % 2 != 0) { expected_bytes.push_back(decode(value[i++])); }
      for (; i < value.size(); i += 2) {
        expected_bytes.push_back((decode(value[i]) << 4) | decode(value[i + 1]));
      }
    }
    expected_valid.push_back(valid);
    expected_offsets.push_back(expected_bytes.size());
  }
  auto expected_offsets_column = test::fixed_width_column_wrapper<cudf::size_type>(
    expected_offsets.begin(), expected_offsets.end());
  auto expected_child =
    test::fixed_width_column_wrapper<uint8_t>(expected_bytes.begin(), expected_bytes.end());
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(expected_valid.begin(), expected_valid.end());
  auto const expected = cudf::make_lists_column(values.size(),
                                                expected_offsets_column.release(),
                                                expected_child.release(),
                                                null_count,
                                                std::move(null_mask));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(actual->view(), expected->view());
}

TEST_F(ParseTimestampWithFormatTests, RejectsConflictingPolicies)
{
  auto const strings = test::strings_column_wrapper{"2024-05-06"};
  strings_column_view scv{strings};

  EXPECT_THROW(spark_rapids_jni::parse_timestamp_strings_with_format(scv, "yyyy-MM-dd", true, true),
               std::invalid_argument);
}

TEST_F(ParseTimestampWithFormatTests, RepeatedFieldsPreserveStrictPolicyDeviation)
{
  // LEGACY matches Spark's last-value behavior. CORRECTED/EXCEPTION preserve a known
  // JNI deviation: Spark rejects conflicting repeated fields. These patterns are
  // outside the plugin's expression format allowlists.
  auto const strings           = test::strings_column_wrapper{"2020-05-01-05", "2020-05-01-06"};
  constexpr int64_t us_per_day = 86'400'000'000;
  auto const expected =
    test::fixed_width_column_wrapper<timestamp_us>({18'383 * us_per_day, 18'414 * us_per_day});
  for (bool const legacy : {false, true}) {
    auto const result = spark_rapids_jni::parse_timestamp_strings_with_format(
      strings_column_view{strings}, "yyyy-MM-dd-MM", legacy, false);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
  auto const exception_result = spark_rapids_jni::parse_timestamp_strings_with_format(
    strings_column_view{strings}, "yyyy-MM-dd-MM", false, true);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(exception_result->view(), expected);
}

TEST_F(ParseCsvDateWithFormatTests, StrictInput)
{
  auto const strings = test::strings_column_wrapper({"2020-05-01",
                                                     "2020-5-1",
                                                     " 2020-05-01",
                                                     "2020-05-01 ",
                                                     "200-01-01",
                                                     "2020-05-01T10:11:12 PST",
                                                     "2020-02-30",
                                                     "2020-02-29",
                                                     "",
                                                     "+",
                                                     "-",
                                                     "2020-05-",
                                                     ""},
                                                    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});
  auto const result  = spark_rapids_jni::parse_csv_date_strings(
    strings_column_view{strings},
    "yyyy-MM-dd",
    {.enable_fallback = false, .trim_control_characters = true});
  auto const expected = test::fixed_width_column_wrapper<timestamp_D>(
    {18383, 0, 0, 0, 0, 0, 0, 18321, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(ParseCsvDateWithFormatTests, PartialFormatConsumesAllInput)
{
  auto const strings = test::strings_column_wrapper{"2020-02", "2020-02-29", "2020-2", "2020-13"};
  auto const result  = spark_rapids_jni::parse_csv_date_strings(
    strings_column_view{strings},
    "yyyy-MM",
    {.enable_fallback = false, .trim_control_characters = true});
  auto const expected =
    test::fixed_width_column_wrapper<timestamp_D>({18293, 0, 0, 0}, {1, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(ParseCsvDateWithFormatTests, RepeatedFieldsRequireMatchingValues)
{
  struct example {
    std::string format;
    std::string matching;
    std::string conflicting;
    int32_t days;
  };
  std::vector<example> const examples{
    {"yyyy-MM-dd-MM", "2020-05-01-05", "2020-05-01-06", 18'383},
    {"yyyy-MM-dd-MM", "2020-05-01-05", "2020-13-01-05", 18'383},
    {"yyyy-MM-dd-yyyy", "1970-01-01-1970", "1970-01-01-1971", 0},
    {"yyyy-MM-dd-MM", "1970-01-01-01", "1970-01-01-02", 0},
    {"yyyy-MM-dd-dd", "1970-01-01-01", "1970-01-01-02", 0},
    {"yyyy-MM-dd HH:mm:ss-HH", "1970-01-01 00:00:00-00", "1970-01-01 00:00:00-01", 0},
    {"yyyy-MM-dd HH:mm:ss-mm", "1970-01-01 00:00:00-00", "1970-01-01 00:00:00-01", 0},
    {"yyyy-MM-dd HH:mm:ss-ss", "1970-01-01 00:00:00-00", "1970-01-01 00:00:00-01", 0},
    {"yyyy-MM-dd-MM-MM", "1970-01-01-01-01", "1970-01-01-02-01", 0},
    {"yyyy-MM-dd-yyyy", "+02020-05-01-2020", "+02020-05-01-2021", 18'383}};
  for (auto const& e : examples) {
    SCOPED_TRACE(::testing::Message() << "format=" << e.format << ", matching=" << e.matching
                                      << ", conflicting=" << e.conflicting);
    auto const strings = test::strings_column_wrapper{e.matching, e.conflicting};
    auto const result  = spark_rapids_jni::parse_csv_date_strings(
      strings_column_view{strings},
      e.format,
      {.enable_fallback = false, .trim_control_characters = true});
    auto const expected = test::fixed_width_column_wrapper<timestamp_D>({e.days, 0}, {1, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}

TEST_F(ParseCsvDateWithFormatTests, RepeatedFieldConflictUsesConfiguredFallback)
{
  auto const strings = test::strings_column_wrapper{
    "2020-05-01 00:00:00-01", "2020-02-30 00:00:00-01", "2020-05-01 00:00:00-00"};
  auto const strict = spark_rapids_jni::parse_csv_date_strings(
    strings_column_view{strings},
    "yyyy-MM-dd HH:mm:ss-HH",
    {.enable_fallback = false, .trim_control_characters = true});
  auto const strict_expected =
    test::fixed_width_column_wrapper<timestamp_D>({0, 0, 18'383}, {0, 0, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strict->view(), strict_expected);
  auto const fallback = spark_rapids_jni::parse_csv_date_strings(
    strings_column_view{strings},
    "yyyy-MM-dd HH:mm:ss-HH",
    {.enable_fallback = true, .trim_control_characters = true});
  auto const fallback_expected =
    test::fixed_width_column_wrapper<timestamp_D>({18'383, 0, 18'383}, {1, 0, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(fallback->view(), fallback_expected);
}

TEST_F(ParseCsvDateWithFormatTests, DateTimeFormatValidatesTime)
{
  auto const strings = test::strings_column_wrapper{"2020-05-01 10:11:12",
                                                    "2020-05-01 24:00:00",
                                                    "2020-05-01T10:11:12",
                                                    "2020-05-01 10:11:12 PST",
                                                    "2020-05-01 10:11:60",
                                                    "+5881580-07-11 23:59:59",
                                                    "-5877641-06-23 00:00:00"};
  auto const result  = spark_rapids_jni::parse_csv_date_strings(
    strings_column_view{strings},
    "yyyy-MM-dd HH:mm:ss",
    {.enable_fallback = false, .trim_control_characters = true});
  auto const expected = test::fixed_width_column_wrapper<timestamp_D>(
    {18383, 0, 0, 0, 0, std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::min()},
    {1, 0, 0, 0, 0, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(ParseCsvDateWithFormatTests, SignedYearsAndEpochDayBounds)
{
  auto const strings = test::strings_column_wrapper{"0000-01-01",
                                                    "-0000-01-01",
                                                    "-0001-01-01",
                                                    "+12345-01-01",
                                                    "+02020-05-01",
                                                    "+2020-05-01",
                                                    "02020-05-01",
                                                    "+5881580-07-11",
                                                    "+5881580-07-12",
                                                    "-5877641-06-23",
                                                    "-5877641-06-22",
                                                    "+999999999-01-01"};
  auto const result  = spark_rapids_jni::parse_csv_date_strings(
    strings_column_view{strings},
    "yyyy-MM-dd",
    {.enable_fallback = false, .trim_control_characters = true});
  auto const expected =
    test::fixed_width_column_wrapper<timestamp_D>({-719528,
                                                   0,
                                                   -719893,
                                                   3789391,
                                                   18383,
                                                   0,
                                                   0,
                                                   std::numeric_limits<int32_t>::max(),
                                                   0,
                                                   std::numeric_limits<int32_t>::min(),
                                                   0,
                                                   0},
                                                  {1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(ParseCsvDateWithFormatTests, IntegerYearOverflowIsNull)
{
  auto const strings = test::strings_column_wrapper{"+2147483648-05-01",
                                                    "-2147483648-05-01",
                                                    "+4294969316-05-01",
                                                    "-4294969316-05-01",
                                                    "+0000002020-05-01"};
  auto const result  = spark_rapids_jni::parse_csv_date_strings(
    strings_column_view{strings},
    "yyyy-MM-dd",
    {.enable_fallback = false, .trim_control_characters = true});
  auto const expected =
    test::fixed_width_column_wrapper<timestamp_D>({0, 0, 0, 0, 18'383}, {0, 0, 0, 0, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(ParseCsvDateWithFormatTests, SliceAndEmpty)
{
  auto const strings = test::strings_column_wrapper(
    {"invalid", "2020-05-01", "", "2020-02-29", "invalid"}, {1, 1, 0, 1, 1});
  auto const views  = cudf::slice(strings, {1, 4, 0, 0, 2, 3});
  auto const result = spark_rapids_jni::parse_csv_date_strings(
    strings_column_view{views[0]},
    "yyyy-MM-dd",
    {.enable_fallback = false, .trim_control_characters = true});
  auto const expected = test::fixed_width_column_wrapper<timestamp_D>({18383, 0, 18321}, {1, 0, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  auto const empty = spark_rapids_jni::parse_csv_date_strings(
    strings_column_view{views[1]},
    "yyyy-MM-dd",
    {.enable_fallback = false, .trim_control_characters = true});
  EXPECT_EQ(empty->type(), data_type{type_id::TIMESTAMP_DAYS});
  EXPECT_EQ(empty->size(), 0);
  auto const null_result = spark_rapids_jni::parse_csv_date_strings(
    strings_column_view{views[2]},
    std::nullopt,
    {.enable_fallback = true, .trim_control_characters = true});
  auto const null_expected = test::fixed_width_column_wrapper<timestamp_D>({0}, {0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(null_result->view(), null_expected);
}

TEST_F(ParseCsvDateWithFormatTests, AllNumericFormatsUseStrictTokens)
{
  struct example {
    char const* format;
    char const* input;
    int days;
  };
  std::vector<example> const examples{{"yyyy-MM-dd", "2020-05-01", 18383},
                                      {"yyyy/MM/dd", "2020/05/01", 18383},
                                      {"yyyy-MM", "2020-05", 18383},
                                      {"yyyy/MM", "2020/05", 18383},
                                      {"dd/MM/yyyy", "01/05/2020", 18383},
                                      {"yyyy-MM-dd HH:mm:ss", "2020-05-01 10:11:12", 18383},
                                      {"MM-dd", "05-01", 120},
                                      {"MM/dd", "05/01", 120},
                                      {"dd-MM", "01-05", 120},
                                      {"dd/MM", "01/05", 120},
                                      {"MM/yyyy", "05/2020", 18383},
                                      {"MM-yyyy", "05-2020", 18383},
                                      {"MM/dd/yyyy", "05/01/2020", 18383},
                                      {"MM-dd-yyyy", "05-01-2020", 18383},
                                      {"MMyyyy", "052020", 18383},
                                      {"yyyyMMdd", "20200501", 18383}};
  for (auto const& e : examples) {
    SCOPED_TRACE(e.format);
    auto const strings = test::strings_column_wrapper{e.input, std::string(e.input) + "junk"};
    auto const result  = spark_rapids_jni::parse_csv_date_strings(
      strings_column_view{strings},
      std::string(e.format),
      {.enable_fallback = false, .trim_control_characters = true});
    auto const expected = test::fixed_width_column_wrapper<timestamp_D>({e.days, 0}, {1, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}

TEST_F(ParseCsvDateWithFormatTests, StrictSlashDateWidths)
{
  auto const strings = test::strings_column_wrapper{
    "2020/05/01", "2020/5/01", "2020/05/1", "2020/005/01", "2020/05/001"};
  auto const result = spark_rapids_jni::parse_csv_date_strings(
    strings_column_view{strings},
    "yyyy/MM/dd",
    {.enable_fallback = false, .trim_control_characters = true});
  auto const expected =
    test::fixed_width_column_wrapper<timestamp_D>({18'383, 0, 0, 0, 0}, {1, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(ParseCsvDateWithFormatTests, DefaultGrammarAndCsvFallback)
{
  auto const strings  = test::strings_column_wrapper{"2020",
                                                    "2020-05",
                                                    " 2020-5-1 ",
                                                    "2020-05-01T10:11:12 PST",
                                                    "200-01-01",
                                                    "\x7f"
                                                     "2020-05-01"
                                                     "\x7f",
                                                    "20GMT20-05-01",
                                                    "GMT2020"};
  auto const expected = test::fixed_width_column_wrapper<timestamp_D>(
    {18262, 18383, 18383, 18383, 0, 18383, 18383, 18262}, {1, 1, 1, 1, 0, 1, 1, 1});
  auto const result = spark_rapids_jni::parse_csv_date_strings(
    strings_column_view{strings},
    std::nullopt,
    {.enable_fallback = true, .trim_control_characters = true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  auto const no_fallback = spark_rapids_jni::parse_csv_date_strings(
    strings_column_view{strings},
    std::nullopt,
    {.enable_fallback = false, .trim_control_characters = true});
  auto const no_fallback_expected = test::fixed_width_column_wrapper<timestamp_D>(
    {18'262, 18'383, 18'383, 18'383, 0, 18'383, 0, 0}, {1, 1, 1, 1, 0, 1, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(no_fallback->view(), no_fallback_expected);
  auto const formatted = spark_rapids_jni::parse_csv_date_strings(
    strings_column_view{strings},
    "yyyy/MM/dd",
    {.enable_fallback = true, .trim_control_characters = true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(formatted->view(), expected);
  auto const old_trim_expected = test::fixed_width_column_wrapper<timestamp_D>(
    {18262, 18383, 18383, 18383, 0, 0, 18383, 18262}, {1, 1, 1, 1, 0, 0, 1, 1});
  auto const old_trim = spark_rapids_jni::parse_csv_date_strings(
    strings_column_view{strings},
    "yyyy/MM/dd",
    {.enable_fallback = true, .trim_control_characters = false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(old_trim->view(), old_trim_expected);
}

TEST_F(ParseCsvDateWithFormatTests, CompleteDateSuffixes)
{
  struct example {
    std::string input;
    bool default_valid;
    bool fallback_valid;
    bool requires_control_trim = false;
  };
  std::vector<example> const examples{
    {"2024-01-01", true, true},
    {"2024-01-01 ", true, true},
    {"2024-01-01 x", true, true},
    {"2024-01-01Tx", true, true},
    {"2024-01-01" + std::string(1'000'000, ' ') + "x", true, true},
    {"2024-01-01\tx", false, false},
    {"2024-01-01\t", true, true},
    {"2024-01-01\x7f", true, true, true},
    {"2024-01-01\x7fx", false, false},
    {"2024-01-01GMT", false, true},
    {"2024-01-01GMTGMT x", false, true},
    {"2024-01-01GMTTx", false, true},
    {"2024-01-01\tGMT", false, true},
    {"2024-01-01GMT\tx", false, false},
    {"GMT20GMT24-01-01GMT x", false, true},
    {"2024 x", false, false},
    {"2024-01Tx", false, false},
    {"2024\t", true, true},
    {"2024-01\t", true, true},
    {"2024-02-30 x", false, false},
    {"+5881580-07-12 x", false, false}};
  std::vector<std::string> inputs;
  for (auto const& e : examples) {
    inputs.push_back(e.input);
  }
  auto const strings = test::strings_column_wrapper(inputs.begin(), inputs.end());
  // Complete and partial valid dates above all resolve to 2024-01-01.
  std::vector<int> const days(examples.size(), 19'723);
  for (bool const trim_control_characters : {false, true}) {
    SCOPED_TRACE(trim_control_characters);
    std::vector<bool> default_validity;
    std::vector<bool> fallback_validity;
    for (auto const& e : examples) {
      bool const trim_allowed = !e.requires_control_trim || trim_control_characters;
      default_validity.push_back(e.default_valid && trim_allowed);
      fallback_validity.push_back(e.fallback_valid && trim_allowed);
    }
    auto const default_expected = test::fixed_width_column_wrapper<timestamp_D>(
      days.begin(), days.end(), default_validity.begin());
    auto const default_result = spark_rapids_jni::parse_csv_date_strings(
      strings_column_view{strings},
      std::nullopt,
      {.enable_fallback = false, .trim_control_characters = trim_control_characters});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(default_result->view(), default_expected);
    auto const fallback_expected = test::fixed_width_column_wrapper<timestamp_D>(
      days.begin(), days.end(), fallback_validity.begin());
    auto const fallback_result = spark_rapids_jni::parse_csv_date_strings(
      strings_column_view{strings},
      "yyyy/MM/dd",
      {.enable_fallback = true, .trim_control_characters = trim_control_characters});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(fallback_result->view(), fallback_expected);
  }
}

TEST_F(ParseCsvDateWithFormatTests, ExtendedYearWidths)
{
  auto const strings = test::strings_column_wrapper{"+0000002020-05-01",
                                                    "-0000002020-05-01",
                                                    "+0000000000000002020-05-01",
                                                    "+00000000000000002020-05-01",
                                                    "0000002020-05-01"};
  auto const result  = spark_rapids_jni::parse_csv_date_strings(
    strings_column_view{strings},
    "yyyy-MM-dd",
    {.enable_fallback = false, .trim_control_characters = true});
  auto const expected =
    test::fixed_width_column_wrapper<timestamp_D>({18383, -1457197, 18383, 0, 0}, {1, 1, 1, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(ParseCsvDateWithFormatTests, TruncationAtEveryTokenBoundary)
{
  std::vector<std::pair<std::string, std::string>> const examples{
    {"yyyy-MM-dd", "2020-05-01"},
    {"yyyy/MM/dd", "2020/05/01"},
    {"yyyy-MM", "2020-05"},
    {"yyyy/MM", "2020/05"},
    {"dd/MM/yyyy", "01/05/2020"},
    {"yyyy-MM-dd HH:mm:ss", "2020-05-01 23:59:59"},
    {"MM-dd", "05-01"},
    {"MM/dd", "05/01"},
    {"dd-MM", "01-05"},
    {"dd/MM", "01/05"},
    {"MM/yyyy", "05/2020"},
    {"MM-yyyy", "05-2020"},
    {"MM/dd/yyyy", "05/01/2020"},
    {"MM-dd-yyyy", "05-01-2020"},
    {"MMyyyy", "052020"}};
  for (auto const& [format, valid_input] : examples) {
    SCOPED_TRACE(format);
    std::vector<std::string> inputs;
    for (size_t end = 0; end < valid_input.size(); ++end) {
      inputs.push_back(valid_input.substr(0, end));
    }
    inputs.push_back(valid_input + "x");
    inputs.push_back(valid_input);
    std::vector<int32_t> days(inputs.size(), format.size() == 5 ? 120 : 18'383);
    std::vector<bool> validity(inputs.size(), false);
    validity.back()    = true;
    auto const strings = test::strings_column_wrapper(inputs.begin(), inputs.end());
    auto const expected =
      test::fixed_width_column_wrapper<timestamp_D>(days.begin(), days.end(), validity.begin());
    auto const result = spark_rapids_jni::parse_csv_date_strings(
      strings_column_view{strings},
      format,
      {.enable_fallback = false, .trim_control_characters = true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}

TEST_F(ParseCsvDateWithFormatTests, PackedYearSignAndReservedDigits)
{
  auto const strings = test::strings_column_wrapper{"20200501",
                                                    "+20200501",
                                                    "+020200501",
                                                    "020200501",
                                                    "-00010501",
                                                    "-00000501",
                                                    "+00000000000000020200501",
                                                    "+000000000000000020200501",
                                                    "+21474836480501",
                                                    "2020050",
                                                    "202005011",
                                                    "20201301",
                                                    "20200230"};
  auto const result  = spark_rapids_jni::parse_csv_date_strings(
    strings_column_view{strings},
    "yyyyMMdd",
    {.enable_fallback = false, .trim_control_characters = true});
  auto const expected = test::fixed_width_column_wrapper<timestamp_D>(
    {18'383, 0, 18'383, 0, -719'773, 0, 18'383, 0, 0, 0, 0, 0, 0},
    {1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(ParseCsvDateWithFormatTests, CalendarAndTimeBoundaries)
{
  auto const strings  = test::strings_column_wrapper{"2000-02-29 00:00:00",
                                                    "1900-02-29 00:00:00",
                                                    "0000-02-29 23:59:59",
                                                    "2020-04-30 23:59:59",
                                                    "2020-04-31 23:59:59",
                                                    "2020-05-01 24:00:00",
                                                    "2020-05-01 23:60:00",
                                                    "2020-05-01 23:59:60",
                                                    "2020-05-01 -1:00:00",
                                                    "2020-05-01 00:-1:00",
                                                    "2020-05-01 00:00:-1",
                                                    "2020-00-01 00:00:00",
                                                    "2020-13-01 00:00:00",
                                                    "2020-01-00 00:00:00",
                                                    "2020-01-32 00:00:00"};
  auto const expected = test::fixed_width_column_wrapper<timestamp_D>(
    {11'016, 0, -719'469, 18'382, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto const result = spark_rapids_jni::parse_csv_date_strings(
    strings_column_view{strings},
    "yyyy-MM-dd HH:mm:ss",
    {.enable_fallback = false, .trim_control_characters = true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(ParseCsvDateWithFormatTests, GmtCleaningIsCaseSensitiveAndNonRecursive)
{
  auto const strings  = test::strings_column_wrapper{"GMTGMT2020-05-01GMTGMT",
                                                    "20GMT20-0GMT5-0GMT1",
                                                    "GMT",
                                                    "GMTGMT",
                                                    "GM2020-05-01",
                                                    "2020-05-01GM",
                                                    "2020-05-01G",
                                                    "gmt2020-05-01",
                                                    "GGMTMT2020-05-01",
                                                    "20GGMTMT20-05-01",
                                                    "2020-05-01GMT x",
                                                    "2020-05-01GMT\tx",
                                                    "2020-05GMT",
                                                    "2020GMT",
                                                    "2020-02-30GMT"};
  auto const expected = test::fixed_width_column_wrapper<timestamp_D>(
    {18'383, 18'383, 0, 0, 0, 0, 0, 0, 0, 0, 18'383, 0, 18'383, 18'262, 0},
    {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0});
  for (auto const& format :
       {std::optional<std::string>{}, std::optional<std::string>{"yyyy/MM/dd"}}) {
    auto const result = spark_rapids_jni::parse_csv_date_strings(
      strings_column_view{strings},
      format,
      {.enable_fallback = true, .trim_control_characters = true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}

TEST_F(ParseCsvDateWithFormatTests, AsciiAndUnicodeWhitespace)
{
  // Character.isWhitespace's ASCII members, before Spark additionally trimmed ISO controls.
  std::vector<int> const old_whitespace{9, 10, 11, 12, 13, 28, 29, 30, 31, 32};
  std::vector<std::string> inputs;
  for (int byte = 0; byte < 128; ++byte) {
    inputs.push_back(std::string(1, static_cast<char>(byte)) + "2020-05-01" +
                     std::string(1, static_cast<char>(byte)));
  }
  // Spark's trimAll works on signed bytes, so even Unicode whitespace remains untrimmed.
  for (auto const& whitespace : {"\xc2\xa0", "\xe2\x80\x83", "\xe3\x80\x80"}) {
    inputs.push_back(std::string(whitespace) + "2020-05-01" + whitespace);
  }
  auto const strings = test::strings_column_wrapper(inputs.begin(), inputs.end());
  std::vector<int32_t> const days(inputs.size(), 18'383);
  for (bool const trim_controls : {false, true}) {
    std::vector<bool> validity(inputs.size(), false);
    for (auto const byte : old_whitespace) {
      validity[byte] = true;
    }
    if (trim_controls) {
      for (int byte = 0; byte <= 32; ++byte) {
        validity[byte] = true;
      }
      validity[127] = true;
    }
    auto const expected =
      test::fixed_width_column_wrapper<timestamp_D>(days.begin(), days.end(), validity.begin());
    auto const default_result = spark_rapids_jni::parse_csv_date_strings(
      strings_column_view{strings},
      std::nullopt,
      {.enable_fallback = false, .trim_control_characters = trim_controls});
    auto const fallback_result = spark_rapids_jni::parse_csv_date_strings(
      strings_column_view{strings},
      "yyyy/MM/dd",
      {.enable_fallback = true, .trim_control_characters = trim_controls});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(default_result->view(), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(fallback_result->view(), expected);
  }
}

TEST_F(ParseCsvDateWithFormatTests, NullMaskWordBoundariesAndSlices)
{
  for (int const size : {1, 31, 32, 33, 63, 64, 65, 257, 1'025}) {
    for (int const offset : {0, 1, 31}) {
      for (int const layout : {0, 1, 2, 3}) {
        SCOPED_TRACE(::testing::Message()
                     << "size=" << size << ", offset=" << offset << ", layout=" << layout);
        std::vector<std::string> inputs(size + offset + 1, "2020-05-01");
        std::vector<bool> input_validity(inputs.size(), true);
        std::vector<int32_t> const days(size, 18'383);
        std::vector<bool> validity(size, true);
        for (int i = 0; i < size; ++i) {
          // All valid, all null, all invalid, or mixed with transitions across mask words.
          bool const is_null         = layout == 1 || (layout == 3 && i % 3 == 1);
          bool const invalid         = layout == 2 || (layout == 3 && i % 3 == 2);
          input_validity[offset + i] = !is_null;
          validity[i]                = !is_null && !invalid;
          if (invalid) { inputs[offset + i] = "2020-02-30"; }
        }
        auto const strings =
          test::strings_column_wrapper(inputs.begin(), inputs.end(), input_validity.begin());
        auto const slice = cudf::slice(strings, {offset, offset + size});
        auto const expected =
          test::fixed_width_column_wrapper<timestamp_D>(days.begin(), days.end(), validity.begin());
        for (auto const& format : {std::optional<std::string>{},
                                   std::optional<std::string>{"yyyy-MM-dd"},
                                   std::optional<std::string>{"yyyy/MM/dd"}}) {
          auto const result = spark_rapids_jni::parse_csv_date_strings(
            strings_column_view{slice[0]},
            format,
            {.enable_fallback = true, .trim_control_characters = true});
          CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
        }
      }
    }
  }
}

TEST_F(ParseCsvDateWithFormatTests, EmptyColumnsWithAndWithoutChildren)
{
  auto const strings   = test::strings_column_wrapper{"2020-05-01"};
  auto const slices    = cudf::slice(strings, {1, 1});
  auto const childless = cudf::make_empty_column(data_type{type_id::STRING});
  for (auto const& view : {slices[0], childless->view()}) {
    for (auto const& format :
         {std::optional<std::string>{}, std::optional<std::string>{"yyyy-MM-dd"}}) {
      for (bool const fallback : {false, true}) {
        auto const result = spark_rapids_jni::parse_csv_date_strings(
          strings_column_view{view},
          format,
          {.enable_fallback = fallback, .trim_control_characters = true});
        EXPECT_EQ(result->type(), data_type{type_id::TIMESTAMP_DAYS});
        EXPECT_EQ(result->size(), 0);
        EXPECT_EQ(result->null_count(), 0);
      }
    }
  }
}

TEST_F(ParseCsvDateWithFormatTests, RejectsUnsupportedFormatsEvenForEmptyInput)
{
  auto const empty = cudf::make_empty_column(data_type{type_id::STRING});
  std::vector<std::string> const formats{"",
                                         "---",
                                         "yyy-MM-dd",
                                         "yyyyy-MM-dd",
                                         "yyyyyyyyyy-MM-dd",
                                         "yyyy-M-dd",
                                         "yyyy-MM-d",
                                         "yyyy-MMM-dd",
                                         "yyyy-MM-dd HH:mm:SS",
                                         "yyyy\xc2\xa0MM",
                                         "yyyy" + std::string(256, 'M')};
  for (auto const& format : formats) {
    SCOPED_TRACE(format);
    EXPECT_THROW((void)spark_rapids_jni::parse_csv_date_strings(
                   strings_column_view{empty->view()},
                   format,
                   {.enable_fallback = false, .trim_control_characters = true}),
                 std::invalid_argument);
  }
}

TYPED_TEST(StringToIntegerTests, Simple)
{
  auto const strings = test::strings_column_wrapper{"1", "0", "42", "null"};
  strings_column_view scv{strings};

  auto const result = spark_rapids_jni::string_to_integer(
    data_type{type_to_id<TypeParam>()}, scv, false, true, cudf::get_default_stream());

  test::fixed_width_column_wrapper<TypeParam> expected({1, 0, 42, 0}, {1, 1, 1, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TYPED_TEST(StringToIntegerTests, SlicedInput)
{
  auto const strings = test::strings_column_wrapper{"outside-before", "1", "2", "outside-after"};
  auto const sliced  = cudf::slice(strings, {1, 3})[0];
  auto const sliced_strings = strings_column_view{sliced};
  auto const result         = spark_rapids_jni::string_to_integer(
    data_type{type_to_id<TypeParam>()}, sliced_strings, false, true, cudf::get_default_stream());
  auto const expected = test::fixed_width_column_wrapper<TypeParam>({1, 2}, {1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TYPED_TEST(StringToIntegerTests, Ansi)
{
  auto const strings = test::strings_column_wrapper(
    {"",       "null",  "+1",      "-0",           "4.2",
     "asdf",   "98fe",  "  00012", ".--e-37602.n", "\r\r\t\n11.12380",
     "-.2",    ".3",    ".",       "+1.2",         "\n123\n456\n",
     "1 2",    "123",   "",        "1. 2",         "+    7.6",
     "  12  ", "7.6.2", "15  ",    "7  2  ",       " 8.2  ",
     "3..14",  "c0",    "\r\r",    "    ",         "+\n"},
    {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  strings_column_view scv{strings};

  constexpr bool is_signed_type = std::is_signed_v<TypeParam>;

  try {
    spark_rapids_jni::string_to_integer(
      data_type{type_to_id<TypeParam>()}, scv, true, true, cudf::get_default_stream());
    CUDF_EXPECTS(false, "expected exception!");
  } catch (spark_rapids_jni::cast_error& e) {
    auto const row = [&]() {
      if constexpr (is_signed_type) {
        return 4;
      } else {
        return 2;
      }
    }();
    auto const first_error_string = [&]() {
      if constexpr (is_signed_type) {
        return "4.2";
      } else {
        return "+1";
      }
    }();

    EXPECT_EQ(e.get_row_number(), row);
    EXPECT_EQ(e.get_string_with_error(), first_error_string);
  }

  auto const result = spark_rapids_jni::string_to_integer(
    data_type{type_to_id<TypeParam>()}, scv, false, true, cudf::get_default_stream());

  test::fixed_width_column_wrapper<TypeParam> expected = []() {
    if constexpr (is_signed_type) {
      return test::fixed_width_column_wrapper<TypeParam>(
        {0, 0,   1, 0, 4, 0,  0, 12, 0, 11, 0, 0, 0, 1, 0,
         0, 123, 0, 0, 0, 12, 0, 15, 0, 8,  0, 0, 0, 0, 0},
        {0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0});
    } else {
      return test::fixed_width_column_wrapper<TypeParam>(
        {0, 0,   0, 0, 4, 0,  0, 12, 0, 11, 0, 0, 0, 0, 0,
         0, 123, 0, 0, 0, 12, 0, 15, 0, 8,  0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0});
    }
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TYPED_TEST(StringToIntegerTests, Overflow)
{
  auto const strings = test::strings_column_wrapper({"127",
                                                     "128",
                                                     "-128",
                                                     "-129",
                                                     "255",
                                                     "256",
                                                     "32767",
                                                     "32768",
                                                     "-32768",
                                                     "-32769",
                                                     "65525",
                                                     "65536",
                                                     "2147483647",
                                                     "2147483648",
                                                     "-2147483648",
                                                     "-2147483649",
                                                     "4294967295",
                                                     "4294967296",
                                                     "-9223372036854775808",
                                                     "-9223372036854775809",
                                                     "9223372036854775807",
                                                     "9223372036854775808",
                                                     "18446744073709551615",
                                                     "18446744073709551616"});
  strings_column_view scv{strings};

  auto result = spark_rapids_jni::string_to_integer(
    data_type{type_to_id<TypeParam>()}, scv, false, true, cudf::get_default_stream());

  auto const expected = [&]() {
    if constexpr (std::is_same_v<TypeParam, int8_t>) {
      return test::fixed_width_column_wrapper<int8_t>(
        {127, 0, -128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    } else if constexpr (std::is_same_v<TypeParam, uint8_t>) {
      return test::fixed_width_column_wrapper<uint8_t>(
        {127, 128, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    } else if constexpr (std::is_same_v<TypeParam, int16_t>) {
      return test::fixed_width_column_wrapper<int16_t>(
        {127, 128, -128, -129, 255, 256, 32767, 0, -32768, 0, 0, 0,
         0,   0,   0,    0,    0,   0,   0,     0, 0,      0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    } else if constexpr (std::is_same_v<TypeParam, uint16_t>) {
      return test::fixed_width_column_wrapper<uint16_t>(
        {127, 128, 0, 0, 255, 256, 32767, 32768, 0, 0, 65525, 0,
         0,   0,   0, 0, 0,   0,   0,     0,     0, 0, 0,     0},
        {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    } else if constexpr (std::is_same_v<TypeParam, int32_t>) {
      auto ret = test::fixed_width_column_wrapper<int32_t>(
        {127,   128,   -128,       -129,   255,
         256,   32767, 32768,      -32768, -32769,
         65525, 65536, 2147483647, 0,      std::numeric_limits<int32_t>::min(),
         0,     0,     0,          0,      0,
         0,     0,     0,          0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
      return ret;
    } else if constexpr (std::is_same_v<TypeParam, uint32_t>) {
      return test::fixed_width_column_wrapper<uint32_t>(
        {127u,        128u, 0u,     0u,     255u,        256u,        32767u, 32768u,
         0u,          0u,   65525u, 65536u, 2147483647u, 2147483648u, 0u,     0u,
         4294967295u, 0u,   0u,     0u,     0u,          0u,          0u,     0u},
        {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
    } else if constexpr (std::is_same_v<TypeParam, int64_t>) {
      return test::fixed_width_column_wrapper<int64_t>(
        {127L,
         128L,
         -128L,
         -129L,
         255L,
         256L,
         32767L,
         32768L,
         -32768L,
         -32769L,
         65525L,
         65536L,
         2147483647L,
         2147483648L,
         -2147483648L,
         -2147483649L,
         4294967295L,
         4294967296L,
         std::numeric_limits<int64_t>::min(),
         0L,
         9223372036854775807L,
         0L,
         0L,
         0L},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0});
    } else if constexpr (std::is_same_v<TypeParam, uint64_t>) {
      return test::fixed_width_column_wrapper<uint64_t>(
        {127UL,
         128UL,
         0UL,
         0UL,
         255UL,
         256UL,
         32767UL,
         32768UL,
         0UL,
         0UL,
         65525UL,
         65536UL,
         2147483647UL,
         2147483648UL,
         0UL,
         0UL,
         4294967295UL,
         4294967296UL,
         0UL,
         0UL,
         9223372036854775807UL,
         9223372036854775808UL,
         18446744073709551615UL,
         0UL},
        {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0});
    }
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TYPED_TEST(StringToIntegerTests, Empty)
{
  auto empty = std::make_unique<column>(data_type{type_id::STRING},
                                        0,
                                        rmm::device_buffer{},
                                        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                        0);

  auto result = spark_rapids_jni::string_to_integer(data_type{type_to_id<TypeParam>()},
                                                    strings_column_view{empty->view()},
                                                    false,
                                                    true,
                                                    cudf::get_default_stream());

  EXPECT_EQ(result->size(), 0);
  EXPECT_EQ(result->type().id(), type_to_id<TypeParam>());
}

TYPED_TEST(StringToIntegerTests, NonEmptyNulls)
{
  auto const strings           = test::strings_column_wrapper{"123", "123", "123", "123"}.release();
  auto const valids            = std::vector<bool>{true, false, true, false};
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(valids.begin(), valids.end());
  strings->set_null_mask(null_mask, null_count);

  auto const scv = strings_column_view{*strings};
  EXPECT_EQ(scv.chars_size(cudf::get_default_stream()), 12);  // make sure it has non-empty null.

  auto const result = spark_rapids_jni::string_to_integer(
    data_type{type_to_id<TypeParam>()}, scv, false, true, cudf::get_default_stream());

  test::fixed_width_column_wrapper<TypeParam> expected({123, 0, 123, 0}, {1, 0, 1, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, Simple)
{
  auto const strings = test::strings_column_wrapper({"1", "0", "-1"});
  strings_column_view scv{strings};

  auto const result =
    spark_rapids_jni::string_to_decimal(1, 0, scv, false, true, cudf::get_default_stream());

  test::fixed_point_column_wrapper<int32_t> expected({1, 0, -1}, {1, 1, 1}, numeric::scale_type{0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, OverPrecise)
{
  auto const strings = test::strings_column_wrapper({"123456", "999999", "-123456", "-999999"});
  strings_column_view scv{strings};

  auto const result =
    spark_rapids_jni::string_to_decimal(5, 0, scv, false, true, cudf::get_default_stream());

  test::fixed_point_column_wrapper<int32_t> expected(
    {0, 0, 0, 0}, {0, 0, 0, 0}, numeric::scale_type{0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, Rounding)
{
  auto const strings = test::strings_column_wrapper({"1.23456", "9.99999", "-1.23456", "-9.99999"});
  strings_column_view scv{strings};

  auto const result =
    spark_rapids_jni::string_to_decimal(5, -4, scv, false, true, cudf::get_default_stream());

  test::fixed_point_column_wrapper<int32_t> expected(
    {12346, 100000, -12346, -100000}, {1, 0, 1, 0}, numeric::scale_type{-4});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, DecimalValues)
{
  auto const strings =
    test::strings_column_wrapper({"1.234", "0.12345", "-1.034", "-0.001234567890123456"});
  strings_column_view scv{strings};

  auto const result =
    spark_rapids_jni::string_to_decimal(6, -5, scv, false, true, cudf::get_default_stream());

  test::fixed_point_column_wrapper<int32_t> expected(
    {123400, 12345, -103400, -123}, {1, 1, 1, 1}, numeric::scale_type{-5});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, ExponentalNotation)
{
  auto const strings =
    test::strings_column_wrapper({"1.234e-1", "0.12345e1", "-1.034e-2", "-0.001234567890123456e2"});
  strings_column_view scv{strings};

  auto const result =
    spark_rapids_jni::string_to_decimal(6, -5, scv, false, true, cudf::get_default_stream());

  test::fixed_point_column_wrapper<int32_t> expected(
    {12340, 123450, -1034, -12346}, {1, 1, 1, 1}, numeric::scale_type{-5});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, Overflow)
{
  auto const strings =
    test::strings_column_wrapper({"123456789", "+987651342", "-1.23e8", "-1256789012"});
  strings_column_view scv{strings};

  auto const result =
    spark_rapids_jni::string_to_decimal(4, 3, scv, false, true, cudf::get_default_stream());

  // overflow, all nulls.
  test::fixed_point_column_wrapper<int32_t> expected(
    {0, 0, 0, 0}, {0, 0, 0, 0}, numeric::scale_type{3});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, PositiveScale)
{
  {
    auto const strings =
      test::strings_column_wrapper({"1234e-1", "12345e1", "-1234.5678", "-0.001234567890123456e6"});
    strings_column_view scv{strings};

    auto const result =
      spark_rapids_jni::string_to_decimal(6, 2, scv, false, true, cudf::get_default_stream());

    test::fixed_point_column_wrapper<int32_t> expected(
      {1, 1235, -12, -12}, {1, 1, 1, 1}, numeric::scale_type{2});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const strings = test::strings_column_wrapper(
      {"813847339", "043469773", "548977048", "985946604", "325679554", "null",      "957413342",
       "541903389", "150050891", "663968655", "976832602", "757172936", "968693314", "106046331",
       "965120263", "354546567", "108127101", "339513621", "980338159", "593267777"});
    strings_column_view scv{strings};

    auto const result =
      spark_rapids_jni::string_to_decimal(8, 3, scv, false, true, cudf::get_default_stream());

    test::fixed_point_column_wrapper<int32_t> expected(
      {813847, 43470,  548977, 985947, 325680, 0,      957413, 541903, 150051, 663969,
       976833, 757173, 968693, 106046, 965120, 354547, 108127, 339514, 980338, 593268},
      {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      numeric::scale_type{3});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}

TEST_F(StringToDecimalTests, Edges)
{
  {
    auto const str = test::strings_column_wrapper({"123456789012345678901234567890123456.01"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(38, -2, scv, false, true, cudf::get_default_stream());
    test::fixed_point_column_wrapper<__int128_t> expected(
      {(static_cast<__int128_t>(123456789012345678ull) * 1000000000000000ull +
        static_cast<__int128_t>(901234567890123ull)) *
         100000 +
       45601},
      {1},
      numeric::scale_type{-2});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
  {
    auto const str = test::strings_column_wrapper({"8.483315330475049E-4"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(15, -1, scv, false, true, cudf::get_default_stream());
    test::fixed_point_column_wrapper<int64_t> expected({0}, {1}, numeric::scale_type{-1});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"8.483315330475049E-2"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(15, -1, scv, false, true, cudf::get_default_stream());
    test::fixed_point_column_wrapper<int64_t> expected({1}, {1}, numeric::scale_type{-1});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"-1.0E14"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(15, -1, scv, false, true, cudf::get_default_stream());
    test::fixed_point_column_wrapper<int64_t> expected({0}, {0}, numeric::scale_type{-1});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"-1.0E14"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(16, -1, scv, false, true, cudf::get_default_stream());
    test::fixed_point_column_wrapper<int64_t> expected(
      {-1'000'000'000'000'000}, {1}, numeric::scale_type{-1});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"8.575859E8"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(15, -1, scv, false, true, cudf::get_default_stream());
    test::fixed_point_column_wrapper<int64_t> expected({8575859000}, {1}, numeric::scale_type{-1});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"10.0"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(3, -1, scv, false, true, cudf::get_default_stream());
    test::fixed_point_column_wrapper<int32_t> expected({100}, {1}, numeric::scale_type{-1});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"1.7142857343"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(9, -8, scv, false, true, cudf::get_default_stream());
    test::fixed_point_column_wrapper<int32_t> expected({171428573}, {1}, numeric::scale_type{-8});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"1.71428573437482136712623"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(9, -8, scv, false, true, cudf::get_default_stream());
    test::fixed_point_column_wrapper<int32_t> expected({171428573}, {1}, numeric::scale_type{-8});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);

    auto const fail_result =
      spark_rapids_jni::string_to_decimal(9, -9, scv, false, true, cudf::get_default_stream());
    test::fixed_point_column_wrapper<int32_t> null_col({0}, {0}, numeric::scale_type{-9});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(fail_result->view(), null_col);
  }

  {
    auto const str = test::strings_column_wrapper({"12.345678901"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(9, -8, scv, false, true, cudf::get_default_stream());
    test::fixed_point_column_wrapper<int32_t> expected({0}, {0}, numeric::scale_type{-8});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"0.12345678901"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(6, -6, scv, false, true, cudf::get_default_stream());
    test::fixed_point_column_wrapper<int32_t> expected({123457}, {1}, numeric::scale_type{-6});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"1.2345678901"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(6, -6, scv, false, true, cudf::get_default_stream());
    test::fixed_point_column_wrapper<int32_t> expected({0}, {0}, numeric::scale_type{-6});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"NaN", "inf", "-inf", "0"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(6, 0, scv, false, true, cudf::get_default_stream());
    test::fixed_point_column_wrapper<int32_t> expected(
      {0, 0, 0, 0}, {0, 0, 0, 1}, numeric::scale_type{0});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
  {
    auto const str = test::strings_column_wrapper({"1234567809"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(8, 3, scv, false, true, cudf::get_default_stream());
    test::fixed_point_column_wrapper<int32_t> expected({1234568}, {1}, numeric::scale_type{3});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
  {
    auto const str = test::strings_column_wrapper({"4347202159", "4347802159"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(4, 6, scv, false, true, cudf::get_default_stream());
    test::fixed_point_column_wrapper<int32_t> expected(
      {4347, 4348}, {1, 1}, numeric::scale_type{6});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}

TEST_F(StringToDecimalTests, Empty)
{
  auto empty = std::make_unique<column>(data_type{type_id::STRING},
                                        0,
                                        rmm::device_buffer{},
                                        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                        0);

  auto const result = spark_rapids_jni::string_to_decimal(
    8, 2, strings_column_view{empty->view()}, false, true, cudf::get_default_stream());

  EXPECT_EQ(result->size(), 0);
  EXPECT_EQ(result->type().id(), type_id::DECIMAL32);
  EXPECT_EQ(result->type().scale(), 2);
}

TEST_F(StringToDecimalTests, MismatchedJsonQuoteCountsThrows)
{
  auto const strings      = test::strings_column_wrapper{"0", "0"};
  auto const scv          = strings_column_view{strings};
  auto const stream       = cudf::get_default_stream();
  auto short_quote_counts = rmm::device_uvector<int8_t>(1, stream);
  auto long_quote_counts  = rmm::device_uvector<int8_t>(3, stream);

  EXPECT_THROW(
    static_cast<void>(spark_rapids_jni::string_to_decimal(
      1, 0, scv, false, false, cudf::device_span<int8_t const>{short_quote_counts}, stream)),
    cudf::logic_error);
  EXPECT_THROW(
    static_cast<void>(spark_rapids_jni::string_to_decimal(
      1, 0, scv, false, false, cudf::device_span<int8_t const>{long_quote_counts}, stream)),
    cudf::logic_error);
}

TEST_F(StringToDecimalTests, ExtremeExponentCarryDoesNotOverflowPosition)
{
  auto const stream  = cudf::get_default_stream();
  auto const strings = test::strings_column_wrapper{"9.9e2147483646"};
  auto const result =
    spark_rapids_jni::string_to_decimal(1, 0, strings_column_view{strings}, false, false, stream);
  auto const expected = test::fixed_point_column_wrapper<int32_t>({0}, {0}, numeric::scale_type{0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, ZeroAtExtremeExponentMatchesSpark)
{
  auto const stream  = cudf::get_default_stream();
  auto const strings = test::strings_column_wrapper{"0e2147483647", "0e2147483646"};
  auto const result =
    spark_rapids_jni::string_to_decimal(1, 0, strings_column_view{strings}, false, false, stream);
  auto const expected =
    test::fixed_point_column_wrapper<int32_t>({0, 0}, {1, 1}, numeric::scale_type{0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, DigitlessMantissaIsInvalid)
{
  auto const stream  = cudf::get_default_stream();
  auto const strings = test::strings_column_wrapper{".", "e", "E2"};
  auto const result =
    spark_rapids_jni::string_to_decimal(1, 0, strings_column_view{strings}, false, false, stream);
  auto const expected =
    test::fixed_point_column_wrapper<int32_t>({0, 0, 0}, {0, 0, 0}, numeric::scale_type{0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, SlicedInput)
{
  auto const stream = cudf::get_default_stream();
  auto const strings =
    test::strings_column_wrapper{"outside-before", "0e2147483647", "7", "outside-after"};
  auto const sliced = cudf::slice(strings, {1, 3})[0];
  auto const result =
    spark_rapids_jni::string_to_decimal(1, 0, strings_column_view{sliced}, false, false, stream);
  auto const expected =
    test::fixed_point_column_wrapper<int32_t>({0, 7}, {1, 1}, numeric::scale_type{0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, JsonZeroExponent)
{
  auto const stream = cudf::get_default_stream();
  std::vector<std::string> const string_values{"0e2147483647",
                                               "-0.00e2147483647",
                                               "0e2147483647",
                                               "+0e2147483647",
                                               "+0e2147483647",
                                               "0.e2147483647",
                                               ".0e2147483647",
                                               "0.01e2147483647"};
  auto const strings = test::strings_column_wrapper(string_values.begin(), string_values.end());
  auto const quote_counts = test::fixed_width_column_wrapper<int8_t>{0, 0, 2, 0, 2, 0, 0, 0};
  cudf::column_view const quote_counts_view = quote_counts;
  auto const quote_counts_span              = cudf::device_span<int8_t const>{
    quote_counts_view.data<int8_t>(), static_cast<std::size_t>(quote_counts_view.size())};
  auto const result = spark_rapids_jni::string_to_decimal(
    1, 0, strings_column_view{strings}, false, false, quote_counts_span, stream);
  auto const expected = test::fixed_point_column_wrapper<int32_t>(
    {0, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 0, 1, 0, 0, 0}, numeric::scale_type{0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, NonEmptyNulls)
{
  auto const strings =
    test::strings_column_wrapper{"1.23456", "1.23456", "1.23456", "1.23456"}.release();
  auto const valids            = std::vector<bool>{true, false, true, false};
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(valids.begin(), valids.end());
  strings->set_null_mask(null_mask, null_count);

  auto const scv = strings_column_view{*strings};
  EXPECT_EQ(scv.chars_size(cudf::get_default_stream()), 28);  // make sure it has non-empty null.

  auto const result =
    spark_rapids_jni::string_to_decimal(6, -5, scv, false, true, cudf::get_default_stream());

  test::fixed_point_column_wrapper<int32_t> expected(
    {123456, 0, 123456, 0}, {1, 0, 1, 0}, numeric::scale_type{-5});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TYPED_TEST(StringToFloatTests, Simple)
{
  cudf::test::strings_column_wrapper in{"-1.8946e-10",
                                        "0001",
                                        "0000.123",
                                        "123",
                                        "123.45",
                                        "45.123",
                                        "-45.123",
                                        "0.45123",
                                        "-0.45123",
                                        "999999999999999999999",
                                        "99999999999999999999",
                                        "9999999999999999999",
                                        "18446744073709551609",
                                        "18446744073709551610",
                                        "18446744073709551619999999999999",
                                        "-18446744073709551609",
                                        "-18446744073709551610",
                                        "-184467440737095516199999999999997"};

  auto const valid_iter = cudf::test::iterators::no_nulls();
  auto expected =
    cudf::strings::to_floats(strings_column_view(in), cudf::data_type{type_to_id<TypeParam>()});
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(valid_iter, valid_iter + expected->size());
  expected->set_null_mask(std::move(null_mask), null_count);

  auto const result = spark_rapids_jni::string_to_float(
    data_type{type_to_id<TypeParam>()}, strings_column_view{in}, false, cudf::get_default_stream());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected->view());
}

TYPED_TEST(StringToFloatTests, SlicedInput)
{
  auto const strings =
    test::strings_column_wrapper{"outside-before", "1.5", "-2.25", "outside-after"};
  auto const sliced         = cudf::slice(strings, {1, 3})[0];
  auto const sliced_strings = strings_column_view{sliced};
  auto const result         = spark_rapids_jni::string_to_float(
    data_type{type_to_id<TypeParam>()}, sliced_strings, false, cudf::get_default_stream());
  auto const expected = test::fixed_width_column_wrapper<TypeParam>(
    {static_cast<TypeParam>(1.5), static_cast<TypeParam>(-2.25)}, {1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TYPED_TEST(StringToFloatTests, DecimalPointAfterTruncatedDigitsAcrossBatches)
{
  auto const make_value = [](std::size_t digits) { return std::string(digits, '9') + ".0"; };
  std::vector<std::string> const input{make_value(31),
                                       make_value(32),
                                       make_value(38),
                                       "-" + make_value(38),
                                       std::string(38, '9') + "." + std::string(100, '0') + "1",
                                       make_value(300),
                                       make_value(307),
                                       make_value(38) + "e-38"};
  auto const in = cudf::test::strings_column_wrapper{input.begin(), input.end()};

  auto const expected = []() {
    if constexpr (std::is_same_v<TypeParam, float>) {
      return test::fixed_width_column_wrapper<TypeParam>{static_cast<TypeParam>(1.0e31),
                                                         static_cast<TypeParam>(1.0e32),
                                                         static_cast<TypeParam>(1.0e38),
                                                         static_cast<TypeParam>(-1.0e38),
                                                         static_cast<TypeParam>(1.0e38),
                                                         std::numeric_limits<TypeParam>::infinity(),
                                                         std::numeric_limits<TypeParam>::infinity(),
                                                         static_cast<TypeParam>(1.0)};
    } else {
      return test::fixed_width_column_wrapper<TypeParam>{static_cast<TypeParam>(1.0e31),
                                                         static_cast<TypeParam>(1.0e32),
                                                         static_cast<TypeParam>(1.0e38),
                                                         static_cast<TypeParam>(-1.0e38),
                                                         static_cast<TypeParam>(1.0e38),
                                                         static_cast<TypeParam>(1.0e300),
                                                         static_cast<TypeParam>(1.0e307),
                                                         static_cast<TypeParam>(1.0)};
    }
  }();

  auto const result = spark_rapids_jni::string_to_float(
    data_type{type_to_id<TypeParam>()}, strings_column_view{in}, false, cudf::get_default_stream());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}

TYPED_TEST(StringToFloatTests, InfNaN)
{
  cudf::test::strings_column_wrapper in{"NaN", "-Infinity", "inf", "Infinity", "-inf", "-nan"};

  auto const valid_iter = cudf::test::iterators::null_at(5);

  auto expected =
    cudf::strings::to_floats(strings_column_view(in), cudf::data_type{type_to_id<TypeParam>()});
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(valid_iter, valid_iter + expected->size());
  expected->set_null_mask(std::move(null_mask), null_count);

  auto const result = spark_rapids_jni::string_to_float(
    data_type{type_to_id<TypeParam>()}, strings_column_view{in}, false, cudf::get_default_stream());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected->view());
}

TYPED_TEST(StringToFloatTests, InvalidValues)
{
  cudf::test::strings_column_wrapper in{"A", "null", "na7.62", "e", ".", "", "f", "E15"};

  auto const valid_iter = cudf::test::iterators::all_nulls();

  auto expected =
    cudf::strings::to_floats(strings_column_view(in), cudf::data_type{type_to_id<TypeParam>()});
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(valid_iter, valid_iter + expected->size());
  expected->set_null_mask(std::move(null_mask), null_count);

  auto const result = spark_rapids_jni::string_to_float(
    data_type{type_to_id<TypeParam>()}, strings_column_view{in}, false, cudf::get_default_stream());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected->view());
}

TYPED_TEST(StringToFloatTests, ANSIInvalids)
{
  cudf::test::strings_column_wrapper in[] = {{"A"}, {"."}, {"e"}};

  for (auto& col : in) {
    try {
      auto const result = spark_rapids_jni::string_to_float(data_type{type_to_id<TypeParam>()},
                                                            strings_column_view{col},
                                                            true,
                                                            cudf::get_default_stream());
      CUDF_EXPECTS(false, "expected exception!");
    } catch (spark_rapids_jni::cast_error& e) {
      EXPECT_EQ(e.get_row_number(), 0);
    }
  }
}

TYPED_TEST(StringToFloatTests, TrickyValues)
{
  cudf::test::strings_column_wrapper in{"7f",
                                        "\riNf",
                                        "1.3e5ef",
                                        "1.3e+7f",
                                        "9\n",
                                        "46037e\t",
                                        "8d",
                                        "0\n",
                                        ".\r",
                                        "2F.",
                                        "                                    7d",
                                        "                            98392.5e-1f",
                                        ".",
                                        "e",
                                        "-1.6721969836937668E-304",
                                        "-2.21363921575273728E17",
                                        "0",
                                        "00000000000000000000",
                                        "-0000000000000000000E0",
                                        "0000000000000000000E0",
                                        "0000000000000000000000000000000017",
                                        "18446744073709551609"};

  test::fixed_width_column_wrapper<TypeParam> expected(
    {static_cast<TypeParam>(7),
     std::numeric_limits<TypeParam>::infinity(),
     static_cast<TypeParam>(0),
     static_cast<TypeParam>(13000000),
     static_cast<TypeParam>(9),
     static_cast<TypeParam>(0),
     static_cast<TypeParam>(8),
     static_cast<TypeParam>(0),
     static_cast<TypeParam>(0),
     static_cast<TypeParam>(0),
     static_cast<TypeParam>(7),
     static_cast<TypeParam>(9839.25),
     static_cast<TypeParam>(0),
     static_cast<TypeParam>(0),
     static_cast<TypeParam>(-1.6721969836937666E-304l),
     static_cast<TypeParam>(-2.21363921575273728E17),
     static_cast<TypeParam>(0),
     static_cast<TypeParam>(0),
     static_cast<TypeParam>(0),
     static_cast<TypeParam>(0),
     static_cast<TypeParam>(17),
     static_cast<TypeParam>(18446744073709551609.0)},
    {1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1});

  auto const result = spark_rapids_jni::string_to_float(
    data_type{type_to_id<TypeParam>()}, strings_column_view{in}, false, cudf::get_default_stream());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TYPED_TEST(StringToFloatTests, Empty)
{
  auto empty = std::make_unique<column>(data_type{type_id::STRING},
                                        0,
                                        rmm::device_buffer{},
                                        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                        0);

  auto const result = spark_rapids_jni::string_to_float(data_type{type_to_id<TypeParam>()},
                                                        strings_column_view{empty->view()},
                                                        false,
                                                        cudf::get_default_stream());

  EXPECT_EQ(result->size(), 0);
}

TYPED_TEST(StringToFloatTests, NonEmptyNulls)
{
  auto const strings = test::strings_column_wrapper{"1.23", "1.23", "1.23", "1.23"}.release();
  auto const valids  = std::vector<bool>{true, false, true, false};
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(valids.begin(), valids.end());
  strings->set_null_mask(null_mask, null_count);

  auto const scv = strings_column_view{*strings};
  EXPECT_EQ(scv.chars_size(cudf::get_default_stream()), 16);  // make sure it has non-empty null.

  auto const result = spark_rapids_jni::string_to_float(
    data_type{type_to_id<TypeParam>()}, scv, false, cudf::get_default_stream());

  test::fixed_width_column_wrapper<TypeParam> expected({1.23, 0.0, 1.23, 0.0}, {1, 0, 1, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}
