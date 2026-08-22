/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include "reverse_strings.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <stdint.h>

#include <cstring>
#include <initializer_list>
#include <string>

namespace {

std::string bytes_to_string(std::initializer_list<uint8_t> bytes)
{
  return std::string(bytes.begin(), bytes.end());
}

}  // namespace

struct ReverseStringsTests : public cudf::test::BaseFixture {};

TEST_F(ReverseStringsTests, EmptyAndNulls)
{
  auto const input  = cudf::test::strings_column_wrapper{};
  auto const result = spark_rapids_jni::reverse_strings(cudf::strings_column_view{input});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, input);

  auto const with_nulls =
    cudf::test::strings_column_wrapper({"abc", "", "世界"}, {true, false, true});
  auto const expected =
    cudf::test::strings_column_wrapper({"cba", "", "界世"}, {true, false, true});
  auto const reversed = spark_rapids_jni::reverse_strings(cudf::strings_column_view{with_nulls});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*reversed, expected);

  // Valid empty string (not a null) plus a nonzero-offset slice of the input.
  auto const slice_source = cudf::test::strings_column_wrapper({"prefix", "", "ignored", "世界"},
                                                               {true, true, false, true});
  auto const slice        = cudf::slice(slice_source, {1, 4}).front();
  auto const slice_expected =
    cudf::test::strings_column_wrapper({"", "ignored", "界世"}, {true, false, true});
  auto const slice_reversed = spark_rapids_jni::reverse_strings(cudf::strings_column_view{slice});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*slice_reversed, slice_expected);
}

TEST_F(ReverseStringsTests, WellFormedUtf8)
{
  // ASCII, complete 2-/3-/4-byte characters.
  auto const input = cudf::test::strings_column_wrapper(
    {"ABC",
     bytes_to_string({0x41, 0xC3, 0xA9}),                // Aé
     bytes_to_string({0x41, 0xE4, 0xB8, 0x96}),          // A世
     bytes_to_string({0x41, 0xF0, 0x90, 0x8D, 0x88})});  // A + U+10048-ish 4-byte
  auto const expected =
    cudf::test::strings_column_wrapper({"CBA",
                                        bytes_to_string({0xC3, 0xA9, 0x41}),
                                        bytes_to_string({0xE4, 0xB8, 0x96, 0x41}),
                                        bytes_to_string({0xF0, 0x90, 0x8D, 0x88, 0x41})});
  auto const result = spark_rapids_jni::reverse_strings(cudf::strings_column_view{input});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(ReverseStringsTests, TruncatedTrailingUtf8NoOverread)
{
  // SPARK-57507 cases. Neighbor rows carry sentinel bytes; an over-read would pull them in.
  auto const row0 = bytes_to_string({0x41, 0xCE});              // A + truncated 2-byte lead
  auto const row1 = bytes_to_string({0xA9, 0x42});              // must not leak into row0
  auto const row2 = bytes_to_string({0x41, 0xE4, 0xB8});        // A + truncated 3-byte lead
  auto const row3 = bytes_to_string({0x96, 0x43});              // must not leak into row2
  auto const row4 = bytes_to_string({0x41, 0xF0, 0x90});        // A + truncated 4-byte lead
  auto const row5 = bytes_to_string({0x8D, 0x88, 0x44});        // must not leak into row4
  auto const row6 = bytes_to_string({0xE4, 0xB8, 0x96, 0xCE});  // 世 + orphan 2-byte lead
  auto const row7 = bytes_to_string({0x45});

  auto const input =
    cudf::test::strings_column_wrapper({row0, row1, row2, row3, row4, row5, row6, row7});
  auto const expected = cudf::test::strings_column_wrapper(
    {bytes_to_string({0xCE, 0x41}),
     bytes_to_string({0x42, 0xA9}),  // 0xA9 is continuation -> Spark width 1, then 'B'
     bytes_to_string({0xE4, 0xB8, 0x41}),
     bytes_to_string({0x43, 0x96}),
     bytes_to_string({0xF0, 0x90, 0x41}),
     bytes_to_string({0x44, 0x88, 0x8D}),
     bytes_to_string({0xCE, 0xE4, 0xB8, 0x96}),
     bytes_to_string({0x45})});

  auto const result = spark_rapids_jni::reverse_strings(cudf::strings_column_view{input});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(ReverseStringsTests, DisallowedUtf8LeadBytesAreWidthOne)
{
  // Spark treats 0xC0-0xC1 and 0xF5-0xFF as width 1 (same as ASCII/continuation).
  auto const input = cudf::test::strings_column_wrapper(
    {bytes_to_string({0x41, 0xC0, 0x42, 0xC1, 0x43, 0xF5, 0x44, 0xFF, 0x45})});
  auto const expected = cudf::test::strings_column_wrapper(
    {bytes_to_string({0x45, 0xFF, 0x44, 0xF5, 0x43, 0xC1, 0x42, 0xC0, 0x41})});

  auto const result = spark_rapids_jni::reverse_strings(cudf::strings_column_view{input});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}
