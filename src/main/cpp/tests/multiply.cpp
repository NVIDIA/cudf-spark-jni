/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "multiply.hpp"

#include "exception_with_row_index.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <tuple>
#include <vector>

struct MultiplyTests : public cudf::test::BaseFixture {};

TEST_F(MultiplyTests, int8)
{
  // min(int8) = -128, max(int8) = 127
  cudf::test::fixed_width_column_wrapper<int8_t> input{{1, 127, 120}};
  cudf::test::fixed_width_column_wrapper<int8_t> input2{{1, 2, -2}};
  cudf::test::fixed_width_column_wrapper<int8_t> expected{{1, 0, 0}, {1, 0, 0}};

  auto result1 = spark_rapids_jni::multiply(input, input2, /*ansi*/ false, /*try*/ true);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result1, expected);

  try {
    spark_rapids_jni::multiply(input, input2, /*ansi*/ true, /*try*/ false);
    FAIL() << "Expected error_at_row exception to be thrown";
  } catch (const spark_rapids_jni::exception_with_row_index& e) {
    EXPECT_EQ(e.get_row_index(), 1);
  }
}

TEST_F(MultiplyTests, int16)
{
  // min(int16) = -32768, max(int16) = 32767
  cudf::test::fixed_width_column_wrapper<int16_t> input{{1, 2, 32767, -30000}};
  cudf::test::fixed_width_column_wrapper<int16_t> input2{{1, 2, 2, 2}};
  cudf::test::fixed_width_column_wrapper<int16_t> expected{{1, 4, 0, 0}, {1, 1, 0, 0}};

  auto result1 = spark_rapids_jni::multiply(input, input2, /*ansi*/ false, /*try*/ true);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result1, expected);

  try {
    spark_rapids_jni::multiply(input, input2, /*ansi*/ true, /*try*/ false);
    FAIL() << "Expected error_at_row exception to be thrown";
  } catch (const spark_rapids_jni::exception_with_row_index& e) {
    EXPECT_EQ(e.get_row_index(), 2);
  }
}

TEST_F(MultiplyTests, int32)
{
  // min(int32) = -2147483648, max(int32) = 2147483647
  cudf::test::fixed_width_column_wrapper<int32_t> input{{0, 1, 2, 2147483647, 0, 5, 0, 2000000000},
                                                        {0, 1, 1, 1, 0, 1, 0, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> input2{{5, 1, 2, 2, 4, 0, 0, -2},
                                                         {1, 1, 1, 1, 1, 0, 0, 1}};
  cudf::test::fixed_width_column_wrapper<int32_t> expected{{0, 1, 4, 0, 0, 0, 0, 0},
                                                           {0, 1, 1, 0, 0, 0, 0, 0}};

  // 0, 1, 1, 1, 0, 1, 0, 1
  // 1, 1, 1, 1, 1, 0, 0, 1
  // 0, 1, 1, 0, 0, 0, 0, 0
  auto result1 = spark_rapids_jni::multiply(input, input2, /*ansi*/ false, /*try*/ true);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result1, expected);

  try {
    spark_rapids_jni::multiply(input, input2, /*ansi*/ true, /*try*/ false);
    FAIL() << "Expected error_at_row exception to be thrown";
  } catch (const spark_rapids_jni::exception_with_row_index& e) {
    EXPECT_EQ(e.get_row_index(), 3);
  }
}

TEST_F(MultiplyTests, int64)
{
  int64_t const min_int64 = -9223372036854775807LL - 1;  // -2^63
  int64_t const max_int64 = 9223372036854775807LL;       // 2^63 - 1

  // tuple format: (left, right, is_result_valid, expected_result)
  std::vector<std::tuple<int64_t, int64_t, bool, int64_t>> cases = {
    {1L, 1L, true, 1L},
    {2L, 2L, true, 4L},
    {min_int64, -1L, false, 0L},
    {-1L, min_int64, false, 0L},
    {max_int64 - 10L, 2L, false, 0L},
    {max_int64 - 10L, -2L, false, 0L},
    {min_int64 + 10L, 2L, false, 0L},
    {min_int64 + 10L, -2L, false, 0L},
    {-1L, 2L, true, -2L},
    {-1L, 2L, true, -2L}};
  std::vector<int64_t> left_longs;
  std::vector<int64_t> right_longs;
  std::vector<int64_t> expected_longs;
  std::vector<bool> is_valid;
  for (const auto& [left, right, is_valid_result, expected] : cases) {
    left_longs.push_back(left);
    right_longs.push_back(right);
    expected_longs.push_back(expected);
    is_valid.push_back(is_valid_result);
  }

  cudf::test::fixed_width_column_wrapper<int64_t> input1(left_longs.begin(), left_longs.end());
  cudf::test::fixed_width_column_wrapper<int64_t> input2(right_longs.begin(), right_longs.end());
  cudf::test::fixed_width_column_wrapper<int64_t> expected(
    expected_longs.begin(), expected_longs.end(), is_valid.begin());

  auto result = spark_rapids_jni::multiply(input1, input2, /*ansi*/ false, /*try*/ true);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  try {
    spark_rapids_jni::multiply(input1, input2, /*ansi*/ true, /*try*/ false);
    FAIL() << "Expected error_at_row exception to be thrown";
  } catch (const spark_rapids_jni::exception_with_row_index& e) {
    EXPECT_EQ(e.get_row_index(), 2);
  }
}

TEST_F(MultiplyTests, float)
{
  cudf::test::fixed_width_column_wrapper<float> input{{1.0, 2.0}};
  cudf::test::fixed_width_column_wrapper<float> input2{{1.0, 2.0}};
  cudf::test::fixed_width_column_wrapper<float> expected{{1.0, 4.0}, {1, 1}};

  auto result1 = spark_rapids_jni::multiply(input, input2, /*ansi*/ false, /*try*/ true);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result1, expected);

  auto result2 = spark_rapids_jni::multiply(input, input2, /*ansi*/ true, /*try*/ false);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result2, expected);
}

TEST_F(MultiplyTests, double)
{
  cudf::test::fixed_width_column_wrapper<double> input{{1.0, 2.0}};
  cudf::test::fixed_width_column_wrapper<double> input2{{1.0, 2.0}};
  cudf::test::fixed_width_column_wrapper<double> expected{{1.0, 4.0}, {1, 1}};

  auto result1 = spark_rapids_jni::multiply(input, input2, /*ansi*/ false, /*try*/ true);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result1, expected);

  auto result2 = spark_rapids_jni::multiply(input, input2, /*ansi*/ true, /*try*/ false);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result2, expected);
}

TEST_F(MultiplyTests, checkTypeEquals)
{
  cudf::test::fixed_width_column_wrapper<int8_t> input{{1, 127}};
  cudf::test::fixed_width_column_wrapper<int16_t> input2{{1, 2}};

  EXPECT_THROW(spark_rapids_jni::multiply(input, input2, /*ansi*/ true, /*try*/ false),
               cudf::logic_error);
}

TEST_F(MultiplyTests, checkRows)
{
  cudf::test::fixed_width_column_wrapper<int8_t> input{{1, 2, 3}};
  cudf::test::fixed_width_column_wrapper<int8_t> input2{{1, 2}};

  EXPECT_THROW(spark_rapids_jni::multiply(input, input2, /*ansi*/ true, /*try*/ false),
               cudf::logic_error);
}

TEST_F(MultiplyTests, invalidType)
{
  cudf::test::fixed_width_column_wrapper<bool> input{{1, 0}};
  cudf::test::fixed_width_column_wrapper<int8_t> input2{{1, 2}};

  EXPECT_THROW(spark_rapids_jni::multiply(input, input2, /*ansi*/ true, /*try*/ false),
               cudf::logic_error);
}

TEST_F(MultiplyTests, invalidMode)
{
  cudf::test::fixed_width_column_wrapper<int8_t> input{{1, 2}};
  cudf::test::fixed_width_column_wrapper<int8_t> input2{{1, 2}};

  EXPECT_THROW(spark_rapids_jni::multiply(input, input2, /*ansi*/ true, /*try*/ true),
               cudf::logic_error);
}
