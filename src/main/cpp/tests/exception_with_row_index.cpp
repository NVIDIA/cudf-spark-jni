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

#include "exception_with_row_index.hpp"

#include "exception_with_row_index_utilities.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

struct ErrorAtRowTests : public cudf::test::BaseFixture {};

TEST_F(ErrorAtRowTests, unary)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{{1, 2}, {1, 1}};
  cudf::test::fixed_width_column_wrapper<int64_t> input2{{1, 2}, {1, 0}};
  cudf::test::fixed_width_column_wrapper<int64_t> result{{1, 2}, {1, 0}};

  try {
    spark_rapids_jni::throw_row_error_if_any(input, result, cudf::get_default_stream());
    FAIL() << "Expected error_at_row exception to be thrown";
  } catch (const spark_rapids_jni::exception_with_row_index& e) {
    EXPECT_EQ(e.get_row_index(), 1);
  }

  // This should not throw an exception since input and result has the same nulls
  spark_rapids_jni::throw_row_error_if_any(input2, result, cudf::get_default_stream());
}

TEST_F(ErrorAtRowTests, binary)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input1{{1, 2}, {1, 1}};
  cudf::test::fixed_width_column_wrapper<int64_t> input2{{1, 2}, {1, 1}};
  cudf::test::fixed_width_column_wrapper<int64_t> result{{1, 2}, {1, 0}};

  try {
    spark_rapids_jni::throw_row_error_if_any(input1, input2, result, cudf::get_default_stream());
    FAIL() << "Expected error_at_row exception to be thrown";
  } catch (const spark_rapids_jni::exception_with_row_index& e) {
    EXPECT_EQ(e.get_row_index(), 1);
  }

  // This should not throw an exception since inputs and result has the same nulls
  spark_rapids_jni::throw_row_error_if_any(input1, input1, input1, cudf::get_default_stream());
}

TEST_F(ErrorAtRowTests, ternary)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input1{{1, 2}, {1, 1}};
  cudf::test::fixed_width_column_wrapper<int64_t> input2{{1, 2}, {1, 1}};
  cudf::test::fixed_width_column_wrapper<int64_t> input3{{1, 2}, {1, 1}};
  cudf::test::fixed_width_column_wrapper<int64_t> result{{1, 2}, {1, 0}};

  try {
    spark_rapids_jni::throw_row_error_if_any(
      input1, input2, input3, result, cudf::get_default_stream());
    FAIL() << "Expected error_at_row exception to be thrown";
  } catch (const spark_rapids_jni::exception_with_row_index& e) {
    EXPECT_EQ(e.get_row_index(), 1);
  }

  // This should not throw an exception since inputs and result has the same nulls
  spark_rapids_jni::throw_row_error_if_any(
    input1, input1, input1, input1, cudf::get_default_stream());
}
