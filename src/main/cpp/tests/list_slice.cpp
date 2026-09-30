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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/lists/lists_column_view.hpp>

#include <list_slice.hpp>

using namespace cudf;

constexpr test::debug_output_level verbosity{test::debug_output_level::FIRST_ERROR};

struct ListSliceTests : public test::BaseFixture {};

TEST_F(ListSliceTests, ListSliceTest)
{
  auto const list_col = test::lists_column_wrapper<int32_t>{{0, 1}, {2, 3, 7, 8}, {4, 5}};
  {
    size_type start  = 1;
    size_type length = 2;

    auto results = spark_rapids_jni::list_slice(list_col, start, length);

    auto const expected = test::lists_column_wrapper<int32_t>{{0, 1}, {2, 3}, {4, 5}};

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
  }
  {
    size_type start   = 1;
    auto const length = test::fixed_width_column_wrapper<int32_t>{0, 1, 2};

    auto results = spark_rapids_jni::list_slice(list_col, start, length);

    auto const expected = test::lists_column_wrapper<int32_t>{{}, {2}, {4, 5}};

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
  }
  {
    auto const start = test::fixed_width_column_wrapper<int32_t>{1, 2, 2};
    size_type length = 2;

    auto results = spark_rapids_jni::list_slice(list_col, start, length);

    auto const expected = test::lists_column_wrapper<int32_t>{{0, 1}, {3, 7}, {5}};

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
  }
  {
    auto const start  = test::fixed_width_column_wrapper<int32_t>{1, 2, 1};
    auto const length = test::fixed_width_column_wrapper<int32_t>{0, 1, 2};

    auto results = spark_rapids_jni::list_slice(list_col, start, length);

    auto const expected = test::lists_column_wrapper<int32_t>{{}, {3}, {4, 5}};

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
  }
}
