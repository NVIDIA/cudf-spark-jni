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

#include <rmm/device_uvector.hpp>

#include <cast_string.hpp>

#include <limits>

using namespace cudf;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};

struct LongToBinaryStringTests : public cudf::test::BaseFixture {};

TEST_F(LongToBinaryStringTests, FromLongToBinary)
{
  auto const longs = cudf::test::fixed_width_column_wrapper<int64_t>{
    0L, 1L, 10L, -1L, std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min()};

  auto results = spark_rapids_jni::long_to_binary_string(longs, cudf::get_default_stream());

  auto const expected = cudf::test::strings_column_wrapper{
    "0",
    "1",
    "1010",
    "1111111111111111111111111111111111111111111111111111111111111111",
    "111111111111111111111111111111111111111111111111111111111111111",
    "1000000000000000000000000000000000000000000000000000000000000000"};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
}
