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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <rmm/device_uvector.hpp>

#include <cast_string.hpp>

#include <limits>

using namespace cudf;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};

struct FloatToStringTests : public cudf::test::BaseFixture {};

TEST_F(FloatToStringTests, FromFloats32)
{
  auto const floats =
    cudf::test::fixed_width_column_wrapper<float>{100.0f,
                                                  654321.25f,
                                                  -12761.125f,
                                                  0.f,
                                                  5.0f,
                                                  -4.0f,
                                                  std::numeric_limits<float>::quiet_NaN(),
                                                  123456789012.34f,
                                                  -0.0f};

  auto results = spark_rapids_jni::float_to_string(floats, cudf::get_default_stream());

  auto const expected = cudf::test::strings_column_wrapper{
    "100.0", "654321.25", "-12761.125", "0.0", "5.0", "-4.0", "NaN", "1.2345679E11", "-0.0"};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
}

TEST_F(FloatToStringTests, FromFloats64)
{
  auto const floats =
    cudf::test::fixed_width_column_wrapper<double>{100.0d,
                                                   654321.25d,
                                                   -12761.125d,
                                                   1.123456789123456789d,
                                                   0.000000000000000000123456789123456789d,
                                                   0.0d,
                                                   5.0d,
                                                   -4.0d,
                                                   std::numeric_limits<double>::quiet_NaN(),
                                                   839542223232.794248339d,
                                                   -0.0d};

  auto results = spark_rapids_jni::float_to_string(floats, cudf::get_default_stream());

  auto const expected = cudf::test::strings_column_wrapper{"100.0",
                                                           "654321.25",
                                                           "-12761.125",
                                                           "1.1234567891234568",
                                                           "1.234567891234568E-19",
                                                           "0.0",
                                                           "5.0",
                                                           "-4.0",
                                                           "NaN",
                                                           "8.395422232327942E11",
                                                           "-0.0"};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
}

TEST_F(FloatToStringTests, FromFloats32ToJsonString)
{
  auto const floats =
    cudf::test::fixed_width_column_wrapper<float>{100.0f,
                                                  -4.0f,
                                                  std::numeric_limits<float>::quiet_NaN(),
                                                  std::numeric_limits<float>::infinity(),
                                                  -std::numeric_limits<float>::infinity(),
                                                  -0.0f};

  auto results = spark_rapids_jni::float_to_string(floats,
                                                   /*json_string=*/true,
                                                   cudf::get_default_stream(),
                                                   cudf::get_current_device_resource_ref());

  auto const expected = cudf::test::strings_column_wrapper{
    "100.0", "-4.0", "\"NaN\"", "\"Infinity\"", "\"-Infinity\"", "-0.0"};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
}

TEST_F(FloatToStringTests, FromFloats64ToJsonString)
{
  auto const floats =
    cudf::test::fixed_width_column_wrapper<double>{100.0d,
                                                   -4.0d,
                                                   std::numeric_limits<double>::quiet_NaN(),
                                                   std::numeric_limits<double>::infinity(),
                                                   -std::numeric_limits<double>::infinity(),
                                                   -0.0d};

  auto results = spark_rapids_jni::float_to_string(floats,
                                                   /*json_string=*/true,
                                                   cudf::get_default_stream(),
                                                   cudf::get_current_device_resource_ref());

  auto const expected = cudf::test::strings_column_wrapper{
    "100.0", "-4.0", "\"NaN\"", "\"Infinity\"", "\"-Infinity\"", "-0.0"};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
}
