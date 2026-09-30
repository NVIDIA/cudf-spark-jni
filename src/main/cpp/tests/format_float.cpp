/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "cast_string.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <limits>

using namespace cudf;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};

struct FormatFloatTests : public cudf::test::BaseFixture {};

TEST_F(FormatFloatTests, FormatFloats32)
{
  auto const floats =
    cudf::test::fixed_width_column_wrapper<float>{100.0f,
                                                  654321.25f,
                                                  -12761.125f,
                                                  0.0f,
                                                  5.0f,
                                                  -4.0f,
                                                  std::numeric_limits<float>::quiet_NaN(),
                                                  123456789012.34f,
                                                  -0.0f};

  auto const expected = cudf::test::strings_column_wrapper{"100.00000",
                                                           "654,321.25000",
                                                           "-12,761.12500",
                                                           "0.00000",
                                                           "5.00000",
                                                           "-4.00000",
                                                           "\xEF\xBF\xBD",
                                                           "123,456,790,000.00000",
                                                           "-0.00000"};

  auto results = spark_rapids_jni::format_float(floats, 5, cudf::get_default_stream());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
}

TEST_F(FormatFloatTests, FormatFloats64)
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
                                                   3232.794248339d,
                                                   11234000000.0d,
                                                   -0.0d};

  auto const expected = cudf::test::strings_column_wrapper{"100.00000",
                                                           "654,321.25000",
                                                           "-12,761.12500",
                                                           "1.12346",
                                                           "0.00000",
                                                           "0.00000",
                                                           "5.00000",
                                                           "-4.00000",
                                                           "\xEF\xBF\xBD",
                                                           "839,542,223,232.79420",
                                                           "3,232.79425",
                                                           "11,234,000,000.00000",
                                                           "-0.00000"};

  auto results = spark_rapids_jni::format_float(floats, 5, cudf::get_default_stream());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
}
