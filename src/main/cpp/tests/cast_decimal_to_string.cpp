/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf/strings/convert/convert_floats.hpp>

#include <rmm/device_uvector.hpp>

#include <cast_string.hpp>

#include <limits>

using namespace cudf;

template <typename T>
struct DecimalToStringTests : public test::BaseFixture {};

TYPED_TEST_SUITE(DecimalToStringTests, cudf::test::FixedPointTypes);

TYPED_TEST(DecimalToStringTests, Simple)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = device_storage_type_t<decimalXX>;
  using fp_wrapper = test::fixed_point_column_wrapper<RepType>;

  auto const scale = scale_type{0};
  auto const input = fp_wrapper{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, scale};
  auto const result =
    spark_rapids_jni::decimal_to_non_ansi_string(input, cudf::get_default_stream());
  auto const expected =
    test::strings_column_wrapper{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(DecimalToStringTests, ScientificEdge)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = device_storage_type_t<decimalXX>;
  using fp_wrapper = test::fixed_point_column_wrapper<RepType>;

  {
    auto const scale = scale_type{-6};
    auto const input = fp_wrapper{{0, 100000000}, scale};
    auto const result =
      spark_rapids_jni::decimal_to_non_ansi_string(input, cudf::get_default_stream());
    auto const expected = test::strings_column_wrapper{"0.000000", "100.000000"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
  {
    auto const scale = scale_type{-7};
    auto const input = fp_wrapper{{0, 100000000}, scale};
    auto const result =
      spark_rapids_jni::decimal_to_non_ansi_string(input, cudf::get_default_stream());
    auto const expected = test::strings_column_wrapper{"0E-7", "10.0000000"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
  {
    auto const scale = scale_type{-8};
    auto const input = fp_wrapper{{0, 1000000000}, scale};
    auto const result =
      spark_rapids_jni::decimal_to_non_ansi_string(input, cudf::get_default_stream());
    auto const expected = test::strings_column_wrapper{"0E-8", "10.00000000"};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

TYPED_TEST(DecimalToStringTests, javaToString)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = device_storage_type_t<decimalXX>;
  using fp_wrapper = test::fixed_point_column_wrapper<RepType>;

  {
    auto const scale = scale_type{0};
    auto const input = fp_wrapper{{123, -123}, scale};
    auto const result =
      spark_rapids_jni::decimal_to_non_ansi_string(input, cudf::get_default_stream());
    auto const expected = test::strings_column_wrapper{"123", "-123"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
  {
    auto const scale = scale_type{1};
    auto const input = fp_wrapper{{123}, scale};
    auto const result =
      spark_rapids_jni::decimal_to_non_ansi_string(input, cudf::get_default_stream());
    auto const expected = test::strings_column_wrapper{"1.23E+3"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
  {
    auto const scale = scale_type{3};
    auto const input = fp_wrapper{{123}, scale};
    auto const result =
      spark_rapids_jni::decimal_to_non_ansi_string(input, cudf::get_default_stream());
    auto const expected = test::strings_column_wrapper{"1.23E+5"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
  {
    auto const scale = scale_type{-1};
    auto const input = fp_wrapper{{123}, scale};
    auto const result =
      spark_rapids_jni::decimal_to_non_ansi_string(input, cudf::get_default_stream());
    auto const expected = test::strings_column_wrapper{"12.3"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
  {
    auto const scale = scale_type{-5};
    auto const input = fp_wrapper{{123}, scale};
    auto const result =
      spark_rapids_jni::decimal_to_non_ansi_string(input, cudf::get_default_stream());
    auto const expected = test::strings_column_wrapper{"0.00123"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
  {
    auto const scale = scale_type{-10};
    auto const input = fp_wrapper{{123}, scale};
    auto const result =
      spark_rapids_jni::decimal_to_non_ansi_string(input, cudf::get_default_stream());
    auto const expected = test::strings_column_wrapper{"1.23E-8"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
  {
    auto const scale = scale_type{-12};
    auto const input = fp_wrapper{{-123}, scale};
    auto const result =
      spark_rapids_jni::decimal_to_non_ansi_string(input, cudf::get_default_stream());
    auto const expected = test::strings_column_wrapper{"-1.23E-10"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
  {
    auto const scale = scale_type{-8};
    auto const input = fp_wrapper{{507688528}, scale};
    auto const result =
      spark_rapids_jni::decimal_to_non_ansi_string(input, cudf::get_default_stream());
    auto const expected = test::strings_column_wrapper{"5.07688528"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}
