/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.
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

#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using namespace cudf;

template <typename T>
struct StringToIntegerTests : public test::BaseFixture {};

struct StringToDecimalTests : public test::BaseFixture {};

template <typename T>
struct StringToFloatTests : public test::BaseFixture {};

TYPED_TEST_SUITE(StringToIntegerTests, cudf::test::IntegralTypesNotBool);
TYPED_TEST_SUITE(StringToFloatTests, cudf::test::FloatingPointTypes);

TYPED_TEST(StringToIntegerTests, Simple)
{
  auto const strings = test::strings_column_wrapper{"1", "0", "42", "null"};
  strings_column_view scv{strings};

  auto const result = spark_rapids_jni::string_to_integer(
    data_type{type_to_id<TypeParam>()}, scv, false, true, cudf::get_default_stream());

  test::fixed_width_column_wrapper<TypeParam> expected({1, 0, 42, 0}, {1, 1, 1, 0});

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
    EXPECT_STREQ(e.get_string_with_error(), first_error_string);
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
  auto empty = std::make_unique<column>(
    data_type{type_id::STRING}, 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);

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
  auto empty = std::make_unique<column>(
    data_type{type_id::STRING}, 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);

  auto const result = spark_rapids_jni::string_to_decimal(
    8, 2, strings_column_view{empty->view()}, false, true, cudf::get_default_stream());

  EXPECT_EQ(result->size(), 0);
  EXPECT_EQ(result->type().id(), type_id::DECIMAL32);
  EXPECT_EQ(result->type().scale(), 2);
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
  auto empty = std::make_unique<column>(
    data_type{type_id::STRING}, 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);

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
