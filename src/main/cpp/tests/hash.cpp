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
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing.hpp>

#include <hash/hash.hpp>

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

class HashTest : public cudf::test::BaseFixture {};

TEST_F(HashTest, MultiValueNulls)
{
  // Nulls with different values should be equal
  cudf::test::strings_column_wrapper const strings_col1(
    {"",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "All work and no play makes Jack a dull boy",
     R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~)"},
    {0, 1, 1, 0, 1});
  cudf::test::strings_column_wrapper const strings_col2(
    {"different but null",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "I am Jack's complete lack of null value",
     R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~)"},
    {0, 1, 1, 0, 1});

  // Nulls with different values should be equal
  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col1(
    {0, 100, -100, limits::min(), limits::max()}, {1, 0, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col2(
    {0, -200, 200, limits::min(), limits::max()}, {1, 0, 0, 1, 1});

  // Nulls with different values should be equal
  // Different truth values should be equal
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 0, 1, 1}, {1, 1, 0, 0, 1});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 2, 1, 0, 255}, {1, 1, 0, 0, 1});

  // Nulls with different values should be equal
  using ts = cudf::timestamp_s;
  cudf::test::fixed_width_column_wrapper<ts, ts::duration> const secs_col1(
    {ts::duration::zero(),
     static_cast<ts::duration>(100),
     static_cast<ts::duration>(-100),
     ts::duration::min(),
     ts::duration::max()},
    {1, 0, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<ts, ts::duration> const secs_col2(
    {ts::duration::zero(),
     static_cast<ts::duration>(-200),
     static_cast<ts::duration>(200),
     ts::duration::min(),
     ts::duration::max()},
    {1, 0, 0, 1, 1});

  auto const input1 = cudf::table_view({strings_col1, ints_col1, bools_col1, secs_col1});
  auto const input2 = cudf::table_view({strings_col2, ints_col2, bools_col2, secs_col2});

  {
    auto const output1 = spark_rapids_jni::murmur_hash3_32(input1, 0);
    auto const output2 = spark_rapids_jni::murmur_hash3_32(input2);

    EXPECT_EQ(input1.num_rows(), output1->size());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
  }

  {
    auto const output1 =
      spark_rapids_jni::xxhash64(input1, spark_rapids_jni::DEFAULT_XXHASH64_SEED);
    auto const output2 = spark_rapids_jni::xxhash64(input2);

    EXPECT_EQ(input1.num_rows(), output1->size());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
  }
}

template <typename T>
class HashTestTyped : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(HashTestTyped, cudf::test::FixedWidthTypes);

TYPED_TEST(HashTestTyped, Equality)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> const col{0, 127, 1, 2, 8};
  auto const input = cudf::table_view({col});

  // Hash of same input should be equal
  {
    auto const output1 = spark_rapids_jni::murmur_hash3_32(input, 0);
    auto const output2 = spark_rapids_jni::murmur_hash3_32(input);

    EXPECT_EQ(input.num_rows(), output1->size());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
  }

  {
    auto const output1 = spark_rapids_jni::xxhash64(input, spark_rapids_jni::DEFAULT_XXHASH64_SEED);
    auto const output2 = spark_rapids_jni::xxhash64(input);

    EXPECT_EQ(input.num_rows(), output1->size());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
  }
}

TYPED_TEST(HashTestTyped, EqualityNulls)
{
  using T = TypeParam;

  // Nulls with different values should be equal
  cudf::test::fixed_width_column_wrapper<T, int32_t> const col1({0, 127, 1, 2, 8}, {0, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<T, int32_t> const col2({1, 127, 1, 2, 8}, {0, 1, 1, 1, 1});

  auto const input1 = cudf::table_view({col1});
  auto const input2 = cudf::table_view({col2});

  {
    auto const output1 = spark_rapids_jni::murmur_hash3_32(input1, 0);
    auto const output2 = spark_rapids_jni::murmur_hash3_32(input2);

    EXPECT_EQ(input1.num_rows(), output1->size());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
  }

  {
    auto const output1 =
      spark_rapids_jni::xxhash64(input1, spark_rapids_jni::DEFAULT_XXHASH64_SEED);
    auto const output2 = spark_rapids_jni::xxhash64(input2);

    EXPECT_EQ(input1.num_rows(), output1->size());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
  }
}

template <typename T>
class HashTestFloatTyped : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(HashTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(HashTestFloatTyped, TestExtremes)
{
  using T = TypeParam;
  T min   = std::numeric_limits<T>::min();
  T max   = std::numeric_limits<T>::max();
  T nan   = std::numeric_limits<T>::quiet_NaN();
  T inf   = std::numeric_limits<T>::infinity();

  cudf::test::fixed_width_column_wrapper<T> const col(
    {T(0.0), T(100.0), T(-100.0), min, max, nan, inf, -inf});
  cudf::test::fixed_width_column_wrapper<T> const col_neg_zero(
    {T(-0.0), T(100.0), T(-100.0), min, max, nan, inf, -inf});
  cudf::test::fixed_width_column_wrapper<T> const col_neg_nan(
    {T(0.0), T(100.0), T(-100.0), min, max, -nan, inf, -inf});

  auto const table_col          = cudf::table_view({col});
  auto const table_col_neg_zero = cudf::table_view({col_neg_zero});
  auto const table_col_neg_nan  = cudf::table_view({col_neg_nan});

  // Spark hash is sensitive to 0 and -0
  {
    auto const spark_col         = spark_rapids_jni::murmur_hash3_32(table_col, 0);
    auto const spark_col_neg_nan = spark_rapids_jni::murmur_hash3_32(table_col_neg_nan);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*spark_col, *spark_col_neg_nan);
  }

  {
    auto const spark_col =
      spark_rapids_jni::xxhash64(table_col, spark_rapids_jni::DEFAULT_XXHASH64_SEED);
    auto const spark_col_neg_nan = spark_rapids_jni::xxhash64(table_col_neg_nan);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*spark_col, *spark_col_neg_nan);
  }
}

class SparkMurmurHash3Test : public cudf::test::BaseFixture {};

TEST_F(SparkMurmurHash3Test, MultiValueWithSeeds)
{
  // The hash values were determined by running the following Scala code in Apache Spark.
  // Note that Spark >= 3.2 normalizes the float/double value of -0. to +0. and both values hash
  // to the same result. This is normalized in the calling code (Spark RAPIDS plugin) for Spark
  // >= 3.2. However, the reference values for -0. below must be obtained with Spark < 3.2 and
  // libcudf will continue to implement the Spark < 3.2 behavior until Spark >= 3.2 is required and
  // the workaround in the calling code is removed. This also affects the combined hash values.

  /*
  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.types._
  import org.apache.spark.sql.Row
  import org.apache.spark.sql.catalyst.util.DateTimeUtils

  val schema = new StructType()
      .add("structs", new StructType()
          .add("a", IntegerType)
          .add("b", StringType)
          .add("c", new StructType()
              .add("x", FloatType)
              .add("y", LongType)))
      .add("strings", StringType)
      .add("doubles", DoubleType)
      .add("timestamps", TimestampType)
      .add("decimal64", DecimalType(18, 7))
      .add("longs", LongType)
      .add("floats", FloatType)
      .add("dates", DateType)
      .add("decimal32", DecimalType(9, 3))
      .add("ints", IntegerType)
      .add("shorts", ShortType)
      .add("bytes", ByteType)
      .add("bools", BooleanType)
      .add("decimal128", DecimalType(38, 11))

  val data = Seq(
      Row(Row(0, "a", Row(0f, 0L)), "", 0.toDouble,
          DateTimeUtils.toJavaTimestamp(0), BigDecimal(0), 0.toLong, 0.toFloat,
          DateTimeUtils.toJavaDate(0), BigDecimal(0), 0, 0.toShort, 0.toByte,
          false, BigDecimal(0)),
      Row(Row(100, "bc", Row(100f, 100L)), "The quick brown fox", -(0.toDouble),
          DateTimeUtils.toJavaTimestamp(100), BigDecimal("0.00001"), 100.toLong, -(0.toFloat),
          DateTimeUtils.toJavaDate(100), BigDecimal("0.1"), 100, 100.toShort, 100.toByte,
          true, BigDecimal("0.000000001")),
      Row(Row(-100, "def", Row(-100f, -100L)), "jumps over the lazy dog.", -Double.NaN,
          DateTimeUtils.toJavaTimestamp(-100), BigDecimal("-0.00001"), -100.toLong, -Float.NaN,
          DateTimeUtils.toJavaDate(-100), BigDecimal("-0.1"), -100, -100.toShort, -100.toByte,
          true, BigDecimal("-0.00000000001")),
      Row(Row(0x12345678, "ghij", Row(Float.PositiveInfinity, 0x123456789abcdefL)),
          "All work and no play makes Jack a dull boy", Double.MinValue,
          DateTimeUtils.toJavaTimestamp(Long.MinValue/1000000), BigDecimal("-99999999999.9999999"),
          Long.MinValue, Float.MinValue, DateTimeUtils.toJavaDate(Int.MinValue/100),
          BigDecimal("-999999.999"), Int.MinValue, Short.MinValue, Byte.MinValue, true,
          BigDecimal("-9999999999999999.99999999999")),
      Row(Row(-0x76543210, "klmno", Row(Float.NegativeInfinity, -0x123456789abcdefL)),
          "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\ud720\ud721", Double.MaxValue,
          DateTimeUtils.toJavaTimestamp(Long.MaxValue/1000000), BigDecimal("99999999999.9999999"),
          Long.MaxValue, Float.MaxValue, DateTimeUtils.toJavaDate(Int.MaxValue/100),
          BigDecimal("999999.999"), Int.MaxValue, Short.MaxValue, Byte.MaxValue, false,
          BigDecimal("99999999999999999999999999.99999999999")))

  val df = spark.createDataFrame(sc.parallelize(data), schema)
  df.columns.foreach(c => println(s"$c => ${df.select(hash(col(c))).collect.mkString(",")}"))
  println(s"combined => ${df.select(hash(col("*"))).collect.mkString(",")}")
  */

  cudf::test::fixed_width_column_wrapper<int32_t> const hash_structs_expected(
    {-105406170, 90479889, -678041645, 1667387937, 301478567});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_strings_expected(
    {142593372, 1217302703, -715697185, -2061143941, -111635966});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_doubles_expected(
    {-1670924195, -853646085, -1281358385, 1897734433, -508695674});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_timestamps_expected(
    {-1670924195, 1114849490, 904948192, -1832979433, 1752430209});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_decimal64_expected(
    {-1670924195, 1114849490, 904948192, 1962370902, -1795328666});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_longs_expected(
    {-1670924195, 1114849490, 904948192, -853646085, -1604625029});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_floats_expected(
    {933211791, 723455942, -349261430, -1225560532, -338752985});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_dates_expected(
    {933211791, 751823303, -1080202046, -1906567553, -1503850410});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_decimal32_expected(
    {-1670924195, 1114849490, 904948192, -1454351396, -193774131});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_ints_expected(
    {933211791, 751823303, -1080202046, 723455942, 133916647});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_shorts_expected(
    {933211791, 751823303, -1080202046, -1871935946, 1249274084});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_bytes_expected(
    {933211791, 751823303, -1080202046, 1110053733, 1135925485});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_bools_expected(
    {933211791, -559580957, -559580957, -559580957, 933211791});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_decimal128_expected(
    {-783713497, -295670906, 1398487324, -52622807, -1359749815});
  cudf::test::fixed_width_column_wrapper<int32_t> const hash_combined_expected(
    {401603227, 588162166, 552160517, 1132537411, -326043017});

  using double_limits = std::numeric_limits<double>;
  using long_limits   = std::numeric_limits<int64_t>;
  using float_limits  = std::numeric_limits<float>;
  using int_limits    = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> a_col{0, 100, -100, 0x1234'5678, -0x7654'3210};
  cudf::test::strings_column_wrapper b_col{"a", "bc", "def", "ghij", "klmno"};
  cudf::test::fixed_width_column_wrapper<float> x_col{
    0.f, 100.f, -100.f, float_limits::infinity(), -float_limits::infinity()};
  cudf::test::fixed_width_column_wrapper<int64_t> y_col{
    0L, 100L, -100L, 0x0123'4567'89ab'cdefL, -0x0123'4567'89ab'cdefL};
  cudf::test::structs_column_wrapper c_col{{x_col, y_col}};
  cudf::test::structs_column_wrapper const structs_col{{a_col, b_col, c_col}};

  cudf::test::strings_column_wrapper const strings_col(
    {"",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\ud720\ud721"});
  cudf::test::fixed_width_column_wrapper<double> const doubles_col(
    {0., -0., -double_limits::quiet_NaN(), double_limits::lowest(), double_limits::max()});
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep> const
    timestamps_col({0L, 100L, -100L, long_limits::min() / 1000000, long_limits::max() / 1000000});
  cudf::test::fixed_point_column_wrapper<int64_t> const decimal64_col(
    {0L, 100L, -100L, -999999999999999999L, 999999999999999999L}, numeric::scale_type{-7});
  cudf::test::fixed_width_column_wrapper<int64_t> const longs_col(
    {0L, 100L, -100L, long_limits::min(), long_limits::max()});
  cudf::test::fixed_width_column_wrapper<float> const floats_col(
    {0.f, -0.f, -float_limits::quiet_NaN(), float_limits::lowest(), float_limits::max()});
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep> dates_col(
    {0, 100, -100, int_limits::min() / 100, int_limits::max() / 100});
  cudf::test::fixed_point_column_wrapper<int32_t> const decimal32_col(
    {0, 100, -100, -999999999, 999999999}, numeric::scale_type{-3});
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, int_limits::min(), int_limits::max()});
  cudf::test::fixed_width_column_wrapper<int16_t> const shorts_col({0, 100, -100, -32768, 32767});
  cudf::test::fixed_width_column_wrapper<int8_t> const bytes_col({0, 100, -100, -128, 127});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0});
  cudf::test::fixed_point_column_wrapper<__int128_t> const decimal128_col(
    {static_cast<__int128>(0),
     static_cast<__int128>(100),
     static_cast<__int128>(-1),
     (static_cast<__int128>(0xFFFF'FFFF'FCC4'D1C3u) << 64 | 0x602F'7FC3'1800'0001u),
     (static_cast<__int128>(0x0785'EE10'D5DA'46D9u) << 64 | 0x00F4'369F'FFFF'FFFFu)},
    numeric::scale_type{-11});

  auto const hash_structs = spark_rapids_jni::murmur_hash3_32(cudf::table_view({structs_col}), 42);
  auto const hash_strings = spark_rapids_jni::murmur_hash3_32(cudf::table_view({strings_col}), 42);
  auto const hash_doubles = spark_rapids_jni::murmur_hash3_32(cudf::table_view({doubles_col}), 42);
  auto const hash_timestamps =
    spark_rapids_jni::murmur_hash3_32(cudf::table_view({timestamps_col}), 42);
  auto const hash_decimal64 =
    spark_rapids_jni::murmur_hash3_32(cudf::table_view({decimal64_col}), 42);
  auto const hash_longs  = spark_rapids_jni::murmur_hash3_32(cudf::table_view({longs_col}), 42);
  auto const hash_floats = spark_rapids_jni::murmur_hash3_32(cudf::table_view({floats_col}), 42);
  auto const hash_dates  = spark_rapids_jni::murmur_hash3_32(cudf::table_view({dates_col}), 42);
  auto const hash_decimal32 =
    spark_rapids_jni::murmur_hash3_32(cudf::table_view({decimal32_col}), 42);
  auto const hash_ints   = spark_rapids_jni::murmur_hash3_32(cudf::table_view({ints_col}), 42);
  auto const hash_shorts = spark_rapids_jni::murmur_hash3_32(cudf::table_view({shorts_col}), 42);
  auto const hash_bytes  = spark_rapids_jni::murmur_hash3_32(cudf::table_view({bytes_col}), 42);
  auto const hash_bools1 = spark_rapids_jni::murmur_hash3_32(cudf::table_view({bools_col1}), 42);
  auto const hash_bools2 = spark_rapids_jni::murmur_hash3_32(cudf::table_view({bools_col2}), 42);
  auto const hash_decimal128 =
    spark_rapids_jni::murmur_hash3_32(cudf::table_view({decimal128_col}), 42);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_structs, hash_structs_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_strings, hash_strings_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_doubles, hash_doubles_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_timestamps, hash_timestamps_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_decimal64, hash_decimal64_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_longs, hash_longs_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_floats, hash_floats_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_dates, hash_dates_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_decimal32, hash_decimal32_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_ints, hash_ints_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_shorts, hash_shorts_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_bytes, hash_bytes_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_bools1, hash_bools_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_bools2, hash_bools_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_decimal128, hash_decimal128_expected, verbosity);

  auto const combined_table = cudf::table_view({structs_col,
                                                strings_col,
                                                doubles_col,
                                                timestamps_col,
                                                decimal64_col,
                                                longs_col,
                                                floats_col,
                                                dates_col,
                                                decimal32_col,
                                                ints_col,
                                                shorts_col,
                                                bytes_col,
                                                bools_col2,
                                                decimal128_col});
  auto const hash_combined  = spark_rapids_jni::murmur_hash3_32(combined_table, 42);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_combined, hash_combined_expected, verbosity);
}

TEST_F(SparkMurmurHash3Test, StringsWithSeed)
{
  // The hash values were determined by running the following Scala code in Apache Spark:
  // val strs = Seq("", "The quick brown fox",
  //              "jumps over the lazy dog.",
  //              "All work and no play makes Jack a dull boy",
  //              "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\ud720\ud721")
  // println(strs.map(org.apache.spark.unsafe.types.UTF8String.fromString)
  //   .map(org.apache.spark.sql.catalyst.expressions.Murmur3HashFunction.hash(
  //     _, org.apache.spark.sql.types.StringType, 314)))

  cudf::test::fixed_width_column_wrapper<int32_t> const hash_strings_expected_seed_314(
    {1467149710, 723257560, -1620282500, -2001858707, 1588473657});

  cudf::test::strings_column_wrapper const strings_col(
    {"",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\ud720\ud721"});

  auto const hash_strings = spark_rapids_jni::murmur_hash3_32(cudf::table_view({strings_col}), 314);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_strings, hash_strings_expected_seed_314, verbosity);
}

TEST_F(SparkMurmurHash3Test, ListValues)
{
  /*
  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.types.{ArrayType, IntegerType, StructType}
  import org.apache.spark.sql.Row

  val schema = new StructType()
    .add("lists",ArrayType(ArrayType(IntegerType)))

  val data = Seq(
    Row(null),
    Row(List(null)),
    Row(List(List())),
    Row(List(List(1))),
    Row(List(List(1, 2))),
    Row(List(List(1, 2, 3))),
    Row(List(List(1, 2), List(3))),
    Row(List(List(1), List(2, 3))),
    Row(List(List(1), List(null, 2, 3))),
    Row(List(List(1, 2), List(3), List(null))),
    Row(List(List(1, 2), null, List(3))),
  )

  val df = spark.createDataFrame(
    spark.sparkContext.parallelize(data), schema)

  val df2 = df.selectExpr("lists", "hash(lists) as hash")
  df2.printSchema()
  df2.show(false)
  */

  auto const null = -1;
  auto nested_list =
    cudf::test::lists_column_wrapper<int>({{},
                                           {1},
                                           {1, 2},
                                           {1, 2, 3},
                                           {1, 2},
                                           {3},
                                           {1},
                                           {2, 3},
                                           {1},
                                           {{null, 2, 3}, cudf::test::iterators::nulls_at({0})},
                                           {1, 2},
                                           {3},
                                           {{null}, cudf::test::iterators::nulls_at({0})},
                                           {1, 2},
                                           {},
                                           {3}},
                                          cudf::test::iterators::nulls_at({0, 14}));
  auto offsets =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 0, 1, 2, 3, 4, 6, 8, 10, 13, 16};
  auto list_validity = cudf::test::iterators::nulls_at({0});
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_validity, list_validity + 11);
  auto list_column = cudf::make_lists_column(
    11, offsets.release(), nested_list.release(), null_count, std::move(null_mask));

  auto expect = cudf::test::fixed_width_column_wrapper<int32_t>{42,
                                                                42,
                                                                42,
                                                                -559580957,
                                                                -222940379,
                                                                -912918097,
                                                                -912918097,
                                                                -912918097,
                                                                -912918097,
                                                                -912918097,
                                                                -912918097};

  auto output = spark_rapids_jni::murmur_hash3_32(cudf::table_view({*list_column}), 42);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, output->view(), verbosity);
}

TEST_F(SparkMurmurHash3Test, StructOfListValues)
{
  /*
  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.types.{ArrayType, IntegerType, StructType}
  import org.apache.spark.sql.Row

  val schema = new StructType()
    .add("structs", new StructType()
        .add("a", ArrayType(IntegerType))
        .add("b", ArrayType(IntegerType)))

  val data = Seq(
    Row(Row(List(), List())),
    Row(Row(List(0), List(0))),
    Row(Row(List(1, null), null)),
    Row(Row(List(1, null), List())),
    Row(Row(List(), List(null, 1))),
    Row(Row(null, List(1))),
    Row(Row(List(2, 3), List(4, 5))),
  )

  val df = spark.createDataFrame(
    spark.sparkContext.parallelize(data), schema)

  val df2 = df.selectExpr("lists", "hash(lists) as hash")
  df2.printSchema()
  df2.show(false)
  */

  auto const null = -1;
  auto col1 =
    cudf::test::lists_column_wrapper<int>({{},
                                           {0},
                                           {{1, null}, cudf::test::iterators::nulls_at({1})},
                                           {{1, null}, cudf::test::iterators::nulls_at({1})},
                                           {},
                                           {} /*NULL*/,
                                           {2, 3}},
                                          cudf::test::iterators::nulls_at({5}));
  auto col2 = cudf::test::lists_column_wrapper<int>(
    {{}, {0}, {} /*NULL*/, {}, {{null, 1}, cudf::test::iterators::nulls_at({0})}, {1}, {4, 5}},
    cudf::test::iterators::nulls_at({2}));
  auto struct_column = cudf::test::structs_column_wrapper{{col1, col2}};

  auto expect = cudf::test::fixed_width_column_wrapper<int32_t>{
    42, 59727262, -559580957, -559580957, -559580957, -559580957, 170038658};

  auto output = spark_rapids_jni::murmur_hash3_32(cudf::table_view({struct_column}), 42);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, output->view(), verbosity);
}

TEST_F(SparkMurmurHash3Test, ListOfStructValues)
{
  /*
  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.types.{ArrayType, IntegerType, StructType}
  import org.apache.spark.sql.Row

  val schema = new StructType()
    .add("lists", ArrayType(new StructType()
      .add("a", IntegerType)
      .add("b", IntegerType)))

  val data = Seq(
    Row(List(Row(0, 0))),
    Row(List(null)),
    Row(List(Row(null, null))),
    Row(List(Row(1, null))),
    Row(List(Row(null, 1))),
    Row(List(Row(null, 1), Row(2, 3))),
    Row(List(Row(2, 3), null)),
    Row(List(Row(2, 3), Row(4, 5))),
  )

  val df = spark.createDataFrame(
    spark.sparkContext.parallelize(data), schema)

  val df2 = df.selectExpr("lists", "hash(lists) as hash")
  df2.printSchema()
  df2.show(false)
  */

  auto const null = -1;
  auto col1       = cudf::test::fixed_width_column_wrapper<int32_t>(
    {0, null, null, 1, null, null, 2, 2, null, 2, 4},
    cudf::test::iterators::nulls_at({1, 2, 4, 5, 8}));
  auto col2 = cudf::test::fixed_width_column_wrapper<int32_t>(
    {0, null, null, null, 1, 1, 3, 3, null, 3, 5}, cudf::test::iterators::nulls_at({1, 2, 3, 8}));
  auto struct_column =
    cudf::test::structs_column_wrapper{{col1, col2}, {1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1}};
  auto offsets =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2, 3, 4, 5, 7, 9, 11};
  auto list_nullmask = std::vector<bool>(1, 8);
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
  auto list_column = cudf::make_lists_column(
    8, offsets.release(), struct_column.release(), null_count, std::move(null_mask));

  // TODO: Lists of structs are not yet supported. Once support is added,
  // remove this EXPECT_THROW and uncomment the rest of this test.
  EXPECT_THROW(spark_rapids_jni::murmur_hash3_32(cudf::table_view({*list_column}), 42),
               cudf::logic_error);

  /*
  auto expect = cudf::test::fixed_width_column_wrapper<int32_t>{
    59727262, 42, 42, -559580957, -559580957, -912918097, 1092624418, 170038658};

  auto output = spark_rapids_jni::murmur_hash3_32(cudf::table_view({*list_column}), 42);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect, output->view(), verbosity);
  */
}

class SparkXXHash64Test : public cudf::test::BaseFixture {};

TEST_F(SparkXXHash64Test, MultiValueWithSeeds)
{
  // The hash values were determined by running the following Scala code in Apache Spark.
  // Note that Spark >= 3.2 normalizes the float/double value of -0. to +0. and both values hash
  // to the same result. This is normalized in the calling code (Spark RAPIDS plugin) for Spark
  // >= 3.2. However, the reference values for -0. below must be obtained with Spark < 3.2 and
  // libcudf will continue to implement the Spark < 3.2 behavior until Spark >= 3.2 is required and
  // the workaround in the calling code is removed. This also affects the combined hash values.

  /*
  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.types._
  import org.apache.spark.sql.Row
  import org.apache.spark.sql.catalyst.util.DateTimeUtils

  val schema = new StructType()
      .add("strings", StringType)
      .add("doubles", DoubleType)
      .add("timestamps", TimestampType)
      .add("decimal64", DecimalType(18, 7))
      .add("longs", LongType)
      .add("floats", FloatType)
      .add("dates", DateType)
      .add("decimal32", DecimalType(9, 3))
      .add("ints", IntegerType)
      .add("shorts", ShortType)
      .add("bytes", ByteType)
      .add("bools", BooleanType)
      .add("decimal128", DecimalType(38, 11))

  val data = Seq(
      Row("", 0.toDouble,
          DateTimeUtils.toJavaTimestamp(0), BigDecimal(0), 0.toLong, 0.toFloat,
          DateTimeUtils.toJavaDate(0), BigDecimal(0), 0, 0.toShort, 0.toByte,
          false, BigDecimal(0)),
      Row("The quick brown fox", -(0.toDouble),
          DateTimeUtils.toJavaTimestamp(100), BigDecimal("0.00001"), 100.toLong, -(0.toFloat),
          DateTimeUtils.toJavaDate(100), BigDecimal("0.1"), 100, 100.toShort, 100.toByte,
          true, BigDecimal("0.000000001")),
      Row("jumps over the lazy dog.", -Double.NaN,
          DateTimeUtils.toJavaTimestamp(-100), BigDecimal("-0.00001"), -100.toLong, -Float.NaN,
          DateTimeUtils.toJavaDate(-100), BigDecimal("-0.1"), -100, -100.toShort, -100.toByte,
          true, BigDecimal("-0.00000000001")),
      Row("All work and no play makes Jack a dull boy", Double.MinValue,
          DateTimeUtils.toJavaTimestamp(Long.MinValue/1000000), BigDecimal("-99999999999.9999999"),
          Long.MinValue, Float.MinValue, DateTimeUtils.toJavaDate(Int.MinValue/100),
          BigDecimal("-999999.999"), Int.MinValue, Short.MinValue, Byte.MinValue, true,
          BigDecimal("-9999999999999999.99999999999")),
      Row("!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\ud720\ud721", Double.MaxValue,
          DateTimeUtils.toJavaTimestamp(Long.MaxValue/1000000), BigDecimal("99999999999.9999999"),
          Long.MaxValue, Float.MaxValue, DateTimeUtils.toJavaDate(Int.MaxValue/100),
          BigDecimal("999999.999"), Int.MaxValue, Short.MaxValue, Byte.MaxValue, false,
          BigDecimal("99999999999999999999999999.99999999999")),
      Row(null, null, null, null, null, null, null, null, null, null, null, null, null, null),
      Row("abcdefgh", 100.toDouble,
          DateTimeUtils.toJavaTimestamp(200), BigDecimal("0.0000123"), 0x123456789abcdefL,
  Float.PositiveInfinity, DateTimeUtils.toJavaDate(-200), BigDecimal("-0.2"), -200, -200.toShort,
  -90.toByte, false, BigDecimal("-9999999999999999.99999999999")), Row("abcdefghi", 200.toDouble,
          DateTimeUtils.toJavaTimestamp(300), BigDecimal("0.0000432"), -0x123456789abcdefL,
  Float.NegativeInfinity, DateTimeUtils.toJavaDate(-300), BigDecimal("-0.3"), -300, -300.toShort,
  -80.toByte, false, BigDecimal("99999999999999999999999999.99999999999")))


  val df = spark.createDataFrame(sc.parallelize(data), schema)
  df.columns.foreach(c => println(s"$c => ${df.select(xxhash64(col(c))).collect.mkString(",")}"))
  println(s"combined => ${df.select(xxhash64(col("*"))).collect.mkString(",")}")
  */

  // cudf::test::fixed_width_column_wrapper<int32_t> const hash_structs_expected(
  //{-105406170, 90479889, -678041645, 1667387937, 301478567});
  cudf::test::fixed_width_column_wrapper<int64_t> const hash_strings_expected(
    {-7444071767201028348,
     -3617261401988713833,
     8198945020833482635,
     -5346617152005100141,
     6614298085531227868,
     spark_rapids_jni::DEFAULT_XXHASH64_SEED,
     2470326616177429180,
     -7093207067522615973});
  cudf::test::fixed_width_column_wrapper<int64_t> const hash_doubles_expected(
    {-5252525462095825812,
     -5252525462095825812,
     -3127944061524951246,
     9065082843545458248,
     -4222314252576420879,
     spark_rapids_jni::DEFAULT_XXHASH64_SEED,
     -7996023612001835843,
     -8838535416664833914});
  cudf::test::fixed_width_column_wrapper<int64_t> const hash_timestamps_expected(
    {-5252525462095825812,
     8713583529807266080,
     5675770457807661948,
     7123048472642709644,
     -5141505295506489983,
     spark_rapids_jni::DEFAULT_XXHASH64_SEED,
     -1244884446866925109,
     1772389229253425430});
  cudf::test::fixed_width_column_wrapper<int64_t> const hash_decimal64_expected(
    {-5252525462095825812,
     8713583529807266080,
     5675770457807661948,
     4265531446127695490,
     2162198894918931945,
     spark_rapids_jni::DEFAULT_XXHASH64_SEED,
     -3178482946328430151,
     4788666723486520022});
  cudf::test::fixed_width_column_wrapper<int64_t> const hash_longs_expected(
    {-5252525462095825812,
     8713583529807266080,
     5675770457807661948,
     -8619748838626508300,
     -3246596055638297850,
     spark_rapids_jni::DEFAULT_XXHASH64_SEED,
     1941233597257011502,
     -1318946533059658749});
  cudf::test::fixed_width_column_wrapper<int64_t> const hash_floats_expected(
    {3614696996920510707,
     3614696996920510707,
     2692338816207849720,
     -8545425418825163117,
     -1065250890878313112,
     spark_rapids_jni::DEFAULT_XXHASH64_SEED,
     -5940311692336719973,
     -7580553461823983095});
  cudf::test::fixed_width_column_wrapper<int64_t> const hash_dates_expected(
    {3614696996920510707,
     -7987742665087449293,
     8990748234399402673,
     -8442426365007754391,
     -1447590449373190349,
     spark_rapids_jni::DEFAULT_XXHASH64_SEED,
     -953008374380745918,
     2895908635257747121});
  cudf::test::fixed_width_column_wrapper<int64_t> const hash_decimal32_expected(
    {-5252525462095825812,
     8713583529807266080,
     5675770457807661948,
     8670643431269007867,
     6810183316718625826,
     spark_rapids_jni::DEFAULT_XXHASH64_SEED,
     7277994511003214036,
     6264187449999859617});
  cudf::test::fixed_width_column_wrapper<int64_t> const hash_ints_expected(
    {3614696996920510707,
     -7987742665087449293,
     8990748234399402673,
     2073849959933241805,
     1508894993788531228,
     spark_rapids_jni::DEFAULT_XXHASH64_SEED,
     -953008374380745918,
     2895908635257747121});
  cudf::test::fixed_width_column_wrapper<int64_t> const hash_shorts_expected(
    {3614696996920510707,
     -7987742665087449293,
     8990748234399402673,
     -904511417458573795,
     8952525448871805501,
     spark_rapids_jni::DEFAULT_XXHASH64_SEED,
     -953008374380745918,
     2895908635257747121});
  cudf::test::fixed_width_column_wrapper<int64_t> const hash_bytes_expected(
    {3614696996920510707,
     -7987742665087449293,
     8990748234399402673,
     4160238337661960656,
     8632298611707923906,
     spark_rapids_jni::DEFAULT_XXHASH64_SEED,
     -4008061843281999337,
     6690883199412647955});
  cudf::test::fixed_width_column_wrapper<int64_t> const hash_bools_expected(
    {3614696996920510707,
     -6698625589789238999,
     -6698625589789238999,
     -6698625589789238999,
     3614696996920510707,
     spark_rapids_jni::DEFAULT_XXHASH64_SEED,
     3614696996920510707,
     3614696996920510707});
  cudf::test::fixed_width_column_wrapper<int64_t> const hash_decimal128_expected(
    {-8959994473701255385,
     4409375254388155230,
     -4006032525457443936,
     -5423362182451591024,
     7041733194569950081,
     spark_rapids_jni::DEFAULT_XXHASH64_SEED,
     -5423362182451591024,
     7041733194569950081});
  cudf::test::fixed_width_column_wrapper<int64_t> const hash_combined_expected(
    {541735645035655239,
     9011982951766246298,
     3834379147931449211,
     -5406325166887725795,
     7797509897614041972,
     spark_rapids_jni::DEFAULT_XXHASH64_SEED,
     -9032872913521304524,
     -604070008711895908});

  using double_limits = std::numeric_limits<double>;
  using long_limits   = std::numeric_limits<int64_t>;
  using float_limits  = std::numeric_limits<float>;
  using int_limits    = std::numeric_limits<int32_t>;

  cudf::test::strings_column_wrapper const strings_col(
    {"",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\ud720\ud721",
     "",
     "abcdefgh",
     "abcdefghi"},
    {1, 1, 1, 1, 1, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<double> const doubles_col({0.,
                                                                    -0.,
                                                                    -double_limits::quiet_NaN(),
                                                                    double_limits::lowest(),
                                                                    double_limits::max(),
                                                                    0.,
                                                                    100.,
                                                                    200.},
                                                                   {1, 1, 1, 1, 1, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep> const
    timestamps_col(
      {0L, 100L, -100L, long_limits::min() / 1000000, long_limits::max() / 1000000, 0L, 200L, 300L},
      {1, 1, 1, 1, 1, 0, 1, 1});
  cudf::test::fixed_point_column_wrapper<int64_t> const decimal64_col(
    {0L, 100L, -100L, -999999999999999999L, 999999999999999999L, 0L, 123L, 432L},
    {1, 1, 1, 1, 1, 0, 1, 1},
    numeric::scale_type{-7});
  cudf::test::fixed_width_column_wrapper<int64_t> const longs_col({0L,
                                                                   100L,
                                                                   -100L,
                                                                   long_limits::min(),
                                                                   long_limits::max(),
                                                                   0L,
                                                                   0x123456789abcdefL,
                                                                   -0x123456789abcdefL},
                                                                  {1, 1, 1, 1, 1, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<float> const floats_col({0.f,
                                                                  -0.f,
                                                                  -float_limits::quiet_NaN(),
                                                                  float_limits::lowest(),
                                                                  float_limits::max(),
                                                                  0.f,
                                                                  float_limits::infinity(),
                                                                  -float_limits::infinity()},
                                                                 {1, 1, 1, 1, 1, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep> dates_col(
    {0, 100, -100, int_limits::min() / 100, int_limits::max() / 100, 0, -200, -300},
    {1, 1, 1, 1, 1, 0, 1, 1});
  cudf::test::fixed_point_column_wrapper<int32_t> const decimal32_col(
    {0, 100, -100, -999999999, 999999999, 0, -200, -300},
    {1, 1, 1, 1, 1, 0, 1, 1},
    numeric::scale_type{-3});
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, int_limits::min(), int_limits::max(), 0, -200, -300}, {1, 1, 1, 1, 1, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<int16_t> const shorts_col(
    {0, 100, -100, -32768, 32767, 0, -200, -300}, {1, 1, 1, 1, 1, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<int8_t> const bytes_col(
    {0, 100, -100, -128, 127, 0, -90, -80}, {1, 1, 1, 1, 1, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0, 0, 0, 0},
                                                                {1, 1, 1, 1, 1, 0, 1, 1});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0, 0, 0, 0},
                                                                {1, 1, 1, 1, 1, 0, 1, 1});
  cudf::test::fixed_point_column_wrapper<__int128_t> const decimal128_col(
    {
      static_cast<__int128>(0),
      static_cast<__int128>(100),
      static_cast<__int128>(-1),
      (static_cast<__int128>(0xFFFF'FFFF'FCC4'D1C3u) << 64 | 0x602F'7FC3'1800'0001u),
      (static_cast<__int128>(0x0785'EE10'D5DA'46D9u) << 64 | 0x00F4'369F'FFFF'FFFFu),
      0,
      (static_cast<__int128>(0xFFFF'FFFF'FCC4'D1C3u) << 64 | 0x602F'7FC3'1800'0001u),
      (static_cast<__int128>(0x0785'EE10'D5DA'46D9u) << 64 | 0x00F4'369F'FFFF'FFFFu),
    },
    {1, 1, 1, 1, 1, 0, 1, 1},
    numeric::scale_type{-11});

  // auto const hash_structs = spark_rapids_jni::xxhash64(cudf::table_view({structs_col}),
  // spark_rapids_jni::DEFAULT_XXHASH64_SEED);
  auto const hash_strings    = spark_rapids_jni::xxhash64(cudf::table_view({strings_col}),
                                                          spark_rapids_jni::DEFAULT_XXHASH64_SEED);
  auto const hash_doubles    = spark_rapids_jni::xxhash64(cudf::table_view({doubles_col}),
                                                          spark_rapids_jni::DEFAULT_XXHASH64_SEED);
  auto const hash_timestamps = spark_rapids_jni::xxhash64(cudf::table_view({timestamps_col}),
                                                          spark_rapids_jni::DEFAULT_XXHASH64_SEED);
  auto const hash_decimal64  = spark_rapids_jni::xxhash64(cudf::table_view({decimal64_col}),
                                                          spark_rapids_jni::DEFAULT_XXHASH64_SEED);
  auto const hash_longs      = spark_rapids_jni::xxhash64(cudf::table_view({longs_col}),
                                                          spark_rapids_jni::DEFAULT_XXHASH64_SEED);
  auto const hash_floats     = spark_rapids_jni::xxhash64(cudf::table_view({floats_col}),
                                                          spark_rapids_jni::DEFAULT_XXHASH64_SEED);
  auto const hash_dates      = spark_rapids_jni::xxhash64(cudf::table_view({dates_col}),
                                                          spark_rapids_jni::DEFAULT_XXHASH64_SEED);
  auto const hash_decimal32  = spark_rapids_jni::xxhash64(cudf::table_view({decimal32_col}),
                                                          spark_rapids_jni::DEFAULT_XXHASH64_SEED);
  auto const hash_ints       = spark_rapids_jni::xxhash64(cudf::table_view({ints_col}),
                                                          spark_rapids_jni::DEFAULT_XXHASH64_SEED);
  auto const hash_shorts     = spark_rapids_jni::xxhash64(cudf::table_view({shorts_col}),
                                                          spark_rapids_jni::DEFAULT_XXHASH64_SEED);
  auto const hash_bytes      = spark_rapids_jni::xxhash64(cudf::table_view({bytes_col}),
                                                          spark_rapids_jni::DEFAULT_XXHASH64_SEED);
  auto const hash_bools1     = spark_rapids_jni::xxhash64(cudf::table_view({bools_col1}),
                                                          spark_rapids_jni::DEFAULT_XXHASH64_SEED);
  auto const hash_bools2     = spark_rapids_jni::xxhash64(cudf::table_view({bools_col2}),
                                                          spark_rapids_jni::DEFAULT_XXHASH64_SEED);
  auto const hash_decimal128 = spark_rapids_jni::xxhash64(cudf::table_view({decimal128_col}),
                                                          spark_rapids_jni::DEFAULT_XXHASH64_SEED);

  // CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_structs, hash_structs_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_strings, hash_strings_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_doubles, hash_doubles_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_timestamps, hash_timestamps_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_decimal64, hash_decimal64_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_longs, hash_longs_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_floats, hash_floats_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_dates, hash_dates_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_decimal32, hash_decimal32_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_ints, hash_ints_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_shorts, hash_shorts_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_bytes, hash_bytes_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_bools1, hash_bools_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_bools2, hash_bools_expected, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_decimal128, hash_decimal128_expected, verbosity);

  auto const combined_table = cudf::table_view({// structs_col,
                                                strings_col,
                                                doubles_col,
                                                timestamps_col,
                                                decimal64_col,
                                                longs_col,
                                                floats_col,
                                                dates_col,
                                                decimal32_col,
                                                ints_col,
                                                shorts_col,
                                                bytes_col,
                                                bools_col2,
                                                decimal128_col});
  auto const hash_combined =
    spark_rapids_jni::xxhash64(combined_table, spark_rapids_jni::DEFAULT_XXHASH64_SEED);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_combined, hash_combined_expected, verbosity);
}

TEST_F(SparkXXHash64Test, Strings)
{
  // The hash values were determined by running the following Scala code in Apache Spark:
  // val strs = Seq("",
  //                null,
  //                "The quick brown fox",
  //                "jumps over the lazy dog.",
  //                "All work and no play makes Jack a dull boy",
  //                "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\ud720\ud721").toDF()
  // strs.selectExpr("xxhash64(value)")

  cudf::test::fixed_width_column_wrapper<int64_t> const hash_strings_expected(
    {-7444071767201028348,
     spark_rapids_jni::DEFAULT_XXHASH64_SEED,
     -3617261401988713833,
     8198945020833482635,
     -5346617152005100141,
     6614298085531227868});

  cudf::test::strings_column_wrapper const strings_col(
    {"",
     "",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "All work and no play makes Jack a dull boy",
     "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\ud720\ud721"},
    {1, 0, 1, 1, 1, 1});

  auto const hash_strings = spark_rapids_jni::xxhash64(cudf::table_view({strings_col}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*hash_strings, hash_strings_expected, verbosity);
}
