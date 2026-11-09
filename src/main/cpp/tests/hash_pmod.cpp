/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/stream>

#include <hash/hash.hpp>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

// Use wide host arithmetic as an independent oracle, including near INT32_MAX divisors.
int32_t positive_modulo(int32_t hash, int32_t partitions)
{
  auto const wide = static_cast<int64_t>(hash);
  return static_cast<int32_t>((wide % partitions + partitions) % partitions);
}

void expect_hash_pmod(cudf::table_view input, uint32_t seed)
{
  auto const hashes      = spark_rapids_jni::murmur_hash3_32(input, seed);
  auto const host_hashes = cudf::test::to_host<int32_t>(*hashes).first;
  for (auto const partitions : {1, 100, 200, std::numeric_limits<int32_t>::max()}) {
    SCOPED_TRACE(partitions);
    std::vector<int32_t> expected_values(host_hashes.size());
    std::transform(host_hashes.begin(), host_hashes.end(), expected_values.begin(), [&](auto hash) {
      return positive_modulo(hash, partitions);
    });
    cudf::test::fixed_width_column_wrapper<int32_t> expected(expected_values.begin(),
                                                             expected_values.end());
    auto const actual = spark_rapids_jni::murmur_hash3_32_pmod(input, partitions, seed);
    EXPECT_FALSE(actual->nullable());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, actual->view());
  }
}

}  // namespace

template <typename T>
class SparkHashPmodTypedTest : public cudf::test::BaseFixture {};

using SparkHashPmodTypes =
  cudf::test::Concat<cudf::test::NumericTypes,
                     cudf::test::FixedPointTypes,
                     cudf::test::Types<cudf::timestamp_D, cudf::timestamp_us, cudf::duration_us>>;
TYPED_TEST_SUITE(SparkHashPmodTypedTest, SparkHashPmodTypes);

TYPED_TEST(SparkHashPmodTypedTest, ValuesAndNulls)
{
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> const values{0, 1, 127, 8, 2};
  cudf::test::fixed_width_column_wrapper<TypeParam, int32_t> const nullable({0, 1, 127, 8, 2},
                                                                            {0, 1, 1, 0, 1});
  expect_hash_pmod(cudf::table_view{{values}}, 42);
  expect_hash_pmod(cudf::table_view{{nullable, values}}, 0xffff'ffffU);
}

class SparkHashPmodTest : public cudf::test::BaseFixture {};

TEST_F(SparkHashPmodTest, SparkReferenceAndSlicedKeys)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const first{999, 0, 1, -1, 42, 123456789, -999};
  cudf::test::fixed_width_column_wrapper<int32_t> const second{
    999, 10, 20, 30, -40, -987654321, -999};
  auto const input =
    cudf::table_view{{cudf::slice(first, {1, 6}).front(), cudf::slice(second, {1, 6}).front()}};
  // The corresponding Spark hashes are -1721723333, 1151116018, 1549484878,
  // -1287750896 and -1980733329.
  cudf::test::fixed_width_column_wrapper<int32_t> const expected100{67, 18, 78, 4, 71};
  cudf::test::fixed_width_column_wrapper<int32_t> const expected200{67, 18, 78, 104, 71};
  auto const output100 = spark_rapids_jni::murmur_hash3_32_pmod(input, 100, 42);
  auto const output200 = spark_rapids_jni::murmur_hash3_32_pmod(input, 200, 42);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected100, output100->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected200, output200->view());
}

TEST_F(SparkHashPmodTest, ThreeStringKeys)
{
  constexpr cudf::size_type rows = 4103;
  std::vector<std::string> first(rows), second(rows), third(rows);
  std::vector<bool> valid(rows);
  for (cudf::size_type i = 0; i < rows; ++i) {
    first[i]  = "key-" + std::to_string(i);
    second[i] = i % 7 == 0 ? std::string{"a\0b", 3} : std::string(i % 65, 'a' + i % 26);
    third[i]  = i % 3 == 0 ? "" : "\xE4\xB8\xAD\xF0\x9F\x8C\x8D";
    valid[i]  = i % 11 != 0;
  }
  cudf::test::strings_column_wrapper const c0(first.begin(), first.end());
  cudf::test::strings_column_wrapper const c1(second.begin(), second.end());
  cudf::test::strings_column_wrapper const c2(third.begin(), third.end());
  cudf::test::strings_column_wrapper const nullable(second.begin(), second.end(), valid.begin());
  for (auto const& input : {cudf::table_view{{c0, c1, c2}}, cudf::table_view{{c0, nullable, c2}}}) {
    expect_hash_pmod(cudf::slice(input, {3, rows - 1}).front(), 42);
  }
}

TEST_F(SparkHashPmodTest, NestedAndFloatingPointKeys)
{
  cudf::test::lists_column_wrapper<int32_t> lists{{1, 2}, {3}, {}, {-1}, {4, 5}};
  cudf::test::fixed_width_column_wrapper<double> doubles({0.,
                                                          -0.,
                                                          -std::numeric_limits<double>::quiet_NaN(),
                                                          std::numeric_limits<double>::infinity(),
                                                          -std::numeric_limits<double>::infinity()},
                                                         {1, 1, 1, 0, 1});
  cudf::test::structs_column_wrapper const structs({lists, doubles}, {1, 1, 1, 1, 0});
  expect_hash_pmod(cudf::table_view{{structs}}, 42);
}

TEST_F(SparkHashPmodTest, SignedHashBoundaries)
{
  // Nulls preserve the seed, so these rows exercise exact signed hash boundaries.
  cudf::test::fixed_width_column_wrapper<int32_t> const null_row({123}, {0});
  auto const input = cudf::table_view{{null_row}};
  auto const max   = std::numeric_limits<int32_t>::max();
  for (auto const hash : {std::numeric_limits<int32_t>::min(), -201, -200, -1, 0, 1, max}) {
    for (auto const partitions : {1, 2, 3, 100, 200, 256, max - 1, max}) {
      SCOPED_TRACE(::testing::Message() << hash << "/" << partitions);
      auto const output =
        spark_rapids_jni::murmur_hash3_32_pmod(input, partitions, std::bit_cast<uint32_t>(hash));
      cudf::test::fixed_width_column_wrapper<int32_t> const expected{
        positive_modulo(hash, partitions)};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, output->view());
      EXPECT_FALSE(output->nullable());
    }
  }
  // The default seed must remain consistent with the unfused hash API.
  auto const output = spark_rapids_jni::murmur_hash3_32_pmod(input, 100);
  cudf::test::fixed_width_column_wrapper<int32_t> const zero{0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(zero, output->view());
}

TEST_F(SparkHashPmodTest, EmptyInput)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const empty{};
  auto const output = spark_rapids_jni::murmur_hash3_32_pmod(cudf::table_view{{empty}}, 100, 42);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty, output->view());
}

TEST_F(SparkHashPmodTest, NonDefaultStreamAndResources)
{
  cudf::test::strings_column_wrapper const keys{"", "one", "two", "three", "four"};
  auto const input = cudf::table_view{{keys, keys, keys}};
  auto const hash  = spark_rapids_jni::murmur_hash3_32(input, 42);
  auto const host  = cudf::test::to_host<int32_t>(*hash).first;
  std::vector<int32_t> expected_values(host.size());
  std::transform(host.begin(), host.end(), expected_values.begin(), [](auto value) {
    return positive_modulo(value, 200);
  });
  cudf::test::fixed_width_column_wrapper<int32_t> const expected(expected_values.begin(),
                                                                 expected_values.end());
  auto stream       = cuda::stream{cuda::device_ref{rmm::get_current_cuda_device().value()}};
  auto output_mr    = rmm::mr::statistics_resource_adaptor{mr()};
  auto temporary_mr = rmm::mr::statistics_resource_adaptor{mr()};
  cudf::get_default_stream().sync();
  std::unique_ptr<cudf::column> output;
  {
    auto current = cudf::test::scoped_current_device_resource{temporary_mr};
    output       = spark_rapids_jni::murmur_hash3_32_pmod(input, 200, 42, stream, output_mr);
    stream.sync();
  }
  EXPECT_GT(output_mr.get_bytes_counter().value, 0);
  EXPECT_GT(temporary_mr.get_bytes_counter().total, 0);
  EXPECT_EQ(temporary_mr.get_bytes_counter().value, 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    expected, output->view(), cudf::test::debug_output_level::FIRST_ERROR, stream);
  output.reset();
  stream.sync();
  EXPECT_EQ(output_mr.get_bytes_counter().value, 0);
}

TEST_F(SparkHashPmodTest, InvalidPartitionCount)
{
  cudf::test::fixed_width_column_wrapper<int32_t> const values{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<int32_t> const empty{};
  for (auto const partitions : {0, -1, std::numeric_limits<int32_t>::min()}) {
    SCOPED_TRACE(partitions);
    for (auto const& input : {cudf::table_view{{values}}, cudf::table_view{{empty}}}) {
      EXPECT_THROW(spark_rapids_jni::murmur_hash3_32_pmod(input, partitions, 42),
                   cudf::logic_error);
    }
  }
}

TEST_F(SparkHashPmodTest, NoInputColumns)
{
  EXPECT_THROW(spark_rapids_jni::murmur_hash3_32_pmod(cudf::table_view{}, 100, 42),
               cudf::logic_error);
}

TEST_F(SparkHashPmodTest, UnsupportedListOfStruct)
{
  cudf::test::fixed_width_column_wrapper<int32_t> child{1, 2};
  auto list =
    cudf::make_lists_column(1,
                            cudf::test::fixed_width_column_wrapper<int32_t>{0, 2}.release(),
                            cudf::test::structs_column_wrapper{{child}}.release(),
                            0,
                            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
  EXPECT_THROW(spark_rapids_jni::murmur_hash3_32_pmod(cudf::table_view{{*list}}, 100, 42),
               cudf::logic_error);
}
