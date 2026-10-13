/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include "get_json_object.hpp"

#include <benchmarks/common/gjo_scenario_generators.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/debug_utilities.hpp>

#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct GetJsonObjectTest : public cudf::test::BaseFixture {};

namespace {

using path_spec = std::tuple<spark_rapids_jni::path_instruction_type, std::string, int32_t>;

path_spec named_path(std::string name)
{
  return {spark_rapids_jni::path_instruction_type::NAMED, std::move(name), 0};
}

path_spec wildcard_path() { return {spark_rapids_jni::path_instruction_type::WILDCARD, "", 0}; }

}  // namespace

TEST_F(GetJsonObjectTest, RejectsInvalidNamedFieldMatchContracts)
{
  auto const input_col = cudf::test::strings_column_wrapper{R"({"a":{"b":"value"}})"};
  auto const input     = cudf::strings_column_view{input_col};

  std::vector<std::vector<path_spec>> const named_paths{{named_path("a")}};
  EXPECT_THROW(
    spark_rapids_jni::get_json_object_multiple_paths(
      input, named_paths, -1, -1, static_cast<spark_rapids_jni::named_field_match_policy>(2)),
    cudf::logic_error);

  std::vector<std::vector<path_spec>> const empty_paths{{}};
  EXPECT_THROW(
    spark_rapids_jni::get_json_object_multiple_paths(
      input, empty_paths, -1, -1, spark_rapids_jni::named_field_match_policy::LAST_NON_NULL),
    cudf::logic_error);

  std::vector<std::vector<path_spec>> const wildcard_paths{{wildcard_path()}};
  EXPECT_THROW(
    spark_rapids_jni::get_json_object_multiple_paths(
      input, wildcard_paths, -1, -1, spark_rapids_jni::named_field_match_policy::LAST_NON_NULL),
    cudf::logic_error);

  std::vector<std::vector<path_spec>> const nested_paths{{named_path("a"), named_path("b")}};
  EXPECT_THROW(
    spark_rapids_jni::get_json_object_multiple_paths(
      input, nested_paths, -1, -1, spark_rapids_jni::named_field_match_policy::LAST_NON_NULL),
    cudf::logic_error);
}

TEST_F(GetJsonObjectTest, ScenarioSeedChangesGeneratedValues)
{
  // The nested scenario's field values and absence pattern must depend on the seed,
  // otherwise the seed axis measures one input twice.
  auto& table          = spark_rapids_jni::gjo_scenarios::nested_table_135();
  auto const num_rows  = std::size_t{4};
  auto const row_bytes = std::size_t{10240};

  std::vector<cudf::test::strings_column_wrapper> inputs;
  for (std::uint64_t seed : {std::uint64_t{1}, std::uint64_t{2}}) {
    std::mt19937_64 rng{seed};
    auto rows =
      spark_rapids_jni::gjo_scenarios::nested_build_rows(rng, table, num_rows, row_bytes, seed);
    inputs.emplace_back(rows.begin(), rows.end());
  }

  std::vector<std::vector<path_spec>> const paths{{named_path("aafgihrfqbh")}};
  std::vector<std::vector<std::string>> outputs;
  for (auto const& input : inputs) {
    auto const view   = cudf::strings_column_view{input};
    auto const result = spark_rapids_jni::get_json_object_multiple_paths(view, paths, -1, -1);
    outputs.push_back(cudf::test::to_strings(*result.front()));
  }

  // Same rows count, different generated values for at least one row.
  EXPECT_EQ(outputs[0].size(), outputs[1].size());
  EXPECT_NE(outputs[0], outputs[1]);
}

TEST_F(GetJsonObjectTest, SlicedInputUsesRelativeScratchOffsets)
{
  auto const input_col =
    cudf::test::strings_column_wrapper{std::string(1'000, ' '), R"({"a":"ok"})"};
  auto const sliced   = cudf::slice(input_col, {1, 2}).front();
  auto const input    = cudf::strings_column_view{sliced};
  auto const expected = cudf::test::strings_column_wrapper{"ok"};
  auto const missing  = cudf::test::strings_column_wrapper({""}, {false});
  auto const single   = spark_rapids_jni::get_json_object(input, {named_path("a")});
  ASSERT_EQ(single->size(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *single);

  std::vector<std::vector<path_spec>> const paths{{named_path("missing")}, {named_path("a")}};
  for (auto const budget : std::array<int64_t, 2>{-1, 1}) {
    auto const result = spark_rapids_jni::get_json_object_multiple_paths(input, paths, budget, -1);
    ASSERT_EQ(result.size(), 2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(missing, *result[0]);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result[1]);

    auto const last = spark_rapids_jni::get_json_object_multiple_paths(
      input, paths, budget, -1, spark_rapids_jni::named_field_match_policy::LAST_NON_NULL);
    ASSERT_EQ(last.size(), 2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(missing, *last[0]);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *last[1]);
  }
}

TEST_F(GetJsonObjectTest, SlicedInputRetriesExpandedObject)
{
  std::string const expanding = "{\"a\":'\x01\x01\x01\x01\x01\x01\x01\x01\"\"'}";
  auto const input_col        = cudf::test::strings_column_wrapper(
    {std::string(1'000, ' '), R"({"a":"ok"})", "", "", R"({"a":""})", expanding},
    {true, true, false, true, true, true});
  auto const sliced       = cudf::strings_column_view{cudf::slice(input_col, {1, 6}).front()};
  auto const unsliced_col = cudf::test::strings_column_wrapper(
    {R"({"a":"ok"})", "", "", R"({"a":""})", expanding}, {true, false, true, true, true});
  auto const unsliced = cudf::strings_column_view{unsliced_col};
  auto const strings  = cudf::test::strings_column_wrapper(
    {"ok", "", "", "", "\x01\x01\x01\x01\x01\x01\x01\x01\"\""}, {true, false, false, true, true});
  auto const objects = cudf::test::strings_column_wrapper(
    {R"({"a":"ok"})",
     "",
     "",
     R"({"a":""})",
     R"({"a":"\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\"\""})"},
    {true, false, false, true, true});
  std::vector<std::vector<path_spec>> const paths{{named_path("a")}, {}};

  for (auto const& input : std::array{sliced, unsliced}) {
    auto const single = spark_rapids_jni::get_json_object(input, {named_path("a")});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strings, *single);
    auto const root = spark_rapids_jni::get_json_object(input, {});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(objects, *root);
    for (auto const budget : std::array<int64_t, 2>{-1, 1}) {
      auto const result =
        spark_rapids_jni::get_json_object_multiple_paths(input, paths, budget, -1);
      ASSERT_EQ(result.size(), 2);
      ASSERT_EQ(result[0]->size(), 5);
      ASSERT_EQ(result[1]->size(), 5);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(strings, *result[0]);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(objects, *result[1]);
    }
  }
}

TEST_F(GetJsonObjectTest, SlicedEmptyAndAllNullInputs)
{
  auto const input_col =
    cudf::test::strings_column_wrapper({std::string(1'000, ' '), "", ""}, {true, false, false});
  auto const slices         = cudf::slice(input_col, {1, 1, 1, 3});
  auto const empty          = cudf::strings_column_view{slices[0]};
  auto const nulls          = cudf::strings_column_view{slices[1]};
  auto const expected_empty = cudf::test::strings_column_wrapper{};
  auto const expected_nulls = cudf::test::strings_column_wrapper({"", ""}, {false, false});
  auto const empty_result   = spark_rapids_jni::get_json_object(empty, {named_path("a")});
  auto const null_result    = spark_rapids_jni::get_json_object(nulls, {named_path("a")});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_empty, *empty_result);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_nulls, *null_result);

  std::vector<std::vector<path_spec>> const paths{{named_path("a")}, {named_path("missing")}};
  auto const empty_batch = spark_rapids_jni::get_json_object_multiple_paths(
    empty, paths, 1, -1, spark_rapids_jni::named_field_match_policy::LAST_NON_NULL);
  auto const null_batch = spark_rapids_jni::get_json_object_multiple_paths(
    nulls, paths, 1, -1, spark_rapids_jni::named_field_match_policy::LAST_NON_NULL);
  ASSERT_EQ(empty_batch.size(), 2);
  ASSERT_EQ(null_batch.size(), 2);
  for (auto const& result : empty_batch) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_empty, *result);
  }
  for (auto const& result : null_batch) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_nulls, *result);
  }

  auto const zero_prefix = cudf::test::strings_column_wrapper{"", R"({"a":"ok"})"};
  auto const zero_slice  = cudf::strings_column_view{cudf::slice(zero_prefix, {1, 2}).front()};
  auto const zero_result = spark_rapids_jni::get_json_object(zero_slice, {named_path("a")});
  auto const expected    = cudf::test::strings_column_wrapper{"ok"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *zero_result);
}

struct GetJsonObjectBoundaryTest : public GetJsonObjectTest,
                                   public testing::WithParamInterface<std::tuple<int, char>> {};

TEST_P(GetJsonObjectBoundaryTest, StringAndQuotedObjectBoundaries)
{
  auto const [padding, quote] = GetParam();
  struct string_case {
    std::string input;
    std::string unescaped;
    std::string escaped;
    bool terminated;
    bool valid;
  };
  std::array<string_case, 12> const cases{{
    {"", "", "", true, true},
    {R"(\")", "\"", R"(\")", true, true},
    {R"(\\)", "\\", R"(\\)", true, true},
    {R"(\')", "'", "'", true, true},
    {"\b\t\n\f\r", "\b\t\n\f\r", R"(\b\t\n\f\r)", true, true},
    {"\x0F", "\x0F", R"(\u000F)", true, true},
    {"\"", "\"", R"(\")", true, quote == '\''},
    {R"(\q)", "", "", true, false},
    {"", "", "", false, false},
    {"\\", "", "", false, false},
    {R"(\u00)", "", "", false, false},
    {"\n", "", "", false, false},
  }};

  for (auto const length : std::array<int, 9>{0, 1, 3, 4, 5, 7, 8, 9, 15}) {
    std::string const prefix(length, 'x');
    for (auto const& value : cases) {
      SCOPED_TRACE(testing::Message() << "length=" << length << " tail=" << value.input
                                      << " terminated=" << value.terminated);
      std::string const json = std::string(padding, ' ') + R"({"a":)" + quote + prefix +
                               value.input + (value.terminated ? std::string(1, quote) + "}" : "");
      auto const input_col = cudf::test::strings_column_wrapper{json};
      auto const input     = cudf::strings_column_view{input_col};
      auto const expected_string =
        cudf::test::strings_column_wrapper({prefix + value.unescaped}, {value.valid});
      auto const expected_object = cudf::test::strings_column_wrapper(
        {R"({"a":")" + prefix + value.escaped + R"("})"}, {value.valid});
      auto const extracted = spark_rapids_jni::get_json_object(input, {named_path("a")});
      auto const object    = spark_rapids_jni::get_json_object(input, {});
      // Nullability follows null presence, so all-valid inputs yield non-nullable outputs.
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_string, *extracted);
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_object, *object);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(PaddingAndQuotes,
                         GetJsonObjectBoundaryTest,
                         testing::Combine(testing::Range(0, 9), testing::Values('"', '\'')));
