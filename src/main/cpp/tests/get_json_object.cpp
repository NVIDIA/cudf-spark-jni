/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct GetJsonObjectTest : public cudf::test::BaseFixture {};

namespace {

using instruction = std::tuple<spark_rapids_jni::path_instruction_type, std::string, int32_t>;

[[nodiscard]] std::unique_ptr<cudf::column> run(std::vector<std::string> const& rows,
                                                std::vector<instruction> const& path)
{
  auto input = cudf::test::strings_column_wrapper(rows.begin(), rows.end());
  return spark_rapids_jni::get_json_object(cudf::strings_column_view{input}, path);
}

cudf::test::strings_column_wrapper expected_column(std::vector<std::string> const& values,
                                                   std::vector<bool> const& valid)
{
  return cudf::test::strings_column_wrapper(values.begin(), values.end(), valid.begin());
}

// Apply validity after building strings, since nullable wrappers erase null rows' bytes.
[[nodiscard]] std::unique_ptr<cudf::column> make_nonempty_nulls(
  std::vector<std::string> const& rows, std::vector<bool> const& valid)
{
  auto input            = cudf::test::strings_column_wrapper(rows.begin(), rows.end()).release();
  auto const chars_size = cudf::strings_column_view{*input}.chars_size(cudf::get_default_stream());
  auto [mask, null_count] = cudf::test::detail::make_null_mask(valid.begin(), valid.end());
  input->set_null_mask(std::move(mask), null_count);
  EXPECT_EQ(cudf::strings_column_view{*input}.chars_size(cudf::get_default_stream()), chars_size);
  return input;
}

// A malformed row must null only itself.
constexpr char malformed[] = R"({"a":"b"c"})";

}  // namespace

// Row 0's output expands past its input interval and engages the retry launch; each malformed
// row then races the following victim row. Interleaved pairs spread the race across warps.
TEST_F(GetJsonObjectTest, RetryCorruption_InterleavedPairs)
{
  auto const path        = std::vector<instruction>{};
  auto const expand      = R"({"a":")" + std::string(32, '\x01') + R"("})";
  std::string expand_out = R"({"a":")";
  for (int i = 0; i < 32; ++i) {
    expand_out += "\\u0001";
  }
  expand_out += R"("})";
  auto const victim = R"({"big":")" + std::string(64, 'v') + R"("})";

  std::vector<std::string> rows{expand};
  std::vector<std::string> values{expand_out};
  std::vector<bool> valid{true};
  for (int i = 0; i < 64; ++i) {
    rows.emplace_back(malformed);
    values.emplace_back("");
    valid.emplace_back(false);
    rows.emplace_back(victim);
    values.emplace_back(victim);
    valid.emplace_back(true);
  }

  for (int rep = 0; rep < 16; ++rep) {
    auto const result   = run(rows, path);
    auto const expected = expected_column(values, valid);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}

// Same race on a named path: the expanding row's value is an object, whose structure copy
// re-escapes each literal newline to two bytes.
TEST_F(GetJsonObjectTest, RetryCorruption_NamedPathValue)
{
  auto const path =
    std::vector<instruction>{{spark_rapids_jni::path_instruction_type::NAMED, "big", 0}};
  auto const expand      = R"({"big":{"k":")" + std::string(64, '\n') + R"("}})";
  std::string expand_out = R"({"k":")";
  for (int i = 0; i < 64; ++i) {
    expand_out += "\\n";
  }
  expand_out += R"("})";
  auto const victim_value = std::string(64, 'v');

  std::vector<std::string> rows{expand};
  std::vector<std::string> values{expand_out};
  std::vector<bool> valid{true};
  for (int i = 0; i < 64; ++i) {
    rows.emplace_back(R"({"big":"b"c"})");
    values.emplace_back("");
    valid.emplace_back(false);
    rows.emplace_back(R"({"big":")" + victim_value + R"("})");
    values.emplace_back(victim_value);
    valid.emplace_back(true);
  }

  for (int rep = 0; rep < 16; ++rep) {
    auto const result   = run(rows, path);
    auto const expected = expected_column(values, valid);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}

// Without an out-of-bound retry, a malformed row still nulls only itself.
TEST_F(GetJsonObjectTest, MalformedRow_NullsOnlyItself)
{
  auto const path =
    std::vector<instruction>{{spark_rapids_jni::path_instruction_type::NAMED, "k", 0}};
  auto const result   = run({R"({"k":"v1"})", malformed, R"({"k":"v2"})"}, path);
  auto const expected = expected_column({"v1", "", "v2"}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

// The malformed row's partial output overruns its own slot: selecting the object re-escapes
// the control run to six bytes per input byte, and the error only surfaces at the trailing
// `x` after that copy completed. The overrun must raise the out-of-bound flag instead of
// polluting the neighboring rows.
TEST_F(GetJsonObjectTest, RetryCorruption_OverrunningMalformedRow)
{
  auto const path =
    std::vector<instruction>{{spark_rapids_jni::path_instruction_type::NAMED, "k", 0}};
  auto const overrun = R"({"k":{"j":")" + std::string(64, '\x01') + R"("}x})";
  for (int rep = 0; rep < 128; ++rep) {
    auto const result   = run({R"({"k":"v1"})", overrun, R"({"k":"v2"})", R"({"k":"v3"})"}, path);
    auto const expected = expected_column({"v1", "", "v2", "v3"}, {true, false, true, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}

// Trailing garbage after the selected index element must not escape its slot.
TEST_F(GetJsonObjectTest, RetryCorruption_TrailingGarbageAfterIndex)
{
  auto const path =
    std::vector<instruction>{{spark_rapids_jni::path_instruction_type::INDEX, "", 1}};
  auto const malformed = R"([0,{"a":")" + std::string(64, '\x01') + R"("}x,2])";
  for (int rep = 0; rep < 128; ++rep) {
    auto const result   = run({R"([0,1])", malformed, R"([0,2])"}, path);
    auto const expected = expected_column({"1", "", "2"}, {true, false, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}

// A malformed field after the matched one must not corrupt the valid rows.
TEST_F(GetJsonObjectTest, RetryCorruption_MalformedTrailingField)
{
  auto const path =
    std::vector<instruction>{{spark_rapids_jni::path_instruction_type::NAMED, "k", 0}};
  auto const malformed_tail = R"({"k":{"j":")" + std::string(64, '\x01') + R"("},"m":x})";
  for (int rep = 0; rep < 128; ++rep) {
    auto const result   = run({R"({"k":"v1"})", malformed_tail, R"({"k":"v2"})"}, path);
    auto const expected = expected_column({"v1", "", "v2"}, {true, false, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}

// Same overrun, with the error inside the structure copy: the string closes, the object does
// not — the copy loop hits end of input after the escaped run was fully copied, and those
// bytes must count toward the output length.
TEST_F(GetJsonObjectTest, RetryCorruption_UnclosedStructureOverrun)
{
  auto const path    = std::vector<instruction>{};
  auto const overrun = R"({"k":")" + std::string(64, '\x01') + R"(")";
  for (int rep = 0; rep < 128; ++rep) {
    auto const result   = run({R"({"k":"v1"})", overrun, R"({"k":"v2"})", R"({"k":"v3"})"}, path);
    auto const expected = expected_column({R"({"k":"v1"})", "", R"({"k":"v2"})", R"({"k":"v3"})"},
                                          {true, false, true, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}

// Same overrun through a wildcard path: the first array element's structure copy re-escapes
// its control run past the row's slot, and the parse error surfaces only when the parent
// re-enters the array loop at the malformed second element — after the child generator has
// already written past this context's extent.
TEST_F(GetJsonObjectTest, RetryCorruption_WildcardArrayOverrun)
{
  auto const path =
    std::vector<instruction>{{spark_rapids_jni::path_instruction_type::NAMED, "k", 0},
                             {spark_rapids_jni::path_instruction_type::WILDCARD, "", 0}};
  auto const expand_row = R"({"k":[{"a":")" + std::string(64, '\x01') + R"("},x]})";
  auto const victim_row = R"({"k":[{"a":"c"}]})";
  for (int rep = 0; rep < 128; ++rep) {
    auto const result   = run({R"({"k":[{"a":"b"}]})", expand_row, victim_row, victim_row}, path);
    auto const expected = expected_column({R"({"a":"b"})", "", R"({"a":"c"})", R"({"a":"c"})"},
                                          {true, false, true, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}

// Discarded wildcard child bytes must not leak into the output.
TEST_F(GetJsonObjectTest, RetryCorruption_DiscardedWildcardBytes)
{
  auto const path =
    std::vector<instruction>{{spark_rapids_jni::path_instruction_type::WILDCARD, "", 0},
                             {spark_rapids_jni::path_instruction_type::WILDCARD, "", 0},
                             {spark_rapids_jni::path_instruction_type::NAMED, "k", 0},
                             {spark_rapids_jni::path_instruction_type::WILDCARD, "", 0},
                             {spark_rapids_jni::path_instruction_type::NAMED, "z", 0},
                             {spark_rapids_jni::path_instruction_type::WILDCARD, "", 0},
                             {spark_rapids_jni::path_instruction_type::WILDCARD, "", 0},
                             {spark_rapids_jni::path_instruction_type::NAMED, "t", 0}};
  auto const expanding = R"([{"k":[{"z":[{"t":")" + std::string(200, '\x01') + R"("}]}]}])";
  auto const result =
    run({R"([{"k":[{"z":[{"t":1}]}]},{"k":[{"z":[{}]},{"z":[{}]}]}])", expanding}, path);
  // the expanding row's output is its t value wrapped in two array levels
  std::string row_b_out = "[[\"";
  for (int i = 0; i < 200; ++i) {
    row_b_out += "\\u0001";
  }
  row_b_out += "\"]]";

  auto const expected = cudf::test::strings_column_wrapper({R"([[1]])", row_b_out});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

// Same-row expansion and abandoned shells must not corrupt the victims.
TEST_F(GetJsonObjectTest, RetryCorruption_SameRowExpandAndAbandon)
{
  auto const path =
    std::vector<instruction>{{spark_rapids_jni::path_instruction_type::WILDCARD, "", 0},
                             {spark_rapids_jni::path_instruction_type::WILDCARD, "", 0},
                             {spark_rapids_jni::path_instruction_type::NAMED, "k", 0},
                             {spark_rapids_jni::path_instruction_type::WILDCARD, "", 0},
                             {spark_rapids_jni::path_instruction_type::NAMED, "z", 0},
                             {spark_rapids_jni::path_instruction_type::WILDCARD, "", 0},
                             {spark_rapids_jni::path_instruction_type::WILDCARD, "", 0},
                             {spark_rapids_jni::path_instruction_type::NAMED, "t", 0}};
  auto const expand_row =
    R"([{"k":[{"z":[{"t":")" + std::string(64, '\x01') + R"("}]}]},{"k":[{"z":[{}]},{"z":[{}]}]}])";
  std::string expand_out = "[[\"";
  for (int i = 0; i < 64; ++i) {
    expand_out += "\\u0001";
  }
  expand_out += "\"]]";
  auto const victim_row = R"([{"k":[{"z":[{"t":"vvvv"}]}]}])";
  auto const victim_out = "[[\"vvvv\"]]";
  auto const result     = run({expand_row, victim_row, victim_row}, path);
  auto const expected   = cudf::test::strings_column_wrapper({expand_out, victim_out, victim_out});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);

  // The abandon row alone never retries: its footprint fits inside its input-size slot.
  auto const no_retry_row = R"([{"k":[{"z":[{"t":1}]}]},{"k":[{"z":[{}]},{"z":[{}]}]}])";
  auto const result2      = run({no_retry_row}, path);
  auto const expected2    = cudf::test::strings_column_wrapper({R"([[1]])"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result2->view(), expected2);
}

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

TEST_F(GetJsonObjectTest, NonEmptyNulls)
{
  // The empty prefix offsets the sliced mask without moving the character base.
  auto const input_col =
    make_nonempty_nulls({"",
                         R"({"k":"first","k":"last","k":null})",
                         R"({"k":"masked-first","k":"masked-last"})",
                         R"({"k":""})",
                         R"({"k":null,"k":"set"})",
                         "",
                         "",
                         R"({"k":x})",
                         R"({"k":"sentinel"})"},
                        {true, true, false, true, true, true, false, false, true});
  ASSERT_EQ(input_col->null_count(), 3);
  auto const first_expected =
    expected_column({"", "first", "", "", "", "", "", "", "sentinel"},
                    {false, true, false, true, false, false, false, false, true});
  auto const last_expected =
    expected_column({"", "last", "", "", "set", "", "", "", "sentinel"},
                    {false, true, false, true, true, false, false, false, true});
  auto const missing_expected =
    expected_column(std::vector<std::string>(9), std::vector<bool>(9, false));
  std::vector<std::vector<path_spec>> const paths{
    {named_path("k")}, {named_path("missing")}, {named_path("k")}};

  for (auto const offset : std::array<cudf::size_type, 2>{0, 1}) {
    SCOPED_TRACE(testing::Message() << "offset=" << offset);
    auto const input =
      cudf::strings_column_view{cudf::slice(*input_col, {offset, input_col->size()}).front()};
    auto const first   = cudf::slice(first_expected, {offset, input_col->size()}).front();
    auto const last    = cudf::slice(last_expected, {offset, input_col->size()}).front();
    auto const missing = cudf::slice(missing_expected, {offset, input_col->size()}).front();
    auto const single  = spark_rapids_jni::get_json_object(input, {named_path("k")});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(first, *single);

    for (auto const budget : std::array<int64_t, 2>{-1, 1}) {
      SCOPED_TRACE(testing::Message() << "budget=" << budget);
      auto const first_batch =
        spark_rapids_jni::get_json_object_multiple_paths(input, paths, budget, -1);
      auto const last_batch = spark_rapids_jni::get_json_object_multiple_paths(
        input, paths, budget, -1, spark_rapids_jni::named_field_match_policy::LAST_NON_NULL);
      ASSERT_EQ(first_batch.size(), paths.size());
      ASSERT_EQ(last_batch.size(), paths.size());
      for (std::size_t i = 0; i < paths.size(); ++i) {
        CUDF_TEST_EXPECT_COLUMNS_EQUAL(i == 1 ? missing : first, *first_batch[i]);
        CUDF_TEST_EXPECT_COLUMNS_EQUAL(i == 1 ? missing : last, *last_batch[i]);
      }
    }
  }
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

using spark_rapids_jni::path_instruction_type;

namespace {

using path_list     = std::vector<path_spec>;
using multi_paths   = std::vector<path_list>;
using multi_outputs = std::vector<std::unique_ptr<cudf::column>>;
using spark_rapids_jni::detail::gjo_scan_mode;

path_list named(std::initializer_list<std::string> keys)
{
  path_list path;
  for (auto const& key : keys) {
    path.push_back(named_path(key));
  }
  return path;
}

auto col_from(std::vector<std::string> const& rows)
{
  return cudf::test::strings_column_wrapper(rows.begin(), rows.end());
}

auto same_rows(int n, std::string const& row) { return std::vector<std::string>(n, row); }

multi_outputs run_fused(cudf::strings_column_view const& input, multi_paths const& paths)
{
  return spark_rapids_jni::get_json_object_multiple_paths(input, paths, -1, -1);
}

void expect_rows_eq(cudf::column const& col, std::string const& expected, bool all_valid)
{
  if (all_valid) { ASSERT_EQ(col.null_count(), 0); }
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col_from(same_rows(64, expected)), col);
}

// The hook observes first-pass routes in caller order, without changing admission.
multi_outputs routed(cudf::strings_column_view const& input,
                     multi_paths const& paths,
                     std::vector<char> const& expected,
                     gjo_scan_mode scan = gjo_scan_mode::sampled,
                     int64_t budget     = -1,
                     bool last          = false)
{
  std::vector<char> routes;
  auto mode         = static_cast<gjo_scan_mode>(-1);
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto out          = spark_rapids_jni::detail::get_json_object(
    input, paths, budget, -1, last, stream, mr, &routes, &mode);
  EXPECT_EQ(routes, expected);
  EXPECT_EQ(mode, scan);
  EXPECT_EQ(out.size(), paths.size());
  return out;
}

void expect_uniform(multi_outputs const& out, int rows, std::string const& value)
{
  auto const expected = col_from(same_rows(rows, value));
  for (auto const& col : out) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *col);
  }
}

// Each raw double quote in a single-quoted value adds one byte when serialized.
std::string quote_bomb(int n, bool escaped = false)
{
  std::string out = escaped ? R"({"a":")" : R"({"a":')";
  for (int i = 0; i < n; ++i) {
    out += escaped ? R"(\")" : "\"";
  }
  return out + (escaped ? "\"}" : "'}");
}

std::string flat_row(int members, int value_bytes, int first_key, int extra = 0)
{
  std::string row = "{";
  for (int i = 0; i < members; ++i) {
    if (i != 0) { row += ','; }
    row += "\"k" + std::to_string(first_key + i) + "\":\"" +
           std::string(value_bytes + (i == 0 ? extra : 0), 'A') + '"';
  }
  return row + '}';
}

void expect_budget_routes(int rows,
                          int members,
                          int value_bytes,
                          int extra,
                          std::initializer_list<std::pair<int64_t, int>> budgets)
{
  auto const col = col_from(same_rows(rows, flat_row(members, value_bytes, 100, extra)));
  multi_paths paths;
  for (int i = 223; i >= 100; --i) {
    paths.push_back(named({"k" + std::to_string(i)}));
  }
  auto const ordinary = col_from(same_rows(rows, std::string(value_bytes, 'A')));
  auto const first    = col_from(same_rows(rows, std::string(value_bytes + extra, 'A')));
  for (auto const [budget, spill] : budgets) {
    SCOPED_TRACE(budget);
    std::vector<char> routes(124, 1);
    std::fill_n(routes.begin(), spill, 0);
    auto const out =
      routed(cudf::strings_column_view{col}, paths, routes, gjo_scan_mode::sampled, budget);
    for (std::size_t i = 0; i < out.size(); ++i) {
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(i == 123 ? first : ordinary, *out[i]);
    }
  }
}

}  // namespace

TEST_F(GetJsonObjectTest, GeneralGroupMoreThan32Paths)
{
  // Forty array INDEX paths must all be evaluated, including paths beyond one warp.
  std::string row_body = "[";
  for (int p = 0; p < 40; ++p) {
    if (p > 0) { row_body += ","; }
    row_body += "\"v" + std::to_string(p) + "\"";
  }
  row_body += "]";
  auto const rows = same_rows(64, row_body);

  multi_paths paths;
  paths.reserve(40);
  for (int p = 0; p < 40; ++p) {
    paths.emplace_back();
    paths.back().emplace_back(path_instruction_type::INDEX, "", p);
  }

  auto const out = run_fused(cudf::strings_column_view{col_from(rows)}, paths);
  ASSERT_EQ(out.size(), 40u);

  for (std::size_t col = 0; col < 40; ++col) {
    SCOPED_TRACE("index path column " + std::to_string(col));
    auto const expected = "v" + std::to_string(col);
    ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*out[col], expected, true));
  }
}

TEST_F(GetJsonObjectTest, MergedRouteValueKindParity)
{
  // Spark's unquoted-key parsing turns 1231 into "23".
  auto const row =
    std::string{
      R"({"r":{"sameA":"first","sameA":"later","sameB":"q\"","SameA":"case","\u0065sc":"\uD83D\uDE00",)"
      R"("\uD83D\uDE00":"surrogate","n":null,"n":{"v":"late"},"v":{"b":[1,true,null]},)"
      R"("z": {"b" : -0,"s":"http:\/\/x",1231:2},"":"","f":0.0003,"zero":-0,"i":-42,"flag":true},"pad":")"} +
    std::string(1024, 'p') + "\"}";
  auto const col   = col_from(same_rows(32, row));
  auto const input = cudf::strings_column_view{col};
  using entry      = std::pair<char const*, char const*>;
  std::array const cases{entry{"z", R"({"b":0,"s":"http://x","23":2})"},
                         entry{"sameA", "first"},
                         entry{"SameA", "case"},
                         entry{"sameB", "q\""},
                         entry{"sameC", nullptr},
                         entry{"esc", "\xF0\x9F\x98\x80"},
                         entry{"\xF0\x9F\x98\x80", "surrogate"},
                         entry{"n", nullptr},
                         entry{"v", R"({"b":[1,true,null]})"},
                         entry{"", ""},
                         entry{"f", "3.0E-4"},
                         entry{"zero", "0"},
                         entry{"i", "-42"},
                         entry{"flag", "true"}};
  multi_paths paths{named({"r"})};
  for (auto const& c : cases) {
    paths.push_back(named({"r", c.first}));
  }
  paths.resize(16, named({"r", "sameA"}));
  auto const out    = routed(input, paths, std::vector<char>(16, 1));
  auto const parent = spark_rapids_jni::get_json_object(input, paths.front());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*parent, *out.at(0));
  auto const nulls = expected_column(same_rows(32, ""), std::vector<bool>(32, false));
  for (std::size_t i = 1; i < out.size(); ++i) {
    auto const* want = i <= cases.size() ? cases[i - 1].second : "first";
    if (want == nullptr) {
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(nulls, *out[i]);
    } else {
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(col_from(same_rows(32, want)), *out[i]);
    }
  }
}

TEST_F(GetJsonObjectTest, MergedRouteDepthFloors)
{
  auto const row =
    R"({"d":{"e":{"f":{"g":{"h":"five"}}}},"s":{"v":"two"},"p":")" + std::string(2048, 'x') + "\"}";
  auto rows = same_rows(32, row);
  rows[1].insert(row.find("\"f\""), R"("f":null,)");
  auto const col  = col_from(rows);
  auto const two  = col_from(same_rows(32, "two"));
  auto valid      = std::vector<bool>(32, true);
  valid[1]        = false;
  auto const five = expected_column(same_rows(32, "five"), valid);
  for (auto const [shallow, deep] : std::array{std::pair{3, 0}, std::pair{4, 7}, std::pair{1, 8}}) {
    SCOPED_TRACE(shallow);
    multi_paths paths(shallow, named({"s", "v"}));
    paths.insert(paths.end(), deep, named({"d", "e", "f", "g", "h"}));
    std::vector<char> routes(shallow, shallow >= 4 || deep == 8 ? 1 : 0);
    routes.insert(routes.end(), deep, deep == 8 ? 2 : 0);
    auto const out = routed(cudf::strings_column_view{col},
                            paths,
                            routes,
                            shallow == 3 ? gjo_scan_mode::offset_only : gjo_scan_mode::sampled);
    for (int i = 0; i < shallow + deep; ++i) {
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(i < shallow ? two : five, *out.at(i));
    }
  }
}

TEST_F(GetJsonObjectTest, MergedRouteMixedDepthCapParity)
{
  auto const row = R"({"a":[1,2],"d":{"e":{"f":{"g":{"h":{"i":"six"}}}}},)"
                   R"("s":"S","t":"T","z":"Z","p":")" +
                   std::string(2048, 'p') + "\"}";
  auto const col = col_from(same_rows(32, row));
  multi_paths paths{named({"z"}),
                    named({"t"}),
                    {named_path("a"), wildcard_path()},
                    {named_path("a"), {path_instruction_type::INDEX, "", 1}},
                    named({"d", "e", "f", "g", "h", "i"})};
  paths.insert(paths.end(), 760, named({"s"}));
  paths.insert(paths.end(), 8, named({"d", "e", "f", "g", "h"}));
  std::vector<char> routes(5, 0);
  routes.insert(routes.end(), 760, 1);
  routes.insert(routes.end(), 8, 2);
  auto const out = routed(cudf::strings_column_view{col}, paths, routes, gjo_scan_mode::exact);
  std::array const values{"Z", "T", "[1,2]", "2", "six", "S", R"({"i":"six"})"};
  std::vector<cudf::test::strings_column_wrapper> expected;
  for (auto const* value : values) {
    expected.push_back(col_from(same_rows(32, value)));
  }
  for (std::size_t i = 0; i < out.size(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected[i < 5 ? i : i < 765 ? 5 : 6], *out[i]);
  }
}

TEST_F(GetJsonObjectTest, MergedRouteDeadPathDepthErrorDoesNotThrow)
{
  // A scalar first match kills all descendants; the live control still throws.
  auto const nested   = R"({"v":)" + std::string(65, '[') + '0' + std::string(65, ']') + '}';
  auto const dead_col = col_from(same_rows(32, R"({"a":1,"a":)" + nested + '}'));
  auto const live_col = col_from(same_rows(32, R"({"a":)" + nested + '}'));
  multi_paths paths{named({"a", "v"})};
  for (int i = 1; i < 16; ++i) {
    paths.push_back(named({"a", "missing" + std::to_string(i)}));
  }
  auto const routes = std::vector<char>(16, 1);
  auto const out = routed(cudf::strings_column_view{dead_col}, paths, routes, gjo_scan_mode::exact);
  auto const nulls = expected_column(same_rows(32, ""), std::vector<bool>(32, false));
  for (auto const& col : out) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(nulls, *col);
  }
  EXPECT_THROW(routed(cudf::strings_column_view{live_col}, paths, routes, gjo_scan_mode::exact),
               cudf::logic_error);
}

TEST_F(GetJsonObjectTest, LastNonNullWideNamedBatches)
{
  auto const row =
    R"({"k":"first","k":"last","k":null,"e":"old","e":"","pad":")" + std::string(512, 'x') + "\"}";
  auto const col   = col_from(same_rows(32, row));
  auto const input = cudf::strings_column_view{col};
  multi_paths paths(40, named({"k"}));
  paths[0] = named({"e"});
  auto const out =
    routed(input, paths, std::vector<char>(40, 0), gjo_scan_mode::offset_only, -1, true);
  auto const last  = col_from(same_rows(32, "last"));
  auto const empty = col_from(same_rows(32, ""));
  for (std::size_t i = 0; i < out.size(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(i == 0 ? empty : last, *out[i]);
  }
}

TEST_F(GetJsonObjectTest, MergedRouteR2RowsParity)
{
  int device = 0, sms = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  // A partial warp after one R=1 walk round; leave row 3 empty but outside the eight samples.
  int const n = sms * 32 + 1;
  std::vector<std::string> rows(n + 2), k(n), u(n);
  auto const pad = R"(,"pad":")" + std::string(512, 'p') + "\"}";
  for (int i = 0; i < n; ++i) {
    k[i]        = std::string(i % 2 == 0 ? 1 : 257, 's') + std::to_string(i);
    u[i]        = i % 2 == 0 ? "U" : "odd";
    rows[i + 1] = R"({"k":")" + k[i] + R"(","u":{"v":")" + u[i] + R"("},"e":"")" + pad;
  }
  rows.front() = R"({'outside':1})";
  rows.back()  = R"({"outside":)";
  // Different rows and write extents force two independent retry offset columns.
  rows[1] = R"({"k":)" + quote_bomb(60) + R"(,"u":{"v":"U"},"e":""})";
  rows[2] = R"({"k":"other","u":{"v":)" + quote_bomb(40) + R"(},"e":""})";
  rows[n] = rows[1].substr(0, rows[1].size() - 1) + R"(,"z":x})";
  rows[4].clear();
  k[0] = quote_bomb(60, true);
  k[1] = "other";
  u[1] = quote_bomb(40, true);
  ASSERT_GT(k[0].size(), rows[1].size());
  ASSERT_GT(u[1].size(), rows[2].size());
  std::vector<bool> mask(n + 2, true), valid(n, true);
  mask[5] = valid[4] = false;  // Nonempty masked row.
  valid[3] = valid.back() = false;
  auto const col          = make_nonempty_nulls(rows, mask);
  auto const input        = cudf::strings_column_view{cudf::slice(*col, {1, n + 1}).front()};
  multi_paths paths{named({"u", "v"}), named({"k"}), named({"e"})};
  for (int i = 3; i < 18; ++i) {
    paths.push_back(named({"k", "missing" + std::to_string(i)}));
  }
  auto const out = routed(input, paths, std::vector<char>(18, 1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column(u, valid), *out.at(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column(k, valid), *out.at(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column(same_rows(n, ""), valid), *out.at(2));
  auto const nulls = expected_column(same_rows(n, ""), std::vector<bool>(n, false));
  for (std::size_t i = 3; i < out.size(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(nulls, *out[i]);
  }
}

TEST_F(GetJsonObjectTest, MergedRetryDeepRouteAfterMatch)
{
  auto const big = std::string(1024, 's');
  auto rows      = same_rows(33, R"({"d":{"e":{"f":")" + big + R"("}},"g":"G"})");
  rows[0]        = R"({"d":{"e":{"f":)" + quote_bomb(80) + R"(}},"g":"G"})";
  auto const col = col_from(rows);
  multi_paths paths(8, named({"d", "e", "missing"}));
  paths[0] = named({"d", "e", "f"});
  paths.push_back(named({"g"}));
  std::vector<char> routes(8, 2);
  routes.push_back(1);
  auto expected = same_rows(33, big);
  expected[0]   = quote_bomb(80, true);
  ASSERT_GT(expected[0].size(), rows[0].size());
  auto const out = routed(cudf::strings_column_view{col}, paths, routes);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col_from(expected), *out.at(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col_from(same_rows(33, "G")), *out.at(8));
}

TEST_F(GetJsonObjectTest, MergedRouteFanoutPairRoutesMerged)
{
  auto const col = col_from(same_rows(32, R"({"k0":"AA","k1":"AA"})"));
  for (int n : {43, 44}) {
    multi_paths paths(n, named({"k0"}));
    for (int i = 1; i < n; i += 2) {
      paths[i] = named({"k1"});
    }
    expect_uniform(
      routed(cudf::strings_column_view{col}, paths, std::vector<char>(n, n == 44)), 32, "AA");
  }
}

TEST_F(GetJsonObjectTest, MergedRouteRowFloorDenyBoundary)
{
  multi_paths const paths(47, named({"k0"}));
  for (int n : {31, 32}) {
    auto const col  = col_from(same_rows(n, R"({"k0":"A"})"));
    auto const mode = n == 32 ? gjo_scan_mode::sampled : gjo_scan_mode::offset_only;
    expect_uniform(
      routed(cudf::strings_column_view{col}, paths, std::vector<char>(47, n == 32), mode), n, "A");
  }
}

TEST_F(GetJsonObjectTest, MergedRouteMemberBandDenyBoundary)
{
  multi_paths paths;
  for (int i = 0; i < 16; ++i) {
    paths.push_back(named({"k" + std::to_string(i % 4)}));
  }
  for (int bytes : {1, 100}) {
    auto const col = col_from(same_rows(32, flat_row(4, bytes, 0)));
    expect_uniform(
      routed(cudf::strings_column_view{col}, paths, std::vector<char>(16, bytes == 100)),
      32,
      std::string(bytes, 'A'));
  }
  // Child members give Q=4 with actual credit and Q=7 with its upper bound.
  auto const col = col_from(same_rows(32, R"({"r":)" + flat_row(40, 194, 100) + '}'));
  auto const out = routed(cudf::strings_column_view{col},
                          multi_paths(4, named({"r", "k100"})),
                          {1, 1, 1, 1},
                          gjo_scan_mode::exact);
  expect_uniform(out, 32, std::string(194, 'A'));
}

TEST_F(GetJsonObjectTest, MergedRouteBatchCountCostBoundary)
{
  // Q=31 at 3151 B/row; capacities 9 and 36 leave tails of 7 and 16, respectively.
  expect_budget_routes(64, 150, 11, 0, {{2 << 20, 124}, {8 << 20, 16}, {-1, 0}});
}

TEST_F(GetJsonObjectTest, MergedRouteRequestFloorSurvivesBatchSplit)
{
  // Q=23 with 204800 B scratch/path. Adding one byte moves capacity from 22 to 23.
  constexpr int64_t budget = 22 * 204800;
  expect_budget_routes(40,
                       124,
                       23,
                       3,
                       {{1 << 20, 124},
                        {2 << 20, 124},
                        {budget, 124},
                        {budget + 1, 9},
                        {budget + 204800, 9},
                        {-1, 0}});
}

TEST_F(GetJsonObjectTest, MergedRouteScanBoundaries)
{
  auto const pad = R"(,"pad":")" + std::string(1024, 'p') + "\"}";
  auto const row = R"({"k":'ok')" + pad;
  multi_paths const paths(4, named({"k"}));
  for (auto const& [json, merged] : std::array{std::pair{row + ' ', true},
                                               std::pair{row + "  ", false},
                                               std::pair{R"({'k':'ok')" + pad, false}}) {
    auto const col = col_from(same_rows(32, json));
    expect_uniform(
      routed(cudf::strings_column_view{col}, paths, std::vector<char>(4, merged)), 32, "ok");
  }
  auto const open  = col_from(same_rows(32, row.substr(0, row.size() - 1)));
  auto const nulls = expected_column(same_rows(32, ""), std::vector<bool>(32, false));
  for (auto const& col : routed(cudf::strings_column_view{open}, paths, std::vector<char>(4))) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(nulls, *col);
  }
  for (int bytes : {4095, 4096}) {
    auto const key   = std::string(bytes, 'L');
    auto rival       = key;
    rival[bytes / 2] = 'M';
    auto const col =
      col_from(same_rows(32, "{\"" + rival + R"(":"other",")" + key + R"(":"hit"})"));
    auto const merged = bytes == 4095;
    auto const mode   = bytes == 4095 ? gjo_scan_mode::sampled : gjo_scan_mode::exact;
    auto const out    = routed(cudf::strings_column_view{col},
                               {named({key}), named({rival}), named({key}), named({rival})},
                            std::vector<char>(4, merged),
                            mode);
    for (std::size_t i = 0; i < out.size(); ++i) {
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(col_from(same_rows(32, i % 2 == 0 ? "hit" : "other")),
                                     *out[i]);
    }
  }
}

TEST_F(GetJsonObjectTest, MergedRouteExactScanCountsMaskedBytes)
{
  using scan_case = std::tuple<int, int, char, gjo_scan_mode>;
  // Slice-relative row 1 is not sampled; row 9 is. Both retain their bytes under a null.
  for (auto const& [bytes, poison, route, scan] :
       std::array{scan_case{12288, 1, 1, gjo_scan_mode::sampled},
                  scan_case{12288, 9, 0, gjo_scan_mode::sampled},
                  scan_case{12289, 1, 0, gjo_scan_mode::exact},
                  scan_case{12289, -1, 1, gjo_scan_mode::exact}}) {
    std::string const prefix = R"({"k":"ok","pad":")";
    auto rows    = same_rows(66, prefix + std::string(bytes - prefix.size() - 2, 'p') + "\"}");
    rows.front() = R"({'outside':1})" + std::string(20'000, ' ');
    rows.back()  = R"({"outside":)";
    std::vector<bool> valid(66, true);
    if (poison >= 0) {
      rows[poison + 1][1] = rows[poison + 1][3] = '\'';
      valid[poison + 1]                         = false;
    }
    auto const col   = make_nonempty_nulls(rows, valid);
    auto const input = cudf::strings_column_view{cudf::slice(*col, {1, 65}).front()};
    // One extra budget byte permits four paths only with slice-relative scratch sizing.
    auto const out = routed(
      input, multi_paths(4, named({"k"})), std::vector<char>(4, route), scan, 3 * 74 * bytes + 1);
    auto const expected =
      poison < 0 ? col_from(same_rows(64, "ok"))
                 : expected_column(same_rows(64, "ok"),
                                   std::vector<bool>(valid.begin() + 1, valid.end() - 1));
    for (auto const& result : out) {
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
    }
  }
}

TEST_F(GetJsonObjectTest, ParallelOverrideForcesBatchWidth)
{
  // Override 4 splits 8 identical k0 paths into two admitted width-4 batches; columns hold.
  auto const col =
    col_from(same_rows(32, R"({"k0":"AA","pad":")" + std::string(512, 'x') + "\"})"));
  multi_paths const paths(8, named({"k0"}));
  std::vector<char> routes;
  auto mode         = static_cast<gjo_scan_mode>(-1);
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto const out    = spark_rapids_jni::detail::get_json_object(
    cudf::strings_column_view{col}, paths, -1, 4, false, stream, mr, &routes, &mode);
  EXPECT_EQ(routes, std::vector<char>(8, 1));
  EXPECT_EQ(mode, gjo_scan_mode::sampled);
  EXPECT_EQ(out.size(), paths.size());
  expect_uniform(out, 32, "AA");
}
