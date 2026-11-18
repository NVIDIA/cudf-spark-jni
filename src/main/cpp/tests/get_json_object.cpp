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
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
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

using path_tuple  = std::tuple<path_instruction_type, std::string, int32_t>;
using path_list   = std::vector<path_tuple>;
using multi_paths = std::vector<path_list>;

std::vector<std::unique_ptr<cudf::column>> run_fused(cudf::strings_column_view const& input,
                                                     multi_paths const& paths)
{
  return spark_rapids_jni::get_json_object_multiple_paths(input, paths, -1, -1);
}

cudf::test::strings_column_wrapper col_from(std::vector<std::string> const& rows)
{
  return cudf::test::strings_column_wrapper(rows.begin(), rows.end());
}

/// Alternate explicit even/odd rows; use `same_rows` when they are identical.
std::vector<std::string> alt_rows(int n, std::string even, std::string odd)
{
  std::vector<std::string> rows;
  rows.reserve(n);
  for (int i = 0; i < n; ++i) {
    rows.push_back(i % 2 == 0 ? even : odd);
  }
  return rows;
}

std::vector<std::string> same_rows(int n, std::string row) { return alt_rows(n, row, row); }

void expect_rows_eq(cudf::column const& col, std::string const& expected, bool all_valid)
{
  if (all_valid) { ASSERT_EQ(col.null_count(), 0); }
  auto const vals = std::get<0>(cudf::test::to_host<std::string>(col.view()));
  ASSERT_EQ(vals.size(), 64u);
  for (int r = 0; r < 64; ++r) {
    EXPECT_EQ(vals[r], expected) << "row " << r;
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
