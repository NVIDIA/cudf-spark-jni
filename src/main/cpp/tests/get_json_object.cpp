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
#include <cudf_test/memory_resource_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/error.hpp>
#include <rmm/mr/callback_memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda/stream>

#include <array>
#include <atomic>
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

// Parity helpers compare raw validity, offsets, and chars bytes.
std::vector<cudf::bitmask_type> host_null_mask(cudf::column_view const& col)
{
  if (col.null_count() == 0) { return {}; }
  auto const words = (col.size() + 31) / 32;
  auto host_buffer = std::vector<cudf::bitmask_type>(words);
  CUDF_CUDA_TRY(cudaMemcpy(
    host_buffer.data(), col.null_mask(), words * sizeof(cudf::bitmask_type), cudaMemcpyDefault));
  return host_buffer;
}

// Preserve offset type and byte representation in parity checks.
std::vector<std::uint8_t> host_offsets_bytes(cudf::column_view const& col)
{
  auto const offsets = col.child(0);
  auto const nbytes  = static_cast<std::size_t>(offsets.size()) * cudf::size_of(offsets.type());
  auto host_buffer   = std::vector<std::uint8_t>(nbytes);
  CUDF_CUDA_TRY(cudaMemcpy(host_buffer.data(), offsets.head(), nbytes, cudaMemcpyDefault));
  return host_buffer;
}

// Copy chars bytes for exact string-column parity.
std::vector<char> host_chars_bytes(cudf::strings_column_view const& scv)
{
  auto const nbytes = scv.chars_size(cudf::get_default_stream());
  auto host_buffer  = std::vector<char>(nbytes);
  CUDF_CUDA_TRY(cudaMemcpy(
    host_buffer.data(), scv.chars_begin(cudf::get_default_stream()), nbytes, cudaMemcpyDefault));
  return host_buffer;
}

// Compare type, row count, validity, offsets, and chars exactly.
void expect_columns_parity(std::unique_ptr<cudf::column> const& a,
                           std::unique_ptr<cudf::column> const& b)
{
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(a->type().id(), b->type().id());
  EXPECT_EQ(a->size(), b->size());
  EXPECT_EQ(a->null_count(), b->null_count());
  EXPECT_EQ(host_null_mask(a->view()), host_null_mask(b->view()));
  EXPECT_EQ(host_offsets_bytes(a->view()), host_offsets_bytes(b->view()));
  EXPECT_EQ(host_chars_bytes(cudf::strings_column_view{a->view()}),
            host_chars_bytes(cudf::strings_column_view{b->view()}));
}

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

path_list named(std::vector<std::string> const& keys)
{
  path_list path;
  path.reserve(keys.size());
  for (auto const& key : keys) {
    path.emplace_back(path_instruction_type::NAMED, key, -1);
  }
  return path;
}

/// A depth-2 path list `$.<key>.v` for each key.
multi_paths named_depth2(std::vector<std::string> const& keys)
{
  multi_paths paths;
  paths.reserve(keys.size());
  for (auto const& key : keys) {
    paths.push_back(named({key, "v"}));
  }
  return paths;
}

/// The shared 8-key depth-2 batch of the abort and malformed-input tests.
multi_paths const eight_key_paths =
  named_depth2({"a", "b", "zz", "yyy", "xxxx", "aaaaaa", "bbbbbbb", "cccccccc"});

void pad_paths(multi_paths& paths, int from)
{
  paths.resize(8);
  for (int p = from; p < 8; ++p) {
    paths[p] = named({"pad" + std::to_string(p), "v"});
  }
}

void expect_rows_eq(cudf::column const& col, std::string const& expected, bool all_valid)
{
  if (all_valid) { ASSERT_EQ(col.null_count(), 0); }
  auto const vals = std::get<0>(cudf::test::to_host<std::string>(col.view()));
  ASSERT_EQ(vals.size(), 64u);
  for (int r = 0; r < 64; ++r) {
    EXPECT_EQ(vals[r], expected) << "row " << r;
  }
}

bool row_valid(cudf::column const& col, std::vector<cudf::bitmask_type> const& mask, int row)
{
  return col.null_count() == 0 || (!mask.empty() && cudf::bit_is_set(mask.data(), row));
}

void expect_rows_or_null(cudf::column const& col, char const* expected)
{
  auto const [vals, mask] = cudf::test::to_host<std::string>(col.view());
  ASSERT_EQ(vals.size(), 64u);
  for (int r = 0; r < 64; ++r) {
    if (expected == nullptr) {
      EXPECT_FALSE(row_valid(col, mask, r)) << "row " << r;
    } else {
      EXPECT_TRUE(row_valid(col, mask, r)) << "row " << r;
      EXPECT_EQ(vals[r], expected) << "row " << r;
    }
  }
}

/// The first `n` paths of `paths` (a deliberately general-routed oracle batch
/// when `n` < the merged route's >= 8 path floor).
multi_paths first_n(multi_paths const& paths, std::size_t n)
{
  return multi_paths(paths.begin(), paths.begin() + n);
}

using multi_outputs = std::vector<std::unique_ptr<cudf::column>>;

std::pair<multi_outputs, multi_outputs> run_merged_vs_general(std::vector<std::string> const& rows,
                                                              multi_paths const& merged_paths)
{
  auto const general_paths = first_n(merged_paths, 7);
  auto merged_out          = run_fused(cudf::strings_column_view{col_from(rows)}, merged_paths);
  auto general_out         = run_fused(cudf::strings_column_view{col_from(rows)}, general_paths);
  CUDF_EXPECTS(merged_out.size() == merged_paths.size() && general_out.size() == 7,
               "merged-vs-general output shape mismatch");
  return {std::move(merged_out), std::move(general_out)};
}

/// Parity of the first `n` columns (default 7 = the general oracle's size); both
/// column counts must be at least `n`.
void expect_first7_parity(multi_outputs const& merged,
                          multi_outputs const& general,
                          std::size_t n     = 7,
                          char const* trace = "parity column ")
{
  ASSERT_GE(merged.size(), n);
  ASSERT_GE(general.size(), n);
  for (std::size_t col = 0; col < n; ++col) {
    SCOPED_TRACE(std::string(trace) + std::to_string(col));
    expect_columns_parity(merged[col], general[col]);
  }
}

/// ESCAPED-writer result for a single-quoted string of `n` double quotes under `$.k`.
std::string quote_bomb_value(int n)
{
  std::string out = R"({"a":")";
  for (int i = 0; i < n; ++i) {
    out += R"(\")";  // one inner quote re-emitted as backslash-escape
  }
  out += R"("})";  // the writer closes the value's opening double quote
  return out;
}

}  // namespace

TEST_F(GetJsonObjectTest, MergedRouteEmptyRowsParity)
{
  // Empty rows must null every path rather than leave unwritten stringviews.
  auto const rows = alt_rows(64, R"({"k0":{"v":"v0"},"k1":{"v":"v1"}})", "");

  auto const merged_paths = named_depth2(
    {"k0", "k1", "missing2", "missing3", "missing4", "missing5", "missing6", "missing7"});
  auto const [merged_out, general_out] = run_merged_vs_general(rows, merged_paths);

  for (std::size_t col = 0; col < 8; ++col) {
    SCOPED_TRACE("merged path column " + std::to_string(col));
    auto const [vals, mask] = cudf::test::to_host<std::string>(merged_out[col]->view());
    ASSERT_EQ(vals.size(), 64u);
    for (int r = 0; r < 64; ++r) {
      bool const is_valid = row_valid(*merged_out[col], mask, r);
      if (r % 2 == 1) {
        EXPECT_FALSE(is_valid) << "empty row " << r;
      } else if (col < 2) {
        EXPECT_TRUE(is_valid) << "object row " << r;
        EXPECT_EQ(vals[r], "v" + std::to_string(col)) << "object row " << r;
      }
    }
  }

  expect_first7_parity(merged_out, general_out);
}

TEST_F(GetJsonObjectTest, MergedRouteSortBeforeFlattenParity)
{
  // An unsorted flat child blob would lose late short key `z` during binary search.
  auto const rows =
    same_rows(64, R"({"a":{"v":"A"},"bbb":{"v":"B"},"seven0":{"v":"w"},"z":{"v":"Z"}})");

  auto const merged_paths =
    named_depth2({"a", "bbb", "seven0", "z", "zzz", "yyyy", "aaaaaa", "bbbbbbb"});
  auto const [merged_out, general_out] = run_merged_vs_general(rows, merged_paths);

  for (std::size_t col : {std::size_t{0}, std::size_t{3}}) {
    SCOPED_TRACE("short-key column " + std::to_string(col));
    char const* expected = col == 0 ? "A" : "Z";
    ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*merged_out[col], expected, true));
  }

  expect_first7_parity(merged_out, general_out);
}

TEST_F(GetJsonObjectTest, MergedRouteDuplicateFirstValueAborts)
{
  // A first non-object value aborts `$.a.v`; a later duplicate cannot resurrect it.
  struct Case {
    char const* label;
    char const* row;
  };
  Case const cases[] = {
    {"null", R"({"a":null,"a":{"v":"1"},"b":{"v":"B"},"z":{"v":"Z"}})"},
    {"scalar", R"({"a":1,"a":{"v":"1"},"b":{"v":"B"},"z":{"v":"Z"}})"},
    {"array", R"({"a":[1,2],"a":{"v":"1"},"b":{"v":"B"}})"},
    {"string", R"({"a":"x","a":{"v":"1"},"b":{"v":"B"}})"},
  };

  for (auto const& c : cases) {
    SCOPED_TRACE(std::string("first value: ") + c.label);
    auto const rows = same_rows(64, c.row);

    auto const [merged_out, general_out] = run_merged_vs_general(rows, eight_key_paths);

    ASSERT_EQ(merged_out[0]->null_count(), 64)
      << "$.a.v must abort on the " << c.label << " first value";
    ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*merged_out[1], "B", true));

    expect_first7_parity(merged_out, general_out);
  }
}

TEST_F(GetJsonObjectTest, MergedRouteEscapedKeyMatches)
{
  // Escaped key `\u0061` must match non-first trie candidate `a`.
  auto const rows = same_rows(64, R"({"z":{"v":"Z"},"\u0061":{"v":"A"},"yy":{"v":"Y"}})");

  auto const merged_paths =
    named_depth2({"z", "a", "yy", "xx", "wwwwwww", "vvvvvvvv", "uuuuuuuu", "ttt"});
  auto const [merged_out, general_out] = run_merged_vs_general(rows, merged_paths);

  ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*merged_out[1], "A", true));
  ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*merged_out[0], "Z", true));

  expect_first7_parity(merged_out, general_out);
}

TEST_F(GetJsonObjectTest, MergedRouteSurrogatePairKeyMatches)
{
  // A surrogate escape must match its four-byte UTF-8 key outside the first trie slot.
  std::string const emoji = "\xF0\x9F\x98\x80";  // U+1F600
  auto const rows = same_rows(64, R"({"z":{"v":"Z"},"\uD83D\uDE00":{"v":"inner"},"yy":{"v":"Y"}})");
  auto const merged_paths = named_depth2({"z", "yy", "aa", "bb", "cc", "dd", "ee", emoji});
  auto const [merged_out, general_out] = run_merged_vs_general(rows, merged_paths);

  ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*merged_out[7], "inner", true));

  expect_first7_parity(merged_out, general_out);
}

TEST_F(GetJsonObjectTest, MergedRouteMalformedJsonNullsAllPaths)
{
  // A parse error after an earlier match must null all paths in the row.
  auto const rows = alt_rows(64, R"({"a":{"v":"A"},"b":x})", R"({"x":1,"b":y})");
  auto const [merged_out, general_out] = run_merged_vs_general(rows, eight_key_paths);

  for (std::size_t col = 0; col < 8; ++col) {
    SCOPED_TRACE("malformed column " + std::to_string(col));
    ASSERT_EQ(merged_out[col]->null_count(), 64) << "col " << col;
  }
  expect_first7_parity(merged_out, general_out);
}

TEST_F(GetJsonObjectTest, MergedRouteTruncatedEofAfterColonComma)
{
  // EOF after `:` or `,` must null paths without reading beyond the row buffer.
  auto const rows                      = alt_rows(64, R"({"a":)", R"({"x":1,)");
  auto const [merged_out, general_out] = run_merged_vs_general(rows, eight_key_paths);

  for (std::size_t col = 0; col < 8; ++col) {
    SCOPED_TRACE("truncated-eof column " + std::to_string(col));
    ASSERT_EQ(merged_out[col]->null_count(), 64) << "col " << col;
  }
  expect_first7_parity(merged_out, general_out);
}

TEST_F(GetJsonObjectTest, MergedRouteMixedDepthPartitionParity)
{
  // Mixed-depth batches retain merged shallow paths while deep paths stay general.
  auto const rows =
    same_rows(64,
              R"({"k0":{"v":"v0"},"k1":{"v":"v1"},"k2":{"v":"v2"},"k3":{"v":"v3"},)"
              R"("m4":{"v":"v4"},"m5":{"v":"v5"},"m6":{"v":"v6"},"m7":{"v":"v7"},)"
              R"("deep3":{"m1":{"m2":"D3"}},"deep5":{"l1":{"l2":{"l3":{"l4":"D5"}}}}})");

  multi_paths paths = named_depth2({"k0", "k1", "k2", "k3", "m4", "m5", "m6", "m7"});
  paths.push_back(named({"deep3", "m1", "m2"}));
  paths.push_back(named({"deep5", "l1", "l2", "l3", "l4"}));

  // Eight shallow paths are the merged oracle; two deep paths stay general.
  auto const merged_only = first_n(paths, 8);
  auto const deep_only   = multi_paths(paths.begin() + 8, paths.end());

  auto const mixed_out  = run_fused(cudf::strings_column_view{col_from(rows)}, paths);
  auto const merged_out = run_fused(cudf::strings_column_view{col_from(rows)}, merged_only);
  auto const deep_out   = run_fused(cudf::strings_column_view{col_from(rows)}, deep_only);
  ASSERT_EQ(mixed_out.size(), 10u);
  ASSERT_EQ(merged_out.size(), 8u);
  ASSERT_EQ(deep_out.size(), 2u);

  for (std::size_t col = 0; col < 10; ++col) {
    SCOPED_TRACE("mixed column " + std::to_string(col));
    std::string const expected = col < 8 ? std::string("v") + std::to_string(col)
                                         : (col == 8 ? std::string("D3") : std::string("D5"));
    ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*mixed_out[col], expected, true));
  }

  expect_first7_parity(mixed_out, merged_out, 8);
  for (std::size_t col = 8; col < 10; ++col) {
    SCOPED_TRACE("deep parity column " + std::to_string(col));
    expect_columns_parity(mixed_out[col], deep_out[col - 8]);
  }
}

TEST_F(GetJsonObjectTest, MergedRouteProductionConflictShapesParity)
{
  // Preserve case-sensitive siblings, first-duplicate wins, and mixed-depth routing.
  auto const rows =
    same_rows(64,
              R"({"module":{"v":"m_low"},"Module":{"v":"M_up"},"zipcode":{"v":"z_low"},)"
              R"("Zipcode":"Z_unused","zipCode":{"v":"Z_camel"},)"
              R"("traceparent":{"v":"t_low"},"traceParent":{"v":"T_camel"},)"
              R"("dup":{"v":"first"},"dup":"second",)"
              R"("refineData":{"mid":{"leaf":"R_mid"},"value":{"leaf":"R_val"}},)"
              R"("mediaPayload":{"mid":{"leaf":"A_mid"},"subModule":{"leaf":"A_mod"}},)"
              R"("question":{"mid":{"leaf":"Q_mid"},"requestID":{"leaf":"Q_req"}}})");

  auto const shallow = named_depth2(
    {"module", "Module", "zipcode", "zipCode", "traceparent", "traceParent", "dup", "dup"});
  multi_paths paths(shallow);
  paths.push_back(named({"module", "v"}));                // depth-2 duplicate of col 0 -> merged
  paths.push_back(named({"refineData", "mid", "leaf"}));  // depth-3 -> general
  paths.push_back(named({"mediaPayload", "subModule", "leaf"}));  // depth-3 -> general
  paths.push_back(named({"question", "mid", "leaf"}));            // depth-3 -> general

  auto const [batch_out, general_out] = run_merged_vs_general(rows, paths);

  char const* expected[] = {"m_low",
                            "M_up",
                            "z_low",
                            "Z_camel",
                            "t_low",
                            "T_camel",
                            "first",
                            "first",
                            "m_low",
                            "R_mid",
                            "A_mod",
                            "Q_mid"};
  for (std::size_t col = 0; col < 12; ++col) {
    SCOPED_TRACE("conflict column " + std::to_string(col) + " (" + expected[col] + ")");
    ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*batch_out[col], expected[col], true));
  }

  expect_first7_parity(batch_out, general_out);
}

TEST_F(GetJsonObjectTest, MergedRouteSignatureBucketParity)
{
  // Colliding signatures, long tails, empty names, and escaped keys must resolve distinctly.
  auto const rows = same_rows(64,
                              R"({"fieldOne":{"v":"1"},"fieldTwo":{"v":"2"},"fieldOnX":{"v":"3"},)"
                              R"("aaaaLongTailX":{"v":"X"},"aaaaLongTailY":{"v":"Y"},"":{"v":"E"},)"
                              R"("b":{"v":"B"},"\u0062":"escaped-b","z":{"v":"Z"}})");

  auto const paths = named_depth2(
    {"fieldOne", "fieldTwo", "fieldOnX", "aaaaLongTailX", "aaaaLongTailY", "", "b", "z", "q"});
  auto const [batch_out, general_out] = run_merged_vs_general(rows, paths);

  char const* expected[] = {"1", "2", "3", "X", "Y", "E", "B", "Z", nullptr};
  for (std::size_t col = 0; col < 9; ++col) {
    SCOPED_TRACE("signature column " + std::to_string(col) + " (" +
                 (expected[col] ? expected[col] : "null") + ")");
    ASSERT_NO_FATAL_FAILURE(expect_rows_or_null(*batch_out[col], expected[col]));
  }

  expect_first7_parity(batch_out, general_out);
}

TEST_F(GetJsonObjectTest, MergedRouteSignatureSingletonBucket)
{
  // Singleton buckets must match full names despite shared signature bytes at other lengths.
  auto const rows = same_rows(
    64, R"({"abcde":{"v":"short"},"abcdeXYZ":{"v":"long"},"abcdeXY":{"v":"mid"},"u":{"v":"U"}})");

  auto const paths = named_depth2({"abcde", "abcdeXYZ", "abcdeXY", "u", "v", "w", "x", "y"});
  auto const [batch_out, general_out] = run_merged_vs_general(rows, paths);

  char const* expected[] = {"short", "long", "mid", "U", nullptr, nullptr, nullptr, nullptr};
  for (std::size_t col = 0; col < 8; ++col) {
    SCOPED_TRACE("singleton column " + std::to_string(col));
    ASSERT_NO_FATAL_FAILURE(expect_rows_or_null(*batch_out[col], expected[col]));
  }

  expect_first7_parity(batch_out, general_out);
}

TEST_F(GetJsonObjectTest, MergedRouteLongNameMetadataFootprint)
{
  auto const input = col_from(same_rows(32, R"({"k1":{"v":"hit"}})"));
  std::string const long_name(64 * 1'024, 'L');
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream{
    cudf::get_current_device_resource_ref()};
  // Tiny rows need less than 8 MiB per request; a group-wide length stride exceeds 250 MiB.
  constexpr std::size_t request_ceiling = 8 * 1'024 * 1'024;
  for (bool const long_at_root : {true, false}) {
    SCOPED_TRACE(long_at_root ? "long root" : "long singleton leaf");
    multi_paths paths;
    paths.reserve(1'000);
    for (int p = 0; p < 1'000; ++p) {
      paths.push_back(named({"k" + std::to_string(p), "v"}));
    }
    std::get<1>(paths[0][long_at_root ? 0 : 1]) = long_name;
    std::atomic<std::size_t> oversized_request{0};
    rmm::mr::callback_memory_resource capped_mr{
      [&](cuda::stream_ref allocation_stream, std::size_t bytes, std::size_t alignment, void*)
        -> void* {
        if (bytes > request_ceiling) {
          oversized_request.store(bytes);
          throw rmm::bad_alloc{"Long-name trie metadata exceeds the test resource ceiling"};
        }
        return upstream.allocate(allocation_stream, bytes, alignment);
      },
      [&](cuda::stream_ref allocation_stream,
          void* ptr,
          std::size_t bytes,
          std::size_t alignment,
          void*) { upstream.deallocate(allocation_stream, ptr, bytes, alignment); }};
    cudf::test::scoped_current_device_resource current_mr{capped_mr};
    std::vector<std::unique_ptr<cudf::column>> out;
    EXPECT_NO_THROW(out = run_fused(cudf::strings_column_view{input}, paths));
    cudf::get_default_stream().sync();
    ASSERT_EQ(oversized_request.load(), 0u);
    ASSERT_EQ(out.size(), paths.size());
    auto const hit = col_from(same_rows(32, "hit"));
    for (std::size_t col = 0; col < paths.size(); ++col) {
      SCOPED_TRACE("column " + std::to_string(col));
      if (col == 1) {
        CUDF_TEST_EXPECT_COLUMNS_EQUAL(hit, out[col]->view());
      } else {
        EXPECT_EQ(out[col]->null_count(), 32);
      }
    }
  }
}

TEST_F(GetJsonObjectTest, MergedRouteLongNameMatching)
{
  std::string const long_name(64 * 1'024, 'L');
  auto collision                  = long_name;
  auto absent                     = long_name;
  collision[long_name.size() / 2] = 'M';
  absent[long_name.size() / 2]    = 'N';
  auto const escaped_name         = R"(\u004c)" + long_name.substr(1);
  for (bool const long_at_root : {true, false}) {
    auto const long_path = [&](std::string const& key) {
      return long_at_root ? named({key, "v"}) : named({"r", key});
    };
    auto const paths = multi_paths{long_path(long_name),
                                   long_path(collision),
                                   long_path(absent),
                                   long_path(long_name),
                                   named({"short", "v"}),
                                   named({"short", "vv"}),
                                   named({"short", ""}),
                                   named({"short", "missing"}),
                                   named({"solo", long_name}),
                                   named({"tail", "v"})};
    for (int spelling = 0; spelling < 6; ++spelling) {
      SCOPED_TRACE(std::string(long_at_root ? "root " : "leaf ") + std::to_string(spelling));
      auto const quote  = spelling == 2 ? '\'' : '"';
      auto const& key   = spelling == 1 ? escaped_name : long_name;
      auto const member = [&](std::string const& name, std::string const& value) {
        auto const payload = long_at_root ? R"({"v":)" + value + "}" : value;
        return quote + name + quote + ":" + payload;
      };
      auto first = member(key, spelling == 3 ? "null" : R"("long")");
      if (spelling == 4) { first = quote + key + quote + ":0"; }
      auto const members = member(collision, R"("other")") + "," + first + "," +
                           member(long_name, R"("later")") + "," +
                           member(long_name + "L", R"("wrong")");
      auto row = long_at_root ? "{" + members : R"({"r":{)" + members + "}";
      row += R"(,"short":{"v":"short","vv":"two","":"empty","too-long":"wrong"},)"
             R"("solo":{")" +
             key + R"(":"single"},"tail":{"v":"tail"}})";
      if (spelling == 5) { row = R"({"tail":{"v":"tail"},")" + long_name; }
      auto const input = col_from(same_rows(64, row));
      auto const out   = run_fused(cudf::strings_column_view{input}, paths);
      ASSERT_EQ(out.size(), paths.size());
      auto const long_value = spelling == 3 || (spelling == 4 && long_at_root) ? nullptr
                              : spelling == 4                                  ? "0"
                                                                               : "long";
      std::array<char const*, 10> expected{long_value,
                                           "other",
                                           nullptr,
                                           long_value,
                                           "short",
                                           "two",
                                           "empty",
                                           nullptr,
                                           "single",
                                           "tail"};
      if (spelling == 5) { expected.fill(nullptr); }
      for (std::size_t col = 0; col < paths.size(); ++col) {
        SCOPED_TRACE("column " + std::to_string(col));
        ASSERT_NO_FATAL_FAILURE(expect_rows_or_null(*out[col], expected[col]));
        auto const single = run_fused(cudf::strings_column_view{input}, {paths[col]});
        ASSERT_EQ(single.size(), 1u);
        ASSERT_NO_FATAL_FAILURE(expect_columns_parity(out[col], single[0]));
      }
    }
  }
}

TEST_F(GetJsonObjectTest, MergedRouteCertifiedSubtreeParity)
{
  // Compact canonical subtrees are byte-identical to the writer's output.
  auto const compact = R"({"a":[1,2,{"b":"x\"y\\z","c":true,"d":null,"e":-12}],"f":0})";
  auto const rows    = same_rows(64, R"({"k":)" + std::string(compact) + R"(,"u":{"v":"U"}})");

  // Query a parent subtree and a separate depth-2 companion in the same batch.
  multi_paths paths{named({"k"}), named({"u", "v"})};
  pad_paths(paths, 2);
  auto const [batch_out, general_out] = run_merged_vs_general(rows, paths);

  ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*batch_out[0], compact, true));

  expect_first7_parity(batch_out, general_out, 7, "certified parity column ");
}

TEST_F(GetJsonObjectTest, MergedRouteUncertifiedSubtreeFallbacks)
{
  // Every uncertified subtree falls back to writer output with general-route parity.

  for (auto const& [label, sub] : std::vector<std::pair<std::string, std::string>>{
         {"whitespace", R"({ "a" : [1, 2] })"},
         {"unicode-escape", R"({"a":"\u4e2d\u56FD"})"},
         {"escaped-slash", R"({"a":"http:\/\/x"})"},
         {"float-1.0", R"({"a":1.0})"},
         {"float-1e3", R"({"a":1e3})"},
         {"float-minus-0", R"({"a":-0})"},
         {"control-char", std::string(R"({"a":"x")") + '\x01' + std::string(R"("})")},
         {"single-quote", R"({'a':1})"}}) {
    SCOPED_TRACE("uncertified case " + label);
    auto const rows = same_rows(64, R"({"k":)" + sub + R"(,"u":{"v":"U"}})");
    multi_paths paths{named({"k"}), named({"u", "v"})};
    pad_paths(paths, 2);
    auto const [batch_out, general_out] = run_merged_vs_general(rows, paths);
    expect_first7_parity(batch_out, general_out, 7, "fallback parity column ");
  }
}

TEST_F(GetJsonObjectTest, MergedRouteCertifiedSubtreeDeepNesting)
{
  // The certificate must handle deeply nested compact subtrees.
  std::string const deep = R"({"a":[[[[{"b":1}]]]],"c":{"d":{"e":true}}})";
  auto const rows        = same_rows(64, R"({"k":)" + deep + R"(,"u":{"v":"U"}})");

  multi_paths paths{named({"k"}), named({"u", "v"})};
  pad_paths(paths, 2);
  auto const [batch_out, general_out] = run_merged_vs_general(rows, paths);

  ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*batch_out[0], deep, false));
  expect_first7_parity(batch_out, general_out, 7, "deep parity column ");
}

TEST_F(GetJsonObjectTest, MergedRouteCertifiedSubtreeDupAndNullAborts)
{
  // First duplicate wins, null aborts, and parent/child paths coexist with certification.
  auto const rows = same_rows(64,
                              R"({"dup":{"x":[1,"s",true]},"dup":"second","n":null,)"
                              R"("n":{"y":{"z":2}},"a":{"b":{"c":[false,null]}},"q":"tail"})");

  multi_paths paths{named({"dup"}), named({"n"}), named({"a", "b"})};
  pad_paths(paths, 3);
  auto const [batch_out, general_out] = run_merged_vs_general(rows, paths);

  ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*batch_out[0], R"({"x":[1,"s",true]})", false));
  ASSERT_EQ(batch_out[1]->null_count(), 64) << "$.n must abort on the null first value";
  ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*batch_out[2], R"({"c":[false,null]})", false));

  expect_first7_parity(batch_out, general_out, 7, "dup/abort parity column ");
}

TEST_F(GetJsonObjectTest, MergedRouteCertifiedCanonicalSpellings)
{
  // Canonical escapes and negative single-digit integers must preserve writer-byte parity.
  std::string const esc = R"({"a":"\b\f\n\r\t"})";
  std::string const neg = R"({"a":[-1,-2,-3,-9]})";
  auto const rows       = alt_rows(64,
                             R"({"k":)" + esc + R"(,"j":)" + neg + R"(,"u":{"v":"U"}})",
                             R"({"k":)" + neg + R"(,"j":)" + esc + R"(,"u":{"v":"U"}})");

  multi_paths paths{named({"k"}), named({"j"}), named({"u", "v"})};
  pad_paths(paths, 3);
  auto const [batch_out, general_out] = run_merged_vs_general(rows, paths);

  auto const k_vals = std::get<0>(cudf::test::to_host<std::string>(batch_out[0]->view()));
  auto const j_vals = std::get<0>(cudf::test::to_host<std::string>(batch_out[1]->view()));
  for (int r = 0; r < 64; ++r) {
    EXPECT_EQ(k_vals[r], (r % 2 == 0) ? esc : neg) << "row " << r;
    EXPECT_EQ(j_vals[r], (r % 2 == 0) ? neg : esc) << "row " << r;
  }
  expect_first7_parity(batch_out, general_out, 7, "canonical-spelling parity column ");
}

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

TEST_F(GetJsonObjectTest, UnifiedDeepTrieRidesMergedRoute)
{
  // Eight deep NAMED paths must share the merged trie with shallow companions.
  auto const row =
    R"({"l1":{"l2":{"l3":{"t":"d3","l4":{"l5":"d5","x":"never"}}}},"dup":{"a":1,"a":2,"b":{"c":"bc"}}})";
  auto const rows = same_rows(64, row);

  multi_paths paths;
  paths.reserve(10);
  paths.push_back(named({"l1", "l2", "l3", "t"}));
  paths.push_back(named({"l1", "l2", "l3", "l4", "l5"}));
  paths.push_back(named({"l1", "l2", "l3", "l4", "x"}));
  paths.push_back(named({"dup", "a"}));
  paths.push_back(named({"dup", "b"}));
  paths.push_back(named({"dup", "b", "c"}));
  // Four missing chains keep the deep-path count at eight.
  for (int p = 0; p < 4; ++p) {
    paths.push_back(named({"l1", "l2", "miss" + std::to_string(p)}));
  }
  ASSERT_EQ(paths.size(), 10u);

  auto const [batch_out, general_out] = run_merged_vs_general(rows, paths);

  char const* const expected[] = {
    "d3", "d5", "never", "1", R"({"c":"bc"})", "bc", nullptr, nullptr, nullptr, nullptr};

  expect_first7_parity(batch_out, general_out, 7, "unified merged column ");
  for (std::size_t col = 0; col < 10; ++col) {
    SCOPED_TRACE("unified merged golden " + std::to_string(col));
    ASSERT_NO_FATAL_FAILURE(expect_rows_or_null(*batch_out[col], expected[col]));
  }
}

TEST_F(GetJsonObjectTest, UnifiedDeepTrieNullAbortAtDepth)
{
  // A first null intermediate must abort the depth-five chain despite a later duplicate.
  auto const rows = same_rows(64, R"({"a":{"b":{"c":null,"c":{"d":{"e":"v"}}}}})");

  multi_paths paths;
  paths.push_back(named({"a", "b", "c", "d", "e"}));
  for (int p = 0; p < 7; ++p) {
    paths.push_back(named({"a", "b", "z" + std::to_string(p)}));
  }
  ASSERT_EQ(paths.size(), 8u);

  auto const [batch_out, general_out] = run_merged_vs_general(rows, paths);
  expect_columns_parity(batch_out[0], general_out[0]);
  SCOPED_TRACE("null-abort chain");
  ASSERT_EQ(batch_out[0]->null_count(), 64) << "null-abort chain must abort";
}

TEST_F(GetJsonObjectTest, DeepGateRejectionRoutesGeneral)
{
  // Fewer than eight deep paths must remain general and match their oracle.
  auto const row  = R"({"a":{"b":{"c":{"d":"deep"}}}})";
  auto const rows = same_rows(64, row);

  multi_paths paths;
  paths.reserve(9);
  paths.push_back(named({"a", "b", "c", "d"}));
  // Seven shallow companions keep the deep-path count below the gate.
  for (int p = 0; p < 7; ++p) {
    paths.push_back(named({"p" + std::to_string(p), "v"}));
  }
  ASSERT_EQ(paths.size(), 8u);

  auto const [batch_out, general_out] = run_merged_vs_general(rows, paths);
  expect_first7_parity(batch_out, general_out, 7, "deep-gate-reject column ");
  SCOPED_TRACE("deep path column");
  ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*batch_out[0], "deep", false));
}

TEST_F(GetJsonObjectTest, DepthBeyondFiveStaysGeneral)
{
  // Depth-six NAMED paths must stay general regardless of batch size.
  auto const row  = R"({"a":{"b":{"c":{"d":{"e":{"f":"six"}}}}}})";
  auto const rows = same_rows(64, row);

  multi_paths paths;
  paths.reserve(10);
  paths.push_back(named({"a", "b", "c", "d", "e", "f"}));
  // Nine depth-3 companions trip the deep gate; the depth-6 path remains general.
  for (int p = 0; p < 9; ++p) {
    paths.push_back(named({"l1", "l2", "m" + std::to_string(p)}));
  }
  auto const [batch_out, general_out] = run_merged_vs_general(rows, paths);
  ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*batch_out[0], "six", false));
  for (std::size_t col = 1; col < 7; ++col) {
    SCOPED_TRACE("depth-6 companion column " + std::to_string(col));
    expect_columns_parity(batch_out[col], general_out[col]);
  }
}

TEST_F(GetJsonObjectTest, OutOfBoundWriteRetryRelaunch)
{
  // Oversized subtree retries while companion outputs retain their values.
  // Row anatomy: `{"k":{"a":'<30 quotes>'},"u":{"v":"U"}}` = 58 bytes;
  // output for `$.k` = `{"a":"` + 30x`\"` + `"}` = 68 bytes.
  auto const bomb   = std::string(30, '"');
  auto const row    = R"({"k":{"a":')" + bomb + R"('},"u":{"v":"U"}})";
  auto const rows   = same_rows(64, row);
  auto const expect = quote_bomb_value(30);

  ASSERT_EQ(row.size(), 58u) << "anatomy comment drifted";
  ASSERT_EQ(expect.size(), 68u) << "anatomy comment drifted";

  multi_paths paths{named({"k"}), named({"u", "v"})};  // depth-1 subtree + companion
  pad_paths(paths, 2);
  auto const [batch_out, general_out] = run_merged_vs_general(rows, paths);

  ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*batch_out[0], expect, true));
  ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*batch_out[1], "U", true));
  for (std::size_t col = 2; col < 8; ++col) {
    SCOPED_TRACE("retry pad column " + std::to_string(col));
    EXPECT_EQ(batch_out[col]->null_count(), 64);
  }
  expect_first7_parity(batch_out, general_out, 7, "retry parity column ");
}

TEST_F(GetJsonObjectTest, UnquotedObjectKeySubtreeParity)
{
  // Unquoted object keys must use normalized writer bytes, never certified raw bytes.
  struct UnquotedCase {
    std::string label;
    std::string sub;
    std::string expected;
  };
  auto const cases = std::vector<UnquotedCase>{
    {"numeric-key", R"({"a":{1231:2}})", R"({"23":2})"},
    {"single-quoted-outer", R"({'a':{1231:2}})", R"({"23":2})"},
    {"mixed-keys", R"({"a":{1231:2,"b":3}})", R"({"23":2,"b":3})"},
    {"array-element", R"({"a":[{1231:2}]})", R"([{"23":2}])"},
    {"nested-unquoted", R"({"a":{"b":{1231:2}}})", R"({"b":{"23":2}})"},
  };

  for (auto const& c : cases) {
    SCOPED_TRACE("unquoted-key case " + c.label);
    auto const row  = std::string(R"({"k":)") + c.sub + std::string(R"(,"u":{"v":"U"}})");
    auto const rows = same_rows(64, row);

    multi_paths paths{named({"k", "a"}), named({"u", "v"})};
    pad_paths(paths, 2);

    auto const [batch_out, general_out] = run_merged_vs_general(rows, paths);

    ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*batch_out[0], c.expected, true));
    ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*batch_out[1], "U", false));
    expect_first7_parity(batch_out, general_out, 7, "unquoted-key parity column ");
  }
}

TEST_F(GetJsonObjectTest, LastNonNullWideNamedBatches)
{
  auto const even =
    R"({"a":"first","a":"last","b":"old","b":null,"c":null,"c":"set","n":null,"e":"first","e":""})";
  auto const odd =
    R"({"a":null,"a":"latest","b":null,"b":"new","c":"keep","c":null,"n":null,"e":""})";
  std::array<std::string, 6> const keys{"a", "b", "c", "n", "missing", "e"};
  std::array<std::string, 6> const even_values{"last", "old", "set", "", "", ""};
  std::array<std::string, 6> const odd_values{"latest", "new", "keep", "", "", ""};

  // Cross both admission floors; null duplicates retain values and empty strings remain valid.
  for (auto const rows : std::array<int, 3>{31, 32, 33}) {
    auto const input_col = col_from(alt_rows(rows, even, odd));
    auto const input     = cudf::strings_column_view{input_col};
    for (auto const path_count : std::array<int, 3>{123, 124, 125}) {
      multi_paths paths;
      paths.reserve(path_count);
      for (int p = 0; p < path_count; ++p) {
        paths.push_back(named({keys[p % keys.size()]}));
      }
      for (auto const parallel : std::array<int32_t, 2>{-1, 1}) {
        SCOPED_TRACE(testing::Message()
                     << "rows=" << rows << " paths=" << path_count << " parallel=" << parallel);
        auto const output = spark_rapids_jni::get_json_object_multiple_paths(
          input, paths, -1, parallel, spark_rapids_jni::named_field_match_policy::LAST_NON_NULL);
        ASSERT_EQ(output.size(), paths.size());
        for (std::size_t p = 0; p < output.size(); ++p) {
          auto const key           = p % keys.size();
          auto const expected_rows = alt_rows(rows, even_values[key], odd_values[key]);
          std::vector<bool> const valid(rows, key != 3 && key != 4);
          auto const expected = cudf::test::strings_column_wrapper(
            expected_rows.begin(), expected_rows.end(), valid.begin());
          CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *output[p]);
        }
      }
    }
  }
}

TEST_F(GetJsonObjectTest, MergedRouteRejectsDepthErrors)
{
  auto const nested = std::string(65, '[') + "0" + std::string(65, ']');

  auto const merged_col = col_from(same_rows(32, R"({"a":{"v":)" + nested + "}}"));
  EXPECT_THROW(run_fused(cudf::strings_column_view{merged_col},
                         named_depth2({"a", "b", "c", "d", "e", "f", "g", "h"})),
               cudf::logic_error);
}

TEST_F(GetJsonObjectTest, MergedRouteSixteenNamedPathsParity)
{
  // Enter the merged evaluator (>= k_merged_min_paths shallow NAMED paths, rows >= 32) and check
  // every output column against a separate single-path evaluation of the same query. Covers:
  // duplicate keys (first wins), escaped name, shared prefixes, depth-2 paths, nulls.
  auto const rows = same_rows(64,
                              R"({"a":"A","ab":"AB","abc":"ABC","dup":"first","dup":"second",)"
                              R"("key\"q":"escaped","n":null,"plain":"P",)"
                              R"("x":{"y":"XY"},"x":0,"zz":"ZZ"})");

  multi_paths paths;
  paths.reserve(20);
  // 18 depth-1 NAMED paths (>= k_merged_min_paths = 16):
  for (char const* k : {"a",
                        "ab",
                        "abc",
                        "dup",
                        "dup",     // duplicate key: first occurrence wins
                        "key\"q",  // escaped name in the JSON text
                        "n",       // explicit null -> null output
                        "plain",
                        "x",  // duplicate key: first occurrence is the object
                        "zz",
                        "missing1",
                        "missing2",
                        "missing3",
                        "missing4",
                        "missing5",
                        "missing6",
                        "missing7",
                        "missing8"}) {
    paths.push_back(named({k}));
  }
  // 2 depth-2 NAMED paths ($.x.y):
  paths.push_back(named({"x", "y"}));
  paths.push_back(named({"x", "y"}));

  auto const merged_out = run_fused(cudf::strings_column_view{col_from(rows)}, paths);
  ASSERT_EQ(merged_out.size(), paths.size());

  // Oracle: evaluate each path alone (single-path call = the pre-existing per-path engine).
  for (std::size_t col = 0; col < paths.size(); ++col) {
    SCOPED_TRACE("parity column " + std::to_string(col));
    multi_paths one;
    one.push_back(paths[col]);
    auto const single = run_fused(cudf::strings_column_view{col_from(rows)}, one);
    ASSERT_EQ(single.size(), 1u);
    ASSERT_NO_FATAL_FAILURE(expect_columns_parity(merged_out[col], single[0]));
  }

  // Spot-check the documented semantics on the first row.
  ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*merged_out[0], "A", true));      // a
  ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*merged_out[1], "AB", true));     // ab (shared prefix)
  ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*merged_out[2], "ABC", true));    // abc (shared prefix)
  ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*merged_out[3], "first", true));  // dup: first wins
  ASSERT_NO_FATAL_FAILURE(expect_rows_eq(*merged_out[5], "escaped", true));
}

TEST_F(GetJsonObjectTest, NonEmptyNullsRetry)
{
  auto const expanding = R"({"k":{"a":')" + std::string(30, '"') + R"('}})";
  auto const expanded  = quote_bomb_value(30);
  ASSERT_EQ(expanding.size(), 44u);
  ASSERT_EQ(expanded.size(), 68u);
  ASSERT_GT(expanded.size(), expanding.size());  // exceeds the first-pass row slot
  auto const input_col = make_nonempty_nulls({std::string(1'000, ' '),
                                              expanding,
                                              R"({"k":"masked"})",
                                              R"({"k":"sentinel"})",
                                              R"({"k":""})",
                                              "",
                                              ""},
                                             {true, true, false, true, true, true, false});
  auto const expected  = expected_column({"", expanded, "", "sentinel", "", "", ""},
                                         {false, true, false, true, true, false, false});
  auto const missing_expected =
    expected_column(std::vector<std::string>(7), std::vector<bool>(7, false));
  std::vector<std::vector<path_spec>> const paths{{named_path("k")}, {named_path("missing")}};

  for (auto const offset : std::array<cudf::size_type, 2>{0, 1}) {
    SCOPED_TRACE(testing::Message() << "offset=" << offset);
    auto const input =
      cudf::strings_column_view{cudf::slice(*input_col, {offset, input_col->size()}).front()};
    auto const values  = cudf::slice(expected, {offset, input_col->size()}).front();
    auto const missing = cudf::slice(missing_expected, {offset, input_col->size()}).front();
    auto const single  = spark_rapids_jni::get_json_object(input, {named_path("k")});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(values, *single);
    for (auto const budget : std::array<int64_t, 2>{-1, 1}) {
      SCOPED_TRACE(testing::Message() << "budget=" << budget);
      auto const first = spark_rapids_jni::get_json_object_multiple_paths(input, paths, budget, -1);
      auto const last  = spark_rapids_jni::get_json_object_multiple_paths(
        input, paths, budget, -1, spark_rapids_jni::named_field_match_policy::LAST_NON_NULL);
      ASSERT_EQ(first.size(), paths.size());
      ASSERT_EQ(last.size(), paths.size());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(values, *first[0]);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(values, *last[0]);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(missing, *first[1]);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(missing, *last[1]);
    }
  }
}

TEST_F(GetJsonObjectTest, NonEmptyNullsMergedRetry)
{
  auto const expanding = R"({"k":{"a":')" + std::string(30, '"') + R"('},"u":{"v":"U"}})";
  auto const expanded  = quote_bomb_value(30);
  ASSERT_EQ(expanding.size(), 58u);
  ASSERT_GT(expanded.size(), expanding.size());
  std::vector<std::string> rows(33, R"({"k":"sentinel","u":{"v":"U"},"e":""})");
  rows[0]  = std::string(1'000, ' ');
  rows[1]  = expanding;
  rows[2]  = R"({"k":"masked","u":{"v":"masked"},"e":"masked"})";
  rows[3]  = R"({"k":"","u":{"v":""},"e":""})";
  rows[31] = "";
  rows[32] = "";
  std::vector<bool> valid(rows.size(), true);
  valid[2] = valid[32] = false;
  auto const input_col = make_nonempty_nulls(rows, valid);

  std::vector<std::string> k_values(rows.size(), "sentinel");
  std::vector<std::string> u_values(rows.size(), "U");
  auto value_valid = valid;
  for (auto const row : {0, 2, 31, 32}) {
    k_values[row] = u_values[row] = "";
    value_valid[row]              = false;
  }
  k_values[1] = expanded;
  k_values[3] = u_values[3] = "";
  auto empty_valid          = value_valid;
  empty_valid[1]            = false;
  auto const k_expected     = expected_column(k_values, value_valid);
  auto const u_expected     = expected_column(u_values, value_valid);
  auto const empty_expected = expected_column(std::vector<std::string>(rows.size()), empty_valid);
  auto const missing_expected =
    expected_column(std::vector<std::string>(rows.size()), std::vector<bool>(rows.size(), false));

  // Sixteen NAMED paths, a depth-two member and at least 32 rows admit the merged first pass.
  std::vector<std::vector<path_spec>> paths{
    {named_path("k")}, {named_path("u"), named_path("v")}, {named_path("e")}};
  for (int i = 3; i < 16; ++i) {
    paths.push_back({named_path("missing" + std::to_string(i))});
  }
  for (auto const offset : std::array<cudf::size_type, 2>{0, 1}) {
    SCOPED_TRACE(testing::Message() << "offset=" << offset);
    auto const input =
      cudf::strings_column_view{cudf::slice(*input_col, {offset, input_col->size()}).front()};
    std::array const expected{cudf::slice(k_expected, {offset, input_col->size()}).front(),
                              cudf::slice(u_expected, {offset, input_col->size()}).front(),
                              cudf::slice(empty_expected, {offset, input_col->size()}).front(),
                              cudf::slice(missing_expected, {offset, input_col->size()}).front()};
    // Expanding k retries through the general kernel, which must not overwrite merged nulls.
    auto const output = spark_rapids_jni::get_json_object_multiple_paths(input, paths, -1, -1);
    ASSERT_EQ(output.size(), paths.size());
    for (std::size_t i = 0; i < output.size(); ++i) {
      SCOPED_TRACE(testing::Message() << "path=" << i);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected[i < 3 ? i : 3], *output[i]);
    }
  }
}
