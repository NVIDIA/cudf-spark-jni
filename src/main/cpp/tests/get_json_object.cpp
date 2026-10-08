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
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

#include <string>
#include <tuple>
#include <vector>

struct GetJsonObjectTest : public cudf::test::BaseFixture {};

namespace {

using instruction = std::tuple<spark_rapids_jni::path_instruction_type, std::string, int32_t>;

std::unique_ptr<cudf::column> run(std::vector<std::string> const& rows,
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

// A malformed row must null only itself.
constexpr char malformed[] = R"({"a":"b"c"})";

// 0x01 re-escapes to the six-byte \u0001 when copied.
auto control_chars(int n) { return R"({"k":")" + std::string(n, '\x01'); }

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

// The malformed row's partial output overruns its own slot (control characters expand while
// copying, and the error only surfaces at the trailing `x`); the overrun must raise the
// out-of-bound flag instead of polluting the neighboring rows.
TEST_F(GetJsonObjectTest, RetryCorruption_OverrunningMalformedRow)
{
  auto const path =
    std::vector<instruction>{{spark_rapids_jni::path_instruction_type::NAMED, "k", 0}};
  auto const overrun = control_chars(64) + R"("x})";
  for (int rep = 0; rep < 128; ++rep) {
    auto const result   = run({R"({"k":"v1"})", overrun, R"({"k":"v2"})", R"({"k":"v3"})"}, path);
    auto const expected = expected_column({"v1", "", "v2", "v3"}, {true, false, true, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}

// Same overrun, with the error inside the structure copy (unclosed string at end of input):
// the bytes copied before the error must still count toward the output length.
TEST_F(GetJsonObjectTest, RetryCorruption_UnclosedStringOverrun)
{
  auto const path =
    std::vector<instruction>{{spark_rapids_jni::path_instruction_type::NAMED, "k", 0}};
  auto const overrun = control_chars(64);
  for (int rep = 0; rep < 128; ++rep) {
    auto const result   = run({R"({"k":"v1"})", overrun, R"({"k":"v2"})", R"({"k":"v3"})"}, path);
    auto const expected = expected_column({"v1", "", "v2", "v3"}, {true, false, true, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}
