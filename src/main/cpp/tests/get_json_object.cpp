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

}  // namespace

// A control character expands six-fold when re-escaped (0x01 -> \u0001), forcing the
// out-of-bound retry. Each malformed row then races the victim row in the retry launch;
// interleaved pairs spread the race across warps and blocks.
TEST_F(GetJsonObjectTest, RetryDoesNotCorruptFollowingRows)
{
  auto const path = std::vector<instruction>{};
  auto const expand =
    R"({"a":")" + std::string(32, '\x01') + R"("})";  // output escapes each byte to \u0001
  std::string escapes;
  for (int i = 0; i < 32; ++i) {
    escapes += "\\u0001";
  }
  auto const expand_out = R"({"a":")" + escapes + R"("})";

  std::vector<std::string> rows{expand};
  std::vector<std::string> values{expand_out};
  std::vector<bool> valid{true};
  for (int i = 0; i < 64; ++i) {
    rows.emplace_back(malformed);
    values.emplace_back("");
    valid.emplace_back(false);
    rows.emplace_back(R"({"big":")" + std::string(64, 'v') + R"("})");
    values.emplace_back(rows.back());
    valid.emplace_back(true);
  }

  for (int rep = 0; rep < 16; ++rep) {
    auto const result   = run(rows, path);
    auto const expected = expected_column(values, valid);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}

// Same race with a named path: the victim row's extracted value must survive intact.
TEST_F(GetJsonObjectTest, RetryDoesNotCorruptNamedPathValue)
{
  auto const path =
    std::vector<instruction>{{spark_rapids_jni::path_instruction_type::NAMED, "big", 0}};
  for (int rep = 0; rep < 128; ++rep) {
    auto const result   = run({R"({"a":"\n"})", malformed, R"({"big": ""})"}, path);
    auto const expected = expected_column({"", "", ""}, {false, false, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}

// Without an out-of-bound retry, a malformed row still nulls only itself.
TEST_F(GetJsonObjectTest, MalformedRowOnlyNullsItself)
{
  auto const path =
    std::vector<instruction>{{spark_rapids_jni::path_instruction_type::NAMED, "k", 0}};
  auto const result   = run({R"({"k":"v1"})", malformed, R"({"k":"v2"})"}, path);
  auto const expected = expected_column({"v1", "", "v2"}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}
