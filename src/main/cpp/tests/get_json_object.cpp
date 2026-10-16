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
#include <cudf_test/debug_utilities.hpp>

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

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
