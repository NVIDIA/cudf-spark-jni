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

#include "json_utils.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

#include <memory>
#include <string>
#include <vector>

namespace {

[[nodiscard]] std::unique_ptr<cudf::column> make_input(cudf::size_type num_rows,
                                                       cudf::size_type mismatch_percent)
{
  std::string const valid      = R"({"data":{"c2":[{"c3":19,"c4":"x"}],"c1":1},"id":10})";
  std::string const mismatched = R"({"data":{"c2":[19],"c1":2},"id":20})";

  std::vector<std::string> rows;
  rows.reserve(num_rows);
  for (cudf::size_type row = 0; row < num_rows; ++row) {
    rows.push_back(mismatch_percent > 0 && row % 100 < mismatch_percent ? mismatched : valid);
  }
  return cudf::test::strings_column_wrapper(rows.begin(), rows.end()).release();
}

std::vector<std::string> nested_schema_names()
{
  return {"data", "c1", "c2", "element", "c3", "c4", "id"};
}

std::vector<int> nested_schema_num_children() { return {2, 0, 1, 2, 0, 0, 0}; }

std::vector<int> nested_schema_types()
{
  return {static_cast<int>(cudf::type_id::STRUCT),
          static_cast<int>(cudf::type_id::INT32),
          static_cast<int>(cudf::type_id::LIST),
          static_cast<int>(cudf::type_id::STRUCT),
          static_cast<int>(cudf::type_id::INT32),
          static_cast<int>(cudf::type_id::STRING),
          static_cast<int>(cudf::type_id::INT32)};
}

std::vector<int> nested_schema_scales() { return {0, 0, 0, 0, 0, 0, 0}; }

std::vector<int> nested_schema_precisions() { return {-1, -1, -1, -1, -1, -1, -1}; }

}  // namespace

void BM_from_json_to_structs(nvbench::state& state)
{
  auto const num_rows         = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const mismatch_percent = static_cast<cudf::size_type>(state.get_int64("mismatch_percent"));
  auto const input            = make_input(num_rows, mismatch_percent);

  auto const col_names    = nested_schema_names();
  auto const num_children = nested_schema_num_children();
  auto const types        = nested_schema_types();
  auto const scales       = nested_schema_scales();
  auto const precisions   = nested_schema_precisions();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    [[maybe_unused]] auto const output =
      spark_rapids_jni::from_json_to_structs(cudf::strings_column_view{input->view()},
                                             col_names,
                                             num_children,
                                             types,
                                             scales,
                                             precisions,
                                             /*normalize_single_quotes=*/true,
                                             /*allow_leading_zeros=*/true,
                                             /*allow_nonnumeric_numbers=*/true,
                                             /*allow_unquoted_control=*/true,
                                             /*is_us_locale=*/true);
  });

  state.add_buffer_size(num_rows, "rows", "Rows");
}

NVBENCH_BENCH(BM_from_json_to_structs)
  .set_name("from_json_to_structs")
  .add_int64_axis("num_rows", {10'000, 100'000})
  .add_int64_axis("mismatch_percent", {0, 1, 10, 50, 100});
