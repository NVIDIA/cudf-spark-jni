/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/generate_input.hpp"

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <nvbench/nvbench.cuh>
#include <row_conversion.hpp>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

// Benchmark grid for the four public row-conversion APIs. Six benchmarks (to/from rows ×
// fixed/wide/strings) cover the Spark plugin's routing keys: the 100-column and 1536-byte
// fixed-width-optimized gates, multi-batch row output, the null-mask-absent branch, and string
// schemas. A fixed-width-optimized cell skips itself when its row exceeds 1536 bytes. Timed
// regions contain exactly the labeled conversion: to-rows cells run one conversion per iteration,
// from-rows cells convert pre-generated row batches back.

namespace {

enum class direction { to_rows, from_rows };

// The fixed-width-optimized kernels stage whole rows in the default 48 KB shared-memory budget
// with at least one 32-thread warp per block, so rows above 48 KB / 32 = 1536 bytes throw.
constexpr int64_t max_fixed_opt_row_bytes = 48 * 1024 / 32;

// 9-type cycle of the pre-rewrite benchmark, kept because the grid's boundary arithmetic
// (~1400-byte rows at 256 columns, the 1536-byte crossing at 320) is derived on it.
std::vector<cudf::type_id> const default_cycle = {cudf::type_id::INT8,
                                                  cudf::type_id::INT32,
                                                  cudf::type_id::INT16,
                                                  cudf::type_id::INT64,
                                                  cudf::type_id::INT32,
                                                  cudf::type_id::BOOL8,
                                                  cudf::type_id::UINT16,
                                                  cudf::type_id::UINT8,
                                                  cudf::type_id::UINT64};

// FNV-1a over the benchmark name and axis values. std::hash is not stable across standard
// libraries; an unstable seed would bench different data per build and inflate A/B gate noise.
constexpr uint64_t fnv1a(std::string_view text, uint64_t hash = 0xcbf29ce484222325ULL)
{
  for (unsigned char const c : text) {
    hash = (hash ^ c) * 0x100000001b3ULL;
  }
  return hash;
}

template <typename... AxisValues>
unsigned deterministic_seed(std::string_view bench_name, AxisValues const&... values)
{
  auto hash       = fnv1a(bench_name);
  auto const fold = [&hash](auto const& value) {
    hash = fnv1a("|", hash);
    if constexpr (std::is_arithmetic_v<std::remove_cvref_t<decltype(value)>>) {
      hash = fnv1a(std::to_string(value), hash);
    } else {
      hash = fnv1a(value, hash);
    }
  };
  (fold(values), ...);
  return static_cast<unsigned>(hash ^ (hash >> 32));
}

std::vector<cudf::data_type> make_schema(std::vector<cudf::type_id> const& types)
{
  std::vector<cudf::data_type> schema;
  schema.reserve(types.size());
  std::ranges::transform(
    types, std::back_inserter(schema), [](cudf::type_id id) { return cudf::data_type{id}; });
  return schema;
}

// Mirrors the JCUDF layout math in row_conversion.cu (compute_fixed_width_layout and
// compute_column_information): each fixed-width column packs at an offset aligned to its own
// element size, a string column packs an 8-byte offset/length pair at 4-byte alignment, and
// ceil(columns / 8) validity bytes follow unaligned.
int64_t jcudf_fixed_and_validity_bytes(std::vector<cudf::data_type> const& schema)
{
  int64_t offset = 0;
  for (auto const& type : schema) {
    bool const is_string = type.id() == cudf::type_id::STRING;
    auto const size      = is_string ? int64_t{8} : static_cast<int64_t>(cudf::size_of(type));
    auto const alignment = is_string ? int64_t{4} : size;
    offset               = (offset + alignment - 1) / alignment * alignment + size;
  }
  return offset + (static_cast<int64_t>(schema.size()) + 7) / 8;
}

// Complete fixed-width JCUDF row: data plus validity, padded to the 8-byte row alignment.
int64_t jcudf_row_size(std::vector<cudf::data_type> const& schema)
{
  return (jcudf_fixed_and_validity_bytes(schema) + 7) / 8 * 8;
}

// Bytes a conversion reads from a realized table: element data, string chars and offsets, and
// whichever null masks are actually present.
int64_t realized_table_bytes(cudf::table_view const& table)
{
  auto const mask_bytes = static_cast<int64_t>((table.num_rows() + 7) / 8);
  int64_t bytes         = 0;
  for (auto const& col : table) {
    if (col.type().id() == cudf::type_id::STRING) {
      auto const strings = cudf::strings_column_view{col};
      bytes += strings.chars_size(cudf::get_default_stream());
      bytes += static_cast<int64_t>(strings.offsets().size()) *
               static_cast<int64_t>(cudf::size_of(strings.offsets().type()));
    } else {
      bytes += static_cast<int64_t>(col.size()) * static_cast<int64_t>(cudf::size_of(col.type()));
    }
    if (col.nullable()) { bytes += mask_bytes; }
  }
  return bytes;
}

// Bytes convert_from_rows* writes when rebuilding this table: it always allocates a null mask
// per output column and rebuilds string offsets as int32, independent of the input's masks.
int64_t reconstructed_table_bytes(cudf::table_view const& table)
{
  auto const num_rows   = static_cast<int64_t>(table.num_rows());
  auto const mask_bytes = (num_rows + 7) / 8;
  int64_t bytes         = 0;
  for (auto const& col : table) {
    if (col.type().id() == cudf::type_id::STRING) {
      auto const strings = cudf::strings_column_view{col};
      bytes += strings.chars_size(cudf::get_default_stream());
      bytes += (num_rows + 1) * static_cast<int64_t>(sizeof(int32_t));
    } else {
      bytes += num_rows * static_cast<int64_t>(cudf::size_of(col.type()));
    }
    bytes += mask_bytes;
  }
  return bytes;
}

int64_t total_row_buffer_bytes(std::vector<std::unique_ptr<cudf::column>> const& row_batches)
{
  int64_t bytes = 0;
  for (auto const& batch : row_batches) {
    bytes += cudf::lists_column_view{batch->view()}.child().size();
  }
  return bytes;
}

void run_fixed_width_bench(nvbench::state& state,
                           direction dir,
                           bool use_fixed_opt,
                           cudf::size_type num_rows,
                           std::vector<cudf::type_id> const& types,
                           std::optional<double> null_probability,
                           unsigned seed)
{
  auto const schema   = make_schema(types);
  auto const row_size = jcudf_row_size(schema);
  if (use_fixed_opt && row_size > max_fixed_opt_row_bytes) {
    state.skip("row size exceeds the fixed-width-optimized 1536-byte limit");
    return;
  }

  // Independent draws per element: the generator's default sample-pool-with-run-lengths mode
  // would repeat 2000 values in runs of ~4, an artifact this grid deliberately avoids.
  data_profile const profile =
    data_profile_builder().cardinality(0).avg_run_length(1).null_probability(null_probability);
  auto table = create_random_table(types, row_count{num_rows}, profile, seed);

  // Decimal columns carry a generated scale, so the from-rows schema must come from the table.
  std::vector<cudf::data_type> realized_schema;
  realized_schema.reserve(types.size());
  for (auto const& col : table->view()) {
    realized_schema.push_back(col.type());
  }

  // Exact for fixed-width schemas: every row occupies the same padded size in every batch.
  auto const row_buffer_bytes = row_size * num_rows;
  state.add_element_count(num_rows, "rows");

  if (dir == direction::to_rows) {
    state.add_global_memory_reads<int8_t>(realized_table_bytes(table->view()));
    state.add_global_memory_writes<int8_t>(row_buffer_bytes);
    // Setup ran on cudf's default stream and the timed region runs on nvbench's; under per-thread
    // default stream nothing orders the two, so drain setup before anything is measured.
    cudf::get_default_stream().synchronize();
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto const stream = rmm::cuda_stream_view{launch.get_stream()};
      auto const row_batches =
        use_fixed_opt
          ? spark_rapids_jni::convert_to_rows_fixed_width_optimized(table->view(), stream)
          : spark_rapids_jni::convert_to_rows(table->view(), stream);
    });
  } else {
    auto const row_batches =
      use_fixed_opt ? spark_rapids_jni::convert_to_rows_fixed_width_optimized(table->view())
                    : spark_rapids_jni::convert_to_rows(table->view());
    // This setup conversion also runs on the default stream; drain it before the timed region.
    cudf::get_default_stream().synchronize();
    state.add_global_memory_reads<int8_t>(row_buffer_bytes);
    state.add_global_memory_writes<int8_t>(reconstructed_table_bytes(table->view()));
    table.reset();  // the timed loop needs only the row batches and the schema
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto const stream = rmm::cuda_stream_view{launch.get_stream()};
      for (auto const& batch : row_batches) {
        cudf::lists_column_view const list{batch->view()};
        auto const out = use_fixed_opt
                           ? spark_rapids_jni::convert_from_rows_fixed_width_optimized(
                               list, realized_schema, stream)
                           : spark_rapids_jni::convert_from_rows(list, realized_schema, stream);
      }
    });
  }
}

void run_strings_bench(nvbench::state& state, direction dir, std::string_view bench_name)
{
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const avg_string_len = state.get_int64("avg_string_len");
  auto const string_cols    = static_cast<cudf::size_type>(state.get_int64("string_cols"));
  auto const seed           = deterministic_seed(bench_name, num_rows, avg_string_len, string_cols);

  // Eight fixed-width columns (leading types of the default cycle) followed by string columns.
  std::vector<cudf::type_id> types(default_cycle.begin(), default_cycle.begin() + 8);
  types.insert(types.end(), string_cols, cudf::type_id::STRING);
  auto const schema = make_schema(types);

  // Independent draws per element (see run_fixed_width_bench); for strings a repeated sample
  // pool would additionally make the char-copy kernels artificially cache-hot.
  data_profile const profile =
    data_profile_builder().cardinality(0).avg_run_length(1).null_probability(0.1).distribution(
      cudf::type_id::STRING, distribution_id::NORMAL, int64_t{0}, 2 * avg_string_len);
  auto table = create_random_table(types, row_count{num_rows}, profile, seed);

  std::vector<cudf::data_type> realized_schema;
  realized_schema.reserve(types.size());
  for (auto const& col : table->view()) {
    realized_schema.push_back(col.type());
  }

  // One up-front conversion supplies the exact row-buffer size (per-row string payloads make it
  // data-dependent) and, for from-rows, the timed input batches.
  auto row_batches            = spark_rapids_jni::convert_to_rows(table->view());
  auto const row_buffer_bytes = total_row_buffer_bytes(row_batches);
  state.add_element_count(num_rows, "rows");

  // Setup ran on cudf's default stream and the timed regions run on nvbench's; under per-thread
  // default stream nothing orders the two, so drain setup before anything is measured.
  cudf::get_default_stream().synchronize();

  if (dir == direction::to_rows) {
    state.add_global_memory_reads<int8_t>(realized_table_bytes(table->view()));
    state.add_global_memory_writes<int8_t>(row_buffer_bytes);
    row_batches.clear();  // keep each iteration's transient row buffer as the only live copy
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto const stream = rmm::cuda_stream_view{launch.get_stream()};
      auto const out    = spark_rapids_jni::convert_to_rows(table->view(), stream);
    });
  } else {
    state.add_global_memory_reads<int8_t>(row_buffer_bytes);
    state.add_global_memory_writes<int8_t>(reconstructed_table_bytes(table->view()));
    table.reset();  // the timed loop needs only the row batches and the schema
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto const stream = rmm::cuda_stream_view{launch.get_stream()};
      for (auto const& batch : row_batches) {
        cudf::lists_column_view const list{batch->view()};
        auto const out = spark_rapids_jni::convert_from_rows(list, realized_schema, stream);
      }
    });
  }
}

}  // namespace

static void to_rows_fixed(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const path        = state.get_string("path");
  run_fixed_width_bench(state,
                        direction::to_rows,
                        path == "fixed_opt",
                        num_rows,
                        cycle_dtypes(default_cycle, num_columns),
                        0.1,
                        deterministic_seed("to_rows_fixed", num_rows, num_columns, path));
}

static void from_rows_fixed(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const path        = state.get_string("path");
  run_fixed_width_bench(state,
                        direction::from_rows,
                        path == "fixed_opt",
                        num_rows,
                        cycle_dtypes(default_cycle, num_columns),
                        0.1,
                        deterministic_seed("from_rows_fixed", num_rows, num_columns, path));
}

static void to_rows_wide(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const nulls       = state.get_string("nulls");
  auto const null_probability =
    nulls == "none" ? std::optional<double>{} : std::optional<double>{std::stod(nulls)};
  run_fixed_width_bench(state,
                        direction::to_rows,
                        false,
                        num_rows,
                        cycle_dtypes(default_cycle, num_columns),
                        null_probability,
                        deterministic_seed("to_rows_wide", num_rows, num_columns, nulls));
}

static void from_rows_wide(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const nulls       = state.get_string("nulls");
  auto const null_probability =
    nulls == "none" ? std::optional<double>{} : std::optional<double>{std::stod(nulls)};
  run_fixed_width_bench(state,
                        direction::from_rows,
                        false,
                        num_rows,
                        cycle_dtypes(default_cycle, num_columns),
                        null_probability,
                        deterministic_seed("from_rows_wide", num_rows, num_columns, nulls));
}

static void to_rows_strings(nvbench::state& state)
{
  run_strings_bench(state, direction::to_rows, "to_rows_strings");
}

static void from_rows_strings(nvbench::state& state)
{
  run_strings_bench(state, direction::from_rows, "from_rows_strings");
}

NVBENCH_BENCH(to_rows_fixed)
  .add_int64_axis("num_rows", {32'768, 262'144, 2'097'152, 16'777'216})
  .add_int64_axis("num_columns", {2, 10, 96, 128, 212, 256})
  .add_string_axis("path", {"general", "fixed_opt"});

NVBENCH_BENCH(from_rows_fixed)
  .add_int64_axis("num_rows", {32'768, 262'144, 2'097'152, 16'777'216})
  .add_int64_axis("num_columns", {2, 10, 96, 128, 212, 256})
  .add_string_axis("path", {"general", "fixed_opt"});

NVBENCH_BENCH(to_rows_wide)
  .add_int64_axis("num_rows", {32'768, 262'144, 2'097'152, 16'777'216})
  .add_int64_axis("num_columns", {212, 320})
  .add_string_axis("nulls", {"none", "0.1"});

NVBENCH_BENCH(from_rows_wide)
  .add_int64_axis("num_rows", {32'768, 262'144, 2'097'152, 16'777'216})
  .add_int64_axis("num_columns", {212, 320})
  .add_string_axis("nulls", {"none", "0.1"});

NVBENCH_BENCH(to_rows_strings)
  .add_int64_axis("num_rows", {32'768, 262'144, 2'097'152, 16'777'216})
  .add_int64_axis("avg_string_len", {16, 128})
  .add_int64_axis("string_cols", {2, 8});

NVBENCH_BENCH(from_rows_strings)
  .add_int64_axis("num_rows", {32'768, 262'144, 2'097'152, 16'777'216})
  .add_int64_axis("avg_string_len", {16, 128})
  .add_int64_axis("string_cols", {2, 8});
