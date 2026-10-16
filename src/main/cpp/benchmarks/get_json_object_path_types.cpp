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

#include "common/gjo_input_cache.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <get_json_object.hpp>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// #define DEBUG_PRINT

namespace {

// get_json_object_path_types: measures get_json_object across JSON path-TYPE mixes (pure
// INDEX / pure WILDCARD / each pair / all three), so every path-shape class the engine routes
// differently gets its own cells. Axis values and the resulting cell count live in the
// NVBENCH_BENCH registration below — single source of truth, never duplicated here.
//
// One array-root row shape serves ALL mixes (hand-rendered: cudf data_factory / the JSON writer
// cannot emit it), with value tokens "s<i>_<zero-padded row index>":
//   [{"a":{"b":"<s0>"},"arr":[{"x":"<s1>"},[1,2]],
//    "m":{"n":["<s2>","<s3>"],"p":"<s4>"}},"<s5>",[7,8,9]]
// Every row has a fixed byte length; size_bytes is not a multiple of it, so the LAST row's
// final token widens by the leftover digits and the column is exactly size_bytes bytes. One
// input per size axis value, generated once per process (shared gjo_input_cache.hpp, row_kind=0).
//
// mix_paths holds the fixed per-mix path lists; num_paths takes the first N and duplicates the
// list cyclically past its end (mirrors production path repeats). Hit = >= 1 non-null result row.
// Per-mix hit composition:
//   mix 0 pure-INDEX       : root[0] (object), root[1] (string), 7/8/9 of [7,8,9]
//   mix 1 pure-WILDCARD    : all root elements; nested [7,8,9] elements
//   mix 2 INDEX + NAMED    : s0 / s1 / s3 / 1 / s4
//   mix 3 NAMED + WILDCARD : s0 / all root elements / s1 / s0 / s2+s3
//   mix 4 INDEX + WILDCARD : all root elements / 7+8+9 / root[2][0] / root[2][2]
//   mix 5 all three        : s0 / s1 / s1 / s2+s3 / all root elements / 1 / s0 / s3
// Hit semantics follow the engine (src/main/cpp/src/get_json_object.cu): WILDCARD matches
// ARRAY children only — a wildcard reaching an object value returns null; a NAMED-led path
// cannot hit the array root, and an INDEX segment cannot descend into an object, so such
// paths are adapted with a `$[0]`/`$[*]` prefix (why mixes 2/3/5 carry INDEX segments);
// `$.a.*` and a missing top-level field can never hit, so they appear as `$[*].a.b` /
// `$[0].m.p`.

using path_instruction_type = spark_rapids_jni::path_instruction_type;
using path_instructions     = std::vector<std::tuple<path_instruction_type, std::string, int32_t>>;

// == Shared generated input (gen once per size, process-global cache) =========================

constexpr std::size_t k_row_index_digits = 10;  // zero-padded row index in every value token
constexpr std::uint64_t k_row_kind       = 0;   // single row shape; scenario slot for the cache key

// Value token "s<slot>_<zero-padded row index>"; extra_digits widens the LAST row's token so
// the column is exactly size_bytes bytes.
std::string value_token(int slot, std::uint64_t row_idx, std::size_t extra_digits = 0)
{
  std::string digits = std::to_string(row_idx);
  digits.insert(0, k_row_index_digits - digits.size(), '0');
  digits.insert(digits.end(), extra_digits, '0');
  return "s" + std::to_string(slot) + "_" + digits;
}

std::string render_row(std::uint64_t row_idx, std::size_t last_token_extra_digits = 0)
{
  std::string row;

  row.reserve(160 + last_token_extra_digits);
  row += R"([{"a":{"b":")";
  row += value_token(0, row_idx);
  row += R"("},"arr":[{"x":")";
  row += value_token(1, row_idx);
  row += R"("},[1,2]],"m":{"n":[")";
  row += value_token(2, row_idx);
  row += R"(",")";
  row += value_token(3, row_idx);
  row += R"("],"p":")";
  row += value_token(4, row_idx);
  row += R"("}},")";
  row += value_token(5, row_idx, last_token_extra_digits);
  row += R"(",[7,8,9]])";
  return row;
}

cudf::column_view cached_input(std::size_t size_bytes)
{
  auto const generate = [size_bytes] {
    std::size_t const row_len  = render_row(0).size();  // fixed-width tokens
    std::uint64_t const rows   = size_bytes / row_len;
    std::size_t const leftover = size_bytes - rows * row_len;  // < row_len
    std::vector<std::string> host_rows;
    host_rows.reserve(rows);
    for (std::uint64_t i = 0; i + 1 < rows; ++i) {
      host_rows.push_back(render_row(i));
    }
    // Last row absorbs the leftover bytes.
    host_rows.push_back(render_row(rows - 1, leftover));
    cudf::test::strings_column_wrapper wrapper(host_rows.begin(), host_rows.end());
    return wrapper.release();
  };
  return spark_rapids_jni::bench::get_or_generate_input(
    spark_rapids_jni::bench::make_input_cache_key(k_row_kind, size_bytes, /*seed=*/0), generate);
}

// == Per-mix fixed path lists =================================================================

std::vector<path_instructions> const& mix_paths(int64_t mix)
{
  static std::array<std::vector<path_instructions>, 6> const mixes = [] {
    auto const named_seg = [](std::string name) {
      return std::make_tuple(path_instruction_type::NAMED, std::move(name), int32_t{-1});
    };
    auto const index_seg = [](int32_t index) {
      return std::make_tuple(path_instruction_type::INDEX, std::string{}, index);
    };
    auto const wild_seg = [] {
      return std::make_tuple(path_instruction_type::WILDCARD, std::string{}, int32_t{-1});
    };

    std::array<std::vector<path_instructions>, 6> mixes;
    // mix 0 — pure INDEX
    mixes[0] = {{index_seg(0)},
                {index_seg(1)},
                {index_seg(2), index_seg(0)},
                {index_seg(2), index_seg(1)},
                {index_seg(2), index_seg(2)}};
    // mix 1 — pure WILDCARD
    mixes[1] = {{wild_seg()}, {wild_seg(), wild_seg()}};
    // mix 2 — INDEX + NAMED
    mixes[2] = {{index_seg(0), named_seg("a"), named_seg("b")},
                {index_seg(0), named_seg("arr"), index_seg(0), named_seg("x")},
                {index_seg(0), named_seg("m"), named_seg("n"), index_seg(1)},
                {index_seg(0), named_seg("arr"), index_seg(1), index_seg(0)},
                {index_seg(0), named_seg("m"), named_seg("p")}};
    // mix 3 — NAMED + WILDCARD
    mixes[3] = {{index_seg(0), named_seg("a"), named_seg("b")},
                {wild_seg()},
                {index_seg(0), named_seg("arr"), wild_seg(), named_seg("x")},
                {wild_seg(), named_seg("a"), named_seg("b")},
                {index_seg(0), named_seg("m"), named_seg("n"), wild_seg()}};
    // mix 4 — INDEX + WILDCARD
    mixes[4] = {{wild_seg()},
                {index_seg(2), wild_seg()},
                {wild_seg(), index_seg(0)},
                {wild_seg(), index_seg(2)}};
    // mix 5 — all three
    mixes[5] = {{index_seg(0), named_seg("a"), named_seg("b")},
                {index_seg(0), named_seg("arr"), index_seg(0), named_seg("x")},
                {index_seg(0), named_seg("arr"), wild_seg(), named_seg("x")},
                {index_seg(0), named_seg("m"), named_seg("n"), wild_seg()},
                {wild_seg()},
                {index_seg(0), named_seg("arr"), index_seg(1), index_seg(0)},
                {wild_seg(), named_seg("a"), named_seg("b")},
                {index_seg(0), named_seg("m"), named_seg("n"), index_seg(1)}};
    return mixes;
  }();
  return mixes[static_cast<std::size_t>(mix)];
}

// Renders a path back to `$[0].a.b` form for pre-flight miss diagnostics.
std::string path_to_string(path_instructions const& path)
{
  std::string out = "$";
  for (auto const& [type, name, index] : path) {
    switch (type) {
      case path_instruction_type::INDEX: out += "[" + std::to_string(index) + "]"; break;
      case path_instruction_type::NAMED: out += "." + name; break;
      case path_instruction_type::WILDCARD: out += "[*]"; break;
    }
  }
  return out;
}

// == Pre-flight hit assertion (mandatory) =====================================================

// Runs once per process: every path must hit >= 1 of the first k_slice_rows rows; a miss throws.
void ensure_pre_flight_once()
{
  static bool done = false;
  if (done) { return; }
  done = true;

  constexpr std::size_t k_slice_rows = 1'000;
  auto const input                   = cached_input(10L * 1024L * 1024L);
  auto const slice =
    cudf::slice(
      input, {0, static_cast<cudf::size_type>(std::min<std::size_t>(k_slice_rows, input.size()))})
      .front();

  std::size_t total_paths = 0;
  for (int64_t mix = 0; mix < 6; ++mix) {
    auto const& paths = mix_paths(mix);
    total_paths += paths.size();
    auto const outputs =
      spark_rapids_jni::get_json_object_multiple_paths(cudf::strings_column_view{slice},
                                                       paths,
                                                       /*memory_budget_bytes=*/0,
                                                       /*parallel_override=*/0);
    CUDF_EXPECTS(outputs.size() == paths.size(),
                 "get_json_object_path_types pre-flight: expected " + std::to_string(paths.size()) +
                   " outputs for mix " + std::to_string(mix) + ", got " +
                   std::to_string(outputs.size()));
    for (std::size_t p = 0; p < outputs.size(); ++p) {
      auto const non_null = outputs[p]->size() - outputs[p]->null_count();
      CUDF_EXPECTS(non_null >= 1,
                   "get_json_object_path_types pre-flight MISS: mix=" + std::to_string(mix) +
                     " path_index=" + std::to_string(p) + " path=" + path_to_string(paths[p]) +
                     " matched 0 of " + std::to_string(outputs[p]->size()) + " rows");
    }
  }
#ifdef DEBUG_PRINT
  std::cout << "get_json_object_path_types pre-flight: all 6 mixes / " << total_paths
            << " paths hit on the first " << slice.size() << " rows" << std::endl;
#endif  // DEBUG_PRINT
}

// =============================================================================================

void BM_get_json_object_path_types(nvbench::state& state)
{
  auto const mix        = state.get_int64("mix");
  auto const num_paths  = state.get_int64("num_paths");
  auto const size_bytes = static_cast<std::size_t>(state.get_int64("size_bytes"));

  ensure_pre_flight_once();  // before the first timed sample
  auto const input = cached_input(size_bytes);

  // First N entries of the fixed list, duplicated cyclically past its end.
  auto const& fixed_paths = mix_paths(mix);
  std::vector<path_instructions const*> cell_paths;
  cell_paths.reserve(num_paths);
  for (int64_t i = 0; i < num_paths; ++i) {
    cell_paths.push_back(&fixed_paths[static_cast<std::size_t>(i) % fixed_paths.size()]);
  }

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    for (auto const* path : cell_paths) {
      [[maybe_unused]] auto const output =
        spark_rapids_jni::get_json_object(cudf::strings_column_view{input}, *path);
    }
  });
  // Every cell_paths entry runs get_json_object once per sample; count all of them so the
  // reported bytes match the timed work.
  state.add_global_memory_reads<nvbench::int8_t>(size_bytes * num_paths);
}

}  // namespace

NVBENCH_BENCH(BM_get_json_object_path_types)
  .set_name("get_json_object_path_types")
  .add_int64_axis("mix", {0, 1, 2, 3, 4, 5})
  .add_int64_axis("num_paths", {1, 8, 32, 64})
  .add_int64_axis("size_bytes", {10L * 1024L * 1024L, 100L * 1024L * 1024L});
