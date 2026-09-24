/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/json_bench_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <get_json_object.hpp>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using path_instruction_type = spark_rapids_jni::path_instruction_type;
using spark_rapids_jni::json_path;

// With a cell's `size_bytes` and `max_depth` coordinates, these fix every generated row.
constexpr int list_depth   = 2;   // array nesting of the leaf list-of-lists
constexpr int list_width   = 10;  // entries per array level
constexpr int string_width = 10;  // content units per generated string token

// Fixed operating point for the content, query and path-count sweeps. `max_depth == 4` puts the
// leaf list-of-lists at `$.struct.0.0`, so the array instructions have an array to walk. 10 MB
// matches the sibling from_json benchmarks and still launches hundreds of blocks, while keeping
// the per-call scratch buffer small enough to run every path count in parallel.
constexpr std::size_t sweep_size_bytes    = 10'000'000;
constexpr cudf::size_type sweep_max_depth = 4;

// Byte classes the parser branches on, one axis value each, so a regression confined to a single
// class is not averaged away.
enum class content_kind { ASCII, UTF8_2BYTE, UTF8_3BYTE, ASTRAL, ESCAPES, MALFORMED_UTF8 };

// Path-instruction shapes the path evaluator branches on.
enum class query_kind {
  NAMED_LEAF,
  NAMED_SUBTREE,
  WILDCARD,
  INDEX,
  WILDCARD_INDEX,
  INDEX_WILDCARD
};

// Axis values paired with the enumerator each selects. Registration and parsing read the same
// table, so an axis value can never reach a helper with no branch for it.
constexpr std::array<std::pair<std::string_view, content_kind>, 6> content_axis{
  {{"ascii", content_kind::ASCII},
   {"utf8_2byte", content_kind::UTF8_2BYTE},
   {"utf8_3byte", content_kind::UTF8_3BYTE},
   {"astral", content_kind::ASTRAL},
   {"escapes", content_kind::ESCAPES},
   {"malformed_utf8", content_kind::MALFORMED_UTF8}}};

constexpr std::array<std::pair<std::string_view, query_kind>, 6> query_axis{
  {{"named_leaf", query_kind::NAMED_LEAF},
   {"named_subtree", query_kind::NAMED_SUBTREE},
   {"wildcard", query_kind::WILDCARD},
   {"index", query_kind::INDEX},
   {"wildcard_index", query_kind::WILDCARD_INDEX},
   {"index_wildcard", query_kind::INDEX_WILDCARD}}};

template <typename Enum, std::size_t N>
std::vector<std::string> axis_values(std::array<std::pair<std::string_view, Enum>, N> const& table)
{
  std::vector<std::string> values;
  values.reserve(N);
  for (auto const& entry : table) {
    values.emplace_back(entry.first);
  }
  return values;
}

template <typename Enum, std::size_t N>
Enum parse_axis_value(std::array<std::pair<std::string_view, Enum>, N> const& table,
                      std::string const& name)
{
  auto const entry =
    std::ranges::find_if(table, [&name](auto const& candidate) { return candidate.first == name; });
  CUDF_EXPECTS(entry != table.end(), "Unknown axis value: " + name);
  return entry->second;
}

void append_byte(std::string& out, std::uint32_t byte) { out.push_back(static_cast<char>(byte)); }

// Append the UTF-8 encoding of `cp`. Only the multi-byte forms are needed here: ASCII content is
// emitted directly by `append_token`.
void append_code_point(std::string& out, std::uint32_t cp)
{
  if (cp < 0x800) {
    append_byte(out, 0xC0 | (cp >> 6));
    append_byte(out, 0x80 | (cp & 0x3F));
  } else if (cp < 0x0001'0000) {
    append_byte(out, 0xE0 | (cp >> 12));
    append_byte(out, 0x80 | ((cp >> 6) & 0x3F));
    append_byte(out, 0x80 | (cp & 0x3F));
  } else {
    append_byte(out, 0xF0 | (cp >> 18));
    append_byte(out, 0x80 | ((cp >> 12) & 0x3F));
    append_byte(out, 0x80 | ((cp >> 6) & 0x3F));
    append_byte(out, 0x80 | (cp & 0x3F));
  }
}

// Which escape is emitted depends on `position`, not on `value`, so the unit keeps a fixed byte
// length. The \u escapes span U+0041..U+4040, which unescape to 1, 2 and 3 bytes, so every output
// width of the unescape path is exercised. That range stops well short of the U+D800 surrogate
// block, which is not a valid standalone escape.
void append_escape_unit(std::string& out, int position, std::uint64_t value)
{
  static constexpr char hex_digits[] = "0123456789ABCDEF";
  switch (position % 4) {
    case 0: out += "\\\""; break;
    case 1: out += "\\\\"; break;
    case 2: out += "\\n"; break;
    default: {
      auto const cp = 0x0041 + value % 0x4000;
      out += "\\u";
      for (int shift = 12; shift >= 0; shift -= 4) {
        out.push_back(hex_digits[(cp >> shift) & 0xF]);
      }
      break;
    }
  }
}

// Three invalid sequences, again selected by position to keep the length fixed. None of these
// bytes is JSON-structural, so they reach the UTF-8 decoder rather than derailing the tokenizer
// first.
void append_malformed_unit(std::string& out, int position, std::uint64_t value)
{
  switch (position % 3) {
    case 0: append_byte(out, 0x80); break;  // lone continuation byte
    case 1:                                 // overlong encoding of '/'
      append_byte(out, 0xC0);
      append_byte(out, 0xAF);
      break;
    default:  // 3-byte lead truncated by an ASCII byte
      append_byte(out, 0xE4);
      append_byte(out, 0xB8);
      append_byte(out, static_cast<std::uint32_t>('a' + value % 26));
      break;
  }
}

// Append one quoted JSON string of `string_width` content units in the given byte class. Within a
// class each unit has a fixed byte width that never depends on `seed`, so every row comes out the
// same length. `seed` varies the content only.
void append_token(std::string& out, content_kind kind, std::uint64_t seed)
{
  out.push_back('"');
  for (int i = 0; i < string_width; ++i) {
    auto const v = seed + static_cast<std::uint64_t>(i);
    switch (kind) {
      case content_kind::ASCII: out.push_back(static_cast<char>('a' + v % 26)); break;
      // U+00C0, U+4E00 and U+1F600 sit inside the 2-, 3- and 4-byte encoding ranges respectively.
      case content_kind::UTF8_2BYTE:
        append_code_point(out, static_cast<std::uint32_t>(0x00C0 + v % 32));
        break;
      case content_kind::UTF8_3BYTE:
        append_code_point(out, static_cast<std::uint32_t>(0x4E00 + v % 64));
        break;
      case content_kind::ASTRAL:
        append_code_point(out, static_cast<std::uint32_t>(0x0001'F600 + v % 64));
        break;
      case content_kind::ESCAPES: append_escape_unit(out, i, v); break;
      case content_kind::MALFORMED_UTF8: append_malformed_unit(out, i, v); break;
    }
  }
  out.push_back('"');
}

// Append one `depth`-deep array of arrays whose leaves are string tokens.
void append_list(std::string& out, content_kind kind, int depth, std::uint64_t seed)
{
  out.push_back('[');
  for (int i = 0; i < list_width; ++i) {
    if (i > 0) { out.push_back(','); }
    auto const child_seed = seed * list_width + static_cast<std::uint64_t>(i);
    if (depth == 1) {
      append_token(out, kind, child_seed);
    } else {
      append_list(out, kind, depth - 1, child_seed);
    }
  }
  out.push_back(']');
}

// Append one JSON row: an int field, a string field, and a "struct" field nesting `struct_depth`
// single-child objects down to a list-of-lists of strings. The field names -- "int32", "string",
// "struct", and "0" for every struct child -- are emitted here rather than inherited from a JSON
// writer, and they are exactly what the benchmark paths query.
void append_row(std::string& out, std::uint64_t row, int struct_depth, content_kind kind)
{
  // A 7-digit value: the same width for every row, and never a leading zero.
  out += "{\"int32\":";
  out += std::to_string(1'000'000 + row % 9'000'000);
  out += ",\"string\":";
  append_token(out, kind, row);
  out += ",\"struct\":";
  for (int d = 0; d < struct_depth; ++d) {
    out += "{\"0\":";
  }
  append_list(out, kind, list_depth, row);
  out.append(static_cast<std::size_t>(struct_depth), '}');
  out.push_back('}');
}

// Number of `{"0":` levels nested under "struct", floored at one level so "struct" always has a
// child to walk into.
int struct_depth_for(cudf::size_type max_depth) { return std::max(max_depth - list_depth, 1); }

// Build a strings column of JSON rows totalling as close to `size_bytes` characters as a whole
// number of rows allows: every row has the same length, so the row count is the target divided by
// that length, rounded down. Uniform row lengths make a cell reproducible byte-for-byte across
// runs and across builds; the cost is that a regression which only bites when rows differ enough
// in length for a few to straggle behind the rest of their warp stays invisible here.
//
// The rows go straight into one flat character buffer with an offsets array beside it, so even the
// largest cells need only a single host-side copy of the generated data.
std::unique_ptr<cudf::column> generate_input(std::size_t size_bytes,
                                             cudf::size_type max_depth,
                                             content_kind kind = content_kind::ASCII)
{
  auto const struct_depth = struct_depth_for(max_depth);

  std::string row_buf;
  append_row(row_buf, 0, struct_depth, kind);
  auto const row_size = row_buf.size();
  auto const num_rows = std::max<std::size_t>(1, size_bytes / row_size);
  CUDF_EXPECTS(
    num_rows * row_size <= static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
    "Generated input exceeds the strings column offset range.");

  std::string chars;
  chars.reserve(num_rows * row_size);
  std::vector<cudf::size_type> offsets;
  offsets.reserve(num_rows + 1);
  offsets.push_back(0);
  for (std::size_t r = 0; r < num_rows; ++r) {
    row_buf.clear();  // Keeps the capacity, so only the first row ever reallocates.
    append_row(row_buf, r, struct_depth, kind);
    chars += row_buf;
    offsets.push_back(static_cast<cudf::size_type>(chars.size()));
  }

  auto const stream = cudf::get_default_stream();
  auto chars_buffer = rmm::device_buffer(chars.data(), chars.size(), stream);
  // INT32 is the element type of the offsets above, `cudf::size_type`.
  auto offsets_column = std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::INT32},
    static_cast<cudf::size_type>(offsets.size()),
    rmm::device_buffer(offsets.data(), offsets.size() * sizeof(cudf::size_type), stream),
    rmm::device_buffer{},
    0);
  // Both copies above are asynchronous, so the host buffers must stay alive until they land.
  stream.sync();

  return cudf::make_strings_column(static_cast<cudf::size_type>(num_rows),
                                   std::move(offsets_column),
                                   std::move(chars_buffer),
                                   0,
                                   rmm::device_buffer{});
}

// Headline path: `$.struct` plus one `0` step per struct level beyond the first. For
// `max_depth > list_depth` this lands on the leaf list-of-lists; at `max_depth == 2` there is no
// `0` step and the path stops at the enclosing object.
json_path make_struct_path(cudf::size_type max_depth)
{
  json_path path;
  path.emplace_back(path_instruction_type::NAMED, "struct", -1);
  for (int i = 0; i < max_depth - list_depth; ++i) {
    path.emplace_back(path_instruction_type::NAMED, "0", -1);
  }
  return path;
}

// Query shapes for the path-instruction sweep. Everything past the two plain NAMED queries is
// anchored on the leaf list-of-lists, and every subscript stays below `list_width`.
json_path make_query_path(query_kind query, cudf::size_type max_depth)
{
  json_path path;
  switch (query) {
    // A scalar string leaf, which the extractor copies out unescaped.
    case query_kind::NAMED_LEAF:
      path.emplace_back(path_instruction_type::NAMED, "string", -1);
      return path;
    // An object subtree, which the extractor re-serializes, re-escaping every string inside it.
    case query_kind::NAMED_SUBTREE:
      path.emplace_back(path_instruction_type::NAMED, "struct", -1);
      return path;
    default: break;
  }

  path = make_struct_path(max_depth);
  switch (query) {
    case query_kind::WILDCARD:  // `[*]` over the outer array
      path.emplace_back(path_instruction_type::WILDCARD, "", -1);
      break;
    case query_kind::INDEX:  // `[3]` into the outer array
      path.emplace_back(path_instruction_type::INDEX, "", 3);
      break;
    case query_kind::WILDCARD_INDEX:  // `[*][2]`
      path.emplace_back(path_instruction_type::WILDCARD, "", -1);
      path.emplace_back(path_instruction_type::INDEX, "", 2);
      break;
    default:  // `[2][*]`, which takes the fused index-wildcard branch
      path.emplace_back(path_instruction_type::INDEX, "", 2);
      path.emplace_back(path_instruction_type::WILDCARD, "", -1);
      break;
  }
  return path;
}

// One path per lane of the path-count sweep, each selecting a different element of the leaf
// list-of-lists so that no two paths in a batch are identical. `list_width * list_width` distinct
// elements are addressable, which covers every path count the axis registers.
std::vector<json_path> make_distinct_paths(int num_paths, cudf::size_type max_depth)
{
  std::vector<json_path> paths;
  paths.reserve(static_cast<std::size_t>(num_paths));
  for (int i = 0; i < num_paths; ++i) {
    auto path = make_struct_path(max_depth);
    path.emplace_back(path_instruction_type::INDEX, "", i % list_width);
    path.emplace_back(path_instruction_type::INDEX, "", (i / list_width) % list_width);
    paths.push_back(std::move(path));
  }
  return paths;
}

// A path that no longer reaches the generated rows returns all nulls, and a walk that gives up at
// the first field name is far cheaper than a real extraction -- so generator-versus-path drift
// would read as a large speedup rather than as a broken benchmark. Checked once, outside the
// timed region, and only for the ASCII sweeps: a byte class may legitimately nullify every row.
void check_path_matches(cudf::column_view const& output)
{
  CUDF_EXPECTS(output.null_count() < output.size(),
               "Benchmark path matched no row: the query and the generator have drifted apart.");
}

// The nvbench measurement tail every sweep shares.
void measure(nvbench::state& state, cudf::column_view const& json_strings, auto&& run)
{
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_global_memory_reads<nvbench::int8_t>(input_char_bytes(json_strings));
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { [[maybe_unused]] auto const output = run(); });
}

}  // namespace

// HEADLINE: throughput against input size and path depth.
void BM_get_json_object(nvbench::state& state)
{
  auto const size_bytes = static_cast<std::size_t>(state.get_int64("size_bytes"));
  auto const max_depth  = static_cast<cudf::size_type>(state.get_int64("max_depth"));

  auto const json_strings = generate_input(size_bytes, max_depth);
  auto const instructions = make_struct_path(max_depth);
  auto const input_view   = cudf::strings_column_view{json_strings->view()};

  check_path_matches(spark_rapids_jni::get_json_object(input_view, instructions)->view());
  measure(state, json_strings->view(), [&] {
    return spark_rapids_jni::get_json_object(input_view, instructions);
  });
}

// Content sweep: one axis value per byte class, at a fixed size and depth so only the content
// varies. Cells are not comparable to each other -- a class that rejects rows early does less work
// than one that parses them through -- so compare each class against itself across builds.
void BM_get_json_object_content(nvbench::state& state)
{
  auto const kind = parse_axis_value(content_axis, state.get_string("content"));

  auto const json_strings = generate_input(sweep_size_bytes, sweep_max_depth, kind);
  auto const instructions = make_struct_path(sweep_max_depth);
  auto const input_view   = cudf::strings_column_view{json_strings->view()};

  measure(state, json_strings->view(), [&] {
    return spark_rapids_jni::get_json_object(input_view, instructions);
  });
}

// Query sweep: one axis value per path-instruction shape, over identical input. Isolates the cost
// of the path evaluator from the cost of tokenizing the input.
void BM_get_json_object_query(nvbench::state& state)
{
  auto const query = parse_axis_value(query_axis, state.get_string("query"));

  auto const json_strings = generate_input(sweep_size_bytes, sweep_max_depth);
  auto const instructions = make_query_path(query, sweep_max_depth);
  auto const input_view   = cudf::strings_column_view{json_strings->view()};

  check_path_matches(spark_rapids_jni::get_json_object(input_view, instructions)->view());
  measure(state, json_strings->view(), [&] {
    return spark_rapids_jni::get_json_object(input_view, instructions);
  });
}

// Path-count sweep over the multi-path API. The kernel gives each row
// `ceil(num_paths / warp_size) * warp_size` threads, so a single path leaves 31 of every 32 lanes
// idle and 32 paths saturate the warp; this is the only sweep that measures that regime.
void BM_get_json_object_multi_path(nvbench::state& state)
{
  auto const num_paths = static_cast<int>(state.get_int64("num_paths"));

  auto const json_strings = generate_input(sweep_size_bytes, sweep_max_depth);
  auto const paths        = make_distinct_paths(num_paths, sweep_max_depth);
  auto const input_view   = cudf::strings_column_view{json_strings->view()};

  // No memory budget and no parallel override, so every path in the batch runs in one launch --
  // which is the regime this sweep exists to measure.
  auto const run = [&] {
    return spark_rapids_jni::get_json_object_multiple_paths(input_view, paths, -1, -1);
  };
  check_path_matches(run().front()->view());
  measure(state, json_strings->view(), run);
}

NVBENCH_BENCH(BM_get_json_object)
  .set_name("get_json_object")
  .add_int64_axis("size_bytes", {1'000'000, 10'000'000, 100'000'000, 1'000'000'000})
  .add_int64_axis("max_depth", {2, 4, 6, 8});

NVBENCH_BENCH(BM_get_json_object_content)
  .set_name("get_json_object_content")
  .add_string_axis("content", axis_values(content_axis));

NVBENCH_BENCH(BM_get_json_object_query)
  .set_name("get_json_object_query")
  .add_string_axis("query", axis_values(query_axis));

NVBENCH_BENCH(BM_get_json_object_multi_path)
  .set_name("get_json_object_multi_path")
  .add_int64_axis("num_paths", {1, 2, 4, 8, 16, 32});
