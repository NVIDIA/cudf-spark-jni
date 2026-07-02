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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <cstddef>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

struct FromJsonTest : public cudf::test::BaseFixture {};

namespace {

// One (key, value) entry of a raw-map row, in the exact textual content the raw-map engine emits.
// The value is stored WITHOUT surrounding quotes for a JSON string value, matching
// `from_json_to_raw_map`'s `include_quote_char=false` extraction.
using kv = std::pair<std::string, std::string>;

// Build the expected `LIST<STRUCT<STRING,STRING>>` raw-map column directly from host data.
//
// `rows[r]` is the ordered list of (key, value) pairs that row `r` must contain (already in the
// exact byte content the engine emits: keys verbatim, string values with their surrounding quotes
// stripped). `row_valid[r] == false` makes row `r` a NULL list row (its pairs, if any, are ignored
// and contribute nothing to the children). The keys child and values child are flat STRING columns
// concatenated in row-major order; the structs child wraps them; the list offsets are the running
// per-row pair counts of the valid rows; the row null mask comes from `row_valid`.
std::unique_ptr<cudf::column> make_expected_raw_map(std::vector<std::vector<kv>> const& rows,
                                                    std::vector<bool> const& row_valid)
{
  CUDF_EXPECTS(rows.size() == row_valid.size(),
               "make_expected_raw_map: rows and row_valid size mismatch");

  auto const num_rows = rows.size();

  std::vector<std::string> flat_keys;
  std::vector<std::string> flat_values;
  std::vector<cudf::size_type> offsets{0};
  offsets.reserve(num_rows + 1);

  cudf::size_type running = 0;
  for (std::size_t r = 0; r < num_rows; ++r) {
    // A null row contributes no children entries (the list slot is masked out).
    if (row_valid[r]) {
      for (auto const& [k, v] : rows[r]) {
        flat_keys.push_back(k);
        flat_values.push_back(v);
        ++running;
      }
    }
    offsets.push_back(running);
  }

  auto keys_child    = cudf::test::strings_column_wrapper(flat_keys.begin(), flat_keys.end());
  auto values_child  = cudf::test::strings_column_wrapper(flat_values.begin(), flat_values.end());
  auto structs_child = cudf::test::structs_column_wrapper{{keys_child, values_child}}.release();

  auto offsets_col =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(offsets.begin(), offsets.end())
      .release();

  auto [null_mask, null_count] = cudf::bools_to_mask(
    cudf::test::fixed_width_column_wrapper<bool>(row_valid.begin(), row_valid.end()));

  return cudf::make_lists_column(static_cast<cudf::size_type>(num_rows),
                                 std::move(offsets_col),
                                 std::move(structs_child),
                                 null_count,
                                 null_count > 0 ? std::move(*null_mask) : rmm::device_buffer{});
}

// Convenience: every row valid.
std::vector<bool> all_valid(std::size_t n) { return std::vector<bool>(n, true); }

// Default leniency options shared by all calls and the single source of truth for the defaults.
// Individual tests override only the field a case exercises (designated initializers below).
constexpr spark_rapids_jni::json_parse_options default_options{
  .normalize_single_quotes  = false,
  .allow_leading_zeros      = true,
  .allow_nonnumeric_numbers = true,
  .allow_unquoted_control   = false,
};

// Thin wrapper to invoke the `Map[String,String]` engine-under-test with the shared default flags.
std::unique_ptr<cudf::column> raw_map(cudf::strings_column_view const& input)
{
  return spark_rapids_jni::from_json_to_raw_map(input, default_options);
}

// Thin wrapper to invoke the `Map[String,Array[String]]` engine-under-test with the shared flags.
std::unique_ptr<cudf::column> raw_map_array(cudf::strings_column_view const& input)
{
  return spark_rapids_jni::from_json_to_raw_map_array_values(input, default_options);
}

}  // namespace

// ===========================================================================
// RawMapOpt_* : pin the exact LIST<STRUCT<STRING,STRING>> output of `from_json_to_raw_map`.
// Raw-map value semantics:
//   * String VALUES/KEYS are emitted WITHOUT surrounding quotes (`include_quote_char=false`).
//   * The extractor is a verbatim byte-range copy; it performs NO JSON unescaping.
//   * A populated object yields a non-null list of its (key, value) pairs in INPUT TEXTUAL order;
//     a duplicate key is emitted once PER OCCURRENCE (no collapsing).
//   * An empty object `{}` yields an EMPTY, NON-NULL list row.
//   * A true-null / empty / whitespace-only / invalid row yields a NULL list row.
// ===========================================================================

// Single key, single row.
TEST_F(FromJsonTest, RawMapOpt_SingleKeySingleRow)
{
  auto const input_col = cudf::test::strings_column_wrapper{R"({"k":"v"})"};
  auto const input     = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map({{{"k", "v"}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map(input), *expected);
}

// Many keys (8, 32, 64), pure-ASCII string values, consistent key order across rows.
TEST_F(FromJsonTest, RawMapOpt_ManyKeys)
{
  for (int const num_keys : {8, 32, 64}) {
    SCOPED_TRACE(std::format("num_keys={}", num_keys));
    constexpr int num_rows = 3;
    std::vector<std::string> input_rows;
    std::vector<std::vector<kv>> expected_rows;
    input_rows.reserve(num_rows);
    expected_rows.reserve(num_rows);

    for (int r = 0; r < num_rows; ++r) {
      std::string obj = "{";
      std::vector<kv> pairs;
      pairs.reserve(static_cast<std::size_t>(num_keys));
      for (int k = 0; k < num_keys; ++k) {
        auto const key = std::format("k{}", k);
        auto const val = std::format("v{}_{}", r, k);
        if (k > 0) { obj += ","; }
        obj += std::format(R"("{}":"{}")", key, val);
        pairs.emplace_back(key, val);
      }
      obj += "}";
      input_rows.push_back(std::move(obj));
      expected_rows.push_back(std::move(pairs));
    }

    auto const input_col = cudf::test::strings_column_wrapper(input_rows.begin(), input_rows.end());
    auto const input     = cudf::strings_column_view{input_col};

    auto const expected = make_expected_raw_map(expected_rows, all_valid(num_rows));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map(input), *expected);
  }
}

// Empty-string values `{"k":""}` -> empty (not null) string value.
TEST_F(FromJsonTest, RawMapOpt_EmptyStringValues)
{
  auto const input_col = cudf::test::strings_column_wrapper{R"({"k":""})", R"({"a":"","b":"x"})"};
  auto const input     = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map({{{"k", ""}}, {{"a", ""}, {"b", "x"}}}, all_valid(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map(input), *expected);
}

// Structural characters inside a quoted string value survive verbatim (only outer quotes stripped).
TEST_F(FromJsonTest, RawMapOpt_StructuralCharValues)
{
  auto const input_col = cudf::test::strings_column_wrapper{
    R"({"a":"{}","b":"[]","c":",:"})", R"({"d":"{}[], <=semantic-symbols-string"})"};
  auto const input = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map(
    {{{"a", "{}"}, {"b", "[]"}, {"c", ",:"}}, {{"d", "{}[], <=semantic-symbols-string"}}},
    all_valid(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map(input), *expected);
}

// UTF-8 / emoji in keys and values pass through verbatim; `704` is the unquoted literal text "704".
TEST_F(FromJsonTest, RawMapOpt_Utf8AndEmoji)
{
  auto const input_col = cudf::test::strings_column_wrapper{
    "{\"Zipcóde\":704,\"ZípCodeTypé\":\"\U00029E3D\",\"City\":\"\U0001F3F3\"}"};
  auto const input = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map(
    {{{"Zipcóde", "704"}, {"ZípCodeTypé", "\U00029E3D"}, {"City", "\U0001F3F3"}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map(input), *expected);
}

// Long values (> 1KB), exercising the chars-buffer sizing / offsets path.
TEST_F(FromJsonTest, RawMapOpt_LongValues)
{
  std::string const long_a(1500, 'A');
  std::string const long_b(2048, 'B');
  auto const input_col = cudf::test::strings_column_wrapper{
    "{\"k\":\"" + long_a + "\"}", "{\"k\":\"" + long_b + "\",\"s\":\"tail\"}", R"({"k":"short"})"};
  auto const input = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map(
    {{{"k", long_a}}, {{"k", long_b}, {"s", "tail"}}, {{"k", "short"}}}, all_valid(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map(input), *expected);
}

// Heterogeneous per-row key sets: each row emits its own keys in textual order.
TEST_F(FromJsonTest, RawMapOpt_HeterogeneousKeySets)
{
  auto const input_col =
    cudf::test::strings_column_wrapper{R"({"k1":"a","k2":"b"})", R"({"k2":"c","k3":"d"})"};
  auto const input = cudf::strings_column_view{input_col};

  auto const expected =
    make_expected_raw_map({{{"k1", "a"}, {"k2", "b"}}, {{"k2", "c"}, {"k3", "d"}}}, all_valid(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map(input), *expected);
}

// True-null input rows yield a NULL list row; surrounding populated rows are unaffected.
TEST_F(FromJsonTest, RawMapOpt_NullInputRows)
{
  auto const input_col = cudf::test::strings_column_wrapper(
    {R"({"k":"a"})", R"(unused)", R"({"k":"c"})"}, {true, false, true});
  auto const input = cudf::strings_column_view{input_col};

  auto const expected =
    make_expected_raw_map({{{"k", "a"}}, {}, {{"k", "c"}}}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map(input), *expected);
}

// Empty object `{}` between populated rows -> EMPTY, NON-NULL list row.
TEST_F(FromJsonTest, RawMapOpt_EmptyObjectRow)
{
  auto const input_col =
    cudf::test::strings_column_wrapper{R"({"a":"1","b":"2"})", R"({})", R"({"c":"3"})"};
  auto const input = cudf::strings_column_view{input_col};

  auto const expected =
    make_expected_raw_map({{{"a", "1"}, {"b", "2"}}, {}, {{"c", "3"}}}, {true, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map(input), *expected);
}

// ---------------------------------------------------------------------------
// Edge-case tests: each pins the exact output. The shape helper below is retained for the large
// stress test, where building a 50k-row expected column would dominate the test.
// ---------------------------------------------------------------------------

namespace {

// Assert the column is LIST<STRUCT<STRING,STRING>> with `num_rows` rows. Uses fatal `ASSERT_*`, so
// a mismatch returns from this function (guarding the subsequent child inspection); wrap calls in
// `ASSERT_NO_FATAL_FAILURE(...)` so the caller stops too.
void expect_raw_map_shape(cudf::column_view const& col, cudf::size_type num_rows)
{
  ASSERT_EQ(col.type().id(), cudf::type_id::LIST);
  ASSERT_EQ(col.size(), num_rows);

  auto const lcv          = cudf::lists_column_view{col};
  auto const structs_view = lcv.child();
  ASSERT_EQ(structs_view.type().id(), cudf::type_id::STRUCT);
  ASSERT_EQ(structs_view.num_children(), 2);

  auto const scv = cudf::structs_column_view{structs_view};
  EXPECT_EQ(scv.child(0).type().id(), cudf::type_id::STRING);
  EXPECT_EQ(scv.child(1).type().id(), cudf::type_id::STRING);
}

}  // namespace

// JSON `null` literal value `{"k":null}` -- the map path keeps the literal text "null" as the value
// (no JSON-null handling: the value node range is copied verbatim). The whole row stays valid.
TEST_F(FromJsonTest, RawMapOpt_Char_NullLiteralValue)
{
  auto const input_col =
    cudf::test::strings_column_wrapper{R"({"a":"x"})", R"({"k":null})", R"({"b":"y"})"};
  auto const input = cudf::strings_column_view{input_col};

  auto const expected =
    make_expected_raw_map({{{"a", "x"}}, {{"k", "null"}}, {{"b", "y"}}}, all_valid(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map(input), *expected);
}

// Escaped quote `\"` and escaped backslash `\\` inside a quoted string value survive literally
// (the extractor copies the value's byte range verbatim; it performs no JSON unescaping).
TEST_F(FromJsonTest, RawMapOpt_Char_EscapedValues)
{
  auto const input_col = cudf::test::strings_column_wrapper{R"({"d":"x\"y","e":"p\\q"})"};
  auto const input     = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map({{{"d", R"(x\"y)"}, {"e", R"(p\\q)"}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map(input), *expected);
}

// Malformed / invalid rows mixed with valid rows -> invalid rows become NULL list rows; the valid
// rows contribute their pairs unchanged. The null rows' pairs are ignored (passed as `{}`).
TEST_F(FromJsonTest, RawMapOpt_Char_MalformedRows)
{
  auto const input_col = cudf::test::strings_column_wrapper{
    R"({"a":"1"})", R"(not json at all)", R"({"b":"2"})", R"(@@@)"};
  auto const input = cudf::strings_column_view{input_col};

  auto const expected =
    make_expected_raw_map({{{"a", "1"}}, {}, {{"b", "2"}}, {}}, {true, false, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map(input), *expected);
}

// Duplicate keys `{"a":"1","a":"2"}` emit one pair per occurrence in textual order (no collapsing).
TEST_F(FromJsonTest, RawMapOpt_Char_DuplicateKeys)
{
  auto const input_col = cudf::test::strings_column_wrapper{R"({"a":"1","a":"2"})"};
  auto const input     = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map({{{"a", "1"}, {"a", "2"}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map(input), *expected);
}

// Whitespace-only row between valid rows -> NULL list row; the surrounding rows are unaffected.
TEST_F(FromJsonTest, RawMapOpt_Char_WhitespaceOnlyRow)
{
  auto const input_col = cudf::test::strings_column_wrapper{R"({"a":"1"})", "   ", R"({"b":"2"})"};
  auto const input     = cudf::strings_column_view{input_col};

  auto const expected =
    make_expected_raw_map({{{"a", "1"}}, {}, {{"b", "2"}}}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map(input), *expected);
}

// Stress: 50k rows x 32 keys with interspersed NULL-input and malformed rows. Asserts the
// structural invariants and the row-validity contract.
TEST_F(FromJsonTest, RawMapOpt_StressInvariantsWithBadRows)
{
  constexpr int num_rows = 50000;
  constexpr int num_keys = 32;

  std::vector<int> const null_rows{100, 12345, 49999};
  std::vector<int> const bad_rows{0, 7, 25000, 49998};

  auto const is_in = [](std::vector<int> const& v, int x) {
    return std::ranges::find(v, x) != v.end();
  };

  std::vector<std::string> rows;
  std::vector<bool> validity;
  rows.reserve(num_rows);
  validity.reserve(num_rows);

  for (int r = 0; r < num_rows; ++r) {
    if (is_in(null_rows, r)) {
      rows.emplace_back("unused-null");
      validity.push_back(false);
      continue;
    }
    if (is_in(bad_rows, r)) {
      rows.emplace_back("not-an-object");
      validity.push_back(true);
      continue;
    }
    std::string obj = "{";
    for (int k = 0; k < num_keys; ++k) {
      if (k > 0) { obj += ","; }
      obj += std::format(R"("key{0}":"r{1}k{0}")", k, r);
    }
    obj += "}";
    rows.push_back(std::move(obj));
    validity.push_back(true);
  }

  auto const input_col =
    cudf::test::strings_column_wrapper(rows.begin(), rows.end(), validity.begin());
  auto const input  = cudf::strings_column_view{input_col};
  auto const result = raw_map(input);

  ASSERT_NO_FATAL_FAILURE(expect_raw_map_shape(result->view(), num_rows));
  EXPECT_EQ(result->view().null_count(),
            static_cast<cudf::size_type>(null_rows.size() + bad_rows.size()));
}

// Empty input for the map function exercises the `make_empty_map` zero-row fast path.
TEST_F(FromJsonTest, RawMapOpt_EmptyInput)
{
  auto const input_col = cudf::test::strings_column_wrapper{};
  auto const input     = cudf::strings_column_view{input_col};

  auto const result = raw_map(input);
  EXPECT_EQ(result->size(), 0);
  ASSERT_EQ(result->type().id(), cudf::type_id::LIST);

  auto const lcv          = cudf::lists_column_view{result->view()};
  auto const structs_view = lcv.child();
  ASSERT_EQ(structs_view.type().id(), cudf::type_id::STRUCT);
  ASSERT_EQ(structs_view.num_children(), 2);
  auto const scv = cudf::structs_column_view{structs_view};
  EXPECT_EQ(scv.child(0).type().id(), cudf::type_id::STRING);  // key.
  EXPECT_EQ(scv.child(1).type().id(), cudf::type_id::STRING);  // value.
}

// allow_unquoted_control=true: a literal control char (tab) inside a quoted value is accepted and
// copied verbatim into the value bytes.
TEST_F(FromJsonTest, RawMapOpt_UnquotedControl)
{
  auto const input_col = cudf::test::strings_column_wrapper{"{\"k\":\"a\tb\"}"};
  auto const input     = cudf::strings_column_view{input_col};

  auto const result = spark_rapids_jni::from_json_to_raw_map(
    input,
    spark_rapids_jni::json_parse_options{default_options.normalize_single_quotes,
                                         default_options.allow_leading_zeros,
                                         default_options.allow_nonnumeric_numbers,
                                         /*allow_unquoted_control=*/true});

  auto const expected = make_expected_raw_map({{{"k", "a\tb"}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

// Sliced (nonzero-offset) input: slice off row 0 of a 3-row column and verify the 2-row result.
TEST_F(FromJsonTest, RawMapOpt_SlicedInput)
{
  auto const input_col =
    cudf::test::strings_column_wrapper{R"({"skip":"me"})", R"({"k":"a"})", R"({"k":"b"})"};
  auto const sliced = cudf::slice(input_col, {1, 3})[0];
  auto const input  = cudf::strings_column_view{sliced};

  auto const expected = make_expected_raw_map({{{"k", "a"}}, {{"k", "b"}}}, all_valid(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map(input), *expected);
}

// Null-mask write path around warp word boundaries (< 32, == 32, > 32, > 64 rows). The first and
// last rows are null in each size so the boundary word is exercised.
TEST_F(FromJsonTest, RawMapOpt_WarpBoundaryNullMask)
{
  for (int const num_rows : {31, 32, 33, 65}) {
    SCOPED_TRACE(std::format("num_rows={}", num_rows));
    std::vector<std::string> rows(static_cast<std::size_t>(num_rows), R"({"k":"v"})");
    std::vector<bool> validity(static_cast<std::size_t>(num_rows), true);
    validity[0]                                      = false;
    validity[static_cast<std::size_t>(num_rows - 1)] = false;

    auto const input_col =
      cudf::test::strings_column_wrapper(rows.begin(), rows.end(), validity.begin());
    auto const result = raw_map(cudf::strings_column_view{input_col});
    EXPECT_EQ(result->null_count(), 2);
  }
}

// ===========================================================================
// Array path: `from_json_to_raw_map_array_values` -> LIST<STRUCT<STRING, LIST<STRING>>>.
// Each test asserts exact output via `make_expected_raw_map_array`. Element/value semantics:
//   * value = JSON array of strings -> non-null inner List<String> of de-quoted elements.
//   * value = `[]` -> empty (non-null) inner list.
//   * value = JSON `null` literal -> row KEPT, NULL inner list (mask #1).
//   * value = scalar / object (present, non-null, NOT an array) -> HARD TYPE MISMATCH: the WHOLE
//     row is nullified (Spark row-level bad-record semantics), even if other keys are valid arrays.
//   * element = string -> de-quoted bytes; number/bool/nested -> raw JSON substring.
//   * element = literal `null` -> NULL element (mask #2); a JSON STRING "null" stays valid.
// ===========================================================================

namespace {

// One (key, value) entry of a map-of-array row. The inner optional models inner-list nullability
// (std::nullopt = null inner list, mask #1); the innermost optional models per-element nullability
// (std::nullopt = null element, mask #2). An empty (present) vector is an empty non-null inner
// list.
using array_value = std::optional<std::vector<std::optional<std::string>>>;
using kva         = std::pair<std::string, array_value>;

// Build the expected `LIST<STRUCT<STRING, LIST<STRING>>>` column directly from host data, mirroring
// `make_expected_raw_map` but for both list levels.
//
// Flattening: all elements across all pairs in all valid rows are concatenated into one STRING
// child built with the null-AWARE `strings_column_wrapper(begin, end, validity)`. The inner offsets
// (length num_pairs+1) give each pair's element span; a null inner list AND an empty inner list
// both produce a 0-length span, distinguished only by the inner null mask (from each pair's
// `array_value.has_value()`). The keys child has no nulls. The outer offsets give per-row pair
// counts; the outer null mask comes from `row_valid`.
std::unique_ptr<cudf::column> make_expected_raw_map_array(std::vector<std::vector<kva>> const& rows,
                                                          std::vector<bool> const& row_valid)
{
  CUDF_EXPECTS(rows.size() == row_valid.size(),
               "make_expected_raw_map_array: rows and row_valid size mismatch");

  auto const num_rows = rows.size();

  std::vector<std::string> flat_keys;
  std::vector<std::string> flat_elements;      // String child content (placeholder for null elems).
  std::vector<bool> flat_element_valid;        // Mask #2 per flattened element.
  std::vector<bool> inner_valid;               // Mask #1 per (key,value) pair.
  std::vector<cudf::size_type> inner_offsets;  // length num_pairs + 1.
  std::vector<cudf::size_type> outer_offsets;  // length num_rows + 1.
  outer_offsets.reserve(num_rows + 1);
  // inner_offsets holds one entry per (key,value) pair plus the leading 0.
  std::size_t total_pairs = 0;
  for (auto const& row : rows) {
    total_pairs += row.size();
  }
  inner_offsets.reserve(total_pairs + 1);

  inner_offsets.push_back(0);
  outer_offsets.push_back(0);

  cudf::size_type pair_running = 0;
  cudf::size_type elem_running = 0;
  for (std::size_t r = 0; r < num_rows; ++r) {
    if (row_valid[r]) {
      for (auto const& [k, arr] : rows[r]) {
        flat_keys.push_back(k);
        inner_valid.push_back(arr.has_value());
        if (arr.has_value()) {
          for (auto const& elem : *arr) {
            flat_elements.push_back(elem.value_or(""));
            flat_element_valid.push_back(elem.has_value());
            ++elem_running;
          }
        }
        ++pair_running;
        inner_offsets.push_back(elem_running);
      }
    }
    outer_offsets.push_back(pair_running);
  }

  auto const num_pairs = pair_running;

  auto keys_child = cudf::test::strings_column_wrapper(flat_keys.begin(), flat_keys.end());

  bool const any_element_null =
    std::any_of(flat_element_valid.begin(), flat_element_valid.end(), [](bool v) { return !v; });
  auto elements_child =
    any_element_null
      ? cudf::test::strings_column_wrapper(
          flat_elements.begin(), flat_elements.end(), flat_element_valid.begin())
      : cudf::test::strings_column_wrapper(flat_elements.begin(), flat_elements.end());

  auto inner_offsets_col = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
                             inner_offsets.begin(), inner_offsets.end())
                             .release();

  auto [inner_mask, inner_null_count] = cudf::bools_to_mask(
    cudf::test::fixed_width_column_wrapper<bool>(inner_valid.begin(), inner_valid.end()));

  auto inner_list =
    cudf::make_lists_column(num_pairs,
                            std::move(inner_offsets_col),
                            elements_child.release(),
                            inner_null_count,
                            inner_null_count > 0 ? std::move(*inner_mask) : rmm::device_buffer{});

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.emplace_back(keys_child.release());
  struct_children.emplace_back(std::move(inner_list));
  auto structs_child =
    cudf::make_structs_column(num_pairs, std::move(struct_children), 0, rmm::device_buffer{});

  auto outer_offsets_col = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
                             outer_offsets.begin(), outer_offsets.end())
                             .release();

  auto [outer_mask, outer_null_count] = cudf::bools_to_mask(
    cudf::test::fixed_width_column_wrapper<bool>(row_valid.begin(), row_valid.end()));

  return cudf::make_lists_column(
    static_cast<cudf::size_type>(num_rows),
    std::move(outer_offsets_col),
    std::move(structs_child),
    outer_null_count,
    outer_null_count > 0 ? std::move(*outer_mask) : rmm::device_buffer{});
}

// Convenience builders for `array_value` literals.
array_value arr(std::vector<std::optional<std::string>> elems)
{
  return array_value{std::move(elems)};
}
array_value null_arr() { return std::nullopt; }
std::optional<std::string> null_elem() { return std::nullopt; }

}  // namespace

// Simple map-of-array: one row, one key, a 3-element string array.
TEST_F(FromJsonTest, RawMapArray_Simple)
{
  auto const input_col = cudf::test::strings_column_wrapper{R"({"k":["a","b","c"]})"};
  auto const input     = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map_array({{{"k", arr({"a", "b", "c"})}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// Multiple keys across multiple rows.
TEST_F(FromJsonTest, RawMapArray_MultiKeyMultiRow)
{
  auto const input_col =
    cudf::test::strings_column_wrapper{R"({"a":["1"],"b":["x","y"]})", R"({"c":["p","q","r"]})"};
  auto const input = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map_array(
    {{{"a", arr({"1"})}, {"b", arr({"x", "y"})}}, {{"c", arr({"p", "q", "r"})}}}, all_valid(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// Empty array `[]` -> empty NON-NULL inner list.
TEST_F(FromJsonTest, RawMapArray_EmptyArray)
{
  auto const input_col = cudf::test::strings_column_wrapper{R"({"k":[]})"};
  auto const input     = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map_array({{{"k", arr({})}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// Empty object `{}` -> empty (non-null) map row, no pairs.
TEST_F(FromJsonTest, RawMapArray_EmptyObject)
{
  auto const input_col =
    cudf::test::strings_column_wrapper{R"({"a":["1"]})", R"({})", R"({"b":["2"]})"};
  auto const input = cudf::strings_column_view{input_col};

  auto const expected =
    make_expected_raw_map_array({{{"a", arr({"1"})}}, {}, {{"b", arr({"2"})}}}, all_valid(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// Spark row-level bad-record semantics for non-array values:
//   * value = JSON `null` literal -> row KEPT, inner list null (mask #1).
//   * value = scalar (string/number/bool) or object (NOT an array, NOT null) -> WHOLE ROW null.
// Each case is its own row so the kept-vs-nullified rows are pinned independently. The
// `true`/`false` rows pin that a scalar boolean value takes the same hard-mismatch path as
// string/number/object.
TEST_F(FromJsonTest, RawMapArray_NonArrayValuesNullInnerList)
{
  auto const input_col = cudf::test::strings_column_wrapper{R"({"a":null})",
                                                            R"({"a":"s"})",
                                                            R"({"a":9})",
                                                            R"({"a":{"x":"y"}})",
                                                            R"({"a":true})",
                                                            R"({"a":false})"};
  auto const input     = cudf::strings_column_view{input_col};

  // Row 0 kept (null inner list); rows 1-5 are hard type mismatches -> whole row null. The pairs of
  // the nullified rows are ignored by `make_expected_raw_map_array`, so their content is
  // irrelevant.
  auto const expected = make_expected_raw_map_array({{{"a", null_arr()}},
                                                     {{"a", null_arr()}},
                                                     {{"a", null_arr()}},
                                                     {{"a", null_arr()}},
                                                     {{"a", null_arr()}},
                                                     {{"a", null_arr()}}},
                                                    {true, false, false, false, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// Row-level bad-record semantics across MULTI-KEY rows: a hard type mismatch in ANY value nulls the
// entire row even when other keys hold valid arrays; a JSON `null` value keeps the row.
//   * `{"a":["x"],"b":"s"}` -> WHOLE ROW null (even though `a` is a valid array).
//   * `{"a":["x"],"b":null}` -> row KEPT: `{a:[x], b:null}`.
TEST_F(FromJsonTest, RawMapArray_TypeMismatchValueNullsRow)
{
  auto const input_col =
    cudf::test::strings_column_wrapper{R"({"a":["x"],"b":"s"})", R"({"a":["x"],"b":null})"};
  auto const input = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map_array(
    {{{"a", arr({"x"})}, {"b", null_arr()}}, {{"a", arr({"x"})}, {"b", null_arr()}}},
    {false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// One array mixing literal `null`, the STRING "null", a number, a bool, and nested obj/array
// elements. Pins mask #2 (literal null -> null element) + raw-text + de-quote together. The nested
// object `{"x":1}` and array `[2,3]` elements keep their interior bytes verbatim.
TEST_F(FromJsonTest, RawMapArray_MixedElementKinds)
{
  auto const input_col =
    cudf::test::strings_column_wrapper{R"({"k":[null,"null",123,true,{"x":1},[2,3]]})"};
  auto const input = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map_array(
    {{{"k", arr({null_elem(), "null", "123", "true", R"({"x":1})", "[2,3]"})}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// Number / bool elements -> raw text.
TEST_F(FromJsonTest, RawMapArray_NumberBoolElements)
{
  auto const input_col = cudf::test::strings_column_wrapper{R"({"k":[1,22,333,true,false]})"};
  auto const input     = cudf::strings_column_view{input_col};

  auto const expected =
    make_expected_raw_map_array({{{"k", arr({"1", "22", "333", "true", "false"})}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// Null input row yields a NULL map row.
TEST_F(FromJsonTest, RawMapArray_NullInputRow)
{
  auto const input_col = cudf::test::strings_column_wrapper(
    {R"({"k":["a"]})", R"(unused)", R"({"k":["c"]})"}, {true, false, true});
  auto const input = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map_array({{{"k", arr({"a"})}}, {}, {{"k", arr({"c"})}}},
                                                    {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// Empty / whitespace-only / non-object / malformed rows -> NULL map rows.
TEST_F(FromJsonTest, RawMapArray_InvalidRows)
{
  auto const input_col = cudf::test::strings_column_wrapper{
    R"({"a":["1"]})", "   ", R"(not json)", R"([1,2,3])", R"({"b":["2"]})"};
  auto const input = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map_array(
    {{{"a", arr({"1"})}}, {}, {}, {}, {{"b", arr({"2"})}}}, {true, false, false, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// Escaped quotes and backslashes inside string elements survive verbatim (no JSON unescaping).
TEST_F(FromJsonTest, RawMapArray_EscapedElements)
{
  auto const input_col = cudf::test::strings_column_wrapper{R"({"k":["x\"y","p\\q"]})"};
  auto const input     = cudf::strings_column_view{input_col};

  auto const expected =
    make_expected_raw_map_array({{{"k", arr({R"(x\"y)", R"(p\\q)"})}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// Structural characters inside quoted string elements survive verbatim.
TEST_F(FromJsonTest, RawMapArray_StructuralCharElements)
{
  auto const input_col = cudf::test::strings_column_wrapper{R"({"k":["{}","[]",",:"]})"};
  auto const input     = cudf::strings_column_view{input_col};

  auto const expected =
    make_expected_raw_map_array({{{"k", arr({"{}", "[]", ",:"})}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// UTF-8 / emoji in keys and elements pass through verbatim.
TEST_F(FromJsonTest, RawMapArray_Utf8)
{
  auto const input_col =
    cudf::test::strings_column_wrapper{"{\"Zipcóde\":[\"\U00029E3D\",\"\U0001F3F3\"]}"};
  auto const input = cudf::strings_column_view{input_col};

  auto const expected =
    make_expected_raw_map_array({{{"Zipcóde", arr({"\U00029E3D", "\U0001F3F3"})}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// Single-quote normalization: with normalize_single_quotes on, `'...'` strings parse like `"..."`.
TEST_F(FromJsonTest, RawMapArray_SingleQuoteNormalization)
{
  auto const input_col = cudf::test::strings_column_wrapper{R"({'k':['a','b']})"};
  auto const input     = cudf::strings_column_view{input_col};

  auto const result = spark_rapids_jni::from_json_to_raw_map_array_values(
    input,
    spark_rapids_jni::json_parse_options{/*normalize_single_quotes=*/true,
                                         default_options.allow_leading_zeros,
                                         default_options.allow_nonnumeric_numbers,
                                         default_options.allow_unquoted_control});

  auto const expected = make_expected_raw_map_array({{{"k", arr({"a", "b"})}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

// Leading-zeros / non-numeric-number leniency on numeric elements (kept as raw text).
TEST_F(FromJsonTest, RawMapArray_NumericLeniency)
{
  auto const input_col = cudf::test::strings_column_wrapper{R"({"k":[007,NaN,Infinity]})"};
  auto const input     = cudf::strings_column_view{input_col};

  auto const result = spark_rapids_jni::from_json_to_raw_map_array_values(
    input,
    spark_rapids_jni::json_parse_options{default_options.normalize_single_quotes,
                                         /*allow_leading_zeros=*/true,
                                         /*allow_nonnumeric_numbers=*/true,
                                         default_options.allow_unquoted_control});

  auto const expected =
    make_expected_raw_map_array({{{"k", arr({"007", "NaN", "Infinity"})}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

// Whitespace around values: insignificant whitespace is not part of the element bytes. The
// expectation pins the de-quoted-string and trimmed-scalar interpretation.
TEST_F(FromJsonTest, RawMapArray_WhitespaceAroundValues)
{
  auto const input_col = cudf::test::strings_column_wrapper{R"({ "k" : [ "a" , "b" ] })"};
  auto const input     = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map_array({{{"k", arr({"a", "b"})}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// Many keys, each with a small array.
TEST_F(FromJsonTest, RawMapArray_ManyKeys)
{
  constexpr int num_keys = 32;
  std::string obj        = "{";
  std::vector<kva> pairs;
  pairs.reserve(num_keys);
  for (int k = 0; k < num_keys; ++k) {
    auto const key = std::format("k{}", k);
    if (k > 0) { obj += ","; }
    obj += std::format(R"("{}":["v{}"])", key, k);
    pairs.emplace_back(key, arr({std::format("v{}", k)}));
  }
  obj += "}";

  auto const input_col = cudf::test::strings_column_wrapper{obj};
  auto const input     = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map_array({pairs}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// Duplicate keys emit one (key, array) pair per occurrence, in textual order.
TEST_F(FromJsonTest, RawMapArray_DuplicateKeys)
{
  auto const input_col = cudf::test::strings_column_wrapper{R"({"a":["1"],"a":["2","3"]})"};
  auto const input     = cudf::strings_column_view{input_col};

  auto const expected =
    make_expected_raw_map_array({{{"a", arr({"1"})}, {"a", arr({"2", "3"})}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// Adversarial inner-offset test: a row interleaving a 0-element array, a null inner list, a
// many-element array, and a 1-element array. Pins the inner-offset off-by-one handling exactly.
TEST_F(FromJsonTest, RawMapArray_InnerOffsetAdversarial)
{
  auto const input_col =
    cudf::test::strings_column_wrapper{R"({"a":[],"b":null,"c":["x","y","z"],"d":["w"]})"};
  auto const input = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map_array(
    {{{"a", arr({})}, {"b", null_arr()}, {"c", arr({"x", "y", "z"})}, {"d", arr({"w"})}}},
    all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// Adversarial inner-offset test at scale: 50k rows of the same interleaved shape, for
// sanitizer race coverage on the inner-offset scatter.
TEST_F(FromJsonTest, RawMapArray_InnerOffsetAdversarialLarge)
{
  constexpr int num_rows = 50000;
  std::vector<std::string> rows;
  std::vector<std::vector<kva>> expected_rows;
  rows.reserve(num_rows);
  expected_rows.reserve(num_rows);

  for (int r = 0; r < num_rows; ++r) {
    rows.emplace_back(R"({"a":[],"b":null,"c":["x","y","z"],"d":["w"]})");
    expected_rows.push_back(
      {{"a", arr({})}, {"b", null_arr()}, {"c", arr({"x", "y", "z"})}, {"d", arr({"w"})}});
  }

  auto const input_col = cudf::test::strings_column_wrapper(rows.begin(), rows.end());
  auto const input     = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map_array(expected_rows, all_valid(num_rows));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// Empty input for the array function (exercises make_empty_map_array).
TEST_F(FromJsonTest, RawMapArray_EmptyInput)
{
  auto const input_col = cudf::test::strings_column_wrapper{};
  auto const input     = cudf::strings_column_view{input_col};

  auto const result = raw_map_array(input);
  EXPECT_EQ(result->size(), 0);
  ASSERT_EQ(result->type().id(), cudf::type_id::LIST);

  auto const lcv          = cudf::lists_column_view{result->view()};
  auto const structs_view = lcv.child();
  ASSERT_EQ(structs_view.type().id(), cudf::type_id::STRUCT);
  ASSERT_EQ(structs_view.num_children(), 2);
  auto const scv = cudf::structs_column_view{structs_view};
  EXPECT_EQ(scv.child(0).type().id(), cudf::type_id::STRING);  // key.
  ASSERT_EQ(scv.child(1).type().id(), cudf::type_id::LIST);    // value List<String>.
  EXPECT_EQ(cudf::lists_column_view{scv.child(1)}.child().type().id(), cudf::type_id::STRING);
}

// size_type-boundary construction: a single row with many elements, sized so the inner offset
// values stay well within int32 but exercise a large element count in one list. This does NOT
// reach overflow; it pins the wide-inner-list path.
TEST_F(FromJsonTest, RawMapArray_WideInnerList)
{
  constexpr int num_elems = 100000;
  std::string obj         = R"({"k":[)";
  std::vector<std::optional<std::string>> elems;
  elems.reserve(num_elems);
  for (int e = 0; e < num_elems; ++e) {
    if (e > 0) { obj += ","; }
    obj += std::format(R"("e{}")", e);
    elems.emplace_back(std::format("e{}", e));
  }
  obj += "]}";

  auto const input_col = cudf::test::strings_column_wrapper{obj};
  auto const input     = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map_array({{{"k", arr(std::move(elems))}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// All-null-element array `{"k":[null,null,null]}` -> non-null inner list of three null elements
// (mask #2 all-false), exercising the degenerate `el_null_count == array_len` element path.
TEST_F(FromJsonTest, RawMapArray_AllNullElements)
{
  auto const input_col = cudf::test::strings_column_wrapper{R"({"k":[null,null,null]})"};
  auto const input     = cudf::strings_column_view{input_col};

  auto const expected = make_expected_raw_map_array(
    {{{"k", arr({null_elem(), null_elem(), null_elem()})}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}

// allow_unquoted_control=true: a literal control char (tab) inside a quoted array element is
// accepted and copied verbatim into the element bytes.
TEST_F(FromJsonTest, RawMapArray_UnquotedControl)
{
  auto const input_col = cudf::test::strings_column_wrapper{"{\"k\":[\"a\tb\"]}"};
  auto const input     = cudf::strings_column_view{input_col};

  auto const result = spark_rapids_jni::from_json_to_raw_map_array_values(
    input,
    spark_rapids_jni::json_parse_options{default_options.normalize_single_quotes,
                                         default_options.allow_leading_zeros,
                                         default_options.allow_nonnumeric_numbers,
                                         /*allow_unquoted_control=*/true});

  auto const expected = make_expected_raw_map_array({{{"k", arr({"a\tb"})}}}, all_valid(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

// Sliced (nonzero-offset) input: slice off row 0 of a 3-row column and verify the 2-row result.
// Multi-level offsets (outer + inner list) make this the most off-by-one-prone path.
TEST_F(FromJsonTest, RawMapArray_SlicedInput)
{
  auto const input_col =
    cudf::test::strings_column_wrapper{R"({"skip":[]})", R"({"k":["a"]})", R"({"k":["b"]})"};
  auto const sliced = cudf::slice(input_col, {1, 3})[0];
  auto const input  = cudf::strings_column_view{sliced};

  auto const expected =
    make_expected_raw_map_array({{{"k", arr({"a"})}}, {{"k", arr({"b"})}}}, all_valid(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*raw_map_array(input), *expected);
}
