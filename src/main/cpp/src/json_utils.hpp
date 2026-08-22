/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <memory>

namespace spark_rapids_jni {

/**
 * @brief Leniency options for the raw-map JSON extraction functions
 */
struct json_parse_options {
  bool normalize_single_quotes  = false;
  bool allow_leading_zeros      = false;
  bool allow_nonnumeric_numbers = false;
  bool allow_unquoted_control   = false;
};

/**
 * @brief Extract a map column from the JSON strings given by an input strings column.
 */
std::unique_ptr<cudf::column> from_json_to_raw_map(
  cudf::strings_column_view const& input,
  json_parse_options options,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Extract a map-of-array column from the JSON strings given by an input strings column.
 *
 * This targets the Spark schema `MapType[StringType, ArrayType[StringType]]`: each JSON object's
 * keys are strings and each value is a JSON array of strings. The output is a
 * `List<Struct<String, List<String>>>` column in which the struct's value child is a
 * `List<String>` holding the array elements.
 *
 * A value that is the JSON `null` literal produces a null inner list (the map row is kept). A value
 * that is non-null but not a JSON array (a scalar or an object) is a row-level bad record: the
 * entire outer map row is nullified, matching Spark `from_json`. An array element that is the
 * literal `null` produces a null element; every other element (string, number, boolean, or a nested
 * object/array taken as its raw JSON substring) is emitted as its de-quoted raw bytes, matching
 * `from_json_to_raw_map`'s `include_quote_char=false` extraction. Row-level null/empty/invalid
 * handling is otherwise identical to `from_json_to_raw_map`.
 */
std::unique_ptr<cudf::column> from_json_to_raw_map_array_values(
  cudf::strings_column_view const& input,
  json_parse_options options,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Parse JSON strings into a struct column followed by a given data schema.
 *
 * The data schema is specified as data arrays flattened by depth-first-search order.
 */
std::unique_ptr<cudf::column> from_json_to_structs(
  cudf::strings_column_view const& input,
  std::vector<std::string> const& col_names,
  std::vector<int> const& num_children,
  std::vector<int> const& types,
  std::vector<int> const& scales,
  std::vector<int> const& precisions,
  bool normalize_single_quotes,
  bool allow_leading_zeros,
  bool allow_nonnumeric_numbers,
  bool allow_unquoted_control,
  bool is_us_locale,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Convert from a strings column to a column with the desired type given by a data schema.
 *
 * The given column schema is specified as data arrays flattened by depth-first-search order.
 */
std::unique_ptr<cudf::column> convert_from_strings(
  cudf::strings_column_view const& input,
  std::vector<int> const& num_children,
  std::vector<int> const& types,
  std::vector<int> const& scales,
  std::vector<int> const& precisions,
  bool allow_nonnumeric_numbers,
  bool is_us_locale,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Remove quotes from each string in the given strings column.
 *
 * If `nullify_if_not_quoted` is true, an input string that is not quoted will result in a null.
 * Otherwise, the output will be the same as the unquoted input.
 */
std::unique_ptr<cudf::column> remove_quotes(
  cudf::strings_column_view const& input,
  bool nullify_if_not_quoted,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Concatenate the JSON objects given by a strings column into one single character buffer,
 * in which each JSON objects is delimited by a special character that does not exist in the input.
 *
 * Beyond returning the concatenated buffer with delimiter, the function also returns a BOOL8
 * column indicating which rows should be nullified after parsing the concatenated buffer. Each
 * row of this column is a `true` value if the corresponding input row is either empty, containing
 * only whitespaces, or invalid JSON object depending on the `nullify_invalid_rows` parameter.
 *
 * Note that an invalid JSON object in this context is a string that does not start with the `{`
 * character after whitespaces.
 *
 * @param input The strings column containing input JSON objects
 * @param nullify_invalid_rows Whether to nullify rows containing invalid JSON objects
 * @param stream The CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the table in the returned
 * @return A tuple containing the concatenated JSON objects as a single buffer, the delimiter
 *         character, and a BOOL8 column indicating which rows should be nullified after parsing
 *         the concatenated buffer
 */
std::tuple<std::unique_ptr<rmm::device_buffer>, char, std::unique_ptr<cudf::column>> concat_json(
  cudf::strings_column_view const& input,
  bool nullify_invalid_rows         = false,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
