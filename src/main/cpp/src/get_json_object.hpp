/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace spark_rapids_jni {

/**
 * @brief The maximum supported depth that a JSON path can reach.
 */
constexpr int MAX_JSON_PATH_DEPTH = 16;

/**
 * @brief Type of instruction in a JSON path.
 */
enum class path_instruction_type : int8_t { WILDCARD, INDEX, NAMED };

/**
 * @brief Policy for selecting a matching named field when an object contains duplicate keys.
 */
enum class named_field_match_policy : int32_t { FIRST_NON_NULL = 0, LAST_NON_NULL = 1 };

/**
 * @brief Extract JSON object from a JSON string based on the specified JSON path.
 *
 * If the input JSON string is invalid, or it does not contain the object at the given path, a null
 * will be returned.
 */
std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& input,
  std::vector<std::tuple<path_instruction_type, std::string, int32_t>> const& instructions,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Extract multiple JSON objects from a JSON string based on the specified JSON paths.
 *
 * This function processes all the JSON paths in parallel, which may be faster than calling
 * to `get_json_object` on the individual JSON paths. However, it may consume much more GPU
 * memory, proportional to the number of JSON paths.
 * @param input the input string column to parse JSON from
 * @param json_paths the path operations to read extract
 * @param memory_budget_bytes a memory budget for temporary memory usage if > 0
 * @param parallel_override if this value is greater than 0 then it specifies the
 *        number of paths to process in parallel (this will cause the
 *        `memory_budget_bytes` paramemter to be ignored)
 */
std::vector<std::unique_ptr<cudf::column>> get_json_object_multiple_paths(
  cudf::strings_column_view const& input,
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>> const&
    json_paths,
  int64_t memory_budget_bytes,
  int32_t parallel_override,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Extract multiple JSON objects using the specified named-field match policy.
 *
 * `FIRST_NON_NULL` requires every path to be non-empty and contain only `NAMED` instructions.
 * `LAST_NON_NULL` requires every path to contain exactly one `NAMED` instruction. Use the overload
 * without `match_policy` for legacy `INDEX` and `WILDCARD` path handling.
 *
 * @throw cudf::logic_error If `match_policy` is invalid or a path shape is unsupported by it
 * @param[in] input The input string column to parse JSON from
 * @param[in] json_paths The JSON path instructions to extract
 * @param[in] memory_budget_bytes A soft temporary-memory budget when greater than zero
 * @param[in] parallel_override A positive override for the number of paths processed in parallel
 * @param[in] match_policy The duplicate named-field selection policy
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate returned columns
 * @return One output strings column for each input path, in path order
 */
std::vector<std::unique_ptr<cudf::column>> get_json_object_multiple_paths(
  cudf::strings_column_view const& input,
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>> const&
    json_paths,
  int64_t memory_budget_bytes,
  int32_t parallel_override,
  named_field_match_policy match_policy,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
