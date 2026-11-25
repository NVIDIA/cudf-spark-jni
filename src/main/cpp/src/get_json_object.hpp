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

/// @brief Maximum JSON path depth
constexpr int MAX_JSON_PATH_DEPTH = 16;

/// @brief JSON path instruction kind
enum class path_instruction_type : int8_t { WILDCARD, INDEX, NAMED };

/// @brief Named-field selection policy for Spark `json_tuple`
enum class named_field_match_policy : int32_t { LAST_NON_NULL = 1 };

/**
 * @brief Extracts one JSON path per input row.
 *
 * Invalid JSON, missing values and null rows return null, regardless of retained bytes.
 * @param[in] input        JSON strings
 * @param[in] instructions Path instructions
 * @param[in] stream       CUDA stream
 * @param[in] mr           Output memory resource
 * @return One strings column in input row order
 */
std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& input,
  std::vector<std::tuple<path_instruction_type, std::string, int32_t>> const& instructions,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Extracts JSON paths in parallel batches, trading scratch memory for throughput.
 *
 * Null input rows return null for every path, regardless of retained bytes.
 * @param[in] input               JSON strings
 * @param[in] json_paths          Path instructions
 * @param[in] memory_budget_bytes Soft scratch budget if positive
 * @param[in] parallel_override   Paths per batch if positive; overrides the budget
 * @param[in] stream              CUDA stream
 * @param[in] mr                  Output memory resource
 * @return One strings column per path, in path order
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
 * @brief Extracts the last non-null occurrence of each named field for Spark `json_tuple`.
 *
 * Each path must be one `NAMED` instruction. Null rows return null regardless of retained bytes.
 * @param[in] input               JSON strings
 * @param[in] json_paths          Top-level named paths
 * @param[in] memory_budget_bytes Soft scratch budget if positive
 * @param[in] parallel_override   Paths per batch if positive; overrides the budget
 * @param[in] match_policy        Duplicate-field policy (`LAST_NON_NULL`)
 * @param[in] stream              CUDA stream
 * @param[in] mr                  Output memory resource
 * @return One strings column per path, in path order
 * @throws cudf::logic_error Unsupported policy or path shape
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

namespace detail {

// Scan work selected for the request; offset sizing is exact in all three modes.
enum class gjo_scan_mode { offset_only, sampled, exact };

// The optional test observer records first-pass routes in request order: 0 general, 1 shallow;
// higher codes are reserved for deeper families. It does not change admission.
std::vector<std::unique_ptr<cudf::column>> get_json_object(
  cudf::strings_column_view const& input,
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>> const&
    json_paths,
  int64_t memory_budget_bytes,
  int32_t parallel_override,
  bool use_last_non_null,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  std::vector<char>* merged_routed_out = nullptr,
  gjo_scan_mode* scan_mode             = nullptr);

}  // namespace detail

}  // namespace spark_rapids_jni
