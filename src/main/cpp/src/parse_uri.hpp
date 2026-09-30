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
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <memory>

namespace spark_rapids_jni {

/**
 * @brief Parse protocol and copy from the input string column to the output string column.
 *
 * @param input Input string column of URIs to parse
 * @param stream Stream on which to operate.
 * @param mr Memory resource for returned column
 * @return std::unique_ptr<column> String column of protocols parsed.
 */
std::unique_ptr<cudf::column> parse_uri_to_protocol(
  cudf::strings_column_view const& input,
  bool ansi_mode                    = false,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Parse host and copy from the input string column to the output string column.
 *
 * @param input Input string column of URIs to parse
 * @param stream Stream on which to operate.
 * @param mr Memory resource for returned column
 * @return std::unique_ptr<column> String column of hosts parsed.
 */
std::unique_ptr<cudf::column> parse_uri_to_host(
  cudf::strings_column_view const& input,
  bool ansi_mode                    = false,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Parse query and copy from the input string column to the output string column.
 *
 * @param input Input string column of URIs to parse
 * @param stream Stream on which to operate.
 * @param mr Memory resource for returned column
 * @return std::unique_ptr<column> String column of queries parsed.
 */
std::unique_ptr<cudf::column> parse_uri_to_query(
  cudf::strings_column_view const& input,
  bool ansi_mode                    = false,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Parse query and copy from the input string column to the output string column.
 *
 * @param input Input string column of URIs to parse.
 * @param query_match String to match in query.
 * @param stream Stream on which to operate.
 * @param mr Memory resource for returned column.
 * @return std::unique_ptr<column> String column of queries parsed.
 */
std::unique_ptr<cudf::column> parse_uri_to_query(
  cudf::strings_column_view const& input,
  std::string const& query_match,
  bool ansi_mode                    = false,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Parse query and copy from the input string column to the output string column.
 *
 * @param input Input string column of URIs to parse.
 * @param query_match string column to match in query.
 * @param stream Stream on which to operate.
 * @param mr Memory resource for returned column.
 * @return std::unique_ptr<column> String column of queries parsed.
 */
std::unique_ptr<cudf::column> parse_uri_to_query(
  cudf::strings_column_view const& input,
  cudf::strings_column_view const& query_match,
  bool ansi_mode                    = false,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Parse path and copy from the input string column to the output string column.
 *
 * @param input Input string column of URIs to parse
 * @param stream Stream on which to operate.
 * @param mr Memory resource for returned column
 * @return std::unique_ptr<column> String column of paths parsed.
 */
std::unique_ptr<cudf::column> parse_uri_to_path(
  cudf::strings_column_view const& input,
  bool ansi_mode                    = false,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
