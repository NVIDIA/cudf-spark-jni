/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.
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

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <memory>

namespace spark_rapids_jni {

std::vector<std::unique_ptr<cudf::column>> convert_to_rows_fixed_width_optimized(
  cudf::table_view const& tbl,
  // TODO need something for validity
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * Convert a table to JCUDF row format. Accepts fixed-width and STRING columns, plus
 * DICTIONARY32 columns with STRING keys, which are encoded by value exactly like the decoded
 * STRING column (other dictionary key types are rejected). The row format carries no dictionary
 * information.
 */
std::vector<std::unique_ptr<cudf::column>> convert_to_rows(
  cudf::table_view const& tbl,
  // TODO need something for validity
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

std::unique_ptr<cudf::table> convert_from_rows_fixed_width_optimized(
  cudf::lists_column_view const& input,
  std::vector<cudf::data_type> const& schema,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * Convert JCUDF rows back to a table. DICTIONARY32 is not a valid schema type: columns that were
 * dictionaries when passed to convert_to_rows must be read back as STRING and re-encoded by the
 * caller (e.g. with cudf::dictionary::encode) if a dictionary is needed.
 */
std::unique_ptr<cudf::table> convert_from_rows(
  cudf::lists_column_view const& input,
  std::vector<cudf::data_type> const& schema,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
