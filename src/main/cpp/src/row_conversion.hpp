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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <vector>

namespace spark_rapids_jni {

// Stream contract for the conversion entry points below: each enqueues its work on `stream` and
// may return before that work completes, so callers must keep the inputs alive and synchronize
// `stream` before reading a result on the host. Some of them synchronize internally today to
// read null counts back; that is an implementation detail callers must not rely on.

std::vector<std::unique_ptr<cudf::column>> convert_to_rows_fixed_width_optimized(
  cudf::table_view const& tbl,
  // TODO need something for validity
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

std::vector<std::unique_ptr<cudf::column>> convert_to_rows(
  cudf::table_view const& tbl,
  // TODO need something for validity
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

std::unique_ptr<cudf::table> convert_from_rows_fixed_width_optimized(
  cudf::lists_column_view const& input,
  std::vector<cudf::data_type> const& schema,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

std::unique_ptr<cudf::table> convert_from_rows(
  cudf::lists_column_view const& input,
  std::vector<cudf::data_type> const& schema,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Convert JCUDF rows back to columns on the fixed-width-optimized path, trusting a
 * caller-supplied per-column nullability declaration
 *
 * `may_have_nulls` must carry one entry per schema column. When every entry is false the caller
 * guarantees the rows contain no nulls: the conversion skips all reverse-validity work (output
 * null-mask allocation, the validity kernel, null-count computation and its synchronize) and
 * returns non-nullable columns. A false guarantee silently yields wrong values for the rows that
 * are actually null; set the environment variable `ROWCONV_VALIDATE_ALL_VALID=1` to re-verify
 * every all-valid declaration and throw on violation instead (debug aid, default off). Any true
 * entry selects the incumbent path for the whole call.
 *
 * `may_have_nulls` MUST be OBSERVED from the data that produced these rows, never inferred from a
 * declared schema. Spark's static nullability metadata is not a valid source: it states what a
 * column MAY contain, and a planner may declare a column non-nullable that carries nulls at
 * runtime. The supported producers are a null-count on the source columns, or a flag the row
 * packer sets while writing (see `GpuRowToColumnarExec`'s generated `fillBatch`).
 *
 * @param input list column holding the JCUDF row data
 * @param schema incoming schema of the data
 * @param may_have_nulls one entry per schema column; false = guaranteed to contain no nulls
 * @param stream stream to use for compute
 * @param mr memory resource for returned data
 * @return the converted table
 */
std::unique_ptr<cudf::table> convert_from_rows_fixed_width_optimized(
  cudf::lists_column_view const& input,
  std::vector<cudf::data_type> const& schema,
  std::vector<bool> const& may_have_nulls,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Convert JCUDF rows back to columns on the general path, trusting a caller-supplied
 * per-column nullability declaration
 *
 * Same declaration contract as the fixed-width-optimized overload above: `may_have_nulls` must
 * be OBSERVED from the data, never inferred from a declared schema.
 *
 * @param input list column holding the JCUDF row data
 * @param schema incoming schema of the data
 * @param may_have_nulls one entry per schema column; false = guaranteed to contain no nulls
 * @param stream stream to use for compute
 * @param mr memory resource for returned data
 * @return the converted table
 */
std::unique_ptr<cudf::table> convert_from_rows(
  cudf::lists_column_view const& input,
  std::vector<cudf::data_type> const& schema,
  std::vector<bool> const& may_have_nulls,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
