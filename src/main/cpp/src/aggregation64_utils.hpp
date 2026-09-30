/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/stream>

#include <memory>

namespace spark_rapids_jni {

/**
 * @brief Extract a 32-bit integer column from a column of 64-bit values.
 *
 * Given a 64-bit input column, a 32-bit integer column is returned corresponding to
 * the index of which 32-bit chunk of the original 64-bit values to extract.
 * 0 corresponds to the least significant chunk, and 1 corresponds to the most
 * significant chunk.
 *
 * A null input row will result in a corresponding null output row.
 *
 * @param col       Column of 64-bit values
 * @param dtype     Integer type to use for the output column (e.g.: UINT32 or INT32)
 * @param chunk_idx Index of the 32-bit chunk to extract
 * @param stream    CUDA stream to use
 * @return          A column containing the extracted 32-bit integer values
 */
std::unique_ptr<cudf::column> extract_chunk32_from_64bit(
  cudf::column_view const& col,
  cudf::data_type dtype,
  int chunk_idx,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Reassemble a 64-bit column from two 64-bit integer columns with overflow detection.
 *
 * The 64-bit value is reconstructed by overlapping the 64-bit values by 32-bits. The least
 * significant 32-bits of the least significant 64-bit value are used directly as the least
 * significant 32-bits of the final 64-bit value, and the remaining 32-bits are added to the next
 * most significant 64-bit value.
 *
 * A null input row will result in a corresponding null output row.
 *
 * @param chunks_table Table of two 64-bit integer columns with the columns ordered from least
 *                     significant to most significant. The last column must be an INT64 column.
 * @param output_type  The type to use for the resulting 64-bit value column
 * @param stream       CUDA stream to use
 * @return             Table containing a boolean column and a 64-bit value column of the
 *                     requested type. The boolean value will be true if an overflow was detected
 *                     for that row's value.
 */
std::unique_ptr<cudf::table> assemble64_from_sum(
  cudf::table_view const& chunks_table,
  cudf::data_type output_type,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
