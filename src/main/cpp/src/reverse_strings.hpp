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
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>

namespace spark_rapids_jni {

/**
 * @brief Reverse strings using Spark `UTF8String.reverse` character-width semantics.
 *
 * Unlike libcudf `cudf::strings::reverse`, character widths follow Spark's
 * `UTF8String.numBytesForFirstByte` and are clamped to the bytes remaining in each
 * row (`min(declared_width, remaining)`). This matches SPARK-57507 and avoids
 * reading past a truncated trailing multi-byte UTF-8 sequence into the next row.
 *
 * Well-formed UTF-8 results match libcudf reverse. Null rows remain null. Output
 * offsets match the input offsets (byte length is preserved per row).
 *
 * @param input Strings column
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource used to allocate the returned column
 * @return New strings column with Spark-compatible reversed contents
 */
std::unique_ptr<cudf::column> reverse_strings(
  cudf::strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
