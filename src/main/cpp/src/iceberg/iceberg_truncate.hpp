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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace spark_rapids_jni {

/**
 * @brief Truncate towards negative infinity direction for types: integral or decimal.
 *
 * The input type can be INT32, INT64, DECIMAL32, DECIMAL64 or DECIMAL128.
 * Note: For DECIMAL types, the truncation is performed on the integer representation
 *
 * For positive values, Iceberg truncation is: value - (value % width)
 * For negative values, this uses a floored modulo approach:
 * value - (((value % width) + width) % width)
 *
 * Examples:
 *
 * Integer truncation with width = 10:
 * - truncate_integer(5, 10) = 0
 * - truncate_integer(15, 10) = 10
 * - truncate_integer(-5, 10) = -10
 *
 * Decimal truncation with width = 10:
 * - truncate_integer(12.29, 10) = 12.20
 * - truncate_integer(-0.05, 10) = -0.10
 *
 * @param input Integral or decimal column to truncate
 * @param width Truncation width
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource used to allocate the returned column
 *
 * @return Truncated integral or decimal column
 */
std::unique_ptr<cudf::column> truncate_integral(
  cudf::column_view const& input,
  int32_t width,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Truncate by character for string values in UTF-8 encoding.
 *
 * Note: One character may use multiple(1-4) bytes in UTF-8 encoding.
 *
 * Examples:
 * - truncate_string("hello world", 5) = "hello"
 * - truncate_string("你好，世界", 2) = "你好"
 * - truncate_string("🚀23😁567", 5) = "🚀23😁5"
 *
 * @param input String column to truncate
 * @param length Maximum character length for truncated strings
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource used to allocate the returned column
 *
 * @return Truncated string column
 */
std::unique_ptr<cudf::column> truncate_string(
  cudf::column_view const& input,
  int32_t length,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Truncates binary values to the specified byte length.
 *
 * @param input Binary (list of bytes) column to truncate
 * @param length Maximum byte length for truncated binary data
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource used to allocate the returned column
 *
 * @return Truncated binary column
 */
std::unique_ptr<cudf::column> truncate_binary(
  cudf::column_view const& input,
  int32_t length,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
