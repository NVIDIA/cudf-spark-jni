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

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/stream>

namespace spark_rapids_jni {
/**
 * @brief Check if input string contains regex pattern `literal[start-end]{len,}`, which means
 * a literal string followed by a range of characters in the range of start to end, with at least
 * len characters.
 *
 * @param strings Column of strings to check for literal.
 * @param literal UTF-8 encoded string to check in strings column.
 * @param len Minimum number of characters to check after the literal.
 * @param start Minimum UTF-8 codepoint value to check for in the range.
 * @param end Maximum UTF-8 codepoint value to check for in the range.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
std::unique_ptr<cudf::column> literal_range_pattern(
  cudf::strings_column_view const& input,
  cudf::string_scalar const& literal,
  int const len,
  int const start,
  int const end,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());
}  // namespace spark_rapids_jni
