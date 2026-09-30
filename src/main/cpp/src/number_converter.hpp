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

#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>
#include <variant>

namespace spark_rapids_jni {

using convert_number_t = std::variant<cudf::column_view, cudf::string_scalar, int>;

/**
 *
 * @brief Convert numbers(in string column) between different number bases. If toBase>0 the result
 * is unsigned, otherwise it is signed. First trim the space characters (ASCII 32). Return null if
 * len(trim_ascii_32(str)) == 0. Return all nulls if `from_base` or `to_base` is not in range [2,
 * 36]
 *
 * e.g.:
 *   convert('11', 2, 10) = '3'
 *   convert('F', 16, 10) = '15'
 *   convert('17', 10, 16) = '11'
 *
 * @param input the input string column or string scalar
 * @param from_base the number base of input, valid range is [2, 36]
 * @param to_base the number base of output, valid range is [2, 36]
 *
 * @return the string column contains numbers with `to_base` base
 */
std::unique_ptr<cudf::column> convert(
  convert_number_t const& input,
  convert_number_t const& from_base,
  convert_number_t const& to_base,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 *
 * @brief Check if overflow occurs for converting numbers(in string column) between different
 * number bases. This is for the checking when it's ANSI mode. For more details, please refer to the
 * convert function.
 *
 * @param input the input string column or string scalar
 * @param from_base the number base of input, valid range is [2, 36]
 * @param to_base the number base of output, valid range is [2, 36]
 *
 * @return If overflow occurs, return true; otherwise, return false.
 */
bool is_convert_overflow(
  convert_number_t const& input,
  convert_number_t const& from_base,
  convert_number_t const& to_base,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
