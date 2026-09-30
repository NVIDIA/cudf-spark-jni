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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/stream>

#include <cstddef>

namespace cudf::jni {

std::unique_ptr<cudf::table> multiply_decimal128(
  cudf::column_view const& a,
  cudf::column_view const& b,
  int32_t product_scale,
  bool const cast_interim_result,
  cuda::stream_ref stream = cudf::get_default_stream());

std::unique_ptr<cudf::table> divide_decimal128(
  cudf::column_view const& a,
  cudf::column_view const& b,
  int32_t quotient_scale,
  cuda::stream_ref stream = cudf::get_default_stream());

std::unique_ptr<cudf::table> integer_divide_decimal128(
  cudf::column_view const& a,
  cudf::column_view const& b,
  int32_t quotient_scale,
  cuda::stream_ref stream = cudf::get_default_stream());

std::unique_ptr<cudf::table> remainder_decimal128(
  cudf::column_view const& a,
  cudf::column_view const& b,
  int32_t remainder_scale,
  cuda::stream_ref stream = cudf::get_default_stream());

std::unique_ptr<cudf::table> add_decimal128(cudf::column_view const& a,
                                            cudf::column_view const& b,
                                            int32_t quotient_scale,
                                            cuda::stream_ref stream = cudf::get_default_stream());

std::unique_ptr<cudf::table> sub_decimal128(cudf::column_view const& a,
                                            cudf::column_view const& b,
                                            int32_t quotient_scale,
                                            cuda::stream_ref stream = cudf::get_default_stream());

/**
 * @brief Cast floating point values to decimals, matching the behavior of Spark.
 *
 * @param input The input column, which is either FLOAT32 or FLOAT64 type
 * @param output_type The output decimal type
 * @param precision The maximum number of digits that will be preserved in the output
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A cudf column containing the cast result and a boolean value indicating whether the cast
           operation has failed for any input rows
 */
std::pair<std::unique_ptr<cudf::column>, cudf::size_type> floating_point_to_decimal(
  cudf::column_view const& input,
  cudf::data_type output_type,
  int32_t precision,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

}  // namespace cudf::jni
