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

#include <cudf/column/column_device_view.cuh>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

namespace spark_rapids_jni {

template <typename T>
constexpr inline bool is_basic_spark_numeric()
{
  return std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> ||
         std::is_same_v<T, int64_t> || std::is_same_v<T, float> || std::is_same_v<T, double>;
}

bool is_basic_spark_numeric(cudf::data_type type);

/**
 * @brief Bitwise-or an array of equally-sized bitmask buffers into a single output buffer
 *
 * @param input The array of input bitmask buffers.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned bloom filter's memory.
 *
 */
std::unique_ptr<rmm::device_buffer> bitmask_bitwise_or(
  std::vector<cudf::device_span<cudf::bitmask_type const>> const& input,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
