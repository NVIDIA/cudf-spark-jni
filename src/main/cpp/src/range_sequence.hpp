/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <span>

namespace spark_rapids_jni {

/**
 * @brief Expand host-side (start, size) ranges into one owning, non-nullable INT64 column.
 *
 * Ranges retain input order, including gaps, overlaps, or decreasing starts. Zero sizes produce
 * no values. The CPU validates metadata and computes cumulative range ends. Thrust writes the
 * final column without per-range device sequences or concatenate. A single range needs no device
 * metadata, and an empty result launches no kernel.
 *
 * Host inputs may be released when this function returns. Output allocations use @p mr; temporary
 * device allocations use the current device resource. All device work uses @p stream. Temporary
 * device storage is O(starts.size()). Allocation failures propagate to the caller.
 * This function may synchronize @p stream. For multiple ranges, each metadata upload completes
 * before its host staging buffer can be released; output generation may still be pending on return.
 *
 * @throws std::invalid_argument if lengths differ, a size is negative, there are more than
 * cudf::size_type's maximum number of ranges, or a generated value exceeds INT64_MAX.
 * @throws std::overflow_error if the total size exceeds cudf::size_type's maximum.
 *
 * @param starts First value of each range; need not be sorted
 * @param sizes Nonnegative number of values in each range
 * @param stream CUDA stream for allocations, metadata upload, and generation
 * @param mr Device memory resource for the output
 * @return Column containing the concatenated sequences, owned by the caller
 */
std::unique_ptr<cudf::column> generate_range_sequence(
  std::span<int64_t const> starts,
  std::span<cudf::size_type const> sizes,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
