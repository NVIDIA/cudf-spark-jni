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

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace spark_rapids_jni {

/**
 * @brief Where each host buffer lands in packed device allocations
 */
struct pack_plan {
  std::vector<std::size_t> pack_sizes;   ///< Each pack's size, rounded up for padding
  std::vector<std::size_t> pack_counts;  ///< How many consecutive buffers each pack takes
  std::vector<std::size_t> pack_of;      ///< Each host buffer's pack
  std::vector<std::size_t> offsets;      ///< Each host buffer's offset in its pack
};

/**
 * @brief Plan packing host buffers into device allocations
 *
 * The buffers are packed back to back in order, starting a new allocation whenever the next one
 * would take the current one past `max_pack_bytes`; a buffer larger than that gets an allocation
 * of its own. Each allocation's size is rounded up to a multiple of 8 bytes, as libcudf does for
 * the column chunk buffers it reads into.
 *
 * @param buffers The host buffers to pack
 * @param max_pack_bytes The most bytes packed into one allocation, unless one buffer is larger
 * @return The plan
 */
pack_plan plan_packs(std::span<std::span<uint8_t const> const> buffers, std::size_t max_pack_bytes);

/**
 * @brief Allocate the packs `plan` describes and queue copies of the host buffers into them
 *
 * Buffers adjacent in host memory and in the same allocation move in a single copy. Every
 * allocation is made before any copy is queued, so if one fails nothing is in flight. If this
 * throws after queueing, the copies have already completed.
 *
 * @param buffers The host buffers `plan` was made for, which must stay valid until the copies
 *        complete
 * @param plan The plan from `plan_packs`
 * @param stream CUDA stream the allocations and copies are made on
 * @param mr Device memory resource used to allocate the packs
 * @return The packs
 */
std::vector<std::unique_ptr<rmm::device_buffer>> copy_packed_async(
  std::span<std::span<uint8_t const> const> buffers,
  pack_plan const& plan,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
