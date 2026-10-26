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
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace spark_rapids_jni {

/**
 * @brief Copies gathered to be queued together by `copy_batch_async`
 */
struct batched_copies {
  std::vector<void*> dsts;
  std::vector<void const*> srcs;
  std::vector<std::size_t> sizes;
};

/**
 * @brief Add copies of host buffers into a device buffer, packed back to back in order
 *
 * Buffers adjacent in host memory move in a single copy.
 *
 * @param buffers The host buffers to copy
 * @param destination The device buffer to copy into, at least as large as the host buffers combined
 * @param copies The batch the copies are added to
 */
void add_packed_copies(std::span<std::span<uint8_t const> const> buffers,
                       cudf::device_span<uint8_t> destination,
                       batched_copies& copies);

/**
 * @brief Queue a batch of copies in one call
 *
 * The sources and destinations must stay valid until the copies complete on `stream`, and the
 * destinations must be safe to write on it. If this throws, the copies it queued have already
 * completed.
 *
 * @param copies The copies to queue
 * @param stream CUDA stream the copies are queued on
 */
void copy_batch_async(batched_copies const& copies,
                      cuda::stream_ref stream = cudf::get_default_stream());

}  // namespace spark_rapids_jni
