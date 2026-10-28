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

#include "memory_utils.hpp"

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda_runtime_api.h>

namespace spark_rapids_jni {

namespace {

// Each pack's size is rounded up to a multiple of this. Matches libcudf's BUFFER_PADDING_MULTIPLE.
constexpr std::size_t padding_multiple = 8;

struct batched_copies {
  std::vector<void*> dsts;
  std::vector<void const*> srcs;
  std::vector<std::size_t> sizes;
};

// Add copies of `buffers` packed back to back into `destination`. Buffers adjacent in host memory
// move as one copy, but runs never extend across destinations.
void add_packed_copies(std::span<std::span<uint8_t const> const> buffers,
                       cudf::device_span<uint8_t> destination,
                       batched_copies& copies)
{
  auto const first = copies.srcs.size();
  auto* dst        = destination.data();
  for (auto const& buffer : buffers) {
    if (buffer.empty()) { continue; }
    // Contiguous with the current run: extend it instead of starting a new copy.
    if (copies.srcs.size() > first &&
        buffer.data() == static_cast<uint8_t const*>(copies.srcs.back()) + copies.sizes.back()) {
      copies.sizes.back() += buffer.size();
    } else {
      copies.dsts.push_back(dst);
      copies.srcs.push_back(buffer.data());
      copies.sizes.push_back(buffer.size());
    }
    dst += buffer.size();
  }
}

}  // namespace

pack_plan plan_packs(std::span<std::span<uint8_t const> const> buffers, std::size_t max_pack_bytes)
{
  pack_plan plan;
  plan.pack_of.reserve(buffers.size());
  plan.offsets.reserve(buffers.size());
  for (auto const& buffer : buffers) {
    // An empty buffer never starts a pack, so no pack is left holding only empty buffers.
    if (plan.pack_sizes.empty() || (!buffer.empty() && plan.pack_sizes.back() > 0 &&
                                    plan.pack_sizes.back() + buffer.size() > max_pack_bytes)) {
      plan.pack_sizes.push_back(0);
      plan.pack_counts.push_back(0);
    }
    plan.pack_of.push_back(plan.pack_sizes.size() - 1);
    plan.offsets.push_back(plan.pack_sizes.back());
    plan.pack_sizes.back() += buffer.size();
    ++plan.pack_counts.back();
  }
  for (auto& size : plan.pack_sizes) {
    size = (size + padding_multiple - 1) / padding_multiple * padding_multiple;
  }
  return plan;
}

std::vector<std::unique_ptr<rmm::device_buffer>> copy_packed_async(
  std::span<std::span<uint8_t const> const> buffers,
  pack_plan const& plan,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  // Allocate every pack before queueing any copy, so a failed allocation leaves nothing in flight.
  std::vector<std::unique_ptr<rmm::device_buffer>> packs;
  packs.reserve(plan.pack_sizes.size());
  for (auto const size : plan.pack_sizes) {
    packs.push_back(std::make_unique<rmm::device_buffer>(size, stream, mr));
  }

  // Allocate the copies
  batched_copies copies;
  copies.dsts.reserve(buffers.size());
  copies.srcs.reserve(buffers.size());
  copies.sizes.reserve(buffers.size());

  // Gather the copies; each buffer adds at most one.
  std::size_t first = 0;
  for (std::size_t p = 0; p < packs.size(); ++p) {
    auto& pack = *packs[p];
    add_packed_copies(buffers.subspan(first, plan.pack_counts[p]),
                      cudf::device_span<uint8_t>(static_cast<uint8_t*>(pack.data()), pack.size()),
                      copies);
    first += plan.pack_counts[p];
  }

  // Queue the copies in one call.
  if (!copies.dsts.empty()) {
    try {
      CUDF_CUDA_TRY(cudf::detail::memcpy_batch_async(
        copies.dsts.data(), copies.srcs.data(), copies.sizes.data(), copies.dsts.size(), stream));
    } catch (...) {
      // On error, block until any copies already queued finish, since the caller may then free
      // their buffers
      cudaStreamSynchronize(stream.get());
      throw;
    }
  }
  return packs;
}

}  // namespace spark_rapids_jni
