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

#include <cuda_runtime_api.h>

namespace spark_rapids_jni {

void add_packed_copies(std::span<std::span<uint8_t const> const> buffers,
                       cudf::device_span<uint8_t> destination,
                       batched_copies& copies)
{
  std::size_t total = 0;
  for (auto const& buffer : buffers) {
    total += buffer.size();
  }
  CUDF_EXPECTS(total <= destination.size(), "host buffers do not fit in the destination");

  // Buffers adjacent in host memory land adjacent in the destination, so a run of them moves as
  // one copy. Runs never extend across destinations.
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

void copy_batch_async(batched_copies const& copies, cuda::stream_ref stream)
{
  if (copies.dsts.empty()) { return; }
  try {
    CUDF_CUDA_TRY(cudf::detail::memcpy_batch_async(
      copies.dsts.data(), copies.srcs.data(), copies.sizes.data(), copies.dsts.size(), stream));
  } catch (...) {
    // Block until the copies already queued finish, since the caller may free their buffers
    cudaStreamSynchronize(stream.get());
    throw;
  }
}

}  // namespace spark_rapids_jni
