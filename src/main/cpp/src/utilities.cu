/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "utilities.hpp"

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <vector>

namespace spark_rapids_jni {

bool is_basic_spark_numeric(cudf::data_type type)
{
  return type.id() == cudf::type_id::INT8 || type.id() == cudf::type_id::INT16 ||
         type.id() == cudf::type_id::INT32 || type.id() == cudf::type_id::INT64 ||
         type.id() == cudf::type_id::FLOAT32 || type.id() == cudf::type_id::FLOAT64;
}

std::unique_ptr<cuda::device_buffer<cudf::bitmask_type>> bitmask_bitwise_or(
  std::vector<cudf::device_span<cudf::bitmask_type const>> const& input,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.size() > 0, "Empty input");
  auto const mask_size = (*input.begin()).size();
  CUDF_EXPECTS(
    std::all_of(
      input.begin(), input.end(), [mask_size](auto mask) { return mask.size() == mask_size; }),
    "Encountered size mismatch in inputs");
  if (mask_size == 0) {
    return std::make_unique<cuda::device_buffer<cudf::bitmask_type>>(
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
  }

  // move the pointers to the gpu
  std::vector<cudf::bitmask_type const*> h_input(input.size());
  std::transform(
    input.begin(), input.end(), h_input.begin(), [](auto mask) { return mask.data(); });
  auto d_input = cudf::detail::make_device_uvector_async(
    h_input, stream, rmm::mr::get_current_device_resource_ref());

  std::unique_ptr<cuda::device_buffer<cudf::bitmask_type>> out =
    std::make_unique<cuda::device_buffer<cudf::bitmask_type>>(stream, mr, mask_size, cuda::no_init);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + mask_size,
                    out->data(),
                    cuda::proclaim_return_type<cudf::bitmask_type>(
                      [buffers     = d_input.data(),
                       num_buffers = input.size()] __device__(cudf::size_type word_index) {
                        cudf::bitmask_type out = buffers[0][word_index];
                        for (auto idx = 1; idx < num_buffers; idx++) {
                          out |= buffers[idx][word_index];
                        }
                        return out;
                      }));

  return out;
}

namespace {

// Separate from cudf's stream pool, which could hand the decode the same stream as the copy.
// Never freed, so it outlives CUDA driver teardown at exit.
rmm::cuda_stream_pool& side_stream_pool()
{
  static auto* const pool = new rmm::cuda_stream_pool(rmm::cuda_stream_pool::default_size,
                                                      rmm::cuda_stream::flags::non_blocking);
  return *pool;
}

}  // namespace

cudaEvent_t copy_host_buffers_to_device_async(std::span<std::span<uint8_t const> const> buffers,
                                              cudf::device_span<uint8_t> destination,
                                              bool on_side_stream,
                                              cuda::stream_ref stream)
{
  std::size_t total = 0;
  for (auto const& buffer : buffers) {
    total += buffer.size();
  }
  CUDF_EXPECTS(total <= destination.size(), "host buffers do not fit in the destination");

  auto const copy_stream = on_side_stream ? side_stream_pool().get_stream() : stream;
  if (on_side_stream) {
    // Order the copies behind everything already queued on `stream`, including the
    // destination's allocation and any earlier copy into it.
    copy_stream.wait(stream);
  }

  try {
    // Buffers adjacent in host memory land adjacent in the destination, so a run of them moves as
    // one copy.
    std::vector<void*> dsts;
    std::vector<void const*> srcs;
    std::vector<std::size_t> sizes;
    dsts.reserve(buffers.size());
    srcs.reserve(buffers.size());
    sizes.reserve(buffers.size());
    auto* dst = destination.data();
    for (auto const& buffer : buffers) {
      if (buffer.empty()) { continue; }
      // Contiguous with the current run: extend it instead of starting a new copy.
      if (!srcs.empty() &&
          buffer.data() == static_cast<uint8_t const*>(srcs.back()) + sizes.back()) {
        sizes.back() += buffer.size();
      } else {
        dsts.push_back(dst);
        srcs.push_back(buffer.data());
        sizes.push_back(buffer.size());
      }
      dst += buffer.size();
    }
    if (!dsts.empty()) {
      CUDF_CUDA_TRY(cudf::detail::memcpy_batch_async(
        dsts.data(), srcs.data(), sizes.data(), dsts.size(), copy_stream));
    }

    cudaEvent_t event;
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    if (auto const err = cudaEventRecord(event, copy_stream.get()); err != cudaSuccess) {
      cudaEventDestroy(event);
      CUDF_CUDA_TRY(err);
    }
    return event;
  } catch (...) {
    // Block until the copies already queued finish, since no event reaches the caller to wait on
    cudaStreamSynchronize(copy_stream.get());
    throw;
  }
}

}  // namespace spark_rapids_jni
