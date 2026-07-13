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

#include "nvtx_ranges.hpp"
#include "uuid.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include <curand_kernel.h>

#include <limits>

namespace spark_rapids_jni {

namespace {

// 36 chars per UUID, e.g.: 123e4567-e89b-12d3-a456-426614174000
constexpr cudf::size_type CHAR_COUNT_PER_UUID = 36;

struct init_curand_fn {
  curandState* state;
  long seed;

  __device__ void operator()(cudf::size_type const idx) const
  { curand_init(seed, idx, 0, state + idx); }
};

__device__ void byte_to_hex(uint8_t byte, char* hex)
{
  uint8_t const nibble = byte >> 4;
  hex[0]               = static_cast<char>(nibble < 10 ? '0' + nibble : 'a' + (nibble - 10));
  byte                 = (byte & 0x0F);
  hex[1]               = byte < 10 ? '0' + byte : 'a' + (byte - 10);
}

/**
 * @brief Converts the most and least significant bits of a UUID into a string format.
 * E.g.: 123e4567-e89b-12d3-a456-426614174000
 */
__device__ void convert_uuid_to_chars(uint64_t most_sig_bits,
                                      uint64_t least_sig_bits,
                                      char* uuid_ptr)
{
  uint64_t tmp = most_sig_bits;
  int idx;
  for (int i = 0; i < 16; i++) {
    if (i == 4 || i == 6 || i == 8 || i == 10) {
      *uuid_ptr = '-';
      uuid_ptr++;
    }

    if (i >= 8) {
      tmp = least_sig_bits;
      idx = i - 8;
    } else {
      idx = i;
    }

    int shift    = (7 - idx) * 8;
    uint8_t byte = static_cast<uint8_t>((tmp >> shift) & 0xFF);
    byte_to_hex(byte, uuid_ptr);
    uuid_ptr += 2;
  }
}

template <int block_size>
__launch_bounds__(block_size) CUDF_KERNEL void generate_uuids_kernel(
  cudf::size_type const row_count, char* uuid_chars, curandState* state, int const num_states)
{
  auto const start_idx = cudf::detail::grid_1d::global_thread_id();
  auto const stride    = cudf::detail::grid_1d::grid_stride();

  if (start_idx >= num_states) { return; }

  curandState local_state = state[start_idx];

  for (cudf::thread_index_type row_idx = start_idx; row_idx < row_count; row_idx += stride) {
    auto const idx = static_cast<cudf::size_type>(row_idx);

    // curand gets 32-bits random number, cast to 64-bits
    uint64_t i1 = static_cast<uint64_t>(curand(&local_state));
    uint64_t i2 = static_cast<uint64_t>(curand(&local_state));
    uint64_t i3 = static_cast<uint64_t>(curand(&local_state));
    uint64_t i4 = static_cast<uint64_t>(curand(&local_state));

    uint64_t const most  = i1 << 32 | i2;
    uint64_t const least = i3 << 32 | i4;

    // set the version bits to 4 (UUID version 4): Truly Random or Pseudo-Random
    auto most_sig_bits = (most & 0xFFFFFFFFFFFF0FFFL) | 0x0000000000004000L;

    // set the variant bits to 2
    auto least_sig_bits = (least | 0x8000000000000000L) & 0xBFFFFFFFFFFFFFFFL;

    // convert the most and least significant bits to a string format
    char* uuid_ptr = uuid_chars + idx * CHAR_COUNT_PER_UUID;
    convert_uuid_to_chars(most_sig_bits, least_sig_bits, uuid_ptr);
  }
}

std::unique_ptr<cudf::column> generate_uuids(cudf::size_type row_count,
                                             long seed,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)

{
  // Check if row_count is positive and does not exceed the maximum limit for a column.
  CUDF_EXPECTS(row_count > 0, "Row count must be positive.");
  CUDF_EXPECTS(row_count <= std::numeric_limits<cudf::size_type>::max() / CHAR_COUNT_PER_UUID,
               "Row count exceeds the maximum limit for UUID generation.",
               std::overflow_error);

  constexpr int block_size = 128;
  auto const num_sms       = cudf::detail::num_multiprocessors();
  int num_blocks_per_sm    = -1;

  // Calculate the maximum number of blocks per multiprocessor
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks_per_sm, generate_uuids_kernel<block_size>, block_size, 0));
  auto num_states = num_sms * num_blocks_per_sm * block_size;

  // Ensure num_states does not exceed row_count
  if (num_states > row_count) { num_states = row_count; }

  // initialize curand states
  rmm::device_uvector<curandState> states(
    num_states, stream, cudf::get_current_device_resource_ref());

  thrust::for_each_n(rmm::exec_policy_nosync(stream),
                     thrust::make_counting_iterator(0),
                     num_states,
                     init_curand_fn{states.data(), seed});

  // generate offsets for the UUIDs
  rmm::device_uvector<cudf::size_type> offsets(row_count + 1, stream, mr);
  thrust::sequence(
    rmm::exec_policy_nosync(stream), offsets.begin(), offsets.end(), 0, CHAR_COUNT_PER_UUID);

  // generate chars for the UUIDs
  auto const num_chars = row_count * CHAR_COUNT_PER_UUID;
  rmm::device_uvector<char> chars(num_chars, stream, mr);
  auto grid = cudf::detail::grid_1d(num_states, block_size);
  generate_uuids_kernel<block_size>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      row_count, chars.data(), states.data(), num_states);

  return cudf::make_strings_column(
    row_count,
    std::make_unique<cudf::column>(std::move(offsets), rmm::device_buffer{}, 0),
    chars.release(),
    0,                    // null count
    rmm::device_buffer{}  // all UUIDs are non-null
  );
}

}  // namespace

std::unique_ptr<cudf::column> random_uuids(int row_count,
                                           long seed,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)

{
  SRJ_FUNC_RANGE();
  return generate_uuids(row_count, seed, stream, mr);
}

}  // namespace spark_rapids_jni
