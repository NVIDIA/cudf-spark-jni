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

#include "cast_string.hpp"
#include "nvtx_ranges.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace spark_rapids_jni {

namespace detail {
namespace {
CUDF_KERNEL void compute_output_size_kernel(cudf::column_device_view d_longs,
                                            cudf::size_type* d_sizes)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= d_longs.size()) { return; }
  if (d_longs.is_null(tid)) {
    d_sizes[tid] = 0;
    return;
  }
  // If the value is 0, the size should be 1
  d_sizes[tid] = max(1, 64 - __clzll(d_longs.element<int64_t>(tid)));
}

/**
 * @brief CUDA kernel to convert long to binary string representation.
 *
 * Each long is processed by `num_threads_per_row` threads.
 * Each thread processes `64 / num_threads_per_row` bits.
 * `bit_index` begins from LSB to MSB, and `char_index` equals to `str_len - 1 - bit_index`.
 *
 * For example, if `num_threads_per_row` is 32, then each thread processes 2 bits of the long.
 * Suppose the input is 25, which is 0b 0000...0001_1001, and the output size should be 5.
 *
 * lane  0 handles bit_indices [ 0,  1], write '1' to char_index 4 and '0' to char_index 3
 * lane  1 handles bit_indices [ 2,  3], write '0' to char_index 2 and '1' to char_index 1
 * lane  2 handles bit_indices [ 4,  5], write '1' to char_index 0, and '0' won't be written
 * since bit_index 5 >= str_len
 * ...
 * lane 31 handles bit_indices [62, 63], the smallest bit_index exceeds str_len, no write operation
 *
 * The output binary string will be "11001".
 *
 * @param d_longs Device view of the input column containing longs.
 * @param d_chars Pointer to the output character array where binary strings will be stored.
 * @param d_offsets Input offset calculator to manage the offsets in the output character array.
 */
template <int num_threads_per_row>
CUDF_KERNEL void long_to_binary_string_kernel(cudf::column_device_view d_longs,
                                              char* d_chars,
                                              cudf::detail::input_offsetalator d_offsets)
{
  auto const tid     = cudf::detail::grid_1d::global_thread_id();
  auto const row_idx = tid / num_threads_per_row;
  if (row_idx >= d_longs.size()) { return; }

  auto const str_len = d_offsets[row_idx + 1] - d_offsets[row_idx];
  if (str_len == 0) { return; }

  constexpr int num_bits_per_thread = 64 / num_threads_per_row;
  auto const value                  = d_longs.element<int64_t>(row_idx);
  auto const lane_idx               = tid % num_threads_per_row;
  auto const first_bit_index        = lane_idx * num_bits_per_thread;

  char* d_buffer = d_chars + d_offsets[row_idx];
  for (auto bit_index = first_bit_index;
       bit_index < first_bit_index + num_bits_per_thread && bit_index < str_len;
       ++bit_index) {
    auto const char_index = str_len - 1 - bit_index;
    d_buffer[char_index]  = '0' + ((value & (1ULL << bit_index)) >> bit_index);
  }
}

}  // namespace

std::unique_ptr<cudf::column> long_to_binary_string(cudf::column_view const& input,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::INT64, "Input column must be long type");

  if (input.is_empty()) return cudf::make_empty_column(cudf::type_id::STRING);

  auto const strings_count = input.size();
  auto const d_column =
    cudf::column_device_view::create(input, stream, cudf::get_current_device_resource_ref());

  // The following code is adapted from `cudf::strings::detail::make_strings_children()`
  // Compute the output sizes
  auto output_sizes         = rmm::device_uvector<cudf::size_type>(strings_count, stream);
  cudf::size_type* d_sizes  = output_sizes.data();
  auto constexpr block_size = 256;
  auto grid                 = cudf::detail::grid_1d{strings_count, block_size};
  compute_output_size_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(*d_column,
                                                                                 d_sizes);

  // Convert the sizes to offsets
  auto [offsets, bytes] = cudf::strings::detail::make_offsets_child_column(
    output_sizes.begin(), output_sizes.end(), stream, mr);
  cudf::detail::input_offsetalator d_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());

  // Now build the chars column
  rmm::device_uvector<char> chars(bytes, stream, mr);
  char* d_chars = chars.data();

  // Fill in the chars data
  constexpr int num_threads_per_row = 32;
  auto new_grid = cudf::detail::grid_1d{strings_count * num_threads_per_row, block_size};
  if (bytes > 0) {
    long_to_binary_string_kernel<num_threads_per_row>
      <<<new_grid.num_blocks, block_size, 0, stream.value()>>>(*d_column, d_chars, d_offsets);
  }

  return cudf::make_strings_column(input.size(),
                                   std::move(offsets),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::copy_bitmask(input, stream, mr));
}

}  // namespace detail

// external API
std::unique_ptr<cudf::column> long_to_binary_string(cudf::column_view const& input,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return detail::long_to_binary_string(input, stream, mr);
}

}  // namespace spark_rapids_jni
