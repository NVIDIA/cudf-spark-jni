/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/stream>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace spark_rapids_jni {

namespace detail {
namespace {

/**
 * @brief Functor to compute output sizes from input offsets.
 *
 * Each input byte produces 2 hex characters, so output size = input byte count * 2.
 */
struct output_size_fn {
  cudf::detail::input_offsetalator d_in_offsets;

  __device__ cudf::size_type operator()(cudf::size_type idx) const
  {
    return static_cast<cudf::size_type>((d_in_offsets[idx + 1] - d_in_offsets[idx]) * 2);
  }
};

/**
 * @brief Functor to convert each byte of a string to its 2-character hex representation.
 *
 * Output position is derived from the input string pointer, avoiding a separate
 * output offsets array read: out_offset = (in_ptr - chars_begin) * 2.
 */
struct write_hex_fn {
  cudf::column_device_view d_strings;
  char const* d_chars_begin;
  char* d_out;

  __device__ void operator()(cudf::size_type idx) const
  {
    if (d_strings.is_null(idx)) { return; }
    auto const d_str  = d_strings.element<cudf::string_view>(idx);
    auto const nbytes = d_str.size_bytes();
    auto const* in    = reinterpret_cast<uint8_t const*>(d_str.data());
    auto* out         = d_out + (d_str.data() - d_chars_begin) * 2;
    for (cudf::size_type i = 0; i < nbytes; ++i) {
      uint8_t const byte = in[i];
      uint8_t const hi   = byte >> 4;
      uint8_t const lo   = byte & 0x0F;
      out[i * 2]         = hi + (hi < 10 ? '0' : 'A' - 10);
      out[i * 2 + 1]     = lo + (lo < 10 ? '0' : 'A' - 10);
    }
  }
};

__device__ int hex_digit(unsigned char value)
{
  if (value >= '0' && value <= '9') { return value - '0'; }
  if (value >= 'A' && value <= 'F') { return value - 'A' + 10; }
  if (value >= 'a' && value <= 'f') { return value - 'a' + 10; }
  return -1;
}

struct valid_hex_fn {
  cudf::column_device_view d_strings;

  __device__ bool operator()(cudf::size_type idx) const
  {
    if (d_strings.is_null(idx)) { return false; }
    auto const value = d_strings.element<cudf::string_view>(idx);
    for (cudf::size_type char_idx = 0; char_idx < value.size_bytes(); ++char_idx) {
      if (hex_digit(static_cast<unsigned char>(value.data()[char_idx])) < 0) { return false; }
    }
    return true;
  }
};

struct decoded_size_fn {
  cudf::column_device_view d_strings;
  cudf::bitmask_type const* valid_mask;

  __device__ cudf::size_type operator()(cudf::size_type idx) const
  {
    if (!cudf::bit_is_set(valid_mask, idx)) { return 0; }
    auto const length = d_strings.element<cudf::string_view>(idx).size_bytes();
    return length / 2 + (length & 1);
  }
};

struct write_bytes_fn {
  cudf::column_device_view d_strings;
  cudf::bitmask_type const* valid_mask;
  cudf::size_type const* offsets;
  uint8_t* output;
  cudf::size_type max_serial_length;

  __device__ void operator()(cudf::size_type idx) const
  {
    if (!cudf::bit_is_set(valid_mask, idx)) { return; }
    auto const value  = d_strings.element<cudf::string_view>(idx);
    auto const length = value.size_bytes();
    if (length >= max_serial_length) { return; }
    auto* out = output + offsets[idx];
    cudf::size_type char_idx{0};
    cudf::size_type byte_idx{0};
    if ((length & 1) != 0) {
      out[byte_idx++] = static_cast<uint8_t>(hex_digit(value.data()[char_idx++]));
    }
    while (char_idx < length) {
      auto const high = hex_digit(value.data()[char_idx++]);
      auto const low  = hex_digit(value.data()[char_idx++]);
      out[byte_idx++] = static_cast<uint8_t>((high << 4) | low);
    }
  }
};

struct row_length_fn {
  cudf::column_device_view d_strings;

  __device__ cudf::size_type operator()(cudf::size_type idx) const
  {
    return d_strings.is_null(idx) ? 0 : d_strings.element<cudf::string_view>(idx).size_bytes();
  }
};

struct long_row_fn {
  cudf::column_device_view d_strings;
  cudf::size_type minimum_length;

  __device__ bool operator()(cudf::size_type idx) const
  {
    return !d_strings.is_null(idx) &&
           d_strings.element<cudf::string_view>(idx).size_bytes() >= minimum_length;
  }
};

struct short_row_size_fn {
  cudf::column_device_view d_strings;
  cudf::size_type maximum_length;

  __device__ cudf::size_type operator()(cudf::size_type idx) const
  {
    if (d_strings.is_null(idx)) { return -1; }
    auto const value = d_strings.element<cudf::string_view>(idx);
    if (value.size_bytes() >= maximum_length) { return -1; }
    for (cudf::size_type char_idx = 0; char_idx < value.size_bytes(); ++char_idx) {
      if (hex_digit(static_cast<unsigned char>(value.data()[char_idx])) < 0) { return -1; }
    }
    auto const length = value.size_bytes();
    return length / 2 + (length & 1);
  }
};

constexpr int decode_block_size                       = 256;
constexpr cudf::size_type max_direct_cooperative_rows = 4096;
constexpr cudf::size_type max_all_cooperative_rows    = 16384;

CUDF_KERNEL void validate_hex_rows(cudf::column_device_view strings,
                                   cudf::size_type* row_sizes,
                                   cudf::size_type const* selected_rows)
{
  auto const row =
    selected_rows == nullptr ? static_cast<cudf::size_type>(blockIdx.x) : selected_rows[blockIdx.x];
  if (row >= strings.size()) { return; }

  __shared__ int valid;
  auto const input_valid = !strings.is_null(row);
  if (threadIdx.x == 0) { valid = input_valid ? 1 : 0; }
  __syncthreads();

  if (input_valid) {
    auto const value = strings.element<cudf::string_view>(row);
    for (int64_t char_idx = threadIdx.x; char_idx < value.size_bytes(); char_idx += blockDim.x) {
      if (hex_digit(static_cast<unsigned char>(value.data()[char_idx])) < 0) {
        atomicExch(&valid, 0);
        break;
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    auto const length = input_valid ? strings.element<cudf::string_view>(row).size_bytes() : 0;
    row_sizes[row]    = valid != 0 ? length / 2 + (length & 1) : -1;
  }
}

struct valid_decoded_size_fn {
  cudf::size_type const* row_sizes;

  __device__ bool operator()(cudf::size_type row) const { return row_sizes[row] >= 0; }
};

struct decoded_size_from_rows_fn {
  cudf::size_type const* row_sizes;

  __device__ cudf::size_type operator()(cudf::size_type row) const
  {
    auto const size = row_sizes[row];
    return size > 0 ? size : 0;
  }
};

CUDF_KERNEL void decode_hex_rows(cudf::column_device_view strings,
                                 cudf::size_type const* row_sizes,
                                 cudf::size_type const* offsets,
                                 uint8_t* output,
                                 cudf::size_type const* selected_rows)
{
  auto const row =
    selected_rows == nullptr ? static_cast<cudf::size_type>(blockIdx.x) : selected_rows[blockIdx.x];
  if (row >= strings.size() || row_sizes[row] <= 0) { return; }

  auto const value  = strings.element<cudf::string_view>(row);
  auto const length = value.size_bytes();
  auto const odd    = length & 1;
  auto* out         = output + offsets[row];
  for (cudf::size_type byte_idx = threadIdx.x; byte_idx < row_sizes[row]; byte_idx += blockDim.x) {
    if (odd != 0 && byte_idx == 0) {
      out[byte_idx] = static_cast<uint8_t>(hex_digit(value.data()[0]));
    } else {
      auto const char_idx = byte_idx * 2 - odd;
      auto const high     = hex_digit(value.data()[char_idx]);
      auto const low      = hex_digit(value.data()[char_idx + 1]);
      out[byte_idx]       = static_cast<uint8_t>((high << 4) | low);
    }
  }
}

}  // namespace

std::unique_ptr<cudf::column> bytes_to_hex(cudf::strings_column_view const& input,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(cudf::type_id::STRING); }

  // Build output offsets from input offsets. Each byte produces 2 hex chars,
  // so output_size[i] = (input_offset[i+1] - input_offset[i]) * 2.
  // make_offsets_child_column handles the INT32/INT64 switch for large strings.
  auto const d_in_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets());
  auto sizes_itr = thrust::make_transform_iterator(
    thrust::make_counting_iterator(cudf::size_type{0}), output_size_fn{d_in_offsets});
  auto [offsets_column, total_bytes] = cudf::strings::detail::make_offsets_child_column(
    sizes_itr, sizes_itr + input.size(), stream, mr);

  // Write hex chars in a single pass.
  auto chars = rmm::device_uvector<char>(total_bytes, stream, mr);
  if (total_bytes > 0) {
    auto const d_column = cudf::column_device_view::create(
      input.parent(), stream, cudf::get_current_device_resource_ref());
    thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     thrust::make_counting_iterator(cudf::size_type{0}),
                     thrust::make_counting_iterator(input.size()),
                     write_hex_fn{*d_column, input.chars_begin(stream), chars.data()});
  }

  return cudf::make_strings_column(input.size(),
                                   std::move(offsets_column),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::copy_bitmask(input.parent(), stream, mr));
}

std::unique_ptr<cudf::column> hex_to_bytes(cudf::strings_column_view const& input,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) {
    return cudf::make_empty_lists_column(cudf::data_type{cudf::type_id::UINT8});
  }

  auto const d_column = cudf::column_device_view::create(
    input.parent(), stream, cudf::get_current_device_resource_ref());
  auto const indices = thrust::make_counting_iterator(cudf::size_type{0});
  if (input.size() > max_direct_cooperative_rows) {
    auto const max_row_length = thrust::transform_reduce(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      indices,
      indices + input.size(),
      row_length_fn{*d_column},
      cudf::size_type{0},
      cuda::maximum{});
    auto const cooperative_threshold =
      std::max<cudf::size_type>(256, std::min<cudf::size_type>(512, input.size() / 64));
    if (max_row_length < cooperative_threshold) {
      auto [valid_mask, null_count] = cudf::detail::valid_if(
        indices, indices + input.size(), valid_hex_fn{*d_column}, stream, mr);
      auto const* mask_ptr = reinterpret_cast<cudf::bitmask_type const*>(valid_mask.data());
      auto sizes = thrust::make_transform_iterator(indices, decoded_size_fn{*d_column, mask_ptr});
      auto [offsets, total_bytes] =
        cudf::detail::make_offsets_child_column(sizes, sizes + input.size(), stream, mr);

      auto bytes = rmm::device_uvector<uint8_t>(total_bytes, stream, mr);
      if (total_bytes > 0) {
        thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         indices,
                         indices + input.size(),
                         write_bytes_fn{*d_column,
                                        mask_ptr,
                                        offsets->view().begin<cudf::size_type>(),
                                        bytes.data(),
                                        std::numeric_limits<cudf::size_type>::max()});
      }

      auto child = std::make_unique<cudf::column>(
        std::move(bytes), cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED), 0);
      return cudf::make_lists_column(
        input.size(), std::move(offsets), std::move(child), null_count, std::move(valid_mask));
    }

    if (input.size() > max_all_cooperative_rows) {
      auto long_rows = rmm::device_uvector<cudf::size_type>(input.size(), stream, mr);
      auto const long_rows_end =
        thrust::copy_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        indices,
                        indices + input.size(),
                        long_rows.begin(),
                        long_row_fn{*d_column, cooperative_threshold});
      auto const long_row_count = static_cast<cudf::size_type>(long_rows_end - long_rows.begin());
      auto row_sizes            = rmm::device_uvector<cudf::size_type>(input.size(), stream, mr);
      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        indices,
                        indices + input.size(),
                        row_sizes.begin(),
                        short_row_size_fn{*d_column, cooperative_threshold});
      validate_hex_rows<<<long_row_count, decode_block_size, 0, stream.get()>>>(
        *d_column, row_sizes.data(), long_rows.data());

      auto [valid_mask, null_count] = cudf::detail::valid_if(
        indices, indices + input.size(), valid_decoded_size_fn{row_sizes.data()}, stream, mr);
      auto sizes =
        thrust::make_transform_iterator(indices, decoded_size_from_rows_fn{row_sizes.data()});
      auto [offsets, total_bytes] =
        cudf::detail::make_offsets_child_column(sizes, sizes + input.size(), stream, mr);

      auto bytes = rmm::device_uvector<uint8_t>(total_bytes, stream, mr);
      if (total_bytes > 0) {
        auto const* mask_ptr = reinterpret_cast<cudf::bitmask_type const*>(valid_mask.data());
        thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         indices,
                         indices + input.size(),
                         write_bytes_fn{*d_column,
                                        mask_ptr,
                                        offsets->view().begin<cudf::size_type>(),
                                        bytes.data(),
                                        cooperative_threshold});
        decode_hex_rows<<<long_row_count, decode_block_size, 0, stream.get()>>>(
          *d_column,
          row_sizes.data(),
          offsets->view().begin<cudf::size_type>(),
          bytes.data(),
          long_rows.data());
      }

      auto child = std::make_unique<cudf::column>(
        std::move(bytes), cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED), 0);
      return cudf::make_lists_column(
        input.size(), std::move(offsets), std::move(child), null_count, std::move(valid_mask));
    }
  }

  auto row_sizes = rmm::device_uvector<cudf::size_type>(input.size(), stream, mr);
  validate_hex_rows<<<input.size(), decode_block_size, 0, stream.get()>>>(
    *d_column, row_sizes.data(), nullptr);

  auto [valid_mask, null_count] = cudf::detail::valid_if(
    indices, indices + input.size(), valid_decoded_size_fn{row_sizes.data()}, stream, mr);
  auto sizes =
    thrust::make_transform_iterator(indices, decoded_size_from_rows_fn{row_sizes.data()});
  auto [offsets, total_bytes] =
    cudf::detail::make_offsets_child_column(sizes, sizes + input.size(), stream, mr);

  auto bytes = rmm::device_uvector<uint8_t>(total_bytes, stream, mr);
  if (total_bytes > 0) {
    decode_hex_rows<<<input.size(), decode_block_size, 0, stream.get()>>>(
      *d_column, row_sizes.data(), offsets->view().begin<cudf::size_type>(), bytes.data(), nullptr);
  }

  auto child = std::make_unique<cudf::column>(
    std::move(bytes), cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED), 0);
  return cudf::make_lists_column(
    input.size(), std::move(offsets), std::move(child), null_count, std::move(valid_mask));
}

}  // namespace detail

std::unique_ptr<cudf::column> bytes_to_hex(cudf::strings_column_view const& input,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return detail::bytes_to_hex(input, stream, mr);
}

std::unique_ptr<cudf::column> hex_to_bytes(cudf::strings_column_view const& input,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return detail::hex_to_bytes(input, stream, mr);
}

}  // namespace spark_rapids_jni
