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
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <cstdint>

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

/**
 * @brief Characters validated or decoded by one thread.
 *
 * When every row of a batch fits in a chunk, one thread per row avoids searching for the row of
 * each chunk. Otherwise, the batch is split into fixed-size character chunks, so that one long row
 * does not serialize the batch on a single thread.
 */
constexpr int64_t chars_per_thread = 32;

/**
 * @brief Functor to validate or decode the characters `[char_idx, stop)` of one row.
 *
 * Without an output, it clears the validity bit of a row with an invalid character. With an
 * output, it writes each byte of a valid row whose low nibble is in the range; an odd-length
 * string has an implicit leading zero nibble.
 */
struct hex_decode_fn {
  cudf::detail::input_offsetalator d_in_offsets;
  char const* d_chars;
  cudf::bitmask_type* d_mask;
  cudf::size_type const* d_out_offsets;
  uint8_t* d_out;

  __device__ void operator()(cudf::size_type row, int64_t char_idx, int64_t stop) const
  {
    if (!cudf::bit_is_set(d_mask, row)) { return; }
    if (d_out == nullptr) {
      for (; char_idx < stop; ++char_idx) {
        if (hex_digit(static_cast<unsigned char>(d_chars[char_idx])) < 0) {
          cudf::clear_bit(d_mask, row);
          return;
        }
      }
      return;
    }
    auto const begin = d_in_offsets[row];
    auto const odd   = (d_in_offsets[row + 1] - begin) & 1;
    auto* out        = d_out + d_out_offsets[row];
    // Start at the low nibble of the first byte in the range.
    for (char_idx += ((char_idx - begin + odd) & 1) == 0; char_idx < stop; char_idx += 2) {
      auto const high =
        char_idx == begin ? 0 : hex_digit(static_cast<unsigned char>(d_chars[char_idx - 1]));
      auto const low                    = hex_digit(static_cast<unsigned char>(d_chars[char_idx]));
      out[(char_idx - begin + odd) / 2] = static_cast<uint8_t>((high << 4) | low);
    }
  }
};

struct hex_row_fn {
  hex_decode_fn decode;

  __device__ void operator()(cudf::size_type row) const
  {
    decode(row, decode.d_in_offsets[row], decode.d_in_offsets[row + 1]);
  }
};

/**
 * @brief Functor to validate or decode a fixed-size chunk of input characters.
 *
 * Each thread finds the row of its first character and walks forward from there.
 */
struct hex_chunk_fn {
  hex_decode_fn decode;
  cudf::size_type num_rows;
  int64_t first_char;
  int64_t last_char;

  __device__ void operator()(int64_t chunk) const
  {
    auto const d_in_offsets = decode.d_in_offsets;
    auto char_idx           = first_char + chunk * chars_per_thread;
    auto const end_char     = cuda::std::min(char_idx + chars_per_thread, last_char);
    auto const row_end =
      thrust::upper_bound(thrust::seq, d_in_offsets, d_in_offsets + num_rows + 1, char_idx);
    auto row = static_cast<cudf::size_type>(cuda::std::distance(d_in_offsets, row_end) - 1);
    while (char_idx < end_char) {
      auto const stop = cuda::std::min(d_in_offsets[row + 1], end_char);
      decode(row, char_idx, stop);
      char_idx = stop;
      ++row;
    }
  }
};

struct row_length_fn {
  cudf::detail::input_offsetalator d_in_offsets;

  __device__ int64_t operator()(cudf::size_type row) const
  {
    return d_in_offsets[row + 1] - d_in_offsets[row];
  }
};

/**
 * @brief Run `decode` over all rows, one thread per row or per character chunk.
 */
void for_each_hex_row(hex_decode_fn const& decode,
                      cudf::size_type num_rows,
                      int64_t first_char,
                      int64_t last_char,
                      bool by_row,
                      cuda::stream_ref stream)
{
  auto const policy = rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref());
  if (by_row) {
    thrust::for_each_n(
      policy, thrust::make_counting_iterator(cudf::size_type{0}), num_rows, hex_row_fn{decode});
    return;
  }
  auto const chunks = (last_char - first_char + chars_per_thread - 1) / chars_per_thread;
  thrust::for_each_n(policy,
                     thrust::make_counting_iterator(int64_t{0}),
                     chunks,
                     hex_chunk_fn{decode, num_rows, first_char, last_char});
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

  auto valid_mask = input.has_nulls() ? cudf::copy_bitmask(input.parent(), stream, mr)
                                      : cudf::create_null_mask(
                                          input.size(), cudf::mask_state::ALL_VALID, stream, mr);
  auto* d_mask    = reinterpret_cast<cudf::bitmask_type*>(valid_mask.data());
  auto const first_char =
    cudf::strings::detail::get_offset_value(input.offsets(), input.offset(), stream);
  auto const last_char =
    cudf::strings::detail::get_offset_value(input.offsets(), input.offset() + input.size(), stream);
  auto const d_in_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
  auto const rows = thrust::make_counting_iterator(cudf::size_type{0});
  // No row is longer than all the characters, and a batch with a longer average row needs chunks.
  auto const num_chars = last_char - first_char;
  auto const policy    = rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref());
  auto const by_row    = num_chars <= chars_per_thread ||
                      (num_chars <= chars_per_thread * input.size() &&
                       thrust::transform_reduce(policy,
                                                rows,
                                                rows + input.size(),
                                                row_length_fn{d_in_offsets},
                                                int64_t{0},
                                                cuda::maximum{}) <= chars_per_thread);
  auto decode = hex_decode_fn{d_in_offsets, input.chars_begin(stream), d_mask, nullptr, nullptr};
  // Validate before sizing so that invalid rows do not reserve any output bytes.
  for_each_hex_row(decode, input.size(), first_char, last_char, by_row, stream);

  auto const d_column = cudf::column_device_view::create(
    input.parent(), stream, cudf::get_current_device_resource_ref());
  auto const sizes = thrust::make_transform_iterator(rows, decoded_size_fn{*d_column, d_mask});
  auto [offsets, total_bytes] =
    cudf::detail::make_offsets_child_column(sizes, sizes + input.size(), stream, mr);

  auto bytes = rmm::device_uvector<uint8_t>(total_bytes, stream, mr);
  if (total_bytes > 0) {
    decode.d_out_offsets = offsets->view().begin<cudf::size_type>();
    decode.d_out         = bytes.data();
    for_each_hex_row(decode, input.size(), first_char, last_char, by_row, stream);
  }

  auto const null_count = cudf::null_count(d_mask, 0, input.size(), stream);
  if (null_count == 0) { valid_mask = cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED); }
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
