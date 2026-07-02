/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "utilities/iterator.cuh"
#include "zorder.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/for_each.h>

namespace {

// pretends to be an array of uint32_t, but really only stores
// the data in an int type with a set number of bits allocated for
// each item (num_bits_per_entry)
template <typename data_type>
struct uint_backed_array {
  uint_backed_array() = delete;
  __device__ explicit uint_backed_array(int32_t num_bits_per_entry)
    : data(0),
      num_bits_per_entry(num_bits_per_entry),
      mask(static_cast<uint32_t>((1L << num_bits_per_entry) - 1))
  {
  }

  __device__ uint32_t operator[](int32_t i) const
  {
    int32_t offset = num_bits_per_entry * i;
    return (data >> offset) & mask;
  }

  __device__ void set(int32_t i, uint32_t value)
  {
    int32_t offset        = i * num_bits_per_entry;
    data_type masked_data = data & ~(static_cast<data_type>(mask) << offset);
    data                  = masked_data | (static_cast<data_type>(value & mask) << offset);
  }

 private:
  data_type data;
  int32_t const num_bits_per_entry;
  uint32_t const mask;
};

// Most of the hilbert index code is based off of the work done by David Moten at
// https://github.com/davidmoten/hilbert-curve, which has the following Note in
// the code too
// This algorithm is derived from work done by John Skilling and published
// in "Programming the Hilbert curve". (c) 2004 American Institute of Physics.
// With thanks also to Paul Chernoch who published a C# algorithm for Skilling's
// work on StackOverflow and
// <a href="https://github.com/paulchernoch/HilbertTransformation">GitHub</a>.
__device__ uint64_t to_hilbert_index(uint_backed_array<uint64_t> const& transposed_index,
                                     int32_t const num_bits_per_entry,
                                     int32_t const num_dimensions)
{
  uint64_t b           = 0;
  int32_t const length = num_bits_per_entry * num_dimensions;
  int32_t b_index      = length - 1;
  uint64_t mask        = 1L << (num_bits_per_entry - 1);
  for (int32_t i = 0; i < num_bits_per_entry; i++) {
    for (int32_t j = 0; j < num_dimensions; j++) {
      if ((transposed_index[j] & mask) != 0) { b |= 1L << b_index; }
      b_index--;
    }
    mask >>= 1;
  }
  // b is expected to be BigEndian
  return b;
}

__device__ uint_backed_array<uint64_t> hilbert_transposed_index(
  uint_backed_array<uint64_t> const& point,
  int32_t const num_bits_per_entry,
  int32_t const num_dimensions)
{
  uint32_t const M = 1L << (num_bits_per_entry - 1);
  int32_t const n  = num_dimensions;
  auto x           = point;

  uint32_t p, q, t;
  uint32_t i;
  // Inverse undo
  for (q = M; q > 1; q >>= 1) {
    p = q - 1;
    for (i = 0; i < n; i++) {
      if ((x[i] & q) != 0) {
        x.set(0, x[0] ^ p);  // invert
      } else {
        t = (x[0] ^ x[i]) & p;
        x.set(0, x[0] ^ t);
        x.set(i, x[i] ^ t);
      }
    }
  }  // exchange

  // Gray encode
  for (i = 1; i < n; i++) {
    x.set(i, x[i] ^ x[i - 1]);
  }
  t = 0;
  for (q = M; q > 1; q >>= 1) {
    if ((x[n - 1] & q) != 0) { t ^= q - 1; }
  }

  for (i = 0; i < n; i++) {
    x.set(i, x[i] ^ t);
  }

  return x;
}

}  // namespace

namespace spark_rapids_jni {

std::unique_ptr<cudf::column> interleave_bits(cudf::table_view const& tbl,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  auto num_columns = tbl.num_columns();
  CUDF_EXPECTS(num_columns > 0, "The input table must have at least one column.");
  CUDF_EXPECTS(is_fixed_width(tbl.begin()->type()), "Only fixed width columns can be used");

  auto const type_id        = tbl.begin()->type().id();
  auto const data_type_size = cudf::size_of(tbl.begin()->type());
  CUDF_EXPECTS(
    std::all_of(tbl.begin(),
                tbl.end(),
                [type_id](cudf::column_view const& col) { return col.type().id() == type_id; }),
    "All columns of the input table must be the same type.");

  // Because the input is a table we know that they all have the same length.
  auto num_rows = tbl.num_rows();

  const cudf::size_type max_bytes_allowed = std::numeric_limits<cudf::size_type>::max();

  int64_t total_output_size = static_cast<int64_t>(num_rows) * data_type_size * num_columns;
  CUDF_EXPECTS(total_output_size <= max_bytes_allowed, "Input is too large to process");

  cudf::size_type output_size = static_cast<cudf::size_type>(total_output_size);

  auto input_dv = cudf::table_device_view::create(tbl, stream);

  auto output_data_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::UINT8}, output_size, cudf::mask_state::UNALLOCATED, stream, mr);

  auto output_dv_ptr = cudf::mutable_column_device_view::create(*output_data_col, stream);

  thrust::for_each_n(
    rmm::exec_policy(stream),
    cuda::make_counting_iterator<cudf::size_type>(0),
    output_size,
    [col = *output_dv_ptr, num_columns, data_type_size, input = *input_dv] __device__(
      cudf::size_type ret_idx) {
      // The most significant byte needs to come from the most significant column, so we switch the
      // order of the output bytes to match that
      cudf::size_type const flipped_start_byte_index = (ret_idx / num_columns) * num_columns;
      cudf::size_type const flipped_ret_idx =
        flipped_start_byte_index + (num_columns - 1 - (ret_idx - flipped_start_byte_index));

      uint8_t ret_byte = 0;
      for (cudf::size_type output_bit_offset = 7; output_bit_offset >= 0; output_bit_offset--) {
        // The index (in bits) of the output bit we are computing right now
        int64_t const output_bit_index = flipped_ret_idx * 8L + output_bit_offset;

        // The most significant bit should come from the most significant column, but 0 is
        // our most significant column, so switch the order of the columns.
        cudf::size_type const column_idx = num_columns - 1 - (output_bit_index % num_columns);
        auto column                      = input.column(column_idx);

        // Also we need to convert the endian byte order when we read the bytes.
        int64_t const bit_index_within_column               = output_bit_index / num_columns;
        cudf::size_type const little_endian_read_byte_index = bit_index_within_column / 8;
        cudf::size_type const read_bit_offset               = bit_index_within_column % 8;
        cudf::size_type const input_row_number     = little_endian_read_byte_index / data_type_size;
        cudf::size_type const start_row_byte_index = input_row_number * data_type_size;
        cudf::size_type const read_byte_index =
          start_row_byte_index +
          (data_type_size - 1 - (little_endian_read_byte_index - start_row_byte_index));

        uint32_t const byte_data =
          column.is_valid(input_row_number) ? column.data<uint8_t>()[read_byte_index] : 0;
        uint32_t const tmp = ((byte_data >> read_bit_offset) & 1) << output_bit_offset;
        ret_byte           = static_cast<uint8_t>(ret_byte | tmp);
      }
      col.data<uint8_t>()[ret_idx] = ret_byte;
    });

  auto offset_begin   = cuda::make_constant_iterator(data_type_size * num_columns);
  auto offsets_column = std::get<0>(
    cudf::detail::make_offsets_child_column(offset_begin, offset_begin + num_rows, stream, mr));

  return cudf::make_lists_column(
    num_rows, std::move(offsets_column), std::move(output_data_col), 0, rmm::device_buffer());
}

std::unique_ptr<cudf::column> hilbert_index(int32_t const num_bits_per_entry,
                                            cudf::table_view const& tbl,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  auto const num_rows    = tbl.num_rows();
  auto const num_columns = tbl.num_columns();

  CUDF_EXPECTS(num_bits_per_entry > 0 && num_bits_per_entry <= 32,
               "the number of bits must be >0 and <= 32.");
  CUDF_EXPECTS(num_bits_per_entry * num_columns <= 64,
               "we only support up to 64 bits of output right now.");
  CUDF_EXPECTS(num_columns > 0, "at least one column is required.");

  CUDF_EXPECTS(std::all_of(tbl.begin(),
                           tbl.end(),
                           [](cudf::column_view const& col) {
                             return col.type().id() == cudf::type_id::INT32;
                           }),
               "All columns of the input table must be INT32.");

  auto const input_dv = cudf::table_device_view::create(tbl, stream);

  auto output_data_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);

  auto const output_dv_ptr = cudf::mutable_column_device_view::create(*output_data_col, stream);

  thrust::transform(
    rmm::exec_policy(stream),
    cuda::make_counting_iterator<cudf::size_type>(0),
    cuda::make_counting_iterator<cudf::size_type>(num_rows),
    output_dv_ptr->begin<int64_t>(),
    cuda::proclaim_return_type<int64_t>(
      [num_bits_per_entry, num_columns, input = *input_dv] __device__(cudf::size_type row_index) {
        uint_backed_array<uint64_t> row(num_bits_per_entry);
        for (cudf::size_type column_index = 0; column_index < num_columns; column_index++) {
          auto const column   = input.column(column_index);
          uint32_t const data = column.is_valid(row_index) ? column.data<uint32_t>()[row_index] : 0;
          row.set(column_index, data);
        }

        auto const transposed_index =
          hilbert_transposed_index(row, num_bits_per_entry, num_columns);
        return static_cast<int64_t>(
          to_hilbert_index(transposed_index, num_bits_per_entry, num_columns));
      }));

  return output_data_col;
}

}  // namespace spark_rapids_jni
