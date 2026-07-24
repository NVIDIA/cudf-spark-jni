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

#pragma once

#include "protobuf/protobuf_device_helpers.cuh"
#include "protobuf/protobuf_host_helpers.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda/functional>
#include <cuda/std/bit>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace spark_rapids_jni::protobuf::detail {

// ============================================================================
// Pass 2: Extract data kernels
// ============================================================================

// ============================================================================
// Data Extraction Location Providers
// ============================================================================

struct top_level_location_provider {
  cudf::size_type const* offsets;
  cudf::size_type base_offset;
  field_location const* locations;
  int field_idx;
  int num_fields;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto loc = locations[flat_index(thread_idx, num_fields, field_idx)];
    if (loc.offset >= 0) { data_offset = offsets[thread_idx] - base_offset + loc.offset; }
    return loc;
  }
};

struct repeated_location_provider {
  cudf::size_type const* row_offsets;
  cudf::size_type base_offset;
  field_occurrence const* occurrences;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto occ    = occurrences[thread_idx];
    data_offset = row_offsets[occ.row_idx] - base_offset + occ.offset;
    return {occ.offset, occ.length};
  }
};

struct nested_location_provider {
  cudf::size_type const* row_offsets;
  cudf::size_type base_offset;
  field_location const* parent_locations;
  field_location const* child_locations;
  int field_idx;
  int num_fields;

  // Rebase child offsets from the parent message to the row for recursive STRUCT decode.
  __device__ inline field_location get_rebased_child_location(int thread_idx,
                                                              protobuf_error* error_flag) const
  {
    auto ploc = parent_locations[thread_idx];
    auto cloc = child_locations[flat_index(thread_idx, num_fields, field_idx)];
    if (ploc.offset < 0 || cloc.offset < 0) { return {-1, 0}; }

    auto const offset = static_cast<int64_t>(ploc.offset) + cloc.offset;
    if (offset > cuda::std::numeric_limits<int32_t>::max()) {
      if (error_flag != nullptr) { set_error_once(error_flag, protobuf_error::OVERFLOW); }
      return {-1, 0};
    }
    return {static_cast<int32_t>(offset), cloc.length};
  }

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto child_parent_loc = get_rebased_child_location(thread_idx, nullptr);
    if (child_parent_loc.offset < 0) { return child_parent_loc; }

    data_offset = row_offsets[thread_idx] - base_offset + child_parent_loc.offset;
    return child_locations[flat_index(thread_idx, num_fields, field_idx)];
  }

  __device__ inline bool valid(int thread_idx) const
  {
    return get_rebased_child_location(thread_idx, nullptr).offset >= 0;
  }
};

struct nested_repeated_location_provider {
  cudf::size_type const* row_offsets;
  cudf::size_type base_offset;
  field_location const* parent_locations;
  field_occurrence const* occurrences;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto occ  = occurrences[thread_idx];
    auto ploc = parent_locations[occ.row_idx];
    if (ploc.offset >= 0) {
      data_offset = row_offsets[occ.row_idx] - base_offset + ploc.offset + occ.offset;
      return {occ.offset, occ.length};
    }
    data_offset = 0;
    return {-1, 0};
  }
};

struct repeated_msg_child_location_provider {
  cudf::size_type const* row_offsets;
  cudf::size_type base_offset;
  field_location const* msg_locations;
  field_location const* child_locations;
  int field_idx;
  int num_fields;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto mloc = msg_locations[thread_idx];
    auto cloc = child_locations[flat_index(thread_idx, num_fields, field_idx)];
    if (mloc.offset >= 0 && cloc.offset >= 0) {
      data_offset = row_offsets[thread_idx] - base_offset + mloc.offset + cloc.offset;
    } else {
      cloc.offset = -1;
    }
    return cloc;
  }
};

__device__ inline scalar_value_input resolve_scalar_value(uint8_t const* message_data,
                                                          field_location location,
                                                          int32_t data_offset)
{
  return {location.offset < 0 ? nullptr : message_data + data_offset,
          location.length,
          location.offset >= 0};
}

template <typename OutputType, bool ZigZag = false>
  requires std::is_integral_v<OutputType>
__device__ inline void decode_varint_value(scalar_value_input input,
                                           int index,
                                           scalar_decode_options<OutputType> options,
                                           scalar_value_output<OutputType> output)
{
  if (!input.present) {
    if (options.has_default) {
      write_varint_value(&output.values[index], static_cast<uint64_t>(options.default_value));
      if (output.valid) output.valid[index] = true;
    } else {
      if (output.valid) output.valid[index] = false;
    }
    return;
  }

  uint8_t const* cur     = input.data;
  uint8_t const* cur_end = cur + input.length;

  uint64_t v;
  int n;
  if (!read_varint(cur, cur_end, v, n)) {
    set_error_once(output.error, protobuf_error::VARINT);
    if (output.valid) output.valid[index] = false;
    return;
  }

  if constexpr (ZigZag) { v = (v >> 1) ^ (-(v & 1)); }
  write_varint_value(&output.values[index], v);
  if (output.valid) output.valid[index] = true;
}

template <typename OutputType>
__device__ inline void decode_fixed_value(scalar_value_input input,
                                          int index,
                                          scalar_decode_options<OutputType> options,
                                          scalar_value_output<OutputType> output)
{
  static_assert(sizeof(OutputType) == 4 || sizeof(OutputType) == 8,
                "Fixed-width protobuf extraction requires a 32-bit or 64-bit output type");
  if (!input.present) {
    if (options.has_default) {
      output.values[index] = options.default_value;
      if (output.valid) output.valid[index] = true;
    } else {
      if (output.valid) output.valid[index] = false;
    }
    return;
  }

  if (input.length < static_cast<int32_t>(sizeof(OutputType))) {
    set_error_once(output.error, protobuf_error::FIXED_LEN);
    if (output.valid) output.valid[index] = false;
    return;
  }

  using raw_type       = cuda::std::conditional_t<sizeof(OutputType) == 4, uint32_t, uint64_t>;
  auto const raw       = load_le<raw_type>(input.data);
  output.values[index] = cuda::std::bit_cast<OutputType>(raw);
  if (output.valid) output.valid[index] = true;
}

template <typename OutputType, auto DecodeFn, typename LocationProvider>
CUDF_KERNEL void extract_scalar_kernel(uint8_t const* message_data,
                                       LocationProvider loc_provider,
                                       int total_items,
                                       scalar_value_output<OutputType> output,
                                       scalar_decode_options<OutputType> options)
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_items) return;

  int32_t data_offset = 0;
  auto loc            = loc_provider.get(idx, data_offset);
  DecodeFn(resolve_scalar_value(message_data, loc, data_offset), idx, options, output);
}

// ============================================================================
// Batched scalar extraction — one 2D kernel for N fields of the same type
// ============================================================================

template <typename OutputType, auto DecodeFn>
CUDF_KERNEL void extract_scalar_batched_kernel(batched_scalar_input_view<OutputType> input)
{
  int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int fi  = static_cast<int>(blockIdx.y);
  if (row >= input.input.num_rows || fi >= input.num_descriptors) return;

  auto const& desc = input.descriptors[fi];
  top_level_location_provider loc_provider{input.input.row_offsets,
                                           input.input.base_offset,
                                           input.locations,
                                           desc.loc_field_idx,
                                           input.num_location_fields};
  int32_t data_offset = 0;
  auto loc            = loc_provider.get(row, data_offset);
  DecodeFn(resolve_scalar_value(input.input.message_data, loc, data_offset),
           row,
           desc.options,
           {desc.output, desc.valid, input.error});
}

// ============================================================================

template <typename LocationProvider>
CUDF_KERNEL void extract_lengths_kernel(LocationProvider loc_provider,
                                        int total_items,
                                        int32_t* out_lengths,
                                        bool has_default       = false,
                                        int32_t default_length = 0)
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_items) return;

  int32_t data_offset = 0;
  auto loc            = loc_provider.get(idx, data_offset);

  if (loc.offset >= 0) {
    out_lengths[idx] = loc.length;
  } else if (has_default) {
    out_lengths[idx] = default_length;
  } else {
    out_lengths[idx] = 0;
  }
}

// ============================================================================
// Host wrapper declarations for kernel launches (repeated + nested)
// ============================================================================

void launch_count_repeated_fields(cudf::column_device_view const& d_in,
                                  field_scan_view fields,
                                  protobuf_error* error_flag,
                                  bool* row_has_invalid_data,
                                  rmm::cuda_stream_view stream);

void launch_scan_all_field_occurrences(cudf::column_device_view const& d_in,
                                       field_occurrence_scan_view fields,
                                       protobuf_error* error_flag,
                                       rmm::cuda_stream_view stream);

void launch_extract_strided_locations(field_location const* nested_locations,
                                      int field_idx,
                                      int num_fields,
                                      field_location* parent_locs,
                                      int num_rows,
                                      rmm::cuda_stream_view stream);

void launch_scan_nested_message_fields(protobuf_input_view input,
                                       nested_parent_view parent,
                                       field_scan_view fields,
                                       protobuf_error* error_flag,
                                       bool* row_has_invalid_data,
                                       rmm::cuda_stream_view stream);

void launch_scan_all_field_occurrences_in_nested(protobuf_input_view input,
                                                 nested_parent_view parent,
                                                 field_occurrence_scan_view fields,
                                                 protobuf_error* error_flag,
                                                 rmm::cuda_stream_view stream);

void launch_compute_grandchild_parent_locations(field_location const* parent_locs,
                                                field_location const* child_locs,
                                                int child_idx,
                                                int num_child_fields,
                                                field_location* gc_parent_locs,
                                                int num_rows,
                                                protobuf_error* error_flag,
                                                rmm::cuda_stream_view stream);

// ============================================================================
// Host-side template helpers that launch CUDA kernels
// ============================================================================

// Build a row-aligned null mask from `valid[row]` boolean flags.
template <typename T>
inline std::pair<rmm::device_buffer, cudf::size_type> make_null_mask_from_valid(
  rmm::device_uvector<T> const& valid,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(num_rows >= 0, "num_rows must be non-negative");
  CUDF_EXPECTS(valid.size() >= static_cast<size_t>(num_rows),
               "valid buffer smaller than requested null mask");
  auto begin = thrust::make_counting_iterator<cudf::size_type>(0);
  auto end   = begin + num_rows;
  auto pred  = [ptr = valid.data()] __device__(cudf::size_type i) {
    return static_cast<bool>(ptr[i]);
  };
  return cudf::detail::valid_if(begin, end, pred, stream, mr);
}

template <typename T, typename LaunchFn>
std::unique_ptr<cudf::column> extract_and_build_scalar_column(cudf::data_type dt,
                                                              int num_rows,
                                                              LaunchFn&& launch_extract,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<T> out(num_rows, stream, mr);
  rmm::device_uvector<bool> valid(num_rows, stream, mr);
  if (num_rows == 0) {
    return std::make_unique<cudf::column>(dt, 0, out.release(), rmm::device_buffer{}, 0);
  }
  launch_extract(out.data(), valid.data());
  auto [mask, null_count] = make_null_mask_from_valid(valid, num_rows, stream, mr);
  return std::make_unique<cudf::column>(dt, num_rows, out.release(), std::move(mask), null_count);
}

template <typename T, typename LocationProvider>
inline void extract_scalar_into_buffers(uint8_t const* message_data,
                                        LocationProvider const& loc_provider,
                                        int num_rows,
                                        proto_encoding encoding,
                                        scalar_decode_options<T> options,
                                        scalar_value_output<T> output,
                                        rmm::cuda_stream_view stream)
{
  auto constexpr threads = THREADS_PER_BLOCK;
  auto const blocks      = static_cast<int>((num_rows + threads - 1u) / threads);
  if constexpr (std::is_floating_point_v<T>) {
    CUDF_EXPECTS(encoding == proto_encoding::DEFAULT || encoding == proto_encoding::FIXED,
                 "Floating-point protobuf extraction requires default or fixed encoding");
    extract_scalar_kernel<T, decode_fixed_value<T>, LocationProvider>
      <<<blocks, threads, 0, stream.value()>>>(
        message_data, loc_provider, num_rows, output, options);
  } else if (encoding == proto_encoding::FIXED) {
    if constexpr (sizeof(T) == 4 || sizeof(T) == 8) {
      extract_scalar_kernel<T, decode_fixed_value<T>, LocationProvider>
        <<<blocks, threads, 0, stream.value()>>>(
          message_data, loc_provider, num_rows, output, options);
    } else {
      CUDF_FAIL("Fixed-width protobuf extraction requires a 32-bit or 64-bit output type");
    }
  } else if constexpr (std::is_signed_v<T>) {
    CUDF_EXPECTS(encoding == proto_encoding::DEFAULT || encoding == proto_encoding::ZIGZAG,
                 "Signed varint protobuf extraction requires default or zigzag encoding");
    if (encoding == proto_encoding::ZIGZAG) {
      extract_scalar_kernel<T, decode_varint_value<T, true>, LocationProvider>
        <<<blocks, threads, 0, stream.value()>>>(
          message_data, loc_provider, num_rows, output, options);
    } else {
      extract_scalar_kernel<T, decode_varint_value<T, false>, LocationProvider>
        <<<blocks, threads, 0, stream.value()>>>(
          message_data, loc_provider, num_rows, output, options);
    }
  } else {
    CUDF_EXPECTS(encoding == proto_encoding::DEFAULT,
                 "Unsigned varint protobuf extraction requires default encoding");
    extract_scalar_kernel<T, decode_varint_value<T, false>, LocationProvider>
      <<<blocks, threads, 0, stream.value()>>>(
        message_data, loc_provider, num_rows, output, options);
  }
}

template <typename T>
inline scalar_decode_options<T> make_scalar_decode_options(protobuf_field_meta_view field)
{
  if constexpr (std::is_same_v<T, uint8_t>) {
    return {field.schema.has_default_value, static_cast<uint8_t>(field.default_bool ? 1 : 0)};
  } else if constexpr (std::is_integral_v<T>) {
    return {field.schema.has_default_value, static_cast<T>(field.default_int)};
  } else if constexpr (std::is_floating_point_v<T>) {
    return {field.schema.has_default_value, static_cast<T>(field.default_float)};
  } else {
    static_assert(std::is_arithmetic_v<T>, "Unsupported protobuf scalar output type");
  }
}

template <typename T, typename LocationProvider>
std::unique_ptr<cudf::column> extract_and_build_integer_column(
  protobuf_field_meta_view field,
  uint8_t const* message_data,
  LocationProvider const& loc_provider,
  int num_rows,
  protobuf_decode_runtime_context decode_ctx,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return extract_and_build_scalar_column<T>(
    field.output_type,
    num_rows,
    [&](T* out_ptr, bool* valid_ptr) {
      extract_scalar_into_buffers<T, LocationProvider>(
        message_data,
        loc_provider,
        num_rows,
        field.schema.encoding,
        make_scalar_decode_options<T>(field),
        {out_ptr, valid_ptr, decode_ctx.error->data()},
        stream);
    },
    stream,
    mr);
}

struct extract_strided_count {
  field_occurrence_count const* info;
  int field_idx;
  int num_fields;

  __device__ int32_t operator()(int row) const
  {
    return info[flat_index(row, num_fields, field_idx)].count;
  }
};

template <typename LocationProvider, typename ValidityFn>
inline std::unique_ptr<cudf::column> extract_and_build_string_or_bytes_column(
  bool as_bytes,
  uint8_t const* message_data,
  int num_rows,
  LocationProvider const& loc_provider,
  ValidityFn validity_fn,
  bool has_default,
  cudf::detail::host_vector<uint8_t> const& default_bytes,
  rmm::device_uvector<protobuf_error>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  int32_t def_len = has_default ? static_cast<int32_t>(default_bytes.size()) : 0;
  rmm::device_uvector<uint8_t> d_default(0, stream, mr);
  if (has_default && def_len > 0) {
    d_default = cudf::detail::make_device_uvector_async(
      default_bytes, stream, cudf::get_current_device_resource_ref());
  }

  rmm::device_uvector<int32_t> lengths(num_rows, stream, mr);
  auto const threads = THREADS_PER_BLOCK;
  auto const blocks  = static_cast<int>((num_rows + threads - 1u) / threads);
  extract_lengths_kernel<LocationProvider><<<blocks, threads, 0, stream.value()>>>(
    loc_provider, num_rows, lengths.data(), has_default, def_len);

  auto [offsets_col, total_size] =
    cudf::strings::detail::make_offsets_child_column(lengths.begin(), lengths.end(), stream, mr);

  rmm::device_uvector<char> chars(total_size, stream, mr);
  if (total_size > 0) {
    auto const* offsets_data = offsets_col->view().data<cudf::size_type>();
    auto* chars_ptr          = chars.data();
    auto const* default_ptr  = d_default.data();

    auto src_iter = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<void const*>(
        [message_data, loc_provider, has_default, default_ptr, def_len] __device__(
          int idx) -> void const* {
          int32_t data_offset = 0;
          auto loc            = loc_provider.get(idx, data_offset);
          if (loc.offset < 0) {
            return (has_default && def_len > 0) ? static_cast<void const*>(default_ptr) : nullptr;
          }
          return static_cast<void const*>(message_data + data_offset);
        }));
    auto dst_iter = cudf::detail::make_counting_transform_iterator(
      0, cuda::proclaim_return_type<void*>([chars_ptr, offsets_data] __device__(int idx) -> void* {
        return static_cast<void*>(chars_ptr + offsets_data[idx]);
      }));
    auto size_iter = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<size_t>(
        [loc_provider, has_default, def_len] __device__(int idx) -> size_t {
          int32_t data_offset = 0;
          auto loc            = loc_provider.get(idx, data_offset);
          if (loc.offset < 0) {
            return (has_default && def_len > 0) ? static_cast<size_t>(def_len) : 0;
          }
          return static_cast<size_t>(loc.length);
        }));

    size_t temp_storage_bytes = 0;
    cub::DeviceMemcpy::Batched(
      nullptr, temp_storage_bytes, src_iter, dst_iter, size_iter, num_rows, stream.value());
    rmm::device_buffer temp_storage(temp_storage_bytes, stream, mr);
    cub::DeviceMemcpy::Batched(temp_storage.data(),
                               temp_storage_bytes,
                               src_iter,
                               dst_iter,
                               size_iter,
                               num_rows,
                               stream.value());
  }

  if (num_rows == 0) {
    if (as_bytes) {
      auto bytes_child = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::UINT8}, 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);
      return cudf::make_lists_column(
        0, std::move(offsets_col), std::move(bytes_child), 0, rmm::device_buffer{});
    }
    return cudf::make_strings_column(
      0, std::move(offsets_col), chars.release(), 0, rmm::device_buffer{});
  }

  rmm::device_uvector<bool> valid(num_rows, stream, mr);
  thrust::transform(rmm::exec_policy_nosync(stream, mr),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(num_rows),
                    valid.data(),
                    validity_fn);
  auto [mask, null_count] = make_null_mask_from_valid(valid, num_rows, stream, mr);
  if (as_bytes) {
    auto bytes_child =
      std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT8},
                                     total_size,
                                     rmm::device_buffer(chars.data(), total_size, stream, mr),
                                     rmm::device_buffer{},
                                     0);
    return cudf::make_lists_column(
      num_rows, std::move(offsets_col), std::move(bytes_child), null_count, std::move(mask));
  }

  return cudf::make_strings_column(
    num_rows, std::move(offsets_col), chars.release(), null_count, std::move(mask));
}

template <typename LocationProvider>
inline std::unique_ptr<cudf::column> extract_typed_column(protobuf_field_decode_request request,
                                                          LocationProvider const& loc_provider,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr)
{
  auto const field           = request.context.schema.field(request.schema_idx);
  auto const message_data    = request.message_data;
  auto const num_items       = request.values.size;
  auto const decode_ctx      = request.context.runtime;
  auto const top_row_indices = request.values.top_row_indices;
  auto constexpr threads     = THREADS_PER_BLOCK;
  auto const blocks          = static_cast<int>((num_items + threads - 1u) / threads);
  auto const dt              = field.output_type;

  switch (dt.id()) {
    case cudf::type_id::BOOL8: {
      return extract_and_build_scalar_column<uint8_t>(
        dt,
        num_items,
        [&](uint8_t* out_ptr, bool* valid_ptr) {
          extract_scalar_kernel<uint8_t, decode_varint_value<uint8_t, false>, LocationProvider>
            <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                     loc_provider,
                                                     num_items,
                                                     {out_ptr, valid_ptr, decode_ctx.error->data()},
                                                     make_scalar_decode_options<uint8_t>(field));
        },
        stream,
        mr);
    }
    case cudf::type_id::INT32: {
      if (num_items == 0) {
        return std::make_unique<cudf::column>(dt, 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);
      }
      rmm::device_uvector<int32_t> out(num_items, stream, mr);
      rmm::device_uvector<bool> valid(num_items, stream, mr);
      extract_scalar_into_buffers<int32_t, LocationProvider>(
        message_data,
        loc_provider,
        num_items,
        field.schema.encoding,
        make_scalar_decode_options<int32_t>(field),
        {out.data(), valid.data(), decode_ctx.error->data()},
        stream);
      if (!field.enum_valid_values.empty()) {
        validate_enum_and_propagate_rows(
          out, valid, field.enum_valid_values, decode_ctx, {num_items, top_row_indices}, stream);
      }
      auto [mask, null_count] = make_null_mask_from_valid(valid, num_items, stream, mr);
      return std::make_unique<cudf::column>(
        dt, num_items, out.release(), std::move(mask), null_count);
    }
    case cudf::type_id::UINT32:
      return extract_and_build_integer_column<uint32_t>(
        field, message_data, loc_provider, num_items, decode_ctx, stream, mr);
    case cudf::type_id::INT64:
      return extract_and_build_integer_column<int64_t>(
        field, message_data, loc_provider, num_items, decode_ctx, stream, mr);
    case cudf::type_id::UINT64:
      return extract_and_build_integer_column<uint64_t>(
        field, message_data, loc_provider, num_items, decode_ctx, stream, mr);
    case cudf::type_id::FLOAT32: {
      return extract_and_build_scalar_column<float>(
        dt,
        num_items,
        [&](float* out_ptr, bool* valid_ptr) {
          extract_scalar_kernel<float, decode_fixed_value<float>, LocationProvider>
            <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                     loc_provider,
                                                     num_items,
                                                     {out_ptr, valid_ptr, decode_ctx.error->data()},
                                                     make_scalar_decode_options<float>(field));
        },
        stream,
        mr);
    }
    case cudf::type_id::FLOAT64: {
      return extract_and_build_scalar_column<double>(
        dt,
        num_items,
        [&](double* out_ptr, bool* valid_ptr) {
          extract_scalar_kernel<double, decode_fixed_value<double>, LocationProvider>
            <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                     loc_provider,
                                                     num_items,
                                                     {out_ptr, valid_ptr, decode_ctx.error->data()},
                                                     make_scalar_decode_options<double>(field));
        },
        stream,
        mr);
    }
    default: return make_null_column(dt, num_items, stream, mr);
  }
}

template <typename T>
inline std::unique_ptr<cudf::column> build_repeated_scalar_column(
  cudf::column_view const& binary_input,
  protobuf_input_view input,
  device_nested_field_descriptor const& field_desc,
  repeated_field_work work,
  rmm::device_uvector<protobuf_error>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const input_null_count = binary_input.null_count();
  auto const field_type_id    = static_cast<cudf::type_id>(field_desc.output_type_id);

  validate_nonempty_repeated_field_work(work, input.num_rows);

  rmm::device_uvector<T> values(work.total_count, stream, mr);
  repeated_location_provider loc_provider{
    input.row_offsets, input.base_offset, work.occurrences->data()};
  extract_scalar_into_buffers<T, repeated_location_provider>(
    input.message_data,
    loc_provider,
    work.total_count,
    field_desc.encoding,
    {false, T{}},
    {values.data(), nullptr, d_error.data()},
    stream);

  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    input.num_rows + 1,
                                                    work.offsets.release(),
                                                    rmm::device_buffer{},
                                                    0);
  auto child_col   = std::make_unique<cudf::column>(
    cudf::data_type{field_type_id}, work.total_count, values.release(), rmm::device_buffer{}, 0);

  if (input_null_count > 0) {
    auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
    return cudf::make_lists_column(input.num_rows,
                                   std::move(offsets_col),
                                   std::move(child_col),
                                   input_null_count,
                                   std::move(null_mask));
  }

  return cudf::make_lists_column(
    input.num_rows, std::move(offsets_col), std::move(child_col), 0, rmm::device_buffer{});
}

// ============================================================================
// Host wrapper declarations for kernel launches
// ============================================================================

void launch_scan_all_fields(cudf::column_device_view const& d_in,
                            field_scan_view fields,
                            protobuf_error* error_flag,
                            bool* row_has_invalid_data,
                            rmm::cuda_stream_view stream);

void launch_validate_enum_values(enum_value_device_view input,
                                 bool* row_has_invalid_enum,
                                 enum_domain_device_view domain,
                                 rmm::cuda_stream_view stream);

void launch_compute_enum_string_lengths(enum_value_device_view input,
                                        enum_string_lookup_device_view lookup,
                                        int32_t* lengths,
                                        rmm::cuda_stream_view stream);

void launch_copy_enum_string_chars(enum_value_device_view input,
                                   enum_string_lookup_device_view lookup,
                                   int32_t const* output_offsets,
                                   char* out_chars,
                                   rmm::cuda_stream_view stream);

}  // namespace spark_rapids_jni::protobuf::detail
