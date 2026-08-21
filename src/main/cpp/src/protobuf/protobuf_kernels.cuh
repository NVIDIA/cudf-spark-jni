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
#include <cuda/std/utility>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <array>
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

struct message_fragment_location_provider {
  protobuf_input_view input;
  message_fragment_source_view source;
  field_occurrence const* fragments;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto const fragment      = fragments[thread_idx];
    auto const parent_offset = source.parent_locations == nullptr
                                 ? int32_t{0}
                                 : source.parent_locations[fragment.row_idx].offset;
    if (parent_offset < 0) {
      data_offset = 0;
      return {-1, 0};
    }
    data_offset =
      input.row_offsets[fragment.row_idx] - input.base_offset + parent_offset + fragment.offset;
    return {fragment.offset, fragment.length};
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
  if (!read_varint64(cur, cur_end, v, n)) {
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

enum class scalar_decode_kind : uint8_t { FIXED, VARINT, ZIGZAG };

struct scalar_kind {
  cudf::type_id type;
  scalar_decode_kind decode;
  bool operator==(scalar_kind const&) const = default;
};

inline constexpr auto SCALAR_KINDS = std::to_array<scalar_kind>({
  {cudf::type_id::INT32, scalar_decode_kind::VARINT},
  {cudf::type_id::UINT32, scalar_decode_kind::VARINT},
  {cudf::type_id::INT64, scalar_decode_kind::VARINT},
  {cudf::type_id::UINT64, scalar_decode_kind::VARINT},
  {cudf::type_id::BOOL8, scalar_decode_kind::VARINT},
  {cudf::type_id::INT32, scalar_decode_kind::ZIGZAG},
  {cudf::type_id::INT64, scalar_decode_kind::ZIGZAG},
  {cudf::type_id::FLOAT32, scalar_decode_kind::FIXED},
  {cudf::type_id::FLOAT64, scalar_decode_kind::FIXED},
  {cudf::type_id::INT32, scalar_decode_kind::FIXED},
  {cudf::type_id::UINT32, scalar_decode_kind::FIXED},
  {cudf::type_id::INT64, scalar_decode_kind::FIXED},
  {cudf::type_id::UINT64, scalar_decode_kind::FIXED},
});

constexpr scalar_decode_kind get_scalar_decode_kind(cudf::type_id type, proto_encoding encoding)
{
  using enum cudf::type_id;
  using enum proto_encoding;
  return type == FLOAT32 || type == FLOAT64 || encoding == FIXED ? scalar_decode_kind::FIXED
         : encoding == ZIGZAG                                    ? scalar_decode_kind::ZIGZAG
                                                                 : scalar_decode_kind::VARINT;
}

template <typename T>
inline scalar_decode_kind get_scalar_decode_kind(proto_encoding encoding)
{
  if constexpr (std::is_floating_point_v<T>) {
    CUDF_EXPECTS(encoding == proto_encoding::DEFAULT || encoding == proto_encoding::FIXED,
                 "Floating-point protobuf extraction requires default or fixed encoding");
  } else if (encoding == proto_encoding::FIXED) {
    CUDF_EXPECTS(sizeof(T) == 4 || sizeof(T) == 8,
                 "Fixed-width protobuf extraction requires a 32-bit or 64-bit output type");
  } else if constexpr (std::is_signed_v<T>) {
    CUDF_EXPECTS(encoding == proto_encoding::DEFAULT || encoding == proto_encoding::ZIGZAG,
                 "Signed varint protobuf extraction requires default or zigzag encoding");
  } else if constexpr (std::is_integral_v<T>) {
    CUDF_EXPECTS(encoding == proto_encoding::DEFAULT,
                 "Unsigned varint protobuf extraction requires default encoding");
  } else {
    CUDF_FAIL("Varint protobuf extraction requires an integral output type");
  }
  return get_scalar_decode_kind(
    std::is_floating_point_v<T> ? cudf::type_id::FLOAT32 : cudf::type_id::INT32, encoding);
}

template <typename T, scalar_decode_kind Decode, typename F>
constexpr void dispatch_scalar_decoder(F&& f)
{
  using enum scalar_decode_kind;
  if constexpr (Decode == FIXED) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8);
    std::forward<F>(f).template operator()<decode_fixed_value<T>>();
  } else {
    static_assert(Decode == VARINT || Decode == ZIGZAG);
    static_assert(std::is_integral_v<T>);
    constexpr bool zigzag = Decode == ZIGZAG;
    if constexpr (zigzag) { static_assert(std::is_signed_v<T>); }
    std::forward<F>(f).template operator()<decode_varint_value<T, zigzag>>();
  }
}

template <typename T, typename F>
inline void dispatch_scalar_decoder(scalar_decode_kind decode, F&& f)
{
  switch (decode) {
    case scalar_decode_kind::FIXED:
      if constexpr (sizeof(T) == 4 || sizeof(T) == 8) {
        return dispatch_scalar_decoder<T, scalar_decode_kind::FIXED>(std::forward<F>(f));
      }
      break;
    case scalar_decode_kind::VARINT:
      if constexpr (std::is_integral_v<T>) {
        return dispatch_scalar_decoder<T, scalar_decode_kind::VARINT>(std::forward<F>(f));
      }
      break;
    case scalar_decode_kind::ZIGZAG:
      if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        return dispatch_scalar_decoder<T, scalar_decode_kind::ZIGZAG>(std::forward<F>(f));
      }
      break;
  }
  CUDF_UNREACHABLE("Invalid protobuf scalar decode kind/type combination");
}

template <typename OutputType, auto DecodeFn, typename LocationProvider>
__device__ void extract_scalar_kernel_impl(uint8_t const* message_data,
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

// Kernel parameters stay by value because forwarding references preserve host lvalue references.
template <typename OutputType, auto DecodeFn, typename... Args>
CUDF_KERNEL void extract_scalar_kernel(Args... args)
{
  extract_scalar_kernel_impl<OutputType, DecodeFn>(cuda::std::forward<Args>(args)...);
}

// ============================================================================
// Batched scalar extraction — one 2D kernel for N fields of the same type
// ============================================================================

template <typename OutputType, auto DecodeFn>
CUDF_KERNEL void extract_scalar_batched_kernel(batched_scalar_input_view<OutputType> input)
{
  int fi = static_cast<int>(blockIdx.y);
  if (fi >= input.num_descriptors) return;

  auto const& desc = input.descriptors[fi];
  top_level_location_provider loc_provider{input.input.row_offsets,
                                           input.input.base_offset,
                                           input.locations,
                                           desc.loc_field_idx,
                                           input.num_location_fields};
  extract_scalar_kernel_impl<OutputType, DecodeFn>(input.input.message_data,
                                                   loc_provider,
                                                   input.input.num_rows,
                                                   {desc.output, desc.valid, input.error},
                                                   desc.options);
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

template <typename LocationProvider>
CUDF_KERNEL void extract_utf8_lengths_kernel(uint8_t const* message_data,
                                             LocationProvider loc_provider,
                                             int total_items,
                                             int32_t* out_lengths,
                                             protobuf_error* error,
                                             uint8_t const* default_data = nullptr,
                                             int32_t default_length      = 0)
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_items) return;

  int32_t data_offset = 0;
  auto const loc      = loc_provider.get(idx, data_offset);
  auto const* data    = loc.offset >= 0 ? message_data + data_offset : default_data;
  auto const size     = loc.offset >= 0 ? loc.length : default_length;
  if (data == nullptr || size == 0) {
    out_lengths[idx] = 0;
    return;
  }

  auto const repaired_length = repaired_utf8_length(data, size);
  if (repaired_length > cuda::std::numeric_limits<int32_t>::max()) {
    out_lengths[idx] = 0;
    if (error != nullptr) { set_error_once(error, protobuf_error::OVERFLOW); }
    return;
  }
  out_lengths[idx] = static_cast<int32_t>(repaired_length);
}

template <typename LocationProvider>
CUDF_KERNEL void copy_repaired_utf8_kernel(uint8_t const* message_data,
                                           LocationProvider loc_provider,
                                           int total_items,
                                           int32_t const* output_offsets,
                                           char* output,
                                           uint8_t const* default_data = nullptr,
                                           int32_t default_length      = 0)
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_items) return;

  int32_t data_offset = 0;
  auto const loc      = loc_provider.get(idx, data_offset);
  auto const* data    = loc.offset >= 0 ? message_data + data_offset : default_data;
  auto const size     = loc.offset >= 0 ? loc.length : default_length;
  if (data != nullptr && size > 0) { copy_repaired_utf8(data, size, output + output_offsets[idx]); }
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

void launch_scan_singular_message_occurrences(cudf::column_device_view const& d_in,
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
                                       int recursion_depth,
                                       rmm::cuda_stream_view stream);

void launch_scan_all_field_occurrences_in_nested(protobuf_input_view input,
                                                 nested_parent_view parent,
                                                 field_occurrence_scan_view fields,
                                                 protobuf_error* error_flag,
                                                 int recursion_depth,
                                                 rmm::cuda_stream_view stream);

void launch_validate_message_fragments(message_fragment_location_provider locations,
                                       message_validation_view fields,
                                       int num_fragments,
                                       bool* invalid_rows,
                                       bool* row_has_invalid_data,
                                       protobuf_error* error_flag,
                                       int recursion_depth,
                                       rmm::cuda_stream_view stream);

void launch_compute_grandchild_parent_locations(nested_location_provider loc_provider,
                                                field_location* gc_parent_locs,
                                                int num_rows,
                                                protobuf_error* error_flag,
                                                rmm::cuda_stream_view stream);

void launch_compute_virtual_parents_for_nested_repeated(protobuf_input_view input,
                                                        nested_parent_view parent,
                                                        repeated_field_work const& work,
                                                        cudf::size_type* virtual_row_offsets,
                                                        field_location* virtual_parent_locs,
                                                        protobuf_decode_runtime_context decode_ctx,
                                                        rmm::cuda_stream_view stream);

void launch_compute_msg_locations_from_occurrences(protobuf_input_view input,
                                                   repeated_field_work const& work,
                                                   field_location* msg_locs,
                                                   cudf::size_type* msg_row_offsets,
                                                   protobuf_decode_runtime_context decode_ctx,
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
  auto [mask, null_count] = cudf::detail::valid_if(begin, end, pred, stream, mr);
  if (null_count == 0) { mask = rmm::device_buffer{}; }
  return {std::move(mask), null_count};
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
  dispatch_scalar_decoder<T>(get_scalar_decode_kind<T>(encoding), [&]<auto DecodeFn>() {
    extract_scalar_kernel<T, DecodeFn><<<blocks, threads, 0, stream.value()>>>(
      message_data, loc_provider, num_rows, output, options);
    CUDF_CHECK_CUDA(stream.value());
  });
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
std::unique_ptr<cudf::column> extract_and_build_scalar_field_column(
  protobuf_field_meta_view field,
  uint8_t const* message_data,
  LocationProvider const& loc_provider,
  int num_values,
  protobuf_decode_runtime_context decode_ctx,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_rows = num_values;
  if (num_rows == 0) { return cudf::make_empty_column(field.output_type); }
  rmm::device_uvector<T> out(num_rows, stream, mr);
  // Validity is temporary extraction state; only output data and its null mask use the caller MR.
  auto const scratch_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<bool> valid(num_rows, stream, scratch_mr);
  extract_scalar_into_buffers<T, LocationProvider>(
    message_data,
    loc_provider,
    num_rows,
    field.schema.encoding,
    make_scalar_decode_options<T>(field),
    {out.data(), valid.data(), decode_ctx.error->data()},
    stream);
  if constexpr (std::is_same_v<T, int32_t>) {
    if (!field.enum_valid_values.empty()) {
      validate_enum_values(out, valid, field.enum_valid_values, stream);
    }
  }
  auto [mask, null_count] = make_null_mask_from_valid(valid, num_rows, stream, mr);
  return std::make_unique<cudf::column>(
    field.output_type, num_rows, out.release(), std::move(mask), null_count);
}

template <typename LocationProvider, typename ValidityFn>
inline std::unique_ptr<cudf::column> extract_and_build_string_or_bytes_column(
  protobuf_field_meta_view field,
  uint8_t const* message_data,
  int num_rows,
  LocationProvider const& loc_provider,
  ValidityFn validity_fn,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const as_bytes       = field.output_type.id() == cudf::type_id::LIST;
  auto const has_default    = field.schema.has_default_value;
  auto const& default_bytes = field.default_string;
  int32_t def_len           = has_default ? static_cast<int32_t>(default_bytes.size()) : 0;
  auto const scratch_mr     = cudf::get_current_device_resource_ref();
  rmm::device_uvector<uint8_t> d_default(0, stream, scratch_mr);
  if (has_default && def_len > 0) {
    d_default = cudf::detail::make_device_uvector_async(
      default_bytes, stream, cudf::get_current_device_resource_ref());
  }

  rmm::device_uvector<int32_t> lengths(num_rows, stream, scratch_mr);
  auto const threads = THREADS_PER_BLOCK;
  auto const blocks  = static_cast<int>((num_rows + threads - 1u) / threads);
  if (num_rows > 0) {
    if (as_bytes) {
      extract_lengths_kernel<LocationProvider><<<blocks, threads, 0, stream.value()>>>(
        loc_provider, num_rows, lengths.data(), has_default, def_len);
    } else {
      extract_utf8_lengths_kernel<LocationProvider>
        <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                 loc_provider,
                                                 num_rows,
                                                 lengths.data(),
                                                 nullptr,
                                                 has_default ? d_default.data() : nullptr,
                                                 def_len);
    }
    CUDF_CHECK_CUDA(stream.value());
  }

  auto [offsets_col, total_size] =
    cudf::strings::detail::make_offsets_child_column(lengths.begin(), lengths.end(), stream, mr);

  rmm::device_uvector<char> chars(total_size, stream, mr);
  if (total_size > 0) {
    auto const* offsets_data = offsets_col->view().data<cudf::size_type>();
    auto* chars_ptr          = chars.data();
    auto const* default_ptr  = d_default.data();

    if (!as_bytes) {
      copy_repaired_utf8_kernel<LocationProvider>
        <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                 loc_provider,
                                                 num_rows,
                                                 offsets_data,
                                                 chars_ptr,
                                                 has_default ? default_ptr : nullptr,
                                                 def_len);
      CUDF_CHECK_CUDA(stream.value());
    } else {
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
        0,
        cuda::proclaim_return_type<void*>([chars_ptr, offsets_data] __device__(int idx) -> void* {
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
      CUDF_CUDA_TRY(cub::DeviceMemcpy::Batched(
        nullptr, temp_storage_bytes, src_iter, dst_iter, size_iter, num_rows, stream.value()));
      rmm::device_buffer temp_storage(temp_storage_bytes, stream, scratch_mr);
      CUDF_CUDA_TRY(cub::DeviceMemcpy::Batched(temp_storage.data(),
                                               temp_storage_bytes,
                                               src_iter,
                                               dst_iter,
                                               size_iter,
                                               num_rows,
                                               stream.value()));
    }
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

  rmm::device_uvector<bool> valid(num_rows, stream, scratch_mr);
  thrust::transform(rmm::exec_policy_nosync(stream, scratch_mr),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(num_rows),
                    valid.data(),
                    validity_fn);
  auto [mask, null_count] = make_null_mask_from_valid(valid, num_rows, stream, mr);
  if (as_bytes) {
    auto bytes_child = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::UINT8}, total_size, chars.release(), rmm::device_buffer{}, 0);
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
  auto const field        = request.context.schema.field(request.schema_idx);
  auto const message_data = request.message_data;
  auto const decode_ctx   = request.context.runtime;
  auto const num_items    = request.num_values;
  auto const dt           = field.output_type;

  switch (dt.id()) {
    case cudf::type_id::BOOL8:
      return extract_and_build_scalar_field_column<uint8_t>(
        field, message_data, loc_provider, request.num_values, decode_ctx, stream, mr);
    case cudf::type_id::INT32:
      return extract_and_build_scalar_field_column<int32_t>(
        field, message_data, loc_provider, request.num_values, decode_ctx, stream, mr);
    case cudf::type_id::UINT32:
      return extract_and_build_scalar_field_column<uint32_t>(
        field, message_data, loc_provider, request.num_values, decode_ctx, stream, mr);
    case cudf::type_id::INT64:
      return extract_and_build_scalar_field_column<int64_t>(
        field, message_data, loc_provider, request.num_values, decode_ctx, stream, mr);
    case cudf::type_id::UINT64:
      return extract_and_build_scalar_field_column<uint64_t>(
        field, message_data, loc_provider, request.num_values, decode_ctx, stream, mr);
    case cudf::type_id::FLOAT32:
      return extract_and_build_scalar_field_column<float>(
        field, message_data, loc_provider, request.num_values, decode_ctx, stream, mr);
    case cudf::type_id::FLOAT64:
      return extract_and_build_scalar_field_column<double>(
        field, message_data, loc_provider, request.num_values, decode_ctx, stream, mr);
    default:
      // Preserve protobuf-java-compatible null output when invalid input reaches this fallback.
      return make_null_column(dt, num_items, stream, mr);
  }
}

template <typename LocationProvider, typename ValidityFn>
inline std::unique_ptr<cudf::column> build_protobuf_field_values_column_shared(
  protobuf_field_decode_request request,
  LocationProvider const& loc_provider,
  ValidityFn validity_fn,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const message_data = request.message_data;
  auto const field        = request.context.schema.field(request.schema_idx);
  auto const decode_ctx   = request.context.runtime;
  auto const num_values   = request.num_values;
  CUDF_EXPECTS(num_values > 0, std::string{__func__} + ": value count must be positive");
  auto const value_type  = field.output_type;
  auto const has_default = field.schema.has_default_value;

  switch (value_type.id()) {
    case cudf::type_id::BOOL8:
    case cudf::type_id::INT32:
    case cudf::type_id::UINT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT64:
    case cudf::type_id::FLOAT32:
    case cudf::type_id::FLOAT64: {
      return extract_typed_column(request, loc_provider, stream, mr);
    }
    case cudf::type_id::STRING:
    case cudf::type_id::LIST: {
      bool const is_enum_string = value_type.id() == cudf::type_id::STRING &&
                                  field.schema.encoding == proto_encoding::ENUM_STRING;
      if (is_enum_string) {
        auto const scratch_mr = cudf::get_current_device_resource_ref();
        rmm::device_uvector<int32_t> values(num_values, stream, scratch_mr);
        rmm::device_uvector<bool> valid(num_values, stream, scratch_mr);
        extract_scalar_into_buffers<int32_t>(
          message_data,
          loc_provider,
          num_values,
          proto_encoding::DEFAULT,
          {has_default, static_cast<int32_t>(field.default_int)},
          {values.data(), valid.data(), decode_ctx.error->data()},
          stream);
        return build_enum_string_column(values, valid, request, stream, mr);
      }
      return extract_and_build_string_or_bytes_column(
        field, message_data, num_values, loc_provider, validity_fn, stream, mr);
    }
    default:
      CUDF_FAIL("Protobuf decode: unsupported child output type id=" +
                std::to_string(static_cast<int>(value_type.id())));
  }
}

template <typename T>
inline std::unique_ptr<cudf::column> build_repeated_scalar_column(
  cudf::column_view const& binary_input,
  protobuf_input_view input,
  protobuf_schema const& schema,
  protobuf_decode_runtime_context decode_ctx,
  repeated_field_work work,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  validate_nonempty_repeated_field_work(work, input.num_rows);

  auto const field       = schema.field(work.schema_idx);
  auto const total_count = work.total_count;
  auto& occurrences      = *work.occurrences;
  repeated_location_provider loc_provider{input.row_offsets, input.base_offset, occurrences.data()};

  std::unique_ptr<cudf::column> child_col;
  if constexpr (std::is_same_v<T, int32_t>) {
    if (!field.enum_valid_values.empty()) {
      auto const context = recursive_decode_context{schema, decode_ctx};
      auto const request =
        protobuf_field_decode_request{context, input.message_data, work.schema_idx, total_count};
      child_col = extract_typed_column(request, loc_provider, stream, mr);
    }
  }

  if (child_col == nullptr) {
    rmm::device_uvector<T> values(total_count, stream, mr);
    extract_scalar_into_buffers<T, repeated_location_provider>(
      input.message_data,
      loc_provider,
      total_count,
      field.schema.encoding,
      {false, T{}},
      {values.data(), nullptr, decode_ctx.error->data()},
      stream);
    child_col = std::make_unique<cudf::column>(
      field.output_type, total_count, values.release(), rmm::device_buffer{}, 0);
  }

  auto offsets_col = make_offsets_column(input.num_rows, std::move(work.offsets));
  auto result      = make_list_column_with_input_nulls(
    input.num_rows, std::move(offsets_col), std::move(child_col), binary_input, stream, mr);
  if constexpr (std::is_same_v<T, int32_t>) {
    if (!field.enum_valid_values.empty()) {
      return drop_unknown_repeated_enum_values(std::move(result), stream, mr);
    }
  }
  return result;
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
