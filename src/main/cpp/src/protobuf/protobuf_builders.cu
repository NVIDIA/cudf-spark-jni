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

#include "protobuf/protobuf_kernels.cuh"

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>

#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <source_location>
#include <string>
#include <utility>

namespace spark_rapids_jni::protobuf::detail {

namespace {

inline std::pair<rmm::device_buffer, cudf::size_type> make_null_mask_from_parent_locations(
  field_location const* parent_locs,
  int num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(num_rows >= 0, std::string{__func__} + ": row count must be non-negative");
  auto [mask, null_count] = cudf::detail::valid_if(
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(num_rows),
    [parent_locs] __device__(cudf::size_type row) { return parent_locs[row].offset >= 0; },
    stream,
    mr);
  if (null_count == 0) { mask = rmm::device_buffer{}; }
  return {std::move(mask), null_count};
}

inline void validate_nested_parent_view(
  protobuf_input_view input,
  nested_parent_view parent,
  std::source_location const& location = std::source_location::current())
{
  auto const caller = location.function_name();
  CUDF_EXPECTS(input.num_rows >= 0, std::string{caller} + ": row count must be non-negative");
  CUDF_EXPECTS(parent.location_count == static_cast<std::size_t>(input.num_rows),
               std::string{caller} + ": parent locations size must match row count");
  CUDF_EXPECTS(parent.locations != nullptr || input.num_rows == 0,
               std::string{caller} + ": parent locations must be non-null for non-empty input");
}

inline void validate_protobuf_decode_context(
  protobuf_decode_runtime_context context,
  int num_rows,
  std::source_location const& location = std::source_location::current())
{
  auto const caller = location.function_name();
  CUDF_EXPECTS(context.row_force_null != nullptr,
               std::string{caller} + ": row-force-null buffer must be non-null");
  CUDF_EXPECTS(context.error != nullptr, std::string{caller} + ": error buffer must be non-null");
  CUDF_EXPECTS(context.error->size() == 1,
               std::string{caller} + ": error buffer must contain exactly one element");
  CUDF_EXPECTS(context.row_force_null->is_empty() ||
                 context.row_force_null->size() == static_cast<size_t>(num_rows),
               std::string{caller} + ": row-force-null buffer must be empty or row-sized");
}

inline std::unique_ptr<cudf::column> make_list_column_with_parent_nulls(
  int num_rows,
  std::unique_ptr<cudf::column> offsets_col,
  std::unique_ptr<cudf::column> child_col,
  field_location const* parent_locs,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto [list_mask, list_null_count] =
    make_null_mask_from_parent_locations(parent_locs, num_rows, stream, mr);
  return cudf::make_lists_column(
    num_rows, std::move(offsets_col), std::move(child_col), list_null_count, std::move(list_mask));
}

template <typename LocationProvider, typename ValidityFn, typename TopRowIndexProvider>
std::unique_ptr<cudf::column> build_protobuf_field_values_column(
  uint8_t const* message_data,
  nested_field_descriptor const& field,
  int schema_idx,
  int num_values,
  LocationProvider const& loc_provider,
  ValidityFn validity_fn,
  bool has_default,
  schema_context_view schema_ctx,
  protobuf_decode_runtime_context decode_ctx,
  TopRowIndexProvider get_top_row_indices,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(num_values > 0, std::string{__func__} + ": value count must be positive");
  auto const value_type = cudf::data_type{field.output_type};
  auto const encoding   = static_cast<int>(field.encoding);
  auto const blocks = static_cast<int>((num_values + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);

  switch (value_type.id()) {
    case cudf::type_id::BOOL8:
    case cudf::type_id::INT32:
    case cudf::type_id::UINT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT64:
    case cudf::type_id::FLOAT32:
    case cudf::type_id::FLOAT64: {
      bool const is_numeric_enum =
        value_type.id() == cudf::type_id::INT32 &&
        schema_idx < static_cast<int>(schema_ctx.enum_valid_values.size()) &&
        !schema_ctx.enum_valid_values[schema_idx].empty();
      return extract_typed_column(value_type,
                                  encoding,
                                  message_data,
                                  loc_provider,
                                  num_values,
                                  blocks,
                                  THREADS_PER_BLOCK,
                                  has_default,
                                  schema_ctx.default_ints[schema_idx],
                                  schema_ctx.default_floats[schema_idx],
                                  schema_ctx.default_bools[schema_idx],
                                  schema_ctx.default_strings[schema_idx],
                                  schema_idx,
                                  schema_ctx.enum_valid_values,
                                  schema_ctx.enum_names,
                                  decode_ctx,
                                  stream,
                                  mr,
                                  is_numeric_enum && decode_ctx.propagate_invalid_enum_rows
                                    ? get_top_row_indices()
                                    : nullptr);
    }
    case cudf::type_id::STRING:
    case cudf::type_id::LIST: {
      bool const is_enum_string =
        value_type.id() == cudf::type_id::STRING && field.encoding == proto_encoding::ENUM_STRING;
      if (is_enum_string) {
        auto const scratch_mr = cudf::get_current_device_resource_ref();
        rmm::device_uvector<int32_t> values(num_values, stream, scratch_mr);
        rmm::device_uvector<bool> valid(num_values, stream, scratch_mr);
        extract_varint_kernel<int32_t, false, LocationProvider>
          <<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(message_data,
                                                             loc_provider,
                                                             num_values,
                                                             values.data(),
                                                             valid.data(),
                                                             decode_ctx.error->data(),
                                                             has_default,
                                                             schema_ctx.default_ints[schema_idx]);
        return build_enum_string_column(
          values,
          valid,
          schema_ctx.enum_valid_values[schema_idx],
          schema_ctx.enum_names[schema_idx],
          decode_ctx,
          num_values,
          stream,
          mr,
          decode_ctx.propagate_invalid_enum_rows ? get_top_row_indices() : nullptr);
      }
      return extract_and_build_string_or_bytes_column(value_type.id() == cudf::type_id::LIST,
                                                      message_data,
                                                      num_values,
                                                      loc_provider,
                                                      validity_fn,
                                                      has_default,
                                                      schema_ctx.default_strings[schema_idx],
                                                      *decode_ctx.error,
                                                      stream,
                                                      mr);
    }
    default:
      CUDF_FAIL("Protobuf decode: unsupported nested child output type id=" +
                std::to_string(static_cast<int>(value_type.id())));
  }
}

}  // namespace

std::unique_ptr<cudf::column> make_list_column_with_input_nulls(
  int num_rows,
  std::unique_ptr<cudf::column> offsets_col,
  std::unique_ptr<cudf::column> child_col,
  cudf::column_view const& binary_input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const input_null_count = binary_input.null_count();
  if (input_null_count > 0) {
    return cudf::make_lists_column(num_rows,
                                   std::move(offsets_col),
                                   std::move(child_col),
                                   input_null_count,
                                   cudf::copy_bitmask(binary_input, stream, mr));
  }
  return cudf::make_lists_column(
    num_rows, std::move(offsets_col), std::move(child_col), 0, rmm::device_buffer{});
}

std::unique_ptr<cudf::column> make_null_column(cudf::data_type dtype,
                                               cudf::size_type num_rows,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  if (num_rows == 0) { return cudf::make_empty_column(dtype); }

  switch (dtype.id()) {
    case cudf::type_id::BOOL8:
    case cudf::type_id::INT8:
    case cudf::type_id::UINT8:
    case cudf::type_id::INT16:
    case cudf::type_id::UINT16:
    case cudf::type_id::INT32:
    case cudf::type_id::UINT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT64:
    case cudf::type_id::FLOAT32:
    case cudf::type_id::FLOAT64:
      return cudf::make_fixed_width_column(dtype, num_rows, cudf::mask_state::ALL_NULL, stream, mr);
    case cudf::type_id::STRING: {
      rmm::device_uvector<cudf::strings::detail::string_index_pair> pairs(num_rows, stream, mr);
      thrust::fill(rmm::exec_policy_nosync(stream, mr),
                   pairs.data(),
                   pairs.end(),
                   cudf::strings::detail::string_index_pair{nullptr, 0});
      return cudf::strings::detail::make_strings_column(pairs.data(), pairs.end(), stream, mr);
    }
    case cudf::type_id::LIST:
      return cudf::lists::detail::make_all_nulls_lists_column(
        num_rows, cudf::data_type{cudf::type_id::UINT8}, stream, mr);
    case cudf::type_id::STRUCT: {
      std::vector<std::unique_ptr<cudf::column>> empty_children;
      auto null_mask = cudf::create_null_mask(num_rows, cudf::mask_state::ALL_NULL, stream, mr);
      return cudf::make_structs_column(
        num_rows, std::move(empty_children), num_rows, std::move(null_mask), stream, mr);
    }
    default: CUDF_FAIL("Unsupported type for null column creation");
  }
}

std::unique_ptr<cudf::column> make_empty_column_safe(cudf::data_type dtype,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  switch (dtype.id()) {
    case cudf::type_id::LIST: {
      auto offsets_col =
        std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                       1,
                                       rmm::device_buffer(sizeof(int32_t), stream, mr),
                                       rmm::device_buffer{},
                                       0);
      CUDF_CUDA_TRY(cudaMemsetAsync(
        offsets_col->mutable_view().data<int32_t>(), 0, sizeof(int32_t), stream.value()));
      auto child_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::UINT8}, 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);
      return cudf::make_lists_column(
        0, std::move(offsets_col), std::move(child_col), 0, rmm::device_buffer{});
    }
    case cudf::type_id::STRUCT: {
      std::vector<std::unique_ptr<cudf::column>> empty_children;
      return cudf::make_structs_column(
        0, std::move(empty_children), 0, rmm::device_buffer{}, stream, mr);
    }
    default: return cudf::make_empty_column(dtype);
  }
}

std::unique_ptr<cudf::column> make_null_list_column_with_child(
  std::unique_ptr<cudf::column> child_col,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<int32_t> offsets(num_rows + 1, stream, mr);
  thrust::fill(rmm::exec_policy_nosync(stream, mr), offsets.begin(), offsets.end(), 0);
  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    num_rows + 1,
                                                    offsets.release(),
                                                    rmm::device_buffer{},
                                                    0);
  auto null_mask   = cudf::create_null_mask(num_rows, cudf::mask_state::ALL_NULL, stream, mr);
  return cudf::make_lists_column(
    num_rows, std::move(offsets_col), std::move(child_col), num_rows, std::move(null_mask));
}

std::unique_ptr<cudf::column> make_empty_list_column(std::unique_ptr<cudf::column> element_col,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    1,
                                                    rmm::device_buffer(sizeof(int32_t), stream, mr),
                                                    rmm::device_buffer{},
                                                    0);
  CUDF_CUDA_TRY(cudaMemsetAsync(
    offsets_col->mutable_view().data<int32_t>(), 0, sizeof(int32_t), stream.value()));
  return cudf::make_lists_column(
    0, std::move(offsets_col), std::move(element_col), 0, rmm::device_buffer{});
}

// ============================================================================
// Enum-as-string column builders
// ============================================================================

struct enum_string_lookup_tables {
  rmm::device_uvector<int32_t> d_valid_enums;
  rmm::device_uvector<int32_t> d_name_offsets;
  rmm::device_uvector<uint8_t> d_name_chars;
};

enum_string_lookup_tables make_enum_string_lookup_tables(
  cudf::detail::host_vector<int32_t> const& valid_enums,
  std::vector<cudf::detail::host_vector<uint8_t>> const& enum_name_bytes,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto d_valid_enums = cudf::detail::make_device_uvector_async(
    valid_enums, stream, cudf::get_current_device_resource_ref());

  // Stream-ordered pinned deallocation keeps these staging buffers safe without a local sync.
  auto h_name_offsets =
    cudf::detail::make_pinned_vector_async<int32_t>(valid_enums.size() + 1, stream);
  h_name_offsets[0]        = 0;
  int64_t total_name_chars = 0;
  for (size_t k = 0; k < enum_name_bytes.size(); ++k) {
    total_name_chars += static_cast<int64_t>(enum_name_bytes[k].size());
    CUDF_EXPECTS(total_name_chars <= std::numeric_limits<int32_t>::max(),
                 "Enum name data exceeds 2 GB limit");
    h_name_offsets[k + 1] = static_cast<int32_t>(total_name_chars);
  }

  auto h_name_chars = cudf::detail::make_pinned_vector_async<uint8_t>(total_name_chars, stream);
  int32_t cursor    = 0;
  for (auto const& name : enum_name_bytes) {
    if (!name.empty()) {
      std::copy(name.data(), name.data() + name.size(), h_name_chars.data() + cursor);
      cursor += static_cast<int32_t>(name.size());
    }
  }

  auto d_name_offsets = cudf::detail::make_device_uvector_async(
    h_name_offsets, stream, cudf::get_current_device_resource_ref());

  auto d_name_chars = [&]() {
    if (total_name_chars > 0) {
      return cudf::detail::make_device_uvector_async(
        h_name_chars, stream, cudf::get_current_device_resource_ref());
    }
    return rmm::device_uvector<uint8_t>(0, stream, cudf::get_current_device_resource_ref());
  }();

  return {std::move(d_valid_enums), std::move(d_name_offsets), std::move(d_name_chars)};
}

std::unique_ptr<cudf::column> build_enum_string_values_column(
  rmm::device_uvector<int32_t>& enum_values,
  rmm::device_uvector<bool>& valid,
  enum_string_lookup_tables const& lookup,
  int num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<int32_t> lengths(num_rows, stream, cudf::get_current_device_resource_ref());
  launch_compute_enum_string_lengths(enum_values.data(),
                                     valid.data(),
                                     lookup.d_valid_enums.data(),
                                     lookup.d_name_offsets.data(),
                                     static_cast<int>(lookup.d_valid_enums.size()),
                                     lengths.data(),
                                     num_rows,
                                     stream);

  auto [offsets_col, total_chars] =
    cudf::strings::detail::make_offsets_child_column(lengths.begin(), lengths.end(), stream, mr);

  rmm::device_uvector<char> chars(total_chars, stream, mr);
  if (total_chars > 0) {
    launch_copy_enum_string_chars(enum_values.data(),
                                  valid.data(),
                                  lookup.d_valid_enums.data(),
                                  lookup.d_name_offsets.data(),
                                  lookup.d_name_chars.data(),
                                  static_cast<int>(lookup.d_valid_enums.size()),
                                  offsets_col->view().data<int32_t>(),
                                  chars.data(),
                                  num_rows,
                                  stream);
  }

  auto [mask, null_count] = make_null_mask_from_valid(valid, num_rows, stream, mr);
  return cudf::make_strings_column(
    num_rows, std::move(offsets_col), chars.release(), null_count, std::move(mask));
}

std::unique_ptr<cudf::column> build_enum_string_column(
  rmm::device_uvector<int32_t>& enum_values,
  rmm::device_uvector<bool>& valid,
  cudf::detail::host_vector<int32_t> const& valid_enums,
  std::vector<cudf::detail::host_vector<uint8_t>> const& enum_name_bytes,
  protobuf_decode_runtime_context decode_ctx,
  int num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int32_t const* top_row_indices)
{
  auto lookup           = make_enum_string_lookup_tables(valid_enums, enum_name_bytes, stream, mr);
  auto const scratch_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<bool> d_item_has_invalid_enum(num_rows, stream, scratch_mr);
  thrust::fill(rmm::exec_policy_nosync(stream, scratch_mr),
               d_item_has_invalid_enum.begin(),
               d_item_has_invalid_enum.end(),
               false);

  launch_validate_enum_values(enum_values.data(),
                              valid.data(),
                              d_item_has_invalid_enum.data(),
                              lookup.d_valid_enums.data(),
                              static_cast<int>(valid_enums.size()),
                              num_rows,
                              stream);
  propagate_invalid_enum_flags_to_rows(d_item_has_invalid_enum,
                                       *decode_ctx.row_force_null,
                                       num_rows,
                                       top_row_indices,
                                       decode_ctx.propagate_invalid_enum_rows,
                                       stream);
  return build_enum_string_values_column(enum_values, valid, lookup, num_rows, stream, mr);
}

std::unique_ptr<cudf::column> build_repeated_enum_string_column(
  cudf::column_view const& binary_input,
  protobuf_input_view input,
  rmm::device_uvector<int32_t> d_field_offsets,
  rmm::device_uvector<repeated_occurrence>& d_occurrences,
  int total_count,
  cudf::detail::host_vector<int32_t> const& valid_enums,
  std::vector<cudf::detail::host_vector<uint8_t>> const& enum_name_bytes,
  protobuf_decode_runtime_context decode_ctx,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const rep_blocks =
    static_cast<int>((total_count + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  auto const scratch_mr = cudf::get_current_device_resource_ref();
  auto const lookup     = make_enum_string_lookup_tables(valid_enums, enum_name_bytes, stream, mr);

  // 1. Extract enum integer values from occurrences
  rmm::device_uvector<int32_t> enum_ints(total_count, stream, scratch_mr);
  rmm::device_uvector<bool> elem_valid(total_count, stream, scratch_mr);
  repeated_location_provider rep_loc{input.row_offsets, input.base_offset, d_occurrences.data()};
  extract_varint_kernel<int32_t, false>
    <<<rep_blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(input.message_data,
                                                           rep_loc,
                                                           total_count,
                                                           enum_ints.data(),
                                                           elem_valid.data(),
                                                           decode_ctx.error->data(),
                                                           false,
                                                           0);

  // 2. Validate enum values — mark invalid as false in elem_valid
  // (elem_valid was already populated by extract_varint_kernel: true for success, false for
  // failure)
  rmm::device_uvector<bool> d_elem_has_invalid_enum(total_count, stream, scratch_mr);
  thrust::fill(rmm::exec_policy_nosync(stream, scratch_mr),
               d_elem_has_invalid_enum.begin(),
               d_elem_has_invalid_enum.end(),
               false);
  launch_validate_enum_values(enum_ints.data(),
                              elem_valid.data(),
                              d_elem_has_invalid_enum.data(),
                              lookup.d_valid_enums.data(),
                              static_cast<int>(valid_enums.size()),
                              total_count,
                              stream);

  rmm::device_uvector<int32_t> d_top_row_indices(total_count, stream, scratch_mr);
  thrust::transform(rmm::exec_policy_nosync(stream, scratch_mr),
                    d_occurrences.begin(),
                    d_occurrences.end(),
                    d_top_row_indices.begin(),
                    [] __device__(repeated_occurrence const& occ) { return occ.row_idx; });
  // STRICT mode leaves the row buffer empty, while nested decoding disables propagation in the
  // runtime context. The callee treats both cases as no-ops.
  propagate_invalid_enum_flags_to_rows(d_elem_has_invalid_enum,
                                       *decode_ctx.row_force_null,
                                       total_count,
                                       d_top_row_indices.data(),
                                       decode_ctx.propagate_invalid_enum_rows,
                                       stream);

  auto child_col =
    build_enum_string_values_column(enum_ints, elem_valid, lookup, total_count, stream, mr);

  auto list_offs_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                      input.num_rows + 1,
                                                      d_field_offsets.release(),
                                                      rmm::device_buffer{},
                                                      0);

  return make_list_column_with_input_nulls(
    input.num_rows, std::move(list_offs_col), std::move(child_col), binary_input, stream, mr);
}

std::unique_ptr<cudf::column> build_repeated_string_column(
  cudf::column_view const& binary_input,
  protobuf_input_view input,
  rmm::device_uvector<int32_t> d_field_offsets,
  rmm::device_uvector<repeated_occurrence>& d_occurrences,
  int total_count,
  bool is_bytes,
  rmm::device_uvector<protobuf_error>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(total_count > 0, "build_repeated_string_column: total_count must be > 0");

  // Extract string lengths from occurrences
  auto const scratch_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<int32_t> str_lengths(total_count, stream, scratch_mr);
  auto const threads = THREADS_PER_BLOCK;
  auto const blocks  = static_cast<int>((total_count + threads - 1u) / threads);
  repeated_location_provider loc_provider{
    input.row_offsets, input.base_offset, d_occurrences.data()};
  extract_lengths_kernel<repeated_location_provider>
    <<<blocks, threads, 0, stream.value()>>>(loc_provider, total_count, str_lengths.data());

  auto [str_offsets_col, total_chars] = cudf::strings::detail::make_offsets_child_column(
    str_lengths.begin(), str_lengths.end(), stream, mr);

  rmm::device_uvector<char> chars(total_chars, stream, mr);
  if (total_chars > 0) {
    repeated_location_provider copy_provider{
      input.row_offsets, input.base_offset, d_occurrences.data()};
    auto const* offsets_data = str_offsets_col->view().data<cudf::size_type>();
    auto const* message_data = input.message_data;
    auto* chars_ptr          = chars.data();

    auto src_iter = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<void const*>(
        [message_data, copy_provider] __device__(int idx) -> void const* {
          int32_t data_offset = 0;
          auto loc            = copy_provider.get(idx, data_offset);
          if (loc.offset < 0) return nullptr;
          return static_cast<void const*>(message_data + data_offset);
        }));
    auto dst_iter = cudf::detail::make_counting_transform_iterator(
      0, cuda::proclaim_return_type<void*>([chars_ptr, offsets_data] __device__(int idx) -> void* {
        return static_cast<void*>(chars_ptr + offsets_data[idx]);
      }));
    auto size_iter = cudf::detail::make_counting_transform_iterator(
      0, cuda::proclaim_return_type<size_t>([copy_provider] __device__(int idx) -> size_t {
        int32_t data_offset = 0;
        auto loc            = copy_provider.get(idx, data_offset);
        if (loc.offset < 0) return 0;
        return static_cast<size_t>(loc.length);
      }));

    size_t temp_storage_bytes = 0;
    cub::DeviceMemcpy::Batched(
      nullptr, temp_storage_bytes, src_iter, dst_iter, size_iter, total_count, stream.value());
    rmm::device_buffer temp_storage(temp_storage_bytes, stream, scratch_mr);
    cub::DeviceMemcpy::Batched(temp_storage.data(),
                               temp_storage_bytes,
                               src_iter,
                               dst_iter,
                               size_iter,
                               total_count,
                               stream.value());
  }

  std::unique_ptr<cudf::column> child_col;
  if (is_bytes) {
    // Transfer ownership of the chars buffer instead of copying — the strings path below uses
    // `chars.release()` for the same reason.
    auto bytes_child = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::UINT8}, total_chars, chars.release(), rmm::device_buffer{}, 0);
    child_col = cudf::make_lists_column(
      total_count, std::move(str_offsets_col), std::move(bytes_child), 0, rmm::device_buffer{});
  } else {
    child_col = cudf::make_strings_column(
      total_count, std::move(str_offsets_col), chars.release(), 0, rmm::device_buffer{});
  }

  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    input.num_rows + 1,
                                                    d_field_offsets.release(),
                                                    rmm::device_buffer{},
                                                    0);

  // Per Spark semantics: only INPUT-null rows are null; rows with count=0 produce [].
  return make_list_column_with_input_nulls(
    input.num_rows, std::move(offsets_col), std::move(child_col), binary_input, stream, mr);
}

// ============================================================================
// Nested struct column builder
// ============================================================================

/**
 * Build a STRUCT column for a nested protobuf message.
 *
 * Scalar, string, bytes, enum-as-string, default values, proto2 required-field checks,
 * repeated non-message children, and recursive STRUCT children are decoded.
 */
std::unique_ptr<cudf::column> build_nested_struct_column(
  protobuf_input_view input,
  nested_parent_view parent,
  std::vector<int> const& child_field_indices,
  std::vector<nested_field_descriptor> const& schema,
  int num_fields,
  schema_context_view schema_ctx,
  protobuf_decode_runtime_context decode_ctx,
  int depth,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(depth < MAX_NESTING_DEPTH,
               "Nested protobuf struct depth exceeds supported decode recursion limit");
  validate_nested_parent_view(input, parent);
  validate_protobuf_decode_context(decode_ctx, input.num_rows);

  if (input.num_rows == 0) {
    return make_empty_struct_column_from_children(
      schema, child_field_indices, num_fields, stream, mr);
  }

  int num_child_fields = static_cast<int>(child_field_indices.size());
  std::vector<int> repeated_child_positions;
  repeated_child_positions.reserve(num_child_fields);

  // Stage child field descriptors through pinned memory so the H2D stays stream-async.
  // Stream-ordered pinned deallocation keeps this staging safe without a local sync.
  auto h_child_field_descs =
    cudf::detail::make_pinned_vector_async<field_descriptor>(num_child_fields, stream);
  for (int i = 0; i < num_child_fields; i++) {
    int child_idx                             = child_field_indices[i];
    h_child_field_descs[i].field_number       = schema[child_idx].field_number;
    h_child_field_descs[i].expected_wire_type = static_cast<int>(schema[child_idx].wire_type);
    h_child_field_descs[i].is_repeated        = schema[child_idx].is_repeated;
    if (schema[child_idx].is_repeated) { repeated_child_positions.push_back(i); }
  }

  auto const scratch_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<field_descriptor> d_child_field_descs(
    std::max(num_child_fields, 1), stream, scratch_mr);
  if (num_child_fields > 0) {
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(d_child_field_descs.data(),
                                             h_child_field_descs.data(),
                                             num_child_fields * sizeof(field_descriptor),
                                             stream));
  }

  auto const child_location_count = static_cast<size_t>(input.num_rows) * num_child_fields;
  rmm::device_uvector<field_location> d_child_locations(
    std::max(child_location_count, size_t{1}), stream, scratch_mr);
  rmm::device_uvector<repeated_field_info> d_repeated_info(
    repeated_child_positions.empty() ? 0 : child_location_count, stream, scratch_mr);
  CUDF_EXPECTS(repeated_child_positions.empty() || d_repeated_info.size() == child_location_count,
               "Protobuf decode internal error: nested repeated count buffer size mismatch");
  // Repeated counts are collected in the same pass as singleton locations so each nested
  // message is parsed only once before LIST offsets are built.
  launch_scan_nested_message_fields(
    input.message_data,
    input.message_data_size,
    input.row_offsets,
    input.base_offset,
    parent.locations,
    input.num_rows,
    d_child_field_descs.data(),
    num_child_fields,
    d_child_locations.data(),
    d_repeated_info.data(),
    decode_ctx.error->data(),
    !decode_ctx.row_force_null->is_empty() ? decode_ctx.row_force_null->data() : nullptr,
    parent.top_row_indices,
    stream);

  maybe_check_required_fields(
    d_child_locations.data(),
    child_field_indices,
    schema,
    input.num_rows,
    nullptr,
    0,
    parent.locations,
    !decode_ctx.row_force_null->is_empty() ? decode_ctx.row_force_null->data() : nullptr,
    parent.top_row_indices,
    decode_ctx.error->data(),
    stream);

  std::vector<std::optional<repeated_field_work>> repeated_work(num_child_fields);
  auto h_scan_descs = cudf::detail::make_pinned_vector_async<repeated_field_scan_desc>(0, stream);
  h_scan_descs.reserve(repeated_child_positions.size());
  for (auto const ci : repeated_child_positions) {
    // The row-major buffer has `num_child_fields` entries per row, so this strided iterator
    // yields exactly one count for each input row.
    auto const child_schema_idx = child_field_indices[ci];
    auto counts_begin           = thrust::make_transform_iterator(
      thrust::make_counting_iterator<int>(0),
      extract_strided_count{d_repeated_info.data(), ci, num_child_fields});
    repeated_work[ci].emplace(
      child_schema_idx,
      make_list_offsets_from_counts(
        counts_begin, input.num_rows, "Repeated nested-field", stream, mr, scratch_mr));

    auto& work = *repeated_work[ci];
    if (work.total_count > 0) {
      work.occurrences = std::make_unique<rmm::device_uvector<repeated_occurrence>>(
        work.total_count, stream, scratch_mr);
      h_scan_descs.push_back({schema[child_schema_idx].field_number,
                              static_cast<int>(schema[child_schema_idx].wire_type),
                              work.offsets.data(),
                              work.occurrences->data()});
    }
  }

  if (!h_scan_descs.empty()) {
    rmm::device_uvector<repeated_field_scan_desc> d_scan_descs(
      h_scan_descs.size(), stream, scratch_mr);
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(d_scan_descs.data(),
                                             h_scan_descs.data(),
                                             h_scan_descs.size() * sizeof(repeated_field_scan_desc),
                                             stream));

    auto h_fn_to_scan =
      build_field_lookup_table(h_scan_descs.data(), static_cast<int>(h_scan_descs.size()), stream);
    rmm::device_uvector<int> d_fn_to_scan(0, stream, scratch_mr);
    int fn_to_scan_size = 0;
    if (!h_fn_to_scan.empty()) {
      d_fn_to_scan = rmm::device_uvector<int>(h_fn_to_scan.size(), stream, scratch_mr);
      CUDF_CUDA_TRY(cudf::detail::memcpy_async(
        d_fn_to_scan.data(), h_fn_to_scan.data(), h_fn_to_scan.size() * sizeof(int), stream));
      fn_to_scan_size = static_cast<int>(h_fn_to_scan.size());
    }

    launch_scan_all_repeated_occurrences_in_nested(
      input.message_data,
      input.message_data_size,
      input.row_offsets,
      input.base_offset,
      parent.locations,
      d_scan_descs.data(),
      static_cast<int>(d_scan_descs.size()),
      decode_ctx.error->data(),
      fn_to_scan_size > 0 ? d_fn_to_scan.data() : nullptr,
      fn_to_scan_size,
      input.num_rows,
      stream);
  }

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  for (int ci = 0; ci < num_child_fields; ci++) {
    int child_schema_idx = child_field_indices[ci];
    auto const dt        = cudf::data_type{schema[child_schema_idx].output_type};
    bool has_def         = schema[child_schema_idx].has_default_value;
    bool is_repeated     = schema[child_schema_idx].is_repeated;

    if (is_repeated) {
      CUDF_EXPECTS(repeated_work[ci].has_value(),
                   "Protobuf decode internal error: missing nested repeated-field work");
      struct_children.push_back(
        build_repeated_child_list_column(input,
                                         parent,
                                         schema,
                                         schema_ctx,
                                         decode_ctx,
                                         std::move(repeated_work[ci].value()),
                                         stream,
                                         mr));
      continue;
    }

    nested_location_provider loc_provider{input.row_offsets,
                                          input.base_offset,
                                          parent.locations,
                                          d_child_locations.data(),
                                          ci,
                                          num_child_fields};

    if (dt.id() == cudf::type_id::STRUCT) {
      // Recursive linear child lookup is fine for realistic schemas; precompute if it gets hot.
      auto gc_indices = find_child_field_indices(schema, num_fields, child_schema_idx);
      rmm::device_uvector<field_location> d_gc_parent_locs(input.num_rows, stream, scratch_mr);
      launch_compute_grandchild_parent_locations(parent.locations,
                                                 d_child_locations.data(),
                                                 ci,
                                                 num_child_fields,
                                                 d_gc_parent_locs.data(),
                                                 input.num_rows,
                                                 decode_ctx.error->data(),
                                                 stream);
      struct_children.push_back(build_nested_struct_column(
        input,
        {d_gc_parent_locs.data(), d_gc_parent_locs.size(), parent.top_row_indices},
        gc_indices,
        schema,
        num_fields,
        schema_ctx,
        decode_ctx,
        depth + 1,
        stream,
        mr));
      continue;
    }

    auto valid_fn = [loc_provider, has_def] __device__(cudf::size_type row) {
      return has_def || loc_provider.valid(row);
    };
    auto get_top_row_indices = [top_row_indices = parent.top_row_indices]() {
      return top_row_indices;
    };
    struct_children.push_back(build_protobuf_field_values_column(input.message_data,
                                                                 schema[child_schema_idx],
                                                                 child_schema_idx,
                                                                 input.num_rows,
                                                                 loc_provider,
                                                                 valid_fn,
                                                                 has_def,
                                                                 schema_ctx,
                                                                 decode_ctx,
                                                                 get_top_row_indices,
                                                                 stream,
                                                                 mr));
  }

  auto [struct_mask, struct_null_count] =
    make_null_mask_from_parent_locations(parent.locations, input.num_rows, stream, mr);
  return cudf::make_structs_column(input.num_rows,
                                   std::move(struct_children),
                                   struct_null_count,
                                   std::move(struct_mask),
                                   stream,
                                   mr);
}

std::unique_ptr<cudf::column> build_repeated_child_list_column(
  protobuf_input_view input,
  nested_parent_view parent,
  std::vector<nested_field_descriptor> const& schema,
  schema_context_view schema_ctx,
  protobuf_decode_runtime_context decode_ctx,
  repeated_field_work work,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  validate_nested_parent_view(input, parent);
  validate_protobuf_decode_context(decode_ctx, input.num_rows);
  auto const child_schema_idx = work.schema_idx;
  CUDF_EXPECTS(child_schema_idx >= 0 && child_schema_idx < static_cast<int>(schema.size()),
               "Protobuf decode internal error: nested repeated schema index is out of bounds");
  CUDF_EXPECTS(schema[child_schema_idx].is_repeated,
               "nested repeated child builder requires a repeated child schema");
  auto const elem_type = cudf::data_type{schema[child_schema_idx].output_type};
  CUDF_EXPECTS(elem_type.id() != cudf::type_id::STRUCT,
               "Protobuf decode: nested repeated MessageType is not yet supported");

  CUDF_EXPECTS(work.offsets.size() == static_cast<size_t>(input.num_rows) + 1,
               "Protobuf decode internal error: nested repeated offsets size mismatch");
  auto const scratch_mr  = cudf::get_current_device_resource_ref();
  auto const total_count = work.total_count;

  if (total_count == 0) {
    auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                      input.num_rows + 1,
                                                      work.offsets.release(),
                                                      rmm::device_buffer{},
                                                      0);
    auto child_col   = make_empty_column_safe(elem_type, stream, mr);
    return make_list_column_with_parent_nulls(
      input.num_rows, std::move(offsets_col), std::move(child_col), parent.locations, stream, mr);
  }

  CUDF_EXPECTS(work.occurrences != nullptr,
               "Protobuf decode internal error: missing nested repeated occurrences");
  CUDF_EXPECTS(work.occurrences->size() == static_cast<size_t>(total_count),
               "Protobuf decode internal error: nested repeated occurrences size mismatch");
  auto list_offsets   = std::move(work.offsets);
  auto& d_occurrences = *work.occurrences;

  std::unique_ptr<rmm::device_uvector<int32_t>> d_top_row_indices;
  auto const* top_row_indices = parent.top_row_indices;
  auto get_top_row_indices    = [&]() -> int32_t const* {
    if (d_top_row_indices == nullptr) {
      d_top_row_indices =
        std::make_unique<rmm::device_uvector<int32_t>>(total_count, stream, scratch_mr);
      thrust::transform(rmm::exec_policy_nosync(stream, scratch_mr),
                        d_occurrences.begin(),
                        d_occurrences.end(),
                        d_top_row_indices->begin(),
                        [top_row_indices] __device__(repeated_occurrence const& occ) {
                          return top_row_indices != nullptr ? top_row_indices[occ.row_idx]
                                                               : occ.row_idx;
                        });
    }
    return d_top_row_indices->data();
  };

  nested_repeated_location_provider loc_provider{
    input.row_offsets, input.base_offset, parent.locations, d_occurrences.data()};
  auto valid_fn     = [] __device__(cudf::size_type) { return true; };
  auto child_values = build_protobuf_field_values_column(input.message_data,
                                                         schema[child_schema_idx],
                                                         child_schema_idx,
                                                         total_count,
                                                         loc_provider,
                                                         valid_fn,
                                                         false,
                                                         schema_ctx,
                                                         decode_ctx,
                                                         get_top_row_indices,
                                                         stream,
                                                         mr);

  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    input.num_rows + 1,
                                                    list_offsets.release(),
                                                    rmm::device_buffer{},
                                                    0);
  return make_list_column_with_parent_nulls(
    input.num_rows, std::move(offsets_col), std::move(child_values), parent.locations, stream, mr);
}

}  // namespace spark_rapids_jni::protobuf::detail
