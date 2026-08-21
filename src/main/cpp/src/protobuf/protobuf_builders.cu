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
#include <cudf/lists/stream_compaction.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/unary.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <source_location>
#include <string>
#include <utility>

namespace spark_rapids_jni::protobuf::detail {

enum_string_lookup_tables make_enum_string_lookup_tables(
  cudf::detail::host_vector<int32_t> const& valid_enums,
  std::vector<cudf::detail::host_vector<uint8_t>> const& enum_name_bytes,
  rmm::cuda_stream_view stream);

enum_string_lookup_tables protobuf_schema::enum_lookup(int schema_idx,
                                                       rmm::cuda_stream_view stream) const
{
  auto const field = this->field(schema_idx);
  return make_enum_string_lookup_tables(field.enum_valid_values, field.enum_names, stream);
}

field_descriptor_bundle make_field_descriptors(std::vector<int> const& field_indices,
                                               protobuf_schema const& schema,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr,
                                               std::span<int const> output_indices)
{
  CUDF_EXPECTS(output_indices.empty() || output_indices.size() == field_indices.size(),
               "protobuf field descriptor output index count must match field count");
  size_t num_enum_values = 0;
  for (auto const schema_idx : field_indices) {
    num_enum_values += schema.field(schema_idx).enum_valid_values.size();
  }

  auto h_enum_values = cudf::detail::make_pinned_vector_async<int32_t>(num_enum_values, stream);
  size_t enum_offset = 0;
  for (auto const schema_idx : field_indices) {
    auto const& values = schema.field(schema_idx).enum_valid_values;
    std::copy(values.begin(), values.end(), h_enum_values.begin() + enum_offset);
    enum_offset += values.size();
  }
  auto d_enum_values = cudf::detail::make_device_uvector_async(h_enum_values, stream, mr);

  auto h_descriptors =
    cudf::detail::make_pinned_vector_async<field_descriptor>(field_indices.size(), stream);
  enum_offset = 0;
  for (size_t i = 0; i < field_indices.size(); ++i) {
    auto const schema_idx = field_indices[i];
    auto const field      = schema.field(schema_idx);
    auto const enum_size  = field.enum_valid_values.size();
    CUDF_EXPECTS(enum_size <= static_cast<size_t>(std::numeric_limits<int>::max()),
                 "protobuf enum metadata exceeds supported value count");
    h_descriptors[i] = {field.schema.field_number,
                        field.schema.wire_type,
                        field.schema.is_repeated,
                        field.schema.output_type == cudf::type_id::STRUCT,
                        enum_size > 0 ? d_enum_values.data() + enum_offset : nullptr,
                        static_cast<int>(enum_size)};
    if (!output_indices.empty()) { h_descriptors[i].output_index = output_indices[i]; }
    enum_offset += enum_size;
  }

  auto d_descriptors = cudf::detail::make_device_uvector_async(h_descriptors, stream, mr);
  return {std::move(h_descriptors), std::move(d_descriptors), std::move(d_enum_values)};
}

namespace {

std::unique_ptr<cudf::column> build_enum_string_column_with_lookup(
  rmm::device_uvector<int32_t>& enum_values,
  rmm::device_uvector<bool>& valid,
  enum_string_lookup_tables const& lookup,
  int num_values,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

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
  protobuf_input_view input,
  nested_parent_view parent,
  std::source_location const& location = std::source_location::current())
{
  auto const caller = location.function_name();
  CUDF_EXPECTS(context.row_force_null != nullptr,
               std::string{caller} + ": row-force-null buffer must be non-null");
  CUDF_EXPECTS(context.error != nullptr, std::string{caller} + ": error buffer must be non-null");
  CUDF_EXPECTS(context.error->size() == 1,
               std::string{caller} + ": error buffer must contain exactly one element");
  CUDF_EXPECTS(
    context.row_force_null->is_empty() || parent.top_row_indices != nullptr ||
      context.row_force_null->size() == static_cast<size_t>(input.num_rows),
    std::string{caller} + ": row-force-null buffer must be empty, row-sized, or remapped");
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

std::unique_ptr<cudf::column> drop_unknown_repeated_enum_values_impl(
  std::unique_ptr<cudf::column> input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const input_view = cudf::lists_column_view{input->view()};
  CUDF_EXPECTS(input_view.offset() == 0,
               "repeated enum filtering requires an unsliced list column");
  auto const child = input_view.get_sliced_child(stream);
  if (child.null_count() == 0) return input;

  // protobuf-java omits unknown proto2 enum occurrences from repeated fields.
  auto const scratch_mr = cudf::get_current_device_resource_ref();
  auto keep_values      = cudf::is_valid(child, stream, scratch_mr);
  auto keep_offsets     = std::make_unique<cudf::column>(input_view.offsets(), stream, scratch_mr);
  auto keep_lists       = cudf::make_lists_column(
    input_view.size(), std::move(keep_offsets), std::move(keep_values), 0, rmm::device_buffer{});
  return cudf::lists::apply_boolean_mask(
    input_view, cudf::lists_column_view{keep_lists->view()}, stream, mr);
}

template <typename LocationProvider, typename ValidityFn>
std::unique_ptr<cudf::column> build_protobuf_field_values_column(
  protobuf_field_decode_request request,
  LocationProvider const& loc_provider,
  ValidityFn validity_fn,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return build_protobuf_field_values_column_shared(request, loc_provider, validity_fn, stream, mr);
}

}  // namespace

std::unique_ptr<cudf::column> drop_unknown_repeated_enum_values(std::unique_ptr<cudf::column> input,
                                                                rmm::cuda_stream_view stream,
                                                                rmm::device_async_resource_ref mr)
{
  return drop_unknown_repeated_enum_values_impl(std::move(input), stream, mr);
}

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
  auto offsets_col = make_offsets_column(num_rows, std::move(offsets));
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

enum_string_lookup_tables make_enum_string_lookup_tables(
  cudf::detail::host_vector<int32_t> const& valid_enums,
  std::vector<cudf::detail::host_vector<uint8_t>> const& enum_name_bytes,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(valid_enums.size() <= static_cast<size_t>(std::numeric_limits<int>::max()),
               "protobuf enum metadata exceeds supported value count");
  CUDF_EXPECTS(valid_enums.size() == enum_name_bytes.size(),
               "protobuf enum values and names must have matching sizes");
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
  auto const input = enum_value_device_view{enum_values.data(), valid.data(), num_rows};
  launch_compute_enum_string_lengths(input, lookup.view(), lengths.data(), stream);

  auto [offsets_col, total_chars] =
    cudf::strings::detail::make_offsets_child_column(lengths.begin(), lengths.end(), stream, mr);

  rmm::device_uvector<char> chars(total_chars, stream, mr);
  if (total_chars > 0) {
    launch_copy_enum_string_chars(
      input, lookup.view(), offsets_col->view().data<int32_t>(), chars.data(), stream);
  }

  auto [mask, null_count] = make_null_mask_from_valid(valid, num_rows, stream, mr);
  return cudf::make_strings_column(
    num_rows, std::move(offsets_col), chars.release(), null_count, std::move(mask));
}

namespace {

std::unique_ptr<cudf::column> build_enum_string_column_with_lookup(
  rmm::device_uvector<int32_t>& enum_values,
  rmm::device_uvector<bool>& valid,
  enum_string_lookup_tables const& lookup,
  int num_values,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  validate_enum_values(enum_values, valid, lookup.view().domain, stream);
  return build_enum_string_values_column(enum_values, valid, lookup, num_values, stream, mr);
}

}  // namespace

std::unique_ptr<cudf::column> build_enum_string_column(rmm::device_uvector<int32_t>& enum_values,
                                                       rmm::device_uvector<bool>& valid,
                                                       protobuf_field_decode_request request,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  auto const lookup = request.context.schema.enum_lookup(request.schema_idx, stream);
  return build_enum_string_column_with_lookup(
    enum_values, valid, lookup, request.num_values, stream, mr);
}

std::unique_ptr<cudf::column> build_repeated_enum_string_column(
  cudf::column_view const& binary_input,
  protobuf_input_view input,
  protobuf_schema const& schema,
  protobuf_decode_runtime_context decode_ctx,
  repeated_field_work work,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  validate_nonempty_repeated_field_work(work, input.num_rows);

  auto const total_count = work.total_count;
  auto& occurrences      = *work.occurrences;
  auto const scratch_mr  = cudf::get_current_device_resource_ref();

  // 1. Extract enum integer values from occurrences
  rmm::device_uvector<int32_t> enum_ints(total_count, stream, scratch_mr);
  rmm::device_uvector<bool> elem_valid(total_count, stream, scratch_mr);
  repeated_location_provider rep_loc{input.row_offsets, input.base_offset, occurrences.data()};
  extract_scalar_into_buffers<int32_t>(
    input.message_data,
    rep_loc,
    total_count,
    proto_encoding::DEFAULT,
    {false, int32_t{0}},
    {enum_ints.data(), elem_valid.data(), decode_ctx.error->data()},
    stream);

  auto const context = recursive_decode_context{schema, decode_ctx};
  auto const request =
    protobuf_field_decode_request{context, input.message_data, work.schema_idx, total_count};
  auto child_col = build_enum_string_column(enum_ints, elem_valid, request, stream, mr);

  auto list_offs_col = make_offsets_column(input.num_rows, std::move(work.offsets));

  auto result = make_list_column_with_input_nulls(
    input.num_rows, std::move(list_offs_col), std::move(child_col), binary_input, stream, mr);
  // Proto2 repeated enums omit unrecognized values without invalidating sibling fields.
  return drop_unknown_repeated_enum_values(std::move(result), stream, mr);
}

std::unique_ptr<cudf::column> build_repeated_string_column(cudf::column_view const& binary_input,
                                                           protobuf_input_view input,
                                                           protobuf_field_meta_view field,
                                                           repeated_field_work work,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr)
{
  validate_nonempty_repeated_field_work(work, input.num_rows);

  auto const total_count = work.total_count;
  auto& occurrences      = *work.occurrences;
  auto const is_bytes    = field.output_type.id() == cudf::type_id::LIST;
  // Extract string lengths from occurrences
  auto const scratch_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<int32_t> str_lengths(total_count, stream, scratch_mr);
  auto const threads = THREADS_PER_BLOCK;
  auto const blocks  = static_cast<int>((total_count + threads - 1u) / threads);
  repeated_location_provider loc_provider{input.row_offsets, input.base_offset, occurrences.data()};
  // STRING must size repaired UTF-8; BYTES can reuse the encoded payload length.
  if (is_bytes) {
    extract_lengths_kernel<repeated_location_provider>
      <<<blocks, threads, 0, stream.value()>>>(loc_provider, total_count, str_lengths.data());
  } else {
    extract_utf8_lengths_kernel<repeated_location_provider><<<blocks, threads, 0, stream.value()>>>(
      input.message_data, loc_provider, total_count, str_lengths.data(), nullptr);
  }
  CUDF_CHECK_CUDA(stream.value());

  auto [str_offsets_col, total_chars] = cudf::strings::detail::make_offsets_child_column(
    str_lengths.begin(), str_lengths.end(), stream, mr);

  rmm::device_uvector<char> chars(total_chars, stream, mr);
  if (total_chars > 0) {
    repeated_location_provider copy_provider{
      input.row_offsets, input.base_offset, occurrences.data()};
    auto const* offsets_data = str_offsets_col->view().data<cudf::size_type>();
    auto const* message_data = input.message_data;
    auto* chars_ptr          = chars.data();

    if (!is_bytes) {
      copy_repaired_utf8_kernel<repeated_location_provider><<<blocks, threads, 0, stream.value()>>>(
        input.message_data, copy_provider, total_count, offsets_data, chars_ptr);
      CUDF_CHECK_CUDA(stream.value());
    } else {
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
        0,
        cuda::proclaim_return_type<void*>([chars_ptr, offsets_data] __device__(int idx) -> void* {
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
      CUDF_CUDA_TRY(cub::DeviceMemcpy::Batched(
        nullptr, temp_storage_bytes, src_iter, dst_iter, size_iter, total_count, stream.value()));
      rmm::device_buffer temp_storage(temp_storage_bytes, stream, scratch_mr);
      CUDF_CUDA_TRY(cub::DeviceMemcpy::Batched(temp_storage.data(),
                                               temp_storage_bytes,
                                               src_iter,
                                               dst_iter,
                                               size_iter,
                                               total_count,
                                               stream.value()));
    }
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

  auto offsets_col = make_offsets_column(input.num_rows, std::move(work.offsets));

  // Per Spark semantics: only INPUT-null rows are null; rows with count=0 produce [].
  return make_list_column_with_input_nulls(
    input.num_rows, std::move(offsets_col), std::move(child_col), binary_input, stream, mr);
}

// ============================================================================
// Nested struct column builder
// ============================================================================

std::unique_ptr<cudf::column> build_merged_singular_struct_column(
  protobuf_input_view input,
  message_fragment_source_view source,
  std::vector<int> const& child_field_indices,
  recursive_decode_context context,
  singular_message_merge_buffers buffers,
  int depth,
  bool materialize_output,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(buffers.total_fragments > 1, "duplicate merge requires multiple fragments");
  CUDF_EXPECTS(buffers.row_offsets.size() == static_cast<size_t>(input.num_rows) + 1,
               "fragment offsets size must match row count");
  CUDF_EXPECTS(buffers.fragments.size() == static_cast<size_t>(buffers.total_fragments),
               "fragment count mismatch");
  auto const scratch_mr = cudf::get_current_device_resource_ref();

  auto validation_fields =
    make_field_descriptors(child_field_indices, context.schema, stream, scratch_mr);
  auto h_field_lookup = build_field_lookup_table(
    validation_fields.host.data(), static_cast<int>(validation_fields.host.size()), stream);
  auto d_field_lookup = cudf::detail::make_device_uvector_async(h_field_lookup, stream, scratch_mr);

  auto invalid_rows =
    cudf::detail::make_zeroed_device_uvector_async<bool>(input.num_rows, stream, scratch_mr);
  message_fragment_location_provider fragment_locations{input, source, buffers.fragments.data()};
  launch_validate_message_fragments(
    fragment_locations,
    {{validation_fields.device.data(),
      static_cast<int>(validation_fields.device.size()),
      d_field_lookup.is_empty() ? nullptr : d_field_lookup.data(),
      static_cast<int>(d_field_lookup.size())}},
    buffers.total_fragments,
    invalid_rows.data(),
    context.runtime.row_force_null->is_empty() ? nullptr : context.runtime.row_force_null->data(),
    context.runtime.error->data(),
    depth + 1,
    stream);

  auto fragment_lengths = thrust::make_transform_iterator(
    buffers.fragments.begin(),
    [] __device__(field_occurrence const& fragment) -> int32_t { return fragment.length; });
  auto fragment_byte_offsets = make_list_offsets_from_counts(fragment_lengths,
                                                             buffers.total_fragments,
                                                             "Merged singular message",
                                                             stream,
                                                             scratch_mr,
                                                             scratch_mr);
  auto const total_bytes     = fragment_byte_offsets.total_count;

  rmm::device_uvector<cudf::size_type> merged_row_offsets(input.num_rows + 1, stream, scratch_mr);
  thrust::transform(rmm::exec_policy_nosync(stream, scratch_mr),
                    thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(input.num_rows + 1),
                    merged_row_offsets.begin(),
                    [row_fragment_offsets = buffers.row_offsets.data(),
                     fragment_offsets = fragment_byte_offsets.offsets.data()] __device__(int row) {
                      return fragment_offsets[row_fragment_offsets[row]];
                    });

  rmm::device_uvector<uint8_t> merged_data(std::max<int32_t>(total_bytes, 1), stream, scratch_mr);
  if (total_bytes > 0) {
    auto const* invalid          = invalid_rows.data();
    auto const* fragments        = buffers.fragments.data();
    auto const* fragment_offsets = fragment_byte_offsets.offsets.data();
    auto* output                 = merged_data.data();

    auto src_iter = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<void const*>(
        [message_data = input.message_data, fragment_locations, fragments, invalid] __device__(
          int idx) -> void const* {
          if (invalid[fragments[idx].row_idx]) { return nullptr; }
          int32_t data_offset = 0;
          auto const location = fragment_locations.get(idx, data_offset);
          return location.offset < 0 ? nullptr
                                     : static_cast<void const*>(message_data + data_offset);
        }));
    auto dst_iter = cudf::detail::make_counting_transform_iterator(
      0, cuda::proclaim_return_type<void*>([output, fragment_offsets] __device__(int idx) -> void* {
        return static_cast<void*>(output + fragment_offsets[idx]);
      }));
    auto size_iter = cudf::detail::make_counting_transform_iterator(
      0, cuda::proclaim_return_type<size_t>([fragments, invalid] __device__(int idx) -> size_t {
        auto const fragment = fragments[idx];
        return invalid[fragment.row_idx] ? 0 : static_cast<size_t>(fragment.length);
      }));

    size_t temp_storage_bytes = 0;
    CUDF_CUDA_TRY(cub::DeviceMemcpy::Batched(nullptr,
                                             temp_storage_bytes,
                                             src_iter,
                                             dst_iter,
                                             size_iter,
                                             buffers.total_fragments,
                                             stream.value()));
    rmm::device_buffer temp_storage(temp_storage_bytes, stream, scratch_mr);
    CUDF_CUDA_TRY(cub::DeviceMemcpy::Batched(temp_storage.data(),
                                             temp_storage_bytes,
                                             src_iter,
                                             dst_iter,
                                             size_iter,
                                             buffers.total_fragments,
                                             stream.value()));
  }

  rmm::device_uvector<field_location> merged_parent_locations(input.num_rows, stream, scratch_mr);
  thrust::transform(
    rmm::exec_policy_nosync(stream, scratch_mr),
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(input.num_rows),
    merged_parent_locations.begin(),
    [row_fragment_offsets = buffers.row_offsets.data(),
     row_byte_offsets     = merged_row_offsets.data(),
     invalid              = invalid_rows.data()] __device__(int row) {
      if (invalid[row] || row_fragment_offsets[row] == row_fragment_offsets[row + 1]) {
        return field_location{-1, 0};
      }
      return field_location{0, row_byte_offsets[row + 1] - row_byte_offsets[row]};
    });

  return build_nested_struct_column(
    {merged_data.data(),
     static_cast<cudf::size_type>(total_bytes),
     merged_row_offsets.data(),
     0,
     input.num_rows},
    {merged_parent_locations.data(), merged_parent_locations.size(), source.top_row_indices},
    child_field_indices,
    context,
    depth,
    materialize_output,
    stream,
    mr);
}

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
  recursive_decode_context context,
  int depth,
  bool materialize_output,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const& schema_context = context.schema;
  auto const& schema         = schema_context.fields();
  auto const decode_ctx      = context.runtime;
  CUDF_EXPECTS(depth < MAX_NESTING_DEPTH,
               "Nested protobuf struct depth exceeds supported decode recursion limit");
  validate_nested_parent_view(input, parent);
  validate_protobuf_decode_context(decode_ctx, input, parent);

  if (input.num_rows == 0) {
    if (!materialize_output) { return nullptr; }
    return make_empty_struct_column_from_children(schema_context, child_field_indices, stream, mr);
  }

  int num_child_fields = static_cast<int>(child_field_indices.size());
  std::vector<int> repeated_child_positions;
  std::vector<int> repeated_work_positions;
  std::vector<int> singular_message_positions;
  repeated_child_positions.reserve(num_child_fields);
  repeated_work_positions.reserve(num_child_fields);
  singular_message_positions.reserve(num_child_fields);

  for (int i = 0; i < num_child_fields; i++) {
    int child_idx = child_field_indices[i];
    if (schema[child_idx].is_repeated) {
      repeated_child_positions.push_back(i);
      if (materialize_output || schema[child_idx].output_type == cudf::type_id::STRUCT) {
        repeated_work_positions.push_back(i);
      }
    } else if (schema[child_idx].output_type == cudf::type_id::STRUCT) {
      singular_message_positions.push_back(i);
    }
  }

  auto const scratch_mr = cudf::get_current_device_resource_ref();
  auto child_field_descs =
    make_field_descriptors(child_field_indices, schema_context, stream, scratch_mr);
  auto const& d_child_field_descs = child_field_descs.device;

  auto const child_location_count = static_cast<size_t>(input.num_rows) * num_child_fields;
  rmm::device_uvector<field_location> d_child_locations(
    std::max(child_location_count, size_t{1}), stream, scratch_mr);
  rmm::device_uvector<field_occurrence_count> d_occurrence_info(
    repeated_child_positions.empty() && singular_message_positions.empty() ? 0
                                                                           : child_location_count,
    stream,
    scratch_mr);
  CUDF_EXPECTS((repeated_child_positions.empty() && singular_message_positions.empty()) ||
                 d_occurrence_info.size() == child_location_count,
               "Protobuf decode internal error: nested occurrence count buffer size mismatch");
  auto d_multiple_message_fields = cudf::detail::make_zeroed_device_uvector_async<int>(
    singular_message_positions.empty() ? 0 : num_child_fields, stream, scratch_mr);
  auto const occurrence_stride = d_occurrence_info.is_empty() ? 0 : num_child_fields;
  // Occurrence counts are collected with singleton locations so duplicate-message and LIST
  // offsets do not require another count pass.
  launch_scan_nested_message_fields(
    input,
    parent,
    {d_child_locations.data(),
     num_child_fields,
     d_occurrence_info.data(),
     occurrence_stride,
     d_occurrence_info.data(),
     occurrence_stride,
     d_multiple_message_fields.data(),
     {d_child_field_descs.data(), num_child_fields, nullptr, 0}},
    decode_ctx.error->data(),
    !decode_ctx.row_force_null->is_empty() ? decode_ctx.row_force_null->data() : nullptr,
    depth + 1,
    stream);

  maybe_check_required_fields({d_child_locations.data(),
                               {input.num_rows, parent.top_row_indices},
                               nullptr,
                               0,
                               parent.locations},
                              child_field_indices,
                              schema,
                              decode_ctx,
                              stream);

  std::vector<std::optional<singular_message_merge_buffers>> message_merge_buffers(
    num_child_fields);
  if (!singular_message_positions.empty()) {
    auto h_multiple_message_fields =
      cudf::detail::make_pinned_vector_async<int>(num_child_fields, stream);
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(h_multiple_message_fields.data(),
                                             d_multiple_message_fields.data(),
                                             num_child_fields * sizeof(int),
                                             stream));
    stream.synchronize();

    std::vector<int> merge_positions;
    for (auto const ci : singular_message_positions) {
      if (h_multiple_message_fields[ci] != 0) { merge_positions.push_back(ci); }
    }
    auto merge_bundle = make_repeated_field_work_bundle(merge_positions,
                                                        child_field_indices,
                                                        d_occurrence_info.data(),
                                                        input.num_rows,
                                                        schema_context,
                                                        "Nested singular message",
                                                        stream,
                                                        scratch_mr,
                                                        scratch_mr);
    for (auto const ci : merge_positions) {
      message_merge_buffers[ci].emplace(std::move(*merge_bundle.fields[ci]));
    }
    launch_occurrence_scan_batches(
      merge_bundle.scan_descriptors, stream, scratch_mr, [&](field_occurrence_scan_view fields) {
        launch_scan_all_field_occurrences_in_nested(
          input, parent, fields, decode_ctx.error->data(), depth + 1, stream);
      });
  }

  auto repeated_work = make_repeated_field_work_bundle(repeated_work_positions,
                                                       child_field_indices,
                                                       d_occurrence_info.data(),
                                                       input.num_rows,
                                                       schema_context,
                                                       "Repeated nested-field",
                                                       stream,
                                                       materialize_output ? mr : scratch_mr,
                                                       scratch_mr);

  launch_occurrence_scan_batches(
    repeated_work.scan_descriptors, stream, scratch_mr, [&](field_occurrence_scan_view fields) {
      launch_scan_all_field_occurrences_in_nested(
        input, parent, fields, decode_ctx.error->data(), depth + 1, stream);
    });

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  for (int ci = 0; ci < num_child_fields; ci++) {
    int child_schema_idx = child_field_indices[ci];
    auto const dt        = cudf::data_type{schema[child_schema_idx].output_type};
    bool has_def         = schema[child_schema_idx].has_default_value;
    bool is_repeated     = schema[child_schema_idx].is_repeated;

    if (is_repeated) {
      if (!repeated_work.fields[ci].has_value()) { continue; }
      auto child = build_repeated_child_list_column(input,
                                                    parent,
                                                    context,
                                                    std::move(repeated_work.fields[ci].value()),
                                                    materialize_output,
                                                    stream,
                                                    mr);
      if (materialize_output) { struct_children.push_back(std::move(child)); }
      continue;
    }

    if (dt.id() == cudf::type_id::STRUCT) {
      auto const& gc_indices = schema_context.children(child_schema_idx);
      if (message_merge_buffers[ci].has_value()) {
        auto child = build_merged_singular_struct_column(input,
                                                         {parent.locations, parent.top_row_indices},
                                                         gc_indices,
                                                         context,
                                                         std::move(*message_merge_buffers[ci]),
                                                         depth + 1,
                                                         materialize_output,
                                                         stream,
                                                         mr);
        if (materialize_output) { struct_children.push_back(std::move(child)); }
        continue;
      }

      nested_location_provider loc_provider{input.row_offsets,
                                            input.base_offset,
                                            parent.locations,
                                            d_child_locations.data(),
                                            ci,
                                            num_child_fields};
      rmm::device_uvector<field_location> d_gc_parent_locs(input.num_rows, stream, scratch_mr);
      launch_compute_grandchild_parent_locations(
        loc_provider, d_gc_parent_locs.data(), input.num_rows, decode_ctx.error->data(), stream);
      auto child = build_nested_struct_column(
        input,
        {d_gc_parent_locs.data(), d_gc_parent_locs.size(), parent.top_row_indices},
        gc_indices,
        context,
        depth + 1,
        materialize_output,
        stream,
        mr);
      if (materialize_output) { struct_children.push_back(std::move(child)); }
      continue;
    }

    if (!materialize_output) { continue; }

    nested_location_provider loc_provider{input.row_offsets,
                                          input.base_offset,
                                          parent.locations,
                                          d_child_locations.data(),
                                          ci,
                                          num_child_fields};

    auto valid_fn = [loc_provider, has_def] __device__(cudf::size_type row) {
      return has_def || loc_provider.valid(row);
    };
    auto const request =
      protobuf_field_decode_request{context, input.message_data, child_schema_idx, input.num_rows};
    auto child = build_protobuf_field_values_column(
      request, loc_provider, valid_fn, stream, materialize_output ? mr : scratch_mr);
    if (materialize_output) { struct_children.push_back(std::move(child)); }
  }

  if (!materialize_output) { return nullptr; }

  auto [struct_mask, struct_null_count] =
    make_null_mask_from_parent_locations(parent.locations, input.num_rows, stream, mr);
  return cudf::make_structs_column(input.num_rows,
                                   std::move(struct_children),
                                   struct_null_count,
                                   std::move(struct_mask),
                                   stream,
                                   mr);
}

std::unique_ptr<cudf::column> build_repeated_child_list_column(protobuf_input_view input,
                                                               nested_parent_view parent,
                                                               recursive_decode_context context,
                                                               repeated_field_work work,
                                                               bool materialize_output,
                                                               rmm::cuda_stream_view stream,
                                                               rmm::device_async_resource_ref mr)
{
  auto const& schema_context = context.schema;
  auto const& schema         = schema_context.fields();
  auto const decode_ctx      = context.runtime;
  validate_nested_parent_view(input, parent);
  validate_protobuf_decode_context(decode_ctx, input, parent);
  auto const child_schema_idx = work.schema_idx;
  CUDF_EXPECTS(child_schema_idx >= 0 && child_schema_idx < static_cast<int>(schema.size()),
               "Protobuf decode internal error: nested repeated schema index is out of bounds");
  CUDF_EXPECTS(schema[child_schema_idx].is_repeated,
               "nested repeated child builder requires a repeated child schema");
  auto const elem_type     = cudf::data_type{schema[child_schema_idx].output_type};
  auto const is_enum_field = !schema_context.field(child_schema_idx).enum_valid_values.empty();

  CUDF_EXPECTS(work.offsets.size() == static_cast<size_t>(input.num_rows) + 1,
               "Protobuf decode internal error: nested repeated offsets size mismatch");
  auto const scratch_mr  = cudf::get_current_device_resource_ref();
  auto const total_count = work.total_count;

  if (total_count == 0) {
    if (!materialize_output) { return nullptr; }
    auto offsets_col = make_offsets_column(input.num_rows, std::move(work.offsets));
    auto child_col =
      elem_type.id() == cudf::type_id::STRUCT
        ? make_empty_struct_column_with_schema(schema_context, child_schema_idx, stream, mr)
        : make_empty_column_safe(elem_type, stream, mr);
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
      d_top_row_indices = std::make_unique<rmm::device_uvector<int32_t>>(
        make_top_row_indices(d_occurrences, top_row_indices, stream, scratch_mr));
    }
    return d_top_row_indices->data();
  };

  std::unique_ptr<cudf::column> child_values;
  if (elem_type.id() == cudf::type_id::STRUCT) {
    rmm::device_uvector<cudf::size_type> d_virtual_row_offsets(total_count, stream, scratch_mr);
    rmm::device_uvector<field_location> d_virtual_parent_locs(total_count, stream, scratch_mr);
    launch_compute_virtual_parents_for_nested_repeated(input,
                                                       parent,
                                                       work,
                                                       d_virtual_row_offsets.data(),
                                                       d_virtual_parent_locs.data(),
                                                       decode_ctx,
                                                       stream);

    auto const& child_field_indices = schema_context.children(child_schema_idx);
    protobuf_input_view const virtual_input{input.message_data,
                                            input.message_data_size,
                                            d_virtual_row_offsets.data(),
                                            input.base_offset,
                                            total_count};
    child_values = build_nested_struct_column(
      virtual_input,
      {d_virtual_parent_locs.data(), d_virtual_parent_locs.size(), get_top_row_indices()},
      child_field_indices,
      context,
      work.depth,
      materialize_output,
      stream,
      mr);
  } else {
    if (!materialize_output) { return nullptr; }
    nested_repeated_location_provider loc_provider{
      input.row_offsets, input.base_offset, parent.locations, d_occurrences.data()};
    auto valid_fn = [] __device__(cudf::size_type) { return true; };
    auto const request =
      protobuf_field_decode_request{context, input.message_data, child_schema_idx, total_count};
    child_values = build_protobuf_field_values_column(
      request, loc_provider, valid_fn, stream, materialize_output ? mr : scratch_mr);
  }

  if (!materialize_output) { return nullptr; }

  auto offsets_col = make_offsets_column(input.num_rows, std::move(list_offsets));
  auto result      = make_list_column_with_parent_nulls(
    input.num_rows, std::move(offsets_col), std::move(child_values), parent.locations, stream, mr);
  if (is_enum_field) { return drop_unknown_repeated_enum_values(std::move(result), stream, mr); }
  return result;
}

std::unique_ptr<cudf::column> build_repeated_struct_column(
  cudf::column_view const& binary_input,
  protobuf_input_view input,
  std::vector<int> const& child_field_indices,
  recursive_decode_context context,
  repeated_field_work work,
  bool materialize_output,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const decode_ctx = context.runtime;
  validate_nonempty_repeated_field_work(work, input.num_rows);

  auto const scratch_mr = cudf::get_current_device_resource_ref();
  auto& occurrences     = *work.occurrences;
  rmm::device_uvector<field_location> d_message_locs(work.total_count, stream, scratch_mr);
  rmm::device_uvector<cudf::size_type> d_message_row_offsets(work.total_count, stream, scratch_mr);
  launch_compute_msg_locations_from_occurrences(
    input, work, d_message_locs.data(), d_message_row_offsets.data(), decode_ctx, stream);

  auto d_top_row_indices = make_top_row_indices(occurrences, nullptr, stream, scratch_mr);

  protobuf_input_view const message_input{
    input.message_data, input.message_data_size, d_message_row_offsets.data(), 0, work.total_count};
  auto struct_values = build_nested_struct_column(
    message_input,
    {d_message_locs.data(), d_message_locs.size(), d_top_row_indices.data()},
    child_field_indices,
    context,
    work.depth,
    materialize_output,
    stream,
    mr);

  if (!materialize_output) { return nullptr; }

  auto offsets_col = make_offsets_column(input.num_rows, std::move(work.offsets));
  return make_list_column_with_input_nulls(
    input.num_rows, std::move(offsets_col), std::move(struct_values), binary_input, stream, mr);
}

}  // namespace spark_rapids_jni::protobuf::detail
