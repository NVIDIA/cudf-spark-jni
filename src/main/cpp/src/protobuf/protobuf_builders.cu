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

field_descriptor_bundle make_field_descriptors(std::vector<int> const& field_indices,
                                               std::vector<nested_field_descriptor> const& schema,
                                               schema_context_view schema_ctx,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  size_t num_enum_values = 0;
  for (auto const schema_idx : field_indices) {
    num_enum_values += schema_ctx.enum_valid_values[schema_idx].size();
  }

  auto h_enum_values = cudf::detail::make_pinned_vector_async<int32_t>(num_enum_values, stream);
  size_t enum_offset = 0;
  for (auto const schema_idx : field_indices) {
    auto const& values = schema_ctx.enum_valid_values[schema_idx];
    std::copy(values.begin(), values.end(), h_enum_values.begin() + enum_offset);
    enum_offset += values.size();
  }
  auto d_enum_values = cudf::detail::make_device_uvector_async(h_enum_values, stream, mr);

  auto h_descriptors =
    cudf::detail::make_pinned_vector_async<field_descriptor>(field_indices.size(), stream);
  enum_offset = 0;
  for (size_t i = 0; i < field_indices.size(); ++i) {
    auto const schema_idx = field_indices[i];
    auto const& field     = schema[schema_idx];
    auto const enum_size  = schema_ctx.enum_valid_values[schema_idx].size();
    CUDF_EXPECTS(enum_size <= static_cast<size_t>(std::numeric_limits<int>::max()),
                 "protobuf enum metadata exceeds supported value count");
    h_descriptors[i] = {field.field_number,
                        static_cast<int>(field.wire_type),
                        field.is_repeated,
                        field.output_type == cudf::type_id::STRUCT,
                        enum_size > 0 ? d_enum_values.data() + enum_offset : nullptr,
                        static_cast<int>(enum_size)};
    enum_offset += enum_size;
  }

  auto d_descriptors = cudf::detail::make_device_uvector_async(h_descriptors, stream, mr);
  return {std::move(h_descriptors), std::move(d_descriptors), std::move(d_enum_values)};
}

namespace {

enum_string_lookup_tables const& get_enum_lookup(schema_context_view schema_ctx,
                                                 int schema_idx,
                                                 protobuf_field_meta_view field,
                                                 rmm::cuda_stream_view stream,
                                                 std::optional<enum_string_lookup_tables>& fallback)
{
  if (schema_ctx.enum_lookup_cache == nullptr) {
    fallback.emplace(
      make_enum_string_lookup_tables(field.enum_valid_values, field.enum_names, stream));
    return *fallback;
  }

  auto& cache = *schema_ctx.enum_lookup_cache;
  auto entry  = cache.find(schema_idx);
  if (entry == cache.end()) {
    entry =
      cache
        .emplace(schema_idx,
                 make_enum_string_lookup_tables(field.enum_valid_values, field.enum_names, stream))
        .first;
  }
  return entry->second;
}

std::unique_ptr<cudf::column> build_enum_string_column(rmm::device_uvector<int32_t>& enum_values,
                                                       rmm::device_uvector<bool>& valid,
                                                       enum_string_lookup_tables const& lookup,
                                                       protobuf_value_decode_context input,
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
  CUDF_EXPECTS(context.deferred_enum_error != nullptr,
               std::string{caller} + ": deferred enum error buffer must be non-null");
  CUDF_EXPECTS(
    context.deferred_enum_error->size() == 1,
    std::string{caller} + ": deferred enum error buffer must contain exactly one element");
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

std::unique_ptr<cudf::column> drop_unknown_repeated_enum_values(std::unique_ptr<cudf::column> input,
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

inline protobuf_field_meta_view make_field_meta_view(schema_context_view schema_ctx,
                                                     nested_field_descriptor const& field,
                                                     int schema_idx)
{
  return {field,
          cudf::data_type{field.output_type},
          schema_ctx.default_ints[schema_idx],
          schema_ctx.default_floats[schema_idx],
          schema_ctx.default_bools[schema_idx],
          schema_ctx.default_strings[schema_idx],
          schema_ctx.enum_valid_values[schema_idx],
          schema_ctx.enum_names[schema_idx]};
}

struct protobuf_field_build_view {
  protobuf_field_decode_view decode;
  schema_context_view schema_context;
  int schema_idx;
};

template <typename LocationProvider, typename ValidityFn, typename TopRowIndexProvider>
std::unique_ptr<cudf::column> build_protobuf_field_values_column(
  protobuf_field_build_view input,
  LocationProvider const& loc_provider,
  ValidityFn validity_fn,
  TopRowIndexProvider get_top_row_indices,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto field_decode       = input.decode;
  auto const message_data = field_decode.message_data;
  auto const field        = field_decode.field;
  auto const schema_ctx   = input.schema_context;
  auto const decode_ctx   = field_decode.values.runtime;
  auto const schema_idx   = input.schema_idx;
  auto const num_values   = field_decode.values.num_values;
  CUDF_EXPECTS(num_values > 0, std::string{__func__} + ": value count must be positive");
  auto const value_type  = field.output_type;
  auto const has_default = field.schema.has_default_value;
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
        value_type.id() == cudf::type_id::INT32 && !field.enum_valid_values.empty();
      bool const invalidates_top_row =
        is_numeric_enum && decode_ctx.invalidate_root_on_invalid_enum;
      field_decode.values.top_row_indices = invalidates_top_row ? get_top_row_indices() : nullptr;
      return extract_typed_column(field_decode, loc_provider, stream, mr);
    }
    case cudf::type_id::STRING:
    case cudf::type_id::LIST: {
      bool const is_enum_string = value_type.id() == cudf::type_id::STRING &&
                                  field.schema.encoding == proto_encoding::ENUM_STRING;
      if (is_enum_string) {
        auto const scratch_mr = cudf::get_current_device_resource_ref();
        rmm::device_uvector<int32_t> values(num_values, stream, scratch_mr);
        rmm::device_uvector<bool> valid(num_values, stream, scratch_mr);
        extract_varint_kernel<int32_t, false, LocationProvider>
          <<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
            message_data,
            loc_provider,
            num_values,
            {values.data(), valid.data(), decode_ctx.error->data()},
            {has_default, field.default_int});
        std::optional<enum_string_lookup_tables> fallback_lookup;
        auto const& lookup =
          get_enum_lookup(schema_ctx, schema_idx, field, stream, fallback_lookup);
        auto value_context = field_decode.values;
        value_context.top_row_indices =
          decode_ctx.invalidate_root_on_invalid_enum ? get_top_row_indices() : nullptr;
        return build_enum_string_column(values, valid, lookup, value_context, stream, mr);
      }
      return extract_and_build_string_or_bytes_column(
        field, message_data, num_values, loc_provider, validity_fn, stream, mr);
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

std::unique_ptr<cudf::column> build_enum_string_column(rmm::device_uvector<int32_t>& enum_values,
                                                       rmm::device_uvector<bool>& valid,
                                                       enum_string_lookup_tables const& lookup,
                                                       protobuf_value_decode_context input,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  validate_enum_and_apply_policy(enum_values,
                                 valid,
                                 lookup.d_valid_enums.data(),
                                 static_cast<int>(lookup.d_valid_enums.size()),
                                 input.runtime,
                                 input.num_values,
                                 input.top_row_indices,
                                 stream);
  return build_enum_string_values_column(enum_values, valid, lookup, input.num_values, stream, mr);
}

}  // namespace

std::unique_ptr<cudf::column> build_enum_string_column(rmm::device_uvector<int32_t>& enum_values,
                                                       rmm::device_uvector<bool>& valid,
                                                       enum_string_decode_view input,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  auto const lookup =
    make_enum_string_lookup_tables(input.valid_enums, input.enum_name_bytes, stream);
  return build_enum_string_column(enum_values, valid, lookup, input.values, stream, mr);
}

std::unique_ptr<cudf::column> build_repeated_enum_string_column(
  cudf::column_view const& binary_input,
  protobuf_input_view input,
  protobuf_field_meta_view field,
  protobuf_decode_runtime_context decode_ctx,
  repeated_field_work work,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  validate_nonempty_repeated_field_work(work, input.num_rows);

  auto const total_count = work.total_count;
  auto& occurrences      = *work.occurrences;
  auto const rep_blocks =
    static_cast<int>((total_count + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  auto const scratch_mr = cudf::get_current_device_resource_ref();
  auto const lookup =
    make_enum_string_lookup_tables(field.enum_valid_values, field.enum_names, stream);

  // 1. Extract enum integer values from occurrences
  rmm::device_uvector<int32_t> enum_ints(total_count, stream, scratch_mr);
  rmm::device_uvector<bool> elem_valid(total_count, stream, scratch_mr);
  repeated_location_provider rep_loc{input.row_offsets, input.base_offset, occurrences.data()};
  extract_varint_kernel<int32_t, false><<<rep_blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    input.message_data,
    rep_loc,
    total_count,
    {enum_ints.data(), elem_valid.data(), decode_ctx.error->data()},
    {false, int64_t{0}});

  auto d_top_row_indices = materialize_top_row_indices(occurrences, nullptr, stream, scratch_mr);
  validate_enum_and_apply_policy(enum_ints,
                                 elem_valid,
                                 lookup.d_valid_enums.data(),
                                 static_cast<int>(lookup.d_valid_enums.size()),
                                 decode_ctx,
                                 total_count,
                                 d_top_row_indices.data(),
                                 stream);

  auto child_col =
    build_enum_string_values_column(enum_ints, elem_valid, lookup, total_count, stream, mr);

  auto list_offs_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                      input.num_rows + 1,
                                                      work.offsets.release(),
                                                      rmm::device_buffer{},
                                                      0);

  return make_list_column_with_input_nulls(
    input.num_rows, std::move(list_offs_col), std::move(child_col), binary_input, stream, mr);
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
  extract_lengths_kernel<repeated_location_provider>
    <<<blocks, threads, 0, stream.value()>>>(loc_provider, total_count, str_lengths.data());

  auto [str_offsets_col, total_chars] = cudf::strings::detail::make_offsets_child_column(
    str_lengths.begin(), str_lengths.end(), stream, mr);

  rmm::device_uvector<char> chars(total_chars, stream, mr);
  if (total_chars > 0) {
    repeated_location_provider copy_provider{
      input.row_offsets, input.base_offset, occurrences.data()};
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
                                                    work.offsets.release(),
                                                    rmm::device_buffer{},
                                                    0);

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
  singular_message_merge_work work,
  int depth,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  validate_singular_message_merge_work(work, input.num_rows);
  auto const scratch_mr = cudf::get_current_device_resource_ref();

  auto validation_fields = make_field_descriptors(
    child_field_indices, context.schema, context.schema_metadata, stream, scratch_mr);
  auto h_field_lookup = build_field_lookup_table(
    validation_fields.host.data(), static_cast<int>(validation_fields.host.size()), stream);
  auto d_field_lookup = cudf::detail::make_device_uvector_async(h_field_lookup, stream, scratch_mr);

  auto invalid_rows =
    cudf::detail::make_zeroed_device_uvector_async<bool>(input.num_rows, stream, scratch_mr);
  message_fragment_location_provider fragment_locations{input, source, work.fragments.data()};
  launch_validate_message_fragments(
    fragment_locations,
    {validation_fields.device.data(),
     static_cast<int>(validation_fields.device.size()),
     d_field_lookup.is_empty() ? nullptr : d_field_lookup.data(),
     static_cast<int>(d_field_lookup.size())},
    work.total_fragments,
    invalid_rows.data(),
    context.runtime.row_force_null->is_empty() ? nullptr : context.runtime.row_force_null->data(),
    context.runtime.error->data(),
    depth + 1,
    stream);

  auto fragment_lengths = thrust::make_transform_iterator(
    work.fragments.begin(),
    [] __device__(field_occurrence const& fragment) -> int32_t { return fragment.length; });
  auto fragment_byte_offsets = make_list_offsets_from_counts(fragment_lengths,
                                                             work.total_fragments,
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
                    [row_fragment_offsets = work.row_offsets.data(),
                     fragment_offsets = fragment_byte_offsets.offsets.data()] __device__(int row) {
                      return fragment_offsets[row_fragment_offsets[row]];
                    });

  rmm::device_uvector<uint8_t> merged_data(std::max<int32_t>(total_bytes, 1), stream, scratch_mr);
  if (total_bytes > 0) {
    auto const* invalid          = invalid_rows.data();
    auto const* fragments        = work.fragments.data();
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
    cub::DeviceMemcpy::Batched(nullptr,
                               temp_storage_bytes,
                               src_iter,
                               dst_iter,
                               size_iter,
                               work.total_fragments,
                               stream.value());
    rmm::device_buffer temp_storage(temp_storage_bytes, stream, scratch_mr);
    cub::DeviceMemcpy::Batched(temp_storage.data(),
                               temp_storage_bytes,
                               src_iter,
                               dst_iter,
                               size_iter,
                               work.total_fragments,
                               stream.value());
  }

  rmm::device_uvector<field_location> merged_parent_locations(input.num_rows, stream, scratch_mr);
  thrust::transform(
    rmm::exec_policy_nosync(stream, scratch_mr),
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(input.num_rows),
    merged_parent_locations.begin(),
    [row_fragment_offsets = work.row_offsets.data(),
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
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const& schema    = context.schema;
  auto const schema_ctx = context.schema_metadata;
  auto const decode_ctx = context.runtime;
  CUDF_EXPECTS(depth < MAX_NESTING_DEPTH,
               "Nested protobuf struct depth exceeds supported decode recursion limit");
  validate_nested_parent_view(input, parent);
  validate_protobuf_decode_context(decode_ctx, input, parent);

  if (input.num_rows == 0) {
    return make_empty_struct_column_from_children(schema, child_field_indices, stream, mr);
  }

  int num_child_fields = static_cast<int>(child_field_indices.size());
  std::vector<int> repeated_child_positions;
  std::vector<int> singular_message_positions;
  repeated_child_positions.reserve(num_child_fields);
  singular_message_positions.reserve(num_child_fields);

  for (int i = 0; i < num_child_fields; i++) {
    int child_idx = child_field_indices[i];
    if (schema[child_idx].is_repeated) {
      repeated_child_positions.push_back(i);
    } else if (schema[child_idx].output_type == cudf::type_id::STRUCT) {
      singular_message_positions.push_back(i);
    }
  }

  auto const scratch_mr = cudf::get_current_device_resource_ref();
  auto child_field_descs =
    make_field_descriptors(child_field_indices, schema, schema_ctx, stream, scratch_mr);
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
  // Occurrence counts are collected with singleton locations so duplicate-message and LIST
  // offsets do not require another count pass.
  launch_scan_nested_message_fields(
    input,
    parent,
    {d_child_field_descs.data(),
     num_child_fields,
     nullptr,
     0,
     d_child_locations.data(),
     d_occurrence_info.data(),
     nullptr,
     d_multiple_message_fields.data()},
    decode_ctx.error->data(),
    !decode_ctx.row_force_null->is_empty() ? decode_ctx.row_force_null->data() : nullptr,
    depth + 1,
    stream);

  maybe_check_required_fields({d_child_locations.data(),
                               input.num_rows,
                               nullptr,
                               0,
                               parent.locations,
                               parent.top_row_indices},
                              child_field_indices,
                              schema,
                              decode_ctx,
                              stream);

  std::vector<std::optional<singular_message_merge_work>> message_merge_work(num_child_fields);
  if (!singular_message_positions.empty()) {
    auto h_multiple_message_fields =
      cudf::detail::make_pinned_vector_async<int>(num_child_fields, stream);
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(h_multiple_message_fields.data(),
                                             d_multiple_message_fields.data(),
                                             num_child_fields * sizeof(int),
                                             stream));
    stream.synchronize();

    for (auto const ci : singular_message_positions) {
      if (h_multiple_message_fields[ci] == 0) { continue; }

      auto const child_schema_idx = child_field_indices[ci];
      auto counts_begin           = thrust::make_transform_iterator(
        thrust::make_counting_iterator<int>(0),
        extract_strided_count{d_occurrence_info.data(), ci, num_child_fields});
      message_merge_work[ci].emplace(
        child_schema_idx,
        make_list_offsets_from_counts(
          counts_begin, input.num_rows, "Nested singular message", stream, scratch_mr, scratch_mr),
        stream,
        scratch_mr);
      auto& work = *message_merge_work[ci];

      auto h_message_scan_desc =
        cudf::detail::make_pinned_vector_async<field_occurrence_scan_desc>(1, stream);
      h_message_scan_desc[0] = {schema[child_schema_idx].field_number,
                                wire_type_value(proto_wire_type::LEN),
                                work.row_offsets.data(),
                                work.fragments.data()};
      auto scan_bundle = make_field_occurrence_scan_bundle(h_message_scan_desc, stream, scratch_mr);
      launch_scan_all_field_occurrences_in_nested(
        input, parent, scan_bundle.view(), decode_ctx.error->data(), depth + 1, stream);
    }
  }

  std::vector<std::optional<repeated_field_work>> repeated_work(num_child_fields);
  auto h_scan_descs = cudf::detail::make_pinned_vector_async<field_occurrence_scan_desc>(0, stream);
  h_scan_descs.reserve(repeated_child_positions.size());
  for (auto const ci : repeated_child_positions) {
    // The row-major buffer has `num_child_fields` entries per row, so this strided iterator
    // yields exactly one count for each input row.
    auto const child_schema_idx = child_field_indices[ci];
    auto counts_begin           = thrust::make_transform_iterator(
      thrust::make_counting_iterator<int>(0),
      extract_strided_count{d_occurrence_info.data(), ci, num_child_fields});
    repeated_work[ci].emplace(
      child_schema_idx,
      make_list_offsets_from_counts(
        counts_begin, input.num_rows, "Repeated nested-field", stream, mr, scratch_mr),
      depth + 1);

    auto& work = *repeated_work[ci];
    if (work.total_count > 0) {
      work.occurrences = std::make_unique<rmm::device_uvector<field_occurrence>>(
        work.total_count, stream, scratch_mr);
    }
    // Zero-count descriptors keep malformed rows aligned with the count pass.
    h_scan_descs.push_back({schema[child_schema_idx].field_number,
                            static_cast<int>(schema[child_schema_idx].wire_type),
                            work.offsets.data(),
                            work.occurrences == nullptr ? nullptr : work.occurrences->data()});
  }

  if (!h_scan_descs.empty()) {
    auto scan_bundle = make_field_occurrence_scan_bundle(h_scan_descs, stream, scratch_mr);
    launch_scan_all_field_occurrences_in_nested(
      input, parent, scan_bundle.view(), decode_ctx.error->data(), depth + 1, stream);
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
      struct_children.push_back(build_repeated_child_list_column(
        input, parent, context, std::move(repeated_work[ci].value()), stream, mr));
      continue;
    }

    if (dt.id() == cudf::type_id::STRUCT) {
      // Recursive linear child lookup is fine for realistic schemas; precompute if it gets hot.
      auto gc_indices = find_child_field_indices(schema, child_schema_idx);
      if (message_merge_work[ci].has_value()) {
        struct_children.push_back(
          build_merged_singular_struct_column(input,
                                              {parent.locations, parent.top_row_indices},
                                              gc_indices,
                                              context,
                                              std::move(*message_merge_work[ci]),
                                              depth + 1,
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
      rmm::device_uvector<field_location> d_gc_parent_locs(input.num_rows, stream, scratch_mr);
      launch_compute_grandchild_parent_locations(
        loc_provider, d_gc_parent_locs.data(), input.num_rows, decode_ctx.error->data(), stream);
      struct_children.push_back(build_nested_struct_column(
        input,
        {d_gc_parent_locs.data(), d_gc_parent_locs.size(), parent.top_row_indices},
        gc_indices,
        context,
        depth + 1,
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

    auto valid_fn = [loc_provider, has_def] __device__(cudf::size_type row) {
      return has_def || loc_provider.valid(row);
    };
    auto get_top_row_indices = [top_row_indices = parent.top_row_indices]() {
      return top_row_indices;
    };
    struct_children.push_back(build_protobuf_field_values_column(
      {{input.message_data,
        make_field_meta_view(schema_ctx, schema[child_schema_idx], child_schema_idx),
        {decode_ctx, input.num_rows}},
       schema_ctx,
       child_schema_idx},
      loc_provider,
      valid_fn,
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

std::unique_ptr<cudf::column> build_repeated_child_list_column(protobuf_input_view input,
                                                               nested_parent_view parent,
                                                               recursive_decode_context context,
                                                               repeated_field_work work,
                                                               rmm::cuda_stream_view stream,
                                                               rmm::device_async_resource_ref mr)
{
  auto const& schema    = context.schema;
  auto const schema_ctx = context.schema_metadata;
  auto const decode_ctx = context.runtime;
  validate_nested_parent_view(input, parent);
  validate_protobuf_decode_context(decode_ctx, input, parent);
  auto const child_schema_idx = work.schema_idx;
  CUDF_EXPECTS(child_schema_idx >= 0 && child_schema_idx < static_cast<int>(schema.size()),
               "Protobuf decode internal error: nested repeated schema index is out of bounds");
  CUDF_EXPECTS(schema[child_schema_idx].is_repeated,
               "nested repeated child builder requires a repeated child schema");
  auto const elem_type     = cudf::data_type{schema[child_schema_idx].output_type};
  auto const is_enum_field = !schema_ctx.enum_valid_values[child_schema_idx].empty();

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
    auto child_col   = elem_type.id() == cudf::type_id::STRUCT
                         ? make_empty_struct_column_with_schema(schema, child_schema_idx, stream, mr)
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
        materialize_top_row_indices(d_occurrences, top_row_indices, stream, scratch_mr));
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

    auto child_field_indices = find_child_field_indices(schema, child_schema_idx);
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
      stream,
      mr);
  } else {
    nested_repeated_location_provider loc_provider{
      input.row_offsets, input.base_offset, parent.locations, d_occurrences.data()};
    auto valid_fn = [] __device__(cudf::size_type) { return true; };
    child_values  = build_protobuf_field_values_column(
      {{input.message_data,
         make_field_meta_view(schema_ctx, schema[child_schema_idx], child_schema_idx),
         {decode_ctx, total_count}},
        schema_ctx,
        child_schema_idx},
      loc_provider,
      valid_fn,
      get_top_row_indices,
      stream,
      mr);
  }

  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    input.num_rows + 1,
                                                    list_offsets.release(),
                                                    rmm::device_buffer{},
                                                    0);
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
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const& schema    = context.schema;
  auto const schema_ctx = context.schema_metadata;
  auto const decode_ctx = context.runtime;
  validate_nonempty_repeated_field_work(work, input.num_rows);

  auto const scratch_mr = cudf::get_current_device_resource_ref();
  auto& occurrences     = *work.occurrences;
  rmm::device_uvector<field_location> d_message_locs(work.total_count, stream, scratch_mr);
  rmm::device_uvector<cudf::size_type> d_message_row_offsets(work.total_count, stream, scratch_mr);
  launch_compute_msg_locations_from_occurrences(
    input, work, d_message_locs.data(), d_message_row_offsets.data(), decode_ctx, stream);

  auto d_top_row_indices = materialize_top_row_indices(occurrences, nullptr, stream, scratch_mr);

  protobuf_input_view const message_input{
    input.message_data, input.message_data_size, d_message_row_offsets.data(), 0, work.total_count};
  auto child_decode_ctx                            = decode_ctx;
  child_decode_ctx.invalidate_root_on_invalid_enum = false;
  auto struct_values                               = build_nested_struct_column(
    message_input,
    {d_message_locs.data(), d_message_locs.size(), d_top_row_indices.data()},
    child_field_indices,
    {schema, schema_ctx, child_decode_ctx},
    work.depth,
    stream,
    mr);

  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    input.num_rows + 1,
                                                    work.offsets.release(),
                                                    rmm::device_buffer{},
                                                    0);
  return make_list_column_with_input_nulls(
    input.num_rows, std::move(offsets_col), std::move(struct_values), binary_input, stream, mr);
}

}  // namespace spark_rapids_jni::protobuf::detail
