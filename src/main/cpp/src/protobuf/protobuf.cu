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

#include "nvtx_ranges.hpp"
#include "protobuf/protobuf_kernels.cuh"

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace spark_rapids_jni::protobuf {

namespace detail {

namespace {

void propagate_nulls_to_descendants(cudf::column& col,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

void apply_parent_mask_to_row_aligned_column(cudf::column& col,
                                             cudf::bitmask_type const* parent_mask_ptr,
                                             cudf::size_type parent_null_count,
                                             cudf::size_type num_rows,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  if (parent_null_count == 0) { return; }
  auto child_view = col.mutable_view();
  CUDF_EXPECTS(child_view.size() == num_rows,
               "struct child size must match parent row count for null propagation");

  if (child_view.nullable()) {
    auto const child_mask_words =
      cudf::num_bitmask_words(static_cast<size_t>(child_view.size() + child_view.offset()));
    std::array<cudf::bitmask_type const*, 2> masks{child_view.null_mask(), parent_mask_ptr};
    std::array<cudf::size_type, 2> begin_bits{child_view.offset(), 0};
    auto const valid_count = cudf::detail::inplace_bitmask_and(
      cudf::device_span<cudf::bitmask_type>(child_view.null_mask(), child_mask_words),
      cudf::host_span<cudf::bitmask_type const* const>(masks.data(), masks.size()),
      cudf::host_span<cudf::size_type const>(begin_bits.data(), begin_bits.size()),
      child_view.size(),
      stream);
    col.set_null_count(child_view.size() - valid_count);
  } else {
    CUDF_EXPECTS(child_view.offset() == 0,
                 "non-nullable child with nonzero offset not supported for null propagation");
    auto child_mask = cudf::detail::copy_bitmask(parent_mask_ptr, 0, num_rows, stream, mr);
    col.set_null_mask(std::move(child_mask), parent_null_count);
  }
}

void propagate_list_nulls_to_descendants(cudf::column& list_col,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  if (list_col.type().id() != cudf::type_id::LIST || list_col.null_count() == 0) { return; }

  cudf::lists_column_view const list_view(list_col.view());
  auto const* list_mask_ptr = list_view.null_mask();
  auto const num_rows       = list_view.size();
  auto& child               = list_col.child(cudf::lists_column_view::child_column_index);
  auto const child_size     = child.size();
  if (child_size == 0) { return; }

  CUDF_EXPECTS(list_view.offset() == 0,
               "decoder list null propagation expects unsliced list columns");
  auto const* offsets_begin = list_view.offsets_begin();
  auto const* offsets_end   = list_view.offsets_end();
  // LIST children are not row-aligned with their parent. Expand the list-row null mask across
  // every covered child element so direct access to the backing child column also observes nulls.
  auto [element_mask, element_null_count] = cudf::detail::valid_if(
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(child_size),
    [list_mask_ptr, offsets_begin, offsets_end] __device__(cudf::size_type idx) {
      auto const it  = thrust::upper_bound(thrust::seq, offsets_begin, offsets_end, idx);
      auto const row = static_cast<cudf::size_type>(it - offsets_begin) - 1;
      return list_mask_ptr == nullptr || cudf::bit_is_set(list_mask_ptr, row);
    },
    stream,
    mr);

  apply_parent_mask_to_row_aligned_column(
    child,
    static_cast<cudf::bitmask_type const*>(element_mask.data()),
    element_null_count,
    child_size,
    stream,
    mr);
  propagate_nulls_to_descendants(child, stream, mr);
}

void propagate_struct_nulls_to_descendants(cudf::column& struct_col,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  if (struct_col.type().id() != cudf::type_id::STRUCT || struct_col.null_count() == 0) { return; }

  auto const struct_view      = struct_col.view();
  auto const* struct_mask_ptr = struct_view.null_mask();
  auto const num_rows         = struct_view.size();
  auto const null_count       = struct_col.null_count();

  for (cudf::size_type i = 0; i < struct_col.num_children(); ++i) {
    auto& child = struct_col.child(i);
    apply_parent_mask_to_row_aligned_column(
      child, struct_mask_ptr, null_count, num_rows, stream, mr);
    propagate_nulls_to_descendants(child, stream, mr);
  }
}

void propagate_nulls_to_descendants(cudf::column& col,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  switch (col.type().id()) {
    case cudf::type_id::STRUCT: propagate_struct_nulls_to_descendants(col, stream, mr); break;
    case cudf::type_id::LIST: propagate_list_nulls_to_descendants(col, stream, mr); break;
    default: break;
  }
}

}  // namespace

std::unique_ptr<cudf::column> make_null_column_with_schema(protobuf_schema const& schema,
                                                           int schema_idx,
                                                           cudf::size_type num_rows,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr)
{
  auto const& field = schema[schema_idx];
  auto const dtype  = cudf::data_type{field.output_type};

  if (field.is_repeated) {
    std::unique_ptr<cudf::column> empty_child;
    if (dtype.id() == cudf::type_id::STRUCT) {
      empty_child = make_empty_struct_column_with_schema(schema, schema_idx, stream, mr);
    } else {
      empty_child = make_empty_column_safe(dtype, stream, mr);
    }
    return make_null_list_column_with_child(std::move(empty_child), num_rows, stream, mr);
  }

  if (dtype.id() == cudf::type_id::STRUCT) {
    auto const& child_indices = schema.children(schema_idx);
    std::vector<std::unique_ptr<cudf::column>> children;
    for (auto const child_idx : child_indices) {
      children.push_back(make_null_column_with_schema(schema, child_idx, num_rows, stream, mr));
    }
    auto null_mask = cudf::create_null_mask(num_rows, cudf::mask_state::ALL_NULL, stream, mr);
    return cudf::make_structs_column(
      num_rows, std::move(children), num_rows, std::move(null_mask), stream, mr);
  }

  return make_null_column(dtype, num_rows, stream, mr);
}

bool is_encoding_compatible(nested_field_descriptor const& field, cudf::data_type const& type)
{
  switch (field.encoding) {
    case proto_encoding::DEFAULT:
      switch (type.id()) {
        case cudf::type_id::BOOL8:
        case cudf::type_id::INT32:
        case cudf::type_id::UINT32:
        case cudf::type_id::INT64:
        case cudf::type_id::UINT64: return field.wire_type == proto_wire_type::VARINT;
        case cudf::type_id::FLOAT32: return field.wire_type == proto_wire_type::I32BIT;
        case cudf::type_id::FLOAT64: return field.wire_type == proto_wire_type::I64BIT;
        case cudf::type_id::STRING:
        case cudf::type_id::LIST:
        case cudf::type_id::STRUCT: return field.wire_type == proto_wire_type::LEN;
        default: return false;
      }
    case proto_encoding::FIXED:
      switch (type.id()) {
        case cudf::type_id::INT32:
        case cudf::type_id::UINT32:
        case cudf::type_id::FLOAT32: return field.wire_type == proto_wire_type::I32BIT;
        case cudf::type_id::INT64:
        case cudf::type_id::UINT64:
        case cudf::type_id::FLOAT64: return field.wire_type == proto_wire_type::I64BIT;
        default: return false;
      }
    case proto_encoding::ZIGZAG:
      return field.wire_type == proto_wire_type::VARINT &&
             (type.id() == cudf::type_id::INT32 || type.id() == cudf::type_id::INT64);
    case proto_encoding::ENUM_STRING:
      return field.wire_type == proto_wire_type::VARINT && type.id() == cudf::type_id::STRING;
    default: return false;
  }
}

void validate_decode_context(protobuf_decode_context const& context)
{
  auto const num_fields = context.schema.size();
  CUDF_EXPECTS(context.default_ints.size() == num_fields,
               "protobuf decode context: default_ints size mismatch",
               std::invalid_argument);
  CUDF_EXPECTS(context.default_floats.size() == num_fields,
               "protobuf decode context: default_floats size mismatch",
               std::invalid_argument);
  CUDF_EXPECTS(context.default_bools.size() == num_fields,
               "protobuf decode context: default_bools size mismatch",
               std::invalid_argument);
  CUDF_EXPECTS(context.default_strings.size() == num_fields,
               "protobuf decode context: default_strings size mismatch",
               std::invalid_argument);
  CUDF_EXPECTS(context.enum_valid_values.size() == num_fields,
               "protobuf decode context: enum_valid_values size mismatch",
               std::invalid_argument);
  CUDF_EXPECTS(context.enum_names.size() == num_fields,
               "protobuf decode context: enum_names size mismatch",
               std::invalid_argument);
  CUDF_EXPECTS(context.output_fields.empty() || context.output_fields.size() == num_fields,
               "protobuf decode context: output_fields size mismatch",
               std::invalid_argument);

  std::set<std::pair<int, int>> seen_field_numbers;
  for (size_t i = 0; i < num_fields; ++i) {
    auto const& field = context.schema[i];
    auto const type   = cudf::data_type{field.output_type};
    CUDF_EXPECTS(field.field_number > 0 && field.field_number <= MAX_FIELD_NUMBER,
                 "protobuf decode context: invalid field number at field " + std::to_string(i),
                 std::invalid_argument);
    CUDF_EXPECTS(field.depth >= 0 && field.depth < MAX_NESTING_DEPTH,
                 "protobuf decode context: field depth exceeds limit at field " + std::to_string(i),
                 std::invalid_argument);
    CUDF_EXPECTS(field.parent_idx >= -1 && field.parent_idx < static_cast<int>(i),
                 "protobuf decode context: invalid parent index at field " + std::to_string(i),
                 std::invalid_argument);
    CUDF_EXPECTS(seen_field_numbers.emplace(field.parent_idx, field.field_number).second,
                 "protobuf decode context: duplicate field number under same parent at field " +
                   std::to_string(i),
                 std::invalid_argument);

    if (field.parent_idx == -1) {
      CUDF_EXPECTS(
        field.depth == 0,
        "protobuf decode context: top-level field must have depth 0 at field " + std::to_string(i),
        std::invalid_argument);
    } else {
      auto const& parent = context.schema[field.parent_idx];
      CUDF_EXPECTS(field.depth == parent.depth + 1,
                   "protobuf decode context: child depth mismatch at field " + std::to_string(i),
                   std::invalid_argument);
      CUDF_EXPECTS(context.schema[field.parent_idx].output_type == cudf::type_id::STRUCT,
                   "protobuf decode context: parent must be STRUCT at field " + std::to_string(i),
                   std::invalid_argument);
      if (!context.output_fields.empty()) {
        // A field and its parent must share the same output flag: a hidden STRUCT cannot have
        // visible descendants (the parent would have to be materialized anyway), and a visible
        // STRUCT cannot have hidden children. Forbid the mismatch up front.
        CUDF_EXPECTS(
          context.output_fields[i] == context.output_fields[field.parent_idx],
          "protobuf decode context: child output flag mismatch at field " + std::to_string(i),
          std::invalid_argument);
      }
    }

    CUDF_EXPECTS(
      field.wire_type == proto_wire_type::VARINT || field.wire_type == proto_wire_type::I64BIT ||
        field.wire_type == proto_wire_type::LEN || field.wire_type == proto_wire_type::I32BIT,
      "protobuf decode context: invalid wire type at field " + std::to_string(i),
      std::invalid_argument);
    CUDF_EXPECTS(
      field.encoding >= proto_encoding::DEFAULT && field.encoding <= proto_encoding::ENUM_STRING,
      "protobuf decode context: invalid encoding at field " + std::to_string(i),
      std::invalid_argument);
    CUDF_EXPECTS(!(field.is_repeated && field.is_required),
                 "protobuf decode context: field cannot be both repeated and required at field " +
                   std::to_string(i),
                 std::invalid_argument);
    CUDF_EXPECTS(!(field.is_repeated && field.has_default_value),
                 "protobuf decode context: repeated field cannot carry default value at field " +
                   std::to_string(i),
                 std::invalid_argument);
    CUDF_EXPECTS(!(field.has_default_value &&
                   (type.id() == cudf::type_id::STRUCT || type.id() == cudf::type_id::LIST)),
                 "protobuf decode context: STRUCT/LIST field cannot carry default value at field " +
                   std::to_string(i),
                 std::invalid_argument);
    CUDF_EXPECTS(is_encoding_compatible(field, type),
                 "protobuf decode context: incompatible wire type/encoding/output type at field " +
                   std::to_string(i),
                 std::invalid_argument);

    auto const has_enum_metadata = !context.enum_valid_values[i].empty();
    auto const is_numeric_enum =
      type.id() == cudf::type_id::INT32 && field.encoding == proto_encoding::DEFAULT;
    auto const is_string_enum =
      type.id() == cudf::type_id::STRING && field.encoding == proto_encoding::ENUM_STRING;
    CUDF_EXPECTS(!has_enum_metadata || is_numeric_enum || is_string_enum,
                 "protobuf decode context: enum metadata requires INT32/DEFAULT or "
                 "STRING/ENUM_STRING at field " +
                   std::to_string(i) + " (output_type=" + cudf::type_to_name(type) +
                   ", encoding=" + std::to_string(static_cast<int>(field.encoding)) + ")",
                 std::invalid_argument);

    auto const& enum_values_for_field = context.enum_valid_values[i];
    if (!enum_values_for_field.empty()) {
      for (size_t j = 1; j < enum_values_for_field.size(); ++j) {
        CUDF_EXPECTS(
          enum_values_for_field[j] > enum_values_for_field[j - 1],
          "protobuf decode context: enum_valid_values must be strictly sorted at field " +
            std::to_string(i),
          std::invalid_argument);
      }
      if (field.has_default_value) {
        auto const default_value = context.default_ints[i];
        CUDF_EXPECTS(default_value >= std::numeric_limits<int32_t>::min() &&
                       default_value <= std::numeric_limits<int32_t>::max() &&
                       std::binary_search(enum_values_for_field.begin(),
                                          enum_values_for_field.end(),
                                          static_cast<int32_t>(default_value)),
                     "protobuf decode context: enum default must be present in enum_valid_values "
                     "at field " +
                       std::to_string(i),
                     std::invalid_argument);
      }
    }

    if (field.encoding == proto_encoding::ENUM_STRING) {
      CUDF_EXPECTS(
        !(enum_values_for_field.empty() || context.enum_names[i].empty()),
        "protobuf decode context: enum-as-string field requires non-empty metadata at field " +
          std::to_string(i),
        std::invalid_argument);
      CUDF_EXPECTS(
        enum_values_for_field.size() == context.enum_names[i].size(),
        "protobuf decode context: enum-as-string metadata mismatch at field " + std::to_string(i),
        std::invalid_argument);
    }
  }

  // Reject schemas that exceed the combined-scan kernel's per-message stack-array capacity.
  // Counting by parent keeps the error schema-deterministic instead of depending on which fields
  // happen to carry data in a particular batch.
  std::unordered_map<int, int> repeated_fields_by_parent;
  for (auto const& field : context.schema) {
    if (!field.is_repeated) { continue; }
    auto& count = repeated_fields_by_parent[field.parent_idx];
    CUDF_EXPECTS(++count <= MAX_REPEATED_FIELDS_PER_KERNEL,
                 error_message(protobuf_error::SCHEMA_TOO_LARGE),
                 std::invalid_argument);
  }
}

protobuf_schema::protobuf_schema(protobuf_decode_context const& context) : context_(context)
{
  validate_decode_context(context);
  children_by_parent_.resize(context.schema.size() + 1);
  std::vector<size_t> child_counts(children_by_parent_.size());
  for (auto const& field : context.schema) {
    ++child_counts[field.parent_idx + 1];
  }
  for (size_t parent_idx = 0; parent_idx < children_by_parent_.size(); ++parent_idx) {
    children_by_parent_[parent_idx].reserve(child_counts[parent_idx]);
  }
  for (int schema_idx = 0; schema_idx < static_cast<int>(context.schema.size()); ++schema_idx) {
    children_by_parent_[context.schema[schema_idx].parent_idx + 1].push_back(schema_idx);
  }
}

protobuf_field_meta_view protobuf_schema::field(int schema_idx) const
{
  auto const idx = static_cast<size_t>(schema_idx);
  return {context_.schema.at(idx),
          cudf::data_type{context_.schema.at(idx).output_type},
          context_.default_ints.at(idx),
          context_.default_floats.at(idx),
          context_.default_bools.at(idx),
          context_.default_strings.at(idx),
          context_.enum_valid_values.at(idx),
          context_.enum_names.at(idx)};
}

std::vector<int> const& protobuf_schema::children(int parent_schema_idx) const
{
  CUDF_EXPECTS(parent_schema_idx >= -1 && parent_schema_idx < static_cast<int>(size()),
               "protobuf schema parent index is out of bounds");
  return children_by_parent_[parent_schema_idx + 1];
}

bool protobuf_schema::is_output(int schema_idx) const
{
  auto const idx = static_cast<size_t>(schema_idx);
  return context_.output_fields.empty() || context_.output_fields.at(idx);
}

std::unique_ptr<cudf::column> decode_protobuf_to_struct(cudf::column_view const& binary_input,
                                                        protobuf_decode_context const& context,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  protobuf_schema schema_context{context};
  auto const& schema  = schema_context.fields();
  bool fail_on_errors = context.fail_on_errors;
  CUDF_EXPECTS(binary_input.type().id() == cudf::type_id::LIST,
               "binary_input must be a LIST<INT8/UINT8> column");
  cudf::lists_column_view const in_list(binary_input);
  auto const child_type = in_list.child().type().id();
  CUDF_EXPECTS(child_type == cudf::type_id::INT8 || child_type == cudf::type_id::UINT8,
               "binary_input must be a LIST<INT8/UINT8> column");

  auto const num_rows   = binary_input.size();
  auto const num_fields = static_cast<int>(schema.size());

  if (num_rows == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty_children;
    for (int i = 0; i < num_fields; i++) {
      if (schema[i].parent_idx != -1 || !schema_context.is_output(i)) { continue; }
      auto field_type  = cudf::data_type{schema[i].output_type};
      auto empty_child = (field_type.id() == cudf::type_id::STRUCT)
                           ? make_empty_struct_column_with_schema(schema_context, i, stream, mr)
                           : make_empty_column_safe(field_type, stream, mr);
      if (schema[i].is_repeated) {
        empty_child = make_empty_list_column(std::move(empty_child), stream, mr);
      }
      empty_children.push_back(std::move(empty_child));
    }
    return cudf::make_structs_column(
      0, std::move(empty_children), 0, rmm::device_buffer{}, stream, mr);
  }

  // Extract shared input data pointers (used by scalar, repeated, and nested sections)
  cudf::lists_column_view const in_list_view(binary_input);
  auto const message_bytes     = in_list_view.get_sliced_child(stream);
  auto const* message_data     = message_bytes.data<uint8_t>();
  auto const message_data_size = message_bytes.size();
  auto const* list_offsets     = in_list_view.offsets_begin();
  auto const base_offset       = message_bytes.offset() - in_list_view.child().offset();
  auto const input =
    protobuf_input_view{message_data, message_data_size, list_offsets, base_offset, num_rows};
  // Scratch allocations consumed inside this function go through the current device resource;
  // only buffers that flow into the returned column should use the caller-supplied `mr`.
  auto const scratch_mr = cudf::get_current_device_resource_ref();

  auto d_in = cudf::column_device_view::create(binary_input, stream);
  // Identify repeated and nested fields at depth 0
  std::vector<int> repeated_field_indices;
  std::vector<int> nested_field_indices;
  std::vector<int> scalar_field_indices;

  for (int i = 0; i < num_fields; i++) {
    if (schema[i].parent_idx == -1) {  // Top-level fields only
      if (schema[i].is_repeated) {
        repeated_field_indices.push_back(i);
      } else if (schema[i].output_type == cudf::type_id::STRUCT) {
        nested_field_indices.push_back(i);
      } else {
        scalar_field_indices.push_back(i);
      }
    }
  }

  int const num_repeated    = static_cast<int>(repeated_field_indices.size());
  int const num_nested      = static_cast<int>(nested_field_indices.size());
  int const num_scalar      = static_cast<int>(scalar_field_indices.size());
  bool const run_count_scan = num_repeated > 0 || num_nested > 0;
  // Validate empty schemas through the field scan without rescanning repeated-only schemas.
  bool const run_field_scan = num_scalar > 0 || !run_count_scan;

  auto d_error = cudf::detail::make_zeroed_device_uvector_async<protobuf_error>(
    1, stream, cudf::get_current_device_resource_ref());
  // Proto2 required-field validation precedes root unknown-enum reporting.
  auto d_deferred_enum_error = cudf::detail::make_zeroed_device_uvector_async<protobuf_error>(
    1, stream, cudf::get_current_device_resource_ref());
  // PERMISSIVE-mode row nulling support. Unknown enum values and malformed rows should both
  // surface as null structs instead of partially decoded data.
  bool const track_permissive_null_rows = !fail_on_errors;
  rmm::device_uvector<bool> d_row_force_null(
    track_permissive_null_rows ? num_rows : 0, stream, cudf::get_current_device_resource_ref());
  if (track_permissive_null_rows) {
    CUDF_CUDA_TRY(
      cudaMemsetAsync(d_row_force_null.data(), 0, num_rows * sizeof(bool), stream.value()));
  }
  auto const decode_ctx = protobuf_decode_runtime_context{
    &d_row_force_null, &d_error, &d_deferred_enum_error, fail_on_errors};

  auto const threads = THREADS_PER_BLOCK;

  // Allocate for counting repeated fields. `std::max(..., 1)` keeps the device_uvector
  // non-empty when the corresponding field count is 0, so `.data()` remains a valid pointer.
  rmm::device_uvector<field_occurrence_count> d_repeated_info(
    std::max<size_t>(static_cast<size_t>(num_rows) * num_repeated, 1), stream, scratch_mr);
  rmm::device_uvector<field_location> d_nested_locations(
    std::max<size_t>(static_cast<size_t>(num_rows) * num_nested, 1), stream, scratch_mr);
  rmm::device_uvector<field_occurrence_count> d_nested_occurrence_info(
    std::max<size_t>(static_cast<size_t>(num_rows) * num_nested, 1), stream, scratch_mr);
  auto d_multiple_nested_fields =
    cudf::detail::make_zeroed_device_uvector_async<int>(num_nested, stream, scratch_mr);

  if (run_count_scan) {
    std::vector<int> count_field_indices = repeated_field_indices;
    count_field_indices.insert(
      count_field_indices.end(), nested_field_indices.begin(), nested_field_indices.end());
    std::vector<int> output_indices;
    output_indices.reserve(count_field_indices.size());
    for (int i = 0; i < num_repeated; ++i) {
      output_indices.push_back(i);
    }
    for (int i = 0; i < num_nested; ++i) {
      output_indices.push_back(i);
    }

    auto field_descs = make_field_descriptors(
      count_field_indices, schema_context, stream, scratch_mr, output_indices);
    auto h_field_lookup = build_field_lookup_table(
      field_descs.host.data(), static_cast<int>(field_descs.host.size()), stream);
    auto d_field_lookup =
      cudf::detail::make_device_uvector_async(h_field_lookup, stream, scratch_mr);

    launch_count_repeated_fields(*d_in,
                                 {d_nested_locations.data(),
                                  num_nested,
                                  d_repeated_info.data(),
                                  num_repeated,
                                  d_nested_occurrence_info.data(),
                                  num_nested,
                                  nullptr,
                                  d_multiple_nested_fields.data(),
                                  {field_descs.device.data(),
                                   static_cast<int>(field_descs.host.size()),
                                   h_field_lookup.empty() ? nullptr : d_field_lookup.data(),
                                   static_cast<int>(h_field_lookup.size())}},
                                 d_error.data(),
                                 track_permissive_null_rows ? d_row_force_null.data() : nullptr,
                                 stream);
  }

  std::vector<std::optional<singular_message_merge_work>> nested_merge_work(num_nested);
  if (num_nested > 0) {
    auto h_multiple_nested_fields = cudf::detail::make_pinned_vector_async<int>(num_nested, stream);
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(h_multiple_nested_fields.data(),
                                             d_multiple_nested_fields.data(),
                                             num_nested * sizeof(int),
                                             stream));
    stream.synchronize();

    for (int ni = 0; ni < num_nested; ++ni) {
      if (h_multiple_nested_fields[ni] == 0) { continue; }

      auto counts_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator<int>(0),
        extract_strided_count{d_nested_occurrence_info.data(), ni, num_nested});
      nested_merge_work[ni].emplace(
        nested_field_indices[ni],
        make_list_offsets_from_counts(
          counts_begin, num_rows, "Top-level singular message", stream, scratch_mr, scratch_mr),
        stream,
        scratch_mr);
      auto& work = *nested_merge_work[ni];

      auto h_scan_descs =
        cudf::detail::make_pinned_vector_async<field_occurrence_scan_desc>(1, stream);
      h_scan_descs[0]  = {schema[work.schema_idx].field_number,
                          proto_wire_type::LEN,
                          work.row_offsets.data(),
                          work.fragments.data()};
      auto scan_bundle = make_field_occurrence_scan_bundle(h_scan_descs, stream, scratch_mr);
      launch_scan_singular_message_occurrences(*d_in, scan_bundle.view(), d_error.data(), stream);
    }
  }

  // Store decoded columns by schema index for ordered assembly at the end.
  std::vector<std::unique_ptr<cudf::column>> column_map(num_fields);

  // Process scalar fields using scan + extract infrastructure
  if (run_field_scan) {
    auto field_descs =
      make_field_descriptors(scalar_field_indices, schema_context, stream, scratch_mr);
    auto const& h_field_descs = field_descs.host;
    auto const& d_field_descs = field_descs.device;

    rmm::device_uvector<field_location> d_locations(
      static_cast<size_t>(num_rows) * num_scalar, stream, scratch_mr);

    auto h_field_lookup = build_field_lookup_table(h_field_descs.data(), num_scalar, stream);
    auto d_field_lookup =
      cudf::detail::make_device_uvector_async(h_field_lookup, stream, scratch_mr);

    launch_scan_all_fields(*d_in,
                           {d_locations.data(),
                            num_scalar,
                            nullptr,
                            0,
                            nullptr,
                            0,
                            d_deferred_enum_error.data(),
                            nullptr,
                            {d_field_descs.data(),
                             num_scalar,
                             h_field_lookup.empty() ? nullptr : d_field_lookup.data(),
                             static_cast<int>(h_field_lookup.size())}},
                           d_error.data(),
                           track_permissive_null_rows ? d_row_force_null.data() : nullptr,
                           stream);

    // Required-field validation applies to all scalar leaves, not just top-level numerics.
    maybe_check_required_fields({d_locations.data(),
                                 {num_rows, nullptr, enum_error_scope::root},
                                 binary_input.null_count() > 0 ? binary_input.null_mask() : nullptr,
                                 binary_input.offset(),
                                 nullptr},
                                scalar_field_indices,
                                schema,
                                decode_ctx,
                                stream);

    // Batched scalar extraction: group non-special fixed-width fields by extraction
    // category and extract all fields of each category with a single 2D kernel launch.
    {
      static constexpr auto fallback = SCALAR_KINDS.size();
      std::array<std::vector<int>, SCALAR_KINDS.size() + 1> group_lists;
      auto find_group = [](cudf::type_id type, proto_encoding encoding) {
        auto const decode = get_scalar_decode_kind(type, encoding);
        auto const it     = std::ranges::find(SCALAR_KINDS, scalar_kind{type, decode});
        return static_cast<size_t>(it - SCALAR_KINDS.begin());
      };

      for (int i = 0; i < num_scalar; i++) {
        int const si        = scalar_field_indices[i];
        auto const field    = schema_context.field(si);
        auto const type     = field.output_type.id();
        auto const encoding = field.schema.encoding;

        // STRING (including enum-as-string) and LIST go to the per-field path.
        if (type == cudf::type_id::STRING || type == cudf::type_id::LIST) continue;

        // INT32 with enum validation goes to fallback.
        if (type == cudf::type_id::INT32 && !field.enum_valid_values.empty()) {
          group_lists[fallback].push_back(i);
          continue;
        }

        group_lists[find_group(type, encoding)].push_back(i);
      }

      auto launch_decoder = [&]<typename T, auto DecodeFn>(auto const& indices) {
        int const nf = static_cast<int>(indices.size());
        if (nf == 0) return;

        std::vector<rmm::device_uvector<T>> outputs;
        std::vector<rmm::device_uvector<bool>> valid;
        outputs.reserve(nf);
        valid.reserve(nf);
        auto h_descs = cudf::detail::make_pinned_vector_async<batched_scalar_desc<T>>(nf, stream);

        for (int j = 0; j < nf; j++) {
          int const li     = indices[j];
          int si           = scalar_field_indices[li];
          auto const field = schema_context.field(si);
          outputs.emplace_back(num_rows, stream, mr);
          valid.emplace_back(num_rows, stream, cudf::get_current_device_resource_ref());
          h_descs[j] = {
            li, outputs.back().data(), valid.back().data(), make_scalar_decode_options<T>(field)};
        }

        if (num_rows > 0) {
          auto d_descs = cudf::detail::make_device_uvector_async(
            h_descs, stream, cudf::get_current_device_resource_ref());
          dim3 grid((num_rows + threads - 1u) / threads, nf);
          auto const batch_input = batched_scalar_input_view<T>{
            input, d_locations.data(), num_scalar, d_descs.data(), nf, d_error.data()};
          extract_scalar_batched_kernel<T, DecodeFn>
            <<<grid, threads, 0, stream.value()>>>(batch_input);
        }

        for (int j = 0; j < nf; j++) {
          int const si            = scalar_field_indices[indices[j]];
          auto dt                 = cudf::data_type{schema[si].output_type};
          auto [mask, null_count] = make_null_mask_from_valid(valid[j], num_rows, stream, mr);
          column_map[si]          = std::make_unique<cudf::column>(
            dt, num_rows, outputs[j].release(), std::move(mask), null_count);
        }
      };

      // `std::size_t = I` works around nvcc #445-D diagnostic.
      auto launch_index = [&]<std::size_t I, std::size_t = I>() {
        constexpr auto type = SCALAR_KINDS[I].type;
        using T = std::conditional_t<type == cudf::type_id::BOOL8, uint8_t, cudf::id_to_type<type>>;
        dispatch_scalar_decoder<T, SCALAR_KINDS[I].decode>([&]<auto DecodeFn>() {
          launch_decoder.template operator()<T, DecodeFn>(group_lists[I]);
        });
      };
      [&]<size_t... I>(std::index_sequence<I...>) {
        (launch_index.template operator()<I>(), ...);
      }(std::make_index_sequence<SCALAR_KINDS.size()>{});

      // Per-field fallback (INT32 with enum, etc.)
      for (int i : group_lists[fallback]) {
        int schema_idx = scalar_field_indices[i];
        top_level_location_provider loc_provider{
          list_offsets, base_offset, d_locations.data(), i, num_scalar};
        column_map[schema_idx] = extract_typed_column({{schema_context, decode_ctx},
                                                       message_data,
                                                       schema_idx,
                                                       {num_rows, nullptr, enum_error_scope::root}},
                                                      loc_provider,
                                                      stream,
                                                      mr);
      }
    }

    // Per-field extraction for STRING and LIST types.
    for (int i = 0; i < num_scalar; i++) {
      int schema_idx        = scalar_field_indices[i];
      auto const field_meta = schema_context.field(schema_idx);
      auto const dt         = field_meta.output_type;
      if (dt.id() != cudf::type_id::STRING && dt.id() != cudf::type_id::LIST) { continue; }
      auto const has_default = field_meta.schema.has_default_value;
      top_level_location_provider loc_provider{
        list_offsets, base_offset, d_locations.data(), i, num_scalar};
      auto valid_fn =
        [locs = d_locations.data(), i, num_scalar, has_default] __device__(cudf::size_type row) {
          return locs[flat_index(row, num_scalar, i)].offset >= 0 || has_default;
        };
      auto get_top_row_indices = []() { return static_cast<int32_t const*>(nullptr); };
      column_map[schema_idx] =
        build_protobuf_field_values_column_shared({{schema_context, decode_ctx},
                                                   message_data,
                                                   schema_idx,
                                                   {num_rows, nullptr, enum_error_scope::root}},
                                                  loc_provider,
                                                  valid_fn,
                                                  get_top_row_indices,
                                                  stream,
                                                  mr);
    }
  }

  // Required top-level nested messages are tracked in d_nested_locations during the scan/count
  // pass.
  maybe_check_required_fields({d_nested_locations.data(),
                               {num_rows, nullptr, enum_error_scope::root},
                               binary_input.null_count() > 0 ? binary_input.null_mask() : nullptr,
                               binary_input.offset(),
                               nullptr},
                              nested_field_indices,
                              schema,
                              decode_ctx,
                              stream);

  // Process repeated fields (three-phase: offsets → combined scan → build columns)
  if (num_repeated > 0) {
    auto rep_work = make_repeated_field_work_bundle(std::views::iota(0, num_repeated),
                                                    repeated_field_indices,
                                                    d_repeated_info.data(),
                                                    num_rows,
                                                    schema_context,
                                                    "Top-level repeated field",
                                                    stream,
                                                    mr,
                                                    scratch_mr);

    if (!rep_work.scan_descriptors.empty()) {
      auto scan_bundle =
        make_field_occurrence_scan_bundle(rep_work.scan_descriptors, stream, scratch_mr);
      launch_scan_all_field_occurrences(*d_in, scan_bundle.view(), d_error.data(), stream);
    }

    // Phase C: Build columns per field.
    for (int ri = 0; ri < num_repeated; ri++) {
      auto& w             = rep_work.fields[ri].value();
      int schema_idx      = w.schema_idx;
      auto element_type   = cudf::data_type{schema[schema_idx].output_type};
      int32_t total_count = w.total_count;

      if (total_count <= 0) {
        // All rows empty: w.offsets is already a zero-filled buffer from Phase A.
        auto offsets_col = make_offsets_column(num_rows, std::move(w.offsets));

        auto child_col =
          element_type.id() == cudf::type_id::STRUCT
            ? make_empty_struct_column_with_schema(schema_context, schema_idx, stream, mr)
            : make_empty_column_safe(element_type, stream, mr);
        column_map[schema_idx] = make_list_column_with_input_nulls(
          num_rows, std::move(offsets_col), std::move(child_col), binary_input, stream, mr);
        continue;
      }

      auto const field_meta = schema_context.field(schema_idx);

      // For repeated fields, schema[].output_type holds the element type (not the outer LIST).
      switch (element_type.id()) {
        case cudf::type_id::INT32:
          column_map[schema_idx] = build_repeated_scalar_column<int32_t>(
            binary_input, input, schema_context, decode_ctx, std::move(w), stream, mr);
          break;
        case cudf::type_id::INT64:
          column_map[schema_idx] = build_repeated_scalar_column<int64_t>(
            binary_input, input, schema_context, decode_ctx, std::move(w), stream, mr);
          break;
        case cudf::type_id::UINT32:
          column_map[schema_idx] = build_repeated_scalar_column<uint32_t>(
            binary_input, input, schema_context, decode_ctx, std::move(w), stream, mr);
          break;
        case cudf::type_id::UINT64:
          column_map[schema_idx] = build_repeated_scalar_column<uint64_t>(
            binary_input, input, schema_context, decode_ctx, std::move(w), stream, mr);
          break;
        case cudf::type_id::FLOAT32:
          column_map[schema_idx] = build_repeated_scalar_column<float>(
            binary_input, input, schema_context, decode_ctx, std::move(w), stream, mr);
          break;
        case cudf::type_id::FLOAT64:
          column_map[schema_idx] = build_repeated_scalar_column<double>(
            binary_input, input, schema_context, decode_ctx, std::move(w), stream, mr);
          break;
        case cudf::type_id::BOOL8:
          column_map[schema_idx] = build_repeated_scalar_column<uint8_t>(
            binary_input, input, schema_context, decode_ctx, std::move(w), stream, mr);
          break;
        case cudf::type_id::STRING: {
          auto enc = field_meta.schema.encoding;
          if (enc == proto_encoding::ENUM_STRING) {
            // Same host-side schema check as the scalar enum path — fail loudly instead of
            // silently emitting a null column.
            CUDF_EXPECTS(!field_meta.enum_valid_values.empty() &&
                           field_meta.enum_valid_values.size() == field_meta.enum_names.size(),
                         "Protobuf decode error: missing or mismatched enum metadata for "
                         "enum-as-string field");
            column_map[schema_idx] = build_repeated_enum_string_column(
              binary_input, input, schema_context, decode_ctx, std::move(w), stream, mr);
          } else {
            column_map[schema_idx] = build_repeated_string_column(
              binary_input, input, field_meta, std::move(w), stream, mr);
          }
          break;
        }
        case cudf::type_id::LIST:  // bytes as LIST<INT8>
          column_map[schema_idx] =
            build_repeated_string_column(binary_input, input, field_meta, std::move(w), stream, mr);
          break;
        case cudf::type_id::STRUCT: {
          auto const& child_field_indices = schema_context.children(schema_idx);
          column_map[schema_idx]          = build_repeated_struct_column(binary_input,
                                                                input,
                                                                child_field_indices,
                                                                         {schema_context, decode_ctx},
                                                                std::move(w),
                                                                stream,
                                                                mr);
          break;
        }
        default:
          CUDF_FAIL("Protobuf decode internal error: unsupported repeated element type id=" +
                    std::to_string(static_cast<int>(element_type.id())));
      }
    }  // for (ri)
  }

  // Process nested struct fields after top-level repeated fields so malformed repeated
  // occurrences can still mark their rows before nested columns are assembled.
  for (int ni = 0; ni < num_nested; ni++) {
    int parent_schema_idx           = nested_field_indices[ni];
    auto const& child_field_indices = schema_context.children(parent_schema_idx);

    // Keep row-force-null tracking for nested required-field failures, but do not let invalid
    // nested enum values null the top-level row.
    std::unique_ptr<cudf::column> nested_col;
    if (nested_merge_work[ni].has_value()) {
      nested_col = build_merged_singular_struct_column(input,
                                                       {nullptr, nullptr},
                                                       child_field_indices,
                                                       {schema_context, decode_ctx},
                                                       std::move(*nested_merge_work[ni]),
                                                       0,
                                                       stream,
                                                       mr);
    } else {
      rmm::device_uvector<field_location> d_parent_locs(num_rows, stream, scratch_mr);
      launch_extract_strided_locations(
        d_nested_locations.data(), ni, num_nested, d_parent_locs.data(), num_rows, stream);
      nested_col = build_nested_struct_column(input,
                                              {d_parent_locs.data(), d_parent_locs.size(), nullptr},
                                              child_field_indices,
                                              {schema_context, decode_ctx},
                                              0,
                                              stream,
                                              mr);
    }
    propagate_nulls_to_descendants(*nested_col, stream, mr);
    column_map[parent_schema_idx] = std::move(nested_col);
  }

  // Assemble top_level_children in schema order (not processing order). Hidden fields are
  // still decoded above (so validation errors surface), but dropped from the output struct.
  std::vector<std::unique_ptr<cudf::column>> top_level_children;
  for (int i = 0; i < num_fields; i++) {
    if (schema[i].parent_idx != -1 || !schema_context.is_output(i)) { continue; }
    top_level_children.push_back(
      column_map[i] ? std::move(column_map[i])
                    : make_null_column_with_schema(schema_context, i, num_rows, stream, mr));
  }

  {
    using enum protobuf_error;
    CUDF_CUDA_TRY(cudaPeekAtLastError());
    protobuf_error h_error               = NONE;
    protobuf_error h_deferred_enum_error = NONE;
    CUDF_CUDA_TRY(
      cudf::detail::memcpy_async(&h_error, d_error.data(), sizeof(protobuf_error), stream));
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(
      &h_deferred_enum_error, d_deferred_enum_error.data(), sizeof(protobuf_error), stream));
    stream.synchronize();
    if (h_error == NONE) { h_error = h_deferred_enum_error; }
    if (h_error == SCHEMA_TOO_LARGE || h_error == REPEATED_COUNT_MISMATCH) {
      throw cudf::logic_error(error_message(h_error));
    }
    if (fail_on_errors && h_error != NONE) { throw cudf::logic_error(error_message(h_error)); }
  }

  // Build final struct null mask by combining input nulls with PERMISSIVE-mode row invalidation.
  cudf::size_type struct_null_count = 0;
  rmm::device_buffer struct_mask{0, stream, mr};
  auto const input_null_count = binary_input.null_count();

  if (track_permissive_null_rows || input_null_count > 0) {
    auto const* input_mask  = binary_input.null_mask();
    auto input_offset       = binary_input.offset();
    auto [mask, null_count] = cudf::detail::valid_if(
      thrust::make_counting_iterator<cudf::size_type>(0),
      thrust::make_counting_iterator<cudf::size_type>(num_rows),
      [row_invalid = track_permissive_null_rows ? d_row_force_null.data() : nullptr,
       input_mask,
       input_offset] __device__(cudf::size_type row) {
        if (input_mask != nullptr && !cudf::bit_is_set(input_mask, input_offset + row)) {
          return false;
        }
        if (row_invalid != nullptr && row_invalid[row]) return false;
        return true;
      },
      stream,
      mr);
    struct_null_count = null_count;
    if (null_count > 0) { struct_mask = std::move(mask); }
  }

  // cuDF child views do not automatically inherit parent nulls. Push nulls down into every
  // top-level child, then recursively through nested STRUCT/LIST children, so callers that
  // access backing grandchildren directly still observe logically-null rows.
  if (struct_null_count > 0) {
    auto const* struct_mask_ptr = static_cast<cudf::bitmask_type const*>(struct_mask.data());
    for (auto& child : top_level_children) {
      apply_parent_mask_to_row_aligned_column(
        *child, struct_mask_ptr, struct_null_count, num_rows, stream, mr);
      propagate_nulls_to_descendants(*child, stream, mr);
    }
  }

  return cudf::make_structs_column(
    num_rows, std::move(top_level_children), struct_null_count, std::move(struct_mask), stream, mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> decode_protobuf_to_struct(cudf::column_view const& binary_input,
                                                        protobuf_decode_context const& context,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return detail::decode_protobuf_to_struct(binary_input, context, stream, mr);
}

}  // namespace spark_rapids_jni::protobuf
