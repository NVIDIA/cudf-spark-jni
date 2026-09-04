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

#include "protobuf/protobuf_types.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <source_location>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace spark_rapids_jni::protobuf::detail {

struct enum_string_lookup_tables {
  rmm::device_uvector<int32_t> d_valid_enums;
  rmm::device_uvector<int32_t> d_name_offsets;
  rmm::device_uvector<uint8_t> d_name_chars;

  enum_string_lookup_device_view view() const
  {
    return {{d_valid_enums.data(), static_cast<int>(d_valid_enums.size())},
            d_name_offsets.data(),
            d_name_chars.data()};
  }
};

class protobuf_schema {
 public:
  explicit protobuf_schema(protobuf_decode_context const& context);
  protobuf_schema(protobuf_decode_context&&)       = delete;
  protobuf_schema(protobuf_decode_context const&&) = delete;

  protobuf_schema(protobuf_schema const&)            = delete;
  protobuf_schema& operator=(protobuf_schema const&) = delete;
  protobuf_schema(protobuf_schema&&)                 = delete;
  protobuf_schema& operator=(protobuf_schema&&)      = delete;

  [[nodiscard]] std::vector<nested_field_descriptor> const& fields() const
  {
    return context_.schema;
  }

  [[nodiscard]] nested_field_descriptor const& operator[](int schema_idx) const
  {
    return context_.schema.at(static_cast<size_t>(schema_idx));
  }

  [[nodiscard]] size_t size() const { return context_.schema.size(); }

  [[nodiscard]] protobuf_field_meta_view field(int schema_idx) const;
  [[nodiscard]] std::vector<int> const& children(int parent_schema_idx) const;
  [[nodiscard]] bool is_output(int schema_idx) const;
  [[nodiscard]] enum_string_lookup_tables enum_lookup(int schema_idx,
                                                      cuda::stream_ref stream) const;

 private:
  // Avoid copying pinned metadata; the decode context outlives this stack-scoped facade.
  protobuf_decode_context const& context_;
  std::vector<std::vector<int>> children_by_parent_;
};

// Keep pinned staging alive alongside its device copy until queued H2D work completes.
struct field_descriptor_bundle {
  cudf::detail::host_vector<field_descriptor> host;
  rmm::device_uvector<field_descriptor> device;
  rmm::device_uvector<int32_t> enum_values;
};

field_descriptor_bundle make_field_descriptors(std::vector<int> const& field_indices,
                                               protobuf_schema const& schema,
                                               cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr,
                                               std::span<int const> output_indices = {});

// ============================================================================
// Nested decode view bundles
// ============================================================================

struct protobuf_decode_runtime_context {
  rmm::device_uvector<bool>* row_force_null;
  rmm::device_uvector<protobuf_error>* error;
};

struct recursive_decode_context {
  protobuf_schema const& schema;
  protobuf_decode_runtime_context runtime;
};

struct protobuf_field_decode_request {
  recursive_decode_context context;
  uint8_t const* message_data;
  int schema_idx;
  int num_values;
};

struct list_offsets_from_counts_result {
  int32_t total_count;
  rmm::device_uvector<int32_t> offsets;
};

// Offsets become LIST output storage; occurrences remain scratch used by value extraction.
struct repeated_field_work {
  int schema_idx;
  int depth;
  int32_t total_count;
  rmm::device_uvector<int32_t> offsets;
  std::unique_ptr<rmm::device_uvector<field_occurrence>> occurrences;

  repeated_field_work(int schema_index,
                      list_offsets_from_counts_result offsets_result,
                      int nested_depth = 0)
    : schema_idx(schema_index),
      depth(nested_depth),
      total_count(offsets_result.total_count),
      offsets(std::move(offsets_result.offsets))
  {
  }
};

struct repeated_field_work_bundle {
  std::vector<std::optional<repeated_field_work>> fields;
  cudf::detail::host_vector<field_occurrence_scan_desc> scan_descriptors;
};

struct extract_strided_count {
  field_occurrence_count const* info;
  int field_position;
  int num_fields;

  __device__ int32_t operator()(int row) const
  {
    return info[flat_index(row, num_fields, field_position)].count;
  }
};

// Buffers that gather duplicate singular-message fragments by row before recursive decoding.
struct singular_message_merge_buffers {
  int schema_idx;
  int32_t total_fragments;
  rmm::device_uvector<int32_t> row_offsets;
  rmm::device_uvector<field_occurrence> fragments;

  singular_message_merge_buffers(int schema_index,
                                 list_offsets_from_counts_result offsets_result,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
    : schema_idx(schema_index),
      total_fragments(offsets_result.total_count),
      row_offsets(std::move(offsets_result.offsets)),
      fragments(total_fragments, stream, mr)
  {
  }

  explicit singular_message_merge_buffers(repeated_field_work&& work)
    : schema_idx(work.schema_idx),
      total_fragments(work.total_count),
      row_offsets(std::move(work.offsets)),
      fragments(std::move(*work.occurrences))
  {
  }
};

inline std::unique_ptr<cudf::column> make_offsets_column(cudf::size_type num_rows,
                                                         rmm::device_uvector<int32_t>&& offsets)
{
  CUDF_EXPECTS(offsets.size() == static_cast<size_t>(num_rows) + 1,
               std::string{__func__} + ": offsets size must match row count");
  return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                        num_rows + 1,
                                        offsets.release(),
                                        rmm::device_buffer{},
                                        0);
}

inline rmm::device_uvector<int32_t> make_top_row_indices(
  rmm::device_uvector<field_occurrence> const& occurrences,
  int32_t const* parent_top_row_indices,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<int32_t> result(occurrences.size(), stream, mr);
  thrust::transform(rmm::exec_policy_nosync(stream, mr),
                    occurrences.begin(),
                    occurrences.end(),
                    result.begin(),
                    [parent_top_row_indices] __device__(field_occurrence const& occurrence) {
                      return parent_top_row_indices != nullptr
                               ? parent_top_row_indices[occurrence.row_idx]
                               : occurrence.row_idx;
                    });
  return result;
}

inline void validate_nonempty_repeated_field_work(
  repeated_field_work const& work,
  int num_rows,
  std::source_location const& location = std::source_location::current())
{
  auto const caller  = location.function_name();
  auto const message = [caller](char const* detail) { return std::string{caller} + ": " + detail; };
  CUDF_EXPECTS(work.total_count > 0, message("total count must be positive"));
  CUDF_EXPECTS(work.offsets.size() == static_cast<size_t>(num_rows) + 1,
               message("offsets size must match row count"));
  CUDF_EXPECTS(work.occurrences != nullptr, message("repeated occurrences must be present"));
  CUDF_EXPECTS(work.occurrences->size() == static_cast<size_t>(work.total_count),
               message("occurrence count mismatch"));
}

template <typename CountIterator>
inline list_offsets_from_counts_result make_list_offsets_from_counts(
  CountIterator counts_begin,
  int num_rows,
  char const* count_context,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref output_mr,
  rmm::device_async_resource_ref scratch_mr)
{
  CUDF_EXPECTS(num_rows >= 0, std::string{__func__} + ": row count must be non-negative");
  auto const counts_end     = counts_begin + num_rows;
  auto const total_count_64 = thrust::reduce(
    rmm::exec_policy_nosync(stream, scratch_mr), counts_begin, counts_end, int64_t{0});
  CUDF_EXPECTS(total_count_64 >= 0, std::string{__func__} + ": total count must be non-negative");
  CUDF_EXPECTS(total_count_64 <= std::numeric_limits<int32_t>::max(),
               std::string{count_context} + " total element count exceeds 2^31-1");
  auto const total_count = static_cast<int32_t>(total_count_64);
  CUDF_EXPECTS(num_rows > 0 || total_count == 0,
               std::string{__func__} + ": empty input cannot have repeated elements");

  rmm::device_uvector<int32_t> offsets(num_rows + 1, stream, output_mr);
  if (num_rows > 0) {
    thrust::exclusive_scan(rmm::exec_policy_nosync(stream, scratch_mr),
                           counts_begin,
                           counts_end,
                           offsets.begin(),
                           int32_t{0});
  }
  thrust::fill_n(
    rmm::exec_policy_nosync(stream, scratch_mr), offsets.data() + num_rows, 1, total_count);
  return {total_count, std::move(offsets)};
}

template <typename PositionRange>
inline repeated_field_work_bundle make_repeated_field_work_bundle(
  PositionRange const& field_positions,
  std::vector<int> const& schema_indices,
  field_occurrence_count const* repeated_info,
  int num_rows,
  protobuf_schema const& schema,
  char const* count_context,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref output_mr,
  rmm::device_async_resource_ref scratch_mr)
{
  auto const num_fields = static_cast<int>(schema_indices.size());
  repeated_field_work_bundle result{
    std::vector<std::optional<repeated_field_work>>(num_fields),
    cudf::detail::make_pinned_vector_async<field_occurrence_scan_desc>(0, stream)};
  result.scan_descriptors.reserve(std::ranges::size(field_positions));

  std::vector<int> positions;
  positions.reserve(std::ranges::size(field_positions));
  for (auto const field_position : field_positions) {
    CUDF_EXPECTS(field_position >= 0 && field_position < num_fields,
                 std::string{__func__} + ": field position is out of bounds");
    positions.push_back(field_position);
  }
  if (positions.empty()) return result;
  CUDF_EXPECTS(repeated_info != nullptr,
               std::string{__func__} + ": repeated count buffer must be non-null");

  std::vector<std::optional<rmm::device_uvector<int64_t>>> wide_offsets(num_fields);
  auto totals = cudf::detail::make_pinned_vector_async<int64_t>(positions.size(), stream);
  std::vector<void*> total_dsts(positions.size());
  std::vector<void const*> total_srcs(positions.size());
  std::vector<size_t> total_sizes(positions.size(), sizeof(int64_t));

  for (size_t i = 0; i < positions.size(); ++i) {
    auto const field_position = positions[i];
    auto counts_begin         = thrust::make_transform_iterator(
      thrust::make_counting_iterator<int>(0),
      [repeated_info, field_position, num_fields] __device__(int row) -> int64_t {
        return repeated_info[flat_index(row, num_fields, field_position)].count;
      });
    auto& offsets = wide_offsets[field_position].emplace(num_rows + 1, stream, scratch_mr);
    thrust::fill_n(rmm::exec_policy_nosync(stream, scratch_mr), offsets.begin(), 1, int64_t{0});
    if (num_rows > 0) {
      thrust::inclusive_scan(rmm::exec_policy_nosync(stream, scratch_mr),
                             counts_begin,
                             counts_begin + num_rows,
                             offsets.begin() + 1);
    }
    total_dsts[i] = &totals[i];
    total_srcs[i] = offsets.data() + num_rows;
  }
  CUDF_CUDA_TRY(cudf::detail::memcpy_batch_async(
    total_dsts.data(), total_srcs.data(), total_sizes.data(), positions.size(), stream));
  stream.sync();

  for (size_t i = 0; i < positions.size(); ++i) {
    auto const field_position = positions[i];
    auto const total_count_64 = totals[i];
    CUDF_EXPECTS(total_count_64 >= 0, std::string{__func__} + ": total count must be non-negative");
    CUDF_EXPECTS(total_count_64 <= std::numeric_limits<int32_t>::max(),
                 std::string{count_context} + " total element count exceeds 2^31-1");
    auto const total_count = static_cast<int32_t>(total_count_64);
    auto const schema_idx  = schema_indices[field_position];
    auto const field_mr    = schema.is_output(schema_idx) ? output_mr : scratch_mr;
    rmm::device_uvector<int32_t> offsets(num_rows + 1, stream, field_mr);
    thrust::transform(rmm::exec_policy_nosync(stream, scratch_mr),
                      wide_offsets[field_position]->begin(),
                      wide_offsets[field_position]->end(),
                      offsets.begin(),
                      [] __device__(int64_t value) { return static_cast<int32_t>(value); });
    auto& work = result.fields[field_position].emplace(
      schema_idx,
      list_offsets_from_counts_result{total_count, std::move(offsets)},
      schema[schema_idx].depth);

    if (work.total_count > 0) {
      work.occurrences = std::make_unique<rmm::device_uvector<field_occurrence>>(
        work.total_count, stream, scratch_mr);
    }
    // Zero-count descriptors keep malformed rows aligned with the count pass.
    auto const& field = schema[schema_idx];
    result.scan_descriptors.push_back(
      field_occurrence_scan_desc{field.field_number,
                                 field.wire_type,
                                 work.offsets.data(),
                                 work.occurrences == nullptr ? nullptr : work.occurrences->data()});
  }
  return result;
}

// ============================================================================
// Field number lookup table helpers
// ============================================================================

/**
 * Build a host-side direct-mapped lookup table: field_number -> index.
 * @param get_field_number Callable: (int i) -> field_number for the i-th entry.
 * @param num_entries Number of entries.
 * @return Empty vector if there are no entries or the max field number exceeds the threshold.
 */
template <typename FieldNumberFn>
inline cudf::detail::host_vector<int> build_lookup_table(FieldNumberFn get_field_number,
                                                         int num_entries,
                                                         cuda::stream_ref stream)
{
  if (num_entries == 0) { return cudf::detail::make_pinned_vector_async<int>(0, stream); }

  int max_fn = 0;
  for (int i = 0; i < num_entries; i++) {
    max_fn = std::max(max_fn, get_field_number(i));
  }
  if (max_fn > FIELD_LOOKUP_TABLE_MAX) {
    return cudf::detail::make_pinned_vector_async<int>(0, stream);
  }
  auto table = cudf::detail::make_pinned_vector_async<int>(max_fn + 1, stream);
  std::fill(table.begin(), table.end(), -1);
  for (int i = 0; i < num_entries; i++) {
    table[get_field_number(i)] = i;
  }
  return table;
}

template <typename FieldDesc>
inline cudf::detail::host_vector<int> build_field_lookup_table(FieldDesc const* descs,
                                                               int num_fields,
                                                               cuda::stream_ref stream)
{
  return build_lookup_table([&](int i) { return descs[i].field_number; }, num_fields, stream);
}

struct field_occurrence_scan_bundle {
  rmm::device_uvector<field_occurrence_scan_desc> descriptors;
  rmm::device_uvector<int> field_number_lookup;

  field_occurrence_scan_view view() const
  {
    auto const lookup_size = static_cast<int>(field_number_lookup.size());
    return {descriptors.data(),
            static_cast<int>(descriptors.size()),
            lookup_size > 0 ? field_number_lookup.data() : nullptr,
            lookup_size};
  }
};

inline field_occurrence_scan_bundle make_field_occurrence_scan_bundle(
  cudf::detail::host_vector<field_occurrence_scan_desc> const& host_descriptors,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto descriptors = cudf::detail::make_device_uvector_async(host_descriptors, stream, mr);
  // Stream-ordered pinned deallocation keeps this staging safe without a local sync.
  auto host_lookup = build_field_lookup_table(
    host_descriptors.data(), static_cast<int>(host_descriptors.size()), stream);
  auto lookup = cudf::detail::make_device_uvector_async(host_lookup, stream, mr);
  return {std::move(descriptors), std::move(lookup)};
}

template <typename LaunchFn>
inline void launch_occurrence_scan_batches(
  cudf::detail::host_vector<field_occurrence_scan_desc> const& descriptors,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref scratch_mr,
  LaunchFn launch)
{
  for (size_t first = 0; first < descriptors.size(); first += MAX_REPEATED_FIELDS_PER_KERNEL) {
    auto const count = std::min<size_t>(MAX_REPEATED_FIELDS_PER_KERNEL, descriptors.size() - first);
    auto batch = cudf::detail::make_pinned_vector_async<field_occurrence_scan_desc>(count, stream);
    std::copy_n(descriptors.begin() + first, count, batch.begin());
    auto bundle = make_field_occurrence_scan_bundle(batch, stream, scratch_mr);
    launch(bundle.view());
  }
}

/**
 * Find all child field indices for a given parent index in the schema.
 * This is a commonly used pattern throughout the codebase.
 *
 * @param schema The host schema.
 * @param parent_idx The parent index to search for
 * @return Vector of child field indices
 */
template <typename SchemaT>
std::vector<int> find_child_field_indices(SchemaT const& schema, int parent_idx)
{
  std::vector<int> child_indices;
  for (int i = 0; i < static_cast<int>(schema.size()); i++) {
    if (schema[i].parent_idx == parent_idx) { child_indices.push_back(i); }
  }
  return child_indices;
}

inline std::vector<int> const& find_child_field_indices(protobuf_schema const& schema,
                                                        int parent_idx)
{
  return schema.children(parent_idx);
}

// Forward declarations needed by make_empty_struct_column_with_schema
std::unique_ptr<cudf::column> make_empty_column_safe(cudf::data_type dtype,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> make_empty_list_column(std::unique_ptr<cudf::column> element_col,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr);

// Forward declaration for the mutual recursion with make_empty_struct_column_from_children.
template <typename SchemaT>
std::unique_ptr<cudf::column> make_empty_struct_column_with_schema(
  SchemaT const& schema,
  int parent_idx,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

// Build an empty (0-row) STRUCT column from an explicit child-index list, recursing into
// STRUCT children and wrapping repeated fields in an empty LIST. Shared by both the
// parent-indexed entry point and build_nested_struct_column's zero-row fast path.
template <typename SchemaT>
std::unique_ptr<cudf::column> make_empty_struct_column_from_children(
  SchemaT const& schema,
  std::vector<int> const& child_indices,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> children;
  for (int child_idx : child_indices) {
    auto child_type = cudf::data_type{schema[child_idx].output_type};

    std::unique_ptr<cudf::column> child_col;
    if (child_type.id() == cudf::type_id::STRUCT) {
      child_col = make_empty_struct_column_with_schema(schema, child_idx, stream, mr);
    } else {
      child_col = make_empty_column_safe(child_type, stream, mr);
    }

    if (schema[child_idx].is_repeated) {
      child_col = make_empty_list_column(std::move(child_col), stream, mr);
    }

    children.push_back(std::move(child_col));
  }

  return cudf::make_structs_column(0, std::move(children), 0, rmm::device_buffer{}, stream, mr);
}

template <typename SchemaT>
std::unique_ptr<cudf::column> make_empty_struct_column_with_schema(
  SchemaT const& schema, int parent_idx, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  auto const& child_indices = find_child_field_indices(schema, parent_idx);
  return make_empty_struct_column_from_children(schema, child_indices, stream, mr);
}

void maybe_check_required_fields(required_field_input_view input,
                                 std::vector<int> const& field_indices,
                                 std::vector<nested_field_descriptor> const& schema,
                                 protobuf_decode_runtime_context decode_ctx,
                                 cuda::stream_ref stream);

void validate_enum_values(rmm::device_uvector<int32_t> const& values,
                          rmm::device_uvector<bool>& valid,
                          enum_domain_device_view enum_domain,
                          cuda::stream_ref stream);

void validate_enum_values(rmm::device_uvector<int32_t> const& values,
                          rmm::device_uvector<bool>& valid,
                          cudf::detail::host_vector<int32_t> const& valid_enums,
                          cuda::stream_ref stream);

// ============================================================================
// Forward declarations of builder/utility functions
// ============================================================================

std::unique_ptr<cudf::column> make_null_column(cudf::data_type dtype,
                                               cudf::size_type num_rows,
                                               cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr);

// Schema-aware all-null builder: recurses into STRUCT children and wraps repeated fields
// in a null-list, mirroring the shape `make_empty_struct_column_with_schema` would produce
// but with `num_rows` all-null rows.
std::unique_ptr<cudf::column> make_null_column_with_schema(protobuf_schema const& schema,
                                                           int schema_idx,
                                                           cudf::size_type num_rows,
                                                           cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> make_null_list_column_with_child(
  std::unique_ptr<cudf::column> child_col,
  cudf::size_type num_rows,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> build_enum_string_column(rmm::device_uvector<int32_t>& enum_values,
                                                       rmm::device_uvector<bool>& valid,
                                                       protobuf_field_decode_request request,
                                                       cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> drop_unknown_repeated_enum_values(std::unique_ptr<cudf::column> input,
                                                                cuda::stream_ref stream,
                                                                rmm::device_async_resource_ref mr);

// Wrap offsets + child into a LIST column, propagating the input's null mask. Note: when
// `binary_input` has no nulls, `mr` is effectively unused — only the with-nulls path
// allocates against it (via `cudf::copy_bitmask`).
std::unique_ptr<cudf::column> make_list_column_with_input_nulls(
  int num_rows,
  std::unique_ptr<cudf::column> offsets_col,
  std::unique_ptr<cudf::column> child_col,
  cudf::column_view const& binary_input,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> build_repeated_enum_string_column(
  cudf::column_view const& binary_input,
  protobuf_input_view input,
  protobuf_schema const& schema,
  protobuf_decode_runtime_context decode_ctx,
  repeated_field_work work,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> build_repeated_string_column(
  cudf::column_view const& binary_input,
  protobuf_input_view input,
  protobuf_field_meta_view field,
  repeated_field_work work,
  rmm::device_uvector<protobuf_error>& d_error,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> build_nested_struct_column(
  protobuf_input_view input,
  nested_parent_view parent,
  std::vector<int> const& child_field_indices,
  recursive_decode_context context,
  int depth,
  bool materialize_output,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> build_merged_singular_struct_column(
  protobuf_input_view input,
  message_fragment_source_view source,
  std::vector<int> const& child_field_indices,
  recursive_decode_context context,
  singular_message_merge_buffers buffers,
  int depth,
  bool materialize_output,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> build_repeated_child_list_column(protobuf_input_view input,
                                                               nested_parent_view parent,
                                                               recursive_decode_context context,
                                                               repeated_field_work work,
                                                               bool materialize_output,
                                                               cuda::stream_ref stream,
                                                               rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> build_repeated_struct_column(
  cudf::column_view const& binary_input,
  protobuf_input_view input,
  std::vector<int> const& child_field_indices,
  recursive_decode_context context,
  repeated_field_work work,
  bool materialize_output,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace spark_rapids_jni::protobuf::detail
