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

#include <algorithm>
#include <optional>
#include <string>

namespace spark_rapids_jni::protobuf::detail {

namespace {

// Compute LIST<...> per-row offsets from a per-row count array. Produces a
// `device_uvector<int32_t>` of size `num_rows + 1` with `offsets[num_rows] = total_count`.
// `mr` should be the output memory resource since the returned offsets feed into the LIST column.
inline rmm::device_uvector<int32_t> make_list_offsets_from_counts(
  rmm::device_uvector<int32_t> const& counts,
  int total_count,
  int num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(num_rows >= 0, "make_list_offsets_from_counts: row count must be non-negative");
  CUDF_EXPECTS(total_count >= 0, "make_list_offsets_from_counts: total count must be non-negative");
  CUDF_EXPECTS(counts.size() == static_cast<size_t>(num_rows),
               "make_list_offsets_from_counts: counts size must match row count");

  rmm::device_uvector<int32_t> offsets(num_rows + 1, stream, mr);
  thrust::exclusive_scan(
    rmm::exec_policy_nosync(stream, mr), counts.begin(), counts.end(), offsets.begin(), 0);
  thrust::fill_n(rmm::exec_policy_nosync(stream, mr),
                 offsets.data() + num_rows,
                 1,
                 static_cast<int32_t>(total_count));
  return offsets;
}

// Construct the singleton ([1]-element) device schema and indices that
// `launch_count_repeated_in_nested` / `launch_scan_repeated_in_nested` consume when
// `build_repeated_child_list_column` is processing a single repeated child field.
// `parent_idx=-1` and `depth=0` are intentional: the kernel walks one local nested level only.
inline std::pair<rmm::device_uvector<device_nested_field_descriptor>, rmm::device_uvector<int>>
make_single_repeated_schema(int child_schema_idx,
                            std::vector<nested_field_descriptor> const& schema,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  device_nested_field_descriptor rep_desc;
  rep_desc.field_number      = schema[child_schema_idx].field_number;
  rep_desc.wire_type         = static_cast<int>(schema[child_schema_idx].wire_type);
  rep_desc.output_type_id    = static_cast<int>(schema[child_schema_idx].output_type);
  rep_desc.is_repeated       = true;
  rep_desc.parent_idx        = -1;
  rep_desc.depth             = 0;
  rep_desc.encoding          = 0;
  rep_desc.is_required       = false;
  rep_desc.has_default_value = false;

  rmm::device_uvector<device_nested_field_descriptor> d_schema(1, stream, mr);
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(
    d_schema.data(), &rep_desc, sizeof(device_nested_field_descriptor), stream));

  int const zero = 0;
  rmm::device_uvector<int> d_indices(1, stream, mr);
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(d_indices.data(), &zero, sizeof(int), stream));
  return {std::move(d_schema), std::move(d_indices)};
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

std::unique_ptr<cudf::column> build_repeated_msg_child_varlen_column(
  uint8_t const* message_data,
  rmm::device_uvector<cudf::size_type> const& d_msg_row_offsets,
  rmm::device_uvector<field_location> const& d_msg_locs,
  rmm::device_uvector<field_location> const& d_child_locs,
  int child_idx,
  int num_child_fields,
  int total_count,
  rmm::device_uvector<protobuf_error>& d_error,
  bool as_bytes,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (total_count == 0) {
    if (as_bytes) return make_empty_column_safe(cudf::data_type{cudf::type_id::LIST}, stream, mr);
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  }

  auto const scratch_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<int32_t> d_lengths(total_count, stream, scratch_mr);
  thrust::transform(
    rmm::exec_policy_nosync(stream, scratch_mr),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(total_count),
    d_lengths.data(),
    [child_locs = d_child_locs.data(), ci = child_idx, ncf = num_child_fields] __device__(int idx) {
      auto const& loc = child_locs[flat_index(
        static_cast<size_t>(idx), static_cast<size_t>(ncf), static_cast<size_t>(ci))];
      return loc.offset >= 0 ? loc.length : 0;
    });

  auto [offsets_col, total_data] = cudf::strings::detail::make_offsets_child_column(
    d_lengths.begin(), d_lengths.end(), stream, mr);

  rmm::device_uvector<char> d_data(total_data, stream, mr);
  rmm::device_uvector<bool> d_valid((total_count > 0 ? total_count : 1), stream, scratch_mr);

  thrust::transform(
    rmm::exec_policy_nosync(stream, scratch_mr),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(total_count),
    d_valid.data(),
    [child_locs = d_child_locs.data(), ci = child_idx, ncf = num_child_fields] __device__(int idx) {
      return child_locs[flat_index(static_cast<size_t>(idx),
                                   static_cast<size_t>(ncf),
                                   static_cast<size_t>(ci))]
               .offset >= 0;
    });

  if (total_data > 0) {
    repeated_msg_child_location_provider loc_provider{d_msg_row_offsets.data(),
                                                      0,
                                                      d_msg_locs.data(),
                                                      d_child_locs.data(),
                                                      child_idx,
                                                      num_child_fields};
    auto const* offsets_data = offsets_col->view().data<cudf::size_type>();
    auto* chars_ptr          = d_data.data();

    auto src_iter = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<void const*>(
        [message_data, loc_provider] __device__(int idx) -> void const* {
          int32_t data_offset = 0;
          auto loc            = loc_provider.get(idx, data_offset);
          if (loc.offset < 0) return nullptr;
          return static_cast<void const*>(message_data + data_offset);
        }));
    auto dst_iter = cudf::detail::make_counting_transform_iterator(
      0, cuda::proclaim_return_type<void*>([chars_ptr, offsets_data] __device__(int idx) -> void* {
        return static_cast<void*>(chars_ptr + offsets_data[idx]);
      }));
    auto size_iter = cudf::detail::make_counting_transform_iterator(
      0, cuda::proclaim_return_type<size_t>([loc_provider] __device__(int idx) -> size_t {
        int32_t data_offset = 0;
        auto loc            = loc_provider.get(idx, data_offset);
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

  auto [mask, null_count] = make_null_mask_from_valid(d_valid, total_count, stream, mr);

  if (as_bytes) {
    auto bytes_child =
      std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT8},
                                     total_data,
                                     rmm::device_buffer(d_data.data(), total_data, stream, mr),
                                     rmm::device_buffer{},
                                     0);
    return cudf::make_lists_column(
      total_count, std::move(offsets_col), std::move(bytes_child), null_count, std::move(mask));
  }

  return cudf::make_strings_column(
    total_count, std::move(offsets_col), d_data.release(), null_count, std::move(mask));
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

// Forward declaration of the public builder (defined further down).
enum_string_lookup_tables make_enum_string_lookup_tables(
  cudf::detail::host_vector<int32_t> const& valid_enums,
  std::vector<cudf::detail::host_vector<uint8_t>> const& enum_name_bytes,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

namespace {

// Returns a const-ref to the lookup table for `schema_idx`, populating `ctx.enum_lookup_cache`
// on first use. Falls back to building (without caching) when the cache is null.
enum_string_lookup_tables const* get_enum_lookup(
  schema_context_view const& ctx,
  int schema_idx,
  cudf::detail::host_vector<int32_t> const& valid_enums,
  std::vector<cudf::detail::host_vector<uint8_t>> const& enum_name_bytes,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::optional<enum_string_lookup_tables>& fallback)
{
  if (ctx.enum_lookup_cache != nullptr) {
    auto& cache = *ctx.enum_lookup_cache;
    auto it     = cache.find(schema_idx);
    if (it == cache.end()) {
      it = cache
             .emplace(schema_idx,
                      make_enum_string_lookup_tables(valid_enums, enum_name_bytes, stream, mr))
             .first;
    }
    return &it->second;
  }
  fallback.emplace(make_enum_string_lookup_tables(valid_enums, enum_name_bytes, stream, mr));
  return &*fallback;
}

}  // namespace

enum_string_lookup_tables make_enum_string_lookup_tables(
  cudf::detail::host_vector<int32_t> const& valid_enums,
  std::vector<cudf::detail::host_vector<uint8_t>> const& enum_name_bytes,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto d_valid_enums = cudf::detail::make_device_uvector_async(
    valid_enums, stream, cudf::get_current_device_resource_ref());

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
  rmm::device_uvector<bool>& d_row_force_null,
  int num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int32_t const* top_row_indices,
  bool propagate_invalid_enum_rows)
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
                                       d_row_force_null,
                                       num_rows,
                                       top_row_indices,
                                       propagate_invalid_enum_rows,
                                       stream);
  return build_enum_string_values_column(enum_values, valid, lookup, num_rows, stream, mr);
}

std::unique_ptr<cudf::column> build_repeated_msg_child_enum_string_column(
  uint8_t const* message_data,
  rmm::device_uvector<cudf::size_type> const& d_msg_row_offsets,
  rmm::device_uvector<field_location> const& d_msg_locs,
  rmm::device_uvector<field_location> const& d_child_locs,
  int child_idx,
  int num_child_fields,
  int total_count,
  cudf::detail::host_vector<int32_t> const& valid_enums,
  std::vector<cudf::detail::host_vector<uint8_t>> const& enum_name_bytes,
  rmm::device_uvector<bool>& d_row_force_null,
  int32_t const* top_row_indices,
  bool propagate_invalid_enum_rows,
  rmm::device_uvector<protobuf_error>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const threads    = THREADS_PER_BLOCK;
  auto const blocks     = static_cast<int>((total_count + threads - 1u) / threads);
  auto const scratch_mr = cudf::get_current_device_resource_ref();
  auto lookup           = make_enum_string_lookup_tables(valid_enums, enum_name_bytes, stream, mr);

  rmm::device_uvector<int32_t> enum_values(total_count, stream, scratch_mr);
  rmm::device_uvector<bool> valid(total_count, stream, scratch_mr);
  repeated_msg_child_location_provider loc_provider{d_msg_row_offsets.data(),
                                                    0,
                                                    d_msg_locs.data(),
                                                    d_child_locs.data(),
                                                    child_idx,
                                                    num_child_fields};
  extract_varint_kernel<int32_t, false, repeated_msg_child_location_provider>
    <<<blocks, threads, 0, stream.value()>>>(message_data,
                                             loc_provider,
                                             total_count,
                                             enum_values.data(),
                                             valid.data(),
                                             d_error.data(),
                                             false,
                                             0);

  rmm::device_uvector<bool> d_elem_has_invalid_enum(total_count, stream, scratch_mr);
  thrust::fill(rmm::exec_policy_nosync(stream, scratch_mr),
               d_elem_has_invalid_enum.begin(),
               d_elem_has_invalid_enum.end(),
               false);
  launch_validate_enum_values(enum_values.data(),
                              valid.data(),
                              d_elem_has_invalid_enum.data(),
                              lookup.d_valid_enums.data(),
                              static_cast<int>(valid_enums.size()),
                              total_count,
                              stream);
  propagate_invalid_enum_flags_to_rows(d_elem_has_invalid_enum,
                                       d_row_force_null,
                                       total_count,
                                       top_row_indices,
                                       propagate_invalid_enum_rows,
                                       stream);
  return build_enum_string_values_column(enum_values, valid, lookup, total_count, stream, mr);
}

std::unique_ptr<cudf::column> build_repeated_enum_string_column(
  cudf::column_view const& binary_input,
  uint8_t const* message_data,
  cudf::size_type const* list_offsets,
  cudf::size_type base_offset,
  rmm::device_uvector<int32_t> d_field_offsets,
  rmm::device_uvector<repeated_occurrence>& d_occurrences,
  int total_count,
  int num_rows,
  cudf::detail::host_vector<int32_t> const& valid_enums,
  std::vector<cudf::detail::host_vector<uint8_t>> const& enum_name_bytes,
  rmm::device_uvector<bool>& d_row_force_null,
  rmm::device_uvector<protobuf_error>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const rep_blocks =
    static_cast<int>((total_count + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  auto const scratch_mr = cudf::get_current_device_resource_ref();
  auto lookup           = make_enum_string_lookup_tables(valid_enums, enum_name_bytes, stream, mr);

  // 1. Extract enum integer values from occurrences
  rmm::device_uvector<int32_t> enum_ints(total_count, stream, scratch_mr);
  rmm::device_uvector<bool> elem_valid(total_count, stream, scratch_mr);
  repeated_location_provider rep_loc{list_offsets, base_offset, d_occurrences.data()};
  extract_varint_kernel<int32_t, false>
    <<<rep_blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(message_data,
                                                           rep_loc,
                                                           total_count,
                                                           enum_ints.data(),
                                                           elem_valid.data(),
                                                           d_error.data(),
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
  propagate_invalid_enum_flags_to_rows(d_elem_has_invalid_enum,
                                       d_row_force_null,
                                       total_count,
                                       d_top_row_indices.data(),
                                       d_row_force_null.size() > 0,
                                       stream);

  auto child_col =
    build_enum_string_values_column(enum_ints, elem_valid, lookup, total_count, stream, mr);

  auto list_offs_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                      num_rows + 1,
                                                      d_field_offsets.release(),
                                                      rmm::device_buffer{},
                                                      0);

  return make_list_column_with_input_nulls(
    num_rows, std::move(list_offs_col), std::move(child_col), binary_input, stream, mr);
}

std::unique_ptr<cudf::column> build_repeated_string_column(
  cudf::column_view const& binary_input,
  uint8_t const* message_data,
  cudf::size_type const* list_offsets,
  cudf::size_type base_offset,
  rmm::device_uvector<int32_t> d_field_offsets,
  rmm::device_uvector<repeated_occurrence>& d_occurrences,
  int total_count,
  int num_rows,
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
  repeated_location_provider loc_provider{list_offsets, base_offset, d_occurrences.data()};
  extract_lengths_kernel<repeated_location_provider>
    <<<blocks, threads, 0, stream.value()>>>(loc_provider, total_count, str_lengths.data());

  auto [str_offsets_col, total_chars] = cudf::strings::detail::make_offsets_child_column(
    str_lengths.begin(), str_lengths.end(), stream, mr);

  rmm::device_uvector<char> chars(total_chars, stream, mr);
  if (total_chars > 0) {
    repeated_location_provider copy_provider{list_offsets, base_offset, d_occurrences.data()};
    auto const* offsets_data = str_offsets_col->view().data<cudf::size_type>();
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
    auto bytes_child =
      std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT8},
                                     total_chars,
                                     rmm::device_buffer(chars.data(), total_chars, stream, mr),
                                     rmm::device_buffer{},
                                     0);
    child_col = cudf::make_lists_column(
      total_count, std::move(str_offsets_col), std::move(bytes_child), 0, rmm::device_buffer{});
  } else {
    child_col = cudf::make_strings_column(
      total_count, std::move(str_offsets_col), chars.release(), 0, rmm::device_buffer{});
  }

  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    num_rows + 1,
                                                    d_field_offsets.release(),
                                                    rmm::device_buffer{},
                                                    0);

  // Only rows where INPUT is null should produce null output;
  // rows with valid input but count=0 produce empty array [].
  return make_list_column_with_input_nulls(
    num_rows, std::move(offsets_col), std::move(child_col), binary_input, stream, mr);
}

// ============================================================================
// Nested struct column builder
// ============================================================================

/**
 * Build a STRUCT column for a nested protobuf message.
 *
 * 3b.2 scope: only scalar numeric/bool children are fully decoded. STRING/LIST<UINT8>/STRUCT
 * children, repeated children, and proto2 required-field checks land in subsequent PRs
 * (3b.3 / 3b.4 / 3b.5 / 3b.6); for now those child slots are filled with typed null columns
 * so the output schema is still well-formed.
 */
std::unique_ptr<cudf::column> build_nested_struct_column(
  uint8_t const* message_data,
  cudf::size_type message_data_size,
  cudf::size_type const* list_offsets,
  cudf::size_type base_offset,
  rmm::device_uvector<field_location> const& d_parent_locs,
  std::vector<int> const& child_field_indices,
  std::vector<nested_field_descriptor> const& schema,
  int num_fields,
  schema_context_view ctx,
  rmm::device_uvector<bool>& d_row_force_null,
  rmm::device_uvector<protobuf_error>& d_error,
  int num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int32_t const* top_row_indices,
  int depth,
  bool propagate_invalid_enum_rows);

// Forward declaration -- build_repeated_child_list_column is defined after
// build_nested_struct_column but both build_repeated_struct_column and build_nested_struct_column
// need to call it.
std::unique_ptr<cudf::column> build_repeated_child_list_column(
  uint8_t const* message_data,
  cudf::size_type message_data_size,
  cudf::size_type const* row_offsets,
  cudf::size_type base_offset,
  field_location const* parent_locs,
  int num_parent_rows,
  int child_schema_idx,
  std::vector<nested_field_descriptor> const& schema,
  int num_fields,
  schema_context_view ctx,
  rmm::device_uvector<bool>& d_row_force_null,
  rmm::device_uvector<protobuf_error>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int32_t const* top_row_indices,
  int depth,
  bool propagate_invalid_enum_rows);

std::unique_ptr<cudf::column> build_repeated_struct_column(
  cudf::column_view const& binary_input,
  uint8_t const* message_data,
  cudf::size_type message_data_size,
  cudf::size_type const* list_offsets,
  cudf::size_type base_offset,
  rmm::device_uvector<int32_t> d_field_offsets,
  rmm::device_uvector<repeated_occurrence>& d_occurrences,
  int total_count,
  int num_rows,
  std::vector<device_nested_field_descriptor> const& h_device_schema,
  std::vector<int> const& child_field_indices,
  std::vector<nested_field_descriptor> const& schema,
  schema_context_view ctx,
  rmm::device_uvector<bool>& d_row_force_null,
  rmm::device_uvector<protobuf_error>& d_error_top,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const& default_ints      = ctx.default_ints;
  auto const& default_floats    = ctx.default_floats;
  auto const& default_bools     = ctx.default_bools;
  auto const& default_strings   = ctx.default_strings;
  auto const& enum_valid_values = ctx.enum_valid_values;
  auto const& enum_names        = ctx.enum_names;
  int num_child_fields          = static_cast<int>(child_field_indices.size());

  CUDF_EXPECTS(total_count > 0, "build_repeated_struct_column: total_count must be > 0");

  if (num_child_fields == 0) {
    auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                      num_rows + 1,
                                                      d_field_offsets.release(),
                                                      rmm::device_buffer{},
                                                      0);

    std::vector<std::unique_ptr<cudf::column>> empty_struct_children;
    auto empty_struct = cudf::make_structs_column(
      total_count, std::move(empty_struct_children), 0, rmm::device_buffer{}, stream, mr);

    return make_list_column_with_input_nulls(
      num_rows, std::move(offsets_col), std::move(empty_struct), binary_input, stream, mr);
  }

  // Build child field descriptors for scanning within each message occurrence.
  // Stage through pinned memory so the H2D is truly stream-async.
  auto h_child_descs =
    cudf::detail::make_pinned_vector_async<field_descriptor>(num_child_fields, stream);
  for (int ci = 0; ci < num_child_fields; ci++) {
    int child_schema_idx                 = child_field_indices[ci];
    h_child_descs[ci].field_number       = h_device_schema[child_schema_idx].field_number;
    h_child_descs[ci].expected_wire_type = h_device_schema[child_schema_idx].wire_type;
    h_child_descs[ci].is_repeated        = h_device_schema[child_schema_idx].is_repeated;
  }
  auto const scratch_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<field_descriptor> d_child_descs(num_child_fields, stream, scratch_mr);
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(d_child_descs.data(),
                                           h_child_descs.data(),
                                           num_child_fields * sizeof(field_descriptor),
                                           stream));
  auto h_lookup_vec = build_field_lookup_table(h_child_descs.data(), num_child_fields, stream);
  rmm::device_uvector<int> d_child_lookup(0, stream, scratch_mr);
  if (!h_lookup_vec.empty()) {
    auto h_child_lookup = cudf::detail::make_pinned_vector_async<int>(h_lookup_vec.size(), stream);
    std::copy(h_lookup_vec.begin(), h_lookup_vec.end(), h_child_lookup.begin());
    d_child_lookup = rmm::device_uvector<int>(h_child_lookup.size(), stream, scratch_mr);
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(
      d_child_lookup.data(), h_child_lookup.data(), h_child_lookup.size() * sizeof(int), stream));
  }

  // For each occurrence, we need to scan for child fields
  // Create "virtual" parent locations from the occurrences using GPU kernel
  // This replaces the host-side loop with D->H->D copy pattern (critical performance fix!)
  rmm::device_uvector<field_location> d_msg_locs(total_count, stream, scratch_mr);
  rmm::device_uvector<cudf::size_type> d_msg_row_offsets(total_count, stream, scratch_mr);
  launch_compute_msg_locations_from_occurrences(d_occurrences.data(),
                                                list_offsets,
                                                base_offset,
                                                d_msg_locs.data(),
                                                d_msg_row_offsets.data(),
                                                total_count,
                                                d_error_top.data(),
                                                stream);
  rmm::device_uvector<int32_t> d_top_row_indices(total_count, stream, scratch_mr);
  thrust::transform(rmm::exec_policy_nosync(stream, scratch_mr),
                    d_occurrences.data(),
                    d_occurrences.end(),
                    d_top_row_indices.data(),
                    [] __device__(repeated_occurrence const& occ) { return occ.row_idx; });

  // Scan for child fields within each message occurrence
  rmm::device_uvector<field_location> d_child_locs(
    static_cast<size_t>(total_count) * num_child_fields, stream, scratch_mr);
  // Reuse top-level error flag so failfast can observe nested repeated-message failures.
  auto& d_error = d_error_top;

  auto const threads = THREADS_PER_BLOCK;
  auto const blocks  = static_cast<int>((total_count + threads - 1u) / threads);

  // Use a custom kernel to scan child fields within message occurrences
  // This is similar to scan_nested_message_fields_kernel but operates on occurrences
  launch_scan_repeated_message_children(message_data,
                                        message_data_size,
                                        d_msg_row_offsets.data(),
                                        d_msg_locs.data(),
                                        total_count,
                                        d_child_descs.data(),
                                        num_child_fields,
                                        d_child_locs.data(),
                                        d_error.data(),
                                        h_lookup_vec.empty() ? nullptr : d_child_lookup.data(),
                                        static_cast<int>(d_child_lookup.size()),
                                        stream);

  // Enforce proto2 required semantics for fields inside each repeated message occurrence.
  maybe_check_required_fields(d_child_locs.data(),
                              child_field_indices,
                              schema,
                              total_count,
                              nullptr,
                              0,
                              nullptr,
                              d_row_force_null.size() > 0 ? d_row_force_null.data() : nullptr,
                              d_top_row_indices.data(),
                              d_error.data(),
                              stream);

  // Note: We no longer need to copy child_locs to host because:
  // 1. All scalar extraction kernels access d_child_locs directly on device
  // 2. String extraction uses GPU kernels
  // 3. Nested struct locations are computed on GPU via compute_nested_struct_locations_kernel

  // Extract child field values - build one column per child field
  std::vector<std::unique_ptr<cudf::column>> struct_children;
  int num_schema_fields = static_cast<int>(h_device_schema.size());
  for (int ci = 0; ci < num_child_fields; ci++) {
    int child_schema_idx   = child_field_indices[ci];
    auto const dt          = cudf::data_type{schema[child_schema_idx].output_type};
    auto const enc         = h_device_schema[child_schema_idx].encoding;
    bool has_def           = h_device_schema[child_schema_idx].has_default_value;
    bool child_is_repeated = h_device_schema[child_schema_idx].is_repeated;

    if (child_is_repeated) {
      struct_children.push_back(build_repeated_child_list_column(message_data,
                                                                 message_data_size,
                                                                 d_msg_row_offsets.data(),
                                                                 0,
                                                                 d_msg_locs.data(),
                                                                 total_count,
                                                                 child_schema_idx,
                                                                 schema,
                                                                 num_schema_fields,
                                                                 ctx,
                                                                 d_row_force_null,
                                                                 d_error_top,
                                                                 stream,
                                                                 mr,
                                                                 d_top_row_indices.data(),
                                                                 1,
                                                                 false));
      continue;
    }

    switch (dt.id()) {
      case cudf::type_id::BOOL8:
      case cudf::type_id::INT32:
      case cudf::type_id::UINT32:
      case cudf::type_id::INT64:
      case cudf::type_id::UINT64:
      case cudf::type_id::FLOAT32:
      case cudf::type_id::FLOAT64: {
        repeated_msg_child_location_provider loc_provider{d_msg_row_offsets.data(),
                                                          0,
                                                          d_msg_locs.data(),
                                                          d_child_locs.data(),
                                                          ci,
                                                          num_child_fields};
        struct_children.push_back(extract_typed_column(dt,
                                                       enc,
                                                       message_data,
                                                       loc_provider,
                                                       total_count,
                                                       blocks,
                                                       threads,
                                                       has_def,
                                                       default_ints[child_schema_idx],
                                                       default_floats[child_schema_idx],
                                                       default_bools[child_schema_idx],
                                                       default_strings[child_schema_idx],
                                                       child_schema_idx,
                                                       enum_valid_values,
                                                       enum_names,
                                                       d_row_force_null,
                                                       d_error,
                                                       stream,
                                                       mr,
                                                       d_top_row_indices.data(),
                                                       false));
        break;
      }
      case cudf::type_id::STRING: {
        if (enc == encoding_value(proto_encoding::ENUM_STRING)) {
          struct_children.push_back(
            build_repeated_msg_child_enum_string_column(message_data,
                                                        d_msg_row_offsets,
                                                        d_msg_locs,
                                                        d_child_locs,
                                                        ci,
                                                        num_child_fields,
                                                        total_count,
                                                        enum_valid_values[child_schema_idx],
                                                        enum_names[child_schema_idx],
                                                        d_row_force_null,
                                                        d_top_row_indices.data(),
                                                        false,
                                                        d_error,
                                                        stream,
                                                        mr));
        } else {
          struct_children.push_back(build_repeated_msg_child_varlen_column(message_data,
                                                                           d_msg_row_offsets,
                                                                           d_msg_locs,
                                                                           d_child_locs,
                                                                           ci,
                                                                           num_child_fields,
                                                                           total_count,
                                                                           d_error,
                                                                           false,
                                                                           stream,
                                                                           mr));
        }
        break;
      }
      case cudf::type_id::LIST: {
        struct_children.push_back(build_repeated_msg_child_varlen_column(message_data,
                                                                         d_msg_row_offsets,
                                                                         d_msg_locs,
                                                                         d_child_locs,
                                                                         ci,
                                                                         num_child_fields,
                                                                         total_count,
                                                                         d_error,
                                                                         true,
                                                                         stream,
                                                                         mr));
        break;
      }
      case cudf::type_id::STRUCT: {
        // Nested struct inside repeated message - use recursive build_nested_struct_column
        int num_schema_fields = static_cast<int>(h_device_schema.size());
        auto grandchild_indices =
          find_child_field_indices(h_device_schema, num_schema_fields, child_schema_idx);

        if (grandchild_indices.empty()) {
          struct_children.push_back(
            cudf::make_structs_column(total_count,
                                      std::vector<std::unique_ptr<cudf::column>>{},
                                      0,
                                      rmm::device_buffer{},
                                      stream,
                                      mr));
        } else {
          // Compute virtual parent locations for each occurrence's nested struct child
          rmm::device_uvector<field_location> d_nested_locs(total_count, stream, scratch_mr);
          rmm::device_uvector<cudf::size_type> d_nested_row_offsets(
            total_count, stream, scratch_mr);
          launch_compute_nested_struct_locations(d_child_locs.data(),
                                                 d_msg_locs.data(),
                                                 d_msg_row_offsets.data(),
                                                 ci,
                                                 num_child_fields,
                                                 d_nested_locs.data(),
                                                 d_nested_row_offsets.data(),
                                                 total_count,
                                                 d_error_top.data(),
                                                 stream);

          struct_children.push_back(build_nested_struct_column(message_data,
                                                               message_data_size,
                                                               d_nested_row_offsets.data(),
                                                               0,
                                                               d_nested_locs,
                                                               grandchild_indices,
                                                               schema,
                                                               num_schema_fields,
                                                               ctx,
                                                               d_row_force_null,
                                                               d_error_top,
                                                               total_count,
                                                               stream,
                                                               mr,
                                                               d_top_row_indices.data(),
                                                               0,
                                                               false));
        }
        break;
      }
      default:
        CUDF_FAIL("Protobuf decode internal error: unsupported repeated message child type id=" +
                  std::to_string(static_cast<int>(dt.id())));
    }
  }

  // Build the struct column from child columns
  auto struct_col = cudf::make_structs_column(
    total_count, std::move(struct_children), 0, rmm::device_buffer{}, stream, mr);

  // Build the list offsets column
  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    num_rows + 1,
                                                    d_field_offsets.release(),
                                                    rmm::device_buffer{},
                                                    0);

  // Build the final LIST column
  return make_list_column_with_input_nulls(
    num_rows, std::move(offsets_col), std::move(struct_col), binary_input, stream, mr);
}

std::unique_ptr<cudf::column> build_nested_struct_column(
  uint8_t const* message_data,
  cudf::size_type message_data_size,
  cudf::size_type const* list_offsets,
  cudf::size_type base_offset,
  rmm::device_uvector<field_location> const& d_parent_locs,
  std::vector<int> const& child_field_indices,
  std::vector<nested_field_descriptor> const& schema,
  int num_fields,
  schema_context_view ctx,
  rmm::device_uvector<bool>& d_row_force_null,
  rmm::device_uvector<protobuf_error>& d_error,
  int num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int32_t const* top_row_indices,
  int depth,
  bool propagate_invalid_enum_rows)
{
  auto const& default_ints      = ctx.default_ints;
  auto const& default_floats    = ctx.default_floats;
  auto const& default_bools     = ctx.default_bools;
  auto const& default_strings   = ctx.default_strings;
  auto const& enum_valid_values = ctx.enum_valid_values;
  auto const& enum_names        = ctx.enum_names;
  CUDF_EXPECTS(depth < MAX_NESTING_DEPTH,
               "Nested protobuf struct depth exceeds supported decode recursion limit");
  CUDF_EXPECTS(d_parent_locs.size() == static_cast<size_t>(num_rows),
               "build_nested_struct_column: parent locations size must match row count");
  // d_row_force_null is local-row-aligned only when no top-row remap is supplied.
  CUDF_EXPECTS(
    d_row_force_null.is_empty() || top_row_indices != nullptr ||
      d_row_force_null.size() == static_cast<size_t>(num_rows),
    "build_nested_struct_column: row-force-null buffer must be empty, row-sized, or remapped");

  if (num_rows == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty_children;
    for (int child_schema_idx : child_field_indices) {
      auto child_type = cudf::data_type{schema[child_schema_idx].output_type};
      std::unique_ptr<cudf::column> child_col;
      if (child_type.id() == cudf::type_id::STRUCT) {
        child_col =
          make_empty_struct_column_with_schema(schema, child_schema_idx, num_fields, stream, mr);
      } else {
        child_col = make_empty_column_safe(child_type, stream, mr);
      }
      if (schema[child_schema_idx].is_repeated) {
        child_col = make_empty_list_column(std::move(child_col), stream, mr);
      }
      empty_children.push_back(std::move(child_col));
    }
    return cudf::make_structs_column(
      0, std::move(empty_children), 0, rmm::device_buffer{}, stream, mr);
  }

  auto const threads   = THREADS_PER_BLOCK;
  auto const blocks    = static_cast<int>((num_rows + threads - 1u) / threads);
  int num_child_fields = static_cast<int>(child_field_indices.size());

  // Stage child field descriptors through pinned memory so the H2D stays stream-async.
  auto h_child_field_descs =
    cudf::detail::make_pinned_vector_async<field_descriptor>(num_child_fields, stream);
  for (int i = 0; i < num_child_fields; i++) {
    int child_idx                             = child_field_indices[i];
    h_child_field_descs[i].field_number       = schema[child_idx].field_number;
    h_child_field_descs[i].expected_wire_type = static_cast<int>(schema[child_idx].wire_type);
    h_child_field_descs[i].is_repeated        = schema[child_idx].is_repeated;
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

  auto const child_location_count = static_cast<size_t>(num_rows) * num_child_fields;
  rmm::device_uvector<field_location> d_child_locations(
    std::max(child_location_count, size_t{1}), stream, scratch_mr);
  // Scan over every child descriptor (including repeated ones) so STRICT-mode wire-type
  // validation still runs on repeated occurrences. Repeated child slots in d_child_locations
  // are not consumed; 3b.5 / 3b.6 produce them via dedicated count/scan kernels.
  launch_scan_nested_message_fields(message_data,
                                    message_data_size,
                                    list_offsets,
                                    base_offset,
                                    d_parent_locs.data(),
                                    num_rows,
                                    d_child_field_descs.data(),
                                    num_child_fields,
                                    d_child_locations.data(),
                                    d_error.data(),
                                    d_row_force_null.size() > 0 ? d_row_force_null.data() : nullptr,
                                    top_row_indices,
                                    stream);

  // Enforce proto2 required semantics for direct children of this nested message.
  maybe_check_required_fields(d_child_locations.data(),
                              child_field_indices,
                              schema,
                              num_rows,
                              nullptr,
                              0,
                              d_parent_locs.data(),
                              d_row_force_null.size() > 0 ? d_row_force_null.data() : nullptr,
                              top_row_indices,
                              d_error.data(),
                              stream);

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  for (int ci = 0; ci < num_child_fields; ci++) {
    int child_schema_idx = child_field_indices[ci];
    auto const dt        = cudf::data_type{schema[child_schema_idx].output_type};
    auto const enc       = static_cast<int>(schema[child_schema_idx].encoding);
    bool has_def         = schema[child_schema_idx].has_default_value;
    bool is_repeated     = schema[child_schema_idx].is_repeated;

    if (is_repeated) {
      struct_children.push_back(build_repeated_child_list_column(message_data,
                                                                 message_data_size,
                                                                 list_offsets,
                                                                 base_offset,
                                                                 d_parent_locs.data(),
                                                                 num_rows,
                                                                 child_schema_idx,
                                                                 schema,
                                                                 num_fields,
                                                                 ctx,
                                                                 d_row_force_null,
                                                                 d_error,
                                                                 stream,
                                                                 mr,
                                                                 top_row_indices,
                                                                 depth,
                                                                 propagate_invalid_enum_rows));
      continue;
    }

    nested_location_provider loc_provider{list_offsets,
                                          base_offset,
                                          d_parent_locs.data(),
                                          d_child_locations.data(),
                                          ci,
                                          num_child_fields};

    switch (dt.id()) {
      case cudf::type_id::BOOL8:
      case cudf::type_id::INT32:
      case cudf::type_id::UINT32:
      case cudf::type_id::INT64:
      case cudf::type_id::UINT64:
      case cudf::type_id::FLOAT32:
      case cudf::type_id::FLOAT64: {
        struct_children.push_back(extract_typed_column(dt,
                                                       enc,
                                                       message_data,
                                                       loc_provider,
                                                       num_rows,
                                                       blocks,
                                                       threads,
                                                       has_def,
                                                       default_ints[child_schema_idx],
                                                       default_floats[child_schema_idx],
                                                       default_bools[child_schema_idx],
                                                       default_strings[child_schema_idx],
                                                       child_schema_idx,
                                                       enum_valid_values,
                                                       enum_names,
                                                       d_row_force_null,
                                                       d_error,
                                                       stream,
                                                       mr,
                                                       top_row_indices,
                                                       propagate_invalid_enum_rows));
        break;
      }
      case cudf::type_id::STRING:
      case cudf::type_id::LIST: {  // bytes represented as LIST<UINT8>
        bool const is_enum_string =
          dt.id() == cudf::type_id::STRING && enc == encoding_value(proto_encoding::ENUM_STRING);
        if (is_enum_string) {
          rmm::device_uvector<int32_t> out(num_rows, stream, mr);
          rmm::device_uvector<bool> valid(num_rows, stream, mr);
          extract_varint_kernel<int32_t, false, nested_location_provider>
            <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                     loc_provider,
                                                     num_rows,
                                                     out.data(),
                                                     valid.data(),
                                                     d_error.data(),
                                                     has_def,
                                                     default_ints[child_schema_idx]);

          struct_children.push_back(build_enum_string_column(out,
                                                             valid,
                                                             enum_valid_values[child_schema_idx],
                                                             enum_names[child_schema_idx],
                                                             d_row_force_null,
                                                             num_rows,
                                                             stream,
                                                             mr,
                                                             top_row_indices,
                                                             propagate_invalid_enum_rows));
        } else {
          auto valid_fn = [loc_provider, has_def] __device__(cudf::size_type row) {
            return has_def || loc_provider.valid(row);
          };
          struct_children.push_back(
            extract_and_build_string_or_bytes_column(dt.id() == cudf::type_id::LIST,
                                                     message_data,
                                                     num_rows,
                                                     loc_provider,
                                                     loc_provider,
                                                     valid_fn,
                                                     has_def,
                                                     default_strings[child_schema_idx],
                                                     d_error,
                                                     stream,
                                                     mr));
        }
        break;
      }
      case cudf::type_id::STRUCT: {
        // Recursive linear child lookup is fine for realistic schemas; precompute if it gets hot.
        auto gc_indices = find_child_field_indices(schema, num_fields, child_schema_idx);
        rmm::device_uvector<field_location> d_gc_parent(num_rows, stream, scratch_mr);
        launch_compute_grandchild_parent_locations(d_parent_locs.data(),
                                                   d_child_locations.data(),
                                                   ci,
                                                   num_child_fields,
                                                   d_gc_parent.data(),
                                                   num_rows,
                                                   d_error.data(),
                                                   stream);
        struct_children.push_back(build_nested_struct_column(message_data,
                                                             message_data_size,
                                                             list_offsets,
                                                             base_offset,
                                                             d_gc_parent,
                                                             gc_indices,
                                                             schema,
                                                             num_fields,
                                                             ctx,
                                                             d_row_force_null,
                                                             d_error,
                                                             num_rows,
                                                             stream,
                                                             mr,
                                                             top_row_indices,
                                                             depth + 1,
                                                             propagate_invalid_enum_rows));
        break;
      }
      default:
        CUDF_FAIL("Protobuf decode internal error: unsupported nested child type id=" +
                  std::to_string(static_cast<int>(dt.id())));
    }
  }

  rmm::device_uvector<bool> struct_valid((num_rows > 0 ? num_rows : 1), stream, scratch_mr);
  thrust::transform(
    rmm::exec_policy_nosync(stream, scratch_mr),
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(num_rows),
    struct_valid.data(),
    [plocs = d_parent_locs.data()] __device__(auto row) { return plocs[row].offset >= 0; });
  auto [struct_mask, struct_null_count] =
    make_null_mask_from_valid(struct_valid, num_rows, stream, mr);
  return cudf::make_structs_column(
    num_rows, std::move(struct_children), struct_null_count, std::move(struct_mask), stream, mr);
}

/**
 * Build a LIST column for a repeated child field inside a parent message.
 * Shared between build_nested_struct_column and build_repeated_struct_column.
 */
std::unique_ptr<cudf::column> build_repeated_child_list_column(
  uint8_t const* message_data,
  cudf::size_type message_data_size,
  cudf::size_type const* row_offsets,
  cudf::size_type base_offset,
  field_location const* parent_locs,
  int num_parent_rows,
  int child_schema_idx,
  std::vector<nested_field_descriptor> const& schema,
  int num_fields,
  schema_context_view ctx,
  rmm::device_uvector<bool>& d_row_force_null,
  rmm::device_uvector<protobuf_error>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int32_t const* top_row_indices,
  int depth,
  bool propagate_invalid_enum_rows)
{
  // This function only consumes enum metadata directly — defaults are forwarded through `ctx`
  // to the recursive `build_nested_struct_column` call.
  auto const& enum_valid_values = ctx.enum_valid_values;
  auto const& enum_names        = ctx.enum_names;
  auto elem_type_id             = schema[child_schema_idx].output_type;
  auto const scratch_mr         = cudf::get_current_device_resource_ref();
  rmm::device_uvector<repeated_field_info> d_rep_info(num_parent_rows, stream, scratch_mr);

  CUDF_EXPECTS(schema[child_schema_idx].is_repeated,
               "count_repeated_in_nested_kernel launch requires repeated child schema");
  auto [d_rep_schema, d_rep_indices] =
    make_single_repeated_schema(child_schema_idx, schema, stream, scratch_mr);

  launch_count_repeated_in_nested(message_data,
                                  message_data_size,
                                  row_offsets,
                                  base_offset,
                                  parent_locs,
                                  num_parent_rows,
                                  d_rep_schema.data(),
                                  1,
                                  d_rep_info.data(),
                                  1,
                                  d_rep_indices.data(),
                                  d_error.data(),
                                  stream);

  rmm::device_uvector<int32_t> d_rep_counts(num_parent_rows, stream, scratch_mr);
  thrust::transform(rmm::exec_policy_nosync(stream, scratch_mr),
                    d_rep_info.data(),
                    d_rep_info.end(),
                    d_rep_counts.data(),
                    [] __device__(repeated_field_info const& info) { return info.count; });
  int64_t total_rep_count_64 = thrust::reduce(rmm::exec_policy_nosync(stream, scratch_mr),
                                              d_rep_counts.data(),
                                              d_rep_counts.end(),
                                              int64_t{0});
  CUDF_EXPECTS(total_rep_count_64 <= std::numeric_limits<int32_t>::max(),
               "Repeated nested-field total element count exceeds 2^31-1");
  int const total_rep_count = static_cast<int>(total_rep_count_64);

  if (total_rep_count == 0) {
    rmm::device_uvector<int32_t> list_offsets_vec(num_parent_rows + 1, stream, mr);
    thrust::fill(
      rmm::exec_policy_nosync(stream, mr), list_offsets_vec.data(), list_offsets_vec.end(), 0);
    auto list_offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                           num_parent_rows + 1,
                                                           list_offsets_vec.release(),
                                                           rmm::device_buffer{},
                                                           0);
    std::unique_ptr<cudf::column> child_col;
    if (elem_type_id == cudf::type_id::STRUCT) {
      child_col =
        make_empty_struct_column_with_schema(schema, child_schema_idx, num_fields, stream, mr);
    } else {
      child_col = make_empty_column_safe(cudf::data_type{elem_type_id}, stream, mr);
    }
    return cudf::make_lists_column(
      num_parent_rows, std::move(list_offsets_col), std::move(child_col), 0, rmm::device_buffer{});
  }

  auto list_offs =
    make_list_offsets_from_counts(d_rep_counts, total_rep_count, num_parent_rows, stream, mr);

  rmm::device_uvector<repeated_occurrence> d_rep_occs(total_rep_count, stream, scratch_mr);
  launch_scan_repeated_in_nested(message_data,
                                 message_data_size,
                                 row_offsets,
                                 base_offset,
                                 parent_locs,
                                 num_parent_rows,
                                 d_rep_schema.data(),
                                 list_offs.data(),
                                 d_rep_indices.data(),
                                 d_rep_occs.data(),
                                 d_error.data(),
                                 stream);

  rmm::device_uvector<int32_t> d_rep_top_row_indices(total_rep_count, stream, scratch_mr);
  thrust::transform(rmm::exec_policy_nosync(stream, scratch_mr),
                    d_rep_occs.begin(),
                    d_rep_occs.end(),
                    d_rep_top_row_indices.begin(),
                    [top_row_indices] __device__(repeated_occurrence const& occ) {
                      return top_row_indices != nullptr ? top_row_indices[occ.row_idx]
                                                        : occ.row_idx;
                    });

  std::unique_ptr<cudf::column> child_values;
  auto const rep_blocks =
    static_cast<int>((total_rep_count + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  nested_repeated_location_provider nr_loc{
    row_offsets, base_offset, parent_locs, d_rep_occs.data()};

  if (elem_type_id == cudf::type_id::BOOL8 || elem_type_id == cudf::type_id::INT32 ||
      elem_type_id == cudf::type_id::UINT32 || elem_type_id == cudf::type_id::INT64 ||
      elem_type_id == cudf::type_id::UINT64 || elem_type_id == cudf::type_id::FLOAT32 ||
      elem_type_id == cudf::type_id::FLOAT64) {
    child_values = extract_typed_column(cudf::data_type{elem_type_id},
                                        static_cast<int>(schema[child_schema_idx].encoding),
                                        message_data,
                                        nr_loc,
                                        total_rep_count,
                                        rep_blocks,
                                        THREADS_PER_BLOCK,
                                        false,
                                        0,
                                        0.0,
                                        false,
                                        cudf::detail::make_pinned_vector_async<uint8_t>(0, stream),
                                        child_schema_idx,
                                        enum_valid_values,
                                        enum_names,
                                        d_row_force_null,
                                        d_error,
                                        stream,
                                        mr,
                                        d_rep_top_row_indices.data(),
                                        propagate_invalid_enum_rows);
  } else if (elem_type_id == cudf::type_id::STRING || elem_type_id == cudf::type_id::LIST) {
    if (elem_type_id == cudf::type_id::STRING &&
        schema[child_schema_idx].encoding == proto_encoding::ENUM_STRING) {
      std::optional<enum_string_lookup_tables> fallback_lookup;
      auto const& lookup = *get_enum_lookup(ctx,
                                            child_schema_idx,
                                            enum_valid_values[child_schema_idx],
                                            enum_names[child_schema_idx],
                                            stream,
                                            mr,
                                            fallback_lookup);
      rmm::device_uvector<int32_t> enum_values(total_rep_count, stream, scratch_mr);
      rmm::device_uvector<bool> valid(
        (total_rep_count > 0 ? total_rep_count : 1), stream, scratch_mr);
      extract_varint_kernel<int32_t, false, nested_repeated_location_provider>
        <<<rep_blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(message_data,
                                                               nr_loc,
                                                               total_rep_count,
                                                               enum_values.data(),
                                                               valid.data(),
                                                               d_error.data(),
                                                               false,
                                                               0);

      rmm::device_uvector<bool> d_elem_has_invalid_enum(total_rep_count, stream, scratch_mr);
      thrust::fill(rmm::exec_policy_nosync(stream, scratch_mr),
                   d_elem_has_invalid_enum.begin(),
                   d_elem_has_invalid_enum.end(),
                   false);
      launch_validate_enum_values(enum_values.data(),
                                  valid.data(),
                                  d_elem_has_invalid_enum.data(),
                                  lookup.d_valid_enums.data(),
                                  static_cast<int>(lookup.d_valid_enums.size()),
                                  total_rep_count,
                                  stream);
      propagate_invalid_enum_flags_to_rows(d_elem_has_invalid_enum,
                                           d_row_force_null,
                                           total_rep_count,
                                           d_rep_top_row_indices.data(),
                                           propagate_invalid_enum_rows,
                                           stream);
      child_values =
        build_enum_string_values_column(enum_values, valid, lookup, total_rep_count, stream, mr);
    } else {
      bool as_bytes      = (elem_type_id == cudf::type_id::LIST);
      auto valid_fn      = [] __device__(cudf::size_type) { return true; };
      auto empty_default = cudf::detail::make_pinned_vector_async<uint8_t>(0, stream);
      child_values       = extract_and_build_string_or_bytes_column(as_bytes,
                                                              message_data,
                                                              total_rep_count,
                                                              nr_loc,
                                                              nr_loc,
                                                              valid_fn,
                                                              false,
                                                              empty_default,
                                                              d_error,
                                                              stream,
                                                              mr);
    }
  } else if (elem_type_id == cudf::type_id::STRUCT) {
    auto gc_indices = find_child_field_indices(schema, num_fields, child_schema_idx);
    if (gc_indices.empty()) {
      child_values = cudf::make_structs_column(total_rep_count,
                                               std::vector<std::unique_ptr<cudf::column>>{},
                                               0,
                                               rmm::device_buffer{},
                                               stream,
                                               mr);
    } else {
      rmm::device_uvector<cudf::size_type> d_virtual_row_offsets(
        total_rep_count, stream, scratch_mr);
      rmm::device_uvector<field_location> d_virtual_parent_locs(
        total_rep_count, stream, scratch_mr);
      launch_compute_virtual_parents_for_nested_repeated(d_rep_occs.data(),
                                                         row_offsets,
                                                         parent_locs,
                                                         d_virtual_row_offsets.data(),
                                                         d_virtual_parent_locs.data(),
                                                         total_rep_count,
                                                         d_error.data(),
                                                         stream);

      child_values = build_nested_struct_column(message_data,
                                                message_data_size,
                                                d_virtual_row_offsets.data(),
                                                base_offset,
                                                d_virtual_parent_locs,
                                                gc_indices,
                                                schema,
                                                num_fields,
                                                ctx,
                                                d_row_force_null,
                                                d_error,
                                                total_rep_count,
                                                stream,
                                                mr,
                                                d_rep_top_row_indices.data(),
                                                depth + 1,
                                                propagate_invalid_enum_rows);
    }
  } else {
    CUDF_FAIL("Protobuf decode internal error: unsupported nested repeated child type id=" +
              std::to_string(static_cast<int>(elem_type_id)));
  }

  auto list_offs_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                      num_parent_rows + 1,
                                                      list_offs.release(),
                                                      rmm::device_buffer{},
                                                      0);
  return cudf::make_lists_column(
    num_parent_rows, std::move(list_offs_col), std::move(child_values), 0, rmm::device_buffer{});
}

}  // namespace spark_rapids_jni::protobuf::detail
