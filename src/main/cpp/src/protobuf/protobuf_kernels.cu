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

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <type_traits>

namespace spark_rapids_jni::protobuf::detail {

namespace {

enum class wire_type_mismatch_policy {
  report_error_and_abort,
  report_error_and_continue,
  continue_silently,
};

// ============================================================================
// Pass 1: Scan all fields kernel - records (offset, length) for each field
// ============================================================================

CUDF_KERNEL void set_error_if_unset_kernel(protobuf_error* error_flag, protobuf_error error)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) { set_error_once(error_flag, error); }
}

__device__ inline void set_row_invalid(bool* rows, int32_t row)
{
  if (rows == nullptr) { return; }
  cuda::atomic_ref<bool, cuda::thread_scope_device> ref(rows[row]);
  ref.store(true, cuda::memory_order_relaxed);
}

__device__ inline int enum_binary_search(int32_t const* valid_enum_values,
                                         int num_valid_values,
                                         int32_t val);

__device__ bool is_recognized_enum_value(field_descriptor const& descriptor,
                                         uint8_t const* value_start,
                                         uint8_t const* value_end,
                                         protobuf_error* error_flag,
                                         bool& recognized)
{
  recognized = true;
  if (descriptor.num_valid_enum_values == 0) return true;

  uint64_t raw_value;
  [[maybe_unused]] int value_size;
  if (!read_varint64(value_start, value_end, raw_value, value_size)) {
    set_error_once(error_flag, protobuf_error::VARINT);
    return false;
  }
  recognized = enum_binary_search(descriptor.valid_enum_values,
                                  descriptor.num_valid_enum_values,
                                  static_cast<int32_t>(raw_value)) >= 0;
  return true;
}

/**
 * Scan one message's bytes once, dispatching matched singular and repeated fields to callbacks.
 *
 * Shared by the top-level (`scan_all_fields_kernel`), nested
 * (`scan_nested_message_fields_kernel`), and occurrence
 * (`scan_all_field_occurrences_kernel`) scanners. The caller owns output initialization and
 * fatal row-level error marking. Parse errors that leave the cursor unsafe return false.
 *
 * `fields` owns the field-number lookup and descriptor attributes used by every scanner.
 * Singular fields are delegated to `on_singular(f, location)` after their location is decoded.
 * The callback owns last-one-wins storage and may ignore unknown proto2 enum values; returning
 * false aborts the scan.
 *
 * Matched repeated fields are delegated to `on_repeated(f, cur, wt)`. Callers capture message
 * bounds only when their repeated handler needs them. The mismatch policy controls singular
 * fields; repeated handlers apply the same depth-specific policy while accepting packed encoding.
 */
struct message_scan_context {
  uint8_t const* begin;
  uint8_t const* end;
  protobuf_error* error;
  bool* row_invalid;
  int max_group_depth;  // Enclosing messages share protobuf-java's recursion budget.
};

template <wire_type_mismatch_policy MismatchPolicy, typename Descriptor>
__device__ bool scan_message_field_locations(message_scan_context context,
                                             lookup_view<Descriptor> fields,
                                             auto&& on_singular,
                                             auto&& on_repeated)
{
  auto const* msg_base = context.begin;
  auto const* msg_end  = context.end;
  auto* error_flag     = context.error;
  bool scan_succeeded  = true;
  auto tag             = proto_tag{0, static_cast<proto_wire_type>(-1)};
  auto advance         = [&](uint8_t const* cur) {
    uint8_t const* next;
    if (!skip_field(cur, msg_end, tag, context.max_group_depth, next)) {
      set_error_once(error_flag, protobuf_error::SKIP);
      scan_succeeded = false;
      return msg_end;
    }
    return next;
  };
  for (uint8_t const* cur = msg_base; cur < msg_end; cur = advance(cur)) {
    if (!decode_tag(cur, msg_end, tag, error_flag)) return false;

    int const f = lookup_field(tag.field_number, fields);
    if (f < 0) continue;

    auto const& field = fields.data[f];
    if (field.is_repeated) {
      if (!on_repeated(f, cur, tag.wire_type)) { return false; }
      continue;
    }
    if (tag.wire_type != field.expected_wire_type) {
      if constexpr (MismatchPolicy == wire_type_mismatch_policy::report_error_and_abort) {
        set_error_once(error_flag, protobuf_error::WIRE_TYPE);
        return false;
      } else if constexpr (MismatchPolicy == wire_type_mismatch_policy::report_error_and_continue) {
        set_error_once(error_flag, protobuf_error::WIRE_TYPE);
        if (context.row_invalid != nullptr) { *context.row_invalid = true; }
      }
      continue;
    }

    int const data_offset = static_cast<int>(cur - msg_base);
    field_location location;
    if (tag.wire_type == proto_wire_type::LEN) {
      // Length prefixes use raw-varint32 semantics and may consume up to ten bytes.
      uint32_t len;
      int len_bytes;
      if (!read_varint32(cur, msg_end, len, len_bytes)) {
        set_error_once(error_flag, protobuf_error::VARINT);
        return false;
      }
      if (len > static_cast<uint32_t>(msg_end - cur - len_bytes) ||
          len > static_cast<uint32_t>(cuda::std::numeric_limits<int>::max())) {
        set_error_once(error_flag, protobuf_error::OVERFLOW);
        return false;
      }
      int32_t data_location;
      if (!checked_add_int32(data_offset, len_bytes, data_location)) {
        set_error_once(error_flag, protobuf_error::OVERFLOW);
        return false;
      }
      location = {data_location, static_cast<int32_t>(len)};
    } else {
      // Fixed-width / varint: record the offset and the wire-type-derived size.
      int field_size = get_wire_type_size(tag.wire_type, cur, msg_end);
      if (field_size < 0) {
        set_error_once(error_flag, protobuf_error::FIELD_SIZE);
        return false;
      }
      location = {data_offset, field_size};
    }
    if (!on_singular(f, location)) { return false; }
  }
  return scan_succeeded;
}

/**
 * Top-level field scanner: one thread per row records each requested top-level field's location
 * via the shared `scan_message_field_locations`. Null rows and out-of-bounds messages leave the
 * row's locations as {-1, 0}; in permissive mode malformed rows are flagged for nulling.
 */
CUDF_KERNEL void scan_all_fields_kernel(cudf::column_device_view const d_in,
                                        field_scan_view fields,
                                        protobuf_error* error_flag,
                                        protobuf_error* deferred_enum_error,
                                        bool* row_has_invalid_data)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  cudf::lists_column_device_view in{d_in};
  if (row >= in.size()) return;

  auto mark_row_error = [&]() { set_row_invalid(row_has_invalid_data, row); };

  auto* field_locations = fields.location_stride > 0
                            ? fields.locations + flat_index(row, fields.location_stride, 0)
                            : nullptr;
  for (int f = 0; f < fields.location_stride; f++) {
    field_locations[f] = {-1, 0};
  }

  if (in.nullable() && in.is_null(row)) return;

  auto const base   = in.offset_at(0);
  auto const child  = in.get_sliced_child();
  auto const* bytes = reinterpret_cast<uint8_t const*>(child.data<int8_t>());
  int32_t start     = in.offset_at(row) - base;
  int32_t end       = in.offset_at(row + 1) - base;

  if (!check_message_bounds(start, end, child.size(), error_flag)) {
    mark_row_error();
    return;
  }

  uint8_t const* const msg_base = bytes + start;
  uint8_t const* const msg_end  = bytes + end;

  auto record_singular = [&](int f, field_location location) {
    auto const& descriptor = fields.lookup.data[f];
    bool recognized;
    auto const* value_start = msg_base + location.offset;
    // protobuf-java retains unknown closed-enum values in UnknownFieldSet, and Spark rejects them
    // at the root even if a later recognized occurrence becomes the field's last value.
    if (!is_recognized_enum_value(
          descriptor, value_start, value_start + location.length, error_flag, recognized)) {
      return false;
    }
    if (!recognized) {
      if (row_has_invalid_data != nullptr) {
        mark_row_error();
      } else {
        set_error_once(deferred_enum_error, protobuf_error::INVALID_ENUM);
      }
    } else {
      field_locations[f] = location;
    }
    return true;
  };
  // Top-level scalar descriptors are never repeated, so the repeated handler is unreachable.
  auto unreachable_repeated = [](int, uint8_t const*, proto_wire_type) { return true; };
  if (!scan_message_field_locations<wire_type_mismatch_policy::report_error_and_abort>(
        {msg_base, msg_end, error_flag, nullptr, PROTOBUF_JAVA_RECURSION_LIMIT},
        fields.lookup,
        record_singular,
        unreachable_repeated)) {
    mark_row_error();
  }
}

// ============================================================================
// Shared device functions for repeated field processing
// ============================================================================

/**
 * Visit each occurrence of a repeated field (packed or unpacked) and invoke `f` for it.
 *
 * `f(int32_t elem_offset, int32_t elem_len) -> bool` runs once per occurrence with the
 * element's offset relative to `msg_base` and its length. Returning false aborts the walk.
 * The walker handles wire-type validation, packed-vs-unpacked dispatch, varint/fixed-width
 * length decoding, and packed-buffer bounds checking.
 */
template <wire_type_mismatch_policy MismatchPolicy, typename F>
  requires std::is_invocable_r_v<bool, F, int32_t /*elem_offset*/, int32_t /*elem_len*/>
__device__ bool walk_repeated_element(uint8_t const* cur,
                                      uint8_t const* msg_base,
                                      uint8_t const* msg_end,
                                      proto_wire_type wt,
                                      proto_wire_type expected_wt,
                                      protobuf_error* error_flag,
                                      F&& f)
{
  bool is_packed = wt == proto_wire_type::LEN && expected_wt != proto_wire_type::LEN;

  if (!is_packed && wt != expected_wt) {
    if constexpr (MismatchPolicy == wire_type_mismatch_policy::continue_silently) {
      return true;
    } else {
      set_error_once(error_flag, protobuf_error::WIRE_TYPE);
      return MismatchPolicy == wire_type_mismatch_policy::report_error_and_continue;
    }
  }

  if (is_packed) {
    uint32_t packed_len;
    int len_bytes;
    if (!read_varint32(cur, msg_end, packed_len, len_bytes)) {
      set_error_once(error_flag, protobuf_error::VARINT);
      return false;
    }
    uint8_t const* packed_start = cur + len_bytes;
    if (packed_len > static_cast<uint64_t>(msg_end - packed_start)) {
      set_error_once(error_flag, protobuf_error::OVERFLOW);
      return false;
    }
    uint8_t const* packed_end = packed_start + packed_len;

    switch (expected_wt) {
      case proto_wire_type::VARINT: {
        // `vbytes` is set inside the loop body before `p += vbytes` runs (the advance step
        // happens after each body execution), but we initialize it defensively to silence a
        // potential "used before set" warning. `read_varint64` validates the varint stays
        // within `packed_end` (the packed payload's end), not `msg_end` — switching to a
        // generic skip helper here would over-read past the packed buffer.
        int vbytes = cuda::std::numeric_limits<int>::max();
        for (uint8_t const* p = packed_start; p < packed_end; p += vbytes) {
          int32_t elem_offset = static_cast<int32_t>(p - msg_base);
          uint64_t dummy;
          if (!read_varint64(p, packed_end, dummy, vbytes)) {
            set_error_once(error_flag, protobuf_error::VARINT);
            return false;
          }
          if (!f(elem_offset, vbytes)) return false;
        }
        break;
      }
      case proto_wire_type::I32BIT:
      case proto_wire_type::I64BIT: {
        int const width = expected_wt == proto_wire_type::I32BIT ? 4 : 8;
        if ((packed_len % width) != 0) {
          set_error_once(error_flag, protobuf_error::FIXED_LEN);
          return false;
        }
        for (uint8_t const* p = packed_start; p < packed_end; p += width) {
          int32_t elem_offset = static_cast<int32_t>(p - msg_base);
          if (!f(elem_offset, width)) return false;
        }
        break;
      }
      default:
        // Unreachable on a well-formed config: only VARINT / I32BIT / I64BIT are valid for
        // packed wire types here (LEN is already filtered out above by the !is_packed path).
        // Fail loudly rather than silently swallowing an unexpected expected_wt.
        set_error_once(error_flag, protobuf_error::WIRE_TYPE);
        return false;
    }
  } else {
    // Unpacked single occurrence. We use `get_field_data_location` rather than `skip_field`
    // because the scan path's `f` needs both the data offset and length to record an
    // occurrence; `skip_field` advances past the field but doesn't surface those. The count
    // path's `f` ignores them, but sharing one helper keeps the walker generic over both
    // actions and avoids re-validating field bounds twice.
    int32_t data_offset, data_length;
    if (!get_field_data_location(cur, msg_end, wt, data_offset, data_length)) {
      set_error_once(error_flag, protobuf_error::FIELD_SIZE);
      return false;
    }
    int32_t abs_offset = static_cast<int32_t>(cur - msg_base) + data_offset;
    if (!f(abs_offset, data_length)) return false;
  }
  return true;
}

CUDF_KERNEL void validate_message_fragments_kernel(message_fragment_location_provider locations,
                                                   message_validation_view fields,
                                                   int num_fragments,
                                                   bool* invalid_rows,
                                                   bool* row_has_invalid_data,
                                                   protobuf_error* error_flag,
                                                   int max_group_depth)
{
  auto const idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= num_fragments) return;

  auto const fragment = locations.fragments[idx];
  auto const row      = fragment.row_idx;
  auto const top_row =
    locations.source.top_row_indices == nullptr ? row : locations.source.top_row_indices[row];
  auto mark_row_error = [&]() {
    set_row_invalid(invalid_rows, row);
    set_row_invalid(row_has_invalid_data, top_row);
  };

  auto const parent =
    locations.source.parent_locations == nullptr
      ? field_location{0, locations.input.row_offsets[row + 1] - locations.input.row_offsets[row]}
      : locations.source.parent_locations[row];
  if (parent.offset < 0 || parent.length < 0 || fragment.offset < 0 || fragment.length < 0) {
    set_error_once(error_flag, protobuf_error::BOUNDS);
    mark_row_error();
    return;
  }

  auto const row_start =
    static_cast<int64_t>(locations.input.row_offsets[row]) - locations.input.base_offset;
  auto const parent_start   = row_start + parent.offset;
  auto const fragment_start = parent_start + fragment.offset;
  auto const fragment_end   = fragment_start + fragment.length;
  auto const parent_end     = parent_start + parent.length;
  if (fragment_start < parent_start || fragment_end > parent_end ||
      !check_message_bounds(
        fragment_start, fragment_end, locations.input.message_data_size, error_flag)) {
    set_error_once(error_flag, protobuf_error::BOUNDS);
    mark_row_error();
    return;
  }

  auto record_singular = []([[maybe_unused]] int f, [[maybe_unused]] field_location location) {
    return true;
  };
  auto const* fragment_begin = locations.input.message_data + fragment_start;
  auto const* fragment_limit = locations.input.message_data + fragment_end;
  auto validate_repeated     = [&](int f, uint8_t const* cur, proto_wire_type wire_type) {
    auto ignore_occurrence = []([[maybe_unused]] int32_t offset, [[maybe_unused]] int32_t length) {
      return true;
    };
    return walk_repeated_element<wire_type_mismatch_policy::continue_silently>(
      cur,
      fragment_begin,
      fragment_limit,
      wire_type,
      fields.lookup.data[f].expected_wire_type,
      error_flag,
      ignore_occurrence);
  };

  if (!scan_message_field_locations<wire_type_mismatch_policy::continue_silently>(
        {fragment_begin, fragment_limit, error_flag, nullptr, max_group_depth},
        fields.lookup,
        record_singular,
        validate_repeated)) {
    mark_row_error();
  }
}

// ============================================================================
// Pass 1b: Count repeated fields kernel
// ============================================================================

/**
 * Count occurrences of repeated fields in each row.
 * Also records locations of nested message fields for hierarchical processing.
 *
 * One descriptor lookup maps field numbers to repeated-count or nested-location outputs.
 * A null direct lookup pointer falls back to linear search.
 */
CUDF_KERNEL void count_repeated_fields_kernel(cudf::column_device_view const d_in,
                                              field_scan_view fields,
                                              protobuf_error* error_flag,
                                              protobuf_error* deferred_enum_error,
                                              bool* row_has_invalid_data)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  cudf::lists_column_device_view in{d_in};
  if (row >= in.size()) return;
  auto mark_row_error = [&]() { set_row_invalid(row_has_invalid_data, row); };

  auto* field_locations = fields.location_stride > 0
                            ? fields.locations + flat_index(row, fields.location_stride, 0)
                            : nullptr;
  for (int f = 0; f < fields.location_stride; f++) {
    field_locations[f] = {-1, 0};
  }
  auto* field_message_info =
    fields.singular_message_stride > 0
      ? fields.singular_message_info + flat_index(row, fields.singular_message_stride, 0)
      : nullptr;
  for (int f = 0; f < fields.singular_message_stride; f++) {
    field_message_info[f] = {0};
  }

  auto* field_repeated_info = fields.repeated_stride > 0
                                ? fields.repeated_info + flat_index(row, fields.repeated_stride, 0)
                                : nullptr;
  for (int f = 0; f < fields.repeated_stride; f++) {
    field_repeated_info[f] = {0};
  }

  if (in.nullable() && in.is_null(row)) return;

  auto const base   = in.offset_at(0);
  auto const child  = in.get_sliced_child();
  auto const* bytes = reinterpret_cast<uint8_t const*>(child.data<int8_t>());
  int32_t start     = in.offset_at(row) - base;
  int32_t end       = in.offset_at(row + 1) - base;
  if (!check_message_bounds(start, end, child.size(), error_flag)) {
    mark_row_error();
    return;
  }

  uint8_t const* const msg_base = bytes + start;
  uint8_t const* const msg_end  = bytes + end;
  auto* row_invalid = row_has_invalid_data != nullptr ? row_has_invalid_data + row : nullptr;

  auto record_nested = [&](int f, field_location location) {
    auto const& field                   = fields.lookup.data[f];
    field_locations[field.output_index] = location;
    auto& info                          = field_message_info[field.output_index];
    if (++info.count == 2) { atomicExch(fields.multiple_message_fields + field.output_index, 1); }
    return true;
  };
  auto count_repeated = [&](int f, uint8_t const* cur, proto_wire_type wire_type) {
    auto const& field = fields.lookup.data[f];
    auto& info        = field_repeated_info[field.output_index];
    auto count_action = [&](int32_t offset, int32_t length) {
      bool recognized;
      auto const* value_start = msg_base + offset;
      // Spark applies the same root UnknownFieldSet check to repeated enums; nested unknown values
      // are instead pruned by the builders.
      if (!is_recognized_enum_value(
            field, value_start, value_start + length, error_flag, recognized)) {
        return false;
      }
      if (!recognized) {
        if (row_invalid != nullptr) {
          *row_invalid = true;
        } else {
          set_error_once(deferred_enum_error, protobuf_error::INVALID_ENUM);
        }
      }
      info.count++;
      return true;
    };
    return walk_repeated_element<wire_type_mismatch_policy::report_error_and_abort>(
      cur, msg_base, msg_end, wire_type, field.expected_wire_type, error_flag, count_action);
  };

  if (!scan_message_field_locations<wire_type_mismatch_policy::report_error_and_continue>(
        {msg_base, msg_end, error_flag, row_invalid, PROTOBUF_JAVA_RECURSION_LIMIT},
        fields.lookup,
        record_nested,
        count_repeated)) {
    mark_row_error();
  }
}

/**
 * Scan each message once and write occurrences for every selected field.
 */
template <wire_type_mismatch_policy MismatchPolicy>
__device__ bool scan_all_field_occurrences_in_message(uint8_t const* msg_base,
                                                      uint8_t const* msg_end,
                                                      field_occurrence_scan_view fields,
                                                      protobuf_error* error_flag,
                                                      cudf::size_type row,
                                                      int max_group_depth)
{
  // Host launchers chunk descriptors to this capacity. Keep the device-side check because
  // overrunning `write_idx` below is silent UB.
  if (fields.size > MAX_REPEATED_FIELDS_PER_KERNEL) {
    set_error_once(error_flag, protobuf_error::SCHEMA_TOO_LARGE);
    return false;
  }

  int write_idx[MAX_REPEATED_FIELDS_PER_KERNEL];
  for (int f = 0; f < fields.size; f++) {
    write_idx[f] = fields.data[f].row_offsets[row];
  }

  auto unreachable_singular = []([[maybe_unused]] int f, [[maybe_unused]] field_location location) {
    return true;
  };

  auto const row_i32    = static_cast<int32_t>(row);
  auto on_repeated_scan = [&](int f, uint8_t const* cur, proto_wire_type wt) {
    auto const& field = fields.data[f];
    auto* occs        = field.occurrences;
    int& wi           = write_idx[f];
    int const we      = field.row_offsets[row + 1];
    auto scan_action  = [&](int32_t off, int32_t len) {
      if (wi >= we) {
        set_error_once(error_flag, protobuf_error::REPEATED_COUNT_MISMATCH);
        return false;
      }
      occs[wi] = {row_i32, off, len};
      wi++;
      return true;
    };
    return walk_repeated_element<MismatchPolicy>(
      cur, msg_base, msg_end, wt, field.expected_wire_type, error_flag, scan_action);
  };

  if (!scan_message_field_locations<MismatchPolicy>(
        {msg_base, msg_end, error_flag, nullptr, max_group_depth},
        fields,
        unreachable_singular,
        on_repeated_scan)) {
    return false;
  }

  for (int f = 0; f < fields.size; f++) {
    if (write_idx[f] != fields.data[f].row_offsets[row + 1]) {
      set_error_once(error_flag, protobuf_error::REPEATED_COUNT_MISMATCH);
      return false;
    }
  }
  return true;
}

CUDF_KERNEL void scan_all_field_occurrences_kernel(cudf::column_device_view const d_in,
                                                   field_occurrence_scan_view fields,
                                                   protobuf_error* error_flag)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  cudf::lists_column_device_view in{d_in};
  if (row >= in.size()) return;

  if (in.nullable() && in.is_null(row)) return;

  auto const base   = in.offset_at(0);
  auto const child  = in.get_sliced_child();
  auto const* bytes = reinterpret_cast<uint8_t const*>(child.data<int8_t>());
  int32_t start     = in.offset_at(row) - base;
  int32_t end       = in.offset_at(row + 1) - base;
  if (!check_message_bounds(start, end, child.size(), error_flag)) return;

  [[maybe_unused]] auto const scan_succeeded =
    scan_all_field_occurrences_in_message<wire_type_mismatch_policy::report_error_and_abort>(
      bytes + start, bytes + end, fields, error_flag, row, PROTOBUF_JAVA_RECURSION_LIMIT);
}

// ============================================================================
// Nested message scanning kernels
// ============================================================================

/**
 * Scan one nested message per parent row to locate singleton children and count occurrences.
 * Singleton locations use last-one-wins semantics; selected occurrences are written by a later
 * scan after their row offsets are available.
 */
CUDF_KERNEL void scan_nested_message_fields_kernel(protobuf_input_view input,
                                                   nested_parent_view parent,
                                                   field_scan_view fields,
                                                   protobuf_error* error_flag,
                                                   bool* row_has_invalid_data,
                                                   int max_group_depth)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= input.num_rows) return;

  auto const top_row =
    parent.top_row_indices != nullptr ? parent.top_row_indices[row] : static_cast<int32_t>(row);
  auto mark_row_error = [&]() { set_row_invalid(row_has_invalid_data, top_row); };

  field_location* field_locations = fields.locations + flat_index(row, fields.location_stride, 0);
  for (int f = 0; f < fields.location_stride; f++) {
    field_locations[f] = {-1, 0};
  }
  auto* field_repeated_info = fields.repeated_stride > 0
                                ? fields.repeated_info + flat_index(row, fields.repeated_stride, 0)
                                : nullptr;
  for (int f = 0; f < fields.repeated_stride; f++) {
    field_repeated_info[f] = {0};
  }
  auto* field_message_info =
    fields.singular_message_stride > 0
      ? fields.singular_message_info + flat_index(row, fields.singular_message_stride, 0)
      : nullptr;
  if (field_message_info != field_repeated_info) {
    for (int f = 0; f < fields.singular_message_stride; f++) {
      field_message_info[f] = {0};
    }
  }

  auto const& parent_loc = parent.locations[row];
  if (parent_loc.offset < 0) return;

  // Do the subtraction in int64 to keep the bounds-check honest even if a future caller
  // ever passes a sliced LIST where parent_base_offset > parent_row_offsets[row].
  int64_t parent_row_start = static_cast<int64_t>(input.row_offsets[row]) - input.base_offset;
  int64_t nested_start_off = parent_row_start + parent_loc.offset;
  int64_t nested_end_off   = nested_start_off + parent_loc.length;
  if (!check_message_bounds(
        nested_start_off, nested_end_off, input.message_data_size, error_flag)) {
    mark_row_error();
    return;
  }
  uint8_t const* const nested_start = input.message_data + nested_start_off;
  uint8_t const* const nested_end   = input.message_data + nested_end_off;

  auto record_singular = [&](int f, field_location location) {
    auto const& descriptor = fields.lookup.data[f];
    bool recognized;
    auto const* value_start = nested_start + location.offset;
    if (!is_recognized_enum_value(
          descriptor, value_start, value_start + location.length, error_flag, recognized)) {
      return false;
    }
    if (recognized) {
      field_locations[f] = location;
      if (descriptor.is_message) {
        auto& info = field_message_info[f];
        if (++info.count == 2) { atomicExch(fields.multiple_message_fields + f, 1); }
      }
    }
    return true;
  };
  auto validate_repeated = [&](int f, uint8_t const* cur, proto_wire_type wt) {
    auto const expected_wire_type = fields.lookup.data[f].expected_wire_type;
    auto count_occurrence = [&]([[maybe_unused]] int32_t off, [[maybe_unused]] int32_t len) {
      if (field_repeated_info != nullptr) { field_repeated_info[f].count++; }
      return true;
    };
    return walk_repeated_element<wire_type_mismatch_policy::continue_silently>(
      cur, nested_start, nested_end, wt, expected_wire_type, error_flag, count_occurrence);
  };

  // protobuf-java preserves wrong-wire known fields in UnknownFieldSet; this projected API has no
  // compatible output channel for nested fields.
  if (!scan_message_field_locations<wire_type_mismatch_policy::continue_silently>(
        {nested_start, nested_end, error_flag, nullptr, max_group_depth},
        fields.lookup,
        record_singular,
        validate_repeated)) {
    mark_row_error();
  }
}

CUDF_KERNEL void scan_all_field_occurrences_in_nested_kernel(protobuf_input_view input,
                                                             nested_parent_view parent,
                                                             field_occurrence_scan_view fields,
                                                             protobuf_error* error_flag,
                                                             int max_group_depth)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= input.num_rows) return;

  auto const& parent_loc = parent.locations[row];
  if (parent_loc.offset < 0) return;

  int64_t const row_off       = static_cast<int64_t>(input.row_offsets[row]) - input.base_offset;
  int64_t const msg_start_off = row_off + parent_loc.offset;
  int64_t const msg_end_off   = msg_start_off + parent_loc.length;
  if (!check_message_bounds(msg_start_off, msg_end_off, input.message_data_size, error_flag)) {
    return;
  }

  [[maybe_unused]] auto const scan_succeeded =
    scan_all_field_occurrences_in_message<wire_type_mismatch_policy::continue_silently>(
      input.message_data + msg_start_off,
      input.message_data + msg_end_off,
      fields,
      error_flag,
      row,
      max_group_depth);
}

CUDF_KERNEL void compute_grandchild_parent_locations_kernel(nested_location_provider loc_provider,
                                                            field_location* gc_parent_locs,
                                                            int num_rows,
                                                            protobuf_error* error_flag)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;

  gc_parent_locs[row] = loc_provider.get_rebased_child_location(row, error_flag);
}

CUDF_KERNEL void compute_virtual_parents_for_nested_repeated_kernel(
  field_occurrence const* occurrences,
  cudf::size_type const* row_list_offsets,
  field_location const* parent_locations,
  cudf::size_type* virtual_row_offsets,
  field_location* virtual_parent_locs,
  int total_count,
  protobuf_error* error_flag)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_count) return;

  auto const& occurrence   = occurrences[idx];
  auto const& parent       = parent_locations[occurrence.row_idx];
  virtual_row_offsets[idx] = row_list_offsets[occurrence.row_idx];

  if (parent.offset < 0) {
    virtual_parent_locs[idx] = {-1, 0};
    return;
  }

  auto const offset = static_cast<int64_t>(parent.offset) + occurrence.offset;
  if (offset < cuda::std::numeric_limits<int32_t>::min() ||
      offset > cuda::std::numeric_limits<int32_t>::max()) {
    virtual_parent_locs[idx] = {-1, 0};
    set_error_once(error_flag, protobuf_error::OVERFLOW);
    return;
  }
  virtual_parent_locs[idx] = {static_cast<int32_t>(offset), occurrence.length};
}

CUDF_KERNEL void compute_msg_locations_from_occurrences_kernel(field_occurrence const* occurrences,
                                                               cudf::size_type const* list_offsets,
                                                               cudf::size_type base_offset,
                                                               field_location* msg_locs,
                                                               cudf::size_type* msg_row_offsets,
                                                               int total_count,
                                                               protobuf_error* error_flag)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_count) return;

  auto const& occurrence = occurrences[idx];
  auto const row_offset  = static_cast<int64_t>(list_offsets[occurrence.row_idx]) - base_offset;
  if (row_offset < cuda::std::numeric_limits<cudf::size_type>::min() ||
      row_offset > cuda::std::numeric_limits<cudf::size_type>::max()) {
    msg_row_offsets[idx] = 0;
    msg_locs[idx]        = {-1, 0};
    set_error_once(error_flag, protobuf_error::OVERFLOW);
    return;
  }
  msg_row_offsets[idx] = static_cast<cudf::size_type>(row_offset);
  msg_locs[idx]        = {occurrence.offset, occurrence.length};
}

/**
 * Pull one field's per-row locations out of the 2D nested-locations array. Replaces a
 * D2H + CPU loop + H2D pattern previously used to extract a parent-location vector per
 * nested struct field.
 */
CUDF_KERNEL void extract_strided_locations_kernel(field_location const* nested_locations,
                                                  int field_idx,
                                                  int num_fields,
                                                  field_location* parent_locs,
                                                  int num_rows)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;
  parent_locs[row] = nested_locations[flat_index(row, num_fields, field_idx)];
}

// ============================================================================
// Kernel to check required fields after scan pass
// ============================================================================

/**
 * Check if any required fields are missing (offset < 0) and set error flag.
 * This is called after the scan pass to validate required field constraints.
 */
CUDF_KERNEL void check_required_fields_kernel(
  required_field_input_view input,
  uint8_t const* is_required,  // [num_fields] (1 = required, 0 = optional)
  int num_fields,
  bool* row_force_null,  // [top_level_num_rows] optional permissive row nulling
  protobuf_error* error_flag)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= input.values.size) return;
  if (input.input_null_mask != nullptr &&
      !cudf::bit_is_set(input.input_null_mask, row + input.input_offset)) {
    return;
  }
  if (input.parent_locations != nullptr && input.parent_locations[row].offset < 0) return;

  for (int f = 0; f < num_fields; f++) {
    if (is_required[f] != 0 && input.locations[flat_index(row, num_fields, f)].offset < 0) {
      if (row_force_null != nullptr) {
        auto const top_row = input.values.top_row_indices != nullptr
                               ? input.values.top_row_indices[row]
                               : static_cast<int32_t>(row);
        set_row_invalid(row_force_null, top_row);
      }
      // Required field is missing - set error flag
      set_error_once(error_flag, protobuf_error::REQUIRED);
      return;  // No need to check other fields for this row
    }
  }
}

/**
 * Binary search a sorted enum-value array. Returns the matched index or -1 if not found.
 * Shared between the validate / lengths / chars enum-as-string kernels.
 */
__device__ inline int enum_binary_search(int32_t const* valid_enum_values,
                                         int num_valid_values,
                                         int32_t val)
{
  int left  = 0;
  int right = num_valid_values - 1;
  while (left <= right) {
    int mid         = left + (right - left) / 2;
    int32_t mid_val = valid_enum_values[mid];
    if (mid_val == val) {
      return mid;
    } else if (mid_val < val) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return -1;
}

/**
 * Validate enum values against a set of valid values.
 * Values outside the set are marked invalid so singular fields fall back to their proto2 default
 * and repeated fields can omit the occurrence.
 *
 * The valid_values array must be sorted for binary search.
 *
 * @note Time complexity: O(log(num_valid_values)) per row.
 */
CUDF_KERNEL void validate_enum_values_kernel(enum_value_device_view input,
                                             enum_domain_device_view domain)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= input.size) return;

  // Skip if already invalid (field was missing) - missing field is not an enum error
  if (!input.valid[row]) return;

  if (enum_binary_search(domain.valid_values, domain.size, input.values[row]) < 0) {
    input.valid[row] = false;
  }
}

/**
 * Compute output UTF-8 length for enum-as-string rows.
 * Invalid/missing values produce length 0; the caller applies row/field semantics.
 */
CUDF_KERNEL void compute_enum_string_lengths_kernel(enum_value_device_view input,
                                                    enum_string_lookup_device_view lookup,
                                                    int32_t* lengths)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= input.size) return;

  if (!input.valid[row]) {
    lengths[row] = 0;
    return;
  }

  int idx = enum_binary_search(lookup.domain.valid_values, lookup.domain.size, input.values[row]);
  // Should not happen when validate_enum_values_kernel has already run, but keep safe.
  lengths[row] = idx >= 0 ? (lookup.name_offsets[idx + 1] - lookup.name_offsets[idx]) : 0;
}

/**
 * Copy enum-as-string UTF-8 bytes into output chars buffer using precomputed row offsets.
 */
CUDF_KERNEL void copy_enum_string_chars_kernel(enum_value_device_view input,
                                               enum_string_lookup_device_view lookup,
                                               int32_t const* output_offsets,
                                               char* out_chars)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= input.size) return;
  if (!input.valid[row]) return;

  int idx = enum_binary_search(lookup.domain.valid_values, lookup.domain.size, input.values[row]);
  if (idx < 0) return;
  int32_t src_begin = lookup.name_offsets[idx];
  int32_t src_end   = lookup.name_offsets[idx + 1];
  int32_t dst_begin = output_offsets[row];
  memcpy(
    out_chars + dst_begin, lookup.name_chars + src_begin, static_cast<size_t>(src_end - src_begin));
}

}  // anonymous namespace

// ============================================================================
// Host wrapper functions — callable from other translation units
// ============================================================================

void set_error_once_async(protobuf_error* error_flag, protobuf_error error, cuda::stream_ref stream)
{
  set_error_if_unset_kernel<<<1, 1, 0, stream.get()>>>(error_flag, error);
  CUDF_CHECK_CUDA(stream.get());
}

void launch_scan_all_fields(cudf::column_device_view const& d_in,
                            field_scan_view fields,
                            protobuf_error* error_flag,
                            protobuf_error* deferred_enum_error,
                            bool* row_has_invalid_data,
                            cuda::stream_ref stream)
{
  auto const num_rows = d_in.size();
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  scan_all_fields_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.get()>>>(
    d_in, fields, error_flag, deferred_enum_error, row_has_invalid_data);
  CUDF_CHECK_CUDA(stream.get());
}

void launch_count_repeated_fields(cudf::column_device_view const& d_in,
                                  field_scan_view fields,
                                  protobuf_error* error_flag,
                                  protobuf_error* deferred_enum_error,
                                  bool* row_has_invalid_data,
                                  cuda::stream_ref stream)
{
  auto const num_rows = d_in.size();
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  count_repeated_fields_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.get()>>>(
    d_in, fields, error_flag, deferred_enum_error, row_has_invalid_data);
  CUDF_CHECK_CUDA(stream.get());
}

void launch_scan_all_field_occurrences(cudf::column_device_view const& d_in,
                                       field_occurrence_scan_view fields,
                                       protobuf_error* error_flag,
                                       cuda::stream_ref stream)
{
  auto const num_rows = d_in.size();
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  scan_all_field_occurrences_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.get()>>>(
    d_in, fields, error_flag);
  CUDF_CHECK_CUDA(stream.get());
}

void launch_scan_singular_message_occurrences(cudf::column_device_view const& d_in,
                                              field_occurrence_scan_view fields,
                                              protobuf_error* error_flag,
                                              cuda::stream_ref stream)
{
  auto const num_rows = d_in.size();
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  scan_all_field_occurrences_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.get()>>>(
    d_in, fields, error_flag);
  CUDF_CHECK_CUDA(stream.get());
}

void launch_extract_strided_locations(field_location const* nested_locations,
                                      int field_idx,
                                      int num_fields,
                                      field_location* parent_locs,
                                      int num_rows,
                                      cuda::stream_ref stream)
{
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  extract_strided_locations_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.get()>>>(
    nested_locations, field_idx, num_fields, parent_locs, num_rows);
  CUDF_CHECK_CUDA(stream.get());
}

void launch_scan_nested_message_fields(protobuf_input_view input,
                                       nested_parent_view parent,
                                       field_scan_view fields,
                                       protobuf_error* error_flag,
                                       bool* row_has_invalid_data,
                                       int recursion_depth,
                                       cuda::stream_ref stream)
{
  if (input.num_rows == 0) return;
  auto const max_group_depth = PROTOBUF_JAVA_RECURSION_LIMIT - recursion_depth;
  auto const blocks =
    static_cast<int>((input.num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  scan_nested_message_fields_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.get()>>>(
    input, parent, fields, error_flag, row_has_invalid_data, max_group_depth);
  CUDF_CHECK_CUDA(stream.get());
}

void launch_scan_all_field_occurrences_in_nested(protobuf_input_view input,
                                                 nested_parent_view parent,
                                                 field_occurrence_scan_view fields,
                                                 protobuf_error* error_flag,
                                                 int recursion_depth,
                                                 cuda::stream_ref stream)
{
  if (input.num_rows == 0) return;
  auto const max_group_depth = PROTOBUF_JAVA_RECURSION_LIMIT - recursion_depth;
  auto const blocks =
    static_cast<int>((input.num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  scan_all_field_occurrences_in_nested_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.get()>>>(
    input, parent, fields, error_flag, max_group_depth);
  CUDF_CHECK_CUDA(stream.get());
}

void launch_validate_message_fragments(message_fragment_location_provider locations,
                                       message_validation_view fields,
                                       int num_fragments,
                                       bool* invalid_rows,
                                       bool* row_has_invalid_data,
                                       protobuf_error* error_flag,
                                       int recursion_depth,
                                       cuda::stream_ref stream)
{
  if (num_fragments == 0) return;
  auto const max_group_depth = PROTOBUF_JAVA_RECURSION_LIMIT - recursion_depth;
  auto const blocks =
    static_cast<int>((num_fragments + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  validate_message_fragments_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.get()>>>(
    locations,
    fields,
    num_fragments,
    invalid_rows,
    row_has_invalid_data,
    error_flag,
    max_group_depth);
  CUDF_CHECK_CUDA(stream.get());
}

void launch_compute_grandchild_parent_locations(nested_location_provider loc_provider,
                                                field_location* gc_parent_locs,
                                                int num_rows,
                                                protobuf_error* error_flag,
                                                cuda::stream_ref stream)
{
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  compute_grandchild_parent_locations_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.get()>>>(
    loc_provider, gc_parent_locs, num_rows, error_flag);
  CUDF_CHECK_CUDA(stream.get());
}

void launch_compute_virtual_parents_for_nested_repeated(protobuf_input_view input,
                                                        nested_parent_view parent,
                                                        repeated_field_work const& work,
                                                        cudf::size_type* virtual_row_offsets,
                                                        field_location* virtual_parent_locs,
                                                        protobuf_decode_runtime_context decode_ctx,
                                                        cuda::stream_ref stream)
{
  if (work.total_count == 0) return;
  auto const blocks =
    static_cast<int>((work.total_count + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  compute_virtual_parents_for_nested_repeated_kernel<<<blocks,
                                                       THREADS_PER_BLOCK,
                                                       0,
                                                       stream.get()>>>(work.occurrences->data(),
                                                                       input.row_offsets,
                                                                       parent.locations,
                                                                       virtual_row_offsets,
                                                                       virtual_parent_locs,
                                                                       work.total_count,
                                                                       decode_ctx.error->data());
  CUDF_CHECK_CUDA(stream.get());
}

void launch_compute_msg_locations_from_occurrences(protobuf_input_view input,
                                                   repeated_field_work const& work,
                                                   field_location* msg_locs,
                                                   cudf::size_type* msg_row_offsets,
                                                   protobuf_decode_runtime_context decode_ctx,
                                                   cuda::stream_ref stream)
{
  if (work.total_count == 0) return;
  auto const blocks =
    static_cast<int>((work.total_count + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  compute_msg_locations_from_occurrences_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.get()>>>(
    work.occurrences->data(),
    input.row_offsets,
    input.base_offset,
    msg_locs,
    msg_row_offsets,
    work.total_count,
    decode_ctx.error->data());
  CUDF_CHECK_CUDA(stream.get());
}

void launch_validate_enum_values(enum_value_device_view input,
                                 enum_domain_device_view domain,
                                 cuda::stream_ref stream)
{
  if (input.size == 0) return;
  auto const blocks = static_cast<int>((input.size + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  validate_enum_values_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.get()>>>(input, domain);
  CUDF_CHECK_CUDA(stream.get());
}

void launch_compute_enum_string_lengths(enum_value_device_view input,
                                        enum_string_lookup_device_view lookup,
                                        int32_t* lengths,
                                        cuda::stream_ref stream)
{
  if (input.size == 0) return;
  auto const blocks = static_cast<int>((input.size + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  compute_enum_string_lengths_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.get()>>>(
    input, lookup, lengths);
  CUDF_CHECK_CUDA(stream.get());
}

void launch_copy_enum_string_chars(enum_value_device_view input,
                                   enum_string_lookup_device_view lookup,
                                   int32_t const* output_offsets,
                                   char* out_chars,
                                   cuda::stream_ref stream)
{
  if (input.size == 0) return;
  auto const blocks = static_cast<int>((input.size + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  copy_enum_string_chars_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.get()>>>(
    input, lookup, output_offsets, out_chars);
  CUDF_CHECK_CUDA(stream.get());
}

void maybe_check_required_fields(required_field_input_view input,
                                 std::vector<int> const& field_indices,
                                 std::vector<nested_field_descriptor> const& schema,
                                 protobuf_decode_runtime_context decode_ctx,
                                 cuda::stream_ref stream)
{
  if (input.values.size == 0 || field_indices.empty()) { return; }

  // Stream-ordered pinned deallocation keeps this staging safe without a local sync.
  bool has_required = false;
  auto h_is_required =
    cudf::detail::make_pinned_vector_async<uint8_t>(field_indices.size(), stream);
  for (size_t i = 0; i < field_indices.size(); ++i) {
    h_is_required[i] = schema[field_indices[i]].is_required ? 1 : 0;
    has_required |= (h_is_required[i] != 0);
  }
  if (!has_required) { return; }

  auto const scratch_mr = cudf::get_current_device_resource_ref();
  auto d_is_required = cudf::detail::make_device_uvector_async(h_is_required, stream, scratch_mr);

  auto const blocks =
    static_cast<int>((input.values.size + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  auto* row_force_null =
    decode_ctx.row_force_null != nullptr && !decode_ctx.row_force_null->is_empty()
      ? decode_ctx.row_force_null->data()
      : nullptr;
  check_required_fields_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.get()>>>(
    input,
    d_is_required.data(),
    static_cast<int>(field_indices.size()),
    row_force_null,
    decode_ctx.error->data());
  CUDF_CHECK_CUDA(stream.get());
}

void validate_enum_values(rmm::device_uvector<int32_t> const& values,
                          rmm::device_uvector<bool>& valid,
                          enum_domain_device_view enum_domain,
                          cuda::stream_ref stream)
{
  CUDF_EXPECTS(values.size() == valid.size(), "enum values and validity sizes must match");
  if (values.is_empty() || enum_domain.size == 0) return;
  CUDF_EXPECTS(enum_domain.valid_values != nullptr, "enum validation requires valid enum values");
  launch_validate_enum_values(
    {values.data(), valid.data(), static_cast<cudf::size_type>(values.size())},
    enum_domain,
    stream);
}

void validate_enum_values(rmm::device_uvector<int32_t> const& values,
                          rmm::device_uvector<bool>& valid,
                          cudf::detail::host_vector<int32_t> const& valid_enums,
                          cuda::stream_ref stream)
{
  CUDF_EXPECTS(values.size() == valid.size(), "enum values and validity sizes must match");
  if (values.is_empty() || valid_enums.empty()) return;

  auto const scratch_mr = cudf::get_current_device_resource_ref();
  auto d_valid_enums    = cudf::detail::make_device_uvector_async(valid_enums, stream, scratch_mr);
  validate_enum_values(
    values, valid, {d_valid_enums.data(), static_cast<int>(d_valid_enums.size())}, stream);
}

}  // namespace spark_rapids_jni::protobuf::detail
