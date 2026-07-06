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
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <type_traits>

namespace spark_rapids_jni::protobuf::detail {

namespace {

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
  if (!read_varint(value_start, value_end, raw_value, value_size)) {
    set_error_once(error_flag, protobuf_error::VARINT);
    return false;
  }
  recognized = enum_binary_search(descriptor.valid_enum_values,
                                  descriptor.num_valid_enum_values,
                                  static_cast<int32_t>(raw_value)) >= 0;
  return true;
}

enum class wire_type_mismatch_policy { report_error, report_error_and_skip, skip };

/**
 * Scan one message's bytes once, dispatching matched singular and repeated fields to callbacks.
 *
 * Shared by the top-level (`scan_all_fields_kernel`), nested
 * (`scan_nested_message_fields_kernel`), and occurrence
 * (`scan_all_field_occurrences_kernel`) scanners. The caller owns output initialization and
 * fatal row-level error marking. Parse errors that leave the cursor unsafe return false.
 * `report_error_and_skip` marks a recoverable mismatch through `context.row_invalid` and keeps
 * scanning so count and occurrence passes remain aligned.
 *
 * `lookup_desc_idx(field_number) -> int` maps a wire field number to its descriptor index (or -1);
 * callers supply it so this helper stays agnostic to whether a lookup table is used. Descriptor
 * attribute access is also caller-supplied so hot paths can use compact descriptor forms directly.
 * Singular fields are delegated to `on_singular(f, cur, msg_end, location)` after their location
 * is decoded. The callback owns last-one-wins storage and may ignore unknown proto2 enum values;
 * returning false aborts the scan.
 *
 * Matched repeated fields are delegated to `on_repeated(f, cur, msg_end, msg_base, wt)` (f is the
 * matched descriptor index) which returns false on error; the handler derives its own expected
 * wire type from f when it needs one. Top-level scalars pass a no-op handler since their
 * descriptors are never repeated.
 */
struct message_scan_context {
  uint8_t const* begin;
  uint8_t const* end;
  protobuf_error* error;
  bool* row_invalid;
  int max_group_depth;  // Enclosing messages share protobuf-java's recursion budget.
};

template <wire_type_mismatch_policy MismatchPolicy>
__device__ bool scan_message_field_locations(message_scan_context context,
                                             auto&& lookup_desc_idx,
                                             auto&& is_repeated_field,
                                             auto&& get_expected_wire_type,
                                             auto&& on_singular,
                                             auto&& on_repeated)
{
  auto const* msg_base = context.begin;
  auto const* msg_end  = context.end;
  auto* error_flag     = context.error;
  for (uint8_t const* cur = msg_base; cur < msg_end;) {
    proto_tag tag;
    if (!decode_tag(cur, msg_end, tag, error_flag)) return false;
    int const wt = tag.wire_type;

    if (int f = lookup_desc_idx(tag.field_number); f >= 0) {
      if (is_repeated_field(f)) {
        if (!on_repeated(f, cur, msg_end, msg_base, wt)) { return false; }
      } else if (wt != get_expected_wire_type(f)) {
        if constexpr (MismatchPolicy != wire_type_mismatch_policy::skip) {
          set_error_once(error_flag, protobuf_error::WIRE_TYPE);
        }
        if constexpr (MismatchPolicy == wire_type_mismatch_policy::report_error) {
          return false;
        } else if constexpr (MismatchPolicy == wire_type_mismatch_policy::report_error_and_skip) {
          if (context.row_invalid != nullptr) { *context.row_invalid = true; }
        }
      } else {
        int const data_offset = static_cast<int>(cur - msg_base);
        field_location location;
        if (wt == wire_type_value(proto_wire_type::LEN)) {
          // Length-delimited: skip past the length prefix and record (data offset, data length).
          uint64_t len;
          int len_bytes;
          if (!read_varint(cur, msg_end, len, len_bytes)) {
            set_error_once(error_flag, protobuf_error::VARINT);
            return false;
          }
          if (len > static_cast<uint64_t>(msg_end - cur - len_bytes) ||
              len > static_cast<uint64_t>(cuda::std::numeric_limits<int>::max())) {
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
          int field_size = get_wire_type_size(wt, cur, msg_end);
          if (field_size < 0) {
            set_error_once(error_flag, protobuf_error::FIELD_SIZE);
            return false;
          }
          location = {data_offset, field_size};
        }
        if (!on_singular(f, cur, msg_end, location)) { return false; }
      }
    }

    // Advance to the next field regardless of whether this one matched the schema.
    uint8_t const* next;
    if (!skip_field(cur, msg_end, tag.field_number, wt, context.max_group_depth, next)) {
      set_error_once(error_flag, protobuf_error::SKIP);
      return false;
    }
    cur = next;
  }
  return true;
}

/**
 * Top-level field scanner: one thread per row records each requested top-level field's location
 * via the shared `scan_message_field_locations`. Null rows and out-of-bounds messages leave the
 * row's locations as {-1, 0}; in permissive mode malformed rows are flagged for nulling.
 */
CUDF_KERNEL void scan_all_fields_kernel(cudf::column_device_view const d_in,
                                        field_scan_view fields,
                                        protobuf_error* error_flag,
                                        bool* row_has_invalid_data)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  cudf::lists_column_device_view in{d_in};
  if (row >= in.size()) return;

  auto mark_row_error = [&]() { set_row_invalid(row_has_invalid_data, row); };

  field_location* field_locations = fields.locations + flat_index(row, fields.size, 0);
  for (int f = 0; f < fields.size; f++) {
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

  auto lookup_desc_idx = [&](int fn) {
    return lookup_field(
      fn, fields.field_number_lookup, fields.lookup_size, fields.size, [&](int f, int n) {
        return fields.descriptors[f].field_number == n;
      });
  };
  auto is_repeated_field      = [&](int f) { return fields.descriptors[f].is_repeated; };
  auto get_expected_wire_type = [&](int f) { return fields.descriptors[f].expected_wire_type; };
  auto record_singular =
    [&](int f, uint8_t const* value_start, uint8_t const* value_end, field_location location) {
      auto const& descriptor = fields.descriptors[f];
      bool recognized;
      if (!is_recognized_enum_value(descriptor, value_start, value_end, error_flag, recognized)) {
        return false;
      }
      if (!recognized) {
        if (row_has_invalid_data != nullptr) {
          mark_row_error();
        } else {
          set_error_once(fields.deferred_enum_error, protobuf_error::INVALID_ENUM);
        }
      }
      if (recognized) {
        // Last recognized value wins; unknown proto2 enum occurrences are ignored.
        field_locations[f] = location;
      }
      return true;
    };
  // Top-level scalar descriptors are never repeated, so the repeated handler is unreachable.
  auto unreachable_repeated = [](int, uint8_t const*, uint8_t const*, uint8_t const*, int) {
    return true;
  };
  if (!scan_message_field_locations<wire_type_mismatch_policy::report_error>(
        {msg_base, msg_end, error_flag, nullptr, PROTOBUF_JAVA_RECURSION_LIMIT},
        lookup_desc_idx,
        is_repeated_field,
        get_expected_wire_type,
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
                                      uint8_t const* msg_end,
                                      uint8_t const* msg_base,
                                      int wt,
                                      int expected_wt,
                                      protobuf_error* error_flag,
                                      F&& f)
{
  bool is_packed = (wt == wire_type_value(proto_wire_type::LEN) &&
                    expected_wt != wire_type_value(proto_wire_type::LEN));

  if (!is_packed && wt != expected_wt) {
    if constexpr (MismatchPolicy == wire_type_mismatch_policy::skip) {
      return true;
    } else {
      set_error_once(error_flag, protobuf_error::WIRE_TYPE);
      return MismatchPolicy == wire_type_mismatch_policy::report_error_and_skip;
    }
  }

  if (is_packed) {
    uint64_t packed_len;
    int len_bytes;
    if (!read_varint(cur, msg_end, packed_len, len_bytes)) {
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
      case wire_type_value(proto_wire_type::VARINT): {
        // `vbytes` is set inside the loop body before `p += vbytes` runs (the advance step
        // happens after each body execution), but we initialize it defensively to silence a
        // potential "used before set" warning. `read_varint` validates the varint stays
        // within `packed_end` (the packed payload's end), not `msg_end` — switching to a
        // generic skip helper here would over-read past the packed buffer.
        int vbytes = cuda::std::numeric_limits<int>::max();
        for (uint8_t const* p = packed_start; p < packed_end; p += vbytes) {
          int32_t elem_offset = static_cast<int32_t>(p - msg_base);
          uint64_t dummy;
          if (!read_varint(p, packed_end, dummy, vbytes)) {
            set_error_once(error_flag, protobuf_error::VARINT);
            return false;
          }
          if (!f(elem_offset, vbytes)) return false;
        }
        break;
      }
      case wire_type_value(proto_wire_type::I32BIT):
      case wire_type_value(proto_wire_type::I64BIT): {
        int const width = (expected_wt == wire_type_value(proto_wire_type::I32BIT)) ? 4 : 8;
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

  auto lookup_desc_idx = [&](int fn) {
    return lookup_field(
      fn, fields.field_number_lookup, fields.lookup_size, fields.size, [&](int f, int n) {
        return fields.descriptors[f].field_number == n;
      });
  };
  auto is_repeated_field      = [&](int f) { return fields.descriptors[f].is_repeated; };
  auto get_expected_wire_type = [&](int f) { return fields.descriptors[f].expected_wire_type; };
  auto record_singular        = []([[maybe_unused]] int f,
                            [[maybe_unused]] uint8_t const* value_start,
                            [[maybe_unused]] uint8_t const* value_end,
                            [[maybe_unused]] field_location location) { return true; };
  auto validate_repeated =
    [&](int f, uint8_t const* cur, uint8_t const* end, uint8_t const* base, int wire_type) {
      auto ignore_occurrence = []([[maybe_unused]] int32_t offset,
                                  [[maybe_unused]] int32_t length) { return true; };
      return walk_repeated_element<wire_type_mismatch_policy::skip>(
        cur, end, base, wire_type, get_expected_wire_type(f), error_flag, ignore_occurrence);
    };

  auto const* fragment_begin = locations.input.message_data + fragment_start;
  auto const* fragment_limit = locations.input.message_data + fragment_end;
  if (!scan_message_field_locations<wire_type_mismatch_policy::skip>(
        {fragment_begin, fragment_limit, error_flag, nullptr, max_group_depth},
        lookup_desc_idx,
        is_repeated_field,
        get_expected_wire_type,
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
 * Optional lookup tables (fn_to_rep_idx, fn_to_nested_idx) provide O(1) field_number
 * to local index mapping. When nullptr, falls back to linear search.
 */
CUDF_KERNEL void count_repeated_fields_kernel(cudf::column_device_view const d_in,
                                              device_schema_view schema,
                                              repeated_field_count_view repeated,
                                              nested_field_location_view nested,
                                              protobuf_error* error_flag,
                                              bool* row_has_invalid_data)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  cudf::lists_column_device_view in{d_in};
  if (row >= in.size()) return;
  auto mark_row_error = [&]() { set_row_invalid(row_has_invalid_data, row); };

  // Initialize repeated counts to 0
  for (int f = 0; f < repeated.size; f++) {
    repeated.info[flat_index(row, repeated.size, f)] = {0};
  }

  // Initialize nested locations and occurrence counts.
  for (int f = 0; f < nested.size; f++) {
    nested.locations[flat_index(row, nested.size, f)]       = {-1, 0};
    nested.occurrence_info[flat_index(row, nested.size, f)] = {0};
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

  // Schema-aware (field_number, depth) lookup. Forwards to `lookup_field` with a
  // predicate that follows the `field_indices` indirection into `schema` and also filters
  // by `depth_level`, since this kernel processes nested schemas where the same field
  // number can appear at multiple depths.
  auto lookup_field_idx = [&](int fn,
                              int const* fn_to_idx,
                              int fn_tbl_size,
                              int const* field_indices,
                              int num_fields_at_depth) -> int {
    return lookup_field(fn, fn_to_idx, fn_tbl_size, num_fields_at_depth, [&](int local_i, int fn) {
      auto const& field_schema = schema.fields[field_indices[local_i]];
      return field_schema.field_number == fn && field_schema.depth == schema.depth;
    });
  };

  // Use one descriptor-index space for the shared scanner: repeated fields first, then nested
  // message fields. A top-level field belongs to exactly one of these groups.
  auto lookup_desc_idx = [&](int fn) {
    int const repeated_idx = lookup_field_idx(fn,
                                              repeated.field_number_lookup,
                                              repeated.lookup_size,
                                              repeated.schema_indices,
                                              repeated.size);
    if (repeated_idx >= 0) { return repeated_idx; }
    int const nested_idx = lookup_field_idx(
      fn, nested.field_number_lookup, nested.lookup_size, nested.schema_indices, nested.size);
    return nested_idx >= 0 ? repeated.size + nested_idx : -1;
  };
  auto is_repeated_field      = [&](int f) { return f < repeated.size; };
  auto get_expected_wire_type = [&](int f) {
    return f < repeated.size ? schema.fields[repeated.schema_indices[f]].wire_type
                             : wire_type_value(proto_wire_type::LEN);
  };
  auto record_nested = [&](int f,
                           [[maybe_unused]] uint8_t const* value_start,
                           [[maybe_unused]] uint8_t const* value_end,
                           field_location location) {
    int const nested_idx                                       = f - repeated.size;
    nested.locations[flat_index(row, nested.size, nested_idx)] = location;
    auto& info = nested.occurrence_info[flat_index(row, nested.size, nested_idx)];
    if (++info.count == 2) { atomicExch(nested.multiple_message_fields + nested_idx, 1); }
    return true;
  };
  auto count_repeated =
    [&](int f, uint8_t const* cur, uint8_t const* end, uint8_t const* base, int wire_type) {
      auto& info        = repeated.info[flat_index(row, repeated.size, f)];
      auto count_action = [&info]([[maybe_unused]] int32_t off, [[maybe_unused]] int32_t len) {
        info.count++;
        return true;
      };
      return walk_repeated_element<wire_type_mismatch_policy::report_error>(
        cur, end, base, wire_type, get_expected_wire_type(f), error_flag, count_action);
    };

  auto* row_invalid = row_has_invalid_data != nullptr ? row_has_invalid_data + row : nullptr;
  if (!scan_message_field_locations<wire_type_mismatch_policy::report_error_and_skip>(
        {msg_base, msg_end, error_flag, row_invalid, PROTOBUF_JAVA_RECURSION_LIMIT},
        lookup_desc_idx,
        is_repeated_field,
        get_expected_wire_type,
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
  // Defense-in-depth: host-side validation enforces this cap, so the check is unreachable on a
  // correct config. Keep it in release builds because overrunning `write_idx` below is silent UB.
  if (fields.size > MAX_REPEATED_FIELDS_PER_KERNEL) {
    set_error_once(error_flag, protobuf_error::SCHEMA_TOO_LARGE);
    return false;
  }

  int write_idx[MAX_REPEATED_FIELDS_PER_KERNEL];
  for (int f = 0; f < fields.size; f++) {
    write_idx[f] = fields.descriptors[f].row_offsets[row];
  }

  auto lookup_by_fn = [&](int fn) {
    return lookup_field(
      fn, fields.field_number_lookup, fields.lookup_size, fields.size, [&](int f, int) {
        return fields.descriptors[f].field_number == fn;
      });
  };
  auto is_repeated_field      = []([[maybe_unused]] int f) { return true; };
  auto get_expected_wire_type = [&](int f) { return fields.descriptors[f].wire_type; };
  auto unreachable_singular   = []([[maybe_unused]] int f,
                                 [[maybe_unused]] uint8_t const* value_start,
                                 [[maybe_unused]] uint8_t const* value_end,
                                 [[maybe_unused]] field_location location) { return true; };

  auto const row_i32 = static_cast<int32_t>(row);
  auto on_repeated_scan =
    [&](int f, uint8_t const* cur, uint8_t const* me, uint8_t const* mb, int wt) {
      auto* occs       = fields.descriptors[f].occurrences;
      int& wi          = write_idx[f];
      int const we     = fields.descriptors[f].row_offsets[row + 1];
      auto scan_action = [&](int32_t off, int32_t len) {
        if (wi >= we) {
          set_error_once(error_flag, protobuf_error::REPEATED_COUNT_MISMATCH);
          return false;
        }
        occs[wi] = {row_i32, off, len};
        wi++;
        return true;
      };
      return walk_repeated_element<MismatchPolicy>(
        cur, me, mb, wt, get_expected_wire_type(f), error_flag, scan_action);
    };

  if (!scan_message_field_locations<MismatchPolicy>(
        {msg_base, msg_end, error_flag, nullptr, max_group_depth},
        lookup_by_fn,
        is_repeated_field,
        get_expected_wire_type,
        unreachable_singular,
        on_repeated_scan)) {
    return false;
  }

  for (int f = 0; f < fields.size; f++) {
    if (write_idx[f] != fields.descriptors[f].row_offsets[row + 1]) {
      set_error_once(error_flag, protobuf_error::REPEATED_COUNT_MISMATCH);
      return false;
    }
  }
  return true;
}

template <wire_type_mismatch_policy MismatchPolicy>
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
    scan_all_field_occurrences_in_message<MismatchPolicy>(
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

  field_location* field_locations = fields.locations + flat_index(row, fields.size, 0);
  for (int f = 0; f < fields.size; f++) {
    field_locations[f] = {-1, 0};
    if (fields.occurrence_info != nullptr) {
      fields.occurrence_info[flat_index(row, fields.size, f)] = {0};
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

  auto lookup_desc_idx = [&](int fn) {
    return lookup_field(
      fn, fields.field_number_lookup, fields.lookup_size, fields.size, [&](int f, int n) {
        return fields.descriptors[f].field_number == n;
      });
  };
  auto is_repeated_field      = [&](int f) { return fields.descriptors[f].is_repeated; };
  auto get_expected_wire_type = [&](int f) { return fields.descriptors[f].expected_wire_type; };
  auto record_singular =
    [&](int f, uint8_t const* value_start, uint8_t const* value_end, field_location location) {
      auto const& descriptor = fields.descriptors[f];
      bool recognized;
      if (!is_recognized_enum_value(descriptor, value_start, value_end, error_flag, recognized)) {
        return false;
      }
      if (recognized) {
        field_locations[f] = location;
        if (descriptor.is_message) {
          auto& info = fields.occurrence_info[flat_index(row, fields.size, f)];
          if (++info.count == 2) { atomicExch(fields.multiple_message_fields + f, 1); }
        }
      }
      return true;
    };
  auto validate_repeated =
    [&](int f, uint8_t const* cur, uint8_t const* msg_end, uint8_t const* msg_base, int wt) {
      auto const expected_wire_type = get_expected_wire_type(f);
      auto count_occurrence = [&]([[maybe_unused]] int32_t off, [[maybe_unused]] int32_t len) {
        if (fields.occurrence_info != nullptr) {
          fields.occurrence_info[flat_index(row, fields.size, f)].count++;
        }
        return true;
      };
      return walk_repeated_element<wire_type_mismatch_policy::skip>(
        cur, msg_end, msg_base, wt, expected_wire_type, error_flag, count_occurrence);
    };

  if (!scan_message_field_locations<wire_type_mismatch_policy::skip>(
        {nested_start, nested_end, error_flag, nullptr, max_group_depth},
        lookup_desc_idx,
        is_repeated_field,
        get_expected_wire_type,
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
    scan_all_field_occurrences_in_message<wire_type_mismatch_policy::skip>(
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
  if (row >= input.num_rows) return;
  if (input.input_null_mask != nullptr &&
      !cudf::bit_is_set(input.input_null_mask, row + input.input_offset)) {
    return;
  }
  if (input.parent_locations != nullptr && input.parent_locations[row].offset < 0) return;

  for (int f = 0; f < num_fields; f++) {
    if (is_required[f] != 0 && input.locations[flat_index(row, num_fields, f)].offset < 0) {
      if (row_force_null != nullptr) {
        auto const top_row =
          input.top_row_indices != nullptr ? input.top_row_indices[row] : static_cast<int32_t>(row);
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
 * If a value is not in the valid set:
 * 1. Mark the value as invalid.
 * 2. Record it for the caller's root/nested enum policy.
 *
 * The valid_values array must be sorted for binary search.
 *
 * @note Time complexity: O(log(num_valid_values)) per row.
 */
CUDF_KERNEL void validate_enum_values_kernel(enum_value_device_view input,
                                             bool* item_has_invalid_enum,
                                             enum_domain_device_view domain)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= input.size) return;

  // Skip if already invalid (field was missing) - missing field is not an enum error
  if (!input.valid[row]) return;

  if (enum_binary_search(domain.valid_values, domain.size, input.values[row]) < 0) {
    input.valid[row]           = false;
    item_has_invalid_enum[row] = true;
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

void set_error_once_async(protobuf_error* error_flag,
                          protobuf_error error,
                          rmm::cuda_stream_view stream)
{
  set_error_if_unset_kernel<<<1, 1, 0, stream.value()>>>(error_flag, error);
  CUDF_CUDA_TRY(cudaPeekAtLastError());
}

void launch_scan_all_fields(cudf::column_device_view const& d_in,
                            field_scan_view fields,
                            protobuf_error* error_flag,
                            bool* row_has_invalid_data,
                            rmm::cuda_stream_view stream)
{
  auto const num_rows = d_in.size();
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  scan_all_fields_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    d_in, fields, error_flag, row_has_invalid_data);
}

void launch_count_repeated_fields(cudf::column_device_view const& d_in,
                                  device_schema_view schema,
                                  repeated_field_count_view repeated,
                                  nested_field_location_view nested,
                                  protobuf_error* error_flag,
                                  bool* row_has_invalid_data,
                                  rmm::cuda_stream_view stream)
{
  auto const num_rows = d_in.size();
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  count_repeated_fields_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    d_in, schema, repeated, nested, error_flag, row_has_invalid_data);
}

void launch_scan_all_field_occurrences(cudf::column_device_view const& d_in,
                                       field_occurrence_scan_view fields,
                                       protobuf_error* error_flag,
                                       rmm::cuda_stream_view stream)
{
  auto const num_rows = d_in.size();
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  scan_all_field_occurrences_kernel<wire_type_mismatch_policy::report_error>
    <<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(d_in, fields, error_flag);
}

void launch_scan_singular_message_occurrences(cudf::column_device_view const& d_in,
                                              field_occurrence_scan_view fields,
                                              protobuf_error* error_flag,
                                              rmm::cuda_stream_view stream)
{
  auto const num_rows = d_in.size();
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  scan_all_field_occurrences_kernel<wire_type_mismatch_policy::report_error_and_skip>
    <<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(d_in, fields, error_flag);
}

void launch_extract_strided_locations(field_location const* nested_locations,
                                      int field_idx,
                                      int num_fields,
                                      field_location* parent_locs,
                                      int num_rows,
                                      rmm::cuda_stream_view stream)
{
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  extract_strided_locations_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    nested_locations, field_idx, num_fields, parent_locs, num_rows);
}

void launch_scan_nested_message_fields(protobuf_input_view input,
                                       nested_parent_view parent,
                                       field_scan_view fields,
                                       protobuf_error* error_flag,
                                       bool* row_has_invalid_data,
                                       int recursion_depth,
                                       rmm::cuda_stream_view stream)
{
  if (input.num_rows == 0) return;
  auto const max_group_depth = PROTOBUF_JAVA_RECURSION_LIMIT - recursion_depth;
  auto const blocks =
    static_cast<int>((input.num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  scan_nested_message_fields_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    input, parent, fields, error_flag, row_has_invalid_data, max_group_depth);
}

void launch_scan_all_field_occurrences_in_nested(protobuf_input_view input,
                                                 nested_parent_view parent,
                                                 field_occurrence_scan_view fields,
                                                 protobuf_error* error_flag,
                                                 int recursion_depth,
                                                 rmm::cuda_stream_view stream)
{
  if (input.num_rows == 0) return;
  auto const max_group_depth = PROTOBUF_JAVA_RECURSION_LIMIT - recursion_depth;
  auto const blocks =
    static_cast<int>((input.num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  scan_all_field_occurrences_in_nested_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    input, parent, fields, error_flag, max_group_depth);
}

void launch_validate_message_fragments(message_fragment_location_provider locations,
                                       message_validation_view fields,
                                       int num_fragments,
                                       bool* invalid_rows,
                                       bool* row_has_invalid_data,
                                       protobuf_error* error_flag,
                                       int recursion_depth,
                                       rmm::cuda_stream_view stream)
{
  if (num_fragments == 0) return;
  auto const max_group_depth = PROTOBUF_JAVA_RECURSION_LIMIT - recursion_depth;
  auto const blocks =
    static_cast<int>((num_fragments + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  validate_message_fragments_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    locations,
    fields,
    num_fragments,
    invalid_rows,
    row_has_invalid_data,
    error_flag,
    max_group_depth);
}

void launch_compute_grandchild_parent_locations(nested_location_provider loc_provider,
                                                field_location* gc_parent_locs,
                                                int num_rows,
                                                protobuf_error* error_flag,
                                                rmm::cuda_stream_view stream)
{
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  compute_grandchild_parent_locations_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    loc_provider, gc_parent_locs, num_rows, error_flag);
}

void launch_compute_virtual_parents_for_nested_repeated(protobuf_input_view input,
                                                        nested_parent_view parent,
                                                        repeated_field_work const& work,
                                                        cudf::size_type* virtual_row_offsets,
                                                        field_location* virtual_parent_locs,
                                                        protobuf_decode_runtime_context decode_ctx,
                                                        rmm::cuda_stream_view stream)
{
  if (work.total_count == 0) return;
  auto const blocks =
    static_cast<int>((work.total_count + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  compute_virtual_parents_for_nested_repeated_kernel<<<blocks,
                                                       THREADS_PER_BLOCK,
                                                       0,
                                                       stream.value()>>>(work.occurrences->data(),
                                                                         input.row_offsets,
                                                                         parent.locations,
                                                                         virtual_row_offsets,
                                                                         virtual_parent_locs,
                                                                         work.total_count,
                                                                         decode_ctx.error->data());
}

void launch_compute_msg_locations_from_occurrences(protobuf_input_view input,
                                                   repeated_field_work const& work,
                                                   field_location* msg_locs,
                                                   cudf::size_type* msg_row_offsets,
                                                   protobuf_decode_runtime_context decode_ctx,
                                                   rmm::cuda_stream_view stream)
{
  if (work.total_count == 0) return;
  auto const blocks =
    static_cast<int>((work.total_count + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  compute_msg_locations_from_occurrences_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    work.occurrences->data(),
    input.row_offsets,
    input.base_offset,
    msg_locs,
    msg_row_offsets,
    work.total_count,
    decode_ctx.error->data());
}

void launch_validate_enum_values(enum_value_device_view input,
                                 bool* item_has_invalid_enum,
                                 enum_domain_device_view domain,
                                 rmm::cuda_stream_view stream)
{
  if (input.size == 0) return;
  auto const blocks = static_cast<int>((input.size + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  validate_enum_values_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    input, item_has_invalid_enum, domain);
}

void launch_compute_enum_string_lengths(enum_value_device_view input,
                                        enum_string_lookup_device_view lookup,
                                        int32_t* lengths,
                                        rmm::cuda_stream_view stream)
{
  if (input.size == 0) return;
  auto const blocks = static_cast<int>((input.size + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  compute_enum_string_lengths_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    input, lookup, lengths);
}

void launch_copy_enum_string_chars(enum_value_device_view input,
                                   enum_string_lookup_device_view lookup,
                                   int32_t const* output_offsets,
                                   char* out_chars,
                                   rmm::cuda_stream_view stream)
{
  if (input.size == 0) return;
  auto const blocks = static_cast<int>((input.size + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  copy_enum_string_chars_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    input, lookup, output_offsets, out_chars);
}

void maybe_check_required_fields(required_field_input_view input,
                                 std::vector<int> const& field_indices,
                                 std::vector<nested_field_descriptor> const& schema,
                                 protobuf_decode_runtime_context decode_ctx,
                                 rmm::cuda_stream_view stream)
{
  if (input.num_rows == 0 || field_indices.empty()) { return; }

  // Stream-ordered pinned deallocation keeps this staging safe without a local sync.
  bool has_required = false;
  auto h_is_required =
    cudf::detail::make_pinned_vector_async<uint8_t>(field_indices.size(), stream);
  for (size_t i = 0; i < field_indices.size(); ++i) {
    h_is_required[i] = schema[field_indices[i]].is_required ? 1 : 0;
    has_required |= (h_is_required[i] != 0);
  }
  if (!has_required) { return; }

  auto d_is_required = cudf::detail::make_device_uvector_async(
    h_is_required, stream, cudf::get_current_device_resource_ref());

  auto const blocks =
    static_cast<int>((input.num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  auto* row_force_null =
    decode_ctx.row_force_null != nullptr && !decode_ctx.row_force_null->is_empty()
      ? decode_ctx.row_force_null->data()
      : nullptr;
  check_required_fields_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    input,
    d_is_required.data(),
    static_cast<int>(field_indices.size()),
    row_force_null,
    decode_ctx.error->data());
}

namespace {

void apply_invalid_enum_policy(rmm::device_uvector<bool> const& item_invalid,
                               protobuf_decode_runtime_context decode_ctx,
                               protobuf_value_domain_view value_domain,
                               rmm::cuda_stream_view stream)
{
  if (value_domain.size == 0 || value_domain.enum_scope == enum_error_scope::local) return;

  CUDF_EXPECTS(decode_ctx.row_force_null != nullptr,
               "enum validation requires a row-invalid buffer");
  CUDF_EXPECTS(decode_ctx.error != nullptr && decode_ctx.error->size() == 1,
               "enum validation requires a single error flag");
  CUDF_EXPECTS(
    decode_ctx.deferred_enum_error != nullptr && decode_ctx.deferred_enum_error->size() == 1,
    "enum validation requires a single deferred enum error flag");

  auto const scratch_mr = cudf::get_current_device_resource_ref();
  if (decode_ctx.fail_on_errors) {
    thrust::for_each(rmm::exec_policy_nosync(stream, scratch_mr),
                     thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(value_domain.size),
                     [item_invalid = item_invalid.data(),
                      error        = decode_ctx.deferred_enum_error->data()] __device__(int idx) {
                       if (item_invalid[idx]) {
                         set_error_once(error, protobuf_error::INVALID_ENUM);
                       }
                     });
    return;
  }

  auto& row_invalid = *decode_ctx.row_force_null;
  CUDF_EXPECTS(!row_invalid.is_empty(), "PERMISSIVE enum validation requires a row-invalid buffer");
  if (value_domain.top_row_indices == nullptr) {
    CUDF_EXPECTS(static_cast<size_t>(value_domain.size) <= row_invalid.size(),
                 "enum invalid-row propagation exceeded row buffer");
    thrust::transform(rmm::exec_policy_nosync(stream, scratch_mr),
                      row_invalid.begin(),
                      row_invalid.begin() + value_domain.size,
                      item_invalid.begin(),
                      row_invalid.begin(),
                      [] __device__(bool row_is_invalid, bool item_is_invalid) {
                        return row_is_invalid || item_is_invalid;
                      });
    return;
  }

  // Multiple items may share the same `top_row_indices[idx]` (e.g. several occurrences of a
  // packed repeated enum within one row), so concurrent threads can race on the same byte.
  // Although every racing write stores the same value (`true`), non-atomic concurrent writes
  // to the same address are UB under the CUDA memory model. Use atomic_ref like set_error_once.
  thrust::for_each(
    rmm::exec_policy_nosync(stream, scratch_mr),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(value_domain.size),
    [item_invalid    = item_invalid.data(),
     top_row_indices = value_domain.top_row_indices,
     row_invalid     = row_invalid.data()] __device__(int idx) {
      if (item_invalid[idx]) {
        cuda::atomic_ref<bool, cuda::thread_scope_device> ref(row_invalid[top_row_indices[idx]]);
        ref.store(true, cuda::memory_order_relaxed);
      }
    });
}

}  // namespace

void validate_enum_and_apply_policy(rmm::device_uvector<int32_t> const& values,
                                    rmm::device_uvector<bool>& valid,
                                    enum_domain_device_view enum_domain,
                                    protobuf_decode_runtime_context decode_ctx,
                                    protobuf_value_domain_view value_domain,
                                    rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(value_domain.size >= 0, "enum value count must be non-negative");
  CUDF_EXPECTS(values.size() == static_cast<size_t>(value_domain.size),
               "enum values size must match value domain");
  CUDF_EXPECTS(valid.size() == static_cast<size_t>(value_domain.size),
               "enum validity size must match value domain");
  if (value_domain.size == 0 || enum_domain.size == 0) return;
  CUDF_EXPECTS(enum_domain.valid_values != nullptr, "enum validation requires valid enum values");

  auto const scratch_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<bool> item_invalid(value_domain.size, stream, scratch_mr);
  thrust::fill(
    rmm::exec_policy_nosync(stream, scratch_mr), item_invalid.begin(), item_invalid.end(), false);
  launch_validate_enum_values(
    {values.data(), valid.data(), value_domain.size}, item_invalid.data(), enum_domain, stream);
  apply_invalid_enum_policy(item_invalid, decode_ctx, value_domain, stream);
}

void validate_enum_and_apply_policy(rmm::device_uvector<int32_t> const& values,
                                    rmm::device_uvector<bool>& valid,
                                    cudf::detail::host_vector<int32_t> const& valid_enums,
                                    protobuf_decode_runtime_context decode_ctx,
                                    protobuf_value_domain_view value_domain,
                                    rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(value_domain.size >= 0, "enum value count must be non-negative");
  CUDF_EXPECTS(values.size() == static_cast<size_t>(value_domain.size),
               "enum values size must match value domain");
  CUDF_EXPECTS(valid.size() == static_cast<size_t>(value_domain.size),
               "enum validity size must match value domain");
  if (value_domain.size == 0 || valid_enums.empty()) return;

  auto d_valid_enums = cudf::detail::make_device_uvector_async(
    valid_enums, stream, cudf::get_current_device_resource_ref());
  validate_enum_and_apply_policy(values,
                                 valid,
                                 {d_valid_enums.data(), static_cast<int>(d_valid_enums.size())},
                                 decode_ctx,
                                 value_domain,
                                 stream);
}

}  // namespace spark_rapids_jni::protobuf::detail
