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

enum class wire_type_mismatch_policy { report_error, report_error_and_skip, skip };

/**
 * Share one message walk across top-level, nested, and occurrence scanners while leaving lookup,
 * output layout, and mismatch policy with each caller. Either callback returns false to abort.
 */
struct message_scan_context {
  uint8_t const* begin;
  uint8_t const* end;
  protobuf_error* error;
  bool* row_invalid;
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
    if (!skip_field(cur, msg_end, wt, next)) {
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

  auto mark_row_error = [&]() {
    if (row_has_invalid_data != nullptr) { row_has_invalid_data[row] = true; }
  };

  auto* field_locations =
    fields.lookup.size > 0 ? fields.locations + flat_index(row, fields.lookup.size, 0) : nullptr;
  for (int f = 0; f < fields.lookup.size; f++) {
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

  auto lookup_desc_idx        = [&](int fn) { return lookup_field(fn, fields.lookup); };
  auto is_repeated_field      = [&](int f) { return fields.lookup.data[f].is_repeated; };
  auto get_expected_wire_type = [&](int f) { return fields.lookup.data[f].expected_wire_type; };
  auto record_singular        = [&](int f,
                             [[maybe_unused]] uint8_t const* value_start,
                             [[maybe_unused]] uint8_t const* value_end,
                             field_location location) {
    field_locations[f] = location;
    return true;
  };
  // Top-level scalar descriptors are never repeated, so the repeated handler is unreachable.
  auto unreachable_repeated = [](int, uint8_t const*, uint8_t const*, uint8_t const*, int) {
    return true;
  };
  if (!scan_message_field_locations<wire_type_mismatch_policy::report_error>(
        {msg_base, msg_end, error_flag, nullptr},
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

// ============================================================================
// Pass 1b: Count repeated fields kernel
// ============================================================================

/**
 * Count occurrences of repeated fields in each row.
 * Also records locations of nested message fields for hierarchical processing.
 *
 * Optional lookup tables in the repeated and nested views provide O(1) field-number mapping.
 * A null lookup pointer falls back to linear search.
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
  auto mark_row_error = [&]() {
    if (row_has_invalid_data != nullptr) { row_has_invalid_data[row] = true; }
  };

  // Initialize repeated counts to 0
  for (int f = 0; f < repeated.schema_lookup.size; f++) {
    repeated.info[flat_index(row, repeated.schema_lookup.size, f)] = {0};
  }

  // Initialize nested locations to not found
  for (int f = 0; f < nested.schema_lookup.size; f++) {
    nested.locations[flat_index(row, nested.schema_lookup.size, f)] = {-1, 0};
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

  // The predicate follows each view's schema-index indirection and filters by depth because the
  // same field number can appear at multiple schema levels.
  auto lookup_field_idx = [&](int fn, lookup_view<int> table) {
    return lookup_field(fn, table, [&](int local_i, int fn) {
      auto const& field_schema = schema.fields[table.data[local_i]];
      return field_schema.field_number == fn && field_schema.depth == schema.depth;
    });
  };

  // Use one descriptor-index space for the shared scanner: repeated fields first, then nested
  // message fields. A field at this depth belongs to exactly one of these groups.
  auto lookup_desc_idx = [&](int fn) {
    int const repeated_idx = lookup_field_idx(fn, repeated.schema_lookup);
    if (repeated_idx >= 0) { return repeated_idx; }
    int const nested_idx = lookup_field_idx(fn, nested.schema_lookup);
    return nested_idx >= 0 ? repeated.schema_lookup.size + nested_idx : -1;
  };
  auto is_repeated_field      = [&](int f) { return f < repeated.schema_lookup.size; };
  auto get_expected_wire_type = [&](int f) {
    return f < repeated.schema_lookup.size ? schema.fields[repeated.schema_lookup.data[f]].wire_type
                                           : wire_type_value(proto_wire_type::LEN);
  };
  auto record_nested = [&](int f,
                           [[maybe_unused]] uint8_t const* value_start,
                           [[maybe_unused]] uint8_t const* value_end,
                           field_location location) {
    int const nested_idx = f - repeated.schema_lookup.size;
    nested.locations[flat_index(row, nested.schema_lookup.size, nested_idx)] = location;
    return true;
  };
  auto count_repeated =
    [&](int f, uint8_t const* cur, uint8_t const* end, uint8_t const* base, int wire_type) {
      auto& info        = repeated.info[flat_index(row, repeated.schema_lookup.size, f)];
      auto count_action = [&info]([[maybe_unused]] int32_t off, [[maybe_unused]] int32_t len) {
        info.count++;
        return true;
      };
      return walk_repeated_element<wire_type_mismatch_policy::report_error>(
        cur, end, base, wire_type, get_expected_wire_type(f), error_flag, count_action);
    };

  auto* row_invalid = row_has_invalid_data != nullptr ? row_has_invalid_data + row : nullptr;
  if (!scan_message_field_locations<wire_type_mismatch_policy::report_error_and_skip>(
        {msg_base, msg_end, error_flag, row_invalid},
        lookup_desc_idx,
        is_repeated_field,
        get_expected_wire_type,
        record_nested,
        count_repeated)) {
    mark_row_error();
  }
}

/**
 * Combined occurrence scan: scans each message once and writes occurrences for all selected
 * fields.
 */
template <wire_type_mismatch_policy MismatchPolicy>
__device__ bool scan_all_field_occurrences_in_message(uint8_t const* msg_base,
                                                      uint8_t const* msg_end,
                                                      field_occurrence_scan_view fields,
                                                      protobuf_error* error_flag,
                                                      cudf::size_type row)
{
  // Defense-in-depth: host-side validation enforces this cap, so the check is unreachable on a
  // correct config. Keep it in release builds because overrunning `write_idx` below is silent UB.
  if (fields.size > MAX_REPEATED_FIELDS_PER_KERNEL) {
    set_error_once(error_flag, protobuf_error::SCHEMA_TOO_LARGE);
    return false;
  }

  int write_idx[MAX_REPEATED_FIELDS_PER_KERNEL];
  for (int f = 0; f < fields.size; f++) {
    write_idx[f] = fields.data[f].row_offsets[row];
  }

  auto lookup_by_fn           = [&](int fn) { return lookup_field(fn, fields); };
  auto is_repeated_field      = []([[maybe_unused]] int f) { return true; };
  auto get_expected_wire_type = [&](int f) { return fields.data[f].wire_type; };
  auto ignore_singular        = []([[maybe_unused]] int f,
                            [[maybe_unused]] uint8_t const* value_start,
                            [[maybe_unused]] uint8_t const* value_end,
                            [[maybe_unused]] field_location location) { return true; };

  auto const row_i32 = static_cast<int32_t>(row);
  auto on_repeated_scan =
    [&](int f, uint8_t const* cur, uint8_t const* me, uint8_t const* mb, int wt) {
      auto* occs       = fields.data[f].occurrences;
      int& wi          = write_idx[f];
      int const we     = fields.data[f].row_offsets[row + 1];
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

  if (!scan_message_field_locations<MismatchPolicy>({msg_base, msg_end, error_flag, nullptr},
                                                    lookup_by_fn,
                                                    is_repeated_field,
                                                    get_expected_wire_type,
                                                    ignore_singular,
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
    scan_all_field_occurrences_in_message<wire_type_mismatch_policy::report_error>(
      bytes + start, bytes + end, fields, error_flag, row);
}

// ============================================================================
// Nested message scanning kernels
// ============================================================================

/**
 * Scan one nested message per parent row to locate singleton children and count repeated
 * children. Singleton locations use last-one-wins semantics; repeated occurrences are written
 * by the combined scan after their LIST offsets are available.
 */
CUDF_KERNEL void scan_nested_message_fields_kernel(protobuf_input_view input,
                                                   nested_parent_view parent,
                                                   field_scan_view fields,
                                                   protobuf_error* error_flag,
                                                   bool* row_has_invalid_data)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= input.num_rows) return;

  auto const top_row =
    parent.top_row_indices != nullptr ? parent.top_row_indices[row] : static_cast<int32_t>(row);
  auto mark_row_error = [&]() {
    if (row_has_invalid_data != nullptr) { row_has_invalid_data[top_row] = true; }
  };

  field_location* field_locations = fields.locations + flat_index(row, fields.lookup.size, 0);
  for (int f = 0; f < fields.lookup.size; f++) {
    field_locations[f] = {-1, 0};
    if (fields.repeated_info != nullptr) {
      fields.repeated_info[flat_index(row, fields.lookup.size, f)] = {0};
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

  auto lookup_desc_idx        = [&](int fn) { return lookup_field(fn, fields.lookup); };
  auto is_repeated_field      = [&](int f) { return fields.lookup.data[f].is_repeated; };
  auto get_expected_wire_type = [&](int f) { return fields.lookup.data[f].expected_wire_type; };
  auto record_singular        = [&](int f,
                             [[maybe_unused]] uint8_t const* value_start,
                             [[maybe_unused]] uint8_t const* value_end,
                             field_location location) {
    field_locations[f] = location;
    return true;
  };
  auto validate_repeated =
    [&](int f, uint8_t const* cur, uint8_t const* msg_end, uint8_t const* msg_base, int wt) {
      auto const expected_wire_type = get_expected_wire_type(f);
      auto count_occurrence = [&]([[maybe_unused]] int32_t off, [[maybe_unused]] int32_t len) {
        if (fields.repeated_info != nullptr) {
          fields.repeated_info[flat_index(row, fields.lookup.size, f)].count++;
        }
        return true;
      };
      return walk_repeated_element<wire_type_mismatch_policy::skip>(
        cur, msg_end, msg_base, wt, expected_wire_type, error_flag, count_occurrence);
    };

  if (!scan_message_field_locations<wire_type_mismatch_policy::skip>(
        {nested_start, nested_end, error_flag, nullptr},
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
                                                             protobuf_error* error_flag)
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
      row);
}

CUDF_KERNEL void compute_grandchild_parent_locations_kernel(field_location const* parent_locs,
                                                            field_location const* child_locs,
                                                            int child_idx,
                                                            int num_child_fields,
                                                            field_location* gc_parent_locs,
                                                            int num_rows,
                                                            protobuf_error* error_flag)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;

  nested_location_provider loc_provider{
    nullptr, 0, parent_locs, child_locs, child_idx, num_child_fields};
  gc_parent_locs[row] = loc_provider.get_rebased_child_location(row, error_flag);
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
  field_location const* locations,  // [num_rows * num_fields]
  uint8_t const* is_required,       // [num_fields] (1 = required, 0 = optional)
  int num_fields,
  int num_rows,
  cudf::bitmask_type const* input_null_mask,  // optional top-level input null mask
  cudf::size_type input_offset,               // bit offset for sliced top-level input
  field_location const* parent_locs,          // [num_rows] optional parent presence for nested rows
  bool* row_force_null,            // [top_level_num_rows] optional permissive row nulling
  int32_t const* top_row_indices,  // [num_rows] optional nested-row -> top-row mapping
  protobuf_error* error_flag)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;
  if (input_null_mask != nullptr && !cudf::bit_is_set(input_null_mask, row + input_offset)) {
    return;
  }
  if (parent_locs != nullptr && parent_locs[row].offset < 0) return;

  for (int f = 0; f < num_fields; f++) {
    if (is_required[f] != 0 && locations[flat_index(row, num_fields, f)].offset < 0) {
      if (row_force_null != nullptr) {
        auto const top_row =
          top_row_indices != nullptr ? top_row_indices[row] : static_cast<int32_t>(row);
        row_force_null[top_row] = true;
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
 * 1. Mark the field as invalid (valid[row] = false)
 * 2. Mark the row as having an invalid enum (row_has_invalid_enum[row] = true)
 *
 * This matches Spark CPU PERMISSIVE mode behavior: when an unknown enum value is
 * encountered, the entire struct row is set to null (not just the enum field).
 *
 * The valid_values array must be sorted for binary search.
 *
 * @note Time complexity: O(log(num_valid_values)) per row.
 */
CUDF_KERNEL void validate_enum_values_kernel(enum_value_device_view input,
                                             bool* row_has_invalid_enum,
                                             enum_domain_device_view domain)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= input.size) return;

  // Skip if already invalid (field was missing) - missing field is not an enum error
  if (!input.valid[row]) return;

  if (enum_binary_search(domain.valid_values, domain.size, input.values[row]) < 0) {
    input.valid[row] = false;
    // Also mark the row as having an invalid enum - this will null the entire struct row
    row_has_invalid_enum[row] = true;
  }
}

/**
 * Compute output UTF-8 length for enum-as-string rows.
 * Invalid/missing values produce length 0 (null row/field semantics handled by valid[] and
 * row_has_invalid_enum).
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
  scan_all_field_occurrences_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    d_in, fields, error_flag);
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
                                       rmm::cuda_stream_view stream)
{
  if (input.num_rows == 0) return;
  auto const blocks =
    static_cast<int>((input.num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  scan_nested_message_fields_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    input, parent, fields, error_flag, row_has_invalid_data);
}

void launch_scan_all_field_occurrences_in_nested(protobuf_input_view input,
                                                 nested_parent_view parent,
                                                 field_occurrence_scan_view fields,
                                                 protobuf_error* error_flag,
                                                 rmm::cuda_stream_view stream)
{
  if (input.num_rows == 0) return;
  auto const blocks =
    static_cast<int>((input.num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  scan_all_field_occurrences_in_nested_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    input, parent, fields, error_flag);
}

void launch_compute_grandchild_parent_locations(field_location const* parent_locs,
                                                field_location const* child_locs,
                                                int child_idx,
                                                int num_child_fields,
                                                field_location* gc_parent_locs,
                                                int num_rows,
                                                protobuf_error* error_flag,
                                                rmm::cuda_stream_view stream)
{
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  compute_grandchild_parent_locations_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    parent_locs, child_locs, child_idx, num_child_fields, gc_parent_locs, num_rows, error_flag);
}

void launch_validate_enum_values(enum_value_device_view input,
                                 bool* row_has_invalid_enum,
                                 enum_domain_device_view domain,
                                 rmm::cuda_stream_view stream)
{
  if (input.size == 0) return;
  auto const blocks = static_cast<int>((input.size + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  validate_enum_values_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    input, row_has_invalid_enum, domain);
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
  check_required_fields_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    input.locations,
    d_is_required.data(),
    static_cast<int>(field_indices.size()),
    input.num_rows,
    input.input_null_mask,
    input.input_offset,
    input.parent_locations,
    !decode_ctx.row_force_null->is_empty() ? decode_ctx.row_force_null->data() : nullptr,
    input.top_row_indices,
    decode_ctx.error->data());
}

void propagate_invalid_enum_flags_to_rows(rmm::device_uvector<bool> const& item_invalid,
                                          protobuf_decode_runtime_context decode_ctx,
                                          protobuf_value_domain_view value_domain,
                                          rmm::cuda_stream_view stream)
{
  auto& row_invalid            = *decode_ctx.row_force_null;
  auto const num_items         = value_domain.size;
  auto const top_row_indices   = value_domain.top_row_indices;
  auto const propagate_to_rows = decode_ctx.propagate_invalid_enum_rows;
  if (num_items == 0 || row_invalid.size() == 0 || !propagate_to_rows) return;

  auto const scratch_mr = cudf::get_current_device_resource_ref();
  if (top_row_indices == nullptr) {
    CUDF_EXPECTS(static_cast<size_t>(num_items) <= row_invalid.size(),
                 "enum invalid-row propagation exceeded row buffer");
    thrust::transform(rmm::exec_policy_nosync(stream, scratch_mr),
                      row_invalid.begin(),
                      row_invalid.begin() + num_items,
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
    thrust::make_counting_iterator(num_items),
    [item_invalid = item_invalid.data(),
     top_row_indices,
     row_invalid = row_invalid.data()] __device__(int idx) {
      if (item_invalid[idx]) {
        cuda::atomic_ref<bool, cuda::thread_scope_device> ref(row_invalid[top_row_indices[idx]]);
        ref.store(true, cuda::memory_order_relaxed);
      }
    });
}

void validate_enum_and_propagate_rows(rmm::device_uvector<int32_t> const& values,
                                      rmm::device_uvector<bool>& valid,
                                      enum_domain_device_view enum_domain,
                                      protobuf_decode_runtime_context decode_ctx,
                                      protobuf_value_domain_view value_domain,
                                      rmm::cuda_stream_view stream)
{
  if (value_domain.size == 0 || enum_domain.size == 0) return;

  auto const scratch_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<bool> item_invalid(value_domain.size, stream, scratch_mr);
  thrust::fill(
    rmm::exec_policy_nosync(stream, scratch_mr), item_invalid.begin(), item_invalid.end(), false);
  launch_validate_enum_values(
    {values.data(), valid.data(), value_domain.size}, item_invalid.data(), enum_domain, stream);

  propagate_invalid_enum_flags_to_rows(item_invalid, decode_ctx, value_domain, stream);
}

void validate_enum_and_propagate_rows(rmm::device_uvector<int32_t> const& values,
                                      rmm::device_uvector<bool>& valid,
                                      cudf::detail::host_vector<int32_t> const& valid_enums,
                                      protobuf_decode_runtime_context decode_ctx,
                                      protobuf_value_domain_view value_domain,
                                      rmm::cuda_stream_view stream)
{
  if (value_domain.size == 0 || valid_enums.empty()) return;

  auto const scratch_mr = cudf::get_current_device_resource_ref();
  auto d_valid_enums    = cudf::detail::make_device_uvector_async(valid_enums, stream, scratch_mr);
  validate_enum_and_propagate_rows(values,
                                   valid,
                                   {d_valid_enums.data(), static_cast<int>(valid_enums.size())},
                                   decode_ctx,
                                   value_domain,
                                   stream);
}

}  // namespace spark_rapids_jni::protobuf::detail
