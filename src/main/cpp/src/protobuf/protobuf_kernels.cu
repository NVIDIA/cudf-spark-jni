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

/**
 * Scan one message's bytes [msg_base, msg_end) once, recording the last-one-wins location
 * (relative to msg_base) of every matching non-repeated field into `out[field_index]`.
 *
 * Shared by the top-level (`scan_all_fields_kernel`), nested
 * (`scan_nested_message_fields_kernel`), and repeated-occurrence
 * (`scan_all_repeated_occurrences_kernel`) scanners. Only matched non-repeated fields are
 * written, so the caller is responsible for initializing `out` if it cares about the result;
 * pass `out == nullptr` when the scan is only validating (e.g. an all-repeated schema). The
 * caller also owns row-level error marking; this helper only sets `error_flag` and returns
 * false on the first parse error that leaves the cursor unsafe to advance.
 *
 * `lookup_desc_idx(field_number) -> int` maps a wire field number to its descriptor index (or -1);
 * callers supply it so this helper stays agnostic to whether a lookup table is used. Descriptor
 * attribute access is also caller-supplied so hot paths can use compact descriptor forms directly.
 *
 * Matched repeated fields are delegated to `on_repeated(f, cur, msg_end, msg_base, wt)` (f is the
 * matched descriptor index) which returns false on error; the handler derives its own expected
 * wire type from f when it needs one. Top-level scalars pass a no-op handler since their
 * descriptors are never repeated.
 */
__device__ bool scan_message_field_locations(uint8_t const* msg_base,
                                             uint8_t const* msg_end,
                                             field_location* out,
                                             protobuf_error* error_flag,
                                             auto&& lookup_desc_idx,
                                             auto&& is_repeated_field,
                                             auto&& get_expected_wire_type,
                                             auto&& on_repeated)
{
  for (uint8_t const* cur = msg_base; cur < msg_end;) {
    proto_tag tag;
    if (!decode_tag(cur, msg_end, tag, error_flag)) return false;
    int const wt = tag.wire_type;

    if (int f = lookup_desc_idx(tag.field_number); f >= 0) {
      if (is_repeated_field(f)) {
        if (!on_repeated(f, cur, msg_end, msg_base, wt)) { return false; }
      } else if (wt != get_expected_wire_type(f)) {
        set_error_once(error_flag, protobuf_error::WIRE_TYPE);
        return false;
      } else {
        // Record this field's location relative to the message start (last one wins).
        int const data_offset = static_cast<int>(cur - msg_base);
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
          if (out != nullptr) { out[f] = {data_location, static_cast<int32_t>(len)}; }
        } else {
          // Fixed-width / varint: record the offset and the wire-type-derived size.
          int field_size = get_wire_type_size(wt, cur, msg_end);
          if (field_size < 0) {
            set_error_once(error_flag, protobuf_error::FIELD_SIZE);
            return false;
          }
          if (out != nullptr) { out[f] = {data_offset, field_size}; }
        }
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
CUDF_KERNEL void scan_all_fields_kernel(
  cudf::column_device_view const d_in,
  field_descriptor const* field_descs,  // [num_fields]
  int num_fields,
  int const* field_lookup,    // direct-mapped lookup table (nullable)
  int field_lookup_size,      // size of lookup table (0 if null)
  field_location* locations,  // [num_rows * num_fields] row-major
  protobuf_error* error_flag,
  bool* row_has_invalid_data)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  cudf::lists_column_device_view in{d_in};
  if (row >= in.size()) return;

  auto mark_row_error = [&]() {
    if (row_has_invalid_data != nullptr) { row_has_invalid_data[row] = true; }
  };

  field_location* field_locations = locations + flat_index(row, num_fields, 0);
  for (int f = 0; f < num_fields; f++) {
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
    return lookup_field(fn, field_lookup, field_lookup_size, num_fields, [&](int f, int n) {
      return field_descs[f].field_number == n;
    });
  };
  auto is_repeated_field      = [&](int f) { return field_descs[f].is_repeated; };
  auto get_expected_wire_type = [&](int f) { return field_descs[f].expected_wire_type; };
  // Top-level scalar descriptors are never repeated, so the repeated handler is unreachable.
  auto unreachable_repeated = [](int, uint8_t const*, uint8_t const*, uint8_t const*, int) {
    return true;
  };
  if (!scan_message_field_locations(msg_base,
                                    msg_end,
                                    field_locations,
                                    error_flag,
                                    lookup_desc_idx,
                                    is_repeated_field,
                                    get_expected_wire_type,
                                    unreachable_repeated)) {
    mark_row_error();
  }
}

// ============================================================================
// Shared device functions for repeated field processing
// ============================================================================

enum class wire_type_mismatch_policy { report_error, skip };

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
      return false;
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
 * Optional lookup tables (fn_to_rep_idx, fn_to_nested_idx) provide O(1) field_number
 * to local index mapping. When nullptr, falls back to linear search.
 */
CUDF_KERNEL void count_repeated_fields_kernel(cudf::column_device_view const d_in,
                                              device_nested_field_descriptor const* schema,
                                              int num_fields,
                                              int depth_level,
                                              repeated_field_info* repeated_info,
                                              int num_repeated_fields,
                                              int const* repeated_field_indices,
                                              field_location* nested_locations,
                                              int num_nested_fields,
                                              int const* nested_field_indices,
                                              protobuf_error* error_flag,
                                              int const* fn_to_rep_idx,
                                              int fn_to_rep_size,
                                              int const* fn_to_nested_idx,
                                              int fn_to_nested_size)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  cudf::lists_column_device_view in{d_in};
  if (row >= in.size()) return;

  // Initialize repeated counts to 0
  for (int f = 0; f < num_repeated_fields; f++) {
    repeated_info[flat_index(row, num_repeated_fields, f)] = {0};
  }

  // Initialize nested locations to not found
  for (int f = 0; f < num_nested_fields; f++) {
    nested_locations[flat_index(row, num_nested_fields, f)] = {-1, 0};
  }

  if (in.nullable() && in.is_null(row)) return;

  auto const base   = in.offset_at(0);
  auto const child  = in.get_sliced_child();
  auto const* bytes = reinterpret_cast<uint8_t const*>(child.data<int8_t>());
  int32_t start     = in.offset_at(row) - base;
  int32_t end       = in.offset_at(row + 1) - base;
  if (!check_message_bounds(start, end, child.size(), error_flag)) return;

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
      auto const& field_schema = schema[field_indices[local_i]];
      return field_schema.field_number == fn && field_schema.depth == depth_level;
    });
  };

  for (uint8_t const* cur = msg_base; cur < msg_end;) {
    proto_tag tag;
    if (!decode_tag(cur, msg_end, tag, error_flag)) return;
    int const fn = tag.field_number;
    int const wt = tag.wire_type;

    if (int f = lookup_field_idx(
          fn, fn_to_rep_idx, fn_to_rep_size, repeated_field_indices, num_repeated_fields);
        f >= 0) {
      int schema_idx    = repeated_field_indices[f];
      auto& info        = repeated_info[flat_index(row, num_repeated_fields, f)];
      auto count_action = [&info]([[maybe_unused]] int32_t off, [[maybe_unused]] int32_t len) {
        info.count++;
        return true;
      };
      if (!walk_repeated_element<wire_type_mismatch_policy::report_error>(
            cur, msg_end, msg_base, wt, schema[schema_idx].wire_type, error_flag, count_action)) {
        return;
      }
    }

    // Check nested message fields at this depth
    if (int f = lookup_field_idx(
          fn, fn_to_nested_idx, fn_to_nested_size, nested_field_indices, num_nested_fields);
        f >= 0) {
      if (wt != wire_type_value(proto_wire_type::LEN)) {
        set_error_once(error_flag, protobuf_error::WIRE_TYPE);
        return;
      }
      uint64_t len;
      int len_bytes;
      if (!read_varint(cur, msg_end, len, len_bytes)) {
        set_error_once(error_flag, protobuf_error::VARINT);
        return;
      }
      if (len > static_cast<uint64_t>(msg_end - cur - len_bytes) ||
          len > static_cast<uint64_t>(cuda::std::numeric_limits<int>::max())) {
        set_error_once(error_flag, protobuf_error::OVERFLOW);
        return;
      }
      // cur - msg_base is bounded by the message length (<= INT32_MAX via check_message_bounds),
      // so this fits int32; checked_add_int32 still guards the offset + len_bytes addition. Matches
      // the LEN handling in scan_message_field_locations.
      int const data_offset = static_cast<int>(cur - msg_base);
      int32_t data_location;
      if (!checked_add_int32(data_offset, len_bytes, data_location)) {
        set_error_once(error_flag, protobuf_error::OVERFLOW);
        return;
      }
      nested_locations[flat_index(row, num_nested_fields, f)] = {data_location,
                                                                 static_cast<int32_t>(len)};
    }

    // Skip to next field
    uint8_t const* next;
    if (!skip_field(cur, msg_end, wt, next)) {
      set_error_once(error_flag, protobuf_error::SKIP);
      return;
    }
    cur = next;
  }
}

/**
 * Combined occurrence scan: scans each message ONCE and writes occurrences for ALL
 * repeated fields simultaneously, scanning each message only once.
 */
template <wire_type_mismatch_policy MismatchPolicy>
__device__ bool scan_all_repeated_occurrences_in_message(uint8_t const* msg_base,
                                                         uint8_t const* msg_end,
                                                         repeated_field_scan_desc const* scan_descs,
                                                         int num_scan_fields,
                                                         protobuf_error* error_flag,
                                                         int const* fn_to_desc_idx,
                                                         int fn_to_desc_size,
                                                         cudf::size_type row)
{
  // Defense-in-depth: host-side validation enforces this cap, so the check is unreachable on a
  // correct config. Keep it in release builds because overrunning `write_idx` below is silent UB.
  if (num_scan_fields > MAX_REPEATED_FIELDS_PER_KERNEL) {
    set_error_once(error_flag, protobuf_error::SCHEMA_TOO_LARGE);
    return false;
  }

  int write_idx[MAX_REPEATED_FIELDS_PER_KERNEL];
  for (int f = 0; f < num_scan_fields; f++) {
    write_idx[f] = scan_descs[f].row_offsets[row];
  }

  auto lookup_by_fn = [&](int fn) {
    return lookup_field(fn, fn_to_desc_idx, fn_to_desc_size, num_scan_fields, [&](int f, int) {
      return scan_descs[f].field_number == fn;
    });
  };
  auto is_repeated_field      = []([[maybe_unused]] int f) { return true; };
  auto get_expected_wire_type = [&](int f) { return scan_descs[f].wire_type; };

  auto const row_i32 = static_cast<int32_t>(row);
  auto on_repeated_scan =
    [&](int f, uint8_t const* cur, uint8_t const* me, uint8_t const* mb, int wt) {
      auto* occs       = scan_descs[f].occurrences;
      int& wi          = write_idx[f];
      int const we     = scan_descs[f].row_offsets[row + 1];
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

  if (!scan_message_field_locations(msg_base,
                                    msg_end,
                                    /*out=*/nullptr,
                                    error_flag,
                                    lookup_by_fn,
                                    is_repeated_field,
                                    get_expected_wire_type,
                                    on_repeated_scan)) {
    return false;
  }

  for (int f = 0; f < num_scan_fields; f++) {
    if (write_idx[f] != scan_descs[f].row_offsets[row + 1]) {
      set_error_once(error_flag, protobuf_error::REPEATED_COUNT_MISMATCH);
      return false;
    }
  }
  return true;
}

CUDF_KERNEL void scan_all_repeated_occurrences_kernel(cudf::column_device_view const d_in,
                                                      repeated_field_scan_desc const* scan_descs,
                                                      int num_scan_fields,
                                                      protobuf_error* error_flag,
                                                      int const* fn_to_desc_idx,
                                                      int fn_to_desc_size)
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
    scan_all_repeated_occurrences_in_message<wire_type_mismatch_policy::report_error>(
      bytes + start,
      bytes + end,
      scan_descs,
      num_scan_fields,
      error_flag,
      fn_to_desc_idx,
      fn_to_desc_size,
      row);
}

// ============================================================================
// Nested message scanning kernels
// ============================================================================

/**
 * Scan one nested message per parent row to locate singleton children and count repeated
 * children. Singleton locations use last-one-wins semantics; repeated occurrences are written
 * by the combined scan after their LIST offsets are available.
 */
CUDF_KERNEL void scan_nested_message_fields_kernel(uint8_t const* message_data,
                                                   cudf::size_type message_data_size,
                                                   cudf::size_type const* parent_row_offsets,
                                                   cudf::size_type parent_base_offset,
                                                   field_location const* parent_locations,
                                                   int num_parent_rows,
                                                   field_descriptor const* field_descs,
                                                   int num_fields,
                                                   field_location* output_locations,
                                                   repeated_field_info* repeated_info,
                                                   protobuf_error* error_flag,
                                                   bool* row_has_invalid_data,
                                                   int32_t const* top_row_indices)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_parent_rows) return;

  auto const top_row =
    top_row_indices != nullptr ? top_row_indices[row] : static_cast<int32_t>(row);
  auto mark_row_error = [&]() {
    if (row_has_invalid_data != nullptr) { row_has_invalid_data[top_row] = true; }
  };

  field_location* field_locations = output_locations + flat_index(row, num_fields, 0);
  for (int f = 0; f < num_fields; f++) {
    field_locations[f] = {-1, 0};
    if (repeated_info != nullptr) { repeated_info[flat_index(row, num_fields, f)] = {0}; }
  }

  auto const& parent_loc = parent_locations[row];
  if (parent_loc.offset < 0) return;

  // Do the subtraction in int64 to keep the bounds-check honest even if a future caller
  // ever passes a sliced LIST where parent_base_offset > parent_row_offsets[row].
  int64_t parent_row_start = static_cast<int64_t>(parent_row_offsets[row]) - parent_base_offset;
  int64_t nested_start_off = parent_row_start + parent_loc.offset;
  int64_t nested_end_off   = nested_start_off + parent_loc.length;
  if (!check_message_bounds(nested_start_off, nested_end_off, message_data_size, error_flag)) {
    mark_row_error();
    return;
  }
  uint8_t const* const nested_start = message_data + nested_start_off;
  uint8_t const* const nested_end   = message_data + nested_end_off;

  auto lookup_desc_idx = [&](int fn) {
    return lookup_field(
      fn, /*field_lookup=*/nullptr, /*field_lookup_size=*/0, num_fields, [&](int f, int n) {
        return field_descs[f].field_number == n;
      });
  };
  auto is_repeated_field      = [&](int f) { return field_descs[f].is_repeated; };
  auto get_expected_wire_type = [&](int f) { return field_descs[f].expected_wire_type; };
  auto validate_repeated =
    [&](int f, uint8_t const* cur, uint8_t const* msg_end, uint8_t const* msg_base, int wt) {
      auto const expected_wire_type = get_expected_wire_type(f);
      auto count_occurrence = [&]([[maybe_unused]] int32_t off, [[maybe_unused]] int32_t len) {
        if (repeated_info != nullptr) { repeated_info[flat_index(row, num_fields, f)].count++; }
        return true;
      };
      return walk_repeated_element<wire_type_mismatch_policy::skip>(
        cur, msg_end, msg_base, wt, expected_wire_type, error_flag, count_occurrence);
    };

  if (!scan_message_field_locations(nested_start,
                                    nested_end,
                                    field_locations,
                                    error_flag,
                                    lookup_desc_idx,
                                    is_repeated_field,
                                    get_expected_wire_type,
                                    validate_repeated)) {
    mark_row_error();
  }
}

CUDF_KERNEL void scan_all_repeated_occurrences_in_nested_kernel(
  uint8_t const* message_data,
  cudf::size_type message_data_size,
  cudf::size_type const* row_offsets,
  cudf::size_type base_offset,
  field_location const* parent_locs,
  repeated_field_scan_desc const* scan_descs,
  int num_scan_fields,
  protobuf_error* error_flag,
  int const* fn_to_desc_idx,
  int fn_to_desc_size,
  int num_rows)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  auto const& parent_loc = parent_locs[row];
  if (parent_loc.offset < 0) return;

  int64_t const row_off       = static_cast<int64_t>(row_offsets[row]) - base_offset;
  int64_t const msg_start_off = row_off + parent_loc.offset;
  int64_t const msg_end_off   = msg_start_off + parent_loc.length;
  if (!check_message_bounds(msg_start_off, msg_end_off, message_data_size, error_flag)) { return; }

  [[maybe_unused]] auto const scan_succeeded =
    scan_all_repeated_occurrences_in_message<wire_type_mismatch_policy::skip>(
      message_data + msg_start_off,
      message_data + msg_end_off,
      scan_descs,
      num_scan_fields,
      error_flag,
      fn_to_desc_idx,
      fn_to_desc_size,
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
CUDF_KERNEL void validate_enum_values_kernel(
  int32_t const* values,             // [num_rows] extracted enum values
  bool* valid,                       // [num_rows] field validity flags (will be modified)
  bool* row_has_invalid_enum,        // [num_rows] row-level invalid enum flag (will be set to true)
  int32_t const* valid_enum_values,  // sorted array of valid enum values
  int num_valid_values,              // size of valid_enum_values
  int num_rows)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  // Skip if already invalid (field was missing) - missing field is not an enum error
  if (!valid[row]) return;

  if (enum_binary_search(valid_enum_values, num_valid_values, values[row]) < 0) {
    valid[row] = false;
    // Also mark the row as having an invalid enum - this will null the entire struct row
    row_has_invalid_enum[row] = true;
  }
}

/**
 * Compute output UTF-8 length for enum-as-string rows.
 * Invalid/missing values produce length 0 (null row/field semantics handled by valid[] and
 * row_has_invalid_enum).
 */
CUDF_KERNEL void compute_enum_string_lengths_kernel(
  int32_t const* values,             // [num_rows] enum numeric values
  bool const* valid,                 // [num_rows] field validity
  int32_t const* valid_enum_values,  // sorted enum numeric values
  int32_t const* enum_name_offsets,  // [num_valid_values + 1]
  int num_valid_values,
  int32_t* lengths,  // [num_rows]
  int num_rows)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  if (!valid[row]) {
    lengths[row] = 0;
    return;
  }

  int idx = enum_binary_search(valid_enum_values, num_valid_values, values[row]);
  // Should not happen when validate_enum_values_kernel has already run, but keep safe.
  lengths[row] = idx >= 0 ? (enum_name_offsets[idx + 1] - enum_name_offsets[idx]) : 0;
}

/**
 * Copy enum-as-string UTF-8 bytes into output chars buffer using precomputed row offsets.
 */
CUDF_KERNEL void copy_enum_string_chars_kernel(
  int32_t const* values,             // [num_rows] enum numeric values
  bool const* valid,                 // [num_rows] field validity
  int32_t const* valid_enum_values,  // sorted enum numeric values
  int32_t const* enum_name_offsets,  // [num_valid_values + 1]
  uint8_t const* enum_name_chars,    // concatenated enum UTF-8 names
  int num_valid_values,
  int32_t const* output_offsets,  // [num_rows + 1]
  char* out_chars,                // [total_chars]
  int num_rows)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;
  if (!valid[row]) return;

  int idx = enum_binary_search(valid_enum_values, num_valid_values, values[row]);
  if (idx < 0) return;
  int32_t src_begin = enum_name_offsets[idx];
  int32_t src_end   = enum_name_offsets[idx + 1];
  int32_t dst_begin = output_offsets[row];
  memcpy(
    out_chars + dst_begin, enum_name_chars + src_begin, static_cast<size_t>(src_end - src_begin));
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
                            field_descriptor const* field_descs,
                            int num_fields,
                            int const* field_lookup,
                            int field_lookup_size,
                            field_location* locations,
                            protobuf_error* error_flag,
                            bool* row_has_invalid_data,
                            int num_rows,
                            rmm::cuda_stream_view stream)
{
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  scan_all_fields_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(d_in,
                                                                           field_descs,
                                                                           num_fields,
                                                                           field_lookup,
                                                                           field_lookup_size,
                                                                           locations,
                                                                           error_flag,
                                                                           row_has_invalid_data);
}

void launch_count_repeated_fields(cudf::column_device_view const& d_in,
                                  device_nested_field_descriptor const* schema,
                                  int num_fields,
                                  int depth_level,
                                  repeated_field_info* repeated_info,
                                  int num_repeated_fields,
                                  int const* repeated_field_indices,
                                  field_location* nested_locations,
                                  int num_nested_fields,
                                  int const* nested_field_indices,
                                  protobuf_error* error_flag,
                                  int const* fn_to_rep_idx,
                                  int fn_to_rep_size,
                                  int const* fn_to_nested_idx,
                                  int fn_to_nested_size,
                                  int num_rows,
                                  rmm::cuda_stream_view stream)
{
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  count_repeated_fields_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    d_in,
    schema,
    num_fields,
    depth_level,
    repeated_info,
    num_repeated_fields,
    repeated_field_indices,
    nested_locations,
    num_nested_fields,
    nested_field_indices,
    error_flag,
    fn_to_rep_idx,
    fn_to_rep_size,
    fn_to_nested_idx,
    fn_to_nested_size);
}

void launch_scan_all_repeated_occurrences(cudf::column_device_view const& d_in,
                                          repeated_field_scan_desc const* scan_descs,
                                          int num_scan_fields,
                                          protobuf_error* error_flag,
                                          int const* fn_to_desc_idx,
                                          int fn_to_desc_size,
                                          int num_rows,
                                          rmm::cuda_stream_view stream)
{
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  scan_all_repeated_occurrences_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    d_in, scan_descs, num_scan_fields, error_flag, fn_to_desc_idx, fn_to_desc_size);
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

void launch_scan_nested_message_fields(uint8_t const* message_data,
                                       cudf::size_type message_data_size,
                                       cudf::size_type const* parent_row_offsets,
                                       cudf::size_type parent_base_offset,
                                       field_location const* parent_locations,
                                       int num_parent_rows,
                                       field_descriptor const* field_descs,
                                       int num_fields,
                                       field_location* output_locations,
                                       repeated_field_info* repeated_info,
                                       protobuf_error* error_flag,
                                       bool* row_has_invalid_data,
                                       int32_t const* top_row_indices,
                                       rmm::cuda_stream_view stream)
{
  if (num_parent_rows == 0) return;
  auto const blocks =
    static_cast<int>((num_parent_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  scan_nested_message_fields_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    message_data,
    message_data_size,
    parent_row_offsets,
    parent_base_offset,
    parent_locations,
    num_parent_rows,
    field_descs,
    num_fields,
    output_locations,
    repeated_info,
    error_flag,
    row_has_invalid_data,
    top_row_indices);
}

void launch_scan_all_repeated_occurrences_in_nested(uint8_t const* message_data,
                                                    cudf::size_type message_data_size,
                                                    cudf::size_type const* row_offsets,
                                                    cudf::size_type base_offset,
                                                    field_location const* parent_locs,
                                                    repeated_field_scan_desc const* scan_descs,
                                                    int num_scan_fields,
                                                    protobuf_error* error_flag,
                                                    int const* fn_to_desc_idx,
                                                    int fn_to_desc_size,
                                                    int num_rows,
                                                    rmm::cuda_stream_view stream)
{
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  scan_all_repeated_occurrences_in_nested_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    message_data,
    message_data_size,
    row_offsets,
    base_offset,
    parent_locs,
    scan_descs,
    num_scan_fields,
    error_flag,
    fn_to_desc_idx,
    fn_to_desc_size,
    num_rows);
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

void launch_validate_enum_values(int32_t const* values,
                                 bool* valid,
                                 bool* row_has_invalid_enum,
                                 int32_t const* valid_enum_values,
                                 int num_valid_values,
                                 int num_rows,
                                 rmm::cuda_stream_view stream)
{
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  validate_enum_values_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    values, valid, row_has_invalid_enum, valid_enum_values, num_valid_values, num_rows);
}

void launch_compute_enum_string_lengths(int32_t const* values,
                                        bool const* valid,
                                        int32_t const* valid_enum_values,
                                        int32_t const* enum_name_offsets,
                                        int num_valid_values,
                                        int32_t* lengths,
                                        int num_rows,
                                        rmm::cuda_stream_view stream)
{
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  compute_enum_string_lengths_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    values, valid, valid_enum_values, enum_name_offsets, num_valid_values, lengths, num_rows);
}

void launch_copy_enum_string_chars(int32_t const* values,
                                   bool const* valid,
                                   int32_t const* valid_enum_values,
                                   int32_t const* enum_name_offsets,
                                   uint8_t const* enum_name_chars,
                                   int num_valid_values,
                                   int32_t const* output_offsets,
                                   char* out_chars,
                                   int num_rows,
                                   rmm::cuda_stream_view stream)
{
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  copy_enum_string_chars_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(values,
                                                                                  valid,
                                                                                  valid_enum_values,
                                                                                  enum_name_offsets,
                                                                                  enum_name_chars,
                                                                                  num_valid_values,
                                                                                  output_offsets,
                                                                                  out_chars,
                                                                                  num_rows);
}

void maybe_check_required_fields(field_location const* locations,
                                 std::vector<int> const& field_indices,
                                 std::vector<nested_field_descriptor> const& schema,
                                 int num_rows,
                                 cudf::bitmask_type const* input_null_mask,
                                 cudf::size_type input_offset,
                                 field_location const* parent_locs,
                                 bool* row_force_null,
                                 int32_t const* top_row_indices,
                                 protobuf_error* error_flag,
                                 rmm::cuda_stream_view stream)
{
  if (num_rows == 0 || field_indices.empty()) { return; }

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

  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  check_required_fields_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    locations,
    d_is_required.data(),
    static_cast<int>(field_indices.size()),
    num_rows,
    input_null_mask,
    input_offset,
    parent_locs,
    row_force_null,
    top_row_indices,
    error_flag);
}

void propagate_invalid_enum_flags_to_rows(rmm::device_uvector<bool> const& item_invalid,
                                          rmm::device_uvector<bool>& row_invalid,
                                          int num_items,
                                          int32_t const* top_row_indices,
                                          bool propagate_to_rows,
                                          rmm::cuda_stream_view stream)
{
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
                                      cudf::detail::host_vector<int32_t> const& valid_enums,
                                      rmm::device_uvector<bool>& row_invalid,
                                      int num_items,
                                      int32_t const* top_row_indices,
                                      bool propagate_to_rows,
                                      rmm::cuda_stream_view stream)
{
  if (num_items == 0 || valid_enums.empty()) return;

  auto const scratch_mr = cudf::get_current_device_resource_ref();
  auto const blocks  = static_cast<int>((num_items + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  auto d_valid_enums = cudf::detail::make_device_uvector_async(valid_enums, stream, scratch_mr);

  rmm::device_uvector<bool> item_invalid(num_items, stream, scratch_mr);
  thrust::fill(
    rmm::exec_policy_nosync(stream, scratch_mr), item_invalid.begin(), item_invalid.end(), false);
  validate_enum_values_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    values.data(),
    valid.data(),
    item_invalid.data(),
    d_valid_enums.data(),
    static_cast<int>(valid_enums.size()),
    num_items);

  propagate_invalid_enum_flags_to_rows(
    item_invalid, row_invalid, num_items, top_row_indices, propagate_to_rows, stream);
}

}  // namespace spark_rapids_jni::protobuf::detail
