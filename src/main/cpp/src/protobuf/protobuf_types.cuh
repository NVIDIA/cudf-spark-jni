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

#include "protobuf/protobuf.hpp"

#include <string>
#include <type_traits>

namespace spark_rapids_jni::protobuf::detail {

// Protobuf varint encoding uses at most 10 bytes to represent a 64-bit value.
constexpr int MAX_VARINT_BYTES              = 10;
constexpr int PROTOBUF_JAVA_RECURSION_LIMIT = 100;

// CUDA kernel launch configuration.
constexpr int THREADS_PER_BLOCK = 256;

enum class protobuf_error : int {
  NONE = 0,
  BOUNDS,
  VARINT,
  FIELD_NUMBER,
  WIRE_TYPE,
  OVERFLOW,
  FIELD_SIZE,
  SKIP,
  FIXED_LEN,
  INVALID_ENUM,
  REQUIRED,
  SCHEMA_TOO_LARGE,
  REPEATED_COUNT_MISMATCH,
};

// Threshold for using a direct-mapped lookup table for field_number -> field_index.
// Field numbers above this threshold fall back to linear search.
constexpr int FIELD_LOOKUP_TABLE_MAX = 4096;

// Maximum number of top-level repeated fields the combined occurrence-scan kernel can process
// in a single launch. The kernel keeps a per-thread `int write_idx[MAX_REPEATED_FIELDS_PER_KERNEL]`
// array on the stack; raising the limit pushes the array into local memory, which would otherwise
// cost 4x the per-thread footprint and pressure occupancy. Validated at the host level so the
// error surface depends on the schema, not on which fields happen to have data in a given batch.
constexpr int MAX_REPEATED_FIELDS_PER_KERNEL = 32;

inline std::string error_message(protobuf_error error)
{
  switch (error) {
    using enum protobuf_error;
    case NONE: return "Protobuf decode error: none";
    case BOUNDS: return "Protobuf decode error: message data out of bounds";
    case VARINT: return "Protobuf decode error: invalid or truncated varint";
    case FIELD_NUMBER: return "Protobuf decode error: invalid field number";
    case WIRE_TYPE: return "Protobuf decode error: unexpected wire type";
    case OVERFLOW: return "Protobuf decode error: length-delimited field overflows message";
    case FIELD_SIZE: return "Protobuf decode error: invalid field size";
    case SKIP: return "Protobuf decode error: unable to skip unknown field";
    case FIXED_LEN: return "Protobuf decode error: invalid fixed-width or packed field length";
    case INVALID_ENUM: return "Protobuf decode error: unknown enum value";
    case REQUIRED: return "Protobuf decode error: missing required field";
    case SCHEMA_TOO_LARGE:
      return "Protobuf decode error: schema exceeds maximum supported repeated fields per "
             "kernel (" +
             std::to_string(MAX_REPEATED_FIELDS_PER_KERNEL) + ")";
    case REPEATED_COUNT_MISMATCH:
      return "Protobuf decode error: repeated-field count/scan mismatch";
  }
  return "Protobuf decode error: unknown error";
}

/**
 * Structure to record field location within a message.
 * offset < 0 means field was not found.
 */
struct field_location {
  int32_t offset;  // Offset of field data within the message (-1 if not found)
  int32_t length;  // Length of field data in bytes
};

/**
 * Field descriptor passed to the scanning kernel.
 */
struct field_descriptor {
  int field_number;                    // Protobuf field number
  proto_wire_type expected_wire_type;  // Expected wire type for this field
  bool is_repeated;                    // Repeated children are scanned via count/scan kernels
  int32_t const* valid_enum_values;    // Sorted closed-enum values, or nullptr
  int num_valid_enum_values;           // Size of valid_enum_values
  int output_index = -1;               // Matching output column, or -1 when unused
};

/**
 * Number of selected field occurrences in a row.
 */
struct field_occurrence_count {
  int32_t count;  // Number of occurrences in this row
};

/**
 * Location of a single field occurrence.
 */
struct field_occurrence {
  int32_t row_idx;  // Which row this occurrence belongs to
  int32_t offset;   // Offset within the message
  int32_t length;   // Length of the field data
};

/**
 * Per-field descriptor passed to the combined occurrence scan kernel.
 * Contains device pointers so the kernel can write to each field's output.
 */
struct field_occurrence_scan_desc {
  int field_number;
  proto_wire_type expected_wire_type;
  static constexpr bool is_repeated = true;
  int32_t const* row_offsets;     // Pre-computed prefix-sum offsets [num_rows + 1]
  field_occurrence* occurrences;  // Output buffer [total_count]
};

template <typename T>
struct lookup_view {
  T const* data;
  int size;
  int const* direct;
  int direct_size;
};

using field_occurrence_scan_view = lookup_view<field_occurrence_scan_desc>;

// ============================================================================
// Device-facing views and scalar kernel arguments
// ============================================================================

struct protobuf_input_view {
  uint8_t const* message_data;
  cudf::size_type message_data_size;
  cudf::size_type const* row_offsets;
  cudf::size_type base_offset;
  int num_rows;
};

struct nested_parent_view {
  field_location const* locations;
  std::size_t location_count;
  int32_t const* top_row_indices;
};

struct protobuf_value_domain_view {
  int size;
  int32_t const* top_row_indices;
};

struct required_field_input_view {
  field_location const* locations;
  protobuf_value_domain_view values;
  cudf::bitmask_type const* input_null_mask;
  cudf::size_type input_offset;
  field_location const* parent_locations;
};

struct scalar_value_input {
  uint8_t const* data;
  int32_t length;
  bool present;
};

struct enum_value_device_view {
  int32_t const* values;
  bool* valid;
  int size;
};

template <typename T>
struct scalar_value_output {
  T* values;
  bool* valid;
  protobuf_error* error;
};

template <typename T>
struct scalar_decode_options {
  bool has_default;
  T default_value;
};

template <typename T>
struct batched_scalar_desc {
  int loc_field_idx;
  T* output;
  bool* valid;
  scalar_decode_options<T> options;
};

template <typename T>
struct batched_scalar_input_view {
  protobuf_input_view input;
  field_location const* locations;
  int num_location_fields;
  batched_scalar_desc<T> const* descriptors;
  int num_descriptors;
  protobuf_error* error;
};

struct enum_domain_device_view {
  int32_t const* valid_values;
  int size;
};

struct enum_string_lookup_device_view {
  enum_domain_device_view domain;
  int32_t const* name_offsets;
  uint8_t const* name_chars;
};

struct field_scan_view {
  field_location* locations;
  int location_stride;
  field_occurrence_count* repeated_info;
  int repeated_stride;
  lookup_view<field_descriptor> lookup;
};

template <typename T>
concept device_layout_compatible = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;

static_assert(device_layout_compatible<protobuf_input_view>);
static_assert(device_layout_compatible<field_descriptor>);
static_assert(device_layout_compatible<field_occurrence_scan_desc>);
static_assert(device_layout_compatible<field_occurrence_scan_view>);
static_assert(device_layout_compatible<lookup_view<field_descriptor>>);
static_assert(device_layout_compatible<nested_parent_view>);
static_assert(device_layout_compatible<protobuf_value_domain_view>);
static_assert(device_layout_compatible<required_field_input_view>);
static_assert(device_layout_compatible<scalar_value_input>);
static_assert(device_layout_compatible<enum_value_device_view>);
static_assert(device_layout_compatible<scalar_value_output<int32_t>>);
static_assert(device_layout_compatible<scalar_decode_options<int64_t>>);
static_assert(device_layout_compatible<batched_scalar_desc<int32_t>>);
static_assert(device_layout_compatible<batched_scalar_input_view<int32_t>>);
static_assert(device_layout_compatible<batched_scalar_desc<double>>);
static_assert(device_layout_compatible<batched_scalar_input_view<double>>);
static_assert(device_layout_compatible<enum_domain_device_view>);
static_assert(device_layout_compatible<enum_string_lookup_device_view>);
static_assert(device_layout_compatible<field_scan_view>);

}  // namespace spark_rapids_jni::protobuf::detail
