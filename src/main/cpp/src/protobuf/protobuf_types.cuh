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

namespace spark_rapids_jni::protobuf::detail {

// Protobuf varint encoding uses at most 10 bytes to represent a 64-bit value.
constexpr int MAX_VARINT_BYTES = 10;

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
  int field_number;        // Protobuf field number
  int expected_wire_type;  // Expected wire type for this field
  bool is_repeated;        // Repeated children are scanned via count/scan kernels
};

/**
 * Information about repeated field occurrences in a row.
 */
struct repeated_field_info {
  int32_t count;  // Number of occurrences in this row
};

/**
 * Location of a single occurrence of a repeated field.
 */
struct repeated_occurrence {
  int32_t row_idx;  // Which row this occurrence belongs to
  int32_t offset;   // Offset within the message
  int32_t length;   // Length of the field data
};

/**
 * Per-field descriptor passed to the combined occurrence scan kernel.
 * Contains device pointers so the kernel can write to each field's output.
 */
struct repeated_field_scan_desc {
  int field_number;
  int wire_type;
  int32_t const* row_offsets;        // Pre-computed prefix-sum offsets [num_rows + 1]
  repeated_occurrence* occurrences;  // Output buffer [total_count]
};

/**
 * Device-side descriptor for nested schema fields.
 */
struct device_nested_field_descriptor {
  int field_number;
  int parent_idx;
  int depth;
  int wire_type;
  int output_type_id;
  int encoding;
  bool is_repeated;
  bool is_required;
  bool has_default_value;

  device_nested_field_descriptor() = default;

  // Wire type and encoding are stored as int (not typed enums) because CUDA device code
  // historically had limited constexpr enum support, and the kernel comparison sites use
  // int-typed wire_type_value()/encoding_value() helpers throughout.
  explicit device_nested_field_descriptor(nested_field_descriptor const& src)
    : field_number(src.field_number),
      parent_idx(src.parent_idx),
      depth(src.depth),
      wire_type(static_cast<int>(src.wire_type)),
      output_type_id(static_cast<int>(src.output_type)),
      encoding(static_cast<int>(src.encoding)),
      is_repeated(src.is_repeated),
      is_required(src.is_required),
      has_default_value(src.has_default_value)
  {
  }
};

}  // namespace spark_rapids_jni::protobuf::detail
