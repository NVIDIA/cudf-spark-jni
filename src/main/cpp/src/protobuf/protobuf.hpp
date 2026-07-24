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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace spark_rapids_jni::protobuf {

enum class proto_encoding : int {
  DEFAULT     = 0,
  FIXED       = 1,
  ZIGZAG      = 2,
  ENUM_STRING = 3,
};

constexpr int MAX_FIELD_NUMBER = (1 << 29) - 1;

enum class proto_wire_type : int {
  VARINT = 0,
  I64BIT = 1,
  LEN    = 2,
  SGROUP = 3,
  EGROUP = 4,
  I32BIT = 5,
};

CUDF_HOST_DEVICE constexpr int wire_type_value(proto_wire_type wire_type)
{
  return static_cast<int>(wire_type);
}

constexpr int MAX_NESTING_DEPTH = 10;

struct nested_field_descriptor {
  int field_number;           // Protobuf field number
  int parent_idx;             // Index of parent field in schema (-1 for top-level)
  int depth;                  // Nesting depth (0 for top-level)
  proto_wire_type wire_type;  // Expected wire type
  cudf::type_id output_type;  // Output cudf type
  proto_encoding encoding;    // Encoding type
  bool is_repeated;           // Whether this field is repeated (array)
  bool is_required;           // Whether this field is required (proto2)
  bool has_default_value;     // Whether this field has a default value
};

struct protobuf_decode_context {
  std::vector<nested_field_descriptor> schema;
  std::vector<int64_t> default_ints;
  std::vector<double> default_floats;
  std::vector<bool> default_bools;
  std::vector<cudf::detail::host_vector<uint8_t>> default_strings;
  std::vector<cudf::detail::host_vector<int32_t>> enum_valid_values;
  std::vector<std::vector<cudf::detail::host_vector<uint8_t>>> enum_names;
  bool fail_on_errors;
  // Per-field flag (parallel to `schema`) controlling whether a field appears in the
  // returned struct. Hidden fields are still decoded so required/enum/wire validation
  // runs, but they are dropped from the final output. An empty vector means "all output".
  std::vector<bool> output_fields;
};

struct protobuf_field_meta_view {
  nested_field_descriptor const& schema;
  cudf::data_type const output_type;
  int64_t default_int;
  double default_float;
  bool default_bool;
  cudf::detail::host_vector<uint8_t> const& default_string;
  cudf::detail::host_vector<int32_t> const& enum_valid_values;
  std::vector<cudf::detail::host_vector<uint8_t>> const& enum_names;
};

namespace detail {

bool is_encoding_compatible(nested_field_descriptor const& field, cudf::data_type const& type);

void validate_decode_context(protobuf_decode_context const& context);

}  // namespace detail

std::unique_ptr<cudf::column> decode_protobuf_to_struct(cudf::column_view const& binary_input,
                                                        protobuf_decode_context const& context,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr);

}  // namespace spark_rapids_jni::protobuf
