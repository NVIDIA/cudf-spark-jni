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

#include "protobuf/protobuf.hpp"
#include "protobuf/protobuf_kernels.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/stream>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace protobuf        = spark_rapids_jni::protobuf;
namespace protobuf_detail = spark_rapids_jni::protobuf::detail;

int sample_count_around_average(int average, std::mt19937& rng)
{
  auto const radius = average / 2;
  return std::uniform_int_distribution<int>(average - radius, average + radius)(rng);
}

// ---------------------------------------------------------------------------
// Protobuf wire-format encoding helpers (host side, for generating test data)
// ---------------------------------------------------------------------------

void encode_varint(std::vector<uint8_t>& buf, uint64_t value)
{
  while (value > 0x7F) {
    buf.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  buf.push_back(static_cast<uint8_t>(value));
}

void encode_tag(std::vector<uint8_t>& buf, int field_number, int wire_type)
{
  encode_varint(buf, (static_cast<uint64_t>(field_number) << 3) | static_cast<uint64_t>(wire_type));
}

void encode_varint_field(std::vector<uint8_t>& buf, int field_number, int64_t value)
{
  encode_tag(buf, field_number, static_cast<int>(protobuf::proto_wire_type::VARINT));
  encode_varint(buf, static_cast<uint64_t>(value));
}

void encode_fixed32_field(std::vector<uint8_t>& buf, int field_number, float value)
{
  encode_tag(buf, field_number, static_cast<int>(protobuf::proto_wire_type::I32BIT));
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int i = 0; i < 4; i++) {
    buf.push_back(static_cast<uint8_t>(bits & 0xFF));
    bits >>= 8;
  }
}

void encode_fixed64_field(std::vector<uint8_t>& buf, int field_number, double value)
{
  encode_tag(buf, field_number, static_cast<int>(protobuf::proto_wire_type::I64BIT));
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int i = 0; i < 8; i++) {
    buf.push_back(static_cast<uint8_t>(bits & 0xFF));
    bits >>= 8;
  }
}

void encode_len_field(std::vector<uint8_t>& buf, int field_number, void const* data, size_t len)
{
  encode_tag(buf, field_number, static_cast<int>(protobuf::proto_wire_type::LEN));
  encode_varint(buf, len);
  auto const* p = static_cast<uint8_t const*>(data);
  buf.insert(buf.end(), p, p + len);
}

void encode_string_field(std::vector<uint8_t>& buf, int field_number, std::string const& s)
{
  encode_len_field(buf, field_number, s.data(), s.size());
}

// Encode a nested message: write its content into a temporary buffer, then emit as LEN.
template <typename Fn>
void encode_nested_message(std::vector<uint8_t>& buf, int field_number, Fn&& content_fn)
{
  std::vector<uint8_t> inner;
  content_fn(inner);
  encode_len_field(buf, field_number, inner.data(), inner.size());
}

// Encode a packed repeated int32 field.
void encode_packed_repeated_int32(std::vector<uint8_t>& buf,
                                  int field_number,
                                  std::vector<int32_t> const& values)
{
  std::vector<uint8_t> packed;
  for (auto v : values) {
    encode_varint(packed, static_cast<uint64_t>(static_cast<uint32_t>(v)));
  }
  encode_len_field(buf, field_number, packed.data(), packed.size());
}

// ---------------------------------------------------------------------------
// Build a cuDF LIST<UINT8> column from host message buffers
// ---------------------------------------------------------------------------

std::unique_ptr<cudf::column> make_binary_column(std::vector<std::vector<uint8_t>> const& messages)
{
  cuda::stream_ref stream = cudf::get_default_stream();
  auto mr                 = cudf::get_current_device_resource_ref();

  std::vector<int32_t> h_offsets(messages.size() + 1);
  h_offsets[0] = 0;
  for (size_t i = 0; i < messages.size(); i++) {
    auto const next_offset = static_cast<size_t>(h_offsets[i]) + messages[i].size();
    CUDF_EXPECTS(std::cmp_less_equal(next_offset, std::numeric_limits<int32_t>::max()),
                 "benchmark input exceeds the LIST offset range");
    h_offsets[i + 1] = static_cast<int32_t>(next_offset);
  }
  int32_t const total_bytes = h_offsets.back();

  std::vector<uint8_t> h_data;
  h_data.reserve(total_bytes);
  for (auto const& m : messages) {
    h_data.insert(h_data.end(), m.begin(), m.end());
  }

  rmm::device_buffer d_data(h_data.data(), h_data.size(), stream, mr);
  rmm::device_buffer d_offsets(h_offsets.data(), h_offsets.size() * sizeof(int32_t), stream, mr);
  stream.sync();

  auto child_col = std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::UINT8}, total_bytes, std::move(d_data), rmm::device_buffer{}, 0);
  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    static_cast<cudf::size_type>(h_offsets.size()),
                                                    std::move(d_offsets),
                                                    rmm::device_buffer{},
                                                    0);

  return cudf::make_lists_column(static_cast<cudf::size_type>(messages.size()),
                                 std::move(offsets_col),
                                 std::move(child_col),
                                 0,
                                 rmm::device_buffer{});
}

// ---------------------------------------------------------------------------
// Schema + message generators for different benchmark scenarios
// ---------------------------------------------------------------------------

using protobuf::nested_field_descriptor;
using protobuf::proto_encoding;
using protobuf::proto_wire_type;

nested_field_descriptor make_field_descriptor(int field_number,
                                              int parent_idx,
                                              int depth,
                                              proto_wire_type wire_type,
                                              cudf::type_id output_type,
                                              bool is_repeated        = false,
                                              proto_encoding encoding = proto_encoding::DEFAULT)
{
  return {
    field_number, parent_idx, depth, wire_type, output_type, encoding, is_repeated, false, false};
}

struct generated_field_type {
  cudf::type_id output_type;
  proto_wire_type wire_type;
  proto_encoding encoding;
};

constexpr auto GENERATED_TYPES = std::to_array<generated_field_type>({
  {cudf::type_id::INT32, proto_wire_type::VARINT, proto_encoding::DEFAULT},
  {cudf::type_id::INT64, proto_wire_type::VARINT, proto_encoding::DEFAULT},
  {cudf::type_id::FLOAT32, proto_wire_type::I32BIT, proto_encoding::FIXED},
  {cudf::type_id::FLOAT64, proto_wire_type::I64BIT, proto_encoding::FIXED},
  {cudf::type_id::BOOL8, proto_wire_type::VARINT, proto_encoding::DEFAULT},
  {cudf::type_id::STRING, proto_wire_type::LEN, proto_encoding::DEFAULT},
});

// Case 1: Flat scalars only — many top-level scalar fields.
//   message FlatMessage {
//     int32  f1 = 1;
//     int64  f2 = 2;
//     ...
//     float  f_k   = k;     (cycling through int32, int64, float, double, bool)
//     string s_k+1 = k+1;   (a few string fields)
//   }
struct FlatScalarCase {
  static constexpr auto NON_STRING_TYPES = std::span{GENERATED_TYPES}.first<5>();

  int num_fields;
  int string_field_percent;

  int num_string_fields() const { return std::max(1, num_fields * string_field_percent / 100); }
  int num_non_string_fields() const { return num_fields - num_string_fields(); }

  protobuf::protobuf_decode_context build_context() const
  {
    std::vector<nested_field_descriptor> schema;

    int fn = 1;
    for (int i = 0; i < num_non_string_fields(); i++, fn++) {
      auto const field_type = NON_STRING_TYPES[i % NON_STRING_TYPES.size()];
      schema.push_back(make_field_descriptor(
        fn, -1, 0, field_type.wire_type, field_type.output_type, false, field_type.encoding));
    }
    for (int i = 0; i < num_string_fields(); i++, fn++) {
      schema.push_back(
        make_field_descriptor(fn, -1, 0, proto_wire_type::LEN, cudf::type_id::STRING));
    }

    return {std::move(schema), true, cudf::get_default_stream()};
  }

  std::vector<std::vector<uint8_t>> generate_messages(int num_rows, std::mt19937& rng) const
  {
    std::vector<std::vector<uint8_t>> messages(num_rows);
    std::uniform_int_distribution<int32_t> int_dist(0, 100000);
    std::uniform_int_distribution<int> str_len_dist(5, 50);
    std::string alphabet = "abcdefghijklmnopqrstuvwxyz0123456789";

    for (int r = 0; r < num_rows; r++) {
      auto& buf = messages[r];
      int fn    = 1;
      for (int i = 0; i < num_non_string_fields(); i++, fn++) {
        switch (NON_STRING_TYPES[i % NON_STRING_TYPES.size()].output_type) {
          case cudf::type_id::INT32:
          case cudf::type_id::INT64: encode_varint_field(buf, fn, int_dist(rng)); break;
          case cudf::type_id::FLOAT32:
            encode_fixed32_field(buf, fn, static_cast<float>(int_dist(rng)));
            break;
          case cudf::type_id::FLOAT64:
            encode_fixed64_field(buf, fn, static_cast<double>(int_dist(rng)));
            break;
          case cudf::type_id::BOOL8: encode_varint_field(buf, fn, rng() % 2); break;
          default: CUDF_FAIL("unsupported generated scalar type");
        }
      }
      for (int i = 0; i < num_string_fields(); i++, fn++) {
        int len = str_len_dist(rng);
        std::string s(len, ' ');
        for (int c = 0; c < len; c++) {
          s[c] = alphabet[rng() % alphabet.size()];
        }
        encode_string_field(buf, fn, s);
      }
    }
    return messages;
  }
};

// Case 2: Nested message — a top-level message with a nested struct child.
//   message OuterMessage {
//     int32  id = 1;
//     string name = 2;
//     InnerMessage inner = 3;
//   }
//   message InnerMessage {
//     int32  x = 1;
//     int64  y = 2;
//     string data = 3;
//     ... (num_inner_fields fields)
//   }
struct NestedMessageCase {
  static constexpr std::array INNER_TYPES{
    GENERATED_TYPES[0], GENERATED_TYPES[1], GENERATED_TYPES[5]};

  int num_inner_fields;  // scalar fields inside InnerMessage

  protobuf::protobuf_decode_context build_context() const
  {
    std::vector<nested_field_descriptor> schema;

    // idx 0: id (int32, top-level)
    schema.push_back(
      make_field_descriptor(1, -1, 0, proto_wire_type::VARINT, cudf::type_id::INT32));
    // idx 1: name (string, top-level)
    schema.push_back(make_field_descriptor(2, -1, 0, proto_wire_type::LEN, cudf::type_id::STRING));
    // idx 2: inner (STRUCT, top-level)
    schema.push_back(make_field_descriptor(3, -1, 0, proto_wire_type::LEN, cudf::type_id::STRUCT));

    // Inner message children (parent_idx=2, depth=1)
    for (int i = 0; i < num_inner_fields; i++) {
      auto const field_type = INNER_TYPES[i % INNER_TYPES.size()];
      schema.push_back(make_field_descriptor(
        i + 1, 2, 1, field_type.wire_type, field_type.output_type, false, field_type.encoding));
    }

    return {std::move(schema), true, cudf::get_default_stream()};
  }

  std::vector<std::vector<uint8_t>> generate_messages(int num_rows, std::mt19937& rng) const
  {
    std::vector<std::vector<uint8_t>> messages(num_rows);
    std::uniform_int_distribution<int32_t> int_dist(0, 100000);
    std::uniform_int_distribution<int> str_len_dist(5, 30);
    std::string alphabet = "abcdefghijklmnopqrstuvwxyz";

    auto random_string = [&](int len) {
      std::string s(len, ' ');
      for (int c = 0; c < len; c++)
        s[c] = alphabet[rng() % alphabet.size()];
      return s;
    };

    for (int r = 0; r < num_rows; r++) {
      auto& buf = messages[r];
      encode_varint_field(buf, 1, int_dist(rng));
      encode_string_field(buf, 2, random_string(str_len_dist(rng)));

      encode_nested_message(buf, 3, [&](std::vector<uint8_t>& inner) {
        for (int i = 0; i < num_inner_fields; i++) {
          switch (INNER_TYPES[i % INNER_TYPES.size()].output_type) {
            case cudf::type_id::INT32:
            case cudf::type_id::INT64: encode_varint_field(inner, i + 1, int_dist(rng)); break;
            case cudf::type_id::STRING:
              encode_string_field(inner, i + 1, random_string(str_len_dist(rng)));
              break;
            default: CUDF_FAIL("unsupported generated nested type");
          }
        }
      });
    }
    return messages;
  }
};

// Case 3: Repeated fields — top-level repeated scalars and a repeated nested message.
//   message RepeatedMessage {
//     int32           id = 1;
//     repeated int32  tags = 2;
//     repeated string labels = 3;
//     repeated Item   items = 4;
//   }
//   message Item {
//     int32  item_id = 1;
//     string item_name = 2;
//     int64  value = 3;
//   }
struct RepeatedFieldCase {
  int avg_tags_per_row;
  int avg_labels_per_row;
  int avg_items_per_row;

  protobuf::protobuf_decode_context build_context() const
  {
    std::vector<nested_field_descriptor> schema;

    // idx 0: id (int32, scalar)
    schema.push_back(
      make_field_descriptor(1, -1, 0, proto_wire_type::VARINT, cudf::type_id::INT32));
    // idx 1: tags (repeated int32, packed)
    schema.push_back(
      make_field_descriptor(2, -1, 0, proto_wire_type::VARINT, cudf::type_id::INT32, true));
    // idx 2: labels (repeated string)
    schema.push_back(
      make_field_descriptor(3, -1, 0, proto_wire_type::LEN, cudf::type_id::STRING, true));
    // idx 3: items (repeated STRUCT)
    schema.push_back(
      make_field_descriptor(4, -1, 0, proto_wire_type::LEN, cudf::type_id::STRUCT, true));
    // idx 4: Item.item_id (int32, child of idx 3)
    schema.push_back(make_field_descriptor(1, 3, 1, proto_wire_type::VARINT, cudf::type_id::INT32));
    // idx 5: Item.item_name (string, child of idx 3)
    schema.push_back(make_field_descriptor(2, 3, 1, proto_wire_type::LEN, cudf::type_id::STRING));
    // idx 6: Item.value (int64, child of idx 3)
    schema.push_back(make_field_descriptor(3, 3, 1, proto_wire_type::VARINT, cudf::type_id::INT64));

    return {std::move(schema), true, cudf::get_default_stream()};
  }

  std::vector<std::vector<uint8_t>> generate_messages(int num_rows, std::mt19937& rng) const
  {
    std::vector<std::vector<uint8_t>> messages(num_rows);
    std::uniform_int_distribution<int32_t> int_dist(0, 100000);
    std::uniform_int_distribution<int> str_len_dist(3, 20);
    std::string alphabet = "abcdefghijklmnopqrstuvwxyz";

    auto random_string = [&](int len) {
      std::string s(len, ' ');
      for (int c = 0; c < len; c++)
        s[c] = alphabet[rng() % alphabet.size()];
      return s;
    };

    for (int r = 0; r < num_rows; r++) {
      auto& buf = messages[r];

      // id
      encode_varint_field(buf, 1, int_dist(rng));

      // tags (packed repeated int32)
      {
        int n = sample_count_around_average(avg_tags_per_row, rng);
        std::vector<int32_t> tags(n);
        for (auto& t : tags)
          t = int_dist(rng);
        if (n > 0) encode_packed_repeated_int32(buf, 2, tags);
      }

      // labels (unpacked repeated string)
      {
        int n = sample_count_around_average(avg_labels_per_row, rng);
        for (int i = 0; i < n; i++) {
          encode_string_field(buf, 3, random_string(str_len_dist(rng)));
        }
      }

      // items (repeated nested message)
      {
        int n = sample_count_around_average(avg_items_per_row, rng);
        for (int i = 0; i < n; i++) {
          encode_nested_message(buf, 4, [&](std::vector<uint8_t>& inner) {
            encode_varint_field(inner, 1, int_dist(rng));
            encode_string_field(inner, 2, random_string(str_len_dist(rng)));
            encode_varint_field(inner, 3, int_dist(rng));
          });
        }
      }
    }
    return messages;
  }
};

// Case 4: Wide repeated message — stress-tests repeated struct child scanning.
//   message WideRepeatedMessage {
//     int32         id = 1;
//     repeated Item items = 2;
//   }
//   message Item {
//     int32 / int64 / float / double / bool / string child fields ...
//     ... (num_child_fields fields)
//   }
//
// This case is intentionally generic and contains no customer schema details.
// Its wide repeated STRUCT payload approximates real-world schema-projection workloads.
struct WideRepeatedMessageCase {
  static constexpr int STRING_FIELD_PERIOD = 10;

  int num_child_fields;
  int avg_items_per_row;

  static constexpr generated_field_type child_type(int index)
  {
    auto const type_index = index % STRING_FIELD_PERIOD == STRING_FIELD_PERIOD - 1
                              ? GENERATED_TYPES.size() - 1
                              : index % (GENERATED_TYPES.size() - 1);
    return GENERATED_TYPES[type_index];
  }

  protobuf::protobuf_decode_context build_context() const
  {
    std::vector<nested_field_descriptor> schema;

    // idx 0: id (scalar)
    schema.push_back(
      make_field_descriptor(1, -1, 0, proto_wire_type::VARINT, cudf::type_id::INT32));
    // idx 1: items (repeated STRUCT)
    schema.push_back(
      make_field_descriptor(2, -1, 0, proto_wire_type::LEN, cudf::type_id::STRUCT, true));

    // Keep strings sparse so the case remains dominated by wide child scanning
    // rather than varlen copy traffic.
    for (int i = 0; i < num_child_fields; i++) {
      auto const field_type = child_type(i);
      schema.push_back(make_field_descriptor(
        i + 1, 1, 1, field_type.wire_type, field_type.output_type, false, field_type.encoding));
    }

    return {std::move(schema), true, cudf::get_default_stream()};
  }

  std::vector<std::vector<uint8_t>> generate_messages(int num_rows, std::mt19937& rng) const
  {
    std::vector<std::vector<uint8_t>> messages(num_rows);
    std::uniform_int_distribution<int32_t> int_dist(0, 100000);
    std::uniform_int_distribution<int> str_len_dist(6, 18);
    std::string alphabet = "abcdefghijklmnopqrstuvwxyz";

    auto random_string = [&](int len) {
      std::string s(len, ' ');
      for (int c = 0; c < len; c++)
        s[c] = alphabet[rng() % alphabet.size()];
      return s;
    };

    for (int r = 0; r < num_rows; r++) {
      auto& buf = messages[r];
      encode_varint_field(buf, 1, int_dist(rng));

      int n = sample_count_around_average(avg_items_per_row, rng);
      for (int item_idx = 0; item_idx < n; item_idx++) {
        encode_nested_message(buf, 2, [&](std::vector<uint8_t>& inner) {
          for (int i = 0; i < num_child_fields; i++) {
            int const field_number = i + 1;
            switch (child_type(i).output_type) {
              case cudf::type_id::INT32:
              case cudf::type_id::INT64:
                encode_varint_field(inner, field_number, int_dist(rng));
                break;
              case cudf::type_id::FLOAT32:
                encode_fixed32_field(inner, field_number, static_cast<float>(int_dist(rng)));
                break;
              case cudf::type_id::FLOAT64:
                encode_fixed64_field(inner, field_number, static_cast<double>(int_dist(rng)));
                break;
              case cudf::type_id::BOOL8: encode_varint_field(inner, field_number, rng() % 2); break;
              case cudf::type_id::STRING:
                encode_string_field(inner, field_number, random_string(str_len_dist(rng)));
                break;
              default: CUDF_FAIL("unsupported generated repeated child type");
            }
          }
        });
      }
    }
    return messages;
  }
};

// Case 5: Repeated child lists — stress-tests repeated fields inside a repeated
// struct child, which exercises build_repeated_child_list_column().
//   message OuterMessage {
//     int32         id = 1;
//     repeated Item items = 2;
//   }
//   message Item {
//     repeated int32  r_int_* = 1..N
//     repeated string r_str_* = ...
//   }
//
// This case is intentionally generic and contains no customer schema details.
struct RepeatedChildListCase {
  int num_repeated_children;
  int avg_items_per_row;
  int avg_child_elems;
  std::string child_mix;

  bool child_is_string(int child_idx) const
  {
    if (child_mix == "string_only") return true;
    if (child_mix == "int_only") return false;
    return (child_idx % 4 == 3);
  }

  protobuf::protobuf_decode_context build_context() const
  {
    std::vector<nested_field_descriptor> schema;

    // idx 0: id (scalar)
    schema.push_back(
      make_field_descriptor(1, -1, 0, proto_wire_type::VARINT, cudf::type_id::INT32));
    // idx 1: items (repeated STRUCT)
    schema.push_back(
      make_field_descriptor(2, -1, 0, proto_wire_type::LEN, cudf::type_id::STRUCT, true));

    for (int i = 0; i < num_repeated_children; i++) {
      bool as_string = child_is_string(i);
      schema.push_back(
        make_field_descriptor(i + 1,
                              1,
                              1,
                              as_string ? proto_wire_type::LEN : proto_wire_type::VARINT,
                              as_string ? cudf::type_id::STRING : cudf::type_id::INT32,
                              true));
    }

    return {std::move(schema), true, cudf::get_default_stream()};
  }

  std::vector<std::vector<uint8_t>> generate_messages(int num_rows, std::mt19937& rng) const
  {
    std::vector<std::vector<uint8_t>> messages(num_rows);
    std::uniform_int_distribution<int32_t> int_dist(0, 100000);
    std::uniform_int_distribution<int> str_len_dist(4, 16);
    std::string alphabet = "abcdefghijklmnopqrstuvwxyz";

    auto random_string = [&](int len) {
      std::string s(len, ' ');
      for (int c = 0; c < len; c++)
        s[c] = alphabet[rng() % alphabet.size()];
      return s;
    };

    for (int r = 0; r < num_rows; r++) {
      auto& buf = messages[r];
      encode_varint_field(buf, 1, int_dist(rng));

      int num_items = sample_count_around_average(avg_items_per_row, rng);
      for (int item_idx = 0; item_idx < num_items; item_idx++) {
        encode_nested_message(buf, 2, [&](std::vector<uint8_t>& inner) {
          for (int child_idx = 0; child_idx < num_repeated_children; child_idx++) {
            int fn        = child_idx + 1;
            bool is_str   = child_is_string(child_idx);
            int num_elems = sample_count_around_average(avg_child_elems, rng);
            if (is_str) {
              for (int j = 0; j < num_elems; j++) {
                encode_string_field(inner, fn, random_string(str_len_dist(rng)));
              }
            } else {
              if (num_elems > 0) {
                std::vector<int32_t> vals(num_elems);
                for (auto& v : vals)
                  v = int_dist(rng);
                encode_packed_repeated_int32(inner, fn, vals);
              }
            }
          }
        });
      }
    }
    return messages;
  }
};

// Case 6: Repeated messages nested inside repeated messages.
//   message Root { repeated Outer outers = 1; }
//   message Outer { repeated Inner inners = 1; }
//   message Inner { int32 value = 1; string label = 2; }
struct RepeatedMessageNestingCase {
  int avg_outer_items;
  int avg_inner_items;

  protobuf::protobuf_decode_context build_context() const
  {
    std::vector<nested_field_descriptor> schema;
    schema.push_back(
      make_field_descriptor(1, -1, 0, proto_wire_type::LEN, cudf::type_id::STRUCT, true));
    schema.push_back(
      make_field_descriptor(1, 0, 1, proto_wire_type::LEN, cudf::type_id::STRUCT, true));
    schema.push_back(make_field_descriptor(1, 1, 2, proto_wire_type::VARINT, cudf::type_id::INT32));
    schema.push_back(make_field_descriptor(2, 1, 2, proto_wire_type::LEN, cudf::type_id::STRING));
    return {std::move(schema), true, cudf::get_default_stream()};
  }

  std::vector<std::vector<uint8_t>> generate_messages(int num_rows, std::mt19937& rng) const
  {
    std::vector<std::vector<uint8_t>> messages(num_rows);
    std::uniform_int_distribution<int32_t> value_dist(0, 100000);
    std::uniform_int_distribution<int> string_length_dist(4, 16);
    std::string const alphabet = "abcdefghijklmnopqrstuvwxyz";

    auto random_string = [&](int length) {
      std::string value(length, ' ');
      for (auto& c : value) {
        c = alphabet[rng() % alphabet.size()];
      }
      return value;
    };
    for (auto& message : messages) {
      auto const num_outer_items = sample_count_around_average(avg_outer_items, rng);
      for (int outer_idx = 0; outer_idx < num_outer_items; ++outer_idx) {
        encode_nested_message(message, 1, [&](std::vector<uint8_t>& outer) {
          auto const num_inner_items = sample_count_around_average(avg_inner_items, rng);
          for (int inner_idx = 0; inner_idx < num_inner_items; ++inner_idx) {
            encode_nested_message(outer, 1, [&](std::vector<uint8_t>& inner) {
              encode_varint_field(inner, 1, value_dist(rng));
              encode_string_field(inner, 2, random_string(string_length_dist(rng)));
            });
          }
        });
      }
    }
    return messages;
  }
};

// Case 7: Singular message merge. Each top-level message field has a fixed set of scalar
// children, and each wire row contains one or more occurrences of that singular message.
// occurrences_per_field=1 exercises the normal nested path; larger values exercise fragment
// collection, concatenation, and merged nested decode.
struct SingularMessageMergeCase {
  static constexpr int NUM_CHILD_FIELDS = 4;

  int num_message_fields;
  int occurrences_per_field;

  protobuf::protobuf_decode_context build_context() const
  {
    std::vector<nested_field_descriptor> schema;

    for (int message_idx = 0; message_idx < num_message_fields; ++message_idx) {
      auto const parent_idx = static_cast<int>(schema.size());
      schema.push_back(
        make_field_descriptor(message_idx + 1, -1, 0, proto_wire_type::LEN, cudf::type_id::STRUCT));
      for (int child_idx = 0; child_idx < NUM_CHILD_FIELDS; ++child_idx) {
        schema.push_back(make_field_descriptor(
          child_idx + 1, parent_idx, 1, proto_wire_type::VARINT, cudf::type_id::INT32));
      }
    }

    return {std::move(schema), true, cudf::get_default_stream()};
  }

  std::vector<std::vector<uint8_t>> generate_messages(int num_rows, std::mt19937&) const
  {
    std::vector<std::vector<uint8_t>> messages(num_rows);
    std::vector<uint8_t> nested;
    for (int row = 0; row < num_rows; ++row) {
      auto& buf = messages[row];
      for (int message_idx = 0; message_idx < num_message_fields; ++message_idx) {
        for (int occurrence_idx = 0; occurrence_idx < occurrences_per_field; ++occurrence_idx) {
          nested.clear();
          for (int child_idx = 0; child_idx < NUM_CHILD_FIELDS; ++child_idx) {
            auto const value = static_cast<int64_t>(row) + message_idx + occurrence_idx + child_idx;
            encode_varint_field(nested, child_idx + 1, value);
          }
          encode_len_field(buf, message_idx + 1, nested.data(), nested.size());
        }
      }
    }
    return messages;
  }
};

struct RepeatedChildStringBenchData {
  std::vector<std::vector<uint8_t>> messages;
  std::vector<protobuf_detail::field_location> parent_locations;
  std::vector<std::vector<int32_t>> counts_by_child;
  std::vector<std::vector<protobuf_detail::field_occurrence>> occurrences_by_child;
};

void encode_string_field_record(std::vector<uint8_t>& buf,
                                int field_number,
                                std::string const& value,
                                std::vector<protobuf_detail::field_occurrence>& occurrences,
                                int32_t row_idx)
{
  encode_tag(buf, field_number, static_cast<int>(proto_wire_type::LEN));
  encode_varint(buf, value.size());
  auto const data_offset = static_cast<int32_t>(buf.size());
  buf.insert(buf.end(), value.begin(), value.end());
  occurrences.push_back({row_idx, data_offset, static_cast<int32_t>(value.size())});
}

// Generates one nested-parent payload per input row. The parent locations, counts, and
// occurrences are retained so the isolation benchmarks can keep input preparation and H2D copies
// outside their timed regions.
struct RepeatedChildStringOnlyCase {
  int num_repeated_children;
  int avg_child_elems;

  protobuf::protobuf_decode_context build_context() const
  {
    std::vector<nested_field_descriptor> schema;
    schema.push_back(make_field_descriptor(1, -1, 0, proto_wire_type::LEN, cudf::type_id::STRUCT));
    for (int child_idx = 0; child_idx < num_repeated_children; ++child_idx) {
      schema.push_back(make_field_descriptor(
        child_idx + 1, 0, 1, proto_wire_type::LEN, cudf::type_id::STRING, true));
    }
    return {std::move(schema), true, cudf::get_default_stream()};
  }

  RepeatedChildStringBenchData generate_messages(int num_rows, std::mt19937& rng) const
  {
    RepeatedChildStringBenchData result;
    result.messages.resize(num_rows);
    result.parent_locations.resize(num_rows);
    result.counts_by_child.resize(num_repeated_children);
    result.occurrences_by_child.resize(num_repeated_children);

    std::uniform_int_distribution<int> string_length_dist(4, 16);
    std::string const alphabet = "abcdefghijklmnopqrstuvwxyz";
    auto random_string         = [&](int length) {
      std::string value(length, ' ');
      for (auto& c : value) {
        c = alphabet[rng() % alphabet.size()];
      }
      return value;
    };
    for (int row = 0; row < num_rows; ++row) {
      auto& message = result.messages[row];
      for (int child_idx = 0; child_idx < num_repeated_children; ++child_idx) {
        auto const num_elements = sample_count_around_average(avg_child_elems, rng);
        result.counts_by_child[child_idx].push_back(num_elements);
        for (int element_idx = 0; element_idx < num_elements; ++element_idx) {
          encode_string_field_record(message,
                                     child_idx + 1,
                                     random_string(string_length_dist(rng)),
                                     result.occurrences_by_child[child_idx],
                                     row);
        }
      }
      result.parent_locations[row] = {0, static_cast<int32_t>(message.size())};
    }
    return result;
  }
};

template <typename T>
void copy_to_device(rmm::device_uvector<T>& destination,
                    std::vector<T> const& source,
                    cuda::stream_ref stream)
{
  CUDF_EXPECTS(destination.size() == source.size(), "benchmark H2D size mismatch");
  if (!source.empty()) {
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(
      destination.data(), source.data(), source.size() * sizeof(T), stream));
  }
}

void expect_no_protobuf_error(rmm::device_uvector<protobuf_detail::protobuf_error> const& error,
                              cuda::stream_ref stream)
{
  auto host_error = protobuf_detail::protobuf_error::NONE;
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(
    &host_error, error.data(), sizeof(protobuf_detail::protobuf_error), stream));
  stream.sync();
  CUDF_EXPECTS(host_error == protobuf_detail::protobuf_error::NONE,
               protobuf_detail::error_message(host_error));
}

struct repeated_child_count_scan_work {
  int32_t total_count;
  rmm::device_uvector<int32_t> offsets;
  rmm::device_uvector<protobuf_detail::field_occurrence> occurrences;

  repeated_child_count_scan_work(int num_rows,
                                 int32_t count,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
    : total_count(count), offsets(num_rows + 1, stream, mr), occurrences(count, stream, mr)
  {
  }
};

struct repeated_child_build_work {
  int32_t total_count;
  rmm::device_uvector<int32_t> counts;
  rmm::device_uvector<protobuf_detail::field_occurrence> occurrences;

  repeated_child_build_work(int num_rows,
                            int32_t count,
                            cuda::stream_ref stream,
                            rmm::device_async_resource_ref mr)
    : total_count(count), counts(num_rows, stream, mr), occurrences(count, stream, mr)
  {
  }
};

// Case 8: Many repeated fields — stress-tests per-repeated-field sync overhead.
//   message WideRepeatedMessage {
//     int32              id = 1;
//     repeated int32     r_int_1 = 2;
//     repeated int32     r_int_2 = 3;
//     ...
//     repeated string    r_str_1 = N;
//     repeated string    r_str_2 = N+1;
//     ...
//   }
struct ManyRepeatedFieldsCase {
  int num_repeated_fields;
  int string_field_percent;
  int avg_elems_per_field;

  int num_repeated_str() const
  {
    return std::max(1, num_repeated_fields * string_field_percent / 100);
  }
  int num_repeated_int() const { return num_repeated_fields - num_repeated_str(); }

  protobuf::protobuf_decode_context build_context() const
  {
    std::vector<nested_field_descriptor> schema;

    int fn = 1;
    // idx 0: id (scalar)
    schema.push_back(
      make_field_descriptor(fn++, -1, 0, proto_wire_type::VARINT, cudf::type_id::INT32));

    for (int i = 0; i < num_repeated_int(); i++) {
      schema.push_back(
        make_field_descriptor(fn++, -1, 0, proto_wire_type::VARINT, cudf::type_id::INT32, true));
    }
    for (int i = 0; i < num_repeated_str(); i++) {
      schema.push_back(
        make_field_descriptor(fn++, -1, 0, proto_wire_type::LEN, cudf::type_id::STRING, true));
    }

    return {std::move(schema), true, cudf::get_default_stream()};
  }

  std::vector<std::vector<uint8_t>> generate_messages(int num_rows, std::mt19937& rng) const
  {
    std::vector<std::vector<uint8_t>> messages(num_rows);
    std::uniform_int_distribution<int32_t> int_dist(0, 100000);
    std::uniform_int_distribution<int> str_len_dist(3, 15);
    std::string alphabet = "abcdefghijklmnopqrstuvwxyz";

    auto random_string = [&](int len) {
      std::string s(len, ' ');
      for (int c = 0; c < len; c++)
        s[c] = alphabet[rng() % alphabet.size()];
      return s;
    };
    for (int r = 0; r < num_rows; r++) {
      auto& buf = messages[r];
      int fn    = 1;

      encode_varint_field(buf, fn++, int_dist(rng));

      for (int i = 0; i < num_repeated_int(); i++) {
        int cur_fn = fn++;
        int n      = sample_count_around_average(avg_elems_per_field, rng);
        if (n > 0) {
          std::vector<int32_t> vals(n);
          for (auto& v : vals)
            v = int_dist(rng);
          encode_packed_repeated_int32(buf, cur_fn, vals);
        }
      }
      for (int i = 0; i < num_repeated_str(); i++) {
        int cur_fn = fn++;
        int n      = sample_count_around_average(avg_elems_per_field, rng);
        for (int j = 0; j < n; j++) {
          encode_string_field(buf, cur_fn, random_string(str_len_dist(rng)));
        }
      }
    }
    return messages;
  }
};

template <typename T>
struct constant_argument {
  T value;

  T get(nvbench::state const&) const { return value; }
};

struct int_axis_argument {
  char const* name;

  int get(nvbench::state const& state) const { return static_cast<int>(state.get_int64(name)); }
};

struct string_axis_argument {
  char const* name;

  std::string get(nvbench::state const& state) const { return std::string{state.get_string(name)}; }
};

struct decode_benchmark_data {
  int num_rows;
  size_t total_bytes;
  protobuf::protobuf_decode_context context;
  std::unique_ptr<cudf::column> binary_col;
};

template <typename Generated>
struct isolated_decode_benchmark_data : decode_benchmark_data {
  Generated generated;
};

template <typename Case, typename... Args>
auto prepare_decode_benchmark(nvbench::state const& state, Args const&... args)
{
  auto const num_rows = static_cast<int>(state.get_int64("num_rows"));
  Case generator{args.get(state)...};
  auto context = generator.build_context();

  std::mt19937 rng(42);
  auto generated       = generator.generate_messages(num_rows, rng);
  auto const& messages = [&]() -> auto const& {
    if constexpr (requires { generated.messages; }) {
      return generated.messages;
    } else {
      return generated;
    }
  }();
  auto const total_bytes = std::transform_reduce(
    messages.begin(), messages.end(), size_t{0}, std::plus{}, [](auto const& message) {
      return message.size();
    });
  auto binary_col = make_binary_column(messages);
  decode_benchmark_data data{num_rows, total_bytes, std::move(context), std::move(binary_col)};
  if constexpr (requires { generated.messages; }) {
    return isolated_decode_benchmark_data<decltype(generated)>{std::move(data),
                                                               std::move(generated)};
  } else {
    return data;
  }
}

}  // anonymous namespace

// ===========================================================================
// Benchmark 1: Flat scalars — measures per-field extraction overhead
// ===========================================================================
static void BM_protobuf_flat_scalars(nvbench::state& state)
{
  constexpr int string_field_percent = 10;
  auto data                          = prepare_decode_benchmark<FlatScalarCase>(
    state, int_axis_argument{"num_fields"}, constant_argument{string_field_percent});

  cuda::stream_ref stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = protobuf::decode_protobuf_to_struct(
      data.binary_col->view(), data.context, stream, cudf::get_current_device_resource_ref());
  });

  state.add_element_count(data.num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(data.total_bytes);
}

NVBENCH_BENCH(BM_protobuf_flat_scalars)
  .set_name("Protobuf Flat Scalars")
  .add_int64_axis("num_rows", {10'000, 100'000, 500'000})
  .add_int64_axis("num_fields", {10, 50, 200});

// ===========================================================================
// Benchmark 2: Nested messages — measures nested struct build overhead
// ===========================================================================
static void BM_protobuf_nested(nvbench::state& state)
{
  auto data = prepare_decode_benchmark<NestedMessageCase>(state, int_axis_argument{"inner_fields"});

  cuda::stream_ref stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = protobuf::decode_protobuf_to_struct(
      data.binary_col->view(), data.context, stream, cudf::get_current_device_resource_ref());
  });

  state.add_element_count(data.num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(data.total_bytes);
}

NVBENCH_BENCH(BM_protobuf_nested)
  .set_name("Protobuf Nested Message")
  .add_int64_axis("num_rows", {10'000, 100'000, 500'000})
  .add_int64_axis("inner_fields", {5, 20, 100});

// ===========================================================================
// Benchmark 3: Repeated fields — measures repeated field pipeline overhead
// ===========================================================================
static void BM_protobuf_repeated(nvbench::state& state)
{
  auto data = prepare_decode_benchmark<RepeatedFieldCase>(
    state, constant_argument<int>{5}, constant_argument<int>{3}, int_axis_argument{"avg_items"});

  cuda::stream_ref stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = protobuf::decode_protobuf_to_struct(
      data.binary_col->view(), data.context, stream, cudf::get_current_device_resource_ref());
  });

  state.add_element_count(data.num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(data.total_bytes);
}

NVBENCH_BENCH(BM_protobuf_repeated)
  .set_name("Protobuf Repeated Fields")
  .add_int64_axis("num_rows", {10'000, 100'000})
  .add_int64_axis("avg_items", {1, 5, 20});

// ===========================================================================
// Benchmark 4: Wide repeated message — measures repeated struct child scan cost
// ===========================================================================
static void BM_protobuf_wide_repeated_message(nvbench::state& state)
{
  auto data = prepare_decode_benchmark<WideRepeatedMessageCase>(
    state, int_axis_argument{"num_child_fields"}, int_axis_argument{"avg_items"});

  cuda::stream_ref stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = protobuf::decode_protobuf_to_struct(
      data.binary_col->view(), data.context, stream, cudf::get_current_device_resource_ref());
  });

  state.add_element_count(data.num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(data.total_bytes);
}

NVBENCH_BENCH(BM_protobuf_wide_repeated_message)
  .set_name("Protobuf Wide Repeated Message")
  .add_int64_axis("num_rows", {10'000, 20'000})
  .add_int64_axis("num_child_fields", {20, 100, 200})
  .add_int64_axis("avg_items", {1, 5, 10});

// ===========================================================================
// Benchmark 5: Repeated child lists — measures repeated-in-nested list overhead
// ===========================================================================
static void BM_protobuf_repeated_child_lists(nvbench::state& state)
{
  auto data =
    prepare_decode_benchmark<RepeatedChildListCase>(state,
                                                    int_axis_argument{"num_repeated_children"},
                                                    int_axis_argument{"avg_items"},
                                                    int_axis_argument{"avg_child_elems"},
                                                    string_axis_argument{"child_mix"});

  cuda::stream_ref stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = protobuf::decode_protobuf_to_struct(
      data.binary_col->view(), data.context, stream, cudf::get_current_device_resource_ref());
  });

  state.add_element_count(data.num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(data.total_bytes);
}

NVBENCH_BENCH(BM_protobuf_repeated_child_lists)
  .set_name("Protobuf Repeated Child Lists")
  .add_int64_axis("num_rows", {10'000, 20'000})
  .add_int64_axis("num_repeated_children", {1, 4, 8})
  .add_int64_axis("avg_items", {1, 5})
  .add_int64_axis("avg_child_elems", {1, 5})
  .add_string_axis("child_mix", {"int_only", "mixed", "string_only"});

// ===========================================================================
// Benchmark 6: Repeated messages nested inside repeated messages
// ===========================================================================
static void BM_protobuf_repeated_message_nesting(nvbench::state& state)
{
  auto data = prepare_decode_benchmark<RepeatedMessageNestingCase>(
    state, int_axis_argument{"avg_outer_items"}, int_axis_argument{"avg_inner_items"});

  cuda::stream_ref stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = protobuf::decode_protobuf_to_struct(
      data.binary_col->view(), data.context, stream, cudf::get_current_device_resource_ref());
  });

  state.add_element_count(data.num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(data.total_bytes);
}

NVBENCH_BENCH(BM_protobuf_repeated_message_nesting)
  .set_name("Protobuf Repeated Message Nesting")
  .add_int64_axis("num_rows", {10'000, 20'000})
  .add_int64_axis("avg_outer_items", {1, 5})
  .add_int64_axis("avg_inner_items", {1, 5, 20});

// ===========================================================================
// Benchmark 7: Singular message merge
// ===========================================================================
static void BM_protobuf_singular_message_merge(nvbench::state& state)
{
  auto data = prepare_decode_benchmark<SingularMessageMergeCase>(
    state, int_axis_argument{"num_message_fields"}, int_axis_argument{"occurrences_per_field"});

  cuda::stream_ref stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = protobuf::decode_protobuf_to_struct(
      data.binary_col->view(), data.context, stream, cudf::get_current_device_resource_ref());
  });

  state.add_element_count(data.num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(data.total_bytes);
}

NVBENCH_BENCH(BM_protobuf_singular_message_merge)
  .set_name("Protobuf Singular Message Merge")
  .add_int64_axis("num_rows", {10'000, 100'000})
  .add_int64_axis("num_message_fields", {1, 8, 32})
  .add_int64_axis("occurrences_per_field", {1, 2, 4});

// ===========================================================================
// Benchmark 8: Repeated child string count + occurrence scan device pipeline
// ===========================================================================
static void BM_protobuf_repeated_child_string_count_scan(nvbench::state& state)
{
  auto prepared = prepare_decode_benchmark<RepeatedChildStringOnlyCase>(
    state, int_axis_argument{"num_repeated_children"}, int_axis_argument{"avg_child_elems"});
  auto const num_rows              = prepared.num_rows;
  auto const num_repeated_children = static_cast<int>(prepared.generated.counts_by_child.size());
  auto const& data                 = prepared.generated;

  cuda::stream_ref stream = cudf::get_default_stream();
  auto mr                 = cudf::get_current_device_resource_ref();

  cudf::lists_column_view input_list(prepared.binary_col->view());
  auto const* row_offsets      = input_list.offsets().data<cudf::size_type>();
  auto const child             = input_list.child();
  auto const* message_data     = child.data<uint8_t>();
  auto const message_data_size = static_cast<cudf::size_type>(child.size());

  rmm::device_uvector<protobuf_detail::field_location> parent_locations(num_rows, stream, mr);
  copy_to_device(parent_locations, data.parent_locations, stream);

  protobuf_detail::protobuf_schema schema{prepared.context};
  std::vector<int> child_field_indices;
  child_field_indices.reserve(num_repeated_children);
  for (int child_idx = 0; child_idx < num_repeated_children; ++child_idx) {
    child_field_indices.push_back(child_idx + 1);
  }

  auto field_descriptors =
    protobuf_detail::make_field_descriptors(child_field_indices, schema, stream, mr);

  auto const field_value_count = static_cast<size_t>(num_rows) * num_repeated_children;
  rmm::device_uvector<protobuf_detail::field_location> field_locations(
    field_value_count, stream, mr);
  rmm::device_uvector<protobuf_detail::field_occurrence_count> occurrence_counts(
    field_value_count, stream, mr);
  auto error =
    cudf::detail::make_zeroed_device_uvector_async<protobuf_detail::protobuf_error>(1, stream, mr);

  std::vector<repeated_child_count_scan_work> child_work;
  child_work.reserve(num_repeated_children);
  auto host_scan_descriptors =
    cudf::detail::make_pinned_vector_async<protobuf_detail::field_occurrence_scan_desc>(
      num_repeated_children, stream);
  for (int child_idx = 0; child_idx < num_repeated_children; ++child_idx) {
    auto const total_count = static_cast<int32_t>(data.occurrences_by_child[child_idx].size());
    auto& work             = child_work.emplace_back(num_rows, total_count, stream, mr);
    host_scan_descriptors[child_idx] = protobuf_detail::field_occurrence_scan_desc{
      child_idx + 1, proto_wire_type::LEN, work.offsets.data(), work.occurrences.data()};
  }
  auto occurrence_scan =
    protobuf_detail::make_field_occurrence_scan_bundle(host_scan_descriptors, stream, mr);
  stream.sync();

  protobuf_detail::protobuf_input_view input{.message_data      = message_data,
                                             .message_data_size = message_data_size,
                                             .row_offsets       = row_offsets,
                                             .base_offset       = 0,
                                             .num_rows          = num_rows};
  protobuf_detail::nested_parent_view parent{.locations       = parent_locations.data(),
                                             .location_count  = parent_locations.size(),
                                             .top_row_indices = nullptr};
  protobuf_detail::field_scan_view field_scan{
    .locations               = {field_locations.data(), num_repeated_children},
    .repeated_info           = {occurrence_counts.data(), num_repeated_children},
    .singular_message_info   = {},
    .multiple_message_fields = nullptr,
    .lookup                  = {field_descriptors.device.data(), num_repeated_children}};

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    protobuf_detail::launch_scan_nested_message_fields(
      input, parent, field_scan, error.data(), nullptr, /*recursion_depth=*/1, stream);

    for (int child_idx = 0; child_idx < num_repeated_children; ++child_idx) {
      auto& work        = child_work[child_idx];
      auto counts_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator<int>(0),
        protobuf_detail::extract_strided_count{
          occurrence_counts.data(), child_idx, num_repeated_children});
      auto const actual_total = thrust::reduce(
        rmm::exec_policy_nosync(stream, mr), counts_begin, counts_begin + num_rows, int64_t{0});
      CUDF_EXPECTS(actual_total == work.total_count,
                   "repeated child count differs from generated benchmark data");
      thrust::exclusive_scan(rmm::exec_policy_nosync(stream, mr),
                             counts_begin,
                             counts_begin + num_rows,
                             work.offsets.begin(),
                             int32_t{0});
      thrust::fill_n(
        rmm::exec_policy_nosync(stream, mr), work.offsets.data() + num_rows, 1, work.total_count);
    }

    protobuf_detail::launch_scan_all_field_occurrences_in_nested(
      input, parent, occurrence_scan.view(), error.data(), /*recursion_depth=*/1, stream);
  });
  expect_no_protobuf_error(error, stream);

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(prepared.total_bytes);
}

NVBENCH_BENCH(BM_protobuf_repeated_child_string_count_scan)
  .set_name("Protobuf Repeated Child String CountScan Device Pipeline")
  .add_int64_axis("num_rows", {10'000, 20'000})
  .add_int64_axis("num_repeated_children", {1, 4, 8})
  .add_int64_axis("avg_child_elems", {1, 5});

// ===========================================================================
// Benchmark 9: Repeated child string materialization from precomputed occurrences
// ===========================================================================
static void BM_protobuf_repeated_child_string_build(nvbench::state& state)
{
  auto prepared = prepare_decode_benchmark<RepeatedChildStringOnlyCase>(
    state, int_axis_argument{"num_repeated_children"}, int_axis_argument{"avg_child_elems"});
  auto const num_rows              = prepared.num_rows;
  auto const num_repeated_children = static_cast<int>(prepared.generated.counts_by_child.size());
  auto const& data                 = prepared.generated;

  cuda::stream_ref stream = cudf::get_default_stream();
  auto mr                 = cudf::get_current_device_resource_ref();

  cudf::lists_column_view input_list(prepared.binary_col->view());
  auto const* row_offsets  = input_list.offsets().data<cudf::size_type>();
  auto const child         = input_list.child();
  auto const* message_data = child.data<uint8_t>();

  rmm::device_uvector<protobuf_detail::field_location> parent_locations(num_rows, stream, mr);
  copy_to_device(parent_locations, data.parent_locations, stream);

  std::vector<repeated_child_build_work> child_work;
  child_work.reserve(num_repeated_children);
  for (int child_idx = 0; child_idx < num_repeated_children; ++child_idx) {
    auto const total_count = static_cast<int32_t>(data.occurrences_by_child[child_idx].size());
    auto& work             = child_work.emplace_back(num_rows, total_count, stream, mr);
    copy_to_device(work.counts, data.counts_by_child[child_idx], stream);
    copy_to_device(work.occurrences, data.occurrences_by_child[child_idx], stream);
  }

  protobuf_detail::protobuf_schema schema{prepared.context};
  stream.sync();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::vector<std::unique_ptr<cudf::column>> results;
    results.reserve(num_repeated_children);

    for (int child_idx = 0; child_idx < num_repeated_children; ++child_idx) {
      auto const& work = child_work[child_idx];
      rmm::device_uvector<int32_t> list_offsets(num_rows + 1, stream, mr);
      thrust::exclusive_scan(rmm::exec_policy_nosync(stream, mr),
                             work.counts.begin(),
                             work.counts.end(),
                             list_offsets.begin(),
                             int32_t{0});
      thrust::fill_n(
        rmm::exec_policy_nosync(stream, mr), list_offsets.data() + num_rows, 1, work.total_count);

      protobuf_detail::field_occurrence_location_provider location_provider{
        .input       = {.message_data      = message_data,
                        .message_data_size = static_cast<cudf::size_type>(child.size()),
                        .row_offsets       = row_offsets,
                        .base_offset       = 0,
                        .num_rows          = num_rows},
        .parent      = {.locations       = parent_locations.data(),
                        .location_count  = parent_locations.size(),
                        .top_row_indices = nullptr},
        .occurrences = work.occurrences.data()};
      auto valid = [] __device__(cudf::size_type) { return true; };
      auto child_values =
        protobuf_detail::extract_and_build_string_or_bytes_column(schema.field(child_idx + 1),
                                                                  message_data,
                                                                  work.total_count,
                                                                  location_provider,
                                                                  valid,
                                                                  stream,
                                                                  mr);
      auto offsets_column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                           num_rows + 1,
                                                           list_offsets.release(),
                                                           rmm::device_buffer{},
                                                           0);
      results.push_back(cudf::make_lists_column(
        num_rows, std::move(offsets_column), std::move(child_values), 0, rmm::device_buffer{}));
    }
  });

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(prepared.total_bytes);
}

NVBENCH_BENCH(BM_protobuf_repeated_child_string_build)
  .set_name("Protobuf Repeated Child String Materialization")
  .add_int64_axis("num_rows", {10'000, 20'000})
  .add_int64_axis("num_repeated_children", {1, 4, 8})
  .add_int64_axis("avg_child_elems", {1, 5});

// ===========================================================================
// Benchmark 10: Many repeated fields — measures per-field sync overhead at scale
// ===========================================================================
static void BM_protobuf_many_repeated(nvbench::state& state)
{
  constexpr int string_field_percent = 20;
  auto data =
    prepare_decode_benchmark<ManyRepeatedFieldsCase>(state,
                                                     int_axis_argument{"num_rep_fields"},
                                                     constant_argument{string_field_percent},
                                                     constant_argument{3});

  cuda::stream_ref stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = protobuf::decode_protobuf_to_struct(
      data.binary_col->view(), data.context, stream, cudf::get_current_device_resource_ref());
  });

  state.add_element_count(data.num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(data.total_bytes);
}

NVBENCH_BENCH(BM_protobuf_many_repeated)
  .set_name("Protobuf Many Repeated Fields")
  .add_int64_axis("num_rows", {10'000, 100'000})
  .add_int64_axis("num_rep_fields",
                  {10, 20, 30, protobuf_detail::MAX_REPEATED_FIELDS_PER_KERNEL + 1});
