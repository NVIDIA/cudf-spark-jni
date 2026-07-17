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

#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace protobuf        = spark_rapids_jni::protobuf;
namespace protobuf_detail = spark_rapids_jni::protobuf::detail;

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
  encode_tag(buf, field_number, protobuf::wire_type_value(protobuf::proto_wire_type::VARINT));
  encode_varint(buf, static_cast<uint64_t>(value));
}

void encode_fixed32_field(std::vector<uint8_t>& buf, int field_number, float value)
{
  encode_tag(buf, field_number, protobuf::wire_type_value(protobuf::proto_wire_type::I32BIT));
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int i = 0; i < 4; i++) {
    buf.push_back(static_cast<uint8_t>(bits & 0xFF));
    bits >>= 8;
  }
}

void encode_fixed64_field(std::vector<uint8_t>& buf, int field_number, double value)
{
  encode_tag(buf, field_number, protobuf::wire_type_value(protobuf::proto_wire_type::I64BIT));
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int i = 0; i < 8; i++) {
    buf.push_back(static_cast<uint8_t>(bits & 0xFF));
    bits >>= 8;
  }
}

void encode_len_field(std::vector<uint8_t>& buf, int field_number, void const* data, size_t len)
{
  encode_tag(buf, field_number, protobuf::wire_type_value(protobuf::proto_wire_type::LEN));
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
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  std::vector<int32_t> h_offsets(messages.size() + 1);
  h_offsets[0] = 0;
  for (size_t i = 0; i < messages.size(); i++) {
    h_offsets[i + 1] = h_offsets[i] + static_cast<int32_t>(messages[i].size());
  }
  int32_t total_bytes = h_offsets.back();

  std::vector<uint8_t> h_data;
  h_data.reserve(total_bytes);
  for (auto const& m : messages) {
    h_data.insert(h_data.end(), m.begin(), m.end());
  }

  rmm::device_buffer d_data(h_data.data(), h_data.size(), stream, mr);
  rmm::device_buffer d_offsets(h_offsets.data(), h_offsets.size() * sizeof(int32_t), stream, mr);
  stream.synchronize();

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

void initialize_context_metadata(protobuf::protobuf_decode_context& context)
{
  auto const size = context.schema.size();
  context.default_ints.resize(size, 0);
  context.default_floats.resize(size, 0.0);
  context.default_bools.resize(size, false);
  context.default_strings.reserve(size);
  context.enum_valid_values.reserve(size);
  auto const stream = cudf::get_default_stream();
  for (size_t i = 0; i < size; ++i) {
    context.default_strings.emplace_back(
      cudf::detail::make_pinned_vector_async<uint8_t>(0, stream));
    context.enum_valid_values.emplace_back(
      cudf::detail::make_pinned_vector_async<int32_t>(0, stream));
  }
  context.enum_names.resize(size);
}

// Case 1: Flat scalars only — many top-level scalar fields.
//   message FlatMessage {
//     int32  f1 = 1;
//     int64  f2 = 2;
//     ...
//     float  f_k   = k;     (cycling through int32, int64, float, double, bool)
//     string s_k+1 = k+1;   (a few string fields)
//   }
struct FlatScalarCase {
  int num_non_string_fields;
  int num_string_fields;

  protobuf::protobuf_decode_context build_context() const
  {
    protobuf::protobuf_decode_context ctx;
    ctx.fail_on_errors = true;

    cudf::type_id non_string_types[] = {cudf::type_id::INT32,
                                        cudf::type_id::INT64,
                                        cudf::type_id::FLOAT32,
                                        cudf::type_id::FLOAT64,
                                        cudf::type_id::BOOL8};
    proto_wire_type wt_for_type[]    = {proto_wire_type::VARINT,
                                        proto_wire_type::VARINT,
                                        proto_wire_type::I32BIT,
                                        proto_wire_type::I64BIT,
                                        proto_wire_type::VARINT};

    int fn = 1;
    for (int i = 0; i < num_non_string_fields; i++, fn++) {
      int ti   = i % 5;
      auto ty  = non_string_types[ti];
      auto wt  = wt_for_type[ti];
      auto enc = proto_encoding::DEFAULT;
      if (ty == cudf::type_id::FLOAT32) { enc = proto_encoding::FIXED; }
      if (ty == cudf::type_id::FLOAT64) { enc = proto_encoding::FIXED; }
      ctx.schema.push_back(make_field_descriptor(fn, -1, 0, wt, ty, false, enc));
    }
    for (int i = 0; i < num_string_fields; i++, fn++) {
      ctx.schema.push_back(
        make_field_descriptor(fn, -1, 0, proto_wire_type::LEN, cudf::type_id::STRING));
    }

    initialize_context_metadata(ctx);
    return ctx;
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
      for (int i = 0; i < num_non_string_fields; i++, fn++) {
        int ti = i % 5;
        switch (ti) {
          case 0: encode_varint_field(buf, fn, int_dist(rng)); break;
          case 1: encode_varint_field(buf, fn, int_dist(rng)); break;
          case 2: encode_fixed32_field(buf, fn, static_cast<float>(int_dist(rng))); break;
          case 3: encode_fixed64_field(buf, fn, static_cast<double>(int_dist(rng))); break;
          case 4: encode_varint_field(buf, fn, rng() % 2); break;
        }
      }
      for (int i = 0; i < num_string_fields; i++, fn++) {
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
  int num_inner_fields;  // scalar fields inside InnerMessage

  protobuf::protobuf_decode_context build_context() const
  {
    protobuf::protobuf_decode_context ctx;
    ctx.fail_on_errors = true;

    // idx 0: id (int32, top-level)
    ctx.schema.push_back(
      make_field_descriptor(1, -1, 0, proto_wire_type::VARINT, cudf::type_id::INT32));
    // idx 1: name (string, top-level)
    ctx.schema.push_back(
      make_field_descriptor(2, -1, 0, proto_wire_type::LEN, cudf::type_id::STRING));
    // idx 2: inner (STRUCT, top-level)
    ctx.schema.push_back(
      make_field_descriptor(3, -1, 0, proto_wire_type::LEN, cudf::type_id::STRUCT));

    // Inner message children (parent_idx=2, depth=1)
    cudf::type_id inner_types[] = {
      cudf::type_id::INT32, cudf::type_id::INT64, cudf::type_id::STRING};
    proto_wire_type inner_wt[] = {
      proto_wire_type::VARINT, proto_wire_type::VARINT, proto_wire_type::LEN};

    for (int i = 0; i < num_inner_fields; i++) {
      int ti = i % 3;
      ctx.schema.push_back(make_field_descriptor(i + 1, 2, 1, inner_wt[ti], inner_types[ti]));
    }

    initialize_context_metadata(ctx);
    return ctx;
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
          int ti = i % 3;
          switch (ti) {
            case 0: encode_varint_field(inner, i + 1, int_dist(rng)); break;
            case 1: encode_varint_field(inner, i + 1, int_dist(rng)); break;
            case 2: encode_string_field(inner, i + 1, random_string(str_len_dist(rng))); break;
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
    protobuf::protobuf_decode_context ctx;
    ctx.fail_on_errors = true;

    // idx 0: id (int32, scalar)
    ctx.schema.push_back(
      make_field_descriptor(1, -1, 0, proto_wire_type::VARINT, cudf::type_id::INT32));
    // idx 1: tags (repeated int32, packed)
    ctx.schema.push_back(
      make_field_descriptor(2, -1, 0, proto_wire_type::VARINT, cudf::type_id::INT32, true));
    // idx 2: labels (repeated string)
    ctx.schema.push_back(
      make_field_descriptor(3, -1, 0, proto_wire_type::LEN, cudf::type_id::STRING, true));
    // idx 3: items (repeated STRUCT)
    ctx.schema.push_back(
      make_field_descriptor(4, -1, 0, proto_wire_type::LEN, cudf::type_id::STRUCT, true));
    // idx 4: Item.item_id (int32, child of idx 3)
    ctx.schema.push_back(
      make_field_descriptor(1, 3, 1, proto_wire_type::VARINT, cudf::type_id::INT32));
    // idx 5: Item.item_name (string, child of idx 3)
    ctx.schema.push_back(
      make_field_descriptor(2, 3, 1, proto_wire_type::LEN, cudf::type_id::STRING));
    // idx 6: Item.value (int64, child of idx 3)
    ctx.schema.push_back(
      make_field_descriptor(3, 3, 1, proto_wire_type::VARINT, cudf::type_id::INT64));

    initialize_context_metadata(ctx);
    return ctx;
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

    // Vary count per row around the average (±50%)
    auto vary = [&](int avg) -> int {
      int lo = std::max(0, avg / 2);
      int hi = avg + avg / 2;
      return std::uniform_int_distribution<int>(lo, std::max(lo, hi))(rng);
    };

    for (int r = 0; r < num_rows; r++) {
      auto& buf = messages[r];

      // id
      encode_varint_field(buf, 1, int_dist(rng));

      // tags (packed repeated int32)
      {
        int n = vary(avg_tags_per_row);
        std::vector<int32_t> tags(n);
        for (auto& t : tags)
          t = int_dist(rng);
        if (n > 0) encode_packed_repeated_int32(buf, 2, tags);
      }

      // labels (unpacked repeated string)
      {
        int n = vary(avg_labels_per_row);
        for (int i = 0; i < n; i++) {
          encode_string_field(buf, 3, random_string(str_len_dist(rng)));
        }
      }

      // items (repeated nested message)
      {
        int n = vary(avg_items_per_row);
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
  int num_child_fields;
  int avg_items_per_row;

  protobuf::protobuf_decode_context build_context() const
  {
    protobuf::protobuf_decode_context ctx;
    ctx.fail_on_errors = true;

    // idx 0: id (scalar)
    ctx.schema.push_back(
      make_field_descriptor(1, -1, 0, proto_wire_type::VARINT, cudf::type_id::INT32));
    // idx 1: items (repeated STRUCT)
    ctx.schema.push_back(
      make_field_descriptor(2, -1, 0, proto_wire_type::LEN, cudf::type_id::STRUCT, true));

    cudf::type_id child_types[] = {cudf::type_id::INT32,
                                   cudf::type_id::INT64,
                                   cudf::type_id::FLOAT32,
                                   cudf::type_id::FLOAT64,
                                   cudf::type_id::BOOL8,
                                   cudf::type_id::STRING};
    proto_wire_type child_wt[]  = {proto_wire_type::VARINT,
                                   proto_wire_type::VARINT,
                                   proto_wire_type::I32BIT,
                                   proto_wire_type::I64BIT,
                                   proto_wire_type::VARINT,
                                   proto_wire_type::LEN};
    proto_encoding child_enc[]  = {proto_encoding::DEFAULT,
                                   proto_encoding::DEFAULT,
                                   proto_encoding::FIXED,
                                   proto_encoding::FIXED,
                                   proto_encoding::DEFAULT,
                                   proto_encoding::DEFAULT};

    // Keep strings sparse so the case remains dominated by wide child scanning
    // rather than varlen copy traffic.
    for (int i = 0; i < num_child_fields; i++) {
      int ti = (i % 10 == 9) ? 5 : (i % 5);
      ctx.schema.push_back(
        make_field_descriptor(i + 1, 1, 1, child_wt[ti], child_types[ti], false, child_enc[ti]));
    }

    initialize_context_metadata(ctx);
    return ctx;
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

    auto vary = [&](int avg) -> int {
      int lo = std::max(0, avg / 2);
      int hi = avg + avg / 2;
      return std::uniform_int_distribution<int>(lo, std::max(lo, hi))(rng);
    };

    for (int r = 0; r < num_rows; r++) {
      auto& buf = messages[r];
      encode_varint_field(buf, 1, int_dist(rng));

      int n = vary(avg_items_per_row);
      for (int item_idx = 0; item_idx < n; item_idx++) {
        encode_nested_message(buf, 2, [&](std::vector<uint8_t>& inner) {
          for (int i = 0; i < num_child_fields; i++) {
            int ti = (i % 10 == 9) ? 5 : (i % 5);
            int fn = i + 1;
            switch (ti) {
              case 0: encode_varint_field(inner, fn, int_dist(rng)); break;
              case 1: encode_varint_field(inner, fn, int_dist(rng)); break;
              case 2: encode_fixed32_field(inner, fn, static_cast<float>(int_dist(rng))); break;
              case 3: encode_fixed64_field(inner, fn, static_cast<double>(int_dist(rng))); break;
              case 4: encode_varint_field(inner, fn, rng() % 2); break;
              case 5: encode_string_field(inner, fn, random_string(str_len_dist(rng))); break;
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
    protobuf::protobuf_decode_context ctx;
    ctx.fail_on_errors = true;

    // idx 0: id (scalar)
    ctx.schema.push_back(
      make_field_descriptor(1, -1, 0, proto_wire_type::VARINT, cudf::type_id::INT32));
    // idx 1: items (repeated STRUCT)
    ctx.schema.push_back(
      make_field_descriptor(2, -1, 0, proto_wire_type::LEN, cudf::type_id::STRUCT, true));

    for (int i = 0; i < num_repeated_children; i++) {
      bool as_string = child_is_string(i);
      ctx.schema.push_back(
        make_field_descriptor(i + 1,
                              1,
                              1,
                              as_string ? proto_wire_type::LEN : proto_wire_type::VARINT,
                              as_string ? cudf::type_id::STRING : cudf::type_id::INT32,
                              true));
    }

    initialize_context_metadata(ctx);
    return ctx;
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

    auto vary = [&](int avg) -> int {
      int lo = std::max(0, avg / 2);
      int hi = avg + avg / 2;
      return std::uniform_int_distribution<int>(lo, std::max(lo, hi))(rng);
    };

    for (int r = 0; r < num_rows; r++) {
      auto& buf = messages[r];
      encode_varint_field(buf, 1, int_dist(rng));

      int num_items = vary(avg_items_per_row);
      for (int item_idx = 0; item_idx < num_items; item_idx++) {
        encode_nested_message(buf, 2, [&](std::vector<uint8_t>& inner) {
          for (int child_idx = 0; child_idx < num_repeated_children; child_idx++) {
            int fn        = child_idx + 1;
            bool is_str   = child_is_string(child_idx);
            int num_elems = vary(avg_child_elems);
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
    protobuf::protobuf_decode_context ctx;
    ctx.fail_on_errors = true;
    ctx.schema.push_back(
      make_field_descriptor(1, -1, 0, proto_wire_type::LEN, cudf::type_id::STRUCT, true));
    ctx.schema.push_back(
      make_field_descriptor(1, 0, 1, proto_wire_type::LEN, cudf::type_id::STRUCT, true));
    ctx.schema.push_back(
      make_field_descriptor(1, 1, 2, proto_wire_type::VARINT, cudf::type_id::INT32));
    ctx.schema.push_back(
      make_field_descriptor(2, 1, 2, proto_wire_type::LEN, cudf::type_id::STRING));
    initialize_context_metadata(ctx);
    return ctx;
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
    auto vary = [&](int average) {
      auto const lower = std::max(0, average / 2);
      auto const upper = average + average / 2;
      return std::uniform_int_distribution<int>(lower, std::max(lower, upper))(rng);
    };

    for (auto& message : messages) {
      auto const num_outer_items = vary(avg_outer_items);
      for (int outer_idx = 0; outer_idx < num_outer_items; ++outer_idx) {
        encode_nested_message(message, 1, [&](std::vector<uint8_t>& outer) {
          auto const num_inner_items = vary(avg_inner_items);
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
    protobuf::protobuf_decode_context ctx;
    ctx.fail_on_errors = true;

    for (int message_idx = 0; message_idx < num_message_fields; ++message_idx) {
      auto const parent_idx = static_cast<int>(ctx.schema.size());
      ctx.schema.push_back(
        make_field_descriptor(message_idx + 1, -1, 0, proto_wire_type::LEN, cudf::type_id::STRUCT));
      for (int child_idx = 0; child_idx < NUM_CHILD_FIELDS; ++child_idx) {
        ctx.schema.push_back(make_field_descriptor(
          child_idx + 1, parent_idx, 1, proto_wire_type::VARINT, cudf::type_id::INT32));
      }
    }

    initialize_context_metadata(ctx);
    return ctx;
  }

  std::vector<std::vector<uint8_t>> generate_messages(int num_rows) const
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
  encode_tag(buf, field_number, protobuf::wire_type_value(proto_wire_type::LEN));
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
    protobuf::protobuf_decode_context ctx;
    ctx.fail_on_errors = true;
    ctx.schema.push_back(
      make_field_descriptor(1, -1, 0, proto_wire_type::LEN, cudf::type_id::STRUCT));
    for (int child_idx = 0; child_idx < num_repeated_children; ++child_idx) {
      ctx.schema.push_back(make_field_descriptor(
        child_idx + 1, 0, 1, proto_wire_type::LEN, cudf::type_id::STRING, true));
    }
    initialize_context_metadata(ctx);
    return ctx;
  }

  RepeatedChildStringBenchData generate_data(int num_rows, std::mt19937& rng) const
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
    auto vary = [&](int average) {
      auto const lower = std::max(0, average / 2);
      auto const upper = average + average / 2;
      return std::uniform_int_distribution<int>(lower, std::max(lower, upper))(rng);
    };

    for (int row = 0; row < num_rows; ++row) {
      auto& message = result.messages[row];
      for (int child_idx = 0; child_idx < num_repeated_children; ++child_idx) {
        auto const num_elements = vary(avg_child_elems);
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
                    rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(destination.size() == source.size(), "benchmark H2D size mismatch");
  if (!source.empty()) {
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(
      destination.data(), source.data(), source.size() * sizeof(T), stream));
  }
}

struct repeated_child_count_scan_work {
  int32_t total_count;
  rmm::device_uvector<int32_t> offsets;
  rmm::device_uvector<protobuf_detail::field_occurrence> occurrences;

  repeated_child_count_scan_work(int num_rows,
                                 int32_t count,
                                 rmm::cuda_stream_view stream,
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
                            rmm::cuda_stream_view stream,
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
  int num_repeated_int;
  int num_repeated_str;

  protobuf::protobuf_decode_context build_context() const
  {
    protobuf::protobuf_decode_context ctx;
    ctx.fail_on_errors = true;

    int fn = 1;
    // idx 0: id (scalar)
    ctx.schema.push_back(
      make_field_descriptor(fn++, -1, 0, proto_wire_type::VARINT, cudf::type_id::INT32));

    for (int i = 0; i < num_repeated_int; i++) {
      ctx.schema.push_back(
        make_field_descriptor(fn++, -1, 0, proto_wire_type::VARINT, cudf::type_id::INT32, true));
    }
    for (int i = 0; i < num_repeated_str; i++) {
      ctx.schema.push_back(
        make_field_descriptor(fn++, -1, 0, proto_wire_type::LEN, cudf::type_id::STRING, true));
    }

    initialize_context_metadata(ctx);
    return ctx;
  }

  std::vector<std::vector<uint8_t>> generate_messages(int num_rows,
                                                      int avg_elems_per_field,
                                                      std::mt19937& rng) const
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
    auto vary = [&](int avg) -> int {
      int lo = std::max(0, avg / 2);
      int hi = avg + avg / 2;
      return std::uniform_int_distribution<int>(lo, std::max(lo, hi))(rng);
    };

    for (int r = 0; r < num_rows; r++) {
      auto& buf = messages[r];
      int fn    = 1;

      encode_varint_field(buf, fn++, int_dist(rng));

      for (int i = 0; i < num_repeated_int; i++) {
        int cur_fn = fn++;
        int n      = vary(avg_elems_per_field);
        if (n > 0) {
          std::vector<int32_t> vals(n);
          for (auto& v : vals)
            v = int_dist(rng);
          encode_packed_repeated_int32(buf, cur_fn, vals);
        }
      }
      for (int i = 0; i < num_repeated_str; i++) {
        int cur_fn = fn++;
        int n      = vary(avg_elems_per_field);
        for (int j = 0; j < n; j++) {
          encode_string_field(buf, cur_fn, random_string(str_len_dist(rng)));
        }
      }
    }
    return messages;
  }
};

}  // anonymous namespace

// ===========================================================================
// Benchmark 1: Flat scalars — measures per-field extraction overhead
// ===========================================================================
static void BM_protobuf_flat_scalars(nvbench::state& state)
{
  auto const num_rows      = static_cast<int>(state.get_int64("num_rows"));
  auto const num_fields    = static_cast<int>(state.get_int64("num_fields"));
  int const num_str        = std::max(1, num_fields / 10);
  int const num_non_string = num_fields - num_str;

  FlatScalarCase flat_case{num_non_string, num_str};
  auto ctx = flat_case.build_context();

  std::mt19937 rng(42);
  auto messages   = flat_case.generate_messages(num_rows, rng);
  auto binary_col = make_binary_column(messages);

  size_t total_bytes = 0;
  for (auto const& m : messages)
    total_bytes += m.size();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = protobuf::decode_protobuf_to_struct(
      binary_col->view(), ctx, stream, cudf::get_current_device_resource_ref());
  });

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(total_bytes);
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
  auto const num_rows     = static_cast<int>(state.get_int64("num_rows"));
  auto const inner_fields = static_cast<int>(state.get_int64("inner_fields"));

  NestedMessageCase nested_case{inner_fields};
  auto ctx = nested_case.build_context();

  std::mt19937 rng(42);
  auto messages   = nested_case.generate_messages(num_rows, rng);
  auto binary_col = make_binary_column(messages);

  size_t total_bytes = 0;
  for (auto const& m : messages)
    total_bytes += m.size();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = protobuf::decode_protobuf_to_struct(
      binary_col->view(), ctx, stream, cudf::get_current_device_resource_ref());
  });

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(total_bytes);
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
  auto const num_rows  = static_cast<int>(state.get_int64("num_rows"));
  auto const avg_items = static_cast<int>(state.get_int64("avg_items"));

  RepeatedFieldCase rep_case{/*avg_tags=*/5, /*avg_labels=*/3, /*avg_items=*/avg_items};
  auto ctx = rep_case.build_context();

  std::mt19937 rng(42);
  auto messages   = rep_case.generate_messages(num_rows, rng);
  auto binary_col = make_binary_column(messages);

  size_t total_bytes = 0;
  for (auto const& m : messages)
    total_bytes += m.size();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = protobuf::decode_protobuf_to_struct(
      binary_col->view(), ctx, stream, cudf::get_current_device_resource_ref());
  });

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(total_bytes);
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
  auto const num_rows         = static_cast<int>(state.get_int64("num_rows"));
  auto const num_child_fields = static_cast<int>(state.get_int64("num_child_fields"));
  auto const avg_items        = static_cast<int>(state.get_int64("avg_items"));

  WideRepeatedMessageCase wide_case{num_child_fields, avg_items};
  auto ctx = wide_case.build_context();

  std::mt19937 rng(42);
  auto messages   = wide_case.generate_messages(num_rows, rng);
  auto binary_col = make_binary_column(messages);

  size_t total_bytes = 0;
  for (auto const& m : messages)
    total_bytes += m.size();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = protobuf::decode_protobuf_to_struct(
      binary_col->view(), ctx, stream, cudf::get_current_device_resource_ref());
  });

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(total_bytes);
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
  auto const num_rows              = static_cast<int>(state.get_int64("num_rows"));
  auto const num_repeated_children = static_cast<int>(state.get_int64("num_repeated_children"));
  auto const avg_items             = static_cast<int>(state.get_int64("avg_items"));
  auto const avg_child_elems       = static_cast<int>(state.get_int64("avg_child_elems"));
  auto const child_mix             = state.get_string("child_mix");

  RepeatedChildListCase list_case{
    num_repeated_children, avg_items, avg_child_elems, std::string(child_mix)};
  auto ctx = list_case.build_context();

  std::mt19937 rng(42);
  auto messages   = list_case.generate_messages(num_rows, rng);
  auto binary_col = make_binary_column(messages);

  size_t total_bytes = 0;
  for (auto const& m : messages)
    total_bytes += m.size();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = protobuf::decode_protobuf_to_struct(
      binary_col->view(), ctx, stream, cudf::get_current_device_resource_ref());
  });

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(total_bytes);
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
  auto const num_rows        = static_cast<int>(state.get_int64("num_rows"));
  auto const avg_outer_items = static_cast<int>(state.get_int64("avg_outer_items"));
  auto const avg_inner_items = static_cast<int>(state.get_int64("avg_inner_items"));

  RepeatedMessageNestingCase nesting_case{avg_outer_items, avg_inner_items};
  auto context = nesting_case.build_context();
  std::mt19937 rng(42);
  auto messages = nesting_case.generate_messages(num_rows, rng);
  auto input    = make_binary_column(messages);

  size_t total_bytes = 0;
  for (auto const& message : messages) {
    total_bytes += message.size();
  }

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = protobuf::decode_protobuf_to_struct(
      input->view(), context, stream, cudf::get_current_device_resource_ref());
  });

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(total_bytes);
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
  auto const num_rows              = static_cast<int>(state.get_int64("num_rows"));
  auto const num_message_fields    = static_cast<int>(state.get_int64("num_message_fields"));
  auto const occurrences_per_field = static_cast<int>(state.get_int64("occurrences_per_field"));

  SingularMessageMergeCase merge_case{num_message_fields, occurrences_per_field};
  auto ctx      = merge_case.build_context();
  auto messages = merge_case.generate_messages(num_rows);
  auto input    = make_binary_column(messages);

  size_t total_bytes = 0;
  for (auto const& message : messages) {
    total_bytes += message.size();
  }

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = protobuf::decode_protobuf_to_struct(
      input->view(), ctx, stream, cudf::get_current_device_resource_ref());
  });

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(total_bytes);
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
  auto const num_rows              = static_cast<int>(state.get_int64("num_rows"));
  auto const num_repeated_children = static_cast<int>(state.get_int64("num_repeated_children"));
  auto const avg_child_elems       = static_cast<int>(state.get_int64("avg_child_elems"));

  RepeatedChildStringOnlyCase string_case{num_repeated_children, avg_child_elems};
  std::mt19937 rng(42);
  auto data       = string_case.generate_data(num_rows, rng);
  auto binary_col = make_binary_column(data.messages);
  auto context    = string_case.build_context();

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  cudf::lists_column_view input_list(binary_col->view());
  auto const* row_offsets      = input_list.offsets().data<cudf::size_type>();
  auto const child             = input_list.child();
  auto const* message_data     = reinterpret_cast<uint8_t const*>(child.data<int8_t>());
  auto const message_data_size = static_cast<cudf::size_type>(child.size());

  rmm::device_uvector<protobuf_detail::field_location> parent_locations(num_rows, stream, mr);
  copy_to_device(parent_locations, data.parent_locations, stream);

  protobuf_detail::protobuf_schema schema{context};
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

  std::vector<std::unique_ptr<repeated_child_count_scan_work>> child_work;
  child_work.reserve(num_repeated_children);
  auto host_scan_descriptors =
    cudf::detail::make_pinned_vector_async<protobuf_detail::field_occurrence_scan_desc>(
      num_repeated_children, stream);
  for (int child_idx = 0; child_idx < num_repeated_children; ++child_idx) {
    auto const total_count = static_cast<int32_t>(data.occurrences_by_child[child_idx].size());
    auto& work             = *child_work.emplace_back(
      std::make_unique<repeated_child_count_scan_work>(num_rows, total_count, stream, mr));
    host_scan_descriptors[child_idx] = {child_idx + 1,
                                        protobuf::wire_type_value(proto_wire_type::LEN),
                                        work.offsets.data(),
                                        work.occurrences.data()};
  }
  auto occurrence_scan =
    protobuf_detail::make_field_occurrence_scan_bundle(host_scan_descriptors, stream, mr);
  stream.synchronize();

  protobuf_detail::protobuf_input_view input{
    message_data, message_data_size, row_offsets, 0, num_rows};
  protobuf_detail::nested_parent_view parent{
    parent_locations.data(), parent_locations.size(), nullptr};
  protobuf_detail::field_scan_view field_scan{
    field_locations.data(),
    occurrence_counts.data(),
    nullptr,
    nullptr,
    {field_descriptors.device.data(), num_repeated_children, nullptr, 0}};

  size_t total_bytes = 0;
  for (auto const& message : data.messages) {
    total_bytes += message.size();
  }

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    protobuf_detail::launch_scan_nested_message_fields(
      input, parent, field_scan, error.data(), nullptr, 1, stream);

    for (int child_idx = 0; child_idx < num_repeated_children; ++child_idx) {
      auto& work        = *child_work[child_idx];
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
      input, parent, occurrence_scan.view(), error.data(), 1, stream);
  });

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(total_bytes);
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
  auto const num_rows              = static_cast<int>(state.get_int64("num_rows"));
  auto const num_repeated_children = static_cast<int>(state.get_int64("num_repeated_children"));
  auto const avg_child_elems       = static_cast<int>(state.get_int64("avg_child_elems"));

  RepeatedChildStringOnlyCase string_case{num_repeated_children, avg_child_elems};
  std::mt19937 rng(42);
  auto data       = string_case.generate_data(num_rows, rng);
  auto binary_col = make_binary_column(data.messages);
  auto context    = string_case.build_context();

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  cudf::lists_column_view input_list(binary_col->view());
  auto const* row_offsets  = input_list.offsets().data<cudf::size_type>();
  auto const child         = input_list.child();
  auto const* message_data = reinterpret_cast<uint8_t const*>(child.data<int8_t>());

  rmm::device_uvector<protobuf_detail::field_location> parent_locations(num_rows, stream, mr);
  copy_to_device(parent_locations, data.parent_locations, stream);

  std::vector<std::unique_ptr<repeated_child_build_work>> child_work;
  child_work.reserve(num_repeated_children);
  for (int child_idx = 0; child_idx < num_repeated_children; ++child_idx) {
    auto const total_count = static_cast<int32_t>(data.occurrences_by_child[child_idx].size());
    auto& work             = *child_work.emplace_back(
      std::make_unique<repeated_child_build_work>(num_rows, total_count, stream, mr));
    copy_to_device(work.counts, data.counts_by_child[child_idx], stream);
    copy_to_device(work.occurrences, data.occurrences_by_child[child_idx], stream);
  }

  protobuf_detail::protobuf_schema schema{context};
  stream.synchronize();

  size_t total_bytes = 0;
  for (auto const& message : data.messages) {
    total_bytes += message.size();
  }

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::vector<std::unique_ptr<cudf::column>> results;
    results.reserve(num_repeated_children);

    for (int child_idx = 0; child_idx < num_repeated_children; ++child_idx) {
      auto const& work = *child_work[child_idx];
      rmm::device_uvector<int32_t> list_offsets(num_rows + 1, stream, mr);
      thrust::exclusive_scan(rmm::exec_policy_nosync(stream, mr),
                             work.counts.begin(),
                             work.counts.end(),
                             list_offsets.begin(),
                             int32_t{0});
      thrust::fill_n(
        rmm::exec_policy_nosync(stream, mr), list_offsets.data() + num_rows, 1, work.total_count);

      protobuf_detail::nested_repeated_location_provider location_provider{
        row_offsets, 0, parent_locations.data(), work.occurrences.data()};
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
  state.add_global_memory_reads<nvbench::int8_t>(total_bytes);
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
  auto const num_rows       = static_cast<int>(state.get_int64("num_rows"));
  auto const num_rep_fields = static_cast<int>(state.get_int64("num_rep_fields"));

  int const num_rep_str = std::max(1, num_rep_fields / 5);
  int const num_rep_int = num_rep_fields - num_rep_str;

  ManyRepeatedFieldsCase many_case{num_rep_int, num_rep_str};
  auto ctx = many_case.build_context();

  std::mt19937 rng(42);
  auto messages   = many_case.generate_messages(num_rows, /*avg_elems=*/3, rng);
  auto binary_col = make_binary_column(messages);

  size_t total_bytes = 0;
  for (auto const& m : messages)
    total_bytes += m.size();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = protobuf::decode_protobuf_to_struct(
      binary_col->view(), ctx, stream, cudf::get_current_device_resource_ref());
  });

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(total_bytes);
}

NVBENCH_BENCH(BM_protobuf_many_repeated)
  .set_name("Protobuf Many Repeated Fields")
  .add_int64_axis("num_rows", {10'000, 100'000})
  .add_int64_axis("num_rep_fields", {10, 20, 30});
