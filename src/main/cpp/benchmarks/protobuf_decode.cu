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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <nvbench/nvbench.cuh>
#include <protobuf.hpp>
#include <protobuf_common.cuh>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

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
  encode_tag(buf,
             field_number,
             spark_rapids_jni::wire_type_value(spark_rapids_jni::proto_wire_type::VARINT));
  encode_varint(buf, static_cast<uint64_t>(value));
}

void encode_fixed32_field(std::vector<uint8_t>& buf, int field_number, float value)
{
  encode_tag(buf,
             field_number,
             spark_rapids_jni::wire_type_value(spark_rapids_jni::proto_wire_type::I32BIT));
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int i = 0; i < 4; i++) {
    buf.push_back(static_cast<uint8_t>(bits & 0xFF));
    bits >>= 8;
  }
}

void encode_fixed64_field(std::vector<uint8_t>& buf, int field_number, double value)
{
  encode_tag(buf,
             field_number,
             spark_rapids_jni::wire_type_value(spark_rapids_jni::proto_wire_type::I64BIT));
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int i = 0; i < 8; i++) {
    buf.push_back(static_cast<uint8_t>(bits & 0xFF));
    bits >>= 8;
  }
}

void encode_len_field(std::vector<uint8_t>& buf, int field_number, void const* data, size_t len)
{
  encode_tag(
    buf, field_number, spark_rapids_jni::wire_type_value(spark_rapids_jni::proto_wire_type::LEN));
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
  auto mr     = rmm::mr::get_current_device_resource();

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

using nfd                    = spark_rapids_jni::nested_field_descriptor;
using pb_field_location      = spark_rapids_jni::protobuf_detail::field_location;
using pb_repeated_occurrence = spark_rapids_jni::protobuf_detail::repeated_occurrence;

inline int32_t checked_size_to_i32(size_t value, char const* what)
{
  if (value > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::overflow_error(std::string("benchmark protobuf size exceeds int32_t for ") + what);
  }
  return static_cast<int32_t>(value);
}

void encode_string_field_record(std::vector<uint8_t>& buf,
                                int field_number,
                                std::string const& s,
                                std::vector<pb_repeated_occurrence>& out_occurrences,
                                int32_t row_idx)
{
  encode_tag(buf,
             field_number,
             /*spark_rapids_jni::wire_type_value(spark_rapids_jni::proto_wire_type::LEN)=*/2);
  encode_varint(buf, s.size());
  auto const data_offset = checked_size_to_i32(buf.size(), "string field data offset");
  buf.insert(buf.end(), s.begin(), s.end());
  out_occurrences.push_back(
    {row_idx, data_offset, checked_size_to_i32(s.size(), "string field length")});
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
  int num_int_fields;
  int num_string_fields;

  spark_rapids_jni::protobuf_decode_context build_context() const
  {
    spark_rapids_jni::protobuf_decode_context ctx;
    ctx.fail_on_errors = true;

    // type_id cycle for integer-like fields
    cudf::type_id int_types[] = {cudf::type_id::INT32,
                                 cudf::type_id::INT64,
                                 cudf::type_id::FLOAT32,
                                 cudf::type_id::FLOAT64,
                                 cudf::type_id::BOOL8};
    int wt_for_type[]         = {
      spark_rapids_jni::wire_type_value(spark_rapids_jni::proto_wire_type::VARINT),
      spark_rapids_jni::wire_type_value(spark_rapids_jni::proto_wire_type::VARINT),
      spark_rapids_jni::wire_type_value(spark_rapids_jni::proto_wire_type::I32BIT),
      spark_rapids_jni::wire_type_value(spark_rapids_jni::proto_wire_type::I64BIT),
      spark_rapids_jni::wire_type_value(spark_rapids_jni::proto_wire_type::VARINT)};

    int fn = 1;
    for (int i = 0; i < num_int_fields; i++, fn++) {
      int ti  = i % 5;
      auto ty = int_types[ti];
      int wt  = wt_for_type[ti];
      int enc = spark_rapids_jni::encoding_value(spark_rapids_jni::proto_encoding::DEFAULT);
      if (ty == cudf::type_id::FLOAT32)
        enc = spark_rapids_jni::encoding_value(spark_rapids_jni::proto_encoding::FIXED);
      if (ty == cudf::type_id::FLOAT64)
        enc = spark_rapids_jni::encoding_value(spark_rapids_jni::proto_encoding::FIXED);
      ctx.schema.push_back({fn, -1, 0, wt, ty, enc, false, false, false});
    }
    for (int i = 0; i < num_string_fields; i++, fn++) {
      ctx.schema.push_back(
        {fn,
         -1,
         0,
         spark_rapids_jni::wire_type_value(spark_rapids_jni::proto_wire_type::LEN),
         cudf::type_id::STRING,
         spark_rapids_jni::encoding_value(spark_rapids_jni::proto_encoding::DEFAULT),
         false,
         false,
         false});
    }

    size_t n = ctx.schema.size();
    for (auto const& f : ctx.schema) {
      ctx.schema_output_types.emplace_back(f.output_type);
    }
    ctx.default_ints.resize(n, 0);
    ctx.default_floats.resize(n, 0.0);
    ctx.default_bools.resize(n, false);
    ctx.default_strings.resize(n);
    ctx.enum_valid_values.resize(n);
    ctx.enum_names.resize(n);
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
      for (int i = 0; i < num_int_fields; i++, fn++) {
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

  spark_rapids_jni::protobuf_decode_context build_context() const
  {
    spark_rapids_jni::protobuf_decode_context ctx;
    ctx.fail_on_errors = true;

    // idx 0: id (int32, top-level)
    ctx.schema.push_back({1, -1, 0, 0, cudf::type_id::INT32, 0, false, false, false});
    // idx 1: name (string, top-level)
    ctx.schema.push_back({2, -1, 0, 2, cudf::type_id::STRING, 0, false, false, false});
    // idx 2: inner (STRUCT, top-level)
    ctx.schema.push_back({3, -1, 0, 2, cudf::type_id::STRUCT, 0, false, false, false});

    // Inner message children (parent_idx=2, depth=1)
    cudf::type_id inner_types[] = {
      cudf::type_id::INT32, cudf::type_id::INT64, cudf::type_id::STRING};
    int inner_wt[] = {0, 0, 2};

    for (int i = 0; i < num_inner_fields; i++) {
      int ti = i % 3;
      ctx.schema.push_back({i + 1, 2, 1, inner_wt[ti], inner_types[ti], 0, false, false, false});
    }

    size_t n = ctx.schema.size();
    for (auto const& f : ctx.schema) {
      ctx.schema_output_types.emplace_back(f.output_type);
    }
    ctx.default_ints.resize(n, 0);
    ctx.default_floats.resize(n, 0.0);
    ctx.default_bools.resize(n, false);
    ctx.default_strings.resize(n);
    ctx.enum_valid_values.resize(n);
    ctx.enum_names.resize(n);
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

  spark_rapids_jni::protobuf_decode_context build_context() const
  {
    spark_rapids_jni::protobuf_decode_context ctx;
    ctx.fail_on_errors = true;

    // idx 0: id (int32, scalar)
    ctx.schema.push_back({1, -1, 0, 0, cudf::type_id::INT32, 0, false, false, false});
    // idx 1: tags (repeated int32, packed)
    ctx.schema.push_back({2, -1, 0, 0, cudf::type_id::INT32, 0, true, false, false});
    // idx 2: labels (repeated string)
    ctx.schema.push_back({3, -1, 0, 2, cudf::type_id::STRING, 0, true, false, false});
    // idx 3: items (repeated STRUCT)
    ctx.schema.push_back({4, -1, 0, 2, cudf::type_id::STRUCT, 0, true, false, false});
    // idx 4: Item.item_id (int32, child of idx 3)
    ctx.schema.push_back({1, 3, 1, 0, cudf::type_id::INT32, 0, false, false, false});
    // idx 5: Item.item_name (string, child of idx 3)
    ctx.schema.push_back({2, 3, 1, 2, cudf::type_id::STRING, 0, false, false, false});
    // idx 6: Item.value (int64, child of idx 3)
    ctx.schema.push_back({3, 3, 1, 0, cudf::type_id::INT64, 0, false, false, false});

    size_t n = ctx.schema.size();
    for (auto const& f : ctx.schema) {
      ctx.schema_output_types.emplace_back(f.output_type);
    }
    ctx.default_ints.resize(n, 0);
    ctx.default_floats.resize(n, 0.0);
    ctx.default_bools.resize(n, false);
    ctx.default_strings.resize(n);
    ctx.enum_valid_values.resize(n);
    ctx.enum_names.resize(n);
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
// It is designed to exercise `scan_repeated_message_children_kernel` with a
// wide repeated STRUCT payload similar in shape to real-world schema-projection
// workloads.
struct WideRepeatedMessageCase {
  int num_child_fields;
  int avg_items_per_row;

  spark_rapids_jni::protobuf_decode_context build_context() const
  {
    spark_rapids_jni::protobuf_decode_context ctx;
    ctx.fail_on_errors = true;

    // idx 0: id (scalar)
    ctx.schema.push_back({1, -1, 0, 0, cudf::type_id::INT32, 0, false, false, false});
    // idx 1: items (repeated STRUCT)
    ctx.schema.push_back({2, -1, 0, 2, cudf::type_id::STRUCT, 0, true, false, false});

    cudf::type_id child_types[] = {cudf::type_id::INT32,
                                   cudf::type_id::INT64,
                                   cudf::type_id::FLOAT32,
                                   cudf::type_id::FLOAT64,
                                   cudf::type_id::BOOL8,
                                   cudf::type_id::STRING};
    int child_wt[]  = {spark_rapids_jni::wire_type_value(spark_rapids_jni::proto_wire_type::VARINT),
                       spark_rapids_jni::wire_type_value(spark_rapids_jni::proto_wire_type::VARINT),
                       spark_rapids_jni::wire_type_value(spark_rapids_jni::proto_wire_type::I32BIT),
                       spark_rapids_jni::wire_type_value(spark_rapids_jni::proto_wire_type::I64BIT),
                       spark_rapids_jni::wire_type_value(spark_rapids_jni::proto_wire_type::VARINT),
                       spark_rapids_jni::wire_type_value(spark_rapids_jni::proto_wire_type::LEN)};
    int child_enc[] = {spark_rapids_jni::encoding_value(spark_rapids_jni::proto_encoding::DEFAULT),
                       spark_rapids_jni::encoding_value(spark_rapids_jni::proto_encoding::DEFAULT),
                       spark_rapids_jni::encoding_value(spark_rapids_jni::proto_encoding::FIXED),
                       spark_rapids_jni::encoding_value(spark_rapids_jni::proto_encoding::FIXED),
                       spark_rapids_jni::encoding_value(spark_rapids_jni::proto_encoding::DEFAULT),
                       spark_rapids_jni::encoding_value(spark_rapids_jni::proto_encoding::DEFAULT)};

    // Keep strings sparse so the case remains dominated by wide child scanning
    // rather than varlen copy traffic.
    for (int i = 0; i < num_child_fields; i++) {
      int ti = (i % 10 == 9) ? 5 : (i % 5);
      ctx.schema.push_back(
        {i + 1, 1, 1, child_wt[ti], child_types[ti], child_enc[ti], false, false, false});
    }

    size_t n = ctx.schema.size();
    for (auto const& f : ctx.schema) {
      ctx.schema_output_types.emplace_back(f.output_type);
    }
    ctx.default_ints.resize(n, 0);
    ctx.default_floats.resize(n, 0.0);
    ctx.default_bools.resize(n, false);
    ctx.default_strings.resize(n);
    ctx.enum_valid_values.resize(n);
    ctx.enum_names.resize(n);
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

  spark_rapids_jni::protobuf_decode_context build_context() const
  {
    spark_rapids_jni::protobuf_decode_context ctx;
    ctx.fail_on_errors = true;

    // idx 0: id (scalar)
    ctx.schema.push_back({1, -1, 0, 0, cudf::type_id::INT32, 0, false, false, false});
    // idx 1: items (repeated STRUCT)
    ctx.schema.push_back({2, -1, 0, 2, cudf::type_id::STRUCT, 0, true, false, false});

    for (int i = 0; i < num_repeated_children; i++) {
      bool as_string = child_is_string(i);
      ctx.schema.push_back({i + 1,
                            1,
                            1,
                            as_string ? 2 : 0,
                            as_string ? cudf::type_id::STRING : cudf::type_id::INT32,
                            0,
                            true,
                            false,
                            false});
    }

    size_t n = ctx.schema.size();
    for (auto const& f : ctx.schema) {
      ctx.schema_output_types.emplace_back(f.output_type);
    }
    ctx.default_ints.resize(n, 0);
    ctx.default_floats.resize(n, 0.0);
    ctx.default_bools.resize(n, false);
    ctx.default_strings.resize(n);
    ctx.enum_valid_values.resize(n);
    ctx.enum_names.resize(n);
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

struct RepeatedChildStringBenchData {
  std::vector<std::vector<uint8_t>> messages;
  std::vector<pb_field_location> parent_locs;
  std::vector<std::vector<int32_t>> counts_by_child;
  std::vector<std::vector<pb_repeated_occurrence>> occurrences_by_child;
};

struct RepeatedChildStringOnlyCase {
  int num_repeated_children;
  int avg_child_elems;

  RepeatedChildStringBenchData generate_data(int num_rows, std::mt19937& rng) const
  {
    RepeatedChildStringBenchData out;
    out.messages.resize(num_rows);
    out.parent_locs.resize(num_rows);
    out.counts_by_child.resize(num_repeated_children);
    out.occurrences_by_child.resize(num_repeated_children);

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

    for (int row = 0; row < num_rows; row++) {
      auto& buf = out.messages[row];
      for (int child_idx = 0; child_idx < num_repeated_children; child_idx++) {
        int fn        = child_idx + 1;
        int num_elems = vary(avg_child_elems);
        out.counts_by_child[child_idx].push_back(num_elems);
        for (int j = 0; j < num_elems; j++) {
          encode_string_field_record(
            buf, fn, random_string(str_len_dist(rng)), out.occurrences_by_child[child_idx], row);
        }
      }
      out.parent_locs[row] = {0, static_cast<int32_t>(buf.size())};
    }
    return out;
  }
};

// Case 6: Many repeated fields — stress-tests per-repeated-field sync overhead.
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

  spark_rapids_jni::protobuf_decode_context build_context() const
  {
    spark_rapids_jni::protobuf_decode_context ctx;
    ctx.fail_on_errors = true;

    int fn = 1;
    // idx 0: id (scalar)
    ctx.schema.push_back({fn++, -1, 0, 0, cudf::type_id::INT32, 0, false, false, false});

    for (int i = 0; i < num_repeated_int; i++) {
      ctx.schema.push_back({fn++, -1, 0, 0, cudf::type_id::INT32, 0, true, false, false});
    }
    for (int i = 0; i < num_repeated_str; i++) {
      ctx.schema.push_back({fn++, -1, 0, 2, cudf::type_id::STRING, 0, true, false, false});
    }

    size_t n = ctx.schema.size();
    for (auto const& f : ctx.schema) {
      ctx.schema_output_types.emplace_back(f.output_type);
    }
    ctx.default_ints.resize(n, 0);
    ctx.default_floats.resize(n, 0.0);
    ctx.default_bools.resize(n, false);
    ctx.default_strings.resize(n);
    ctx.enum_valid_values.resize(n);
    ctx.enum_names.resize(n);
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
  auto const num_rows   = static_cast<int>(state.get_int64("num_rows"));
  auto const num_fields = static_cast<int>(state.get_int64("num_fields"));
  int const num_str     = std::max(1, num_fields / 10);
  int const num_int     = num_fields - num_str;

  FlatScalarCase flat_case{num_int, num_str};
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
    auto result = spark_rapids_jni::decode_protobuf_to_struct(binary_col->view(), ctx, stream);
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
    auto result = spark_rapids_jni::decode_protobuf_to_struct(binary_col->view(), ctx, stream);
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
    auto result = spark_rapids_jni::decode_protobuf_to_struct(binary_col->view(), ctx, stream);
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
    auto result = spark_rapids_jni::decode_protobuf_to_struct(binary_col->view(), ctx, stream);
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
    auto result = spark_rapids_jni::decode_protobuf_to_struct(binary_col->view(), ctx, stream);
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
// Benchmark 6: Repeated child string count+scan only
// ===========================================================================
static void BM_protobuf_repeated_child_string_count_scan(nvbench::state& state)
{
  auto const num_rows              = static_cast<int>(state.get_int64("num_rows"));
  auto const num_repeated_children = static_cast<int>(state.get_int64("num_repeated_children"));
  auto const avg_child_elems       = static_cast<int>(state.get_int64("avg_child_elems"));

  RepeatedChildStringOnlyCase list_case{num_repeated_children, avg_child_elems};
  std::mt19937 rng(42);
  auto data       = list_case.generate_data(num_rows, rng);
  auto binary_col = make_binary_column(data.messages);

  cudf::lists_column_view in_list(binary_col->view());
  auto const* row_offsets      = in_list.offsets().data<cudf::size_type>();
  auto const* message_data     = reinterpret_cast<uint8_t const*>(in_list.child().data<int8_t>());
  auto const message_data_size = static_cast<cudf::size_type>(in_list.child().size());

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  rmm::device_uvector<pb_field_location> d_parent_locs(num_rows, stream, mr);
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(
    d_parent_locs.data(), data.parent_locs.data(), num_rows * sizeof(pb_field_location), stream));

  std::vector<spark_rapids_jni::protobuf_detail::device_nested_field_descriptor> h_schema(
    num_repeated_children);
  for (int i = 0; i < num_repeated_children; i++) {
    h_schema[i].field_number = i + 1;
    h_schema[i].parent_idx   = -1;
    h_schema[i].depth        = 0;
    h_schema[i].wire_type =
      spark_rapids_jni::wire_type_value(spark_rapids_jni::proto_wire_type::LEN);
    h_schema[i].output_type_id = static_cast<int>(cudf::type_id::STRING);
    h_schema[i].encoding =
      spark_rapids_jni::encoding_value(spark_rapids_jni::proto_encoding::DEFAULT);
    h_schema[i].is_repeated       = true;
    h_schema[i].is_required       = false;
    h_schema[i].has_default_value = false;
  }
  rmm::device_uvector<spark_rapids_jni::protobuf_detail::device_nested_field_descriptor> d_schema(
    num_repeated_children, stream, mr);
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(
    d_schema.data(), h_schema.data(), num_repeated_children * sizeof(h_schema[0]), stream));

  std::vector<int> h_rep_indices(num_repeated_children);
  for (int i = 0; i < num_repeated_children; i++) {
    CUDF_EXPECTS(h_schema[i].is_repeated,
                 "count_repeated_in_nested_kernel benchmark expects repeated child fields");
    CUDF_EXPECTS(h_schema[i].depth == 0,
                 "count_repeated_in_nested_kernel benchmark expects pre-filtered child depth 0");
    h_rep_indices[i] = i;
  }
  rmm::device_uvector<int> d_rep_indices(num_repeated_children, stream, mr);
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(
    d_rep_indices.data(), h_rep_indices.data(), num_repeated_children * sizeof(int), stream));

  size_t total_bytes = 0;
  for (auto const& m : data.messages)
    total_bytes += m.size();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    rmm::device_uvector<int> d_error(1, stream, mr);
    CUDF_CUDA_TRY(cudaMemsetAsync(d_error.data(), 0, sizeof(int), stream.value()));

    rmm::device_uvector<spark_rapids_jni::protobuf_detail::repeated_field_info> d_rep_info(
      static_cast<size_t>(num_rows) * num_repeated_children, stream, mr);
    spark_rapids_jni::protobuf_detail::
      count_repeated_in_nested_kernel<<<(num_rows + 255) / 256, 256, 0, stream.value()>>>(
        message_data,
        message_data_size,
        row_offsets,
        0,
        d_parent_locs.data(),
        num_rows,
        d_schema.data(),
        num_repeated_children,
        d_rep_info.data(),
        num_repeated_children,
        d_rep_indices.data(),
        d_error.data());

    struct rep_work {
      rmm::device_uvector<int32_t> counts;
      rmm::device_uvector<int32_t> offsets;
      int32_t total_count{0};
      std::unique_ptr<rmm::device_uvector<pb_repeated_occurrence>> occs;
      rep_work(int n, rmm::cuda_stream_view s, rmm::device_async_resource_ref m)
        : counts(n, s, m), offsets(n + 1, s, m)
      {
      }
    };

    std::vector<std::unique_ptr<rep_work>> work;
    work.reserve(num_repeated_children);
    for (int ri = 0; ri < num_repeated_children; ri++) {
      auto& w = *work.emplace_back(std::make_unique<rep_work>(num_rows, stream, mr));
      thrust::transform(rmm::exec_policy_nosync(stream, mr),
                        thrust::make_counting_iterator(0),
                        thrust::make_counting_iterator(num_rows),
                        w.counts.data(),
                        spark_rapids_jni::protobuf_detail::extract_strided_count{
                          d_rep_info.data(), ri, num_repeated_children});
      CUDF_CUDA_TRY(cudaMemsetAsync(w.offsets.data(), 0, sizeof(int32_t), stream.value()));
      thrust::inclusive_scan(rmm::exec_policy_nosync(stream, mr),
                             w.counts.begin(),
                             w.counts.end(),
                             w.offsets.data() + 1);
      CUDF_CUDA_TRY(cudf::detail::memcpy_async(
        &w.total_count, w.offsets.data() + num_rows, sizeof(int32_t), stream));
    }
    stream.synchronize();

    for (int ri = 0; ri < num_repeated_children; ri++) {
      auto& w = *work[ri];
      if (w.total_count > 0) {
        w.occs =
          std::make_unique<rmm::device_uvector<pb_repeated_occurrence>>(w.total_count, stream, mr);
        spark_rapids_jni::protobuf_detail::
          scan_repeated_in_nested_kernel<<<(num_rows + 255) / 256, 256, 0, stream.value()>>>(
            message_data,
            message_data_size,
            row_offsets,
            0,
            d_parent_locs.data(),
            num_rows,
            d_schema.data(),
            w.offsets.data(),
            d_rep_indices.data() + ri,
            w.occs->data(),
            d_error.data());
      }
    }
  });

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(total_bytes);
}

NVBENCH_BENCH(BM_protobuf_repeated_child_string_count_scan)
  .set_name("Protobuf Repeated Child String CountScan")
  .add_int64_axis("num_rows", {10'000, 20'000})
  .add_int64_axis("num_repeated_children", {1, 4, 8})
  .add_int64_axis("avg_child_elems", {1, 5});

// ===========================================================================
// Benchmark 7: Repeated child string build-only
// ===========================================================================
static void BM_protobuf_repeated_child_string_build(nvbench::state& state)
{
  auto const num_rows              = static_cast<int>(state.get_int64("num_rows"));
  auto const num_repeated_children = static_cast<int>(state.get_int64("num_repeated_children"));
  auto const avg_child_elems       = static_cast<int>(state.get_int64("avg_child_elems"));

  RepeatedChildStringOnlyCase list_case{num_repeated_children, avg_child_elems};
  std::mt19937 rng(42);
  auto data       = list_case.generate_data(num_rows, rng);
  auto binary_col = make_binary_column(data.messages);

  cudf::lists_column_view in_list(binary_col->view());
  auto const* row_offsets  = in_list.offsets().data<cudf::size_type>();
  auto const* message_data = reinterpret_cast<uint8_t const*>(in_list.child().data<int8_t>());

  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  rmm::device_uvector<pb_field_location> d_parent_locs(num_rows, stream, mr);
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(
    d_parent_locs.data(), data.parent_locs.data(), num_rows * sizeof(pb_field_location), stream));

  struct precomputed_child {
    rmm::device_uvector<int32_t> counts;
    rmm::device_uvector<pb_repeated_occurrence> occs;
    int total_count;
    precomputed_child(int nrows,
                      int total,
                      rmm::cuda_stream_view s,
                      rmm::device_async_resource_ref m)
      : counts(nrows, s, m), occs(total, s, m), total_count(total)
    {
    }
  };

  std::vector<std::unique_ptr<precomputed_child>> children;
  children.reserve(num_repeated_children);
  for (int i = 0; i < num_repeated_children; i++) {
    int total = static_cast<int>(data.occurrences_by_child[i].size());
    auto& c =
      *children.emplace_back(std::make_unique<precomputed_child>(num_rows, total, stream, mr));
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(
      c.counts.data(), data.counts_by_child[i].data(), num_rows * sizeof(int32_t), stream));
    if (total > 0) {
      CUDF_CUDA_TRY(cudf::detail::memcpy_async(c.occs.data(),
                                               data.occurrences_by_child[i].data(),
                                               total * sizeof(pb_repeated_occurrence),
                                               stream));
    }
  }

  size_t total_bytes = 0;
  for (auto const& m : data.messages)
    total_bytes += m.size();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    rmm::device_uvector<int> d_error(1, stream, mr);
    CUDF_CUDA_TRY(cudaMemsetAsync(d_error.data(), 0, sizeof(int), stream.value()));

    for (int i = 0; i < num_repeated_children; i++) {
      auto& c = *children[i];
      rmm::device_uvector<int32_t> list_offs(num_rows + 1, stream, mr);
      thrust::exclusive_scan(rmm::exec_policy_nosync(stream, mr),
                             c.counts.begin(),
                             c.counts.end(),
                             list_offs.begin(),
                             0);
      CUDF_CUDA_TRY(cudf::detail::memcpy_async(
        list_offs.data() + num_rows, &c.total_count, sizeof(int32_t), stream));

      spark_rapids_jni::protobuf_detail::nested_repeated_location_provider nr_loc{
        row_offsets, 0, d_parent_locs.data(), c.occs.data()};
      auto valid_fn = [] __device__(cudf::size_type) { return true; };
      std::vector<uint8_t> empty_default;
      auto child_values =
        spark_rapids_jni::protobuf_detail::extract_and_build_string_or_bytes_column(false,
                                                                                    message_data,
                                                                                    c.total_count,
                                                                                    nr_loc,
                                                                                    nr_loc,
                                                                                    valid_fn,
                                                                                    false,
                                                                                    empty_default,
                                                                                    d_error,
                                                                                    stream,
                                                                                    mr);
      auto list_offs_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                          num_rows + 1,
                                                          list_offs.release(),
                                                          rmm::device_buffer{},
                                                          0);
      auto result        = cudf::make_lists_column(
        num_rows, std::move(list_offs_col), std::move(child_values), 0, rmm::device_buffer{});
    }
  });

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(total_bytes);
}

NVBENCH_BENCH(BM_protobuf_repeated_child_string_build)
  .set_name("Protobuf Repeated Child String Build")
  .add_int64_axis("num_rows", {10'000, 20'000})
  .add_int64_axis("num_repeated_children", {1, 4, 8})
  .add_int64_axis("avg_child_elems", {1, 5});

// ===========================================================================
// Benchmark 8: Many repeated fields — measures per-field sync overhead at scale
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
    auto result = spark_rapids_jni::decode_protobuf_to_struct(binary_col->view(), ctx, stream);
  });

  state.add_element_count(num_rows, "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(total_bytes);
}

NVBENCH_BENCH(BM_protobuf_many_repeated)
  .set_name("Protobuf Many Repeated Fields")
  .add_int64_axis("num_rows", {10'000, 100'000})
  .add_int64_axis("num_rep_fields", {10, 30, 50});
