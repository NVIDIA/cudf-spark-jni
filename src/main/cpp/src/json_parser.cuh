/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

#include "ftos_converter.cuh"

#include <cudf/strings/detail/convert/string_to_float.cuh>
#include <cudf/strings/string_view.hpp>
#include <cudf/types.hpp>

#include <cuda/std/tuple>
#include <cuda/std/utility>

#include <cstdint>

namespace spark_rapids_jni {

/**
 * write style when writing out JSON string
 */
enum class escape_style {
  // e.g.: '\\r' is a string with 2 chars '\' 'r', writes 1 char '\r'
  UNESCAPED,

  // e.g.: '"' is a string with 1 char '"', writes out 4 chars '"' '\' '\"'
  // '"'
  ESCAPED
};

/**
 * @brief Maximum JSON nesting depth
 * JSON with a greater depth is invalid
 * If set this to be a greater value, should update `context_stack`
 */
constexpr int MAX_JSON_NESTING_DEPTH = 64;

//
/**
 * Define the maximum JSON number length. Negative or zero means no
 * limitation.
 *
 * By default, maximum JSON number length is negative one, means no
 * limitation.
 *
 * e.g.: The length of number -123.45e-67 is 7. if maximum JSON number length
 * is 6, then this number is a invalid number.
 */
constexpr int max_num_len = 1000;

/**
 * JSON token enum
 */
enum class json_token : int8_t {
  // start token
  INIT = 0,

  // successfully parsed the whole JSON string
  SUCCESS,

  // get error when parsing JSON string
  ERROR,

  // '{'
  START_OBJECT,

  // '}'
  END_OBJECT,

  // '['
  START_ARRAY,

  // ']'
  END_ARRAY,

  // e.g.: key1 in {"key1" : "value1"}
  FIELD_NAME,

  // e.g.: value1 in {"key1" : "value1"}
  VALUE_STRING,

  // e.g.: 123 in {"key1" : 123}
  VALUE_NUMBER_INT,

  // e.g.: 1.25 in {"key1" : 1.25}
  VALUE_NUMBER_FLOAT,

  // e.g.: true in {"key1" : true}
  VALUE_TRUE,

  // e.g.: false in {"key1" : false}
  VALUE_FALSE,

  // e.g.: null in {"key1" : null}
  VALUE_NULL

};

/**
 * This is similar to cudf::string_view, but cudf::string_view enforces
 * UTF-8 encoding, which adds overhead that is not needed for this process.
 */
class char_range {
 public:
  __device__ inline char_range(char const* const start, cudf::size_type const len)
    : _data(start), _len(len)
  {
  }

  __device__ inline char_range(cudf::string_view const& input)
    : _data(input.data()), _len(input.size_bytes())
  {
  }

  // Warning it looks like there is some kind of a bug in CUDA where you don't want to initialize
  // a member variable with a static method like this.
  __device__ inline static char_range null() { return char_range(nullptr, 0); }

  __device__ inline char_range(char_range const&)            = default;
  __device__ inline char_range(char_range&&)                 = default;
  __device__ inline char_range& operator=(char_range const&) = default;
  __device__ inline char_range& operator=(char_range&&)      = default;
  __device__ inline ~char_range()                            = default;

  __device__ inline cudf::size_type size() const { return _len; }
  __device__ inline char const* data() const { return _data; }
  __device__ inline bool is_null() const { return _data == nullptr; }
  __device__ inline bool is_empty() const { return _len <= 0; }
  __device__ inline char operator[](cudf::size_type pos) const { return _data[pos]; }

  __device__ inline cudf::string_view slice_sv(cudf::size_type pos, cudf::size_type len) const
  {
    return cudf::string_view(_data + pos, len);
  }

  __device__ inline char_range slice(cudf::size_type pos, cudf::size_type len) const
  {
    return char_range(_data + pos, len);
  }

 protected:
  char const* _data;
  cudf::size_type _len;
};

/**
 * A char range that moves the begin pointer of the current range forward while reading.
 *
 * This support continuous reading of characters without the need of an additional variable
 * to keep track of the current reading position.
 */
class char_range_reader : public char_range {
 public:
  __device__ inline explicit char_range_reader(char_range range)
    : char_range(cuda::std::move(range))
  {
  }
  __device__ inline void next()
  {
    _data++;
    _len--;
  }

  // Warning: this does not check for out-of-bound access.
  // The caller must be responsible to check for empty range before calling this.
  __device__ inline char current_char() const { return _data[0]; }
};

/**
 * JSON parser, provides token by token parsing.
 * Follow Jackson JSON format by default.
 *
 *
 * For JSON format:
 * Refer to https://www.json.org/json-en.html.
 *
 * Note: This is not conventional as it allows
 * single quotes and unescaped control characters
 * to match what SPARK does for get_json_object
 *
 * White space can only be 4 chars: ' ', '\n', '\r', '\t',
 * Jackson does not allow other control chars as white spaces.
 *
 * Valid number examples:
 *   0, 102, -0, -102, 0.3, -0.3
 *   1e-5, 1E+5, 1e0, 1E0, 1.3e5
 *   1e01 : allow leading zeor after 'e'
 *
 * Invalid number examples:
 *   00, -00   Leading zeroes not allowed
 *   infinity, +infinity, -infinity
 *   1e, 1e+, 1e-, -1., 1.
 *
 * Valid string examples:
 *     "\'" , "\"" ,  '\'' , '\"' , '"' , "'"
 *
 * Valid string: "ascii_control_chars"
 *    here `ascii_control_chars` represents control chars which in Ascii code
 * range: [0, 32)
 *
 */
class json_parser {
 public:
  __device__ inline explicit json_parser(char_range _chars)
    : chars(_chars), curr_pos(0), current_token(json_token::INIT), max_depth_exceeded(false)
  {
  }

 private:
  /**
   * @brief get the bit value for specified bit from a int64 number
   */
  static __device__ inline bool get_bit_value(int64_t number, int bitIndex)
  {
    // Shift the number right by the bitIndex to bring the desired bit to the rightmost position
    long shifted = number >> bitIndex;

    // Extract the rightmost bit by performing a bitwise AND with 1
    bool bit_value = shifted & 1;

    return bit_value;
  }

  /**
   * @brief set the bit value for specified bit to a int64 number
   */
  static __device__ inline void set_bit_value(int64_t& number, int bit_index, bool bit_value)
  {
    // Create a mask with a 1 at the desired bit index
    long mask = 1L << bit_index;

    if (bit_value) {
      // Set the bit to 1 by performing a bitwise OR with the mask
      number |= mask;
    } else {
      // Set the bit to 0 by performing a bitwise AND with the complement of the mask
      number &= ~mask;
    }
  }

  /**
   * is current position EOF
   */
  __device__ inline bool eof(cudf::size_type pos) const { return pos >= chars.size(); }
  __device__ inline bool eof() const { return curr_pos >= chars.size(); }

  /**
   * is hex digits: 0-9, A-F, a-f
   */
  static __device__ inline bool is_hex_digit(char c)
  {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
  }

  /**
   * is 0 to 9 digit
   */
  static __device__ inline bool is_digit(char c) { return (c >= '0' && c <= '9'); }

  /**
   * is white spaces: ' ', '\t', '\n' '\r'
   */
  static __device__ inline bool is_whitespace(char c)
  {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
  }

  /**
   * @brief Whether the UTF-16 code unit `cu` continues a Java identifier
   *
   * Matches `Character.isJavaIdentifierPart(char)`. The ranges are generated from the JDK rather
   * than derived from Unicode categories, because the predicate also accepts
   * identifier-ignorable controls that no single category covers.
   */
  static __device__ inline bool is_java_identifier_part(unsigned int cu)
  {
    // Inclusive `lo, hi` pairs, sorted, covering everything above ASCII.
    static constexpr uint16_t identifier_part_ranges[] = {
      0x0080, 0x009F, 0x00A2, 0x00A5, 0x00AA, 0x00AA, 0x00AD, 0x00AD, 0x00B5, 0x00B5, 0x00BA,
      0x00BA, 0x00C0, 0x00D6, 0x00D8, 0x00F6, 0x00F8, 0x02C1, 0x02C6, 0x02D1, 0x02E0, 0x02E4,
      0x02EC, 0x02EC, 0x02EE, 0x02EE, 0x0300, 0x0374, 0x0376, 0x0377, 0x037A, 0x037D, 0x037F,
      0x037F, 0x0386, 0x0386, 0x0388, 0x038A, 0x038C, 0x038C, 0x038E, 0x03A1, 0x03A3, 0x03F5,
      0x03F7, 0x0481, 0x0483, 0x0487, 0x048A, 0x052F, 0x0531, 0x0556, 0x0559, 0x0559, 0x0560,
      0x0588, 0x058F, 0x058F, 0x0591, 0x05BD, 0x05BF, 0x05BF, 0x05C1, 0x05C2, 0x05C4, 0x05C5,
      0x05C7, 0x05C7, 0x05D0, 0x05EA, 0x05EF, 0x05F2, 0x0600, 0x0605, 0x060B, 0x060B, 0x0610,
      0x061A, 0x061C, 0x061C, 0x0620, 0x0669, 0x066E, 0x06D3, 0x06D5, 0x06DD, 0x06DF, 0x06E8,
      0x06EA, 0x06FC, 0x06FF, 0x06FF, 0x070F, 0x074A, 0x074D, 0x07B1, 0x07C0, 0x07F5, 0x07FA,
      0x07FA, 0x07FD, 0x082D, 0x0840, 0x085B, 0x0860, 0x086A, 0x0870, 0x0887, 0x0889, 0x088E,
      0x0890, 0x0891, 0x0897, 0x0963, 0x0966, 0x096F, 0x0971, 0x0983, 0x0985, 0x098C, 0x098F,
      0x0990, 0x0993, 0x09A8, 0x09AA, 0x09B0, 0x09B2, 0x09B2, 0x09B6, 0x09B9, 0x09BC, 0x09C4,
      0x09C7, 0x09C8, 0x09CB, 0x09CE, 0x09D7, 0x09D7, 0x09DC, 0x09DD, 0x09DF, 0x09E3, 0x09E6,
      0x09F3, 0x09FB, 0x09FC, 0x09FE, 0x09FE, 0x0A01, 0x0A03, 0x0A05, 0x0A0A, 0x0A0F, 0x0A10,
      0x0A13, 0x0A28, 0x0A2A, 0x0A30, 0x0A32, 0x0A33, 0x0A35, 0x0A36, 0x0A38, 0x0A39, 0x0A3C,
      0x0A3C, 0x0A3E, 0x0A42, 0x0A47, 0x0A48, 0x0A4B, 0x0A4D, 0x0A51, 0x0A51, 0x0A59, 0x0A5C,
      0x0A5E, 0x0A5E, 0x0A66, 0x0A75, 0x0A81, 0x0A83, 0x0A85, 0x0A8D, 0x0A8F, 0x0A91, 0x0A93,
      0x0AA8, 0x0AAA, 0x0AB0, 0x0AB2, 0x0AB3, 0x0AB5, 0x0AB9, 0x0ABC, 0x0AC5, 0x0AC7, 0x0AC9,
      0x0ACB, 0x0ACD, 0x0AD0, 0x0AD0, 0x0AE0, 0x0AE3, 0x0AE6, 0x0AEF, 0x0AF1, 0x0AF1, 0x0AF9,
      0x0AFF, 0x0B01, 0x0B03, 0x0B05, 0x0B0C, 0x0B0F, 0x0B10, 0x0B13, 0x0B28, 0x0B2A, 0x0B30,
      0x0B32, 0x0B33, 0x0B35, 0x0B39, 0x0B3C, 0x0B44, 0x0B47, 0x0B48, 0x0B4B, 0x0B4D, 0x0B55,
      0x0B57, 0x0B5C, 0x0B5D, 0x0B5F, 0x0B63, 0x0B66, 0x0B6F, 0x0B71, 0x0B71, 0x0B82, 0x0B83,
      0x0B85, 0x0B8A, 0x0B8E, 0x0B90, 0x0B92, 0x0B95, 0x0B99, 0x0B9A, 0x0B9C, 0x0B9C, 0x0B9E,
      0x0B9F, 0x0BA3, 0x0BA4, 0x0BA8, 0x0BAA, 0x0BAE, 0x0BB9, 0x0BBE, 0x0BC2, 0x0BC6, 0x0BC8,
      0x0BCA, 0x0BCD, 0x0BD0, 0x0BD0, 0x0BD7, 0x0BD7, 0x0BE6, 0x0BEF, 0x0BF9, 0x0BF9, 0x0C00,
      0x0C0C, 0x0C0E, 0x0C10, 0x0C12, 0x0C28, 0x0C2A, 0x0C39, 0x0C3C, 0x0C44, 0x0C46, 0x0C48,
      0x0C4A, 0x0C4D, 0x0C55, 0x0C56, 0x0C58, 0x0C5A, 0x0C5D, 0x0C5D, 0x0C60, 0x0C63, 0x0C66,
      0x0C6F, 0x0C80, 0x0C83, 0x0C85, 0x0C8C, 0x0C8E, 0x0C90, 0x0C92, 0x0CA8, 0x0CAA, 0x0CB3,
      0x0CB5, 0x0CB9, 0x0CBC, 0x0CC4, 0x0CC6, 0x0CC8, 0x0CCA, 0x0CCD, 0x0CD5, 0x0CD6, 0x0CDD,
      0x0CDE, 0x0CE0, 0x0CE3, 0x0CE6, 0x0CEF, 0x0CF1, 0x0CF3, 0x0D00, 0x0D0C, 0x0D0E, 0x0D10,
      0x0D12, 0x0D44, 0x0D46, 0x0D48, 0x0D4A, 0x0D4E, 0x0D54, 0x0D57, 0x0D5F, 0x0D63, 0x0D66,
      0x0D6F, 0x0D7A, 0x0D7F, 0x0D81, 0x0D83, 0x0D85, 0x0D96, 0x0D9A, 0x0DB1, 0x0DB3, 0x0DBB,
      0x0DBD, 0x0DBD, 0x0DC0, 0x0DC6, 0x0DCA, 0x0DCA, 0x0DCF, 0x0DD4, 0x0DD6, 0x0DD6, 0x0DD8,
      0x0DDF, 0x0DE6, 0x0DEF, 0x0DF2, 0x0DF3, 0x0E01, 0x0E3A, 0x0E3F, 0x0E4E, 0x0E50, 0x0E59,
      0x0E81, 0x0E82, 0x0E84, 0x0E84, 0x0E86, 0x0E8A, 0x0E8C, 0x0EA3, 0x0EA5, 0x0EA5, 0x0EA7,
      0x0EBD, 0x0EC0, 0x0EC4, 0x0EC6, 0x0EC6, 0x0EC8, 0x0ECE, 0x0ED0, 0x0ED9, 0x0EDC, 0x0EDF,
      0x0F00, 0x0F00, 0x0F18, 0x0F19, 0x0F20, 0x0F29, 0x0F35, 0x0F35, 0x0F37, 0x0F37, 0x0F39,
      0x0F39, 0x0F3E, 0x0F47, 0x0F49, 0x0F6C, 0x0F71, 0x0F84, 0x0F86, 0x0F97, 0x0F99, 0x0FBC,
      0x0FC6, 0x0FC6, 0x1000, 0x1049, 0x1050, 0x109D, 0x10A0, 0x10C5, 0x10C7, 0x10C7, 0x10CD,
      0x10CD, 0x10D0, 0x10FA, 0x10FC, 0x1248, 0x124A, 0x124D, 0x1250, 0x1256, 0x1258, 0x1258,
      0x125A, 0x125D, 0x1260, 0x1288, 0x128A, 0x128D, 0x1290, 0x12B0, 0x12B2, 0x12B5, 0x12B8,
      0x12BE, 0x12C0, 0x12C0, 0x12C2, 0x12C5, 0x12C8, 0x12D6, 0x12D8, 0x1310, 0x1312, 0x1315,
      0x1318, 0x135A, 0x135D, 0x135F, 0x1380, 0x138F, 0x13A0, 0x13F5, 0x13F8, 0x13FD, 0x1401,
      0x166C, 0x166F, 0x167F, 0x1681, 0x169A, 0x16A0, 0x16EA, 0x16EE, 0x16F8, 0x1700, 0x1715,
      0x171F, 0x1734, 0x1740, 0x1753, 0x1760, 0x176C, 0x176E, 0x1770, 0x1772, 0x1773, 0x1780,
      0x17D3, 0x17D7, 0x17D7, 0x17DB, 0x17DD, 0x17E0, 0x17E9, 0x180B, 0x1819, 0x1820, 0x1878,
      0x1880, 0x18AA, 0x18B0, 0x18F5, 0x1900, 0x191E, 0x1920, 0x192B, 0x1930, 0x193B, 0x1946,
      0x196D, 0x1970, 0x1974, 0x1980, 0x19AB, 0x19B0, 0x19C9, 0x19D0, 0x19D9, 0x1A00, 0x1A1B,
      0x1A20, 0x1A5E, 0x1A60, 0x1A7C, 0x1A7F, 0x1A89, 0x1A90, 0x1A99, 0x1AA7, 0x1AA7, 0x1AB0,
      0x1ABD, 0x1ABF, 0x1ACE, 0x1B00, 0x1B4C, 0x1B50, 0x1B59, 0x1B6B, 0x1B73, 0x1B80, 0x1BF3,
      0x1C00, 0x1C37, 0x1C40, 0x1C49, 0x1C4D, 0x1C7D, 0x1C80, 0x1C8A, 0x1C90, 0x1CBA, 0x1CBD,
      0x1CBF, 0x1CD0, 0x1CD2, 0x1CD4, 0x1CFA, 0x1D00, 0x1F15, 0x1F18, 0x1F1D, 0x1F20, 0x1F45,
      0x1F48, 0x1F4D, 0x1F50, 0x1F57, 0x1F59, 0x1F59, 0x1F5B, 0x1F5B, 0x1F5D, 0x1F5D, 0x1F5F,
      0x1F7D, 0x1F80, 0x1FB4, 0x1FB6, 0x1FBC, 0x1FBE, 0x1FBE, 0x1FC2, 0x1FC4, 0x1FC6, 0x1FCC,
      0x1FD0, 0x1FD3, 0x1FD6, 0x1FDB, 0x1FE0, 0x1FEC, 0x1FF2, 0x1FF4, 0x1FF6, 0x1FFC, 0x200B,
      0x200F, 0x202A, 0x202E, 0x203F, 0x2040, 0x2054, 0x2054, 0x2060, 0x2064, 0x2066, 0x206F,
      0x2071, 0x2071, 0x207F, 0x207F, 0x2090, 0x209C, 0x20A0, 0x20C0, 0x20D0, 0x20DC, 0x20E1,
      0x20E1, 0x20E5, 0x20F0, 0x2102, 0x2102, 0x2107, 0x2107, 0x210A, 0x2113, 0x2115, 0x2115,
      0x2119, 0x211D, 0x2124, 0x2124, 0x2126, 0x2126, 0x2128, 0x2128, 0x212A, 0x212D, 0x212F,
      0x2139, 0x213C, 0x213F, 0x2145, 0x2149, 0x214E, 0x214E, 0x2160, 0x2188, 0x2C00, 0x2CE4,
      0x2CEB, 0x2CF3, 0x2D00, 0x2D25, 0x2D27, 0x2D27, 0x2D2D, 0x2D2D, 0x2D30, 0x2D67, 0x2D6F,
      0x2D6F, 0x2D7F, 0x2D96, 0x2DA0, 0x2DA6, 0x2DA8, 0x2DAE, 0x2DB0, 0x2DB6, 0x2DB8, 0x2DBE,
      0x2DC0, 0x2DC6, 0x2DC8, 0x2DCE, 0x2DD0, 0x2DD6, 0x2DD8, 0x2DDE, 0x2DE0, 0x2DFF, 0x2E2F,
      0x2E2F, 0x3005, 0x3007, 0x3021, 0x302F, 0x3031, 0x3035, 0x3038, 0x303C, 0x3041, 0x3096,
      0x3099, 0x309A, 0x309D, 0x309F, 0x30A1, 0x30FA, 0x30FC, 0x30FF, 0x3105, 0x312F, 0x3131,
      0x318E, 0x31A0, 0x31BF, 0x31F0, 0x31FF, 0x3400, 0x4DBF, 0x4E00, 0xA48C, 0xA4D0, 0xA4FD,
      0xA500, 0xA60C, 0xA610, 0xA62B, 0xA640, 0xA66F, 0xA674, 0xA67D, 0xA67F, 0xA6F1, 0xA717,
      0xA71F, 0xA722, 0xA788, 0xA78B, 0xA7CD, 0xA7D0, 0xA7D1, 0xA7D3, 0xA7D3, 0xA7D5, 0xA7DC,
      0xA7F2, 0xA827, 0xA82C, 0xA82C, 0xA838, 0xA838, 0xA840, 0xA873, 0xA880, 0xA8C5, 0xA8D0,
      0xA8D9, 0xA8E0, 0xA8F7, 0xA8FB, 0xA8FB, 0xA8FD, 0xA92D, 0xA930, 0xA953, 0xA960, 0xA97C,
      0xA980, 0xA9C0, 0xA9CF, 0xA9D9, 0xA9E0, 0xA9FE, 0xAA00, 0xAA36, 0xAA40, 0xAA4D, 0xAA50,
      0xAA59, 0xAA60, 0xAA76, 0xAA7A, 0xAAC2, 0xAADB, 0xAADD, 0xAAE0, 0xAAEF, 0xAAF2, 0xAAF6,
      0xAB01, 0xAB06, 0xAB09, 0xAB0E, 0xAB11, 0xAB16, 0xAB20, 0xAB26, 0xAB28, 0xAB2E, 0xAB30,
      0xAB5A, 0xAB5C, 0xAB69, 0xAB70, 0xABEA, 0xABEC, 0xABED, 0xABF0, 0xABF9, 0xAC00, 0xD7A3,
      0xD7B0, 0xD7C6, 0xD7CB, 0xD7FB, 0xF900, 0xFA6D, 0xFA70, 0xFAD9, 0xFB00, 0xFB06, 0xFB13,
      0xFB17, 0xFB1D, 0xFB28, 0xFB2A, 0xFB36, 0xFB38, 0xFB3C, 0xFB3E, 0xFB3E, 0xFB40, 0xFB41,
      0xFB43, 0xFB44, 0xFB46, 0xFBB1, 0xFBD3, 0xFD3D, 0xFD50, 0xFD8F, 0xFD92, 0xFDC7, 0xFDF0,
      0xFDFC, 0xFE00, 0xFE0F, 0xFE20, 0xFE2F, 0xFE33, 0xFE34, 0xFE4D, 0xFE4F, 0xFE69, 0xFE69,
      0xFE70, 0xFE74, 0xFE76, 0xFEFC, 0xFEFF, 0xFEFF, 0xFF04, 0xFF04, 0xFF10, 0xFF19, 0xFF21,
      0xFF3A, 0xFF3F, 0xFF3F, 0xFF41, 0xFF5A, 0xFF66, 0xFFBE, 0xFFC2, 0xFFC7, 0xFFCA, 0xFFCF,
      0xFFD2, 0xFFD7, 0xFFDA, 0xFFDC, 0xFFE0, 0xFFE1, 0xFFE5, 0xFFE6, 0xFFF9, 0xFFFB};
    constexpr int range_count =
      sizeof(identifier_part_ranges) / sizeof(identifier_part_ranges[0]) / 2;

    int lo = 0;
    int hi = range_count - 1;
    while (lo <= hi) {
      auto const mid = (lo + hi) / 2;
      if (cu < identifier_part_ranges[2 * mid]) {
        hi = mid - 1;
      } else if (cu > identifier_part_ranges[2 * mid + 1]) {
        lo = mid + 1;
      } else {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief The code point at `str`, or -1 when it is not a well-formed 2- or 3-byte sequence
   *
   * A four-byte lead also reports -1: Jackson judges an astral character by its high surrogate,
   * which never continues an identifier. Answering only "can this be looked up" is what keeps this
   * independent of how many replacement characters a malformed run would produce.
   */
  static __device__ inline int decode_bmp_code_point(char_range const& str)
  {
    auto const lead = static_cast<unsigned int>(static_cast<unsigned char>(str[0]));
    // 0xC0 and 0xC1 can only encode a value that fits in one byte, so the JDK decoder rejects them.
    if (lead >= 0xC2 && lead <= 0xDF) {
      if (str.size() < 2) { return -1; }
      auto const trail_1 = static_cast<unsigned int>(static_cast<unsigned char>(str[1]));
      if ((trail_1 & 0xC0) != 0x80) { return -1; }
      return static_cast<int>(((lead & 0x1F) << 6) | (trail_1 & 0x3F));
    }
    if (lead >= 0xE0 && lead <= 0xEF) {
      if (str.size() < 3) { return -1; }
      auto const trail_1 = static_cast<unsigned int>(static_cast<unsigned char>(str[1]));
      auto const trail_2 = static_cast<unsigned int>(static_cast<unsigned char>(str[2]));
      if ((trail_1 & 0xC0) != 0x80 || (trail_2 & 0xC0) != 0x80) { return -1; }
      // A low trail byte after 0xE0 is overlong and a high one after 0xED is a surrogate.
      if ((lead == 0xE0 && trail_1 < 0xA0) || (lead == 0xED && trail_1 >= 0xA0)) { return -1; }
      return static_cast<int>(((lead & 0x0F) << 12) | ((trail_1 & 0x3F) << 6) | (trail_2 & 0x3F));
    }
    return -1;
  }

  /**
   * @brief Whether the character at `str` continues a root-level `true`/`false`/`null`
   *
   * Continuing one makes the document invalid. Jackson consults `Character.isJavaIdentifierPart`
   * only for bytes at or above '0' other than ']' and '}', so every byte below that ends the
   * keyword however Java would classify it. Jackson also classifies the single UTF-16 code unit it
   * read, so an astral character is judged by its high surrogate and a malformed run by U+FFFD,
   * neither of which is an identifier part.
   */
  static __device__ inline bool continues_root_keyword(char_range const& str)
  {
    auto const byte = static_cast<unsigned char>(str[0]);
    if (byte < '0' || byte == ']' || byte == '}') { return false; }
    if (byte < 0x80) {
      return byte == 0x7F || byte == '_' || is_digit(static_cast<char>(byte)) ||
             (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z');
    }

    auto const code_point = decode_bmp_code_point(str);
    return code_point >= 0 && is_java_identifier_part(static_cast<unsigned int>(code_point));
  }

  /**
   * skips 4 characters: ' ', '\t', '\n' '\r'
   */
  __device__ inline void skip_whitespaces()
  {
    while (!eof() && is_whitespace(chars[curr_pos])) {
      curr_pos++;
    }
  }

  /**
   * check current char, if it's expected, then plus the position
   */
  static __device__ inline bool try_skip(char_range_reader& reader, char expected)
  {
    if (!reader.is_empty() && reader.current_char() == expected) {
      reader.next();
      return true;
    }
    return false;
  }

  __device__ inline bool try_skip(cudf::size_type& pos, char expected) const
  {
    if (!eof(pos) && chars[pos] == expected) {
      pos++;
      return true;
    }
    return false;
  }

  /**
   * try to push current context into stack
   * if nested depth exceeds limitation, return false
   */
  __device__ inline bool try_push_context(json_token token)
  {
    if (stack_size < MAX_JSON_NESTING_DEPTH) {
      push_context(token);
      return true;
    } else {
      return false;
    }
  }

  /**
   * record the nested state into stack: JSON object or JSON array
   */
  __device__ inline void push_context(json_token token)
  {
    bool v = json_token::START_OBJECT == token ? true : false;
    set_bit_value(context_stack, stack_size, v);
    stack_size++;
  }

  /**
   * whether the top of nested context stack is JSON object context
   * true is object, false is array
   * only has two contexts: object or array
   */
  __device__ inline bool is_object_context() const
  {
    return get_bit_value(context_stack, stack_size - 1);
  }

  __device__ inline void pop_curr_context() { stack_size--; }

  __device__ inline bool is_context_stack_empty() const { return stack_size == 0; }

  __device__ inline void set_current_error() { current_token = json_token::ERROR; }

  /**
   * @brief Reject trailing content after a root-level scalar, as Jackson does
   *
   * Jackson ends a root number only at whitespace or end of input, and a root
   * `true`/`false`/`null` only at a byte that cannot continue a Java identifier.
   */
  __device__ inline void check_root_scalar_end()
  {
    if (eof()) { return; }
    char const c = chars[curr_pos];
    if (current_token == json_token::VALUE_NUMBER_INT ||
        current_token == json_token::VALUE_NUMBER_FLOAT) {
      if (!is_whitespace(c)) { set_current_error(); }
    } else if (current_token == json_token::VALUE_TRUE ||
               current_token == json_token::VALUE_FALSE ||
               current_token == json_token::VALUE_NULL) {
      // The whole tail is passed because classifying a non-ASCII character needs its full sequence.
      if (continues_root_keyword(chars.slice(curr_pos, chars.size() - curr_pos))) {
        set_current_error();
      }
    }
  }

  /**
   * parse the first value token from current position
   * e.g., after finished this function:
   *   current token is START_OBJECT if current value is object
   *   current token is START_ARRAY if current value is array
   *   current token is string/num/true/false/null if current value is terminal
   *   current token is ERROR if parse failed
   */
  __device__ inline void parse_first_token_in_value_and_set_current()
  {
    current_token_start_pos = curr_pos;
    // A ':' or ',' as the last byte of the row leaves no value to parse, and reading one anyway
    // would sample bytes belonging to the next row.
    if (eof()) {
      set_current_error();
      return;
    }
    char c = chars[curr_pos];
    switch (c) {
      case '{':
        if (!try_push_context(json_token::START_OBJECT)) {
          max_depth_exceeded = true;
          set_current_error();
        } else {
          curr_pos++;
          current_token = json_token::START_OBJECT;
        }
        break;
      case '[':
        if (!try_push_context(json_token::START_ARRAY)) {
          max_depth_exceeded = true;
          set_current_error();
        } else {
          curr_pos++;
          current_token = json_token::START_ARRAY;
        }
        break;
      case '"':
        // fall through
      case '\'': parse_string_and_set_current(); break;
      case 't':
        curr_pos++;
        parse_true_and_set_current();
        break;
      case 'f':
        curr_pos++;
        parse_false_and_set_current();
        break;
      case 'n':
        curr_pos++;
        parse_null_and_set_current();
        break;
      default: parse_number_and_set_current(); break;
    }
    // An empty stack means the value just parsed is the whole document; a nested one is already
    // bounded by the ',', '}' or ']' its container requires.
    if (is_context_stack_empty()) { check_root_scalar_end(); }
  }

  // =========== Parse string begin ===========

  /**
   * parse quoted string and set current token
   */
  __device__ inline void parse_string_and_set_current()
  {
    [[maybe_unused]] auto const [success, matched, end] =
      try_parse_string(char_range_reader{chars.slice(curr_pos, chars.size() - curr_pos)});
    if (success) {
      curr_pos      = static_cast<cudf::size_type>(cuda::std::distance(chars.data(), end));
      current_token = json_token::VALUE_STRING;
    } else {
      set_current_error();
    }
  }

  /**
   * transform int value from [0, 15] to hex char
   */
  static __device__ inline char to_hex_char(unsigned int v)
  {
    if (v < 10)
      return '0' + v;
    else
      return 'A' + (v - 10);
  }

  /**
   * escape control char ( ASCII code value [0, 32) )
   * e.g.: \0  (ASCII code 0) will be escaped to 6 chars: \u0000
   * e.g.: \10 (ASCII code 0) will be escaped to 2 chars: \n
   * @param char to be escaped, c should in range [0, 31)
   * @param[out] escape output
   */
  static __device__ inline int escape_char(unsigned char c, char* output)
  {
    if (nullptr == output) {
      switch (c) {
        case 8:             // \b
        case 9:             // \t
        case 10:            // \n
        case 12:            // \f
        case 13: return 2;  // \r
        default: return 6;  // \u0000
      }
    }
    switch (c) {
      case 8:
        output[0] = '\\';
        output[1] = 'b';
        return 2;
      case 9:
        output[0] = '\\';
        output[1] = 't';
        return 2;
      case 10:
        output[0] = '\\';
        output[1] = 'n';
        return 2;
      case 12:
        output[0] = '\\';
        output[1] = 'f';
        return 2;
      case 13:
        output[0] = '\\';
        output[1] = 'r';
        return 2;
      default:
        output[0] = '\\';
        output[1] = 'u';
        output[2] = '0';
        output[3] = '0';

        // write high digit
        if (c >= 16) {
          output[4] = '1';
        } else {
          output[4] = '0';
        }

        // write low digit
        unsigned int v = c % 16;
        output[5]      = to_hex_char(v);
        return 6;
    }
  }

  static __device__ inline int write_string(char_range_reader& str,
                                            char* copy_destination,
                                            escape_style w_style)
  {
    if (str.is_empty()) { return 0; }
    char const quote_char = str.current_char();
    int output_size_bytes = 0;

    // write the first " if write style is escaped
    if (escape_style::ESCAPED == w_style) {
      output_size_bytes++;
      if (nullptr != copy_destination) { *copy_destination++ = '"'; }
    }

    // skip left quote char
    // No need to check because we just read it in.
    str.next();

    // scan string content
    while (!str.is_empty()) {
      char const c = str.current_char();
      int const v  = static_cast<int>(c);
      if (c == quote_char) {
        // path 1: match closing quote char
        str.next();

        // write the end " if write style is escaped
        if (escape_style::ESCAPED == w_style) {
          output_size_bytes++;
          if (nullptr != copy_destination) { *copy_destination++ = '"'; }
        }

        return output_size_bytes;
      } else if (v >= 0 && v < 32) {
        // path 2: unescaped control char

        // copy if enabled, unescape mode, write 1 char
        if (escape_style::UNESCAPED == w_style) {
          output_size_bytes++;
          if (copy_destination != nullptr) { *copy_destination++ = str.current_char(); }
        } else {
          // escape_style::ESCAPED
          int const escape_chars = escape_char(str.current_char(), copy_destination);
          if (copy_destination != nullptr) { copy_destination += escape_chars; }
          output_size_bytes += escape_chars;
        }

        str.next();
      } else if ('\\' == c) {
        // path 3: escape path
        str.next();
        char_range_reader to_match(char_range::null());  // unused
        bool matched_field_name{false};                  // unused
        if (!try_skip_escape_part(
              str, to_match, copy_destination, w_style, output_size_bytes, matched_field_name)) {
          return output_size_bytes;
        }
      } else {
        // path 4: safe code point

        // handle single unescaped " char; happens when string is quoted by char '
        // e.g.:  'A"' string, escape to "A\\"" (5 chars: " A \ " ")
        if ('\"' == c && escape_style::ESCAPED == w_style) {
          if (copy_destination != nullptr) { *copy_destination++ = '\\'; }
          output_size_bytes++;
        }

        if (copy_destination != nullptr) { *copy_destination++ = c; }
        str.next();
        output_size_bytes++;
      }
    }

    // technically this is an error state, but we will do our best from here...
    return output_size_bytes;
  }

  /**
   * utility for parsing string, this function does not update the parser
   * internal try parse quoted string using passed `quote_char` `quote_char` can
   * be ' or " For UTF-8 encoding: Single byte char: The most significant bit of
   * the byte is always 0 Two-byte characters: The leading bits of the first
   * byte are 110, and the leading bits of the second byte are 10. Three-byte
   * characters: The leading bits of the first byte are 1110, and the leading
   * bits of the second and third bytes are 10. Four-byte characters: The
   * leading bits of the first byte are 11110, and the leading bits of the
   * second, third, and fourth bytes are 10. Because JSON structural chars([ ] {
   * } , :), string quote char(" ') and Escape char \ are all Ascii(The leading
   * bit is 0), so it's safe that do not convert byte array to UTF-8 char.
   *
   * When quote is " grammar is:
   *
   *   STRING
   *     : '"' (ESC | SAFECODEPOINT)* '"'
   *     ;
   *
   *   fragment ESC
   *     : '\\' (["\\/bfnrt] | UNICODE)
   *     ;
   *
   *   fragment UNICODE
   *     : 'u' HEX HEX HEX HEX
   *     ;
   *
   *   fragment HEX
   *     : [0-9a-fA-F]
   *     ;
   *
   *   fragment SAFECODEPOINT
   *       // 1 not " or '
   *       // 2 not \
   *       // 3 non control character: Ascii value not in [0, 32)
   *     : ~ ["\\\u0000-\u001F]
   *     ;
   *
   * @param str string to parse, positioned on its delimiting quote; that byte is adopted as the
   *        delimiter without validation, so the caller must reject a non-quote itself
   * @param to_match expected match str
   * @return a tuple of values indicating if the parse process was successful, if field name was
   *         matched, and a pointer to the past-end position of the parsed data
   */
  static __device__ inline cuda::std::tuple<bool, bool, char const*> try_parse_string(
    char_range_reader str, char_range_reader to_match = char_range_reader(char_range::null()))
  {
    if (str.is_empty()) { return cuda::std::make_tuple(false, false, nullptr); }
    char const quote_char   = str.current_char();
    bool matched_field_name = !to_match.is_null();

    // skip left quote char
    // We don't need to actually verify what it is, because we just read it.
    str.next();

    // scan string content
    while (!str.is_empty()) {
      char c = str.current_char();
      int v  = static_cast<int>(c);
      if (c == quote_char) {  // path 1: match closing quote char
        str.next();
        matched_field_name = matched_field_name && (to_match.is_null() || to_match.is_empty());
        return cuda::std::make_tuple(true, matched_field_name, str.data());
      } else if (v >= 0 && v < 32) {  // path 2: unescaped control char
        matched_field_name = matched_field_name && try_match_char(to_match, c);
        str.next();
        continue;
      } else if ('\\' == c) {  // path 3: escape path
        str.next();

        char* copy_dest_nullptr = nullptr;  // unused
        int output_size_bytes   = 0;        // unused
        if (!try_skip_escape_part(str,
                                  to_match,
                                  copy_dest_nullptr,
                                  escape_style::UNESCAPED,
                                  output_size_bytes,
                                  matched_field_name)) {
          return cuda::std::make_tuple(false, false, nullptr);
        }
      } else {  // path 4: safe code point
        if (!try_skip_safe_code_point(str, c)) {
          return cuda::std::make_tuple(false, false, nullptr);
        }
        matched_field_name = matched_field_name && try_match_char(to_match, c);
      }
    }

    return cuda::std::make_tuple(false, false, nullptr);
  }

  static __device__ inline bool try_match_char(char_range_reader& reader, char c)
  {
    if (!reader.is_null()) {
      if (!reader.is_empty() && reader.current_char() == c) {
        reader.next();
        return true;
      } else {
        return false;
      }
    } else {
      return true;
    }
  }

  /**
   * skip the second char in \", \', \\, \/, \b, \f, \n, \r, \t;
   * skip the HEX chars in \u HEX HEX HEX HEX.
   * @return positive escaped ASCII value if success, -1 otherwise
   */
  static __device__ inline bool try_skip_escape_part(char_range_reader& str,
                                                     char_range_reader& to_match,
                                                     char*& copy_dest,
                                                     escape_style w_style,
                                                     int& output_size_bytes,
                                                     bool& matched_field_name)
  {
    // already skipped the first '\'
    // try skip second part
    if (!str.is_empty()) {
      char const c = str.current_char();
      switch (c) {
        // path 1: \", \', \\, \/, \b, \f, \n, \r, \t
        case '\"':
          if (nullptr != copy_dest && escape_style::UNESCAPED == w_style) { *copy_dest++ = c; }
          if (escape_style::ESCAPED == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = '"';
            }
            output_size_bytes++;
          }
          output_size_bytes++;
          str.next();
          matched_field_name = matched_field_name && try_match_char(to_match, c);
          return true;
        case '\'':
          // for both unescaped/escaped writes a single char '
          if (nullptr != copy_dest) { *copy_dest++ = c; }

          output_size_bytes++;
          str.next();
          matched_field_name = matched_field_name && try_match_char(to_match, c);
          return true;
        case '\\':
          if (nullptr != copy_dest && escape_style::UNESCAPED == w_style) { *copy_dest++ = c; }
          if (escape_style::ESCAPED == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = '\\';
            }
            output_size_bytes++;
          }
          output_size_bytes++;
          str.next();
          matched_field_name = matched_field_name && try_match_char(to_match, c);
          return true;
        case '/':
          // for both unescaped/escaped writes a single char /
          if (nullptr != copy_dest) { *copy_dest++ = c; }
          output_size_bytes++;
          str.next();
          matched_field_name = matched_field_name && try_match_char(to_match, c);
          return true;
        case 'b':
          if (nullptr != copy_dest && escape_style::UNESCAPED == w_style) { *copy_dest++ = '\b'; }
          if (escape_style::ESCAPED == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = 'b';
            }
            output_size_bytes++;
          }
          output_size_bytes++;
          str.next();
          matched_field_name = matched_field_name && try_match_char(to_match, '\b');
          return true;
        case 'f':
          if (nullptr != copy_dest && escape_style::UNESCAPED == w_style) { *copy_dest++ = '\f'; }
          if (escape_style::ESCAPED == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = 'f';
            }
            output_size_bytes++;
          }
          output_size_bytes++;
          str.next();
          matched_field_name = matched_field_name && try_match_char(to_match, '\f');
          return true;
        case 'n':
          if (nullptr != copy_dest && escape_style::UNESCAPED == w_style) { *copy_dest++ = '\n'; }
          if (escape_style::ESCAPED == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = 'n';
            }
            output_size_bytes++;
          }
          output_size_bytes++;
          str.next();
          matched_field_name = matched_field_name && try_match_char(to_match, '\n');
          return true;
        case 'r':
          if (nullptr != copy_dest && escape_style::UNESCAPED == w_style) { *copy_dest++ = '\r'; }
          if (escape_style::ESCAPED == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = 'r';
            }
            output_size_bytes++;
          }
          output_size_bytes++;
          str.next();
          matched_field_name = matched_field_name && try_match_char(to_match, '\r');
          return true;
        case 't':
          if (nullptr != copy_dest && escape_style::UNESCAPED == w_style) { *copy_dest++ = '\t'; }
          if (escape_style::ESCAPED == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = 't';
            }
            output_size_bytes++;
          }
          output_size_bytes++;
          str.next();
          matched_field_name = matched_field_name && try_match_char(to_match, '\t');
          return true;
        // path 1 done: \", \', \\, \/, \b, \f, \n, \r, \t
        case 'u':
          // path 2: \u HEX HEX HEX HEX
          str.next();

          // for both unescaped/escaped writes corresponding utf8 bytes, no need
          // to pass in write style
          return try_skip_unicode(str, to_match, copy_dest, output_size_bytes, matched_field_name);
        default:
          // path 3: invalid
          return false;
      }
    } else {
      // eof, no escaped char after char '\'
      return false;
    }
  }

  /**
   * parse:
   *   fragment SAFECODEPOINT
   *       // 1 not " or '
   *       // 2 not \
   *       // 3 non control character: Ascii value not in [0, 32)
   *     : ~ ["\\\u0000-\u001F]
   *     ;
   */
  static __device__ inline bool try_skip_safe_code_point(char_range_reader& str, char c)
  {
    // 1 the char is not quoted(' or ") char, here satisfy, do not need to check
    // again

    // 2. the char is not \, here satisfy, do not need to check again

    // 3. chars not in [0, 32)
    int v = static_cast<int>(c);
    if (!(v >= 0 && v < 32)) {
      str.next();
      return true;
    } else {
      return false;
    }
  }

  /**
   * convert chars 0-9, a-f, A-F to int value
   */
  static __device__ inline uint8_t hex_value(char c)
  {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
  }

  /**
   * @brief Returns the number of bytes in the specified character.
   *
   * @param character Single character
   * @return Number of bytes
   */
  static __device__ cudf::size_type bytes_in_char_utf8(cudf::char_utf8 character)
  {
    return 1 + static_cast<cudf::size_type>((character & 0x0000'FF00u) > 0) +
           static_cast<cudf::size_type>((character & 0x00FF'0000u) > 0) +
           static_cast<cudf::size_type>((character & 0xFF00'0000u) > 0);
  }

  /**
   * @brief Converts a character code-point value into a UTF-8 character.
   *
   * @param unchr Character code-point to convert.
   * @return Single UTF-8 character.
   */
  static __device__ cudf::char_utf8 codepoint_to_utf8(uint32_t unchr)
  {
    cudf::char_utf8 utf8 = 0;
    if (unchr < 0x0000'0080) {
      // single byte utf8
      utf8 = unchr;
    } else if (unchr < 0x0000'0800) {
      // double byte utf8
      utf8 = (unchr << 2) & 0x1F00;
      utf8 |= (unchr & 0x3F);
      utf8 |= 0x0000'C080;
    } else if (unchr < 0x0001'0000) {
      // triple byte utf8
      utf8 = (unchr << 4) & 0x0F'0000;
      utf8 |= (unchr << 2) & 0x00'3F00;
      utf8 |= (unchr & 0x3F);
      utf8 |= 0x00E0'8080;
    } else if (unchr < 0x0011'0000) {
      // quadruple byte utf8
      utf8 = (unchr << 6) & 0x0700'0000;
      utf8 |= (unchr << 4) & 0x003F'0000;
      utf8 |= (unchr << 2) & 0x0000'3F00;
      utf8 |= (unchr & 0x3F);
      utf8 |= 0xF080'8080u;
    }
    return utf8;
  }

  /**
   * @brief Place a char_utf8 value into a char array.
   *
   * @param character Single character
   * @param[out] str Output array.
   * @return The number of bytes in the character
   */
  static __device__ cudf::size_type from_char_utf8(cudf::char_utf8 character, char* str)
  {
    cudf::size_type const chr_width = bytes_in_char_utf8(character);
    for (cudf::size_type idx = 0; idx < chr_width; ++idx) {
      str[chr_width - idx - 1] = static_cast<char>(character) & 0xFF;
      character                = character >> 8;
    }
    return chr_width;
  }

  /**
   * try skip 4 HEX chars, and possibly another \u escaped hex char
   * sequence, if the sequence is part of a surrogate pair.
   * in pattern: '\\' 'u' HEX HEX HEX HEX, it's a code point of unicode
   */
  static __device__ bool try_skip_unicode(char_range_reader& str,
                                          char_range_reader& to_match,
                                          char*& copy_dest,
                                          int& output_size_bytes,
                                          bool& matched_field_name)
  {
    // Parse initial \u escape sequence
    cudf::char_utf8 code_point = 0;
    for (size_t i = 0; i < 4; i++) {
      if (str.is_empty()) { return false; }
      char const c = str.current_char();
      str.next();
      if (!is_hex_digit(c)) { return false; }
      code_point = (code_point * 16) + hex_value(c);
    }

    // Check for high surrogate
    if (code_point >= 0xD800 && code_point <= 0xDBFF) {
      // We need to peek ahead 6 characters: \uXXXX
      if (str.size() >= 6) {
        // Peek for '\'
        if (str[0] == '\\' && str[1] == 'u' && is_hex_digit(str[2]) && is_hex_digit(str[3]) &&
            is_hex_digit(str[4]) && is_hex_digit(str[5])) {
          cudf::char_utf8 low_surrogate = 0;
          for (size_t i = 0; i < 4; i++) {
            char const c  = str[i + 2];
            low_surrogate = (low_surrogate * 16) + hex_value(c);
          }
          // If valid low surrogate, combine
          if (low_surrogate >= 0xDC00 && low_surrogate <= 0xDFFF) {
            code_point = 0x10000 + ((code_point - 0xD800) << 10) + (low_surrogate - 0xDC00);
            str.next();
            str.next();
            str.next();
            str.next();
            str.next();
            str.next();
          }
        }
      }
    }

    auto utf_char = codepoint_to_utf8(code_point);
    // write utf8 bytes.
    // In UTF-8, the maximum number of bytes used to encode a single character
    // is 4
    char buff[4];
    cudf::size_type const bytes = from_char_utf8(utf_char, buff);
    output_size_bytes += bytes;

    // TODO I think if we do an escape sequence for \n/etc it will return
    // the wrong thing....
    if (nullptr != copy_dest) {
      for (cudf::size_type i = 0; i < bytes; i++) {
        *copy_dest++ = buff[i];
      }
    }

    if (matched_field_name && !to_match.is_null()) {
      for (cudf::size_type i = 0; i < bytes; i++) {
        if (to_match.is_empty() || to_match.current_char() != buff[i]) {
          matched_field_name = false;
          break;
        }
        to_match.next();
      }
    }

    return true;
  }

  // =========== Parse string end ===========

  // =========== Parse number begin ===========

  /**
   * parse number, grammar is:
   * NUMBER
   *   : '-'? INT ('.' [0-9]+)? EXP?
   *   ;
   *
   * fragment INT
   *   // integer part forbis leading 0s (e.g. `01`)
   *   : '0'
   *   | [1-9] [0-9]*
   *   ;
   *
   * fragment EXP
   *   : [Ee] [+\-]? [0-9]+
   *   ;
   *
   * valid number:    0, 0.3, 0e005, 0E005
   * invalid number:  0., 0e, 0E
   *
   * Note: Leading zeroes are not allowed, keep consistent with Spark, e.g.: 00, -01 are invalid
   */
  __device__ inline void parse_number_and_set_current()
  {
    // parse sign
    try_skip(curr_pos, '-');

    // parse unsigned number
    bool is_float = false;
    // store number digits length
    // e.g.: +1.23e-45 length is 5
    int number_digits_length = 0;
    if (try_unsigned_number(is_float, number_digits_length)) {
      if (check_max_num_len(number_digits_length)) {
        current_token = (is_float ? json_token::VALUE_NUMBER_FLOAT : json_token::VALUE_NUMBER_INT);
        // success parsed a number, update the token length
        number_token_len = curr_pos - current_token_start_pos;
      } else {
        set_current_error();
      }
    } else {
      set_current_error();
    }
  }

  /**
   * verify max number digits length if enabled
   * e.g.: +1.23e-45 length is 5
   */
  static __device__ inline bool check_max_num_len(int number_digits_length)
  {
    return
      // disabled num len check
      max_num_len <= 0 ||
      // enabled num len check
      (max_num_len > 0 && number_digits_length <= max_num_len);
  }

  /**
   * parse:  INT ('.' [0-9]+)? EXP?
   * and verify leading zeroes
   *
   * @param[out] is_float, if contains `.` or `e`, set true
   */
  __device__ inline bool try_unsigned_number(bool& is_float, int& number_digits_length)
  {
    if (!eof()) {
      char const c = chars[curr_pos];
      if (c >= '1' && c <= '9') {
        curr_pos++;
        number_digits_length++;
        // first digit is [1-9]
        // path: INT = [1-9] [0-9]*
        number_digits_length += skip_zero_or_more_digits();
        return parse_number_from_fraction(is_float, number_digits_length);
      } else if (c == '0') {
        curr_pos++;
        number_digits_length++;

        // check leading zeros
        if (!eof()) {
          char const next_char_after_zero = chars[curr_pos];
          if (next_char_after_zero >= '0' && next_char_after_zero <= '9') {
            // e.g.: 01 is invalid
            return false;
          }
        }

        // first digit is [0]
        // path: INT = '0'
        return parse_number_from_fraction(is_float, number_digits_length);
      } else {
        // first digit is non [0-9]
        return false;
      }
    } else {
      // eof, has no digits
      return false;
    }
  }

  /**
   * parse: ('.' [0-9]+)? EXP?
   * @param[is_float] is float
   */
  __device__ inline bool parse_number_from_fraction(bool& is_float, int& number_digits_length)
  {
    // parse fraction
    if (try_skip(curr_pos, '.')) {
      // has fraction
      is_float = true;
      // try pattern: [0-9]+
      if (!try_skip_one_or_more_digits(number_digits_length)) { return false; }
    }

    // parse exp
    if (!eof() && (chars[curr_pos] == 'e' || chars[curr_pos] == 'E')) {
      curr_pos++;
      is_float = true;
      return try_parse_exp(number_digits_length);
    }

    return true;
  }

  /**
   * parse: [0-9]*
   * skip zero or more [0-9]
   */
  __device__ inline int skip_zero_or_more_digits()
  {
    int digits = 0;
    while (!eof()) {
      if (is_digit(chars[curr_pos])) {
        digits++;
        curr_pos++;
      } else {
        // point to first non-digit char
        break;
      }
    }
    return digits;
  }

  /**
   * parse: [0-9]+
   * try skip one or more [0-9]
   * @param[out] len: skipped num of digits
   */
  __device__ inline bool try_skip_one_or_more_digits(int& number_digits_length)
  {
    if (!eof() && is_digit(chars[curr_pos])) {
      curr_pos++;
      number_digits_length++;
      number_digits_length += skip_zero_or_more_digits();
      return true;
    } else {
      return false;
    }
  }

  /**
   * parse [eE][+-]?[0-9]+
   * @param[out] exp_len exp len
   */
  __device__ inline bool try_parse_exp(int& number_digits_length)
  {
    // already parsed [eE]

    // parse [+-]?
    if (!eof() && (chars[curr_pos] == '+' || chars[curr_pos] == '-')) { curr_pos++; }

    // parse [0-9]+
    return try_skip_one_or_more_digits(number_digits_length);
  }

  // =========== Parse number end ===========

  /**
   * parse true
   */
  __device__ inline void parse_true_and_set_current()
  {
    // already parsed 't'
    if (try_skip(curr_pos, 'r') && try_skip(curr_pos, 'u') && try_skip(curr_pos, 'e')) {
      current_token = json_token::VALUE_TRUE;
    } else {
      set_current_error();
    }
  }

  /**
   * parse false
   */
  __device__ inline void parse_false_and_set_current()
  {
    // already parsed 'f'
    if (try_skip(curr_pos, 'a') && try_skip(curr_pos, 'l') && try_skip(curr_pos, 's') &&
        try_skip(curr_pos, 'e')) {
      current_token = json_token::VALUE_FALSE;
    } else {
      set_current_error();
    }
  }

  /**
   * parse null
   */
  __device__ inline void parse_null_and_set_current()
  {
    // already parsed 'n'
    if (try_skip(curr_pos, 'u') && try_skip(curr_pos, 'l') && try_skip(curr_pos, 'l')) {
      current_token = json_token::VALUE_NULL;
    } else {
      set_current_error();
    }
  }

  /**
   * parse the key string in key:value pair
   */
  __device__ inline void parse_field_name_and_set_current(
    bool& matched_field_name, char_range to_match_field_name = char_range::null())
  {
    current_token_start_pos = curr_pos;
    // Spark leaves `ALLOW_UNQUOTED_FIELD_NAMES` off, but `try_parse_string` adopts any byte as
    // the delimiter, so a missing opening quote must be rejected here.
    if (eof() || (chars[curr_pos] != '"' && chars[curr_pos] != '\'')) {
      // The caller returns this out-parameter even on the error path.
      matched_field_name = false;
      set_current_error();
      return;
    }
    auto const [success, matched, end] =
      try_parse_string(char_range_reader{chars.slice(curr_pos, chars.size() - curr_pos)},
                       char_range_reader{cuda::std::move(to_match_field_name)});
    if (success) {
      matched_field_name = matched;
      curr_pos           = static_cast<cudf::size_type>(cuda::std::distance(chars.data(), end));
      current_token      = json_token::FIELD_NAME;
    } else {
      set_current_error();
    }
  }

  /**
   * continute parsing the next token and update current token
   * Note: only parse one token at a time
   */
  __device__ inline void parse_next_token_and_set_current(
    bool& has_comma_before_token,
    bool& has_colon_before_token,
    bool& matched_field_name,
    char_range to_match_field_name = char_range::null())
  {
    skip_whitespaces();
    if (!eof()) {
      char const c = chars[curr_pos];
      if (is_context_stack_empty()) {
        // stack is empty

        if (current_token == json_token::INIT) {
          // main root entry point
          parse_first_token_in_value_and_set_current();
        } else {
          // previous token is not INIT, means already get a token; stack is
          // empty; Successfully parsed. Note: ignore the tailing sub-string
          current_token = json_token::SUCCESS;
        }
      } else {
        // stack is non-empty

        if (is_object_context()) {
          // in JSON object context
          if (current_token == json_token::START_OBJECT) {
            // previous token is '{'
            if (c == '}') {
              // empty object
              // close curr object context
              current_token_start_pos = curr_pos;
              curr_pos++;
              pop_curr_context();
              current_token = json_token::END_OBJECT;
            } else {
              // parse key in key:value pair
              parse_field_name_and_set_current(matched_field_name, to_match_field_name);
            }
          } else if (current_token == json_token::FIELD_NAME) {
            if (c == ':') {
              has_colon_before_token = true;
              // skip ':' and parse value in key:value pair
              curr_pos++;
              skip_whitespaces();
              parse_first_token_in_value_and_set_current();
            } else {
              set_current_error();
            }
          } else {
            // expect next key:value pair or '}'
            if (c == '}') {
              // end of object
              current_token_start_pos = curr_pos;
              curr_pos++;
              pop_curr_context();
              current_token = json_token::END_OBJECT;
            } else if (c == ',') {
              has_comma_before_token = true;
              // parse next key:value pair
              curr_pos++;
              skip_whitespaces();
              parse_field_name_and_set_current(matched_field_name, to_match_field_name);
            } else {
              set_current_error();
            }
          }
        } else {
          // in Json array context
          if (current_token == json_token::START_ARRAY) {
            // previous token is '['
            if (c == ']') {
              // curr: ']', empty array
              current_token_start_pos = curr_pos;
              curr_pos++;
              pop_curr_context();
              current_token = json_token::END_ARRAY;
            } else {
              // non-empty array, parse the first value in the array
              parse_first_token_in_value_and_set_current();
            }
          } else {
            if (c == ',') {
              has_comma_before_token = true;
              // skip ',' and parse the next value
              curr_pos++;
              skip_whitespaces();
              parse_first_token_in_value_and_set_current();
            } else if (c == ']') {
              // end of array
              current_token_start_pos = curr_pos;
              curr_pos++;
              pop_curr_context();
              current_token = json_token::END_ARRAY;
            } else {
              set_current_error();
            }
          }
        }
      }
    } else {
      // eof
      if (is_context_stack_empty() && current_token != json_token::INIT) {
        // reach eof; stack is empty; previous token is not INIT
        current_token = json_token::SUCCESS;
      } else {
        // eof, and meet the following cases:
        //   - has unclosed JSON array/object;
        //   - the whole JSON is empty
        set_current_error();
      }
    }
  }

 public:
  /**
   * continute parsing, get next token.
   * The final tokens are ERROR or SUCCESS;
   */
  __device__ json_token next_token()
  {
    // parse next token
    bool has_comma_before_token;  // no-initialization because of do not care here
    bool has_colon_before_token;  // no-initialization because of do not care here
    bool matched_field_name;      // no-initialization because of do not care here
    parse_next_token_and_set_current(
      has_comma_before_token, has_colon_before_token, matched_field_name);
    return current_token;
  }

  /**
   * Continute parsing the next token. If the token is a field name then check if it is
   * matched with the given name.
   */
  __device__ bool parse_next_token_with_matching(cudf::string_view to_match_field_name)
  {
    // parse next token
    bool has_comma_before_token;  // no-initialization because of do not care here
    bool has_colon_before_token;  // no-initialization because of do not care here
    bool matched_field_name;
    parse_next_token_and_set_current(has_comma_before_token,
                                     has_colon_before_token,
                                     matched_field_name,
                                     char_range{to_match_field_name});
    return matched_field_name;
  }

  /**
   * get current token
   */
  __device__ json_token get_current_token() const { return current_token; }

  // TODO make this go away!!!!
  __device__ inline char_range current_range() const
  {
    return chars.slice(current_token_start_pos, curr_pos - current_token_start_pos);
  }

  /**
   * skip children if current token is [ or {, or do nothing otherwise.
   * after this call, the current token is ] or } if token is { or [
   * @return true if JSON is valid so far, false otherwise.
   */
  __device__ bool try_skip_children()
  {
    if (current_token == json_token::ERROR || current_token == json_token::INIT ||
        current_token == json_token::SUCCESS) {
      return false;
    }

    if (current_token != json_token::START_OBJECT && current_token != json_token::START_ARRAY) {
      return true;
    }

    json_token t;
    int open = 1;
    do {
      t = next_token();
      if (t == json_token::START_OBJECT || t == json_token::START_ARRAY) {
        ++open;
      } else if (t == json_token::END_OBJECT || t == json_token::END_ARRAY) {
        if (--open == 0) { return true; }
      } else if (t == json_token::ERROR) {
        return false;
      }
    } while (t != json_token::SUCCESS);
    return false;
  }

  __device__ cudf::size_type compute_unescaped_len() const { return write_unescaped_text(nullptr); }

  /**
   * unescape current token text, then write to destination
   * e.g.: '\\r' is a string with 2 chars '\' 'r', writes 1 char '\r'
   * e.g.: "\u4e2d\u56FD" are code points for Chinese chars "中国",
   *   writes 6 utf8 bytes: -28  -72 -83 -27 -101 -67
   * For number, write verbatim without normalization
   */
  __device__ cudf::size_type write_unescaped_text(char* destination) const
  {
    switch (current_token) {
      case json_token::VALUE_STRING: {
        // can not copy from JSON directly due to escaped chars
        // rewind the pos; parse again with copy
        char_range_reader reader(current_range());
        return write_string(reader, destination, escape_style::UNESCAPED);
      }
      case json_token::VALUE_NUMBER_INT:
        if (number_token_len == 2 && chars[current_token_start_pos] == '-' &&
            chars[current_token_start_pos + 1] == '0') {
          if (nullptr != destination) *destination++ = '0';
          return 1;
        }
        if (nullptr != destination) {
          for (cudf::size_type i = 0; i < number_token_len; ++i) {
            *destination++ = chars[current_token_start_pos + i];
          }
        }
        return number_token_len;
      case json_token::VALUE_NUMBER_FLOAT: {
        // number normalization:
        // 0.03E-2 => 0.3E-5, 200.000 => 200.0, 351.980 => 351.98,
        // 12345678900000000000.0 => 1.23456789E19, 1E308 => 1.0E308
        // 0.0000000000003 => 3.0E-13; 0.003 => 0.003; 0.0003 => 3.0E-4
        // 1.0E309 => "Infinity", -1E309 => "-Infinity"
        double d_value =
          cudf::strings::detail::stod(chars.slice_sv(current_token_start_pos, number_token_len));
        return spark_rapids_jni::ftos_converter::double_normalization(d_value, destination);
      }
      case json_token::VALUE_TRUE:
        if (nullptr != destination) {
          *destination++ = 't';
          *destination++ = 'r';
          *destination++ = 'u';
          *destination++ = 'e';
        }
        return 4;
      case json_token::VALUE_FALSE:
        if (nullptr != destination) {
          *destination++ = 'f';
          *destination++ = 'a';
          *destination++ = 'l';
          *destination++ = 's';
          *destination++ = 'e';
        }
        return 5;
      case json_token::VALUE_NULL:
        if (nullptr != destination) {
          *destination++ = 'n';
          *destination++ = 'u';
          *destination++ = 'l';
          *destination++ = 'l';
        }
        return 4;
      case json_token::FIELD_NAME: {
        // can not copy from JSON directly due to escaped chars
        // rewind the pos; parse again with copy
        char_range_reader reader(current_range());
        return write_string(reader, destination, escape_style::UNESCAPED);
      }
      case json_token::START_ARRAY:
        if (nullptr != destination) { *destination++ = '['; }
        return 1;
      case json_token::END_ARRAY:
        if (nullptr != destination) { *destination++ = ']'; }
        return 1;
      case json_token::START_OBJECT:
        if (nullptr != destination) { *destination++ = '{'; }
        return 1;
      case json_token::END_OBJECT:
        if (nullptr != destination) { *destination++ = '}'; }
        return 1;
      // for the following tokens, return false
      case json_token::SUCCESS:
      case json_token::ERROR:
      case json_token::INIT: return 0;
    }
    return 0;
  }

  __device__ cudf::size_type compute_escaped_len() const { return write_escaped_text(nullptr); }
  /**
   * escape current token text, then write to destination
   * e.g.: '"' is a string with 1 char '"', writes out 4 chars '"' '\' '\"' '"'
   * e.g.: "\u4e2d\u56FD" are code points for Chinese chars "中国",
   *   writes 8 utf8 bytes: '"' -28  -72 -83 -27 -101 -67 '"'
   * For number, write verbatim without normalization
   */
  __device__ cudf::size_type write_escaped_text(char* destination) const
  {
    switch (current_token) {
      case json_token::VALUE_STRING: {
        // can not copy from JSON directly due to escaped chars
        char_range_reader reader(current_range());
        return write_string(reader, destination, escape_style::ESCAPED);
      }
      case json_token::VALUE_NUMBER_INT: {
        if (number_token_len == 2 && chars[current_token_start_pos] == '-' &&
            chars[current_token_start_pos + 1] == '0') {
          if (nullptr != destination) *destination++ = '0';
          return 1;
        }
        if (nullptr != destination) {
          for (cudf::size_type i = 0; i < number_token_len; ++i) {
            *destination++ = chars[current_token_start_pos + i];
          }
        }
        return number_token_len;
      }
      case json_token::VALUE_NUMBER_FLOAT: {
        // number normalization:
        double d_value =
          cudf::strings::detail::stod(chars.slice_sv(current_token_start_pos, number_token_len));
        return spark_rapids_jni::ftos_converter::double_normalization(d_value, destination);
      }
      case json_token::VALUE_TRUE:
        if (nullptr != destination) {
          *destination++ = 't';
          *destination++ = 'r';
          *destination++ = 'u';
          *destination++ = 'e';
        }
        return 4;
      case json_token::VALUE_FALSE:
        if (nullptr != destination) {
          *destination++ = 'f';
          *destination++ = 'a';
          *destination++ = 'l';
          *destination++ = 's';
          *destination++ = 'e';
        }
        return 5;
      case json_token::VALUE_NULL:
        if (nullptr != destination) {
          *destination++ = 'n';
          *destination++ = 'u';
          *destination++ = 'l';
          *destination++ = 'l';
        }
        return 4;
      case json_token::FIELD_NAME: {
        // can not copy from JSON directly due to escaped chars
        char_range_reader reader(current_range());
        return write_string(reader, destination, escape_style::ESCAPED);
      }
      case json_token::START_ARRAY:
        if (nullptr != destination) { *destination++ = '['; }
        return 1;
      case json_token::END_ARRAY:
        if (nullptr != destination) { *destination++ = ']'; }
        return 1;
      case json_token::START_OBJECT:
        if (nullptr != destination) { *destination++ = '{'; }
        return 1;
      case json_token::END_OBJECT:
        if (nullptr != destination) { *destination++ = '}'; }
        return 1;
      // for the following tokens, return false
      case json_token::SUCCESS:
      case json_token::ERROR:
      case json_token::INIT: return 0;
    }
    return 0;
  }

  /**
   * copy current structure to destination.
   * return false if meets JSON format error,
   * reurn true otherwise.
   * @param[out] copy_to
   */
  __device__ cuda::std::pair<bool, size_t> copy_current_structure(char* copy_to)
  {
    switch (current_token) {
      case json_token::INIT:
      case json_token::ERROR:
      case json_token::SUCCESS:
      case json_token::FIELD_NAME:
      case json_token::END_ARRAY:
      case json_token::END_OBJECT: return {false, 0};
      case json_token::VALUE_NUMBER_INT:
      case json_token::VALUE_NUMBER_FLOAT:
      case json_token::VALUE_STRING:
      case json_token::VALUE_TRUE:
      case json_token::VALUE_FALSE:
      case json_token::VALUE_NULL:
        // copy terminal token
        if (nullptr != copy_to) {
          size_t copy_len = write_escaped_text(copy_to);
          return {true, copy_len};
        } else {
          size_t copy_len = compute_escaped_len();
          return {true, copy_len};
        }
      case json_token::START_ARRAY:
      case json_token::START_OBJECT:
        // stack size increased by 1 when meet start object/array
        // copy until meet matched end object/array
        size_t sum_copy_len   = 0;
        int backup_stack_size = stack_size;

        // copy start object/array
        if (nullptr != copy_to) {
          int len = write_escaped_text(copy_to);
          sum_copy_len += len;
          copy_to += len;
        } else {
          sum_copy_len += compute_escaped_len();
        }

        while (true) {
          bool has_comma_before_token = false;
          bool has_colon_before_token = false;

          // parse and get has_comma_before_token, has_colon_before_token
          bool matched_field_name;  // unused
          parse_next_token_and_set_current(
            has_comma_before_token, has_colon_before_token, matched_field_name);

          // check the JSON format
          if (current_token == json_token::ERROR) { return {false, 0}; }

          // write out the token
          if (nullptr != copy_to) {
            if (has_comma_before_token) {
              sum_copy_len++;
              *copy_to++ = ',';
            }
            if (has_colon_before_token) {
              sum_copy_len++;
              *copy_to++ = ':';
            }
            int len = write_escaped_text(copy_to);
            sum_copy_len += len;
            copy_to += len;
          } else {
            if (has_comma_before_token) { sum_copy_len++; }
            if (has_colon_before_token) { sum_copy_len++; }
            sum_copy_len += compute_escaped_len();
          }

          if (backup_stack_size - 1 == stack_size) {
            // indicate meet the matched end object/array
            return {true, sum_copy_len};
          }
        }
        return {false, 0};
    }

    // never happen
    return {false, 0};
  }

  __device__ inline bool max_nesting_depth_exceeded() const { return max_depth_exceeded; }

 private:
  char_range const chars;
  cudf::size_type curr_pos;

  // 64 bits long saves the nested object/array contexts
  // true(bit value 1) is JSON object context
  // false(bit value 0) is JSON array context
  // JSON parser checks array/object are mached, e.g.: [1,2) are wrong
  int64_t context_stack;
  int stack_size = 0;

  // TODO remove if possible
  // save current token start pos, used by coping current token text
  cudf::size_type current_token_start_pos;
  // TODO remove if possible
  // used to store number token length
  cudf::size_type number_token_len;

  json_token current_token;

  // Error check if the maximum nesting depth has been reached.
  bool max_depth_exceeded;
};

}  // namespace spark_rapids_jni
