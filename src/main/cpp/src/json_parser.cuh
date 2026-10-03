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

#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/utility>

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

  // Skips a whole multi-byte sequence at once, where `next()` would step one byte.
  __device__ inline void advance(cudf::size_type n)
  {
    assert(n >= 0 && n <= _len);
    _data += n;
    _len -= n;
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
    : chars(_chars),
      curr_pos(0),
      current_token(json_token::INIT),
      max_depth_exceeded(false),
      token_has_lone_surrogate(false)
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
    // already checked eof
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
  }

  // =========== Parse string begin ===========

  /**
   * parse quoted string and set current token
   */
  __device__ inline void parse_string_and_set_current()
  {
    // A local, not the member: passing the member by reference lets its address escape into the
    // string loop, which blocks optimization there and measurably slows it down.
    bool found_lone_surrogate                           = false;
    [[maybe_unused]] auto const [success, matched, end] = try_parse_string(
      char_range_reader{chars.slice(curr_pos, chars.size() - curr_pos)}, found_lone_surrogate);
    token_has_lone_surrogate = found_lone_surrogate;
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
   * write one UTF-16 code unit as an uppercase JSON unicode escape
   */
  static __device__ inline int write_unicode_escape(uint16_t code_unit, char* output)
  {
    if (output != nullptr) {
      output[0] = '\\';
      output[1] = 'u';
      for (int digit = 0; digit < 4; ++digit) {
        auto const shift  = static_cast<unsigned int>((3 - digit) * 4);
        output[2 + digit] = to_hex_char((code_unit >> shift) & 0x0F);
      }
    }
    return 6;
  }

  /**
   * escape control char ( ASCII code value [0, 32) )
   * e.g.: \0  (ASCII code 0) will be escaped to 6 chars: \u0000
   * e.g.: \10 (ASCII code 0) will be escaped to 2 chars: \n
   * @param char to be escaped, c should in range [0, 32)
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
      // A control char is its own UTF-16 code unit.
      default: return write_unicode_escape(c, output);
    }
  }

  /**
   * @brief One UTF-8 sequence as the JDK decoder classifies it
   */
  struct utf8_sequence {
    cudf::size_type length;  // input bytes consumed
    bool malformed;          // decoder emits one U+FFFD instead of a code point
    bool astral;             // well formed and outside the BMP, so ESCAPED emits a surrogate pair
  };

  // Shared by the write path and the field-name match path so the two cannot compare against
  // different bytes.
  static constexpr char replacement_bytes[3] = {
    static_cast<char>(0xEF), static_cast<char>(0xBF), static_cast<char>(0xBD)};

  static __device__ inline bool is_utf8_continuation(unsigned char byte)
  {
    return (byte & 0xC0) == 0x80;
  }

  /**
   * @brief Whether `str` starts on a 2-byte sequence the decoder accepts unchanged
   *
   * Shared with `classify_utf8_sequence`, which must decide this case identically.
   */
  static __device__ inline bool is_plain_two_byte(char_range const& str, unsigned char lead)
  {
    return lead >= 0xC2 && lead <= 0xDF && str.size() >= 2 &&
           is_utf8_continuation(static_cast<unsigned char>(str[1]));
  }

  /**
   * @brief Whether `str` starts on a 3-byte sequence the decoder accepts unchanged
   *
   * 0xE0 and 0xED are excluded because they still need the overlong and surrogate checks.
   */
  static __device__ inline bool is_plain_three_byte(char_range const& str, unsigned char lead)
  {
    return lead >= 0xE1 && lead <= 0xEF && lead != 0xED && str.size() >= 3 &&
           is_utf8_continuation(static_cast<unsigned char>(str[1])) &&
           is_utf8_continuation(static_cast<unsigned char>(str[2]));
  }

  /**
   * @brief Bytes the JDK decoder consumes for a malformed sequence led by 0xF0..0xF4
   */
  static __device__ inline cudf::size_type malformed_four_byte_length(char_range const& str,
                                                                      unsigned char lead)
  {
    if (str.size() < 2) { return 1; }
    auto const b2 = static_cast<unsigned char>(str[1]);
    // An overlong 0xF0 or out-of-range 0xF4 is rejected together with its lead, so only one byte
    // is consumed.
    if ((lead == 0xF0 && (b2 < 0x90 || b2 > 0xBF)) || (lead == 0xF4 && (b2 & 0xF0) != 0x80) ||
        !is_utf8_continuation(b2)) {
      return 1;
    }
    if (str.size() < 3 || !is_utf8_continuation(static_cast<unsigned char>(str[2]))) { return 2; }
    return 3;
  }

  /**
   * @brief Classify the sequence starting at `str`, which must not be empty, as the JDK decoder
   * does
   *
   * Spark decodes the document before Jackson tokenizes it, so a malformed run has already
   * collapsed to one U+FFFD, and matching the decoder's run lengths is what keeps the output
   * identical: `ED A0 80` fails only the surrogate check, so all three bytes become one U+FFFD,
   * while the overlong `E0 80 80` fails after the lead and yields three. The size of `str` bounds
   * the lookahead.
   *
   * Every rule below is decided on the raw bytes; only the astral escape needs the code point, so
   * it is computed at that one branch rather than for every character.
   */
  static __device__ inline utf8_sequence classify_utf8_sequence(char_range const& str)
  {
    auto const remaining = str.size();
    auto const lead      = static_cast<unsigned char>(str[0]);

    if (lead <= 0x7F) { return {1, false, false}; }

    // Bare continuation byte, an overlong 2-byte lead, or a lead beyond the Unicode range.
    if (lead <= 0xBF || lead == 0xC0 || lead == 0xC1 || lead >= 0xF5) { return {1, true, false}; }

    if (lead <= 0xDF) {
      if (is_plain_two_byte(str, lead)) { return {2, false, false}; }
      if (remaining < 2) { return {remaining, true, false}; }
      return {1, true, false};
    }

    if (lead <= 0xEF) {
      if (is_plain_three_byte(str, lead)) { return {3, false, false}; }
      if (remaining < 2) { return {remaining, true, false}; }
      auto const b2 = static_cast<unsigned char>(str[1]);
      // A 0xE0 lead with a 0x80..0x9F second byte is overlong; the decoder drops the lead alone.
      if ((lead == 0xE0 && (b2 & 0xE0) == 0x80) || !is_utf8_continuation(b2)) {
        return {1, true, false};
      }
      if (remaining < 3) { return {remaining, true, false}; }
      if (!is_utf8_continuation(static_cast<unsigned char>(str[2]))) { return {2, true, false}; }
      // A 0xED lead with a second byte at or above 0xA0 encodes a surrogate, which is structurally
      // valid, so the whole 3-byte run is replaced.
      if (lead == 0xED && b2 >= 0xA0) { return {3, true, false}; }
      return {3, false, false};
    }

    // A length of 3 means the lead, the second byte and the third all satisfy the four-byte rules,
    // which already exclude the overlong 0xF0 and the out-of-range 0xF4 forms, so a fourth
    // continuation byte completes a well-formed code point, necessarily outside the BMP.
    auto const bad_length = malformed_four_byte_length(str, lead);
    if (remaining < 4) { return {cuda::std::min(bad_length, remaining), true, false}; }
    if (bad_length != 3 || !is_utf8_continuation(static_cast<unsigned char>(str[3]))) {
      return {bad_length, true, false};
    }
    return {4, false, true};
  }

  /**
   * Writes the UTF-8 sequence `str` starts on, advances `str` past the bytes it consumed and
   * `copy_destination` past the bytes it wrote, and returns that written count. `str` must not be
   * empty. A null `copy_destination` sizes without writing, so a sizing pass and a writing pass
   * return the same count. Under ESCAPED an astral character is written as a surrogate-pair escape,
   * so the count it returns is not the count it consumed.
   */
  static __device__ inline int write_utf8_sequence(char_range_reader& str,
                                                   char*& copy_destination,
                                                   escape_style w_style)
  {
    auto const lead = static_cast<unsigned char>(str.current_char());

    // A sequence the decoder accepts unchanged copies through under both write styles.
    if (is_plain_two_byte(str, lead)) {
      if (copy_destination != nullptr) {
        copy_destination[0] = str[0];
        copy_destination[1] = str[1];
        copy_destination += 2;
      }
      str.advance(2);
      return 2;
    }
    if (is_plain_three_byte(str, lead)) {
      if (copy_destination != nullptr) {
        copy_destination[0] = str[0];
        copy_destination[1] = str[1];
        copy_destination[2] = str[2];
        copy_destination += 3;
      }
      str.advance(3);
      return 3;
    }

    // Every other lead needs the full classification.
    auto const sequence = classify_utf8_sequence(str);
    int written         = 0;
    if (sequence.malformed) {
      // Jackson leaves U+FFFD unescaped, so both styles emit its raw bytes. One input byte can
      // become three, and the size check happens after writing, so the growth must fit the
      // scratch buffer's padding.
      if (copy_destination != nullptr) {
        copy_destination[0] = replacement_bytes[0];
        copy_destination[1] = replacement_bytes[1];
        copy_destination[2] = replacement_bytes[2];
        copy_destination += 3;
      }
      written = 3;
    } else if (escape_style::ESCAPED == w_style && sequence.astral) {
      // Jackson renders a raw astral char as a surrogate-pair escape, turning 4 input bytes into
      // 12. That is the largest expansion here, so an astral-heavy row overruns its input-sized
      // budget and forces the kernel's second launch; the scratch padding absorbs the write.
      uint32_t const code_point = ((static_cast<unsigned char>(str[0]) & 0x07u) << 18) |
                                  ((static_cast<unsigned char>(str[1]) & 0x3Fu) << 12) |
                                  ((static_cast<unsigned char>(str[2]) & 0x3Fu) << 6) |
                                  (static_cast<unsigned char>(str[3]) & 0x3Fu);
      uint32_t const surrogate_offset = code_point - 0x0001'0000u;
      auto const high_unit            = static_cast<uint16_t>(0xD800u + (surrogate_offset >> 10));
      auto const low_unit  = static_cast<uint16_t>(0xDC00u + (surrogate_offset & 0x3FFu));
      auto const high_size = write_unicode_escape(high_unit, copy_destination);
      if (copy_destination != nullptr) { copy_destination += high_size; }
      auto const low_size = write_unicode_escape(low_unit, copy_destination);
      if (copy_destination != nullptr) { copy_destination += low_size; }
      written = high_size + low_size;
    } else {
      if (copy_destination != nullptr) {
        for (cudf::size_type i = 0; i < sequence.length; ++i) {
          copy_destination[i] = str[i];
        }
        copy_destination += sequence.length;
      }
      written = sequence.length;
    }

    // Advance unconditionally so the sizing and writing passes stay in lock step.
    str.advance(sequence.length);
    return written;
  }

  /**
   * @brief Write the string token `str` starts on, returning the byte count either way
   *
   * `str` must be positioned on the token's delimiting quote; whatever byte it starts on becomes
   * the delimiter without validation, so the caller must reject a non-quote itself.
   */
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

    // One set serves every escape: this writer never reads them back, and the tokenizer already
    // recorded any lone surrogate.
    char_range_reader to_match(char_range::null());
    bool matched_field_name   = false;
    bool found_lone_surrogate = false;

    // scan string content
    while (!str.is_empty()) {
      char const c        = str.current_char();
      bool const is_ascii = static_cast<unsigned char>(c) < 0x80;
      // Test ASCII first so the common byte never reaches the classification below.
      if (static_cast<unsigned char>(c) >= 32 && is_ascii && c != quote_char && '\\' != c &&
          !(escape_style::ESCAPED == w_style && '"' == c)) {
        if (copy_destination != nullptr) { *copy_destination++ = c; }
        str.next();
        output_size_bytes++;
        continue;
      }
      if (c == quote_char) {
        // path 1: match closing quote char
        str.next();

        // write the end " if write style is escaped
        if (escape_style::ESCAPED == w_style) {
          output_size_bytes++;
          if (nullptr != copy_destination) { *copy_destination++ = '"'; }
        }

        return output_size_bytes;
      } else if (is_ascii && c < 32) {
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
        if (!try_skip_escape_part(str,
                                  to_match,
                                  copy_destination,
                                  w_style,
                                  output_size_bytes,
                                  matched_field_name,
                                  found_lone_surrogate)) {
          return output_size_bytes;
        }
      } else {
        // path 4: safe code point
        if (is_ascii) {
          // handle single unescaped " char; happens when string is quoted by char '
          // e.g.:  'A"' string, escape to "A\\"" (5 chars: " A \ " ")
          if ('\"' == c && escape_style::ESCAPED == w_style) {
            if (copy_destination != nullptr) { *copy_destination++ = '\\'; }
            output_size_bytes++;
          }

          if (copy_destination != nullptr) { *copy_destination++ = c; }
          str.next();
          output_size_bytes++;
          continue;
        }

        output_size_bytes += write_utf8_sequence(str, copy_destination, w_style);
      }
    }

    // technically this is an error state, but we will do our best from here...
    return output_size_bytes;
  }

  /**
   * @brief The body of `try_parse_string`, instantiated twice from this one definition
   *
   * `WithUtf8Classification` selects what a byte at or above 0x80 does. It is a template
   * parameter because the classification measurably slows the byte loop even on ASCII input that
   * never reaches it, so the hot instantiation must not contain it at all: that one leaves the
   * loop as soon as a field name needs classifying and hands the string to the out-of-line
   * `try_parse_string_slow`.
   */
  template <bool WithUtf8Classification>
  static __device__ __forceinline__ cuda::std::tuple<bool, bool, char const*> try_parse_string_impl(
    char_range_reader str, bool& found_lone_surrogate, char_range_reader to_match)
  {
    if (str.is_empty()) { return cuda::std::make_tuple(false, false, nullptr); }
    char const quote_char   = str.current_char();
    bool matched_field_name = !to_match.is_null();

    // The bailing instantiation restarts the parse rather than resuming it, so both readers are
    // saved as they were before the opening quote was skipped.
    [[maybe_unused]] char_range_reader const str_start      = str;
    [[maybe_unused]] char_range_reader const to_match_start = to_match;

    // skip left quote char
    // We don't need to actually verify what it is, because we just read it.
    str.next();

    [[maybe_unused]] bool needs_classification = false;

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
        // Record a lone surrogate for the writer; tokenizing itself must not fail on one.
        if (!try_skip_escape_part(str,
                                  to_match,
                                  copy_dest_nullptr,
                                  escape_style::UNESCAPED,
                                  output_size_bytes,
                                  matched_field_name,
                                  found_lone_surrogate)) {
          return cuda::std::make_tuple(false, false, nullptr);
        }
      } else if constexpr (WithUtf8Classification) {  // path 4: safe code point
        // Test ASCII first so the common byte never reaches the classification below.
        if (static_cast<unsigned char>(c) < 0x80) {
          if (!try_skip_safe_code_point(str, c)) {
            return cuda::std::make_tuple(false, false, nullptr);
          }
          matched_field_name = matched_field_name && try_match_char(to_match, c);
        } else {
          // Spark decodes the whole document before Jackson tokenizes it, so a malformed run inside
          // a name has already collapsed to one U+FFFD by the time a path is compared against it.
          // Comparing the raw bytes instead would leave such a name unreachable from any path.
          auto const sequence = classify_utf8_sequence(str);
          if (sequence.malformed) {
            matched_field_name = matched_field_name &&
                                 try_match_char(to_match, replacement_bytes[0]) &&
                                 try_match_char(to_match, replacement_bytes[1]) &&
                                 try_match_char(to_match, replacement_bytes[2]);
          } else {
            for (cudf::size_type i = 0; i < sequence.length; ++i) {
              matched_field_name = matched_field_name && try_match_char(to_match, str[i]);
            }
          }
          str.advance(sequence.length);
        }
      } else {  // path 4: safe code point, bailing out instead of classifying
        // Classification only matters while matching a field name, where a malformed run must
        // collapse to one U+FFFD. With no name to match, consuming byte by byte accepts the same
        // input: every byte of a sequence has its high bit set and the delimiter is ASCII, so
        // none of them can close the string.
        if (static_cast<unsigned char>(c) >= 0x80 && !to_match.is_null()) {
          needs_classification = true;
          break;
        }
        if (!try_skip_safe_code_point(str, c)) {
          return cuda::std::make_tuple(false, false, nullptr);
        }
        matched_field_name = matched_field_name && try_match_char(to_match, c);
      }
    }

    // Replaying the prefix the loop already consumed is what keeps the classification out of that
    // loop. `found_lone_surrogate` is only ever set, so replaying cannot lose or invent one.
    if constexpr (!WithUtf8Classification) {
      if (needs_classification) {
        return try_parse_string_slow(str_start, found_lone_surrogate, to_match_start);
      }
    }

    return cuda::std::make_tuple(false, false, nullptr);
  }

  /**
   * @brief `try_parse_string_impl<true>` kept out of line, so the classification never lands in
   * the byte loop of the instantiation that runs on every string
   */
  static __device__ __noinline__ cuda::std::tuple<bool, bool, char const*> try_parse_string_slow(
    char_range_reader str, bool& found_lone_surrogate, char_range_reader to_match)
  {
    return try_parse_string_impl<true>(str, found_lone_surrogate, to_match);
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
   * @param str string to parse
   * @param[in,out] found_lone_surrogate set to true when a `\uXXXX` escape decodes to an unpaired
   *        UTF-16 surrogate; never cleared here, so the caller must pass it already false
   * @param to_match expected match str
   * @return a tuple of values indicating if the parse process was successful, if field name was
   *         matched, and a pointer to the past-end position of the parsed data
   */
  static __device__ inline cuda::std::tuple<bool, bool, char const*> try_parse_string(
    char_range_reader str,
    bool& found_lone_surrogate,
    char_range_reader to_match = char_range_reader(char_range::null()))
  {
    return try_parse_string_impl<false>(str, found_lone_surrogate, to_match);
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
   * @param[in,out] found_lone_surrogate set to true when a `\uXXXX` escape decodes to an unpaired
   *        UTF-16 surrogate; never cleared here, so the caller must pass it already false
   * @return true if the escape was consumed, false if no valid escape char follows the backslash
   */
  static __device__ inline bool try_skip_escape_part(char_range_reader& str,
                                                     char_range_reader& to_match,
                                                     char*& copy_dest,
                                                     escape_style w_style,
                                                     int& output_size_bytes,
                                                     bool& matched_field_name,
                                                     bool& found_lone_surrogate)
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

          return try_skip_unicode(str,
                                  to_match,
                                  copy_dest,
                                  w_style,
                                  output_size_bytes,
                                  matched_field_name,
                                  found_lone_surrogate);
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
   * The UTF-16 surrogate ranges.
   */
  static __device__ inline bool is_high_surrogate(uint32_t unit)
  {
    return unit >= 0xD800 && unit <= 0xDBFF;
  }

  static __device__ inline bool is_low_surrogate(uint32_t unit)
  {
    return unit >= 0xDC00 && unit <= 0xDFFF;
  }

  static __device__ inline bool is_surrogate(uint32_t unit)
  {
    return is_high_surrogate(unit) || is_low_surrogate(unit);
  }

  /**
   * @brief Consume the `\uXXXX` low surrogate following `str` and return that code unit
   *
   * Returns 0 and leaves `str` unmoved when what follows is not one; zero is unambiguous because a
   * low surrogate is never zero.
   */
  static __device__ inline uint16_t try_consume_low_surrogate(char_range_reader& str)
  {
    // '\' 'u' and four hex digits.
    if (str.size() < 6) { return 0; }
    if (str[0] != '\\' || str[1] != 'u' || !is_hex_digit(str[2]) || !is_hex_digit(str[3]) ||
        !is_hex_digit(str[4]) || !is_hex_digit(str[5])) {
      return 0;
    }
    cudf::char_utf8 low_surrogate = 0;
    for (size_t i = 0; i < 4; i++) {
      low_surrogate = (low_surrogate * 16) + hex_value(str[i + 2]);
    }
    if (!is_low_surrogate(low_surrogate)) { return 0; }
    str.advance(6);
    return static_cast<uint16_t>(low_surrogate);
  }

  /**
   * try skip 4 HEX chars, and possibly another \u escaped hex char
   * sequence, if the sequence is part of a surrogate pair.
   * in pattern: '\\' 'u' HEX HEX HEX HEX, it's a code point of unicode
   *
   * @param w_style under ESCAPED, surrogate code units are re-emitted as `\uXXXX` rather than
   *        decoded, matching Jackson's writeString
   * @param[in,out] found_lone_surrogate set to true when a `\uXXXX` escape decodes to an unpaired
   *        UTF-16 surrogate; never cleared here, so the caller must pass it already false. ESCAPED
   *        re-emits the units and returns before that point, so it always leaves the flag untouched
   */
  static __device__ bool try_skip_unicode(char_range_reader& str,
                                          char_range_reader& to_match,
                                          char*& copy_dest,
                                          escape_style w_style,
                                          int& output_size_bytes,
                                          bool& matched_field_name,
                                          bool& found_lone_surrogate)
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
    auto const first_code_unit = static_cast<uint16_t>(code_point);
    bool has_surrogate_pair    = false;
    uint16_t low_code_unit     = 0;

    // A high surrogate followed by a low one is a single astral character.
    if (is_high_surrogate(code_point)) {
      low_code_unit = try_consume_low_surrogate(str);
      if (low_code_unit != 0) {
        has_surrogate_pair = true;
        code_point         = 0x0001'0000 + ((code_point - 0xD800) << 10) + (low_code_unit - 0xDC00);
      }
    }

    // Jackson escapes every surrogate code unit, including a valid pair, so re-emit the units
    // rather than the decoded character.
    if (w_style == escape_style::ESCAPED && (has_surrogate_pair || is_surrogate(first_code_unit))) {
      auto const first_size = write_unicode_escape(first_code_unit, copy_dest);
      if (copy_dest != nullptr) { copy_dest += first_size; }
      output_size_bytes += first_size;
      if (has_surrogate_pair) {
        auto const second_size = write_unicode_escape(low_code_unit, copy_dest);
        if (copy_dest != nullptr) { copy_dest += second_size; }
        output_size_bytes += second_size;
      }
      return true;
    }
    // An unpaired surrogate has no UTF-8 encoding, so Spark nulls the row. Flag rather than fail:
    // tokenizing must still succeed, and `write_raw` gates on the flag.
    if (!has_surrogate_pair && is_surrogate(code_point)) {
      found_lone_surrogate = true;
      // A key holding a lone surrogate matches no query key, not even one carrying the same
      // surrogate. Spark would match that, so it is a known divergence on an input that cannot
      // round-trip through UTF-8 anyway.
      if (matched_field_name && !to_match.is_null()) { matched_field_name = false; }
      return true;
    }

    // Only the last branch needs the decoded UTF-8 bytes; the ESCAPED forms above write pure
    // ASCII. Field-name matching sits there because it compares those same bytes.
    if (w_style == escape_style::ESCAPED && (code_point == '"' || code_point == '\\')) {
      output_size_bytes += 2;
      if (copy_dest != nullptr) {
        *copy_dest++ = '\\';
        *copy_dest++ = static_cast<char>(code_point);
      }
    } else if (w_style == escape_style::ESCAPED && code_point < 32) {
      auto const escaped_size = escape_char(static_cast<unsigned char>(code_point), copy_dest);
      output_size_bytes += escaped_size;
      if (copy_dest != nullptr) { copy_dest += escaped_size; }
    } else {
      // In UTF-8, the maximum number of bytes used to encode a single character is 4.
      char buff[4];
      cudf::size_type const bytes = from_char_utf8(codepoint_to_utf8(code_point), buff);
      output_size_bytes += bytes;
      if (copy_dest != nullptr) {
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
    // A local for the same reason as in `parse_string_and_set_current`.
    bool found_lone_surrogate = false;
    auto const [success, matched, end] =
      try_parse_string(char_range_reader{chars.slice(curr_pos, chars.size() - curr_pos)},
                       found_lone_surrogate,
                       char_range_reader{cuda::std::move(to_match_field_name)});
    token_has_lone_surrogate = found_lone_surrogate;
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
    // Every token transition funnels through here, so clearing the flag keeps it a strict
    // description of `current_token`.
    token_has_lone_surrogate = false;
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
    // Initialized because this one is returned; only the field-name paths assign it.
    bool matched_field_name = false;
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
   *
   * @note A lone surrogate has no UTF-8 encoding, so it is dropped rather than written. A caller
   * that needs the whole token must reject such a token first, by checking `has_lone_surrogate()`.
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

  /**
   * whether the current string token has a `\uXXXX` escape that decodes to an unpaired UTF-16
   * surrogate; a surrogate already encoded in the input bytes is not detected
   */
  __device__ inline bool has_lone_surrogate() const { return token_has_lone_surrogate; }

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

  // Set while parsing a string token whose `\uXXXX` escape decodes to an unpaired UTF-16 surrogate,
  // which Spark renders as a null row. A surrogate already encoded in the input bytes is not
  // detected. Reset on every token transition, so it always describes `current_token`.
  bool token_has_lone_surrogate;
};

}  // namespace spark_rapids_jni
