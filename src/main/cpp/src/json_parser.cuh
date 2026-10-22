/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace spark_rapids_jni {

enum class escape_style {
  // Omits quotes and decodes escapes, e.g. \r becomes one carriage-return byte.
  UNESCAPED,
  // Adds quotes and re-escapes: a lone double quote becomes four bytes; Unicode stays decoded.
  ESCAPED
};

// Increasing this limit requires widening context_stack.
constexpr int MAX_JSON_NESTING_DEPTH = 64;

// Counts digits, including exponent digits: -123.45e-67 has 7; nonpositive limits disable it.
constexpr int max_num_len = 1000;

enum class json_token : int8_t {
  INIT = 0,
  SUCCESS,
  ERROR,
  START_OBJECT,
  END_OBJECT,
  START_ARRAY,
  END_ARRAY,
  FIELD_NAME,
  VALUE_STRING,
  VALUE_NUMBER_INT,
  VALUE_NUMBER_FLOAT,
  VALUE_TRUE,
  VALUE_FALSE,
  VALUE_NULL
};

// Borrowed bytes, not UTF-8 characters; callers retain storage and bound each index and slice.
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

  // CUDA member-initialization workaround: use explicit construction, not this static factory.
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

// Consumes a borrowed range by moving its begin pointer; it neither owns nor validates bytes.
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

  // Requires a non-empty range.
  __device__ inline char current_char() const { return _data[0]; }
};

/**
 * Token-by-token parser with Spark/Jackson's relaxed strings: single quotes and raw controls are
 * accepted, but whitespace is only space, tab, CR or LF. Scans do not validate UTF-8.
 * Numbers accept -0 and 1e01, but reject 01, +1, 1., 1e and 1e+.
 * Success covers the first root value; trailing bytes are ignored. Malformed input reports ERROR,
 * and nesting overflow also sets max_nesting_depth_exceeded for the caller's error policy.
 */
class json_parser {
 public:
  __device__ inline explicit json_parser(char_range _chars)
    : chars(_chars), curr_pos(0), current_token(json_token::INIT), max_depth_exceeded(false)
  {
  }

 private:
  static __device__ inline bool get_bit_value(int64_t number, int bitIndex)
  {
    long shifted = number >> bitIndex;

    bool bit_value = shifted & 1;

    return bit_value;
  }

  static __device__ inline void set_bit_value(int64_t& number, int bit_index, bool bit_value)
  {
    long mask = 1L << bit_index;

    if (bit_value) {
      number |= mask;
    } else {
      number &= ~mask;
    }
  }

  __device__ inline bool eof(cudf::size_type pos) const { return pos >= chars.size(); }
  __device__ inline bool eof() const { return curr_pos >= chars.size(); }

  static __device__ inline bool is_hex_digit(char c)
  {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
  }

  static __device__ inline bool is_digit(char c) { return (c >= '0' && c <= '9'); }

  static __device__ inline bool is_whitespace(char c)
  {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
  }

  __device__ inline void skip_whitespaces()
  {
    while (!eof() && is_whitespace(chars[curr_pos])) {
      curr_pos++;
    }
  }

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

  __device__ inline bool try_push_context(json_token token)
  {
    if (stack_size < MAX_JSON_NESTING_DEPTH) {
      push_context(token);
      return true;
    } else {
      return false;
    }
  }

  __device__ inline void push_context(json_token token)
  {
    bool v = json_token::START_OBJECT == token ? true : false;
    set_bit_value(context_stack, stack_size, v);
    stack_size++;
  }

  __device__ inline bool is_object_context() const
  {
    return get_bit_value(context_stack, stack_size - 1);
  }

  __device__ inline void pop_curr_context() { stack_size--; }

  __device__ inline bool is_context_stack_empty() const { return stack_size == 0; }

  __device__ inline void set_current_error() { current_token = json_token::ERROR; }

  __device__ inline void parse_first_token_in_value_and_set_current()
  {
    if (eof()) {
      set_current_error();
      return;
    }
    current_token_start_pos = curr_pos;
    char c                  = chars[curr_pos];
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

  static __device__ inline char to_hex_char(unsigned int v)
  {
    if (v < 10)
      return '0' + v;
    else
      return 'A' + (v - 10);
  }

  // Requires a control byte below 32: NUL becomes \u0000; a null output only counts bytes.
  static __device__ inline int escape_char(unsigned char c, char* output)
  {
    if (nullptr == output) {
      switch (c) {
        case 8:             // \b
        case 9:             // \t
        case 10:            // \n
        case 12:            // \f
        case 13: return 2;  // \r
        default: return 6;  // \u00XX
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

        if (c >= 16) {
          output[4] = '1';
        } else {
          output[4] = '0';
        }

        unsigned int v = c % 16;
        output[5]      = to_hex_char(v);
        return 6;
    }
  }

  /**
   * Decodes a previously validated quoted token and consumes its closing quote in the local reader.
   * A null destination only counts bytes. UNESCAPED omits quotes; ESCAPED emits double quotes and
   * re-escapes controls and simple escapes. Unicode escapes decode in either style.
   * Invalid input can return a partial byte count; callers must validate before using this writer.
   */
  static __device__ inline int write_string(char_range_reader& str,
                                            char* copy_destination,
                                            escape_style w_style)
  {
    if (str.is_empty()) { return 0; }
    char const quote_char = str.current_char();
    int output_size_bytes = 0;

    if (escape_style::ESCAPED == w_style) {
      output_size_bytes++;
      if (nullptr != copy_destination) { *copy_destination++ = '"'; }
    }

    str.next();

    while (!str.is_empty()) {
      char const c = str.current_char();
      int const v  = static_cast<int>(c);
      if (c == quote_char) {
        str.next();

        if (escape_style::ESCAPED == w_style) {
          output_size_bytes++;
          if (nullptr != copy_destination) { *copy_destination++ = '"'; }
        }

        return output_size_bytes;
      } else if (v >= 0 && v < 32) {
        if (escape_style::UNESCAPED == w_style) {
          output_size_bytes++;
          if (copy_destination != nullptr) { *copy_destination++ = str.current_char(); }
        } else {
          int const escape_chars = escape_char(str.current_char(), copy_destination);
          if (copy_destination != nullptr) { copy_destination += escape_chars; }
          output_size_bytes += escape_chars;
        }

        str.next();
      } else if ('\\' == c) {
        str.next();
        char_range_reader to_match(char_range::null());
        bool matched_field_name{false};
        if (!try_skip_escape_part(
              str, to_match, copy_destination, w_style, output_size_bytes, matched_field_name)) {
          return output_size_bytes;
        }
      } else {
        // Quoted output escapes a double quote inside single quotes: 'A"' becomes "A\"".
        if ('\"' == c && escape_style::ESCAPED == w_style) {
          if (copy_destination != nullptr) { *copy_destination++ = '\\'; }
          output_size_bytes++;
        }

        if (copy_destination != nullptr) { *copy_destination++ = c; }
        str.next();
        output_size_bytes++;
      }
    }

    return output_size_bytes;
  }

  /**
   * Requires an opening single or double quote; scans bytes without validating UTF-8.
   * Returns {success, decoded name matched, raw cursor after the closing quote}.
   * Matching is false with a null name; failure returns {false, false, nullptr}.
   */
  static __device__ inline cuda::std::tuple<bool, bool, char const*> try_parse_string(
    char_range_reader str, char_range_reader to_match = char_range_reader(char_range::null()))
  {
    if (str.is_empty()) { return cuda::std::make_tuple(false, false, nullptr); }
    char const quote_char   = str.current_char();
    bool matched_field_name = !to_match.is_null();

    str.next();

    // ASCII delimiters cannot occur inside valid UTF-8 multibyte sequences.
    while (!str.is_empty()) {
      char c = str.current_char();
      int v  = static_cast<int>(c);
      if (c == quote_char) {
        str.next();
        matched_field_name = matched_field_name && (to_match.is_null() || to_match.is_empty());
        return cuda::std::make_tuple(true, matched_field_name, str.data());
      } else if (v >= 0 && v < 32) {
        matched_field_name = matched_field_name && try_match_char(to_match, c);
        str.next();
        continue;
      } else if ('\\' == c) {
        str.next();

        char* copy_dest_nullptr = nullptr;
        int output_size_bytes   = 0;
        if (!try_skip_escape_part(str,
                                  to_match,
                                  copy_dest_nullptr,
                                  escape_style::UNESCAPED,
                                  output_size_bytes,
                                  matched_field_name)) {
          return cuda::std::make_tuple(false, false, nullptr);
        }
      } else {
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

  // Starts after a backslash; accepts escaped apostrophes/slashes and advances past the escape.
  static __device__ inline bool try_skip_escape_part(char_range_reader& str,
                                                     char_range_reader& to_match,
                                                     char*& copy_dest,
                                                     escape_style w_style,
                                                     int& output_size_bytes,
                                                     bool& matched_field_name)
  {
    if (!str.is_empty()) {
      char const c = str.current_char();
      switch (c) {
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
        case 'u':
          str.next();
          return try_skip_unicode(str, to_match, copy_dest, output_size_bytes, matched_field_name);
        default: return false;
      }
    } else {
      return false;
    }
  }

  static __device__ inline bool try_skip_safe_code_point(char_range_reader& str, char c)
  {
    int v = static_cast<int>(c);
    if (!(v >= 0 && v < 32)) {
      str.next();
      return true;
    } else {
      return false;
    }
  }

  static __device__ inline uint8_t hex_value(char c)
  {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
  }

  static __device__ cudf::size_type bytes_in_char_utf8(cudf::char_utf8 character)
  {
    return 1 + static_cast<cudf::size_type>((character & 0x0000'FF00u) > 0) +
           static_cast<cudf::size_type>((character & 0x00FF'0000u) > 0) +
           static_cast<cudf::size_type>((character & 0xFF00'0000u) > 0);
  }

  // Packs UTF-8 bytes, not a Unicode code-point value; from_char_utf8 emits them in byte order.
  static __device__ cudf::char_utf8 codepoint_to_utf8(uint32_t unchr)
  {
    cudf::char_utf8 utf8 = 0;
    if (unchr < 0x0000'0080) {
      utf8 = unchr;
    } else if (unchr < 0x0000'0800) {
      utf8 = (unchr << 2) & 0x1F00;
      utf8 |= (unchr & 0x3F);
      utf8 |= 0x0000'C080;
    } else if (unchr < 0x0001'0000) {
      utf8 = (unchr << 4) & 0x0F'0000;
      utf8 |= (unchr << 2) & 0x00'3F00;
      utf8 |= (unchr & 0x3F);
      utf8 |= 0x00E0'8080;
    } else if (unchr < 0x0011'0000) {
      utf8 = (unchr << 6) & 0x0700'0000;
      utf8 |= (unchr << 4) & 0x003F'0000;
      utf8 |= (unchr << 2) & 0x0000'3F00;
      utf8 |= (unchr & 0x3F);
      utf8 |= 0xF080'8080u;
    }
    return utf8;
  }

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
   * Starts after \u; reads four hex digits.
   * Pairs a high surrogate only with an immediately following \u low surrogate.
   * E.g. \uD83E\uDD66 becomes U+1F966; unpaired surrogates are encoded directly.
   */
  static __device__ bool try_skip_unicode(char_range_reader& str,
                                          char_range_reader& to_match,
                                          char*& copy_dest,
                                          int& output_size_bytes,
                                          bool& matched_field_name)
  {
    cudf::char_utf8 code_point = 0;
    for (size_t i = 0; i < 4; i++) {
      if (str.is_empty()) { return false; }
      char const c = str.current_char();
      str.next();
      if (!is_hex_digit(c)) { return false; }
      code_point = (code_point * 16) + hex_value(c);
    }

    if (code_point >= 0xD800 && code_point <= 0xDBFF) {
      if (str.size() >= 6) {
        if (str[0] == '\\' && str[1] == 'u' && is_hex_digit(str[2]) && is_hex_digit(str[3]) &&
            is_hex_digit(str[4]) && is_hex_digit(str[5])) {
          cudf::char_utf8 low_surrogate = 0;
          for (size_t i = 0; i < 4; i++) {
            char const c  = str[i + 2];
            low_surrogate = (low_surrogate * 16) + hex_value(c);
          }
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
    // One decoded code point, including a combined surrogate pair, needs at most four UTF-8 bytes.
    char buff[4];
    cudf::size_type const bytes = from_char_utf8(utf_char, buff);
    output_size_bytes += bytes;

    // Decoded bytes are not re-escaped, even for quotes, backslashes or controls.
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

  // Number grammar: `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`; 1e01 is valid, 01 is not.
  __device__ inline void parse_number_and_set_current()
  {
    try_skip(curr_pos, '-');

    bool is_float            = false;
    int number_digits_length = 0;
    if (try_unsigned_number(is_float, number_digits_length)) {
      if (check_max_num_len(number_digits_length)) {
        current_token = (is_float ? json_token::VALUE_NUMBER_FLOAT : json_token::VALUE_NUMBER_INT);
        number_token_len = curr_pos - current_token_start_pos;
      } else {
        set_current_error();
      }
    } else {
      set_current_error();
    }
  }

  static __device__ inline bool check_max_num_len(int number_digits_length)
  {
    return max_num_len <= 0 || (max_num_len > 0 && number_digits_length <= max_num_len);
  }

  // Only sets is_float to true and adds to the digit count; callers initialize both.
  __device__ inline bool try_unsigned_number(bool& is_float, int& number_digits_length)
  {
    if (!eof()) {
      char const c = chars[curr_pos];
      if (c >= '1' && c <= '9') {
        curr_pos++;
        number_digits_length++;
        number_digits_length += skip_zero_or_more_digits();
        return parse_number_from_fraction(is_float, number_digits_length);
      } else if (c == '0') {
        curr_pos++;
        number_digits_length++;

        if (!eof()) {
          char const next_char_after_zero = chars[curr_pos];
          if (next_char_after_zero >= '0' && next_char_after_zero <= '9') { return false; }
        }

        return parse_number_from_fraction(is_float, number_digits_length);
      } else {
        return false;
      }
    } else {
      return false;
    }
  }

  __device__ inline bool parse_number_from_fraction(bool& is_float, int& number_digits_length)
  {
    if (try_skip(curr_pos, '.')) {
      is_float = true;
      if (!try_skip_one_or_more_digits(number_digits_length)) { return false; }
    }

    if (!eof() && (chars[curr_pos] == 'e' || chars[curr_pos] == 'E')) {
      curr_pos++;
      is_float = true;
      return try_parse_exp(number_digits_length);
    }

    return true;
  }

  __device__ inline int skip_zero_or_more_digits()
  {
    int digits = 0;
    while (!eof()) {
      if (is_digit(chars[curr_pos])) {
        digits++;
        curr_pos++;
      } else {
        break;
      }
    }
    return digits;
  }

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

  // The exponent marker has already been consumed.
  __device__ inline bool try_parse_exp(int& number_digits_length)
  {
    if (!eof() && (chars[curr_pos] == '+' || chars[curr_pos] == '-')) { curr_pos++; }

    return try_skip_one_or_more_digits(number_digits_length);
  }

  __device__ inline void parse_true_and_set_current()
  {
    if (try_skip(curr_pos, 'r') && try_skip(curr_pos, 'u') && try_skip(curr_pos, 'e')) {
      current_token = json_token::VALUE_TRUE;
    } else {
      set_current_error();
    }
  }

  __device__ inline void parse_false_and_set_current()
  {
    if (try_skip(curr_pos, 'a') && try_skip(curr_pos, 'l') && try_skip(curr_pos, 's') &&
        try_skip(curr_pos, 'e')) {
      current_token = json_token::VALUE_FALSE;
    } else {
      set_current_error();
    }
  }

  __device__ inline void parse_null_and_set_current()
  {
    if (try_skip(curr_pos, 'u') && try_skip(curr_pos, 'l') && try_skip(curr_pos, 'l')) {
      current_token = json_token::VALUE_NULL;
    } else {
      set_current_error();
    }
  }

  __device__ inline void parse_field_name_and_set_current(
    bool& matched_field_name, char_range to_match_field_name = char_range::null())
  {
    current_token_start_pos = curr_pos;
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

  // Inspecting callers zero separator flags; this only sets them true. Matching is FIELD_NAME-only.
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
        if (current_token == json_token::INIT) {
          parse_first_token_in_value_and_set_current();
        } else {
          // Trailing input after the root value is ignored.
          current_token = json_token::SUCCESS;
        }
      } else {
        if (is_object_context()) {
          if (current_token == json_token::START_OBJECT) {
            if (c == '}') {
              current_token_start_pos = curr_pos;
              curr_pos++;
              pop_curr_context();
              current_token = json_token::END_OBJECT;
            } else {
              parse_field_name_and_set_current(matched_field_name, to_match_field_name);
            }
          } else if (current_token == json_token::FIELD_NAME) {
            if (c == ':') {
              has_colon_before_token = true;
              curr_pos++;
              skip_whitespaces();
              parse_first_token_in_value_and_set_current();
            } else {
              set_current_error();
            }
          } else {
            if (c == '}') {
              current_token_start_pos = curr_pos;
              curr_pos++;
              pop_curr_context();
              current_token = json_token::END_OBJECT;
            } else if (c == ',') {
              has_comma_before_token = true;
              curr_pos++;
              skip_whitespaces();
              parse_field_name_and_set_current(matched_field_name, to_match_field_name);
            } else {
              set_current_error();
            }
          }
        } else {
          if (current_token == json_token::START_ARRAY) {
            if (c == ']') {
              current_token_start_pos = curr_pos;
              curr_pos++;
              pop_curr_context();
              current_token = json_token::END_ARRAY;
            } else {
              parse_first_token_in_value_and_set_current();
            }
          } else {
            if (c == ',') {
              has_comma_before_token = true;
              curr_pos++;
              skip_whitespaces();
              parse_first_token_in_value_and_set_current();
            } else if (c == ']') {
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
      if (is_context_stack_empty() && current_token != json_token::INIT) {
        current_token = json_token::SUCCESS;
      } else {
        set_current_error();
      }
    }
  }

 public:
  // ERROR and SUCCESS end the token stream even before input ends; SUCCESS ignores trailing bytes.
  __device__ json_token next_token()
  {
    // These write-only outputs are ignored.
    bool has_comma_before_token;
    bool has_colon_before_token;
    bool matched_field_name;
    parse_next_token_and_set_current(
      has_comma_before_token, has_colon_before_token, matched_field_name);
    return current_token;
  }

  // The match result is valid only for FIELD_NAME.
  __device__ bool parse_next_token_with_matching(cudf::string_view to_match_field_name)
  {
    // These write-only separator outputs are ignored.
    bool has_comma_before_token;
    bool has_colon_before_token;
    bool matched_field_name;
    parse_next_token_and_set_current(has_comma_before_token,
                                     has_colon_before_token,
                                     matched_field_name,
                                     char_range{to_match_field_name});
    return matched_field_name;
  }

  __device__ json_token get_current_token() const { return current_token; }

  // Raw string and field-name ranges include both quotes and retain escapes.
  __device__ inline char_range current_range() const
  {
    return chars.slice(current_token_start_pos, curr_pos - current_token_start_pos);
  }

  /**
   * Opening object/array tokens advance to their matching end; other tokens stay in place.
   * INIT, ERROR, SUCCESS or malformed children return false.
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
   * A null destination only counts bytes; strings are decoded without surrounding quotes.
   * E.g. \u4E2D\u56FD becomes six bytes: E4 B8 AD E5 9B BD; quoted output adds two bytes.
   * Integer -0 becomes 0; floats normalize through double, e.g. 200.000 becomes 200.0.
   */
  __device__ cudf::size_type write_unescaped_text(char* destination) const
  {
    switch (current_token) {
      case json_token::VALUE_STRING: {
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
        // Floats can expand: 1E308 becomes 1.0E308; 0.003 stays 0.003, but 0.0003 becomes 3.0E-4.
        // Trailing zeros drop: 351.980 becomes 351.98; 0.0000000000003 becomes 3.0E-13.
        // Large finite values also normalize: 12345678900000000000.0 becomes 1.23456789E19.
        // Even raw output quotes overflow: 1.0E309 becomes "Infinity", -1E309 becomes "-Infinity".
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
      case json_token::SUCCESS:
      case json_token::ERROR:
      case json_token::INIT: return 0;
    }
    return 0;
  }

  __device__ cudf::size_type compute_escaped_len() const { return write_escaped_text(nullptr); }

  /**
   * A null destination only counts bytes; numbers normalize exactly as in unescaped output.
   * A lone double quote becomes four bytes (quote, backslash, quote, quote).
   * \u4E2D\u56FD becomes eight bytes: two quotes around E4 B8 AD E5 9B BD.
   */
  __device__ cudf::size_type write_escaped_text(char* destination) const
  {
    switch (current_token) {
      case json_token::VALUE_STRING: {
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
      case json_token::SUCCESS:
      case json_token::ERROR:
      case json_token::INIT: return 0;
    }
    return 0;
  }

  /**
   * Serializes one value with normalized tokens and separators, not a verbatim source copy.
   * A null destination counts bytes; containers advance to their matching close token.
   * Returns {false, sum_copy_len} when the copy itself fails midway through a container — the
   * count covers the bytes written before the error and must reach the out-of-bound check —
   * and {false, 0} for invalid tokens or malformed children; partial writes are invalid.
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
        if (nullptr != copy_to) {
          size_t copy_len = write_escaped_text(copy_to);
          return {true, copy_len};
        } else {
          size_t copy_len = compute_escaped_len();
          return {true, copy_len};
        }
      case json_token::START_ARRAY:
      case json_token::START_OBJECT:
        // The matching end token lowers stack_size by one from this saved depth.
        size_t sum_copy_len   = 0;
        int backup_stack_size = stack_size;

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

          bool matched_field_name;  // ignored
          parse_next_token_and_set_current(
            has_comma_before_token, has_colon_before_token, matched_field_name);

          // check the JSON format
          if (current_token == json_token::ERROR) {
            // The bytes copied before the error are real; report them.
            return {false, sum_copy_len};
          }

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

          if (backup_stack_size - 1 == stack_size) { return {true, sum_copy_len}; }
        }
        return {false, 0};
    }

    return {false, 0};
  }

  __device__ inline bool max_nesting_depth_exceeded() const { return max_depth_exceeded; }

 private:
  char_range const chars;
  cudf::size_type curr_pos;
  // One bit per nesting level: 1 for object, 0 for array.
  int64_t context_stack;
  int stack_size = 0;
  cudf::size_type current_token_start_pos;
  cudf::size_type number_token_len;

  json_token current_token;

  bool max_depth_exceeded;
};

}  // namespace spark_rapids_jni
