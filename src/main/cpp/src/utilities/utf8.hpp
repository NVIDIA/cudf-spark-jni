/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cudf/types.hpp>

namespace spark_rapids_jni::util {

// Copied from cuDF strings/detail/utf8.hpp to avoid depending on its internal API.

/**
 * @brief This will return true if passed a continuation byte of a UTF-8 character.
 *
 * @param chr Any single byte from a valid UTF-8 character
 * @return true if this is not the first byte of the character
 */
CUDF_HOST_DEVICE constexpr bool is_utf8_continuation_char(unsigned char chr)
{
  // The (0xC0 & 0x80) bit pattern identifies a continuation byte of a character.
  return (chr & 0xC0) == 0x80;
}

/**
 * @brief Returns the number of bytes used to represent the provided byte.
 *
 * This could be 0 to 4 bytes. 0 is returned for intermediate bytes within a
 * single character. For example, for the two-byte 0xC3A8 single character,
 * the first byte would return 2 and the second byte would return 0.
 *
 * @param byte Byte from an encoded character.
 * @return Number of bytes.
 */
CUDF_HOST_DEVICE constexpr cudf::size_type bytes_in_utf8_byte(uint8_t byte)
{
  return 1 + static_cast<cudf::size_type>((byte & 0xF0) == 0xF0)  // 4-byte character prefix
         + static_cast<cudf::size_type>((byte & 0xE0) == 0xE0)    // 3-byte character prefix
         + static_cast<cudf::size_type>((byte & 0xC0) == 0xC0)    // 2-byte character prefix
         - static_cast<cudf::size_type>((byte & 0xC0) == 0x80);   // intermediate byte
}

/**
 * @brief Convert a char array into a char_utf8 value.
 *
 * @param str String containing encoded char bytes.
 * @param[out] character Single char_utf8 value.
 * @return The number of bytes in the character
 */
CUDF_HOST_DEVICE constexpr cudf::size_type to_char_utf8(char const* str, cudf::char_utf8& character)
{
  cudf::size_type const chr_width = bytes_in_utf8_byte(static_cast<uint8_t>(*str));

  character = static_cast<cudf::char_utf8>(*str++) & 0xFF;
  if (chr_width > 1) {
    character = character << 8;
    character |= (static_cast<cudf::char_utf8>(*str++) & 0xFF);  // << 8;
    if (chr_width > 2) {
      character = character << 8;
      character |= (static_cast<cudf::char_utf8>(*str++) & 0xFF);  // << 16;
      if (chr_width > 3) {
        character = character << 8;
        character |= (static_cast<cudf::char_utf8>(*str++) & 0xFF);  // << 24;
      }
    }
  }
  return chr_width;
}

/**
 * @brief Converts a single UTF-8 character into a code-point value that
 * can be used for lookup in the character flags or the character case tables.
 *
 * Note that this could return a 3-byte value outside the supported lookup
 * tables' ranges.
 *
 * @param utf8_char Single UTF-8 character to convert.
 * @return Code-point for the UTF-8 character.
 */
CUDF_HOST_DEVICE constexpr uint32_t utf8_to_codepoint(cudf::char_utf8 utf8_char)
{
  uint32_t unchr = 0;
  if (utf8_char < 0x0000'0080)  // single-byte pass thru
    unchr = utf8_char;
  else if (utf8_char < 0x0000'E000)  // two bytes
  {
    unchr = (utf8_char & 0x1F00) >> 2;  // shift and
    unchr |= (utf8_char & 0x003F);      // unmask
  } else if (utf8_char < 0x00F0'0000)   // three bytes
  {
    unchr = (utf8_char & 0x0F'0000) >> 4;   // get upper 4 bits
    unchr |= (utf8_char & 0x00'3F00) >> 2;  // shift and
    unchr |= (utf8_char & 0x00'003F);       // unmask
  } else if (utf8_char <= 0xF800'0000u)     // four bytes
  {
    unchr = (utf8_char & 0x0700'0000) >> 6;   // upper 3 bits
    unchr |= (utf8_char & 0x003F'0000) >> 4;  // next 6 bits
    unchr |= (utf8_char & 0x0000'3F00) >> 2;  // next 6 bits
    unchr |= (utf8_char & 0x0000'003F);       // unmask
  }
  return unchr;
}

}  // namespace spark_rapids_jni::util
