/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.
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

#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <thrust/find.h>
#include <thrust/reverse.h>

namespace spark_rapids_jni {

struct java_big_decimal_bytes {
  cuda::std::array<cuda::std::byte, sizeof(__int128_t)> bytes;
  cudf::size_type length;
};

/**
 * Normalization of floating point NaNs, passthrough for all other values.
 */
template <typename T>
T __device__ inline normalize_nans(T const& key)
{
  if constexpr (cudf::is_floating_point<T>()) {
    if (cuda::std::isnan(key)) { return cuda::std::numeric_limits<T>::quiet_NaN(); }
  }
  return key;
}

/**
 * Normalization of floating point NaNs and zeros, passthrough for all other values.
 */
template <typename T>
T __device__ inline normalize_nans_and_zeros(T const& key)
{
  if constexpr (cudf::is_floating_point<T>()) {
    if (key == T{0.0}) { return T{0.0}; }
  }
  return normalize_nans(key);
}

/**
 * @brief Converts a cudf decimal128 value to a java bigdecimal value.
 *
 * @param key The cudf decimal value
 *
 * @returns The converted decimal bytes and the relevant number of bytes in the value.
 *
 */
__device__ __inline__ java_big_decimal_bytes to_java_bigdecimal(numeric::decimal128 key)
{
  // java.math.BigDecimal.valueOf(unscaled_value, _scale).unscaledValue().toByteArray()
  // https://github.com/apache/spark/blob/master/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/hash.scala#L381
  __int128_t const val               = key.value();
  constexpr cudf::size_type key_size = sizeof(__int128_t);
  auto const data = cuda::std::bit_cast<cuda::std::array<cuda::std::byte, key_size>>(val);

  // Small negative values start with 0xff..., small positive values start with 0x00...
  bool const is_negative           = val < 0;
  cuda::std::byte const zero_value = is_negative ? cuda::std::byte{0xff} : cuda::std::byte{0x00};

  // If the value can be represented with a shorter than 16-byte integer, the
  // leading bytes of the little-endian value are truncated and are not hashed.
  auto const reverse_begin = cuda::std::reverse_iterator(data.end());
  auto const reverse_end   = cuda::std::reverse_iterator(data.begin());
  auto const first_nonzero_byte =
    thrust::find_if_not(thrust::seq,
                        reverse_begin,
                        reverse_end,
                        [zero_value](cuda::std::byte const& v) { return v == zero_value; })
      .base();
  // Max handles special case of 0 and -1 which would shorten to 0 length otherwise
  cudf::size_type length = cuda::std::max(
    1, static_cast<cudf::size_type>(cuda::std::distance(data.begin(), first_nonzero_byte)));

  // Preserve the 2's complement sign bit by adding a byte back on if necessary.
  // e.g. 0x0000ff would shorten to 0x00ff. The 0x00 byte is retained to
  // preserve the sign bit, rather than leaving an "f" at the front which would
  // change the sign bit. However, 0x00007f would shorten to 0x7f. No extra byte
  // is needed because the leftmost bit matches the sign bit. Similarly for
  // negative values: 0xffff00 --> 0xff00 and 0xffff80 --> 0x80.
  if ((length < key_size) && (is_negative ^ bool(data[length - 1] & cuda::std::byte{0x80}))) {
    ++length;
  }

  // Convert to big endian by reversing the range of nonzero bytes. Only those bytes are hashed.
  cuda::std::array<cuda::std::byte, key_size> big_endian_data{};
  thrust::reverse_copy(thrust::seq, data.begin(), data.begin() + length, big_endian_data.begin());

  return {big_endian_data, length};
}
}  // namespace spark_rapids_jni
