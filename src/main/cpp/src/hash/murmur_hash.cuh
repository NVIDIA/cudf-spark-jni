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

#include "hash.cuh"

#include <cudf/hashing.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/types.hpp>

namespace spark_rapids_jni {

using murmur_hash_value_type = int32_t;

__device__ inline uint32_t rotate_bits_left(uint32_t x, uint32_t r)
{
  // This function is equivalent to (x << r) | (x >> (32 - r))
  return __funnelshift_l(x, x, r);
}

template <typename Key, CUDF_ENABLE_IF(not cudf::is_nested<Key>())>
struct MurmurHash3_32 {
  using result_type = murmur_hash_value_type;

  __host__ __device__ constexpr MurmurHash3_32() = delete;
  __host__ __device__ constexpr MurmurHash3_32(uint32_t seed) : m_seed(seed) {}

  [[nodiscard]] __device__ inline uint32_t fmix32(uint32_t h) const
  {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

  [[nodiscard]] __device__ inline uint32_t getblock32(cuda::std::byte const* data,
                                                      cudf::size_type offset) const
  {
    // Read a 4-byte value from the data pointer as individual bytes for safe
    // unaligned access (very likely for string types).
    auto block = reinterpret_cast<uint8_t const*>(data + offset);
    return block[0] | (block[1] << 8) | (block[2] << 16) | (block[3] << 24);
  }

  [[nodiscard]] result_type __device__ inline operator()(Key const& key) const
  { return compute(key); }

  template <typename T>
  result_type __device__ inline compute(T const& key) const
  { return compute_bytes(reinterpret_cast<cuda::std::byte const*>(&key), sizeof(T)); }

  result_type __device__ inline compute_remaining_bytes(cuda::std::byte const* data,
                                                        cudf::size_type len,
                                                        cudf::size_type tail_offset,
                                                        result_type h) const
  {
    // Process remaining bytes that do not fill a four-byte chunk using Spark's approach
    // (does not conform to normal MurmurHash3).
    for (auto i = tail_offset; i < len; i++) {
      // We require a two-step cast to get the k1 value from the byte. First,
      // we must cast to a signed int8_t. Then, the sign bit is preserved when
      // casting to uint32_t under 2's complement. Java preserves the sign when
      // casting byte-to-int, but C++ does not.
      uint32_t k1 = static_cast<uint32_t>(cuda::std::to_integer<int8_t>(data[i]));
      k1 *= c1;
      k1 = spark_rapids_jni::rotate_bits_left(k1, rot_c1);
      k1 *= c2;
      h ^= k1;
      h = spark_rapids_jni::rotate_bits_left(h, rot_c2);
      h = h * 5 + c3;
    }
    return h;
  }

  result_type __device__ compute_bytes(cuda::std::byte const* data, cudf::size_type const len) const
  {
    constexpr cudf::size_type BLOCK_SIZE = 4;
    cudf::size_type const nblocks        = len / BLOCK_SIZE;
    cudf::size_type const tail_offset    = nblocks * BLOCK_SIZE;
    result_type h                        = m_seed;

    // Process all four-byte chunks.
    for (cudf::size_type i = 0; i < nblocks; i++) {
      uint32_t k1 = getblock32(data, i * BLOCK_SIZE);
      k1 *= c1;
      k1 = spark_rapids_jni::rotate_bits_left(k1, rot_c1);
      k1 *= c2;
      h ^= k1;
      h = spark_rapids_jni::rotate_bits_left(h, rot_c2);
      h = h * 5 + c3;
    }

    h = compute_remaining_bytes(data, len, tail_offset, h);

    // Finalize hash.
    h ^= len;
    h = fmix32(h);
    return h;
  }

 private:
  uint32_t m_seed{cudf::DEFAULT_HASH_SEED};
  static constexpr uint32_t c1     = 0xcc9e2d51;
  static constexpr uint32_t c2     = 0x1b873593;
  static constexpr uint32_t c3     = 0xe6546b64;
  static constexpr uint32_t rot_c1 = 15;
  static constexpr uint32_t rot_c2 = 13;
};

template <>
murmur_hash_value_type __device__ inline MurmurHash3_32<bool>::operator()(bool const& key) const
{ return compute<uint32_t>(key); }

template <>
murmur_hash_value_type __device__ inline MurmurHash3_32<int8_t>::operator()(int8_t const& key) const
{ return compute<uint32_t>(key); }

template <>
murmur_hash_value_type __device__ inline MurmurHash3_32<uint8_t>::operator()(
  uint8_t const& key) const
{ return compute<uint32_t>(key); }

template <>
murmur_hash_value_type __device__ inline MurmurHash3_32<int16_t>::operator()(
  int16_t const& key) const
{ return compute<uint32_t>(key); }

template <>
murmur_hash_value_type __device__ inline MurmurHash3_32<uint16_t>::operator()(
  uint16_t const& key) const
{ return compute<uint32_t>(key); }

template <>
murmur_hash_value_type __device__ inline MurmurHash3_32<float>::operator()(float const& key) const
{ return compute<float>(spark_rapids_jni::normalize_nans(key)); }

template <>
murmur_hash_value_type __device__ inline MurmurHash3_32<double>::operator()(double const& key) const
{ return compute<double>(spark_rapids_jni::normalize_nans(key)); }

template <>
murmur_hash_value_type __device__ inline MurmurHash3_32<cudf::string_view>::operator()(
  cudf::string_view const& key) const
{
  auto const data = reinterpret_cast<cuda::std::byte const*>(key.data());
  auto const len  = key.size_bytes();
  return compute_bytes(data, len);
}

template <>
murmur_hash_value_type __device__ inline MurmurHash3_32<numeric::decimal32>::operator()(
  numeric::decimal32 const& key) const
{ return compute<uint64_t>(key.value()); }

template <>
murmur_hash_value_type __device__ inline MurmurHash3_32<numeric::decimal64>::operator()(
  numeric::decimal64 const& key) const
{ return compute<uint64_t>(key.value()); }

template <>
murmur_hash_value_type __device__ inline MurmurHash3_32<numeric::decimal128>::operator()(
  numeric::decimal128 const& key) const
{
  auto [java_d, length] = to_java_bigdecimal(key);
  auto bytes            = reinterpret_cast<cuda::std::byte*>(&java_d);
  return compute_bytes(bytes, length);
}

}  // namespace spark_rapids_jni
