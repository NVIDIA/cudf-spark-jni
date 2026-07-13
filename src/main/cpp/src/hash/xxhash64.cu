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

#include "hash.cuh"
#include "hash.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/std/utility>
#include <thrust/tabulate.h>

namespace spark_rapids_jni {

namespace {

using hash_value_type = int64_t;
using half_size_type  = int32_t;

constexpr __device__ inline int64_t rotate_bits_left_signed(hash_value_type h, int8_t r)
{ return (h << r) | (h >> (64 - r)) & ~(-1 << r); }

template <typename Key>
struct XXHash_64 {
  using result_type = hash_value_type;

  constexpr XXHash_64() = delete;
  __host__ __device__ constexpr XXHash_64(hash_value_type seed) : m_seed(seed) {}

  template <typename T>
  __device__ inline T getblock32(cuda::std::byte const* data, cudf::size_type offset) const
  {
    // Read a 4-byte value from the data pointer as individual bytes for safe
    // unaligned access (very likely for string types).
    auto block      = reinterpret_cast<std::uint8_t const*>(data + offset);
    uint32_t result = static_cast<uint32_t>(block[0]) | (static_cast<uint32_t>(block[1]) << 8) |
                      (static_cast<uint32_t>(block[2]) << 16) |
                      (static_cast<uint32_t>(block[3]) << 24);
    return reinterpret_cast<T const*>(&result)[0];
  }

  __device__ inline hash_value_type getblock64(cuda::std::byte const* data,
                                               cudf::size_type offset) const
  {
    uint64_t result = static_cast<uint64_t>(getblock32<uint32_t>(data, offset)) |
                      static_cast<uint64_t>(getblock32<uint32_t>(data, offset + 4)) << 32;
    return reinterpret_cast<hash_value_type const*>(&result)[0];
  }

  result_type __device__ inline operator()(Key const& key) const { return compute(key); }

  template <typename T>
  result_type __device__ inline compute(T const& key) const
  { return compute_bytes(reinterpret_cast<cuda::std::byte const*>(&key), sizeof(T)); }

  result_type __device__ inline compute_remaining_bytes(cuda::std::byte const* data,
                                                        cudf::size_type const nbytes,
                                                        cudf::size_type offset,
                                                        result_type h64) const
  {
    // remaining data can be processed in 8-byte chunks
    if ((nbytes % 32) >= 8) {
      for (; offset <= nbytes - 8; offset += 8) {
        hash_value_type k1 = getblock64(data, offset) * prime2;
        k1                 = rotate_bits_left_signed(k1, 31) * prime1;
        h64 ^= k1;
        h64 = rotate_bits_left_signed(h64, 27) * prime1 + prime4;
      }
    }

    // remaining data can be processed in 4-byte chunks
    if (((nbytes % 32) % 8) >= 4) {
      for (; offset <= nbytes - 4; offset += 4) {
        h64 ^= (getblock32<half_size_type>(data, offset) & 0xffffffffL) * prime1;
        h64 = rotate_bits_left_signed(h64, 23) * prime2 + prime3;
      }
    }

    // and the rest
    if (nbytes % 4) {
      while (offset < nbytes) {
        h64 ^= (static_cast<uint8_t>(data[offset]) & 0xff) * prime5;
        h64 = rotate_bits_left_signed(h64, 11) * prime1;
        ++offset;
      }
    }
    return h64;
  }

  result_type __device__ compute_bytes(cuda::std::byte const* data,
                                       cudf::size_type const nbytes) const
  {
    uint64_t offset = 0;
    hash_value_type h64;
    // data can be processed in 32-byte chunks
    if (nbytes >= 32) {
      auto limit         = nbytes - 32;
      hash_value_type v1 = m_seed + prime1 + prime2;
      hash_value_type v2 = m_seed + prime2;
      hash_value_type v3 = m_seed;
      hash_value_type v4 = m_seed - prime1;

      do {
        // pipeline 4*8byte computations
        v1 += getblock64(data, offset) * prime2;
        v1 = rotate_bits_left_signed(v1, 31);
        v1 *= prime1;
        offset += 8;
        v2 += getblock64(data, offset) * prime2;
        v2 = rotate_bits_left_signed(v2, 31);
        v2 *= prime1;
        offset += 8;
        v3 += getblock64(data, offset) * prime2;
        v3 = rotate_bits_left_signed(v3, 31);
        v3 *= prime1;
        offset += 8;
        v4 += getblock64(data, offset) * prime2;
        v4 = rotate_bits_left_signed(v4, 31);
        v4 *= prime1;
        offset += 8;
      } while (offset <= limit);

      h64 = rotate_bits_left_signed(v1, 1) + rotate_bits_left_signed(v2, 7) +
            rotate_bits_left_signed(v3, 12) + rotate_bits_left_signed(v4, 18);

      v1 *= prime2;
      v1 = rotate_bits_left_signed(v1, 31);
      v1 *= prime1;
      h64 ^= v1;
      h64 = h64 * prime1 + prime4;

      v2 *= prime2;
      v2 = rotate_bits_left_signed(v2, 31);
      v2 *= prime1;
      h64 ^= v2;
      h64 = h64 * prime1 + prime4;

      v3 *= prime2;
      v3 = rotate_bits_left_signed(v3, 31);
      v3 *= prime1;
      h64 ^= v3;
      h64 = h64 * prime1 + prime4;

      v4 *= prime2;
      v4 = rotate_bits_left_signed(v4, 31);
      v4 *= prime1;
      h64 ^= v4;
      h64 = h64 * prime1 + prime4;
    } else {
      h64 = m_seed + prime5;
    }

    h64 += nbytes;
    h64 = compute_remaining_bytes(data, nbytes, offset, h64);

    return finalize(h64);
  }

  constexpr __host__ __device__ hash_value_type finalize(hash_value_type h) const noexcept
  {
    h ^= static_cast<hash_value_type>(static_cast<uint64_t>(h) >> 33);
    h *= prime2;
    h ^= static_cast<hash_value_type>(static_cast<uint64_t>(h) >> 29);
    h *= prime3;
    h ^= static_cast<hash_value_type>(static_cast<uint64_t>(h) >> 32);
    return h;
  }

 private:
  hash_value_type m_seed{};

  static constexpr hash_value_type prime1 = 0x9E3779B185EBCA87L;
  static constexpr hash_value_type prime2 = 0xC2B2AE3D27D4EB4FL;
  static constexpr hash_value_type prime3 = 0x165667B19E3779F9L;
  static constexpr hash_value_type prime4 = 0x85EBCA77C2B2AE63L;
  static constexpr hash_value_type prime5 = 0x27D4EB2F165667C5L;
};

template <>
hash_value_type __device__ inline XXHash_64<bool>::operator()(bool const& key) const
{ return compute<uint32_t>(key); }

template <>
hash_value_type __device__ inline XXHash_64<int8_t>::operator()(int8_t const& key) const
{ return compute<uint32_t>(key); }

template <>
hash_value_type __device__ inline XXHash_64<uint8_t>::operator()(uint8_t const& key) const
{ return compute<uint32_t>(key); }

template <>
hash_value_type __device__ inline XXHash_64<int16_t>::operator()(int16_t const& key) const
{ return compute<uint32_t>(key); }

template <>
hash_value_type __device__ inline XXHash_64<uint16_t>::operator()(uint16_t const& key) const
{ return compute<uint32_t>(key); }

template <>
hash_value_type __device__ inline XXHash_64<float>::operator()(float const& key) const
{ return compute<float>(spark_rapids_jni::normalize_nans_and_zeros(key)); }

template <>
hash_value_type __device__ inline XXHash_64<double>::operator()(double const& key) const
{ return compute<double>(spark_rapids_jni::normalize_nans_and_zeros(key)); }

template <>
hash_value_type __device__ inline XXHash_64<cudf::string_view>::operator()(
  cudf::string_view const& key) const
{
  auto const data = reinterpret_cast<cuda::std::byte const*>(key.data());
  auto const len  = key.size_bytes();
  return compute_bytes(data, len);
}

template <>
hash_value_type __device__ inline XXHash_64<numeric::decimal32>::operator()(
  numeric::decimal32 const& key) const
{ return compute<uint64_t>(key.value()); }

template <>
hash_value_type __device__ inline XXHash_64<numeric::decimal64>::operator()(
  numeric::decimal64 const& key) const
{ return compute<uint64_t>(key.value()); }

template <>
hash_value_type __device__ inline XXHash_64<numeric::decimal128>::operator()(
  numeric::decimal128 const& key) const
{
  auto [java_d, length] = to_java_bigdecimal(key);
  auto bytes            = reinterpret_cast<cuda::std::byte*>(&java_d);
  return compute_bytes(bytes, length);
}

/**
 * @brief Computes the hash value of a row in the given table.
 *
 * This functor uses Spark conventions for xxhash64 hashing, which differs from
 * the xxhash64 implementation used in the rest of libcudf. These differences
 * include:
 * - Serially using the output hash as an input seed for the next item
 * - Ignorance of null values
 *
 * The serial use of hashes as seeds means that data of different nested types
 * can exhibit hash collisions. For example, a row of an integer column
 * containing a 1 will have the same hash as a lists column of integers
 * containing a list of [1] and a struct column of a single integer column
 * containing a struct of {1}.
 *
 * As a consequence of ignoring null values, inputs like [1], [1, null], and
 * [null, 1] have the same hash (an expected hash collision). This kind of
 * collision can also occur across a table of nullable columns and with nulls
 * in structs ({1, null} and {null, 1} have the same hash). The seed value (the
 * previous element's hash value) is returned as the hash if an element is
 * null.
 *
 * For additional differences such as special tail processing and decimal type
 * handling, refer to the SparkXXHash64 functor.
 *
 * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
 */
template <typename Nullate>
class device_row_hasher {
 public:
  device_row_hasher(Nullate nulls, cudf::table_device_view const& t, hash_value_type seed)
    : _check_nulls(nulls), _table(t), _seed(seed)
  {
  }

  __device__ auto operator()(cudf::size_type row_index) const noexcept
  {
    auto result = _seed;
    auto itr    = _table.begin();

    auto op = [row_index, nulls = _check_nulls](auto hash, auto column) -> hash_value_type {
      return cudf::type_dispatcher(
        column.type(), element_hasher_adapter{}, column, row_index, nulls, hash);
    };
    for (; itr != _table.end(); ++itr) {
      result = op(result, *itr);
    }
    return result;
  }

  /**
   * @brief Computes the hash value of an element in the given column.
   *
   * When the column is non-nested, this is a simple wrapper around the element_hasher.
   * When the column is nested, this uses a seed value to serially compute each
   * nested element, with the output hash becoming the seed for the next value.
   * This requires constructing a new hash functor for each nested element,
   * using the new seed from the previous element's hash. The hash of a null
   * element is the input seed (the previous element's hash).
   */
  class element_hasher_adapter {
   public:
    class element_hasher {
     private:
      Nullate _check_nulls;
      hash_value_type _seed;

     public:
      __device__ element_hasher(Nullate check_nulls, hash_value_type seed)
        : _check_nulls(check_nulls), _seed(seed)
      {
      }

      template <typename T, CUDF_ENABLE_IF(cudf::column_device_view::has_element_accessor<T>())>
      __device__ hash_value_type operator()(cudf::column_device_view const& col,
                                            cudf::size_type row_index) const noexcept
      {
        if (_check_nulls && col.is_null(row_index)) { return _seed; }
        return XXHash_64<T>{_seed}(col.element<T>(row_index));
      }

      template <typename T, CUDF_ENABLE_IF(not cudf::column_device_view::has_element_accessor<T>())>
      __device__ hash_value_type operator()(cudf::column_device_view const&,
                                            cudf::size_type) const noexcept
      { CUDF_UNREACHABLE("Unsupported type for xxhash64"); }
    };

    template <typename T, CUDF_ENABLE_IF(not cudf::is_nested<T>())>
    __device__ hash_value_type operator()(cudf::column_device_view const& col,
                                          cudf::size_type row_index,
                                          Nullate const _check_nulls,
                                          hash_value_type const _seed) const noexcept
    {
      auto const hasher = element_hasher{_check_nulls, _seed};
      return hasher.template operator()<T>(col, row_index);
    }

    struct col_stack_frame {
     private:
      cudf::column_device_view _column;  // the column to process
      int _idx_to_process;               // the index of child or element to process next

     public:
      __device__ col_stack_frame() =
        delete;  // Because the default constructor of `cudf::column_device_view` is deleted

      __device__ col_stack_frame(cudf::column_device_view col)
        : _column(cuda::std::move(col)), _idx_to_process(0)
      {
      }

      __device__ int get_and_inc_idx_to_process() { return _idx_to_process++; }

      __device__ int get_idx_to_process() { return _idx_to_process; }

      __device__ cudf::column_device_view get_column() { return _column; }
    };

    /**
     * @brief Functor to compute hash value for nested columns.
     *
     * This functor uses a stack to process nested columns. It iterates through the nested columns
     * in a depth-first manner. The stack is used to keep track of the nested columns that need to
     * be processed.
     *
     * - If the current column is a list column, it replaces the list column with its most inner
     *   non-list child since null values can be ignored in the xxhash64 computation.
     * - If the current column is a struct column, there are two cases:
     *    a. If the struct column has only one row, it would be treated as a struct element. The
     *      children of the struct element would be pushed into the stack.
     *    b. If the struct column has multiple rows, it would be treated as a struct column. The
     *      next struct element would be pushed into the stack.
     * - If the current column is a primitive column, it computes the hash value.
     *
     * For example, consider that the input column is of type `List<Struct<int, float>>`.
     * Assume that the element at `row_index` is: [(1, 2.0), (3, 4.0)].
     * The sliced column is noted as L1 here.
     *
     *            L1            List<Struct<int, float>>
     *            |
     *            S1            Struct<int, float>  ----> `struct_column` with multiple rows
     *         /     \
     *       S1[0]  S1[1]       Struct<int, float>  ----> `struct_element` with single row
     *      /  \    /  \
     *     i1  f1  i2  f2       Primitive columns
     *
     * List level L1:
     * |Index|List<Struct<int, float>> |
     * |-----|-------------------------|
     * |0    |  [(1, 2.0), (3, 4.0)]   |
     * length: 1
     * Offsets: 0, 2
     *
     * Struct level S1:
     * |Index|Struct<int, float>|
     * |-----|------------------|
     * |0    |  (1, 2.0)        |
     * |1    |  (3, 4.0)        |
     * length: 2
     *
     * @tparam T Type of the column.
     * @param col The column to hash.
     * @param row_index The index of the row to hash.
     * @param _check_nulls A flag to indicate whether to check for null values.
     * @param _seed The initial seed value for the hash computation.
     * @return The computed hash value.
     *
     * @note This function is only enabled for nested columns.
     */
    template <typename T, CUDF_ENABLE_IF(cudf::is_nested<T>())>
    __device__ hash_value_type operator()(cudf::column_device_view const& col,
                                          cudf::size_type row_index,
                                          Nullate const _check_nulls,
                                          hash_value_type const _seed) const noexcept
    {
      hash_value_type ret               = _seed;
      cudf::column_device_view curr_col = col.slice(row_index, 1);
      // The default constructor of `col_stack_frame` is deleted, so it can not allocate an array
      // of `col_stack_frame` directly.
      // Instead leverage the byte array to create the col_stack_frame array.
      alignas(col_stack_frame) char stack_wrapper[sizeof(col_stack_frame) * MAX_STACK_DEPTH];
      auto col_stack = reinterpret_cast<col_stack_frame*>(stack_wrapper);
      int stack_size = 0;

      col_stack[stack_size++] = col_stack_frame(curr_col);

      while (stack_size > 0) {
        col_stack_frame& top = col_stack[stack_size - 1];
        curr_col             = top.get_column();
        // Replace list column with its most inner non-list child
        if (curr_col.type().id() == cudf::type_id::LIST) {
          do {
            curr_col = cudf::lists_column_device_view(curr_col).get_sliced_child();
          } while (curr_col.type().id() == cudf::type_id::LIST);
          col_stack[stack_size - 1] = col_stack_frame(curr_col);
          continue;
        }

        if (curr_col.type().id() == cudf::type_id::STRUCT) {
          if (curr_col.size() <= 1) {  // struct element
            // All child columns processed, pop the element
            if (top.get_idx_to_process() == curr_col.num_child_columns()) {
              --stack_size;
            } else {
              // Push the next child column into the stack
              col_stack[stack_size++] =
                col_stack_frame(cudf::structs_column_device_view(curr_col).get_sliced_child(
                  top.get_and_inc_idx_to_process()));
            }
          } else {  // struct column
            if (top.get_idx_to_process() == curr_col.size()) {
              --stack_size;
            } else {
              col_stack[stack_size++] =
                col_stack_frame(curr_col.slice(top.get_and_inc_idx_to_process(), 1));
            }
          }
        } else {  // Primitive column
          auto op = [curr_col, _check_nulls](auto hash, auto element_index) {
            return cudf::type_dispatcher<cudf::detail::dispatch_void_if_nested>(
              curr_col.type(), element_hasher{_check_nulls, hash}, curr_col, element_index);
          };
          for (auto i = 0; i < curr_col.size(); ++i) {
            ret = op(ret, i);
          }
          --stack_size;
        }
      }
      return ret;
    }
  };

  Nullate const _check_nulls;
  cudf::table_device_view const _table;
  hash_value_type const _seed;
};

void check_nested_depth(cudf::table_view const& input)
{
  using column_checker_fn_t = std::function<int(cudf::column_view const&)>;

  column_checker_fn_t get_nested_depth = [&](cudf::column_view const& col) {
    if (col.type().id() == cudf::type_id::LIST) {
      auto const child_col = cudf::lists_column_view(col).child();
      // When encountering a List of Struct column, we need to account for an extra depth,
      // as both the struct column and its elements will be pushed into the stack.
      if (child_col.type().id() == cudf::type_id::STRUCT) {
        return 1 + get_nested_depth(child_col);
      }
      return get_nested_depth(child_col);
    } else if (col.type().id() == cudf::type_id::STRUCT) {
      int max_child_depth = 0;
      for (auto child = col.child_begin(); child != col.child_end(); ++child) {
        max_child_depth = std::max(max_child_depth, get_nested_depth(*child));
      }
      return 1 + max_child_depth;
    } else {  // Primitive type
      return 1;
    }
  };

  for (auto i = 0; i < input.num_columns(); i++) {
    cudf::column_view const& col = input.column(i);
    CUDF_EXPECTS(get_nested_depth(col) <= MAX_STACK_DEPTH,
                 "The " + std::to_string(i) +
                   "-th column exceeds the maximum allowed nested depth. " +
                   "Current depth: " + std::to_string(get_nested_depth(col)) + ", " +
                   "Maximum allowed depth: " + std::to_string(MAX_STACK_DEPTH));
  }
}

}  // namespace

std::unique_ptr<cudf::column> xxhash64(cudf::table_view const& input,
                                       int64_t _seed,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  hash_value_type seed = static_cast<hash_value_type>(_seed);

  auto output = cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<hash_value_type>()),
                                          input.num_rows(),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);

  // Return early if there's nothing to hash
  if (input.num_columns() == 0 || input.num_rows() == 0) { return output; }

  check_nested_depth(input);

  bool const nullable   = has_nested_nulls(input);
  auto const input_view = cudf::table_device_view::create(input, stream);
  auto output_view      = output->mutable_view();

  // Compute the hash value for each row
  thrust::tabulate(rmm::exec_policy(stream),
                   output_view.begin<hash_value_type>(),
                   output_view.end<hash_value_type>(),
                   device_row_hasher(nullable, *input_view, seed));

  return output;
}

}  // namespace spark_rapids_jni
