/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

#include <cudf/utilities/traits.hpp>

#include <cuda/std/type_traits>

namespace spark_rapids_jni {

namespace detail {

/**
 * @brief Functor for computing size of data elements for a given cudf type.
 *
 * Note: columns types which themselves inherently have no data (strings, lists,
 * structs) return 0.
 */
struct size_of_helper {
  template <typename T>
  constexpr cuda::std::
    enable_if_t<!cudf::is_fixed_width<T>() && !cuda::std::is_same_v<T, cudf::string_view>, size_t>
    operator()() const
  {
    return 0;
  }

  template <typename T>
  constexpr cuda::std::
    enable_if_t<!cudf::is_fixed_width<T>() && cuda::std::is_same_v<T, cudf::string_view>, size_t>
    operator()() const
  {
    return sizeof(cudf::device_storage_type_t<int8_t>);
  }

  template <typename T>
  constexpr cuda::std::enable_if_t<cudf::is_fixed_width<T>(), size_t> __device__
  operator()() const noexcept
  {
    return sizeof(cudf::device_storage_type_t<T>);
  }
};

/**
 * @brief Header for each partition.
 *
 * The values are stored in big-endian format.
 */
struct partition_header {
  uint32_t magic_number;
  uint32_t row_index;  // row index in the source table that this partition started at
  uint32_t num_rows;
  uint32_t validity_size;
  uint32_t offset_size;
  uint32_t total_size;  // size of validity+offset+data sections
  uint32_t num_flattened_columns;
};

// alignment values for each validity type, as applied at the end of that data type
// in each partition. so for example all of the grouped-together validity buffers for
// a given partition will have a final 4 byte alignment applied before the offset buffers begin
constexpr size_t validity_pad = 4;
constexpr size_t offset_pad   = 4;
constexpr size_t data_pad     = 4;

/**
 * @brief Compute per-partition metadata size.
 */
constexpr size_t compute_per_partition_metadata_size(size_t total_columns)
{
  auto const has_validity_length = (total_columns + 7) / 8;  // has-validity bit per column
  return sizeof(partition_header) + has_validity_length;
}

// align all column size allocations to this boundary so that all output column buffers
// start at that alignment.
static constexpr std::size_t split_align = 64;

/**
 * @brief Buffer type enum
 *
 * Note: these values matter. Don't rearrange them.
 */
enum class buffer_type { VALIDITY = 0, OFFSETS = 1, DATA = 2 };

}  // namespace detail

}  // namespace spark_rapids_jni
