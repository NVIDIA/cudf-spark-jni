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

#include "nvtx_ranges.hpp"
#include "reverse_strings.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <thrust/for_each.h>

#include <stdint.h>

#include <memory>

namespace spark_rapids_jni {
namespace detail {
namespace {

/**
 * @brief Spark `UTF8String.numBytesForFirstByte` character width.
 *
 * Continuation bytes and UTF-8-disallowed lead bytes are treated as width 1, matching
 * Spark's `bytesOfCodePointInUTF8` table with the `(numBytes == 0) ? 1 : numBytes` rule.
 */
__device__ __forceinline__ cudf::size_type spark_num_bytes_for_first_byte(uint8_t byte)
{
  if (byte <= 0x7F) { return 1; }
  // 0x80-0xBF continuation and 0xC0-0xC1 disallowed -> Spark width 1
  if (byte <= 0xC1) { return 1; }
  if (byte <= 0xDF) { return 2; }
  if (byte <= 0xEF) { return 3; }
  // 0xF0-0xF4 valid 4-byte leads; 0xF5-0xFF disallowed -> Spark width 1
  if (byte <= 0xF4) { return 4; }
  return 1;
}

/**
 * @brief Reverse characters in each string with Spark clamp semantics (SPARK-57507).
 */
struct reverse_characters_fn {
  cudf::column_device_view const d_strings;
  cudf::detail::input_offsetalator d_offsets;
  char* d_chars;

  __device__ void operator()(cudf::size_type idx) const
  {
    if (d_strings.is_null(idx)) { return; }
    auto const d_str  = d_strings.element<cudf::string_view>(idx);
    auto const nbytes = d_str.size_bytes();
    if (nbytes == 0) { return; }

    auto const* in = reinterpret_cast<uint8_t const*>(d_str.data());
    // Write character chunks from the end of this row's output region, matching Spark:
    // keep bytes within a character in order, reverse the order of characters, and clamp
    // each character width to the bytes remaining in this row.
    auto* out_end     = d_chars + d_offsets[idx] + nbytes;
    cudf::size_type i = 0;
    while (i < nbytes) {
      auto const declared = spark_num_bytes_for_first_byte(in[i]);
      auto const len      = cuda::std::min(declared, nbytes - i);
      out_end -= len;
      for (cudf::size_type j = 0; j < len; ++j) {
        out_end[j] = static_cast<char>(in[i + j]);
      }
      i += len;
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> reverse_strings(cudf::strings_column_view const& input,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(cudf::type_id::STRING); }

  // Preserve offsets/nulls; rewrite only the character bytes.
  auto result          = std::make_unique<cudf::column>(input.parent(), stream, mr);
  auto const sv        = cudf::strings_column_view(result->view());
  auto const d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(sv.offsets());
  auto* d_chars        = result->mutable_view().head<char>();

  auto const d_column = cudf::column_device_view::create(
    input.parent(), stream, cudf::get_current_device_resource_ref());
  thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<cudf::size_type>{0},
                     input.size(),
                     reverse_characters_fn{*d_column, d_offsets, d_chars});

  return result;
}

}  // namespace detail

std::unique_ptr<cudf::column> reverse_strings(cudf::strings_column_view const& input,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return detail::reverse_strings(input, stream, mr);
}

}  // namespace spark_rapids_jni
