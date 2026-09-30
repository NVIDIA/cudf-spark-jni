/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "aggregation64_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/stream>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace {

// Functor to reassemble a 64-bit value from two 64-bit chunks with overflow detection.
class chunk_assembler_64bit {
 public:
  chunk_assembler_64bit(bool* overflows, uint64_t const* chunks0, int64_t const* chunks1)
    : overflows(overflows), chunks0(chunks0), chunks1(chunks1)
  {
  }

  __device__ int64_t operator()(cudf::size_type i) const
  {
    // Starting with the least significant input and moving to the most significant, propagate the
    // upper 32-bits of the previous column into the next column, i.e.: propagate the "carry" bits
    // of each 64-bit chunk into the next chunk.
    uint64_t const c0    = chunks0[i];
    int64_t const c1     = chunks1[i] + (c0 >> 32);
    int64_t const result = (c1 << 32) | static_cast<uint32_t>(c0);

    // check for overflow by ensuring the sign bit matches the top carry bits
    int32_t const replicated_sign_bit = static_cast<int32_t>(c1) >> 31;
    int32_t const top_carry_bits      = static_cast<int32_t>(c1 >> 32);
    overflows[i]                      = (replicated_sign_bit != top_carry_bits);

    return result;
  }

 private:
  // output column for overflow detected
  bool* const overflows;

  // input columns for the two 64-bit values
  uint64_t const* const chunks0;
  int64_t const* const chunks1;
};

}  // anonymous namespace

namespace spark_rapids_jni {

// Extract a 32-bit chunk from a 64-bit value.
std::unique_ptr<cudf::column> extract_chunk32_from_64bit(cudf::column_view const& in_col,
                                                         cudf::data_type type,
                                                         int chunk_idx,
                                                         cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    in_col.type().id() == cudf::type_id::INT64 || in_col.type().id() == cudf::type_id::UINT64,
    "Input column must be a 64-bit type (INT64 or UINT64).");
  CUDF_EXPECTS(chunk_idx >= 0 && chunk_idx < 2,
               "Invalid chunk index. Must be 0 (lower 32-bits) or 1 (upper 32-bits).");
  CUDF_EXPECTS(type.id() == cudf::type_id::INT32 || type.id() == cudf::type_id::UINT32,
               "Output type must be a 32-bit integer type (INT32 or UINT32).");

  auto const num_rows = in_col.size();
  auto out_col        = cudf::make_fixed_width_column(
    type, num_rows, cudf::copy_bitmask(in_col, stream, mr), in_col.null_count(), stream, mr);
  auto out_view = out_col->mutable_view();

  if (chunk_idx == 0) {  // Extract lower 32 bits
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      in_col.begin<uint64_t>(),
                      in_col.end<uint64_t>(),
                      out_view.data<uint32_t>(),
                      [] __device__(uint64_t val) { return static_cast<uint32_t>(val); });
  } else {  // Extract upper 32 bits
    // Cast to int32_t for the upper chunk to correctly handle signedness during potential future
    // aggregation.
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      in_col.begin<uint64_t>(),
                      in_col.end<uint64_t>(),
                      out_view.data<int32_t>(),
                      [] __device__(uint64_t val) { return static_cast<int32_t>(val >> 32); });
  }
  return out_col;
}

// Reassemble a column of 64-bit values from two 64-bit integer columns with overflow detection.
std::unique_ptr<cudf::table> assemble64_from_sum(cudf::table_view const& chunks_table,
                                                 cudf::data_type output_type,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    output_type.id() == cudf::type_id::INT64 || output_type.id() == cudf::type_id::UINT64,
    "Output type must be a 64-bit integer type (INT64 or UINT64).");
  CUDF_EXPECTS(chunks_table.num_columns() == 2, "Input table must contain exactly 2 columns.");

  auto const num_rows = chunks_table.num_rows();
  auto const chunks0  = chunks_table.column(0);
  auto const chunks1  = chunks_table.column(1);

  CUDF_EXPECTS(cudf::size_of(chunks0.type()) == 8 && chunks1.type().id() == cudf::type_id::INT64,
               "Input chunk columns must be 64-bit types.");

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                                  num_rows,
                                                  cudf::copy_bitmask(chunks0, stream, mr),
                                                  chunks0.null_count(),
                                                  stream,
                                                  mr));
  columns.push_back(cudf::make_fixed_width_column(output_type,
                                                  num_rows,
                                                  cudf::copy_bitmask(chunks0, stream, mr),
                                                  chunks0.null_count(),
                                                  stream,
                                                  mr));
  auto overflows_view = columns[0]->mutable_view();
  auto assembled_view = columns[1]->mutable_view();
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(num_rows),
    assembled_view.begin<int64_t>(),
    chunk_assembler_64bit(
      overflows_view.begin<bool>(), chunks0.begin<uint64_t>(), chunks1.begin<int64_t>()));

  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace spark_rapids_jni
