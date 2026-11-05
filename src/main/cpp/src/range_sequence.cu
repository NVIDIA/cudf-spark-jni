/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "range_sequence.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace spark_rapids_jni {

std::unique_ptr<cudf::column> generate_range_sequence(std::span<int64_t const> starts,
                                                      std::span<cudf::size_type const> sizes,
                                                      cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  auto constexpr max_rows = std::numeric_limits<cudf::size_type>::max();
  CUDF_EXPECTS(starts.size() == sizes.size(),
               "starts and sizes must have the same length",
               std::invalid_argument);
  CUDF_EXPECTS(starts.size() <= max_rows, "Too many ranges", std::invalid_argument);
  cudf::size_type num_rows = 0;
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    CUDF_EXPECTS(sizes[i] >= 0, "Range sizes must be nonnegative", std::invalid_argument);
    CUDF_EXPECTS(sizes[i] == 0 || starts[i] <= std::numeric_limits<int64_t>::max() - (sizes[i] - 1),
                 "Range sequence overflows INT64",
                 std::invalid_argument);
    CUDF_EXPECTS(sizes[i] <= max_rows - num_rows,
                 "Range sequence exceeds column size limit",
                 std::overflow_error);
    num_rows += sizes[i];
  }
  if (num_rows == 0) { return cudf::make_empty_column(cudf::type_id::INT64); }

  auto output = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT64}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  auto const data    = output->mutable_view().data<int64_t>();
  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto const policy  = rmm::exec_policy_nosync(stream, temp_mr);
  if (starts.size() == 1) {
    thrust::sequence(policy, data, data + num_rows, starts.front());
    return output;
  }

  auto const count = static_cast<cudf::size_type>(sizes.size());
  // Pinned staging may share a pool with CPU users that do not synchronize before writing.
  auto host_starts = cudf::detail::make_pinned_vector_async<int64_t>(count, stream);
  auto host_ends   = cudf::detail::make_pinned_vector_async<cudf::size_type>(count, stream);
  std::copy(starts.begin(), starts.end(), host_starts.begin());
  std::partial_sum(sizes.begin(), sizes.end(), host_ends.begin());
  // Complete each upload before another allocation can fail and release its host source.
  auto device_starts = cudf::detail::make_device_uvector(host_starts, stream, temp_mr);
  auto device_ends   = cudf::detail::make_device_uvector(host_ends, stream, temp_mr);
  thrust::tabulate(policy,
                   data,
                   data + num_rows,
                   cuda::proclaim_return_type<int64_t>([starts = device_starts.data(),
                                                        ends   = device_ends.data(),
                                                        count] __device__(cudf::size_type row) {
                     auto const range =
                       thrust::upper_bound(thrust::seq, ends, ends + count, row) - ends;
                     return starts[range] + (row - (range == 0 ? 0 : ends[range - 1]));
                   }));
  return output;
}

}  // namespace spark_rapids_jni
