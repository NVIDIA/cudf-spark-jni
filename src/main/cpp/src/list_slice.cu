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

#include "list_slice.hpp"
#include "nvtx_ranges.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/null_mask.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/logical.h>

using namespace cudf;

namespace spark_rapids_jni {

namespace detail {

namespace {

void assert_start_is_not_zero(column_device_view const& start, rmm::cuda_stream_view stream)
{
  bool start_valid =
    thrust::all_of(rmm::exec_policy(stream),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(start.size()),
                   cuda::proclaim_return_type<bool>([start] __device__(size_type index) {
                     if (start.is_null(index)) return true;
                     return start.element<int32_t>(index) != 0;
                   }));
  CUDF_EXPECTS(start_valid, "Invalid start value: start must not be 0");
}

void assert_length_is_not_negative(column_device_view const& length, rmm::cuda_stream_view stream)
{
  bool length_valid =
    thrust::all_of(rmm::exec_policy(stream),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(length.size()),
                   cuda::proclaim_return_type<bool>([length] __device__(size_type index) {
                     if (length.is_null(index)) return true;
                     return length.element<int32_t>(index) >= 0;
                   }));
  CUDF_EXPECTS(length_valid, "Invalid length value: length must be >= 0");
}

struct int_iterator_from_scalar {
  int32_t value;
  int_iterator_from_scalar(int32_t v) : value(v) {}
  __device__ bool is_null(cudf::thread_index_type const index) const { return false; }
  __device__ int32_t operator()(cudf::thread_index_type const index) const
  {
    // ignore index, always return the same value
    return value;
  }
};

struct int_iterator_from_column {
  column_device_view cdv;
  int_iterator_from_column(column_device_view cdv) : cdv(cdv) {}
  __device__ bool is_null(cudf::thread_index_type const index) const { return cdv.is_null(index); }
  __device__ int32_t operator()(cudf::thread_index_type const index) const
  { return cdv.element<int32_t>(index); }
};

template <typename SRART_ITERATOR, typename LENGTH_ITERATOR>
CUDF_KERNEL void compute_starts_and_sizes_kernel(size_type const* offsets_of_input_lists,
                                                 size_type const num_rows,
                                                 SRART_ITERATOR const start_iterator,
                                                 LENGTH_ITERATOR const length_iterator,
                                                 size_type* d_starts,
                                                 size_type* d_sizes)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_rows) { return; }

  // If either start or length is null, produce an empty list
  if (start_iterator.is_null(tid) || length_iterator.is_null(tid)) {
    d_sizes[tid] = 0;
    return;
  }

  // The number of elements in the current row's list
  auto const length_of_list = offsets_of_input_lists[tid + 1] - offsets_of_input_lists[tid];
  auto start                = start_iterator(tid);
  auto const length         = length_iterator(tid);

  // start cannot be 0
  start = start < 0 ? length_of_list + start : start - 1;
  // If the original start is out of [-length_of_list, length_of_list], will produce an empty list
  // If `check_start_length` is false, will not check the legality of start and length.
  // If original start is 0 or length is negative, set the output size to 0 to avoid out-of-bound
  // access. The result for this row will be an empty list or null(since the output mask will be
  // reset according to the input mask, start mask and length mask)
  if (start < 0 || start >= length_of_list || length <= 0) {
    d_sizes[tid] = 0;
    return;
  }
  // The new start index will be in range [0, length_of_list)
  d_starts[tid] = start;
  // The sliced length cannot exceed the remaining elements in the list
  d_sizes[tid] = cuda::std::min(length_of_list - start, length);
}

CUDF_KERNEL void compute_gather_map(size_type const num_rows_of_input,
                                    size_type const* d_offsets,
                                    size_type const* d_starts,
                                    size_type const* output_offset,
                                    size_type* d_gather_map)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_rows_of_input) { return; }

  auto const offset      = output_offset[tid];
  auto const begin_index = d_starts[tid] + d_offsets[tid];
  auto const rows        = output_offset[tid + 1] - output_offset[tid];
  for (auto i = 0; i < rows; i++) {
    d_gather_map[offset + i] = begin_index + i;
  }
}

template <typename SRART_ITERATOR, typename LENGTH_ITERATOR>
auto generate_starts_and_sizes(size_type const* offsets_of_input_lists,
                               size_type const num_rows,
                               SRART_ITERATOR const start_iterator,
                               LENGTH_ITERATOR const length_iterator,
                               rmm::cuda_stream_view stream)
{
  auto starts              = make_numeric_column(data_type{type_id::INT32},
                                                 num_rows,
                                                 mask_state::UNALLOCATED,
                                                 stream,
                                                 cudf::get_current_device_resource_ref());
  auto sizes               = make_numeric_column(data_type{type_id::INT32},
                                                 num_rows,
                                                 mask_state::UNALLOCATED,
                                                 stream,
                                                 cudf::get_current_device_resource_ref());
  constexpr int block_size = 256;
  auto grid                = cudf::detail::grid_1d{num_rows, block_size};
  compute_starts_and_sizes_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
    offsets_of_input_lists,
    num_rows,
    start_iterator,
    length_iterator,
    starts->mutable_view().data<int32_t>(),
    sizes->mutable_view().data<int32_t>());
  return std::make_pair(std::move(starts), std::move(sizes));
}

std::unique_ptr<cudf::column> legal_list_slice(lists_column_view const& input,
                                               column_view const& starts,
                                               column_view const& sizes,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  auto const num_rows = input.size();
  // make output_offset
  auto [output_offset, num_total_elements] = cudf::detail::make_offsets_child_column(
    sizes.begin<int32_t>(), sizes.end<int32_t>(), stream, mr);

  // generate gather map
  constexpr int block_size = 256;
  auto grid                = cudf::detail::grid_1d{num_rows, block_size};
  rmm::device_uvector<int32_t> gather_map(num_total_elements, stream);
  compute_gather_map<<<grid.num_blocks, block_size, 0, stream.value()>>>(
    num_rows,
    input.offsets_begin(),
    starts.begin<int32_t>(),
    output_offset->view().begin<int32_t>(),
    gather_map.data());

  // The following code is adapted from cudf::lists::segmented_gather
  // Call gather on child of input column
  auto const col_map =
    cudf::column_view(cudf::device_span<int32_t const>(gather_map.data(), gather_map.size()));
  auto child_table = cudf::gather(table_view({input.get_sliced_child(stream)}),
                                  col_map,
                                  out_of_bounds_policy::DONT_CHECK,
                                  negative_index_policy::NOT_ALLOWED,
                                  stream,
                                  mr);

  auto child = std::move(child_table->release().front());

  // Assemble list column & return
  auto null_mask  = cudf::copy_bitmask(input.parent(), stream, mr);
  auto null_count = input.null_count();
  return make_lists_column(
    num_rows, std::move(output_offset), std::move(child), null_count, std::move(null_mask));
}

}  // namespace

std::unique_ptr<cudf::column> list_slice(lists_column_view const& input,
                                         size_type const start,
                                         size_type const length,
                                         bool check_start_length,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  if (check_start_length) {
    CUDF_EXPECTS(start != 0, "Invalid start value: start must not be 0");
    CUDF_EXPECTS(length >= 0, "Invalid length value: length must be >= 0");
  }

  auto const num_rows = input.size();
  if (num_rows == 0) { return make_empty_column(data_type{type_id::LIST}); }

  auto [starts, sizes] = generate_starts_and_sizes(input.offsets_begin(),
                                                   num_rows,
                                                   int_iterator_from_scalar(start),
                                                   int_iterator_from_scalar(length),
                                                   stream);

  return legal_list_slice(input, starts->view(), sizes->view(), stream, mr);
}

std::unique_ptr<cudf::column> list_slice(lists_column_view const& input,
                                         size_type const start,
                                         column_view const& length,
                                         bool check_start_length,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(length.type().id() == type_id::INT32,
               "Invalid length type: length must be INT32",
               cudf::data_type_error);
  CUDF_EXPECTS(input.size() == length.size(), "Input and length size mismatch");

  auto const length_cdv = column_device_view::create(length, stream);
  if (check_start_length) {
    CUDF_EXPECTS(start != 0, "Invalid start value: start must not be 0");
    assert_length_is_not_negative(*length_cdv, stream);
  }

  auto const num_rows = input.size();
  if (num_rows == 0) { return make_empty_column(data_type{type_id::LIST}); }

  auto [starts, sizes] = generate_starts_and_sizes(input.offsets_begin(),
                                                   num_rows,
                                                   int_iterator_from_scalar(start),
                                                   int_iterator_from_column(*length_cdv),
                                                   stream);
  auto [null_mask, null_count] =
    cudf::bitmask_and(table_view{{input.parent(), length}}, stream, mr);
  auto result = legal_list_slice(input, starts->view(), sizes->view(), stream, mr);
  result->set_null_mask(std::move(null_mask), null_count);
  return result;
}

std::unique_ptr<cudf::column> list_slice(lists_column_view const& input,
                                         column_view const& start,
                                         size_type const length,
                                         bool check_start_length,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(start.type().id() == type_id::INT32,
               "Invalid start type: start must be INT32",
               cudf::data_type_error);
  CUDF_EXPECTS(input.size() == start.size(), "Input and start size mismatch");

  auto const start_cdv = column_device_view::create(start, stream);
  if (check_start_length) {
    assert_start_is_not_zero(*start_cdv, stream);
    CUDF_EXPECTS(length >= 0, "Invalid length value: length must be >= 0");
  }

  auto const num_rows = input.size();
  if (num_rows == 0) { return make_empty_column(data_type{type_id::LIST}); }

  auto [starts, sizes] = generate_starts_and_sizes(input.offsets_begin(),
                                                   num_rows,
                                                   int_iterator_from_column(*start_cdv),
                                                   int_iterator_from_scalar(length),
                                                   stream);

  auto [null_mask, null_count] = cudf::bitmask_and(table_view{{input.parent(), start}}, stream, mr);
  auto result                  = legal_list_slice(input, starts->view(), sizes->view(), stream, mr);
  result->set_null_mask(std::move(null_mask), null_count);
  return result;
}

std::unique_ptr<cudf::column> list_slice(lists_column_view const& input,
                                         column_view const& start,
                                         column_view const& length,
                                         bool check_start_length,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(start.type().id() == type_id::INT32,
               "Invalid start type: start must be INT32",
               cudf::data_type_error);
  CUDF_EXPECTS(length.type().id() == type_id::INT32,
               "Invalid length type: length must be INT32",
               cudf::data_type_error);
  CUDF_EXPECTS(input.size() == start.size(), "Input and start size mismatch");
  CUDF_EXPECTS(input.size() == length.size(), "Input and length size mismatch");

  auto const start_cdv  = column_device_view::create(start, stream);
  auto const length_cdv = column_device_view::create(length, stream);

  if (check_start_length) {
    assert_start_is_not_zero(*start_cdv, stream);
    assert_length_is_not_negative(*length_cdv, stream);
  }

  auto const num_rows = input.size();
  if (num_rows == 0) { return make_empty_column(data_type{type_id::LIST}); }

  auto [starts, sizes] = generate_starts_and_sizes(input.offsets_begin(),
                                                   num_rows,
                                                   int_iterator_from_column(*start_cdv),
                                                   int_iterator_from_column(*length_cdv),
                                                   stream);

  auto [null_mask, null_count] =
    cudf::bitmask_and(table_view{{input.parent(), start, length}}, stream, mr);
  auto result = legal_list_slice(input, starts->view(), sizes->view(), stream, mr);
  result->set_null_mask(std::move(null_mask), null_count);
  return result;
}

}  // namespace detail

// external API
std::unique_ptr<cudf::column> list_slice(lists_column_view const& input,
                                         size_type const start,
                                         size_type const length,
                                         bool check_start_length,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return detail::list_slice(input, start, length, check_start_length, stream, mr);
}

std::unique_ptr<cudf::column> list_slice(lists_column_view const& input,
                                         size_type const start,
                                         column_view const& length,
                                         bool check_start_length,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return detail::list_slice(input, start, length, check_start_length, stream, mr);
}

std::unique_ptr<cudf::column> list_slice(lists_column_view const& input,
                                         column_view const& start,
                                         size_type const length,
                                         bool check_start_length,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return detail::list_slice(input, start, length, check_start_length, stream, mr);
}

std::unique_ptr<cudf::column> list_slice(lists_column_view const& input,
                                         column_view const& start,
                                         column_view const& length,
                                         bool check_start_length,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return detail::list_slice(input, start, length, check_start_length, stream, mr);
}

}  // namespace spark_rapids_jni
