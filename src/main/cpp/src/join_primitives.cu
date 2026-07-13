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

#include "join_primitives.hpp"
#include "nvtx_ranges.hpp"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/sort_merge_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <limits>

namespace spark_rapids_jni {

namespace {

// Type alias for intermediate storage used in AST expression evaluation.
template <bool has_nulls>
using intermediate_storage_type = cudf::ast::detail::IntermediateDataType<has_nulls>;

template <bool has_nulls>
__global__ void filter_join_indices_kernel(
  cudf::size_type const* __restrict__ left_indices,
  cudf::size_type const* __restrict__ right_indices,
  cudf::size_type num_pairs,
  cudf::table_device_view left_table,
  cudf::table_device_view right_table,
  cudf::ast::detail::expression_device_view device_expression_data,
  bool* __restrict__ keep_mask)
{
  extern __shared__ char raw_intermediate_storage[];
  auto intermediate_storage =
    reinterpret_cast<intermediate_storage_type<has_nulls>*>(raw_intermediate_storage);
  auto thread_intermediate_storage =
    &intermediate_storage[threadIdx.x * device_expression_data.num_intermediates];

  auto evaluator = cudf::ast::detail::expression_evaluator<has_nulls>(
    left_table, right_table, device_expression_data);

  auto const idx    = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();

  for (cudf::thread_index_type i = idx; i < num_pairs; i += stride) {
    auto output_dest                = cudf::ast::detail::value_expression_result<bool, has_nulls>();
    cudf::size_type const left_idx  = left_indices[i];
    cudf::size_type const right_idx = right_indices[i];

    evaluator.evaluate(output_dest, left_idx, right_idx, 0, thread_intermediate_storage);

    keep_mask[i] = output_dest.is_valid() && output_dest.value();
  }
}

template <bool has_nulls>
std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
filter_by_conditional_impl(cudf::device_span<cudf::size_type const> left_indices,
                           cudf::device_span<cudf::size_type const> right_indices,
                           cudf::table_device_view const& left_table,
                           cudf::table_device_view const& right_table,
                           cudf::ast::detail::expression_device_view device_expression_data,
                           cudf::size_type num_intermediates,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  auto const num_pairs = left_indices.size();
  if (num_pairs == 0) {
    return {rmm::device_uvector<cudf::size_type>(0, stream, mr),
            rmm::device_uvector<cudf::size_type>(0, stream, mr)};
  }

  // Create a boolean mask for indices to keep
  // Note: Temporary buffers use current device resource for allocation,
  // while output results use the caller-provided memory resource (mr).
  auto keep_mask =
    rmm::device_uvector<bool>(num_pairs, stream, cudf::get_current_device_resource_ref());

  // Launch kernel to evaluate expression for each pair with dynamic shared memory sizing
  int current_device = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&current_device));
  int max_shmem_per_block = 0;
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(
    &max_shmem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, current_device));

  auto const per_thread_bytes = static_cast<int>(num_intermediates) *
                                static_cast<int>(sizeof(intermediate_storage_type<has_nulls>));
  int block_size              = 256;  // default
  if (per_thread_bytes > 0) {
    int const max_by_shmem = max_shmem_per_block / per_thread_bytes;
    if (max_by_shmem > 0) {
      // Prefer multiples of warp size (32)
      int const rounded = (max_by_shmem / 32) * 32;
      block_size        = std::max(32, std::min(block_size, rounded > 0 ? rounded : max_by_shmem));
    } else {
      block_size = 32;  // minimal reasonable size
    }
  }
  cudf::size_type const grid_size = (num_pairs + block_size - 1) / block_size;
  auto const shmem_size = static_cast<size_t>(block_size) * static_cast<size_t>(per_thread_bytes);

  filter_join_indices_kernel<has_nulls>
    <<<grid_size, block_size, shmem_size, stream.value()>>>(left_indices.data(),
                                                            right_indices.data(),
                                                            num_pairs,
                                                            left_table,
                                                            right_table,
                                                            device_expression_data,
                                                            keep_mask.data());

  // Surface any kernel launch errors immediately
  CUDF_CUDA_TRY(cudaPeekAtLastError());

  // Count the number of true values in the mask
  auto const num_matches =
    thrust::count(rmm::exec_policy(stream), keep_mask.begin(), keep_mask.end(), true);

  // Allocate output vectors
  auto out_left_indices  = rmm::device_uvector<cudf::size_type>(num_matches, stream, mr);
  auto out_right_indices = rmm::device_uvector<cudf::size_type>(num_matches, stream, mr);

  // Copy indices where condition is true
  auto input_iter  = thrust::make_zip_iterator(left_indices.begin(), right_indices.begin());
  auto output_iter = thrust::make_zip_iterator(out_left_indices.begin(), out_right_indices.begin());

  thrust::copy_if(rmm::exec_policy_nosync(stream),
                  input_iter,
                  input_iter + num_pairs,
                  keep_mask.begin(),
                  output_iter,
                  cuda::std::identity{});

  return {std::move(out_left_indices), std::move(out_right_indices)};
}

}  // anonymous namespace

// =============================================================================
// BASIC EQUALITY JOINS (Sort-Merge and Hash)
// =============================================================================

std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
sort_merge_inner_join(cudf::table_view const& left_keys,
                      cudf::table_view const& right_keys,
                      cudf::sorted is_left_sorted,
                      cudf::sorted is_right_sorted,
                      cudf::null_equality compare_nulls,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  // Validate inputs
  CUDF_EXPECTS(left_keys.num_columns() > 0, "Left keys table must have at least one column");
  CUDF_EXPECTS(right_keys.num_columns() > 0, "Right keys table must have at least one column");

  // Handle empty table cases
  if (left_keys.num_rows() == 0 || right_keys.num_rows() == 0) {
    return {rmm::device_uvector<cudf::size_type>(0, stream, mr),
            rmm::device_uvector<cudf::size_type>(0, stream, mr)};
  }

  // Perform sort-merge inner join
  cudf::sort_merge_join join_obj(right_keys, is_right_sorted, compare_nulls, stream);
  auto [left_result, right_result] = join_obj.inner_join(left_keys, is_left_sorted, stream, mr);
  return {std::move(*left_result), std::move(*right_result)};
}

std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
hash_inner_join(cudf::table_view const& left_keys,
                cudf::table_view const& right_keys,
                cudf::null_equality compare_nulls,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  // Validate inputs
  CUDF_EXPECTS(left_keys.num_columns() > 0, "Left keys table must have at least one column");
  CUDF_EXPECTS(right_keys.num_columns() > 0, "Right keys table must have at least one column");

  // Handle empty table cases
  if (left_keys.num_rows() == 0 || right_keys.num_rows() == 0) {
    return {rmm::device_uvector<cudf::size_type>(0, stream, mr),
            rmm::device_uvector<cudf::size_type>(0, stream, mr)};
  }

  // Build hash join on right table, probe with left table
  // This returns (left_indices, right_indices) in the correct order
  auto hash_join                   = cudf::hash_join(right_keys, compare_nulls, stream);
  auto [left_result, right_result] = hash_join.inner_join(left_keys, std::nullopt, stream, mr);
  return {std::move(*left_result), std::move(*right_result)};
}

// =============================================================================
// AST FILTERING
// =============================================================================

std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
filter_gather_maps_by_ast(cudf::device_span<cudf::size_type const> left_indices,
                          cudf::device_span<cudf::size_type const> right_indices,
                          cudf::table_view const& left_table,
                          cudf::table_view const& right_table,
                          cudf::ast::expression const& binary_predicate,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  CUDF_EXPECTS(left_indices.size() == right_indices.size(),
               "Left and right gather maps must have the same size");

  if (left_indices.size() == 0) {
    return {rmm::device_uvector<cudf::size_type>(0, stream, mr),
            rmm::device_uvector<cudf::size_type>(0, stream, mr)};
  }

  // Check for nulls in tables
  auto const has_nulls = cudf::has_nested_nulls(left_table) || cudf::has_nested_nulls(right_table);

  // Parse the AST expression
  auto const parser = cudf::ast::detail::expression_parser{binary_predicate,
                                                           left_table,
                                                           right_table,
                                                           has_nulls,
                                                           stream,
                                                           cudf::get_current_device_resource_ref()};
  CUDF_EXPECTS(parser.output_type().id() == cudf::type_id::BOOL8,
               "The expression must produce a boolean output.");

  // Create device views of tables
  auto left_table_view  = cudf::table_device_view::create(left_table, stream);
  auto right_table_view = cudf::table_device_view::create(right_table, stream);

  // Filter by conditional expression
  if (has_nulls) {
    return filter_by_conditional_impl<true>(left_indices,
                                            right_indices,
                                            *left_table_view,
                                            *right_table_view,
                                            parser.device_expression_data,
                                            parser.device_expression_data.num_intermediates,
                                            stream,
                                            mr);
  } else {
    return filter_by_conditional_impl<false>(left_indices,
                                             right_indices,
                                             *left_table_view,
                                             *right_table_view,
                                             parser.device_expression_data,
                                             parser.device_expression_data.num_intermediates,
                                             stream,
                                             mr);
  }
}

// =============================================================================
// MAKE OUTER JOINS
// =============================================================================

namespace {

// Helper function to process one side of a join:
// - Allocates hash_match buffer
// - Populates it with for_each
// - Counts unmatched rows
// Returns: (hash_match buffer, num_unmatched count)
std::pair<rmm::device_uvector<bool>, cudf::size_type> compute_side_match_info(
  cudf::device_span<cudf::size_type const> indices,
  cudf::size_type table_size,
  rmm::cuda_stream_view stream)
{
  // Create a boolean mask to track which rows have matches
  // Note: Temporary buffers use current device resource for allocation
  auto has_match =
    rmm::device_uvector<bool>(table_size, stream, cudf::get_current_device_resource_ref());
  CUDF_CUDA_TRY(cudaMemsetAsync(has_match.data(), 0, has_match.size(), stream.value()));

  // Mark rows that have matches
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   indices.begin(),
                   indices.end(),
                   [has_match = has_match.data(), table_size] __device__(cudf::size_type idx) {
                     if (idx < 0 || idx >= table_size) { return; }
                     has_match[idx] = true;
                   });

  // Count unmatched rows
  auto const num_unmatched =
    thrust::count(rmm::exec_policy_nosync(stream), has_match.begin(), has_match.end(), false);

  return {std::move(has_match), num_unmatched};
}

// Helper function to populate result with unmatched rows
// Copies unmatched rows and fills corresponding other indices with sentinel values
void populate_outer_result(cudf::size_type table_size,
                           rmm::device_uvector<bool> const& has_match,
                           cudf::size_type num_unmatched,
                           cudf::size_type unmatched_offset,
                           rmm::device_uvector<cudf::size_type>& out_indices,
                           rmm::device_uvector<cudf::size_type>& out_other_indices,
                           rmm::cuda_stream_view stream)
{
  // Copy unmatched rows
  auto unmatched_iter = out_indices.begin() + unmatched_offset;
  thrust::copy_if(rmm::exec_policy_nosync(stream),
                  thrust::counting_iterator<cudf::size_type>(0),
                  thrust::counting_iterator<cudf::size_type>(table_size),
                  has_match.begin(),
                  unmatched_iter,
                  cuda::std::logical_not<bool>());

  // Fill corresponding other indices with sentinel values (null markers)
  // Use INT32_MIN as the sentinel value for unmatched rows
  thrust::fill(rmm::exec_policy_nosync(stream),
               out_other_indices.begin() + unmatched_offset,
               out_other_indices.begin() + unmatched_offset + num_unmatched,
               std::numeric_limits<cudf::size_type>::min());
}

}  // anonymous namespace

std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
make_left_outer(cudf::device_span<cudf::size_type const> left_indices,
                cudf::device_span<cudf::size_type const> right_indices,
                cudf::size_type left_table_size,
                cudf::size_type right_table_size,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  CUDF_EXPECTS(left_indices.size() == right_indices.size(),
               "Left and right gather maps must have the same size");
  CUDF_EXPECTS(left_table_size >= 0 && right_table_size >= 0, "Table sizes must be non-negative");

  // Process left side to get match info
  auto [left_has_match, num_left_unmatched] =
    compute_side_match_info(left_indices, left_table_size, stream);

  // Allocate output with space for both matched and unmatched
  auto const total_size  = left_indices.size() + num_left_unmatched;
  auto out_left_indices  = rmm::device_uvector<cudf::size_type>(total_size, stream, mr);
  auto out_right_indices = rmm::device_uvector<cudf::size_type>(total_size, stream, mr);

  // Copy matched pairs
  thrust::copy(rmm::exec_policy_nosync(stream),
               left_indices.begin(),
               left_indices.end(),
               out_left_indices.begin());
  thrust::copy(rmm::exec_policy_nosync(stream),
               right_indices.begin(),
               right_indices.end(),
               out_right_indices.begin());

  // Add unmatched left rows with null right indices
  populate_outer_result(left_table_size,
                        left_has_match,
                        num_left_unmatched,
                        left_indices.size(),
                        out_left_indices,
                        out_right_indices,
                        stream);

  return {std::move(out_left_indices), std::move(out_right_indices)};
}

std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
make_full_outer(cudf::device_span<cudf::size_type const> left_indices,
                cudf::device_span<cudf::size_type const> right_indices,
                cudf::size_type left_table_size,
                cudf::size_type right_table_size,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  CUDF_EXPECTS(left_indices.size() == right_indices.size(),
               "Left and right gather maps must have the same size");
  CUDF_EXPECTS(left_table_size >= 0 && right_table_size >= 0, "Table sizes must be non-negative");

  // Process both sides to get match info
  auto [left_has_match, num_left_unmatched] =
    compute_side_match_info(left_indices, left_table_size, stream);
  auto [right_has_match, num_right_unmatched] =
    compute_side_match_info(right_indices, right_table_size, stream);

  // Allocate output with space for matched and all unmatched
  auto const total_size  = left_indices.size() + num_left_unmatched + num_right_unmatched;
  auto out_left_indices  = rmm::device_uvector<cudf::size_type>(total_size, stream, mr);
  auto out_right_indices = rmm::device_uvector<cudf::size_type>(total_size, stream, mr);

  // Copy matched pairs
  thrust::copy(rmm::exec_policy_nosync(stream),
               left_indices.begin(),
               left_indices.end(),
               out_left_indices.begin());
  thrust::copy(rmm::exec_policy_nosync(stream),
               right_indices.begin(),
               right_indices.end(),
               out_right_indices.begin());

  auto offset = left_indices.size();

  // Add unmatched left rows
  populate_outer_result(left_table_size,
                        left_has_match,
                        num_left_unmatched,
                        offset,
                        out_left_indices,
                        out_right_indices,
                        stream);

  offset += num_left_unmatched;

  // Add unmatched right rows
  populate_outer_result(right_table_size,
                        right_has_match,
                        num_right_unmatched,
                        offset,
                        out_right_indices,
                        out_left_indices,
                        stream);

  return {std::move(out_left_indices), std::move(out_right_indices)};
}

// =============================================================================
// MAKE SEMI/ANTI JOINS
// =============================================================================

rmm::device_uvector<cudf::size_type> make_semi(
  cudf::device_span<cudf::size_type const> left_indices,
  cudf::size_type left_table_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  // Create a boolean mask for all left rows (default-initialized to false)
  auto left_has_match =
    rmm::device_uvector<bool>(left_table_size, stream, cudf::get_current_device_resource_ref());
  thrust::fill(
    rmm::exec_policy_nosync(stream), left_has_match.begin(), left_has_match.end(), false);

  // Mark left rows that have matches
  thrust::for_each(
    rmm::exec_policy_nosync(stream),
    left_indices.begin(),
    left_indices.end(),
    [left_has_match = left_has_match.data(), left_table_size] __device__(cudf::size_type idx) {
      if (idx < 0 || idx >= left_table_size) { return; }
      left_has_match[idx] = true;
    });

  // Count and collect matched rows
  auto const num_matched = thrust::count(
    rmm::exec_policy_nosync(stream), left_has_match.begin(), left_has_match.end(), true);

  auto result = rmm::device_uvector<cudf::size_type>(num_matched, stream, mr);
  thrust::copy_if(rmm::exec_policy_nosync(stream),
                  thrust::counting_iterator<cudf::size_type>(0),
                  thrust::counting_iterator<cudf::size_type>(left_table_size),
                  left_has_match.begin(),
                  result.begin(),
                  cuda::std::identity());

  return result;
}

rmm::device_uvector<cudf::size_type> make_anti(
  cudf::device_span<cudf::size_type const> left_indices,
  cudf::size_type left_table_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  // Create a boolean mask for all left rows (default-initialized to false)
  auto left_has_match =
    rmm::device_uvector<bool>(left_table_size, stream, cudf::get_current_device_resource_ref());
  thrust::fill(
    rmm::exec_policy_nosync(stream), left_has_match.begin(), left_has_match.end(), false);

  // Mark left rows that have matches
  thrust::for_each(
    rmm::exec_policy_nosync(stream),
    left_indices.begin(),
    left_indices.end(),
    [left_has_match = left_has_match.data(), left_table_size] __device__(cudf::size_type idx) {
      if (idx < 0 || idx >= left_table_size) { return; }
      left_has_match[idx] = true;
    });

  // Count and collect unmatched rows
  auto const num_unmatched = thrust::count(
    rmm::exec_policy_nosync(stream), left_has_match.begin(), left_has_match.end(), false);

  auto result = rmm::device_uvector<cudf::size_type>(num_unmatched, stream, mr);
  thrust::copy_if(rmm::exec_policy_nosync(stream),
                  thrust::counting_iterator<cudf::size_type>(0),
                  thrust::counting_iterator<cudf::size_type>(left_table_size),
                  left_has_match.begin(),
                  result.begin(),
                  cuda::std::logical_not<bool>());

  return result;
}

// =============================================================================
// PARTITIONED JOIN SUPPORT
// =============================================================================

std::unique_ptr<cudf::column> get_matched_rows(cudf::device_span<cudf::size_type const> gather_map,
                                               cudf::size_type table_size,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  // Create a boolean column initialized to false
  auto result = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::BOOL8}, table_size, cudf::mask_state::UNALLOCATED, stream, mr);

  auto result_view = result->mutable_view();
  auto result_data = result_view.data<bool>();

  // Initialize all to false
  CUDF_CUDA_TRY(cudaMemsetAsync(result_data, 0, table_size, stream.value()));

  // Mark rows that appear in the gather map as true
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   gather_map.begin(),
                   gather_map.end(),
                   [result_data, table_size] __device__(cudf::size_type idx) {
                     if (idx < 0 || idx >= table_size) { return; }
                     result_data[idx] = true;
                   });

  return result;
}

}  // namespace spark_rapids_jni
