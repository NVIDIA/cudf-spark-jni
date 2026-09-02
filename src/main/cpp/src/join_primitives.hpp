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

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <memory>
#include <utility>

namespace spark_rapids_jni {

/**
 * @file
 * @brief Join primitive operations for composable join implementations
 *
 * This file provides low-level join primitives that can be composed to build
 * various join operations (inner, outer, semi, anti, hash, sort-merge, etc).
 * These primitives allow for flexible join strategies and optimization at higher levels.
 */

// =============================================================================
// BASIC EQUALITY JOINS (Sort-Merge and Hash)
// =============================================================================

/**
 * @brief Perform an inner join using sort-merge algorithm
 *
 * Returns gather maps for matching rows. Does not optimize by swapping tables
 * based on size - that optimization should be done at a higher level if desired.
 *
 * @param left_keys The left table for equality comparison
 * @param right_keys The right table for equality comparison
 * @param is_left_sorted Whether the left table is pre-sorted
 * @param is_right_sorted Whether the right table is pre-sorted
 * @param compare_nulls Whether null values in equality keys join to each other
 * @param stream CUDA stream for device operations
 * @param mr Device memory resource for allocation
 *
 * @return Pair of device vectors [left_indices, right_indices] for matching rows
 */
std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
sort_merge_inner_join(cudf::table_view const& left_keys,
                      cudf::table_view const& right_keys,
                      cudf::sorted is_left_sorted       = cudf::sorted::NO,
                      cudf::sorted is_right_sorted      = cudf::sorted::NO,
                      cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
                      cuda::stream_ref stream           = cudf::get_default_stream(),
                      rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Perform an inner join using hash join algorithm
 *
 * Returns gather maps for matching rows. Does not optimize by swapping tables
 * based on size - that optimization should be done at a higher level if desired.
 *
 * @param left_keys The left table for equality comparison
 * @param right_keys The right table for equality comparison
 * @param compare_nulls Whether null values in equality keys join to each other
 * @param stream CUDA stream for device operations
 * @param mr Device memory resource for allocation
 *
 * @return Pair of device vectors [left_indices, right_indices] for matching rows
 */
std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
hash_inner_join(cudf::table_view const& left_keys,
                cudf::table_view const& right_keys,
                cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
                cuda::stream_ref stream           = cudf::get_default_stream(),
                rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

// =============================================================================
// AST FILTERING
// =============================================================================

/**
 * @brief Filter gather maps using an AST conditional expression
 *
 * Takes existing gather maps and filters them by evaluating the AST expression
 * on the corresponding rows from the left and right tables. Only pairs where
 * the expression evaluates to true are kept.
 *
 * @param left_indices Input gather map for left table
 * @param right_indices Input gather map for right table (must be same size as left_indices)
 * @param left_table The left table for conditional expression evaluation
 * @param right_table The right table for conditional expression evaluation
 * @param binary_predicate The AST expression to evaluate (must return boolean)
 * @param stream CUDA stream for device operations
 * @param mr Device memory resource for allocation
 *
 * @return Pair of filtered device vectors [left_indices, right_indices]
 */
std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
filter_gather_maps_by_ast(
  cudf::device_span<cudf::size_type const> left_indices,
  cudf::device_span<cudf::size_type const> right_indices,
  cudf::table_view const& left_table,
  cudf::table_view const& right_table,
  cudf::ast::expression const& binary_predicate,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

// =============================================================================
// MAKE OUTER JOINS
// =============================================================================

/**
 * @brief Convert inner join gather maps to left outer join gather maps
 *
 * Takes gather maps from an inner join and adds entries for unmatched left rows.
 * Unmatched left rows will have right indices set to INT32_MIN (sentinel value for null).
 *
 * @param left_indices Inner join gather map for left table
 * @param right_indices Inner join gather map for right table (must be same size as left_indices)
 * @param left_table_size Number of rows in the left table
 * @param right_table_size Number of rows in the right table
 * @param stream CUDA stream for device operations
 * @param mr Device memory resource for allocation
 *
 * @return Pair of device vectors [left_indices, right_indices] for left outer join
 */
std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
make_left_outer(cudf::device_span<cudf::size_type const> left_indices,
                cudf::device_span<cudf::size_type const> right_indices,
                cudf::size_type left_table_size,
                cudf::size_type right_table_size,
                cuda::stream_ref stream           = cudf::get_default_stream(),
                rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Convert inner join gather maps to full outer join gather maps
 *
 * Takes gather maps from an inner join and adds entries for both unmatched left and right rows.
 * Unmatched left rows will have right indices set to INT32_MIN (sentinel value for null).
 * Unmatched right rows will have left indices set to INT32_MIN (sentinel value for null).
 *
 * @param left_indices Inner join gather map for left table
 * @param right_indices Inner join gather map for right table (must be same size as left_indices)
 * @param left_table_size Number of rows in the left table
 * @param right_table_size Number of rows in the right table
 * @param stream CUDA stream for device operations
 * @param mr Device memory resource for allocation
 *
 * @return Pair of device vectors [left_indices, right_indices] for full outer join
 */
std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
make_full_outer(cudf::device_span<cudf::size_type const> left_indices,
                cudf::device_span<cudf::size_type const> right_indices,
                cudf::size_type left_table_size,
                cudf::size_type right_table_size,
                cuda::stream_ref stream           = cudf::get_default_stream(),
                rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

// =============================================================================
// MAKE SEMI/ANTI JOINS
// =============================================================================

/**
 * @brief Convert inner join gather maps to semi join result
 *
 * Takes the left gather map from an inner join and returns unique left indices.
 * Each left row appears at most once in the result. The right gather map is not needed
 * since semi join only cares about which left rows have matches, not the actual matches.
 *
 * @param indices Inner join gather map
 * @param table_size Number of rows in the table that the indices are from
 * @param stream CUDA stream for device operations
 * @param mr Device memory resource for allocation
 *
 * @return Device vector of unique left indices
 */
rmm::device_uvector<cudf::size_type> make_semi(
  cudf::device_span<cudf::size_type const> indices,
  cudf::size_type table_size,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Convert semi join result to anti join result
 *
 * Takes a gather map of matched left indices and returns indices of unmatched left rows.
 * This is the complement of the semi join.
 *
 * @param indices Semi join result (gather map of matched indices)
 * @param table_size Number of rows in the table that the indices are from
 * @param stream CUDA stream for device operations
 * @param mr Device memory resource for allocation
 *
 * @return Device vector of unmatched left indices
 */
rmm::device_uvector<cudf::size_type> make_anti(
  cudf::device_span<cudf::size_type const> indices,
  cudf::size_type table_size,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

// =============================================================================
// PARTITIONED JOIN SUPPORT
// =============================================================================

/**
 * @brief Get boolean column indicating which rows were matched
 *
 * For partitioned joins, returns a boolean column where true indicates the row
 * at that index was matched. This allows combining results from multiple partitions
 * by OR-ing the boolean columns together.
 *
 * @param gather_map Gather map from a join operation
 * @param table_size Total number of rows in the source table
 * @param stream CUDA stream for device operations
 * @param mr Device memory resource for allocation
 *
 * @return Boolean column where true indicates row was matched
 */
std::unique_ptr<cudf::column> get_matched_rows(
  cudf::device_span<cudf::size_type const> gather_map,
  cudf::size_type table_size,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
