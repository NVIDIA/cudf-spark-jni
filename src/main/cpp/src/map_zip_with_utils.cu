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

#include "map_zip_with_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/lists/count_elements.hpp>
#include <cudf/lists/gather.hpp>
#include <cudf/lists/set_operations.hpp>
#include <cudf/reduction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/functional>
#include <thrust/scan.h>

using namespace cudf;

namespace spark_rapids_jni {

namespace {

/**
 * Taken from CUDF at cpp/src/lists/utilities.cu
 * @brief Generate list labels for elements in the child column of the input lists column.
 *
 * @param input The input lists column
 * @param n_elements The number of elements in the child column of the input lists column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned object
 * @return A column containing list labels corresponding to each element in the child column
 */
std::unique_ptr<column> generate_labels(
  lists_column_view const& input,
  size_type n_elements,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto labels = make_numeric_column(
    data_type(type_to_id<size_type>()), n_elements, cudf::mask_state::UNALLOCATED, stream, mr);
  auto const labels_begin = labels->mutable_view().template begin<size_type>();
  cudf::detail::label_segments(
    input.offsets_begin(), input.offsets_end(), labels_begin, labels_begin + n_elements, stream);
  return labels;
}

/**
 * @brief Find the indices of search keys within their corresponding search value lists.
 *
 * This function performs a parallel search operation where each key from search_keys
 * is compared against all values in the corresponding search_values list. The algorithm
 * works by generating all possible key-value comparisons and then finding matches.
 * When multiple matches exist for the same key, the last match is returned (following
 * Spark's behavior where repeated keys use the last key/value in the output).
 *
 * This function takes two LISTS columns:
 * 1. a `search_keys` column (keys being searched *for*)
 * 2. a `search_space` column (the space being searched)
 *
 * Neither list should have repeated keys within each list row.
 *
 * The function's output is a LISTS column, where each row[i] contains the list
 * indices of all matches between `search_keys[i]` and `search_space[i]`.
 * i.e.
 * For each row (i.e. list) in the search-space,
 * 1. If an element matches with an element in the search-keys,
 *    its list index is included in the output row.
 * 2. If there are repeated values in the search-space row,
 *    the last element's index is returned.
 * 3. If there are values that simply aren't in the search-space row,
 *    an out of bounds index is returned.
 *
 * For example, consider the following rows at index `i`:
 *
 * auto const ∅ = -OuB; // Out of Bounds index
 * search_keys[i]    == { B, C, D, Z};
 * search_space[i]   == { A, B, C, C, D, E, F };
 *
 *
 * results[i]        == { 1, 3, 4, ∅};
 *
 * The results column has as many (list) rows as search_keys or search_space.
 * Note that each result (list) row has as many indices as values in the search_keys row.
 * Also note that this function assumes that neither search_keys nor search_space
 * has repetitions in the keys.
 *
 * @param search_keys Column containing lists of keys to search for
 * @param search_values Column containing lists of values to search through (i.e. search space)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned object
 * @return A column containing the indices of matching values for each key, or -1 if no match found
 *
 * @note Both input list columns must have the same number of rows
 */
std::unique_ptr<column> indices_of(
  lists_column_view const& search_keys,    // Column containing lists of keys to search for
  lists_column_view const& search_values,  // Column containing lists of values to search through
  rmm::cuda_stream_view stream =
    cudf::get_default_stream(),  // CUDA stream for asynchronous execution
  rmm::device_async_resource_ref mr =
    cudf::get_current_device_resource_ref())  // Memory resource for allocations
{
  /*
  * Example using two list columns and indexing
  *
  * search_keys     | search_values
  * [1, 2, 3]       | [10, 2, 3]
  * [4, 5]          | [4, 14, 5, 16]
  * [6, 7, 8, 9]    | [6, 8]

  * search_keys child:      1, 2, 3, 4,  5,  6, 7, 8, 9
  * search_values child:   10, 2, 3, 4, 14, 16, 6, 8
  *
  *
  * keys_labels:                     0, 0, 0, 1, 1, 2, 2, 2, 2
  * search_key_to_num_search_values: 3, 3, 3, 4, 4, 2, 2, 2, 2
  * list_sizes_offsets:           0, 3, 6, 9, 13, 17, 19, 21, 23, 25
  * key_index:   0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8
  *
  * total_compares_per_row:    9, 8, 8
  * total_compares_offsets: 0, 9, 17, 25
  * val_offsets: 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2
  * values_idx:  0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 8, 7, 8, 7, 8, 7, 8
  *
  */
  CUDF_EXPECTS(search_keys.size() == search_values.size(),
               "Number of search keys lists must match search values lists.");

  // Get the number of lists (rows) in the input columns
  auto const num_lists = search_values.size();

  // Extract the child columns containing the actual key and value data
  auto const all_keys   = search_keys.child();
  auto const num_keys   = all_keys.size();
  auto const all_values = search_values.child();

  // Calculate the number of elements in each list for both keys and values
  auto const values_sizes = cudf::lists::count_elements(search_values, stream);
  auto const values_nulls = cudf::is_null(*values_sizes, stream);
  auto const keys_sizes   = cudf::lists::count_elements(search_keys, stream);
  auto const keys_nulls   = cudf::is_null(*keys_sizes, stream);
  // Generate labels to map each key back to its corresponding row index
  // This helps us know which list-row each key belongs to
  auto const keys_labels = generate_labels(search_keys, num_keys, stream);

  // For each key, get the size of the associated values list
  // This tells us how many values we need to compare against for each key
  auto search_key_to_num_search_values = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    cuda::proclaim_return_type<size_type>([values_sizes = values_sizes->view().begin<size_type>(),
                                           keys_labels  = keys_labels->view().begin<size_type>(),
                                           values_nulls = values_nulls->view().begin<bool>(),
                                           num_keys] __device__(auto const idx) {
      if (idx < num_keys) {
        auto keys_label = keys_labels[idx];
        return values_nulls[keys_label] ? 0 : values_sizes[keys_label];
      }
      return 0;
    }));

  // Calculate cumulative offsets for the associated list sizes
  // This gives us the starting position for each key's value comparisons
  auto values_sizes_offsets = make_numeric_column(
    data_type(type_to_id<size_type>()), num_keys + 1, cudf::mask_state::UNALLOCATED, stream);
  auto d_values_sizes_offsets = values_sizes_offsets->mutable_view().template data<size_type>();
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         search_key_to_num_search_values,
                         search_key_to_num_search_values + num_keys + 1,
                         d_values_sizes_offsets);
  // Calculate the total number of comparisons needed for each row
  // For each list row in search_keys, we need: number_of_keys * number_of_values comparisons
  auto total_compares_per_row = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    cuda::proclaim_return_type<size_type>(
      [keys_sizes   = keys_sizes->view().begin<size_type>(),
       values_sizes = values_sizes->view().begin<size_type>(),
       num_lists,
       keys_nulls   = keys_nulls->view().begin<bool>(),
       values_nulls = values_nulls->view().begin<bool>()] __device__(auto const offset_val) {
        if (offset_val >= num_lists) { return 0; }
        if (keys_nulls[offset_val] || values_nulls[offset_val]) {
          return 0;
        } else {
          return keys_sizes[offset_val] * values_sizes[offset_val];
        }
      }));
  // Sum up all comparisons across all rows
  auto const total_compares = thrust::reduce(
    rmm::exec_policy(stream), total_compares_per_row, total_compares_per_row + num_lists);

  // Calculate indices for search values
  // Calculate offsets for the total comparisons per row
  // This gives us the starting position for each row's comparisons
  auto total_compares_offsets = make_numeric_column(
    data_type(type_to_id<size_type>()), num_lists + 1, cudf::mask_state::UNALLOCATED, stream);
  auto d_total_compares_offsets = total_compares_offsets->mutable_view().template data<size_type>();
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         total_compares_per_row,
                         total_compares_per_row + num_lists + 1,
                         d_total_compares_offsets);

  // Check for any overflow in getting the total number of compares
  CUDF_EXPECTS(
    !thrust::any_of(rmm::exec_policy(stream),
                    d_total_compares_offsets,
                    d_total_compares_offsets + num_lists + 1,
                    cuda::proclaim_return_type<bool>(
                      [] __device__(auto const total_compares) { return total_compares < 0; })),
    "Input Maps are too large to process");

  // Create an index array that maps each comparison to its corresponding key
  // This tells us which key we're comparing in each comparison operation
  auto key_index = make_numeric_column(
    data_type(type_to_id<size_type>()), total_compares, cudf::mask_state::UNALLOCATED, stream);
  auto d_key_index = key_index->mutable_view().template data<size_type>();

  // Use label_segments to expand the key indices based on the list size offsets
  cudf::detail::label_segments(values_sizes_offsets->view().begin<size_type>(),
                               values_sizes_offsets->view().end<size_type>(),
                               d_key_index,
                               d_key_index + total_compares,
                               stream);

  // Create an array that maps each comparison to its corresponding value list
  // This tells us which value list we're comparing against
  auto val_offsets = make_numeric_column(
    data_type(type_to_id<size_type>()), total_compares, cudf::mask_state::UNALLOCATED, stream);
  auto d_val_offsets = val_offsets->mutable_view().template data<size_type>();
  cudf::detail::label_segments(d_total_compares_offsets,
                               d_total_compares_offsets + num_lists + 1,
                               d_val_offsets,
                               d_val_offsets + total_compares,
                               stream);

  // Calculate the index of the value within its list for each comparison
  // This gives us the specific value position we're comparing against
  auto values_idx = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    cuda::proclaim_return_type<size_type>(
      [d_key_index,
       d_values_sizes_offsets,
       d_val_offsets,
       d_values_sizes_val_offsets = search_values.offsets_begin()] __device__(auto const idx) {
        return d_values_sizes_offsets[d_key_index[idx]] > 0
                 ? idx % d_values_sizes_offsets[d_key_index[idx]] +
                     d_values_sizes_val_offsets[d_val_offsets[idx]]  // This is to add the offset of
                                                                     // the previous row size
                 : idx;
      }));
  // Use row comparator to allow nested/NULL/NaN comparisons
  auto const keys_tview   = cudf::table_view{{all_keys}};
  auto const values_tview = cudf::table_view{{all_values}};
  auto const has_nulls    = has_nested_nulls(values_tview) || has_nested_nulls(keys_tview);
  auto const comparator =
    cudf::detail::row::equality::two_table_comparator(values_tview, keys_tview, stream);
  auto const d_comp    = comparator.equal_to<false>(cudf::nullate::DYNAMIC{has_nulls});
  using lhs_index_type = cudf::detail::row::lhs_index_type;
  using rhs_index_type = cudf::detail::row::rhs_index_type;

  // Perform the actual key-value comparisons
  // For each comparison: if key == value, return the value index; otherwise return Out of Bounds
  // index which is filled in at the start
  auto results = make_numeric_column(
    data_type(type_to_id<size_type>()), num_keys, cudf::mask_state::UNALLOCATED, stream);
  thrust::fill(rmm::exec_policy(stream),
               results->mutable_view().template begin<size_type>(),
               results->mutable_view().template end<size_type>(),
               -total_compares - 1);

  // Since there are no duplicate keys, we can immediately write the found index into the output
  // as there will be only one match per map
  thrust::for_each(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(total_compares),
    [values_idx,
     d_key_index,
     d_val_offsets,
     d_comp,
     d_value_sizes_val_offsets = search_values.offsets_begin(),
     results = results->mutable_view().template begin<size_type>()] __device__(auto const idx) {
      if (d_comp(static_cast<lhs_index_type>(values_idx[idx]),
                 static_cast<rhs_index_type>(d_key_index[idx]))) {
        results[d_key_index[idx]] = values_idx[idx] - d_value_sizes_val_offsets[d_val_offsets[idx]];
      }
    });
  return results;
}
}  // namespace

std::unique_ptr<cudf::column> map_zip(
  cudf::lists_column_view const& col1,  // First map column containing key-value pairs
  cudf::lists_column_view const& col2,  // Second map column containing key-value pairs
  rmm::cuda_stream_view stream,         // CUDA stream for asynchronous execution
  rmm::device_async_resource_ref mr)    // Memory resource for allocations
{
  CUDF_EXPECTS(col1.child().type().id() == cudf::type_id::STRUCT,
               "col1 must have exactly 1 child (STRUCT) column.");
  CUDF_EXPECTS(col1.child().num_children() == 2,
               "col1 key-value struct must have exactly 2 children.");
  CUDF_EXPECTS(col2.child().type().id() == cudf::type_id::STRUCT,
               "col2 must have exactly 1 child (STRUCT) column.");
  CUDF_EXPECTS(col2.child().num_children() == 2,
               "col2 key-value struct must have exactly 2 children.");

  // Extract keys and values from the first map column (col1)
  // Create a column view that represents the keys and values part of col1
  auto const map1_keys   = column_view{data_type{type_id::LIST},
                                       col1.size(),
                                       nullptr,
                                       col1.null_mask(),
                                       col1.null_count(),
                                       col1.offset(),
                                       {col1.offsets(), col1.child().child(0)}};
  auto const map1_values = column_view{data_type{type_id::LIST},
                                       col1.size(),
                                       nullptr,
                                       col1.null_mask(),
                                       col1.null_count(),
                                       col1.offset(),
                                       {col1.offsets(), col1.child().child(1)}};

  // Extract keys and values from the second map column (col2)
  // Create a column view that represents the keys and values part of col2
  auto const map2_keys = column_view{data_type{type_id::LIST},
                                     col2.size(),
                                     nullptr,
                                     col2.null_mask(),
                                     col2.null_count(),
                                     col2.offset(),
                                     {col2.offsets(), col2.child().child(0)}};

  auto const map2_values = column_view{data_type{type_id::LIST},
                                       col2.size(),
                                       nullptr,
                                       col2.null_mask(),
                                       col2.null_count(),
                                       col2.offset(),
                                       {col2.offsets(), col2.child().child(1)}};
  // Find the union of all unique keys from both maps
  // This creates a combined set of keys that will be used for the final result
  auto const search_keys      = cudf::lists::union_distinct(cudf::lists_column_view(map1_keys),
                                                            cudf::lists_column_view(map2_keys));
  auto const search_keys_list = cudf::lists_column_view(*search_keys);

  // Find the indices of each key in the first map
  // This tells us where each key appears in the first map, or if it doesn't exist
  auto map1_indices      = indices_of(search_keys_list, cudf::lists_column_view(map1_keys), stream);
  auto map1_indices_list = make_lists_column(search_keys_list.size(),
                                             std::make_unique<column>(search_keys_list.offsets()),
                                             std::move(map1_indices),
                                             0,
                                             rmm::device_buffer{0, stream});

  // Find the indices of each key in the second map
  // This tells us where each key appears in the second map, or if it doesn't exist
  auto map2_indices      = indices_of(search_keys_list, cudf::lists_column_view(map2_keys), stream);
  auto map2_indices_list = make_lists_column(search_keys_list.size(),
                                             std::make_unique<column>(search_keys_list.offsets()),
                                             std::move(map2_indices),
                                             0,
                                             rmm::device_buffer{0, stream});

  // Gather the values from map1 and map2 using the calculated indices
  // This extracts the values corresponding to each key in the union
  auto map1_values_list_gather =
    cudf::lists::segmented_gather(cudf::lists_column_view(map1_values),
                                  cudf::lists_column_view(*map1_indices_list),
                                  out_of_bounds_policy::NULLIFY);

  auto map2_values_list_gather =
    cudf::lists::segmented_gather(cudf::lists_column_view(map2_values),
                                  cudf::lists_column_view(*map2_indices_list),
                                  out_of_bounds_policy::NULLIFY);

  std::vector<std::unique_ptr<column>> value_pair_children;
  value_pair_children.push_back(
    std::move(std::make_unique<column>(cudf::lists_column_view(*map1_values_list_gather).child())));
  value_pair_children.push_back(
    std::move(std::make_unique<column>(cudf::lists_column_view(*map2_values_list_gather).child())));

  auto value_pair = make_structs_column(cudf::lists_column_view(*search_keys).child().size(),
                                        std::move(value_pair_children),
                                        0,
                                        rmm::device_buffer{0, stream, mr},
                                        stream,
                                        mr);
  std::vector<std::unique_ptr<column>> map_structs_children;
  map_structs_children.push_back(
    std::move(std::make_unique<column>(cudf::lists_column_view(*search_keys).child())));
  map_structs_children.push_back(std::move(value_pair));

  auto map_structs = make_structs_column(cudf::lists_column_view(*search_keys).child().size(),
                                         std::move(map_structs_children),
                                         0,
                                         rmm::device_buffer{0, stream, mr},
                                         stream,
                                         mr);
  auto [result_mask, null_count] =
    cudf::bitmask_and(cudf::table_view({col1.parent(), col2.parent()}), stream);
  return make_lists_column(search_keys_list.size(),
                           std::make_unique<column>(search_keys_list.offsets()),
                           std::move(map_structs),
                           null_count,
                           std::move(result_mask));
}

}  // namespace spark_rapids_jni
