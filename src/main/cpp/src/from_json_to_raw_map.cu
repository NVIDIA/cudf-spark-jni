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

#include "from_json_to_raw_map_debug.cuh"
#include "json_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/io/detail/json.hpp>
#include <cudf/io/detail/tokenize_json.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/functional>
#include <cuda/std/limits>
#include <cuda/std/utility>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace spark_rapids_jni {

using namespace cudf::io::json;

namespace {

template <typename InputIterator,
          typename StencilIterator,
          typename OutputIterator,
          typename Predicate>
OutputIterator copy_if(InputIterator begin,
                       InputIterator end,
                       StencilIterator stencil,
                       OutputIterator result,
                       Predicate predicate,
                       rmm::cuda_stream_view stream)
{
  auto const num_items = cuda::std::distance(begin, end);

  auto num_selected =
    cudf::detail::device_scalar<cuda::std::size_t>(stream, cudf::get_current_device_resource_ref());

  auto temp_storage_bytes = std::size_t{0};
  CUDF_CUDA_TRY(cub::DeviceSelect::FlaggedIf(nullptr,
                                             temp_storage_bytes,
                                             begin,
                                             stencil,
                                             result,
                                             num_selected.data(),
                                             num_items,
                                             predicate,
                                             stream.value()));

  auto d_temp_storage =
    rmm::device_buffer(temp_storage_bytes, stream, cudf::get_current_device_resource_ref());

  CUDF_CUDA_TRY(cub::DeviceSelect::FlaggedIf(d_temp_storage.data(),
                                             temp_storage_bytes,
                                             begin,
                                             stencil,
                                             result,
                                             num_selected.data(),
                                             num_items,
                                             predicate,
                                             stream.value()));

  return result + num_selected.value(stream);
}

template <typename Predicate, typename InputIterator, typename OutputIterator>
OutputIterator copy_if(InputIterator begin,
                       InputIterator end,
                       OutputIterator output,
                       Predicate predicate,
                       rmm::cuda_stream_view stream)
{
  auto const num_items = cuda::std::distance(begin, end);

  // Device scalar to store the number of selected elements
  auto num_selected =
    cudf::detail::device_scalar<cuda::std::size_t>(stream, cudf::get_current_device_resource_ref());

  // First call to get temporary storage size
  size_t temp_storage_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceSelect::If(nullptr,
                                      temp_storage_bytes,
                                      begin,
                                      output,
                                      num_selected.data(),
                                      num_items,
                                      predicate,
                                      stream.value()));

  // Allocate temporary storage
  rmm::device_buffer d_temp_storage(
    temp_storage_bytes, stream, cudf::get_current_device_resource_ref());

  // Run copy_if
  CUDF_CUDA_TRY(cub::DeviceSelect::If(d_temp_storage.data(),
                                      temp_storage_bytes,
                                      begin,
                                      output,
                                      num_selected.data(),
                                      num_items,
                                      predicate,
                                      stream.value()));

  // Copy number of selected elements back to host via pinned memory
  return output + num_selected.value(stream);
}

std::unique_ptr<cudf::column> make_empty_map(rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  auto keys   = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  auto values = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  std::vector<std::unique_ptr<cudf::column>> out_keys_vals;
  out_keys_vals.emplace_back(std::move(keys));
  out_keys_vals.emplace_back(std::move(values));
  auto child =
    cudf::make_structs_column(0, std::move(out_keys_vals), 0, rmm::device_buffer{}, stream, mr);
  auto offsets = cudf::make_empty_column(cudf::data_type(cudf::type_id::INT32));
  return cudf::make_lists_column(0, std::move(offsets), std::move(child), 0, rmm::device_buffer{});
}

// Concatenating all input strings into one string, for which each input string is appended by a
// delimiter character that does not exist in the input column.
std::tuple<rmm::device_buffer, char, std::unique_ptr<cudf::column>> unify_json_strings(
  cudf::strings_column_view const& input, rmm::cuda_stream_view stream)
{
  auto const default_mr = cudf::get_current_device_resource_ref();
  auto [concatenated_buff, delimiter, should_be_nullified] =
    concat_json(input, /*nullify_invalid_rows*/ true, stream, default_mr);

  if (concatenated_buff->size() == 0) {
    return {std::move(*concatenated_buff), delimiter, std::move(should_be_nullified)};
  }

  // Append the delimiter to the end of the concatenated buffer.
  // This is to fix a bug when the last string is invalid
  // (https://github.com/rapidsai/cudf/issues/16999).
  // The bug was fixed in libcudf's JSON reader by the same way like this.
  auto unified_buff = rmm::device_buffer(concatenated_buff->size() + 1, stream, default_mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(unified_buff.data(),
                                concatenated_buff->data(),
                                concatenated_buff->size(),
                                cudaMemcpyDefault,
                                stream));
  cudf::detail::cuda_memcpy_async(
    cudf::device_span<char>(static_cast<char*>(unified_buff.data()) + concatenated_buff->size(),
                            1u),
    cudf::host_span<char const>(&delimiter, 1, false),
    stream);

  return {std::move(unified_buff), delimiter, std::move(should_be_nullified)};
}

// Check if a token is a json node.
struct is_node {
  __host__ __device__ bool operator()(PdaTokenT token) const
  {
    switch (token) {
      case token_t::StructBegin:
      case token_t::ListBegin:
      case token_t::StringBegin:
      case token_t::ValueBegin:
      case token_t::FieldNameBegin:
      case token_t::ErrorBegin: return true;
      default: return false;
    };
  }
};

// Compute the level of each token node.
// The top json node (top json object level) has level 0.
// Each row in the input column should have levels starting from 1.
// This is copied from cudf's `json_tree.cu`.
rmm::device_uvector<TreeDepthT> compute_node_levels(std::size_t num_nodes,
                                                    cudf::device_span<PdaTokenT const> tokens,
                                                    rmm::cuda_stream_view stream)
{
  auto token_levels = rmm::device_uvector<TreeDepthT>(tokens.size(), stream);

  // Whether the token pops from the parent node stack.
  auto const does_pop =
    cuda::proclaim_return_type<bool>([] __device__(PdaTokenT const token) -> bool {
      switch (token) {
        case token_t::StructMemberEnd:
        case token_t::StructEnd:
        case token_t::ListEnd: return true;
        default: return false;
      };
    });

  // Whether the token pushes onto the parent node stack.
  auto const does_push =
    cuda::proclaim_return_type<bool>([] __device__(PdaTokenT const token) -> bool {
      switch (token) {
        case token_t::FieldNameBegin:
        case token_t::StructBegin:
        case token_t::ListBegin: return true;
        default: return false;
      };
    });

  auto const push_pop_it = thrust::make_transform_iterator(
    tokens.begin(),
    cuda::proclaim_return_type<cudf::size_type>(
      [does_push, does_pop] __device__(PdaTokenT const token) -> cudf::size_type {
        return does_push(token) - does_pop(token);
      }));
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream),
                         push_pop_it,
                         push_pop_it + tokens.size(),
                         token_levels.begin());

  auto node_levels    = rmm::device_uvector<TreeDepthT>(num_nodes, stream);
  auto const copy_end = copy_if(token_levels.begin(),
                                token_levels.end(),
                                tokens.begin(),
                                node_levels.begin(),
                                is_node{},
                                stream);
  CUDF_EXPECTS(cuda::std::distance(node_levels.begin(), copy_end) == num_nodes,
               "Node level count mismatch.");

#ifdef DEBUG_FROM_JSON
  print_debug(node_levels, "Node levels", ", ", stream);
#endif
  return node_levels;
}

// Compute the map from nodes to their indices in the list of all tokens.
rmm::device_uvector<NodeIndexT> compute_node_to_token_index_map(
  std::size_t num_nodes, cudf::device_span<PdaTokenT const> tokens, rmm::cuda_stream_view stream)
{
  auto node_token_ids   = rmm::device_uvector<NodeIndexT>(num_nodes, stream);
  auto const node_id_it = thrust::counting_iterator<NodeIndexT>(0);
  auto const copy_end   = copy_if(node_id_it,
                                node_id_it + tokens.size(),
                                tokens.begin(),
                                node_token_ids.begin(),
                                is_node{},
                                stream);
  CUDF_EXPECTS(cuda::std::distance(node_token_ids.begin(), copy_end) == num_nodes,
               "Invalid computation for node-to-token-index map.");

#ifdef DEBUG_FROM_JSON
  print_map_debug(node_token_ids, "Node-to-token-index map", stream);
#endif
  return node_token_ids;
}

// This is copied from cudf's `json_tree.cu`.
template <typename KeyType, typename IndexType = cudf::size_type>
std::pair<rmm::device_uvector<KeyType>, rmm::device_uvector<IndexType>> stable_sorted_key_order(
  cudf::device_span<KeyType const> keys, rmm::cuda_stream_view stream)
{
  // Buffers used for storing intermediate results during sorting.
  rmm::device_uvector<KeyType> keys_buffer1(keys.size(), stream);
  rmm::device_uvector<KeyType> keys_buffer2(keys.size(), stream);
  rmm::device_uvector<IndexType> order_buffer1(keys.size(), stream);
  rmm::device_uvector<IndexType> order_buffer2(keys.size(), stream);
  cub::DoubleBuffer<KeyType> keys_buffer(keys_buffer1.data(), keys_buffer2.data());
  cub::DoubleBuffer<IndexType> order_buffer(order_buffer1.data(), order_buffer2.data());

  thrust::copy(rmm::exec_policy_nosync(stream), keys.begin(), keys.end(), keys_buffer1.begin());
  thrust::sequence(rmm::exec_policy_nosync(stream), order_buffer1.begin(), order_buffer1.end());

  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
    nullptr, temp_storage_bytes, keys_buffer, order_buffer, keys.size());
  rmm::device_buffer d_temp_storage(temp_storage_bytes, stream);
  cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                  temp_storage_bytes,
                                  keys_buffer,
                                  order_buffer,
                                  keys.size(),
                                  0,
                                  sizeof(KeyType) * 8,
                                  stream.value());

  return std::pair{keys_buffer.Current() == keys_buffer1.data() ? std::move(keys_buffer1)
                                                                : std::move(keys_buffer2),
                   order_buffer.Current() == order_buffer1.data() ? std::move(order_buffer1)
                                                                  : std::move(order_buffer2)};
}

// This is copied from cudf's `json_tree.cu`.
void propagate_parent_to_siblings(cudf::device_span<TreeDepthT const> node_levels,
                                  cudf::device_span<NodeIndexT> parent_node_ids,
                                  rmm::cuda_stream_view stream)
{
  auto const [sorted_node_levels, sorted_order] = stable_sorted_key_order(node_levels, stream);

  // Instead of gather, using permutation_iterator, which is ~17% faster.
  thrust::inclusive_scan_by_key(
    rmm::exec_policy_nosync(stream),
    sorted_node_levels.begin(),
    sorted_node_levels.end(),
    thrust::make_permutation_iterator(parent_node_ids.begin(), sorted_order.begin()),
    thrust::make_permutation_iterator(parent_node_ids.begin(), sorted_order.begin()),
    cuda::std::equal_to<TreeDepthT>{},
    cuda::maximum<NodeIndexT>{});
}

// This is copied from cudf's `json_tree.cu`.
rmm::device_uvector<NodeIndexT> compute_parent_node_ids(
  cudf::device_span<PdaTokenT const> tokens,
  cudf::device_span<NodeIndexT const> node_token_ids,
  rmm::cuda_stream_view stream)
{
  auto const first_childs_parent_token_id =
    cuda::proclaim_return_type<NodeIndexT>([tokens] __device__(auto i) -> NodeIndexT {
      if (i <= 0) { return -1; }
      if (tokens[i - 1] == token_t::StructBegin || tokens[i - 1] == token_t::ListBegin) {
        return i - 1;
      } else if (tokens[i - 1] == token_t::FieldNameEnd) {
        return i - 2;
      } else if (tokens[i - 1] == token_t::StructMemberBegin &&
                 (tokens[i - 2] == token_t::StructBegin || tokens[i - 2] == token_t::ListBegin)) {
        return i - 2;
      } else {
        return -1;
      }
    });

  auto const num_nodes = node_token_ids.size();
  auto parent_node_ids = rmm::device_uvector<NodeIndexT>(num_nodes, stream);
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    node_token_ids.begin(),
    node_token_ids.end(),
    parent_node_ids.begin(),
    cuda::proclaim_return_type<NodeIndexT>(
      [node_ids_gpu = node_token_ids.begin(), num_nodes, first_childs_parent_token_id] __device__(
        NodeIndexT const tid) -> NodeIndexT {
        auto const pid = first_childs_parent_token_id(tid);
        return pid < 0
                 ? cudf::io::json::parent_node_sentinel
                 : thrust::lower_bound(thrust::seq, node_ids_gpu, node_ids_gpu + num_nodes, pid) -
                     node_ids_gpu;
      }));

  // Propagate parent node to siblings from first sibling - inplace.
  auto const node_levels = compute_node_levels(num_nodes, tokens, stream);
  propagate_parent_to_siblings(node_levels, parent_node_ids, stream);

#ifdef DEBUG_FROM_JSON
  print_debug(parent_node_ids, "Parent node ids", ", ", stream);
#endif
  return parent_node_ids;
}

// Special values to denote if a node is a key or value to extract for the output.
constexpr int8_t key_sentinel{1};
constexpr int8_t value_sentinel{2};

// Check for each node if it is a key or a value field.
rmm::device_uvector<int8_t> check_key_or_value_nodes(
  cudf::device_span<NodeIndexT const> parent_node_ids, rmm::cuda_stream_view stream)
{
  auto key_or_value       = rmm::device_uvector<int8_t>(parent_node_ids.size(), stream);
  auto const transform_it = thrust::counting_iterator<int>(0);
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    transform_it,
    transform_it + parent_node_ids.size(),
    key_or_value.begin(),
    cuda::proclaim_return_type<int8_t>(
      [key_sentinel   = key_sentinel,
       value_sentinel = value_sentinel,
       parent_ids     = parent_node_ids.begin()] __device__(auto const node_id) -> int8_t {
        if (parent_ids[node_id] >= 0) {
          auto const grand_parent = parent_ids[parent_ids[node_id]];
          if (grand_parent < 0) {
            return key_sentinel;
          } else if (parent_ids[grand_parent] < 0) {
            return value_sentinel;
          }
        }

        return 0;
      }));

#ifdef DEBUG_FROM_JSON
  print_debug(key_or_value, "Nodes are key/value (1==key, 2==value)", ", ", stream);
#endif
  return key_or_value;
}

// Convert token positions to node ranges for each valid node.
struct node_ranges_fn {
  cudf::device_span<PdaTokenT const> tokens;
  cudf::device_span<SymbolOffsetT const> token_positions;
  cudf::device_span<NodeIndexT const> node_token_ids;
  cudf::device_span<NodeIndexT const> parent_node_ids;
  cudf::device_span<int8_t const> key_or_value;

  // Whether the extracted string values from json map will have the quote character.
  static const bool include_quote_char{false};

  __device__ cuda::std::pair<SymbolOffsetT, SymbolOffsetT> operator()(cudf::size_type node_id) const
  {
    [[maybe_unused]] auto const is_begin_of_section =
      cuda::proclaim_return_type<bool>([] __device__(PdaTokenT const token) {
        switch (token) {
          case token_t::StructBegin:
          case token_t::ListBegin:
          case token_t::StringBegin:
          case token_t::ValueBegin:
          case token_t::FieldNameBegin: return true;
          default: return false;
        };
      });

    // The end-of-* partner token for a given beginning-of-* token
    auto const end_of_partner =
      cuda::proclaim_return_type<token_t>([] __device__(PdaTokenT const token) {
        switch (token) {
          case token_t::StructBegin: return token_t::StructEnd;
          case token_t::ListBegin: return token_t::ListEnd;
          case token_t::StringBegin: return token_t::StringEnd;
          case token_t::ValueBegin: return token_t::ValueEnd;
          case token_t::FieldNameBegin: return token_t::FieldNameEnd;
          default: return token_t::ErrorBegin;
        };
      });

    // Encode a fixed value for nested node types (list+struct).
    auto const nested_node_to_value =
      cuda::proclaim_return_type<int32_t>([] __device__(PdaTokenT const token) -> int32_t {
        switch (token) {
          case token_t::StructBegin: return 1;
          case token_t::StructEnd: return -1;
          case token_t::ListBegin: return 1 << 8;
          case token_t::ListEnd: return -(1 << 8);
          default: return 0;
        };
      });

    auto const get_token_position = cuda::proclaim_return_type<SymbolOffsetT>(
      [include_quote_char = include_quote_char] __device__(PdaTokenT const token,
                                                           SymbolOffsetT const token_index) {
        constexpr SymbolOffsetT quote_char_size = 1;
        switch (token) {
          // Strip off quote char included for StringBegin
          case token_t::StringBegin:
            return token_index + (include_quote_char ? 0 : quote_char_size);
          // Strip off or Include trailing quote char for string values for StringEnd
          case token_t::StringEnd: return token_index + (include_quote_char ? quote_char_size : 0);
          // Strip off quote char included for FieldNameBegin
          case token_t::FieldNameBegin: return token_index + quote_char_size;
          default: return token_index;
        };
      });

    if (key_or_value[node_id] != key_sentinel && key_or_value[node_id] != value_sentinel) {
      return {0, 0};
    }

    auto const token_idx = node_token_ids[node_id];
    auto const token     = tokens[token_idx];
    cudf_assert(is_begin_of_section(token) && "Invalid node category.");

    // The section from the original JSON input that this token demarcates.
    auto const range_begin = get_token_position(token, token_positions[token_idx]);
    auto range_end         = range_begin + 1;  // non-leaf, non-field nodes ignore this value.
    if ((token_idx + 1) < tokens.size() && end_of_partner(token) == tokens[token_idx + 1]) {
      // Update the range_end for this pair of tokens
      range_end = get_token_position(tokens[token_idx + 1], token_positions[token_idx + 1]);
    } else {
      auto nested_range_value = nested_node_to_value(token);  // iterate until this is zero
      auto end_idx            = token_idx + 1;
      while (end_idx < tokens.size()) {
        nested_range_value += nested_node_to_value(tokens[end_idx]);
        if (nested_range_value == 0) {
          range_end = get_token_position(tokens[end_idx], token_positions[end_idx]) + 1;
          break;
        }
        ++end_idx;
      }
      cudf_assert(nested_range_value == 0 && "Invalid range computation.");
      cudf_assert((end_idx + 1 < tokens.size()) && "Invalid range computation.");
    }
    return {range_begin, range_end};
  }
};

// Compute position range for each node.
// These ranges identify positions to extract nodes from the unified json string.
rmm::device_uvector<cuda::std::pair<SymbolOffsetT, SymbolOffsetT>> compute_node_ranges(
  cudf::device_span<PdaTokenT const> tokens,
  cudf::device_span<SymbolOffsetT const> token_positions,
  cudf::device_span<NodeIndexT const> node_token_ids,
  cudf::device_span<NodeIndexT const> parent_node_ids,
  cudf::device_span<int8_t const> key_or_value,
  rmm::cuda_stream_view stream)
{
  auto const num_nodes = node_token_ids.size();
  auto node_ranges =
    rmm::device_uvector<cuda::std::pair<SymbolOffsetT, SymbolOffsetT>>(num_nodes, stream);
  auto const transform_it = thrust::counting_iterator<int>(0);
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    transform_it,
    transform_it + num_nodes,
    node_ranges.begin(),
    node_ranges_fn{tokens, token_positions, node_token_ids, parent_node_ids, key_or_value});

#ifdef DEBUG_FROM_JSON
  print_pair_debug(node_ranges, "Node ranges", stream);
#endif
  return node_ranges;
}

// Function logic for substring API.
// This both calculates the output size and executes the substring.
// No bound check is performed, assuming that the substring bounds are all valid.
struct substring_fn {
  cudf::device_span<char const> d_string;
  cudf::device_span<cuda::std::pair<SymbolOffsetT, SymbolOffsetT> const> d_ranges;

  cudf::size_type* d_sizes;
  char* d_chars;
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(cudf::size_type idx)
  {
    auto const range = d_ranges[idx];
    auto const size  = range.second - range.first;
    if (d_chars) {
      memcpy(d_chars + d_offsets[idx], d_string.data() + range.first, size);
    } else {
      d_sizes[idx] = size;
    }
  }
};

// Extract key-value string pairs from the input json string.
std::unique_ptr<cudf::column> extract_keys_or_values(
  int8_t key_value_sentinel,
  cudf::device_span<cuda::std::pair<SymbolOffsetT, SymbolOffsetT> const> node_ranges,
  cudf::device_span<int8_t const> key_or_value,
  cudf::device_span<char const> input_json,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const is_key_or_value = cuda::proclaim_return_type<bool>(
    [key_or_value, key_value_sentinel] __device__(auto const node_id) {
      return key_or_value[node_id] == key_value_sentinel;
    });

  auto extracted_ranges = rmm::device_uvector<cuda::std::pair<SymbolOffsetT, SymbolOffsetT>>(
    node_ranges.size(), stream, mr);
  auto const range_end   = copy_if(node_ranges.begin(),
                                 node_ranges.end(),
                                 thrust::make_counting_iterator(0),
                                 extracted_ranges.begin(),
                                 is_key_or_value,
                                 stream);
  auto const num_extract = cuda::std::distance(extracted_ranges.begin(), range_end);
  if (num_extract == 0) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}); }

  auto [offsets, chars] = cudf::strings::detail::make_strings_children(
    substring_fn{input_json, extracted_ranges}, num_extract, stream, mr);
  return cudf::make_strings_column(
    num_extract, std::move(offsets), chars.release(), 0, rmm::device_buffer{});
}

// Compute the offsets for the final lists of Struct<String,String>.
std::unique_ptr<cudf::column> compute_list_offsets(
  cudf::size_type n_lists,
  cudf::device_span<NodeIndexT const> parent_node_ids,
  cudf::device_span<int8_t const> key_or_value,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Count the number of children nodes for the json object nodes.
  // These object nodes are given as one row of the input json strings column.
  auto node_child_counts = rmm::device_uvector<NodeIndexT>(parent_node_ids.size(), stream);

  // For the nodes having parent_id < 0 (they are json object given by one input row), set their
  // child counts to zero. Otherwise, set child counts to a negative sentinel number.
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    parent_node_ids.begin(),
    parent_node_ids.end(),
    node_child_counts.begin(),
    cuda::proclaim_return_type<NodeIndexT>([] __device__(auto const parent_id) -> NodeIndexT {
      return parent_id < 0 ? 0 : cuda::std::numeric_limits<NodeIndexT>::lowest();
    }));

  auto const is_key = cuda::proclaim_return_type<bool>(
    [key_or_value = key_or_value.begin()] __device__(auto const node_id) {
      return key_or_value[node_id] == key_sentinel;
    });

  // Count the number of keys for each json object using `atomicAdd`.
  auto const transform_it = thrust::counting_iterator<int>(0);
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   transform_it,
                   transform_it + parent_node_ids.size(),
                   [is_key,
                    child_counts = node_child_counts.begin(),
                    parent_ids   = parent_node_ids.begin()] __device__(auto const node_id) {
                     if (is_key(node_id)) {
                       auto const parent_id = parent_ids[node_id];
                       atomicAdd(&child_counts[parent_id], 1);
                     }
                   });
#ifdef DEBUG_FROM_JSON
  print_debug(node_child_counts, "Nodes' child keys counts", ", ", stream);
#endif

  auto list_offsets   = rmm::device_uvector<cudf::size_type>(n_lists + 1, stream, mr);
  auto const copy_end = copy_if(
    node_child_counts.begin(),
    node_child_counts.end(),
    list_offsets.begin(),
    cuda::proclaim_return_type<bool>([] __device__(auto const count) { return count >= 0; }),
    stream);
  CUDF_EXPECTS(cuda::std::distance(list_offsets.begin(), copy_end) == static_cast<int64_t>(n_lists),
               "Invalid list size computation.");
#ifdef DEBUG_FROM_JSON
  print_debug(list_offsets, "Output list sizes (except the last one)", ", ", stream);
#endif

  thrust::exclusive_scan(rmm::exec_policy_nosync(stream),
                         list_offsets.begin(),
                         list_offsets.end(),
                         list_offsets.begin());
#ifdef DEBUG_FROM_JSON
  print_debug(list_offsets, "Output list offsets", ", ", stream);
#endif
  return std::make_unique<cudf::column>(std::move(list_offsets), rmm::device_buffer{}, 0);
}

// If a JSON line is invalid, the tokens corresponding to that line are output as
// [StructBegin, StructEnd] but their locations in the unified JSON string are all set to 0.
struct is_invalid_struct_begin {
  cudf::device_span<PdaTokenT const> tokens;
  cudf::device_span<NodeIndexT const> node_token_ids;
  cudf::device_span<SymbolOffsetT const> token_positions;

  __device__ bool operator()(int node_idx) const
  {
    auto const node_token_id = node_token_ids[node_idx];
    auto const node_token    = tokens[node_token_id];
    if (node_token != token_t::StructBegin) { return false; }

    // The next token in the token stream after node_token.
    // Since the token stream has been post process, there should always be the more token.
    auto const next_token = tokens[node_token_id + 1];
    if (next_token != token_t::StructEnd) { return false; }

    return token_positions[node_token_id] == 0 && token_positions[node_token_id + 1] == 0;
  }
};

// A line begin with a StructBegin token which does not have parent.
struct is_line_begin {
  cudf::device_span<PdaTokenT const> tokens;
  cudf::device_span<NodeIndexT const> node_token_ids;
  cudf::device_span<NodeIndexT const> parent_node_ids;

  __device__ bool operator()(int node_idx) const
  {
    return tokens[node_token_ids[node_idx]] == token_t::StructBegin &&
           parent_node_ids[node_idx] < 0;
  }
};

std::pair<rmm::device_buffer, cudf::size_type> create_null_mask(
  cudf::size_type num_rows,
  std::unique_ptr<cudf::column> const& should_be_nullified,
  cudf::device_span<PdaTokenT const> tokens,
  cudf::device_span<SymbolOffsetT const> token_positions,
  cudf::device_span<NodeIndexT const> node_token_ids,
  cudf::device_span<NodeIndexT const> parent_node_ids,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_nodes = node_token_ids.size();

  // To store indices of the StructBegin nodes that are detected as of invalid JSON objects.
  rmm::device_uvector<NodeIndexT> invalid_indices(num_nodes, stream);

  auto const node_id_it = thrust::counting_iterator<NodeIndexT>(0);
  auto const invalid_copy_end =
    copy_if(node_id_it,
            node_id_it + node_token_ids.size(),
            invalid_indices.begin(),
            is_invalid_struct_begin{tokens, node_token_ids, token_positions},
            stream);
  auto const num_invalid = cuda::std::distance(invalid_indices.begin(), invalid_copy_end);
#ifdef DEBUG_FROM_JSON
  print_debug(invalid_indices,
              "Invalid StructBegin nodes' indices (size = " + std::to_string(num_invalid) + ")",
              ", ",
              stream);
#endif

  // In addition to `should_be_nullified` which identified the null and empty rows,
  // we also need to identify the rows containing invalid JSON objects.
  if (num_invalid > 0) {
    // Build a list of StructBegin tokens that start a line.
    // We must have such list having size equal to the number of original input JSON strings.
    rmm::device_uvector<NodeIndexT> line_begin_indices(num_nodes, stream);
    auto const line_begin_copy_end = copy_if(node_id_it,
                                             node_id_it + node_token_ids.size(),
                                             line_begin_indices.begin(),
                                             is_line_begin{tokens, node_token_ids, parent_node_ids},
                                             stream);
    auto const num_line_begin =
      cuda::std::distance(line_begin_indices.begin(), line_begin_copy_end);
    CUDF_EXPECTS(num_line_begin == num_rows, "Incorrect count of JSON objects.");
#ifdef DEBUG_FROM_JSON
    print_debug(line_begin_indices,
                "Line begin StructBegin indices (size = " + std::to_string(num_line_begin) + ")",
                ", ",
                stream);
#endif

    // Scatter the indices of the invalid StructBegin nodes into `should_be_nullified`.
    thrust::for_each(rmm::exec_policy_nosync(stream),
                     invalid_indices.begin(),
                     invalid_indices.begin() + num_invalid,
                     [should_be_nullified = should_be_nullified->mutable_view().begin<bool>(),
                      line_begin_indices  = line_begin_indices.begin(),
                      num_rows] __device__(auto node_idx) {
                       auto const row_idx = thrust::lower_bound(thrust::seq,
                                                                line_begin_indices,
                                                                line_begin_indices + num_rows,
                                                                node_idx) -
                                            line_begin_indices;
                       should_be_nullified[row_idx] = true;
                     });
  }

  auto const valid_it          = should_be_nullified->view().begin<bool>();
  auto [null_mask, null_count] = cudf::detail::valid_if(
    valid_it, valid_it + should_be_nullified->size(), thrust::logical_not<bool>{}, stream, mr);
  return {null_count > 0 ? std::move(null_mask) : rmm::device_buffer{0, stream, mr}, null_count};
}

}  // namespace

std::unique_ptr<cudf::column> from_json_to_raw_map(cudf::strings_column_view const& input,
                                                   bool normalize_single_quotes,
                                                   bool allow_leading_zeros,
                                                   bool allow_nonnumeric_numbers,
                                                   bool allow_unquoted_control,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return make_empty_map(stream, mr); }

  // Firstly, concatenate all the input json strings into one buffer.
  // When testing/debugging, the output can be validated using
  // https://jsonformatter.curiousconcept.com.
  auto [concat_json_buff, delimiter, should_be_nullified] = unify_json_strings(input, stream);
  auto concat_buff_wrapper =
    cudf::io::datasource::owning_buffer<rmm::device_buffer>(std::move(concat_json_buff));
  if (normalize_single_quotes) {
    cudf::io::json::detail::normalize_single_quotes(
      concat_buff_wrapper, delimiter, stream, cudf::get_current_device_resource_ref());
  }
  auto const preprocessed_input = cudf::device_span<char const>(
    reinterpret_cast<char const*>(concat_buff_wrapper.data()), concat_buff_wrapper.size());

  // Tokenize the input json strings.
  static_assert(sizeof(SymbolT) == sizeof(char),
                "Invalid internal data for nested json tokenizer.");
  auto const [tokens, token_positions] = cudf::io::json::detail::get_token_stream(
    preprocessed_input,
    cudf::io::json_reader_options_builder{}
      .lines(true)
      .normalize_whitespace(false)  // don't need it
      .experimental(true)
      .mixed_types_as_string(true)
      .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
      .strict_validation(true)
      // specifying parameters
      .delimiter(delimiter)
      .numeric_leading_zeros(allow_leading_zeros)
      .nonnumeric_numbers(allow_nonnumeric_numbers)
      .unquoted_control_chars(allow_unquoted_control)
      .build(),
    stream,
    cudf::get_current_device_resource_ref());

#ifdef DEBUG_FROM_JSON
  print_debug(tokens, "Tokens", ", ", stream);
  print_debug(token_positions, "Token positions", ", ", stream);
  std::cerr << "normalize_single_quotes: " << normalize_single_quotes << std::endl;
  std::cerr << "allow_leading_zeros: " << allow_leading_zeros << std::endl;
  std::cerr << "allow_nonnumeric_numbers: " << allow_nonnumeric_numbers << std::endl;
  std::cerr << "allow_unquoted_control: " << allow_unquoted_control << std::endl;
#endif

  auto const num_nodes =
    thrust::count_if(rmm::exec_policy_nosync(stream), tokens.begin(), tokens.end(), is_node{});

  // Compute the map from nodes to their indices in the list of all tokens.
  auto const node_token_ids = compute_node_to_token_index_map(num_nodes, tokens, stream);

  // A map from each node to the index of its parent node.
  auto const parent_node_ids = compute_parent_node_ids(tokens, node_token_ids, stream);

  // Check for each node if it is a map key or a map value to extract.
  auto const is_key_or_value_node = check_key_or_value_nodes(parent_node_ids, stream);

  // Compute index range for each node.
  // These ranges identify positions to extract nodes from the unified json string.
  auto const node_ranges = compute_node_ranges(
    tokens, token_positions, node_token_ids, parent_node_ids, is_key_or_value_node, stream);

  auto extracted_keys = extract_keys_or_values(
    key_sentinel, node_ranges, is_key_or_value_node, preprocessed_input, stream, mr);
  auto extracted_values = extract_keys_or_values(
    value_sentinel, node_ranges, is_key_or_value_node, preprocessed_input, stream, mr);
  CUDF_EXPECTS(extracted_keys->size() == extracted_values->size(),
               "Invalid key-value pair extraction.");

  // Compute the offsets of the final output lists column.
  auto list_offsets =
    compute_list_offsets(input.size(), parent_node_ids, is_key_or_value_node, stream, mr);

#ifdef DEBUG_FROM_JSON
  print_output_spark_map(list_offsets, extracted_keys, extracted_values, stream);
#endif

  auto const num_pairs = extracted_keys->size();
  std::vector<std::unique_ptr<cudf::column>> out_keys_vals;
  out_keys_vals.emplace_back(std::move(extracted_keys));
  out_keys_vals.emplace_back(std::move(extracted_values));
  auto structs_col = cudf::make_structs_column(
    num_pairs, std::move(out_keys_vals), 0, rmm::device_buffer{}, stream, mr);

  // Do not use `cudf::make_lists_column` since we do not need to call `purge_nonempty_nulls`
  // on the children columns as they do not have non-empty nulls.
  std::vector<std::unique_ptr<cudf::column>> list_children;
  list_children.emplace_back(std::move(list_offsets));
  list_children.emplace_back(std::move(structs_col));

  auto [null_mask, null_count] = create_null_mask(input.size(),
                                                  should_be_nullified,
                                                  tokens,
                                                  token_positions,
                                                  node_token_ids,
                                                  parent_node_ids,
                                                  stream,
                                                  mr);

  return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::LIST},
                                        input.size(),
                                        rmm::device_buffer{},
                                        std::move(null_mask),
                                        null_count,
                                        std::move(list_children));
}

}  // namespace spark_rapids_jni
