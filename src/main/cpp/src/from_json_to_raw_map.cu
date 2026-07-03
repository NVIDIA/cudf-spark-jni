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
#include "nvtx_ranges.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/io/detail/json.hpp>
#include <cudf/io/detail/tokenize_json.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/std/functional>
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
#include <thrust/uninitialized_fill.h>

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

// Build a zero-row `List<Struct<String, V>>` from an already-constructed empty value child. The
// value-child type selects the map flavor: an empty `STRING` gives `make_empty_map`'s output, an
// empty `List<String>` gives `make_empty_map_array`'s.
std::unique_ptr<cudf::column> make_empty_map_from_value(std::unique_ptr<cudf::column> value_child,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(value_child->size() == 0, "value_child must be an empty column.");
  auto keys = cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  std::vector<std::unique_ptr<cudf::column>> out_keys_vals;
  out_keys_vals.emplace_back(std::move(keys));
  out_keys_vals.emplace_back(std::move(value_child));
  auto child =
    cudf::make_structs_column(0, std::move(out_keys_vals), 0, rmm::device_buffer{}, stream, mr);
  auto offsets = cudf::make_empty_column(cudf::data_type(cudf::type_id::INT32));
  return cudf::make_lists_column(0, std::move(offsets), std::move(child), 0, rmm::device_buffer{});
}

std::unique_ptr<cudf::column> make_empty_map(rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  return make_empty_map_from_value(
    cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}), stream, mr);
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

// Node classification for the key/value/element extraction passes; stored in device arrays and the
// element flag, int8_t-backed to keep the arrays compact. `extract_keys_or_values` selects the
// nodes of a given kind (key, value, or a direct element of a value array).
enum class node_kind : int8_t { none = 0, key = 1, value = 2, element = 3 };

// Predicates over `node_kind` so the classification passes read as intent, not magic comparisons.
__device__ inline bool is_key_node(node_kind const k) { return k == node_kind::key; }
__device__ inline bool is_value_node(node_kind const k) { return k == node_kind::value; }
__device__ inline bool is_element_node(node_kind const k) { return k == node_kind::element; }

// Per-token nesting weights for the nested-range balance scan. libcudf bounds JSON tree depth to
// 127
// (`TreeDepthT` is `int8_t`), so the list weight (256) keeps list nesting (+/-256) from colliding
// with struct nesting (+/-struct_nesting_weight) when a running delta sum locates a matching close.
constexpr int32_t struct_nesting_weight{1};
constexpr int32_t list_nesting_weight{1 << 8};

// Check for each node if it is a key or a value field.
rmm::device_uvector<node_kind> check_key_or_value_nodes(
  cudf::device_span<NodeIndexT const> parent_node_ids, rmm::cuda_stream_view stream)
{
  auto key_or_value       = rmm::device_uvector<node_kind>(parent_node_ids.size(), stream);
  auto const transform_it = thrust::counting_iterator<int>(0);
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    transform_it,
    transform_it + parent_node_ids.size(),
    key_or_value.begin(),
    cuda::proclaim_return_type<node_kind>(
      [parent_ids = parent_node_ids.begin()] __device__(auto const node_id) -> node_kind {
        if (parent_ids[node_id] >= 0) {
          auto const grand_parent = parent_ids[parent_ids[node_id]];
          if (grand_parent < 0) {
            return node_kind::key;
          } else if (parent_ids[grand_parent] < 0) {
            return node_kind::value;
          }
        }

        return node_kind::none;
      }));

#ifdef DEBUG_FROM_JSON
  print_debug(key_or_value, "Nodes are key/value (1==key, 2==value)", ", ", stream);
#endif
  return key_or_value;
}

// The tokenize + tree-classify prelude shared by both raw-map variants. Owns every buffer it
// produces with the correct lifetimes: `preprocessed_input` is a span into the device memory held
// by `concat_buff_wrapper`, so the wrapper must outlive the span -- the struct keeps both, and
// both public functions keep the struct alive for the remainder of their bodies. The wrapper's
// move preserves the underlying device allocation (no realloc), so the span stays valid across the
// struct's move out of the helper.
struct tokenized_input {
  cudf::io::datasource::owning_buffer<rmm::device_buffer> concat_buff_wrapper;
  cudf::device_span<char const> preprocessed_input;
  rmm::device_uvector<PdaTokenT> tokens;
  rmm::device_uvector<SymbolOffsetT> token_positions;
  std::unique_ptr<cudf::column> should_be_nullified;
  cudf::size_type num_nodes;
  rmm::device_uvector<NodeIndexT> node_token_ids;
  rmm::device_uvector<NodeIndexT> parent_node_ids;
  rmm::device_uvector<node_kind> is_key_or_value_node;
};

// Run the shared prelude: concatenate + optionally normalize the input, tokenize it, then build
// the node-to-token map, parent-node ids, and key/value classification. Both public functions call
// this and diverge only after it returns.
tokenized_input tokenize_and_classify(cudf::strings_column_view const& input,
                                      json_parse_options const& options,
                                      rmm::cuda_stream_view stream)
{
  auto [concat_json_buff, delimiter, should_be_nullified] = unify_json_strings(input, stream);
  auto concat_buff_wrapper =
    cudf::io::datasource::owning_buffer<rmm::device_buffer>(std::move(concat_json_buff));
  if (options.normalize_single_quotes) {
    cudf::io::json::detail::normalize_single_quotes(
      concat_buff_wrapper, delimiter, stream, cudf::get_current_device_resource_ref());
  }
  auto const preprocessed_input = cudf::device_span<char const>(
    reinterpret_cast<char const*>(concat_buff_wrapper.data()), concat_buff_wrapper.size());

  // Tokenize the input json strings.
  static_assert(sizeof(SymbolT) == sizeof(char),
                "Invalid internal data for nested json tokenizer.");
  auto [tokens, token_positions] = cudf::io::json::detail::get_token_stream(
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
      .numeric_leading_zeros(options.allow_leading_zeros)
      .nonnumeric_numbers(options.allow_nonnumeric_numbers)
      .unquoted_control_chars(options.allow_unquoted_control)
      .build(),
    stream,
    cudf::get_current_device_resource_ref());

#ifdef DEBUG_FROM_JSON
  print_debug(tokens, "Tokens", ", ", stream);
  print_debug(token_positions, "Token positions", ", ", stream);
  std::cerr << "normalize_single_quotes: " << options.normalize_single_quotes << std::endl;
  std::cerr << "allow_leading_zeros: " << options.allow_leading_zeros << std::endl;
  std::cerr << "allow_nonnumeric_numbers: " << options.allow_nonnumeric_numbers << std::endl;
  std::cerr << "allow_unquoted_control: " << options.allow_unquoted_control << std::endl;
#endif

  auto const num_nodes = static_cast<cudf::size_type>(
    thrust::count_if(rmm::exec_policy_nosync(stream), tokens.begin(), tokens.end(), is_node{}));

  // Compute the map from nodes to their indices in the list of all tokens.
  auto node_token_ids = compute_node_to_token_index_map(num_nodes, tokens, stream);

  // A map from each node to the index of its parent node.
  auto parent_node_ids = compute_parent_node_ids(tokens, node_token_ids, stream);

  // Check for each node if it is a map key or a map value to extract.
  auto is_key_or_value_node = check_key_or_value_nodes(parent_node_ids, stream);

  return {std::move(concat_buff_wrapper),
          preprocessed_input,
          std::move(tokens),
          std::move(token_positions),
          std::move(should_be_nullified),
          num_nodes,
          std::move(node_token_ids),
          std::move(parent_node_ids),
          std::move(is_key_or_value_node)};
}

// The end-of-* partner token for a given begin-of-* token.
__device__ inline token_t end_of_partner(PdaTokenT const tk)
{
  switch (tk) {
    case token_t::StructBegin: return token_t::StructEnd;
    case token_t::ListBegin: return token_t::ListEnd;
    case token_t::StringBegin: return token_t::StringEnd;
    case token_t::ValueBegin: return token_t::ValueEnd;
    case token_t::FieldNameBegin: return token_t::FieldNameEnd;
    default: return token_t::ErrorBegin;
  };
}

// Per-token nesting-depth delta (struct +/-struct_nesting_weight, list +/-list_nesting_weight) so a
// running sum hits 0 at a nested node's matching close; the distinct magnitudes stop a struct close
// balancing a list open.
__device__ inline int32_t nested_node_to_value(PdaTokenT const tk)
{
  switch (tk) {
    case token_t::StructBegin: return struct_nesting_weight;
    case token_t::StructEnd: return -struct_nesting_weight;
    case token_t::ListBegin: return list_nesting_weight;
    case token_t::ListEnd: return -list_nesting_weight;
    default: return 0;
  };
}

// Walk from the begin token at `token_idx` to its matching close, using the per-token nesting-depth
// delta so a running sum returns to zero at the matching close. Returns the close token's index, or
// `tokens.size()` if the tokens are unbalanced (debug builds assert). The caller dequotes the
// returned token to obtain the range end.
__device__ inline NodeIndexT matching_close_index(cudf::device_span<PdaTokenT const> tokens,
                                                  NodeIndexT const token_idx)
{
  auto nested_range_value = nested_node_to_value(tokens[token_idx]);
  for (auto end_idx = token_idx + 1; end_idx < tokens.size(); ++end_idx) {
    nested_range_value += nested_node_to_value(tokens[end_idx]);
    if (nested_range_value == 0) {
      cudf_assert((end_idx + 1 < tokens.size()) && "Nested close must be followed by more tokens.");
      return end_idx;
    }
  }
  cudf_assert(false && "Nested range never closed (unbalanced tokens).");
  return static_cast<NodeIndexT>(tokens.size());
}

// De-quote a token's start position; shared by node_ranges_fn and element_classify_fn so the two
// paths can't drift apart. With include_quote_char == false, StringEnd matches default (return
// pos); with include_quote_char == true it keeps the trailing quote, which default would not.
__device__ inline SymbolOffsetT dequote_token_position(PdaTokenT const token,
                                                       SymbolOffsetT const token_index,
                                                       bool const include_quote_char)
{
  constexpr SymbolOffsetT quote_char_size = 1;
  switch (token) {
    case token_t::StringBegin: return token_index + (include_quote_char ? 0 : quote_char_size);
    case token_t::StringEnd: return token_index + (include_quote_char ? quote_char_size : 0);
    case token_t::FieldNameBegin: return token_index + quote_char_size;
    default: return token_index;
  };
}

// Byte range [begin, end) for the node at `token_idx`: the de-quoted begin, and the end taken from
// the adjacent partner close, or the matching close for a nested ([/{) node. Shared by both
// functors.
__device__ inline cuda::std::pair<SymbolOffsetT, SymbolOffsetT> compute_token_range(
  cudf::device_span<PdaTokenT const> tokens,
  cudf::device_span<SymbolOffsetT const> token_positions,
  NodeIndexT const token_idx,
  bool const include_quote_char)
{
  auto const token = tokens[token_idx];
  auto const range_begin =
    dequote_token_position(token, token_positions[token_idx], include_quote_char);
  auto range_end = range_begin + 1;
  if ((token_idx + 1) < tokens.size() && end_of_partner(token) == tokens[token_idx + 1]) {
    range_end = dequote_token_position(
      tokens[token_idx + 1], token_positions[token_idx + 1], include_quote_char);
  } else {
    auto const end_idx = matching_close_index(tokens, token_idx);
    if (end_idx < tokens.size()) {
      range_end =
        dequote_token_position(tokens[end_idx], token_positions[end_idx], include_quote_char) + 1;
    }
  }
  return {range_begin, range_end};
}

// Convert token positions to node ranges for each valid node.
struct node_ranges_fn {
  cudf::device_span<PdaTokenT const> tokens;
  cudf::device_span<SymbolOffsetT const> token_positions;
  cudf::device_span<NodeIndexT const> node_token_ids;
  cudf::device_span<NodeIndexT const> parent_node_ids;
  cudf::device_span<node_kind const> key_or_value;

  // Whether the extracted string values from json map will have the quote character.
  static bool const include_quote_char{false};

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

    if (!is_key_node(key_or_value[node_id]) && !is_value_node(key_or_value[node_id])) {
      return {0, 0};
    }

    auto const token_idx = node_token_ids[node_id];
    cudf_assert(is_begin_of_section(tokens[token_idx]) && "Invalid node category.");

    return compute_token_range(tokens, token_positions, token_idx, include_quote_char);
  }
};

// Compute position range for each node.
// These ranges identify positions to extract nodes from the unified json string.
rmm::device_uvector<cuda::std::pair<SymbolOffsetT, SymbolOffsetT>> compute_node_ranges(
  cudf::device_span<PdaTokenT const> tokens,
  cudf::device_span<SymbolOffsetT const> token_positions,
  cudf::device_span<NodeIndexT const> node_token_ids,
  cudf::device_span<NodeIndexT const> parent_node_ids,
  cudf::device_span<node_kind const> key_or_value,
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
  node_kind key_value_sentinel,
  cudf::device_span<cuda::std::pair<SymbolOffsetT, SymbolOffsetT> const> node_ranges,
  cudf::device_span<node_kind const> key_or_value,
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
  cudf::device_span<node_kind const> key_or_value,
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
      return is_key_node(key_or_value[node_id]);
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
  cudf::device_span<NodeIndexT const> precomputed_line_begin,
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
    // The per-row line-begin StructBegin indices. A caller that already computed this identical
    // list (same predicate, same node range) passes it in to skip the redundant device scan;
    // otherwise compute it here.
    rmm::device_uvector<NodeIndexT> computed_line_begin(0, stream);
    NodeIndexT const* line_begin_indices = precomputed_line_begin.data();
    if (precomputed_line_begin.empty()) {
      // Build a list of StructBegin tokens that start a line.
      // We must have such list having size equal to the number of original input JSON strings.
      computed_line_begin.resize(num_nodes, stream);
      auto const line_begin_copy_end =
        copy_if(node_id_it,
                node_id_it + node_token_ids.size(),
                computed_line_begin.begin(),
                is_line_begin{tokens, node_token_ids, parent_node_ids},
                stream);
      auto const num_line_begin =
        cuda::std::distance(computed_line_begin.begin(), line_begin_copy_end);
      CUDF_EXPECTS(num_line_begin == num_rows, "Incorrect count of JSON objects.");
      line_begin_indices = computed_line_begin.data();
#ifdef DEBUG_FROM_JSON
      print_debug(computed_line_begin,
                  "Line begin StructBegin indices (size = " + std::to_string(num_line_begin) + ")",
                  ", ",
                  stream);
#endif
    }

    // Scatter the indices of the invalid StructBegin nodes into `should_be_nullified`.
    thrust::for_each(rmm::exec_policy_nosync(stream),
                     invalid_indices.begin(),
                     invalid_indices.begin() + num_invalid,
                     [should_be_nullified = should_be_nullified->mutable_view().begin<bool>(),
                      line_begin_indices,
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

// Array-value path: parse `Map[String, Array[String]]` JSON into
// `List<Struct<String, List<String>>>`, reusing the shared device helpers above.

// True iff the byte range `[begin, end)` of `json` is exactly the 4-byte literal `null`. Callers
// gate on a `ValueBegin` token first, so a JSON STRING "null" (a `StringBegin`) never reaches here;
// the length-4 guard separates the literal from a 4-byte number like `1000`.
__device__ inline bool is_json_null_literal(char const* json,
                                            SymbolOffsetT begin,
                                            SymbolOffsetT end)
{
  if (end - begin != 4) { return false; }
  auto const* p = json + begin;
  return p[0] == 'n' && p[1] == 'u' && p[2] == 'l' && p[3] == 'l';
}

// Zero-row `List<Struct<String, List<String>>>` for empty input. Sibling of `make_empty_map`;
// the struct's value child is an empty `List<String>` instead of an empty `STRING`.
std::unique_ptr<cudf::column> make_empty_map_array(rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  return make_empty_map_from_value(
    cudf::make_empty_lists_column(cudf::data_type{cudf::type_id::STRING}), stream, mr);
}

// Classify each node as a value-array element and, if so, compute its de-quoted byte range and
// element validity (mask #2: null iff it is the literal `null`). Emitted as three parallel device
// vectors in one fused pass so the element extraction, inner offsets, and masks need no extra
// classification pass. The de-quote/range logic is shared with `node_ranges_fn` via
// `dequote_token_position`/`compute_token_range` so element strings match the value semantics.
struct element_classify_fn {
  cudf::device_span<char const> input_json;
  cudf::device_span<PdaTokenT const> tokens;
  cudf::device_span<SymbolOffsetT const> token_positions;
  cudf::device_span<NodeIndexT const> node_token_ids;
  cudf::device_span<NodeIndexT const> parent_node_ids;
  cudf::device_span<node_kind const> key_or_value;

  // Outputs.
  node_kind* element_flag;
  cuda::std::pair<SymbolOffsetT, SymbolOffsetT>* element_ranges;
  bool* element_valid;

  __device__ void operator()(cudf::size_type node_id) const
  {
    element_flag[node_id]   = node_kind::none;
    element_ranges[node_id] = {0, 0};
    element_valid[node_id]  = true;

    auto const token_idx = node_token_ids[node_id];
    auto const token     = tokens[token_idx];

    // Exclude error nodes; they are not array elements.
    if (token == token_t::ErrorBegin) { return; }

    // An element is a node whose parent is the value array's `ListBegin`. The value guard is
    // required: a nested inner `[` is also a `ListBegin`, but its parent is not value-tagged.
    auto const parent = parent_node_ids[node_id];
    if (parent < 0 || !is_value_node(key_or_value[parent]) ||
        tokens[node_token_ids[parent]] != token_t::ListBegin) {
      return;
    }

    auto const [range_begin, range_end] =
      compute_token_range(tokens, token_positions, token_idx, /*include_quote_char=*/false);

    element_flag[node_id]   = node_kind::element;
    element_ranges[node_id] = {range_begin, range_end};

    // Mask #2: a literal `null` element is a null element, and gets an empty byte span -- it
    // materializes as a null string with no non-empty-null payload, which the manual list assembly
    // requires.
    if (token == token_t::ValueBegin &&
        is_json_null_literal(input_json.data(), range_begin, range_end)) {
      element_valid[node_id]  = false;
      element_ranges[node_id] = {range_begin, range_begin};
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> from_json_to_raw_map(cudf::strings_column_view const& input,
                                                   json_parse_options options,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  if (input.is_empty()) { return make_empty_map(stream, mr); }

  // Concatenate, optionally normalize, tokenize, and classify the input json strings. The returned
  // struct owns the concatenated buffer that `preprocessed_input` points into, so it must stay
  // alive for the rest of this function. When testing/debugging, the output can be validated using
  // https://jsonformatter.curiousconcept.com.
  auto const tok                   = tokenize_and_classify(input, options, stream);
  auto const& preprocessed_input   = tok.preprocessed_input;
  auto const& tokens               = tok.tokens;
  auto const& token_positions      = tok.token_positions;
  auto const& should_be_nullified  = tok.should_be_nullified;
  auto const& node_token_ids       = tok.node_token_ids;
  auto const& parent_node_ids      = tok.parent_node_ids;
  auto const& is_key_or_value_node = tok.is_key_or_value_node;

  // Compute index range for each node.
  // These ranges identify positions to extract nodes from the unified json string.
  auto const node_ranges = compute_node_ranges(
    tokens, token_positions, node_token_ids, parent_node_ids, is_key_or_value_node, stream);

  auto extracted_keys = extract_keys_or_values(
    node_kind::key, node_ranges, is_key_or_value_node, preprocessed_input, stream, mr);
  auto extracted_values = extract_keys_or_values(
    node_kind::value, node_ranges, is_key_or_value_node, preprocessed_input, stream, mr);
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
                                                  /*precomputed_line_begin*/ {},
                                                  stream,
                                                  mr);

  return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::LIST},
                                        input.size(),
                                        rmm::device_buffer{},
                                        std::move(null_mask),
                                        null_count,
                                        std::move(list_children));
}

std::unique_ptr<cudf::column> from_json_to_raw_map_array_values(
  cudf::strings_column_view const& input,
  json_parse_options options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  if (input.is_empty()) { return make_empty_map_array(stream, mr); }

  // Concatenate, optionally normalize, tokenize, and classify the input JSON (same shared prelude
  // as the string-map path). The returned struct owns the concatenated buffer that
  // `preprocessed_input` points into, so it must stay alive for the rest of this function.
  auto const tok                   = tokenize_and_classify(input, options, stream);
  auto const& preprocessed_input   = tok.preprocessed_input;
  auto const& tokens               = tok.tokens;
  auto const& token_positions      = tok.token_positions;
  auto const& should_be_nullified  = tok.should_be_nullified;
  auto const num_nodes             = tok.num_nodes;
  auto const& node_token_ids       = tok.node_token_ids;
  auto const& parent_node_ids      = tok.parent_node_ids;
  auto const& is_key_or_value_node = tok.is_key_or_value_node;

  // Keys use the shared node-range computation; element ranges come from the fused classifier
  // below.
  auto const node_ranges = compute_node_ranges(
    tokens, token_positions, node_token_ids, parent_node_ids, is_key_or_value_node, stream);
  auto extracted_keys = extract_keys_or_values(
    node_kind::key, node_ranges, is_key_or_value_node, preprocessed_input, stream, mr);

  // Fused classification pass: per node emit the element flag, the de-quoted element byte range,
  // and element validity (mask #2) in one transform.
  auto element_flag = rmm::device_uvector<node_kind>(num_nodes, stream);
  auto element_ranges =
    rmm::device_uvector<cuda::std::pair<SymbolOffsetT, SymbolOffsetT>>(num_nodes, stream);
  auto element_valid = rmm::device_uvector<bool>(num_nodes, stream);
  {
    auto const node_id_it = thrust::counting_iterator<cudf::size_type>(0);
    thrust::for_each(rmm::exec_policy_nosync(stream),
                     node_id_it,
                     node_id_it + num_nodes,
                     element_classify_fn{preprocessed_input,
                                         tokens,
                                         token_positions,
                                         node_token_ids,
                                         parent_node_ids,
                                         is_key_or_value_node,
                                         element_flag.begin(),
                                         element_ranges.begin(),
                                         element_valid.begin()});
  }

  // Extract element strings with the shared extractor (a 0-null STRING column); attach mask #2.
  auto extracted_elements = extract_keys_or_values(
    node_kind::element, element_ranges, element_flag, preprocessed_input, stream, mr);
  {
    // Mask #2 over only the selected (element-flagged) nodes, in element document order, matching
    // the order `extract_keys_or_values` emits.
    auto element_validity = rmm::device_uvector<bool>(extracted_elements->size(), stream);
    auto const is_element = cuda::proclaim_return_type<bool>(
      [element_flag = element_flag.begin()] __device__(auto const node_id) {
        return is_element_node(element_flag[node_id]);
      });
    auto const valid_end    = copy_if(element_valid.begin(),
                                   element_valid.end(),
                                   thrust::make_counting_iterator(0),
                                   element_validity.begin(),
                                   is_element,
                                   stream);
    auto const num_elements = cuda::std::distance(element_validity.begin(), valid_end);
    CUDF_EXPECTS(num_elements == extracted_elements->size(),
                 "Invalid element validity extraction.");
    if (num_elements > 0) {
      auto [el_mask, el_null_count] = cudf::bools_to_mask(
        cudf::device_span<bool const>(element_validity.data(), num_elements), stream, mr);
      if (el_null_count > 0) {
        extracted_elements->set_null_mask(std::move(*el_mask), el_null_count);
      }
    }
  }

  // Outer list offsets: one slot per input row (shared list-offset computation).
  auto outer_offsets =
    compute_list_offsets(input.size(), parent_node_ids, is_key_or_value_node, stream, mr);

  // Struct/inner-list row count = number of value nodes (one per map pair). In a well-formed map
  // this equals the key count, which is already known on the host for free (column::size() is a
  // noexcept O(1) accessor). The device value-node count is computed only as a debug-build check of
  // that parser invariant, since the equality is the only consumer of the reduction.
  auto const num_values = extracted_keys->size();
#ifndef NDEBUG
  auto const num_value_nodes = static_cast<cudf::size_type>(
    thrust::count_if(rmm::exec_policy_nosync(stream),
                     is_key_or_value_node.begin(),
                     is_key_or_value_node.end(),
                     cuda::proclaim_return_type<bool>(
                       [] __device__(node_kind const kv) { return is_value_node(kv); })));
  CUDF_EXPECTS(num_value_nodes == num_values,
               "Invalid key-value pair extraction for map of arrays.");
#endif

  // Value-node ordinal map: exclusive scan of the value-tagged flag over nodes. A value node's
  // ordinal is its position among value nodes in document order, which matches the key/value
  // extraction order.
  auto value_ordinals = rmm::device_uvector<cudf::size_type>(num_nodes, stream);
  {
    auto const value_flag_it =
      thrust::make_transform_iterator(is_key_or_value_node.begin(),
                                      cuda::proclaim_return_type<cudf::size_type>(
                                        [] __device__(node_kind const kv) -> cudf::size_type {
                                          return is_value_node(kv) ? 1 : 0;
                                        }));
    thrust::exclusive_scan(rmm::exec_policy_nosync(stream),
                           value_flag_it,
                           value_flag_it + num_nodes,
                           value_ordinals.begin());
  }

  // Inner offsets: per value node, the number of its array elements. Length is `num_values + 1`;
  // a null/scalar/object value is still a value node and contributes a 0-count slot (its inner
  // list is masked null below, but the offset slot must exist).
  auto inner_offsets = rmm::device_uvector<cudf::size_type>(num_values + 1, stream, mr);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream), inner_offsets.begin(), inner_offsets.end(), 0);
  {
    auto const node_id_it = thrust::counting_iterator<cudf::size_type>(0);
    thrust::for_each(rmm::exec_policy_nosync(stream),
                     node_id_it,
                     node_id_it + num_nodes,
                     [element_flag   = element_flag.begin(),
                      parent_ids     = parent_node_ids.begin(),
                      value_ordinals = value_ordinals.begin(),
                      counts         = inner_offsets.begin()] __device__(auto const node_id) {
                       if (!is_element_node(element_flag[node_id])) { return; }
                       // The element's parent is the value array's `ListBegin` node.
                       auto const value_node = parent_ids[node_id];
                       auto ref = cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device>{
                         counts[value_ordinals[value_node]]};
                       ref.fetch_add(1, cuda::memory_order_relaxed);
                     });
    thrust::exclusive_scan(rmm::exec_policy_nosync(stream),
                           inner_offsets.begin(),
                           inner_offsets.end(),
                           inner_offsets.begin());
  }

  // Mask #1 (inner-list validity): per value node, null iff its token is not `ListBegin`
  // (null/scalar/object value). Indexed by value-node ordinal, matching the keys/values order.
  auto inner_valid = rmm::device_uvector<bool>(num_values, stream);
  {
    auto const node_id_it = thrust::counting_iterator<cudf::size_type>(0);
    thrust::for_each(rmm::exec_policy_nosync(stream),
                     node_id_it,
                     node_id_it + num_nodes,
                     [key_or_value   = is_key_or_value_node.begin(),
                      tokens         = tokens.begin(),
                      node_token_ids = node_token_ids.begin(),
                      value_ordinals = value_ordinals.begin(),
                      inner_valid    = inner_valid.begin()] __device__(auto const node_id) {
                       if (!is_value_node(key_or_value[node_id])) { return; }
                       inner_valid[value_ordinals[node_id]] =
                         tokens[node_token_ids[node_id]] == token_t::ListBegin;
                     });
  }

  // Assemble the inner `List<String>` (the struct's value child) by hand: the public factory would
  // launch a `purge_nonempty_nulls` scan, but the inner list carries nulls by design (mask #1) and
  // its element child has no non-empty nulls.
  auto [inner_mask, inner_null_count] =
    cudf::bools_to_mask(cudf::device_span<bool const>(inner_valid), stream, mr);
  auto inner_offsets_col =
    std::make_unique<cudf::column>(std::move(inner_offsets), rmm::device_buffer{}, 0);
  std::vector<std::unique_ptr<cudf::column>> inner_list_children;
  inner_list_children.emplace_back(std::move(inner_offsets_col));
  inner_list_children.emplace_back(std::move(extracted_elements));
  auto inner_list = std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::LIST},
    num_values,
    rmm::device_buffer{},
    inner_null_count > 0 ? std::move(*inner_mask) : rmm::device_buffer{},
    inner_null_count > 0 ? inner_null_count : 0,
    std::move(inner_list_children));

  // Struct child: <key STRING, value List<String>>.
  std::vector<std::unique_ptr<cudf::column>> out_keys_vals;
  out_keys_vals.emplace_back(std::move(extracted_keys));
  out_keys_vals.emplace_back(std::move(inner_list));
  auto structs_col = cudf::make_structs_column(
    num_values, std::move(out_keys_vals), 0, rmm::device_buffer{}, stream, mr);

  // Assemble the outer list (row offsets + struct child).
  std::vector<std::unique_ptr<cudf::column>> list_children;
  list_children.emplace_back(std::move(outer_offsets));
  list_children.emplace_back(std::move(structs_col));

  // Line-begin StructBegin node ids, one per input row, sorted by construction. Computed once here
  // and reused by `create_null_mask` below (passed in to skip its identical recomputation).
  rmm::device_uvector<NodeIndexT> line_begin_indices(num_nodes, stream);
  {
    auto const node_id_it          = thrust::counting_iterator<NodeIndexT>(0);
    auto const line_begin_copy_end = copy_if(node_id_it,
                                             node_id_it + num_nodes,
                                             line_begin_indices.begin(),
                                             is_line_begin{tokens, node_token_ids, parent_node_ids},
                                             stream);
    auto const num_line_begin =
      cuda::std::distance(line_begin_indices.begin(), line_begin_copy_end);
    CUDF_EXPECTS(num_line_begin == input.size(), "Incorrect count of JSON objects.");
  }

  // Row-level bad-record semantics (Spark `from_json`): a row is nullified if any of its value
  // nodes is a hard type mismatch -- a value that is neither a `ListBegin` (valid array) nor the
  // JSON `null` literal (kept, with a null inner list). Folding this into `should_be_nullified`
  // before `create_null_mask` lets the shared row null-mask computation absorb it with the correct
  // `null_count`. A value node's row is the last line-begin `StructBegin` id not exceeding it.
  {
    auto const node_id_it = thrust::counting_iterator<NodeIndexT>(0);
    thrust::for_each(
      rmm::exec_policy_nosync(stream),
      node_id_it,
      node_id_it + num_nodes,
      [key_or_value       = is_key_or_value_node.begin(),
       tokens             = tokens.begin(),
       node_token_ids     = node_token_ids.begin(),
       node_ranges        = node_ranges.begin(),
       input_json         = preprocessed_input.data(),
       line_begin_indices = line_begin_indices.begin(),
       num_rows           = input.size(),
       should_be_nullified =
         should_be_nullified->mutable_view().begin<bool>()] __device__(NodeIndexT const node_id) {
        if (!is_value_node(key_or_value[node_id])) { return; }
        auto const token = tokens[node_token_ids[node_id]];
        if (token == token_t::ListBegin) { return; }  // valid array.
        // Keep the row for a JSON `null` literal value (its inner list becomes null). A JSON STRING
        // "null" is a `StringBegin`, not a `ValueBegin`, so it is a hard mismatch like any other
        // scalar, matching Spark.
        if (token == token_t::ValueBegin) {
          auto const range = node_ranges[node_id];
          if (is_json_null_literal(input_json, range.first, range.second)) { return; }
        }
        // Hard type mismatch (scalar string/number/bool, object, or any non-array, non-null value):
        // nullify the entire row. Row = last line-begin id not exceeding `node_id`.
        auto const row_idx =
          thrust::upper_bound(
            thrust::seq, line_begin_indices, line_begin_indices + num_rows, node_id) -
          line_begin_indices - 1;
        should_be_nullified[row_idx] = true;
      });
  }

  auto [null_mask, null_count] =
    create_null_mask(input.size(),
                     should_be_nullified,
                     tokens,
                     token_positions,
                     node_token_ids,
                     parent_node_ids,
                     cudf::device_span<NodeIndexT const>{line_begin_indices.data(),
                                                         static_cast<std::size_t>(input.size())},
                     stream,
                     mr);

  auto result = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::LIST},
                                               input.size(),
                                               rmm::device_buffer{},
                                               std::move(null_mask),
                                               null_count,
                                               std::move(list_children));

  // A row nulled for a hard value type mismatch still spans the key/value pairs that were
  // materialized before the mismatch was detected, so the outer list can carry non-empty nulls.
  // This manual assembly skips the factory's `purge_nonempty_nulls` scan, so sanitize the outer
  // list here (the children are already empty under their nulls by construction). Gated on a cheap
  // check so the common all-valid / empty-null paths keep the zero-copy build.
  if (null_count > 0 && cudf::has_nonempty_nulls(result->view(), stream)) {
    return cudf::purge_nonempty_nulls(result->view(), stream, mr);
  }
  return result;
}

}  // namespace spark_rapids_jni
