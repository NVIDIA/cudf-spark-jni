/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "get_json_object.hpp"
#include "json_parser.cuh"
#include "nvtx_ranges.hpp"
#include "utilities/iterator.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/functional>
#include <cuda/hierarchy>
#include <cuda/std/algorithm>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/stream>
#include <cuda_runtime_api.h>
#include <thrust/uninitialized_fill.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>

namespace spark_rapids_jni {

inline constexpr std::size_t warp_size = 32;

namespace detail {

__device__ inline void set_device_flag(int8_t* flag)
{
  cuda::atomic_ref<int8_t, cuda::thread_scope_device> ref(*flag);
  ref.store(1, cuda::memory_order_relaxed);
}

enum class write_style : int8_t { RAW, QUOTED, FLATTEN };

struct path_instruction {
  __host__ __device__ inline path_instruction(path_instruction_type _type) : type(_type) {}

  cudf::string_view name;  // used when type is named type
  int index{-1};           // used when type is index
  path_instruction_type type;
};

// Objects are copied whole; only generated arrays need comma state.
class json_generator {
 public:
  __device__ json_generator(int _offset = 0) : offset(_offset), output_len(0) {}

  // The child writes after this generator in the shared output buffer.
  __device__ json_generator new_child_generator() const
  {
    return json_generator(offset + output_len);
  }

  __device__ void write_start_array(char* out_begin)
  {
    try_write_comma(out_begin);
    // Starting an array after `[[1,2,3]` yields `[[1,2,3],[`.

    out_begin[offset + output_len] = '[';
    output_len++;
    array_depth++;
    is_curr_array_empty = true;
  }

  __device__ void write_end_array(char* out_begin)
  {
    out_begin[offset + output_len] = ']';
    output_len++;
    array_depth--;
    is_curr_array_empty = false;
  }

  // Delay outer brackets until the wildcard match count is known.
  __device__ void write_first_start_array_without_output()
  {
    array_depth++;
    is_curr_array_empty = true;
  }

  __device__ inline bool need_comma() const { return (array_depth > 0 && !is_curr_array_empty); }

  __device__ void try_write_comma(char* out_begin)
  {
    if (need_comma()) {
      out_begin[offset + output_len] = ',';
      output_len++;
    }
  }

  __device__ bool copy_current_structure(json_parser& parser, char* out_begin)
  {
    try_write_comma(out_begin);

    if (array_depth > 0) { is_curr_array_empty = false; }

    auto [b, copy_len] = parser.copy_current_structure(out_begin + offset + output_len);
    output_len += copy_len;
    return b;
  }

  // String escapes must be decoded into output rather than exposed as a source span.
  __device__ void write_raw(json_parser& parser, char* out_begin)
  {
    if (array_depth > 0) { is_curr_array_empty = false; }

    auto copied = parser.write_token_text(out_begin + offset + output_len, escape_style::UNESCAPED);
    output_len += copied;
  }

  /**
   * Make room before the in-place child for deferred punctuation: [1 + 2,3,4 becomes
   * [1,2,3,4 or [1,[2,3,4]. A single wildcard match stays unwrapped.
   */
  __device__ void write_child_raw_value(char* child_block_begin,
                                        int child_block_len,
                                        bool write_outer_array_tokens)
  {
    bool insert_comma = need_comma();

    if (array_depth > 0) { is_curr_array_empty = false; }

    if (write_outer_array_tokens) {
      if (insert_comma) {
        *(child_block_begin + child_block_len + 2) = ']';
        move_forward(child_block_begin, child_block_len, 2);
        *(child_block_begin + 1) = '[';
        *(child_block_begin)     = ',';
      } else {
        *(child_block_begin + child_block_len + 1) = ']';
        move_forward(child_block_begin, child_block_len, 1);
        *(child_block_begin) = '[';
      }
    } else {
      if (insert_comma) {
        move_forward(child_block_begin, child_block_len, 1);
        *(child_block_begin) = ',';
      } else {
      }
    }

    if (insert_comma) { output_len++; }
    if (write_outer_array_tokens) { output_len += 2; }
    output_len += child_block_len;
  }

  // Copy overlapping bytes backward: shifting 1 2 0 0 by one yields 1 1 2 0.
  static __device__ void move_forward(char* begin, size_t len, int forward)
  {
    char* pos = begin + len + forward - 1;
    char* e   = begin + forward - 1;
    while (pos > e) {
      *pos = *(pos - forward);
      pos--;
    }
  }

  __device__ inline int get_offset() const { return offset; }
  __device__ inline int get_output_len() const { return output_len; }

  // Discard failed output logically; stale bytes in the shared buffer are not erased.
  __device__ inline void set_output_len_zero() { output_len = 0; }

  __device__ inline void set_output_len(size_t len) { output_len = len; }

 private:
  int offset;  // global buffer offset
  int output_len;
  int array_depth = 0;
  // Closing an inner array makes its parent non-empty, so one emptiness flag suffices.
  bool is_curr_array_empty;
};

__device__ inline bool path_is_empty(size_t path_size) { return path_size == 0; }

__device__ inline bool path_match_element(cudf::device_span<path_instruction const> path,
                                          path_instruction_type path_type0)
{
  if (path.size() < 1) { return false; }
  return path.data()[0].type == path_type0;
}

__device__ inline bool path_match_elements(cudf::device_span<path_instruction const> path,
                                           path_instruction_type path_type0,
                                           path_instruction_type path_type1)
{
  if (path.size() < 2) { return false; }
  return path.data()[0].type == path_type0 && path.data()[1].type == path_type1;
}

__device__ inline cuda::std::tuple<bool, int> path_match_index(
  cudf::device_span<path_instruction const> path)
{
  auto match = path_match_element(path, path_instruction_type::INDEX);
  if (match) {
    return cuda::std::make_tuple(true, path.data()[0].index);
  } else {
    return cuda::std::make_tuple(false, 0);
  }
}

__device__ inline cuda::std::tuple<bool, int> path_match_index_wildcard(
  cudf::device_span<path_instruction const> path)
{
  auto match =
    path_match_elements(path, path_instruction_type::INDEX, path_instruction_type::WILDCARD);
  if (match) {
    return cuda::std::make_tuple(true, path.data()[0].index);
  } else {
    return cuda::std::make_tuple(false, 0);
  }
}

// Spark JSONPath cases determine how parents resume after a child returns.
enum class evaluation_case_path : int8_t {
  INVALID                                           = -1,
  START_ARRAY___EMPTY_PATH___FLATTEN_STYLE          = 2,
  START_OBJECT___MATCHED_NAME_PATH                  = 4,
  START_ARRAY___MATCHED_DOUBLE_WILDCARD             = 5,
  START_ARRAY___MATCHED_WILDCARD___STYLE_NOT_QUOTED = 6,
  START_ARRAY___MATCHED_WILDCARD                    = 7,
  START_ARRAY___MATCHED_INDEX_AND_WILDCARD          = 8,
  START_ARRAY___MATCHED_INDEX                       = 9
};

struct context {
  json_generator g;
  // Saved across returns from children in case 6.
  json_generator child_g;
  cudf::device_span<path_instruction const> path;
  // Match count for case 6; other wildcard cases clamp successful output to 1.
  int dirty;
  evaluation_case_path case_path;
  json_token token;
  write_style style;
  // One-time setup must not repeat when a child returns to this context.
  bool is_first_enter;
  bool task_is_done;
};

/**
 * @brief Evaluates the command buffer against one row.
 * @return {success, result bytes on success or write extent on failure, maximum write extent}
 *
 * The extent includes discarded child writes so retry slots cover every touched byte.
 */
__device__ cuda::std::tuple<bool, cudf::size_type, cudf::size_type> evaluate_path(
  json_parser& p,
  cudf::device_span<path_instruction const> path_commands,
  char* out_buf,
  int8_t* max_path_depth_exceeded)
{
  // The root needs a frame in addition to the path frames.
  context stack[MAX_JSON_PATH_DEPTH + 1];
  int stack_size = 0;

  // Include abandoned wildcard writes.
  int max_footprint = 0;

  auto const parse_failure = [&](json_generator const& g) {
    auto const extent = g.get_offset() + g.get_output_len();
    return cuda::std::tuple{false, extent, cuda::std::max(extent, max_footprint)};
  };
  // A failed wildcard may already have written beyond the parent.
  auto const parse_failure_with_child = [&](json_generator const& g,
                                            json_generator const& child_g) {
    auto const extent = cuda::std::max(g.get_offset() + g.get_output_len(),
                                       child_g.get_offset() + child_g.get_output_len());
    return cuda::std::tuple{false, extent, cuda::std::max(extent, max_footprint)};
  };

  p.next_token();
  if (json_token::ERROR == p.get_current_token()) { return parse_failure(json_generator{}); }

  auto const push_context = [&](evaluation_case_path _case_path,
                                json_generator _g,
                                write_style _style,
                                cudf::device_span<path_instruction const> _path) {
    if (stack_size > MAX_JSON_PATH_DEPTH) {
      set_device_flag(max_path_depth_exceeded);
      // The host rejects this batch, so partial output need not be repaired.
      return;
    }
    auto& ctx          = stack[stack_size++];
    ctx.g              = cuda::std::move(_g);
    ctx.path           = cuda::std::move(_path);
    ctx.dirty          = 0;
    ctx.case_path      = _case_path;
    ctx.token          = p.get_current_token();
    ctx.style          = _style;
    ctx.is_first_enter = true;
    ctx.task_is_done   = false;
  };

  push_context(evaluation_case_path::INVALID, json_generator{}, write_style::RAW, path_commands);

  while (stack_size > 0) {
    auto& ctx = stack[stack_size - 1];
    if (!ctx.task_is_done) {
      // Case 1: terminal string with raw output.
      if (json_token::VALUE_STRING == ctx.token && path_is_empty(ctx.path.size()) &&
          ctx.style == write_style::RAW) {
        ctx.g.write_raw(p, out_buf);
        ctx.dirty        = 1;
        ctx.task_is_done = true;
      }
      // Case 2: flatten a terminal array.
      else if (json_token::START_ARRAY == ctx.token && path_is_empty(ctx.path.size()) &&
               ctx.style == write_style::FLATTEN) {
        if (json_token::END_ARRAY != p.next_token()) {
          if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }
          push_context(evaluation_case_path::START_ARRAY___EMPTY_PATH___FLATTEN_STYLE,
                       ctx.g,
                       ctx.style,
                       cudf::device_span<path_instruction const>{});
        } else {
          ctx.task_is_done = true;
        }
      }
      // Case 3: terminal value.
      else if (path_is_empty(ctx.path.size())) {
        if (!(ctx.g.copy_current_structure(p, out_buf))) { return parse_failure(ctx.g); }
        ctx.dirty        = 1;
        ctx.task_is_done = true;
      }
      // Case 4: named object field.
      else if (json_token::START_OBJECT == ctx.token &&
               ctx.path.front().type == path_instruction_type::NAMED) {
        if (!ctx.is_first_enter) {
          // Validate the remaining fields even after a successful match.
          if (ctx.dirty > 0) {
            while (json_token::END_OBJECT != p.next_token()) {
              if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }

              p.next_token();
              if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }

              if (!p.try_skip_children()) { return parse_failure(ctx.g); }
            }
          }
          ctx.task_is_done = true;
        } else {
          ctx.is_first_enter = false;
          // A missed child path must not fall back to a later duplicate.
          bool found_expected_child = false;
          auto const to_match_name  = ctx.path.front().name;
          while (true) {
            auto const is_name_matched = p.parse_next_token_with_matching(to_match_name);
            if (json_token::END_OBJECT == p.get_current_token()) { break; }

            if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }

            if (is_name_matched) {
              p.next_token();
              if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }

              // A null first occurrence does not fall back to a later duplicate.
              if (json_token::VALUE_NULL == p.get_current_token()) { return parse_failure(ctx.g); }
              push_context(evaluation_case_path::START_OBJECT___MATCHED_NAME_PATH,
                           ctx.g,
                           ctx.style,
                           {ctx.path.data() + 1, ctx.path.size() - 1});
              found_expected_child = true;
              break;
            } else {
              p.next_token();
              // JSON validation check
              if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }

              // current child is not expected, skip current child
              if (!p.try_skip_children()) {
                // JSON validation check
                return parse_failure(ctx.g);
              }
            }
          }
          if (!found_expected_child) {
            ctx.task_is_done = true;
            ctx.dirty        = false;
          }
        }
      }
      // Case 5: consecutive array wildcards.
      else if (json_token::START_ARRAY == ctx.token &&
               path_match_elements(
                 ctx.path, path_instruction_type::WILDCARD, path_instruction_type::WILDCARD)) {
        // Consecutive wildcards flatten terminal arrays to match Hive rather than preserve nesting.
        if (ctx.is_first_enter) {
          ctx.is_first_enter = false;
          ctx.g.write_start_array(out_buf);
        }

        if (p.next_token() != json_token::END_ARRAY) {
          if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }
          push_context(evaluation_case_path::START_ARRAY___MATCHED_DOUBLE_WILDCARD,
                       ctx.g,
                       write_style::FLATTEN,
                       {ctx.path.data() + 2, ctx.path.size() - 2});
        } else {
          ctx.g.write_end_array(out_buf);
          ctx.task_is_done = true;
        }
      }
      // Case 6: array wildcard outside quoted output.
      else if (json_token::START_ARRAY == ctx.token &&
               path_match_element(ctx.path, path_instruction_type::WILDCARD) &&
               ctx.style != write_style::QUOTED) {
        // Raw strings would make multiple matches invalid JSON array items.
        write_style next_style = write_style::RAW;
        switch (ctx.style) {
          case write_style::RAW: next_style = write_style::QUOTED; break;
          case write_style::FLATTEN: next_style = write_style::FLATTEN; break;
          case write_style::QUOTED: next_style = write_style::QUOTED;  // unreachable
        }

        // Hive leaves one wildcard match unwrapped; multiple matches need outer brackets.
        json_generator child_g;
        if (ctx.is_first_enter) {
          ctx.is_first_enter = false;
          child_g            = ctx.g.new_child_generator();
          child_g.write_first_start_array_without_output();
        } else {
          child_g = ctx.child_g;
        }

        if (p.next_token() != json_token::END_ARRAY) {
          if (json_token::ERROR == p.get_current_token()) {
            // Include child writes or retry can miss the true overrun.
            return parse_failure_with_child(ctx.g, child_g);
          }
          push_context(evaluation_case_path::START_ARRAY___MATCHED_WILDCARD___STYLE_NOT_QUOTED,
                       child_g,
                       next_style,
                       {ctx.path.data() + 1, ctx.path.size() - 1});
        } else {
          char* child_g_start = out_buf + child_g.get_offset();
          int child_g_len     = child_g.get_output_len();
          if (ctx.dirty > 1) {
            ctx.g.write_child_raw_value(
              child_g_start, child_g_len, /* write_outer_array_tokens */ true);
          } else if (ctx.dirty == 1) {
            ctx.g.write_child_raw_value(
              child_g_start, child_g_len, /* write_outer_array_tokens */ false);
          }

          ctx.task_is_done = true;
        }
      }
      // Case 7: array wildcard with quoted output.
      else if (json_token::START_ARRAY == ctx.token &&
               path_match_element(ctx.path, path_instruction_type::WILDCARD)) {
        if (ctx.is_first_enter) {
          ctx.is_first_enter = false;
          ctx.g.write_start_array(out_buf);
        }
        if (p.next_token() != json_token::END_ARRAY) {
          if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }

          push_context(evaluation_case_path::START_ARRAY___MATCHED_WILDCARD,
                       ctx.g,
                       write_style::QUOTED,
                       {ctx.path.data() + 1, ctx.path.size() - 1});
        } else {
          ctx.g.write_end_array(out_buf);
          ctx.task_is_done = true;
        }
      }
      // Case 8: array index followed by a wildcard.
      else if (json_token::START_ARRAY == ctx.token &&
               cuda::std::get<0>(path_match_index_wildcard(ctx.path))) {
        int idx = cuda::std::get<1>(path_match_index_wildcard(ctx.path));

        p.next_token();
        if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }
        ctx.is_first_enter = false;

        int i = idx;
        while (i > 0) {
          if (p.get_current_token() == json_token::END_ARRAY) { return parse_failure(ctx.g); }

          if (!p.try_skip_children()) { return parse_failure(ctx.g); }

          p.next_token();
          if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }

          --i;
        }

        push_context(evaluation_case_path::START_ARRAY___MATCHED_INDEX_AND_WILDCARD,
                     ctx.g,
                     write_style::QUOTED,
                     {ctx.path.data() + 1, ctx.path.size() - 1});
      }
      // Case 9: array index.
      else if (json_token::START_ARRAY == ctx.token &&
               cuda::std::get<0>(path_match_index(ctx.path))) {
        int idx = cuda::std::get<1>(path_match_index(ctx.path));

        p.next_token();
        if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }

        int i = idx;
        while (i > 0) {
          if (p.get_current_token() == json_token::END_ARRAY) { return parse_failure(ctx.g); }

          if (!p.try_skip_children()) { return parse_failure(ctx.g); }

          p.next_token();
          if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }

          --i;
        }

        push_context(evaluation_case_path::START_ARRAY___MATCHED_INDEX,
                     ctx.g,
                     ctx.style,
                     {ctx.path.data() + 1, ctx.path.size() - 1});
      }
      // Case 12: no match.
      else {
        if (!p.try_skip_children()) { return parse_failure(ctx.g); }
        ctx.dirty        = 0;
        ctx.task_is_done = true;
      }
    } else {
      stack_size--;

      if (stack_size == 0) { break; }

      auto& p_ctx = stack[stack_size - 1];

      switch (ctx.case_path) {
          // Cases 2, 5, 7: only case 6 needs a match count rather than a success flag.
        case evaluation_case_path::START_ARRAY___EMPTY_PATH___FLATTEN_STYLE:
        case evaluation_case_path::START_ARRAY___MATCHED_DOUBLE_WILDCARD:
        case evaluation_case_path::START_ARRAY___MATCHED_WILDCARD: {
          p_ctx.dirty += ctx.dirty;
          if (p_ctx.dirty > 0) { p_ctx.dirty = 1; }
          p_ctx.g = ctx.g;
          break;
        }

        case evaluation_case_path::START_OBJECT___MATCHED_NAME_PATH: {
          p_ctx.dirty = ctx.dirty;

          p_ctx.g = ctx.g;

          break;
        }

        case evaluation_case_path::START_ARRAY___MATCHED_WILDCARD___STYLE_NOT_QUOTED: {
          p_ctx.dirty += ctx.dirty;
          p_ctx.child_g = ctx.g;
          max_footprint =
            cuda::std::max(max_footprint, ctx.g.get_offset() + ctx.g.get_output_len());
          break;
        }

          // The indexed child returned; validate the rest of the array.
        case evaluation_case_path::START_ARRAY___MATCHED_INDEX_AND_WILDCARD:
        case evaluation_case_path::START_ARRAY___MATCHED_INDEX: {
          p_ctx.dirty += ctx.dirty;

          while (p.next_token() != json_token::END_ARRAY) {
            if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }
            if (!p.try_skip_children()) { return parse_failure(ctx.g); }
          }
          p_ctx.task_is_done = true;
          p_ctx.g            = ctx.g;
          break;
        }

        default:;
      }
    }
  }

  auto const success = stack[0].dirty > 0;

  auto const root_extent     = stack[0].g.get_offset() + stack[0].g.get_output_len();
  auto const write_footprint = cuda::std::max(max_footprint, root_extent);
  // Discarded writes affect retry sizing, never a valid result's length.
  return {success, success ? root_extent : write_footprint, write_footprint};
}

struct named_path_selection {
  bool valid;
  char_range value;
};

// Null duplicates preserve an earlier value; serialize only the final selection.
__device__ named_path_selection select_last_non_null(json_parser& p,
                                                     cudf::string_view name,
                                                     int8_t* max_path_depth_exceeded)
{
  auto selected = char_range::null();
  auto token    = p.next_token();
  if (token == json_token::ERROR) {
    if (p.max_nesting_depth_exceeded()) { set_device_flag(max_path_depth_exceeded); }
    return {false, selected};
  }

  if (token != json_token::START_OBJECT) { return {false, selected}; }

  while (true) {
    auto const is_name_matched = p.parse_next_token_with_matching(name);
    token                      = p.get_current_token();
    if (token == json_token::ERROR) {
      if (p.max_nesting_depth_exceeded()) { set_device_flag(max_path_depth_exceeded); }
      return {false, char_range::null()};
    }
    // The first object is validated to its end; trailing input is ignored.
    if (token == json_token::END_OBJECT) { return {true, selected}; }

    token = p.next_token();
    if (token == json_token::ERROR) {
      if (p.max_nesting_depth_exceeded()) { set_device_flag(max_path_depth_exceeded); }
      return {false, char_range::null()};
    }
    auto const value_begin = p.current_range().data();
    if (!p.try_skip_children()) {
      if (p.max_nesting_depth_exceeded()) { set_device_flag(max_path_depth_exceeded); }
      return {false, char_range::null()};
    }
    if (is_name_matched && token != json_token::VALUE_NULL) {
      auto const value_end_range = p.current_range();
      auto const value_end       = value_end_range.data() + value_end_range.size();
      selected = char_range{value_begin, static_cast<cudf::size_type>(value_end - value_begin)};
    }
  }
}

// One path's borrowed instructions, row-slot offsets, output views, scratch and retry marker.
struct json_path_processing_data {
  cudf::device_span<path_instruction const> path_commands;
  cudf::detail::input_offsetalator offsets;
  // First input offset for sliced scratch; exact-size retry offsets start at zero.
  int64_t base_offset;
  cuda::std::pair<char const*, cudf::size_type>* out_stringviews;
  char* out_buf;
  int8_t* has_out_of_bound;
};

/**
 * Evaluates one (row, path) per active thread. Short multi-path batches pack rows into warps.
 * Keep both launch bounds: the NVCC default spilled; the residency target can allow more
 * registers and fewer instructions, but is not measured occupancy. Retain path data through
 * completion. Retry overwrites pairs using new scratch; the host checks depth before sizes.
 */
template <int block_size, int min_block_per_sm, bool use_last_non_null>
__launch_bounds__(block_size, min_block_per_sm) CUDF_KERNEL
  void get_json_object_kernel(cudf::column_device_view input,
                              cudf::device_span<json_path_processing_data> path_data,
                              std::size_t num_threads_per_row,
                              int8_t* max_path_depth_exceeded)
{
  auto const tidx    = cuda::gpu_thread.index_as<cudf::thread_index_type>(cuda::grid).x;
  auto const row_idx = tidx / num_threads_per_row;
  if (row_idx >= input.size()) { return; }

  auto const path_idx = tidx % num_threads_per_row;
  if (path_idx >= path_data.size()) { return; }

  auto const& path = path_data[path_idx];
  if (input.is_null(row_idx)) {
    if (path.out_stringviews) { path.out_stringviews[row_idx] = {nullptr, 0}; }
    return;
  }

  // Slice offsets are absolute, but scratch owns only the selected bytes.
  char* const dst                 = path.out_buf + (path.offsets[row_idx] - path.base_offset);
  auto const max_size             = path.offsets[row_idx + 1] - path.offsets[row_idx];
  bool is_valid                   = false;
  cudf::size_type out_size        = 0;
  cudf::size_type write_footprint = 0;

  auto const str = input.element<cudf::string_view>(row_idx);
  if (str.size_bytes() > 0) {
    json_parser p{char_range{str}};
    if constexpr (use_last_non_null) {
      auto const selection =
        select_last_non_null(p, path.path_commands.front().name, max_path_depth_exceeded);
      if (selection.valid && !selection.value.is_null()) {
        json_parser selected_parser{selection.value};
        cuda::std::tie(is_valid, out_size, write_footprint) =
          evaluate_path(selected_parser,
                        cudf::device_span<path_instruction const>{},
                        dst,
                        max_path_depth_exceeded);
      }
    } else {
      cuda::std::tie(is_valid, out_size, write_footprint) =
        evaluate_path(p, path.path_commands, dst, max_path_depth_exceeded);
    }

    // The host throws on depth overflow before using any output.
    if (p.max_nesting_depth_exceeded()) {
      set_device_flag(max_path_depth_exceeded);
      return;
    }

    // The footprint also covers discarded child writes.
    if (write_footprint > max_size) { set_device_flag(path.has_out_of_bound); }
  }

  // OOB rows carry the write footprint for retry sizing; others carry the result.
  if (path.out_stringviews) {
    path.out_stringviews[row_idx] =
      cuda::std::pair{is_valid ? dst : nullptr,
                      write_footprint > max_size ? write_footprint : (is_valid ? out_size : 0)};
  }
}

struct kernel_launcher {
  static void exec(cudf::column_device_view const& input,
                   cudf::device_span<json_path_processing_data> path_data,
                   int8_t* max_path_depth_exceeded,
                   bool use_last_non_null,
                   cuda::stream_ref stream)
  {
    // 128 threads / 8 blocks per SM beat 256 / 4.
    static constexpr int block_size       = 128;
    static constexpr int min_block_per_sm = 8;
    // Power-of-two groups pack short path batches: 3 paths use 4 threads; 1 and 33 use 32 and 64.
    auto const num_threads_per_row = path_data.size() <= warp_size && path_data.size() > 1
                                       ? std::bit_ceil(path_data.size())
                                       : cuda::ceil_div(path_data.size(), warp_size) * warp_size;
    auto const num_blocks =
      cuda::ceil_div(num_threads_per_row * input.size(), static_cast<std::size_t>(block_size));
    // Separate instantiations keep the json_tuple scan out of ordinary path-evaluation kernels.
    if (use_last_non_null) {
      get_json_object_kernel<block_size, min_block_per_sm, true>
        <<<num_blocks, block_size, 0, stream.get()>>>(
          input, path_data, num_threads_per_row, max_path_depth_exceeded);
    } else {
      get_json_object_kernel<block_size, min_block_per_sm, false>
        <<<num_blocks, block_size, 0, stream.get()>>>(
          input, path_data, num_threads_per_row, max_path_depth_exceeded);
    }
  }
};

using path_batch =
  std::vector<cudf::host_span<std::tuple<path_instruction_type, std::string, int32_t> const>>;

// Retain the pinned source and device commands through the batch's error readback.
struct path_staging {
  std::vector<cudf::device_span<path_instruction const>> paths;
  std::optional<cudf::detail::host_vector<std::byte>> host;
  std::optional<cuda::device_buffer<std::byte>> device;
};

path_staging stage_commands(path_batch const& paths,
                            std::vector<std::size_t> const& indices,
                            cuda::stream_ref stream)
{
  path_staging out;
  out.paths.resize(paths.size());
  std::size_t num_instructions = 0, name_bytes = 0;
  for (auto const i : indices) {
    num_instructions += paths[i].size();
    for (auto const& [type, name, index] : paths[i]) {
      if (type == path_instruction_type::NAMED) { name_bytes += name.size(); }
    }
  }
  if (num_instructions == 0) { return out; }
  auto const name_begin = num_instructions * sizeof(path_instruction);
  auto const bytes      = name_begin + name_bytes;
  out.host.emplace(cudf::detail::make_pinned_vector_async<std::byte>(bytes, stream));
  out.device.emplace(stream, cudf::get_current_device_resource_ref(), bytes, cuda::no_init);
  auto* const h           = out.host->data();
  auto* const d           = out.device->data();
  std::size_t command_pos = 0, name_pos = name_begin;
  for (auto const i : indices) {
    out.paths[i] = {reinterpret_cast<path_instruction const*>(d + command_pos), paths[i].size()};
    for (auto const& [type, name, index] : paths[i]) {
      path_instruction cmd{type};
      if (type == path_instruction_type::INDEX) {
        cmd.index = index;
      } else if (type == path_instruction_type::NAMED) {
        cmd.name = cudf::string_view{reinterpret_cast<char const*>(d + name_pos),
                                     static_cast<cudf::size_type>(name.size())};
        std::memcpy(h + name_pos, name.data(), name.size());
        name_pos += name.size();
      } else if (type != path_instruction_type::WILDCARD) {
        CUDF_FAIL("Invalid path instruction type");
      }
      std::memcpy(h + command_pos, &cmd, sizeof(cmd));
      command_pos += sizeof(cmd);
    }
  }
  CUDF_CUDA_TRY(cudaMemcpyAsync(d, h, bytes, cudaMemcpyDefault, stream.get()));
  return out;
}

// Scratch uses only the selected offsets, plus slack for speculative writes in the last row.
__global__ void scratch_size_kernel(cudf::detail::input_offsetalator offsets,
                                    cudf::size_type rows,
                                    int64_t* stats)
{
  constexpr int threads = 256;
  __shared__ int64_t maxima[threads];
  int64_t max_row = 0;
  for (int64_t i = threadIdx.x; i < rows; i += threads) {
    auto const row   = static_cast<cudf::size_type>(i);
    auto const bytes = offsets[row + 1] - offsets[row];
    if (bytes > max_row) { max_row = bytes; }
  }
  maxima[threadIdx.x] = max_row;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < threads; ++i) {
      max_row = cuda::std::max(max_row, maxima[i]);
    }
    stats[0] = max_row;
    stats[1] = offsets[rows] - offsets[0];
    stats[2] = offsets[0];
  }
}

// Return scratch bytes and the absolute slice base.
cuda::std::pair<int64_t, int64_t> calc_scratch_size(cudf::strings_column_view const& input,
                                                    cudf::detail::input_offsetalator offsets,
                                                    cuda::stream_ref stream)
{
  rmm::device_uvector<int64_t> stats(3, stream);
  scratch_size_kernel<<<1, 256, 0, stream.get()>>>(offsets, input.size(), stats.data());
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  auto const host = cudf::detail::make_pinned_vector(stats, stream);
  return {host[0] * 10 + host[1], host[2]};
}

// Depth errors take precedence over retry and unwritten output pairs.
bool check_error(cudf::host_span<int8_t const> flags)
{
  CUDF_EXPECTS(flags.back() == 0, "The processed input has nesting depth exceeds depth limit.");
  return std::none_of(flags.begin(), flags.end(), [](auto flag) { return flag != 0; });
}

namespace {

constexpr int k_merged_max_depth         = 2;
constexpr std::size_t k_merged_min_paths = 16;
// Launch overhead dominates small inputs; four warps times 16 B per path gives 768 in 48 KiB.
constexpr int k_min_rows_per_family      = 32;
constexpr std::size_t k_merged_max_paths = 768;

// A borrowed trie maps object scopes to name edges and compact output IDs.
// The grouped upload owns all arrays, including descendant IDs killed by first matches.
struct merged_trie_view {
  int32_t const* children_beg;
  int32_t const* children_cnt;
  int32_t const* child_node;
  int32_t const* paths_beg;
  int32_t const* paths_cnt;
  int32_t const* flat_children;
  int32_t const* terminal_ids;
  char const* names;
  int32_t const* subtree_beg;
  int32_t const* subtree_cnt;
  int32_t const* subtree_pids;
  // Flat-order names avoid a slot-ID lookup.
  int32_t const* flat_name_off;
  int32_t const* flat_name_len;
};

// A negative begin means unset; dead paths cannot be revived by later duplicate keys.
// Only the row leader writes; the warp barrier orders these reads before gather.
struct alignas(8) merged_path_out {
  uint64_t range = 0xffffffffu;
  int8_t kind    = 0;
};

constexpr int8_t k_merged_kind_string = 1;
constexpr int8_t k_merged_kind_dead   = 3;

constexpr int k_merged_block_size        = 128;
constexpr int k_merged_min_blocks_per_sm = 8;

struct merged_rec_view {
  int32_t begin;
  int32_t end;
  int8_t kind;
};

__device__ __forceinline__ merged_rec_view load_record(merged_path_out const& r)
{
  return {static_cast<int32_t>(r.range), static_cast<int32_t>(r.range >> 32), r.kind};
}

__device__ __forceinline__ void store_record(merged_path_out& r,
                                             int32_t begin,
                                             int32_t end,
                                             int8_t kind)
{
  r = {static_cast<uint32_t>(begin) | (static_cast<uint64_t>(end) << 32), kind};
}

// Escapes compare through the parser instead of raw-byte equality.
__device__ __forceinline__ int32_t merged_exact_scan(merged_trie_view const& trie,
                                                     char_range const& raw,
                                                     int32_t from,
                                                     int32_t to)
{
  for (int32_t c = from; c < to; ++c) {
    int32_t const slot = trie.flat_children[c];
    cudf::string_view const cand(trie.names + trie.flat_name_off[c], trie.flat_name_len[c]);
    if (json_parser::match_decoded_name(raw, cand)) { return slot; }
  }
  return -1;
}

// Advance one member; -1 also covers END_OBJECT and ERROR.
__device__ inline int32_t merged_scan_member(json_parser& p,
                                             merged_trie_view const& trie,
                                             int32_t node)
{
  auto const begin = trie.children_beg[node];
  auto const count = trie.children_cnt[node];
  if (count == 1) {
    bool const match = p.parse_next_token_with_matching(
      cudf::string_view{trie.names + trie.flat_name_off[begin], trie.flat_name_len[begin]});
    return match && p.get_current_token() == json_token::FIELD_NAME ? trie.flat_children[begin]
                                                                    : -1;
  }
  return p.next_token() == json_token::FIELD_NAME
           ? merged_exact_scan(trie, p.current_range(), begin, begin + count)
           : -1;
}

// One trie walk per row group. All its lanes parse in lockstep; only its leader records
// intervals. First matches own their subtree even for nulls or missing descendants.

__launch_bounds__(k_merged_block_size, k_merged_min_blocks_per_sm) CUDF_KERNEL
  void get_json_object_kernel_merged_named(cudf::column_device_view input,
                                           cudf::device_span<json_path_processing_data> paths,
                                           merged_trie_view trie,
                                           int8_t* depth_error)
{
  using record        = merged_path_out;
  constexpr int width = warp_size;
  auto const row = static_cast<cudf::size_type>(cudf::detail::grid_1d::global_thread_id() / width);
  if (row >= input.size()) { return; }
  auto const row_lane     = threadIdx.x % width;
  constexpr unsigned mask = 0xffffffffu;
  auto const count        = static_cast<int32_t>(paths.size());
  extern __shared__ __align__(8) unsigned char records[];
  auto* const out  = reinterpret_cast<record*>(records) + (threadIdx.x / width) * count;
  bool const owner = row_lane == 0;
  auto const str = input.is_null(row) ? cudf::string_view{} : input.element<cudf::string_view>(row);
  if (str.size_bytes() == 0) {
    for (int32_t i = row_lane; i < count; i += width) {
      paths[i].out_stringviews[row] = {nullptr, 0};
    }
    return;
  }
  if (owner) {
    for (int32_t i = 0; i < count; ++i) {
      out[i] = record{};
    }
  }

  json_parser p{char_range{str}};
  auto const offset = [&] { return static_cast<int32_t>(p.current_range().data() - str.data()); };
  auto const record_slot = [&](int32_t slot, int32_t begin, int32_t end, int8_t kind) {
    for (int32_t i = 0; i < trie.paths_cnt[slot]; ++i) {
      auto& rec      = out[trie.terminal_ids[trie.paths_beg[slot] + i]];
      auto const old = load_record(rec);
      if (old.begin < 0 && old.kind != k_merged_kind_dead) { store_record(rec, begin, end, kind); }
    }
  };
  auto const kill = [&](int32_t slot) {
    for (int32_t i = 0; i < trie.subtree_cnt[slot]; ++i) {
      auto& rec = out[trie.subtree_pids[trie.subtree_beg[slot] + i]];
      if (load_record(rec).begin < 0) { store_record(rec, -1, 0, k_merged_kind_dead); }
    }
  };
  auto const record_subtree = [&](int32_t slot, int32_t begin, int32_t end) {
    record_slot(slot, begin, end, 0);
  };
  bool valid = true;
  if (p.next_token() != json_token::START_OBJECT) {
    valid = p.try_skip_children();
  } else {
    struct frame {
      int32_t node, parent_slot, begin;
    };
    frame stack[k_merged_max_depth];
    int sp = 0, node = 0;
    while (true) {
      auto const slot  = merged_scan_member(p, trie, node);
      auto const token = p.get_current_token();
      if (token == json_token::ERROR) {
        valid = false;
        break;
      }
      if (token == json_token::END_OBJECT) {
        if (sp == 0) { break; }
        auto const fr = stack[--sp];
        if (owner) {
          if (trie.paths_cnt[fr.parent_slot] != 0) {
            record_subtree(fr.parent_slot, fr.begin, offset() + 1);
          }
          // An exhausted child kills its unmatched paths, not its siblings.
          for (int32_t i = 0; i < trie.children_cnt[node]; ++i) {
            kill(trie.flat_children[trie.children_beg[node] + i]);
          }
        }
        node = fr.node;
        continue;
      }
      auto const value = p.next_token();
      if (value == json_token::ERROR) {
        valid = false;
        break;
      }
      if (slot < 0) {
        if (!p.try_skip_children()) {
          valid = false;
          break;
        }
        continue;
      }
      auto const child = trie.child_node[slot];
      auto const begin = offset();
      if (value == json_token::START_OBJECT && child >= 0 && trie.children_cnt[child] > 0) {
        if (sp == k_merged_max_depth) {
          valid = false;
          break;
        }
        stack[sp++] = {node, slot, begin};
        node        = child;
        continue;
      }
      bool const container = value == json_token::START_OBJECT || value == json_token::START_ARRAY;
      if (container && !p.try_skip_children()) {
        valid = false;
        break;
      }
      if (owner) {
        if (value != json_token::VALUE_NULL && trie.paths_cnt[slot] != 0) {
          if (container) {
            record_subtree(slot, begin, offset() + 1);
          } else {
            auto const end  = begin + static_cast<int32_t>(p.current_range().size());
            auto const kind = value == json_token::VALUE_STRING ? k_merged_kind_string : int8_t{0};
            record_slot(slot, begin, end, kind);
          }
        }
        if (container || value == json_token::VALUE_NULL || child >= 0) { kill(slot); }
      }
    }
  }
  if (__shfl_sync(mask, static_cast<int>(p.max_nesting_depth_exceeded()), 0, width)) {
    // Depth overflow throws only for a live path; null/dead first matches already returned.
    if (owner) {
      for (int32_t i = 0; i < count; ++i) {
        if (load_record(out[i]).kind != k_merged_kind_dead) {
          set_device_flag(depth_error);
          break;
        }
      }
    }
    valid = false;
  }
  valid = __shfl_sync(mask, static_cast<int>(valid), 0, width) != 0;
  __syncwarp(mask);
  for (int32_t i = row_lane; i < count; i += width) {
    auto const& path = paths[i];
    auto const rec   = load_record(out[i]);
    if (!valid || rec.begin < 0) {
      path.out_stringviews[row] = {nullptr, 0};
      continue;
    }
    // Use this path's offsets: retry slots can differ even for the same input row.
    auto const base      = path.offsets[row];
    char* const dest     = path.out_buf + (base - path.base_offset);
    auto const* chars    = dest;
    auto const value     = char_range{str.data() + rec.begin, rec.end - rec.begin};
    cudf::size_type size = 0;
    bool ok              = true;
    if (rec.kind == k_merged_kind_string) {
      char_range_reader reader{value};
      size = json_parser::write_string(reader, dest, escape_style::UNESCAPED);
    } else {
      json_parser emitter{value};
      emitter.next_token();
      auto const result = emitter.copy_current_structure(dest);
      ok                = result.first;
      size              = result.second;
    }
    auto const oob = size > path.offsets[row + 1] - base;
    if (oob) { set_device_flag(path.has_out_of_bound); }
    path.out_stringviews[row] = {ok ? chars : nullptr, (ok || oob) ? size : 0};
  }
}

// Host offsets accompany one aligned upload; each section has its native alignment.
struct merged_group_plan {
  std::vector<uint8_t> h_blob;
  std::array<std::size_t, 13> offsets;
  merged_trie_view view{};
  int32_t num_paths;
  int max_trie_depth;
};

merged_group_plan build_merged_named_plan(path_batch const& paths,
                                          std::vector<std::size_t> const& indices,
                                          int max_trie_depth)
{
  CUDF_EXPECTS(indices.size() <= k_merged_max_paths, "Too many paths in the merged trie");
  struct slot_rec {
    std::string_view name;
    int32_t child = -1;
    std::vector<int32_t> terminals, descendants;
  };
  std::vector<slot_rec> slots;
  std::vector<std::vector<int32_t>> nodes(1);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    auto const path = paths[indices[i]];
    CUDF_EXPECTS(!path.empty() && path.size() <= static_cast<std::size_t>(max_trie_depth),
                 "Invalid depth in the merged trie");
    int32_t node = 0;
    for (std::size_t d = 0; d < path.size(); ++d) {
      auto const& [type, name, index] = path[d];
      CUDF_EXPECTS(type == path_instruction_type::NAMED, "The merged trie requires named paths");
      auto const& children = nodes[node];
      auto const it        = std::find_if(
        children.begin(), children.end(), [&](int32_t child) { return slots[child].name == name; });
      int32_t slot = it == children.end() ? -1 : *it;
      if (slot < 0) {
        slot = static_cast<int32_t>(slots.size());
        nodes[node].push_back(slot);
        slots.push_back({name, -1, {}, {}});
      }
      auto& rec = slots[slot];
      rec.descendants.push_back(static_cast<int32_t>(i));
      if (d + 1 == path.size()) {
        rec.terminals.push_back(static_cast<int32_t>(i));
      } else {
        if (rec.child < 0) {
          rec.child = static_cast<int32_t>(nodes.size());
          nodes.emplace_back();
        }
        node = rec.child;
      }
    }
  }
  std::vector<int32_t> children_beg, children_cnt, flat_children, flat_name_off, flat_name_len;
  std::vector<int32_t> child_node(slots.size()), paths_beg(slots.size()), paths_cnt(slots.size()),
    subtree_beg(slots.size()), subtree_cnt(slots.size());
  std::vector<int32_t> terminal_ids, subtree_pids;
  std::string names;
  for (auto& children : nodes) {
    std::stable_sort(children.begin(), children.end(), [&](int32_t a, int32_t b) {
      return slots[a].name.size() < slots[b].name.size();
    });
    children_beg.push_back(static_cast<int32_t>(flat_children.size()));
    children_cnt.push_back(static_cast<int32_t>(children.size()));
    for (auto const s : children) {
      auto const& slot = slots[s];
      flat_children.push_back(s);
      flat_name_off.push_back(static_cast<int32_t>(names.size()));
      flat_name_len.push_back(static_cast<int32_t>(slot.name.size()));
      names += slot.name;
      child_node[s]           = slot.child;
      auto const append_paths = [](std::vector<int32_t>& ids,
                                   std::vector<int32_t> const& selected,
                                   int32_t& begin,
                                   int32_t& count) {
        begin = static_cast<int32_t>(ids.size());
        count = static_cast<int32_t>(selected.size());
        ids.insert(ids.end(), selected.begin(), selected.end());
      };
      append_paths(terminal_ids, slot.terminals, paths_beg[s], paths_cnt[s]);
      append_paths(subtree_pids, slot.descendants, subtree_beg[s], subtree_cnt[s]);
    }
  }
  merged_group_plan plan{{}, {}, {}, static_cast<int32_t>(indices.size()), max_trie_depth};
  struct section {
    void const* data;
    std::size_t size, alignment;
  };
  auto const bytes_of = [](auto const& v) {
    using T = std::remove_reference_t<decltype(v[0])>;
    return section{v.data(), v.size() * sizeof(T), alignof(T)};
  };
  auto const sections = std::array{bytes_of(children_beg),
                                   bytes_of(children_cnt),
                                   bytes_of(child_node),
                                   bytes_of(paths_beg),
                                   bytes_of(paths_cnt),
                                   bytes_of(flat_children),
                                   bytes_of(terminal_ids),
                                   bytes_of(names),
                                   bytes_of(subtree_beg),
                                   bytes_of(subtree_cnt),
                                   bytes_of(subtree_pids),
                                   bytes_of(flat_name_off),
                                   bytes_of(flat_name_len)};
  std::size_t total   = 0;
  for (std::size_t i = 0; i < sections.size(); ++i) {
    auto const alignment = sections[i].alignment;
    total                = (total + alignment - 1) & ~(alignment - 1);
    plan.offsets[i]      = total;
    total += sections[i].size;
  }
  plan.h_blob.resize(total);
  for (std::size_t i = 0; i < sections.size(); ++i) {
    auto const [data, bytes, alignment] = sections[i];
    if (bytes != 0) { std::memcpy(plan.h_blob.data() + plan.offsets[i], data, bytes); }
  }
  return plan;
}

void bind_merged_plan_view(merged_group_plan& plan, uint8_t const* base)
{
  auto const& s   = plan.offsets;
  auto const ints = [&](int i) { return reinterpret_cast<int32_t const*>(base + s[i]); };
  plan.view       = {ints(0),
                     ints(1),
                     ints(2),
                     ints(3),
                     ints(4),
                     ints(5),
                     ints(6),
                     reinterpret_cast<char const*>(base + s[7]),
                     ints(8),
                     ints(9),
                     ints(10),
                     ints(11),
                     ints(12)};
}

// Both kernels require the bound trie and group-ordered path data through completion.
void launch_merged_named(cudf::column_device_view const& input,
                         cudf::device_span<json_path_processing_data> paths,
                         merged_group_plan const& plan,
                         int8_t* depth_error,
                         cuda::stream_ref stream)
{
  constexpr auto warps = k_merged_block_size / warp_size;
  static_assert(warps * k_merged_max_paths * sizeof(merged_path_out) <= 48 * 1024);
  auto const rows   = static_cast<std::size_t>(input.size());
  auto const smem   = warps * plan.num_paths * sizeof(merged_path_out);
  auto const blocks = (rows + warps - 1) / warps;
  get_json_object_kernel_merged_named<<<blocks, k_merged_block_size, smem, stream.get()>>>(
    input, paths, plan.view, depth_error);
}

// The 63/20 cutoff separates the measured 124/40 deny and 120/38 admit; it is not break-even.
bool admits_fanout(path_batch const& paths, std::vector<std::size_t> const& indices)
{
  if (indices.size() < k_merged_min_paths) { return false; }
  std::set<std::string_view> roots;
  std::set<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>> unique;
  for (auto const i : indices) {
    roots.insert(std::get<1>(paths[i].front()));
    unique.emplace(paths[i].begin(), paths[i].end());
  }
  return unique.size() * 20 >= roots.size() * 63;
}

// Per-path codes survive retry compaction; a different evaluator can exceed measured slots.
struct grouped_routes {
  std::vector<char> codes;
  std::vector<std::size_t> general;
  std::vector<std::size_t> merged;
};

int named_depth(cudf::host_span<std::tuple<path_instruction_type, std::string, int32_t> const> path)
{
  return path.size() <= k_merged_max_depth &&
             std::all_of(
               path.begin(),
               path.end(),
               [](auto const& op) { return std::get<0>(op) == path_instruction_type::NAMED; })
           ? static_cast<int>(path.size())
           : 0;
}

// Preserve batch order when assigning the shallow family.
grouped_routes plan_routes(path_batch const& paths, cudf::size_type rows, bool last_non_null)
{
  grouped_routes plan{std::vector<char>(paths.size()), {}, {}};
  for (std::size_t i = 0; i < paths.size(); ++i) {
    auto const depth = last_non_null || rows < k_min_rows_per_family ? 0 : named_depth(paths[i]);
    if (depth == 0 || plan.merged.size() == k_merged_max_paths) {
      plan.general.push_back(i);
    } else {
      plan.merged.push_back(i);
    }
  }
  if (!admits_fanout(paths, plan.merged)) {
    plan.general.insert(plan.general.end(), plan.merged.begin(), plan.merged.end());
    plan.merged.clear();
  }
  for (auto const i : plan.merged) {
    plan.codes[i] = 1;
  }
  return plan;
}

// Retry only the flagged paths, in first-pass route order, without reapplying admission.
grouped_routes retry_routes(std::vector<char> codes)
{
  grouped_routes plan{std::move(codes), {}, {}};
  for (std::size_t i = 0; i < plan.codes.size(); ++i) {
    (plan.codes[i] == 0 ? plan.general : plan.merged).push_back(i);
  }
  return plan;
}

// Mixed launch data stays live until readback has ordered both streams.
struct grouped_upload {
  cudf::detail::host_vector<std::byte> host;
  cuda::device_buffer<std::byte> device;
};

std::optional<grouped_upload> launch_grouped(
  cudf::column_device_view const& input,
  path_batch const& paths,
  grouped_routes const& routes,
  std::vector<json_path_processing_data> const& host_paths,
  int8_t* depth_error,
  bool use_last_non_null,
  cuda::stream_ref stream)
{
  auto const mr = cudf::get_current_device_resource_ref();
  if (routes.merged.empty()) {
    auto data = cudf::detail::make_device_uvector_async(host_paths, stream, mr);
    kernel_launcher::exec(input, data, depth_error, use_last_non_null, stream);
    return std::nullopt;
  }
  auto const depth         = k_merged_max_depth;
  auto plan                = build_merged_named_plan(paths, routes.merged, depth);
  constexpr auto alignment = alignof(json_path_processing_data);
  auto const group_begin   = (plan.h_blob.size() + alignment - 1) & ~(alignment - 1);
  auto const bytes         = group_begin + host_paths.size() * sizeof(json_path_processing_data);
  auto host                = cudf::detail::make_pinned_vector_async<std::byte>(bytes, stream);
  cuda::device_buffer<std::byte> device(stream, mr, bytes, cuda::no_init);
  std::memcpy(host.data(), plan.h_blob.data(), plan.h_blob.size());
  auto* h               = host.data() + group_begin;
  auto* d               = reinterpret_cast<json_path_processing_data*>(device.data() + group_begin);
  auto const pack_group = [&](std::vector<std::size_t> const& indices) {
    auto const span = cudf::device_span<json_path_processing_data>{d, indices.size()};
    for (auto const i : indices) {
      std::memcpy(h, &host_paths[i], sizeof(host_paths[i]));
      h += sizeof(host_paths[i]);
    }
    d += indices.size();
    return span;
  };
  auto const merged  = pack_group(routes.merged);
  auto const general = pack_group(routes.general);
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(device.data(), host.data(), bytes, cudaMemcpyDefault, stream.get()));
  bind_merged_plan_view(plan, reinterpret_cast<uint8_t const*>(device.data()));
  if (general.empty()) {
    launch_merged_named(input, merged, plan, depth_error, stream);
  } else {
    auto fork = cudf::detail::fork_streams(stream, 1);
    launch_merged_named(input, merged, plan, depth_error, stream);
    kernel_launcher::exec(input, general, depth_error, use_last_non_null, fork[0]);
    cudf::detail::join_streams(fork, stream);
  }
  return grouped_upload{std::move(host), std::move(device)};
}

}  // namespace

std::vector<std::unique_ptr<cudf::column>> get_json_object_batch(
  cudf::column_device_view const& input,
  cudf::detail::input_offsetalator const& offsets,
  int64_t base_offset,
  path_batch const& paths,
  int64_t scratch_size,
  bool use_last_non_null,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  std::vector<char>* routes_out)
{
  auto const count  = paths.size();
  auto const rows   = static_cast<std::size_t>(input.size());
  auto const routes = plan_routes(paths, input.size(), use_last_non_null);
  for (auto const& path : paths) {
    CUDF_EXPECTS(path.size() <= MAX_JSON_PATH_DEPTH,
                 "JSON Path has depth exceeds the maximum allowed value.");
  }
  auto host_flags     = cudf::detail::make_pinned_vector_async<int8_t>(count + 1, stream);
  auto const commands = stage_commands(paths, routes.general, stream);
  using pair_type     = cuda::std::pair<char const*, cudf::size_type>;
  cuda::device_buffer<pair_type> pairs(stream, mr, count * rows, cuda::no_init);
  std::vector<rmm::device_uvector<char>> scratch;
  std::vector<json_path_processing_data> host_paths;
  rmm::device_uvector<int8_t> flags(count + 1, stream);
  scratch.reserve(count);
  host_paths.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    scratch.emplace_back(scratch_size, stream);
    host_paths.push_back({commands.paths[i],
                          offsets,
                          base_offset,
                          pairs.data() + i * rows,
                          scratch.back().data(),
                          flags.data() + i});
  }
  // The readback also orders both kernels and their asynchronous host uploads.
  auto const run_pass = [&](path_batch const& pass_paths,
                            grouped_routes const& pass_routes,
                            rmm::device_uvector<int8_t>& errors) {
    thrust::uninitialized_fill(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      errors.begin(),
      errors.end(),
      0);
    [[maybe_unused]] auto const upload = launch_grouped(input,
                                                        pass_paths,
                                                        pass_routes,
                                                        host_paths,
                                                        errors.data() + pass_paths.size(),
                                                        use_last_non_null,
                                                        stream);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      host_flags.data(), errors.data(), errors.size(), cudaMemcpyDefault, stream.get()));
    stream.sync();
    return cudf::host_span<int8_t const>{host_flags.data(), errors.size()};
  };
  auto const first = run_pass(paths, routes, flags);
  if (routes_out != nullptr) { *routes_out = routes.codes; }
  auto const gather = [&](std::vector<std::size_t> const& ids) {
    if (ids.empty()) { return std::vector<std::unique_ptr<cudf::column>>{}; }
    std::vector<cudf::device_span<pair_type const>> views;
    views.reserve(ids.size());
    for (auto const i : ids) {
      views.emplace_back(pairs.data() + i * rows, rows);
    }
    return cudf::make_strings_column_batch(views, stream, mr);
  };
  std::vector<std::size_t> small, large;
  auto const ok = check_error(first);
  for (std::size_t i = 0; i < count; ++i) {
    (first[i] == 0 ? small : large).push_back(i);
  }
  if (ok) { return gather(small); }
  // Flagged scratch may contain overlapping writes; only gather accepted columns.
  auto unmoved = gather(small);
  scratch.clear();

  path_batch retry_paths;
  std::vector<char> retry_codes;
  std::vector<std::unique_ptr<cudf::column>> retry_offsets;
  rmm::device_uvector<int8_t> retry_flags(large.size() + 1, stream);
  host_paths.clear();
  for (auto const i : large) {
    auto* const view = pairs.data() + i * rows;
    // Rows that did not overflow may have discarded writes up to their original slot size.
    auto const sizes = spark_rapids_jni::util::make_counting_transform_iterator(
      cudf::size_type{0},
      cuda::proclaim_return_type<cudf::size_type>([view, offsets] __device__(cudf::size_type row) {
        return cuda::std::max(view[row].second,
                              static_cast<cudf::size_type>(offsets[row + 1] - offsets[row]));
      }));
    auto [col, bytes] =
      cudf::strings::detail::make_offsets_child_column(sizes, sizes + input.size(), stream, mr);
    scratch.emplace_back(bytes, stream, mr);
    host_paths.push_back({commands.paths[i],
                          cudf::detail::offsetalator_factory::make_input_iterator(col->view()),
                          0,
                          view,
                          scratch.back().data(),
                          retry_flags.data() + retry_paths.size()});
    retry_offsets.emplace_back(std::move(col));
    retry_paths.emplace_back(paths[i]);
    retry_codes.push_back(routes.codes[i]);
  }
  CUDF_EXPECTS(
    check_error(run_pass(retry_paths, retry_routes(std::move(retry_codes)), retry_flags)),
    "Unexpected out-of-bound write in get_json_object kernel.");
  // Retry offsets are capacities, not committed lengths; gather from the rewritten pairs.
  auto retried = gather(large);
  std::vector<std::unique_ptr<cudf::column>> out(count);
  for (std::size_t j = 0; j < small.size(); ++j) {
    out[small[j]] = std::move(unmoved[j]);
  }
  for (std::size_t j = 0; j < large.size(); ++j) {
    out[large[j]] = std::move(retried[j]);
  }
  return out;
}

std::vector<std::unique_ptr<cudf::column>> get_json_object(
  cudf::strings_column_view const& input,
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>> const& paths,
  int64_t memory_budget_bytes,
  int32_t parallel_override,
  bool use_last_non_null,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  std::vector<char>* routes_out)
{
  auto const count = paths.size();
  std::vector<std::unique_ptr<cudf::column>> out(count);
  if (routes_out != nullptr) { routes_out->assign(count, 0); }
  if (count == 0) { return out; }
  if (input.is_empty() || input.size() == input.null_count()) {
    for (auto& col : out) {
      col = std::make_unique<cudf::column>(input.parent(), stream, mr);
    }
    return out;
  }
  auto const offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
  std::vector<std::size_t> order(count);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](auto a, auto b) { return paths[a] < paths[b]; });
  auto const [scratch_size, base_offset] = calc_scratch_size(input, offsets, stream);
  auto const device_input                = cudf::column_device_view::create(
    input.parent(), stream, cudf::get_current_device_resource_ref());
  // A budget charges each path after accepting it, even if a single path exceeds the budget.
  auto const width = parallel_override > 0 ? static_cast<std::size_t>(parallel_override)
                     : (memory_budget_bytes <= 0 || scratch_size == 0)
                       ? count
                       : static_cast<std::size_t>(1 + (memory_budget_bytes - 1) / scratch_size);
  path_batch batch;
  std::vector<char> batch_routes;
  for (std::size_t begin = 0; begin < count; begin += width) {
    auto const size = std::min(width, count - begin);
    batch.clear();
    for (std::size_t i = 0; i < size; ++i) {
      batch.emplace_back(paths[order[begin + i]]);
    }
    auto next = get_json_object_batch(*device_input,
                                      offsets,
                                      base_offset,
                                      batch,
                                      scratch_size,
                                      use_last_non_null,
                                      stream,
                                      mr,
                                      routes_out == nullptr ? nullptr : &batch_routes);
    for (std::size_t i = 0; i < size; ++i) {
      auto const id = order[begin + i];
      out[id]       = std::move(next[i]);
      if (routes_out != nullptr) { (*routes_out)[id] = batch_routes[i]; }
    }
  }
  return out;
}
}  // namespace detail

std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& input,
  std::vector<std::tuple<path_instruction_type, std::string, int32_t>> const& instructions,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return std::move(
    detail::get_json_object(input, {instructions}, -1, -1, false, stream, mr).front());
}

std::vector<std::unique_ptr<cudf::column>> get_json_object_multiple_paths(
  cudf::strings_column_view const& input,
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>> const&
    json_paths,
  int64_t memory_budget_bytes,
  int32_t parallel_override,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return detail::get_json_object(
    input, json_paths, memory_budget_bytes, parallel_override, false, stream, mr);
}

std::vector<std::unique_ptr<cudf::column>> get_json_object_multiple_paths(
  cudf::strings_column_view const& input,
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>> const&
    json_paths,
  int64_t memory_budget_bytes,
  int32_t parallel_override,
  named_field_match_policy match_policy,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  CUDF_EXPECTS(match_policy == named_field_match_policy::LAST_NON_NULL,
               "Invalid named-field match policy.");
  CUDF_EXPECTS(std::all_of(json_paths.begin(),
                           json_paths.end(),
                           [](auto const& path) {
                             return path.size() == 1 &&
                                    std::get<0>(path.front()) == path_instruction_type::NAMED;
                           }),
               "LAST_NON_NULL requires paths containing exactly one named instruction.");
  return detail::get_json_object(
    input, json_paths, memory_budget_bytes, parallel_override, true, stream, mr);
}
}  // namespace spark_rapids_jni
