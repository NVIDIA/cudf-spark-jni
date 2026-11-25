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
#include <cudf/strings/utilities.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_memcpy.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/functional>
#include <cuda/hierarchy>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <stdexcept>
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

  // Both parser variants validate children through the matching close.
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
constexpr int k_merged_max_paths      = 768;
constexpr int k_merged_min_paths      = 4;
constexpr int k_merged_max_depth      = 2;
constexpr int k_merged_deep_max_depth = 5;
constexpr int k_deep_merged_min_paths = 8;
// A smaller group cannot amortize the merged launch overhead.
constexpr int k_min_rows_per_family     = 32;
constexpr int k_request_deny            = k_merged_max_paths + 1;
constexpr int64_t k_request_credit      = 128;
constexpr int64_t k_request_member_cost = 395;
constexpr int64_t k_request_width_scale = 57;
constexpr int64_t k_request_margin      = 88;
constexpr int k_request_sample_bytes    = 12'288;
constexpr int k_exact_scan_threads      = 64;

// Byte sizing is exact in every state; a zero threshold requests the exact pass.
struct request_scan {
  int64_t max_row_bytes;
  int64_t selected_chars;
  int64_t base_offset;
  int32_t min_eligible;
};
static_assert(sizeof(request_scan) == 32);

__host__ __device__ int64_t request_width_cost(int64_t avg)
{
  return k_request_member_cost * avg / 5'800;
}

// Price count and credit from the same row, using the selected-view average.
__device__ int64_t request_score(int64_t m, int64_t avg, int64_t u)
{
  return avg + k_request_member_cost * m +
         cuda::std::min(request_width_cost(avg) * m,
                        k_request_member_cost * cuda::std::min(avg, u) / k_request_width_scale);
}

// Two ceilings preserve equality at the 88% margin. INT64_MAX denies every batch.
__device__ int32_t request_threshold(int64_t score, int64_t avg)
{
  if (score == INT64_MAX) { return k_request_deny; }
  auto const scaled = (100 * score + k_request_margin - 1) / k_request_margin;
  auto const q      = scaled / avg + (scaled % avg != 0);
  return static_cast<int32_t>(
    cuda::std::min<int64_t>(k_request_deny, cuda::std::max<int64_t>(k_merged_min_paths, q)));
}

struct request_limit {
  int64_t avg;
  int64_t min_members;
  int64_t min_credit_units;

  __device__ bool denies(int64_t m, int64_t u) const
  {
    return m >= min_members &&
           k_request_width_scale * m + cuda::std::min(avg, u) >= min_credit_units;
  }
};

// Both sides of the minimum must exceed the budget. Invert each once; equality admits.
__host__ __device__ request_limit make_request_limit(int64_t avg, int64_t eligible)
{
  auto const allowance   = k_request_margin * eligible * avg / 100 - avg;
  auto const min_members = 1 + allowance / (k_request_member_cost + request_width_cost(avg));
  // Invert floor(395 * (57 * m + min(avg, u)) / 57) without multiplying the full allowance.
  auto const tail             = k_request_width_scale * (allowance % k_request_member_cost + 1);
  auto const min_credit_units = k_request_width_scale * (allowance / k_request_member_cost) +
                                (tail + k_request_member_cost - 1) / k_request_member_cost;
  return {avg, min_members, min_credit_units};
}

struct request_prefix {
  int pos;
  int count;
  int credit;
};

// All lanes scan plain pairs; lane 0 gets exact prefix evidence unless denial already follows.
__device__ request_prefix flat_request_prefix(char const* row, int64_t len, request_limit limit)
{
  constexpr int width     = static_cast<int>(warp_size);
  constexpr unsigned mask = 0xffffffffu;
  if (len < width || row[0] != '{') { return {}; }
  auto const lane         = static_cast<int>(threadIdx.x % width);
  auto const end          = static_cast<int>(cuda::std::min<int64_t>(len, k_request_sample_bytes));
  auto const lanes_before = (1u << lane) - 1;
  request_prefix prefix{};
  int quotes = 0, last_quote = -1, credit = 0;
  for (int base = 0; base < end; base += width) {
    // Leave long strings to the word-at-a-time scalar walk.
    if ((quotes & 1) && base - last_quote >= k_request_credit) { break; }
    auto const pos   = base + lane;
    auto const ch    = pos < end ? row[pos] : '\0';
    auto const prev  = pos > 0 && pos < end ? row[pos - 1] : '\0';
    auto const hits  = __ballot_sync(mask, ch == '"');
    auto const prior = hits & lanes_before;
    auto const role  = (quotes + __popc(prior)) & 3;
    // Quotes cycle through name open/close, then value open/close.
    auto const separator  = role == 2 ? ':' : ',';
    auto const quote_from = pos == 1 ? '{' : separator;
    bool const valid      = pos == 0 || (role & 1    ? ch != '\\'
                                         : ch == '"' ? prev == quote_from
                                                     : prev == '"' && ch == separator);
    auto const stop       = __ballot_sync(mask, pos < end && !valid);
    bool const settled    = ch == '"' && role == 3 && (stop == 0 || lane < __ffs(stop) - 1);
    auto const closes     = __ballot_sync(mask, settled);
    if (settled) {
      auto const open = prior == 0 ? last_quote : base + width - 1 - __clz(prior);
      credit += cuda::std::max(0, static_cast<int>(k_request_credit) - (pos - open - 1));
    }
    if (closes != 0) {
      prefix.pos = base + width - __clz(closes);
      prefix.count += __popc(closes);
      // Payload bytes fit within the prefix; this bound avoids reducing credit each chunk.
      auto const lower =
        cuda::std::max(0, static_cast<int>(k_request_credit) * prefix.count - prefix.pos);
      if (limit.denies(prefix.count, lower)) { return {prefix.pos, prefix.count, lower}; }
    }
    if (stop != 0) { break; }
    quotes += __popc(hits);
    if (hits != 0) { last_quote = base + width - 1 - __clz(hits); }
  }
  if (prefix.count != 0) {
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {
      credit += __shfl_down_sync(mask, credit, delta);
    }
  }
  prefix.credit = credit;
  return prefix;
}

// Starts on the opening quote; returns false at the byte limit, without decoding escapes.
template <bool sampled>
__device__ bool scan_raw_quote(char const* row, int64_t& pos, int64_t end, bool& plain)
{
  auto const quote = row[pos++];
  auto const body  = pos;
  while (pos < end) {
    auto const c = row[pos];
    if (c == quote) {
      ++pos;
      return true;
    }
    if (c == '\\') {
      plain = false;
      pos += cuda::std::min<int64_t>(2, end - pos);
    } else if (end - pos >= 8 &&
               (sampled ? pos - body >= 23 && (reinterpret_cast<uintptr_t>(row + pos) & 7) == 0
                        : pos >= 7)) {
      // Ramp on samples to amortize setup; exact can use any readable masked head.
      auto const n = static_cast<cudf::size_type>(cuda::std::min<int64_t>(end - pos, INT32_MAX));
      char_range_reader run{char_range{row + pos, n}};
      json_parser::swar_safe_run<false>(run, nullptr, quote, false);
      pos += n - run.size();
    } else {
      ++pos;
    }
  }
  return false;
}

// Below the counted frames, balance the opening delimiter without counting nested names.
__device__ bool skip_request_scope(char const* row, int64_t& pos, int64_t end)
{
  auto const open  = row[pos++];
  auto const close = open == '{' ? '}' : ']';
  int64_t depth    = 1;
  while (pos < end) {
    auto const c = row[pos];
    if (c == '"' || c == '\'') {
      bool plain = false;
      if (!scan_raw_quote<false>(row, pos, end, plain)) { return false; }
    } else {
      ++pos;
      depth += (c == open) - (c == close);
      if (depth == 0) { return true; }
    }
  }
  return false;
}

enum class request_walk_state { complete, clipped, deny };
struct request_members {
  int64_t count;
  int64_t credit;
  request_walk_state state;
};

// Offsets, not validity, define the row. This walk estimates work; evaluators validate JSON.
template <bool sampled>
__device__ request_members
walk_request_row(char const* row, int64_t len, request_limit limit, request_prefix prefix = {})
{
  using state = request_walk_state;
  enum { object = 1, want_name = 2, after_comma = 4, unpaid = 8 };
  if (len == 0) { return {0, 0, state::deny}; }
  // A warm cursor follows a whole root member; the next token is still unconsumed.
  auto const end    = sampled ? cuda::std::min<int64_t>(len, k_request_sample_bytes) : len;
  auto const at_end = end == len ? state::deny : state::clipped;
  int parents[k_merged_deep_max_depth];
  int depth       = prefix.pos != 0;
  int frame       = depth != 0 ? object : 0;
  int64_t members = prefix.count, credit = prefix.credit, pos = prefix.pos;
  if constexpr (sampled) {
    if (limit.denies(members, credit)) { return {members, credit, state::deny}; }
  }
  bool root_seen = false;
  while (pos < end) {
    auto const ch = row[pos];
    if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
      ++pos;
      continue;
    }
    bool const name = (frame & want_name) != 0;
    // A counted name pays for at most one value, even on a permissively swallowed row.
    bool const pays_credit = !name && (frame & unpaid);
    if (ch == '"' || ch == '\'') {
      if (name && ch == '\'') { return {members, credit, state::deny}; }
      auto const start = pos;
      // Opening quote + 4095 raw bytes + closing quote.
      auto const cap = name ? pos + cuda::std::min<int64_t>(4'097, end - pos) : end;
      bool plain     = ch == '"';
      if (!scan_raw_quote<sampled>(row, pos, cap, plain)) {
        return {members, credit, sampled && name && cap - start == 4'097 ? state::clipped : at_end};
      }
      if (name) {
        if (++members > (1 << 20) || limit.denies(members, credit)) {
          return {members, credit, state::deny};
        }
        frame |= unpaid;
      } else if (pays_credit) {
        credit += plain ? cuda::std::max<int64_t>(0, k_request_credit - (pos - start - 2))
                        : k_request_credit;
        frame &= ~unpaid;
      }
    } else if (ch == '{' || ch == '[') {
      if (pays_credit) {
        credit += k_request_credit;
        frame &= ~unpaid;
      }
      if (depth == k_merged_deep_max_depth) {
        if (sampled) { return {members, credit, state::clipped}; }
        if (!skip_request_scope(row, pos, end)) { return {members, credit, state::deny}; }
      } else {
        parents[depth++] = frame;
        frame            = ch == '{' ? object | want_name : 0;
        ++pos;
      }
    } else if (ch == '}' || ch == ']') {
      if (depth == 0 || ((ch == '}') != ((frame & object) != 0)) ||
          (name && (frame & after_comma))) {
        return {members, credit, state::deny};
      }
      ++pos;
      if (--depth == 0) {
        return {members,
                credit,
                end - pos > 1 ? state::deny
                : end < len   ? state::clipped
                              : state::complete};
      }
      frame = parents[depth];
    } else if (ch == ',' || ch == ':') {
      if (depth == 0 || (ch == ':' && !(frame & object))) { return {members, credit, state::deny}; }
      frame = ch == ':'          ? frame & ~want_name
              : (frame & object) ? object | want_name | after_comma
                                 : 0;
      ++pos;
    } else {
      // Swallow raw scalars such as x so a late invalid token can still exercise merged retry.
      if (pays_credit) { credit += k_request_credit; }
      root_seen |= depth == 0;
      frame &= ~(want_name | unpaid);
      while (++pos < end && row[pos] != ',' && row[pos] != '}' && row[pos] != ']') {}
    }
  }
  return {members, credit, root_seen && depth == 0 && end == len ? state::complete : at_end};
}

// Launch once with 256 threads. Offsets are slice-based; chars still points to the allocation base.
template <bool sample_members>
__global__ void scan_request_kernel(cudf::detail::input_offsetalator offsets,
                                    char const* chars,
                                    cudf::size_type rows,
                                    int64_t budget,
                                    int32_t parallel_override,
                                    int32_t eligible_upper,
                                    request_scan* out)
{
  int64_t longest   = 0;
  auto const tidx   = cuda::gpu_thread.index_as<cudf::thread_index_type>(cuda::block).x;
  auto const stride = cuda::gpu_thread.count_as<cudf::thread_index_type>(cuda::block);
  for (auto i = tidx; i < rows; i += stride) {
    auto const r = static_cast<cudf::size_type>(i);
    longest      = cuda::std::max(longest, offsets[r + 1] - offsets[r]);
  }
  using reduce = cub::BlockReduce<int64_t, 256>;
  __shared__ typename reduce::TempStorage storage;
  longest        = reduce(storage).Reduce(longest, cuda::maximum{});
  auto const tid = threadIdx.x;
  if (tid == 0) {
    auto const base = rows > 0 ? offsets[0] : int64_t{0};
    *out            = {longest, rows > 0 ? offsets[rows] - base : int64_t{0}, base, k_request_deny};
  }
  if constexpr (sample_members) {
    __shared__ int64_t eligible;
    __shared__ request_limit limit;
    __shared__ request_members samples[8];
    if (tid == 0) {
      eligible         = 0;
      auto const bytes = out->selected_chars;
      // An average of zero on nonempty input guarantees a physical empty row.
      if (rows >= k_min_rows_per_family && bytes >= rows) {
        auto const avg     = bytes / rows;
        auto const scratch = bytes + 10 * longest;
        auto const cap     = parallel_override > 0 ? int64_t{parallel_override}
                             : budget > 0          ? 1 + (budget - 1) / scratch
                                                   : eligible_upper;
        eligible           = cuda::std::min<int64_t>(eligible_upper, cap);
        if (eligible >= k_merged_min_paths) { limit = make_request_limit(avg, eligible); }
      }
    }
    __syncthreads();
    if (eligible < k_merged_min_paths) { return; }
    auto const slot   = tid / warp_size;
    auto const r      = static_cast<cudf::size_type>(static_cast<int64_t>(slot) * (rows - 1) / 7);
    auto const begin  = offsets[r];
    auto const len    = offsets[r + 1] - begin;
    auto const* row   = chars + begin;
    auto const prefix = flat_request_prefix(row, len, limit);
    if (tid % warp_size == 0) { samples[slot] = walk_request_row<true>(row, len, limit, prefix); }
    __syncthreads();
    if (tid == 0) {
      int64_t max_m = 0;
      bool complete = true;
      for (auto const& s : samples) {
        if (s.state == request_walk_state::deny || limit.denies(s.count, s.credit)) { return; }
        max_m = cuda::std::max(max_m, s.count);
        complete &= s.state == request_walk_state::complete;
      }
      auto const upper =
        request_threshold(request_score(max_m, limit.avg, k_request_credit * max_m), limit.avg);
      out->min_eligible = complete && upper <= eligible ? upper : 0;
    }
  }
}

// Each row stays on one thread; a decisive deny stops that thread's walk.
__global__ void scan_exact_admission(cudf::detail::input_offsetalator offsets,
                                     char const* chars,
                                     cudf::size_type rows,
                                     request_limit limit,
                                     request_scan* out)
{
  auto const tidx   = cuda::gpu_thread.index_as<cudf::thread_index_type>(cuda::grid).x;
  auto const stride = cuda::gpu_thread.count_as<cudf::thread_index_type>(cuda::grid);
  int64_t score     = 0;
  for (auto i = tidx; i < rows; i += stride) {
    auto const r     = static_cast<cudf::size_type>(i);
    auto const begin = offsets[r];
    auto const row   = walk_request_row<false>(chars + begin, offsets[r + 1] - begin, limit);
    if (row.state != request_walk_state::complete || limit.denies(row.count, row.credit)) {
      score = INT64_MAX;
      break;
    }
    score = cuda::std::max(score, request_score(row.count, limit.avg, row.credit));
  }
  using reduce = cub::BlockReduce<int64_t, k_exact_scan_threads>;
  __shared__ typename reduce::TempStorage storage;
  score = reduce(storage).Reduce(score, cuda::maximum{});
  if (threadIdx.x == 0) {
    cuda::atomic_ref<int32_t, cuda::thread_scope_device>{out->min_eligible}.fetch_max(
      request_threshold(score, limit.avg), cuda::memory_order_relaxed);
  }
}

namespace {

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
  uint64_t const* flat_name_sig;
  // For multi-child nodes, the length row includes a final sentinel.
  int32_t const* len_prefix;
  std::size_t const* len_prefix_beg;
  int32_t const* max_name_len;
};

// A negative begin means unset; dead paths cannot be revived by later duplicate keys.
// Only the row leader writes; the warp barrier orders these reads before gather.
struct alignas(8) merged_path_out {
  uint64_t range = 0xffffffffu;
  int8_t kind    = 0;
};

constexpr int8_t k_merged_kind_string     = 1;
constexpr int8_t k_merged_kind_dead       = 3;
constexpr int8_t k_merged_kind_string_raw = 4;
constexpr int8_t k_merged_kind_value_raw  = 5;

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

// Two rows per warp fit the same shared memory by using 29 bits for the exclusive end.
struct merged_path_packed {
  uint64_t word = 0xffffffffu;
};
constexpr int64_t k_packed_end_max = (uint64_t{1} << 29) - 1;

__device__ __forceinline__ merged_rec_view load_record(merged_path_packed const& r)
{
  return {static_cast<int32_t>(r.word),
          static_cast<int32_t>((r.word >> 32) & k_packed_end_max),
          static_cast<int8_t>(r.word >> 61)};
}

__device__ __forceinline__ void store_record(merged_path_packed& r,
                                             int32_t begin,
                                             int32_t end,
                                             int8_t kind)
{
  r.word = static_cast<uint32_t>(begin) | (static_cast<uint64_t>(end) << 32) |
           (static_cast<uint64_t>(kind) << 61);
}

// Returns the matching slot ID, not the flat child position, or -1 within the supplied child range.
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

// Packs the first and last nonoverlapping bytes, up to four each; equal signatures alone never
// certify equal names. E.g. abcde packs abcd in the low half and e in the high half; there is
// no repeated overlap. Host and device use the same formula.
__host__ __device__ inline uint64_t name_sig(char const* p, int32_t n)
{
  uint64_t sig     = 0;
  int32_t const hn = n < 4 ? n : 4;
  int32_t const tn = n - 4 > hn ? 4 : n > 4 ? n - 4 : 0;
  for (int32_t i = 0; i < hn; ++i) {
    sig |= static_cast<uint64_t>(static_cast<uint8_t>(p[i])) << (i * 8);
  }
  for (int32_t i = 0; i < tn; ++i) {
    sig |= static_cast<uint64_t>(static_cast<uint8_t>(p[n - tn + i])) << (32 + i * 8);
  }
  return sig;
}

// Full byte comparison defends against length-bucket and signature collisions.
__device__ __forceinline__ bool signature_match(merged_trie_view const& trie,
                                                char_range const& raw,
                                                int32_t inner,
                                                int32_t c)
{
  int32_t const cand_len = trie.flat_name_len[c];
  if (cand_len != inner) { return false; }  // bucket drift defense
  char const* cand = trie.names + trie.flat_name_off[c];
  for (int32_t i = 0; i < inner; ++i) {
    if (cand[i] != raw.data()[1 + i]) { return false; }
  }
  return true;
}

/**
 * Certifies byte-identical serialization of a parsed subtree; this is not a JSON validator.
 * Requires double quotes, simple round-trip escapes, compact separators and canonical
 * integers/literals. Other forms (including floats, -0, Unicode/slash escapes) use the writer.
 * Keep raw eligibility in sync if the writer stops preserving bytes >=0x80, e.g. astral UTF-8.
 */
__device__ __forceinline__ bool certify_raw_subtree(char const* data, int32_t len)
{
  bool in_string = false;
  int32_t i      = 0;
  // Track object-key positions in the already-validated structure.
  uint8_t kind_stack[MAX_JSON_NESTING_DEPTH];
  int32_t stack_top   = 0;
  bool expect_obj_key = false;
  while (i < len) {
    char const b = data[i];
    if (in_string) {
      if (b == '\\') {
        // Only \" \\ \b \f \n \r \t round-trip; \/ and \u fall back.
        if (i + 1 >= len) { return false; }
        char const e = data[i + 1];
        bool const canonical =
          e == '"' || e == '\\' || e == 'b' || e == 'f' || e == 'n' || e == 'r' || e == 't';
        if (!canonical) { return false; }
        i += 2;  // consume backslash + escape char
      } else if (b == '"') {
        in_string = false;
        ++i;
      } else if (static_cast<uint8_t>(b) < 0x20) {
        return false;
      } else {
        // run >= 1: data[i] is safe because the aligned head load masks bytes before the
        // cursor; the outer scanner consumes the stop byte.
        char_range_reader r{char_range(data + i, len - i)};
        i += json_parser::swar_safe_run<true>(r, nullptr, '"', false);
        if (i >= len) { return false; }  // unterminated string: parse would have failed
        continue;                        // scalar decisions handle the stop byte
      }
      continue;
    }
    if (expect_obj_key) {
      // Only a quoted key or an immediate `{"}` empty-object close is canonical.
      if (b != '"' && !(b == '}' && i > 0 && data[i - 1] == '{')) { return false; }
    }
    if (b == '"') {
      expect_obj_key = false;  // this quote opens the key itself when the gate passed
      in_string      = true;
      ++i;
    } else if (b == '{' || b == '[') {
      // Track whether a comma must precede an object key.
      if (stack_top >= MAX_JSON_NESTING_DEPTH) {
        return false;  // deeper than the parser's own cap — uncertified, not parseable input
      }
      kind_stack[stack_top++] = (b == '{') ? 1 : 0;
      expect_obj_key          = (b == '{');
      ++i;
    } else if (b == '}' || b == ']') {
      if (stack_top == 0) { return false; }  // unbalanced — parse would have failed
      bool const was_object = kind_stack[--stack_top] == 1;
      if ((b == '}') != was_object) { return false; }  // mismatched closer
      expect_obj_key = false;
      ++i;
    } else if (b == ',' || b == ':') {
      // The parent container determines whether a comma requires an object key.
      if (b == ',') {
        if (stack_top == 0) { return false; }  // top-level comma — parse would have failed
        expect_obj_key = kind_stack[stack_top - 1] == 1;
      }
      ++i;
    } else if ((b >= '0' && b <= '9') || b == '-') {
      // Ints only (no float, no -0, no leading zero).
      int32_t const beg = i;
      while (i < len) {
        char const nb = data[i];
        if ((nb >= '0' && nb <= '9') || nb == '.' || nb == 'e' || nb == 'E' || nb == '+' ||
            nb == '-') {
          ++i;
        } else {
          break;
        }
      }
      int32_t digits_beg = beg;
      if (data[digits_beg] == '-') { ++digits_beg; }
      int32_t const digits_len = i - digits_beg;
      if (digits_len == 0) { return false; }  // bare `-`
      for (int32_t j = digits_beg; j < i; ++j) {
        if (data[j] < '0' || data[j] > '9') { return false; }  // float/exponent
      }
      if (digits_len > 1 && data[digits_beg] == '0') { return false; }  // leading zero
      if (data[beg] == '-' && digits_len == 1 && data[digits_beg] == '0') {
        return false;  // literal `-0` rewrites to `0`; `-1`..`-9` are verbatim
      }
    } else if ((b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')) {
      int32_t const beg = i;
      while (i < len) {
        char const lb = data[i];
        if ((lb >= 'a' && lb <= 'z') || (lb >= 'A' && lb <= 'Z')) {
          ++i;
        } else {
          break;
        }
      }
      int32_t const run_len = i - beg;
      bool const is_literal =
        (run_len == 4 && data[beg] == 't' && data[beg + 1] == 'r' && data[beg + 2] == 'u' &&
         data[beg + 3] == 'e') ||
        (run_len == 5 && data[beg] == 'f' && data[beg + 1] == 'a' && data[beg + 2] == 'l' &&
         data[beg + 3] == 's' && data[beg + 4] == 'e') ||
        (run_len == 4 && data[beg] == 'n' && data[beg + 1] == 'u' && data[beg + 2] == 'l' &&
         data[beg + 3] == 'l');
      if (!is_literal) { return false; }
    } else {
      // Other bytes either normalize on write or fail the parser grammar.
      return false;
    }
  }
  return !in_string;
}

// One member scan uses the parser's matcher for one child, or bucket-filtered exact matching.
// Advances one object-member token; returns its slot or -1, leaving END_OBJECT/ERROR in the parser.
// The caller must inspect the token before treating -1 as an unmatched name or reading its value.
__device__ inline int32_t merged_scan_member(json_parser_impl<true>& p,
                                             merged_trie_view const& trie,
                                             int32_t node)
{
  int32_t const beg = trie.children_beg[node];
  int32_t const cnt = trie.children_cnt[node];
  if (cnt == 1) {
    int32_t const slot = trie.flat_children[beg];
    cudf::string_view const first_name(trie.names + trie.flat_name_off[beg],
                                       trie.flat_name_len[beg]);
    bool const matched = p.parse_next_token_with_matching(first_name);
    // The parser's match flag is only meaningful for a FIELD_NAME token.
    if (json_token::FIELD_NAME == p.get_current_token() && matched) { return slot; }
    return -1;
  }

  // With multiple candidates the parser's match flag is useless, so scan all of them after
  // parsing the FIELD_NAME token; a single candidate lets the parser match during tokenization.
  p.next_token();
  if (json_token::FIELD_NAME != p.get_current_token() || cnt == 0) { return -1; }

  // Escaped names need decoded comparison; raw-length buckets are sound only without
  // backslashes.
  char_range const raw  = p.current_range();
  int32_t const tok_len = raw.size();  // raw, incl. quotes
  if (tok_len < 2 || p.current_token_is_escaped_field_name()) {
    return merged_exact_scan(trie, raw, beg, beg + cnt);
  }

  int32_t const inner = tok_len - 2;

  if (inner > trie.max_name_len[node]) { return -1; }
  // Length-sorted children occupy [beg + row[L], beg + row[L+1]).
  int32_t const* row = trie.len_prefix + trie.len_prefix_beg[node];
  int32_t const lo   = beg + row[inner];
  int32_t const hi   = beg + row[inner + 1];
  if (lo == hi) { return -1; }

  // Signature filters only; full raw-byte comparison decides every match.
  uint64_t const sig = name_sig(raw.data() + 1, inner);
  if (lo + 1 < hi) {
    for (int32_t c = lo; c < hi; ++c) {
      if (trie.flat_name_sig[c] != sig) { continue; }
      if (signature_match(trie, raw, inner, c)) { return trie.flat_children[c]; }
    }
    return -1;
  }
  // Singleton buckets skip signature loads and compare the only candidate.
  if (signature_match(trie, raw, inner, lo)) { return trie.flat_children[lo]; }
  return -1;
}

// One trie walk per row group. All its lanes parse in lockstep; only its leader records
// intervals. First matches own their subtree even for nulls or missing descendants.
template <int max_trie_depth, int R>
__launch_bounds__(k_merged_block_size, k_merged_min_blocks_per_sm) CUDF_KERNEL
  void get_json_object_kernel_merged_named(cudf::column_device_view input,
                                           cudf::device_span<json_path_processing_data> paths,
                                           merged_trie_view trie,
                                           int8_t* depth_error)
{
  using record        = cuda::std::conditional_t<R == 1, merged_path_out, merged_path_packed>;
  constexpr int width = warp_size / R;
  auto const row = static_cast<cudf::size_type>(cudf::detail::grid_1d::global_thread_id() / width);
  if (row >= input.size()) { return; }
  auto const lane     = threadIdx.x % warp_size;
  auto const row_lane = lane % width;
  auto const mask     = width == 32 ? 0xffffffffu : (0xffffu << ((lane / width) * width));
  auto const count    = static_cast<int32_t>(paths.size());
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

  json_parser_impl<true> p{char_range{str}};
  p.enable_warp_cooperative_skip(static_cast<int>(lane), width, mask);
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
    auto const kind =
      certify_raw_subtree(str.data() + begin, end - begin) ? k_merged_kind_value_raw : int8_t{0};
    record_slot(slot, begin, end, kind);
  };
  bool valid = true;
  if (p.next_token() != json_token::START_OBJECT) {
    valid = p.try_skip_children();
  } else {
    struct frame {
      int32_t node, parent_slot, begin;
    };
    frame stack[max_trie_depth];
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
        if (sp == max_trie_depth) {
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
            auto const end = begin + static_cast<int32_t>(p.current_range().size());
            int8_t kind    = 0;
            if (value == json_token::VALUE_STRING) {
              kind = p.current_token_is_escaped_string() ? k_merged_kind_string
                                                         : k_merged_kind_string_raw;
            } else if (value == json_token::VALUE_TRUE || value == json_token::VALUE_FALSE ||
                       (value == json_token::VALUE_NUMBER_INT &&
                        !(end - begin == 2 && str.data()[begin] == '-' &&
                          str.data()[begin + 1] == '0'))) {
              kind = k_merged_kind_value_raw;
            }
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
    char const* chars    = dest;
    auto const value     = char_range{str.data() + rec.begin, rec.end - rec.begin};
    cudf::size_type size = 0;
    bool ok              = true;
    // Borrow raw bytes through gather; a valid empty string still has a non-null pointer.
    if (rec.kind == k_merged_kind_string_raw) {
      chars = value.data() + 1;
      size  = value.size() - 2;
    } else if (rec.kind == k_merged_kind_value_raw) {
      chars = value.data();
      size  = value.size();
    } else if (rec.kind == k_merged_kind_string) {
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
  std::array<std::size_t, 17> offsets;
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
  std::vector<uint64_t> flat_name_sig;
  std::vector<int32_t> len_prefix, max_name_len;
  std::vector<std::size_t> len_prefix_beg;
  for (auto& children : nodes) {
    std::stable_sort(children.begin(), children.end(), [&](int32_t a, int32_t b) {
      return slots[a].name.size() < slots[b].name.size();
    });
    children_beg.push_back(static_cast<int32_t>(flat_children.size()));
    children_cnt.push_back(static_cast<int32_t>(children.size()));
    len_prefix_beg.push_back(len_prefix.size());
    max_name_len.push_back(
      children.size() <= 1 ? 0 : static_cast<int32_t>(slots[children.back()].name.size()));
    if (children.size() > 1) {
      // Row[L] counts children shorter than L. Size by node to bound long-name metadata.
      std::size_t len = 0;
      for (std::size_t c = 0; c < children.size(); ++c) {
        auto const end = slots[children[c]].name.size() + 1;
        len_prefix.insert(len_prefix.end(), end - len, static_cast<int32_t>(c));
        len = end;
      }
      len_prefix.push_back(static_cast<int32_t>(children.size()));
    }
    for (auto const s : children) {
      auto const& slot = slots[s];
      flat_children.push_back(s);
      flat_name_off.push_back(static_cast<int32_t>(names.size()));
      flat_name_len.push_back(static_cast<int32_t>(slot.name.size()));
      names += slot.name;
      flat_name_sig.push_back(name_sig(slot.name.data(), flat_name_len.back()));
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
                                   bytes_of(flat_name_len),
                                   bytes_of(flat_name_sig),
                                   bytes_of(len_prefix),
                                   bytes_of(len_prefix_beg),
                                   bytes_of(max_name_len)};
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
                     ints(12),
                     reinterpret_cast<uint64_t const*>(base + s[13]),
                     ints(14),
                     reinterpret_cast<std::size_t const*>(base + s[15]),
                     ints(16)};
}

// Both kernels require the bound trie and group-ordered path data through completion.
void launch_merged_named(cudf::column_device_view const& input,
                         cudf::device_span<json_path_processing_data> paths,
                         merged_group_plan const& plan,
                         int64_t max_row_size,
                         int8_t* depth_error,
                         cuda::stream_ref stream)
{
  constexpr auto warps = k_merged_block_size / warp_size;
  static_assert(warps * k_merged_max_paths * sizeof(merged_path_out) <= 48 * 1'024);
  static_assert(sizeof(merged_path_out) == 2 * sizeof(merged_path_packed));
  auto const rows = static_cast<std::size_t>(input.size());
  auto const smem = warps * plan.num_paths * sizeof(merged_path_out);
  int device = 0, sms = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  // Two rows per warp pay off after one estimated walk round, if row offsets fit 29 bits.
  bool const pair_rows =
    rows > sms * warps * k_merged_min_blocks_per_sm && max_row_size <= k_packed_end_max;
  auto const launch = [&]<int Depth, int R>() {
    auto const blocks = (rows + R * warps - 1) / (R * warps);
    get_json_object_kernel_merged_named<Depth, R>
      <<<blocks, k_merged_block_size, smem, stream.get()>>>(input, paths, plan.view, depth_error);
  };
  if (plan.max_trie_depth == k_merged_max_depth) {
    if (pair_rows) {
      launch.template operator()<k_merged_max_depth, 2>();
    } else {
      launch.template operator()<k_merged_max_depth, 1>();
    }
  } else {
    if (pair_rows) {
      launch.template operator()<k_merged_deep_max_depth, 2>();
    } else {
      launch.template operator()<k_merged_deep_max_depth, 1>();
    }
  }
}

// Per-path codes survive retry compaction; a different evaluator can exceed measured slots.
struct grouped_routes {
  std::vector<char> codes;
  std::vector<std::size_t> general;
  std::vector<std::size_t> merged;
  bool deep = false;
};

int named_depth(cudf::host_span<std::tuple<path_instruction_type, std::string, int32_t> const> path)
{
  return path.size() <= k_merged_deep_max_depth &&
             std::all_of(
               path.begin(),
               path.end(),
               [](auto const& op) { return std::get<0>(op) == path_instruction_type::NAMED; })
           ? static_cast<int>(path.size())
           : 0;
}

// Preserve batch order within each family. A deep group must qualify before absorbing shallow.
grouped_routes plan_routes(path_batch const& paths,
                           cudf::size_type rows,
                           bool last_non_null,
                           std::size_t min_eligible)
{
  grouped_routes plan{std::vector<char>(paths.size()), {}, {}, false};
  std::vector<std::size_t> deep;
  for (std::size_t i = 0; i < paths.size(); ++i) {
    auto const depth = last_non_null || rows < k_min_rows_per_family ? 0 : named_depth(paths[i]);
    if (depth == 0 || plan.merged.size() + deep.size() == k_merged_max_paths) {
      plan.general.push_back(i);
    } else if (depth > k_merged_max_depth) {
      deep.push_back(i);
    } else {
      plan.merged.push_back(i);
    }
  }
  if (deep.size() >= std::max<std::size_t>(k_deep_merged_min_paths, min_eligible)) {
    plan.deep = true;
    plan.merged.insert(plan.merged.end(), deep.begin(), deep.end());
    for (auto const i : deep) {
      plan.codes[i] = 2;
    }
  } else {
    plan.general.insert(plan.general.end(), deep.begin(), deep.end());
  }
  if (!plan.deep && plan.merged.size() < min_eligible) {
    plan.general.insert(plan.general.end(), plan.merged.begin(), plan.merged.end());
    plan.merged.clear();
  }
  for (auto const i : plan.merged) {
    if (plan.codes[i] == 0) { plan.codes[i] = 1; }
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
  int64_t max_row_size,
  cuda::stream_ref stream)
{
  auto const mr = cudf::get_current_device_resource_ref();
  if (routes.merged.empty()) {
    auto data = cudf::detail::make_device_uvector_async(host_paths, stream, mr);
    kernel_launcher::exec(input, data, depth_error, use_last_non_null, stream);
    return std::nullopt;
  }
  auto const depth         = routes.deep ? k_merged_deep_max_depth : k_merged_max_depth;
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
    launch_merged_named(input, merged, plan, max_row_size, depth_error, stream);
  } else {
    auto fork = cudf::detail::fork_streams(stream, 1);
    launch_merged_named(input, merged, plan, max_row_size, depth_error, stream);
    kernel_launcher::exec(input, general, depth_error, use_last_non_null, fork[0]);
    cudf::detail::join_streams(fork, stream);
  }
  return grouped_upload{std::move(host), std::move(device)};
}

using row_pair = cuda::std::pair<char const*, cudf::size_type>;

// One descriptor per checked path. Pair bytes may borrow either input or scratch.
struct output_segment {
  row_pair const* pairs;
  cudf::bitmask_type* mask;
  unsigned long long bytes{};
  cudf::size_type valid{};
  void* offsets{};
  char* chars{};
  cudf::data_type offset_type{};
};

constexpr int output_block_size = 256;

// Each block owns whole mask words; the descriptors start with zeroed totals.
CUDF_KERNEL void measure_output(output_segment* cols, int64_t tiles, int rows, int tiles_per_path)
{
  using reduce = cub::BlockReduce<unsigned long long, output_block_size>;
  __shared__ reduce::TempStorage temp;
  auto const block_idx = cuda::block.index_as<cudf::thread_index_type>(cuda::grid).x;
  auto const stride    = cuda::block.count_as<cudf::thread_index_type>(cuda::grid);
  for (auto tile = block_idx; tile < tiles; tile += stride) {
    auto& col       = cols[tile / tiles_per_path];
    auto const row  = (tile % tiles_per_path) * output_block_size + threadIdx.x;
    auto const pair = row < rows ? col.pairs[row] : row_pair{nullptr, 0};
    auto const mask = __ballot_sync(0xffffffffu, pair.first != nullptr);
    if (row < rows && threadIdx.x % warp_size == 0) { col.mask[row / warp_size] = mask; }
    auto const bytes = reduce(temp).Sum(pair.second);
    auto const valid = cudf::detail::single_lane_block_sum_reduce<output_block_size>(__popc(mask));
    if (threadIdx.x == 0) {
      cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>{col.bytes}.fetch_add(
        bytes, cuda::memory_order_relaxed);
      cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device>{col.valid}.fetch_add(
        valid, cuda::memory_order_relaxed);
    }
  }
}

}  // namespace

std::vector<std::unique_ptr<cudf::column>> eval_batch(
  cudf::column_device_view const& input,
  cudf::detail::input_offsetalator const& offsets,
  path_batch const& paths,
  int64_t scratch_size,
  bool use_last_non_null,
  request_scan const& scan,
  bool use_segmented_gather,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  std::vector<char>* routes_out)
{
  auto const count  = paths.size();
  auto const rows   = input.size();
  auto const routes = plan_routes(paths, rows, use_last_non_null, scan.min_eligible);
  for (auto const& path : paths) {
    CUDF_EXPECTS(path.size() <= MAX_JSON_PATH_DEPTH,
                 "JSON Path has depth exceeds the maximum allowed value.");
  }
  auto host_flags     = cudf::detail::make_pinned_vector_async<int8_t>(count + 1, stream);
  auto const commands = stage_commands(paths, routes.general, stream);
  cuda::device_buffer<row_pair> pairs(stream, mr, count * rows, cuda::no_init);
  std::vector<cuda::device_buffer<char>> scratch;
  std::vector<json_path_processing_data> jobs;
  rmm::device_uvector<int8_t> flags(count + 1, stream);
  scratch.reserve(count);
  jobs.reserve(count);
  using cudf::detail::input_offsetalator;
  auto const add_slot = [&](std::size_t i, input_offsetalator slots, int64_t base, int64_t bytes) {
    scratch.emplace_back(stream, mr, bytes, cuda::no_init);
    jobs.push_back({commands.paths[i],
                    slots,
                    base,
                    pairs.data() + i * rows,
                    scratch.back().data(),
                    flags.data() + jobs.size()});
  };
  for (std::size_t i = 0; i < count; ++i) {
    add_slot(i, offsets, scan.base_offset, scratch_size);
  }
  // The flag readback orders both kernels and their uploads before any pair access.
  auto const run_pass = [&](path_batch const& pass, grouped_routes const& plan) {
    auto const n = pass.size();
    CUDF_CUDA_TRY(cudaMemsetAsync(flags.data(), 0, n + 1, stream.get()));
    auto* const depth = flags.data() + n;
    [[maybe_unused]] auto const upload =
      launch_grouped(input, pass, plan, jobs, depth, use_last_non_null, scan.max_row_bytes, stream);
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(host_flags.data(), flags.data(), n + 1, cudaMemcpyDefault, stream.get()));
    stream.sync();
    CUDF_EXPECTS(host_flags[n] == 0, "The processed input has nesting depth exceeds depth limit.");
  };
  run_pass(paths, routes);
  if (routes_out != nullptr) { *routes_out = routes.codes; }
  std::vector<std::unique_ptr<cudf::column>> out(count);
  // Materialize only checked pairs; flagged scratch may contain overlapping writes.
  auto const gather = [&](std::vector<std::size_t> const& ids) {
    if (ids.empty()) { return; }
    if (!use_segmented_gather) {
      CUDF_EXPECTS(
        rows < INT32_MAX, "Size of output exceeds the column size limit", std::overflow_error);
      auto const i = ids.front();
      auto const view =
        cudf::device_span<row_pair const>{pairs.data() + i * rows, static_cast<std::size_t>(rows)};
      out[i] = std::move(cudf::make_strings_column_batch({view}, stream, mr).front());
      return;
    }
    auto const n       = static_cast<int64_t>(ids.size());
    auto const temp_mr = cudf::get_current_device_resource_ref();
    auto host          = cudf::detail::make_pinned_vector_async<output_segment>(n, stream);
    cuda::device_buffer<output_segment> d(stream, temp_mr, n, cuda::no_init);
    auto const bytes = host.size() * sizeof(output_segment);
    std::vector<cuda::device_buffer<cudf::bitmask_type>> masks;
    masks.reserve(n);
    for (int64_t p = 0; p < n; ++p) {
      masks.push_back(cudf::create_null_mask(rows, cudf::mask_state::UNINITIALIZED, stream, mr));
      host[p] = {pairs.data() + ids[p] * rows, masks.back().data()};
    }
    CUDF_CUDA_TRY(cudaMemcpyAsync(d.data(), host.data(), bytes, cudaMemcpyDefault, stream.get()));
    auto const per_path = cuda::ceil_div(rows, output_block_size);
    auto const tiles    = n * per_path;
    measure_output<<<std::min<int64_t>(tiles, INT32_MAX), output_block_size, 0, stream.get()>>>(
      d.data(), tiles, rows, per_path);
    CUDF_CUDA_TRY(cudaGetLastError());
    CUDF_CUDA_TRY(cudaMemcpyAsync(host.data(), d.data(), bytes, cudaMemcpyDefault, stream.get()));
    stream.sync();

    auto const limit = static_cast<unsigned long long>(cudf::strings::get_offset64_threshold());
    auto const wide  = [limit](output_segment const& col) { return col.bytes >= limit; };
    CUDF_EXPECTS(rows < INT32_MAX && (cudf::strings::is_large_strings_enabled() ||
                                      std::none_of(host.begin(), host.end(), wide)),
                 "Size of output exceeds the column size limit",
                 std::overflow_error);
    for (int64_t p = 0; p < n; ++p) {
      auto& col       = host[p];
      col.offset_type = cudf::data_type{wide(col) ? cudf::type_id::INT64 : cudf::type_id::INT32};
      auto child      = cudf::make_numeric_column(
        col.offset_type, rows + 1, cudf::mask_state::UNALLOCATED, stream, mr);
      rmm::device_buffer chars(col.bytes, stream, mr);
      col.offsets      = child->mutable_view().head();
      col.chars        = static_cast<char*>(chars.data());
      auto const nulls = rows - col.valid;
      if (nulls == 0) { masks[p] = cuda::device_buffer<cudf::bitmask_type>(stream, mr); }
      out[ids[p]] = cudf::make_strings_column(
        rows, std::move(child), std::move(chars), nulls, std::move(masks[p]));
    }

    // Zero-byte columns still need offsets, but have no destination buffer for CUB.
    auto const end =
      std::partition(host.begin(), host.end(), [](auto const& c) { return c.bytes; });
    CUDF_CUDA_TRY(cudaMemcpyAsync(d.data(), host.data(), bytes, cudaMemcpyDefault, stream.get()));
    stream.sync();  // The descriptor upload borrows the host table.
    auto const* cols  = d.data();
    auto const idx    = cuda::counting_iterator<int64_t>{0};
    auto const stride = cuda::fast_mod_div<int64_t>{rows};
    auto const keys   = cuda::transform_iterator(
      idx, [stride] __device__(int64_t i) -> int64_t { return i / stride; });
    auto const pairs_it =
      cuda::transform_iterator(idx, [cols, stride] __device__(int64_t i) -> row_pair const& {
        return cols[i / stride].pairs[i % stride];
      });
    auto const sizes = cuda::transform_iterator(pairs_it, &row_pair::second);
    // Inclusive prefixes write offsets at row + 1 without a sentinel input per path.
    auto const store = [cols, stride] __device__(int64_t i, int64_t value) -> void {
      auto const& col = cols[i / stride];
      auto const row  = static_cast<cudf::size_type>(i % stride);
      cudf::detail::output_offsetalator const target{col.offsets, col.offset_type};
      if (row == 0) { target[0] = 0; }
      target[row + 1] = value;
    };
    auto const stream_prop = cuda::std::execution::prop{cuda::get_stream_t{}, stream};
    auto const mr_prop     = cuda::std::execution::prop{cuda::mr::get_memory_resource_t{}, temp_mr};
    auto const env         = cuda::std::execution::env{stream_prop, mr_prop};
    auto const target      = cuda::tabulate_output_iterator(store);
    CUDF_CUDA_TRY(cub::DeviceScan::InclusiveScanByKey(
      keys, sizes, target, cuda::std::plus<int64_t>{}, n * rows, cuda::std::equal_to<>{}, env));
    if (end == host.begin()) { return; }
    auto const dst = [cols, stride] __device__(int64_t i) -> char* {
      auto const& col = cols[i / stride];
      return col.chars + cudf::detail::input_offsetalator{col.offsets, col.offset_type}[i % stride];
    };
    auto const src  = cuda::transform_iterator(pairs_it, &row_pair::first);
    auto const dest = cuda::transform_iterator(idx, dst);
    CUDF_CUDA_TRY(cub::DeviceMemcpy::Batched(src, dest, sizes, (end - host.begin()) * rows, env));
  };
  std::vector<std::size_t> accepted, flagged;
  for (std::size_t i = 0; i < count; ++i) {
    (host_flags[i] == 0 ? accepted : flagged).push_back(i);
  }
  gather(accepted);
  if (flagged.empty()) { return out; }
  scratch.clear();  // Device frees follow the accepted gather on this stream.

  path_batch retry_paths;
  grouped_routes retry_routes;
  std::vector<std::unique_ptr<cudf::column>> retry_offsets;
  jobs.clear();
  // A different evaluator can exceed the slots measured by the first pass.
  for (auto const i : flagged) {
    auto const code = routes.codes[i];
    (code == 0 ? retry_routes.general : retry_routes.merged).push_back(retry_paths.size());
    retry_routes.deep |= code == 2;
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
    add_slot(i, cudf::detail::offsetalator_factory::make_input_iterator(col->view()), 0, bytes);
    retry_offsets.emplace_back(std::move(col));
    retry_paths.emplace_back(paths[i]);
  }
  run_pass(retry_paths, retry_routes);
  auto const end = host_flags.begin() + flagged.size();
  CUDF_EXPECTS(std::none_of(host_flags.begin(), end, [](auto v) { return v != 0; }),
               "Unexpected out-of-bound write in get_json_object kernel.");
  // Retry offsets are capacities, not committed lengths; gather from the rewritten pairs.
  gather(flagged);
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
  std::vector<char>* routes_out,
  gjo_scan_mode* scan_mode)
{
  auto const count = paths.size();
  std::vector<std::unique_ptr<cudf::column>> out(count);
  if (routes_out != nullptr) { routes_out->assign(count, 0); }
  if (scan_mode != nullptr) { *scan_mode = gjo_scan_mode::offset_only; }
  if (count == 0) { return out; }
  if (input.is_empty() || input.size() == input.null_count()) {
    for (auto& col : out) {
      col = std::make_unique<cudf::column>(input.parent(), stream, mr);
    }
    return out;
  }
  auto const offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
  auto const rows  = input.size();
  auto const chars = input.chars_begin(stream);
  int32_t eligible = 0;
  if (!use_last_non_null && rows >= k_min_rows_per_family && count >= k_merged_min_paths &&
      (parallel_override <= 0 || parallel_override >= k_merged_min_paths)) {
    int32_t shallow = 0, deep = 0;
    for (auto const& path : paths) {
      auto const depth = named_depth(path);
      if (depth > k_merged_max_depth) {
        if (deep < k_merged_max_paths) { ++deep; }
      } else if (depth != 0 && shallow < k_merged_max_paths) {
        ++shallow;
      }
    }
    if (deep < k_deep_merged_min_paths ||
        (parallel_override > 0 && parallel_override < k_deep_merged_min_paths)) {
      deep = 0;
    }
    eligible = std::max(shallow, deep);
  }
  auto h_scan = cudf::detail::make_pinned_vector_async<request_scan>(1, stream);
  rmm::device_uvector<request_scan> d_scan(1, stream);
  if (eligible >= k_merged_min_paths) {
    scan_request_kernel<true><<<1, 256, 0, stream.get()>>>(
      offsets, chars, rows, memory_budget_bytes, parallel_override, eligible, d_scan.data());
    if (scan_mode != nullptr) { *scan_mode = gjo_scan_mode::sampled; }
  } else {
    scan_request_kernel<false><<<1, 256, 0, stream.get()>>>(
      offsets, nullptr, rows, memory_budget_bytes, parallel_override, 0, d_scan.data());
  }
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    h_scan.data(), d_scan.data(), sizeof(request_scan), cudaMemcpyDefault, stream.get()));
  std::vector<std::size_t> order(count);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](auto a, auto b) { return paths[a] < paths[b]; });
  stream.sync();
  request_scan scan       = h_scan[0];
  auto const scratch_size = scan.max_row_bytes * 10 + scan.selected_chars;
  // A budget charges each path after accepting it, even if a single path exceeds the budget.
  auto const width = parallel_override > 0 ? static_cast<std::size_t>(parallel_override)
                     : (memory_budget_bytes <= 0 || scratch_size == 0)
                       ? count
                       : static_cast<std::size_t>(1 + (memory_budget_bytes - 1) / scratch_size);
  if (scan.min_eligible == 0) {
    int device = 0, sms = 0;
    CUDF_CUDA_TRY(cudaGetDevice(&device));
    CUDF_CUDA_TRY(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    auto const blocks = std::min(1 + (rows - 1) / k_exact_scan_threads, 8 * sms);
    auto const limit =
      make_request_limit(scan.selected_chars / rows, std::min<std::size_t>(width, eligible));
    scan_exact_admission<<<blocks, k_exact_scan_threads, 0, stream.get()>>>(
      offsets, chars, rows, limit, d_scan.data());
    CUDF_CUDA_TRY(cudaPeekAtLastError());
    CUDF_CUDA_TRY(cudaMemcpyAsync(&h_scan[0].min_eligible,
                                  &d_scan.data()->min_eligible,
                                  sizeof(scan.min_eligible),
                                  cudaMemcpyDefault,
                                  stream.get()));
    stream.sync();
    scan.min_eligible = h_scan[0].min_eligible;
    if (scan_mode != nullptr) { *scan_mode = gjo_scan_mode::exact; }
  }
  auto const device = cudf::column_device_view::create(
    input.parent(), stream, cudf::get_current_device_resource_ref());
  path_batch batch;
  std::vector<char> batch_routes;
  auto* const observed = routes_out == nullptr ? nullptr : &batch_routes;
  // Keep singleton tails and retries of wider requests on the segmented gather.
  auto const use_segmented_gather = std::min(count, width) > 1;
  for (std::size_t begin = 0; begin < count; begin += width) {
    auto const size = std::min(width, count - begin);
    batch.clear();
    for (std::size_t i = 0; i < size; ++i) {
      batch.emplace_back(paths[order[begin + i]]);
    }
    auto next = eval_batch(*device,
                           offsets,
                           batch,
                           scratch_size,
                           use_last_non_null,
                           scan,
                           use_segmented_gather,
                           stream,
                           mr,
                           observed);
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
