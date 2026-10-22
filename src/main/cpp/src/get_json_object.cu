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
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/atomic>
#include <cuda/cmath>
#include <cuda/functional>
#include <cuda/hierarchy>
#include <cuda/std/algorithm>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>
#include <cuda_runtime_api.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/uninitialized_fill.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <numeric>
#include <optional>
#include <string>

namespace spark_rapids_jni {

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

  // Container copies validate each child through the matching close token.
  // The merged walk's CoopSkip=true parser and the general route's CoopSkip=false parser share
  // this writer; the body is parser-mode-agnostic.
  template <bool CoopSkip>
  __device__ bool copy_current_structure(json_parser_impl<CoopSkip>& parser, char* out_begin)
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
   * Child bytes are already at the parent's end; make room for the deferred comma and brackets.
   * With a comma: parent prefix [1 and child 2,3,4 become [1,2,3,4 without wrapping.
   * With wrapping too, the same prefix and child become [1,[2,3,4].
   * write_outer_array_tokens completes the brackets deferred by
   * write_first_start_array_without_output; a single wildcard match stays unwrapped.
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

  // Copy backward so shifting an overlapping block cannot overwrite unread bytes.
  // Moving two bytes of 1 2 0 0 by one yields 1 1 2 0; by two yields 1 2 1 2.
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
 * @brief Parse a single json string using the provided command buffer.
 *
 * @param p The JSON parser for input string
 * @param path_commands The command buffer to be applied to the string
 * @param out_buf Buffer user to store the string resulted from the query
 * @param max_path_depth_exceeded A marker to record if the maximum path depth has been reached
 *        during parsing the input string
 * @return A tuple of the result code, the committed output size, and the maximum write
 *         footprint (bytes touched, including discarded child writes)
 */
__device__ cuda::std::tuple<bool, cudf::size_type, cudf::size_type> evaluate_path(
  json_parser& p,
  cudf::device_span<path_instruction const> path_commands,
  char* out_buf,
  int8_t* max_path_depth_exceeded)
{
  // Define stack; plus 1 indicates root context task needs an extra memory.
  context stack[MAX_JSON_PATH_DEPTH + 1];
  int stack_size = 0;

  // Bytes written by any generator in this row; covers child bytes abandoned by a
  // no-match wildcard step.
  int max_footprint = 0;

  // Returns a failed parse result carrying the write extent (bytes written so far).
  auto const parse_failure = [&](json_generator const& g) {
    auto const extent = g.get_offset() + g.get_output_len();
    return cuda::std::tuple{false, extent, cuda::std::max(extent, max_footprint)};
  };
  // Variant for the wildcard step: the extent is the larger of the parent and child
  // generators' writes.
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
          // JSON validation check
          if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }
          // push back task
          // add child task
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
        // general case: just copy the child tree verbatim
        if (!(ctx.g.copy_current_structure(p, out_buf))) {
          // JSON validation check
          return parse_failure(ctx.g);
        }
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
              // JSON validation check
              if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }

              p.next_token();
              // JSON validation check
              if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }

              // skip value of FIELD_NAME
              if (!p.try_skip_children()) {
                // JSON validation check
                return parse_failure(ctx.g);
              }
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

            // JSON validation check
            if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }

            if (is_name_matched) {
              p.next_token();
              // JSON validation check
              if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }

              // A null first occurrence does not fall back to a later duplicate.
              if (json_token::VALUE_NULL == p.get_current_token()) { return parse_failure(ctx.g); }
              // push sub task; sub task will update the result of path 4
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
          // JSON validation check
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
          // JSON validation check
          if (json_token::ERROR == p.get_current_token()) {
            // The child generator may have written past this context's extent; report the
            // larger of the two so the out-of-bound check sees the real overrun.
            return parse_failure_with_child(ctx.g, child_g);
          }
          // track the number of array elements and only emit an outer array if
          // we've written more than one element, this matches Hive's behavior
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
          // JSON validation check
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
        // JSON validation check
        if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }
        ctx.is_first_enter = false;

        int i = idx;
        while (i > 0) {
          if (p.get_current_token() == json_token::END_ARRAY) {
            // terminate, nothing has been written
            return parse_failure(ctx.g);
          }

          if (!p.try_skip_children()) { return parse_failure(ctx.g); }

          p.next_token();
          // JSON validation check
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
        // JSON validation check
        if (json_token::ERROR == p.get_current_token()) { return parse_failure(ctx.g); }

        int i = idx;
        while (i > 0) {
          if (p.get_current_token() == json_token::END_ARRAY) {
            // terminate, nothing has been written
            return parse_failure(ctx.g);
          }

          if (!p.try_skip_children()) { return parse_failure(ctx.g); }

          p.next_token();
          // JSON validation check
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
        // default case path, return false for this task
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

          // Case 4.
        case evaluation_case_path::START_OBJECT___MATCHED_NAME_PATH: {
          p_ctx.dirty = ctx.dirty;

          p_ctx.g = ctx.g;

          break;
        }

          // Case 6.
        case evaluation_case_path::START_ARRAY___MATCHED_WILDCARD___STYLE_NOT_QUOTED: {
          p_ctx.dirty += ctx.dirty;
          p_ctx.child_g = ctx.g;
          max_footprint =
            cuda::std::max(max_footprint, ctx.g.get_offset() + ctx.g.get_output_len());
          break;
        }

          // Cases 8 and 9: the index walk consumed the array elements; the child task's
          // array iteration resumes here and must run to the array's end.
        case evaluation_case_path::START_ARRAY___MATCHED_INDEX_AND_WILDCARD:
        case evaluation_case_path::START_ARRAY___MATCHED_INDEX: {
          p_ctx.dirty += ctx.dirty;

          while (p.next_token() != json_token::END_ARRAY) {
            // JSON validation check
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
  // A valid row's length is its committed output; discarded child bytes (a no-match
  // wildcard step) must not leak into it, they only matter for the out-of-bound check.
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
 * Evaluates one (row, path) per active thread; padded path slots and rows return without parsing.
 * Each row uses ceil(num_paths / warp_size) whole warps, so no warp mixes rows.
 *
 * Keep both __launch_bounds__ arguments: NVCC's default register heuristic caused costly spilling
 * in this parser. Specifying min_block_per_sm lets it use more registers within the launch limit
 * to reduce instructions; it is a compiler target, not a guarantee of measured residency.
 *
 * path_data borrows its storage through kernel completion. Retries write into new buffers and
 * overwrite the output pairs; rebuilt columns use the retry's lengths and validity.
 * The host checks the shared depth flag before consuming output or deciding on a size retry.
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
    // Specifying both bounds lets NVCC trade extra registers for fewer instructions.
    // Testing favored these 128-thread, 8-block bounds over a 256-thread, 4-block alternative.
    static constexpr int block_size       = 128;
    static constexpr int min_block_per_sm = 8;
    constexpr std::size_t warp_size       = 32;  // matches cudf::detail::warp_size
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

/**
 * Uploads all path instructions and packs their names without synchronizing between paths.
 * The returned host commands and names back asynchronous copies; retain every returned owner
 * through the batch's ordering synchronization, and device owners through their kernel use.
 */
std::tuple<std::vector<rmm::device_uvector<path_instruction>>,
           std::unique_ptr<std::vector<std::vector<path_instruction>>>,
           cudf::string_scalar,
           std::string>
construct_path_commands(
  std::vector<cudf::host_span<std::tuple<path_instruction_type, std::string, int32_t> const>> const&
    json_paths,
  cuda::stream_ref stream)
{
  auto h_inst_names = [&] {
    std::size_t length{0};
    for (auto const& instructions : json_paths) {
      for (auto const& [type, name, index] : instructions) {
        if (type == path_instruction_type::NAMED) { length += name.length(); }
      }
    }
    std::string all_names;
    all_names.reserve(length);
    for (auto const& instructions : json_paths) {
      for (auto const& [type, name, index] : instructions) {
        if (type == path_instruction_type::NAMED) { all_names += name; }
      }
    }
    return all_names;
  }();
  auto d_inst_names =
    cudf::string_scalar(h_inst_names, true, stream, cudf::get_current_device_resource_ref());

  std::size_t name_pos{0};
  auto h_path_commands = std::make_unique<std::vector<std::vector<path_instruction>>>();
  h_path_commands->reserve(json_paths.size());

  for (auto const& instructions : json_paths) {
    h_path_commands->emplace_back();
    auto& path_commands = h_path_commands->back();
    path_commands.reserve(instructions.size());

    for (auto const& [type, name, index] : instructions) {
      path_commands.emplace_back(path_instruction{type});

      if (type == path_instruction_type::INDEX) {
        path_commands.back().index = index;
      } else if (type == path_instruction_type::NAMED) {
        path_commands.back().name = cudf::string_view(d_inst_names.data() + name_pos, name.size());
        name_pos += name.size();
      } else if (type != path_instruction_type::WILDCARD) {
        CUDF_FAIL("Invalid path instruction type");
      }
    }
  }

  auto d_path_commands = std::vector<rmm::device_uvector<path_instruction>>{};
  d_path_commands.reserve(h_path_commands->size());
  for (auto const& path_commands : *h_path_commands) {
    d_path_commands.emplace_back(cudf::detail::make_device_uvector_async(
      path_commands, stream, cudf::get_current_device_resource_ref()));
  }

  return {std::move(d_path_commands),
          std::move(h_path_commands),
          std::move(d_inst_names),
          std::move(h_inst_names)};
}

/**
 * One 256-thread block writes {maximum row bytes, selected bytes, first absolute offset} to stats.
 * The sum covers selected rows, not their parent column; ten longest-row lengths add scratch slack.
 * The offset range must have rows + 1 entries, and stats must have three writable int64 entries.
 */
__global__ void scratch_size_kernel(cudf::detail::input_offsetalator const offsets,
                                    cudf::size_type rows,
                                    int64_t* stats)
{
  // Threads scan disjoint row sets to share the input reads.
  constexpr int block_threads = 256;
  int64_t max_row_size        = 0;
  int64_t chars_size          = 0;
  for (uint32_t idx = threadIdx.x; idx < static_cast<uint32_t>(rows); idx += block_threads) {
    auto const row_size = offsets[idx + 1] - offsets[idx];
    max_row_size        = cuda::std::max(max_row_size, row_size);
    chars_size += row_size;
  }
  __shared__ int64_t s_max[block_threads];
  __shared__ int64_t s_sum[block_threads];
  s_max[threadIdx.x] = max_row_size;
  s_sum[threadIdx.x] = chars_size;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int t = 1; t < block_threads; ++t) {
      max_row_size = cuda::std::max(max_row_size, s_max[t]);
      chars_size += s_sum[t];
    }
    stats[0] = max_row_size;
    stats[1] = chars_size;
    stats[2] = offsets[0];
  }
}

// Returns {scratch bytes, first absolute offset} after one readback of the selected-row statistics.
// A non-null max_row_size_out receives the R=2 row-byte bound from the same readback.
cuda::std::pair<int64_t, int64_t> calc_scratch_size(
  cudf::strings_column_view const& input,
  cudf::detail::input_offsetalator const& in_offsets,
  cuda::stream_ref stream,
  int64_t* max_row_size_out = nullptr)
{
  // Read all statistics together to avoid separate synchronizations.
  auto const rows = input.size();
  rmm::device_uvector<int64_t> d_stats(3, stream);
  scratch_size_kernel<<<1, 256, 0, stream.get()>>>(in_offsets, rows, d_stats.data());
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  auto const h_stats = cudf::detail::make_pinned_vector(d_stats, stream);

  // Export max row size to the merged-walk R=2 gate without another sync.
  // Padding covers speculative writes beyond the final row's input-sized slice.
  auto constexpr padding_rows = 10;
  if (max_row_size_out != nullptr) { *max_row_size_out = h_stats[0]; }
  return {h_stats[0] * padding_rows + h_stats[1], h_stats[2]};
}

// The last marker is depth overflow: throw first; otherwise true means every size marker is zero.
bool check_error(cudf::host_span<int8_t const> error_check)
{
  CUDF_EXPECTS(error_check.back() == 0,
               "The processed input has nesting depth exceeds depth limit.");

  // The checked depth marker is zero, so it can remain in the range.
  return std::none_of(
    error_check.begin(), error_check.end(), [](auto const val) { return val != 0; });
}

namespace {

// Accumulate per-path character totals in 64 bits before narrowing offsets.
using scan_accum_t = int64_t;

// Splits path-major scan indices into (path, row), with one terminal slot after every row segment.
// seg32 uses 32-bit div/mod only when the full path count times (rows + 1) fits signed 31 bits.
struct scan_index_split {
  int64_t seg;  // rows + 1
  int32_t seg32{};

  __device__ cuda::std::pair<int64_t, int64_t> split(int64_t const i) const
  {
    if (seg32 > 0) {
      auto const i32 = static_cast<int32_t>(i);
      return {i32 / seg32, i32 % seg32};
    }
    return {i / seg, i % seg};
  }
};

// Each path contributes its row lengths plus a final zero, so an exclusive scan emits its total.
// A depth error makes pair records unreadable; return zeros without touching those records.
struct batched_lengths_fn {
  cuda::std::pair<char const*, cudf::size_type> const* const* pair_ptrs;
  scan_index_split split_idx;
  int8_t const* depth_error;

  __device__ scan_accum_t operator()(int64_t const i) const
  {
    if (depth_error != nullptr && *depth_error != 0) { return 0; }
    auto const [path, row] = split_idx.split(i);
    if (row == split_idx.seg - 1) { return 0; }  // terminal zero
    return pair_ptrs[path][row].second;
  }
};

/**
 * CUB output proxy for path-major scan results: scatter offsets to each path's destination column.
 * The terminal slot stores an independent int64 total before narrowing the offset to T; callers
 * must check that total before exposing offsets. Each target has rows + 1 writable entries.
 */
template <typename T>
struct scattered_scan_output_iterator {
  T* const* targets{};  // table of per-path destination pointers (const table from uvector)
  scan_index_split split_idx;
  int64_t idx{};
  // Capture each segment's final 64-bit total without another scan.
  int64_t* terminal_totals{};
  using difference_type   = std::ptrdiff_t;
  using value_type        = T;
  using pointer           = void;
  using reference         = scattered_scan_output_iterator const&;
  using iterator_category = std::random_access_iterator_tag;

  __device__ reference operator*() const { return *this; }
  __device__ scattered_scan_output_iterator operator[](int64_t const offset) const
  {
    return *this + offset;
  }
  __device__ scattered_scan_output_iterator& operator++()
  {
    ++idx;
    return *this;
  }
  __device__ scattered_scan_output_iterator operator++(int)
  {
    auto tmp{*this};
    ++idx;
    return tmp;
  }
  __device__ scattered_scan_output_iterator& operator+=(difference_type const offset)
  {
    idx += offset;
    return *this;
  }
  __device__ scattered_scan_output_iterator operator+(difference_type const offset) const
  {
    return scattered_scan_output_iterator{targets, split_idx, idx + offset, terminal_totals};
  }
  __device__ scattered_scan_output_iterator& operator--()
  {
    --idx;
    return *this;
  }
  __device__ scattered_scan_output_iterator operator-(difference_type const offset) const
  {
    return scattered_scan_output_iterator{targets, split_idx, idx - offset, terminal_totals};
  }
  __device__ difference_type operator-(scattered_scan_output_iterator const& other) const
  {
    return idx - other.idx;
  }
  __device__ bool operator==(scattered_scan_output_iterator const& other) const
  {
    return idx == other.idx;
  }
  __device__ bool operator!=(scattered_scan_output_iterator const& other) const
  {
    return idx != other.idx;
  }

  __device__ reference operator=(scan_accum_t const value) const
  {
    auto const [path, row] = split_idx.split(idx);
    if (terminal_totals != nullptr && row == split_idx.seg - 1) {
      // The terminal exclusive-scan slot holds the full path's byte total.
      terminal_totals[path] = value;
    }
    targets[path][row] = static_cast<T>(value);
    return *this;
  }
};

constexpr cudf::size_type validity_block_size = 256;

/**
 * Produces validity from non-null row-pair pointers; a depth flag suppresses unwritten pair reads.
 * Blocks own disjoint mask-word slices and atomically add one count each to a zeroed path counter.
 * Supply 256 threads per block and mask words covering every row, including the tail.
 */
CUDF_KERNEL void batched_validity_kernel(
  int64_t rows,
  cuda::std::pair<char const*, cudf::size_type> const* const* pair_ptrs,
  cudf::bitmask_type* const* mask_ptrs,
  int32_t words_per_path,
  int32_t num_blocks_per_path,
  int32_t words_per_block,
  cudf::size_type* valid_counts,
  int8_t const* depth_error)
{
  // Grid bounds keep block-to-path index arithmetic in 32 bits.
  auto const path       = blockIdx.x / num_blocks_per_path;
  auto const first_word = (blockIdx.x % num_blocks_per_path) * words_per_block;
  auto const last_word =
    first_word + words_per_block < words_per_path ? first_word + words_per_block : words_per_path;
  auto const lane            = threadIdx.x % cudf::detail::warp_size;
  auto const warp_in_block   = threadIdx.x / cudf::detail::warp_size;
  auto const warps_per_block = blockDim.x / cudf::detail::warp_size;

  cudf::size_type warp_valid_count{0};
  bool const has_depth_error = depth_error != nullptr && *depth_error != 0;
  for (int64_t w = first_word + warp_in_block; w < last_word; w += warps_per_block) {
    auto const row = w * cudf::detail::warp_size + lane;
    // Mask the last partial word; cache the path's warp-uniform pair pointer.
    auto const* path_pairs = pair_ptrs[path];
    bool const is_valid    = !has_depth_error && row < rows && path_pairs[row].first != nullptr;

    cudf::bitmask_type const ballot = __ballot_sync(0xFFFF'FFFFu, is_valid);

    if (lane == 0) {
      mask_ptrs[path][w] = ballot;
      warp_valid_count += __popc(ballot);
    }
  }

  // The word loop is warp-uniform, allowing one counter update per block.
  auto const block_count =
    cudf::detail::single_lane_block_sum_reduce<validity_block_size, 0>(warp_valid_count);
  if (threadIdx.x == 0) { atomicAdd(valid_counts + path, block_count); }
}

// Borrowed gather descriptor: row pairs may point into scratch or input; offsets select dst ranges.
// Keep all three device allocations alive until the queued gather has consumed them.
struct gather_entry {
  cuda::std::pair<char const*, cudf::size_type> const* pairs;
  char* dst;
  int32_t const* offsets32;
};

// IdxT is int32 only when entries * rows fits 31 bits.
template <typename IdxT>
struct gather_fn {
  gather_entry const* entries;
  IdxT rows;
  __device__ gather_entry const& entry(int64_t i) const
  {
    return entries[static_cast<IdxT>(i) / rows];
  }
  __device__ IdxT row(int64_t i) const { return static_cast<IdxT>(i) % rows; }
};

// Runs CUB's storage query then execution on one stream; temporary storage is stream-ordered.
template <typename Operation, typename... Args>
void run_cub_two_phase(Operation operation, cuda::stream_ref stream, Args const&... args)
{
  std::size_t bytes = 0;
  CUDF_CUDA_TRY(operation(nullptr, bytes, args..., stream.get()));
  rmm::device_buffer temp(bytes, stream);
  CUDF_CUDA_TRY(operation(temp.data(), bytes, args..., stream.get()));
}

// Each flat index names one (path, row) copy; IdxT must fit the index and offsets must be checked.
// Enqueue after descriptor upload; results and source owners retain device storage through gather.
template <typename IdxT>
void gather_chars_for_group_impl(gather_entry const* entries,
                                 IdxT rows,
                                 int64_t num_buffers,
                                 cuda::stream_ref stream)
{
  gather_fn<IdxT> const gather{entries, rows};
  auto const src_fn = [gather] __device__(int64_t i) {
    // The CUB workaround casts a read-only source pointer to char*.
    return const_cast<char*>(gather.entry(i).pairs[gather.row(i)].first);
  };
  auto const dst_fn = [gather] __device__(int64_t i) {
    auto const& entry = gather.entry(i);
    return entry.dst + entry.offsets32[gather.row(i)];
  };
  auto const size_fn = [gather] __device__(int64_t i) {
    return gather.entry(i).pairs[gather.row(i)].second;
  };

  auto const src_it = thrust::make_transform_iterator(thrust::make_counting_iterator(int64_t{0}),
                                                      cuda::proclaim_return_type<char*>(src_fn));
  auto const dst_it = thrust::make_transform_iterator(thrust::make_counting_iterator(int64_t{0}),
                                                      cuda::proclaim_return_type<char*>(dst_fn));
  auto const size_it =
    thrust::make_transform_iterator(thrust::make_counting_iterator(int64_t{0}),
                                    cuda::proclaim_return_type<cudf::size_type>(size_fn));

  auto const copy = [](auto&&... args) { return cub::DeviceMemcpy::Batched(args...); };
  run_cub_two_phase(copy, stream, src_it, dst_it, size_it, num_buffers);
}

/**
 * Enqueues one batched gather over equal-row-count paths using already checked int32 offsets.
 * Null pairs have zero lengths; valid pairs borrow readable source bytes through gather completion.
 * The local uploaded descriptor table is ordered before return on nonlegacy streams, including
 * per-thread default streams; this synchronization protects host staging, not a final host read.
 */
void gather_chars_for_group(std::vector<gather_entry> const& h_entries,
                            int64_t rows,
                            cuda::stream_ref stream)
{
  if (h_entries.empty()) { return; }
  auto const num_buffers = static_cast<int64_t>(h_entries.size()) * rows;
  auto d_entries         = cudf::detail::make_device_uvector_async(
    h_entries, stream, cudf::get_current_device_resource_ref());
  // h_entries belongs to the finisher; delay destruction until upload completes on explicit/PTDS
  // streams. A null handle is not the legacy stream when per-thread default streams are enabled.
  if (stream.get() != cudaStreamLegacy && (stream.get() != nullptr || cudf::is_ptds_enabled())) {
    stream.sync();
  }

  auto const idx32 = num_buffers < (int64_t{1} << 31);
  if (idx32) {
    return gather_chars_for_group_impl(
      d_entries.data(), static_cast<int32_t>(rows), num_buffers, stream);
  }
  gather_chars_for_group_impl(d_entries.data(), rows, num_buffers, stream);
}

// Returns padded int32-count bytes so the following int64 totals stay eight-byte aligned.
[[nodiscard]] std::size_t valid_counts_bytes(int64_t num_paths)
{
  return (static_cast<std::size_t>(num_paths) * sizeof(cudf::size_type) + sizeof(int64_t) - 1) &
         ~(sizeof(int64_t) - 1);
}

/**
 * Owns offsets, masks, pointer tables and one packed counts/totals/errors readback per path group.
 * Host accessors need a synchronized preparation stream; already_synced records
 * that ordering, not merely that readback was queued. Keep host pointer-table sources until then.
 * Row pairs and their character sources are borrowed; they remain alive through the final gather.
 */
struct prepared_output_metadata {
  std::vector<std::unique_ptr<cudf::column>> offsets_cols;
  std::vector<cuda::device_buffer<cudf::bitmask_type>> null_masks;
  rmm::device_uvector<cuda::std::pair<char const*, cudf::size_type> const*> d_pair_ptrs;
  rmm::device_uvector<cudf::bitmask_type*> d_mask_ptrs;
  rmm::device_uvector<int32_t*> d_targets32;
  // Pinned readback layout: [valid_counts][8-aligned total_sizes][error_check].
  rmm::device_uvector<std::byte> d_packed_readback;
  cudf::detail::host_vector<std::byte> h_packed_readback;
  int64_t num_err{};
  bool already_synced{};
  // Keep H2D pointer-table sources alive through the sync on non-default streams.
  std::vector<cuda::std::pair<char const*, cudf::size_type> const*> h_pair_ptrs;
  std::vector<cudf::bitmask_type*> h_mask_ptrs;
  std::vector<int32_t*> h_targets32;
  int64_t num_paths;
  int64_t rows;

  [[nodiscard]] int64_t const* h_total_sizes() const
  {
    return reinterpret_cast<int64_t const*>(h_packed_readback.data() +
                                            valid_counts_bytes(num_paths));
  }
  [[nodiscard]] cudf::size_type const* h_valid_counts() const
  {
    return reinterpret_cast<cudf::size_type const*>(h_packed_readback.data());
  }
  // Error markers: per-path OOB slots followed by a depth-exceeded slot.
  [[nodiscard]] int8_t const* h_error_check() const
  {
    return num_err > 0 ? reinterpret_cast<int8_t const*>(
                           h_packed_readback.data() + valid_counts_bytes(num_paths) +
                           static_cast<std::size_t>(num_paths) * sizeof(int64_t))
                       : nullptr;
  }
};

// Small batches avoid packed metadata setup; the same gate controls success and retry outputs.
constexpr std::size_t packed_readback_min_paths = 8;

/**
 * Queues offsets/validity for equal-sized pair spans and one packed host readback; no sync.
 * The returned owner retains asynchronous pointer-table sources. If supplied, error_check_slots is
 * caller-initialized per-path size flags followed by one depth flag; depth failures suppress reads
 * of unwritten pair records. Oversized scratch is never gathered here: callers check flags first.
 * No paths/rows queues no readback and returns num_err=0. Retain pairs and errors through ordering.
 */
prepared_output_metadata prepare_output_metadata(
  cudf::column_device_view const& input,
  std::vector<cudf::device_span<cuda::std::pair<char const*, cudf::size_type> const>> const&
    pairs_by_path,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  cudf::device_span<int8_t> error_check_slots = {})
{
  auto const num_paths = static_cast<int64_t>(pairs_by_path.size());
  auto const rows      = static_cast<int64_t>(input.size());
  auto const seg       = rows + 1;
  auto const num_err   = static_cast<int64_t>(error_check_slots.size());

  if (num_paths == 0 || rows == 0) {
    // device_uvector requires a stream-bound zero-size allocation even for empty outputs.
    return prepared_output_metadata{
      {},
      {},
      rmm::device_uvector<cuda::std::pair<char const*, cudf::size_type> const*>(0, stream),
      rmm::device_uvector<cudf::bitmask_type*>(0, stream),
      rmm::device_uvector<int32_t*>(0, stream),
      rmm::device_uvector<std::byte>(0, stream),
      cudf::detail::host_vector<std::byte>(0, {cudf::get_pinned_memory_resource(), stream}),
      0,      // degenerate: no packed readback was enqueued; the caller reads d_error_check itself
      false,  // nothing was enqueued; the finisher's degenerate paths return before any sync
      {},
      {},
      {},
      num_paths,
      rows};
  }

  auto const* depth_error =
    error_check_slots.empty() ? nullptr : error_check_slots.data() + error_check_slots.size() - 1;

  std::vector<std::unique_ptr<cudf::column>> offsets_cols;
  offsets_cols.reserve(pairs_by_path.size());
  std::vector<cuda::device_buffer<cudf::bitmask_type>> null_masks;
  null_masks.reserve(pairs_by_path.size());
  std::vector<cuda::std::pair<char const*, cudf::size_type> const*> h_pair_ptrs;
  std::vector<cudf::bitmask_type*> h_mask_ptrs;
  std::vector<int32_t*> h_targets32;
  h_pair_ptrs.reserve(pairs_by_path.size());
  h_mask_ptrs.reserve(pairs_by_path.size());
  h_targets32.reserve(pairs_by_path.size());

  auto const words_per_path = (rows + cudf::detail::warp_size - 1) / cudf::detail::warp_size;
  for (auto const& span : pairs_by_path) {
    CUDF_EXPECTS(static_cast<int64_t>(span.size()) == rows, "Invalid pair span size");
    offsets_cols.emplace_back(cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                        static_cast<cudf::size_type>(seg),
                                                        cudf::mask_state::UNALLOCATED,
                                                        stream,
                                                        mr));
    null_masks.emplace_back(cudf::create_null_mask(
      static_cast<cudf::size_type>(rows), cudf::mask_state::UNINITIALIZED, stream, mr));
    h_pair_ptrs.emplace_back(span.data());
    h_mask_ptrs.emplace_back(null_masks.back().data());
    h_targets32.emplace_back(offsets_cols.back()->mutable_view().data<cudf::size_type>());
  }
  auto d_pair_ptrs = cudf::detail::make_device_uvector_async(
    h_pair_ptrs, stream, cudf::get_current_device_resource_ref());
  auto d_mask_ptrs = cudf::detail::make_device_uvector_async(
    h_mask_ptrs, stream, cudf::get_current_device_resource_ref());
  auto d_targets32 = cudf::detail::make_device_uvector_async(
    h_targets32, stream, cudf::get_current_device_resource_ref());

  auto const valid_counts_size = valid_counts_bytes(num_paths);
  auto const total_sizes_bytes = static_cast<std::size_t>(num_paths) * sizeof(int64_t);
  auto const error_bytes       = static_cast<std::size_t>(num_err);
  auto const readback_bytes    = valid_counts_size + total_sizes_bytes + error_bytes;
  rmm::device_uvector<std::byte> d_packed_readback(readback_bytes, stream, mr);
  auto* const d_valid_counts = reinterpret_cast<cudf::size_type*>(d_packed_readback.data());
  auto* const d_total_sizes =
    reinterpret_cast<int64_t*>(d_packed_readback.data() + valid_counts_size);
  auto* const d_error_slots =
    num_err > 0
      ? reinterpret_cast<int8_t*>(d_packed_readback.data() + valid_counts_size + total_sizes_bytes)
      : nullptr;
  // Zero valid counts before atomicAdds; scans fill totals and callers initialize errors.
  thrust::uninitialized_fill(rmm::exec_policy_nosync(stream, mr),
                             reinterpret_cast<std::byte*>(d_valid_counts),
                             reinterpret_cast<std::byte*>(d_valid_counts) + valid_counts_size,
                             std::byte{0});
  if (d_error_slots != nullptr) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      d_error_slots, error_check_slots.data(), error_bytes, cudaMemcpyDefault, stream.get()));
  }

  // Only flat counts below 2^31 use 32-bit indexing; accumulation remains 64-bit.
  auto const idx32 = num_paths * seg < (int64_t{1} << 31);
  scan_index_split const split_idx{seg, idx32 ? static_cast<int32_t>(seg) : 0};
  auto const lengths_it =
    thrust::make_transform_iterator(thrust::make_counting_iterator(int64_t{0}),
                                    cuda::proclaim_return_type<scan_accum_t>(batched_lengths_fn{
                                      d_pair_ptrs.data(), split_idx, depth_error}));
  // Scan-by-key parallelizes the flat index space; the same gate chooses key width.
  auto const keys32_it = thrust::make_transform_iterator(
    thrust::make_counting_iterator(int64_t{0}),
    cuda::proclaim_return_type<int32_t>(
      [seg32 = static_cast<int32_t>(seg)] __device__(int64_t const i) {
        return static_cast<int32_t>(i / seg32);
      }));
  auto const keys64_it = thrust::make_transform_iterator(
    thrust::make_counting_iterator(int64_t{0}),
    cuda::proclaim_return_type<int64_t>([seg] __device__(int64_t const i) { return i / seg; }));
  auto const scan_out_it = scattered_scan_output_iterator<cudf::size_type>{
    d_targets32.data(), split_idx, 0, d_total_sizes};

  // CUB's flat item count stays 64-bit even on the 32-bit key path.
  auto const scan_num_items           = num_paths * seg;
  auto const run_exclusive_sum_by_key = [&](auto const& keys_it) {
    auto const scan = [](auto&&... args) { return cub::DeviceScan::ExclusiveSumByKey(args...); };
    run_cub_two_phase(
      scan, stream, keys_it, lengths_it, scan_out_it, scan_num_items, cuda::std::equal_to<>{});
  };
  if (idx32) {
    run_exclusive_sum_by_key(keys32_it);
  } else {
    run_exclusive_sum_by_key(keys64_it);
  }

  {
    // Each grid block owns one path's word slice; pair metadata bounds fit 32-bit grid indices.
    auto const num_blocks_per_path =
      static_cast<int32_t>(cudf::util::div_rounding_up_safe<int64_t>(rows, validity_block_size));
    auto const words_per_block = static_cast<int32_t>(
      cudf::util::div_rounding_up_safe<int64_t>(words_per_path, num_blocks_per_path));
    auto const total_blocks = num_paths * num_blocks_per_path;  // bounded, see note above
    auto const grid =
      cudf::detail::grid_1d{total_blocks * validity_block_size, validity_block_size};
    batched_validity_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
      rows,
      d_pair_ptrs.data(),
      d_mask_ptrs.data(),
      static_cast<int32_t>(words_per_path),
      num_blocks_per_path,
      words_per_block,
      d_valid_counts,
      depth_error);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // Retain pinned D2H buffers through the caller or retry finisher's ordering sync.
  auto h_packed_readback = cudf::detail::make_pinned_vector_async<std::byte>(
    static_cast<std::size_t>(readback_bytes), stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(h_packed_readback.data(),
                                d_packed_readback.data(),
                                readback_bytes,
                                cudaMemcpyDefault,
                                stream.get()));
  return prepared_output_metadata{std::move(offsets_cols),
                                  std::move(null_masks),
                                  std::move(d_pair_ptrs),
                                  std::move(d_mask_ptrs),
                                  std::move(d_targets32),
                                  std::move(d_packed_readback),
                                  std::move(h_packed_readback),
                                  num_err,
                                  false,  // the caller sets this after its error-readback sync
                                  std::move(h_pair_ptrs),
                                  std::move(h_mask_ptrs),
                                  std::move(h_targets32),
                                  num_paths,
                                  rows};
}

/**
 * Builds columns only after the caller has accepted depth/size flags for these paths.
 * Synchronizes the prepared readback unless already ordered, then rejects totals at the int32
 * factory threshold before allocating chars or exposing narrowed offsets. Pairs must retain the
 * same lengths/order used by preparation and readable input/scratch bytes through queued gather.
 * Moves offsets/masks into the results; zero paths returns none, zero rows returns empty strings.
 */
std::vector<std::unique_ptr<cudf::column>> finish_output_columns(
  prepared_output_metadata&& prepared,
  std::vector<cudf::device_span<cuda::std::pair<char const*, cudf::size_type> const>> const&
    pairs_by_path,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_paths = prepared.num_paths;
  auto const rows      = prepared.rows;
  auto& offsets_cols   = prepared.offsets_cols;
  auto& null_masks     = prepared.null_masks;

  if (num_paths == 0) { return {}; }
  if (rows == 0) {
    std::vector<std::unique_ptr<cudf::column>> output;
    output.reserve(pairs_by_path.size());
    for (std::size_t i = 0; i < pairs_by_path.size(); ++i) {
      output.emplace_back(cudf::make_empty_column(cudf::type_id::STRING));
    }
    return output;
  }

  // already_synced avoids a second sync on success; retry synchronizes here.
  if (!prepared.already_synced) { stream.sync(); }
  auto const* h_total_sizes  = prepared.h_total_sizes();
  auto const* h_valid_counts = prepared.h_valid_counts();

  // Match the factory's int32 offset overflow threshold before exposing columns.
  auto const threshold = cudf::strings::get_offset64_threshold();
  for (int64_t p = 0; p < num_paths; ++p) {
    CUDF_EXPECTS(h_total_sizes[p] < threshold,
                 "Size of output exceeds the column size limit",
                 std::overflow_error);
  }

  std::vector<rmm::device_uvector<char>> chars_buffers;
  chars_buffers.reserve(pairs_by_path.size());
  std::vector<gather_entry> h_entries;
  h_entries.reserve(pairs_by_path.size());
  for (int64_t p = 0; p < num_paths; ++p) {
    chars_buffers.emplace_back(
      rmm::device_uvector<char>(static_cast<std::size_t>(h_total_sizes[p]), stream, mr));
    // Skip empty chars buffers in the gather table to avoid zero-byte CUB descriptors.
    if (h_total_sizes[p] == 0) { continue; }  // skip-emission
    h_entries.emplace_back(gather_entry{pairs_by_path[p].data(),
                                        chars_buffers.back().data(),
                                        offsets_cols[p]->view().data<cudf::size_type>()});
  }
  // Pair lengths must sum to the scanned total; the factory trusts this producer too.
  gather_chars_for_group(h_entries, rows, stream);

  std::vector<std::unique_ptr<cudf::column>> output;
  output.reserve(pairs_by_path.size());
  for (int64_t p = 0; p < num_paths; ++p) {
    auto const null_count = static_cast<cudf::size_type>(rows - h_valid_counts[p]);
    output.emplace_back(cudf::make_strings_column(
      static_cast<cudf::size_type>(rows),
      std::move(offsets_cols[p]),
      std::move(chars_buffers[p]).release(),
      null_count,
      null_count > 0 ? std::move(null_masks[p])
                     : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr)));
  }
  return output;
}
}  // namespace

namespace {

// One decoded-name trie walk per pure-NAMED group; first duplicate wins, dead paths stay dead.
// Parse errors null all paths; depth overflow throws only if a path is not already dead.

// Shallow admission: depth <= 2 and >= 8 paths (depth 1 requires >= 124).
inline constexpr std::size_t k_merged_max_depth        = 2;
inline constexpr std::size_t k_merged_min_paths        = 8;
inline constexpr std::size_t k_merged_min_depth1_paths = 124;

/**
 * Borrowed trie for one pure-NAMED group. Nodes are object scopes; slots are their named edges.
 * terminal_ids maps a slot to paths ending there; subtree_pids includes all descendant paths that
 * must become permanently dead when the first matching value cannot satisfy their remaining chain.
 * Path IDs are positions in the compact group, not original batch indices. Retain the device pack.
 */
struct merged_trie_view {
  int32_t const* children_beg;   // [nodes] index into `flat_children`
  int32_t const* children_cnt;   // [nodes]
  int32_t const* child_node;     // [slots] trie node a deeper chain descends to; -1 leaf
  int32_t const* paths_beg;      // [slots] into `terminal_ids`
  int32_t const* paths_cnt;      // [slots] paths ENDING at this slot
  int32_t const* flat_children;  // [total children] SLOT IDS, length-sorted per node
  int32_t const* terminal_ids;   // [total terminals] compact group positions
  char const* names;             // packed operand bytes
  int32_t const* subtree_beg;    // [slots] into `subtree_pids`
  int32_t const* subtree_cnt;    // [slots] paths in this slot's whole subtree
  int32_t const* subtree_pids;   // [total subtree entries] compact group positions
  // Flat-order name fields avoid an extra flat_children indirection.
  int32_t const* flat_name_off;   // [total children] into `names`
  int32_t const* flat_name_len;   // [total children]
  uint64_t const* flat_name_sig;  // [total children] first+last 4 bytes packed
  // Only multi-child nodes have max_name_len[node] + 2 entries, including the final sentinel.
  // len_prefix_beg counts entries; row[L] counts children shorter than L.
  int32_t const* len_prefix;
  std::size_t const* len_prefix_beg;  // [nodes]; empty/singleton nodes bypass their row
  int32_t const* max_name_len;        // [nodes]; unused when the row is bypassed
};

// R=1 uses an aligned 8-byte head and separate 4-byte kind word in a 16-byte stride.
// R=2 packs all fields in 8 bytes; both layouts require 8-byte alignment.
/**
 * One path's selection in one row: [val_begin, val_end) are byte offsets into that input row.
 * begin=-1 is unset; dead marks a failed first occurrence that later duplicates cannot replace.
 * Row lanes publish the same selection atomically; initialize/reset only outside those accesses.
 */
struct alignas(8) merged_path_out {
  int32_t val_begin;
  int32_t val_end;
  int8_t val_kind;  // 0: subtree (normalized serialization)
                    // 1: VALUE_STRING (unescaped write)
                    // 2: number/true/false (normalized serialization)
};

constexpr int8_t k_merged_kind_subtree  = 0;
constexpr int8_t k_merged_kind_string   = 1;
constexpr int8_t k_merged_kind_terminal = 2;
// Raw string or integer kinds point into input only if writer output is byte-identical.
// Floats and -0 normalize and must use the writer instead.
constexpr int8_t k_merged_kind_string_raw   = 4;
constexpr int8_t k_merged_kind_terminal_raw = 5;
// Certified raw subtrees bypass reparsing and point directly into input bytes.
constexpr int8_t k_merged_kind_subtree_raw = 6;
// Dead unset paths never resurrect on later duplicate keys.
constexpr int8_t k_merged_kind_dead = 3;

/**
 * R=2 record: begin occupies bits 0..31, exclusive end bits 32..60, and kind bits 61..63.
 * begin=-1 is unset; kind=dead prevents a failed first occurrence from being resurrected.
 * The launcher admits this layout only when row bytes fit 29 end bits; updates use one atomic word.
 */
struct merged_path_packed {
  uint64_t word;
};

inline constexpr int64_t k_packed_end_max = static_cast<int64_t>((uint64_t{1} << 29) - 1);

// Row barriers bracket initialization/reset; these ordinary stores must not race atomic reads.
__device__ __forceinline__ void merged_rec_init(merged_path_out& r)
{
  r = merged_path_out{-1, -1, 0};
}
__device__ __forceinline__ void merged_rec_set(merged_path_out& r,
                                               int32_t begin,
                                               int32_t end,
                                               int8_t kind)
{
  // Redundant row lanes publish identical values to each path's distinct record.
  // Keep begin/end in one atomic 8-byte head and kind in a separate atomic 4-byte word.
  // Publish kind first, fence, then publish head; readers access both words atomically.
  // Folding kind into the head would overwrite val_end's high byte.
  static_assert(offsetof(merged_path_out, val_begin) == 0 &&
                  offsetof(merged_path_out, val_end) == 4 &&
                  offsetof(merged_path_out, val_kind) == 8,
                "12-byte record active layout changed");
  uint64_t const image = static_cast<uint64_t>(static_cast<uint32_t>(begin)) |
                         (static_cast<uint64_t>(static_cast<uint32_t>(end)) << 32);
  unsigned int* const kind_word = reinterpret_cast<unsigned int*>(&r.val_kind);
  (void)atomicExch(kind_word, static_cast<unsigned int>(static_cast<unsigned char>(kind)));
  __threadfence_block();
  (void)atomicExch(reinterpret_cast<unsigned long long*>(&r.val_begin), image);
}

// Record reads must be atomic while other row lanes can publish the same fields.
__device__ __forceinline__ uint64_t merged_rec_head(merged_path_out const& r)
{
  return cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(
           *reinterpret_cast<unsigned long long*>(const_cast<int32_t*>(&r.val_begin)))
    .load(cuda::std::memory_order_relaxed);
}
__device__ __forceinline__ unsigned int merged_rec_kind_word(merged_path_out const& r)
{
  return cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(
           *reinterpret_cast<unsigned int*>(const_cast<int8_t*>(&r.val_kind)))
    .load(cuda::std::memory_order_relaxed);
}

__device__ __forceinline__ void merged_rec_mark_dead(merged_path_out& r)
{
  // The set predicate reads an atomic head while other row lanes can publish it.
  if (static_cast<int32_t>(static_cast<uint32_t>(merged_rec_head(r))) < 0) {
    // Mark kind before republishing the atomic head, matching normal publication order.
    unsigned int* const kind_word = reinterpret_cast<unsigned int*>(&r.val_kind);
    (void)atomicExch(kind_word,
                     static_cast<unsigned int>(static_cast<unsigned char>(k_merged_kind_dead)));
    // An atomic head read keeps begin/end from different exchanges from being mixed.
    uint64_t const head  = merged_rec_head(r);
    uint64_t const image = (static_cast<uint64_t>(static_cast<uint32_t>(head)) |
                            (static_cast<uint64_t>(static_cast<uint32_t>(head >> 32)) << 32));
    __threadfence_block();
    (void)atomicExch(reinterpret_cast<unsigned long long*>(&r.val_begin), image);
  }
}

__device__ __forceinline__ bool merged_rec_is_set(merged_path_out const& r)
{
  return static_cast<int32_t>(static_cast<uint32_t>(merged_rec_head(r))) >= 0;
}
__device__ __forceinline__ int8_t merged_rec_kind(merged_path_out const& r)
{
  return static_cast<int8_t>(merged_rec_kind_word(r) & 0xffu);
}
// R=1 uses separate atomic loads for head and kind, not one atomic record snapshot.
// Redundant row lanes publish identical values during the walk; the post-walk barrier
// makes both fields stable before materialization.
struct merged_rec_view {
  int32_t begin;
  int32_t end;
  int8_t kind;
  bool is_set;
};
__device__ __forceinline__ merged_rec_view merged_rec_load(merged_path_out const& r)
{
  merged_rec_view v{};
  uint64_t const head = merged_rec_head(r);
  v.begin             = static_cast<int32_t>(static_cast<uint32_t>(head));
  v.end               = static_cast<int32_t>(head >> 32);
  v.is_set            = v.begin >= 0;
  v.kind              = static_cast<int8_t>(merged_rec_kind_word(r) & 0xffu);
  return v;
}

__device__ __forceinline__ void merged_rec_init(merged_path_packed& r)
{
  r.word = 0x0000'0000'ffff'ffffull;
}
__device__ __forceinline__ void merged_rec_set(merged_path_packed& r,
                                               int32_t begin,
                                               int32_t end,
                                               int8_t kind)
{
  // Redundant row lanes publish identical values to each path's distinct packed record.
  // Predicate and exchange are separate; repeated publications rewrite the same word.
  uint64_t const image = static_cast<uint64_t>(static_cast<uint32_t>(begin)) |
                         (static_cast<uint64_t>(static_cast<uint32_t>(end)) << 32) |
                         (static_cast<uint64_t>(static_cast<uint8_t>(kind)) << 61);
  (void)atomicExch(reinterpret_cast<unsigned long long*>(&r.word), image);
}
// Read the packed record atomically while other row lanes can publish it.
__device__ __forceinline__ uint64_t merged_rec_word(merged_path_packed const& r)
{
  return cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(
           *reinterpret_cast<unsigned long long*>(const_cast<uint64_t*>(&r.word)))
    .load(cuda::std::memory_order_relaxed);
}
__device__ __forceinline__ bool merged_rec_is_set(merged_path_packed const& r)
{
  return static_cast<int32_t>(static_cast<uint32_t>(merged_rec_word(r))) >= 0;
}
__device__ __forceinline__ void merged_rec_mark_dead(merged_path_packed& r)
{
  // A dead record must never resurrect, and an already-set record must never
  // be killed: publish kind=dead only if the record is still unset, as one
  // atomic 8-byte transition. CAS-loop: the predicate is re-checked on the
  // freshly loaded value every attempt, so only a truly-unset record flips.
  unsigned long long* const addr = reinterpret_cast<unsigned long long*>(&r.word);
  // Seed the CAS predicate atomically because other row lanes can publish this word.
  unsigned long long old = merged_rec_word(r);
  unsigned long long assumed;
  do {
    if (static_cast<int32_t>(static_cast<uint32_t>(old)) >= 0) { return; }
    assumed = old;
    unsigned long long const dead =
      (old & ~(7ull << 61)) |
      (static_cast<uint64_t>(static_cast<uint8_t>(k_merged_kind_dead)) << 61);
    old = atomicCAS(addr, assumed, dead);
  } while (old != assumed);
}
__device__ __forceinline__ int8_t merged_rec_kind(merged_path_packed const& r)
{
  return static_cast<int8_t>((merged_rec_word(r) >> 61) & 7);
}
// The packed variant decodes all fields from one atomic 8-byte word.
__device__ __forceinline__ merged_rec_view merged_rec_load(merged_path_packed const& r)
{
  uint64_t const word = merged_rec_word(r);
  merged_rec_view v{};
  v.begin  = static_cast<int32_t>(static_cast<uint32_t>(word));
  v.end    = static_cast<int32_t>((word >> 32) & ((uint64_t{1} << 29) - 1));
  v.kind   = static_cast<int8_t>((word >> 61) & 7);
  v.is_set = v.begin >= 0;
  return v;
}

template <int R>
using merged_row_rec_t = cuda::std::conditional_t<R == 1, merged_path_out, merged_path_packed>;

// The R=2 admission estimate uses this desired resident-block bound.
inline constexpr int k_merged_min_blocks_per_sm = 8;

// Shallow admission bounds the live trie descent stack.
constexpr int k_merged_max_trie_depth = static_cast<int>(k_merged_max_depth);

// Deep admission bounds its stack independently of MAX_JSON_PATH_DEPTH.
inline constexpr std::size_t k_merged_deep_max_depth = 5;

// Deep trie requires at least eight pure-NAMED paths.
inline constexpr std::size_t k_deep_merged_min_paths = 8;

/**
 * Compares a validated quoted field name with decoded candidate bytes, without moving parser state.
 * Requires the whole raw token, including both quotes; escaped names and surrogate pairs compare
 * by decoded bytes, and a successful match consumes the complete candidate and raw spelling.
 */
// try_skip_unicode's contract; both handle surrogate pairs and exact closing quotes.
__device__ inline bool merged_name_matches(char_range raw, cudf::string_view candidate)
{
  if (raw.is_empty()) { return false; }
  char_range_reader r(raw);
  char const quote_char = r.current_char();
  r.next();  // skip opening quote
  for (cudf::size_type ci = 0; ci < candidate.size_bytes(); /* incremented in body */) {
    if (r.is_empty()) { return false; }
    char const c = r.current_char();
    if (c == quote_char) { return false; }  // name ended before candidate consumed
    if (c == '\\') {
      r.next();
      if (r.is_empty()) { return false; }
      char const e = r.current_char();
      switch (e) {
        case '"':
        case '\'':
        case '\\':
        case '/':
          r.next();
          if (e != candidate.data()[ci]) { return false; }
          ++ci;
          continue;
        case 'b':
        case 'f':
        case 'n':
        case 'r':
        case 't': {
          char const decoded = json_parser::simple_escape(e);
          r.next();
          if (decoded != candidate.data()[ci]) { return false; }
          ++ci;
          continue;
        }

        case 'u': {
          r.next();
          if (r.size() < 4) { return false; }
          uint32_t cp = 0;
          for (int h = 0; h < 4; ++h) {
            char const hc = r.current_char();
            if (!json_parser::is_hex_digit(hc)) { return false; }
            cp = (cp << 4) | json_parser::hex_value(hc);
            r.next();
          }
          // Pair only a valid low surrogate; otherwise encode the high value alone.
          if (cp >= 0xD800 && cp <= 0xDBFF && r.size() >= 6 && r[0] == '\\' && r[1] == 'u') {
            uint32_t low = 0;
            bool ok      = true;
            for (int h = 2; h < 6; ++h) {
              char const hc = r[h];
              if (!json_parser::is_hex_digit(hc)) {
                ok = false;
                break;
              }
              low = (low << 4) | json_parser::hex_value(hc);
            }
            if (ok && low >= 0xDC00 && low <= 0xDFFF) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
              for (int h = 0; h < 6; ++h) {
                r.next();
              }
            }
          }
          char buff[4];
          cudf::size_type const w =
            json_parser::from_char_utf8(json_parser::codepoint_to_utf8(cp), buff);
          // Compare decoded bytes to the candidate without advancing the raw reader again.
          if (candidate.size_bytes() - ci < w) { return false; }
          for (cudf::size_type b = 0; b < w; ++b) {
            if (candidate.data()[ci + b] != buff[b]) { return false; }
          }
          ci += w;
          continue;
        }
        default: return false;  // unreachable: main parse validated the escape
      }
    }
    if (c != candidate.data()[ci]) { return false; }
    r.next();
    ++ci;
  }
  if (r.is_empty() || r.current_char() != quote_char) { return false; }
  r.next();
  return r.is_empty();  // no trailing raw content
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
    if (merged_name_matches(raw, cand)) { return slot; }
  }
  return -1;
}

// Packs the first and last nonoverlapping bytes, up to four each; matching still needs full bytes.
// E.g. abcde packs abcd in the low half and e in the high half; there is no repeated overlap.
// Host/device use the same formula; equal signatures alone never certify equal names.
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
 * Requires double-quoted strings/keys, canonical simple escapes, compact separators and unchanged
 * integers/literals. Single quotes, exterior whitespace, controls, Unicode/slash escapes, floats
 * and -0 return false for writer normalization. A true result permits a borrowed input view.
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
        // run >= 1: data[i] is safe (M3); the outer scanner consumes the stop byte.
        char_range_reader r{char_range(data + i, len - i)};
        i += json_parser::swar_safe_run<true>(r, nullptr, '"', false);
        if (i >= len) { return false; }  // unterminated string: parse would have failed
        continue;                        // scalar decisions handle the stop byte
      }
      continue;
    }
    if (expect_obj_key) {
      // Only a quoted key or an immediate empty-object close is canonical.
      if (b != '"' && !(b == '}' && i > 0 && data[i - 1] == '{')) { return false; }
    }
    if (b == '"') {
      expect_obj_key = false;  // this '"' opens the key itself when the gate passed
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
    // The parser's match flag is valid only for a FIELD_NAME token.
    if (json_token::FIELD_NAME == p.get_current_token() && matched) { return slot; }
    return -1;
  }

  // Multiple candidates avoid serial first-candidate matching.
  p.next_token();
  if (json_token::FIELD_NAME != p.get_current_token() || cnt == 0) { return -1; }

  // Escaped names need decoded comparison; raw-length buckets are sound only without backslashes.
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

// Kill an unmatched subtree permanently so later duplicate keys cannot resurrect paths.
template <typename RecT>
__device__ __forceinline__ void kill_subtree_unmatched(merged_trie_view const& trie,
                                                       int32_t slot,
                                                       RecT* row_out)
{
  int32_t const sb = trie.subtree_beg[slot];
  int32_t const sc = trie.subtree_cnt[slot];
  for (int32_t t = 0; t < sc; ++t) {
    auto const pid = trie.subtree_pids[sb + t];
    if (!merged_rec_is_set(row_out[pid])) { merged_rec_mark_dead(row_out[pid]); }
  }
}

/**
 * Evaluates pure-NAMED paths with one shared prefix trie per row: R=1 uses a warp, R=2 a half-warp.
 * Each row lane repeats the same cooperative parse and atomically records selections; a row barrier
 * separates the walk from lane-strided output writing. Unselected records produce null views.
 *
 * The first matching key owns its entire path subtree, even for null or a missing deeper member.
 * A path ending at an object can coexist with paths below it; its range completes on object close.
 * Parse failure nulls all paths. Depth overflow sets the host flag if any path is not already dead.
 * The caller supplies group-ordered path_data and a bound trie, retaining both through completion;
 * Null row_scratch selects shared records; otherwise storage holds rows * num_paths records.
 */
template <int block_size, int MaxTrieDepth, int R>
__launch_bounds__(block_size, k_merged_min_blocks_per_sm) CUDF_KERNEL
  void get_json_object_kernel_merged_named(
    cudf::column_device_view input,
    cudf::device_span<json_path_processing_data> path_data,
    merged_trie_view trie,             // copied view; retain pointees through completion
    merged_row_rec_t<R>* row_scratch,  // [rows * num_paths]
    // Null selects smem, non-null global fallback.
    int32_t num_paths,
    int8_t* max_path_depth_exceeded)
{
  static_assert(R == 1 || R == 2, "merged walk supports 1 or 2 rows per warp");
  using rec_t               = merged_row_rec_t<R>;
  auto constexpr warp_lanes = static_cast<int>(cudf::detail::warp_size);
  // R groups per warp redundantly walk separate rows, then split path materialization.
  auto constexpr row_lanes = warp_lanes / R;
  auto const tidx          = cudf::detail::grid_1d::global_thread_id();
  auto const lane          = threadIdx.x % warp_lanes;
  // R=2 maps consecutive half-warps to independent rows.
  auto const row_idx  = static_cast<cudf::size_type>(tidx / row_lanes);
  auto const half     = lane / row_lanes;  // row slot within this warp: 0 at R=1, 0/1 at R=2
  auto const row_lane = lane % row_lanes;
  if (row_idx >= input.size()) { return; }
  // row_mask excludes the sibling half-warp after an early row return.
  unsigned const row_mask = R == 1 ? 0xffffffffu : (0x0000ffffu << (half * row_lanes));
  // Byte-typed extern shared storage avoids conflicting templated rec_t symbols across R.
  // R=1 strides 16 bytes and R=2 strides 8; the shared base must be 8-byte aligned.
  extern __shared__ char smem_row_out_bytes[];
  auto* smem_row_out       = reinterpret_cast<rec_t*>(smem_row_out_bytes);
  auto const warp_in_block = threadIdx.x / warp_lanes;
  rec_t* row_out = row_scratch == nullptr ? smem_row_out + (warp_in_block * R + half) * num_paths
                                          : row_scratch + static_cast<int64_t>(row_idx) * num_paths;
  for (int32_t i = row_lane; i < num_paths; i += row_lanes) {
    merged_rec_init(row_out[i]);
  }
  __syncwarp(row_mask);

  auto const str =
    input.is_null(row_idx) ? cudf::string_view{} : input.element<cudf::string_view>(row_idx);
  if (str.size_bytes() == 0) {
    // Null and empty rows must initialize every output view without parsing.
    for (int32_t i = row_lane; i < num_paths; i += row_lanes) {
      auto const& path = path_data[i];
      if (path.out_stringviews) { path.out_stringviews[row_idx] = {nullptr, 0}; }
    }
    return;
  }

  // Redundant same-row walks make cooperative string skip warp-uniform.
  bool parse_ok       = true;
  bool depth_exceeded = false;
  json_parser_impl<true> p{char_range{str}};
  p.enable_warp_cooperative_skip(static_cast<int>(lane), row_lanes, row_mask);
  p.next_token();
  json_token const root_tok = p.get_current_token();
  if (json_token::ERROR == root_tok) {
    parse_ok = false;
  } else if (root_tok != json_token::START_OBJECT) {
    // Non-object roots validate but leave every NAMED path null.
    if (!p.try_skip_children()) { parse_ok = false; }
  } else {
    struct walk_frame {
      int32_t node;        // trie node whose members we are scanning
      int32_t pend_slot;   // slot awaiting its subtree range completion; -1 none
      int32_t pend_begin;  // value start offset of the pending structure
    };
    // Admission bounds live descent frames by MaxTrieDepth.
    walk_frame stack[MaxTrieDepth + 1];
    int32_t sp   = 0;
    int32_t node = 0;

    auto const record_slot = [&](int32_t slot, int32_t begin, int32_t end, int8_t kind) {
      int32_t const vb = trie.paths_beg[slot];
      int32_t const vc = trie.paths_cnt[slot];
      for (int32_t t = 0; t < vc; ++t) {
        auto const pid = trie.terminal_ids[vb + t];
        // First duplicate wins; unset dead paths never resurrect.
        merged_rec_view const cur = merged_rec_load(row_out[pid]);
        if (!cur.is_set && cur.kind != k_merged_kind_dead) {
          merged_rec_set(row_out[pid], begin, end, kind);
        }
      }
    };

    auto const record_subtree = [&](int32_t slot, int32_t begin, int32_t end) {
      // A certificate borrows input through gather; false only selects normalized serialization.
      bool const raw = certify_raw_subtree(str.data() + begin, end - begin);
      record_slot(slot, begin, end, raw ? k_merged_kind_subtree_raw : k_merged_kind_subtree);
    };

    while (parse_ok) {
      int32_t const slot  = merged_scan_member(p, trie, node);
      json_token const mt = p.get_current_token();
      if (json_token::ERROR == mt) {
        parse_ok = false;
        break;
      }
      if (json_token::END_OBJECT == mt) {
        if (sp == 0) { break; }
        auto const& fr = stack[sp - 1];
        if (fr.pend_slot >= 0) {
          int32_t const end = static_cast<int32_t>(p.current_range().data() - str.data()) + 1;
          record_subtree(fr.pend_slot, fr.pend_begin, end);
        }
        // Exhausted child namespace kills only its own unmatched subtree; sibling paths survive.
        // subtree_pids covers all descendants, so direct-child kills need no DFS.
        {
          int32_t const cb = trie.children_beg[node];
          int32_t const cc = trie.children_cnt[node];
          for (int32_t c = 0; c < cc; ++c) {
            kill_subtree_unmatched(trie, trie.flat_children[cb + c], row_out);
          }
        }
        --sp;
        node = stack[sp].node;
        continue;
      }
      if (slot < 0) {
        p.next_token();
        if (json_token::ERROR == p.get_current_token()) {
          parse_ok = false;
          break;
        }
        if (!p.try_skip_children()) {
          parse_ok = false;
          break;
        }
        continue;
      }

      p.next_token();
      json_token const vt = p.get_current_token();
      if (json_token::ERROR == vt) {
        parse_ok = false;
        break;
      }

      if (json_token::VALUE_NULL == vt) {
        // Matched null kills its path permanently, including future duplicate keys.
        kill_subtree_unmatched(trie, slot, row_out);
        continue;
      }
      if (json_token::START_OBJECT == vt || json_token::START_ARRAY == vt) {
        int32_t const child = trie.child_node[slot];
        bool const can_descend =
          child >= 0 && trie.children_cnt[child] > 0 && json_token::START_OBJECT == vt;
        int32_t const vc = trie.paths_cnt[slot];
        if (vc > 0 && !can_descend) {
          // A non-descending subtree can still terminate this slot; deeper chains stay dead.
          int32_t const begin = static_cast<int32_t>(p.current_range().data() - str.data());
          if (!p.try_skip_children()) {
            parse_ok = false;
            break;
          }
          int32_t const end = static_cast<int32_t>(p.current_range().data() - str.data()) + 1;
          record_subtree(slot, begin, end);
          kill_subtree_unmatched(trie, slot, row_out);
        } else if (can_descend) {
          if (sp > MaxTrieDepth - 1) {
            parse_ok = false;
            break;
          }  // defensive: admission guarantees this never fires
          stack[sp] = walk_frame{
            node, vc > 0 ? slot : -1, static_cast<int32_t>(p.current_range().data() - str.data())};
          ++sp;
          node = child;
          continue;
        } else {
          // Validate childless subtrees but mark deeper unmatched chains dead.
          if (!p.try_skip_children()) {
            parse_ok = false;
            break;
          }
          kill_subtree_unmatched(trie, slot, row_out);
        }
        continue;
      }

      int32_t const vc = trie.paths_cnt[slot];
      if (vc > 0) {
        int32_t const begin = static_cast<int32_t>(p.current_range().data() - str.data());
        int32_t const end   = begin + static_cast<int32_t>(p.current_range().size());
        int8_t kind;
        if (json_token::VALUE_STRING == vt) {
          kind =
            p.current_token_is_escaped_string() ? k_merged_kind_string : k_merged_kind_string_raw;
        } else if (json_token::VALUE_NUMBER_INT == vt &&
                   !(p.number_len() == 2 && str.data()[begin] == '-' &&
                     str.data()[begin + 1] == '0')) {
          kind = k_merged_kind_terminal_raw;
        } else if (json_token::VALUE_TRUE == vt || json_token::VALUE_FALSE == vt) {
          kind = k_merged_kind_terminal_raw;
        } else {
          kind = k_merged_kind_terminal;
        }
        record_slot(slot, begin, end, kind);
      }
      // A scalar kills deeper chains; a childless slot has no descendants to kill.
      if (trie.child_node[slot] >= 0) { kill_subtree_unmatched(trie, slot, row_out); }
    }
  }
  // Depth flags cover spine and root-validation paths, including nested skips.
  depth_exceeded = p.max_nesting_depth_exceeded();

  // Broadcast decisions within each row group before its conditional error path.
  parse_ok       = __shfl_sync(row_mask, parse_ok, 0, row_lanes);
  depth_exceeded = __shfl_sync(row_mask, depth_exceeded, 0, row_lanes);
  if (depth_exceeded) {
    // Throw iff any path is non-dead — mirrors the general kernel's per-path depth flag.
    bool any_non_dead = false;
    if (row_lane == 0) {
      for (int32_t i = 0; i < num_paths; ++i) {
        if (merged_rec_kind(row_out[i]) != k_merged_kind_dead) {
          any_non_dead = true;
          break;
        }
      }
    }
    any_non_dead = __shfl_sync(row_mask, any_non_dead, 0, row_lanes);
    if (any_non_dead) {
      if (row_lane == 0) { set_device_flag(max_path_depth_exceeded); }
      return;
    }
    // All-dead paths may still emit null views without throwing.
  }
  if (!parse_ok) {
    // Ordinary resets must follow every row lane's atomic record accesses.
    __syncwarp(row_mask);
    // Validation failure nulls every path regardless of prior records.
    for (int32_t i = row_lane; i < num_paths; i += row_lanes) {
      merged_rec_init(row_out[i]);
    }
  }
  __syncwarp(row_mask);

  auto const row_beg = static_cast<int64_t>(path_data[0].offsets[row_idx]);
  auto const row_end = static_cast<int64_t>(path_data[0].offsets[row_idx + 1]);
  auto const row_max = row_end - row_beg;  // int64: base compared the raw offsets difference
  // Slice offsets are absolute, but scratch owns only the selected bytes.
  auto const row_scratch_beg = row_beg - path_data[0].base_offset;
  for (int32_t i = row_lane; i < num_paths; i += row_lanes) {
    auto const& path         = path_data[i];
    auto const& po           = row_out[i];
    bool is_valid            = false;
    cudf::size_type out_size = 0;
    char const* out_ptr      = path.out_buf + row_scratch_beg;
    // Row records are stable after the barrier; R=1 still reads head and kind separately.
    merged_rec_view const rec = merged_rec_load(po);
    if (rec.is_set) {
      char* const dst = path.out_buf + row_scratch_beg;
      if (rec.kind == k_merged_kind_string_raw || rec.kind == k_merged_kind_terminal_raw ||
          rec.kind == k_merged_kind_subtree_raw) {
        // Byte-identical raw values point into input; the later gather reads those pointers.
        // Even empty strings retain non-null pointers for valid output.
        is_valid = true;
        if (rec.kind == k_merged_kind_string_raw) {
          out_ptr  = str.data() + rec.begin + 1;
          out_size = rec.end - rec.begin - 2;
        } else {
          out_ptr  = str.data() + rec.begin;
          out_size = rec.end - rec.begin;
        }
      } else if (rec.kind == k_merged_kind_string) {
        // Decode validated escapes through the parser's writer seam.
        char_range_reader r(char_range(str.data() + rec.begin, rec.end - rec.begin));
        out_size = json_parser::write_string(r, dst, escape_style::UNESCAPED);
        is_valid = true;
      } else {
        char_range const value_range(str.data() + rec.begin, rec.end - rec.begin);
        json_parser_impl<true> vp{value_range};
        vp.next_token();
        if (json_token::ERROR != vp.get_current_token() && !vp.max_nesting_depth_exceeded()) {
          json_generator g;
          // Subtrees and normalized numbers reuse the general path's writer.
          is_valid = g.copy_current_structure(vp, dst);
          out_size = g.get_output_len();
        }
      }
    }
    if (out_size > row_max) { set_device_flag(path.has_out_of_bound); }
    if (path.out_stringviews) {
      path.out_stringviews[row_idx] = {is_valid ? out_ptr : nullptr, is_valid ? out_size : 0};
    }
  }
}

// Owns the host trie blob and optional device row records; view borrows the grouped upload pack.
struct merged_group_plan {
  std::vector<uint8_t> h_blob;
  std::size_t blob_off{0};
  // SoA section offsets are relative to the blob's packed device base.
  std::array<std::size_t, 17> sec{};
  merged_trie_view view{};
  rmm::device_uvector<merged_path_out> row_scratch;  // rows * num_paths
  int32_t num_paths{0};
  int max_trie_depth{0};
};

/**
 * Builds an owned, length-sorted trie pack for nonempty pure-NAMED paths selected by indices.
 * Group position k remains path ID k, including repeated paths; shared prefixes retain terminals.
 * Invalid kinds or paths beyond max_trie_depth throw. Upload h_blob and bind view before
 * launching; the view starts unbound, and row scratch is allocated later.
 */
merged_group_plan build_merged_named_plan(
  std::vector<cudf::host_span<std::tuple<path_instruction_type, std::string, int32_t> const>> const&
    json_paths,
  std::vector<std::size_t> const& indices,
  int max_trie_depth,
  cuda::stream_ref stream)
{
  auto const num_paths = static_cast<int32_t>(indices.size());

  // Defend the kernel's MaxTrieDepth stack even if a caller bypasses admission.
  for (auto const idx : indices) {
    CUDF_EXPECTS(static_cast<int>(json_paths[idx].size()) <= max_trie_depth,
                 "merged walk requires paths of depth <= the group's max trie depth");
  }

  // Root node 0 needs one empty child list.
  int32_t constexpr k_root = 0;
  std::vector<std::vector<int32_t>> node_children(1);
  struct slot_rec {
    std::string name;
    int32_t child_node{-1};
    std::vector<int32_t> terminals;
    // Subtree paths include deeper descendants for whole-chain termination.
    std::vector<int32_t> subtree_paths;
  };
  std::vector<slot_rec> slots;
  slots.reserve(num_paths * 2);

  for (int32_t k = 0; k < num_paths; ++k) {
    auto const& path = json_paths[indices[k]];
    int32_t cur      = k_root;
    for (std::size_t j = 0; j < path.size(); ++j) {
      auto const& [type, name, index] = path[j];
      CUDF_EXPECTS(type == path_instruction_type::NAMED,
                   "merged walk requires a pure-NAMED path group");
      int32_t found = -1;
      for (auto const s : node_children[cur]) {
        if (slots[s].name == name) {
          found = s;
          break;
        }
      }
      if (found < 0) {
        found = static_cast<int32_t>(slots.size());
        slots.push_back(slot_rec{name, -1, {}, {}});
        node_children[cur].push_back(found);
      }
      slots[found].subtree_paths.push_back(k);
      if (j + 1 == path.size()) {
        slots[found].terminals.push_back(k);
      } else {
        if (slots[found].child_node < 0) {
          slots[found].child_node = static_cast<int32_t>(node_children.size());
          node_children.emplace_back();
        }
        cur = slots[found].child_node;
      }
    }
  }

  auto const n_nodes = static_cast<int32_t>(node_children.size());
  auto const n_slots = static_cast<int32_t>(slots.size());

  // Per-slot arrays use slot IDs; flat_children orders those IDs by node.
  std::vector<int32_t> child_node(n_slots, -1);
  std::vector<int32_t> paths_beg(n_slots, 0), paths_cnt(n_slots, 0);
  std::vector<int32_t> subtree_beg(n_slots, 0), subtree_cnt(n_slots, 0);
  std::vector<int32_t> children_beg(n_nodes, 0), children_cnt(n_nodes, 0);
  std::vector<std::size_t> len_prefix_beg(n_nodes, 0);
  std::vector<int32_t> max_name_len(n_nodes, 0);
  std::size_t prefix_entries = 0;
  // Stable length sorting makes each node's equal-length candidates contiguous.
  for (int32_t n = 0; n < n_nodes; ++n) {
    auto& children = node_children[n];
    std::stable_sort(children.begin(), children.end(), [&](int32_t a, int32_t b) {
      return slots[a].name.size() < slots[b].name.size();
    });
    len_prefix_beg[n] = prefix_entries;
    if (children.size() > 1) {
      max_name_len[n] = static_cast<int32_t>(slots[children.back()].name.size());
      prefix_entries += static_cast<std::size_t>(max_name_len[n]) + 2;
    }
  }
  std::vector<int32_t> flat_children;
  // Flat-order fields avoid a slot-ID indirection in device scans.
  std::vector<int32_t> flat_name_off, flat_name_len;
  std::vector<uint64_t> flat_name_sig;
  std::vector<int32_t> terminal_ids;
  std::vector<int32_t> subtree_pids;
  std::string names;

  int32_t child_cursor = 0, term_cursor = 0, sub_cursor = 0;
  for (int32_t n = 0; n < n_nodes; ++n) {
    children_beg[n] = child_cursor;
    children_cnt[n] = static_cast<int32_t>(node_children[n].size());

    for (auto const s : node_children[n]) {
      int32_t const l = static_cast<int32_t>(slots[s].name.size());
      flat_children.push_back(s);  // SLOT ID, length-sorted (pre-flatten sort)
      flat_name_off.push_back(static_cast<int32_t>(names.size()));
      names += slots[s].name;
      flat_name_len.push_back(l);
      // Signatures only filter; the shared helper keeps host/device packing identical.
      flat_name_sig.push_back(name_sig(slots[s].name.data(), l));
      child_node[s] = slots[s].child_node;
      paths_beg[s]  = term_cursor;
      paths_cnt[s]  = static_cast<int32_t>(slots[s].terminals.size());
      terminal_ids.insert(terminal_ids.end(), slots[s].terminals.begin(), slots[s].terminals.end());
      term_cursor += paths_cnt[s];
      subtree_beg[s] = sub_cursor;
      subtree_cnt[s] = static_cast<int32_t>(slots[s].subtree_paths.size());
      subtree_pids.insert(
        subtree_pids.end(), slots[s].subtree_paths.begin(), slots[s].subtree_paths.end());
      sub_cursor += subtree_cnt[s];
      ++child_cursor;
    }
  }
  CUDF_EXPECTS(child_cursor == static_cast<int32_t>(flat_children.size()), "trie flatten mismatch");

  // Length rows are filled directly in the owned blob, avoiding a second pageable copy.
  struct section {
    void const* data;
    std::size_t bytes;
    std::size_t alignment;
  };
  auto const sections = std::array{
    section{children_beg.data(), children_beg.size() * sizeof(int32_t), 1},
    section{children_cnt.data(), children_cnt.size() * sizeof(int32_t), 1},
    section{child_node.data(), child_node.size() * sizeof(int32_t), 1},
    section{paths_beg.data(), paths_beg.size() * sizeof(int32_t), 1},
    section{paths_cnt.data(), paths_cnt.size() * sizeof(int32_t), 1},
    section{flat_children.data(), flat_children.size() * sizeof(int32_t), 1},
    section{terminal_ids.data(), terminal_ids.size() * sizeof(int32_t), 1},
    section{names.data(), names.size(), 1},
    section{flat_name_sig.data(), flat_name_sig.size() * sizeof(uint64_t), 8},
    section{subtree_beg.data(), subtree_beg.size() * sizeof(int32_t), 1},
    section{subtree_cnt.data(), subtree_cnt.size() * sizeof(int32_t), 1},
    section{subtree_pids.data(), subtree_pids.size() * sizeof(int32_t), 1},
    section{flat_name_off.data(), flat_name_off.size() * sizeof(int32_t), 1},
    section{flat_name_len.data(), flat_name_len.size() * sizeof(int32_t), 1},
    section{nullptr, prefix_entries * sizeof(int32_t), alignof(int32_t)},
    section{
      len_prefix_beg.data(), len_prefix_beg.size() * sizeof(std::size_t), alignof(std::size_t)},
    section{max_name_len.data(), max_name_len.size() * sizeof(int32_t), 1}};

  std::array<std::size_t, 17> offsets{};
  std::size_t total = 0;
  for (std::size_t k = 0; k < sections.size(); ++k) {
    total      = (total + sections[k].alignment - 1) & ~(sections[k].alignment - 1);
    offsets[k] = total;
    total += sections[k].bytes;
  }
  std::vector<uint8_t> h_blob(total, 0);
  for (std::size_t k = 0; k < sections.size(); ++k) {
    if (sections[k].bytes != 0 && sections[k].data != nullptr) {
      std::memcpy(h_blob.data() + offsets[k], sections[k].data, sections[k].bytes);
    }
  }
  auto* const len_prefix = h_blob.data() + offsets[14];
  for (int32_t n = 0; n < n_nodes; ++n) {
    int32_t const cnt = children_cnt[n];
    if (cnt <= 1) { continue; }
    auto* const row = len_prefix + len_prefix_beg[n] * sizeof(int32_t);
    // Leading zero buckets come from blob initialization; write each later entry once.
    auto next_len = static_cast<std::size_t>(flat_name_len[children_beg[n]]) + 1;
    for (int32_t c = 1; c < cnt; ++c) {
      auto const end = static_cast<std::size_t>(flat_name_len[children_beg[n] + c]) + 1;
      for (; next_len < end; ++next_len) {
        std::memcpy(row + next_len * sizeof(int32_t), &c, sizeof(c));
      }
    }
    std::memcpy(row + next_len * sizeof(int32_t), &cnt, sizeof(cnt));
  }

  return merged_group_plan{
    std::move(h_blob),
    0,  // blob_off: set by the merged packer (launch_grouped)
    offsets,
    merged_trie_view{},
    rmm::device_uvector<merged_path_out>(0, stream),  // global fallback: allocated lazily
    num_paths,
    max_trie_depth};
}

// Bind SoA sections relative to the device-pack base; signatures require 8-byte alignment.
void bind_merged_plan_view(merged_group_plan& plan, uint8_t const* pack_base)
{
  auto const& sec = plan.sec;
  auto const base = pack_base + plan.blob_off;
  auto const p32  = [&](std::size_t off) { return reinterpret_cast<int32_t const*>(base + off); };
  plan.view       = merged_trie_view{p32(0),
                               p32(sec[1]),
                               p32(sec[2]),
                               p32(sec[3]),
                               p32(sec[4]),
                               p32(sec[5]),
                               p32(sec[6]),
                               reinterpret_cast<char const*>(base + sec[7]),
                               p32(sec[9]),
                               p32(sec[10]),
                               p32(sec[11]),
                               p32(sec[12]),
                               p32(sec[13]),
                               reinterpret_cast<uint64_t const*>(base + sec[8]),
                               p32(sec[14]),
                               reinterpret_cast<std::size_t const*>(base + sec[15]),
                               p32(sec[16])};
}

/**
 * Launches a bound shallow/deep trie; path_data must match the plan's compact path order.
 * R=1 uses shared records when they fit 48 KiB per block; otherwise plan owns their global scratch.
 * R=2 uses two 16-lane rows per warp only beyond the R=1 capacity estimate, with row bytes fitting
 * its packed end field and all block records fitting 48 KiB. Otherwise R=1 uses one warp per row.
 * Launch bounds estimate capacity, not residency.
 * max_row_size must bound every input row so the packed end cannot truncate. Retain the bound pack,
 * path buffers and plan through completion; the caller owns error readback.
 */
void launch_merged_named(cudf::column_device_view const& input,
                         cudf::device_span<json_path_processing_data> path_data,
                         merged_group_plan& plan,
                         int64_t max_row_size,
                         int8_t* max_path_depth_exceeded,
                         cuda::stream_ref stream)
{
  constexpr int block_size       = 128;
  auto constexpr warps_per_block = block_size / cudf::detail::warp_size;
  // The admission gate estimates a walk round from the launch bounds, not measured residency.
  auto constexpr k_resident_warps_per_sm = k_merged_min_blocks_per_sm * warps_per_block;

  // R=2 needs more than one R=1 walk round and a row within the packed end bound.
  auto const rows = static_cast<std::size_t>(input.size());
  // R=1 uses 16-byte records; R=2 uses two 8-byte records per warp.
  auto const smem_r1 =
    static_cast<std::size_t>(warps_per_block) * plan.num_paths * sizeof(merged_path_out);
  auto const smem_r2 =
    static_cast<std::size_t>(warps_per_block) * 2 * plan.num_paths * sizeof(merged_path_packed);
  auto constexpr k_smem_cap = 48 * 1024;  // default dynamic smem per block
  auto const walk_capacity =
    static_cast<std::size_t>(cudf::detail::num_multiprocessors()) * k_resident_warps_per_sm;
  bool const use_r2 =
    rows > walk_capacity && max_row_size <= k_packed_end_max && smem_r2 <= k_smem_cap;

  // R=2 halves row-warps; R=1 retains the original block mapping.
  auto const warps_needed = use_r2 ? cudf::util::div_rounding_up_safe(rows, std::size_t{2}) : rows;
  auto const num_blocks   = cudf::util::div_rounding_up_safe(warps_needed * cudf::detail::warp_size,
                                                           static_cast<std::size_t>(block_size));
  bool const row_scratch  = !use_r2 && smem_r1 > k_smem_cap;
  if (row_scratch) {
    plan.row_scratch =
      rmm::device_uvector<merged_path_out>(static_cast<int64_t>(input.size()) * plan.num_paths,
                                           stream,
                                           cudf::get_current_device_resource_ref());
  }
  // Only R=1 may use the 16-byte-strided global-scratch fallback.
  auto* scratch = row_scratch ? plan.row_scratch.data() : nullptr;
  // Select the admitted shallow/deep stack cap and the gated R specialization.
  auto const launch = [&]<int depth, int r>(std::size_t smem, auto* row_scratch) {
    get_json_object_kernel_merged_named<block_size, depth, r>
      <<<num_blocks, block_size, smem, stream.get()>>>(
        input, path_data, plan.view, row_scratch, plan.num_paths, max_path_depth_exceeded);
  };
  auto const launch_for_r = [&]<int r>(std::size_t smem, auto* row_scratch) {
    if (plan.max_trie_depth == k_merged_max_trie_depth) {
      launch.template operator()<k_merged_max_trie_depth, r>(smem, row_scratch);
    } else {
      launch.template operator()<static_cast<int>(k_merged_deep_max_depth), r>(smem, row_scratch);
    }
  };
  if (use_r2) {
    launch_for_r.template operator()<2>(smem_r2, static_cast<merged_path_packed*>(nullptr));
  } else {
    launch_for_r.template operator()<1>(row_scratch ? 0 : smem_r1, scratch);
  }
}

/**
 * Owns the grouped upload's host bytes, device pack, path views and optional global row records.
 * Retain this owner through the first error-readback synchronization, which also orders
 * the joined kernels; local staging could die while uploads or kernels still use it.
 */
struct launch_grouped_staging {
  std::optional<cudf::detail::host_vector<std::byte>> h_prewalk_pack;
  rmm::device_buffer d_prewalk_pack;
  std::vector<std::vector<json_path_processing_data>> group_staging;
  std::vector<merged_group_plan> merged_plans;
};

/**
 * Partitions a first-match batch into one admitted pure-NAMED trie and remaining independent paths.
 * A depth-three-to-five group needs 32 rows and eight deep paths, then absorbs shallow paths;
 * otherwise shallow admission needs 32 rows and eight paths, or 124 for all-depth-one groups.
 * Other shapes, unadmitted groups and last-non-null selection use the general kernel.
 * Host/device paths share batch order; zeroed error storage has path flags plus one depth flag.
 * Pass fresh caller-owned staging for upload lifetimes. Mixed groups fork after upload and join on
 * stream before return; the join orders later readback, but does not synchronize the host.
 */
void launch_grouped(
  cudf::column_device_view const& input,
  std::vector<cudf::host_span<std::tuple<path_instruction_type, std::string, int32_t> const>> const&
    json_paths,
  rmm::device_uvector<int8_t>& d_error_check,
  std::vector<json_path_processing_data> const& h_path_data,
  cudf::device_span<json_path_processing_data> d_path_data,
  int64_t max_row_size,  // Row-size bound for the two-rows-per-warp gate.
  bool use_last_non_null,
  launch_grouped_staging& staging,
  cuda::stream_ref stream)
{
  auto const num_outputs = json_paths.size();
  if (num_outputs == 0) { return; }

  int8_t* const d_max_path_depth_exceeded = d_error_check.data() + num_outputs;

  // The merged walk keeps the first duplicate; last-non-null selection needs the general scan.
  if (use_last_non_null) {
    kernel_launcher::exec(input, d_path_data, d_max_path_depth_exceeded, true, stream);
    return;
  }

  // Depth-1 groups need 124 paths; depth-2 keeps the merged route.
  std::vector<std::size_t> general_indices;
  std::vector<std::size_t> merged_shallow_indices;
  std::vector<std::size_t> merged_deep_indices;
  for (std::size_t idx = 0; idx < num_outputs; ++idx) {
    bool has_wc    = false;
    bool has_index = false;
    bool has_named = false;
    for (auto const& [type, name, index] : json_paths[idx]) {
      if (type == path_instruction_type::WILDCARD) { has_wc = true; }
      if (type == path_instruction_type::INDEX) { has_index = true; }
      if (type == path_instruction_type::NAMED) { has_named = true; }
    }
    // Only pure-NAMED depths <= 2 or <= 5 route shallow/deep; all else stays general.
    if (!has_wc && !has_index && has_named) {
      auto const depth = json_paths[idx].size();
      if (depth <= k_merged_max_depth) {
        merged_shallow_indices.push_back(idx);
      } else if (depth <= k_merged_deep_max_depth) {
        merged_deep_indices.push_back(idx);
      } else {
        general_indices.push_back(idx);
      }
    } else {
      general_indices.push_back(idx);
    }
  }

  // The 32-row floor protects small groups from extra merged-launch overhead.
  auto constexpr k_min_rows_per_family = std::size_t{32};

  // Caller-owned staging retains pack bytes while the kernels are in flight.
  auto& group_staging = staging.group_staging;
  auto& merged_plans  = staging.merged_plans;

  // Stage group views for one pinned H2D pack.
  auto const build_group_view = [&](std::vector<std::size_t> const& indices) {
    group_staging.emplace_back();
    auto& h_group = group_staging.back();
    h_group.reserve(indices.size());
    for (auto idx : indices) {
      h_group.push_back(h_path_data[idx]);
    }
    // Return the group's ordinal for its later pack offset.
    return group_staging.size() - 1;
  };

  // Eight eligible depth-three-to-five paths admit the deep stack and absorb shallow paths before
  // the shallow gate; a smaller deep group falls back without helping shallow admission.
  if (!merged_deep_indices.empty() && merged_deep_indices.size() >= k_deep_merged_min_paths &&
      static_cast<std::size_t>(input.size()) >= k_min_rows_per_family) {
    merged_shallow_indices.insert(
      merged_shallow_indices.end(), merged_deep_indices.begin(), merged_deep_indices.end());
    merged_plans.emplace_back(build_merged_named_plan(
      json_paths, merged_shallow_indices, static_cast<int>(k_merged_deep_max_depth), stream));
  } else if (!merged_deep_indices.empty()) {
    general_indices.insert(
      general_indices.end(), merged_deep_indices.begin(), merged_deep_indices.end());
  }

  if (merged_plans.empty() && !merged_shallow_indices.empty()) {
    std::size_t max_inst_count = 0;
    for (auto idx : merged_shallow_indices) {
      max_inst_count = std::max(max_inst_count, json_paths[idx].size());
    }
    if (max_inst_count <= k_merged_max_depth &&
        (max_inst_count == 2 || merged_shallow_indices.size() >= k_merged_min_depth1_paths) &&
        merged_shallow_indices.size() >= k_merged_min_paths &&
        static_cast<std::size_t>(input.size()) >= k_min_rows_per_family) {
      merged_plans.emplace_back(build_merged_named_plan(
        json_paths, merged_shallow_indices, k_merged_max_trie_depth, stream));
    } else {
      general_indices.insert(
        general_indices.end(), merged_shallow_indices.begin(), merged_shallow_indices.end());
    }
  }

  // No merged plan: launch all paths from the existing device view on the caller's stream.
  if (merged_plans.empty()) {
    kernel_launcher::exec(input, d_path_data, d_max_path_depth_exceeded, use_last_non_null, stream);
    return;
  }

  // Pack at most one merged view and one general view.
  bool const has_merged = !merged_plans.empty();
  std::optional<std::size_t> merged_group;
  std::optional<std::size_t> general_group;
  if (has_merged) { merged_group = build_group_view(merged_shallow_indices); }
  if (!general_indices.empty()) { general_group = build_group_view(general_indices); }

  // The 8-byte-aligned pack stores trie blobs and group views, whose spans already point to device
  // data.
  auto const align8      = [](std::size_t v) { return (v + 7) & ~std::size_t{7}; };
  std::size_t pack_bytes = 0;
  for (auto const& plan : merged_plans) {
    pack_bytes += align8(plan.h_blob.size());
  }
  for (auto const& group : group_staging) {
    pack_bytes += align8(group.size() * sizeof(json_path_processing_data));
  }
  staging.h_prewalk_pack = cudf::detail::make_pinned_vector_async<std::byte>(pack_bytes, stream);
  staging.d_prewalk_pack =
    rmm::device_buffer(pack_bytes, stream, cudf::get_current_device_resource_ref());
  auto* const h_base = staging.h_prewalk_pack->data();
  auto* const d_base = static_cast<std::byte*>(staging.d_prewalk_pack.data());

  std::size_t off = 0;
  for (auto& plan : merged_plans) {
    std::memcpy(h_base + off, plan.h_blob.data(), plan.h_blob.size());
    plan.blob_off = off;
    off += align8(plan.h_blob.size());
  }

  // Group views retain device-absolute spans; the merged kernel does not read path_commands.
  std::vector<json_path_processing_data*> d_groups;
  for (auto const& group : group_staging) {
    auto const bytes = group.size() * sizeof(json_path_processing_data);
    std::memcpy(h_base + off, group.data(), bytes);
    d_groups.push_back(reinterpret_cast<json_path_processing_data*>(d_base + off));
    off += align8(bytes);
  }

  // Pinned/pageable H2D source must outlive the first ordering sync (non-default streams).
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_base, h_base, pack_bytes, cudaMemcpyDefault, stream.get()));

  for (auto& plan : merged_plans) {
    bind_merged_plan_view(plan, reinterpret_cast<uint8_t const*>(d_base));
  }

  // Upload precedes both kernels; join folds the fork into stream before the caller reads errors.
  auto const span_of = [&](std::optional<std::size_t> group_ord) {
    return cudf::device_span<json_path_processing_data>{d_groups[*group_ord],
                                                        group_staging[*group_ord].size()};
  };
  auto const launch_merged = [&] {
    launch_merged_named(input,
                        span_of(merged_group),
                        merged_plans.back(),
                        max_row_size,
                        d_max_path_depth_exceeded,
                        stream);
  };
  if (has_merged && general_group) {
    auto fork = cudf::detail::fork_streams(stream, 1);
    launch_merged();
    kernel_launcher::exec(
      input, span_of(general_group), d_max_path_depth_exceeded, use_last_non_null, fork[0]);
    cudf::detail::join_streams(fork, stream);
  } else if (has_merged) {
    launch_merged();
  } else if (general_group) {
    kernel_launcher::exec(
      input, span_of(general_group), d_max_path_depth_exceeded, use_last_non_null, stream);
  }
}

}  // namespace

std::vector<std::unique_ptr<cudf::column>> get_json_object_batch(
  cudf::column_device_view const& input,
  cudf::detail::input_offsetalator const& in_offsets,
  int64_t base_offset,
  std::vector<cudf::host_span<std::tuple<path_instruction_type, std::string, int32_t> const>> const&
    json_paths,
  int64_t scratch_size,
  bool use_last_non_null,
  int64_t max_row_size,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const [d_json_paths, h_json_paths, d_inst_names, h_inst_names] =
    construct_path_commands(json_paths, stream);

  auto const num_outputs = json_paths.size();

  // Error slots: per-path OOB, then the depth-exceeded marker.
  rmm::device_uvector<int8_t> d_error_check(num_outputs + 1, stream);
  auto const d_max_path_depth_exceeded = d_error_check.data() + num_outputs;

  // Share string-view metadata while keeping each path's scratch independently allocated.
  using stringview_pair = cuda::std::pair<char const*, cudf::size_type>;
  auto const rows       = static_cast<std::size_t>(input.size());
  // `second` = write footprint for OOB rows, else committed result length.
  rmm::device_uvector<stringview_pair> out_stringviews(num_outputs * rows, stream);
  std::vector<rmm::device_uvector<char>> scratch_buffers;
  scratch_buffers.reserve(num_outputs);
  std::vector<json_path_processing_data> h_path_data;
  h_path_data.reserve(num_outputs);

  auto* const stringviews_base  = out_stringviews.data();
  auto const stringview_span_of = [&](std::size_t idx) {
    return cudf::device_span<stringview_pair const>(stringviews_base + idx * rows, rows);
  };
  auto const stringview_span_mutable_of = [&](std::size_t idx) {
    return cudf::device_span<stringview_pair>(stringviews_base + idx * rows, rows);
  };
  auto const assemble_stringviews = [&](auto indices, std::size_t count) {
    std::vector<cudf::device_span<stringview_pair const>> stringviews;
    stringviews.reserve(count);
    for (std::size_t idx = 0; idx < count; ++idx) {
      stringviews.emplace_back(stringview_span_of(indices[idx]));
    }
    return stringviews;
  };

  for (std::size_t idx = 0; idx < num_outputs; ++idx) {
    auto const& path = json_paths[idx];
    if (path.size() > MAX_JSON_PATH_DEPTH) {
      CUDF_FAIL("JSON Path has depth exceeds the maximum allowed value.");
    }

    scratch_buffers.emplace_back(rmm::device_uvector<char>(scratch_size, stream));
    h_path_data.emplace_back(json_path_processing_data{d_json_paths[idx],
                                                       in_offsets,
                                                       base_offset,
                                                       stringviews_base + idx * rows,
                                                       scratch_buffers.back().data(),
                                                       d_error_check.data() + idx});
  }
  auto d_path_data = cudf::detail::make_device_uvector_async(
    h_path_data, stream, cudf::get_current_device_resource_ref());
  auto all_stringviews =
    assemble_stringviews(thrust::make_counting_iterator(std::size_t{0}), num_outputs);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    d_error_check.begin(),
    d_error_check.end(),
    0);

  // Pinned/pageable H2D source must outlive the first ordering sync (non-default streams).
  launch_grouped_staging launch_staging;
  launch_grouped(input,
                 json_paths,
                 d_error_check,
                 h_path_data,
                 d_path_data,
                 max_row_size,
                 use_last_non_null,
                 launch_staging,
                 stream);

  // Check errors before consuming prepared totals; one sync orders both readbacks.
  // Below the gate, error readback and both output sites keep the factory path.
  bool const use_packed_readback = num_outputs >= packed_readback_min_paths;
  std::optional<prepared_output_metadata> prepared_metadata;
  if (use_packed_readback) {
    prepared_metadata = prepare_output_metadata(
      input, all_stringviews, stream, mr, cudf::device_span<int8_t>{d_error_check});
  }

  // Both routes order H2D sources; a small batch reads error markers directly.
  bool has_no_oob = false;
  cudf::host_span<int8_t const> error_markers;
  std::optional<cudf::detail::host_vector<int8_t>> h_error_check;
  if (use_packed_readback && prepared_metadata->num_err > 0) {
    stream.sync();  // completes the packed D2H (totals + counts + error markers)
    prepared_metadata->already_synced = true;
    error_markers                     = cudf::host_span<int8_t const>{
      prepared_metadata->h_error_check(), static_cast<std::size_t>(prepared_metadata->num_err)};
  } else {
    h_error_check = cudf::detail::make_pinned_vector(d_error_check, stream);
    error_markers = *h_error_check;
  }
  has_no_oob = check_error(error_markers);

  // Only unflagged scratch may be gathered; successful string views already carry final validity.
  if (has_no_oob) {
    // The admission gate is identical here AND at the OOB site.
    if (!use_packed_readback) {
      return cudf::make_strings_column_batch(all_stringviews, stream, mr);
    }
    return finish_output_columns(std::move(*prepared_metadata), all_stringviews, stream, mr);
  }
  // Retry flagged paths using the first pass's lengths; unflagged paths keep their existing views.

  std::vector<std::pair<std::unique_ptr<cudf::column>, int64_t>> out_offsets_and_sizes;
  std::vector<rmm::device_uvector<char>> out_char_buffers;
  std::vector<std::size_t> oob_indices;
  std::vector<std::size_t> no_oob_indices;

  // Rebuild only paths that had out-of-bound writes; retry slots are sized
  // max(stored size, input row size).
  h_path_data.clear();
  for (std::size_t idx = 0; idx < num_outputs; ++idx) {
    auto const out_sview = stringview_span_of(idx);

    if (error_markers[idx]) {
      oob_indices.emplace_back(idx);

      // Non-OOB rows' discarded writes never exceed their input row size.
      auto const size_it = spark_rapids_jni::util::make_counting_transform_iterator(
        0,
        cuda::proclaim_return_type<cudf::size_type>(
          [string_pairs = out_sview.data(), in_offsets] __device__(auto const idx) {
            auto const in_size = in_offsets[idx + 1] - in_offsets[idx];
            return cuda::std::max(string_pairs[idx].second, static_cast<cudf::size_type>(in_size));
          }));
      out_offsets_and_sizes.emplace_back(cudf::strings::detail::make_offsets_child_column(
        size_it, size_it + input.size(), stream, mr));
      out_char_buffers.emplace_back(
        rmm::device_uvector<char>(out_offsets_and_sizes.back().second, stream, mr));

      h_path_data.emplace_back(
        json_path_processing_data{d_json_paths[idx],
                                  cudf::detail::offsetalator_factory::make_input_iterator(
                                    out_offsets_and_sizes.back().first->view()),
                                  0,
                                  stringview_span_mutable_of(idx).data(),
                                  out_char_buffers.back().data(),
                                  d_error_check.data() + idx});
    } else {
      no_oob_indices.emplace_back(idx);
    }
  }

  std::vector<std::unique_ptr<cudf::column>> output(num_outputs);
  auto no_oob_stringviews = assemble_stringviews(no_oob_indices.begin(), no_oob_indices.size());
  // Same admission gate as the happy site: a small batch builds retry columns
  // with make_strings_column_batch, a large one reuses the packed machinery.
  auto no_oob_output =
    (!use_packed_readback)
      ? cudf::make_strings_column_batch(no_oob_stringviews, stream, mr)
      : finish_output_columns(prepare_output_metadata(input, no_oob_stringviews, stream, mr),
                              no_oob_stringviews,
                              stream,
                              mr);
  for (std::size_t idx = 0; idx < no_oob_indices.size(); ++idx) {
    auto const out_idx = no_oob_indices[idx];
    output[out_idx]    = std::move(no_oob_output[idx]);
  }

  // The flat pair storage stays alive through the retry: the retry kernel overwrites the
  // flagged paths' pairs in place, and the rebuild below reads them after the retry.
  scratch_buffers.clear();

  // The retry overwrites flagged paths' output pairs; rebuilt columns use those lengths and
  // validity.
  d_path_data = cudf::detail::make_device_uvector_async(
    h_path_data, stream, cudf::get_current_device_resource_ref());
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    d_error_check.begin(),
    d_error_check.end(),
    0);
  kernel_launcher::exec(input, d_path_data, d_max_path_depth_exceeded, use_last_non_null, stream);
  auto h_error_check_retry = cudf::detail::make_pinned_vector(d_error_check, stream);
  has_no_oob               = check_error(h_error_check_retry);

  // Identical input and paths must reproduce measured lengths; still check the retry for overflow.
  CUDF_EXPECTS(has_no_oob, "Unexpected out-of-bound write in get_json_object kernel.");

  // Rebuild each retried column from its own post-retry pairs while the flat pair storage is
  // still alive; the release below may only happen after the last consumer is enqueued.
  for (std::size_t i = 0; i < oob_indices.size(); ++i) {
    auto const out_idx   = oob_indices[i];
    auto const out_sview = stringview_span_of(out_idx);
    // Rebuild from the retry's committed pairs: the pre-retry offsets only size the scratch
    // slots, so adopting them as logical offsets would leak unwritten trailing bytes into rows.
    output[out_idx] =
      cudf::strings::detail::make_strings_column(out_sview.begin(), out_sview.end(), stream, mr);
  }
  // First-pass scratch and views are unreferenced by the rebuilt columns or the retry.
  scratch_buffers.clear();
  (void)out_stringviews.release();
  return output;
}

std::vector<std::unique_ptr<cudf::column>> get_json_object(
  cudf::strings_column_view const& input,
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>> const&
    json_paths,
  int64_t memory_budget_bytes,
  int32_t parallel_override,
  bool use_last_non_null,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_outputs = json_paths.size();

  if (input.is_empty() || input.size() == input.null_count()) {
    std::vector<std::unique_ptr<cudf::column>> output;
    for (std::size_t idx = 0; idx < num_outputs; ++idx) {
      output.emplace_back(std::make_unique<cudf::column>(input.parent(), stream, mr));
    }
    return output;
  }

  std::vector<std::size_t> sorted_indices(json_paths.size());
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

  std::sort(sorted_indices.begin(), sorted_indices.end(), [&json_paths](size_t i, size_t j) {
    return json_paths[i] < json_paths[j];
  });

  auto const in_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
  int64_t max_row_size{};
  auto const [scratch_size, base_offset] =
    calc_scratch_size(input, in_offsets, stream, &max_row_size);
  if (memory_budget_bytes <= 0 && parallel_override <= 0) {
    parallel_override = static_cast<int>(sorted_indices.size());
  }
  auto const d_input_ptr = cudf::column_device_view::create(
    input.parent(), stream, cudf::get_current_device_resource_ref());
  std::vector<std::unique_ptr<cudf::column>> output(num_outputs);

  std::vector<cudf::host_span<std::tuple<path_instruction_type, std::string, int32_t> const>> batch;
  std::vector<std::size_t> output_ids;

  std::size_t starting_path = 0;
  while (starting_path < num_outputs) {
    std::size_t at = starting_path;
    batch.resize(0);
    output_ids.resize(0);
    int count   = 0;
    long budget = 0;
    while (at < num_outputs &&
           (parallel_override > 0 ? count < parallel_override : budget < memory_budget_bytes)) {
      auto output_location = sorted_indices[at];
      batch.emplace_back(json_paths[output_location]);
      output_ids.push_back(output_location);
      at++;
      if (parallel_override > 0) {
        count++;
      } else {
        budget += scratch_size;
      }
    }
    auto tmp = get_json_object_batch(*d_input_ptr,
                                     in_offsets,
                                     base_offset,
                                     batch,
                                     scratch_size,
                                     use_last_non_null,
                                     max_row_size,
                                     stream,
                                     mr);
    for (std::size_t i = 0; i < tmp.size(); i++) {
      std::size_t out_i = output_ids[i];
      output[out_i]     = std::move(tmp[i]);
    }
    starting_path = at;
  }
  return output;
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
