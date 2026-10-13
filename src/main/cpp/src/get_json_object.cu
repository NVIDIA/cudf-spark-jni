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
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/scalar/scalar.hpp>
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
#include <cuda/functional>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <bit>
#include <numeric>

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
   * Child bytes are already at the parent's end; make room for the deferred comma and brackets.
   * With a comma: parent prefix [1 and child 2,3,4 become [1,2,3,4 without wrapping.
   * With wrapping too, the same prefix and child become [1,[2,3,4].
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

// Parent contexts resume according to the case that created their child.
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

// Depth errors are deferred to the host; ordinary evaluation failure returns a zero size.
__device__ cuda::std::pair<bool, cudf::size_type> evaluate_path(
  json_parser& p,
  cudf::device_span<path_instruction const> path_commands,
  char* out_buf,
  int8_t* max_path_depth_exceeded)
{
  p.next_token();
  if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

  // One extra slot for the root context.
  context stack[MAX_JSON_PATH_DEPTH + 1];
  int stack_size = 0;

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
          if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }
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
        if (!(ctx.g.copy_current_structure(p, out_buf))) { return {false, 0}; }
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
              if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

              p.next_token();
              if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

              if (!p.try_skip_children()) { return {false, 0}; }
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

            if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

            if (is_name_matched) {
              p.next_token();
              if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

              // A null first occurrence does not fall back to a later duplicate.
              if (json_token::VALUE_NULL == p.get_current_token()) { return {false, 0}; }
              push_context(evaluation_case_path::START_OBJECT___MATCHED_NAME_PATH,
                           ctx.g,
                           ctx.style,
                           {ctx.path.data() + 1, ctx.path.size() - 1});
              found_expected_child = true;
              break;
            } else {
              p.next_token();
              if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

              if (!p.try_skip_children()) { return {false, 0}; }
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
        // Consecutive wildcards flatten terminal arrays instead of preserving their nesting.
        if (ctx.is_first_enter) {
          ctx.is_first_enter = false;
          ctx.g.write_start_array(out_buf);
        }

        if (p.next_token() != json_token::END_ARRAY) {
          if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }
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

        // Delay brackets until the number of successful matches is known.
        json_generator child_g;
        if (ctx.is_first_enter) {
          ctx.is_first_enter = false;
          child_g            = ctx.g.new_child_generator();
          child_g.write_first_start_array_without_output();
        } else {
          child_g = ctx.child_g;
        }

        if (p.next_token() != json_token::END_ARRAY) {
          if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }
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
          if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

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
        if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }
        ctx.is_first_enter = false;

        int i = idx;
        while (i > 0) {
          if (p.get_current_token() == json_token::END_ARRAY) { return {false, 0}; }

          if (!p.try_skip_children()) { return {false, 0}; }

          p.next_token();
          if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

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
        if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

        int i = idx;
        while (i > 0) {
          if (p.get_current_token() == json_token::END_ARRAY) { return {false, 0}; }

          if (!p.try_skip_children()) { return {false, 0}; }

          p.next_token();
          if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

          --i;
        }

        push_context(evaluation_case_path::START_ARRAY___MATCHED_INDEX,
                     ctx.g,
                     ctx.style,
                     {ctx.path.data() + 1, ctx.path.size() - 1});
      }
      // Case 12: no match.
      else {
        if (!p.try_skip_children()) { return {false, 0}; }
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
          p_ctx.g     = ctx.g;
          break;
        }

          // Case 6.
        case evaluation_case_path::START_ARRAY___MATCHED_WILDCARD___STYLE_NOT_QUOTED: {
          p_ctx.dirty += ctx.dirty;
          p_ctx.child_g = ctx.g;
          break;
        }

          // Cases 8 and 9.
        case evaluation_case_path::START_ARRAY___MATCHED_INDEX_AND_WILDCARD:
        case evaluation_case_path::START_ARRAY___MATCHED_INDEX: {
          p_ctx.dirty += ctx.dirty;

          while (p.next_token() != json_token::END_ARRAY) {
            if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }
            if (!p.try_skip_children()) { return {false, 0}; }
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

  // Partial output must not contribute to the retry buffer size.
  return {success, success ? stack[0].g.get_output_len() : 0};
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

struct json_path_processing_data {
  cudf::device_span<path_instruction const> path_commands;
  cudf::detail::input_offsetalator offsets;
  int64_t base_offset;
  cuda::std::pair<char const*, cudf::size_type>* out_stringviews;
  char* out_buf;
  int8_t* has_out_of_bound;
};

// Each active thread evaluates one (row, path); extra per-row threads return without parsing.
template <int block_size, int min_block_per_sm, bool use_last_non_null>
__launch_bounds__(block_size, min_block_per_sm) CUDF_KERNEL
  void get_json_object_kernel(cudf::column_device_view input,
                              cudf::device_span<json_path_processing_data> path_data,
                              std::size_t num_threads_per_row,
                              int8_t* max_path_depth_exceeded)
{
  auto const tidx    = cudf::detail::grid_1d::global_thread_id();
  auto const row_idx = tidx / num_threads_per_row;
  if (row_idx >= input.size()) { return; }

  auto const path_idx = tidx % num_threads_per_row;
  if (path_idx >= path_data.size()) { return; }

  // Slice offsets are absolute, but scratch owns only the selected bytes.
  auto const& path         = path_data[path_idx];
  char* const dst          = path.out_buf + (path.offsets[row_idx] - path.base_offset);
  bool is_valid            = false;
  cudf::size_type out_size = 0;

  auto const str = input.element<cudf::string_view>(row_idx);
  if (static_cast<bool>(input.null_mask()) && input.is_null(row_idx)) {
    if (path.out_stringviews) { path.out_stringviews[row_idx] = {nullptr, 0}; }
    return;
  }
  if (str.size_bytes() > 0) {
    json_parser p{char_range{str}};
    if constexpr (use_last_non_null) {
      auto const selection =
        select_last_non_null(p, path.path_commands.front().name, max_path_depth_exceeded);
      if (selection.valid && !selection.value.is_null()) {
        json_parser selected_parser{selection.value};
        cuda::std::tie(is_valid, out_size) =
          evaluate_path(selected_parser,
                        cudf::device_span<path_instruction const>{},
                        dst,
                        max_path_depth_exceeded);
      }
    } else {
      cuda::std::tie(is_valid, out_size) =
        evaluate_path(p, path.path_commands, dst, max_path_depth_exceeded);
    }

    // The host throws on depth overflow before using any output.
    if (p.max_nesting_depth_exceeded()) {
      set_device_flag(max_path_depth_exceeded);
      return;
    }

    auto const max_size = path.offsets[row_idx + 1] - path.offsets[row_idx];
    if (out_size > max_size) { set_device_flag(path.has_out_of_bound); }
  }

  // The retry reuses first-pass lengths and validity, so it need not rewrite string views.
  if (path.out_stringviews) {
    path.out_stringviews[row_idx] = {is_valid ? dst : nullptr, out_size};
  }
}

void launch_general(cudf::column_device_view const& input,
                    cudf::device_span<json_path_processing_data> path_data,
                    int8_t* max_path_depth_exceeded,
                    bool use_last_non_null,
                    cuda::stream_ref stream)
{
  // Both launch bounds let the compiler trade registers for fewer instructions;
  // these values were measured faster than a 256-thread, 4-block pairing.
  constexpr int block_size       = 128;
  constexpr int min_block_per_sm = 8;
  // Power-of-two groups pack short path batches: 3 paths use 4 threads; 1 and 33 use 32 and 64.
  auto const num_threads_per_row =
    path_data.size() <= cudf::detail::warp_size && path_data.size() > 1
      ? std::bit_ceil(path_data.size())
      : cudf::util::div_rounding_up_safe(path_data.size(),
                                         static_cast<std::size_t>(cudf::detail::warp_size)) *
          cudf::detail::warp_size;
  auto const num_blocks = cudf::util::div_rounding_up_safe(num_threads_per_row * input.size(),
                                                           static_cast<std::size_t>(block_size));
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
      path_commands, stream, rmm::mr::get_current_device_resource_ref()));
  }

  return {std::move(d_path_commands),
          std::move(h_path_commands),
          std::move(d_inst_names),
          std::move(h_inst_names)};
}

// Ten maximum-length rows provide padding for writes that expand before overflow is detected.
__global__ void scratch_size_kernel(cudf::detail::input_offsetalator const offsets,
                                    cudf::size_type rows,
                                    int64_t* stats)
{
  // Threads scan disjoint row sets to share the input reads.
  constexpr int block_threads = 256;
  int64_t max_row_size        = 0;
  int64_t chars_size          = 0;
  for (cudf::size_type idx = threadIdx.x; idx < rows; idx += block_threads) {
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

cuda::std::pair<int64_t, int64_t> calc_scratch_size(
  cudf::strings_column_view const& input,
  cudf::detail::input_offsetalator const& in_offsets,
  cuda::stream_ref stream)
{
  // Read all statistics together to avoid separate synchronizations.
  auto const rows = input.size();
  rmm::device_uvector<int64_t> d_stats(3, stream);
  scratch_size_kernel<<<1, 256, 0, stream.get()>>>(in_offsets, rows, d_stats.data());
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  auto const h_stats = cudf::detail::make_host_vector(d_stats, stream);

  auto constexpr padding_rows = 10;
  return {h_stats[0] * padding_rows + h_stats[1], h_stats[2]};
}

bool check_error(cudf::detail::host_vector<int8_t> const& error_check)
{
  CUDF_EXPECTS(error_check.back() == 0,
               "The processed input has nesting depth exceeds depth limit.");

  // The checked depth marker is zero, so it can remain in the range.
  return std::none_of(
    error_check.cbegin(), error_check.cend(), [](auto const val) { return val != 0; });
}

std::vector<std::unique_ptr<cudf::column>> get_json_object_batch(
  cudf::column_device_view const& input,
  cudf::detail::input_offsetalator const& in_offsets,
  int64_t base_offset,
  std::vector<cudf::host_span<std::tuple<path_instruction_type, std::string, int32_t> const>> const&
    json_paths,
  int64_t scratch_size,
  bool use_last_non_null,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const [d_json_paths, h_json_paths, d_inst_names, h_inst_names] =
    construct_path_commands(json_paths, stream);

  auto const num_outputs = json_paths.size();

  // One size-overflow flag per output, then the shared depth-overflow flag.
  rmm::device_uvector<int8_t> d_error_check(num_outputs + 1, stream);
  auto const d_max_path_depth_exceeded = d_error_check.data() + num_outputs;

  std::vector<rmm::device_uvector<char>> scratch_buffers;
  std::vector<rmm::device_uvector<cuda::std::pair<char const*, cudf::size_type>>> out_stringviews;
  std::vector<json_path_processing_data> h_path_data;
  scratch_buffers.reserve(json_paths.size());
  out_stringviews.reserve(json_paths.size());
  h_path_data.reserve(json_paths.size());

  for (std::size_t idx = 0; idx < num_outputs; ++idx) {
    auto const& path = json_paths[idx];
    if (path.size() > MAX_JSON_PATH_DEPTH) {
      CUDF_FAIL("JSON Path has depth exceeds the maximum allowed value.");
    }

    scratch_buffers.emplace_back(rmm::device_uvector<char>(scratch_size, stream));
    out_stringviews.emplace_back(rmm::device_uvector<cuda::std::pair<char const*, cudf::size_type>>{
      static_cast<std::size_t>(input.size()), stream});

    h_path_data.emplace_back(json_path_processing_data{d_json_paths[idx],
                                                       in_offsets,
                                                       base_offset,
                                                       out_stringviews.back().data(),
                                                       scratch_buffers.back().data(),
                                                       d_error_check.data() + idx});
  }
  auto d_path_data = cudf::detail::make_device_uvector_async(
    h_path_data, stream, rmm::mr::get_current_device_resource_ref());
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    d_error_check.begin(),
    d_error_check.end(),
    0);

  launch_general(input, d_path_data, d_max_path_depth_exceeded, use_last_non_null, stream);
  auto h_error_check = cudf::detail::make_host_vector(d_error_check, stream);
  auto has_no_oob    = check_error(h_error_check);

  std::vector<cudf::device_span<cuda::std::pair<char const*, cudf::size_type> const>>
    batch_stringviews;
  batch_stringviews.reserve(out_stringviews.size());

  if (has_no_oob) {
    for (auto const& out_sview : out_stringviews) {
      batch_stringviews.emplace_back(out_sview);
    }
    return cudf::make_strings_column_batch(batch_stringviews, stream, mr);
  }

  // Retry only oversized outputs; the first pass already recorded exact lengths and validity.
  std::vector<std::pair<cuda::device_buffer<std::byte>, cudf::size_type>>
    out_null_masks_and_null_counts;
  std::vector<std::pair<std::unique_ptr<cudf::column>, int64_t>> out_offsets_and_sizes;
  std::vector<rmm::device_uvector<char>> out_char_buffers;
  std::vector<std::size_t> oob_indices;
  std::vector<std::size_t> no_oob_indices;

  auto const validator = [] __device__(cuda::std::pair<char const*, cudf::size_type> const item) {
    return item.first != nullptr;
  };

  h_path_data.clear();
  for (std::size_t idx = 0; idx < num_outputs; ++idx) {
    auto const& out_sview = out_stringviews[idx];

    if (h_error_check[idx]) {
      oob_indices.emplace_back(idx);

      out_null_masks_and_null_counts.emplace_back(
        cudf::detail::valid_if(out_sview.begin(), out_sview.end(), validator, stream, mr));

      auto const size_it = spark_rapids_jni::util::make_counting_transform_iterator(
        0,
        cuda::proclaim_return_type<cudf::size_type>(
          [string_pairs = out_sview.data()] __device__(auto const idx) {
            return string_pairs[idx].second;
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
                                  nullptr /*out_stringviews*/,
                                  out_char_buffers.back().data(),
                                  d_error_check.data() + idx});
    } else {
      no_oob_indices.emplace_back(idx);
      batch_stringviews.emplace_back(out_sview);
    }
  }

  std::vector<std::unique_ptr<cudf::column>> output(num_outputs);
  auto no_oob_output = cudf::make_strings_column_batch(batch_stringviews, stream, mr);
  for (std::size_t idx = 0; idx < no_oob_indices.size(); ++idx) {
    auto const out_idx = no_oob_indices[idx];
    output[out_idx]    = std::move(no_oob_output[idx]);
  }

  scratch_buffers.clear();
  out_stringviews.clear();

  d_path_data = cudf::detail::make_device_uvector_async(
    h_path_data, stream, rmm::mr::get_current_device_resource_ref());
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    d_error_check.begin(),
    d_error_check.end(),
    0);
  launch_general(input, d_path_data, d_max_path_depth_exceeded, use_last_non_null, stream);
  h_error_check = cudf::detail::make_host_vector(d_error_check, stream);
  has_no_oob    = check_error(h_error_check);

  // Identical input and paths must reproduce the recorded lengths.
  CUDF_EXPECTS(has_no_oob, "Unexpected out-of-bound write in get_json_object kernel.");

  for (std::size_t idx = 0; idx < oob_indices.size(); ++idx) {
    auto const out_idx = oob_indices[idx];
    output[out_idx] =
      cudf::make_strings_column(input.size(),
                                std::move(out_offsets_and_sizes[idx].first),
                                out_char_buffers[idx].release(),
                                out_null_masks_and_null_counts[idx].second,
                                std::move(out_null_masks_and_null_counts[idx].first));
  }
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
  auto const [scratch_size, base_offset] = calc_scratch_size(input, in_offsets, stream);
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
    if (parallel_override > 0) {
      int count = 0;
      while (at < num_outputs && count < parallel_override) {
        auto output_location = sorted_indices[at];
        batch.emplace_back(json_paths[output_location]);
        output_ids.push_back(output_location);
        at++;
        count++;
      }
    } else {
      long budget = 0;
      while (at < num_outputs && budget < memory_budget_bytes) {
        auto output_location = sorted_indices[at];
        batch.emplace_back(json_paths[output_location]);
        output_ids.push_back(output_location);
        at++;
        budget += scratch_size;
      }
    }
    auto tmp = get_json_object_batch(
      *d_input_ptr, in_offsets, base_offset, batch, scratch_size, use_last_non_null, stream, mr);
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
