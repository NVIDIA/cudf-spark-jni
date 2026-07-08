/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

#include "cast_string.hpp"
#include "json_utils.hpp"
#include "nvtx_ranges.hpp"
#include "utilities/iterator.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/io/json.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>

#include <algorithm>
#include <map>
#include <unordered_map>

namespace spark_rapids_jni {

namespace detail {

namespace {

/**
 * @brief The struct similar to `cudf::io::schema_element` with adding decimal precision and
 * preserving column order.
 */
struct schema_element_with_precision {
  cudf::data_type type;
  int precision;
  std::vector<std::pair<std::string, schema_element_with_precision>> child_types;
};

std::pair<cudf::io::schema_element, schema_element_with_precision> parse_schema_element(
  std::size_t& index,
  std::vector<std::string> const& col_names,
  std::vector<int> const& num_children,
  std::vector<int> const& types,
  std::vector<int> const& scales,
  std::vector<int> const& precisions)
{
  // Get data for the current column.
  auto const d_type    = cudf::data_type{static_cast<cudf::type_id>(types[index]), scales[index]};
  auto const precision = precisions[index];
  auto const col_num_children = num_children[index];
  index++;

  std::map<std::string, cudf::io::schema_element> children;
  std::vector<std::pair<std::string, schema_element_with_precision>> children_with_precisions;
  std::vector<std::string> child_names(col_num_children);

  if (d_type.id() == cudf::type_id::STRUCT || d_type.id() == cudf::type_id::LIST) {
    for (int i = 0; i < col_num_children; ++i) {
      auto const& name = col_names[index];
      auto [child, child_with_precision] =
        parse_schema_element(index, col_names, num_children, types, scales, precisions);
      children.emplace(name, std::move(child));
      children_with_precisions.emplace_back(name, std::move(child_with_precision));
      child_names[i] = name;
    }
  } else {
    CUDF_EXPECTS(col_num_children == 0,
                 "Found children for a non-nested type that should have none.",
                 std::invalid_argument);
  }

  // Note that if the first schema element does not has type STRUCT/LIST then it always has type
  // STRING, since we intentionally parse JSON into strings column for later post-processing.
  auto const schema_dtype =
    d_type.id() == cudf::type_id::STRUCT || d_type.id() == cudf::type_id::LIST
      ? d_type
      : cudf::data_type{cudf::type_id::STRING};
  return {cudf::io::schema_element{schema_dtype, std::move(children), {std::move(child_names)}},
          schema_element_with_precision{d_type, precision, std::move(children_with_precisions)}};
}

// Generate struct type schemas by traveling the schema data by depth-first search order.
// Two separate schemas is generated:
// - The first one is used as input to `cudf::read_json`, in which the data types of all columns
//   are specified as STRING type. As such, the table returned by `cudf::read_json` will contain
//   only strings columns or nested (LIST/STRUCT) columns.
// - The second schema contains decimal precision (if available) and preserves schema column types
//   as well as the column order, used for converting from STRING type to the desired types for the
//   final output.
std::pair<cudf::io::schema_element, schema_element_with_precision> generate_struct_schema(
  std::vector<std::string> const& col_names,
  std::vector<int> const& num_children,
  std::vector<int> const& types,
  std::vector<int> const& scales,
  std::vector<int> const& precisions)
{
  std::map<std::string, cudf::io::schema_element> schema_cols;
  std::vector<std::pair<std::string, schema_element_with_precision>> schema_cols_with_precisions;
  std::vector<std::string> name_order;

  std::size_t index = 0;
  while (index < types.size()) {
    auto const& name = col_names[index];
    auto [child, child_with_precision] =
      parse_schema_element(index, col_names, num_children, types, scales, precisions);
    schema_cols.emplace(name, std::move(child));
    schema_cols_with_precisions.emplace_back(name, std::move(child_with_precision));
    name_order.push_back(name);
  }
  return {
    cudf::io::schema_element{
      cudf::data_type{cudf::type_id::STRUCT}, std::move(schema_cols), {std::move(name_order)}},
    schema_element_with_precision{
      cudf::data_type{cudf::type_id::STRUCT}, -1, std::move(schema_cols_with_precisions)}};
}

void nullify_rows(cudf::column& input,
                  std::vector<cudf::size_type> const& row_indices,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr)
{
  if (row_indices.empty()) { return; }

  auto const input_view = input.view();
  auto null_mask =
    input_view.nullable()
      ? cudf::copy_bitmask(input_view, stream, mr)
      : cudf::create_null_mask(input_view.size(), cudf::mask_state::ALL_VALID, stream, mr);
  auto d_row_indices = cudf::detail::make_device_uvector_async(row_indices, stream, mr);

  auto mask_ptr = static_cast<cudf::bitmask_type*>(null_mask.data());
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   d_row_indices.begin(),
                   d_row_indices.end(),
                   [mask_ptr] __device__(auto const row) { cudf::clear_bit(mask_ptr, row); });

  auto const null_count = cudf::null_count(
    static_cast<cudf::bitmask_type const*>(null_mask.data()), 0, input_view.size(), stream);
  input.set_null_mask(std::move(null_mask), null_count);
}

std::unique_ptr<cudf::column> make_lists_column_with_null_sanitization(
  cudf::size_type num_rows,
  std::unique_ptr<cudf::column> offsets_column,
  std::unique_ptr<cudf::column> child_column,
  cudf::size_type null_count,
  rmm::device_buffer&& null_mask,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> children;
  children.emplace_back(std::move(offsets_column));
  children.emplace_back(std::move(child_column));
  auto output = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::LIST},
                                               num_rows,
                                               rmm::device_buffer{},
                                               std::move(null_mask),
                                               null_count,
                                               std::move(children));
  // Row-level schema mismatch nulls can leave child data under null parents; sanitize it here.
  if (cudf::has_nonempty_nulls(output->view(), stream)) {
    output = cudf::purge_nonempty_nulls(output->view(), stream, mr);
  }
  return output;
}

std::unique_ptr<cudf::column> make_structs_column_with_null_consistency(
  cudf::size_type num_rows,
  std::vector<std::unique_ptr<cudf::column>>&& children,
  cudf::size_type null_count,
  rmm::device_buffer&& null_mask,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (null_count > 0) {
    // make_structs_column superimposes parent nulls onto children for a consistent nested column.
    return cudf::make_structs_column(
      num_rows, std::move(children), null_count, std::move(null_mask), stream, mr);
  }

  return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRUCT},
                                        num_rows,
                                        rmm::device_buffer{},
                                        std::move(null_mask),
                                        null_count,
                                        std::move(children));
}

using string_index_pair = cuda::std::pair<char const*, cudf::size_type>;

std::unique_ptr<cudf::column> cast_strings_to_booleans(cudf::column_view const& input,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  auto const string_count = input.size();
  if (string_count == 0) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::BOOL8}); }

  auto output = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::BOOL8}, string_count, cudf::mask_state::UNALLOCATED, stream, mr);
  auto validity = rmm::device_uvector<bool>(string_count, stream);

  auto const input_sv = cudf::strings_column_view{input};
  auto const offsets_it =
    cudf::detail::offsetalator_factory::make_input_iterator(input_sv.offsets());
  auto const d_input_ptr = cudf::column_device_view::create(input, stream);
  auto const is_valid_it = cudf::detail::make_validity_iterator<true>(*d_input_ptr);
  auto const output_it =
    thrust::make_zip_iterator(output->mutable_view().begin<bool>(), validity.begin());
  thrust::tabulate(
    rmm::exec_policy_nosync(stream),
    output_it,
    output_it + string_count,
    [chars = input_sv.chars_begin(stream), offsets = offsets_it, is_valid = is_valid_it] __device__(
      auto idx) -> cuda::std::tuple<bool, bool> {
      if (is_valid[idx]) {
        auto const start_offset = offsets[idx];
        auto const end_offset   = offsets[idx + 1];
        auto const size         = end_offset - start_offset;
        auto const str          = chars + start_offset;

        if (size == 4 && str[0] == 't' && str[1] == 'r' && str[2] == 'u' && str[3] == 'e') {
          return {true, true};
        }
        if (size == 5 && str[0] == 'f' && str[1] == 'a' && str[2] == 'l' && str[3] == 's' &&
            str[4] == 'e') {
          return {false, true};
        }
      }

      // Either null input, or the input string is neither `true` nor `false`.
      return {false, false};
    });

  auto [null_mask, null_count] =
    cudf::detail::valid_if(validity.begin(), validity.end(), cuda::std::identity{}, stream, mr);
  output->set_null_mask(null_count > 0 ? std::move(null_mask) : rmm::device_buffer{0, stream, mr},
                        null_count);

  return output;
}

std::unique_ptr<cudf::column> cast_strings_to_integers(cudf::column_view const& input,
                                                       cudf::data_type output_type,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  auto const string_count = input.size();
  if (string_count == 0) { return cudf::make_empty_column(output_type); }

  auto const input_sv = cudf::strings_column_view{input};
  auto const input_offsets_it =
    cudf::detail::offsetalator_factory::make_input_iterator(input_sv.offsets());
  auto const d_input_ptr    = cudf::column_device_view::create(input, stream);
  auto const valid_input_it = cudf::detail::make_validity_iterator<true>(*d_input_ptr);

  // We need to nullify the invalid string rows.
  // Technically, we should just mask out these rows as nulls through the nullmask.
  // These masked out non-empty nulls will be handled in the conversion API.
  auto valids = rmm::device_uvector<bool>(string_count, stream);

  // Since the strings store integer numbers, they should be very short.
  // As such, using one thread per string should be fine.
  thrust::tabulate(rmm::exec_policy_nosync(stream),
                   valids.begin(),
                   valids.end(),
                   [chars       = input_sv.chars_begin(stream),
                    offsets     = input_offsets_it,
                    valid_input = valid_input_it] __device__(cudf::size_type idx) -> bool {
                     if (!valid_input[idx]) { return false; }

                     auto in_ptr       = chars + offsets[idx];
                     auto const in_end = chars + offsets[idx + 1];
                     while (in_ptr != in_end) {
                       if (*in_ptr == '.' || *in_ptr == 'e' || *in_ptr == 'E') { return false; }
                       ++in_ptr;
                     }

                     return true;
                   });

  auto const [null_mask, null_count] =
    cudf::detail::valid_if(valids.begin(),
                           valids.end(),
                           cuda::std::identity{},
                           stream,
                           cudf::get_current_device_resource_ref());
  // If the null count doesn't change, just use the input column for conversion.
  auto const input_applied_null =
    null_count == input.null_count()
      ? cudf::column_view{}
      : cudf::column_view{cudf::data_type{cudf::type_id::STRING},
                          input_sv.size(),
                          input_sv.chars_begin(stream),
                          reinterpret_cast<cudf::bitmask_type const*>(null_mask.data()),
                          null_count,
                          input_sv.offset(),
                          std::vector<cudf::column_view>{input_sv.offsets()}};

  return spark_rapids_jni::string_to_integer(
    output_type,
    null_count == input.null_count() ? input_sv : cudf::strings_column_view{input_applied_null},
    /*ansi_mode*/ false,
    /*strip*/ false,
    stream,
    mr);
}

std::pair<std::unique_ptr<cudf::column>, bool> try_remove_quotes_for_floats(
  cudf::column_view const& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  auto const string_count = input.size();
  if (string_count == 0) { return {nullptr, false}; }

  auto const input_sv = cudf::strings_column_view{input};
  auto const input_offsets_it =
    cudf::detail::offsetalator_factory::make_input_iterator(input_sv.offsets());
  auto const d_input_ptr = cudf::column_device_view::create(input, stream);
  auto const is_valid_it = cudf::detail::make_validity_iterator<true>(*d_input_ptr);

  auto string_pairs = rmm::device_uvector<string_index_pair>(string_count, stream);
  thrust::tabulate(rmm::exec_policy_nosync(stream),
                   string_pairs.begin(),
                   string_pairs.end(),
                   [chars    = input_sv.chars_begin(stream),
                    offsets  = input_offsets_it,
                    is_valid = is_valid_it] __device__(cudf::size_type idx) -> string_index_pair {
                     if (!is_valid[idx]) { return {nullptr, 0}; }

                     auto const start_offset = offsets[idx];
                     auto const end_offset   = offsets[idx + 1];
                     auto const size         = end_offset - start_offset;
                     auto const str          = chars + start_offset;

                     // Need to check for size, since the input string may contain just a single
                     // character `"`. Such input should not be considered as quoted.
                     auto const is_quoted = size > 1 && str[0] == '"' && str[size - 1] == '"';

                     // We check and remove quotes only for the special cases (non-numeric numbers
                     // wrapped in double quotes) that are accepted in `from_json`.
                     // They are "NaN", "+INF", "-INF", "+Infinity", "Infinity", "-Infinity".
                     if (is_quoted) {
                       // "NaN"
                       auto accepted = size == 5 && str[1] == 'N' && str[2] == 'a' && str[3] == 'N';

                       // "+INF" and "-INF"
                       accepted = accepted || (size == 6 && (str[1] == '+' || str[1] == '-') &&
                                               str[2] == 'I' && str[3] == 'N' && str[4] == 'F');

                       // "Infinity"
                       accepted = accepted || (size == 10 && str[1] == 'I' && str[2] == 'n' &&
                                               str[3] == 'f' && str[4] == 'i' && str[5] == 'n' &&
                                               str[6] == 'i' && str[7] == 't' && str[8] == 'y');

                       // "+Infinity" and "-Infinity"
                       accepted = accepted || (size == 11 && (str[1] == '+' || str[1] == '-') &&
                                               str[2] == 'I' && str[3] == 'n' && str[4] == 'f' &&
                                               str[5] == 'i' && str[6] == 'n' && str[7] == 'i' &&
                                               str[8] == 't' && str[9] == 'y');

                       if (accepted) { return {str + 1, size - 2}; }
                     }

                     return {str, size};
                   });

  auto const size_it = spark_rapids_jni::util::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<cudf::size_type>(
      [string_pairs = string_pairs.begin()] __device__(cudf::size_type idx) -> cudf::size_type {
        return string_pairs[idx].second;
      }));
  auto [offsets_column, bytes] =
    cudf::strings::detail::make_offsets_child_column(size_it, size_it + string_count, stream, mr);

  // If the output has the same total bytes, the output should be the same as the input.
  if (bytes == input_sv.chars_size(stream)) { return {nullptr, false}; }

  auto chars_data = cudf::strings::detail::make_chars_buffer(
    offsets_column->view(), bytes, string_pairs.begin(), string_count, stream, mr);

  return {cudf::make_strings_column(string_count,
                                    std::move(offsets_column),
                                    chars_data.release(),
                                    input.null_count(),
                                    cudf::copy_bitmask(input, stream, mr)),
          true};
}

std::unique_ptr<cudf::column> cast_strings_to_floats(cudf::column_view const& input,
                                                     cudf::data_type output_type,
                                                     bool allow_nonnumeric_numbers,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  auto const string_count = input.size();
  if (string_count == 0) { return cudf::make_empty_column(output_type); }

  if (allow_nonnumeric_numbers) {
    // Non-numeric numbers are always quoted.
    auto const [removed_quotes, success] = try_remove_quotes_for_floats(input, stream, mr);
    return spark_rapids_jni::string_to_float(
      output_type,
      cudf::strings_column_view{success ? removed_quotes->view() : input},
      /*ansi_mode*/ false,
      stream,
      mr);
  }
  return spark_rapids_jni::string_to_float(
    output_type, cudf::strings_column_view{input}, /*ansi_mode*/ false, stream, mr);
}

// TODO there is a bug here around 0 https://github.com/NVIDIA/spark-rapids/issues/10898
std::unique_ptr<cudf::column> cast_strings_to_decimals(cudf::column_view const& input,
                                                       cudf::data_type output_type,
                                                       int precision,
                                                       bool is_us_locale,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  auto const string_count = input.size();
  if (string_count == 0) { return cudf::make_empty_column(output_type); }

  CUDF_EXPECTS(is_us_locale, "String to decimal conversion is only supported in US locale.");

  auto const input_sv = cudf::strings_column_view{input};
  auto const in_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input_sv.offsets());

  // Count the number of characters `"`.
  rmm::device_uvector<int8_t> quote_counts(string_count, stream);
  // Count the number of characters `"` and `,` in each string.
  rmm::device_uvector<int8_t> remove_counts(string_count, stream);

  {
    using count_type    = cuda::std::tuple<int8_t, int8_t>;
    auto const check_it = spark_rapids_jni::util::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<count_type>(
        [chars = input_sv.chars_begin(stream)] __device__(auto idx) {
          auto const c             = chars[idx];
          auto const is_quote      = c == '"';
          auto const should_remove = is_quote || c == ',';
          return count_type{static_cast<int8_t>(is_quote), static_cast<int8_t>(should_remove)};
        }));
    auto const plus_op =
      cuda::proclaim_return_type<count_type>([] __device__(count_type lhs, count_type rhs) {
        return count_type{cuda::std::get<0>(lhs) + cuda::std::get<0>(rhs),
                          cuda::std::get<1>(lhs) + cuda::std::get<1>(rhs)};
      });

    auto const out_count_it =
      thrust::make_zip_iterator(quote_counts.begin(), remove_counts.begin());

    std::size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::Reduce(nullptr,
                                       temp_storage_bytes,
                                       check_it,
                                       out_count_it,
                                       string_count,
                                       in_offsets,
                                       in_offsets + 1,
                                       plus_op,
                                       count_type{0, 0},
                                       stream.value());
    auto d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
    cub::DeviceSegmentedReduce::Reduce(d_temp_storage.data(),
                                       temp_storage_bytes,
                                       check_it,
                                       out_count_it,
                                       string_count,
                                       in_offsets,
                                       in_offsets + 1,
                                       plus_op,
                                       count_type{0, 0},
                                       stream.value());
  }

  auto const out_size_it = spark_rapids_jni::util::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<cudf::size_type>(
      [offsets       = in_offsets,
       quote_counts  = quote_counts.begin(),
       remove_counts = remove_counts.begin()] __device__(auto idx) {
        auto const input_size = offsets[idx + 1] - offsets[idx];
        // If the current row is non-quoted, just return the original string.
        // As such, non-quoted string containing `,` character will not be preprocessed.
        if (quote_counts[idx] == 0) { return static_cast<cudf::size_type>(input_size); }

        // For quoted strings, we will modify them, removing characters '"' and ','.
        return static_cast<cudf::size_type>(input_size - remove_counts[idx]);
      }));
  auto [offsets_column, bytes] = cudf::strings::detail::make_offsets_child_column(
    out_size_it, out_size_it + string_count, stream, mr);

  // If the output strings column does not change in its total bytes, we can use the input directly.
  if (bytes == input_sv.chars_size(stream)) {
    return spark_rapids_jni::string_to_decimal(precision,
                                               output_type.scale(),
                                               input_sv,
                                               /*ansi_mode*/ false,
                                               /*strip*/ false,
                                               stream,
                                               mr);
  }

  auto const out_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(offsets_column->view());
  auto chars_data = rmm::device_uvector<char>(bytes, stream, mr);

  // Since the strings store decimal numbers, they should not be very long.
  // As such, using one thread per string should be fine.
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(string_count),
                   [in_offsets,
                    out_offsets,
                    input  = input_sv.chars_begin(stream),
                    output = chars_data.begin()] __device__(auto idx) {
                     auto const in_size  = in_offsets[idx + 1] - in_offsets[idx];
                     auto const out_size = out_offsets[idx + 1] - out_offsets[idx];
                     if (in_size == 0) { return; }

                     // If the output size is not changed, we are returning the original unquoted
                     // string. Such string may still contain other alphabet characters, but that
                     // should be handled in the conversion function later on.
                     if (in_size == out_size) {
                       memcpy(output + out_offsets[idx], input + in_offsets[idx], in_size);
                     } else {  // copy byte by byte, ignoring '"' and ',' characters.
                       auto in_ptr  = input + in_offsets[idx];
                       auto in_end  = input + in_offsets[idx + 1];
                       auto out_ptr = output + out_offsets[idx];
                       while (in_ptr != in_end) {
                         if (*in_ptr != '"' && *in_ptr != ',') {
                           *out_ptr = *in_ptr;
                           ++out_ptr;
                         }
                         ++in_ptr;
                       }
                     }
                   });

  // Don't care about the null mask, as nulls imply empty strings, which will also result in nulls.
  auto const unquoted_strings =
    cudf::make_strings_column(string_count, std::move(offsets_column), chars_data.release(), 0, {});

  return spark_rapids_jni::string_to_decimal(precision,
                                             output_type.scale(),
                                             cudf::strings_column_view{unquoted_strings->view()},
                                             /*ansi_mode*/ false,
                                             /*strip*/ false,
                                             stream,
                                             mr);
}

std::pair<std::unique_ptr<cudf::column>, bool> try_remove_quotes(
  cudf::strings_column_view const& input,
  bool nullify_if_not_quoted,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  auto const string_count = input.size();
  if (string_count == 0) { return {nullptr, false}; }

  auto const input_offsets_it =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets());
  auto const d_input_ptr = cudf::column_device_view::create(input.parent(), stream);
  auto const is_valid_it = cudf::detail::make_validity_iterator<true>(*d_input_ptr);

  auto string_pairs = rmm::device_uvector<string_index_pair>(string_count, stream);
  thrust::tabulate(rmm::exec_policy_nosync(stream),
                   string_pairs.begin(),
                   string_pairs.end(),
                   [nullify_if_not_quoted,
                    chars    = input.chars_begin(stream),
                    offsets  = input_offsets_it,
                    is_valid = is_valid_it] __device__(cudf::size_type idx) -> string_index_pair {
                     if (!is_valid[idx]) { return {nullptr, 0}; }

                     auto const start_offset = offsets[idx];
                     auto const end_offset   = offsets[idx + 1];
                     auto const size         = end_offset - start_offset;
                     auto const str          = chars + start_offset;

                     // Need to check for size, since the input string may contain just a single
                     // character `"`. Such input should not be considered as quoted.
                     auto const is_quoted = size > 1 && str[0] == '"' && str[size - 1] == '"';
                     if (nullify_if_not_quoted && !is_quoted) { return {nullptr, 0}; }

                     if (is_quoted) { return {chars + start_offset + 1, size - 2}; }
                     return {chars + start_offset, size};
                   });

  auto const size_it = spark_rapids_jni::util::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<cudf::size_type>(
      [string_pairs = string_pairs.begin()] __device__(cudf::size_type idx) -> cudf::size_type {
        return string_pairs[idx].second;
      }));
  auto [offsets_column, bytes] =
    cudf::strings::detail::make_offsets_child_column(size_it, size_it + string_count, stream, mr);

  // If the output has the same total bytes, the output should be the same as the input.
  if (bytes == input.chars_size(stream)) { return {nullptr, false}; }

  auto chars_data = cudf::strings::detail::make_chars_buffer(
    offsets_column->view(), bytes, string_pairs.begin(), string_count, stream, mr);

  if (nullify_if_not_quoted) {
    auto output = cudf::make_strings_column(string_count,
                                            std::move(offsets_column),
                                            chars_data.release(),
                                            0,
                                            rmm::device_buffer{0, stream, mr});

    auto [null_mask, null_count] = cudf::detail::valid_if(
      string_pairs.begin(),
      string_pairs.end(),
      [] __device__(string_index_pair const& pair) { return pair.first != nullptr; },
      stream,
      mr);
    if (null_count > 0) { output->set_null_mask(std::move(null_mask), null_count); }

    return {std::move(output), true};
  }

  return {cudf::make_strings_column(string_count,
                                    std::move(offsets_column),
                                    chars_data.release(),
                                    input.null_count(),
                                    cudf::copy_bitmask(input.parent(), stream, mr)),
          true};
}

template <typename InputType>
std::unique_ptr<cudf::column> convert_data_type(InputType&& input,
                                                schema_element_with_precision const& schema,
                                                bool allow_nonnumeric_numbers,
                                                bool is_us_locale,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  using DecayInputT                  = std::decay_t<InputType>;
  auto constexpr input_is_const_cv   = std::is_same_v<DecayInputT, cudf::column_view>;
  auto constexpr input_is_column_ptr = std::is_same_v<DecayInputT, std::unique_ptr<cudf::column>>;
  static_assert(input_is_const_cv ^ input_is_column_ptr,
                "Input to `convert_data_type` must either be `cudf::column_view const&` or "
                "`std::unique_ptr<cudf::column>`");

  auto const [d_type, num_rows] = [&]() -> std::pair<cudf::type_id, cudf::size_type> {
    if constexpr (input_is_column_ptr) {
      return {input->type().id(), input->size()};
    } else {
      return {input.type().id(), input.size()};
    }
  }();

  if (d_type == cudf::type_id::STRING) {
    if (cudf::is_chrono(schema.type)) {
      // Date/time is not processed here - it should be handled separately in spark-rapids.
      if constexpr (input_is_column_ptr) {
        return std::move(input);
      } else {
        CUDF_FAIL("Cannot convert data type to a chrono (date/time) type.");
        return nullptr;
      }
    }

    if (schema.type.id() == cudf::type_id::BOOL8) {
      if constexpr (input_is_column_ptr) {
        return cast_strings_to_booleans(input->view(), stream, mr);
      } else {
        return cast_strings_to_booleans(input, stream, mr);
      }
    }

    if (cudf::is_integral(schema.type)) {
      if constexpr (input_is_column_ptr) {
        return cast_strings_to_integers(input->view(), schema.type, stream, mr);
      } else {
        return cast_strings_to_integers(input, schema.type, stream, mr);
      }
    }

    if (cudf::is_floating_point(schema.type)) {
      if constexpr (input_is_column_ptr) {
        return cast_strings_to_floats(
          input->view(), schema.type, allow_nonnumeric_numbers, stream, mr);
      } else {
        return cast_strings_to_floats(input, schema.type, allow_nonnumeric_numbers, stream, mr);
      }
    }

    if (cudf::is_fixed_point(schema.type)) {
      if constexpr (input_is_column_ptr) {
        return cast_strings_to_decimals(
          input->view(), schema.type, schema.precision, is_us_locale, stream, mr);
      } else {
        return cast_strings_to_decimals(
          input, schema.type, schema.precision, is_us_locale, stream, mr);
      }
    }

    if (schema.type.id() == cudf::type_id::STRING) {
      if constexpr (input_is_column_ptr) {
        auto [removed_quotes, success] =
          try_remove_quotes(input->view(), /*nullify_if_not_quoted*/ false, stream, mr);
        return std::move(success ? removed_quotes : input);
      } else {
        auto [removed_quotes, success] =
          try_remove_quotes(input, /*nullify_if_not_quoted*/ false, stream, mr);
        return success ? std::move(removed_quotes)
                       : std::make_unique<cudf::column>(input, stream, mr);
      }
    }

    CUDF_FAIL("Unexpected column type for conversion.");
    return nullptr;
  }  // d_type == cudf::type_id::STRING

  // From here, the input column should have type either LIST or STRUCT.

  CUDF_EXPECTS(schema.type.id() == d_type, "Mismatched data type for nested columns.");

  if constexpr (input_is_column_ptr) {
    auto const null_count   = input->null_count();
    auto const num_children = input->num_children();
    auto input_content      = input->release();

    if (schema.type.id() == cudf::type_id::LIST) {
      auto const& child_schema = schema.child_types.front().second;
      auto& child = input_content.children[cudf::lists_column_view::child_column_index];

      if (cudf::is_nested(child_schema.type)) {
        CUDF_EXPECTS(child_schema.type.id() == child->type().id(),
                     "Mismatched data type for nested child column of a lists column.");
      }

      std::vector<std::unique_ptr<cudf::column>> new_children;
      new_children.emplace_back(
        std::move(input_content.children[cudf::lists_column_view::offsets_column_index]));
      new_children.emplace_back(convert_data_type(
        std::move(child), child_schema, allow_nonnumeric_numbers, is_us_locale, stream, mr));

      return make_lists_column_with_null_sanitization(
        num_rows,
        std::move(new_children[cudf::lists_column_view::offsets_column_index]),
        std::move(new_children[cudf::lists_column_view::child_column_index]),
        null_count,
        std::move(*input_content.null_mask),
        stream,
        mr);
    }

    if (schema.type.id() == cudf::type_id::STRUCT) {
      std::vector<std::unique_ptr<cudf::column>> new_children;
      new_children.reserve(num_children);
      for (cudf::size_type i = 0; i < num_children; ++i) {
        new_children.emplace_back(convert_data_type(std::move(input_content.children[i]),
                                                    schema.child_types[i].second,
                                                    allow_nonnumeric_numbers,
                                                    is_us_locale,
                                                    stream,
                                                    mr));
      }

      return make_structs_column_with_null_consistency(num_rows,
                                                       std::move(new_children),
                                                       null_count,
                                                       std::move(*input_content.null_mask),
                                                       stream,
                                                       mr);
    }
  } else {  // input_is_const_cv
    auto const null_count   = input.null_count();
    auto const num_children = input.num_children();

    if (schema.type.id() == cudf::type_id::LIST) {
      auto const& child_schema = schema.child_types.front().second;
      auto const child         = input.child(cudf::lists_column_view::child_column_index);

      if (cudf::is_nested(child_schema.type)) {
        CUDF_EXPECTS(child_schema.type.id() == child.type().id(),
                     "Mismatched data type for nested child column of a lists column.");
      }

      std::vector<std::unique_ptr<cudf::column>> new_children;
      new_children.emplace_back(
        std::make_unique<cudf::column>(input.child(cudf::lists_column_view::offsets_column_index)));
      new_children.emplace_back(
        convert_data_type(child, child_schema, allow_nonnumeric_numbers, is_us_locale, stream, mr));

      return make_lists_column_with_null_sanitization(
        num_rows,
        std::move(new_children[cudf::lists_column_view::offsets_column_index]),
        std::move(new_children[cudf::lists_column_view::child_column_index]),
        null_count,
        cudf::copy_bitmask(input, stream, mr),
        stream,
        mr);
    }

    if (schema.type.id() == cudf::type_id::STRUCT) {
      std::vector<std::unique_ptr<cudf::column>> new_children;
      new_children.reserve(num_children);
      for (cudf::size_type i = 0; i < num_children; ++i) {
        new_children.emplace_back(convert_data_type(input.child(i),
                                                    schema.child_types[i].second,
                                                    allow_nonnumeric_numbers,
                                                    is_us_locale,
                                                    stream,
                                                    mr));
      }

      return make_structs_column_with_null_consistency(num_rows,
                                                       std::move(new_children),
                                                       null_count,
                                                       cudf::copy_bitmask(input, stream, mr),
                                                       stream,
                                                       mr);
    }
  }

  CUDF_FAIL("Unexpected column type for conversion.");
  return nullptr;
}

std::unique_ptr<cudf::column> from_json_to_structs(cudf::strings_column_view const& input,
                                                   std::vector<std::string> const& col_names,
                                                   std::vector<int> const& num_children,
                                                   std::vector<int> const& types,
                                                   std::vector<int> const& scales,
                                                   std::vector<int> const& precisions,
                                                   bool normalize_single_quotes,
                                                   bool allow_leading_zeros,
                                                   bool allow_nonnumeric_numbers,
                                                   bool allow_unquoted_control,
                                                   bool is_us_locale,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  auto const [concat_input, delimiter, should_be_nullified] =
    concat_json(input, false, stream, cudf::get_current_device_resource_ref());
  auto const [schema, schema_with_precision] =
    generate_struct_schema(col_names, num_children, types, scales, precisions);

  auto opts_builder =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::device_span<std::byte const>{
        static_cast<std::byte const*>(concat_input->data()), concat_input->size()}})
      // fixed options
      .lines(true)
      .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
      .normalize_whitespace(true)
      .mixed_types_as_string(true)
      .keep_quotes(true)
      .experimental(true)
      .strict_validation(true)
      // specifying parameters
      .normalize_single_quotes(normalize_single_quotes)
      .delimiter(delimiter)
      .numeric_leading_zeros(allow_leading_zeros)
      .nonnumeric_numbers(allow_nonnumeric_numbers)
      .unquoted_control_chars(allow_unquoted_control)
      .dtypes(schema)
      .prune_columns(schema.child_types.size() != 0);

  auto parsed_result = cudf::io::read_json_with_row_diagnostics(opts_builder.build(), stream, mr);
  auto const& parsed_meta = parsed_result.data.metadata;
  auto parsed_columns     = parsed_result.data.tbl->release();

  CUDF_EXPECTS(parsed_columns.size() == schema.child_types.size(),
               "Numbers of output columns is different from schema size.");

  auto const& mismatch_diagnostics =
    parsed_result.diagnostics.top_level_columns_with_schema_mismatch_rows;
  std::unordered_map<std::string, std::vector<cudf::size_type> const*> mismatch_rows_by_column;
  mismatch_rows_by_column.reserve(mismatch_diagnostics.size());
  for (auto const& mismatch : mismatch_diagnostics) {
    mismatch_rows_by_column.emplace(mismatch.column_name, &mismatch.row_indices);
  }

  std::vector<std::unique_ptr<cudf::column>> converted_cols;
  converted_cols.reserve(parsed_columns.size());
  for (std::size_t i = 0; i < parsed_columns.size(); ++i) {
    auto const d_type = parsed_columns[i]->type().id();
    CUDF_EXPECTS(d_type == cudf::type_id::LIST || d_type == cudf::type_id::STRUCT ||
                   d_type == cudf::type_id::STRING,
                 "Parsed JSON columns should be STRING or nested.");

    auto const& [col_name, col_schema] = schema_with_precision.child_types[i];
    CUDF_EXPECTS(parsed_meta.schema_info[i].name == col_name, "Mismatched column name.");
    auto const mismatch_rows = mismatch_rows_by_column.find(col_name);
    if (mismatch_rows != mismatch_rows_by_column.end()) {
      nullify_rows(*parsed_columns[i], *mismatch_rows->second, stream, mr);
    }
    converted_cols.emplace_back(convert_data_type(std::move(parsed_columns[i]),
                                                  col_schema,
                                                  allow_nonnumeric_numbers,
                                                  is_us_locale,
                                                  stream,
                                                  mr));
  }

  auto const valid_it          = should_be_nullified->view().begin<bool>();
  auto [null_mask, null_count] = cudf::detail::valid_if(
    valid_it, valid_it + should_be_nullified->size(), thrust::logical_not<bool>{}, stream, mr);

  return make_structs_column_with_null_consistency(
    input.size(),
    std::move(converted_cols),
    null_count,
    null_count > 0 ? std::move(null_mask) : rmm::device_buffer{0, stream, mr},
    stream,
    mr);
}

}  // namespace

}  // namespace detail

std::unique_ptr<cudf::column> from_json_to_structs(cudf::strings_column_view const& input,
                                                   std::vector<std::string> const& col_names,
                                                   std::vector<int> const& num_children,
                                                   std::vector<int> const& types,
                                                   std::vector<int> const& scales,
                                                   std::vector<int> const& precisions,
                                                   bool normalize_single_quotes,
                                                   bool allow_leading_zeros,
                                                   bool allow_nonnumeric_numbers,
                                                   bool allow_unquoted_control,
                                                   bool is_us_locale,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  return detail::from_json_to_structs(input,
                                      col_names,
                                      num_children,
                                      types,
                                      scales,
                                      precisions,
                                      normalize_single_quotes,
                                      allow_leading_zeros,
                                      allow_nonnumeric_numbers,
                                      allow_unquoted_control,
                                      is_us_locale,
                                      stream,
                                      mr);
}

std::unique_ptr<cudf::column> convert_from_strings(cudf::strings_column_view const& input,
                                                   std::vector<int> const& num_children,
                                                   std::vector<int> const& types,
                                                   std::vector<int> const& scales,
                                                   std::vector<int> const& precisions,
                                                   bool allow_nonnumeric_numbers,
                                                   bool is_us_locale,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  [[maybe_unused]] auto const [schema, schema_with_precision] = detail::generate_struct_schema(
    /*dummy col_names*/ std::vector<std::string>(num_children.size(), std::string{}),
    num_children,
    types,
    scales,
    precisions);
  CUDF_EXPECTS(schema_with_precision.child_types.size() == 1,
               "The input schema to convert must have exactly one column.");

  auto const input_cv = input.parent();
  return detail::convert_data_type(input_cv,
                                   schema_with_precision.child_types.front().second,
                                   allow_nonnumeric_numbers,
                                   is_us_locale,
                                   stream,
                                   mr);
}

std::unique_ptr<cudf::column> remove_quotes(cudf::strings_column_view const& input,
                                            bool nullify_if_not_quoted,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  auto const input_cv = input.parent();
  auto [removed_quotes, success] =
    detail::try_remove_quotes(input_cv, nullify_if_not_quoted, stream, mr);
  return success ? std::move(removed_quotes) : std::make_unique<cudf::column>(input_cv, stream, mr);
}

}  // namespace spark_rapids_jni
