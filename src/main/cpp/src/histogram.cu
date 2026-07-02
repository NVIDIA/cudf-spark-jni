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

#include "histogram.hpp"
#include "utilities/iterator.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/reduction/detail/histogram.hpp>
#include <cudf/sorting.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <cuda/std/functional>
#include <thrust/binary_search.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>

#include <type_traits>

namespace spark_rapids_jni {

namespace {

template <typename ElementIterator, typename ValidityIterator>  //
struct fill_percentile_fn {
  __device__ void operator()(cudf::size_type const idx) const
  {
    auto const histogram_idx = idx / percentages.size();

    // If a histogram has null element, it never has more than one null (as the histogram
    // only stores unique elements) and that null is sorted to stay at the end.
    // We need to ignore null thus we will shift the end point if we see a null.

    auto const start     = offsets[histogram_idx];
    auto const try_end   = offsets[histogram_idx + 1];
    auto const all_valid = sorted_validity[try_end - 1];
    auto const end       = all_valid ? try_end : try_end - 1;

    // If the end point after shifting coincides with the start point, we don't have any
    // other valid element.
    auto const has_all_nulls = start >= end;

    auto const percentage_idx = idx % percentages.size();
    if (percentage_idx == 0) {
      // If the histogram only contains null elements, the output percentile will be null.
      out_validity[histogram_idx] = !has_all_nulls;
    }

    if (has_all_nulls) { return; }

    auto const max_positions = accumulated_counts[end - 1] - 1L;
    auto const percentage    = percentages[percentage_idx];
    auto const position      = static_cast<double>(max_positions) * percentage;
    auto const lower         = static_cast<int64_t>(floor(position));
    auto const higher        = static_cast<int64_t>(ceil(position));

    auto const lower_index   = search_counts(lower + 1, start, end);
    auto const lower_element = sorted_input[lower_index];
    if (higher == lower) {
      output[idx] = lower_element;
      return;
    }

    auto const higher_index   = search_counts(higher + 1, start, end);
    auto const higher_element = sorted_input[higher_index];
    if (higher_element == lower_element) {
      output[idx] = lower_element;
      return;
    }

    // Using `volatile` qualifier to prevent the compiler from combining `lower_part` and
    // `upper_part` which may lead to output with different round-off error.
    volatile double const lower_part =
      (static_cast<double>(higher) - position) * static_cast<double>(lower_element);
    volatile double const higher_part =
      (position - static_cast<double>(lower)) * static_cast<double>(higher_element);
    output[idx] = lower_part + higher_part;
  }

  fill_percentile_fn(cudf::size_type const* const offsets_,
                     ElementIterator const sorted_input_,
                     ValidityIterator const sorted_validity_,
                     cudf::device_span<int64_t const> const accumulated_counts_,
                     cudf::device_span<double const> const percentages_,
                     double* const output_,
                     bool* const out_validity_)
    : offsets{offsets_},
      sorted_input{sorted_input_},
      sorted_validity{sorted_validity_},
      accumulated_counts{accumulated_counts_},
      percentages{percentages_},
      output{output_},
      out_validity{out_validity_}
  {
  }

 private:
  __device__ cudf::size_type search_counts(int64_t position,
                                           cudf::size_type start,
                                           cudf::size_type end) const
  {
    auto const it = thrust::lower_bound(
      thrust::seq, accumulated_counts.begin() + start, accumulated_counts.begin() + end, position);
    return static_cast<cudf::size_type>(cuda::std::distance(accumulated_counts.begin(), it));
  }

  cudf::size_type const* const offsets;
  ElementIterator const sorted_input;
  ValidityIterator const sorted_validity;
  cudf::device_span<int64_t const> const accumulated_counts;
  cudf::device_span<double const> const percentages;
  double* const output;
  bool* const out_validity;
};

struct percentile_dispatcher {
  template <typename T>
  static constexpr bool is_supported()
  {
    return std::is_arithmetic_v<T>;
  }

  // The output here is only intermediate result, consisting of:
  //  1. The output percentile values,
  //  2. Null mask to apply for the final output column containing percentile values, and
  //  3. Null count corresponding to that null mask.
  using output_type =
    std::tuple<std::unique_ptr<cudf::column>, rmm::device_buffer, cudf::size_type>;

  template <typename T, typename... Args>
  std::enable_if_t<!is_supported<T>(), output_type> operator()(Args&&...) const
  {
    CUDF_FAIL("Unsupported type in histogram-to-percentile evaluation.");
  }

  template <typename T, CUDF_ENABLE_IF(is_supported<T>())>
  output_type operator()(cudf::size_type const* const offsets,
                         cudf::size_type const* const ordered_indices,
                         cudf::column_device_view const& data,
                         cudf::device_span<int64_t const> accumulated_counts,
                         cudf::device_span<double const> percentages,
                         bool has_null,
                         cudf::size_type num_histograms,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr) const
  {
    // Returns all nulls for totally empty input.
    if (data.size() == 0 || percentages.size() == 0) {
      return {cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                        num_histograms,
                                        cudf::mask_state::UNALLOCATED,
                                        stream,
                                        mr),
              cudf::create_null_mask(num_histograms, cudf::mask_state::ALL_NULL, stream, mr),
              num_histograms};
    }

    auto percentiles =
      cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                num_histograms * static_cast<cudf::size_type>(percentages.size()),
                                cudf::mask_state::UNALLOCATED,
                                stream,
                                mr);

    // We may always have nulls in the output due to either:
    // - Having nulls in the input, and/or,
    // - Having empty histograms.
    auto out_validities =
      rmm::device_uvector<bool>(num_histograms, stream, rmm::mr::get_current_device_resource_ref());

    auto const fill_percentile = [&](auto const sorted_validity_it) {
      auto const sorted_input_it =
        cuda::make_permutation_iterator(data.begin<T>(), ordered_indices);
      thrust::for_each_n(rmm::exec_policy(stream),
                         cuda::make_counting_iterator(0),
                         num_histograms * static_cast<cudf::size_type>(percentages.size()),
                         fill_percentile_fn{offsets,
                                            sorted_input_it,
                                            sorted_validity_it,
                                            accumulated_counts,
                                            percentages,
                                            percentiles->mutable_view().begin<double>(),
                                            out_validities.begin()});
    };

    if (!has_null) {
      fill_percentile(cuda::make_constant_iterator(true));
    } else {
      auto const sorted_validity_it = cuda::make_permutation_iterator(
        cudf::detail::make_validity_iterator<false>(data), ordered_indices);
      fill_percentile(sorted_validity_it);
    }

    auto [null_mask, null_count] =
      cudf::bools_to_mask(cudf::device_span<bool const>(out_validities), stream, mr);
    if (null_count > 0) {
      return {std::move(percentiles), std::move(*null_mask.release()), null_count};
    }

    return {std::move(percentiles), rmm::device_buffer{}, 0};
  }
};

void check_input(cudf::column_view const& input, std::vector<double> const& percentages)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::LIST,
               "The input column must be of type LIST.",
               std::invalid_argument);

  auto const child = input.child(cudf::lists_column_view::child_column_index);
  CUDF_EXPECTS(
    !child.has_nulls(), "Child of the input column must not have nulls.", std::invalid_argument);
  CUDF_EXPECTS(child.type().id() == cudf::type_id::STRUCT && child.num_children() == 2,
               "Child of the input column must be of STRUCT type having two children.",
               std::invalid_argument);
  CUDF_EXPECTS(!child.child(1).has_nulls(),
               "Child of the input column must have its second child containing non-null elements.",
               std::invalid_argument);
  CUDF_EXPECTS(child.child(1).type().id() == cudf::type_id::INT64,
               "Child of the input column must have its second child of type INT64.",
               std::invalid_argument);

  CUDF_EXPECTS(static_cast<std::size_t>(input.size()) * percentages.size() <=
                 static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
               "Size of output exceeds cudf column size limit.",
               std::overflow_error);
}

// Wrap the input column in a lists column, to satisfy the requirement type in Spark.
std::unique_ptr<cudf::column> wrap_in_list(std::unique_ptr<cudf::column>&& input,
                                           rmm::device_buffer&& null_mask,
                                           cudf::size_type null_count,
                                           cudf::size_type num_histograms,
                                           cudf::size_type num_percentages,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  if (input->size() == 0) { return cudf::make_empty_lists_column(input->type()); }

  auto const sizes_itr = cuda::make_constant_iterator(num_percentages);
  auto offsets         = std::get<0>(
    cudf::detail::make_offsets_child_column(sizes_itr, sizes_itr + num_histograms, stream, mr));
  auto output = cudf::make_lists_column(
    num_histograms, std::move(offsets), std::move(input), null_count, std::move(null_mask));
  if (null_count > 0) { return cudf::purge_nonempty_nulls(output->view(), stream, mr); }

  return output;
}

}  // namespace

std::unique_ptr<cudf::column> create_histogram_if_valid(cudf::column_view const& values,
                                                        cudf::column_view const& frequencies,
                                                        bool output_as_lists,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    !frequencies.has_nulls(), "The input frequencies must not have nulls.", std::invalid_argument);
  CUDF_EXPECTS(frequencies.type().id() == cudf::type_id::INT64,
               "The input frequencies must be of type INT64.",
               std::invalid_argument);
  CUDF_EXPECTS(values.size() == frequencies.size(),
               "The input values and frequencies must have the same size.",
               std::invalid_argument);

  if (values.size() == 0) {
    if (output_as_lists) {
      return cudf::make_lists_column(0,
                                     cudf::make_empty_column(cudf::type_to_id<cudf::size_type>()),
                                     cudf::reduction::detail::make_empty_histogram_like(values),
                                     0,
                                     {});
    } else {
      return cudf::reduction::detail::make_empty_histogram_like(values);
    }
  }

  auto const default_mr = rmm::mr::get_current_device_resource_ref();

  // We only check if there is any row in frequencies that are negative (invalid) or zero.
  auto check_invalid_and_zero =
    cudf::detail::make_zeroed_device_uvector_async<int8_t>(2, stream, default_mr);

  // We need to check and remember which rows are valid (positive) so we can do filtering later on.
  auto check_valid = rmm::device_uvector<bool>(frequencies.size(), stream, default_mr);

  thrust::for_each_n(rmm::exec_policy(stream),
                     cuda::make_counting_iterator(0),
                     frequencies.size(),
                     [frequencies   = frequencies.begin<int64_t>(),
                      check_invalid = check_invalid_and_zero.begin(),
                      check_zero    = check_invalid_and_zero.begin() + 1,
                      check_valid   = check_valid.begin()] __device__(auto const idx) {
                       if (frequencies[idx] < 0) { *check_invalid = 1; }
                       if (frequencies[idx] == 0) { *check_zero = 1; }

                       check_valid[idx] = static_cast<int8_t>(frequencies[idx] > 0);
                     });

  auto const h_checks = cudf::detail::make_std_vector(check_invalid_and_zero, stream);
  CUDF_EXPECTS(!h_checks.front(),  // check invalid (negative) frequencies
               "The input frequencies must not contain negative values.",
               std::invalid_argument);

  auto const make_structs_histogram = [&](rmm::device_buffer&& null_mask,
                                          cudf::size_type null_count) {
    // Copy values and frequencies into a new structs column.
    std::vector<std::unique_ptr<cudf::column>> values_and_frequencies;
    values_and_frequencies.emplace_back(std::make_unique<cudf::column>(values, stream, mr));
    values_and_frequencies.emplace_back(std::make_unique<cudf::column>(frequencies, stream, mr));

    // Set null mask for the returned values.
    // Such null mask is resulted from checking frequencies > 0.
    if (null_count > 0) {
      if (!values.has_nulls()) {
        values_and_frequencies.front()->set_null_mask(std::move(null_mask), null_count);
      } else {
        // We need to AND the current null mask with the given null mask.
        auto [new_null_mask, new_null_count] = cudf::bitmask_and(
          std::vector<cudf::bitmask_type const*>{
            // Don't use values.null_mask(), to make sure no slicing.
            values_and_frequencies.front()->view().null_mask(),
            reinterpret_cast<cudf::bitmask_type const*>(null_mask.data())},
          std::vector<cudf::size_type>{0, 0},
          values.size(),
          stream,
          mr);
        values_and_frequencies.front()->set_null_mask(std::move(new_null_mask), new_null_count);
      }

      // Nulls will be eventually excluded from percentile computation.
      // However, having frequencies containing zero (corresponding to these nulls) will crash
      // cudf MERGE_HISTOGRAM aggregation.
      // Therefore, we manually set `1` for the frequencies of nulls.
      thrust::for_each_n(
        rmm::exec_policy(stream),
        cuda::make_counting_iterator(0),
        frequencies.size(),
        [frequencies = values_and_frequencies.back()->mutable_view().begin<int64_t>(),
         null_mask =
           values_and_frequencies.front()->view().null_mask()] __device__(auto const idx) {
          // If this is a null, set 1 frequency.
          if (!cudf::bit_is_set(null_mask, idx)) { frequencies[idx] = int64_t{1}; }
        });
    }

    return cudf::make_structs_column(
      values.size(), std::move(values_and_frequencies), 0, rmm::device_buffer{}, stream, mr);
  };

  auto const make_lists_histograms = [&](cudf::size_type num_elements,
                                         std::unique_ptr<cudf::column>&& structs_histogram) {
    // Each output list will have size 1.
    auto const sizes_itr = cuda::make_constant_iterator(1);
    auto offsets         = std::get<0>(
      cudf::detail::make_offsets_child_column(sizes_itr, sizes_itr + num_elements, stream, mr));
    return cudf::make_lists_column(
      num_elements, std::move(offsets), std::move(structs_histogram), 0, rmm::device_buffer{});
  };

  if (output_as_lists) {
    auto child            = make_structs_histogram(rmm::device_buffer{}, 0);
    auto lists_histograms = make_lists_histograms(values.size(), std::move(child));

    if (!h_checks.back()) {  // all frequencies are positive
      return lists_histograms;
    }

    // Frequencies contain zero.
    // We generate a null mask in which nulls correspond to the zero row in frequencies.
    // Then, apply it to the output lists column, empty out the null lists, and finally remove
    // the null mask.
    // By doing so, the input rows corresponding to zero frequencies will be output as empty lists.
    auto [null_mask, null_count] =
      cudf::bools_to_mask(cudf::device_span<bool const>(check_valid), stream, default_mr);
    lists_histograms->set_null_mask(std::move(*null_mask.release()), null_count);
    lists_histograms = cudf::purge_nonempty_nulls(lists_histograms->view(), stream, mr);
    lists_histograms->set_null_mask(rmm::device_buffer{}, 0);
    return lists_histograms;
  } else {                   // output_as_lists==false
    if (!h_checks.back()) {  // all frequencies are positive
      return make_structs_histogram(rmm::device_buffer{}, 0);
    }

    // We nullify the values corresponding to zero frequencies.
    auto [null_mask, null_count] =
      cudf::bools_to_mask(cudf::device_span<bool const>(check_valid), stream, mr);
    return make_structs_histogram(std::move(*null_mask.release()), null_count);
  }
}

std::unique_ptr<cudf::column> percentile_from_histogram(cudf::column_view const& input,
                                                        std::vector<double> const& percentages,
                                                        bool output_as_list,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  check_input(input, percentages);

  auto const lcv_histograms = cudf::lists_column_view{input};
  auto const histograms     = lcv_histograms.get_sliced_child(stream);
  auto const data_col       = cudf::structs_column_view{histograms}.get_sliced_child(0);
  auto const counts_col     = cudf::structs_column_view{histograms}.get_sliced_child(1);

  auto const default_mr    = rmm::mr::get_current_device_resource_ref();
  auto const d_data        = cudf::column_device_view::create(data_col, stream);
  auto const d_percentages = cudf::detail::make_device_uvector(percentages, stream, default_mr);

  // Attach histogram labels to the input.
  auto histogram_labels =
    rmm::device_uvector<cudf::size_type>(histograms.size(), stream, default_mr);
  cudf::detail::label_segments(lcv_histograms.offsets_begin(),
                               lcv_histograms.offsets_end(),
                               histogram_labels.begin(),
                               histogram_labels.end(),
                               stream);
  auto const labels_cv = cudf::column_view{cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                           static_cast<cudf::size_type>(histogram_labels.size()),
                                           histogram_labels.data(),
                                           nullptr,
                                           0};
  auto const labeled_histograms = cudf::table_view{{labels_cv, histograms}};
  // Find the order of segmented sort elements within each histogram list.
  // The null order must be `AFTER`.
  auto const ordered_indices = cudf::sorted_order(
    labeled_histograms,
    std::vector<cudf::order>{cudf::order::ASCENDING, cudf::order::ASCENDING},
    std::vector<cudf::null_order>{cudf::null_order::AFTER, cudf::null_order::AFTER},
    stream,
    default_mr);

  auto const d_accumulated_counts = [&] {
    auto const sorted_counts = cuda::make_permutation_iterator(
      counts_col.begin<int64_t>(), ordered_indices->view().begin<cudf::size_type>());
    auto accumulated_counts = rmm::device_uvector<int64_t>(counts_col.size(), stream, default_mr);
    // We don't need a permutation iterator for the labels, since the same labels always
    // stay together after sorting.
    thrust::inclusive_scan_by_key(rmm::exec_policy(stream),
                                  histogram_labels.begin(),
                                  histogram_labels.end(),
                                  sorted_counts,
                                  accumulated_counts.begin());
    return accumulated_counts;
  }();

  auto [percentiles, null_mask, null_count] =
    type_dispatcher(data_col.type(),
                    percentile_dispatcher{},
                    lcv_histograms.offsets_begin(),
                    ordered_indices->view().begin<cudf::size_type>(),
                    *d_data,
                    d_accumulated_counts,
                    d_percentages,
                    data_col.has_nulls(),
                    input.size(),
                    stream,
                    mr);

  if (output_as_list) {
    return wrap_in_list(std::move(percentiles),
                        std::move(null_mask),
                        null_count,
                        lcv_histograms.size(),
                        static_cast<cudf::size_type>(percentages.size()),
                        stream,
                        mr);
  }
  percentiles->set_null_mask(std::move(null_mask), null_count);
  return std::move(percentiles);
}

}  // namespace spark_rapids_jni
