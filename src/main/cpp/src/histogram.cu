/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/reduction/detail/histogram.hpp>
#include <cudf/sorting.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/atomic>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/stream>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>

#include <limits>
#include <stdexcept>
#include <type_traits>

namespace spark_rapids_jni {

namespace {

enum class percentile_status : int32_t { VALID = 0, EMPTY, INVALID_RANK };

template <percentile_interpolation Interpolation,
          typename ElementIterator,
          typename ValidityIterator>  //
struct fill_percentile_fn {
  __device__ void operator()(cudf::size_type const idx) const
  {
    auto const histogram_idx = idx / percentages.size();

    // If a histogram has null element, it never has more than one null (as the histogram
    // only stores unique elements) and that null is sorted to stay at the end.
    // We need to ignore null thus we will shift the end point if we see a null.

    // The child columns and sorted indices are relative to the sliced input.
    auto const base    = offsets[0];
    auto const start   = offsets[histogram_idx] - base;
    auto const try_end = offsets[histogram_idx + 1] - base;
    // Empty histograms have no last element whose validity can be inspected.
    auto const end = start < try_end && !sorted_validity[try_end - 1] ? try_end - 1 : try_end;

    auto const percentage_idx = idx % percentages.size();
    if (start >= end) {
      if (percentage_idx == 0) { status[histogram_idx] = percentile_status::EMPTY; }
      return;
    }

    // Spark computes positions with wrapping Long arithmetic and a Double
    // multiplication.
    auto const max_position =
      static_cast<int64_t>(static_cast<uint64_t>(accumulated_counts[end - 1]) - 1);
    auto const position     = static_cast<double>(max_position) * percentages[percentage_idx];
    auto const lower        = to_spark_long(floor(position));
    auto const higher       = to_spark_long(ceil(position));
    auto const lower_index  = search_counts(lower, start, end);
    auto const higher_index = lower == higher ? lower_index : search_counts(higher, start, end);
    if (lower_index == end || higher_index == end) {
      // A large frequency can round the last position up to the cumulative
      // count.
      cuda::atomic_ref<percentile_status, cuda::thread_scope_device>{status[histogram_idx]}.store(
        percentile_status::INVALID_RANK, cuda::memory_order_relaxed);
      return;
    }

    auto const lower_element = sorted_input[lower_index];
    if (higher == lower) {
      output[idx] = lower_element;
      return;
    }

    auto const higher_element = sorted_input[higher_index];
    if (higher_element == lower_element) {
      output[idx] = lower_element;
      return;
    }

    if constexpr (Interpolation == percentile_interpolation::WEIGHTED_ENDPOINTS) {
      // Use explicit round-to-nearest operations to match Spark and prevent FMA
      // contraction. Volatile temporaries do not reliably prevent contraction
      // on all CUDA targets.
      auto const lower_part  = __dmul_rn(__dsub_rn(static_cast<double>(higher), position),
                                        static_cast<double>(lower_element));
      auto const higher_part = __dmul_rn(__dsub_rn(position, static_cast<double>(lower)),
                                         static_cast<double>(higher_element));
      output[idx]            = __dadd_rn(lower_part, higher_part);
    } else {
      // Keep the subtraction, multiplication, and addition as separate operations to match
      // Spark's endpoint-delta interpolation and prevent fused multiply-add from changing
      // rounding behavior.
      auto const fraction = __dsub_rn(position, static_cast<double>(lower));
      auto const endpoint_delta =
        __dsub_rn(static_cast<double>(higher_element), static_cast<double>(lower_element));
      auto const scaled_delta = __dmul_rn(fraction, endpoint_delta);
      output[idx]             = __dadd_rn(static_cast<double>(lower_element), scaled_delta);
    }
  }

  fill_percentile_fn(cudf::size_type const* const offsets_,
                     ElementIterator const sorted_input_,
                     ValidityIterator const sorted_validity_,
                     cudf::device_span<int64_t const> const accumulated_counts_,
                     cudf::device_span<double const> const percentages_,
                     double* const output_,
                     percentile_status* const status_)
    : offsets{offsets_},
      sorted_input{sorted_input_},
      sorted_validity{sorted_validity_},
      accumulated_counts{accumulated_counts_},
      percentages{percentages_},
      output{output_},
      status{status_}
  {
  }

 private:
  // JVM narrowing maps NaN to zero and saturates at the limits of Long.
  __device__ static int64_t to_spark_long(double x)
  {
    if (isnan(x)) { return 0; }
    if (x >= 0x1p63) { return cuda::std::numeric_limits<int64_t>::max(); }
    if (x <= -0x1p63) { return cuda::std::numeric_limits<int64_t>::min(); }
    return static_cast<int64_t>(x);
  }

  __device__ cudf::size_type search_counts(int64_t rank,
                                           cudf::size_type start,
                                           cudf::size_type end) const
  {
    auto const target = static_cast<int64_t>(static_cast<uint64_t>(rank) + 1);
    // Match java.util.Arrays.binarySearch, including its choice of midpoint if
    // counts wrap.
    auto high = end - 1;
    while (start <= high) {
      auto const mid   = start + (high - start) / 2;
      auto const count = accumulated_counts[mid];
      if (count == target) { return mid; }
      if (count < target) {
        start = mid + 1;
      } else {
        high = mid - 1;
      }
    }
    return start;
  }

  cudf::size_type const* const offsets;
  ElementIterator const sorted_input;
  ValidityIterator const sorted_validity;
  cudf::device_span<int64_t const> const accumulated_counts;
  cudf::device_span<double const> const percentages;
  double* const output;
  percentile_status* const status;
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
  using output_type = std::
    tuple<std::unique_ptr<cudf::column>, cuda::device_buffer<cudf::bitmask_type>, cudf::size_type>;

  template <typename T, typename... Args>
  std::enable_if_t<!is_supported<T>(), output_type> operator()(Args&&...) const
  {
    CUDF_FAIL("Unsupported type in histogram-to-percentile evaluation.");
  }

  template <typename T>
    requires(is_supported<T>())
  output_type operator()(cudf::size_type const* const offsets,
                         cudf::size_type const* const ordered_indices,
                         cudf::column_device_view const& data,
                         cudf::device_span<int64_t const> accumulated_counts,
                         cudf::device_span<double const> percentages,
                         percentile_interpolation interpolation,
                         bool has_null,
                         cudf::size_type num_histograms,
                         cuda::stream_ref stream,
                         rmm::device_async_resource_ref mr) const
  {
    // Returns all nulls for totally empty input.
    if (data.size() == 0 || percentages.size() == 0) {
      return {
        cudf::make_numeric_column(
          cudf::data_type{cudf::type_id::FLOAT64},
          percentages.empty() ? num_histograms
                              : num_histograms * static_cast<cudf::size_type>(percentages.size()),
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

    auto status = cudf::detail::make_zeroed_device_uvector_async<percentile_status>(
      num_histograms, stream, cudf::get_current_device_resource_ref());

    auto const fill_percentile = [&](auto const sorted_validity_it) {
      auto const sorted_input_it =
        cuda::make_permutation_iterator(data.begin<T>(), ordered_indices);
      auto const launch_fill = [&](auto const interpolation_constant) {
        constexpr auto interpolation_value = std::decay_t<decltype(interpolation_constant)>::value;
        using fill_fn                      = fill_percentile_fn<interpolation_value,
                                                                std::decay_t<decltype(sorted_input_it)>,
                                                                std::decay_t<decltype(sorted_validity_it)>>;
        thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                           cuda::make_counting_iterator(0),
                           num_histograms * static_cast<cudf::size_type>(percentages.size()),
                           fill_fn{offsets,
                                   sorted_input_it,
                                   sorted_validity_it,
                                   accumulated_counts,
                                   percentages,
                                   percentiles->mutable_view().begin<double>(),
                                   status.data()});
      };

      switch (interpolation) {
        case percentile_interpolation::WEIGHTED_ENDPOINTS:
          launch_fill(std::integral_constant<percentile_interpolation,
                                             percentile_interpolation::WEIGHTED_ENDPOINTS>{});
          break;
        case percentile_interpolation::ENDPOINT_DELTA:
          launch_fill(std::integral_constant<percentile_interpolation,
                                             percentile_interpolation::ENDPOINT_DELTA>{});
          break;
      }
    };

    if (!has_null) {
      fill_percentile(cuda::make_constant_iterator(true));
    } else {
      auto const sorted_validity_it = cuda::make_permutation_iterator(
        cudf::detail::make_validity_iterator<false>(data), ordered_indices);
      fill_percentile(sorted_validity_it);
    }

    auto [null_mask, null_count] = cudf::detail::valid_if(
      status.begin(),
      status.end(),
      [] __device__(auto state) { return state == percentile_status::VALID; },
      stream,
      mr);
    if (null_count > 0) {
      auto const invalid =
        thrust::find(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     status.begin(),
                     status.end(),
                     percentile_status::INVALID_RANK);
      CUDF_EXPECTS(invalid == status.end(),
                   "Percentile rank is outside the histogram.",
                   percentile_rank_out_of_range);
      return {std::move(percentiles), std::move(null_mask), null_count};
    }

    return {std::move(percentiles), cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED), 0};
  }
};

// Wrap the input column in a lists column, to satisfy the requirement type in Spark.
std::unique_ptr<cudf::column> wrap_in_list(std::unique_ptr<cudf::column>&& input,
                                           cuda::device_buffer<cudf::bitmask_type>&& null_mask,
                                           cudf::size_type null_count,
                                           cudf::size_type num_histograms,
                                           cudf::size_type num_percentages,
                                           cuda::stream_ref stream,
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
                                                        cuda::stream_ref stream,
                                                        rmm::device_async_resource_ref mr)
{
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
                                     cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
    } else {
      return cudf::reduction::detail::make_empty_histogram_like(values);
    }
  }

  auto const default_mr = rmm::mr::get_current_device_resource_ref();

  // Check for negative frequencies among rows Spark considers, and remember whether any frequency
  // must be sanitized because it is null or non-positive. Null values with positive frequencies
  // need no extra work because the existing value mask already excludes them downstream.
  auto check_invalid_and_sanitize =
    cudf::detail::make_zeroed_device_uvector_async<int8_t>(2, stream, default_mr);

  // Frequencies that are valid and positive can safely flow downstream unchanged.
  auto check_frequency_usable = rmm::device_uvector<bool>(frequencies.size(), stream, default_mr);

  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::make_counting_iterator(0),
    frequencies.size(),
    [frequencies            = frequencies.begin<int64_t>(),
     values_null_mask       = values.null_mask(),
     values_offset          = values.offset(),
     frequencies_null_mask  = frequencies.null_mask(),
     frequencies_offset     = frequencies.offset(),
     check_invalid          = check_invalid_and_sanitize.begin(),
     check_sanitize         = check_invalid_and_sanitize.begin() + 1,
     check_frequency_usable = check_frequency_usable.begin()] __device__(auto const idx) -> void {
      auto const value_is_valid =
        values_null_mask == nullptr || cudf::bit_is_set(values_null_mask, values_offset + idx);
      auto const frequency_is_valid =
        frequencies_null_mask == nullptr ||
        cudf::bit_is_set(frequencies_null_mask, frequencies_offset + idx);
      auto const frequency_is_usable = frequency_is_valid && frequencies[idx] > 0;

      if (value_is_valid && frequency_is_valid && frequencies[idx] < 0) { *check_invalid = 1; }
      if (!frequency_is_usable) { *check_sanitize = 1; }
      check_frequency_usable[idx] = frequency_is_usable;
    });

  auto const h_checks = cudf::detail::make_std_vector(check_invalid_and_sanitize, stream);
  CUDF_EXPECTS(!h_checks.front(),  // check invalid (negative) frequencies
               "The input frequencies must not contain negative values.",
               std::invalid_argument);

  auto const make_structs_histogram = [&](cuda::device_buffer<cudf::bitmask_type>&& null_mask,
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
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        cuda::make_counting_iterator(0),
        frequencies.size(),
        [frequencies = values_and_frequencies.back()->mutable_view().begin<int64_t>(),
         null_mask =
           values_and_frequencies.front()->view().null_mask()] __device__(auto const idx) {
          // If this is a null, set 1 frequency.
          if (!cudf::bit_is_set(null_mask, idx)) { frequencies[idx] = int64_t{1}; }
        });

      // Every null frequency has been replaced by the safe placeholder above.
      values_and_frequencies.back()->set_null_mask(
        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED), 0);
    }

    return cudf::make_structs_column(values.size(),
                                     std::move(values_and_frequencies),
                                     0,
                                     cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                     stream,
                                     mr);
  };

  auto const make_lists_histograms = [&](cudf::size_type num_elements,
                                         std::unique_ptr<cudf::column>&& structs_histogram) {
    // Each output list will have size 1.
    auto const sizes_itr = cuda::make_constant_iterator(1);
    auto offsets         = std::get<0>(
      cudf::detail::make_offsets_child_column(sizes_itr, sizes_itr + num_elements, stream, mr));
    return cudf::make_lists_column(num_elements,
                                   std::move(offsets),
                                   std::move(structs_histogram),
                                   0,
                                   cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
  };

  if (output_as_lists) {
    auto child =
      make_structs_histogram(cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED), 0);
    auto lists_histograms = make_lists_histograms(values.size(), std::move(child));

    if (!h_checks.back()) {  // all frequencies can flow downstream unchanged
      return lists_histograms;
    }

    // Some frequencies must be sanitized.
    // We generate a null mask in which nulls correspond to unusable frequencies.
    // Then, apply it to the output lists column, empty out the null lists, and finally remove
    // the null mask.
    // By doing so, input rows with unusable frequencies will be output as empty lists.
    auto [null_mask, null_count] = cudf::bools_to_mask(
      cudf::device_span<bool const>(check_frequency_usable), stream, default_mr);
    lists_histograms->set_null_mask(std::move(*null_mask), null_count);
    lists_histograms = cudf::purge_nonempty_nulls(lists_histograms->view(), stream, mr);
    lists_histograms->set_null_mask(cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED), 0);
    return lists_histograms;
  } else {                   // output_as_lists==false
    if (!h_checks.back()) {  // all frequencies can flow downstream unchanged
      return make_structs_histogram(cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED), 0);
    }

    // We nullify the values corresponding to unusable frequencies.
    auto [null_mask, null_count] =
      cudf::bools_to_mask(cudf::device_span<bool const>(check_frequency_usable), stream, mr);
    return make_structs_histogram(std::move(*null_mask), null_count);
  }
}

std::unique_ptr<cudf::column> percentile_from_histogram(cudf::column_view const& input,
                                                        std::vector<double> const& percentages,
                                                        bool output_as_list,
                                                        percentile_interpolation interpolation,
                                                        cuda::stream_ref stream,
                                                        rmm::device_async_resource_ref mr)
{
  return percentile_from_histogram(
    input, percentages, output_as_list, interpolation, value_order::ASCENDING, stream, mr);
}

std::unique_ptr<cudf::column> percentile_from_histogram(cudf::column_view const& input,
                                                        std::vector<double> const& percentages,
                                                        bool output_as_list,
                                                        percentile_interpolation interpolation,
                                                        value_order order,
                                                        cuda::stream_ref stream,
                                                        rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::LIST,
               "The input column must be of type LIST.",
               std::invalid_argument);
  CUDF_EXPECTS(static_cast<std::size_t>(input.size()) * percentages.size() <=
                 static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
               "Size of output exceeds cudf column size limit.",
               std::overflow_error);
  CUDF_EXPECTS(interpolation == percentile_interpolation::WEIGHTED_ENDPOINTS ||
                 interpolation == percentile_interpolation::ENDPOINT_DELTA,
               "Unsupported percentile interpolation algorithm.",
               std::invalid_argument);
  CUDF_EXPECTS(order == value_order::ASCENDING || order == value_order::DESCENDING,
               "Unsupported histogram value order.",
               std::invalid_argument);
  auto const is_descending = order == value_order::DESCENDING;

  auto const lcv_histograms = cudf::lists_column_view{input};
  auto const histograms     = lcv_histograms.get_sliced_child(stream);
  CUDF_EXPECTS(!histograms.has_nulls(),
               "Child of the input column must not have nulls.",
               std::invalid_argument);
  CUDF_EXPECTS(histograms.type().id() == cudf::type_id::STRUCT && histograms.num_children() == 2,
               "Child of the input column must be of STRUCT type having two children.",
               std::invalid_argument);
  auto const entries    = cudf::structs_column_view{histograms};
  auto const data_col   = entries.get_sliced_child(0, stream);
  auto const counts_col = entries.get_sliced_child(1, stream);
  CUDF_EXPECTS(!counts_col.has_nulls(),
               "Child of the input column must have its second child containing non-null elements.",
               std::invalid_argument);
  CUDF_EXPECTS(counts_col.type().id() == cudf::type_id::INT64,
               "Child of the input column must have its second child of type INT64.",
               std::invalid_argument);

  auto const default_mr    = rmm::mr::get_current_device_resource_ref();
  auto const d_data        = cudf::column_device_view::create(data_col, stream, default_mr);
  auto const d_percentages = cudf::detail::make_device_uvector(percentages, stream, default_mr);

  // Attach histogram labels to the input.
  auto histogram_labels =
    rmm::device_uvector<cudf::size_type>(histograms.size(), stream, default_mr);
  cudf::detail::label_segments(lcv_histograms.offsets_begin(),
                               lcv_histograms.offsets_end(),
                               histogram_labels.begin(),
                               histogram_labels.end(),
                               stream,
                               default_mr);
  auto const labels_cv = cudf::column_view{cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                           static_cast<cudf::size_type>(histogram_labels.size()),
                                           histogram_labels.data(),
                                           nullptr,
                                           0};
  auto const labeled_histograms = cudf::table_view{{labels_cv, histograms}};
  // Find the order of segmented sort elements within each histogram list.
  // Nulls must physically sort after valid values so they can be excluded below. cuDF reverses
  // null precedence along with value order, so descending values require null_order::BEFORE.
  auto const ordered_indices = cudf::sorted_order(
    labeled_histograms,
    std::vector<cudf::order>{cudf::order::ASCENDING,
                             is_descending ? cudf::order::DESCENDING : cudf::order::ASCENDING},
    std::vector<cudf::null_order>{
      cudf::null_order::AFTER, is_descending ? cudf::null_order::BEFORE : cudf::null_order::AFTER},
    stream,
    default_mr);

  auto const d_accumulated_counts = [&] {
    auto const sorted_counts = cuda::make_permutation_iterator(
      counts_col.begin<int64_t>(), ordered_indices->view().begin<cudf::size_type>());
    auto accumulated_counts = rmm::device_uvector<int64_t>(counts_col.size(), stream, default_mr);
    // We don't need a permutation iterator for the labels, since the same labels always
    // stay together after sorting.
    thrust::inclusive_scan_by_key(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      histogram_labels.begin(),
      histogram_labels.end(),
      sorted_counts,
      accumulated_counts.begin(),
      cuda::std::equal_to<>{},
      cuda::std::plus<uint64_t>{});
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
                    interpolation,
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
  if (null_count > 0 && percentages.size() > 1) {
    // The dispatcher returns one validity bit per histogram. Flat output needs
    // one per value.
    auto const num_percentages             = static_cast<cudf::size_type>(percentages.size());
    auto const begin                       = cuda::make_counting_iterator(cudf::size_type{0});
    auto [flat_null_mask, flat_null_count] = cudf::detail::valid_if(
      begin,
      begin + percentiles->size(),
      [mask = null_mask.data(), num_percentages] __device__(auto const idx) {
        return cudf::bit_is_set(mask, idx / num_percentages);
      },
      stream,
      mr);
    null_mask  = std::move(flat_null_mask);
    null_count = flat_null_count;
  }
  percentiles->set_null_mask(std::move(null_mask), null_count);
  return std::move(percentiles);
}

}  // namespace spark_rapids_jni
