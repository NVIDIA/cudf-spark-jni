/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include "find_in_set.hpp"
#include "nvtx_ranges.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/string_view.cuh>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace spark_rapids_jni {
namespace {

__device__ bool token_matches(cudf::string_view set,
                              cudf::size_type token_start,
                              cudf::size_type token_size,
                              cudf::string_view word)
{
  auto const word_size = word.size_bytes();
  if (token_size != word_size) { return false; }
  if (word_size == 0) { return true; }

  auto const* token     = set.data() + token_start;
  auto const* word_data = word.data();
  for (cudf::size_type idx = 0; idx < word_size; ++idx) {
    if (token[idx] != word_data[idx]) { return false; }
  }
  return true;
}

__device__ cudf::size_type find_token_position(cudf::string_view set, cudf::string_view word)
{
  auto const* set_data        = set.data();
  auto const set_size         = set.size_bytes();
  cudf::size_type token_pos   = 1;
  cudf::size_type token_start = 0;

  for (cudf::size_type idx = 0; idx <= set_size; ++idx) {
    if (idx == set_size || set_data[idx] == ',') {
      if (token_matches(set, token_start, idx - token_start, word)) { return token_pos; }
      ++token_pos;
      token_start = idx + 1;
    }
  }
  return 0;
}

}  // namespace

std::unique_ptr<cudf::column> find_in_set(cudf::strings_column_view const& sets,
                                          std::string const& word,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  auto const row_count = sets.size();
  if (row_count == 0) { return cudf::make_empty_column(cudf::type_id::INT32); }

  auto results         = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                           row_count,
                                           cudf::copy_bitmask(sets.parent(), stream, mr),
                                           sets.null_count(),
                                           stream,
                                           mr);
  auto const d_results = results->mutable_view().data<cudf::size_type>();

  if (word.find(',') != std::string::npos) {
    thrust::fill_n(rmm::exec_policy(stream), d_results, row_count, cudf::size_type{0});
    results->set_null_count(sets.null_count());
    return results;
  }

  auto word_scalar               = cudf::make_string_scalar(word, stream);
  auto const& word_string_scalar = static_cast<cudf::string_scalar const&>(*word_scalar);
  auto const d_word = cudf::string_view(word_string_scalar.data(), word_string_scalar.size());

  auto const sets_column = cudf::column_device_view::create(sets.parent(), stream);
  auto const d_sets      = *sets_column;

  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(row_count),
                    d_results,
                    [d_sets, d_word] __device__(cudf::size_type idx) {
                      if (d_sets.is_null(idx)) { return cudf::size_type{0}; }
                      return find_token_position(d_sets.element<cudf::string_view>(idx), d_word);
                    });
  results->set_null_count(sets.null_count());
  return results;
}

std::unique_ptr<cudf::column> find_in_set_repeated(cudf::strings_column_view const& sets,
                                                   std::string const& word,
                                                   cudf::size_type max_distinct_sets,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  auto const row_count = sets.size();
  if (row_count == 0) { return cudf::make_empty_column(cudf::type_id::INT32); }

  auto make_zero_or_null_result = [&]() {
    auto results = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                             row_count,
                                             cudf::copy_bitmask(sets.parent(), stream, mr),
                                             sets.null_count(),
                                             stream,
                                             mr);
    thrust::fill_n(rmm::exec_policy(stream),
                   results->mutable_view().data<cudf::size_type>(),
                   row_count,
                   cudf::size_type{0});
    results->set_null_count(sets.null_count());
    return results;
  };

  if (word.find(',') != std::string::npos) { return make_zero_or_null_result(); }

  auto dictionary = cudf::dictionary::encode(
    sets.parent(), cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto const dictionary_view = cudf::dictionary_column_view{dictionary->view()};
  auto const keys_size       = dictionary_view.keys_size();
  if (keys_size > max_distinct_sets) { return nullptr; }
  if (keys_size == 0) { return make_zero_or_null_result(); }

  auto key_positions =
    find_in_set(cudf::strings_column_view{dictionary_view.keys()}, word, stream, mr);

  auto gather_map = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                              row_count,
                                              cudf::mask_state::UNALLOCATED,
                                              stream,
                                              mr);
  auto const d_gather_map = gather_map->mutable_view().data<cudf::size_type>();
  auto const d_dictionary =
    cudf::column_device_view::create(dictionary_view.parent(), stream);
  auto const d_indices =
    cudf::column_device_view::create(dictionary_view.indices(), stream);

  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(row_count),
                    d_gather_map,
                    [d_dictionary = *d_dictionary, d_indices = *d_indices] __device__(
                      cudf::size_type idx) {
                      return d_dictionary.is_null(idx) ? cudf::size_type{0}
                                                       : d_indices.element<cudf::size_type>(idx);
                    });

  auto gathered_table = cudf::gather(cudf::table_view{{key_positions->view()}},
                                     gather_map->view(),
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     stream,
                                     mr);
  auto result         = std::move(gathered_table->release()[0]);
  result->set_null_mask(cudf::copy_bitmask(sets.parent(), stream, mr), sets.null_count());
  return result;
}

}  // namespace spark_rapids_jni
