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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

//
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

namespace spark_rapids_jni {

/**
 * @brief Check the input if they are valid and create a histogram from them.
 *
 * Validity of the input columns are defined as following:
 *  - Values and frequencies columns must have the same size.
 *  - Frequencies column must be of type INT64, must not have nulls, and must not contain
 *    negative numbers.
 *
 * If the input columns are valid, a histogram will be created from them. Otherwise, an exception
 * will be thrown.
 *
 * The output histogram is stored in a structs column in the form of `STRUCT<value, frequency>`.
 * If `output_as_lists == true`, each struct element is wrapped in a list, producing a
 * lists-of-structs column.
 *
 * @param values The input values
 * @param frequencies The frequencies corresponding to the input values
 * @param output_as_list Specify whether to wrap each pair of <value, frequency> in the output
 * histogram in a separate list
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A histogram column with data copied from the input
 */
std::unique_ptr<cudf::column> create_histogram_if_valid(
  cudf::column_view const& values,
  cudf::column_view const& frequencies,
  bool output_as_lists,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Compute percentiles from the given histograms and percentage values.
 *
 * The input histograms must be given in the form of `LIST<STRUCT<ElementType, long>>`.
 *
 * @param input The lists of input histograms
 * @param percentages The input percentage values
 * @param output_as_lists Specify whether the output percentiles will be wrapped in a list
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A lists column, each list stores the percentile value(s) of the corresponding row in the
 * input column
 */
std::unique_ptr<cudf::column> percentile_from_histogram(
  cudf::column_view const& input,
  std::vector<double> const& percentage,
  bool output_as_lists,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
