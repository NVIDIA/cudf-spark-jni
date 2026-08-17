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
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream_ref>

namespace spark_rapids_jni {

/**
 * @brief Compute hash codes for the input, generate HyperLogLogPlusPlus(HLLPP)
 * sketches from hash codes, and merge the sketches in the same group. Output is
 * a struct column with multiple long columns which is consistent with Spark.
 */
std::unique_ptr<cudf::column> group_hyper_log_log_plus_plus(
  cudf::column_view const& input,
  int64_t const num_groups,
  cudf::device_span<cudf::size_type const> group_labels,
  int64_t const precision,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Merge HyperLogLogPlusPlus(HLLPP) sketches in the same group.
 * Input is a struct column with multiple long columns which is consistent with
 * Spark.
 */
std::unique_ptr<cudf::column> group_merge_hyper_log_log_plus_plus(
  cudf::column_view const& input,
  int64_t const num_groups,
  cudf::device_span<cudf::size_type const> group_labels,
  int64_t const precision,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Compute hash codes for the input, generate HyperLogLogPlusPlus(HLLPP)
 * sketches from hash codes, and merge all the sketches into one sketch, output
 * is a struct scalar with multiple long values.
 */
std::unique_ptr<cudf::scalar> reduce_hyper_log_log_plus_plus(
  cudf::column_view const& input,
  int64_t const precision,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Merge all HyperLogLogPlusPlus(HLLPP) sketches in the input column into one
 * sketch. Input is a struct column with multiple long columns which is
 * consistent with Spark. Output is a struct scalar with multiple long values.
 */
std::unique_ptr<cudf::scalar> reduce_merge_hyper_log_log_plus_plus(
  cudf::column_view const& input,
  int64_t const precision,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Estimate count distinct values from HyperLogLogPlusPlus(HLLPP) sketches.
 * Input is a struct column with multiple long columns which is consistent with
 * Spark. Output is a long column with all values are not null. Spark returns 0
 * for null values when doing APPROX_COUNT_DISTINCT.
 */
std::unique_ptr<cudf::column> estimate_from_hll_sketches(
  cudf::column_view const& input,
  int precision,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
}  // namespace spark_rapids_jni
