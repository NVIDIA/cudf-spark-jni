/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

#include <cuda/stream>

#include <memory>

namespace spark_rapids_jni {

/**
 * @brief Calculates the difference in years between the epoch year (1970) and the
 * given date/timestamp column. E.g.: for date '1971-01-01', the result would be 1:
 * (1 year after epoch year)
 *
 * @param input The input date/timestamp column.
 * @param stream The CUDA stream to use for device memory operations and kernel launches.
 * @param mr Device memory resource to use for allocations.
 * @return A column of type INT32 containing the year differences from epoch.
 */
std::unique_ptr<cudf::column> years_from_epoch(
  cudf::column_view const& input,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Calculates the difference in months between the epoch month (1970-01) and the
 * given date/timestamp column. E.g.: for date '1971-02-01', the result would be 13:
 * (1 year and 1 month after epoch month)
 *
 * @param input The input date/timestamp column.
 * @param stream The CUDA stream to use for device memory operations and kernel launches.
 * @param mr Device memory resource to use for allocations.
 * @return A column of type INT32 containing the month differences from epoch.
 */
std::unique_ptr<cudf::column> months_from_epoch(
  cudf::column_view const& input,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Calculates the difference in days between the epoch day (1970-01-01) and the
 * given date/timestamp column. E.g.: for date '1970-01-21', the result would be 20:
 * (20 days after epoch day)
 *
 * @param input The input date/timestamp column.
 * @param stream The CUDA stream to use for device memory operations and kernel launches.
 * @param mr Device memory resource to use for allocations.
 * @return A column of type Date.
 */
std::unique_ptr<cudf::column> days_from_epoch(
  cudf::column_view const& input,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * Calculates the difference in hours between the epoch hour (1970-01-01T00:00:00)
 * and the given timestamp column. E.g.: for timestamp '1970-01-01 01:00:00',
 * the result would be 1 (1 hour after epoch hour)
 *
 * @param input The input timestamp column.
 * @param stream The CUDA stream to use for device memory operations and kernel launches.
 * @param mr Device memory resource to use for allocations.
 * @return A column of type INT32 containing the hour differences from epoch.
 */
std::unique_ptr<cudf::column> hours_from_epoch(
  cudf::column_view const& input,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
