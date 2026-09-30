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

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace spark_rapids_jni {

/**
 * @brief Multiply two columns with overflow check if ANSI mode is enabled.
 * Only supports Spark data types: byte, short, integer, long, float32 and float64.
 * If `is_ansi_mode` is true, it checks for overflow and sets the validity accordingly.
 *
 * E.g.:
 * Integer.MAX_VALUE * 2 = -2 in regular mode
 * Integer.MAX_VALUE * 2 = null in try mode
 * Integer.MAX_VALUE * 2 = throws exception in ansi mode
 *
 * The left and right inputs MUST have the same data type, and have the same size.
 * MUST not enable both ANSI mode and TRY mode at the same time.
 *
 * @throws spark_rapids_jni::exception_with_row_index exception if it's ansi mode and overflow
 * occurs.
 *
 * @param left_input The left input.
 * @param right_input The right input.
 * @param is_ansi_mode If true, throws exception when overflow occurs.
 * @param is_try_mode If true, set null when overflow occurs.
 * @param stream CUDA stream to use for the operation.
 * @param mr Memory resource to use for allocations.
 * @return A new column containing the results of the multiplication.
 */
std::unique_ptr<cudf::column> multiply(
  cudf::column_view const& left_input,
  cudf::column_view const& right_input,
  bool is_ansi_mode,
  bool is_try_mode,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Multiply a column and a scalar with overflow check if ANSI mode is enabled.
 * Only supports Spark data types: byte, short, integer, long, float32 and float64.
 * If `is_ansi_mode` is true, it checks for overflow and sets the validity accordingly.
 *
 * E.g.:
 * Integer.MAX_VALUE * 2 = -2 in regular mode
 * Integer.MAX_VALUE * 2 = null in try mode
 * Integer.MAX_VALUE * 2 = throws exception in ansi mode
 *
 * The left and right inputs MUST have the same data type, and have the same size.
 * MUST not enable both ANSI mode and TRY mode at the same time.
 * Input types MUST be: int8, int16, int32, int64, float32, or float64.
 *
 * @throws spark_rapids_jni::exception_with_row_index exception if it's ansi mode and overflow
 * occurs.
 *
 * @param left_input The left input.
 * @param right_input The right input.
 * @param is_ansi_mode If true, throws exception when overflow occurs.
 * @param is_try_mode If true, set null when overflow occurs.
 * @param stream CUDA stream to use for the operation.
 * @param mr Memory resource to use for allocations.
 * @return A new column containing the results of the multiplication.
 */
std::unique_ptr<cudf::column> multiply(
  cudf::column_view const& left_input,
  cudf::scalar const& right_input,
  bool is_ansi_mode,
  bool is_try_mode,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Multiply a scalar and a column with overflow check if ANSI mode is enabled.
 * Only supports Spark data types: byte, short, integer, long, float32 and float64.
 * If `is_ansi_mode` is true, it checks for overflow and sets the validity accordingly.
 *
 * E.g.:
 * Integer.MAX_VALUE * 2 = -2 in regular mode
 * Integer.MAX_VALUE * 2 = null in try mode
 * Integer.MAX_VALUE * 2 = throws exception in ansi mode
 *
 * The left and right inputs MUST have the same data type, and have the same size.
 * MUST not enable both ANSI mode and TRY mode at the same time.
 * Input types MUST be: int8, int16, int32, int64, float32, or float64.
 *
 * @throws spark_rapids_jni::exception_with_row_index exception if it's ansi mode and overflow
 * occurs.
 *
 * @param left_input The left input.
 * @param right_input The right input.
 * @param is_ansi_mode If true, throws exception when overflow occurs.
 * @param is_try_mode If true, set null when overflow occurs.
 * @param stream CUDA stream to use for the operation.
 * @param mr Memory resource to use for allocations.
 * @return A new column containing the results of the multiplication.
 */
std::unique_ptr<cudf::column> multiply(
  cudf::scalar const& left_input,
  cudf::column_view const& right_input,
  bool is_ansi_mode,
  bool is_try_mode,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
