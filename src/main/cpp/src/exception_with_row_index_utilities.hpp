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
#include <cudf/utilities/default_stream.hpp>

#include <cuda/stream>

namespace spark_rapids_jni {

/**
 * @brief Throws exception_with_row_index if has any row is invalid for a unary operation.
 * If the input is not null and the result is null, it means the row is invalid.
 *
 * @throws exception_with_row_index if any row is invalid.
 *
 * @param input The input.
 * @param result The result column view.
 */
void throw_row_error_if_any(cudf::column_view const& input,
                            cudf::column_view const& result,
                            cuda::stream_ref stream = cudf::get_default_stream());

/**
 * @brief Throws exception_with_row_index if has any row is invalid for a binary operation.
 * If the input is not null and the result is null, it means the row is invalid.
 *
 * @throws exception_with_row_index if any row is invalid.
 *
 * @param input1 The first input.
 * @param input2 The second input.
 * @param result The result column view.
 */
void throw_row_error_if_any(cudf::column_view const& input1,
                            cudf::column_view const& input2,
                            cudf::column_view const& result,
                            cuda::stream_ref stream = cudf::get_default_stream());

/**
 * @brief Throws exception_with_row_index if has any row is invalid for a ternary operation.
 * If the input is not null and the result is null, it means the row is invalid.
 *
 * @throws exception_with_row_index if any row is invalid.
 *
 * @param input1 The first input.
 * @param input2 The second input.
 * @param input3 The third input.
 * @param result The result column view.
 */
void throw_row_error_if_any(cudf::column_view const& input1,
                            cudf::column_view const& input2,
                            cudf::column_view const& input3,
                            cudf::column_view const& result,
                            cuda::stream_ref stream = cudf::get_default_stream());

/**
 * @brief Throws exception_with_row_index if has any row is invalid for a binary operation.
 * If the input is not null and the result is null, it means the row is invalid.
 *
 * @throws exception_with_row_index if any row is invalid.
 *
 * @param input1 The first input.
 * @param input2 The second input.
 * @param result The result column view.
 */
void throw_row_error_if_any(cudf::column_view const& input1,
                            cudf::scalar const& input2,
                            cudf::column_view const& result,
                            cuda::stream_ref stream = cudf::get_default_stream());

/**
 * @brief Throws exception_with_row_index if has any row is invalid for a binary operation.
 * If the input is not null and the result is null, it means the row is invalid.
 *
 * @throws exception_with_row_index if any row is invalid.
 *
 * @param input1 The first input.
 * @param input2 The second input.
 * @param result The result column view.
 */
void throw_row_error_if_any(cudf::scalar const& input1,
                            cudf::column_view const& input2,
                            cudf::column_view const& result,
                            cuda::stream_ref stream = cudf::get_default_stream());

}  // namespace spark_rapids_jni
