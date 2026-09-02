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

#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <memory>

namespace spark_rapids_jni {

/**
 *
 * Select the column index for the first true in bool columns.
 * For the row does not contain true, use end index(number of columns).
 *
 * e.g.:
 *   column 0 in table: true,  false, false, false
 *   column 1 in table: false, true,  false, false
 *   column 2 in table: false, false, true, false
 *
 *   1st row is: true, flase, false; first true index is 0
 *   2nd row is: false, true, false; first true index is 1
 *   3rd row is: false, flase, true; first true index is 2
 *   4th row is: false, false, false; do not find true, set index to the end index 3
 *
 *   output column: 0, 1, 2, 3
 *   In the `case when` context, here 3 index means using NULL value.
 *
 */
std::unique_ptr<cudf::column> select_first_true_index(
  cudf::table_view const& when_bool_columns,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
