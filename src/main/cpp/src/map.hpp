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

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace spark_rapids_jni {

/**
 * Sort entries for each map in map column according to the keys of each map.
 * Note:
 *   The keys of map MUST not be null.
 *   Assume that maps do not have duplicate keys.
 *   Do not normalize/sort the nested maps in `KEY` column; This means
 *   Only consider the first level LIST(STRUCT(KEY, VALUE)) as map type.
 *
 * @param input Input map column, should in LIST(STRUCT(KEY, VALUE)) type.
 * @param sort_order Ascending or descending order
 * @return Sorted map according to the sort order of the key column in map.
 * @throws cudf::logic_error If the input column is not a LIST(STRUCT(KEY, VALUE)) column or the
 * keys contain nulls.
 */
std::unique_ptr<cudf::column> sort_map_column(
  cudf::column_view const& input,
  cudf::order sort_order,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
