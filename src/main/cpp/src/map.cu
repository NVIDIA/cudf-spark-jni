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

#include "map.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/stream>

namespace spark_rapids_jni {

std::unique_ptr<cudf::column> sort_map_column(cudf::column_view const& input,
                                              cudf::order sort_order,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::LIST,
               "maps_column_view input must be LIST type");
  if (input.size() == 0) { return cudf::make_empty_column(input.type()); }

  auto const lists_of_structs = cudf::lists_column_view(input);
  auto const structs          = lists_of_structs.child();
  CUDF_EXPECTS(structs.type().id() == cudf::type_id::STRUCT,
               "maps_column_view input must have exactly 1 child (STRUCT) column.");
  CUDF_EXPECTS(structs.num_children() == 2,
               "maps_column_view key-value struct must have exactly 2 children.");
  auto keys   = structs.child(0);
  auto values = structs.child(1);
  CUDF_EXPECTS(structs.null_count() == 0, "maps_column_view key-value struct must have no null.");
  CUDF_EXPECTS(keys.null_count() == 0, "maps_column_view keys must have no null.");
  auto segments = lists_of_structs.offsets();

  auto sorted = cudf::segmented_sort_by_key(cudf::table_view{{structs}},
                                            cudf::table_view{{keys}},
                                            segments,
                                            {sort_order},
                                            {},  // Map keys MUST not be null
                                            stream,
                                            mr);

  return cudf::make_lists_column(input.size(),
                                 std::make_unique<cudf::column>(segments),  // copy segment offsets
                                 std::move(sorted->release().front()),      // child column
                                 input.null_count(),
                                 cudf::copy_bitmask(input, stream, mr));
}

}  // namespace spark_rapids_jni
