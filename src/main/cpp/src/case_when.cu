/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "case_when.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <thrust/transform.h>

namespace spark_rapids_jni {
namespace detail {
namespace {

/**
 * Select the column index for the first true in bool columns for the specified row
 */
struct select_first_true_fn {
  // bool columns stores the results of executing `when` expressions
  cudf::table_device_view const d_table;

  /**
   * The number of bool columns is the size of case when branches.
   * Note: reuturned index may be out of bound, valid bound is [0, col_num)
   * When returning col_num index, it means final result is NULL value or ELSE value.
   *
   * e.g.:
   * CASE WHEN 'a' THEN 'A' END
   *   The number of bool columns is 1
   *   The number of scalars is 1
   *   Max index is 1 which means using NULL(all when exprs are false).
   * CASE WHEN 'a' THEN 'A' ELSE '_' END
   *   The number of bool columns is 1
   *   The number of scalars is 2
   *   Max index is also 1 which means using else value '_'
   */
  __device__ cudf::size_type operator()(std::size_t row_idx) const
  {
    auto col_num = d_table.num_columns();
    for (auto col_idx = 0; col_idx < col_num; col_idx++) {
      auto const& col = d_table.column(col_idx);
      if (!col.is_null(row_idx) && col.element<bool>(row_idx)) {
        // Predicate is true and not null
        return col_idx;
      }
    }
    return col_num;
  }
};

}  // anonymous namespace

std::unique_ptr<cudf::column> select_first_true_index(cudf::table_view const& when_bool_columns,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  // checks
  auto const num_columns = when_bool_columns.num_columns();
  CUDF_EXPECTS(num_columns > 0, "At least one column must be specified");
  auto const row_count = when_bool_columns.num_rows();
  if (row_count == 0) {  // empty begets empty
    return cudf::make_empty_column(cudf::type_id::INT32);
  }
  // make output column
  auto ret = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, row_count, cudf::mask_state::UNALLOCATED, stream, mr);

  // select first true index
  auto const d_table_ptr = cudf::table_device_view::create(when_bool_columns, stream);
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(row_count),
                    ret->mutable_view().begin<cudf::size_type>(),
                    select_first_true_fn{*d_table_ptr});
  return ret;
}

}  // namespace detail

std::unique_ptr<cudf::column> select_first_true_index(cudf::table_view const& when_bool_columns,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{ return detail::select_first_true_index(when_bool_columns, stream, mr); }

}  // namespace spark_rapids_jni
