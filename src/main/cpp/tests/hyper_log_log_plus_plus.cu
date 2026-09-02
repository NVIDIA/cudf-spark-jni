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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <hyper_log_log_plus_plus.hpp>
#include <hyper_log_log_plus_plus_host_udf.hpp>
#include <utilities/iterator.cuh>

using doubles_col = cudf::test::fixed_width_column_wrapper<double>;
using int64_col   = cudf::test::fixed_width_column_wrapper<int64_t>;

struct HyperLogLogPlusPlusUDFTest : cudf::test::BaseFixture {};

namespace {

/**
 * @brief Concatenate struct scalars into a struct column.
 * @param scalar_col_ptrs Pointers to the columns in scalars, this span appends the column pointes
 * of scalar one by one, the size of the vector is num_scalars * num_longs_in_scalar.
 * @param num_scalars Number of struct scalars
 * @param num_longs_in_scalar Number of long columns in each struct scalar
 */
CUDF_KERNEL void concat_struct_scalars_to_struct_column_kernel(
  cudf::device_span<int64_t const*> scalar_col_ptrs,
  int num_scalars,
  int num_longs_in_scalar,
  cudf::device_span<int64_t*> output)
{
  for (auto col = 0; col < num_longs_in_scalar; ++col) {
    for (auto scalar_idx = 0; scalar_idx < num_scalars; ++scalar_idx) {
      auto flattened_col_idx  = scalar_idx * num_longs_in_scalar + col;
      output[col][scalar_idx] = scalar_col_ptrs[flattened_col_idx][0];
    }
  }
}

/**
 * @brief Flatten columns in scalars into a vector.
 */
std::vector<int64_t const*> get_column_ptrs_from_struct_scalars(
  std::vector<std::unique_ptr<cudf::scalar>>& scalars, int num_longs_in_scalar)
{
  std::vector<int64_t const*> col_ptrs(num_longs_in_scalar * scalars.size());
  int idx = 0;
  for (auto const& s : scalars) {
    auto const struct_scalar_ptr = dynamic_cast<cudf::struct_scalar*>(s.get());
    auto const table_view        = struct_scalar_ptr->view();
    for (auto const& col_view : table_view) {
      col_ptrs[idx++] = col_view.data<int64_t>();
    }
  }
  return col_ptrs;
}

/**
 * @brief Make a struct column from multiple scalars with checks: each scalar is a struct(long)
 */
std::unique_ptr<cudf::column> make_struct_column_from_scalars(
  std::vector<std::unique_ptr<cudf::scalar>>& scalars,
  int num_longs_in_scalar,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  // asserts
  for (auto const& s : scalars) {
    EXPECT_EQ(s->type().id(), cudf::type_id::STRUCT);
    auto const struct_scalar_ptr = dynamic_cast<cudf::struct_scalar*>(s.get());
    auto const table_view        = struct_scalar_ptr->view();
    EXPECT_EQ(num_longs_in_scalar, table_view.num_columns());
    for (auto const& col_view : table_view) {
      EXPECT_EQ(col_view.type().id(), cudf::type_id::INT64);
    }
  }

  // get column pointers from struct scalars
  auto col_ptrs   = get_column_ptrs_from_struct_scalars(scalars, num_longs_in_scalar);
  auto d_col_ptrs = cudf::detail::make_device_uvector(col_ptrs, stream, mr);

  // create output columns
  auto const results_iter = spark_rapids_jni::util::make_counting_transform_iterator(0, [&](int i) {
    return cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                     scalars.size(),  // num_rows
                                     cudf::mask_state::UNALLOCATED,
                                     stream,
                                     mr);
  });
  auto children =
    std::vector<std::unique_ptr<cudf::column>>(results_iter, results_iter + num_longs_in_scalar);
  auto host_results_pointer_iter =
    thrust::make_transform_iterator(children.begin(), [](auto const& results_column) {
      return results_column->mutable_view().template data<int64_t>();
    });
  auto host_results_pointers =
    std::vector<int64_t*>(host_results_pointer_iter, host_results_pointer_iter + children.size());
  auto d_output = cudf::detail::make_device_uvector(host_results_pointers, stream, mr);

  // concatenate struct scalars into a struct column
  concat_struct_scalars_to_struct_column_kernel<<<1, 1, 0, stream.get()>>>(
    d_col_ptrs, scalars.size(), num_longs_in_scalar, d_output);

  // create struct column
  return cudf::make_structs_column(scalars.size(),  // num_rows
                                   std::move(children),
                                   0,                     // null count
                                   rmm::device_buffer{},  // null mask
                                   stream);
}

/**
 * @brief Make a struct column from a single scalar with checks: each scalar is a struct(long)
 */
std::unique_ptr<cudf::column> make_struct_column_from_scalar(std::unique_ptr<cudf::scalar>& scalar,
                                                             int num_longs_in_scalar,
                                                             cuda::stream_ref stream,
                                                             rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  scalars.push_back(std::move(scalar));
  return make_struct_column_from_scalars(scalars, num_longs_in_scalar, stream, mr);
}

}  // namespace

TEST_F(HyperLogLogPlusPlusUDFTest, Reduction)
{
  // 1. Create data
  auto const vals1 = doubles_col{1.0, 2.0, 3.0, 4.0, 5.0};
  auto const vals2 = doubles_col{6.0, 7.0, 8.0, 9.0, 10.0};
  auto stream      = cudf::get_default_stream();
  auto mr          = cudf::get_current_device_resource_ref();

  // 2. Execute reduce
  // There are pow(2, 9) = 512 registers
  // Each register stores the num of leading zeros of hash, the max num is 64,
  // 6 bits are enough to store the number, so each register is 6 bits.
  // All the registers are compacted into 512 / (64 / 6) + 1 = 52 longs
  constexpr int precision           = 9;
  constexpr int num_longs_in_sketch = 512 / 10 + 1;
  auto const reduce_agg =
    cudf::make_host_udf_aggregation<cudf::reduce_aggregation>(std::unique_ptr<cudf::host_udf_base>(
      spark_rapids_jni::create_hllpp_reduction_host_udf(precision)));
  std::vector<std::unique_ptr<cudf::scalar>> reduced_scalars;
  for (size_t i = 0; i < 64; i++) {
    if (i % 2 == 0) {
      auto reduced =
        cudf::reduce(vals1, *reduce_agg, cudf::data_type{cudf::type_id::STRUCT}, stream, mr);
      EXPECT_TRUE(reduced->is_valid());
      reduced_scalars.push_back(std::move(reduced));
    } else {
      auto reduced =
        cudf::reduce(vals2, *reduce_agg, cudf::data_type{cudf::type_id::STRUCT}, stream, mr);
      EXPECT_TRUE(reduced->is_valid());
      reduced_scalars.push_back(std::move(reduced));
    }
  }

  // 3. Merge all the sketches into one sketch
  auto const input_for_merge =
    make_struct_column_from_scalars(reduced_scalars, num_longs_in_sketch, stream, mr);
  auto const merge_agg =
    cudf::make_host_udf_aggregation<cudf::reduce_aggregation>(std::unique_ptr<cudf::host_udf_base>(
      spark_rapids_jni::create_hllpp_reduction_merge_host_udf(precision)));

  auto reduce_merged =
    cudf::reduce(*input_for_merge, *merge_agg, cudf::data_type{cudf::type_id::STRUCT}, stream, mr);
  EXPECT_TRUE(reduce_merged->is_valid());

  // 4. Estimate count distinct values from the merged sketch
  auto const input_for_estimate =
    make_struct_column_from_scalar(reduce_merged, num_longs_in_sketch, stream, mr);
  auto const result =
    spark_rapids_jni::estimate_from_hll_sketches(*input_for_estimate, precision, stream, mr);

  // 5. check count distinct value
  auto const expected = int64_col{10};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(HyperLogLogPlusPlusUDFTest, Groupby)
{
  constexpr int precision = 9;

  // 1. Create data
  auto const keys = int64_col{1, 2, 3, 1, 2, 3, 1, 2, 3};
  // Each key in (1, 2, 3) maps to three values: (1.0, 2.0, 3.0)
  auto const vals1 = doubles_col{1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0};
  // Each key in (1, 2, 3) maps to three values: (4.0, 5.0, 6.0)
  auto const vals2 = doubles_col{4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0};
  // Each key in (1, 2, 3) maps to three values: (7.0, 8.0, 9.0)
  auto const vals3 = doubles_col{7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 9.0, 9.0, 9.0};

  // 2. Execute groupby
  auto agg1 =
    cudf::make_host_udf_aggregation<cudf::groupby_aggregation>(std::unique_ptr<cudf::host_udf_base>(
      spark_rapids_jni::create_hllpp_groupby_host_udf(precision)));
  auto agg2 =
    cudf::make_host_udf_aggregation<cudf::groupby_aggregation>(std::unique_ptr<cudf::host_udf_base>(
      spark_rapids_jni::create_hllpp_groupby_host_udf(precision)));
  auto agg3 =
    cudf::make_host_udf_aggregation<cudf::groupby_aggregation>(std::unique_ptr<cudf::host_udf_base>(
      spark_rapids_jni::create_hllpp_groupby_host_udf(precision)));
  std::vector<cudf::groupby::aggregation_request> agg_requests;
  agg_requests.emplace_back();
  agg_requests[0].values = vals1;
  agg_requests[0].aggregations.push_back(std::move(agg1));
  agg_requests.emplace_back();
  agg_requests[1].values = vals2;
  agg_requests[1].aggregations.push_back(std::move(agg2));
  agg_requests.emplace_back();
  agg_requests[2].values = vals3;
  agg_requests[2].aggregations.push_back(std::move(agg3));
  cudf::groupby::groupby gb_obj(
    cudf::table_view({keys}), cudf::null_policy::INCLUDE, cudf::sorted::NO, {}, {});
  auto const grp_result = gb_obj.aggregate(agg_requests, cudf::test::get_default_stream());
  // each grouped sketches has 3 rows for keys: 1, 2, 3
  auto const grouped_sketches_for_vals1 = grp_result.second[0].results[0]->view();
  auto const grouped_sketches_for_vals2 = grp_result.second[1].results[0]->view();
  auto const grouped_sketches_for_vals3 = grp_result.second[2].results[0]->view();

  // 3. Execute merge sketches
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();
  // each result is 3 rows, concat to 9 rows.
  auto const sketches = cudf::concatenate(
    std::vector<cudf::column_view>{
      grouped_sketches_for_vals1, grouped_sketches_for_vals2, grouped_sketches_for_vals3},
    stream,
    mr);
  auto merge_agg =
    cudf::make_host_udf_aggregation<cudf::groupby_aggregation>(std::unique_ptr<cudf::host_udf_base>(
      spark_rapids_jni::create_hllpp_groupby_merge_host_udf(precision)));
  std::vector<cudf::groupby::aggregation_request> merge_requests;
  merge_requests.emplace_back();
  merge_requests[0].values = sketches->view();
  merge_requests[0].aggregations.push_back(std::move(merge_agg));
  cudf::groupby::groupby gb_obj2(
    cudf::table_view({keys}), cudf::null_policy::INCLUDE, cudf::sorted::NO, {}, {});
  auto const grp_result2 = gb_obj2.aggregate(merge_requests, cudf::test::get_default_stream());
  auto const& merged     = grp_result2.second[0].results[0];

  // 4. Estimate
  auto const result = spark_rapids_jni::estimate_from_hll_sketches(*merged, precision, stream, mr);

  // 5. Check result
  // each key in (1, 2, 3) has 9 distinct values: (1.0, 2.0, ..., 9.0)
  auto const expected = int64_col{9, 9, 9};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}
