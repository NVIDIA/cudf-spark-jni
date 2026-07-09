/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/generate_input.hpp"

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/filling.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvbench/nvbench.cuh>
#include <parse_uri.hpp>

static void bench_random_parse_uri(nvbench::state& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.get_int64("num_rows")};

  auto const table = create_random_table({cudf::type_id::STRING}, row_count{n_rows});

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto out =
      spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{table->get_column(0)});
  });

  state.add_buffer_size(n_rows, "trc", "Total Rows");
}

static void bench_parse_uri(nvbench::state& state)
{
  auto const n_rows   = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const hit_rate = static_cast<cudf::size_type>(state.get_int64("hit_rate"));

  // build input table using the following data
  auto raw_data =
    cudf::test::strings_column_wrapper(
      {
        "https://www.google.com/s/"
        "query?parameternumber0=0&parameternumber1=1&parameternumber2=2parameternumber2="
        "2parameternumber2=2parameternumber2=2parameternumber2=2parameternumber2=2parameternumber2="
        "2parameternumber2=2parameternumber2=2parameternumber2=2parameternumber2=2parameternumber2="
        "2parameternumber2=2parameternumber2=2parameternumber2=2parameternumber2=2parameternumber2="
        "2parameternumber2=2parameternumber2=2parameternumber2=2parameternumber2=2parameternumber2="
        "2parameternumber2=2parameternumber2=2parameternumber2=2",  // valid uri
        "abcdefghijklmnopqrstuvwxyz 01234abcdefghijklmnopqrstuvwxyz "
        "01234abcdefghijklmnopqrstuvwxyz 01234abcdefghijklmnopqrstuvwxyz "
        "01234abcdefghijklmnopqrstuvwxyz 01234abcdefghijklmnopqrstuvwxyz "
        "01234abcdefghijklmnopqrstuvwxyz 01234abcdefghijklmnopqrstuvwxyz "
        "01234abcdefghijklmnopqrstuvwxyz 01234abcdefghijklmnopqrstuvwxyz "
        "01234abcdefghijklmnopqrstuvwxyz 01234abcdefghijklmnopqrstuvwxyz "
        "01234abcdefghijklmnopqrstuvwxyz 01234abcdefghijklmnopqrstuvwxyz "
        "01234abcdefghijklmnopqrstuvwxyz 01234abcdefghijklmnopqrstuvwxyz "
        "01234abcdefghijklmnopqrstuvwxyz 01234abcdefghijklmnopqrstuvwxyz 01234",  // the rest are
                                                                                  // invalid
        "",
        "AbcéDEFGHIJKLMNOPQRSTUVWXYZ 01AbcéDEFGHIJKLMNOPQRSTUVWXYZ 01AbcéDEFGHIJKLMNOPQRSTUVWXYZ "
        "01AbcéDEFGHIJKLMNOPQRSTUVWXYZ 01AbcéDEFGHIJKLMNOPQRSTUVWXYZ 01AbcéDEFGHIJKLMNOPQRSTUVWXYZ "
        "01AbcéDEFGHIJKLMNOPQRSTUVWXYZ 01AbcéDEFGHIJKLMNOPQRSTUVWXYZ 01AbcéDEFGHIJKLMNOPQRSTUVWXYZ "
        "01AbcéDEFGHIJKLMNOPQRSTUVWXYZ 01AbcéDEFGHIJKLMNOPQRSTUVWXYZ 01AbcéDEFGHIJKLMNOPQRSTUVWXYZ "
        "01AbcéDEFGHIJKLMNOPQRSTUVWXYZ 01AbcéDEFGHIJKLMNOPQRSTUVWXYZ 01AbcéDEFGHIJKLMNOPQRSTUVWXYZ "
        "01AbcéDEFGHIJKLMNOPQRSTUVWXYZ 01AbcéDEFGHIJKLMNOPQRSTUVWXYZ 01AbcéDEFGHIJKLMNOPQRSTUVWXYZ "
        "01AbcéDEFGHIJKLMNOPQRSTUVWXYZ 01AbcéDEFGHIJKLMNOPQRSTUVWXYZ 01",
        "9876543210,abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
        "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
        "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
        "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
        "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
        "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
        "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
        "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU",
        "9876543210,abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
        "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
        "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
        "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
        "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
        "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
        "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
        "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
        "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
        "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU9876543210,"
        "abcdefghijklmnopqrstU9876543210,abcdefghijklmnopqrstU",
        "123 édf 4567890 DéFG 0987 X5123 édf 4567890 DéFG 0987 X5123 édf 4567890 DéFG 0987 X5123 "
        "édf 4567890 DéFG 0987 X5123 édf 4567890 DéFG 0987 X5123 édf 4567890 DéFG 0987 X5123 édf "
        "4567890 DéFG 0987 X5123 édf 4567890 DéFG 0987 X5123 édf 4567890 DéFG 0987 X5123 édf "
        "4567890 DéFG 0987 X5123 édf 4567890 DéFG 0987 X5123 édf 4567890 DéFG 0987 X5123 édf "
        "4567890 DéFG 0987 X5123 édf 4567890 DéFG 0987 X5123 édf 4567890 DéFG 0987 X5123 édf "
        "4567890 DéFG 0987 X5123 édf 4567890 DéFG 0987 X5123 édf 4567890 DéFG 0987 X5123 édf "
        "4567890 DéFG 0987 X5123 édf 4567890 DéFG 0987 X5123 édf 4567890 DéFG 0987 X5123 édf "
        "4567890 DéFG 0987 X5123 édf 4567890 DéFG 0987 X5123 édf 4567890 DéFG 0987 X5123 édf "
        "4567890 DéFG 0987 X5123 édf 4567890 DéFG 0987 X5",
        "1",
      })
      .release();

  auto data_view = raw_data->view();

  // compute number of rows in n_rows that should match
  auto const matches = static_cast<cudf::size_type>(n_rows * hit_rate) / 100;

  // Create a randomized gather-map to build a column out of the strings in data.
  data_profile gather_profile =
    data_profile_builder().cardinality(0).null_probability(0.0).distribution(
      cudf::type_id::INT32, distribution_id::UNIFORM, 1, data_view.size() - 1);
  auto gather_table =
    create_random_table({cudf::type_id::INT32}, row_count{n_rows}, gather_profile);
  gather_table->get_column(0).set_null_mask(rmm::device_buffer{}, 0);

  // Create scatter map by placing 0-index values throughout the gather-map
  auto scatter_data = cudf::sequence(
    matches, cudf::numeric_scalar<int32_t>(0), cudf::numeric_scalar<int32_t>(n_rows / matches));
  auto zero_scalar = cudf::numeric_scalar<int32_t>(0);
  auto table       = cudf::scatter({zero_scalar}, scatter_data->view(), gather_table->view());
  auto gather_map  = table->view().column(0);
  table            = cudf::gather(cudf::table_view({data_view}), gather_map);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto out =
      spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{table->get_column(0)});
  });

  state.add_buffer_size(n_rows, "trc", "Total Rows");
}

NVBENCH_BENCH(bench_random_parse_uri)
  .set_name("Strings")
  .add_int64_axis("num_rows", {512 * 1024, 1 * 1024 * 1024});

NVBENCH_BENCH(bench_parse_uri)
  .set_name("URIStringMix")
  .add_int64_axis("num_rows", {512 * 1024, 1 * 1024 * 1024})
  .add_int64_axis("hit_rate", {5, 50, 100});
