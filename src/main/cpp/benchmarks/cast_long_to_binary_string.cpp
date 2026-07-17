/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cudf/io/types.hpp>

#include <cast_string.hpp>
#include <nvbench/nvbench.cuh>

static void long_to_binary_string(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));

  auto const input_table = create_random_table({cudf::type_id::INT64}, row_count{num_rows});
  auto const long_col    = input_table->get_column(0);
  auto const stream      = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    spark_rapids_jni::long_to_binary_string(long_col, stream);
  });
}

NVBENCH_BENCH(long_to_binary_string)
  .set_name("Long to Binary String Cast")
  .add_int64_axis("num_rows", {100'000, 500'000, 1'000'000, 5'000'000, 10'000'000});
