/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "hash/hash.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda/memory_resource>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <optional>
#include <utility>

namespace {

// Restore the previous resource on both normal return and benchmark failure.
class async_resource_scope {
 public:
  async_resource_scope()
    : previous{cudf::set_current_device_resource(rmm::mr::cuda_async_memory_resource{})}
  {
  }
  ~async_resource_scope() { cudf::set_current_device_resource(std::move(previous)); }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> previous;
};

}  // namespace

static void bench_spark_hash_pmod(nvbench::state& state)
{
  // Use a stream-ordered pool, as in Spark; pool setup and input generation are not timed.
  auto current          = async_resource_scope{};
  auto const rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const partitions = static_cast<cudf::size_type>(state.get_int64("partitions"));
  auto const nulls      = state.get_float64("nulls");
  auto const fused      = state.get_string("method") == "fused";
  auto const partition  = state.get_int64("partition") != 0;
  auto const type =
    state.get_string("data_type") == "string" ? cudf::type_id::STRING : cudf::type_id::INT64;
  data_profile const profile = data_profile_builder().null_probability(
    nulls == 0 ? std::nullopt : std::optional<double>{nulls});
  // Ten data columns, with the first three used as partition keys. Data generation is not timed.
  auto const data   = create_random_table(cycle_dtypes({type}, 10), row_count{rows}, profile);
  auto const keys   = data->view().select({0, 1, 2});
  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_element_count(rows);

  auto const memory_stats = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto const map = [&] {
      if (fused) { return spark_rapids_jni::murmur_hash3_32_pmod(keys, partitions, 42, stream); }
      auto const hashes  = spark_rapids_jni::murmur_hash3_32(keys, 42, stream);
      auto const divisor = cudf::numeric_scalar<int32_t>{partitions, true, stream};
      return cudf::binary_operation(hashes->view(),
                                    divisor,
                                    cudf::binary_operator::PMOD,
                                    cudf::data_type{cudf::type_id::INT32},
                                    stream);
    }();
    if (partition) {
      // Includes partition offsets readback, but no packing or file output.
      auto const result = cudf::partition(data->view(), map->view(), partitions, stream);
    }
  });
  state.add_buffer_size(memory_stats.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_spark_hash_pmod)
  .set_name("spark_hash_pmod")
  .add_int64_axis("num_rows", {65'536, 1'000'000, 10'000'000})
  .add_int64_axis("partitions", {100, 200})
  .add_string_axis("data_type", {"string", "int64"})
  .add_float64_axis("nulls", {0.0, 0.1})
  .add_int64_axis("partition", {0, 1})
  .add_string_axis("method", {"separate", "fused"});
