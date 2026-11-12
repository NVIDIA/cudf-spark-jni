/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/gjo_input_cache.hpp"
#include "common/gjo_scenario_generators.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <get_json_object.hpp>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using namespace spark_rapids_jni::gjo_scenarios;

// Benchmarks the nested-path shapes and the flat (mixed and wide) shapes with reproducible row
// seeds.

namespace {

// Byte unit for the size->rows derivation below (rows sized at ~10 KiB each).
constexpr std::size_t ki_b = 1'024;

// Nested layouts served by gjo_scenario_generators.hpp — NESTED_135 (the
// largest fused call, 135 distinct paths), NESTED_33 (the narrower
// per-branch extraction, 33 paths), FLAT_MIXED_16 (15 top-level fields
// plus one depth-two path: the mixed flat+nested shape admission must not
// pull onto the merged route), FLAT_MIXED_64 (64 distinct depth-two paths
// f{root}.c{child} over 2 roots of 32 children each, on flat_mixed-style
// 150-member rows: a wide flat+nested shape), FLAT_600 (15 top-level paths
// over 600-member flat rows: the wide flat row shape), and FLAT_WIDE_124
// (124 depth-1 paths over 1 root: the merged-route wide-flat shape).
enum class scenario_id : int32_t {
  NESTED_135    = 0,
  NESTED_33     = 1,
  FLAT_MIXED_16 = 2,
  FLAT_WIDE_124 = 3,
  VALUE_KINDS   = 4,
  FLAT_MIXED_64 = 5,
  FLAT_600      = 6
};

}  // namespace

// Pool allocation avoids synchronizing cudaFree costs absent from deployment.
namespace {
rmm::mr::pool_memory_resource pool_mr{rmm::mr::cuda_memory_resource{},
                                      rmm::percent_of_free_device_memory(50)};
auto* pool_guard = [] {
  cudf::set_current_device_resource(pool_mr);
  return &pool_mr;
}();
}  // namespace

void BM_get_json_object_multiple_paths(nvbench::state& state)
{
  auto const size_bytes = static_cast<std::size_t>(state.get_int64("size_bytes"));
  auto const num_paths  = static_cast<std::size_t>(state.get_int64("num_paths"));
  auto const seed       = static_cast<uint64_t>(state.get_int64("seed"));
  auto const scenario   = static_cast<scenario_id>(state.get_int64("scenario"));
  auto const budget     = state.get_int64("memory_budget_bytes");
  auto const override_r = state.get_int64("parallel_override");

  auto const& table = scenario == scenario_id::NESTED_135 ? nested_table_135() : nested_table_33();
  multi_paths base_paths;
  std::size_t members = 0;
  if (scenario == scenario_id::FLAT_MIXED_16) {
    // The mixed shape fixes 16 paths and varies the member count: the num_paths=50 cell
    // builds rows of 50 top-level members, every other num_paths value 150 members.
    base_paths = flat_mixed_build_paths();
    members    = num_paths == 50 ? 50 : 150;
  } else if (scenario == scenario_id::FLAT_WIDE_124) {
    // The wide-flat merged-route shape fixes 124 depth-one paths over one root; num_paths does
    // not apply to it.
    base_paths = flat_wide_124_build_paths();
  } else if (scenario == scenario_id::VALUE_KINDS) {
    base_paths = value_kinds_build_paths();
  } else if (scenario == scenario_id::FLAT_MIXED_64) {
    // The wide flat+nested shape fixes 64 depth-two paths over 2 roots and the 150-member
    // layout; the num_paths axis does not apply to it.
    base_paths = flat_mixed_64_build_paths();
    members    = 150;
  } else if (scenario == scenario_id::FLAT_600) {
    // The wide flat shape fixes 15 top-level paths and the 600-member layout; the num_paths
    // axis does not apply to it.
    base_paths = flat_600_build_paths();
    members    = 600;
  } else {
    base_paths = nested_build_paths(table, scenario == scenario_id::NESTED_135 ? 135 : 33);
  }
  auto const fixed_paths =
    scenario == scenario_id::FLAT_MIXED_16 || scenario == scenario_id::FLAT_WIDE_124 ||
    scenario == scenario_id::FLAT_MIXED_64 || scenario == scenario_id::FLAT_600 ||
    scenario == scenario_id::VALUE_KINDS;
  multi_paths tiled;
  tiled.reserve(fixed_paths ? base_paths.size() : num_paths);
  for (std::size_t i = 0; i < (fixed_paths ? base_paths.size() : num_paths); ++i) {
    tiled.push_back(base_paths[i % base_paths.size()]);
  }

  // Keep generated rows parseable at small requested sizes.
  auto const num_rows  = std::max<std::size_t>(size_bytes / (10 * ki_b), 8);
  auto const row_bytes = std::max<std::size_t>(size_bytes / num_rows, 16);

  // The flat rows depend on the member count (the 16-path mixed shape derives it from
  // `num_paths`; the other flat scenarios fix it), so the cache key must distinguish member
  // counts via the flavor field; the nested scenarios derive rows from the table alone
  // (FLAT_WIDE_124's rows keep their fixed 124 members for every axis value).
  auto const input = spark_rapids_jni::bench::get_or_generate_input(
    spark_rapids_jni::bench::make_input_cache_key(
      static_cast<std::uint64_t>(scenario), size_bytes, seed, members),
    [&] {
      std::mt19937_64 rng{seed};
      auto rows = scenario == scenario_id::FLAT_MIXED_16
                    ? flat_mixed_build_rows(rng, num_rows, members, row_bytes, seed)
                  : scenario == scenario_id::FLAT_WIDE_124
                    ? flat_wide_124_build_rows(rng, num_rows, row_bytes, seed)
                  : scenario == scenario_id::FLAT_MIXED_64
                    ? flat_mixed_64_build_rows(rng, num_rows, members, row_bytes, seed)
                  : scenario == scenario_id::FLAT_600
                    ? flat_600_build_rows(rng, num_rows, members, row_bytes, seed)
                  : scenario == scenario_id::VALUE_KINDS
                    ? value_kinds_build_rows(rng, num_rows, row_bytes, seed)
                    : nested_build_rows(rng, table, num_rows, row_bytes, seed);
      return cudf::test::strings_column_wrapper(rows.begin(), rows.end()).release();
    });

  // Report actual generated bytes rather than the target `size_bytes`.
  auto const input_bytes = cudf::strings_column_view{input}.chars_size(cudf::get_default_stream());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_global_memory_reads<nvbench::int8_t>(input_bytes);
  state.add_element_count(num_rows, "rows");
  state.add_element_count(fixed_paths ? base_paths.size() : num_paths, "paths");
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    [[maybe_unused]] auto const output = spark_rapids_jni::get_json_object_multiple_paths(
      cudf::strings_column_view{input}, tiled, budget, override_r);
  });
}

NVBENCH_BENCH(BM_get_json_object_multiple_paths)
  .set_name("get_json_object_multiple_paths")
  .add_int64_axis("scenario",
                  {static_cast<nvbench::int64_t>(scenario_id::NESTED_135),
                   static_cast<nvbench::int64_t>(scenario_id::NESTED_33),
                   static_cast<nvbench::int64_t>(scenario_id::FLAT_MIXED_16),
                   static_cast<nvbench::int64_t>(scenario_id::FLAT_WIDE_124),
                   static_cast<nvbench::int64_t>(scenario_id::FLAT_MIXED_64),
                   static_cast<nvbench::int64_t>(scenario_id::FLAT_600),
                   static_cast<nvbench::int64_t>(scenario_id::VALUE_KINDS)})
  .add_int64_axis("size_bytes", {10 * 1'024 * 1'024, 100 * 1'024 * 1'024})
  .add_int64_axis("num_paths", {1, 8, 32, 33, 50, 135})
  .add_int64_axis("seed", {1, 2})
  .add_int64_axis("memory_budget_bytes", {-1, 4LL * 1'024 * 1'024 * 1'024})
  // parallel_override exercises the fixed-size batching path: -1 takes the scratch-size
  // budget loop (or collapses to one batch when budget <= 0), 123 forces 123-path batches.
  // NESTED_135 then runs two batches whose shallow families may admit differently per batch,
  // so that cell's pair is NOT route-identical and must not be read as an A/B.
  .add_int64_axis("parallel_override", {-1, 123});
