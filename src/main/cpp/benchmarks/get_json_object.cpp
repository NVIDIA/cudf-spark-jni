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
#include <cudf/utilities/error.hpp>

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
#include <span>
#include <utility>

using namespace spark_rapids_jni::gjo_scenarios;

namespace {

constexpr std::size_t ki_b = 1'024;

// Stable NVBench scenario IDs. FLAT_MIXED_16 probes mixed-depth admission; FLAT_WIDE_124
// probes the flat merged route. FLAT_WIDE_768 reaches the shared-memory record cap.
enum class scenario_id : int32_t {
  NESTED_135    = 0,
  NESTED_33     = 1,
  FLAT_MIXED_16 = 2,
  FLAT_WIDE_124 = 3,
  VALUE_KINDS   = 4,
  FLAT_MIXED_64 = 5,
  FLAT_600      = 6,
  FLAT_WIDE_150 = 7,
  FLAT_WIDE_600 = 8,
  FLAT_WIDE_2R  = 9,
  FLAT_WIDE_768 = 10,
  SIBLING_RICH  = 11,
  NESTED_DENSE  = 12
};

// Only nested tables tile; 8-path wide and sibling-rich cells cover single-root denial.
inline std::span<int64_t const> scenario_num_paths(scenario_id scenario)
{
  static constexpr int64_t nested[]{1, 8, 32, 33, 50, 135};
  static constexpr int64_t paths_15[]{15}, paths_16[]{16}, paths_64[]{64}, paths_124[]{124};
  static constexpr int64_t wide_150[]{8, 124}, wide_2r[]{16, 64}, wide_768[]{768};
  static constexpr int64_t sibling_rich[]{8, 16, 124}, nested_dense[]{8, 16};
  switch (scenario) {
    case scenario_id::NESTED_135:
    case scenario_id::NESTED_33: return nested;
    case scenario_id::FLAT_MIXED_16:
    case scenario_id::VALUE_KINDS: return paths_16;
    case scenario_id::FLAT_MIXED_64: return paths_64;
    case scenario_id::FLAT_600: return paths_15;
    case scenario_id::FLAT_WIDE_124:
    case scenario_id::FLAT_WIDE_600: return paths_124;
    case scenario_id::FLAT_WIDE_150: return wide_150;
    case scenario_id::FLAT_WIDE_2R: return wide_2r;
    case scenario_id::FLAT_WIDE_768: return wide_768;
    case scenario_id::SIBLING_RICH: return sibling_rich;
    case scenario_id::NESTED_DENSE: return nested_dense;
  }
  CUDF_FAIL("unhandled scenario");
}

// Pool allocation avoids synchronizing cudaFree costs absent from deployment.
rmm::mr::pool_memory_resource pool_mr{rmm::mr::cuda_memory_resource{},
                                      rmm::percent_of_free_device_memory(50)};
auto* pool_guard = [] {
  cudf::set_current_device_resource(pool_mr);
  return &pool_mr;
}();
}  // namespace

void BM_get_json_object_multiple_paths(nvbench::state& state)
{
  auto const size_bytes        = static_cast<std::size_t>(state.get_int64("size_bytes"));
  auto const num_paths         = static_cast<std::size_t>(state.get_int64("num_paths"));
  auto const seed              = static_cast<uint64_t>(state.get_int64("seed"));
  auto const scenario          = static_cast<scenario_id>(state.get_int64("scenario"));
  auto const budget            = state.get_int64("memory_budget_bytes");
  auto const override_r        = state.get_int64("parallel_override");
  auto const member_count      = static_cast<std::size_t>(state.get_int64("member_count"));
  auto const nested_dense_deep = state.get_int64("nested_dense_deep");

  auto const consumed = scenario_num_paths(scenario);
  if (std::find(consumed.begin(), consumed.end(), state.get_int64("num_paths")) == consumed.end()) {
    state.skip("num_paths value does not apply to this scenario");
    return;
  }
  auto const mixed = scenario == scenario_id::FLAT_MIXED_16;
  auto const dense = scenario == scenario_id::SIBLING_RICH || scenario == scenario_id::NESTED_DENSE;
  // Only dense families and the two mixed layouts consume nonzero member counts.
  auto const valid_members = mixed   ? member_count == 50 || member_count == 150
                             : dense ? member_count != 0
                                     : member_count == 0;
  if (!valid_members) {
    state.skip("member_count value does not apply to this scenario");
    return;
  }
  // Skip unused depth values to avoid duplicate cells.
  if (scenario != scenario_id::NESTED_DENSE && nested_dense_deep != 0) {
    state.skip("nested_dense_deep value does not apply to this scenario");
    return;
  }

  auto const& table = scenario == scenario_id::NESTED_135 ? nested_table_135() : nested_table_33();
  auto const deep   = scenario == scenario_id::NESTED_DENSE && nested_dense_deep != 0;
  auto const [num_roots, rooted_members] = [scenario]() -> std::pair<std::size_t, std::size_t> {
    switch (scenario) {
      case scenario_id::FLAT_WIDE_150: return {1, 150};
      case scenario_id::FLAT_WIDE_600: return {1, 600};
      case scenario_id::FLAT_WIDE_2R: return {2, 150};
      case scenario_id::FLAT_WIDE_768: return {4, 200};
      default: return {};
    }
  }();
  multi_paths tiled;
  std::size_t members = rooted_members;
  if (mixed) {
    tiled   = flat_mixed_build_paths();
    members = member_count;
  } else if (scenario == scenario_id::FLAT_WIDE_124) {
    tiled = flat_wide_124_build_paths();
  } else if (scenario == scenario_id::VALUE_KINDS) {
    tiled = value_kinds_build_paths();
  } else if (scenario == scenario_id::FLAT_MIXED_64) {
    tiled   = flat_mixed_64_build_paths();
    members = 150;
  } else if (scenario == scenario_id::FLAT_600) {
    tiled   = flat_600_build_paths();
    members = 600;
  } else if (num_roots != 0) {
    tiled = flat_wide_rooted_build_paths(num_paths, num_roots);
  } else if (scenario == scenario_id::SIBLING_RICH) {
    tiled = sibling_rich_build_paths(num_paths);
  } else if (scenario == scenario_id::NESTED_DENSE) {
    tiled = nested_dense_build_paths(num_paths, deep);
  } else {
    auto const base_paths =
      nested_build_paths(table, scenario == scenario_id::NESTED_135 ? 135 : 33);
    tiled.reserve(num_paths);
    for (std::size_t i = 0; i < num_paths; ++i) {
      tiled.push_back(base_paths[i % base_paths.size()]);
    }
  }

  // Dense bands size by member count: padding to ~10 KiB could admit rows meant to deny.
  // For other shapes, keep small requests parseable with minimum row count and width.
  auto const num_rows = std::max<std::size_t>(size_bytes / (10 * ki_b), 8);
  auto const row_bytes =
    dense ? dense_band_row_bytes(member_count) : std::max<std::size_t>(size_bytes / num_rows, 16);
  if (dense) { expect_dense_band_row_bytes(member_count, row_bytes); }

  // Cache flavor identifies row variants: rooted widths by path count, dense widths by member
  // and path counts (plus depth for NESTED_DENSE), and other layouts by member count.
  auto const flavor = scenario == scenario_id::SIBLING_RICH ? member_count * 1024 + num_paths
                      : scenario == scenario_id::NESTED_DENSE
                        ? member_count * 2048 + num_paths * 2 + (deep ? 1 : 0)
                        : (num_roots != 0 ? num_paths : members);
  auto const input  = spark_rapids_jni::bench::get_or_generate_input(
    spark_rapids_jni::bench::make_input_cache_key(
      static_cast<std::uint64_t>(scenario), size_bytes, seed, flavor),
    [&] {
      std::mt19937_64 rng{seed};
      auto rows =
        mixed ? flat_mixed_build_rows(rng, num_rows, members, row_bytes, seed)
         : scenario == scenario_id::FLAT_WIDE_124
           ? flat_wide_124_build_rows(rng, num_rows, row_bytes, seed)
         : scenario == scenario_id::FLAT_MIXED_64
           ? flat_mixed_64_build_rows(rng, num_rows, members, row_bytes, seed)
         : scenario == scenario_id::FLAT_600
           ? flat_600_build_rows(rng, num_rows, members, row_bytes, seed)
         : num_roots != 0
           ? flat_wide_rooted_build_rows(rng, num_rows, members, num_roots, tiled, row_bytes)
         : scenario == scenario_id::SIBLING_RICH
           ? sibling_rich_build_rows(rng, num_rows, member_count, num_paths, row_bytes)
         : scenario == scenario_id::NESTED_DENSE
           ? nested_dense_build_rows(rng, num_rows, member_count, num_paths, row_bytes, deep)
         : scenario == scenario_id::VALUE_KINDS
           ? value_kinds_build_rows(rng, num_rows, row_bytes, seed)
           : nested_build_rows(rng, table, num_rows, row_bytes, seed);
      return cudf::test::strings_column_wrapper(rows.begin(), rows.end()).release();
    });

  // Natural row widths can exceed the sizing target.
  auto const input_bytes = cudf::strings_column_view{input}.chars_size(cudf::get_default_stream());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_global_memory_reads<nvbench::int8_t>(input_bytes);
  state.add_element_count(num_rows, "rows");
  state.add_element_count(tiled.size(), "paths");
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
                   static_cast<nvbench::int64_t>(scenario_id::VALUE_KINDS),
                   static_cast<nvbench::int64_t>(scenario_id::FLAT_WIDE_150),
                   static_cast<nvbench::int64_t>(scenario_id::FLAT_WIDE_600),
                   static_cast<nvbench::int64_t>(scenario_id::FLAT_WIDE_2R),
                   static_cast<nvbench::int64_t>(scenario_id::FLAT_WIDE_768),
                   static_cast<nvbench::int64_t>(scenario_id::SIBLING_RICH),
                   static_cast<nvbench::int64_t>(scenario_id::NESTED_DENSE)})
  .add_int64_axis("member_count", {0, 50, 150, 169, 340, 600})
  .add_int64_axis("nested_dense_deep", {0, 1})
  .add_int64_axis("size_bytes", {10 * 1'024 * 1'024, 100 * 1'024 * 1'024})
  .add_int64_axis("num_paths", {1, 8, 16, 32, 33, 50, 64, 124, 135, 150, 768})
  .add_int64_axis("seed", {1, 2})
  .add_int64_axis("memory_budget_bytes", {-1, 4LL * 1'024 * 1'024 * 1'024})
  // -1 uses the scratch budget (one batch if nonpositive); 123 forces 123-path batches.
  // At 135 paths, the split may change routing; this is not a route-identical A/B.
  .add_int64_axis("parallel_override", {-1, 123});
