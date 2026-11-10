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

#pragma once

#include <cudf/column/column.hpp>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>

// Callers of get_or_generate_input share one process-wide cache; each key is generated once.
// Benchmark invocations are serial, so the map needs no locking.
namespace spark_rapids_jni {
namespace bench {

using input_cache_key = std::uint64_t;

/**
 * @brief Pack (scenario, flavor, size_bytes, seed) into one cache key:
 * 8 / 8 / 32 / 16 bits at shifts 56 / 48 / 16 / 0 respectively.
 *
 * `scenario` distinguishes input shapes at the same byte size (row kind, `max_depth`); `flavor`
 * distinguishes row-shape variants within one scenario (e.g. the mixed shape's member count);
 * `size_bytes` is packed in KiB units so 32 bits cover up to 4 TiB; `seed` distinguishes
 * generator seeds. All fields are disjoint, so distinct tuples map to distinct keys.
 */
constexpr input_cache_key make_input_cache_key(std::uint64_t scenario,
                                               std::uint64_t size_bytes,
                                               std::uint64_t seed,
                                               std::uint64_t flavor = 0)
{
  return (scenario & 0xFFULL) << 56 | (flavor & 0xFFULL) << 48 |
         ((size_bytes >> 10) & 0xFFFF'FFFFULL) << 16 | (seed & 0xFFFFULL);
}

/**
 * @brief Return a reference to the process-cached input column map for `key` lookups.
 *
 * The map is heap-allocated and never destroyed on purpose: freeing the cached device memory at
 * static teardown crashes (the CUDA runtime has already started exiting), so the columns leak
 * and the driver reclaims them at process exit. Defined in `gjo_input_cache.cpp` so every
 * translation unit shares one map.
 */
std::unordered_map<input_cache_key, std::unique_ptr<cudf::column>>& input_cache();

template <typename Generate>
cudf::column_view get_or_generate_input(input_cache_key key, Generate&& generate)
{
  auto& cache = input_cache();
  auto it     = cache.find(key);
  if (it == cache.end()) { it = cache.try_emplace(key, generate()).first; }
  return it->second->view();
}

}  // namespace bench
}  // namespace spark_rapids_jni
