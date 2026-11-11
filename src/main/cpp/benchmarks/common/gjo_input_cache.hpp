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
#include <tuple>
#include <unordered_map>
#include <utility>

// Callers of get_or_generate_input share one process-wide cache; each key is generated once.
// Benchmark invocations are serial, so the map needs no locking.
namespace spark_rapids_jni {
namespace bench {

using input_cache_key = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t>;

// std::hash has no tuple overload, so the shared map carries its own hasher: each field is
// diffused (splitmix64 finalizer) and FNV-combined; a collision only costs one regeneration in
// this benchmark-local map.
struct input_cache_key_hash {
  std::size_t operator()(input_cache_key const& key) const noexcept
  {
    auto const mix = [](std::uint64_t v) {
      v ^= v >> 33;
      v *= 0xff51afd7ed558ccdULL;
      v ^= v >> 33;
      return v;
    };
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (std::uint64_t v :
         {std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key)}) {
      h = (h ^ mix(v)) * 0x100000001b3ULL;
    }
    return static_cast<std::size_t>(h);
  }
};

using input_cache_map =
  std::unordered_map<input_cache_key, std::unique_ptr<cudf::column>, input_cache_key_hash>;

/**
 * @brief Build the process-cache key `(scenario, size_bytes, seed, flavor)`.
 *
 * Every field is carried at full width: `size_bytes` is NOT truncated to KiB units, so
 * distinct requested byte counts never share an entry even when they round to the same
 * KiB value. `scenario` distinguishes input shapes at the same byte size (row kind,
 * `max_depth`); `flavor` distinguishes row-shape variants within one scenario (e.g. the
 * flat shapes' member counts); `seed` distinguishes generator seeds.
 */
constexpr input_cache_key make_input_cache_key(std::uint64_t scenario,
                                               std::uint64_t size_bytes,
                                               std::uint64_t seed,
                                               std::uint64_t flavor = 0)
{
  return {scenario, size_bytes, seed, flavor};
}

/**
 * @brief Return a reference to the process-cached input column map for `key` lookups.
 *
 * The map is heap-allocated and never destroyed on purpose: freeing the cached device memory at
 * static teardown crashes (the CUDA runtime has already started exiting), so the columns leak
 * and the driver reclaims them at process exit. Defined in `gjo_input_cache.cpp` so every
 * translation unit shares one map.
 */
std::unordered_map<input_cache_key, std::unique_ptr<cudf::column>, input_cache_key_hash>&
input_cache();

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
