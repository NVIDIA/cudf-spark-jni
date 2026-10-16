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

// Process-global gen-once input cache shared by every get_json_object benchmark in this binary:
// each input is generated and uploaded once, however many cells read it (nvbench runs
// invocations serially, so the map needs no locking).
namespace spark_rapids_jni {
namespace bench {

using input_cache_key = std::uint64_t;

/**
 * @brief Pack (scenario, size_bytes, seed) into one cache key: 16 / 32 / 16 bits respectively.
 *
 * `scenario` distinguishes input shapes at the same byte size (row kind, `max_depth`); `seed`
 * distinguishes generator seeds.
 */
constexpr input_cache_key make_input_cache_key(std::uint64_t scenario,
                                               std::uint64_t size_bytes,
                                               std::uint64_t seed)
{
  return (scenario << 48) | ((size_bytes & 0xFFFF'FFFFULL) << 16) | (seed & 0xFFFFULL);
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
