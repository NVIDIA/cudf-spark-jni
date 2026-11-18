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

#include "common/gjo_input_cache.hpp"

namespace spark_rapids_jni {
namespace bench {

std::unordered_map<input_cache_key, std::unique_ptr<cudf::column>>& input_cache()
{
  // Heap-allocated and never destroyed on purpose: freeing the cached device memory at static
  // teardown crashes (the CUDA runtime has already started exiting), so the columns leak and the
  // driver reclaims them at process exit.
  static auto* cache = new std::unordered_map<input_cache_key, std::unique_ptr<cudf::column>>();
  return *cache;
}

}  // namespace bench
}  // namespace spark_rapids_jni
