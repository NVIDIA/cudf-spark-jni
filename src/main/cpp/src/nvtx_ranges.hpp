/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include <nvtx3/nvtx3.hpp>

namespace spark_rapids_jni {
/**
 * @brief Tag type for the NVTX domain
 */
struct srj_domain {
  static constexpr char const* name{"srj"};  ///< Name of the domain
};

}  // namespace spark_rapids_jni

/**
 * @brief Convenience macro for generating an NVTX range in the
 * spark_rapids_jni domain for the lifetime of a function.
 *
 * Uses the name of the immediately enclosing function returned by `__func__` to
 * name the range.
 *
 * Example:
 * ```
 * void some_function(){
 *    SRJ_FUNC_RANGE();
 *    ...
 * }
 * ```
 */
#define SRJ_FUNC_RANGE() NVTX3_FUNC_RANGE_IN(spark_rapids_jni::srj_domain)
