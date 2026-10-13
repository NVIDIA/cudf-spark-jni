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

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/cmath>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/utility>

namespace spark_rapids_jni::util {

/**
 * @brief Round a nonnegative value up to a positive multiple, checking for overflow.
 *
 * Retains the host exception and device failure behavior needed by allocation-size callers;
 * `cuda::round_up` alone requires its result to be representable.
 */
template <typename T>
CUDF_HOST_DEVICE constexpr T round_up_safe(T value, T multiple)
{
  if (cuda::ceil_div(value, multiple) > cuda::std::numeric_limits<T>::max() / multiple) {
#ifndef __CUDA_ARCH__
    CUDF_FAIL("Attempt to round up beyond the type's maximum value", cudf::data_type_error);
#else
    assert(false && "Unreachable: Attempt to round up beyond the type's maximum value");
    cuda::std::unreachable();
#endif
  }
  return cuda::round_up(value, multiple);
}

}  // namespace spark_rapids_jni::util
