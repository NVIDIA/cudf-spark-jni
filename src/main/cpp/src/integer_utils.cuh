/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

#include <cuda/std/type_traits>

namespace spark_rapids_jni {

/**
 * @brief The utilities for integer.
 */
namespace integer_utils {

/**
 * @brief floor division for integer types.
 */
template <typename Type>
__device__ Type floor_div(Type x, Type y)
  requires(cuda::std::is_integral_v<Type> and cuda::std::is_signed_v<Type>)
{
  auto const quotient          = x / y;
  auto const nonzero_remainder = (x % y) != 0;
  auto const mixed_sign        = (x ^ y) < 0;
  return quotient - mixed_sign * nonzero_remainder;
}

/**
 * @brief floor modulo for positive divisors.
 */
template <typename Type>
__device__ Type floor_mod(Type x, Type y)
  requires(cuda::std::is_integral_v<Type> and cuda::std::is_signed_v<Type>)
{
  auto const remainder = x % y;
  return remainder < 0 ? remainder + y : remainder;
}

}  // namespace integer_utils

}  // namespace spark_rapids_jni
