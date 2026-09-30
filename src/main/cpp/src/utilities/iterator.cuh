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

#include <cudf/types.hpp>

#include <cuda/iterator>
#include <thrust/iterator/transform_iterator.h>

namespace spark_rapids_jni::util {

/**
 * @brief Convenience wrapper for creating a `thrust::transform_iterator` over a
 * `cuda::counting_iterator` within the range [0, INT_MAX].
 * Note: This is practically identical to `cudf::detail::make_counting_transform_iterator`,
 * except that it uses `cuda::counting_iterator` instead of `thrust::counting_iterator`.
 *
 * Example:
 * @code{.cpp}
 * // Returns square of the value of the counting iterator
 * auto iter = make_counting_transform_iterator(0, [](auto i){ return (i * i);});
 * iter[0] == 0
 * iter[1] == 1
 * iter[2] == 4
 * ...
 * iter[n] == n * n
 * @endcode
 *
 * @param start The starting value of the counting iterator (must be size_type or smaller type).
 * @param f The unary function to apply to the counting iterator.
 * @return A transform iterator that applies `f` to a counting iterator
 */
template <typename CountingIterType, typename UnaryFunction>
CUDF_HOST_DEVICE inline auto make_counting_transform_iterator(CountingIterType start,
                                                              UnaryFunction f)
{
  // Check if the `start` for counting_iterator is of size_type or a smaller integral type
  static_assert(
    cuda::std::is_integral_v<CountingIterType> and
      cuda::std::numeric_limits<CountingIterType>::digits <=
        cuda::std::numeric_limits<cudf::size_type>::digits,
    "The `start` for the counting_transform_iterator must be size_type or smaller type");

  return thrust::make_transform_iterator(cuda::make_counting_iterator(start), f);
}

}  // namespace spark_rapids_jni::util
