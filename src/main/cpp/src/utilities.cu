/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/stream>

namespace spark_rapids_jni {

bool is_basic_spark_numeric(cudf::data_type type)
{
  return type.id() == cudf::type_id::INT8 || type.id() == cudf::type_id::INT16 ||
         type.id() == cudf::type_id::INT32 || type.id() == cudf::type_id::INT64 ||
         type.id() == cudf::type_id::FLOAT32 || type.id() == cudf::type_id::FLOAT64;
}

std::unique_ptr<rmm::device_buffer> bitmask_bitwise_or(
  std::vector<cudf::device_span<cudf::bitmask_type const>> const& input,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.size() > 0, "Empty input");
  auto const mask_size = (*input.begin()).size();
  CUDF_EXPECTS(
    std::all_of(
      input.begin(), input.end(), [mask_size](auto mask) { return mask.size() == mask_size; }),
    "Encountered size mismatch in inputs");
  if (mask_size == 0) {
    return std::make_unique<rmm::device_buffer>(rmm::device_buffer{0, stream, mr});
  }

  // move the pointers to the gpu
  std::vector<cudf::bitmask_type const*> h_input(input.size());
  std::transform(
    input.begin(), input.end(), h_input.begin(), [](auto mask) { return mask.data(); });
  auto d_input = cudf::detail::make_device_uvector_async(
    h_input, stream, rmm::mr::get_current_device_resource_ref());

  std::unique_ptr<rmm::device_buffer> out =
    std::make_unique<rmm::device_buffer>(mask_size * sizeof(cudf::bitmask_type), stream, mr);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + mask_size,
                    static_cast<cudf::bitmask_type*>(out->data()),
                    cuda::proclaim_return_type<cudf::bitmask_type>(
                      [buffers     = d_input.data(),
                       num_buffers = input.size()] __device__(cudf::size_type word_index) {
                        cudf::bitmask_type out = buffers[0][word_index];
                        for (auto idx = 1; idx < num_buffers; idx++) {
                          out |= buffers[idx][word_index];
                        }
                        return out;
                      }));

  return out;
}

}  // namespace spark_rapids_jni
