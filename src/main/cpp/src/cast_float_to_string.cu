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

#include "cast_string.hpp"
#include "ftos_converter.cuh"
#include "nvtx_ranges.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace spark_rapids_jni {

namespace detail {
namespace {

template <typename FloatType, bool json_string>
struct float_to_string_fn {
  cudf::column_device_view d_floats;
  cudf::size_type* d_sizes;
  char* d_chars;
  cudf::detail::input_offsetalator d_offsets;

  __device__ cudf::size_type compute_output_size(cudf::size_type idx) const
  {
    auto const value = d_floats.element<FloatType>(idx);
    return static_cast<cudf::size_type>(
      ftos_converter::compute_ftos_size<FloatType, json_string>(value));
  }

  __device__ void float_to_string(cudf::size_type idx) const
  {
    auto const value  = d_floats.element<FloatType>(idx);
    auto const output = d_chars + d_offsets[idx];
    ftos_converter::float_to_string<FloatType, json_string>(value, output);
  }

  __device__ void operator()(cudf::size_type idx) const
  {
    if (d_floats.is_null(idx)) {
      if (d_chars == nullptr) { d_sizes[idx] = 0; }
      return;
    }
    if (d_chars != nullptr) {
      float_to_string(idx);
    } else {
      d_sizes[idx] = compute_output_size(idx);
    }
  }
};

/**
 * @brief This dispatch method is for converting floats into strings.
 *
 * The template function declaration ensures only float types are allowed.
 */
template <bool json_string>
struct dispatch_float_to_string_fn {
  template <typename FloatType, CUDF_ENABLE_IF(cuda::std::is_floating_point_v<FloatType>)>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& floats,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
  {
    auto const strings_count = floats.size();
    if (strings_count == 0) { return cudf::make_empty_column(cudf::type_id::STRING); }

    auto const input_ptr = cudf::column_device_view::create(floats, stream);

    auto [offsets, chars] = cudf::strings::detail::make_strings_children(
      float_to_string_fn<FloatType, json_string>{*input_ptr}, strings_count, stream, mr);

    return make_strings_column(strings_count,
                               std::move(offsets),
                               chars.release(),
                               floats.null_count(),
                               cudf::copy_bitmask(floats, stream, mr));
  }

  // non-float types throw an exception
  template <typename T, CUDF_ENABLE_IF(not cuda::std::is_floating_point_v<T>)>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const&,
                                           rmm::cuda_stream_view,
                                           rmm::device_async_resource_ref)
  { CUDF_FAIL("Values for float_to_string function must be a float type."); }
};

}  // namespace

// This will convert all float column types into a strings column.
std::unique_ptr<cudf::column> float_to_string(cudf::column_view const& floats,
                                              bool json_string,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  return json_string
           ? type_dispatcher(floats.type(), dispatch_float_to_string_fn<true>{}, floats, stream, mr)
           : type_dispatcher(
               floats.type(), dispatch_float_to_string_fn<false>{}, floats, stream, mr);
}

}  // namespace detail

// external API
std::unique_ptr<cudf::column> float_to_string(cudf::column_view const& floats,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{ return float_to_string(floats, false, stream, mr); }

std::unique_ptr<cudf::column> float_to_string(cudf::column_view const& floats,
                                              bool json_string,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return detail::float_to_string(floats, json_string, stream, mr);
}

}  // namespace spark_rapids_jni
