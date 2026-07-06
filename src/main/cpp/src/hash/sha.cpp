/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <hash/hash.hpp>

namespace {
using HashFunction = std::unique_ptr<cudf::column> (*)(cudf::table_view const&,
                                                       rmm::cuda_stream_view,
                                                       rmm::device_async_resource_ref);

std::unique_ptr<cudf::column> sha_impl(HashFunction hash_function,
                                       cudf::column_view const& input,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}); }

  if (input.has_nulls()) {
    // Using the tmp memory resource, because `hash_from_cudf` is a temporary column
    // that will be re-copied when purged of non-empty nulls.
    auto hash_from_cudf =
      hash_function(cudf::table_view{{input}}, stream, cudf::get_current_device_resource_ref());
    hash_from_cudf->set_null_mask(cudf::copy_bitmask(input, stream), input.null_count());
    return cudf::purge_nonempty_nulls(*hash_from_cudf, stream, mr);
  } else {
    // Using the provided memory resource, because `hash_from_cudf` is not a temporary.
    return hash_function(cudf::table_view{{input}}, stream, mr);
  }
}
}  // namespace

namespace spark_rapids_jni {

std::unique_ptr<cudf::column> sha224_nulls_preserved(cudf::column_view const& input,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{ return sha_impl(cudf::hashing::sha224, input, stream, mr); }

std::unique_ptr<cudf::column> sha256_nulls_preserved(cudf::column_view const& input,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{ return sha_impl(cudf::hashing::sha256, input, stream, mr); }

std::unique_ptr<cudf::column> sha384_nulls_preserved(cudf::column_view const& input,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{ return sha_impl(cudf::hashing::sha384, input, stream, mr); }

std::unique_ptr<cudf::column> sha512_nulls_preserved(cudf::column_view const& input,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{ return sha_impl(cudf::hashing::sha512, input, stream, mr); }
}  // namespace spark_rapids_jni
