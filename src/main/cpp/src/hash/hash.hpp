/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

namespace spark_rapids_jni {

constexpr int64_t DEFAULT_XXHASH64_SEED = 42;
constexpr int MAX_STACK_DEPTH           = 8;

/**
 * @brief Computes the murmur32 hash value of each row in the input set of columns.
 *
 * @param input The table of columns to hash
 * @param seed Optional seed value to use for the hash function
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a column from the input.
 */
std::unique_ptr<cudf::column> murmur_hash3_32(
  cudf::table_view const& input,
  uint32_t seed                     = 0,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Computes the xxhash64 hash value of each row in the input set of columns.
 *
 * @param input The table of columns to hash
 * @param seed Optional seed value to use for the hash function
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a column from the input.
 */
std::unique_ptr<cudf::column> xxhash64(
  cudf::table_view const& input,
  int64_t seed                      = DEFAULT_XXHASH64_SEED,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Computes the Hive hash value of each row in the input set of columns.
 *
 * @param input The table of columns to hash
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a column from the input.
 */
std::unique_ptr<cudf::column> hive_hash(
  cudf::table_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Computes the SHA-224 hash value of each row in the input set of columns.
 * Differs from cudf::hashing::sha224 in that it returns null output rows for null input rows.
 *
 * @param input The column to hash
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a column from the input.
 */
std::unique_ptr<cudf::column> sha224_nulls_preserved(
  cudf::column_view const& input,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Computes the SHA-256 hash value of each row in the input set of columns.
 * Differs from cudf::hashing::sha256 in that it returns null output rows for null input rows.
 *
 * @param input The column to hash
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a column from the input.
 */
std::unique_ptr<cudf::column> sha256_nulls_preserved(
  cudf::column_view const& input,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Computes the SHA-384 hash value of each row in the input set of columns.
 * Differs from cudf::hashing::sha384 in that it returns null output rows for null input rows.
 *
 * @param input The column to hash
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a column from the input.
 */
std::unique_ptr<cudf::column> sha384_nulls_preserved(
  cudf::column_view const& input,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Computes the SHA-512 hash value of each row in the input set of columns.
 * Differs from cudf::hashing::sha512 in that it returns null output rows for null input rows.
 *
 * @param input The column to hash
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a column from the input.
 */
std::unique_ptr<cudf::column> sha512_nulls_preserved(
  cudf::column_view const& input,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
