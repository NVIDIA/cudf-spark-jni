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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace spark_rapids_jni {

/**
 * @brief Compute Iceberg bucket assignments for the input column.
 *
 * The bucket transform computes a hash-based bucket assignment for partitioning.
 * For a value v, the bucket is computed as: (hash(v) & Integer.MAX_VALUE) % numBuckets
 *
 * This matches the Iceberg bucket transform specification:
 * https://iceberg.apache.org/spec/#bucket-transform-details
 *
 * Supported input types:
 * - Integer types (INT32, INT64)
 * - Decimal types (DECIMAL32, DECIMAL64, DECIMAL128)
 * - Date types (TIMESTAMP_DAYS)
 * - Timestamp types (TIMESTAMP_MICROSECONDS)
 * - String types
 * - Binary types (LIST of UINT8)
 *
 * @param input Column to compute bucket assignments for
 * @param num_buckets Number of buckets (must be positive)
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource used to allocate the returned column
 *
 * @return INT32 column containing bucket assignments (0 to numBuckets-1), with nulls preserved
 */
std::unique_ptr<cudf::column> compute_bucket(
  cudf::column_view const& input,
  int32_t num_buckets,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
