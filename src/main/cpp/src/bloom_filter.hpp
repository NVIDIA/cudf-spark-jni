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

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

namespace spark_rapids_jni {

constexpr int bloom_filter_version_1 = 1;
constexpr int bloom_filter_version_2 = 2;

// V1 header: [version, num_hashes, num_longs] — 12 bytes
struct bloom_filter_header_v1 {
  int version;
  int num_hashes;
  int num_longs;
};
constexpr int bloom_filter_header_v1_size_bytes = sizeof(bloom_filter_header_v1);
static_assert(bloom_filter_header_v1_size_bytes == 12, "V1 header size must be 12 bytes");

// V2 header: [version, num_hashes, seed, num_longs] — 16 bytes
struct bloom_filter_header_v2 {
  int version;
  int num_hashes;
  int seed;
  int num_longs;
};
constexpr int bloom_filter_header_v2_size_bytes = sizeof(bloom_filter_header_v2);
static_assert(bloom_filter_header_v2_size_bytes == 16, "V2 header size must be 16 bytes");

// Unified header used internally after unpacking from either format.
// Seed is not stored here; for V2 it is returned separately from unpack_bloom_filter.
struct bloom_filter_header {
  int version;
  int num_hashes;
  int num_longs;
};

inline int bloom_filter_header_size_for_version(int version)
{
  return version == bloom_filter_version_2 ? bloom_filter_header_v2_size_bytes
                                           : bloom_filter_header_v1_size_bytes;
}

/**
 * @brief Create an empty bloom filter of the specified size and parameters.
 *
 * The bloom filter is stored in a cudf list_scalar as a single byte buffer. The buffer
 * layout is Spark-compatible: a version-specific header (big-endian) followed by the
 * bit array. V1 header is 12 bytes (version, num_hashes, num_longs). V2 header is 16 bytes
 * (version, num_hashes, seed, num_longs). The remainder of the buffer is
 * bloom_filter_longs * 8 bytes of bit data, also written in big-endian order for Spark
 * interchange.
 *
 * @param version Bloom filter format version: 1 or 2 (e.g. bloom_filter_version_1,
 *        bloom_filter_version_2). V2 uses 64-bit hash indexing and supports a configurable
 *        seed for better distribution on large filters.
 * @param num_hashes Number of bit positions set (and checked) per key. Derived from two
 *        underlying hashes; higher values reduce false positives but increase work per
 *        put/probe.
 * @param bloom_filter_longs Size of the bit array in 64-bit longs; total bits =
 *        bloom_filter_longs * 64.
 * @param seed Hash seed. Used only for V2; ignored for V1 (V1 always uses seed 0).
 * @param stream CUDA stream for device memory operations and kernel launches.
 * @param mr Device memory resource for allocating the bloom filter buffer.
 * @returns A list_scalar wrapping the packed Spark-format bloom filter (header + bits).
 */
std::unique_ptr<cudf::list_scalar> bloom_filter_create(
  int version,
  int num_hashes,
  int bloom_filter_longs,
  int seed                          = 0,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Inserts input values into a bloom filter.
 *
 * Can be called multiple times on the same bloom_filter buffer.
 *
 * @param[in,out] bloom_filter The bloom filter to be added to.
 * @param input Input column of int64_t values to be inserted into the bloom filter.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 *
 */
void bloom_filter_put(cudf::list_scalar& bloom_filter,
                      cudf::column_view const& input,
                      cuda::stream_ref stream = cudf::get_default_stream());

/**
 * @brief Probe a bloom filter with an input column of int64_t values.
 *
 * @param input The column of int64_t values to probe with.
 * @param bloom_filter The bloom filter to be probed.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned boolean column's memory.
 *
 * @returns A column of booleans where a true value indicates a value may be present in the bloom
 * filter, and a false indicates the value is not present.
 */
std::unique_ptr<cudf::column> bloom_filter_probe(
  cudf::column_view const& input,
  cudf::device_span<uint8_t const> bloom_filter,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Probe a bloom filter with an input column of int64_t values.
 *
 * @param input The column of int64_t values to probe with.
 * @param bloom_filter A list_scalar encapsulating the bloom filter to be probed.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned boolean column's memory.
 *
 * @returns A column of booleans where a true value indicates a value may be present in the bloom
 * filter, and a false indicates the value is not present.
 */
std::unique_ptr<cudf::column> bloom_filter_probe(
  cudf::column_view const& input,
  cudf::list_scalar& bloom_filter,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Merge multiple bloom filters into a single output.
 *
 * The incoming bloom filters are expected to be in the form of a list column, with
 * each row corresponding to an invidual bloom filter.  Each bloom filter must have the
 * same number of hashes and size.
 *
 * @param bloom_filters The bloom filters to be probed.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned boolean column's memory.
 *
 * @returns The new bloom filter.
 */
std::unique_ptr<cudf::list_scalar> bloom_filter_merge(
  cudf::column_view const& bloom_filters,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
