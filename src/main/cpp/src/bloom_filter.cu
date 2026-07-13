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

#include "bloom_filter.hpp"
#include "hash/murmur_hash.cuh"
#include "nvtx_ranges.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/std/utility>
#include <thrust/logical.h>

#include <byteswap.h>

#include <limits>

namespace spark_rapids_jni {

namespace {

using bloom_hash_type = spark_rapids_jni::murmur_hash_value_type;

inline int32_t byte_swap_int32(int32_t val)
{ return static_cast<int32_t>(bswap_32(static_cast<uint32_t>(val))); }

// Given a non-negative bit position within the bloom filter, compute
// the 32-bit word index and bitmask. Handles big-endian swizzle so
// the GPU buffer is directly compatible with Spark's serialized format.
__device__ inline cuda::std::pair<int64_t, cudf::bitmask_type> gpu_bit_to_word_mask(int64_t bit_pos)
{
  auto const word_index = (bit_pos / 32) ^ 0x1;
  auto const bit_index  = static_cast<int32_t>(bit_pos % 32) ^ 0x18;

  return {word_index, static_cast<cudf::bitmask_type>(1u << bit_index)};
}

// V1: combined hash is 32-bit int, loop from 1..num_hashes
// V2: combined hash is 64-bit long, seeded with h1*INT32_MAX, loop from 0..num_hashes-1
template <int Version, bool nullable>
CUDF_KERNEL void gpu_bloom_filter_put(cudf::bitmask_type* const bloom_filter,
                                      int64_t bloom_filter_bits,
                                      cudf::column_device_view input,
                                      cudf::size_type num_hashes,
                                      int32_t seed)
{
  size_t const tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= input.size()) { return; }

  if constexpr (nullable) {
    if (!input.is_valid(tid)) { return; }
  }

  auto const el = input.element<int64_t>(tid);
  // V1 has no seed in the format; use 0. V2 uses the stored seed.
  int32_t const hash_seed  = (Version == 1) ? 0 : seed;
  bloom_hash_type const h1 = MurmurHash3_32<int64_t>(hash_seed)(el);
  bloom_hash_type const h2 = MurmurHash3_32<int64_t>(h1)(el);

  if constexpr (Version == 1) {
    // https://github.com/apache/spark/blob/5075ea6a85f3f1689766cf08a7d5b2ce500be1fb/common/sketch/src/main/java/org/apache/spark/util/sketch/BloomFilterImpl.java#L38
    // This is the original V1 hash algorithm from Spark.
    for (auto idx = 1; idx <= num_hashes; idx++) {
      bloom_hash_type combined_hash = h1 + (idx * h2);
      auto const bit_pos =
        static_cast<int64_t>(combined_hash < 0 ? ~combined_hash : combined_hash) %
        bloom_filter_bits;
      auto const [word_index, mask] = gpu_bit_to_word_mask(bit_pos);
      cuda::atomic_ref<cudf::bitmask_type, cuda::thread_scope_device> ref(bloom_filter[word_index]);
      ref.fetch_or(mask, cuda::memory_order_relaxed);
    }
  } else {
    // https://github.com/apache/spark/blob/5075ea6a85f3f1689766cf08a7d5b2ce500be1fb/common/sketch/src/main/java/org/apache/spark/util/sketch/BloomFilterImplV2.java#L63
    int64_t combined_hash =
      static_cast<int64_t>(h1) * static_cast<int64_t>(cuda::std::numeric_limits<int32_t>::max());
    for (int idx = 0; idx < num_hashes; idx++) {
      combined_hash += h2;
      int64_t combined_index        = combined_hash < 0 ? ~combined_hash : combined_hash;
      auto const bit_pos            = combined_index % bloom_filter_bits;
      auto const [word_index, mask] = gpu_bit_to_word_mask(bit_pos);
      cuda::atomic_ref<cudf::bitmask_type, cuda::thread_scope_device> ref(bloom_filter[word_index]);
      ref.fetch_or(mask, cuda::memory_order_relaxed);
    }
  }
}

template <int Version>
struct bloom_probe_functor {
  cudf::bitmask_type const* const bloom_filter;
  int64_t const bloom_filter_bits;
  cudf::size_type const num_hashes;
  int32_t const seed;

  __device__ bool operator()(int64_t input) const
  {
    int32_t const hash_seed  = (Version == 1) ? 0 : seed;
    bloom_hash_type const h1 = MurmurHash3_32<int64_t>(hash_seed)(input);
    bloom_hash_type const h2 = MurmurHash3_32<int64_t>(h1)(input);

    if constexpr (Version == 1) {
      for (auto idx = 1; idx <= num_hashes; idx++) {
        bloom_hash_type combined_hash = h1 + (idx * h2);
        auto const bit_pos =
          static_cast<int64_t>(combined_hash < 0 ? ~combined_hash : combined_hash) %
          bloom_filter_bits;
        auto const [word_index, mask] = gpu_bit_to_word_mask(bit_pos);
        if (!(bloom_filter[word_index] & mask)) { return false; }
      }
    } else {
      int64_t combined_hash =
        static_cast<int64_t>(h1) * static_cast<int64_t>(cuda::std::numeric_limits<int32_t>::max());
      for (int idx = 0; idx < num_hashes; idx++) {
        combined_hash += h2;
        int64_t combined_index        = combined_hash < 0 ? ~combined_hash : combined_hash;
        auto const bit_pos            = combined_index % bloom_filter_bits;
        auto const [word_index, mask] = gpu_bit_to_word_mask(bit_pos);
        if (!(bloom_filter[word_index] & mask)) { return false; }
      }
    }
    return true;
  }
};

void pack_bloom_filter_header(cudf::device_span<uint8_t> buf,
                              bloom_filter_header const& header,
                              rmm::cuda_stream_view stream,
                              int32_t seed)
{
  if (header.version == bloom_filter_version_1) {
    bloom_filter_header_v1 raw = {byte_swap_int32(header.version),
                                  byte_swap_int32(header.num_hashes),
                                  byte_swap_int32(header.num_longs)};
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      buf.data(), &raw, bloom_filter_header_v1_size_bytes, cudaMemcpyHostToDevice, stream));
  } else {
    bloom_filter_header_v2 raw = {byte_swap_int32(header.version),
                                  byte_swap_int32(header.num_hashes),
                                  byte_swap_int32(seed),
                                  byte_swap_int32(header.num_longs)};
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      buf.data(), &raw, bloom_filter_header_v2_size_bytes, cudaMemcpyHostToDevice, stream));
  }
}

/*
  Unpack bloom filter information from a Spark-format bloom filter buffer (big-endian).
  Accepts both V1 and V2; the version is read from the first 4 bytes.

  @return A std::tuple of four elements:
    - Element 0 (bloom_filter_header): Decoded header with version, num_hashes, and num_longs.
      Does not include seed; that is returned separately as element 3.
    - Element 1 (cudf::device_span<cudf::bitmask_type const>): Device span over the bloom filter
      bit array. Length is header.num_longs * 2 (number of 32-bit words). The data start
      immediately after the version-specific header in the buffer.
    - Element 2 (int64_t): Total number of bits in the bloom filter, i.e. header.num_longs * 64.
    - Element 3 (int32_t): Hash seed used when building/probing the filter. Zero for V1;
      for V2, the value stored in the serialized header.
*/
std::tuple<bloom_filter_header, cudf::device_span<cudf::bitmask_type const>, int64_t, int32_t>
unpack_bloom_filter(cudf::device_span<uint8_t const> bloom_filter, rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(bloom_filter.size() >= static_cast<size_t>(bloom_filter_header_v1_size_bytes),
               "Encountered truncated bloom filter");

  int32_t raw_ints[4] = {};
  auto const read_size =
    std::min(bloom_filter.size(), static_cast<size_t>(bloom_filter_header_v2_size_bytes));

  // TODO (future): Consider using pinned host memory for cudaMemcpyAsync.
  // Refer to https://github.com/NVIDIA/spark-rapids-jni/issues/4407.
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(raw_ints, bloom_filter.data(), read_size, cudaMemcpyDeviceToHost, stream));
  stream.synchronize();

  int const version = byte_swap_int32(raw_ints[0]);
  CUDF_EXPECTS(version == bloom_filter_version_1 || version == bloom_filter_version_2,
               "Unexpected bloom filter version");

  auto const hdr_size = bloom_filter_header_size_for_version(version);
  CUDF_EXPECTS(bloom_filter.size() >= static_cast<size_t>(hdr_size),
               "Encountered truncated bloom filter header");

  bloom_filter_header header;
  header.version    = version;
  header.num_hashes = byte_swap_int32(raw_ints[1]);
  header.num_longs  = (version == bloom_filter_version_1) ? byte_swap_int32(raw_ints[2])
                                                          : byte_swap_int32(raw_ints[3]);

  int32_t seed = 0;
  if (version == bloom_filter_version_2) {
    // Safe: the header-size check above guarantees bloom_filter.size() >= v2 header size,
    // so read_size (= min(bloom_filter.size(), v2 header size)) == v2 header size.
    seed = byte_swap_int32(raw_ints[2]);
  }

  auto const bloom_filter_bits = static_cast<int64_t>(header.num_longs) * 64;
  auto const num_bitmask_words = static_cast<size_t>(header.num_longs) * 2;

  CUDF_EXPECTS(bloom_filter_bits > 0, "Invalid empty bloom filter size");

  return {header,
          {reinterpret_cast<cudf::bitmask_type const*>(bloom_filter.data() + hdr_size),
           num_bitmask_words},
          bloom_filter_bits,
          seed};
}

std::tuple<bloom_filter_header, cudf::device_span<cudf::bitmask_type const>, int64_t, int32_t>
unpack_bloom_filter(cudf::column_view const& bloom_filter, rmm::cuda_stream_view stream)
{
  return unpack_bloom_filter(
    cudf::device_span<uint8_t const>{bloom_filter.data<uint8_t>(),
                                     static_cast<size_t>(bloom_filter.size())},
    stream);
}

/*
  Device functor used by bloom_filter_merge to verify every filter in the list has the same
  header. raw_header holds the reference header in big-endian form (as in the serialized buffer).
*/
struct bloom_filter_same {
  /// Reference header: big-endian int32s.
  /// V1 uses [0..2]: version, num_hashes, num_longs.
  /// V2 uses [0..3]: version, num_hashes, seed, num_longs.
  int32_t raw_header[4];
  int header_field_count;
  cudf::lists_column_device_view ldv;
  cudf::size_type stride;

  bool __device__ operator()(cudf::size_type i)
  {
    auto const* a = reinterpret_cast<int32_t const*>(ldv.child().data<uint8_t>() + stride * i);
    for (int j = 0; j < header_field_count; j++) {
      if (a[j] != raw_header[j]) return false;
    }
    return true;
  }
};

/*
  Returns a std::tuple indicating:
  - first: size in bytes of the bloom filter bit array (num_longs * 8).
  - second: total size in bytes of the serialized bloom filter buffer (header + bit array).
  Uses the version-specific header size.
*/
std::tuple<int32_t, int32_t> get_bloom_filter_stride(int version, int bloom_filter_longs)
{
  auto const bloom_filter_size = static_cast<int64_t>(bloom_filter_longs) * sizeof(int64_t);
  auto const hdr_size          = bloom_filter_header_size_for_version(version);
  auto const buf_size          = hdr_size + bloom_filter_size;
  CUDF_EXPECTS(buf_size <= std::numeric_limits<int32_t>::max(),
               "Bloom filter buffer size exceeds int32 range");
  return {static_cast<int32_t>(bloom_filter_size), static_cast<int32_t>(buf_size)};
}

}  // anonymous namespace

/*
  Creates a new bloom filter. The result is stored in a cudf list_scalar with a single
  UINT8 buffer of the following form (all header and bit data in big-endian for Spark):

  - V1: first 12 bytes = bloom_filter_header_v1 (version, num_hashes, num_longs).
  - V2: first 16 bytes = bloom_filter_header_v2 (version, num_hashes, seed, num_longs).
  - Remaining bytes: bloom_filter_longs * 8, the bit array. Initialized to zero.

  unpack_bloom_filter() reads this layout; pack_bloom_filter_header() writes the header.
*/
std::unique_ptr<cudf::list_scalar> bloom_filter_create(int version,
                                                       int num_hashes,
                                                       int bloom_filter_longs,
                                                       int seed,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  CUDF_EXPECTS(version == bloom_filter_version_1 || version == bloom_filter_version_2,
               "Bloom filter version must be 1 or 2");
  CUDF_EXPECTS(num_hashes > 0, "Bloom filter hash count must be positive");
  CUDF_EXPECTS(bloom_filter_longs > 0, "Bloom filter bit count must be positive");

  auto [bloom_filter_size, buf_size] = get_bloom_filter_stride(version, bloom_filter_longs);
  auto const hdr_size                = bloom_filter_header_size_for_version(version);

  rmm::device_buffer buf{static_cast<size_t>(buf_size), stream, mr};

  bloom_filter_header header{version, num_hashes, bloom_filter_longs};
  pack_bloom_filter_header({reinterpret_cast<uint8_t*>(buf.data()), static_cast<size_t>(buf_size)},
                           header,
                           stream,
                           (version == bloom_filter_version_1 ? 0 : seed));

  CUDF_CUDA_TRY(cudaMemsetAsync(
    reinterpret_cast<uint8_t*>(buf.data()) + hdr_size, 0, bloom_filter_size, stream));

  return std::make_unique<cudf::list_scalar>(
    cudf::column(
      cudf::data_type{cudf::type_id::UINT8}, buf_size, std::move(buf), rmm::device_buffer{}, 0),
    true,
    stream,
    mr);
}

void bloom_filter_put(cudf::list_scalar& bloom_filter,
                      cudf::column_view const& input,
                      rmm::cuda_stream_view stream)
{
  SRJ_FUNC_RANGE();
  auto [header, buffer, bloom_filter_bits, seed] = unpack_bloom_filter(bloom_filter.view(), stream);
  auto const hdr_size = bloom_filter_header_size_for_version(header.version);
  CUDF_EXPECTS(static_cast<size_t>(bloom_filter.view().size()) == (buffer.size() * 4) + hdr_size,
               "Encountered invalid/mismatched bloom filter buffer data");

  // bloom_filter is non-const, so mutable access to the underlying data is valid.
  // list_scalar::view() returns a const column_view, requiring const_cast here.
  auto* mutable_buffer = const_cast<cudf::bitmask_type*>(buffer.data());

  constexpr int block_size = 256;
  auto grid                = cudf::detail::grid_1d{input.size(), block_size, 1};
  auto d_input             = cudf::column_device_view::create(input);

  auto launch = [&](auto version_tag, auto nullable_tag) {
    gpu_bloom_filter_put<decltype(version_tag)::value, decltype(nullable_tag)::value>
      <<<grid.num_blocks, block_size, 0, stream.value()>>>(
        mutable_buffer, bloom_filter_bits, *d_input, header.num_hashes, seed);
  };

  if (header.version == bloom_filter_version_1) {
    CUDF_EXPECTS(bloom_filter_bits <= std::numeric_limits<int32_t>::max(),
                 "V1 bloom filter bit count exceeds int32 range");
    if (input.has_nulls()) {
      launch(std::integral_constant<int, 1>{}, std::true_type{});
    } else {
      launch(std::integral_constant<int, 1>{}, std::false_type{});
    }
  } else {
    if (input.has_nulls()) {
      launch(std::integral_constant<int, 2>{}, std::true_type{});
    } else {
      launch(std::integral_constant<int, 2>{}, std::false_type{});
    }
  }
}

std::unique_ptr<cudf::list_scalar> bloom_filter_merge(cudf::column_view const& bloom_filters,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  cudf::lists_column_view lcv(bloom_filters);

  // The list child column is a concatenation of packed bloom filter buffers (header + bits)
  // one after another. Unpack the first buffer to get the header, bit span, bit count, and
  // seed; we use these to validate total size and to build the merged output.
  auto [header, buffer, bloom_filter_bits, seed] = unpack_bloom_filter(lcv.child(), stream);
  auto const hdr_size = bloom_filter_header_size_for_version(header.version);
  CUDF_EXPECTS(
    static_cast<size_t>(lcv.child().size()) == (buffer.size() * 4 + static_cast<size_t>(hdr_size)) *
                                                 static_cast<size_t>(bloom_filters.size()),
    "Encountered invalid/mismatched bloom filter buffer data");

  auto [bloom_filter_size, buf_size] = get_bloom_filter_stride(header.version, header.num_longs);

  int32_t raw_hdr[4]     = {};
  int header_field_count = 0;
  if (header.version == bloom_filter_version_1) {
    raw_hdr[0]         = byte_swap_int32(header.version);
    raw_hdr[1]         = byte_swap_int32(header.num_hashes);
    raw_hdr[2]         = byte_swap_int32(header.num_longs);
    header_field_count = 3;
  } else {
    raw_hdr[0]         = byte_swap_int32(header.version);
    raw_hdr[1]         = byte_swap_int32(header.num_hashes);
    raw_hdr[2]         = byte_swap_int32(seed);
    raw_hdr[3]         = byte_swap_int32(header.num_longs);
    header_field_count = 4;
  }

  auto dv = cudf::column_device_view::create(bloom_filters);
  CUDF_EXPECTS(
    thrust::all_of(
      rmm::exec_policy_nosync(stream),
      thrust::make_counting_iterator(1),
      thrust::make_counting_iterator(bloom_filters.size()),
      bloom_filter_same{
        {raw_hdr[0], raw_hdr[1], raw_hdr[2], raw_hdr[3]}, header_field_count, *dv, buf_size}),
    "Mismatch of bloom filter parameters");

  rmm::device_buffer buf{static_cast<size_t>(buf_size), stream, mr};
  pack_bloom_filter_header(
    {reinterpret_cast<uint8_t*>(buf.data()), static_cast<size_t>(buf_size)}, header, stream, seed);

  auto src = lcv.child().data<uint8_t>() + hdr_size;
  auto dst =
    reinterpret_cast<cudf::bitmask_type*>(reinterpret_cast<uint8_t*>(buf.data()) + hdr_size);

  cudf::size_type num_words = header.num_longs * 2;
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + num_words,
    dst,
    cuda::proclaim_return_type<cudf::bitmask_type>(
      [src, num_buffers = bloom_filters.size(), stride = buf_size] __device__(
        cudf::size_type word_index) {
        cudf::bitmask_type out = (reinterpret_cast<cudf::bitmask_type const*>(src))[word_index];
        for (auto idx = 1; idx < num_buffers; idx++) {
          out |= (reinterpret_cast<cudf::bitmask_type const*>(src + idx * stride))[word_index];
        }
        return out;
      }));

  // create the 1-row list column and move it into a scalar.
  return std::make_unique<cudf::list_scalar>(
    cudf::column(
      cudf::data_type{cudf::type_id::UINT8}, buf_size, std::move(buf), rmm::device_buffer{}, 0),
    true,
    stream,
    mr);
}

std::unique_ptr<cudf::column> bloom_filter_probe(cudf::column_view const& input,
                                                 cudf::device_span<uint8_t const> bloom_filter,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  auto [header, buffer, bloom_filter_bits, seed] = unpack_bloom_filter(bloom_filter, stream);
  auto const hdr_size = bloom_filter_header_size_for_version(header.version);
  CUDF_EXPECTS(bloom_filter.size() == static_cast<size_t>((buffer.size() * 4) + hdr_size),
               "Encountered invalid/mismatched bloom filter buffer data");

  auto out = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                           input.size(),
                                           cudf::copy_bitmask(input, stream, mr),
                                           input.null_count(),
                                           stream,
                                           mr);

  if (header.version == bloom_filter_version_1) {
    CUDF_EXPECTS(bloom_filter_bits <= std::numeric_limits<int32_t>::max(),
                 "V1 bloom filter bit count exceeds int32 range");
    thrust::transform(
      rmm::exec_policy_nosync(stream),
      input.begin<int64_t>(),
      input.end<int64_t>(),
      out->mutable_view().begin<bool>(),
      bloom_probe_functor<1>{buffer.data(), bloom_filter_bits, header.num_hashes, seed});
  } else {
    thrust::transform(
      rmm::exec_policy_nosync(stream),
      input.begin<int64_t>(),
      input.end<int64_t>(),
      out->mutable_view().begin<bool>(),
      bloom_probe_functor<2>{buffer.data(), bloom_filter_bits, header.num_hashes, seed});
  }

  return out;
}

std::unique_ptr<cudf::column> bloom_filter_probe(cudf::column_view const& input,
                                                 cudf::list_scalar& bloom_filter,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return bloom_filter_probe(input, bloom_filter.view(), stream, mr);
}

}  // namespace spark_rapids_jni
