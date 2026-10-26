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

#include "cudf_jni_apis.hpp"
#include "memory_utils.hpp"
#include "utilities.hpp"

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

extern "C" {

// The destinations must have been allocated on `j_stream`, since nothing orders the copies behind
// any other stream.
JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_jni_PackedHostToDeviceCopy_copyAsync(JNIEnv* env,
                                                                  jclass,
                                                                  jlongArray j_addrs,
                                                                  jlongArray j_lens,
                                                                  jlongArray j_dst_addrs,
                                                                  jlongArray j_dst_lens,
                                                                  jintArray j_dst_counts,
                                                                  jlong j_stream)
{
  // Reject null arrays.
  JNI_NULL_CHECK(env, j_addrs, "host buffer addresses are null", );
  JNI_NULL_CHECK(env, j_lens, "host buffer lengths are null", );
  JNI_NULL_CHECK(env, j_dst_addrs, "destination addresses are null", );
  JNI_NULL_CHECK(env, j_dst_lens, "destination lengths are null", );
  JNI_NULL_CHECK(env, j_dst_counts, "destination buffer counts are null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    // Read the Java arrays and check their counts agree.
    cudf::jni::native_jlongArray addrs(env, j_addrs);
    cudf::jni::native_jlongArray lens(env, j_lens);
    cudf::jni::native_jlongArray dst_addrs(env, j_dst_addrs);
    cudf::jni::native_jlongArray dst_lens(env, j_dst_lens);
    cudf::jni::native_jintArray dst_counts(env, j_dst_counts);
    CUDF_EXPECTS(addrs.size() == lens.size(),
                 "host buffer addresses and lengths must have the same count");
    CUDF_EXPECTS(dst_addrs.size() == dst_lens.size() && dst_addrs.size() == dst_counts.size(),
                 "destination addresses, lengths and buffer counts must have the same count");

    // Wrap the host buffers in spans.
    std::vector<std::span<uint8_t const>> buffers;
    buffers.reserve(lens.size());
    for (int i = 0; i < lens.size(); ++i) {
      CUDF_EXPECTS(lens[i] >= 0, "host buffer length must not be negative");
      buffers.emplace_back(reinterpret_cast<uint8_t const*>(addrs[i]),
                           static_cast<std::size_t>(lens[i]));
    }

    // Gather the copies: destination d takes the next dst_counts[d] buffers, packed back to back.
    // Each buffer adds at most one copy.
    spark_rapids_jni::batched_copies copies;
    copies.dsts.reserve(buffers.size());
    copies.srcs.reserve(buffers.size());
    copies.sizes.reserve(buffers.size());
    std::size_t first = 0;
    for (int d = 0; d < dst_addrs.size(); ++d) {
      CUDF_EXPECTS(dst_lens[d] >= 0, "destination length must not be negative");
      CUDF_EXPECTS(dst_counts[d] >= 0, "destination buffer count must not be negative");
      auto const count = static_cast<std::size_t>(dst_counts[d]);
      CUDF_EXPECTS(first + count <= buffers.size(),
                   "destination buffer counts exceed the host buffers");
      auto const destination = cudf::device_span<uint8_t>(reinterpret_cast<uint8_t*>(dst_addrs[d]),
                                                          static_cast<std::size_t>(dst_lens[d]));
      spark_rapids_jni::add_packed_copies(
        std::span<std::span<uint8_t const> const>(buffers).subspan(first, count),
        destination,
        copies);
      first += count;
    }
    CUDF_EXPECTS(first == buffers.size(),
                 "destination buffer counts do not cover the host buffers");

    // The arrays were only read, so skip copying them back to Java.
    addrs.cancel();
    lens.cancel();
    dst_addrs.cancel();
    dst_lens.cancel();
    dst_counts.cancel();

    // Queue all the copies in one call.
    auto const stream = cuda::stream_ref{reinterpret_cast<cudaStream_t>(j_stream)};
    spark_rapids_jni::copy_batch_async(copies, stream);
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_PackedHostToDeviceCopy_joinStreams(
  JNIEnv* env, jclass, jlong j_waited_on, jlong j_stream)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const waited_on = cuda::stream_ref{reinterpret_cast<cudaStream_t>(j_waited_on)};
    spark_rapids_jni::join_streams({&waited_on, 1},
                                   cuda::stream_ref{reinterpret_cast<cudaStream_t>(j_stream)});
  }
  JNI_CATCH(env, );
}
}  // extern "C"
