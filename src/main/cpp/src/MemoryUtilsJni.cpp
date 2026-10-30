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

#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

cuda::stream_ref as_stream(jlong j_stream)
{
  return cuda::stream_ref{reinterpret_cast<cudaStream_t>(j_stream)};
}

}  // namespace

extern "C" {

// Returns the pack count, then each pack's rmm::device_buffer*, address and length, then each host
// buffer's pack and offset in it. Java takes ownership of the device_buffers.
JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_PackedHostToDeviceCopy_copyAsync(JNIEnv* env,
                                                                  jclass,
                                                                  jlongArray j_addrs,
                                                                  jlongArray j_lens,
                                                                  jlong j_max_pack_bytes,
                                                                  jlong j_stream)
{
  JNI_NULL_CHECK(env, j_addrs, "host buffer addresses are null", nullptr);
  JNI_NULL_CHECK(env, j_lens, "host buffer lengths are null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::jni::native_jlongArray addrs(env, j_addrs);
    cudf::jni::native_jlongArray lens(env, j_lens);
    CUDF_EXPECTS(addrs.size() == lens.size(),
                 "host buffer addresses and lengths must have the same count");
    CUDF_EXPECTS(j_max_pack_bytes >= 0, "max pack bytes must not be negative");

    std::vector<std::span<uint8_t const>> buffers;
    buffers.reserve(lens.size());
    for (int i = 0; i < lens.size(); ++i) {
      CUDF_EXPECTS(lens[i] >= 0, "host buffer length must not be negative");
      buffers.emplace_back(reinterpret_cast<uint8_t const*>(addrs[i]),
                           static_cast<std::size_t>(lens[i]));
    }
    // The arrays were only read, so skip copying them back to Java.
    addrs.cancel();
    lens.cancel();

    // Plan the packs, then make the result buffers before copying, so nothing after the copies are
    // queued can throw.
    auto const plan =
      spark_rapids_jni::plan_packs(buffers, static_cast<std::size_t>(j_max_pack_bytes));
    auto const pack_count   = static_cast<int>(plan.pack_sizes.size());
    auto const buffer_count = static_cast<int>(buffers.size());
    auto const base         = 1 + 3 * pack_count;
    cudf::jni::native_jlongArray layout(env, base + 2 * buffer_count);
    auto* const out = layout.data();

    // Copy the buffers to packed device allocations.
    auto packs = spark_rapids_jni::copy_packed_async(buffers, plan, as_stream(j_stream));

    // Build and return the layout.
    out[0] = pack_count;
    for (int p = 0; p < pack_count; ++p) {
      out[1 + 3 * p] = std::bit_cast<jlong>(packs[p].get());
      out[2 + 3 * p] = std::bit_cast<jlong>(packs[p]->data());
      out[3 + 3 * p] = static_cast<jlong>(packs[p]->size());
    }
    for (int i = 0; i < buffer_count; ++i) {
      out[base + 2 * i]     = static_cast<jlong>(plan.pack_of[i]);
      out[base + 2 * i + 1] = static_cast<jlong>(plan.offsets[i]);
    }
    for (auto& pack : packs) {
      static_cast<void>(pack.release());
    }
    return layout.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

// Free each pack in `j_layout`, as copyAsync returned it, on `j_stream` when it is closed.
JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_PackedHostToDeviceCopy_setFreeStream(
  JNIEnv* env, jclass, jlongArray j_layout, jlong j_stream)
{
  JNI_NULL_CHECK(env, j_layout, "layout is null", );
  JNI_TRY
  {
    auto const stream = as_stream(j_stream);
    cudf::jni::native_jlongArray layout(env, j_layout);
    for (int p = 0; p < layout[0]; ++p) {
      std::bit_cast<rmm::device_buffer*>(layout[1 + 3 * p])->set_stream(stream);
    }
    layout.cancel();
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_PackedHostToDeviceCopy_freePack(
  JNIEnv* env, jclass, jlong j_handle)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete std::bit_cast<rmm::device_buffer*>(j_handle);
  }
  JNI_CATCH(env, );
}
}  // extern "C"
