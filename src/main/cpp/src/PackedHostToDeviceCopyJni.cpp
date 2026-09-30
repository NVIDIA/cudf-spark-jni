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
#include "utilities.hpp"

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" {

// Returns [buffer_address, allocation_size, rmm_buffer_handle, event_handle], all zero when there
// is nothing to copy.
JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_PackedHostToDeviceCopy_copyAsync(
  JNIEnv* env, jclass, jlongArray j_addrs, jlongArray j_lens, jboolean on_side_stream)
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

    std::vector<cudf::host_span<uint8_t const>> buffers;
    buffers.reserve(lens.size());
    for (int i = 0; i < lens.size(); ++i) {
      CUDF_EXPECTS(lens[i] >= 0, "host buffer length must not be negative");
      buffers.emplace_back(reinterpret_cast<uint8_t const*>(addrs[i]),
                           static_cast<std::size_t>(lens[i]));
    }
    addrs.cancel();
    lens.cancel();

    cudf::jni::native_jlongArray result(env, 4);
    auto copy = spark_rapids_jni::copy_host_buffers_to_device_async(buffers, on_side_stream);
    if (copy.buffer) {
      result[0] = cudf::jni::ptr_as_jlong(copy.buffer->data());
      result[1] = static_cast<jlong>(copy.buffer->size());
      result[2] = cudf::jni::release_as_jlong(copy.buffer);
      result[3] = cudf::jni::ptr_as_jlong(copy.event);
    }
    return result.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_PackedHostToDeviceCopy_waitOnEvent(
  JNIEnv* env, jclass, jlong j_event)
{
  JNI_NULL_CHECK(env, j_event, "event is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    CUDF_CUDA_TRY(cudaStreamWaitEvent(
      cudf::get_default_stream().get(), reinterpret_cast<cudaEvent_t>(j_event), 0));
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_PackedHostToDeviceCopy_finish(JNIEnv* env,
                                                                                      jclass,
                                                                                      jlong j_event)
{
  JNI_NULL_CHECK(env, j_event, "event is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const event    = reinterpret_cast<cudaEvent_t>(j_event);
    auto const sync_err = cudaEventSynchronize(event);
    CUDF_CUDA_TRY(cudaEventDestroy(event));
    CUDF_CUDA_TRY(sync_err);
  }
  JNI_CATCH(env, );
}

}  // extern "C"
