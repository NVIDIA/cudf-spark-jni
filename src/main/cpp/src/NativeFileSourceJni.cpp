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

#include <cudf/io/config_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace {

cudf::io::datasource* as_source(jlong j_handle)
{
  return std::bit_cast<cudf::io::datasource*>(j_handle);
}

}  // namespace

extern "C" {

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_NativeFileSource_setUpKvikioNative(
  JNIEnv* env, jclass, jint j_mode)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::io::kvikio_integration::set_up_kvikio(
      std::nullopt, static_cast<cudf::io::kvikio_integration::compat_mode>(j_mode));
  }
  JNI_CATCH(env, );
}
JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_NativeFileSource_openNative(JNIEnv* env,
                                                                                     jclass,
                                                                                     jstring j_path)
{
  JNI_NULL_CHECK(env, j_path, "path is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring path(env, j_path);
    return std::bit_cast<jlong>(cudf::io::datasource::create(std::string{path.get()}).release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_NativeFileSource_size(JNIEnv* env,
                                                                               jclass,
                                                                               jlong j_handle)
{
  JNI_NULL_CHECK(env, j_handle, "file source is null", 0);
  JNI_TRY { return static_cast<jlong>(as_source(j_handle)->size()); }
  JNI_CATCH(env, 0);
}

JNIEXPORT jboolean JNICALL Java_com_nvidia_spark_rapids_jni_NativeFileSource_isDeviceReadDirect(
  JNIEnv* env, jclass, jlong j_handle, jlong j_size)
{
  JNI_NULL_CHECK(env, j_handle, "file source is null", false);
  JNI_TRY
  {
    CUDF_EXPECTS(j_size >= 0, "read size must not be negative");
    return as_source(j_handle)->is_device_read_direct(static_cast<std::size_t>(j_size));
  }
  JNI_CATCH(env, false);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_NativeFileSource_deviceRead(JNIEnv* env,
                                                             jclass,
                                                             jlong j_handle,
                                                             jlong j_file_offset,
                                                             jlong j_length,
                                                             jlong j_dst_address,
                                                             jlong j_stream)
{
  JNI_NULL_CHECK(env, j_handle, "file source is null", 0);
  JNI_NULL_CHECK(env, j_dst_address, "destination is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* const source = as_source(j_handle);
    CUDF_EXPECTS(j_file_offset >= 0 && static_cast<std::size_t>(j_file_offset) <= source->size(),
                 "file offset is out of range");
    CUDF_EXPECTS(j_length >= 0, "read length must not be negative");
    return static_cast<jlong>(
      source->device_read(static_cast<std::size_t>(j_file_offset),
                          static_cast<std::size_t>(j_length),
                          std::bit_cast<uint8_t*>(j_dst_address),
                          cuda::stream_ref{reinterpret_cast<cudaStream_t>(j_stream)}));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_NativeFileSource_close(JNIEnv* env,
                                                                               jclass,
                                                                               jlong j_handle)
{
  JNI_TRY { cudf::jni::safe_delete<cudf::io::datasource>(j_handle); }
  JNI_CATCH(env, );
}
}  // extern "C"
