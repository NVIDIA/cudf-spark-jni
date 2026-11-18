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
#include <exception>
#include <future>
#include <string>
#include <utility>
#include <vector>

namespace {

cudf::io::datasource* as_source(jlong j_handle)
{
  return std::bit_cast<cudf::io::datasource*>(j_handle);
}

/**
 * Start `read_async(file_offset, length, dst)` for every non-empty range, wait for all of them and
 * rethrow the first failure. Each started read must finish before returning, since it writes into
 * the caller's buffer.
 */
template <typename ReadAsync>
void read_ranges(JNIEnv* env,
                 cudf::io::datasource const& source,
                 jlongArray j_file_offsets,
                 jlongArray j_lengths,
                 jlongArray j_dst_addresses,
                 ReadAsync read_async)
{
  // Views of the Java arrays; the Java side has already checked each range fits the destination.
  cudf::jni::native_jlongArray const file_offsets(env, j_file_offsets);
  cudf::jni::native_jlongArray const lengths(env, j_lengths);
  cudf::jni::native_jlongArray const dst_addresses(env, j_dst_addresses);
  CUDF_EXPECTS(file_offsets.size() == lengths.size() && dst_addresses.size() == lengths.size(),
               "range arrays must have the same length");

  // Validate every range before starting any read, so a bad range fails the call without writing
  // to the destination. The length check is written as a subtraction so it cannot overflow.
  auto const file_size = source.size();
  for (int i = 0; i < lengths.size(); ++i) {
    CUDF_EXPECTS(file_offsets[i] >= 0 && lengths[i] >= 0 &&
                   static_cast<std::size_t>(file_offsets[i]) <= file_size &&
                   static_cast<std::size_t>(lengths[i]) <=
                     file_size - static_cast<std::size_t>(file_offsets[i]),
                 "read range is outside the file");
  }

  // Each started read and the length it should return.
  std::vector<std::pair<std::future<std::size_t>, std::size_t>> reads;
  reads.reserve(lengths.size());
  // The first failure, from starting a read or from one that was started.
  std::exception_ptr error;

  // Start all the reads so KvikIO runs them concurrently. If starting one throws, stop starting
  // more, but fall through to wait for those already running.
  try {
    for (int i = 0; i < lengths.size(); ++i) {
      if (lengths[i] == 0) { continue; }
      auto const length = static_cast<std::size_t>(lengths[i]);
      reads.emplace_back(read_async(static_cast<std::size_t>(file_offsets[i]),
                                    length,
                                    std::bit_cast<uint8_t*>(dst_addresses[i])),
                         length);
    }
  } catch (...) {
    error = std::current_exception();
  }

  // Wait for every started read, even after a failure: returning early would let the caller free
  // the destination while reads are still writing to it. Keep only the first failure.
  for (auto& [read, length] : reads) {
    try {
      // The ranges were checked against the file size, so fewer bytes means the read failed.
      if (read.get() != length && !error) {
        error = std::make_exception_ptr(cudf::logic_error("short read"));
      }
    } catch (...) {
      if (!error) { error = std::current_exception(); }
    }
  }
  if (error) { std::rethrow_exception(error); }
}

}  // namespace

extern "C" {

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_NativeFileSource_setUpKvikioNative(
  JNIEnv* env, jclass, jint j_mode, jint j_num_threads)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::io::kvikio_integration::set_up_kvikio(
      static_cast<uint32_t>(j_num_threads),
      static_cast<cudf::io::kvikio_integration::compat_mode>(j_mode));
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

JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_jni_NativeFileSource_readToHost(JNIEnv* env,
                                                             jclass,
                                                             jlong j_handle,
                                                             jlongArray j_file_offsets,
                                                             jlongArray j_lengths,
                                                             jlongArray j_dst_addresses)
{
  JNI_NULL_CHECK(env, j_handle, "file source is null", );
  JNI_NULL_CHECK(env, j_file_offsets, "file offsets are null", );
  JNI_NULL_CHECK(env, j_lengths, "lengths are null", );
  JNI_NULL_CHECK(env, j_dst_addresses, "destination addresses are null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* const source = as_source(j_handle);
    read_ranges(env,
                *source,
                j_file_offsets,
                j_lengths,
                j_dst_addresses,
                [source](std::size_t file_offset, std::size_t length, uint8_t* dst) {
                  return source->host_read_async(file_offset, length, dst);
                });
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_jni_NativeFileSource_readToDevice(JNIEnv* env,
                                                               jclass,
                                                               jlong j_handle,
                                                               jlongArray j_file_offsets,
                                                               jlongArray j_lengths,
                                                               jlongArray j_dst_addresses,
                                                               jlong j_stream)
{
  JNI_NULL_CHECK(env, j_handle, "file source is null", );
  JNI_NULL_CHECK(env, j_file_offsets, "file offsets are null", );
  JNI_NULL_CHECK(env, j_lengths, "lengths are null", );
  JNI_NULL_CHECK(env, j_dst_addresses, "destination addresses are null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* const source = as_source(j_handle);
    CUDF_EXPECTS(source->supports_device_read(), "file source does not support device reads");
    cuda::stream_ref const stream{reinterpret_cast<cudaStream_t>(j_stream)};
    read_ranges(env,
                *source,
                j_file_offsets,
                j_lengths,
                j_dst_addresses,
                [source, stream](std::size_t file_offset, std::size_t length, uint8_t* dst) {
                  return source->device_read_async(file_offset, length, dst, stream);
                });
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_NativeFileSource_close(JNIEnv* env,
                                                                               jclass,
                                                                               jlong j_handle)
{
  JNI_TRY { cudf::jni::safe_delete<cudf::io::datasource>(j_handle); }
  JNI_CATCH(env, );
}
}  // extern "C"
