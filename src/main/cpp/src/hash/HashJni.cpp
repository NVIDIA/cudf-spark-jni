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

#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"
#include "hash.hpp"
#include "jni_utils.hpp"

#include <zlib.h>

extern "C" {

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_Hash_getMaxStackDepth(JNIEnv* env, jclass)
{ return spark_rapids_jni::MAX_STACK_DEPTH; }

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_Hash_murmurHash32(
  JNIEnv* env, jclass, jint seed, jlongArray column_handles)
{
  JNI_NULL_CHECK(env, column_handles, "array of column handles is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto column_views =
      cudf::jni::native_jpointerArray<cudf::column_view>{env, column_handles}.get_dereferenced();
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::murmur_hash3_32(cudf::table_view{column_views}, seed));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_Hash_xxhash64(JNIEnv* env,
                                                                       jclass,
                                                                       jlong seed,
                                                                       jlongArray column_handles)
{
  JNI_NULL_CHECK(env, column_handles, "array of column handles is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto column_views =
      cudf::jni::native_jpointerArray<cudf::column_view>{env, column_handles}.get_dereferenced();
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::xxhash64(cudf::table_view{column_views}, seed));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_Hash_hiveHash(JNIEnv* env,
                                                                       jclass,
                                                                       jlongArray column_handles)
{
  JNI_NULL_CHECK(env, column_handles, "array of column handles is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto column_views =
      cudf::jni::native_jpointerArray<cudf::column_view>{env, column_handles}.get_dereferenced();
    return cudf::jni::release_as_jlong(spark_rapids_jni::hive_hash(cudf::table_view{column_views}));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_Hash_sha224NullsPreserved(JNIEnv* env, jclass, jlong column_handle)
{
  JNI_NULL_CHECK(env, column_handle, "column handle is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    return cudf::jni::release_as_jlong(spark_rapids_jni::sha224_nulls_preserved(
      *reinterpret_cast<cudf::column_view const*>(column_handle)));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_Hash_sha256NullsPreserved(JNIEnv* env, jclass, jlong column_handle)
{
  JNI_NULL_CHECK(env, column_handle, "column handle is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    return cudf::jni::release_as_jlong(spark_rapids_jni::sha256_nulls_preserved(
      *reinterpret_cast<cudf::column_view const*>(column_handle)));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_Hash_sha384NullsPreserved(JNIEnv* env, jclass, jlong column_handle)
{
  JNI_NULL_CHECK(env, column_handle, "column handle is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    return cudf::jni::release_as_jlong(spark_rapids_jni::sha384_nulls_preserved(
      *reinterpret_cast<cudf::column_view const*>(column_handle)));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_Hash_sha512NullsPreserved(JNIEnv* env, jclass, jlong column_handle)
{
  JNI_NULL_CHECK(env, column_handle, "column handle is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    return cudf::jni::release_as_jlong(spark_rapids_jni::sha512_nulls_preserved(
      *reinterpret_cast<cudf::column_view const*>(column_handle)));
  }
  JNI_CATCH(env, 0);
}

/**
 * @brief Compute the CRC32 checksum of the data in the given buffer on the host (CPU).
 *
 * @param crc the initial CRC value
 * @param buffer_handle the address of the buffer containing the data to checksum. Null is allowed
 * for empty buffers.
 * @param len the length of the data in bytes. Must be zero for empty buffers, and positive for
 * non-empty buffers.
 * @return the computed CRC32 checksum
 */
JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_Hash_hostCrc32(
  JNIEnv* env, jclass, jlong crc, jlong buffer_handle, jint len)
{
  if (buffer_handle == 0) {
    JNI_ARG_CHECK(env, len == 0, "len is not zero for empty buffer", 0);
  } else {
    JNI_ARG_CHECK(env, len > 0, "len must be positive for non-empty buffer", 0);
  }
  JNI_TRY
  {
    auto const buffer_addr = reinterpret_cast<unsigned char*>(buffer_handle);
    return crc32(static_cast<uLong>(crc), buffer_addr, static_cast<uInt>(len));
  }
  JNI_CATCH(env, 0);
}
}
