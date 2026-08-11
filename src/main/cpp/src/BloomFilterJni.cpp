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
#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"
#include "jni_utils.hpp"
#include "utilities.hpp"

#include <bit>
#include <limits>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_BloomFilter_creategpu(
  JNIEnv* env, jclass, jint version, jint numHashes, jlong bloomFilterBits, jint seed)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    // Per the Spark implementation, according to the BitArray class,
    // https://github.com/apache/spark/blob/5075ea6a85f3f1689766cf08a7d5b2ce500be1fb/common/sketch/src/main/java/org/apache/spark/util/sketch/BitArray.java#L34
    // the number of longs representing the bit array can only be Integer.MAX_VALUE, at the most.
    // (This is presumably because the BitArray is indexed with an int32_t.)
    // This implies that the maximum supported bloom filter bit count is Integer.MAX_VALUE * 64.

    JNI_ARG_CHECK(
      env,
      bloomFilterBits > 0 &&
        bloomFilterBits <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()) * 64,
      "bloom filter bit count must be positive and less than or equal to the maximum supported "
      "size",
      0);
    auto const bloom_filter_longs = static_cast<int32_t>((bloomFilterBits + 63) / 64);
    auto bloom_filter =
      spark_rapids_jni::bloom_filter_create(version, numHashes, bloom_filter_longs, seed);
    return std::bit_cast<jlong>(bloom_filter.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_BloomFilter_put(JNIEnv* env,
                                                                        jclass,
                                                                        jlong bloomFilter,
                                                                        jlong cv)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::column_view const& input_column = *std::bit_cast<cudf::column_view const*>(cv);
    spark_rapids_jni::bloom_filter_put(*(std::bit_cast<cudf::list_scalar*>(bloomFilter)),
                                       input_column);
    return 0;
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_BloomFilter_merge(JNIEnv* env,
                                                                           jclass,
                                                                           jlong bloomFilters)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::column_view const& input_bloom_filter =
      *std::bit_cast<cudf::column_view const*>(bloomFilters);
    auto bloom_filter = spark_rapids_jni::bloom_filter_merge(input_bloom_filter);
    return std::bit_cast<jlong>(bloom_filter.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_BloomFilter_probe(JNIEnv* env,
                                                                           jclass,
                                                                           jlong bloomFilter,
                                                                           jlong cv)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::column_view const& input_column = *std::bit_cast<cudf::column_view const*>(cv);
    return cudf::jni::release_as_jlong(spark_rapids_jni::bloom_filter_probe(
      input_column, *(std::bit_cast<cudf::list_scalar*>(bloomFilter))));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_BloomFilter_probebuffer(
  JNIEnv* env, jclass, jlong bloomFilter, jlong bloomFilterSize, jlong cv)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::column_view const& input_column = *std::bit_cast<cudf::column_view const*>(cv);
    auto buf                              = std::bit_cast<uint8_t const*>(bloomFilter);
    return cudf::jni::release_as_jlong(spark_rapids_jni::bloom_filter_probe(
      input_column, cudf::device_span<uint8_t const>{buf, static_cast<size_t>(bloomFilterSize)}));
  }
  JNI_CATCH(env, 0);
}
}
