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

#include "cudf_jni_apis.hpp"
#include "iceberg/iceberg_bucket.hpp"
#include "jni_utils.hpp"

#include <bit>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_iceberg_IcebergBucket_computeBucket(
  JNIEnv* env, jclass, jlong input_column, jint num_buckets)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input_cv = std::bit_cast<cudf::column_view const*>(input_column);
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::compute_bucket(*input_cv, num_buckets, cudf::get_default_stream()));
  }
  JNI_CATCH(env, 0);
}
}
