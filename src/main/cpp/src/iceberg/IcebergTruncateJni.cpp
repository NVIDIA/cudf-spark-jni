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
#include "iceberg/iceberg_truncate.hpp"
#include "jni_utils.hpp"

#include <bit>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_iceberg_IcebergTruncate_truncate(
  JNIEnv* env, jclass, jlong input_column, jint width)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input_cv = std::bit_cast<cudf::column_view const*>(input_column);
    auto const type_id  = input_cv->type().id();
    // Switch on type_id to call appropriate truncate function
    switch (type_id) {
      case cudf::type_id::INT32:
      case cudf::type_id::INT64:
      case cudf::type_id::DECIMAL32:
      case cudf::type_id::DECIMAL64:
      case cudf::type_id::DECIMAL128:
        return cudf::jni::release_as_jlong(
          spark_rapids_jni::truncate_integral(*input_cv, width, cudf::get_default_stream()));
      case cudf::type_id::STRING:
        return cudf::jni::release_as_jlong(
          spark_rapids_jni::truncate_string(*input_cv, width, cudf::get_default_stream()));
      case cudf::type_id::LIST:
        return cudf::jni::release_as_jlong(
          spark_rapids_jni::truncate_binary(*input_cv, width, cudf::get_default_stream()));
      default:
        JNI_THROW_NEW(
          env, "java/lang/IllegalArgumentException", "Unsupported type for truncation", 0);
    }
  }
  JNI_CATCH(env, 0);
}
}
