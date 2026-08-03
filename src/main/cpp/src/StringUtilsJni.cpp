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
#include "reverse_strings.hpp"
#include "uuid.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_StringUtils_randomUUIDs(JNIEnv* env,
                                                                                 jclass,
                                                                                 jint row_count,
                                                                                 jlong seed)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    return cudf::jni::release_as_jlong(spark_rapids_jni::random_uuids(row_count, seed));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_StringUtils_reverseStrings(JNIEnv* env, jclass, jlong input_handle)
{
  JNI_NULL_CHECK(env, input_handle, "input column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::column_view const*>(input_handle);
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::reverse_strings(cudf::strings_column_view{*input}));
  }
  JNI_CATCH(env, 0);
}
}
