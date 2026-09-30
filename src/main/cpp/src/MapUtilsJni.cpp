/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
#include "jni_utils.hpp"
#include "map_utils.hpp"

#include <bit>

extern "C" {

JNIEXPORT jboolean JNICALL Java_com_nvidia_spark_rapids_jni_MapUtils_isValidMap(
  JNIEnv* env, jclass, jlong input_handle, jboolean throw_on_null_key)
{
  JNI_NULL_CHECK(env, input_handle, "input column is null", JNI_FALSE);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const& input = *std::bit_cast<cudf::column_view const*>(input_handle);
    return spark_rapids_jni::is_valid_map(input, static_cast<bool>(throw_on_null_key)) ? JNI_TRUE
                                                                                       : JNI_FALSE;
  }
  JNI_CATCH(env, JNI_FALSE);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_MapUtils_mapFromEntries(
  JNIEnv* env, jclass, jlong input_handle, jboolean throw_on_null_key)
{
  JNI_NULL_CHECK(env, input_handle, "input column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const& input = *std::bit_cast<cudf::column_view const*>(input_handle);
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::map_from_entries(input, static_cast<bool>(throw_on_null_key)));
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
