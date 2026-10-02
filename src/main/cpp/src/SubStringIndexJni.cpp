/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.
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
#include "substring_index.hpp"

#include <bit>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_GpuSubstringIndexUtils_substringIndex(
  JNIEnv* env, jclass, jlong strings_handle, jlong delimiter, jint count)
{
  JNI_NULL_CHECK(env, strings_handle, "strings column handle is null", 0);
  JNI_NULL_CHECK(env, delimiter, "delimiter scalar handle is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input               = std::bit_cast<cudf::column_view const*>(strings_handle);
    auto const strings_column      = cudf::strings_column_view{*input};
    cudf::string_scalar* ss_scalar = std::bit_cast<cudf::string_scalar*>(delimiter);
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::substring_index(strings_column, *ss_scalar, count));
  }
  JNI_CATCH(env, 0);
}
}  // extern "C"
