/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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
#include "jni_utils.hpp"
#include "regex_rewrite_utils.hpp"

#include <bit>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_RegexRewriteUtils_literalRangePattern(
  JNIEnv* env, jclass, jlong input, jlong target, jint d, jint start, jint end)
{
  JNI_NULL_CHECK(env, input, "input column is null", 0);
  JNI_NULL_CHECK(env, target, "target is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::column_view* cv = std::bit_cast<cudf::column_view*>(input);
    cudf::strings_column_view scv(*cv);
    cudf::string_scalar* ss_scalar = std::bit_cast<cudf::string_scalar*>(target);
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::literal_range_pattern(scv, *ss_scalar, d, start, end));
  }
  JNI_CATCH(env, 0);
}
}
