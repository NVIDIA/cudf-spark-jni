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
#include "map_zip_with_utils.hpp"

#include <bit>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_GpuMapZipWithUtils_mapZip(
  JNIEnv* env, jclass, jlong input_column1, jlong input_column2)
{
  JNI_NULL_CHECK(env, input_column1, "input column1 is null", 0);
  JNI_NULL_CHECK(env, input_column2, "input column2 is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::lists_column_view col1{*std::bit_cast<cudf::column_view const*>(input_column1)};
    cudf::lists_column_view col2{*std::bit_cast<cudf::column_view const*>(input_column2)};
    return cudf::jni::release_as_jlong(spark_rapids_jni::map_zip(col1, col2));
  }
  JNI_CATCH(env, 0);
}
}
