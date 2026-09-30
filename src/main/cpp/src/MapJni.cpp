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
#include "jni_utils.hpp"
#include "map.hpp"

#include <bit>

extern "C" {

JNIEXPORT jlong Java_com_nvidia_spark_rapids_jni_Map_sort(JNIEnv* env,
                                                          jclass,
                                                          jlong map_haldle,
                                                          jboolean is_descending)
{
  JNI_NULL_CHECK(env, map_haldle, "column is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto sort_order = is_descending ? cudf::order::DESCENDING : cudf::order::ASCENDING;
    cudf::column_view const& map_view = *std::bit_cast<cudf::column_view const*>(map_haldle);
    return cudf::jni::release_as_jlong(spark_rapids_jni::sort_map_column(map_view, sort_order));
  }
  JNI_CATCH(env, 0);
}
}
