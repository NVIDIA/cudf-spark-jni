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

#include "aggregation64_utils.hpp"
#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"

#include <bit>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_Aggregation64Utils_extractInt32Chunk(
  JNIEnv* env, jclass, jlong j_column_view, jint j_out_dtype, jint j_chunk_idx)
{
  JNI_NULL_CHECK(env, j_column_view, "column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto cview = std::bit_cast<cudf::column_view const*>(j_column_view);
    auto dtype = cudf::jni::make_data_type(j_out_dtype, 0);
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::extract_chunk32_from_64bit(*cview, dtype, j_chunk_idx));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_Aggregation64Utils_combineInt64SumChunks(
  JNIEnv* env, jclass, jlong j_table_view, jint j_dtype, jint j_scale)
{
  JNI_NULL_CHECK(env, j_table_view, "table is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto tview = std::bit_cast<cudf::table_view const*>(j_table_view);
    std::unique_ptr<cudf::table> result =
      spark_rapids_jni::assemble64_from_sum(*tview, cudf::jni::make_data_type(j_dtype, j_scale));
    return cudf::jni::convert_table_for_return(env, result);
  }
  JNI_CATCH(env, 0);
}
}
