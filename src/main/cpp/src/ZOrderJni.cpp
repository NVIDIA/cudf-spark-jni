/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include "zorder.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_ZOrder_interleaveBits(
  JNIEnv* env, jclass, jlongArray input_columns)
{
  JNI_NULL_CHECK(env, input_columns, "input is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jpointerArray<cudf::column_view> n_input_columns(env, input_columns);
    cudf::table_view tbl(n_input_columns.get_dereferenced());

    return cudf::jni::ptr_as_jlong(spark_rapids_jni::interleave_bits(tbl).release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_ZOrder_hilbertIndex(
  JNIEnv* env, jclass, jint num_bits, jlongArray input_columns)
{
  JNI_NULL_CHECK(env, input_columns, "input is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jpointerArray<cudf::column_view> n_input_columns(env, input_columns);
    cudf::table_view tbl(n_input_columns.get_dereferenced());

    return cudf::jni::ptr_as_jlong(spark_rapids_jni::hilbert_index(num_bits, tbl).release());
  }
  JNI_CATCH(env, 0);
}
}
