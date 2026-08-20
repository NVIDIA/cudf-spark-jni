/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.
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
#include "row_conversion.hpp"

#include <bit>

extern "C" {

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_RowConversion_convertToRowsFixedWidthOptimized(JNIEnv* env,
                                                                                jclass,
                                                                                jlong input_table)
{
  JNI_NULL_CHECK(env, input_table, "input table is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::table_view const* n_input_table = std::bit_cast<cudf::table_view const*>(input_table);
    std::vector<std::unique_ptr<cudf::column>> cols =
      spark_rapids_jni::convert_to_rows_fixed_width_optimized(*n_input_table);
    int const num_columns = cols.size();
    cudf::jni::native_jlongArray outcol_handles(env, num_columns);
    std::transform(cols.begin(), cols.end(), outcol_handles.begin(), [](auto& col) {
      return cudf::jni::release_as_jlong(col);
    });
    return outcol_handles.get_jArray();
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_RowConversion_convertToRows(JNIEnv* env, jclass, jlong input_table)
{
  JNI_NULL_CHECK(env, input_table, "input table is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::table_view const* n_input_table = std::bit_cast<cudf::table_view const*>(input_table);
    std::vector<std::unique_ptr<cudf::column>> cols =
      spark_rapids_jni::convert_to_rows(*n_input_table);
    int const num_columns = cols.size();
    cudf::jni::native_jlongArray outcol_handles(env, num_columns);
    std::transform(cols.begin(), cols.end(), outcol_handles.begin(), [](auto& col) {
      return cudf::jni::release_as_jlong(col);
    });
    return outcol_handles.get_jArray();
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_RowConversion_convertFromRowsFixedWidthOptimized(
  JNIEnv* env, jclass, jlong input_column, jintArray types, jintArray scale)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  JNI_NULL_CHECK(env, types, "types is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::lists_column_view const list_input{*std::bit_cast<cudf::column_view*>(input_column)};
    cudf::jni::native_jintArray n_types(env, types);
    cudf::jni::native_jintArray n_scale(env, scale);
    if (n_types.size() != n_scale.size()) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "types and scales must match size", NULL);
    }
    std::vector<cudf::data_type> types_vec;
    std::transform(n_types.begin(),
                   n_types.end(),
                   n_scale.begin(),
                   std::back_inserter(types_vec),
                   [](jint type, jint scale) { return cudf::jni::make_data_type(type, scale); });
    std::unique_ptr<cudf::table> result =
      spark_rapids_jni::convert_from_rows_fixed_width_optimized(list_input, types_vec);
    return cudf::jni::convert_table_for_return(env, result);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_RowConversion_convertFromRows(
  JNIEnv* env, jclass, jlong input_column, jintArray types, jintArray scale)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  JNI_NULL_CHECK(env, types, "types is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::lists_column_view const list_input{*std::bit_cast<cudf::column_view*>(input_column)};
    cudf::jni::native_jintArray n_types(env, types);
    cudf::jni::native_jintArray n_scale(env, scale);
    if (n_types.size() != n_scale.size()) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "types and scales must match size", NULL);
    }
    std::vector<cudf::data_type> types_vec;
    std::transform(n_types.begin(),
                   n_types.end(),
                   n_scale.begin(),
                   std::back_inserter(types_vec),
                   [](jint type, jint scale) { return cudf::jni::make_data_type(type, scale); });
    std::unique_ptr<cudf::table> result =
      spark_rapids_jni::convert_from_rows(list_input, types_vec);
    return cudf::jni::convert_table_for_return(env, result);
  }
  JNI_CATCH(env, 0);
}
}
