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
#include "decimal_utils.hpp"

#include <bit>

extern "C" {

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_DecimalUtils_multiply128(JNIEnv* env,
                                                          jclass,
                                                          jlong j_view_a,
                                                          jlong j_view_b,
                                                          jint j_product_scale,
                                                          bool cast_interim_result)
{
  JNI_NULL_CHECK(env, j_view_a, "column is null", 0);
  JNI_NULL_CHECK(env, j_view_b, "column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto view_a = std::bit_cast<cudf::column_view const*>(j_view_a);
    auto view_b = std::bit_cast<cudf::column_view const*>(j_view_b);
    auto scale  = static_cast<int>(j_product_scale);
    return cudf::jni::convert_table_for_return(
      env, cudf::jni::multiply_decimal128(*view_a, *view_b, scale, cast_interim_result));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_DecimalUtils_divide128(
  JNIEnv* env, jclass, jlong j_view_a, jlong j_view_b, jint j_quotient_scale, jboolean j_is_int_div)
{
  JNI_NULL_CHECK(env, j_view_a, "column is null", 0);
  JNI_NULL_CHECK(env, j_view_b, "column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto view_a          = std::bit_cast<cudf::column_view const*>(j_view_a);
    auto view_b          = std::bit_cast<cudf::column_view const*>(j_view_b);
    auto scale           = static_cast<int>(j_quotient_scale);
    auto is_int_division = static_cast<bool>(j_is_int_div);
    if (is_int_division) {
      return cudf::jni::convert_table_for_return(
        env, cudf::jni::integer_divide_decimal128(*view_a, *view_b, scale));
    } else {
      return cudf::jni::convert_table_for_return(
        env, cudf::jni::divide_decimal128(*view_a, *view_b, scale));
    }
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_DecimalUtils_remainder128(
  JNIEnv* env, jclass, jlong j_view_a, jlong j_view_b, jint j_remainder_scale)
{
  JNI_NULL_CHECK(env, j_view_a, "column is null", 0);
  JNI_NULL_CHECK(env, j_view_b, "column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto view_a = std::bit_cast<cudf::column_view const*>(j_view_a);
    auto view_b = std::bit_cast<cudf::column_view const*>(j_view_b);
    auto scale  = static_cast<int>(j_remainder_scale);
    return cudf::jni::convert_table_for_return(
      env, cudf::jni::remainder_decimal128(*view_a, *view_b, scale));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_DecimalUtils_add128(
  JNIEnv* env, jclass, jlong j_view_a, jlong j_view_b, jint j_target_scale)
{
  JNI_NULL_CHECK(env, j_view_a, "column is null", 0);
  JNI_NULL_CHECK(env, j_view_b, "column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const view_a = std::bit_cast<cudf::column_view const*>(j_view_a);
    auto const view_b = std::bit_cast<cudf::column_view const*>(j_view_b);
    auto const scale  = static_cast<int>(j_target_scale);
    return cudf::jni::convert_table_for_return(env,
                                               cudf::jni::add_decimal128(*view_a, *view_b, scale));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_DecimalUtils_subtract128(
  JNIEnv* env, jclass, jlong j_view_a, jlong j_view_b, jint j_target_scale)
{
  JNI_NULL_CHECK(env, j_view_a, "column is null", 0);
  JNI_NULL_CHECK(env, j_view_b, "column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const view_a = std::bit_cast<cudf::column_view const*>(j_view_a);
    auto const view_b = std::bit_cast<cudf::column_view const*>(j_view_b);
    auto const scale  = static_cast<int>(j_target_scale);
    return cudf::jni::convert_table_for_return(env,
                                               cudf::jni::sub_decimal128(*view_a, *view_b, scale));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_DecimalUtils_floatingPointToDecimal(
  JNIEnv* env, jclass, jlong j_input, jint output_type_id, jint precision, jint decimal_scale)
{
  JNI_NULL_CHECK(env, j_input, "j_input is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input = std::bit_cast<cudf::column_view const*>(j_input);
    cudf::jni::native_jlongArray output(env, 2);

    auto [casted_col, failure_row_id] = cudf::jni::floating_point_to_decimal(
      *input,
      cudf::data_type{static_cast<cudf::type_id>(output_type_id), static_cast<int>(decimal_scale)},
      precision);
    output[0] = cudf::jni::release_as_jlong(std::move(casted_col));
    output[1] = static_cast<jlong>(failure_row_id);
    return output.get_jArray();
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
