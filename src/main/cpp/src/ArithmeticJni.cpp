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
#include "exception_with_row_index.hpp"
#include "jni_utils.hpp"
#include "multiply.hpp"
#include "round_float.hpp"

#include <bit>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_Arithmetic_multiply(JNIEnv* env,
                                                                             jclass,
                                                                             jlong left,
                                                                             jboolean is_left_cv,
                                                                             jlong right,
                                                                             jboolean is_right_cv,
                                                                             jboolean ansi_enabled,
                                                                             jboolean is_try_mode)
{
  JNI_NULL_CHECK(env, left, "left input is null", 0);
  JNI_NULL_CHECK(env, right, "right input is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    if (is_left_cv && is_right_cv) {
      auto const& left_cv  = *std::bit_cast<cudf::column_view const*>(left);
      auto const& right_cv = *std::bit_cast<cudf::column_view const*>(right);
      return cudf::jni::release_as_jlong(
        spark_rapids_jni::multiply(left_cv, right_cv, ansi_enabled, is_try_mode));
    } else if (is_left_cv && !is_right_cv) {
      auto const& left_cv      = *std::bit_cast<cudf::column_view const*>(left);
      auto const& right_scalar = *std::bit_cast<cudf::scalar const*>(right);
      return cudf::jni::release_as_jlong(
        spark_rapids_jni::multiply(left_cv, right_scalar, ansi_enabled, is_try_mode));
    } else if (!is_left_cv && is_right_cv) {
      auto const& left_scalar = *std::bit_cast<cudf::scalar const*>(left);
      auto const& right_cv    = *std::bit_cast<cudf::column_view*>(right);
      return cudf::jni::release_as_jlong(
        spark_rapids_jni::multiply(left_scalar, right_cv, ansi_enabled, is_try_mode));
    } else {
      throw cudf::logic_error("Unsupported: Both left and right are scalars");
    }
  }
  // throw ExceptionWithRowIndex if an exception occurs if in ansi mode
  // ExceptionWithRowIndex contains the row number that caused the exception
  CATCH_EXCEPTION_WITH_ROW_INDEX(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_Arithmetic_round(JNIEnv* env,
                                                                          jclass,
                                                                          jlong input_ptr,
                                                                          jint decimal_places,
                                                                          jint rounding_method,
                                                                          jboolean is_ansi_mode)
{
  JNI_NULL_CHECK(env, input_ptr, "input is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::column_view* input     = std::bit_cast<cudf::column_view*>(input_ptr);
    cudf::rounding_method method = static_cast<cudf::rounding_method>(rounding_method);
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::round(*input, decimal_places, method, is_ansi_mode));
  }
  // throw ExceptionWithRowIndex if an exception occurs if in ansi mode
  CATCH_EXCEPTION_WITH_ROW_INDEX(env, 0);
}
}
