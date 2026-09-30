/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/types.hpp>

namespace spark_rapids_jni {

constexpr char const* JNI_EXCEPTION_WITH_ROW_INDEX_CLASS =
  "com/nvidia/spark/rapids/jni/ExceptionWithRowIndex";

/**
 * @brief Exception class indicating which row in a column caused an error.
 *
 * Typically it's used in ANSI mode to indicate which row has an error.
 */
class exception_with_row_index : public std::runtime_error {
 public:
  /**
   * @brief Constructs a exception_with_row_index with a row index which caused an error.
   */
  exception_with_row_index(cudf::size_type row_index)
    : std::runtime_error(""), _row_index(row_index)
  {
  }

  /**
   * @brief Get the row index which cuased error
   *
   * @return cudf::size_type row index
   */
  [[nodiscard]] cudf::size_type get_row_index() const { return _row_index; }

 private:
  cudf::size_type _row_index;
};

// catch a exception_with_row_index exception and throw a Java exception.
// This macro is used in JNI functions to throw an ExceptionWithRowIndex if error occurs
// ExceptionWithRowIndex contains the row number that caused the exception
#define CATCH_EXCEPTION_WITH_ROW_INDEX(env, ret_val)                                        \
  JNI_CATCH_BEGIN(env, ret_val)                                                             \
  catch (spark_rapids_jni::exception_with_row_index const& e)                               \
  {                                                                                         \
    if (env->ExceptionOccurred()) { return ret_val; }                                       \
    jclass ex_class = env->FindClass(spark_rapids_jni::JNI_EXCEPTION_WITH_ROW_INDEX_CLASS); \
    if (ex_class != NULL) {                                                                 \
      jmethodID ctor_id = env->GetMethodID(ex_class, "<init>", "(I)V");                     \
      if (ctor_id != NULL) {                                                                \
        jint e_row_index  = static_cast<jint>(e.get_row_index());                           \
        jobject jni_error = env->NewObject(ex_class, ctor_id, e_row_index);                 \
        if (jni_error != NULL) { env->Throw((jthrowable)jni_error); }                       \
      }                                                                                     \
    }                                                                                       \
    return ret_val;                                                                         \
  }                                                                                         \
  CATCH_SPECIAL_EXCEPTION(env, ret_val)                                                     \
  CATCH_STD_EXCEPTION(env, ret_val)
}  // namespace spark_rapids_jni
