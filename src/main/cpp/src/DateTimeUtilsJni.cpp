/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.
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
#include "datetime_utils.hpp"

#include <bit>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_DateTimeUtils_rebaseGregorianToJulian(
  JNIEnv* env, jclass, jlong input)
{
  JNI_NULL_CHECK(env, input, "input column is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input_cv = std::bit_cast<cudf::column_view const*>(input);
    auto output         = spark_rapids_jni::rebase_gregorian_to_julian(*input_cv);
    return std::bit_cast<jlong>(output.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_DateTimeUtils_rebaseJulianToGregorian(
  JNIEnv* env, jclass, jlong input)
{
  JNI_NULL_CHECK(env, input, "input column is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input_cv = std::bit_cast<cudf::column_view const*>(input);
    auto output         = spark_rapids_jni::rebase_julian_to_gregorian(*input_cv);
    return std::bit_cast<jlong>(output.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_DateTimeUtils_truncateWithColumnFormat(
  JNIEnv* env, jclass, jlong datetime, jlong format)
{
  JNI_NULL_CHECK(env, datetime, "input datetime is null", 0);
  JNI_NULL_CHECK(env, format, "input format is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const datetime_cv = std::bit_cast<cudf::column_view const*>(datetime);
    auto const format_cv   = std::bit_cast<cudf::column_view const*>(format);
    return std::bit_cast<jlong>(spark_rapids_jni::truncate(*datetime_cv, *format_cv).release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_DateTimeUtils_truncateWithScalarFormat(
  JNIEnv* env, jclass, jlong datetime, jstring format)
{
  JNI_NULL_CHECK(env, datetime, "input datetime is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const datetime_cv = std::bit_cast<cudf::column_view const*>(datetime);
    auto const format_jstr = cudf::jni::native_jstring(env, format);
    auto const format      = std::string(format_jstr.get(), format_jstr.size_bytes());
    return std::bit_cast<jlong>(spark_rapids_jni::truncate(*datetime_cv, format).release());
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
