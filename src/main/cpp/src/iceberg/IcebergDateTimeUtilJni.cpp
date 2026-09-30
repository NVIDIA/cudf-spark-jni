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
#include "iceberg/iceberg_datetime_util.hpp"

#include <bit>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_iceberg_IcebergDateTimeUtil_yearsFromEpoch(
  JNIEnv* env, jclass, jlong input)
{
  JNI_NULL_CHECK(env, input, "input column is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input_cv = std::bit_cast<cudf::column_view const*>(input);
    auto output         = spark_rapids_jni::years_from_epoch(*input_cv);
    return std::bit_cast<jlong>(output.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_iceberg_IcebergDateTimeUtil_monthsFromEpoch(JNIEnv* env,
                                                                             jclass,
                                                                             jlong input)
{
  JNI_NULL_CHECK(env, input, "input column is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input_cv = std::bit_cast<cudf::column_view const*>(input);
    auto output         = spark_rapids_jni::months_from_epoch(*input_cv);
    return std::bit_cast<jlong>(output.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_iceberg_IcebergDateTimeUtil_daysFromEpoch(
  JNIEnv* env, jclass, jlong input)
{
  JNI_NULL_CHECK(env, input, "input column is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input_cv = std::bit_cast<cudf::column_view const*>(input);
    auto output         = spark_rapids_jni::days_from_epoch(*input_cv);
    return std::bit_cast<jlong>(output.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_iceberg_IcebergDateTimeUtil_hoursFromEpoch(
  JNIEnv* env, jclass, jlong input)
{
  JNI_NULL_CHECK(env, input, "input column is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input_cv = std::bit_cast<cudf::column_view const*>(input);
    auto output         = spark_rapids_jni::hours_from_epoch(*input_cv);
    return std::bit_cast<jlong>(output.release());
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
