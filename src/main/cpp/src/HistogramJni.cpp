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
#include "histogram.hpp"

#include <bit>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_Histogram_createHistogramIfValid(
  JNIEnv* env, jclass, jlong values_handle, jlong frequencies_handle, jboolean output_as_lists)
{
  JNI_NULL_CHECK(env, values_handle, "values_handle is null", 0);
  JNI_NULL_CHECK(env, frequencies_handle, "frequencies_handle is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const values      = std::bit_cast<cudf::column_view const*>(values_handle);
    auto const frequencies = std::bit_cast<cudf::column_view const*>(frequencies_handle);
    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::create_histogram_if_valid(*values, *frequencies, output_as_lists)
        .release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_Histogram_percentileFromHistogram(
  JNIEnv* env, jclass, jlong input_handle, jdoubleArray jpercentages, jboolean output_as_lists)
{
  JNI_NULL_CHECK(env, input_handle, "input_handle is null", 0);
  JNI_NULL_CHECK(env, jpercentages, "jpercentages is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const input       = std::bit_cast<cudf::column_view const*>(input_handle);
    auto const percentages = [&] {
      auto const native_percentages = cudf::jni::native_jdoubleArray(env, jpercentages);
      return std::vector<double>(native_percentages.begin(), native_percentages.end());
    }();
    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::percentile_from_histogram(*input, percentages, output_as_lists).release());
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
