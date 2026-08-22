/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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
#include "hyper_log_log_plus_plus.hpp"
#include "hyper_log_log_plus_plus_host_udf.hpp"

#include <bit>

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_HyperLogLogPlusPlusHostUDF_createHLLPPHostUDF(JNIEnv* env,
                                                                               jclass,
                                                                               jint agg_type,
                                                                               int precision)
{
  JNI_TRY
  {
    auto udf_ptr = [&] {
      // The value of agg_type must be sync with
      // `HyperLogLogPlusPlusHostUDF.java#AggregationType`.
      switch (agg_type) {
        case 0: return spark_rapids_jni::create_hllpp_reduction_host_udf(precision);
        case 1: return spark_rapids_jni::create_hllpp_reduction_merge_host_udf(precision);
        case 2: return spark_rapids_jni::create_hllpp_groupby_host_udf(precision);
        case 3: return spark_rapids_jni::create_hllpp_groupby_merge_host_udf(precision);
        default: CUDF_FAIL("Invalid aggregation type.");
      }
    }();
    CUDF_EXPECTS(udf_ptr != nullptr, "Invalid HyperLogLogPlusPlus(HLLPP) UDF instance.");

    return std::bit_cast<jlong>(udf_ptr);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_HyperLogLogPlusPlusHostUDF_estimateDistinctValueFromSketches(
  JNIEnv* env, jclass, jlong sketches, jint precision)
{
  JNI_NULL_CHECK(env, sketches, "Sketch column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const sketch_view = std::bit_cast<cudf::column_view const*>(sketches);
    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::estimate_from_hll_sketches(*sketch_view, precision).release());
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
