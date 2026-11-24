/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "jni_utils.hpp"
#include "range_sequence.hpp"

#include <cstdint>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_RangeSequence_generateNative(
  JNIEnv* env, jclass, jlongArray j_starts, jintArray j_sizes)
{
  JNI_NULL_CHECK(env, j_starts, "starts is null", 0);
  JNI_NULL_CHECK(env, j_sizes, "sizes is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const starts = cudf::jni::native_jlongArray(env, j_starts).to_vector<int64_t>();
    auto const sizes  = cudf::jni::native_jintArray(env, j_sizes).to_vector<cudf::size_type>();
    return cudf::jni::release_as_jlong(spark_rapids_jni::generate_range_sequence(starts, sizes));
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
