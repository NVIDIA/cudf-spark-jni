/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include "charset_decode.hpp"
#include "cudf_jni_apis.hpp"

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <bit>

extern "C" {

/**
 * Decode a binary column. Returns the native column pointer on success.
 * In REPORT mode, malformed/unmappable input causes this function to throw a
 * Java RuntimeException; the C++ jni_exception unwinds out through JNI_CATCH
 * and the caller observes the pending Java exception. The caller is expected
 * to translate it into Spark's MALFORMED_CHARACTER_CODING error class.
 */
JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_CharsetDecode_decodeNative(
  JNIEnv* env, jclass, jlong input_column, jint charset, jint error_action)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input = *std::bit_cast<cudf::column_view const*>(input_column);
    auto result =
      spark_rapids_jni::decode_charset(input,
                                       static_cast<spark_rapids_jni::charset_type>(charset),
                                       static_cast<spark_rapids_jni::error_action>(error_action),
                                       cudf::get_default_stream(),
                                       cudf::get_current_device_resource_ref());
    if (result.malformed) {
      cudf::jni::throw_java_exception(
        env,
        "com/nvidia/spark/rapids/jni/CharsetDecode$MalformedInputException",
        "malformed or unmappable input for charset decode");
    }
    return cudf::jni::release_as_jlong(std::move(result.output));
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
