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
#include "cudf_jni_apis.hpp"

extern "C" {

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_DeviceAttr_isDeviceIntegrated(JNIEnv* env,
                                                                                      jclass)
{
  JNI_TRY
  {
    int device{};
    CUDF_CUDA_TRY(cudaGetDevice(&device));

    int integrated{};
    CUDF_CUDA_TRY(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device));

    return integrated;
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"