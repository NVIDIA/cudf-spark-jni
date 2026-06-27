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

#include "cudf_jni_apis.hpp"
#include "pageable_pool_resource.hpp"

#include <cuda/memory_resource>

#define CATCH_PAGEABLE_POOL_EXHAUSTED(env, ret_val)                        \
  JNI_CATCH_BEGIN(env, ret_val)                                            \
  catch (spark_rapids_jni::pageable_pool_exhausted const&)                 \
  {                                                                        \
    return -1; /* Pool exhausted — caller falls back to regular malloc. */ \
  }                                                                        \
  CATCH_SPECIAL_EXCEPTION(env, ret_val)                                    \
  CATCH_STD_EXCEPTION(env, ret_val)

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_PageableMemoryPool_newPageablePoolMemoryResource(
  JNIEnv* env, jclass, jlong pool_size, jlong /* max_size unused */, jint pretouch_threads)
{
  JNI_TRY
  {
    auto* pool = new spark_rapids_jni::pageable_pool_resource(
      cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>(
        spark_rapids_jni::pageable_memory_resource{}),
      static_cast<std::size_t>(pool_size),
      static_cast<int>(pretouch_threads));
    return reinterpret_cast<jlong>(pool);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_PageableMemoryPool_releasePageablePoolMemoryResource(
  JNIEnv* env, jclass, jlong pool_ptr)
{
  JNI_TRY { delete reinterpret_cast<spark_rapids_jni::pageable_pool_resource*>(pool_ptr); }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_PageableMemoryPool_allocFromPageablePool(JNIEnv* env,
                                                                                     jclass,
                                                                                     jlong pool_ptr,
                                                                                     jlong size)
{
  JNI_TRY
  {
    auto* pool = reinterpret_cast<spark_rapids_jni::pageable_pool_resource*>(pool_ptr);
    void* ret  = pool->allocate_sync(static_cast<std::size_t>(size));
    return reinterpret_cast<jlong>(ret);
  }
  CATCH_PAGEABLE_POOL_EXHAUSTED(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_PageableMemoryPool_freeFromPageablePool(
  JNIEnv* env, jclass, jlong pool_ptr, jlong ptr, jlong size)
{
  JNI_TRY
  {
    auto* pool = reinterpret_cast<spark_rapids_jni::pageable_pool_resource*>(pool_ptr);
    pool->deallocate_sync(reinterpret_cast<void*>(ptr), static_cast<std::size_t>(size));
  }
  JNI_CATCH(env, );
}

}  // extern "C"
