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
#include "jni_utils.hpp"

#include <cudf/io/experimental/deletion_vectors.hpp>

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_DeletionVectorUtils_computeNumDeletedRows(
  JNIEnv* env,
  jclass,
  jlong serialized_bitmap_address,
  jlong serialized_bitmap_length,
  jint total_num_rows,
  jlongArray row_group_offsets,
  jintArray row_group_num_rows,
  jboolean is_retention,
  jint max_chunk_rows)
{
  JNI_NULL_CHECK(env, serialized_bitmap_address, "serialized bitmap address is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jlongArray n_row_group_offsets(env, row_group_offsets);
    cudf::jni::native_jintArray n_row_group_num_rows(env, row_group_num_rows);

    auto offsets = std::vector<std::size_t>{};
    offsets.reserve(n_row_group_offsets.size());
    std::transform(n_row_group_offsets.begin(),
                   n_row_group_offsets.end(),
                   std::back_inserter(offsets),
                   [](jlong offset) { return static_cast<std::size_t>(offset); });

    auto const bitmap = cudf::host_span<cuda::std::byte const>{
      reinterpret_cast<cuda::std::byte const*>(serialized_bitmap_address),
      static_cast<std::size_t>(serialized_bitmap_length)};
    auto const info = cudf::io::parquet::experimental::deletion_vector_info{
      .serialized_roaring_bitmaps = {bitmap},
      .deletion_vector_row_counts = {total_num_rows},
      .row_group_offsets          = std::move(offsets),
      .row_group_num_rows         = n_row_group_num_rows.to_vector(),
      .are_retention_vectors      = static_cast<bool>(is_retention)};

    return static_cast<jlong>(cudf::io::parquet::experimental::compute_num_deleted_rows(
      info, static_cast<cudf::size_type>(max_chunk_rows)));
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
