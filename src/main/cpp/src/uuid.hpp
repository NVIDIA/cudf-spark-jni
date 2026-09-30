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
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace spark_rapids_jni {

/**
 * @brief Generate a column of UUIDs (String type) with `rowCount` rows.
 * Spark uses `Truly Random or Pseudo-Random` UUID type which is described in
 * the section 4.4 of [RFC4122](https://datatracker.ietf.org/doc/html/rfc4122),
 * The variant in UUID is 2 and the version in UUID is 4. This implementation
 * generates UUIDs in the same format, but does not generate the same UUIDs as
 * Spark. This function is deterministic, meaning that it will generate
 * the same UUIDs for the same seed and row count.
 *
 * E.g.: "123e4567-e89b-12d3-a456-426614174000"
 *
 * @param rowCount Number of UUIDs to generate
 * @param seed Seed for random number generation
 * @return ColumnVector containing UUIDs
 */
std::unique_ptr<cudf::column> random_uuids(
  int row_count,
  long seed,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
