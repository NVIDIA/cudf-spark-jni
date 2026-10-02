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

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda/stream>

#include <cstdint>

namespace spark_rapids_jni {

/**
 * @brief Supported character set types for decoding.
 */
enum class charset_type : int32_t {
  GBK = 0,
};

/**
 * @brief How to handle malformed / unmappable byte sequences when decoding.
 *
 * Mirrors java.nio.charset.CodingErrorAction. REPORT is signaled to the caller
 * via decode_result::malformed; the caller is expected to raise an exception
 * from Java.
 */
enum class error_action : int32_t {
  REPLACE = 0,
  REPORT  = 1,
};

/**
 * @brief Result of a charset decode.
 *
 * When `malformed == false`, `output` is a UTF-8 strings column (malformed bytes
 * within rows are emitted as U+FFFD so the column is well-formed). When
 * `malformed == true` (only possible with `error_action::REPORT`), the caller
 * MUST treat `output` as discardable; the column may be null.
 */
struct decode_result {
  std::unique_ptr<cudf::column> output;
  bool malformed;
};

/**
 * @brief Decode a binary column from the specified charset encoding to a UTF-8 strings column.
 *
 * @param input The input column of type LIST<UINT8> (Spark BinaryType)
 * @param charset The source charset encoding
 * @param action How to treat malformed/unmappable sequences
 * @param stream CUDA stream used for device operations
 * @param mr Device memory resource used for allocating output column
 * @return A `decode_result` whose `output` is the decoded UTF-8 strings column
 *         and whose `malformed` flag is true (REPORT mode only) iff any input
 *         row contained a malformed or unmappable byte sequence; when set,
 *         `output` is discardable.
 */
[[nodiscard]] decode_result decode_charset(
  cudf::column_view const& input,
  charset_type charset,
  error_action action,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
