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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.*;

/**
 * Utility methods for breaking apart and reassembling 64-bit values during aggregations
 * to enable hash-based aggregations and detect overflows.
 *
 * Note that this is intended to be temporary until CUDF can add in SUM aggregations
 * with overflow detection themselves.
 */
public class Aggregation64Utils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Extract a 32-bit chunk from a 64-bit value.
   * @param col column of 64-bit values (e.g.: INT64)
   * @param outType integer type to use for the output column (e.g.: UINT32 or INT32)
   * @param chunkIdx index of the 32-bit chunk to extract where 0 is the least significant chunk
   *                 and 1 is the most significant chunk
   * @return column containing the specified 32-bit chunk of the input column values. A null input
   *                row will result in a corresponding null output row.
   */
  public static ColumnVector extractInt32Chunk(ColumnView col, DType outType, int chunkIdx) {
    return new ColumnVector(extractInt32Chunk(col.getNativeView(),
        outType.getTypeId().getNativeId(), chunkIdx));
  }

  /**
   * Reassemble a column of 64-bit values from a table of two 64-bit integer columns and check
   * for overflow. The 64-bit value is reconstructed by overlapping the 64-bit values by 32-bits.
   * The least significant 32-bits of the least significant 64-bit value are used directly as the
   * least significant 32-bits of the final 64-bit value, and the remaining 32-bits are added to
   * the next most significant 64-bit value.
   *
   * @param chunks table of two 64-bit integer columns with the columns ordered from least
   *               significant to most significant. The last column must be of type INT64.
   * @param type the type to use for the resulting 64-bit value column
   * @return table containing a boolean column and a 64-bit value column of the requested type.
   *         The boolean value will be true if an overflow was detected for that row's value when
   *         it was reassembled. A null input row will result in a corresponding null output row.
   */
  public static Table combineInt64SumChunks(Table chunks, DType type) {
    return new Table(combineInt64SumChunks(chunks.getNativeView(),
        type.getTypeId().getNativeId(),
        type.getScale()));
  }

  private static native long extractInt32Chunk(long columnView, int outTypeId, int chunkIdx);

  private static native long[] combineInt64SumChunks(long chunksTableView, int dtype, int scale);
}