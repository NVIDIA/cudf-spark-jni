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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.DeletionVector;
import ai.rapids.cudf.NativeDepsLoader;

/**
 * JNI utilities for processing deletion vectors.
 */
public final class DeletionVectorUtils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private DeletionVectorUtils() {}

  /**
   * Computes the number of rows deleted by a serialized deletion vector on the GPU.
   *
   * @param deletionVectorInfo deletion vector and row-group metadata
   * @param maxChunkRows maximum number of row indexes to process at once
   * @return number of deleted rows in the specified row groups
   */
  public static long computeNumDeletedRows(
      DeletionVector.DeletionVectorInfo deletionVectorInfo, int maxChunkRows) {
    if (deletionVectorInfo == null) {
      throw new NullPointerException("deletionVectorInfo");
    }
    if (maxChunkRows <= 0) {
      throw new IllegalArgumentException("maxChunkRows must be positive");
    }
    return computeNumDeletedRows(
        deletionVectorInfo.serializedBitmap.getAddress(),
        deletionVectorInfo.serializedBitmap.getLength(),
        deletionVectorInfo.totalNumRows,
        deletionVectorInfo.rowGroupOffsets,
        deletionVectorInfo.rowGroupNumRows,
        deletionVectorInfo.isRetention,
        maxChunkRows);
  }

  private static native long computeNumDeletedRows(
      long serializedBitmapAddress,
      long serializedBitmapLength,
      int totalNumRows,
      long[] rowGroupOffsets,
      int[] rowGroupNumRows,
      boolean isRetention,
      int maxChunkRows);
}
