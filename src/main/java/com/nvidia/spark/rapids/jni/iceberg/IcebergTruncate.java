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
package com.nvidia.spark.rapids.jni.iceberg;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.NativeDepsLoader;

/** Utility class for Iceberg truncate transform operations */
public class IcebergTruncate {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Truncate int/long/decimal/string/binary types for Iceberg partitioning.
   *
   * For integer types, result is: value - (value % width)
   * For decimal types, result is: value - (value % width) on the underlying integer
   * For String(UTF8 encoding) type, 
   *   result is retaining only the first 'width' number of characters,
   *   note that this counts characters, not bytes.
   *   UTF8 characters have 1-4 bytes.
   * For Binary type, result is retaining only the first 'width' number of bytes.
   *
   * @param input int/long/decimal/string/binary column to truncate
   * @param width Truncation width
   * @return Truncated column
   */
  public static ColumnVector truncate(ColumnView input, int width) {
    return new ColumnVector(truncate(input.getNativeView(), width));
  }

  private static native long truncate(long nativeColumnView, int width);
}
