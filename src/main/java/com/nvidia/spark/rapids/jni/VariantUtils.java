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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.NativeDepsLoader;

import java.util.Objects;

/**
 * JNI bridge to cuDF's experimental Parquet Variant field extraction APIs.
 */
public class VariantUtils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private VariantUtils() {}

  private static void validateTargetType(DType targetType) {
    Objects.requireNonNull(targetType, "targetType");
    if (targetType != DType.STRING && targetType != DType.INT8 && targetType != DType.INT16 &&
        targetType != DType.INT32 && targetType != DType.INT64) {
      throw new IllegalArgumentException("unsupported Variant target type: " + targetType +
          "; supported types are STRING, INT8, INT16, INT32, and INT64");
    }
  }

  /**
   * Extract raw Variant-encoded value bytes at {@code path} from a Variant struct column.
   *
   * @param variantStruct Variant materialization: STRUCT(metadata LIST&lt;UINT8&gt;,
   *                      value LIST&lt;UINT8&gt;, optional shredded children...)
   * @param path JSONPath-like path accepted by cuDF's Variant extractor. Paths are expected to
   *             be ASCII object-field paths like {@code x}, {@code $.x}, or {@code $.x.y}.
   * @return LIST&lt;UINT8&gt; column of raw encoded Variant values
   */
  public static ColumnVector getVariantFieldValue(ColumnView variantStruct, String path) {
    return new ColumnVector(getVariantFieldValue(variantStruct.getNativeView(), path));
  }

  /**
   * Decode raw Variant-encoded value bytes into {@code targetType}. Supported target types are
   * {@link DType#STRING}, {@link DType#INT8}, {@link DType#INT16}, {@link DType#INT32}, and
   * {@link DType#INT64}.
   */
  public static ColumnVector castVariantValue(ColumnView valueBytes, DType targetType) {
    validateTargetType(targetType);
    return new ColumnVector(castVariantValue(
        valueBytes.getNativeView(), targetType.getTypeId().getNativeId()));
  }

  /**
   * Extract a Variant field and decode it into {@code targetType} in one native call.
   * Supported target types are {@link DType#STRING}, {@link DType#INT8}, {@link DType#INT16},
   * {@link DType#INT32}, and {@link DType#INT64}.
   */
  public static ColumnVector extractVariantField(
      ColumnView variantStruct, String path, DType targetType) {
    validateTargetType(targetType);
    return new ColumnVector(extractVariantField(
        variantStruct.getNativeView(), path, targetType.getTypeId().getNativeId()));
  }

  /**
   * Returns true when this JNI library was built against cuDF with Variant extraction APIs.
   */
  public static boolean isAvailable() {
    try {
      return isAvailableNative();
    } catch (UnsatisfiedLinkError e) {
      return false;
    }
  }

  private static native long getVariantFieldValue(long variantStructHandle, String path);

  private static native long castVariantValue(long valueBytesHandle, int cudfTypeId);

  private static native long extractVariantField(
      long variantStructHandle, String path, int cudfTypeId);

  private static native boolean isAvailableNative();
}
