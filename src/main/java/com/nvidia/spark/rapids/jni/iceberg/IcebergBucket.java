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

/**
 * Utility class for Iceberg bucket transform operations.
 *
 * The bucket transform computes a hash-based bucket assignment for partitioning.
 * For a value v, the bucket is computed as: (hash(v) & Integer.MAX_VALUE) % numBuckets
 *
 * This matches the Iceberg bucket transform specification:
 * https://iceberg.apache.org/spec/#bucket-transform-details
 */
public class IcebergBucket {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Compute bucket assignments for the input column.
   *
   * The bucket is computed as: (murmur3_hash(value) & Integer.MAX_VALUE) % numBuckets
   * Null values result in null bucket assignments.
   *
   * Supported input types:
   * - Integer types (INT32, INT64)
   * - Decimal types (DECIMAL32, DECIMAL64, DECIMAL128)
   * - Date types (TIMESTAMP_DAYS)
   * - Timestamp types (TIMESTAMP_MICROSECONDS)
   * - String types
   * - Binary types (LIST of UINT8)
   *
   * @param input Column to compute bucket assignments for
   * @param numBuckets Number of buckets (must be positive)
   * @return INT32 column containing bucket assignments (0 to numBuckets-1), with nulls preserved
   */
  public static ColumnVector computeBucket(ColumnView input, int numBuckets) {
    if (numBuckets <= 0) {
      throw new IllegalArgumentException("numBuckets must be positive, got: " + numBuckets);
    }
    return new ColumnVector(computeBucket(input.getNativeView(), numBuckets));
  }

  private static native long computeBucket(long nativeColumnView, int numBuckets);
}

