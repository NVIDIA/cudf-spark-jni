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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.BinaryOp;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Scalar;
import java.lang.management.ManagementFactory;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;

public class StringUtils {

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  // Stores the sequence ID of calling generate UUIDs.
  private static AtomicLong sequence = new AtomicLong(0);

  /**
   * Generate a seed for UUID generation.
   * The seed is generated based on the current time in nanoseconds, the process
   * name, the GPU UUID, and a sequence ID that increments with each call to
   * this method. This method ensures (do best effort) that the seed is unique
   * across different runs, including different Spark jobs, different executions
   * of the same job, set backward of the clock, etc.
   *
   * @return A seed for UUID generation.
   */
  private static long randomSeed() {
    long seed = System.nanoTime();
    String processName = ManagementFactory.getRuntimeMXBean().getName();
    byte[] gpuUUID = Cuda.getGpuUuid();
    seed = seed * 37 + processName.hashCode();
    seed = seed * 37 + Arrays.hashCode(gpuUUID);
    seed = seed * 37 + sequence.incrementAndGet();
    return seed;
  }

  /**
   * Generate a column of UUIDs (String type) with `rowCount` rows.
   * Spark uses `Truly Random or Pseudo-Random` UUID type which is described in
   * the section 4.4 of [RFC4122](https://datatracker.ietf.org/doc/html/rfc4122),
   * The variant in UUID is 2 and the version in UUID is 4. This implementation
   * generates UUIDs in the same format, but does not generate the same UUIDs as
   * Spark. This function is indeterministic, meaning that it will generate
   * different UUIDs each time it is called, even with the same row count.
   * The UUIDs are generated using a seed based on the current time, process name,
   * GPU UUID and running sequence index, ensuring uniqueness across different
   * runs.
   *
   * E.g.: "123e4567-e89b-12d3-a456-426614174000"
   *
   * @param rowCount Number of UUIDs to generate
   * @return ColumnVector containing UUIDs
   */
  public static ColumnVector randomUUIDs(int rowCount) {
    long seed = randomSeed();
    return new ColumnVector(randomUUIDs(rowCount, seed));
  }

  /**
   * Generate a column of UUIDs (String type) with `rowCount` rows.
   * The same `seed` will generate the same sequence of UUIDs.
   * Spark uses `Truly Random or Pseudo-Random` UUID type which is described in
   * the section 4.4 of [RFC4122](https://datatracker.ietf.org/doc/html/rfc4122),
   * The variant in UUID is 2 and the version in UUID is 4. This implementation
   * generates UUIDs in the same format, but does not generate the same UUIDs as
   * Spark.
   *
   * E.g.: "123e4567-e89b-12d3-a456-426614174000"
   *
   * @param rowCount Number of UUIDs to generate
   * @param seed Seed for UUID generation
   * @return ColumnVector containing UUIDs
   */
  public static ColumnVector randomUUIDsWithSeed(int rowCount, long seed) {
    return new ColumnVector(randomUUIDs(rowCount, seed));
  }

  /**
   * Reverse each string using Spark {@code UTF8String.reverse} semantics.
   * Character widths follow Spark's {@code numBytesForFirstByte} and are clamped to the
   * bytes remaining in each row (SPARK-57507), so truncated trailing UTF-8 sequences do
   * not read past the row boundary.
   *
   * @param input strings column
   * @return new strings column with reversed contents
   */
  public static ColumnVector reverseStrings(ColumnView input) {
    return new ColumnVector(reverseStrings(input.getNativeView()));
  }

  /**
   * Match each input string against the LIKE pattern at the corresponding row.
   *
   * <p>The escape character applies to every pattern. A null input or pattern produces a null
   * result. Invalid escape sequences in non-null patterns throw {@link ExceptionWithRowIndex} for
   * the first row where both the input and pattern are non-null.</p>
   *
   * @param input strings to match
   * @param patterns row-aligned LIKE patterns
   * @param escapeChar scalar string containing the escape character
   * @return a BOOL8 column containing the row-aligned match results
   */
  public static ColumnVector like(ColumnView input, ColumnView patterns, Scalar escapeChar) {
    assert input.getType().equals(DType.STRING) : "input column must be a String";
    assert patterns.getType().equals(DType.STRING) : "patterns column must be a String";
    assert input.getRowCount() == patterns.getRowCount()
        : "input and patterns must have the same number of rows";
    assert escapeChar != null : "escapeChar must not be null";
    assert escapeChar.getType().equals(DType.STRING) : "escapeChar must be a string scalar";

    try (Scalar empty = Scalar.fromString("");
         ColumnVector normalizedPatterns = patterns.replaceNulls(empty);
         ColumnVector result = new ColumnVector(like(
             input.getNativeView(), normalizedPatterns.getNativeView(), escapeChar.getScalarHandle()))) {
      return result.mergeAndSetValidity(BinaryOp.BITWISE_AND, input, patterns);
    }
  }

  private static native long randomUUIDs(int rowCount, long seed);

  private static native long reverseStrings(long inputHandle) throws CudfException;

  private static native long like(long inputHandle, long patternsHandle, long escapeCharHandle)
      throws CudfException, ExceptionWithRowIndex;
}
