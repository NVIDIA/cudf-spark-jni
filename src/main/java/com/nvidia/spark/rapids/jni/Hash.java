/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.
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
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.NativeDepsLoader;

public class Hash {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  // there doesn't appear to be a useful constant in spark to reference. this could break.
  static final long DEFAULT_XXHASH64_SEED = 42;

  public static final int MAX_STACK_DEPTH = getMaxStackDepth();

  /**
   * Create a new vector containing spark's 32-bit murmur3 hash of each row in the table.
   * Spark's murmur3 hash uses a different tail processing algorithm.
   *
   * @param seed integer seed for the murmur3 hash function
   * @param columns array of columns to hash, must have identical number of rows.
   * @return the new ColumnVector of 32-bit values representing each row's hash value.
   */
  public static ColumnVector murmurHash32(int seed, ColumnView columns[]) {
    if (columns.length < 1) {
      throw new IllegalArgumentException("Murmur3 hashing requires at least 1 column of input");
    }
    long[] columnViews = new long[columns.length];
    long size = columns[0].getRowCount();

    for(int i = 0; i < columns.length; i++) {
      assert columns[i] != null : "Column vectors passed may not be null";
      assert columns[i].getRowCount() == size : "Row count mismatch, all columns must be the same size";
      assert !columns[i].getType().isDurationType() : "Unsupported column type Duration";
      columnViews[i] = columns[i].getNativeView(); 
    }
    return new ColumnVector(murmurHash32(seed, columnViews));
  }

  public static ColumnVector murmurHash32(ColumnView columns[]) {
    return murmurHash32(0, columns);
  }

  /**
   * Create a new vector containing the xxhash64 hash of each row in the table.
   *
   * @param seed integer seed for the xxhash64 hash function
   * @param columns array of columns to hash, must have identical number of rows.
   * @return the new ColumnVector of 64-bit values representing each row's hash value.
   */
  public static ColumnVector xxhash64(long seed, ColumnView columns[]) {
    if (columns.length < 1) {
      throw new IllegalArgumentException("xxhash64 hashing requires at least 1 column of input");
    }
    long[] columnViews = new long[columns.length];
    long size = columns[0].getRowCount();

    for(int i = 0; i < columns.length; i++) {
      assert columns[i] != null : "Column vectors passed may not be null";
      assert columns[i].getRowCount() == size : "Row count mismatch, all columns must be the same size";
      assert !columns[i].getType().isDurationType() : "Unsupported column type Duration";
      columnViews[i] = columns[i].getNativeView(); 
    }
    return new ColumnVector(xxhash64(seed, columnViews));
  }

  public static ColumnVector xxhash64(ColumnView columns[]) {
    return xxhash64(DEFAULT_XXHASH64_SEED, columns);
  }

  public static ColumnVector hiveHash(ColumnView columns[]) {
    if (columns.length < 1) {
      throw new IllegalArgumentException("Hive hashing requires at least 1 column of input");
    }
    long[] columnViews = new long[columns.length];
    long size = columns[0].getRowCount();

    for(int i = 0; i < columns.length; i++) {
      assert columns[i] != null : "Column vectors passed may not be null";
      assert columns[i].getRowCount() == size : "Row count mismatch, all columns must be the same size";
      assert !columns[i].getType().isDurationType() : "Unsupported column type Duration";
      columnViews[i] = columns[i].getNativeView();
    }
    return new ColumnVector(hiveHash(columnViews));
  }
  
  private static void validateColumnForSha2(ColumnView column) {
    if (column == null) {
      throw new IllegalArgumentException("SHA-2 hashing requires a non-null column");
    }
    if (!column.getType().equals(DType.STRING)) {
      throw new IllegalArgumentException("SHA-2 hashing requires a string column");
    }
  }

  /**
   * Create a new vector containing the SHA-224 hash of each row in the input column
   * Differs from cudf::hashing::sha224 in that it returns null output rows for null input rows.
   *
   * @param column the column to hash
   * @return the new ColumnVector of strings representing each row's hash value.
   */
  public static ColumnVector sha224NullsPreserved(ColumnView column) {
    validateColumnForSha2(column);
    return new ColumnVector(sha224NullsPreserved(column.getNativeView()));
  }

  /**
   * Create a new vector containing the SHA-256 hash of each row in the input column
   * Differs from cudf::hashing::sha256 in that it returns null output rows for null input rows.
   *
   * @param column the column to hash
   * @return the new ColumnVector of strings representing each row's hash value.
   */
  public static ColumnVector sha256NullsPreserved(ColumnView column) {
    validateColumnForSha2(column);
    return new ColumnVector(sha256NullsPreserved(column.getNativeView()));
  }

  /**
   * Create a new vector containing the SHA-384 hash of each row in the input column
   * Differs from cudf::hashing::sha384 in that it returns null output rows for null input rows.
   *
   * @param column the column to hash
   * @return the new ColumnVector of strings representing each row's hash value.
   */
  public static ColumnVector sha384NullsPreserved(ColumnView column) {
    validateColumnForSha2(column);
    return new ColumnVector(sha384NullsPreserved(column.getNativeView()));
  }

  /**
   * Create a new vector containing the SHA-512 hash of each row in the input column
   * Differs from cudf::hashing::sha512 in that it returns null output rows for null input rows.
   *
   * @param column the column to hash
   * @return the new ColumnVector of strings representing each row's hash value.
   */
  public static ColumnVector sha512NullsPreserved(ColumnView column) {
    validateColumnForSha2(column);
    return new ColumnVector(sha512NullsPreserved(column.getNativeView()));
  }

  /**
   * Return the CRC32 checksum of the data in the given HostMemoryBuffer, starting with the provided initial CRC value.
   * The checksum is computed on the host (CPU) using zlib's crc32 function.
   * 
   * @param crc the initial CRC value
   * @param buffer the HostMemoryBuffer containing the data to checksum
   * @return the computed CRC32 checksum
   */
  public static long hostCrc32(long crc, HostMemoryBuffer buffer) {
    return hostCrc32(crc, buffer.getAddress(), Math.toIntExact(buffer.getLength()));
  }

  private static native int getMaxStackDepth();

  private static native long murmurHash32(int seed, long[] viewHandles) throws CudfException;
  
  private static native long xxhash64(long seed, long[] viewHandles) throws CudfException;

  private static native long hiveHash(long[] viewHandles) throws CudfException;

  // Native methods for SHA-2 hashing.
  private static native long sha224NullsPreserved(long columnHandle) throws CudfException;
  private static native long sha256NullsPreserved(long columnHandle) throws CudfException;
  private static native long sha384NullsPreserved(long columnHandle) throws CudfException;
  private static native long sha512NullsPreserved(long columnHandle) throws CudfException;

  private static native long hostCrc32(long crc, long bufferHandle, int len) throws CudfException;
}
