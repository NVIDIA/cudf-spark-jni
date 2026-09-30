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

import ai.rapids.cudf.BaseDeviceMemoryBuffer;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.Scalar;
import ai.rapids.cudf.NativeDepsLoader;

public class BloomFilter {
  public static final int VERSION_1 = 1;
  public static final int VERSION_2 = 2;
  public static final int DEFAULT_SEED = 0;

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Create a V1 bloom filter with the specified number of hashes and bloom filter bits,
   * using the default seed.
   *
   * This overload is retained temporarily. Use {@link #create(int, int, long, int)} instead;
   * remove this overload after all callers migrate.
   * @param numHashes Number of bit positions set (and checked) per key.
   * @param bloomFilterBits Total size of the bloom filter in bits (will be rounded up to a
   *        multiple of 64).
   * @return A Scalar wrapping the GPU bloom filter (Spark format).
   */
  public static Scalar create(int numHashes, long bloomFilterBits) {
    return create(VERSION_1, numHashes, bloomFilterBits, DEFAULT_SEED);
  }

  /**
   * Create a bloom filter with the specified version, number of hashes, bloom filter bits,
   * and hash seed.
   *
   * @param version Bloom filter format: {@link #VERSION_1} or {@link #VERSION_2}. V2 uses
   *        64-bit hash indexing and supports a configurable seed.
   * @param numHashes Number of bit positions set (and checked) per key. Higher values reduce
   *        false positives but increase work per put/probe.
   * @param bloomFilterBits Total size of the bloom filter in bits (will be rounded up to a
   *        multiple of 64).
   * @param seed Hash seed. Used only for V2; ignored for V1.
   * @return A Scalar wrapping the GPU bloom filter (Spark format).
   */
  public static Scalar create(int version, int numHashes, long bloomFilterBits, int seed) {
    if (version != VERSION_1 && version != VERSION_2) {
      throw new IllegalArgumentException("Bloom filter version must be 1 or 2");
    }
    if (numHashes <= 0) {
      throw new IllegalArgumentException("Bloom filters must have a positive hash count");
    }
    if (bloomFilterBits <= 0) {
      throw new IllegalArgumentException("Bloom filters must have a positive number of bits");
    }
    return new Scalar(DType.LIST, creategpu(version, numHashes, bloomFilterBits, seed));
  }

  public static void put(Scalar bloomFilter, ColumnVector cv) {
    put(bloomFilter.getScalarHandle(), cv.getNativeView());
  }

  public static Scalar merge(ColumnVector bloomFilters) {
    return new Scalar(DType.LIST, merge(bloomFilters.getNativeView()));
  }

  /**
   * Probe a bloom filter with a column of longs. For each row, true means the value may be
   * in the set used to build the filter; false means it is definitely not in the set.
   *
   * @param bloomFilter The bloom filter to probe (a Scalar wrapping the GPU filter).
   * @param cv Column of int64 values to check for membership.
   * @return A boolean column with the same row count as cv; true for possible membership,
   *         false for definite non-membership. Nulls in cv are preserved in the output.
   */
  public static ColumnVector probe(Scalar bloomFilter, ColumnVector cv) {
    return new ColumnVector(probe(bloomFilter.getScalarHandle(), cv.getNativeView()));
  }

  /**
   * Probe a bloom filter with a column of longs. For each row, true means the value may be
   * in the set used to build the filter; false means it is definitely not in the set.
   * Use this overload when the filter is in a device buffer (e.g. Spark serialized form).
   *
   * @param bloomFilter Device buffer containing the packed bloom filter including header.
   * @param cv Column of int64 values to check for membership.
   * @return A boolean column with the same row count as cv; true for possible membership,
   *         false for definite non-membership. Nulls in cv are preserved in the output.
   */
  public static ColumnVector probe(BaseDeviceMemoryBuffer bloomFilter, ColumnVector cv) {
    return new ColumnVector(
        probebuffer(bloomFilter.getAddress(), bloomFilter.getLength(), cv.getNativeView()));
  }

  private static native long creategpu(int version, int numHashes, long bloomFilterBits, int seed)
      throws CudfException;
  private static native int put(long bloomFilter, long cv) throws CudfException;
  private static native long merge(long bloomFilters) throws CudfException;
  private static native long probe(long bloomFilter, long cv) throws CudfException;
  private static native long probebuffer(long bloomFilter, long bloomFilterSize, long cv)
      throws CudfException;
}
