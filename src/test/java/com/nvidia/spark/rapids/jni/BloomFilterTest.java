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

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.Scalar;

import static org.junit.jupiter.api.Assertions.assertThrows;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

public class BloomFilterTest {
  @ParameterizedTest
  @ValueSource(ints = {BloomFilter.VERSION_1, BloomFilter.VERSION_2})
  void testBuildAndProbe(int version) {
    int numHashes = 3;
    long bloomFilterBits = 4 * 1024 * 1024;

    try (ColumnVector input = ColumnVector.fromLongs(20, 80, 100, 99, 47, -9, 234000000);
         Scalar bloomFilter = BloomFilter.create(version, numHashes, bloomFilterBits, 0)) {
      BloomFilter.put(bloomFilter, input);
      try (ColumnVector probe = ColumnVector.fromLongs(20, 80, 100, 99, 47, -9, 234000000, -10, 1, 2, 3);
           ColumnVector expected = ColumnVector.fromBooleans(true, true, true, true, true, true, true, false, false, false, false);
           ColumnVector result = BloomFilter.probe(bloomFilter, probe)) {
        AssertUtils.assertColumnsAreEqual(expected, result);
      }
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {BloomFilter.VERSION_1, BloomFilter.VERSION_2})
  void testBuildAndProbeBuffer(int version) {
    int numHashes = 3;
    long bloomFilterBits = 4 * 1024 * 1024;

    try (ColumnVector input = ColumnVector.fromLongs(20, 80, 100, 99, 47, -9, 234000000);
         Scalar bloomFilter = BloomFilter.create(version, numHashes, bloomFilterBits, 0)) {
      BloomFilter.put(bloomFilter, input);
      try (ColumnVector probe = ColumnVector.fromLongs(20, 80, 100, 99, 47, -9, 234000000, -10, 1, 2, 3);
           ColumnVector expected = ColumnVector.fromBooleans(true, true, true, true, true, true, true, false, false, false, false);
           ColumnVector result = BloomFilter.probe(bloomFilter.getListAsColumnView().getData(), probe)) {
        AssertUtils.assertColumnsAreEqual(expected, result);
      }
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {BloomFilter.VERSION_1, BloomFilter.VERSION_2})
  void testBuildWithNullsAndProbe(int version) {
    int numHashes = 3;
    long bloomFilterBits = 4 * 1024 * 1024;

    try (ColumnVector input = ColumnVector.fromBoxedLongs(null, 80L, 100L, null, 47L, -9L, 234000000L);
         Scalar bloomFilter = BloomFilter.create(version, numHashes, bloomFilterBits, 0)) {
      BloomFilter.put(bloomFilter, input);
      try (ColumnVector probe = ColumnVector.fromLongs(20, 80, 100, 99, 47, -9, 234000000, -10, 1, 2, 3);
           ColumnVector expected = ColumnVector.fromBooleans(false, true, true, false, true, true, true, false, false, false, false);
           ColumnVector result = BloomFilter.probe(bloomFilter, probe)) {
        AssertUtils.assertColumnsAreEqual(expected, result);
      }
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {BloomFilter.VERSION_1, BloomFilter.VERSION_2})
  void testBuildAndProbeWithNulls(int version) {
    int numHashes = 3;
    long bloomFilterBits = 4 * 1024 * 1024;

    try (ColumnVector input = ColumnVector.fromLongs(20, 80, 100, 99, 47, -9, 234000000);
         Scalar bloomFilter = BloomFilter.create(version, numHashes, bloomFilterBits, 0)) {
      BloomFilter.put(bloomFilter, input);
      try (ColumnVector probe = ColumnVector.fromBoxedLongs(null, null, null, 99L, 47L, -9L, 234000000L, null, null, 2L, 3L);
           ColumnVector expected = ColumnVector.fromBoxedBooleans(null, null, null, true, true, true, true, null, null, false, false);
           ColumnVector result = BloomFilter.probe(bloomFilter, probe)) {
        AssertUtils.assertColumnsAreEqual(expected, result);
      }
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {BloomFilter.VERSION_1, BloomFilter.VERSION_2})
  void testBuildMergeProbe(int version) {
    int numHashes = 3;
    long bloomFilterBits = 4 * 1024 * 1024;

    try (ColumnVector colA = ColumnVector.fromLongs(20, 80, 100, 99, 47, -9, 234000000);
         ColumnVector colB = ColumnVector.fromLongs(100, 200, 300, 400);
         ColumnVector colC = ColumnVector.fromLongs(-100, -200, -300, -400);
         Scalar bloomFilterA = BloomFilter.create(version, numHashes, bloomFilterBits, 0);
         Scalar bloomFilterB = BloomFilter.create(version, numHashes, bloomFilterBits, 0);
         Scalar bloomFilterC = BloomFilter.create(version, numHashes, bloomFilterBits, 0)) {

      BloomFilter.put(bloomFilterA, colA);
      BloomFilter.put(bloomFilterB, colB);
      BloomFilter.put(bloomFilterC, colC);

      try (ColumnVector bloomA = ColumnVector.fromScalar(bloomFilterA, 1);
           ColumnVector bloomB = ColumnVector.fromScalar(bloomFilterB, 1);
           ColumnVector bloomC = ColumnVector.fromScalar(bloomFilterC, 1)) {

        try (ColumnVector premerge = ColumnVector.concatenate(bloomA, bloomB, bloomC)) {
          try (ColumnVector probe = ColumnVector.fromLongs(-9, 200, 300, 6000, -2546, 99,
                  65535, 0, -100, -200, -300, -400);
               ColumnVector expected = ColumnVector.fromBooleans(true, true, true,
                  false, false, true, false, false, true, true, true, true);
               Scalar merged = BloomFilter.merge(premerge);
               ColumnVector result = BloomFilter.probe(merged, probe)) {
            AssertUtils.assertColumnsAreEqual(expected, result);
          }
        }
      }
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {BloomFilter.VERSION_1, BloomFilter.VERSION_2})
  void testBuildTrivialMergeProbe(int version) {
    int numHashes = 3;
    long bloomFilterBits = 4 * 1024 * 1024;

    try (ColumnVector colA = ColumnVector.fromLongs(20, 80, 100, 99, 47, -9, 234000000);
         Scalar bloomFilter = BloomFilter.create(version, numHashes, bloomFilterBits, 0)) {

      BloomFilter.put(bloomFilter, colA);

      try (ColumnVector premerge = ColumnVector.fromScalar(bloomFilter, 1);
           ColumnVector probe = ColumnVector.fromLongs(-9, 200, 300, 6000, -2546, 99, 65535, 0, -100, -200, -300, -400);
           ColumnVector expected = ColumnVector.fromBooleans(true, false, false, false, false, true, false, false, false, false, false, false);
           Scalar merged = BloomFilter.merge(premerge);
           ColumnVector result = BloomFilter.probe(merged, probe)) {
        AssertUtils.assertColumnsAreEqual(expected, result);
      }
    }
  }

  @Test
  void testBuildAndProbeV2WithSeed() {
    int numHashes = 3;
    long bloomFilterBits = 4 * 1024 * 1024;
    int seed = 42;

    try (ColumnVector input = ColumnVector.fromLongs(20, 80, 100, 99, 47, -9, 234000000);
         Scalar bloomFilter = BloomFilter.create(BloomFilter.VERSION_2, numHashes, bloomFilterBits, seed)) {
      BloomFilter.put(bloomFilter, input);
      try (ColumnVector probe = ColumnVector.fromLongs(20, 80, 100, 99, 47, -9, 234000000, -10, 1, 2, 3);
           ColumnVector expected = ColumnVector.fromBooleans(true, true, true, true, true, true, true, false, false, false, false);
           ColumnVector result = BloomFilter.probe(bloomFilter, probe)) {
        AssertUtils.assertColumnsAreEqual(expected, result);
      }
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {BloomFilter.VERSION_1, BloomFilter.VERSION_2})
  void testBuildExpectedFailures(int version) {
    // bloom filter with no hashes
    assertThrows(IllegalArgumentException.class, () -> {
      try (Scalar bloomFilter = BloomFilter.create(version, 0, 64, 0)) {}
    });

    // bloom filter with no size
    assertThrows(IllegalArgumentException.class, () -> {
      try (Scalar bloomFilter = BloomFilter.create(version, 3, 0, 0)) {}
    });

    // merge with mixed hash counts
    assertThrows(CudfException.class, () -> {
      try (Scalar bloomFilterA = BloomFilter.create(version, 3, 1024, 0);
           Scalar bloomFilterB = BloomFilter.create(version, 4, 1024, 0);
           Scalar bloomFilterC = BloomFilter.create(version, 4, 1024, 0);
           ColumnVector bloomA = ColumnVector.fromScalar(bloomFilterA, 1);
           ColumnVector bloomB = ColumnVector.fromScalar(bloomFilterB, 1);
           ColumnVector bloomC = ColumnVector.fromScalar(bloomFilterC, 1);
           ColumnVector premerge = ColumnVector.concatenate(bloomA, bloomB, bloomC);
           Scalar merged = BloomFilter.merge(premerge)) {}
    });

    // merge with mixed bit sizes
    assertThrows(CudfException.class, () -> {
      try (Scalar bloomFilterA = BloomFilter.create(version, 3, 1024, 0);
           Scalar bloomFilterB = BloomFilter.create(version, 3, 1024, 0);
           Scalar bloomFilterC = BloomFilter.create(version, 3, 2048, 0);
           ColumnVector bloomA = ColumnVector.fromScalar(bloomFilterA, 1);
           ColumnVector bloomB = ColumnVector.fromScalar(bloomFilterB, 1);
           ColumnVector bloomC = ColumnVector.fromScalar(bloomFilterC, 1);
           ColumnVector premerge = ColumnVector.concatenate(bloomA, bloomB, bloomC);
           Scalar merged = BloomFilter.merge(premerge)) {}
    });
  }

  @Test
  void testBuildExpectedFailuresVersionIndependent() {
    // invalid version
    assertThrows(IllegalArgumentException.class, () -> {
      try (Scalar bloomFilter = BloomFilter.create(3, 3, 64, 0)) {}
    });

    // merge with mixed versions (V1 + V2)
    assertThrows(CudfException.class, () -> {
      try (Scalar bloomFilterA = BloomFilter.create(BloomFilter.VERSION_1, 3, 1024, 0);
           Scalar bloomFilterB = BloomFilter.create(BloomFilter.VERSION_2, 3, 1024, 0);
           ColumnVector bloomA = ColumnVector.fromScalar(bloomFilterA, 1);
           ColumnVector bloomB = ColumnVector.fromScalar(bloomFilterB, 1);
           ColumnVector premerge = ColumnVector.concatenate(bloomA, bloomB);
           Scalar merged = BloomFilter.merge(premerge)) {}
    });
  }
}
