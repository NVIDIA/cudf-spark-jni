/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ai.rapids.cudf.CudfColumnSizeOverflowException;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DType;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertThrowsExactly;

class RangeSequenceTest {
  private static void assertSequence(long[] starts, int[] sizes) {
    int rows = Arrays.stream(sizes).sum();
    long[] values = new long[rows];
    int outputIndex = 0;
    for (int range = 0; range < starts.length; range++) {
      for (int i = 0; i < sizes[range]; i++) {
        values[outputIndex++] = starts[range] + i;
      }
    }
    try (ColumnVector actual = RangeSequence.generate(starts, sizes);
         ColumnVector expected = ColumnVector.fromLongs(values)) {
      assertEquals(DType.INT64, actual.getType());
      assertEquals(0, actual.getNullCount());
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void emptyAndZeroSizeRanges() {
    assertSequence(new long[0], new int[0]);
    assertSequence(new long[]{Long.MAX_VALUE, Long.MIN_VALUE}, new int[]{0, 0});
    assertSequence(new long[]{0, 100, 0, 0, 200, 0}, new int[]{0, 2, 0, 0, 3, 0});
  }

  @Test
  void gapsOverlapsAndDecreasingStarts() {
    assertSequence(new long[]{100, 20, 500, 21}, new int[]{3, 2, 1, 3});
  }

  @Test
  void singleRangeAndIntegerBoundaries() {
    assertSequence(new long[]{-1000}, new int[]{4097});
    assertSequence(new long[]{Long.MAX_VALUE}, new int[]{1});
    assertSequence(new long[]{Long.MIN_VALUE, -1, Long.MAX_VALUE - 2}, new int[]{3, 3, 3});
  }

  @Test
  void manyRangesWithEmptyRanges() {
    long[] starts = new long[4097];
    int[] sizes = new int[starts.length];
    for (int i = 0; i < starts.length; i++) {
      starts[i] = 10L * i;
      sizes[i] = i % 4;
    }
    assertSequence(starts, sizes);
  }

  @Test
  void randomizedFragmentedRanges() {
    long seed = System.currentTimeMillis();
    System.out.println("RangeSequenceTest.randomizedFragmentedRanges seed: " + seed);
    Random random = new Random(seed);
    for (int iteration = 0; iteration < 8; iteration++) {
      int ranges = 1 + random.nextInt(2048);
      long[] starts = random.longs(ranges, 0, Long.MAX_VALUE - 2048).toArray();
      int[] sizes = new int[ranges];
      for (int i = 0; i < ranges; i++) {
        sizes[i] = random.nextInt(2048);
      }
      assertSequence(starts, sizes);
    }
  }

  @Test
  void invalidInputs() {
    assertThrows(NullPointerException.class, () -> RangeSequence.generate(null, new int[0]));
    assertThrows(NullPointerException.class, () -> RangeSequence.generate(new long[0], null));
    assertThrowsExactly(CudfException.class,
        () -> RangeSequence.generate(new long[]{0}, new int[0]));
    assertThrowsExactly(CudfException.class,
        () -> RangeSequence.generate(new long[]{0, 1}, new int[]{0, -1}));
    assertThrowsExactly(CudfColumnSizeOverflowException.class,
        () -> RangeSequence.generate(new long[]{0, 0}, new int[]{Integer.MAX_VALUE, 1}));
    assertThrowsExactly(CudfException.class,
        () -> RangeSequence.generate(new long[]{Long.MAX_VALUE}, new int[]{2}));
    assertThrowsExactly(CudfException.class,
        () -> RangeSequence.generate(new long[]{0, Long.MAX_VALUE}, new int[]{0, 2}));
  }
}
