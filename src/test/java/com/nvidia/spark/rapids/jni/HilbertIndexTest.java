/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
import org.davidmoten.hilbert.HilbertCurve;
import org.davidmoten.hilbert.SmallHilbertCurve;
import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class HilbertIndexTest {
  static long[] getExpected(int numBits, int numRows, Integer[]... inputs) {
    final int dimensions = inputs.length;
    final int length = numBits * dimensions;
    assert(length <= 64);
    SmallHilbertCurve shc = HilbertCurve.small().bits(numBits).dimensions(dimensions);
    long[] ret = new long[numRows];
    long[] tmpInputs = new long[dimensions];
    for (int rowIndex = 0; rowIndex < numRows; rowIndex++) {
      for (int colIndex = 0; colIndex < dimensions; colIndex++) {
        Integer i = inputs[colIndex][rowIndex];
        if (i == null) {
          i = 0;
        }
        tmpInputs[colIndex] = i;
      }
      if (tmpInputs.length == 0) {
        ret[rowIndex] = 0;
      } else {
        ret[rowIndex] = shc.index(tmpInputs);
      }
    }
    return ret;
  }

  public static void doTest(int numBits, int numRows, Integer[]... inputs) {
    long[] expected = getExpected(numBits, numRows, inputs);
    ColumnVector[] cvInputs = new ColumnVector[inputs.length];
    try {
      for (int columnIndex = 0; columnIndex < inputs.length; columnIndex++) {
        cvInputs[columnIndex] = ColumnVector.fromBoxedInts(inputs[columnIndex]);
      }
      try (ColumnVector results = ZOrder.hilbertIndex(numBits, numRows, cvInputs);
           ColumnVector expectedCv = ColumnVector.fromLongs(expected)) {
        assertColumnsAreEqual(expectedCv, results);
      }
    } finally {
      for (ColumnVector cv: cvInputs) {
        if (cv != null) {
          cv.close();
        }
      }
    }
  }

  @Test
  void test0() {
    doTest(6, 10);
  }

  @Test
  void test1NonNull() {
    Integer[] inputs = {1, 2, 3, 4, 5};
    doTest(3, inputs.length, inputs);
  }

  @Test
  void test1Null() {
    Integer[] inputs = {null, 7, null, 8};
    doTest(4, inputs.length, inputs);
  }

  @Test
  void testInt2NonNull() {
    Integer[] inputs1 = {  1, 500, 1000, 250};
    Integer[] inputs2 = {500, 400,  300, 200};
    doTest(10, inputs1.length, inputs1, inputs2);
  }

  @Test
  void testInt2Null() {
    Integer[] inputs1 = {  0, null,  50, 1000};
    Integer[] inputs2 = {200,  300, 100,    0};
    doTest(10, inputs1.length, inputs1, inputs2);
  }

  @Test
  void testInt3NonNull() {
    Integer[] inputs1 = {0, 4, 1, 0, 1023, 512};
    Integer[] inputs2 = {1, 8, 2, 0, 1023, 512};
    Integer[] inputs3 = {2, 0, 4, 0, 1023, 512};
    doTest(10, inputs1.length, inputs1, inputs2, inputs3);
  }
}