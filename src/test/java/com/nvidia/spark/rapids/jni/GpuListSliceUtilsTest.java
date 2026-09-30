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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector.*;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;

import static ai.rapids.cudf.AssertUtils.*;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class GpuListSliceUtilsTest {
  @Test
  void testListSliceStartIntLengthInt() {
    try (ColumnVector intListCV = ColumnVector.fromLists(
        new ListType(true, new BasicType(true, DType.INT32)),
        Arrays.asList(1, 2, 3), // Normal case
        Arrays.asList(4, null, 5), // Contains null
        null, // Entire array is null
        Collections.emptyList(), // Empty list
        Collections.singletonList(null), // Single null
        Arrays.asList(null, null, null), // All nulls
        Arrays.asList(6, 7));
        ColumnVector result1 = GpuListSliceUtils.listSlice(intListCV, 1, 2);
        ColumnVector expected1 = ColumnVector.fromLists(
            new ListType(true, new BasicType(true, DType.INT32)),
            Arrays.asList(1, 2),
            Arrays.asList(4, null),
            null,
            Collections.emptyList(),
            Collections.singletonList(null),
            Arrays.asList(null, null),
            Arrays.asList(6, 7));
        ColumnVector result2 = GpuListSliceUtils.listSlice(intListCV, 1, 0);
        ColumnVector expected2 = ColumnVector.fromLists(
            new ListType(true, new BasicType(true, DType.INT32)),
            Collections.emptyList(),
            Collections.emptyList(),
            null,
            Collections.emptyList(),
            Collections.emptyList(),
            Collections.emptyList(),
            Collections.emptyList());
        ColumnVector result3 = GpuListSliceUtils.listSlice(intListCV, 3, 10);
        ColumnVector expected3 = ColumnVector.fromLists(
            new ListType(true, new BasicType(true, DType.INT32)),
            Collections.singletonList(3),
            Collections.singletonList(5),
            null,
            Collections.emptyList(),
            Collections.emptyList(),
            Collections.singletonList(null),
            Collections.emptyList());
        ColumnVector result4 = GpuListSliceUtils.listSlice(intListCV, -2, 2);
        ColumnVector expected4 = ColumnVector.fromLists(
            new ListType(true, new BasicType(true, DType.INT32)),
            Arrays.asList(2, 3),
            Arrays.asList(null, 5),
            null,
            Collections.emptyList(),
            Collections.emptyList(),
            Arrays.asList(null, null),
            Arrays.asList(6, 7))) {
      assertColumnsAreEqual(expected1, result1);
      assertColumnsAreEqual(expected2, result2);
      assertColumnsAreEqual(expected3, result3);
      assertColumnsAreEqual(expected4, result4);
    }
  }

  @Test
  void testListSliceStartIntLengthCol() {
    try (ColumnVector intListCV = ColumnVector.fromLists(
        new ListType(true, new BasicType(true, DType.INT32)),
        Arrays.asList(1, 2, 3), // Normal case
        Arrays.asList(4, null, 5), // Contains null
        null, // Entire array is null
        Collections.emptyList(), // Empty list
        Collections.singletonList(null), // Single null
        Arrays.asList(null, null, null), // All nulls
        Arrays.asList(6, 7));
        ColumnVector lengthCV = ColumnVector.fromBoxedInts(0, 1, 2, 3, 4, 10, null);
        ColumnVector result = GpuListSliceUtils.listSlice(intListCV, 1, lengthCV);
        ColumnVector expected = ColumnVector.fromLists(
            new ListType(true, new BasicType(true, DType.INT32)),
            Collections.emptyList(),
            Collections.singletonList(4),
            null,
            Collections.emptyList(),
            Collections.singletonList(null),
            Arrays.asList(null, null, null),
            null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testListSliceStartColLengthInt() {
    try (ColumnVector intListCV = ColumnVector.fromLists(
        new ListType(true, new BasicType(true, DType.INT32)),
        Arrays.asList(1, 2, 3), // Normal case
        Arrays.asList(4, null, 5), // Contains null
        null, // Entire array is null
        Collections.emptyList(), // Empty list
        Collections.singletonList(null), // Single null
        Arrays.asList(null, null, null), // All nulls
        Arrays.asList(6, 7));
        ColumnVector startCV = ColumnVector.fromBoxedInts(1, -1, 2, 3, -10, 10, null);
        ColumnVector result = GpuListSliceUtils.listSlice(intListCV, startCV, 2);
        ColumnVector expected = ColumnVector.fromLists(
            new ListType(true, new BasicType(true, DType.INT32)),
            Arrays.asList(1, 2),
            Collections.singletonList(5),
            null,
            Collections.emptyList(),
            Collections.emptyList(),
            Collections.emptyList(),
            null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testListSliceStartColLengthCol() {
    try (ColumnVector intListCV = ColumnVector.fromLists(
        new ListType(true, new BasicType(true, DType.INT32)),
        Arrays.asList(1, 2, 3), // Normal case
        Arrays.asList(4, null, 5), // Contains null
        null, // Entire array is null
        Collections.emptyList(), // Empty list
        Collections.singletonList(null), // Single null
        Arrays.asList(null, null, null), // All nulls
        Arrays.asList(6, 7));
        ColumnVector startCV = ColumnVector.fromBoxedInts(1, -1, 2, 3, -10, -2, null);
        ColumnVector lengthCV = ColumnVector.fromBoxedInts(0, 1, 2, null, 4, 10, 1);
        ColumnVector result = GpuListSliceUtils.listSlice(intListCV, startCV, lengthCV);
        ColumnVector expected = ColumnVector.fromLists(
            new ListType(true, new BasicType(true, DType.INT32)),
            Collections.emptyList(),
            Collections.singletonList(5),
            null,
            null,
            Collections.emptyList(),
            Arrays.asList(null, null),
            null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testListSliceIllegalArguments() {
    try (ColumnVector intListCV = ColumnVector.fromLists(
        new ListType(true, new BasicType(true, DType.INT32)),
        Arrays.asList(1, 2, 3),
        Arrays.asList(4, 5));
        ColumnVector longStartCV = ColumnVector.fromBoxedLongs(1L, 1L);
        ColumnVector longLengthCV = ColumnVector.fromBoxedLongs(1L, 1L);
        ColumnVector legalStartCV = ColumnVector.fromBoxedInts(1, 1);
        ColumnVector legalLengthCV = ColumnVector.fromBoxedInts(1, 1);
        ColumnVector startContainsZeroCV = ColumnVector.fromBoxedInts(0, 2);
        ColumnVector lengthContainsNegCV = ColumnVector.fromBoxedInts(1, -1);
        ColumnVector startMismatchCV = ColumnVector.fromBoxedInts(1);
        ColumnVector lengthMismatchCV = ColumnVector.fromBoxedInts(1, 1, 1)) {
      // start can not be 0
      assertThrows(CudfException.class, () -> GpuListSliceUtils.listSlice(intListCV, 0, 1));
      assertThrows(CudfException.class, () -> GpuListSliceUtils.listSlice(intListCV, 0, legalLengthCV));
      assertThrows(CudfException.class, () -> GpuListSliceUtils.listSlice(intListCV, startContainsZeroCV, 1));
      assertThrows(CudfException.class, () -> GpuListSliceUtils.listSlice(intListCV, startContainsZeroCV, legalLengthCV));
      // length can not be negative
      assertThrows(CudfException.class, () -> GpuListSliceUtils.listSlice(intListCV, 1, -1));
      assertThrows(CudfException.class, () -> GpuListSliceUtils.listSlice(intListCV, 1, lengthContainsNegCV));
      assertThrows(CudfException.class, () -> GpuListSliceUtils.listSlice(intListCV, legalStartCV, -1));
      assertThrows(CudfException.class, () -> GpuListSliceUtils.listSlice(intListCV, legalStartCV, lengthContainsNegCV));
      // mismatched size of start or length
      assertThrows(CudfException.class, () -> GpuListSliceUtils.listSlice(intListCV, startMismatchCV, 1));
      assertThrows(CudfException.class, () -> GpuListSliceUtils.listSlice(intListCV, startMismatchCV, legalLengthCV));
      assertThrows(CudfException.class, () -> GpuListSliceUtils.listSlice(intListCV, 1, lengthMismatchCV));
      assertThrows(CudfException.class, () -> GpuListSliceUtils.listSlice(intListCV, legalStartCV, lengthMismatchCV));
      // start or length column is not of INT32 type
      assertThrows(CudfException.class, () -> GpuListSliceUtils.listSlice(intListCV, longStartCV, 1));
      assertThrows(CudfException.class, () -> GpuListSliceUtils.listSlice(intListCV, 1, longLengthCV));
      assertThrows(CudfException.class, () -> GpuListSliceUtils.listSlice(intListCV, longStartCV, longLengthCV));
    }
  }
}
