/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

import ai.rapids.cudf.*;

import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class CaseWhenTest {

  @Test
  void selectIndexTest() {
    try (
        ColumnVector b0 = ColumnVector.fromBooleans(
            true, false, false, false);
        ColumnVector b1 = ColumnVector.fromBooleans(
            true, true, false, false);
        ColumnVector b2 = ColumnVector.fromBooleans(
            false, false, true, false);
        ColumnVector b3 = ColumnVector.fromBooleans(
            true, true, true, false);
        ColumnVector expected = ColumnVector.fromInts(0, 1, 2, 4)) {
      ColumnVector[] boolColumns = new ColumnVector[] { b0, b1, b2, b3 };
      try (ColumnVector actual = CaseWhen.selectFirstTrueIndex(boolColumns)) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }

  @Test
  void selectIndexTestWithNull() {
    try (
        ColumnVector b0 = ColumnVector.fromBoxedBooleans(
            null, false, false, null, false);
        ColumnVector b1 = ColumnVector.fromBoxedBooleans(
            null, null, false, true, true);
        ColumnVector b2 = ColumnVector.fromBoxedBooleans(
            null, null, false, true, false);
        ColumnVector b3 = ColumnVector.fromBoxedBooleans(
            null, null, null, true, null);
        ColumnVector expected = ColumnVector.fromInts(4, 4, 4, 1, 1)) {
      ColumnVector[] boolColumns = new ColumnVector[] { b0, b1, b2, b3 };
      try (ColumnVector actual = CaseWhen.selectFirstTrueIndex(boolColumns)) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }
}
