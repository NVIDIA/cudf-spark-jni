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

import ai.rapids.cudf.*;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

import java.util.Arrays;
import java.util.List;
import java.util.HashSet;
import java.util.UUID;

public class StringUtilsTest {

  @Test
  void testUuidInvalidInput() {
    // row count must be positive
    assertThrows(CudfException.class, () -> {
      StringUtils.randomUUIDs(0);
    });

    // row count must be positive
    assertThrows(CudfException.class, () -> {
      StringUtils.randomUUIDs(-1);
    });

    // throws ColumnsizeOverflowException
    // 36 is the number of chars in a UUID
    int numRows = (Integer.MAX_VALUE / 36) + 1;
    assertThrows(CudfColumnSizeOverflowException.class, () -> {
      StringUtils.randomUUIDs(numRows);
    });
  }

  /**
   * @brief Run two rounds with the same row count, MUST generate different UUIDs.
   */
  @Test
  void testUuid() {
    List<Integer> rowCounts = Arrays.asList(1, 2, 3, 5, 17, 33, 65, 129, 515, 1025);
    for (Integer rowCount : rowCounts) {
      try (
          ColumnVector round1 = StringUtils.randomUUIDs(rowCount);
          ColumnVector round2 = StringUtils.randomUUIDs(rowCount);
          HostColumnVector h1 = round1.copyToHost();
          HostColumnVector h2 = round2.copyToHost()) {
        HashSet<String> set = new HashSet<>();
        for (int i = 0; i < rowCount; i++) {
          String uuidStr1 = h1.getJavaString(i);
          String uuidStr2 = h2.getJavaString(i);
          UUID uuid1 = UUID.fromString(uuidStr1);
          UUID uuid2 = UUID.fromString(uuidStr2);
          assertEquals(uuid1.version(), 4);
          assertEquals(uuid2.version(), 4);
          assertEquals(uuid1.variant(), 2);
          assertEquals(uuid2.variant(), 2);
          set.add(uuidStr1);
          set.add(uuidStr2);
        }
        // The UUIDs in two rounds must be different.
        assertEquals(set.size(), rowCount * 2);
      }
    }
  }

  @Test
  void testUuidSameSeed() {
    int rowCount = 128;
    long seed = 1123L;
    try (
        ColumnVector round1 = StringUtils.randomUUIDsWithSeed(rowCount, seed);
        ColumnVector round2 = StringUtils.randomUUIDsWithSeed(rowCount, seed)) {
      // Same seed should generate the same UUIDs.
      assertColumnsAreEqual(round1, round2);
    }
  }

  @Test
  void testFindInSet() {
    try (
        ColumnVector sets = ColumnVector.fromStrings(
            "a,b,c",
            "b,a,b",
            "x,y",
            "",
            ",",
            "a,,b",
            null,
            "\u00e9,b",
            "a,\u00e9",
            "aa,a",
            "a,aa");
        ColumnVector expectedB = ColumnVector.fromBoxedInts(
            2, 1, 0, 0, 0, 3, null, 2, 0, 0, 0);
        ColumnVector expectedEmpty = ColumnVector.fromBoxedInts(
            0, 0, 0, 1, 1, 2, null, 0, 0, 0, 0);
        ColumnVector expectedAccent = ColumnVector.fromBoxedInts(
            0, 0, 0, 0, 0, 0, null, 1, 2, 0, 0);
        ColumnVector expectedComma = ColumnVector.fromBoxedInts(
            0, 0, 0, 0, 0, 0, null, 0, 0, 0, 0);
        ColumnVector actualB = StringUtils.findInSet(sets, "b");
        ColumnVector actualEmpty = StringUtils.findInSet(sets, "");
        ColumnVector actualAccent = StringUtils.findInSet(sets, "\u00e9");
        ColumnVector actualComma = StringUtils.findInSet(sets, "a,b")) {
      assertColumnsAreEqual(expectedB, actualB);
      assertColumnsAreEqual(expectedEmpty, actualEmpty);
      assertColumnsAreEqual(expectedAccent, actualAccent);
      assertColumnsAreEqual(expectedComma, actualComma);
    }
  }

  @Test
  void testFindInSetRepeated() {
    try (
        ColumnVector sets = ColumnVector.fromStrings(
            "a,b,c",
            "x,b",
            "a,b,c",
            null,
            "b",
            "",
            ",");
        ColumnVector expectedB = ColumnVector.fromBoxedInts(
            2, 2, 2, null, 1, 0, 0);
        ColumnVector expectedEmpty = ColumnVector.fromBoxedInts(
            0, 0, 0, null, 0, 1, 1);
        ColumnVector expectedComma = ColumnVector.fromBoxedInts(
            0, 0, 0, null, 0, 0, 0);
        ColumnVector actualB = StringUtils.findInSetRepeated(sets, "b", 5);
        ColumnVector actualEmpty = StringUtils.findInSetRepeated(sets, "", 5);
        ColumnVector actualComma = StringUtils.findInSetRepeated(sets, "a,b", 5)) {
      assertColumnsAreEqual(expectedB, actualB);
      assertColumnsAreEqual(expectedEmpty, actualEmpty);
      assertColumnsAreEqual(expectedComma, actualComma);
      try (ColumnVector tooManyDistinct = StringUtils.findInSetRepeated(sets, "b", 4)) {
        assertNull(tooManyDistinct);
      }
    }
  }
}
