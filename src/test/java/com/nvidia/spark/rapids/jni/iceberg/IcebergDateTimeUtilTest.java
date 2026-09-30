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

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

import java.util.Random;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.apache.iceberg.util.DateTimeUtil;
import ai.rapids.cudf.ColumnVector;

public class IcebergDateTimeUtilTest {

  private static long seed;

  @BeforeAll
  public static void init() {
    seed = System.currentTimeMillis();
    System.out.println("IcebergDateTimeUtilTest seed: " + seed);
  }

  @Test
  void toYearsTest() {
    // basic test
    try (
        ColumnVector input = ColumnVector.timestampDaysFromBoxedInts(
            -50,
            -500,
            null,
            50,
            500,
            null);
        ColumnVector expected = ColumnVector.fromBoxedInts(
            -1,
            -2,
            null,
            0,
            1,
            null);
        ColumnVector result = IcebergDateTimeUtil.yearsFromEpoch(input)) {
      assertColumnsAreEqual(expected, result);
    }

    Random random = new Random(seed);
    int numRows = 1024;
    int[] days = new int[numRows];
    long[] micros = new long[numRows];
    for (int i = 0; i < numRows; ++i) {
      days[i] = random.nextInt();
      micros[i] = random.nextLong();
    }
    int[] expectedValues1 = new int[numRows];
    int[] expectedValues2 = new int[numRows];
    for (int i = 0; i < numRows; ++i) {
      // run on CPU
      expectedValues1[i] = DateTimeUtil.daysToYears(days[i]);
      expectedValues2[i] = DateTimeUtil.microsToYears(micros[i]);
    }
    try (
        ColumnVector input1 = ColumnVector.daysFromInts(days);
        ColumnVector expected1 = ColumnVector.fromInts(expectedValues1);
        ColumnVector result1 = IcebergDateTimeUtil.yearsFromEpoch(input1);
        ColumnVector input2 = ColumnVector.timestampMicroSecondsFromLongs(micros);
        ColumnVector expected2 = ColumnVector.fromInts(expectedValues2);
        ColumnVector result2 = IcebergDateTimeUtil.yearsFromEpoch(input2)) {
      assertColumnsAreEqual(expected1, result1);
      assertColumnsAreEqual(expected2, result2);
    }
  }

  @Test
  void toMonthsTest() {
    // basic test
    try (
        ColumnVector input = ColumnVector.timestampDaysFromBoxedInts(
            -50,
            -500,
            null,
            50,
            500,
            null);
        ColumnVector expected = ColumnVector.fromBoxedInts(
            -2,
            -17,
            null,
            1,
            16,
            null);
        ColumnVector result = IcebergDateTimeUtil.monthsFromEpoch(input)) {
      assertColumnsAreEqual(expected, result);
    }

    Random random = new Random(seed);
    int numRows = 1024;
    int[] days = new int[numRows];
    long[] micros = new long[numRows];
    for (int i = 0; i < numRows; ++i) {
      days[i] = random.nextInt();
      micros[i] = random.nextLong();
    }
    int[] expectedValues1 = new int[numRows];
    int[] expectedValues2 = new int[numRows];
    for (int i = 0; i < numRows; ++i) {
      // run on CPU
      expectedValues1[i] = DateTimeUtil.daysToMonths(days[i]);
      expectedValues2[i] = DateTimeUtil.microsToMonths(micros[i]);
    }
    try (
        ColumnVector input1 = ColumnVector.daysFromInts(days);
        ColumnVector expected1 = ColumnVector.fromInts(expectedValues1);
        ColumnVector result1 = IcebergDateTimeUtil.monthsFromEpoch(input1);
        ColumnVector input2 = ColumnVector.timestampMicroSecondsFromLongs(micros);
        ColumnVector expected2 = ColumnVector.fromInts(expectedValues2);
        ColumnVector result2 = IcebergDateTimeUtil.monthsFromEpoch(input2)) {
      assertColumnsAreEqual(expected1, result1);
      assertColumnsAreEqual(expected2, result2);
    }
  }

  @Test
  void toDaysTest() {
    Random random = new Random(seed);
    int numRows = 1024;
    int[] days = new int[numRows];
    long[] micros = new long[numRows];
    for (int i = 0; i < numRows; ++i) {
      days[i] = random.nextInt();
      micros[i] = random.nextLong();
    }
    int[] expectedValues1 = new int[numRows];
    int[] expectedValues2 = new int[numRows];
    for (int i = 0; i < numRows; ++i) {
      // run on CPU
      expectedValues1[i] = days[i]; // do nothing, days to days
      expectedValues2[i] = DateTimeUtil.microsToDays(micros[i]);
    }
    try (
        ColumnVector input1 = ColumnVector.daysFromInts(days);
        ColumnVector expected1 = ColumnVector.daysFromInts(expectedValues1);
        ColumnVector result1 = IcebergDateTimeUtil.daysFromEpoch(input1);
        ColumnVector input2 = ColumnVector.timestampMicroSecondsFromLongs(micros);
        ColumnVector expected2 = ColumnVector.daysFromInts(expectedValues2);
        ColumnVector result2 = IcebergDateTimeUtil.daysFromEpoch(input2)) {
      assertColumnsAreEqual(expected1, result1);
      assertColumnsAreEqual(expected2, result2);
    }
  }

  @Test
  void toHoursTest() {
    // test overflow
    try (
        ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(
            Long.MAX_VALUE, // CPU will overflow without error, GPU does the same
            Long.MIN_VALUE // CPU will overflow without error, GPU does the same
        );
        ColumnVector expected = ColumnVector.fromBoxedInts(
            -1732919508, // overflow, Long.MAX gets negative value.
            1732919507 // overflow, Long.MIN gets positive value.
        );
        ColumnVector result = IcebergDateTimeUtil.hoursFromEpoch(input)) {
      assertColumnsAreEqual(expected, result);
    }

    // random test
    Random random = new Random(seed);
    int numRows = 1024;
    long[] micros = new long[numRows];
    for (int i = 0; i < numRows; ++i) {
      micros[i] = random.nextLong();
    }
    int[] expectedValues = new int[numRows];
    for (int i = 0; i < numRows; ++i) {
      // run on CPU
      expectedValues[i] = DateTimeUtil.microsToHours(micros[i]);
    }
    try (
        ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(micros);
        ColumnVector expected = ColumnVector.fromInts(expectedValues);
        ColumnVector result = IcebergDateTimeUtil.hoursFromEpoch(input)) {
      assertColumnsAreEqual(expected, result);
    }
  }
}
