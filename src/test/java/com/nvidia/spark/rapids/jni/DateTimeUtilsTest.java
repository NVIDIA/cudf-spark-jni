/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

import org.junit.jupiter.api.Test;

import ai.rapids.cudf.ColumnVector;

public class DateTimeUtilsTest {
  @Test
  void rebaseDaysToJulianTest() {
    try (
        ColumnVector input = ColumnVector.timestampDaysFromBoxedInts(-719162, -354285, null,
            -141714, -141438, -141437,
            null, null,
            -141432, -141427, -31463, -31453, -1, 0, 18335);
        ColumnVector expected = ColumnVector.timestampDaysFromBoxedInts(-719164, -354280, null,
            -141704, -141428, -141427,
            null, null,
            -141427, -141427, -31463, -31453, -1, 0, 18335);
        ColumnVector result = DateTimeUtils.rebaseGregorianToJulian(input)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void rebaseDaysToGregorianTest() {
    try (
        ColumnVector input = ColumnVector.timestampDaysFromBoxedInts(-719164, -354280, null,
            -141704, -141428, -141427,
            null, null,
            -141427, -141427, -31463, -31453, -1, 0, 18335);
        ColumnVector expected = ColumnVector.timestampDaysFromBoxedInts(-719162, -354285, null,
            -141714, -141438, -141427,
            null, null,
            -141427, -141427, -31463, -31453, -1, 0, 18335);
        ColumnVector result = DateTimeUtils.rebaseJulianToGregorian(input)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void rebaseMicroToJulian() {
    try (
        ColumnVector input = ColumnVector.timestampMicroSecondsFromBoxedLongs(-62135593076345679L,
            -30610213078876544L,
            null,
            -12244061221876544L,
            -12220243200000000L,
            -12219639001448163L,
            -12219292799000001L,
            -45446999900L,
            1L,
            null,
            1584178381500000L);
        ColumnVector expected =
            ColumnVector.timestampMicroSecondsFromBoxedLongs(-62135765876345679L,
                -30609781078876544L,
                null,
                -12243197221876544L,
                -12219379200000000L,
                -12219207001448163L,
                -12219292799000001L,
                -45446999900L,
                1L,
                null,
                1584178381500000L);
        ColumnVector result = DateTimeUtils.rebaseGregorianToJulian(input)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void rebaseMicroToGregorian() {
    try (
        ColumnVector input = ColumnVector.timestampMicroSecondsFromBoxedLongs(-62135765876345679L,
            -30609781078876544L,
            null,
            -12243197221876544L,
            -12219379200000000L,
            -12219207001448163L,
            -12219292799000001L,
            -45446999900L,
            1L,
            null,
            1584178381500000L);
        ColumnVector expected =
            ColumnVector.timestampMicroSecondsFromBoxedLongs(-62135593076345679L,
                -30610213078876544L,
                null,
                -12244061221876544L,
                -12220243200000000L,
                -12219207001448163L,
                -12219292799000001L,
                -45446999900L,
                1L,
                null,
                1584178381500000L);
        ColumnVector result = DateTimeUtils.rebaseJulianToGregorian(input)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void truncateDateTest() {
    try (ColumnVector input = ColumnVector.timestampDaysFromBoxedInts(-31463, -31453, null, 0, 18335);
         ColumnVector format = ColumnVector.fromStrings("YEAR", "MONTH", "WEEK", "QUARTER", "YY");
         ColumnVector expected = ColumnVector.timestampDaysFromBoxedInts(-31776, -31472, null, 0, 18262);
         ColumnVector result = DateTimeUtils.truncate(input, format)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void truncateTimestampTest() {
    try (
        ColumnVector input = ColumnVector.timestampMicroSecondsFromBoxedLongs(
            -12219292799000001L,
            -45446999900L,
            1L,
            null,
            1584178381500000L);
        ColumnVector format = ColumnVector.fromStrings("YEAR", "HOUR", "WEEK", "QUARTER", "SECOND");
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
            -12244089600000000L,
            -46800000000L,
            -259200000000L,
            null,
            1584178381000000L);
        ColumnVector result = DateTimeUtils.truncate(input, format)) {
      assertColumnsAreEqual(expected, result);
    }
  }
}
