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

import ai.rapids.cudf.*;

import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class NumberConverterTest {

  @Test
  void convertCvCvCvTest() {
    try (
        ColumnVector input = ColumnVector.fromStrings(
            "Z1", "34", " azc ");
        ColumnVector fromBase = ColumnVector.fromBoxedInts(
            36, 5, 34);
        ColumnVector toBase = ColumnVector.fromBoxedInts(
            10, 10, 9);
        ColumnVector expected = ColumnVector.fromStrings("1261", "19", "11")) {
      try (
          ColumnVector actual = NumberConverter.convertCvCvCv(input, fromBase, toBase)) {
        assertColumnsAreEqual(expected, actual);
        boolean actualOverflow = NumberConverter.isConvertOverflowCvCvCv(input, fromBase, toBase);
        assertFalse(actualOverflow);
      }
    }
  }

  @Test
  void convertCvCvSTest() {
    final int constToBase = 27;

    try (
        ColumnVector input = ColumnVector.fromStrings(
            "Z1", "34", " azc ");
        ColumnVector fromBase = ColumnVector.fromBoxedInts(
            7, 36, 36);
        ColumnVector expected = ColumnVector.fromStrings("0", "44", "JE3")) {
      try (
          ColumnVector actual = NumberConverter.convertCvCvS(input, fromBase, constToBase)) {
        assertColumnsAreEqual(expected, actual);
        boolean actualOverflow = NumberConverter.isConvertOverflowCvCvS(input, fromBase, constToBase);
        assertFalse(actualOverflow);
      }
    }
  }

  @Test
  void convertCvSCvTest() {
    final int constFromBase = 4;

    try (
        ColumnVector input = ColumnVector.fromStrings(
            "Z1", "34", " azc ");
        ColumnVector toBase = ColumnVector.fromBoxedInts(
            7, 9, 36);
        ColumnVector expected = ColumnVector.fromStrings("0", "3", "0")) {
      try (ColumnVector actual = NumberConverter.convertCvSCv(input, constFromBase, toBase)) {
        assertColumnsAreEqual(expected, actual);
        boolean actualOverflow = NumberConverter.isConvertOverflowCvSCv(input, constFromBase, toBase);
        assertFalse(actualOverflow);
      }
    }
  }

  @Test
  void convertCvSSTest() {
    final int constFromBase = 9;
    final int constToBase = 27;

    try (
        ColumnVector input = ColumnVector.fromStrings(
            "Z1", "34", " azc ");
        ColumnVector expected = ColumnVector.fromStrings("0", "14", "0")) {
      try (ColumnVector actual = NumberConverter.convertCvSS(input, constFromBase, constToBase)) {
        assertColumnsAreEqual(expected, actual);
        boolean actualOverflow = NumberConverter.isConvertOverflowCvSS(input, constFromBase, constToBase);
        assertFalse(actualOverflow);
      }
    }
  }

  @Test
  void convertSCvCvTest() {
    try (
        Scalar input = Scalar.fromString("-127");
        ColumnVector fromBase = ColumnVector.fromBoxedInts(
            10, 5, 34);
        ColumnVector toBase = ColumnVector.fromBoxedInts(
            16, -10, 9);
        ColumnVector expected = ColumnVector.fromStrings("FFFFFFFFFFFFFF81", "-7", "145808576354216722140")) {
      try (
          ColumnVector actual = NumberConverter.convertSCvCv(input, fromBase, toBase)) {
        assertColumnsAreEqual(expected, actual);
        boolean actualOverflow = NumberConverter.isConvertOverflowSCvCv(input, fromBase, toBase);
        assertFalse(actualOverflow);
      }
    }
  }

  @Test
  void convertSCvSTest() {
    final int constToBase = 27;

    try (
        Scalar input = Scalar.fromString("-FF4D");
        ColumnVector fromBase = ColumnVector.fromBoxedInts(
            7, 35, 36);
        ColumnVector expected = ColumnVector.fromStrings("0", "4EO8HFAM6EF567", "4EO8HFAM6EC6Q3")) {
      try (
          ColumnVector actual = NumberConverter.convertSCvS(input, fromBase, constToBase)) {
        assertColumnsAreEqual(expected, actual);
        boolean actualOverflow = NumberConverter.isConvertOverflowSCvS(input, fromBase, constToBase);
        assertFalse(actualOverflow);
      }
    }
  }

  @Test
  void convertSSCvTest() {
    final int constFromBase = 4;

    try (
        Scalar input = Scalar.fromString("11223344FFTTZZ");
        ColumnVector toBase = ColumnVector.fromBoxedInts(
            7, 9, -36);
        ColumnVector expected = ColumnVector.fromStrings("4146", "1886", "14F")) {
      try (ColumnVector actual = NumberConverter.convertSSCv(input, constFromBase, toBase)) {
        assertColumnsAreEqual(expected, actual);
        boolean actualOverflow = NumberConverter.isConvertOverflowSSCv(input, constFromBase, toBase);
        assertFalse(actualOverflow);
      }
    }
  }
}
