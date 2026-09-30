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
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class InterleaveBitsTest {

  // The following source of truth comes from deltalake, but translated to java, and uses a List
  // to make our tests simpler. Deltalake only supports ints. For completeness and better
  // performance in the future we support more than this.
  static List<Byte> defaultInterleaveBits(Integer[] inputs) {
    // First do the null check/handling to match deltalake
    for (int index = 0; index < inputs.length; index++) {
      if (inputs[index] == null) {
        inputs[index] = 0;
      }
    }
    List<Byte> ret = new ArrayList<>(inputs.length * 4);
    int ret_idx = 0;
    int ret_bit = 7;
    byte ret_byte = 0;

    int bit = 31; // going from most to least significant bit
    while (bit >= 0) {
      int idx = 0;
      while (idx < inputs.length) {
        int tmp = (((inputs[idx] >> bit) & 1) << ret_bit);
        ret_byte = (byte)(ret_byte | tmp);
        ret_bit -= 1;
        if (ret_bit == -1) {
          // finished processing a byte
          ret.add(ret_idx, ret_byte);
          ret_byte = 0;
          ret_idx += 1;
          ret_bit = 7;
        }
        idx += 1;
      }
      bit -= 1;
    }
    assert(ret_idx == inputs.length * 4);
    assert(ret_bit == 7);
    return ret;
  }

  static List<Byte> defaultInterleaveBits(Short[] inputs) {
    // First do the null check/handling to match deltalake
    for (int index = 0; index < inputs.length; index++) {
      if (inputs[index] == null) {
        inputs[index] = 0;
      }
    }
    List<Byte> ret = new ArrayList<>(inputs.length * 2);
    int ret_idx = 0;
    int ret_bit = 7;
    byte ret_byte = 0;

    int bit = 15; // going from most to least significant bit
    while (bit >= 0) {
      int idx = 0;
      while (idx < inputs.length) {
        int tmp = (((inputs[idx] >> bit) & 1) << ret_bit);
        ret_byte = (byte)(ret_byte | tmp);
        ret_bit -= 1;
        if (ret_bit == -1) {
          // finished processing a byte
          ret.add(ret_idx, ret_byte);
          ret_byte = 0;
          ret_idx += 1;
          ret_bit = 7;
        }
        idx += 1;
      }
      bit -= 1;
    }
    assert(ret_idx == inputs.length * 2);
    assert(ret_bit == 7);
    return ret;
  }

  static List<Byte> defaultInterleaveBits(Byte[] inputs) {
    // First do the null check/handling to match deltalake
    for (int index = 0; index < inputs.length; index++) {
      if (inputs[index] == null) {
        inputs[index] = 0;
      }
    }
    List<Byte> ret = new ArrayList<>(inputs.length);
    int ret_idx = 0;
    int ret_bit = 7;
    byte ret_byte = 0;

    int bit = 7; // going from most to least significant bit
    while (bit >= 0) {
      int idx = 0;
      while (idx < inputs.length) {
        int tmp = (((inputs[idx] >> bit) & 1) << ret_bit);
        ret_byte = (byte)(ret_byte | tmp);
        ret_bit -= 1;
        if (ret_bit == -1) {
          // finished processing a byte
          ret.add(ret_idx, ret_byte);
          ret_byte = 0;
          ret_idx += 1;
          ret_bit = 7;
        }
        idx += 1;
      }
      bit -= 1;
    }
    assert(ret_idx == inputs.length);
    assert(ret_bit == 7);
    return ret;
  }

  static List<Byte>[] getExpected(int numRows, Integer[]... inputs) {
    List<Byte>[] ret = (List<Byte>[]) new List[numRows];
    Integer[] tmpInputs = new Integer[inputs.length];
    for (int rowIndex = 0; rowIndex < numRows; rowIndex++) {
      for (int colIndex = 0; colIndex < inputs.length; colIndex++) {
        tmpInputs[colIndex] = inputs[colIndex][rowIndex];
      }
      ret[rowIndex] = defaultInterleaveBits(tmpInputs);
    }
    return ret;
  }

  static List<Byte>[] getExpected(int numRows, Short[]... inputs) {
    List<Byte>[] ret = (List<Byte>[]) new List[numRows];
    Short[] tmpInputs = new Short[inputs.length];
    for (int rowIndex = 0; rowIndex < numRows; rowIndex++) {
      for (int colIndex = 0; colIndex < inputs.length; colIndex++) {
        tmpInputs[colIndex] = inputs[colIndex][rowIndex];
      }
      ret[rowIndex] = defaultInterleaveBits(tmpInputs);
    }
    return ret;
  }

  static List<Byte>[] getExpected(int numRows, Byte[]... inputs) {
    List<Byte>[] ret = (List<Byte>[]) new List[numRows];
    Byte[] tmpInputs = new Byte[inputs.length];
    for (int rowIndex = 0; rowIndex < numRows; rowIndex++) {
      for (int colIndex = 0; colIndex < inputs.length; colIndex++) {
        tmpInputs[colIndex] = inputs[colIndex][rowIndex];
      }
      ret[rowIndex] = defaultInterleaveBits(tmpInputs);
    }
    return ret;
  }

  public static HostColumnVector.DataType outputType =
      new HostColumnVector.ListType(true, new HostColumnVector.BasicType(false, DType.UINT8));

  public static void doIntTest(int numRows, Integer[]... inputs) {
    List<Byte>[] expected = getExpected(numRows, inputs);
    ColumnVector[] cvInputs = new ColumnVector[inputs.length];
    try {
      for (int columnIndex = 0; columnIndex < inputs.length; columnIndex++) {
        cvInputs[columnIndex] = ColumnVector.fromBoxedInts(inputs[columnIndex]);
      }
      try (ColumnVector results = ZOrder.interleaveBits(numRows, cvInputs);
           ColumnVector expectedCv = ColumnVector.fromLists(outputType, expected)) {
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

  public static void doShortTest(int numRows, Short[]... inputs) {
    List<Byte>[] expected = getExpected(numRows, inputs);
    ColumnVector[] cvInputs = new ColumnVector[inputs.length];
    try {
      for (int columnIndex = 0; columnIndex < inputs.length; columnIndex++) {
        cvInputs[columnIndex] = ColumnVector.fromBoxedShorts(inputs[columnIndex]);
      }
      try (ColumnVector results = ZOrder.interleaveBits(numRows, cvInputs);
           ColumnVector expectedCv = ColumnVector.fromLists(outputType, expected)) {
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

  public static void doByteTest(int numRows, Byte[]... inputs) {
    List<Byte>[] expected = getExpected(numRows, inputs);
    ColumnVector[] cvInputs = new ColumnVector[inputs.length];
    try {
      for (int columnIndex = 0; columnIndex < inputs.length; columnIndex++) {
        cvInputs[columnIndex] = ColumnVector.fromBoxedBytes(inputs[columnIndex]);
      }
      try (ColumnVector results = ZOrder.interleaveBits(numRows, cvInputs);
           ColumnVector expectedCv = ColumnVector.fromLists(outputType, expected)) {
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
  void testInt0() {
    doIntTest(10);
  }

  @Test
  void testShort0() {
    doShortTest(10);
  }

  @Test
  void testByte0() {
    doByteTest(10);
  }

  @Test
  void testInt1NonNull() {
    Integer[] inputs = {1, 2, 3, 4, 0x01020304};
    doIntTest(inputs.length, inputs);
  }

  @Test
  void testShort1NonNull() {
    Short[] inputs = {1, 2, 3, 4, 0x0102};
    doShortTest(inputs.length, inputs);
  }

  @Test
  void testByte1NonNull() {
    Byte[] inputs = {1, 2, 3, 4, 5};
    doByteTest(inputs.length, inputs);
  }

  @Test
  void testInt1Null() {
    Integer[] inputs = {null, 7, null, 8};
    doIntTest(inputs.length, inputs);
  }

  @Test
  void testShort1Null() {
    Short[] inputs = {null, 7, null, 8};
    doShortTest(inputs.length, inputs);
  }

  @Test
  void testByte1Null() {
    Byte[] inputs = {null, 7, null, 8};
    doByteTest(inputs.length, inputs);
  }

  @Test
  void testInt2NonNull() {
    Integer[] inputs1 = {0x01020304, 0x00000000, 0xFFFFFFFF, 0xFF00FF00};
    Integer[] inputs2 = {0x10203040, 0xFFFFFFFF, 0x00000000, 0x00FF00FF};
    doIntTest(inputs1.length, inputs1, inputs2);
  }

  @Test
  void testShort2NonNull() {
    Short[] inputs1 = {(short)0x0102, (short)0x0000, (short)0xFFFF, (short)0xFF00};
    Short[] inputs2 = {(short)0x1020, (short)0xFFFF, (short)0x0000, (short)0x00FF};
    doShortTest(inputs1.length, inputs1, inputs2);
  }

  @Test
  void testByte2NonNull() {
    Byte[] inputs1 = {(byte)0x01, (byte)0x00, (byte)0xFF, (byte)0x0F};
    Byte[] inputs2 = {(byte)0x10, (byte)0xFF, (byte)0x00, (byte)0xF0};
    doByteTest(inputs1.length, inputs1, inputs2);
  }

  @Test
  void testInt2Null() {
    Integer[] inputs1 = {0x00000000, null,       0xFFFFFFFF, 0xFF00FF00};
    Integer[] inputs2 = {0xFFFFFFFF, 0x00000000, 0x00FF00FF, null};
    doIntTest(inputs1.length, inputs1, inputs2);
  }

  @Test
  void testInt3NonNull() {
    Integer[] inputs1 = {0x00000000, 0x44444444, 0x11111111};
    Integer[] inputs2 = {0x11111111, 0x88888888, 0x22222222};
    Integer[] inputs3 = {0x22222222, 0x00000000, 0x44444444};
    doIntTest(inputs1.length, inputs1, inputs2, inputs3);
  }

  @Test
  void testShort3NonNull() {
    Short[] inputs1 = {(short)0x0000, (short)0x4444, (short)0x1111};
    Short[] inputs2 = {(short)0x1111, (short)0x8888, (short)0x2222};
    Short[] inputs3 = {(short)0x2222, (short)0x0000, (short)0x4444};
    doShortTest(inputs1.length, inputs1, inputs2, inputs3);
  }

  @Test
  void testByte3NonNull() {
    Byte[] inputs1 = {(byte)0x00, (byte)0x44, (byte)0x11};
    Byte[] inputs2 = {(byte)0x11, (byte)0x88, (byte)0x22};
    Byte[] inputs3 = {(byte)0x22, (byte)0x00, (byte)0x44};
    doByteTest(inputs1.length, inputs1, inputs2, inputs3);
  }
}