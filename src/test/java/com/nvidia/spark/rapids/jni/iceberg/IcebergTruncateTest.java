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

package com.nvidia.spark.rapids.jni.iceberg;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.HostColumnVector.BasicType;
import ai.rapids.cudf.HostColumnVector.ListType;
import org.apache.iceberg.transforms.Transforms;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.apache.iceberg.types.Types;
import org.apache.commons.lang3.RandomStringUtils;

public class IcebergTruncateTest {

  private static long seed;

  @BeforeAll
  static void setup() {
    // Use a fixed seed because Iceberg truncate can not handle overflows well
    // Refer to https://github.com/NVIDIA/cudf-spark-jni/issues/4016
    seed = 0L;
    System.out.println("IcebergTruncateTest seed: " + seed);
  }

  @Test
  void testTruncateInt() {
    int width = 10;
    try (
        ColumnVector input = ColumnVector.fromBoxedInts(null,
            0,
            1,
            5,
            9,
            10,
            11,
            -1,
            -5,
            -10,
            -11,
            null);
        ColumnVector expected = ColumnVector.fromBoxedInts(
            null,
            0,
            0,
            0,
            0,
            10,
            10,
            -10,
            -10,
            -10,
            -20,
            null);
        ColumnVector result = IcebergTruncate.truncate(input, width)) {
      assertColumnsAreEqual(expected, result);
    }

    Function<Object, Object> truncFunc = Transforms.truncate(width).bind(Types.IntegerType.get());

    Random rand = new Random(seed);
    int numRows = 1024;
    Integer[] inputData = new Integer[numRows];
    Integer[] expectedData = new Integer[numRows];
    for (int i = 0; i < numRows; i++) {
      Integer val;
      if (i % 5 == 0) {
        // 20% nulls
        val = null;
      } else {
        val = rand.nextInt();
      }
      inputData[i] = val;
      // run on CPU to get expected value
      expectedData[i] = (Integer) truncFunc.apply(val);
    }
    try (
        ColumnVector input = ColumnVector.fromBoxedInts(inputData);
        ColumnVector expected = ColumnVector.fromBoxedInts(expectedData);
        ColumnVector result = IcebergTruncate.truncate(input, width)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testTruncateLong() {
    int width = 10;
    try (
        ColumnVector input = ColumnVector.fromBoxedLongs(
            null,
            0L,
            1L,
            5L,
            9L,
            10L,
            11L,
            -1L,
            -5L,
            -10L,
            -11L,
            null);
        ColumnVector expected = ColumnVector.fromBoxedLongs(
            null,
            0L,
            0L,
            0L,
            0L,
            10L,
            10L,
            -10L,
            -10L,
            -10L,
            -20L,
            null);
        ColumnVector result = IcebergTruncate.truncate(input, width)) {
      assertColumnsAreEqual(expected, result);
    }

    Function<Object, Object> truncFunc = Transforms.truncate(width).bind(Types.LongType.get());

    Random rand = new Random(seed);
    int numRows = 1024;
    Long[] inputData = new Long[numRows];
    Long[] expectedData = new Long[numRows];
    for (int i = 0; i < numRows; i++) {
      Long val;
      if (i % 5 == 0) {
        // 20% nulls
        val = null;
      } else {
        val = rand.nextLong();
      }
      inputData[i] = val;
      // run on CPU to get expected value
      expectedData[i] = (Long) truncFunc.apply(val);
    }
    try (
        ColumnVector input = ColumnVector.fromBoxedLongs(inputData);
        ColumnVector expected = ColumnVector.fromBoxedLongs(expectedData);
        ColumnVector result = IcebergTruncate.truncate(input, width)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testTruncateString() {
    int width = 5;
    try (
        ColumnVector input = ColumnVector.fromStrings(
            null,
            "🚀23四😁678",
            "中华人民共和国",
            "",
            null);
        ColumnVector expected = ColumnVector.fromStrings(
            null,
            "🚀23四😁",
            "中华人民共",
            "",
            null);
        ColumnVector result = IcebergTruncate.truncate(input, width)) {
      assertColumnsAreEqual(expected, result);
    }

    Function<Object, Object> truncFunc = Transforms.truncate(width).bind(Types.StringType.get());

    int numRows = 1024;
    String[] inputData = new String[numRows];
    String[] expectedData = new String[numRows];
    for (int i = 0; i < numRows; i++) {
      String val;
      if (i % 5 == 0) {
        // 20% nulls
        val = null;
      } else {
        val = RandomStringUtils.randomPrint(i);
      }
      inputData[i] = val;
      // run on CPU to get expected value
      expectedData[i] = (String) truncFunc.apply(val);
    }
    try (
        ColumnVector input = ColumnVector.fromStrings(inputData);
        ColumnVector expected = ColumnVector.fromStrings(expectedData);
        ColumnVector result = IcebergTruncate.truncate(input, width)) {
      assertColumnsAreEqual(expected, result);
    }

    try (
        ColumnVector input = ColumnVector.fromStrings(inputData)) {
      // test negative width, should throw exception
      assertThrows(CudfException.class, () -> {
        IcebergTruncate.truncate(input, -width);
      });
    }
  }

  @Test
  @SuppressWarnings("unchecked")
  void testTruncateBinary() {
    int width = 10;
    try (
        ColumnVector input = ColumnVector.fromLists(
            new ListType(true, new BasicType(false, DType.UINT8)),
            Arrays.asList((byte) 1, (byte) 2, (byte) 3), // Normal case
            null, // Entire array is null
            Arrays.asList(), // Empty list
            Arrays.asList((byte) 11, (byte) 22, (byte) 33));
        ColumnVector expected = ColumnVector.fromLists(
            new ListType(true, new BasicType(false, DType.UINT8)),
            Arrays.asList((byte) 1, (byte) 2),
            null, // Entire array is null
            Collections.emptyList(), // Empty list
            Arrays.asList((byte) 11, (byte) 22));
        ColumnVector result = IcebergTruncate.truncate(input, 2)) {
      assertColumnsAreEqual(expected, result);
    }

    int numRows = 1024;
    List<Byte>[] inputData = new List[numRows];
    List<Byte>[] expectedData = new List[numRows];
    Random rand = new Random(seed);

    Function<Object, Object> truncFunc = Transforms.truncate(width).bind(Types.BinaryType.get());

    for (int i = 0; i < numRows; i++) {
      // generate random byte array
      byte[] arr;
      if (i % 5 == 0) {
        // 20% nulls
        arr = null;
        inputData[i] = null;
      } else {
        int len = rand.nextInt(20); // length up to 20
        arr = new byte[len];
        rand.nextBytes(arr);
        inputData[i] = new ArrayList<>();
        for (byte b : arr) {
          inputData[i].add(b);
        }
      }

      // run on CPU to get expected value
      ByteBuffer bb = (arr == null ? null : ByteBuffer.wrap(arr));
      ByteBuffer fromCpu = (ByteBuffer) truncFunc.apply(bb);
      if (fromCpu == null) {
        expectedData[i] = null;
      } else {
        List<Byte> cpuRet = new ArrayList<>(fromCpu.remaining());
        while (fromCpu.hasRemaining()) {
          cpuRet.add(fromCpu.get());
        }
        expectedData[i] = cpuRet;
      }
    }
    try (
        ColumnVector input = ColumnVector.fromLists(
            new ListType(true, new BasicType(false, DType.UINT8)),
            inputData);
        ColumnVector expected = ColumnVector.fromLists(
            new ListType(true, new BasicType(false, DType.UINT8)),
            expectedData);
        ColumnVector result = IcebergTruncate.truncate(input, 10)) {
      assertColumnsAreEqual(expected, result);
    }

    try (
        ColumnVector input = ColumnVector.fromLists(
            new ListType(true, new BasicType(false, DType.UINT8)),
            inputData)) {
      // test negative width, should throw exception
      assertThrows(CudfException.class, () -> {
        IcebergTruncate.truncate(input, -10);
      });
    }
  }

  private static BigInteger randomBigInteger(int length, Random random) {
    boolean negative = random.nextBoolean();
    StringBuilder sb = negative ? new StringBuilder("-") : new StringBuilder();
    for (int i = 0; i < length; i++) {
      sb.append(random.nextInt(10));
    }
    return new BigInteger(sb.toString());
  }

  private static void compareDecimals(
      DType.DTypeEnum decimalType,
      BigDecimal[] expectedValues,
      HostColumnVector actualValues,
      int numRows) {
    int precision;
    switch (decimalType) {
      case DECIMAL32:
        precision = DType.DECIMAL32_MAX_PRECISION;
        break;
      case DECIMAL64:
        precision = DType.DECIMAL64_MAX_PRECISION;
        break;
      case DECIMAL128:
        precision = DType.DECIMAL128_MAX_PRECISION;
        break;
      default:
        throw new IllegalArgumentException("Unsupported decimal type: " + decimalType);
    }

    for (int i = 0; i < numRows; i++) {
      if (expectedValues[i] == null) {
        assert actualValues.isNull(i);
      } else {
        BigDecimal actual = actualValues.getBigDecimal(i);
        BigDecimal expected = expectedValues[i];
        assertTrue(expected.precision() <= precision && actual.precision() <= precision);
        assertTrue(expected.compareTo(actual) == 0);
      }
    }
  }

  @Test
  void testTruncateDecimal32() {
    try (
        ColumnVector input = ColumnVector.decimalFromBoxedInts(-2,
            null,
            1234,
            1230,
            1229,
            5,
            -5,
            null);
        ColumnVector expected = ColumnVector.decimalFromBoxedInts(-2,
            null,
            1230,
            1230,
            1220,
            0,
            -10,
            null);
        ColumnVector result = IcebergTruncate.truncate(input, 10)) {
      assertColumnsAreEqual(expected, result);
    }

    int width = 10;
    int scale = 2;
    Function<Object, Object> truncFunc = Transforms.truncate(width)
        .bind(Types.DecimalType.of(DType.DECIMAL32_MAX_PRECISION, scale));
    int numRows = 1024;
    Integer[] inputData = new Integer[numRows];
    BigDecimal[] expectedData = new BigDecimal[numRows];
    Random rand = new Random(seed);
    for (int i = 0; i < numRows; i++) {
      // generate random BigInteger within range of precision 9
      BigInteger val;
      if (i % 5 == 0) {
        // 20% nulls
        val = null;
        inputData[i] = null;
      } else {
        val = randomBigInteger(DType.DECIMAL32_MAX_PRECISION, rand);
        inputData[i] = val.intValue();
      }

      // run on CPU to get expected value
      BigDecimal v = val == null ? null : new BigDecimal(val, scale);
      expectedData[i] = (BigDecimal) truncFunc.apply(v);
    }

    try (
        ColumnVector input = ColumnVector.decimalFromBoxedInts(-2, inputData);
        ColumnVector result = IcebergTruncate.truncate(input, 10);
        HostColumnVector ret = result.copyToHost()) {

      assertTrue(input.getType().getTypeId() == DType.DTypeEnum.DECIMAL32);
      assertTrue(result.getType().getTypeId() == DType.DTypeEnum.DECIMAL32);
      compareDecimals(DType.DTypeEnum.DECIMAL32, expectedData, ret, numRows);
    }
  }

  @Test
  void testTruncateDecimal64() {
    try (
        ColumnVector input = ColumnVector.decimalFromBoxedLongs(
            -2,
            null,
            1234L,
            1230L,
            1229L,
            5L,
            -5L,
            null);
        ColumnVector expected = ColumnVector.decimalFromBoxedLongs(
            -2,
            null,
            1230L,
            1230L,
            1220L,
            0L,
            -10L,
            null);
        ColumnVector result = IcebergTruncate.truncate(input, 10)) {
      assertColumnsAreEqual(expected, result);
    }

    int width = 10;
    int scale = 2;
    Function<Object, Object> truncFunc = Transforms.truncate(width)
        .bind(Types.DecimalType.of(DType.DECIMAL64_MAX_PRECISION, scale));
    int numRows = 1024;
    Long[] inputData = new Long[numRows];
    BigDecimal[] expectedData = new BigDecimal[numRows];
    Random rand = new Random(seed);

    for (int i = 0; i < numRows; i++) {
      // generate random BigInteger within range of precision 18
      BigInteger val;
      if (i % 5 == 0) {
        // 20% nulls
        val = null;
        inputData[i] = null;
      } else {
        val = randomBigInteger(DType.DECIMAL64_MAX_PRECISION, rand);
        inputData[i] = val.longValue();
      }

      // run on CPU to get expected value
      BigDecimal v = val == null ? null : new BigDecimal(val, scale);
      expectedData[i] = (BigDecimal) truncFunc.apply(v);
    }

    try (
        ColumnVector input = ColumnVector.decimalFromBoxedLongs(-2, inputData);
        ColumnVector result = IcebergTruncate.truncate(input, 10);
        HostColumnVector ret = result.copyToHost()) {

      assertTrue(input.getType().getTypeId() == DType.DTypeEnum.DECIMAL64);
      assertTrue(result.getType().getTypeId() == DType.DTypeEnum.DECIMAL64);
      compareDecimals(DType.DTypeEnum.DECIMAL64, expectedData, ret, numRows);
    }
  }

  @Test
  void testTruncateDecimal128() {
    try (
        ColumnVector input = ColumnVector.decimalFromBigInt(
            -2,
            null,
            new BigInteger("1234"),
            new BigInteger("1230"),
            new BigInteger("1229"),
            new BigInteger("5"),
            new BigInteger("-5"));
        ColumnVector expected = ColumnVector.decimalFromBigInt(
            -2,
            null,
            new BigInteger("1230"),
            new BigInteger("1230"),
            new BigInteger("1220"),
            new BigInteger("0"),
            new BigInteger("-10"));
        ColumnVector result = IcebergTruncate.truncate(input, 10)) {
      assertColumnsAreEqual(expected, result);
    }

    int width = 10;
    int scale = 2;
    Function<Object, Object> truncFunc = Transforms.truncate(width)
        .bind(Types.DecimalType.of(DType.DECIMAL128_MAX_PRECISION, scale));

    int numRows = 1024;
    BigInteger[] inputData = new BigInteger[numRows];
    BigDecimal[] expectedData = new BigDecimal[numRows];
    Random rand = new Random(seed);
    for (int i = 0; i < numRows; i++) {
      // GPU decimal128 supports precision up to 38,
      // but Java BigInteger can support arbitrary precision, so we just
      // generate generate random BigInteger within range of precision 38
      BigInteger val;
      if (i % 5 == 0) {
        // 20% nulls
        val = null;
        inputData[i] = null;
      } else {
        val = randomBigInteger(DType.DECIMAL128_MAX_PRECISION, rand);
        inputData[i] = val;
      }

      // run on CPU to get expected value
      BigDecimal v = val == null ? null : new BigDecimal(val, scale);
      expectedData[i] = (BigDecimal) truncFunc.apply(v);
    }

    try (
        ColumnVector input = ColumnVector.decimalFromBigInt(-2, inputData);
        ColumnVector result = IcebergTruncate.truncate(input, 10);
        HostColumnVector ret = result.copyToHost()) {
      assertTrue(input.getType().getTypeId() == DType.DTypeEnum.DECIMAL128);
      assertTrue(result.getType().getTypeId() == DType.DTypeEnum.DECIMAL128);
      compareDecimals(DType.DTypeEnum.DECIMAL128, expectedData, ret, numRows);
    }
  }
}
