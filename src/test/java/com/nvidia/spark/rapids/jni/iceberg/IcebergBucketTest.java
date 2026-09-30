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

import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ai.rapids.cudf.*;
import ai.rapids.cudf.HostColumnVector.*;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

import org.apache.iceberg.transforms.Transforms;
import org.apache.iceberg.types.Types;

import org.apache.commons.lang3.RandomStringUtils;

import java.util.function.Function;

/**
 * Tests for IcebergBucket transform.
 *
 * These tests verify that the GPU implementation matches the Iceberg CPU implementation.
 */
public class IcebergBucketTest {

  // Use the maximum number of buckets to test the hash code logic.
  private static int numBuckets = Integer.MAX_VALUE;
  private static long seed;
  private static final double NULL_PROBABILITY = 0.2;
  private static final int NUM_ROWS = 1024;

  @BeforeAll
  static void setup() {
    seed = System.currentTimeMillis();
    System.out.println("IcebergBucketTest seed: " + seed);
  }

  @Test
  void testBucketInt() {
    Function<Object, Integer> bucketTransform = Transforms.bucket(numBuckets).bind(Types.IntegerType.get());

    Random rand = new Random(seed);
    Integer[] inputData = new Integer[NUM_ROWS];
    Integer[] expectedData = new Integer[NUM_ROWS];
    for (int i = 0; i < NUM_ROWS; i++) {
      if (rand.nextDouble() < NULL_PROBABILITY) {
        inputData[i] = null;
      } else {
        inputData[i] = rand.nextInt();
      }
      expectedData[i] = bucketTransform.apply(inputData[i]);
    }
    try (
        ColumnVector input = ColumnVector.fromBoxedInts(inputData);
        ColumnVector expected = ColumnVector.fromBoxedInts(expectedData);
        ColumnVector result = IcebergBucket.computeBucket(input, numBuckets)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testBucketLong() {
    Function<Object, Integer> bucketTransform = Transforms.bucket(numBuckets).bind(Types.LongType.get());

    Random rand = new Random(seed);
    Long[] inputData = new Long[NUM_ROWS];
    Integer[] expectedData = new Integer[NUM_ROWS];
    for (int i = 0; i < NUM_ROWS; i++) {
      if (rand.nextDouble() < NULL_PROBABILITY) {
        inputData[i] = null;
      } else {
        inputData[i] = rand.nextLong();
      }
      expectedData[i] = bucketTransform.apply(inputData[i]);
    }
    try (
        ColumnVector input = ColumnVector.fromBoxedLongs(inputData);
        ColumnVector expected = ColumnVector.fromBoxedInts(expectedData);
        ColumnVector result = IcebergBucket.computeBucket(input, numBuckets)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testBucketString() {
    Function<Object, Integer> bucketTransform = Transforms.bucket(numBuckets).bind(Types.StringType.get());

    Random rand = new Random(seed);
    String[] inputData = new String[NUM_ROWS];
    Integer[] expectedData = new Integer[NUM_ROWS];
    for (int i = 0; i < NUM_ROWS; i++) {
      if (rand.nextDouble() < NULL_PROBABILITY) {
        inputData[i] = null;
      } else {
        int len = rand.nextInt(128);
        inputData[i] = RandomStringUtils.random(len, 0, Character.MAX_VALUE, true, true, null, rand);
      }
      expectedData[i] = bucketTransform.apply(inputData[i]);
    }
    try (
        ColumnVector input = ColumnVector.fromStrings(inputData);
        ColumnVector expected = ColumnVector.fromBoxedInts(expectedData);
        ColumnVector result = IcebergBucket.computeBucket(input, numBuckets)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testBucketBinary() {
    Function<Object, Integer> bucketTransform = Transforms.bucket(numBuckets).bind(Types.BinaryType.get());

    Random rand = new Random(seed);
    
    @SuppressWarnings("unchecked")
    List<Byte>[] inputData = new List[NUM_ROWS];
    Integer[] expectedData = new Integer[NUM_ROWS];
    for (int i = 0; i < NUM_ROWS; i++) {
      if (rand.nextDouble() < NULL_PROBABILITY) {
        inputData[i] = null;
        expectedData[i] = null;
      } else {
        int len = rand.nextInt(20);
        byte[] arr = new byte[len];
        rand.nextBytes(arr);
        inputData[i] = new ArrayList<>();
        for (byte b : arr) {
          inputData[i].add(b);
        }
        expectedData[i] = bucketTransform.apply(ByteBuffer.wrap(arr));
      }
    }
    try (
        ColumnVector input = ColumnVector.fromLists(
            new ListType(true, new BasicType(false, DType.UINT8)),
            inputData);
        ColumnVector expected = ColumnVector.fromBoxedInts(expectedData);
        ColumnVector result = IcebergBucket.computeBucket(input, numBuckets)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testBucketDate() {
    Function<Object, Integer> bucketTransform = Transforms.bucket(numBuckets).bind(Types.DateType.get());

    Random rand = new Random(seed);
    Integer[] inputData = new Integer[NUM_ROWS];
    Integer[] expectedData = new Integer[NUM_ROWS];
    for (int i = 0; i < NUM_ROWS; i++) {
      if (rand.nextDouble() < NULL_PROBABILITY) {
        inputData[i] = null;
      } else {
        inputData[i] = rand.nextInt();
      }
      expectedData[i] = bucketTransform.apply(inputData[i]);
    }
    try (
        ColumnVector input = ColumnVector.timestampDaysFromBoxedInts(inputData);
        ColumnVector expected = ColumnVector.fromBoxedInts(expectedData);
        ColumnVector result = IcebergBucket.computeBucket(input, numBuckets)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testBucketTimestamp() {
    Function<Object, Integer> bucketTransform = Transforms.bucket(numBuckets).bind(Types.TimestampType.withZone());

    Random rand = new Random(seed);
    Long[] inputData = new Long[NUM_ROWS];
    Integer[] expectedData = new Integer[NUM_ROWS];
    for (int i = 0; i < NUM_ROWS; i++) {
      if (rand.nextDouble() < NULL_PROBABILITY) {
        inputData[i] = null;
      } else {
        inputData[i] = rand.nextLong();
      }
      expectedData[i] = bucketTransform.apply(inputData[i]);
    }
    try (
        ColumnVector input = ColumnVector.timestampMicroSecondsFromBoxedLongs(inputData);
        ColumnVector expected = ColumnVector.fromBoxedInts(expectedData);
        ColumnVector result = IcebergBucket.computeBucket(input, numBuckets)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  private static BigInteger randomBigInteger(int length, Random random) {
    boolean negative = random.nextBoolean();
    StringBuilder sb = negative ? new StringBuilder("-") : new StringBuilder();
    sb.append(random.nextInt(9) + 1);
    for (int i = 1; i < length; i++) {
      sb.append(random.nextInt(10));
    }
    return new BigInteger(sb.toString());
  }

  @Test
  void testBucketDecimal32() {
    int scale = 2;
    Function<Object, Integer> bucketTransform = Transforms.bucket(numBuckets).bind(Types.DecimalType.of(9, scale));

    Random rand = new Random(seed);
    int[] inputData = new int[NUM_ROWS];
    Integer[] expectedData = new Integer[NUM_ROWS];

    for (int i = 0; i < NUM_ROWS; i++) {
      BigInteger val = randomBigInteger(Math.min(8, DType.DECIMAL32_MAX_PRECISION), rand);
      inputData[i] = val.intValue();
      BigDecimal bd = new BigDecimal(val, scale);
      expectedData[i] = bucketTransform.apply(bd);
    }

    try (
        ColumnVector input = ColumnVector.decimalFromInts(-scale, inputData);
        ColumnVector expected = ColumnVector.fromBoxedInts(expectedData);
        ColumnVector result = IcebergBucket.computeBucket(input, numBuckets)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testBucketDecimal64() {
    int scale = 2;
    Function<Object, Integer> bucketTransform = Transforms.bucket(numBuckets).bind(Types.DecimalType.of(18, scale));

    Random rand = new Random(seed);
    long[] inputData = new long[NUM_ROWS];
    Integer[] expectedData = new Integer[NUM_ROWS];

    for (int i = 0; i < NUM_ROWS; i++) {
      BigInteger val = randomBigInteger(Math.min(17, DType.DECIMAL64_MAX_PRECISION), rand);
      inputData[i] = val.longValue();
      BigDecimal bd = new BigDecimal(val, scale);
      expectedData[i] = bucketTransform.apply(bd);
    }

    try (
        ColumnVector input = ColumnVector.decimalFromLongs(-scale, inputData);
        ColumnVector expected = ColumnVector.fromBoxedInts(expectedData);
        ColumnVector result = IcebergBucket.computeBucket(input, numBuckets)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testBucketDecimal128() {
    int scale = 2;
    Function<Object, Integer> bucketTransform = Transforms.bucket(numBuckets).bind(Types.DecimalType.of(38, scale));

    Random rand = new Random(seed);
    BigInteger[] inputData = new BigInteger[NUM_ROWS];
    Integer[] expectedData = new Integer[NUM_ROWS];

    for (int i = 0; i < NUM_ROWS; i++) {
      if (rand.nextDouble() < NULL_PROBABILITY) {
        inputData[i] = null;
      } else {
        inputData[i] = randomBigInteger(Math.min(36, DType.DECIMAL128_MAX_PRECISION), rand);
      }
      BigDecimal bd = inputData[i] == null ? null : new BigDecimal(inputData[i], scale);
      expectedData[i] = bucketTransform.apply(bd);
    }

    try (
        ColumnVector input = ColumnVector.decimalFromBigInt(-scale, inputData);
        ColumnVector expected = ColumnVector.fromBoxedInts(expectedData);
        ColumnVector result = IcebergBucket.computeBucket(input, numBuckets)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testInvalidNumBuckets() {
    try (ColumnVector input = ColumnVector.fromInts(1, 2, 3)) {
      assertThrows(IllegalArgumentException.class, () -> {
        IcebergBucket.computeBucket(input, 0);
      });
      assertThrows(IllegalArgumentException.class, () -> {
        IcebergBucket.computeBucket(input, -1);
      });
    }
  }

  @Test
  void testEmptyColumn() {
    try (
        ColumnVector input = ColumnVector.fromInts();
        ColumnVector result = IcebergBucket.computeBucket(input, numBuckets)) {
      assert result.getRowCount() == 0;
    }
  }
}
