/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.
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
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.HostColumnVector.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.function.Function;
import java.util.zip.CRC32;

import static ai.rapids.cudf.AssertUtils.*;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class HashTest {
// IEEE 754 NaN values
  static final float POSITIVE_FLOAT_NAN_LOWER_RANGE = Float.intBitsToFloat(0x7f800001);
  static final float POSITIVE_FLOAT_NAN_UPPER_RANGE = Float.intBitsToFloat(0x7fffffff);
  static final float NEGATIVE_FLOAT_NAN_LOWER_RANGE = Float.intBitsToFloat(0xff800001);
  static final float NEGATIVE_FLOAT_NAN_UPPER_RANGE = Float.intBitsToFloat(0xffffffff);

  static final double POSITIVE_DOUBLE_NAN_LOWER_RANGE = Double.longBitsToDouble(0x7ff0000000000001L);
  static final double POSITIVE_DOUBLE_NAN_UPPER_RANGE = Double.longBitsToDouble(0x7fffffffffffffffL);
  static final double NEGATIVE_DOUBLE_NAN_LOWER_RANGE = Double.longBitsToDouble(0xfff0000000000001L);
  static final double NEGATIVE_DOUBLE_NAN_UPPER_RANGE = Double.longBitsToDouble(0xffffffffffffffffL);

  @Test
  void testSpark32BitMurmur3HashStrings() {
    try (ColumnVector v0 = ColumnVector.fromStrings(
           "a", "B\nc",  "dE\"\u0100\t\u0101 \ud720\ud721\\Fg2\'",
           "A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
           "in the MD5 hash function. This string needed to be longer.A 60 character string to " +
           "test MD5's message padding algorithm",
           "hiJ\ud720\ud721\ud720\ud721", null);
         ColumnVector result = Hash.murmurHash32(42, new ColumnVector[]{v0});
         ColumnVector expected = ColumnVector.fromBoxedInts(1485273170, 1709559900, 1423943036, 176121990, 1199621434, 42)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashInts() {
    try (ColumnVector v0 = ColumnVector.fromBoxedInts(0, 100, null, null, Integer.MIN_VALUE, null);
         ColumnVector v1 = ColumnVector.fromBoxedInts(0, null, -100, null, null, Integer.MAX_VALUE);
         ColumnVector result = Hash.murmurHash32(42, new ColumnVector[]{v0, v1});
         ColumnVector expected = ColumnVector.fromBoxedInts(59727262, 751823303, -1080202046, 42, 723455942, 133916647)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashDoubles() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles(
          0.0, null, 100.0, -100.0, Double.MIN_NORMAL, Double.MAX_VALUE,
          POSITIVE_DOUBLE_NAN_UPPER_RANGE, POSITIVE_DOUBLE_NAN_LOWER_RANGE,
          NEGATIVE_DOUBLE_NAN_UPPER_RANGE, NEGATIVE_DOUBLE_NAN_LOWER_RANGE,
          Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
         ColumnVector result = Hash.murmurHash32(new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedInts(1669671676, 0, -544903190, -1831674681, 150502665, 474144502, 1428788237, 1428788237, 1428788237, 1428788237, 420913893, 1915664072)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashTimestamps() {
    // The hash values were derived from Apache Spark in a manner similar to the one documented at
    // https://github.com/nvidia/cudf/blob/aa7ca46dcd9e/cpp/tests/hashing/hash_test.cpp#L281-L307
    try (ColumnVector v = ColumnVector.timestampMicroSecondsFromBoxedLongs(
        0L, null, 100L, -100L, 0x123456789abcdefL, null, -0x123456789abcdefL);
         ColumnVector result = Hash.murmurHash32(42, new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedInts(-1670924195, 42, 1114849490, 904948192, 657182333, 42, -57193045)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashDecimal64() {
    // The hash values were derived from Apache Spark in a manner similar to the one documented at
    // https://github.com/nvidia/cudf/blob/aa7ca46dcd9e/cpp/tests/hashing/hash_test.cpp#L281-L307
    try (ColumnVector v = ColumnVector.decimalFromLongs(-7,
        0L, 100L, -100L, 0x123456789abcdefL, -0x123456789abcdefL);
         ColumnVector result = Hash.murmurHash32(42, new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedInts(-1670924195, 1114849490, 904948192, 657182333, -57193045)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashDecimal32() {
    // The hash values were derived from Apache Spark in a manner similar to the one documented at
    // https://github.com/nvidia/cudf/blob/aa7ca46dcd9e/cpp/tests/hashing/hash_test.cpp#L281-L307
    try (ColumnVector v = ColumnVector.decimalFromInts(-3,
        0, 100, -100, 0x12345678, -0x12345678);
         ColumnVector result = Hash.murmurHash32(42, new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedInts(-1670924195, 1114849490, 904948192, -958054811, -1447702630)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashDates() {
    // The hash values were derived from Apache Spark in a manner similar to the one documented at
    // https://github.com/nvidia/cudf/blob/aa7ca46dcd9e/cpp/tests/hashing/hash_test.cpp#L281-L307
    try (ColumnVector v = ColumnVector.timestampDaysFromBoxedInts(
        0, null, 100, -100, 0x12345678, null, -0x12345678);
         ColumnVector result = Hash.murmurHash32(42, new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedInts(933211791, 42, 751823303, -1080202046, -1721170160, 42, 1852996993)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashFloats() {
    try (ColumnVector v = ColumnVector.fromBoxedFloats(
          0f, 100f, -100f, Float.MIN_NORMAL, Float.MAX_VALUE, null,
          POSITIVE_FLOAT_NAN_LOWER_RANGE, POSITIVE_FLOAT_NAN_UPPER_RANGE,
          NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE,
          Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
         ColumnVector result = Hash.murmurHash32(411, new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedInts(-235179434, 1812056886, 2028471189, 1775092689, -1531511762, 411, -1053523253, -1053523253, -1053523253, -1053523253, -1526256646, 930080402)){
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashBools() {
    try (ColumnVector v0 = ColumnVector.fromBoxedBooleans(null, true, false, true, null, false);
         ColumnVector v1 = ColumnVector.fromBoxedBooleans(null, true, false, null, false, true);
         ColumnVector result = Hash.murmurHash32(0, new ColumnVector[]{v0, v1});
         ColumnVector expected = ColumnVector.fromBoxedInts(0, -1589400010, -239939054, -68075478, 593689054, -1194558265)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashMixed() {
    try (ColumnVector strings = ColumnVector.fromStrings(
          "a", "B\n", "dE\"\u0100\t\u0101 \ud720\ud721",
          "A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
          "in the MD5 hash function. This string needed to be longer.",
          null, null);
         ColumnVector integers = ColumnVector.fromBoxedInts(0, 100, -100, Integer.MIN_VALUE, Integer.MAX_VALUE, null);
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(
          0.0, 100.0, -100.0, POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(
          0f, 100f, -100f, NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnVector bools = ColumnVector.fromBoxedBooleans(true, false, null, false, true, null);
         ColumnVector result = Hash.murmurHash32(1868, new ColumnVector[]{strings, integers, doubles, floats, bools});
         ColumnVector expected = ColumnVector.fromBoxedInts(1936985022, 720652989, 339312041, 1400354989, 769988643, 1868)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashStruct() {
    try (ColumnVector strings = ColumnVector.fromStrings(
        "a", "B\n", "dE\"\u0100\t\u0101 \ud720\ud721",
        "A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
            "in the MD5 hash function. This string needed to be longer.",
        null, null);
         ColumnVector integers = ColumnVector.fromBoxedInts(0, 100, -100, Integer.MIN_VALUE, Integer.MAX_VALUE, null);
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(
             0.0, 100.0, -100.0, POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(
             0f, 100f, -100f, NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnVector bools = ColumnVector.fromBoxedBooleans(true, false, null, false, true, null);
         ColumnView structs = ColumnView.makeStructView(strings, integers, doubles, floats, bools);
         ColumnVector result = Hash.murmurHash32(1868, new ColumnView[]{structs});
         ColumnVector expected = Hash.murmurHash32(1868, new ColumnVector[]{strings, integers, doubles, floats, bools})) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashNestedStruct() {
    try (ColumnVector strings = ColumnVector.fromStrings(
        "a", "B\n", "dE\"\u0100\t\u0101 \ud720\ud721",
        "A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
            "in the MD5 hash function. This string needed to be longer.",
        null, null);
         ColumnVector integers = ColumnVector.fromBoxedInts(0, 100, -100, Integer.MIN_VALUE, Integer.MAX_VALUE, null);
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(
             0.0, 100.0, -100.0, POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(
             0f, 100f, -100f, NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnVector bools = ColumnVector.fromBoxedBooleans(true, false, null, false, true, null);
         ColumnView structs1 = ColumnView.makeStructView(strings, integers);
         ColumnView structs2 = ColumnView.makeStructView(structs1, doubles);
         ColumnView structs3 = ColumnView.makeStructView(bools);
         ColumnView structs = ColumnView.makeStructView(structs2, floats, structs3);
         ColumnVector expected = Hash.murmurHash32(1868, new ColumnVector[]{strings, integers, doubles, floats, bools});
         ColumnVector result = Hash.murmurHash32(1868, new ColumnView[]{structs})) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashListsAndNestedLists() {
    try (ColumnVector stringListCV = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.STRING)),
             Arrays.asList(null, "a"),
             Arrays.asList("B\n", ""),
             Arrays.asList("dE\"\u0100\t\u0101", " \ud720\ud721"),
             Collections.singletonList("A very long (greater than 128 bytes/char string) to test a multi" +
             " hash-step data point in the Murmur3 hash function. This string needed to be longer."),
             Collections.singletonList(""),
             null);
         ColumnVector strings1 = ColumnVector.fromStrings(
             "a", "B\n", "dE\"\u0100\t\u0101",
             "A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
             "in the Murmur3 hash function. This string needed to be longer.", null, null);
         ColumnVector strings2 = ColumnVector.fromStrings(
             null, "", " \ud720\ud721", null, "", null);
         ColumnView stringStruct = ColumnView.makeStructView(strings1, strings2);
         ColumnVector stringExpected = Hash.murmurHash32(1868, new ColumnView[]{stringStruct});
         ColumnVector stringResult = Hash.murmurHash32(1868, new ColumnView[]{stringListCV});
         ColumnVector intListCV = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)),
             null,
             Arrays.asList(0, -2, 3),
             Collections.singletonList(Integer.MAX_VALUE),
             Arrays.asList(5, -6, null),
             Collections.singletonList(Integer.MIN_VALUE),
             null);
         ColumnVector integers1 = ColumnVector.fromBoxedInts(null, 0, null, 5, Integer.MIN_VALUE, null);
         ColumnVector integers2 = ColumnVector.fromBoxedInts(null, -2, Integer.MAX_VALUE, null, null, null);
         ColumnVector integers3 = ColumnVector.fromBoxedInts(null, 3, null, -6, null, null);
         ColumnVector intExpected =
             Hash.murmurHash32(1868, new ColumnVector[]{integers1, integers2, integers3});
         ColumnVector intResult = Hash.murmurHash32(1868, new ColumnVector[]{intListCV});
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(
          0.0, 100.0, -100.0, POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(
          0f, 100f, -100f, NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnView structCV = ColumnView.makeStructView(intListCV, stringListCV, doubles, floats);
         ColumnVector nestedExpected =
             Hash.murmurHash32(1868, new ColumnView[]{intListCV, strings1, strings2, doubles, floats});
         ColumnVector nestedResult =
             Hash.murmurHash32(1868, new ColumnView[]{structCV})) {
      assertColumnsAreEqual(stringExpected, stringResult);
      assertColumnsAreEqual(intExpected, intResult);
      assertColumnsAreEqual(nestedExpected, nestedResult);
    }
  }

  @Test
  void testXXHash64Strings() {
    try (ColumnVector v0 = ColumnVector.fromStrings(
           "a", "B\nc",  "dE\"\u0100\t\u0101 \ud720\ud721\\Fg2\'",
           "A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
           "in the MD5 hash function. This string needed to be longer.A 60 character string to " +
           "test MD5's message padding algorithm",
           "hiJ\ud720\ud721\ud720\ud721", null);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{v0});
         ColumnVector expected = ColumnVector.fromBoxedLongs(-8582455328737087284L, 2221214721321197934L, 5798966295358745941L, -4834097201550955483L, -3782648123388245694L, Hash.DEFAULT_XXHASH64_SEED)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testXXHash64Ints() {
    try (ColumnVector v0 = ColumnVector.fromBoxedInts(0, 100, null, null, Integer.MIN_VALUE, null);
         ColumnVector v1 = ColumnVector.fromBoxedInts(0, null, -100, null, null, Integer.MAX_VALUE);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{v0, v1});
         ColumnVector expected = ColumnVector.fromBoxedLongs(1151812168208346021L, -7987742665087449293L, 8990748234399402673L, Hash.DEFAULT_XXHASH64_SEED, 2073849959933241805L, 1508894993788531228L)) {
      assertColumnsAreEqual(expected, result);
    }
  }
  
  @Test
  void testXXHash64Doubles() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles(
          0.0, null, 100.0, -100.0, Double.MIN_NORMAL, Double.MAX_VALUE,
          POSITIVE_DOUBLE_NAN_UPPER_RANGE, POSITIVE_DOUBLE_NAN_LOWER_RANGE,
          NEGATIVE_DOUBLE_NAN_UPPER_RANGE, NEGATIVE_DOUBLE_NAN_LOWER_RANGE,
          Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedLongs(-5252525462095825812L, Hash.DEFAULT_XXHASH64_SEED, -7996023612001835843L, 5695175288042369293L, 6181148431538304986L, -4222314252576420879L, -3127944061524951246L, -3127944061524951246L, -3127944061524951246L, -3127944061524951246L, 5810986238603807492L, 5326262080505358431L)) {
      assertColumnsAreEqual(expected, result);
    }
  }
    
  @Test
  void testXXHash64Timestamps() {
    // The hash values were derived from Apache Spark in a manner similar to the one documented at
    // https://github.com/nvidia/cudf/blob/aa7ca46dcd9e/cpp/tests/hashing/hash_test.cpp#L281-L307
    try (ColumnVector v = ColumnVector.timestampMicroSecondsFromBoxedLongs(
        0L, null, 100L, -100L, 0x123456789abcdefL, null, -0x123456789abcdefL);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedLongs(-5252525462095825812L, Hash.DEFAULT_XXHASH64_SEED, 8713583529807266080L, 5675770457807661948L, 1941233597257011502L, Hash.DEFAULT_XXHASH64_SEED, -1318946533059658749L)) {
      assertColumnsAreEqual(expected, result);
    }
  }
  
  @Test
  void testXXHash64Decimal64() {
    // The hash values were derived from Apache Spark in a manner similar to the one documented at
    // https://github.com/nvidia/cudf/blob/aa7ca46dcd9e/cpp/tests/hashing/hash_test.cpp#L281-L307
    try (ColumnVector v = ColumnVector.decimalFromLongs(-7,
        0L, 100L, -100L, 0x123456789abcdefL, -0x123456789abcdefL);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedLongs(-5252525462095825812L, 8713583529807266080L, 5675770457807661948L, 1941233597257011502L, -1318946533059658749L)) {
      assertColumnsAreEqual(expected, result);
    }
  }
    
  @Test
  void testXXHash64Decimal32() {
    // The hash values were derived from Apache Spark in a manner similar to the one documented at
    // https://github.com/nvidia/cudf/blob/aa7ca46dcd9e/cpp/tests/hashing/hash_test.cpp#L281-L307
    try (ColumnVector v = ColumnVector.decimalFromInts(-3,
        0, 100, -100, 0x12345678, -0x12345678);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedLongs(-5252525462095825812L, 8713583529807266080L, 5675770457807661948L, -7728554078125612835L, 3142315292375031143L)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testXXHash64Dates() {
    // The hash values were derived from Apache Spark in a manner similar to the one documented at
    // https://github.com/nvidia/cudf/blob/aa7ca46dcd9e/cpp/tests/hashing/hash_test.cpp#L281-L307
    try (ColumnVector v = ColumnVector.timestampDaysFromBoxedInts(
        0, null, 100, -100, 0x12345678, null, -0x12345678);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedLongs(3614696996920510707L, Hash.DEFAULT_XXHASH64_SEED, -7987742665087449293L, 8990748234399402673L, 6954428822481665164L, Hash.DEFAULT_XXHASH64_SEED, -4294222333805341278L)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testXXHash64Floats() {
    try (ColumnVector v = ColumnVector.fromBoxedFloats(
          0f, 100f, -100f, Float.MIN_NORMAL, Float.MAX_VALUE, null,
          POSITIVE_FLOAT_NAN_LOWER_RANGE, POSITIVE_FLOAT_NAN_UPPER_RANGE,
          NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE,
          Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedLongs(3614696996920510707L, -8232251799677946044L, -6625719127870404449L, -6699704595004115126L, -1065250890878313112L, Hash.DEFAULT_XXHASH64_SEED, 2692338816207849720L, 2692338816207849720L, 2692338816207849720L, 2692338816207849720L, -5940311692336719973L, -7580553461823983095L)){
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testXXHash64Bools() {
    try (ColumnVector v0 = ColumnVector.fromBoxedBooleans(null, true, false, true, null, false);
         ColumnVector v1 = ColumnVector.fromBoxedBooleans(null, true, false, null, false, true);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{v0, v1});
         ColumnVector expected = ColumnVector.fromBoxedLongs(Hash.DEFAULT_XXHASH64_SEED, 9083826852238114423L, 1151812168208346021L, -6698625589789238999L, 3614696996920510707L, 7945966957015589024L)) {
      assertColumnsAreEqual(expected, result);
    }
  }
  
  @Test
  void testXXHash64Mixed() {
    try (ColumnVector strings = ColumnVector.fromStrings(
          "a", "B\n", "dE\"\u0100\t\u0101 \ud720\ud721",
          "A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
          "in the MD5 hash function. This string needed to be longer.",
          null, null);
         ColumnVector integers = ColumnVector.fromBoxedInts(0, 100, -100, Integer.MIN_VALUE, Integer.MAX_VALUE, null);
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(
          0.0, 100.0, -100.0, POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(
          0f, 100f, -100f, NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnVector bools = ColumnVector.fromBoxedBooleans(true, false, null, false, true, null);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{strings, integers, doubles, floats, bools});
         ColumnVector expected = ColumnVector.fromBoxedLongs(7451748878409563026L, 6024043102550151964L, 3380664624738534402L, 8444697026100086329L, -5888679192448042852L, Hash.DEFAULT_XXHASH64_SEED)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testXXHash64Struct() {
    try (ColumnVector strings = ColumnVector.fromStrings(
          "a", "B\n", "dE\"\u0100\t\u0101 \ud720\ud721",
          "A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
          "in the MD5 hash function. This string needed to be longer.",
          null, null);
         ColumnVector integers = ColumnVector.fromBoxedInts(0, 100, -100, Integer.MIN_VALUE, Integer.MAX_VALUE, null);
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(
          0.0, 100.0, -100.0, POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(
          0f, 100f, -100f, NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnVector bools = ColumnVector.fromBoxedBooleans(true, false, null, false, true, null);
         ColumnView structs = ColumnView.makeStructView(strings, integers, doubles, floats, bools);
         ColumnVector result = Hash.xxhash64(new ColumnView[]{structs});
         ColumnVector expected = ColumnVector.fromBoxedLongs(7451748878409563026L, 6024043102550151964L, 3380664624738534402L, 8444697026100086329L, -5888679192448042852L, Hash.DEFAULT_XXHASH64_SEED)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testXXHash64NestedStruct() {
    try (ColumnVector strings = ColumnVector.fromStrings(
          "a", "B\n", "dE\"\u0100\t\u0101 \ud720\ud721",
          "A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
          "in the MD5 hash function. This string needed to be longer.",
          null, null);
         ColumnVector integers = ColumnVector.fromBoxedInts(0, 100, -100, Integer.MIN_VALUE, Integer.MAX_VALUE, null);
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(
          0.0, 100.0, -100.0, POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(
          0f, 100f, -100f, NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnVector bools = ColumnVector.fromBoxedBooleans(true, false, null, false, true, null);
         ColumnView structs1 = ColumnView.makeStructView(strings, integers);
         ColumnView structs2 = ColumnView.makeStructView(structs1, doubles);
         ColumnView structs3 = ColumnView.makeStructView(bools);
         ColumnView structs = ColumnView.makeStructView(structs2, floats, structs3);
         ColumnVector result = Hash.xxhash64(new ColumnView[]{structs});
         ColumnVector expected = ColumnVector.fromBoxedLongs(7451748878409563026L, 6024043102550151964L, 3380664624738534402L, 8444697026100086329L, -5888679192448042852L, Hash.DEFAULT_XXHASH64_SEED)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testXXHash64Lists() {
    try (ColumnVector stringListCV = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.STRING)),
             Arrays.asList(null, "a"),
             Arrays.asList("B\n", ""),
             Arrays.asList("dE\"\u0100\t\u0101", " \ud720\ud721"),
             Collections.singletonList("A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
             "in the MD5 hash function. This string needed to be longer."),
             Collections.singletonList(""),
             null);
         ColumnVector stringExpected = ColumnVector.fromBoxedLongs(-8582455328737087284L, 7160715839242204087L, -862482741676457612L, -3700309651391443614L, -7444071767201028348L, Hash.DEFAULT_XXHASH64_SEED);
         ColumnVector stringResult = Hash.xxhash64(new ColumnView[]{stringListCV});
         ColumnVector intListCV = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)),
             Collections.emptyList(),
             Arrays.asList(0, -2, 3),
             Collections.singletonList(Integer.MAX_VALUE),
             Arrays.asList(5, -6, null),
             Collections.singletonList(Integer.MIN_VALUE),
             null);
         ColumnVector intExpected = ColumnVector.fromBoxedLongs(Hash.DEFAULT_XXHASH64_SEED, -4022702357093761688L, 1508894993788531228L, 7329154841501342665L, 2073849959933241805L, Hash.DEFAULT_XXHASH64_SEED);
         ColumnVector intResult = Hash.xxhash64(new ColumnVector[]{intListCV})) {
      assertColumnsAreEqual(stringExpected, stringResult);
      assertColumnsAreEqual(intExpected, intResult);
    }
  }

  @Test
  void testXXHash64NestedLists() {
    try (ColumnVector nestedStringListCV = ColumnVector.fromLists(
             new ListType(true, new ListType(true, new BasicType(true, DType.STRING))),
             Arrays.asList(null, Collections.singletonList("a")),
             Collections.singletonList(Arrays.asList("B\n", "")),
             Arrays.asList(Collections.singletonList("dE\"\u0100\t\u0101"), Collections.singletonList(" \ud720\ud721")),
             Collections.singletonList(Collections.singletonList("A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
             "in the MD5 hash function. This string needed to be longer.")),
             Collections.singletonList(Collections.singletonList("")),
             null);
         ColumnVector stringExpected = ColumnVector.fromBoxedLongs(-8582455328737087284L, 7160715839242204087L, -862482741676457612L, -3700309651391443614L, -7444071767201028348L, Hash.DEFAULT_XXHASH64_SEED);
         ColumnVector stringResult = Hash.xxhash64(new ColumnView[]{nestedStringListCV});
         ColumnVector nestedIntListCV = ColumnVector.fromLists(
             new ListType(true, new ListType(true, new BasicType(true, DType.INT32))),
             Collections.emptyList(),
             Arrays.asList(Collections.singletonList(0), Collections.singletonList(-2), Collections.singletonList(3)),
             Collections.singletonList(Collections.singletonList(Integer.MAX_VALUE)),
             Arrays.asList(Collections.singletonList(5), Arrays.asList(-6, null)),
             Collections.singletonList(Collections.singletonList(Integer.MIN_VALUE)),
             null);
         ColumnVector intExpected = ColumnVector.fromBoxedLongs(Hash.DEFAULT_XXHASH64_SEED, -4022702357093761688L, 1508894993788531228L, 7329154841501342665L, 2073849959933241805L, Hash.DEFAULT_XXHASH64_SEED);
         ColumnVector intResult = Hash.xxhash64(new ColumnVector[]{nestedIntListCV});) {
      assertColumnsAreEqual(stringExpected, stringResult);
      assertColumnsAreEqual(intExpected, intResult);
    }
  }

  @Test
  void testXXHash64StructOfList() {
    try (ColumnVector stringListCV = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.STRING)),
             Arrays.asList(null, "a"),
             Arrays.asList("B\n", ""),
             Arrays.asList("dE\"\u0100\t\u0101", " \ud720\ud721"),
             Collections.singletonList("A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
             "in the MD5 hash function. This string needed to be longer."),
             Collections.singletonList(""),
             null);
         ColumnVector intListCV = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)),
             Collections.emptyList(),
             Arrays.asList(0, -2, 3),
             Collections.singletonList(Integer.MAX_VALUE),
             Arrays.asList(5, -6, null),
             Collections.singletonList(Integer.MIN_VALUE),
             null);
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(
         0.0, 100.0, -100.0, POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(
         0f, 100f, -100f, NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnView structCV = ColumnView.makeStructView(intListCV, stringListCV, doubles, floats);
         ColumnVector nestedExpected = ColumnVector.fromBoxedLongs(-8492741646850220468L, -6547737320918905493L, -8718220625378038731L, 5441580647216064522L, 3645801243834961127L, Hash.DEFAULT_XXHASH64_SEED);
         ColumnVector nestedResult = Hash.xxhash64(new ColumnView[]{structCV})) {
      assertColumnsAreEqual(nestedExpected, nestedResult);
    }
  }

  @Test
  void testXXHash64ListOfStruct() {
    try (ColumnVector structListCV = ColumnVector.fromLists(new ListType(true, new StructType(true,
              new BasicType(true, DType.STRING), new BasicType(true, DType.INT32), new BasicType(true, DType.FLOAT64), new BasicType(true, DType.FLOAT32), new BasicType(true, DType.BOOL8))),
             Collections.emptyList(),
             Collections.singletonList(new StructData("a", 0, 0.0, 0f, true)),
             Arrays.asList(new StructData("B\n", 100, 100.0, 100f, false), new StructData("dE\"\u0100\t\u0101 \ud720\ud721", -100, -100.0, -100f, null)),
             Collections.singletonList(new StructData("A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
             "in the MD5 hash function. This string needed to be longer.", Integer.MIN_VALUE, POSITIVE_DOUBLE_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_LOWER_RANGE, false)),
             Arrays.asList(new StructData(null, Integer.MAX_VALUE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, true), new StructData(null, null, null, null, null)),
             null);
         ColumnVector result = Hash.xxhash64(new ColumnView[]{structListCV});
         ColumnVector expected = ColumnVector.fromBoxedLongs(Hash.DEFAULT_XXHASH64_SEED, 7451748878409563026L, 948372773124634350L, 8444697026100086329L, -5888679192448042852L, Hash.DEFAULT_XXHASH64_SEED)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testXXHash64NestedDepthExceedsLimit() {
    try (ColumnVector nestedIntListCV = ColumnVector.fromLists(
            new ListType(true, new ListType(true, new BasicType(true, DType.INT32))),
            Arrays.asList(Arrays.asList(null, null), null),
            Arrays.asList(Collections.singletonList(0), Collections.singletonList(-2), Collections.singletonList(3)),
            Arrays.asList(null, Collections.singletonList(Integer.MAX_VALUE)),
            Arrays.asList(Collections.singletonList(5), Arrays.asList(-6, null)),
            Arrays.asList(Collections.singletonList(Integer.MIN_VALUE), null),
            null);
         ColumnVector integers = ColumnVector.fromBoxedInts(
            0, 100, -100, Integer.MIN_VALUE, Integer.MAX_VALUE, null);
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(0.0, 100.0, -100.0,
            POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(0f, 100f, -100f,
            NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnVector bools = ColumnVector.fromBoxedBooleans(
            true, false, null, false, true, null);
         ColumnView structs1 = ColumnView.makeStructView(nestedIntListCV, integers);
         ColumnView structs2 = ColumnView.makeStructView(structs1, doubles);
         ColumnView structs3 = ColumnView.makeStructView(structs2, bools);
         ColumnView structs4 = ColumnView.makeStructView(structs3);
         ColumnView structs5 = ColumnView.makeStructView(structs4, floats);
         ColumnView structs6 = ColumnView.makeStructView(structs5);
         ColumnView structs7 = ColumnView.makeStructView(structs6);
         ColumnView nestedResult = ColumnView.makeStructView(structs7);) {
      assertThrows(CudfException.class, () -> Hash.xxhash64(new ColumnView[]{nestedResult}));
    }
  }

  @Test
  void testHiveHashBools() {
    try (ColumnVector v0 = ColumnVector.fromBoxedBooleans(true, false, null);
         ColumnVector result = Hash.hiveHash(new ColumnVector[]{v0});
         ColumnVector expected = ColumnVector.fromInts(1, 0, 0)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testHiveHashInts() {
    try (ColumnVector v0 = ColumnVector.fromBoxedInts(
          Integer.MIN_VALUE, Integer.MAX_VALUE, -1, 1, -10, 10, null);
         ColumnVector result = Hash.hiveHash(new ColumnVector[]{v0});
         ColumnVector expected = ColumnVector.fromInts(
          Integer.MIN_VALUE, Integer.MAX_VALUE, -1, 1, -10, 10, 0)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testHiveHashBytes() {
    try (ColumnVector v0 = ColumnVector.fromBoxedBytes(
         Byte.MIN_VALUE, Byte.MAX_VALUE, (byte)-1, (byte)1, (byte)-10, (byte)10, null);
         ColumnVector result = Hash.hiveHash(new ColumnVector[]{v0});
         ColumnVector expected = ColumnVector.fromInts(
          Byte.MIN_VALUE, Byte.MAX_VALUE, -1, 1, -10, 10, 0)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testHiveHashLongs() {
    try (ColumnVector v0 = ColumnVector.fromBoxedLongs(
          Long.MIN_VALUE, Long.MAX_VALUE, -1L, 1L, -10L, 10L, null);
         ColumnVector result = Hash.hiveHash(new ColumnVector[]{v0});
         ColumnVector expected = ColumnVector.fromInts(
          Integer.MIN_VALUE, Integer.MIN_VALUE, 0, 1, 9, 10, 0)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testHiveHashStrings() {
    try (ColumnVector v0 = ColumnVector.fromStrings(
          "a", "B\n", "dE\"\u0100\t\u0101 \ud720\ud721", null,
          "This is a long string (greater than 128 bytes/char string) case to test this " +
          "hash function. Just want an abnormal case here to see if any error may happen when" +
          "doing the hive hashing");
         ColumnVector result = Hash.hiveHash(new ColumnVector[]{v0});
         ColumnVector expected = ColumnVector.fromInts(97, 2056, 745239896, 0, 2112075710)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testHiveHashFloats() {
    try (ColumnVector v = ColumnVector.fromBoxedFloats(0f, 100f, -100f, Float.MIN_NORMAL,
          Float.MAX_VALUE, null, Float.MIN_VALUE,
          POSITIVE_FLOAT_NAN_LOWER_RANGE, POSITIVE_FLOAT_NAN_UPPER_RANGE,
          NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE,
          Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
         ColumnVector result = Hash.hiveHash(new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromInts(0, 1120403456, -1027080192, 8388608,
          2139095039, 0, 1, 2143289344, 2143289344, 2143289344, 2143289344, 2139095040, -8388608)){
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testHiveHashDoubles() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles(0.0, 100.0, -100.0,
          POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector result = Hash.hiveHash(new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromInts(0, 1079574528, -1067909120,
          2146959360, 2146959360, 0)){
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testHiveHashDates() {
    try (ColumnVector v = ColumnVector.timestampDaysFromBoxedInts(
          0, null, 100, -100, 0x12345678, null, -0x12345678);
         ColumnVector result = Hash.hiveHash(new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromInts(
          0, 0, 100, -100, 0x12345678, 0, -0x12345678)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testHiveHashTimestamps() {
    try (ColumnVector v = ColumnVector.timestampMicroSecondsFromBoxedLongs(
        0L, null, 100L, -100L, 0x123456789abcdefL, null, -0x123456789abcdefL);
         ColumnVector result = Hash.hiveHash(new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromInts(
          0, 0, 100000, 99999, -660040456, 0, 486894999)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testHiveHashMixed() {
    try (ColumnVector strings = ColumnVector.fromStrings(
          "a", "B\n", "dE\"\u0100\t\u0101 \ud720\ud721",
          "This is a long string (greater than 128 bytes/char string) case to test this " +
          "hash function. Just want an abnormal case here to see if any error may happen when" +
          "doing the hive hashing",
          null, null);
         ColumnVector integers = ColumnVector.fromBoxedInts(
          0, 100, -100, Integer.MIN_VALUE, Integer.MAX_VALUE, null);
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(0.0, 100.0, -100.0,
          POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(0f, 100f, -100f,
          NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnVector bools = ColumnVector.fromBoxedBooleans(
          true, false, null, false, true, null);
         ColumnVector result = Hash.hiveHash(new ColumnVector[]{
          strings, integers, doubles, floats, bools});
         ColumnVector expected = ColumnVector.fromInts(89581538, 363542820, 413439036,
          1272817854, 1513589666, 0)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testHiveHashStruct() {
    try (ColumnVector strings = ColumnVector.fromStrings(
          "a", "B\n", "dE\"\u0100\t\u0101 \ud720\ud721",
          "This is a long string (greater than 128 bytes/char string) case to test this " +
          "hash function. Just want an abnormal case here to see if any error may happen when" +
          "doing the hive hashing",
          null, null);
         ColumnVector integers = ColumnVector.fromBoxedInts(
          0, 100, -100, Integer.MIN_VALUE, Integer.MAX_VALUE, null);
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(0.0, 100.0, -100.0,
          POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(0f, 100f, -100f,
          NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnVector bools = ColumnVector.fromBoxedBooleans(
          true, false, null, false, true, null);
         ColumnView structs = ColumnView.makeStructView(strings, integers, doubles, floats, bools);
         ColumnVector result = Hash.hiveHash(new ColumnView[]{structs});
         ColumnVector expected = ColumnVector.fromInts(89581538, 363542820, 413439036,
          1272817854, 1513589666, 0)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testHiveHashNestedStruct() {
    try (ColumnVector strings = ColumnVector.fromStrings(
          "a", "B\n", "dE\"\u0100\t\u0101 \ud720\ud721",
          "This is a long string (greater than 128 bytes/char string) case to test this " +
          "hash function. Just want an abnormal case here to see if any error may happen when" +
          "doing the hive hashing",
          null, null);
         ColumnVector integers = ColumnVector.fromBoxedInts(
          0, 100, -100, Integer.MIN_VALUE, Integer.MAX_VALUE, null);
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(0.0, 100.0, -100.0,
          POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(0f, 100f, -100f,
          NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnVector bools = ColumnVector.fromBoxedBooleans(
          true, false, null, false, true, null);
         ColumnView structs1 = ColumnView.makeStructView(strings, integers);
         ColumnView structs2 = ColumnView.makeStructView(structs1, doubles);
         ColumnView structs3 = ColumnView.makeStructView(bools);
         ColumnView structs = ColumnView.makeStructView(structs2, floats, structs3);
         ColumnVector result = Hash.hiveHash(new ColumnView[]{structs});
         ColumnVector expected = ColumnVector.fromInts(89581538, 363542820, 413439036,
          1272817854, 1513589666, 0)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testHiveHashLists() {
    try (ColumnVector stringListCV = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.STRING)),
             Arrays.asList(null, "a"),
             Arrays.asList("B\n", ""),
             Arrays.asList("dE\"\u0100\t\u0101", " \ud720\ud721"),
             Collections.singletonList("This is a long string (greater than 128 bytes/char string) case to test this " +
             "hash function. Just want an abnormal case here to see if any error may happen when" +
             "doing the hive hashing"),
             Collections.singletonList(""),
             null);
         ColumnVector stringResult = Hash.hiveHash(new ColumnView[]{stringListCV});
         ColumnVector stringExpected = ColumnVector.fromInts(97, 63736, -96263528, 2112075710, 0, 0);
         ColumnVector intListCV = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)),
             Collections.emptyList(),
             Arrays.asList(0, -2, 3),
             Collections.singletonList(Integer.MAX_VALUE),
             Arrays.asList(5, -6, null),
             Collections.singletonList(Integer.MIN_VALUE),
             null);
         ColumnVector intExpected = ColumnVector.fromInts(0, -59, 2147483647, 4619, -2147483648, 0);
         ColumnVector intResult = Hash.hiveHash(new ColumnVector[]{intListCV});) {
      assertColumnsAreEqual(stringExpected, stringResult);
      assertColumnsAreEqual(intExpected, intResult);
    }
  }

  @Test
  void testHiveHashNestedLists() {
    try (ColumnVector nestedStringListCV = ColumnVector.fromLists(
            new ListType(true, new ListType(true, new BasicType(true, DType.STRING))),
            Arrays.asList(null, Arrays.asList("a", null)),
            Arrays.asList(Arrays.asList("B\n", "")),
            Arrays.asList(Collections.singletonList("dE\"\u0100\t\u0101"), Collections.singletonList(" \ud720\ud721")),
            Arrays.asList(Collections.singletonList("This is a long string (greater than 128 bytes/char string) case to test this " +
              "hash function. Just want an abnormal case here to see if any error may happen when" +
              "doing the hive hashing")),
            Arrays.asList(Collections.singletonList(""), null),
            null);
         ColumnVector stringExpected = ColumnVector.fromInts(3007, 63736, -96263528, 2112075710, 0, 0);
         ColumnVector stringResult = Hash.hiveHash(new ColumnView[]{nestedStringListCV});
         ColumnVector nestedIntListCV = ColumnVector.fromLists(
             new ListType(true, new ListType(true, new BasicType(true, DType.INT32))),
             Arrays.asList(Arrays.asList(null, null), null),
             Arrays.asList(Collections.singletonList(0), Collections.singletonList(-2), Collections.singletonList(3)),
             Arrays.asList(null, Collections.singletonList(Integer.MAX_VALUE)),
             Arrays.asList(Collections.singletonList(5), Arrays.asList(-6, null)),
             Arrays.asList(Collections.singletonList(Integer.MIN_VALUE), null),
             null);
         ColumnVector intExpected = ColumnVector.fromInts(0, -59, 2147483647, -31, -2147483648, 0);
         ColumnVector intResult = Hash.hiveHash(new ColumnVector[]{nestedIntListCV});) {
      assertColumnsAreEqual(stringExpected, stringResult);
      assertColumnsAreEqual(intExpected, intResult);
    }
  }

  @Test
  void testHiveHashStructOfList() {
    try (ColumnVector stringListCV = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.STRING)),
             Arrays.asList(null, "a"),
             Arrays.asList("B\n", ""),
             Arrays.asList("dE\"\u0100\t\u0101", " \ud720\ud721"),
             Collections.singletonList("This is a long string (greater than 128 bytes/char string) case to test this " +
             "hash function. Just want an abnormal case here to see if any error may happen when" +
             "doing the hive hashing"),
             Collections.singletonList(""),
             null);
         ColumnVector intListCV = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)),
             Collections.singletonList(null),
             Arrays.asList(0, -2, 3),
             Collections.singletonList(Integer.MAX_VALUE),
             Arrays.asList(5, -6, null),
             Collections.singletonList(Integer.MIN_VALUE),
             null);
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(
          0.0, 100.0, -100.0, POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(
          0f, 100f, -100f, NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnView structCV = ColumnView.makeStructView(intListCV, stringListCV, doubles, floats);
         ColumnVector nestedExpected = ColumnVector.fromInts(93217, 286968083, 59992121, -1697616301, 2127036416, 0);
         ColumnVector nestedResult = Hash.hiveHash(new ColumnView[]{structCV})) {
      assertColumnsAreEqual(nestedExpected, nestedResult);
    }
  }

  @Test
  void testHiveHashListOfStruct() {
    try (ColumnVector structListCV = ColumnVector.fromLists(new ListType(true, new StructType(true,
              new BasicType(true, DType.STRING), new BasicType(true, DType.INT32), new BasicType(true, DType.FLOAT64), new BasicType(true, DType.FLOAT32), new BasicType(true, DType.BOOL8))),
             Collections.emptyList(),
             Collections.singletonList(new StructData("a", 0, 0.0, 0f, true)),
             Arrays.asList(new StructData("B\n", 100, 100.0, 100f, false), new StructData("dE\"\u0100\t\u0101 \ud720\ud721", -100, -100.0, -100f, null)),
             Collections.singletonList(new StructData("This is a long string (greater than 128 bytes/char string) case to test this " +
             "hash function. Just want an abnormal case here to see if any error may happen when" + "doing the hive hashing", Integer.MIN_VALUE, POSITIVE_DOUBLE_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_LOWER_RANGE, false)),
             Arrays.asList(new StructData(null, Integer.MAX_VALUE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, true), new StructData(null, null, null, null, null)),
             null);
         ColumnVector result = Hash.hiveHash(new ColumnView[]{structListCV});
         ColumnVector expected = ColumnVector.fromInts(0, 89581538, -1201635432, 1272817854, -323360610, 0)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testHiveHashNestedDepthExceedsLimit() {
    try (ColumnVector nestedIntListCV = ColumnVector.fromLists(
            new ListType(true, new ListType(true, new BasicType(true, DType.INT32))),
            Arrays.asList(Arrays.asList(null, null), null),
            Arrays.asList(Collections.singletonList(0), Collections.singletonList(-2), Collections.singletonList(3)),
            Arrays.asList(null, Collections.singletonList(Integer.MAX_VALUE)),
            Arrays.asList(Collections.singletonList(5), Arrays.asList(-6, null)),
            Arrays.asList(Collections.singletonList(Integer.MIN_VALUE), null),
            null);
         ColumnVector integers = ColumnVector.fromBoxedInts(
            0, 100, -100, Integer.MIN_VALUE, Integer.MAX_VALUE, null);
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(0.0, 100.0, -100.0,
            POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(0f, 100f, -100f,
            NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnVector bools = ColumnVector.fromBoxedBooleans(
            true, false, null, false, true, null);
         ColumnView structs1 = ColumnView.makeStructView(nestedIntListCV, integers);
         ColumnView structs2 = ColumnView.makeStructView(structs1, doubles);
         ColumnView structs3 = ColumnView.makeStructView(structs2, bools);
         ColumnView structs4 = ColumnView.makeStructView(structs3);
         ColumnView structs5 = ColumnView.makeStructView(structs4, floats);
         ColumnView structs6 = ColumnView.makeStructView(structs5);
         ColumnView nestedResult = ColumnView.makeStructView(structs6);) {
      assertThrows(CudfException.class, () -> Hash.hiveHash(new ColumnView[]{nestedResult}));
    }
  }

  private static final String[] sha2TestInputStrings = {
    null,
    "",
    "0",
    "A 56 character string to test message padding algorithm.",
    "A 63 character string to test message padding algorithm, again.",
    "A 64 character string to test message padding algorithm, again!!",
    "A very long (greater than 128 bytes/char string) to execute a multi hash-step data point in " +
    "the hash function being tested. This string needed to be longer.",
    "All work and no play makes Jack a dull boy",
    "",
    "Multi-byte characters: é¼³⅝",
    "(!\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`{|}~)" // All the special characters.
  };

  private static String[] computeSha2OnCpu(String[] inputStrings, String algoString) 
    throws NoSuchAlgorithmException {
    MessageDigest digest = MessageDigest.getInstance(algoString);
    String[] outputStrings = new String[inputStrings.length];
    StringBuilder hexString = new StringBuilder();
    for (int i = 0; i < inputStrings.length; i++) {
      if (inputStrings[i] == null) {
        outputStrings[i] = null; // The key difference between CUDF and Spark.
      }
      else {
        hexString.setLength(0);
        byte[] hashBytes = digest.digest(inputStrings[i].getBytes(StandardCharsets.UTF_8));
        for (byte b : hashBytes) {
          String hex = Integer.toHexString(0xff & b);
          if (hex.length() == 1) {
            hexString.append('0');
          }
          hexString.append(hex);
        }
        outputStrings[i] = hexString.toString();
      }
    }
    return outputStrings;
  }

  private void testSha2Impl(Function<ColumnView, ColumnVector> hashFunction, String algo) {
    try (ColumnVector inputStrings = ColumnVector.fromStrings(sha2TestInputStrings);
         ColumnVector result = hashFunction.apply(inputStrings);
         ColumnVector expectedOnCpu = 
           ColumnVector.fromStrings(computeSha2OnCpu(sha2TestInputStrings, algo))) {
      // Outputs can be verified on the shell with `sha224sum` or equivalent:
      // ```bash
      // echo -n "input string" | sha224sum
      // ```
      assertColumnsAreEqual(expectedOnCpu, result);
    }
    catch (NoSuchAlgorithmException e) {
      org.junit.jupiter.api.Assertions.fail("Unexpected failure: " + e.getMessage());
    }
  }

  @Test
  void testSha224NullsPreserved() {
    testSha2Impl(Hash::sha224NullsPreserved, "SHA-224");
  }

  @Test
  void testSha256NullsPreserved() {
    testSha2Impl(Hash::sha256NullsPreserved, "SHA-256");
  }
  
  @Test
  void testSha384NullsPreserved() {
    testSha2Impl(Hash::sha384NullsPreserved, "SHA-384");
  }

  @Test
  void testSha512NullsPreserved() {
    testSha2Impl(Hash::sha512NullsPreserved, "SHA-512");
  }

  @ParameterizedTest
  @ValueSource(ints = {0, 1, 10, 100, 256, 1000, 4096})
  void testCrc32WithBinaryData(int size) {
    Random random = new Random(17); // Fixed seed for reproducibility

    byte[] binaryData = new byte[size];
    random.nextBytes(binaryData);

    CRC32 javaCrc32 = new CRC32();
    javaCrc32.update(binaryData);
    long expected = javaCrc32.getValue();

    try (HostMemoryBuffer buffer = HostMemoryBuffer.allocate(binaryData.length)) {
      buffer.setBytes(0, binaryData, 0, binaryData.length);
      long actual = Hash.hostCrc32(0, buffer);

      assertEquals(expected, actual, "CRC32 mismatch for binary data of size " + size);
    }
  }

  @Test
  void testCrc32IncrementalUpdate() {
    Random random = new Random(17); // Fixed seed for reproducibility

    int part1Size = random.nextInt(1024); // 0 to 1023 bytes
    int part2Size = random.nextInt(1024);

    byte[] bytes1 = new byte[part1Size];
    byte[] bytes2 = new byte[part2Size];
    random.nextBytes(bytes1);
    random.nextBytes(bytes2);

    CRC32 javaCrc32 = new CRC32();
    javaCrc32.update(bytes1);
    javaCrc32.update(bytes2);
    long expected = javaCrc32.getValue();

    // zlib's approach - update in parts using previous CRC value
    try (HostMemoryBuffer buffer1 = HostMemoryBuffer.allocate(bytes1.length);
         HostMemoryBuffer buffer2 = HostMemoryBuffer.allocate(bytes2.length)) {
      buffer1.setBytes(0, bytes1, 0, part1Size);
      buffer2.setBytes(0, bytes2, 0, part2Size);

      // First update: compute CRC32 of part1
      long actual = Hash.hostCrc32(0, buffer1);

      // Second update: continue from previous CRC value
      actual = Hash.hostCrc32(actual, buffer2);

      assertEquals(expected, actual,
        "CRC32 mismatch for incremental update");
    }

    // Also verify that single computation matches
    byte[] combinedBytes = new byte[part1Size + part2Size];
    System.arraycopy(bytes1, 0, combinedBytes, 0, part1Size);
    System.arraycopy(bytes2, 0, combinedBytes, part1Size, part2Size);
    try (HostMemoryBuffer buffer = HostMemoryBuffer.allocate(combinedBytes.length)) {
      buffer.setBytes(0, combinedBytes, 0, combinedBytes.length);
      long actual = Hash.hostCrc32(0, buffer);

      assertEquals(expected, actual,
        "CRC32 mismatch for single computation vs incremental");
    }
  }
}
