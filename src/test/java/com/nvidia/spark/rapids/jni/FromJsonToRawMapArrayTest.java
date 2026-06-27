/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
import ai.rapids.cudf.HostColumnVectorCore;
import ai.rapids.cudf.JSONOptions;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Tests for the array-valued surface of {@link JSONUtils#extractRawMapFromJsonString}:
 *  - the three-argument overload with {@link JSONUtils.MapValueType#ARRAY_OF_STRING}, which produces
 *    a {@code LIST<STRUCT<STRING, LIST<STRING>>>} column;
 *  - the three-argument overload with {@link JSONUtils.MapValueType#STRING}, which must agree with
 *    the deprecated two-argument overload it delegates to;
 *  - the unmapped-{@code valueType} guard of the dispatch switch.
 */
public class FromJsonToRawMapArrayTest {
  // Column schema for the array-valued result: LIST<STRUCT<key:STRING, value:LIST<STRING>>>. The
  // struct and its key child are non-nullable (the kernel builds them with no null mask); only the
  // value list (null inner list) and its elements (null element) carry nulls.
  private static final HostColumnVector.StructType STRUCT_TYPE =
      new HostColumnVector.StructType(false,
          new HostColumnVector.BasicType(false, DType.STRING),                             // key
          new HostColumnVector.ListType(true, new HostColumnVector.BasicType(true,
              DType.STRING)));                                                             // value
  private static final HostColumnVector.ListType LIST_TYPE =
      new HostColumnVector.ListType(true, STRUCT_TYPE);

  // Same leniency flags as FromJsonToRawMapTest so both suites parse identically.
  private static JSONOptions getOptions() {
    return JSONOptions.builder()
        .withNormalizeSingleQuotes(true)
        .withLeadingZeros(true)
        .withNonNumericNumbers(true)
        .withUnquotedControlChars(true)
        .build();
  }

  private static HostColumnVector.StructData pair(String key, List<String> values) {
    return new HostColumnVector.StructData(key, values);
  }

  // The array-valued result is LIST<STRUCT<key, value>>; the kernel builds the struct and its key
  // child with no null mask. Assert that invariant directly -- assertColumnsAreEqual runs with
  // nullability checking off, so it would not catch a stray key or struct validity mask.
  private static void assertKeyAndStructNonNullable(ColumnVector result) {
    try (HostColumnVector host = result.copyToHost()) {
      HostColumnVectorCore structChild = host.getChildColumnView(0);
      HostColumnVectorCore keyChild = structChild.getChildColumnView(0);
      assertFalse(structChild.hasValidityVector(), "struct child must be non-nullable");
      assertFalse(keyChild.hasValidityVector(), "map key column must be non-nullable");
    }
  }

  // ARRAY_OF_STRING: a single object whose one key maps to a two-element string array. The engine
  // copies the elements verbatim with their surrounding quotes stripped, so {"k":["a","b"]} yields
  // one non-null row holding the pair (k -> ["a", "b"]).
  @Test
  void testExtractRawMapArrayFromJsonString() {
    List<HostColumnVector.StructData> row0 =
        Arrays.asList(pair("k", Arrays.asList("a", "b")));
    try (ColumnVector input = ColumnVector.fromStrings("{\"k\":[\"a\",\"b\"]}");
         ColumnVector result = JSONUtils.extractRawMapFromJsonString(input, getOptions(),
             JSONUtils.MapValueType.ARRAY_OF_STRING);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, row0)) {
      assertColumnsAreEqual(expected, result);
      assertKeyAndStructNonNullable(result);
    }
  }

  // ARRAY_OF_STRING nested validity through the public HostColumnVector path:
  //  - {"k":null}       -> row kept, the value inner list is null (mask #1);
  //  - {"k":[null,"x"]} -> inner list present with a null first element (mask #2).
  // Pins that nested null lists/elements survive the JNI -> HostColumnVector round-trip.
  @Test
  void testExtractRawMapArrayNullListAndElement() {
    List<HostColumnVector.StructData> row0 = Arrays.asList(pair("k", null));
    List<HostColumnVector.StructData> row1 = Arrays.asList(pair("k", Arrays.asList(null, "x")));
    try (ColumnVector input =
             ColumnVector.fromStrings("{\"k\":null}", "{\"k\":[null,\"x\"]}");
         ColumnVector result = JSONUtils.extractRawMapFromJsonString(input, getOptions(),
             JSONUtils.MapValueType.ARRAY_OF_STRING);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, row0, row1)) {
      assertColumnsAreEqual(expected, result);
      assertKeyAndStructNonNullable(result);
    }
  }

  // The three-argument overload with MapValueType.STRING must produce exactly what the deprecated
  // two-argument overload produces, since the latter delegates to the former with STRING.
  @Test
  void testExtractRawMapStringMatchesDeprecatedOverload() {
    try (ColumnVector input = ColumnVector.fromStrings("{\"k\":\"v\"}");
         ColumnVector deprecated = JSONUtils.extractRawMapFromJsonString(input, getOptions());
         ColumnVector typed = JSONUtils.extractRawMapFromJsonString(input, getOptions(),
             JSONUtils.MapValueType.STRING)) {
      assertColumnsAreEqual(deprecated, typed);
    }
  }

  // A null valueType is rejected: switching on it throws NullPointerException before any arm. The
  // MapValueType enum declares only STRING and ARRAY_OF_STRING, so the IllegalArgumentException
  // default arm is unreachable through the typed API; null is the only reachable invalid input, and
  // this test pins that it fails fast rather than being silently accepted.
  @Test
  void testExtractRawMapNullValueTypeRejected() {
    try (ColumnVector input = ColumnVector.fromStrings("{\"k\":\"v\"}")) {
      assertThrows(NullPointerException.class,
          () -> JSONUtils.extractRawMapFromJsonString(input, getOptions(), null));
    }
  }
}
