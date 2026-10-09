/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.JSONOptions;
import ai.rapids.cudf.Schema;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class FromJsonToStructsTest {
  private static final int DECIMAL_SCALE = -5;
  private static final int DECIMAL_PRECISION = 10;
  private static final DType DECIMAL_TYPE =
      DType.create(DType.DTypeEnum.DECIMAL64, DECIMAL_SCALE);

  private static final class DecimalTestData {
    private final String[] decimalStrings;
    private final String[] jsonStrings;
    private final Long[] expectedUnscaledValues;

    private DecimalTestData(String[] decimalStrings, String[] jsonStrings,
                            Long[] expectedUnscaledValues) {
      this.decimalStrings = decimalStrings;
      this.jsonStrings = jsonStrings;
      this.expectedUnscaledValues = expectedUnscaledValues;
    }
  }

  private static JSONOptions getOptions() {
    return JSONOptions.builder()
        .withNormalizeSingleQuotes(true)
        .withLeadingZeros(true)
        .withNonNumericNumbers(true)
        .withUnquotedControlChars(false)
        .build();
  }

  private static Schema mixedNestedTypesSchema() {
    Schema.Builder root = Schema.builder();
    Schema.Builder data = root.addColumn(DType.STRUCT, "data");
    data.column(DType.INT32, "c1");
    Schema.Builder c2 = data.addColumn(DType.LIST, "c2");
    Schema.Builder element = c2.addColumn(DType.STRUCT, "element");
    element.column(DType.INT32, "c3");
    element.column(DType.STRING, "c4");
    root.column(DType.INT32, "id");
    return root.build();
  }

  private static HostColumnVector.StructData nestedRow(int value) {
    return new HostColumnVector.StructData(
        new HostColumnVector.StructData(
            value,
            Collections.singletonList(new HostColumnVector.StructData(value, "x"))),
        value);
  }

  private static Schema decimalSchema() {
    return Schema.builder().column(DECIMAL_TYPE, "data", DECIMAL_PRECISION).build();
  }

  private static DecimalTestData unicodeDecimalTestData() {
    List<String> decimalStrings = new ArrayList<>();
    List<Long> expectedUnscaledValues = new ArrayList<>();

    // BigDecimal accepts every BMP character for which Character.isDigit(char) is true.
    for (int codePoint = Character.MIN_VALUE; codePoint <= Character.MAX_VALUE; ++codePoint) {
      char character = (char) codePoint;
      if (Character.isDigit(character)) {
        decimalStrings.add("\"" + character + "\"");
        expectedUnscaledValues.add(Character.digit(character, 10) * 100000L);
      }
    }

    // These digit blocks were added after Java 8's Unicode version. On an older JVM they must
    // remain invalid even if the native cuDF Unicode table recognizes them.
    for (char character : new char[] {'\u0DE6', '\uA9F0'}) {
      decimalStrings.add("\"" + character + "\"");
      expectedUnscaledValues.add(Character.isDigit(character)
          ? Character.digit(character, 10) * 100000L : null);
    }

    String[] issueValues = {
        "\u0967,\u0966\u0966\u0966.\u0966\u0966\u0967",
        "\u0E51,\u0E50\u0E50\u0E50.\u0E50\u0E50\u0E51",
        "\u0967", "\u0E51", "\u0967\u0966", "\u0E51\u0E50", "1\u0662\u096D"
    };
    Long[] issueExpected = {
        100000100L, 100000100L, 100000L, 100000L, 1000000L, 1000000L, 12700000L
    };
    for (int index = 0; index < issueValues.length; ++index) {
      decimalStrings.add("\"" + issueValues[index] + "\"");
      expectedUnscaledValues.add(issueExpected[index]);
    }

    // BigDecimal evaluates UTF-16 code units, so supplementary decimal digits remain invalid.
    decimalStrings.add("\"\uD835\uDFCF\"");
    expectedUnscaledValues.add(null);
    // Numeric characters outside the Unicode Decimal_Number category also remain invalid.
    decimalStrings.add("\"\u00B2\"");
    expectedUnscaledValues.add(null);

    String[] jsonStrings = new String[decimalStrings.size()];
    for (int index = 0; index < jsonStrings.length; ++index) {
      jsonStrings[index] = "{\"data\":" + decimalStrings.get(index) + "}";
    }
    return new DecimalTestData(
        decimalStrings.toArray(new String[0]),
        jsonStrings,
        expectedUnscaledValues.toArray(new Long[0]));
  }

  @Test
  void testConvertFromStringsNormalizesUnicodeDecimalDigits() {
    DecimalTestData data = unicodeDecimalTestData();
    try (ColumnVector input = ColumnVector.fromStrings(data.decimalStrings);
         ColumnVector actual =
             JSONUtils.convertFromStrings(input, decimalSchema(), false, true);
         ColumnVector expected =
             ColumnVector.decimalFromBoxedLongs(DECIMAL_SCALE, data.expectedUnscaledValues)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testConvertFromStringsNormalizesUnquotedDigitsWithoutRemovingCommas() {
    try (ColumnVector input =
             ColumnVector.fromStrings("\u0967", "1\u0662\u096D", "\u0967,\u0966", "", (String) null);
         ColumnVector actual =
             JSONUtils.convertFromStrings(input, decimalSchema(), false, true);
         ColumnVector expected = ColumnVector.decimalFromBoxedLongs(
             DECIMAL_SCALE, 100000L, 12700000L, null, null, null)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testFromJsonToStructsNormalizesUnicodeDecimalDigits() {
    DecimalTestData data = unicodeDecimalTestData();
    try (ColumnVector input = ColumnVector.fromStrings(data.jsonStrings);
         ColumnVector actual =
             JSONUtils.fromJSONToStructs(input, decimalSchema(), getOptions(), true);
         ColumnView actualDecimals = actual.getChildColumnView(0);
         ColumnVector expected =
             ColumnVector.decimalFromBoxedLongs(DECIMAL_SCALE, data.expectedUnscaledValues)) {
      assertColumnsAreEqual(expected, actualDecimals);
    }
  }

  @Test
  void testFromJsonToStructsNormalizesNestedDecimalDigits() {
    Schema.Builder root = Schema.builder();
    root.addColumn(DType.STRUCT, "nested")
        .column(DECIMAL_TYPE, "amount", DECIMAL_PRECISION);
    Long laterDigit = Character.isDigit('\u0DE6') ? 0L : null;
    try (ColumnVector input = ColumnVector.fromStrings(
             "{\"nested\":{\"amount\":\"\u0967\"}}",
             "{\"nested\":{\"amount\":\"\u0DE6\"}}",
             "{\"nested\":{\"amount\":\"2\"}}");
         ColumnVector actual = JSONUtils.fromJSONToStructs(input, root.build(), getOptions(), true);
         ColumnView nested = actual.getChildColumnView(0);
         ColumnView decimals = nested.getChildColumnView(0);
         ColumnVector expected = ColumnVector.decimalFromBoxedLongs(
             DECIMAL_SCALE, 100000L, laterDigit, 200000L)) {
      assertColumnsAreEqual(expected, decimals);
    }
  }

  @Test
  void testEmbeddedNulIsNotUsedAsRowDelimiter() {
    String malformedBanner =
        "\uFFFD\uFFFD[\u0000\"\u0000\uFFFD\u0000\"\u0000]\u0000";
    String json = "{\"aniviaData\":{\"asset\":{\"assetId\":\"va\"}," +
        "\"bannerDetails\":" + malformedBanner + "}}";

    Schema.Builder root = Schema.builder();
    Schema.Builder aniviaData = root.addColumn(DType.STRUCT, "aniviaData");
    Schema.Builder asset = aniviaData.addColumn(DType.STRUCT, "asset");
    asset.addColumn(DType.STRING, "assetId");
    Schema schema = root.build();

    try (ColumnVector input = ColumnVector.fromStrings(json);
         ColumnVector output = JSONUtils.fromJSONToStructs(
             input, schema, getOptions(), true);
         ColumnView outputAniviaData = output.getChildColumnView(0);
         ColumnView outputAsset = outputAniviaData.getChildColumnView(0)) {
      try (ColumnView outputAssetId = outputAsset.getChildColumnView(0);
           HostColumnVector hostAssetId = outputAssetId.copyToHost()) {
        assertTrue(hostAssetId.isNull(0), "malformed record should nullify assetId");
      }
      assertEquals(input.getRowCount(), output.getRowCount());
      assertEquals(input.getRowCount(), outputAniviaData.getRowCount());
      assertEquals(input.getRowCount(), outputAsset.getRowCount());
    }
  }

  @Test
  void testFromJsonToStructsNullsOnlyMismatchedRowsForDepthOneParent() {
    String valid = "{\"data\":{\"c2\":[{\"c3\":19,\"c4\":\"x\"}],\"c1\":1},\"id\":10}";
    String preExistingNull = "{\"data\":null,\"id\":15}";
    String firstMismatch = "{\"data\":{\"c2\":[19],\"c1\":2},\"id\":20}";
    String secondMismatch = "{\"data\":{\"c2\":[29],\"c1\":3},\"id\":25}";
    String validAfterMismatch =
        "{\"data\":{\"c2\":[{\"c3\":39,\"c4\":\"z\"}],\"c1\":4},\"id\":30}";
    Schema schema = mixedNestedTypesSchema();

    try (ColumnVector input = ColumnVector.fromStrings(
             valid, preExistingNull, firstMismatch, secondMismatch, validAfterMismatch);
         ColumnVector actual = JSONUtils.fromJSONToStructs(input, schema, getOptions(), true);
         ColumnVector expected = ColumnVector.fromStructs(schema.asHostDataType(),
             new HostColumnVector.StructData(
                 new HostColumnVector.StructData(
                     1,
                     Collections.singletonList(new HostColumnVector.StructData(19, "x"))),
                 10),
             new HostColumnVector.StructData(Arrays.asList(null, 15)),
             new HostColumnVector.StructData(Arrays.asList(null, 20)),
             new HostColumnVector.StructData(Arrays.asList(null, 25)),
             new HostColumnVector.StructData(
                 new HostColumnVector.StructData(
                     4,
                     Collections.singletonList(new HostColumnVector.StructData(39, "z"))),
                 30));
         ColumnView data = actual.getChildColumnView(0);
         ColumnView c2 = data.getChildColumnView(1)) {
      assertColumnsAreEqual(expected, actual);
      assertFalse(c2.hasNonEmptyNulls(), "mismatched row must have an empty null LIST");
    }
  }

  @Test
  void testFromJsonToStructsHandlesEmptyAndNullInputs() {
    Schema schema = mixedNestedTypesSchema();

    try (ColumnVector emptyInput = ColumnVector.fromStrings();
         ColumnVector emptyOutput =
             JSONUtils.fromJSONToStructs(emptyInput, schema, getOptions(), true);
         ColumnVector expectedEmpty = ColumnVector.fromStructs(schema.asHostDataType())) {
      assertColumnsAreEqual(expectedEmpty, emptyOutput);
    }

    try (ColumnVector nullInput = ColumnVector.fromStrings((String) null);
         ColumnVector nullOutput =
             JSONUtils.fromJSONToStructs(nullInput, schema, getOptions(), true)) {
      assertEquals(1, nullOutput.getRowCount());
      assertEquals(1, nullOutput.getNullCount());
    }
  }

  @Test
  void testFromJsonToStructsAssociatesMismatchRowsByColumnName() {
    Schema.Builder root = Schema.builder();
    root.addColumn(DType.STRUCT, "a").column(DType.INT32, "value");
    root.addColumn(DType.STRUCT, "b").column(DType.INT32, "value");
    Schema schema = root.build();

    try (ColumnVector input = ColumnVector.fromStrings(
             "{\"a\":1,\"b\":{\"value\":10}}",
             "{\"a\":{\"value\":20},\"b\":2}");
         ColumnVector actual = JSONUtils.fromJSONToStructs(input, schema, getOptions(), true);
         ColumnVector expected = ColumnVector.fromStructs(schema.asHostDataType(),
             new HostColumnVector.StructData(
                 Arrays.asList(null, new HostColumnVector.StructData(10))),
             new HostColumnVector.StructData(
                 Arrays.asList(new HostColumnVector.StructData(20), null)))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testFromJsonToStructsGroupedMaskUpdatesAcrossWordBoundaries() {
    int[] mismatchRows = {0, 1, 2, 31, 32, 63, 64};
    String[] inputRows = new String[65];
    HostColumnVector.StructData[] expectedRows = new HostColumnVector.StructData[65];
    for (int row = 0; row < inputRows.length; ++row) {
      boolean mismatched = Arrays.binarySearch(mismatchRows, row) >= 0;
      inputRows[row] = mismatched
          ? String.format("{\"data\":{\"c2\":[%d],\"c1\":%d},\"id\":%d}", row, row, row)
          : String.format(
              "{\"data\":{\"c2\":[{\"c3\":%d,\"c4\":\"x\"}],\"c1\":%d},\"id\":%d}",
              row, row, row);
      expectedRows[row] = mismatched
          ? new HostColumnVector.StructData(Arrays.asList(null, row))
          : nestedRow(row);
    }
    Schema schema = mixedNestedTypesSchema();

    try (ColumnVector input = ColumnVector.fromStrings(inputRows);
         ColumnVector actual = JSONUtils.fromJSONToStructs(input, schema, getOptions(), true);
         ColumnVector expected = ColumnVector.fromStructs(schema.asHostDataType(), expectedRows)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testFromJsonToStructsSanitizesTopLevelListMismatch() {
    Schema.Builder root = Schema.builder();
    Schema.Builder items = root.addColumn(DType.LIST, "items");
    items.addColumn(DType.STRUCT, "element").column(DType.INT32, "value");
    Schema schema = root.build();

    try (ColumnVector input = ColumnVector.fromStrings(
             "{\"items\":[{\"value\":1}]}",
             "{\"items\":[2]}",
             "{\"items\":[{\"value\":3}]}");
         ColumnVector actual = JSONUtils.fromJSONToStructs(input, schema, getOptions(), true);
         ColumnVector expected = ColumnVector.fromStructs(schema.asHostDataType(),
             new HostColumnVector.StructData(
                 (Object) Collections.singletonList(new HostColumnVector.StructData(1))),
             new HostColumnVector.StructData((Object) null),
             new HostColumnVector.StructData(
                 (Object) Collections.singletonList(new HostColumnVector.StructData(3))));
         ColumnView actualItems = actual.getChildColumnView(0)) {
      assertColumnsAreEqual(expected, actual);
      assertFalse(actualItems.hasNonEmptyNulls(), "mismatched row must have an empty null LIST");
    }
  }
}
