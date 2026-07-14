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

import java.util.Arrays;
import java.util.Collections;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class FromJsonToStructsTest {
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
}
