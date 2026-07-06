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
import ai.rapids.cudf.JSONOptions;
import ai.rapids.cudf.Schema;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class FromJsonToStructsTest {
  private static JSONOptions getOptions() {
    return JSONOptions.builder()
        .withNormalizeSingleQuotes(true)
        .withLeadingZeros(true)
        .withNonNumericNumbers(true)
        .withUnquotedControlChars(false)
        .build();
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
      assertEquals(input.getRowCount(), output.getRowCount());
      assertEquals(input.getRowCount(), outputAniviaData.getRowCount());
      assertEquals(input.getRowCount(), outputAsset.getRowCount());
    }
  }
}
