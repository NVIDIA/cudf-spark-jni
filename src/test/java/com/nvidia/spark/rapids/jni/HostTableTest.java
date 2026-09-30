/*
 *  Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.HostColumnVector.BasicType;
import ai.rapids.cudf.HostColumnVector.DataType;
import ai.rapids.cudf.HostColumnVector.ListType;
import ai.rapids.cudf.HostColumnVector.StructData;
import ai.rapids.cudf.HostColumnVector.StructType;
import org.junit.jupiter.api.Test;

public class HostTableTest {
  @Test
  public void testRoundTripSync() {
    try (Table expected = buildTable()) {
      try (HostTable ht = HostTable.fromTable(expected, Cuda.DEFAULT_STREAM)) {
        try (Table actual = ht.toTable(Cuda.DEFAULT_STREAM)) {
          AssertUtils.assertTablesAreEqual(expected, actual);
        }
      }
    }
  }

  @Test
  public void testRoundTripSyncDefault() {
    try (Table expected = buildTable()) {
      try (HostTable ht = HostTable.fromTable(expected)) {
        try (Table actual = ht.toTable()) {
          AssertUtils.assertTablesAreEqual(expected, actual);
        }
      }
    }
  }

  @Test
  public void testRoundTripAsync() {
    testRoundTripAsync(buildTable());
  }

  @Test
  public void testRoundTripAsyncEmpty() {
    testRoundTripAsync(buildEmptyTable());
  }

  private void testRoundTripAsync(Table expected) {
    try (Table t = expected) {
      try (HostTable ht = HostTable.fromTableAsync(t, Cuda.DEFAULT_STREAM)) {
        try (Table actual = ht.toTableAsync(Cuda.DEFAULT_STREAM)) {
          AssertUtils.assertTablesAreEqual(expected, actual);
        }
      }
    }
  }

  private Table buildEmptyTable() {
    DataType listStringsType = new ListType(true, new BasicType(true, DType.STRING));
    DataType mapType = new ListType(true,
        new StructType(true,
            new BasicType(false, DType.STRING),
            new BasicType(false, DType.STRING)));
    DataType structType = new StructType(true,
        new BasicType(true, DType.INT8),
        new BasicType(false, DType.FLOAT32));
    try (ColumnVector emptyInt = ColumnVector.fromInts();
         ColumnVector emptyDouble = ColumnVector.fromDoubles();
         ColumnVector emptyString = ColumnVector.fromStrings();
         ColumnVector emptyListString = ColumnVector.fromLists(listStringsType);
         ColumnVector emptyMap = ColumnVector.fromLists(mapType);
         ColumnVector emptyStruct = ColumnVector.fromStructs(structType)) {
      return new Table(emptyInt, emptyInt, emptyDouble, emptyString,
          emptyListString, emptyMap, emptyStruct);
    }
  }

  private Table buildTable() {
    StructType mapStructType = new StructType(true,
        new BasicType(false, DType.STRING),
        new BasicType(false, DType.STRING));
    StructType structType = new StructType(true,
        new BasicType(true, DType.INT32),
        new BasicType(false, DType.FLOAT32));
    return new Table.TestBuilder()
        .column(     100,      202,      3003,    40004,        5,      -60,    1, null,    3,  null,     5, null,    7, null,   9,   null,    11, null,   13, null,  15)
        .column(    true,     true,     false,    false,     true,     null, true, true, null, false, false, null, true, true, null, false, false, null, true, true, null)
        .column( (byte)1,  (byte)2,      null,  (byte)4,  (byte)5,  (byte)6, (byte)1, (byte)2, (byte)3, null, (byte)5, (byte)6, (byte)7, null, (byte)9, (byte)10, (byte)11, null, (byte)13, (byte)14, (byte)15)
        .column((short)6, (short)5,  (short)4,     null, (short)2, (short)1, (short)1, (short)2, (short)3, null, (short)5, (short)6, (short)7, null, (short)9, (short)10, null, (short)12, (short)13, (short)14, null)
        .column(      1L,     null,     1001L,      50L,   -2000L,     null, 1L, 2L, 3L, 4L, null, 6L, 7L, 8L, 9L, null, 11L, 12L, 13L, 14L, null)
        .column(   10.1f,      20f, Float.NaN,  3.1415f,     -60f,     null, 1f, 2f, 3f, 4f, 5f, null, 7f, 8f, 9f, 10f, 11f, null, 13f, 14f, 15f)
        .column(   10.1f,      20f, Float.NaN,  3.1415f,     -60f,     -50f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f)
        .column(    10.1,     20.0,      33.1,   3.1415,    -60.5,     null, 1., 2., 3., 4., 5., 6., null, 8., 9., 10., 11., 12., null, 14., 15.)
        .timestampDayColumn(99,      100,      101,      102,      103,      104, 1, 2, 3, 4, 5, 6, 7, null, 9, 10, 11, 12, 13, null, 15)
        .timestampMillisecondsColumn(9L,    1006L,     101L,    5092L,     null,      88L, 1L, 2L, 3L, 4L, 5L ,6L, 7L, 8L, null, 10L, 11L, 12L, 13L, 14L, 15L)
        .timestampSecondsColumn(1L, null, 3L, 4L, 5L, 6L, 1L, 2L, 3L, 4L, 5L ,6L, 7L, 8L, 9L, null, 11L, 12L, 13L, 14L, 15L)
        .decimal32Column(-3, 100,      202,      3003,    40004,        5,      -60,    1, null,    3,  null,     5, null,    7, null,   9,   null,    11, null,   13, null,  15)
        .decimal64Column(-8,      1L,     null,     1001L,      50L,   -2000L,     null, 1L, 2L, 3L, 4L, null, 6L, 7L, 8L, 9L, null, 11L, 12L, 13L, 14L, null)
        .column(     "A",      "B",      "C",      "D",     null,   "TESTING", "1", "2", "3", "4", "5", "6", "7", null, "9", "10", "11", "12", "13", null, "15")
        .column(
            strings("1", "2", "3"), strings("4"), strings("5"), strings("6, 7"),
            strings("", "9", null), strings("11"), strings(""), strings(null, null),
            strings("15", null), null, null, strings("18", "19", "20"),
            null, strings("22"), strings("23", ""), null,
            null, null, null, strings(),
            strings("the end"))
        .column(mapStructType,
            structs(struct("1", "2")), structs(struct("3", "4")),
            null, null,
            structs(struct("key", "value"), struct("a", "b")), null,
            null, structs(struct("3", "4"), struct("1", "2")),
            structs(), structs(null, struct("foo", "bar")),
            structs(null, null, null), null,
            null, null,
            null, null,
            null, null,
            null, null,
            structs(struct("the", "end")))
        .column(structType,
            struct(1, 1f), null, struct(2, 3f), null, struct(8, 7f),
            struct(0, 0f), null, null, struct(-1, -1f), struct(-100, -100f),
            struct(Integer.MAX_VALUE, Float.MAX_VALUE), null, null, null, null,
            null, null, null, null, null,
            struct(Integer.MIN_VALUE, Float.MIN_VALUE))
        .column(     "A",      "A",      "C",      "C",     null,   "TESTING", "1", "2", "3", "4", "5", "6", "7", null, "9", "10", "11", "12", "13", null, "15")
        .build();
  }

  private static StructData struct(Object... values) {
    return new StructData(values);
  }

  private static StructData[] structs(StructData... values) {
    return values;
  }

  private static String[] strings(String... values) {
    return values;
  }
}
