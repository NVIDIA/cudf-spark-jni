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

import ai.rapids.cudf.HostMemoryBuffer;
import org.apache.parquet.format.ColumnChunk;
import org.apache.parquet.format.ColumnMetaData;
import org.apache.parquet.format.CompressionCodec;
import org.apache.parquet.format.Encoding;
import org.apache.parquet.format.FieldRepetitionType;
import org.apache.parquet.format.RowGroup;
import org.apache.parquet.format.SchemaElement;
import org.apache.parquet.format.Type;
import org.apache.parquet.format.Util;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class ParquetFooterTest {

  private static final String COL_NAME = "a";

  // ---- helpers ----

  /**
   * Build a single-column RowGroup with the given row count, data page offset,
   * and compressed size.
   */
  private static RowGroup makeRowGroup(long numRows, long dataPageOffset, long compressedSize) {
    ColumnMetaData cmd = new ColumnMetaData(
        Type.INT32,
        Collections.singletonList(Encoding.PLAIN),
        Collections.singletonList(COL_NAME),
        CompressionCodec.UNCOMPRESSED,
        numRows,          // num_values
        compressedSize * 2,   // total_uncompressed_size. unused currently.
        compressedSize,   // total_compressed_size
        dataPageOffset);  // data_page_offset
    ColumnChunk cc = new ColumnChunk(dataPageOffset);
    cc.setMeta_data(cmd);
    RowGroup rg = new RowGroup(Collections.singletonList(cc), compressedSize, numRows);
    rg.setTotal_compressed_size(compressedSize);
    return rg;
  }

  /**
   * Build a minimal FileMetaData with a single INT32 column "a" and the given
   * row groups.
   */
  private static org.apache.parquet.format.FileMetaData makeFooter(RowGroup... rowGroups) {
    SchemaElement root = new SchemaElement("schema");
    root.setNum_children(1);
    SchemaElement col = new SchemaElement(COL_NAME);
    col.setType(Type.INT32);
    col.setRepetition_type(FieldRepetitionType.OPTIONAL);
    long totalRows = 0;
    for (RowGroup rg : rowGroups) {
      totalRows += rg.getNum_rows();
    }
    return new org.apache.parquet.format.FileMetaData(
        1,
        Arrays.asList(root, col),
        totalRows,
        Arrays.asList(rowGroups));
  }

  /**
   * Serialize a FileMetaData to thrift compact protocol bytes.
   */
  private static byte[] serialize(org.apache.parquet.format.FileMetaData meta) throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    Util.writeFileMetaData(meta, baos);
    return baos.toByteArray();
  }

  /**
   * Schema for readAndFilter that matches the single "a" column in makeFooter.
   */
  private static ParquetFooter.StructElement makeReadSchema() {
    return ParquetFooter.StructElement.builder()
        .addChild(COL_NAME, new ParquetFooter.ValueElement())
        .build();
  }

  /**
   * Deserialize footer bytes with byte-range filtering.
   */
  private static ParquetFooter readFooter(byte[] footerBytes, long partOffset, long partLength)
      throws Exception {
    try (HostMemoryBuffer buffer = HostMemoryBuffer.allocate(footerBytes.length)) {
      buffer.setBytes(0, footerBytes, 0, footerBytes.length);
      return ParquetFooter.readAndFilter(buffer, partOffset, partLength, makeReadSchema(), false);
    }
  }

  // ---- shared test data ----

  //  The `filter_groups` function (NativeParquetJni.cpp) includes a row group when its midpoint
  //  falls within [partOffset, partOffset + partLength).
  //  midpoint = data_page_offset + compressed_size / 2
  //
  //  Three row groups layout:
  //   RG0: 1000 rows, data_page_offset=100, compressed_size=200  → midpoint=200
  //   RG1: 2000 rows, data_page_offset=400, compressed_size=200  → midpoint=500
  //   RG2:  500 rows, data_page_offset=700, compressed_size=200  → midpoint=800
  //
  //  Cumulative row index offsets: RG0=0, RG1=1000, RG2=3000

  private static org.apache.parquet.format.FileMetaData threeRowGroupFooter() {
    return makeFooter(
        makeRowGroup(1000, 100, 200),
        makeRowGroup(2000, 400, 200),
        makeRowGroup(500,  700, 200));
  }

  // ---- tests ----

  @Test
  void testRowIndexOffsetsNoFiltering() throws Exception {
    byte[] bytes = serialize(threeRowGroupFooter());
    try (ParquetFooter footer = readFooter(bytes, 0, -1)) {
      assertArrayEquals(new long[]{0, 1000, 3000}, footer.getRowIndexOffsets());
      assertEquals(3500, footer.getNumRows());
    }
  }

  @Test
  void testRowIndexOffsetsSelectFirstRowGroup() throws Exception {
    byte[] bytes = serialize(threeRowGroupFooter());
    // midpoint 200 is in [0, 300)
    try (ParquetFooter footer = readFooter(bytes, 0, 300)) {
      assertArrayEquals(new long[]{0}, footer.getRowIndexOffsets());
      assertEquals(1000, footer.getNumRows());
    }
  }

  @Test
  void testRowIndexOffsetsSelectMiddleRowGroup() throws Exception {
    byte[] bytes = serialize(threeRowGroupFooter());
    // midpoint 500 is in [300, 600)
    try (ParquetFooter footer = readFooter(bytes, 300, 300)) {
      assertArrayEquals(new long[]{1000}, footer.getRowIndexOffsets());
      assertEquals(2000, footer.getNumRows());
    }
  }

  @Test
  void testRowIndexOffsetsSelectLastRowGroup() throws Exception {
    byte[] bytes = serialize(threeRowGroupFooter());
    // midpoint 800 is in [600, 900)
    try (ParquetFooter footer = readFooter(bytes, 600, 300)) {
      assertArrayEquals(new long[]{3000}, footer.getRowIndexOffsets());
      assertEquals(500, footer.getNumRows());
    }
  }

  @Test
  void testRowIndexOffsetsSelectFirstTwoRowGroups() throws Exception {
    byte[] bytes = serialize(threeRowGroupFooter());
    // midpoints 200 and 500 are in [0, 600)
    try (ParquetFooter footer = readFooter(bytes, 0, 600)) {
      assertArrayEquals(new long[]{0, 1000}, footer.getRowIndexOffsets());
      assertEquals(3000, footer.getNumRows());
    }
  }

  @Test
  void testRowIndexOffsetsSelectAllByByteRange() throws Exception {
    byte[] bytes = serialize(threeRowGroupFooter());
    // midpoints 200, 500, 800 are all in [0, 1000)
    try (ParquetFooter footer = readFooter(bytes, 0, 1000)) {
      assertArrayEquals(new long[]{0, 1000, 3000}, footer.getRowIndexOffsets());
      assertEquals(3500, footer.getNumRows());
    }
  }

  @Test
  void testRowIndexOffsetsNoRowGroupsSurvive() throws Exception {
    byte[] bytes = serialize(threeRowGroupFooter());
    // midpoints are 200, 500, 800 — none in [900, 1000)
    try (ParquetFooter footer = readFooter(bytes, 900, 100)) {
      assertArrayEquals(new long[]{}, footer.getRowIndexOffsets());
      assertEquals(0, footer.getNumRows());
    }
  }

  @Test
  void testRowIndexOffsetsSingleRowGroup() throws Exception {
    org.apache.parquet.format.FileMetaData meta = makeFooter(makeRowGroup(5000, 100, 200));
    byte[] bytes = serialize(meta);
    try (ParquetFooter footer = readFooter(bytes, 0, -1)) {
      assertArrayEquals(new long[]{0}, footer.getRowIndexOffsets());
      assertEquals(5000, footer.getNumRows());
    }
  }

  @Test
  void testRowIndexOffsetsSingleRowGroupByteRangeFiltered() throws Exception {
    org.apache.parquet.format.FileMetaData meta = makeFooter(makeRowGroup(5000, 100, 200));
    byte[] bytes = serialize(meta);
    // midpoint 200 is in [0, 300)
    try (ParquetFooter footer = readFooter(bytes, 0, 300)) {
      assertArrayEquals(new long[]{0}, footer.getRowIndexOffsets());
      assertEquals(5000, footer.getNumRows());
    }
  }


  @Test
  void testRowIndexOffsetsManyRowGroups() throws Exception {
    // 5 row groups with different sizes — verify cumulative offsets are correct
    //   RG0: 100 rows   → offset 0
    //   RG1: 200 rows   → offset 100
    //   RG2: 300 rows   → offset 300
    //   RG3: 400 rows   → offset 600
    //   RG4: 500 rows   → offset 1000
    org.apache.parquet.format.FileMetaData meta = makeFooter(
        makeRowGroup(100, 100,  200),
        makeRowGroup(200, 400,  200),
        makeRowGroup(300, 700,  200),
        makeRowGroup(400, 1000, 200),
        makeRowGroup(500, 1300, 200));
    byte[] bytes = serialize(meta);

    // No filtering
    try (ParquetFooter footer = readFooter(bytes, 0, -1)) {
      assertArrayEquals(new long[]{0, 100, 300, 600, 1000}, footer.getRowIndexOffsets());
      assertEquals(1500, footer.getNumRows());
    }

    // Select RG2 only: midpoint = 700 + 100 = 800, in [600, 900)
    try (ParquetFooter footer = readFooter(bytes, 600, 300)) {
      assertArrayEquals(new long[]{300}, footer.getRowIndexOffsets());
      assertEquals(300, footer.getNumRows());
    }

    // Select RG1, RG2, RG3: midpoints 500, 800, 1100 are all in [400, 1200)
    try (ParquetFooter footer = readFooter(bytes, 400, 800)) {
      assertArrayEquals(new long[]{100, 300, 600}, footer.getRowIndexOffsets());
      assertEquals(900, footer.getNumRows());
    }
  }
}
