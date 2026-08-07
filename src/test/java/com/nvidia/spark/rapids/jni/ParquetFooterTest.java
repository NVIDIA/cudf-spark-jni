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

import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.HostMemoryBuffer;
import org.apache.parquet.format.ColumnChunk;
import org.apache.parquet.format.ColumnMetaData;
import org.apache.parquet.format.ColumnOrder;
import org.apache.parquet.format.CompressionCodec;
import org.apache.parquet.format.ConvertedType;
import org.apache.parquet.format.Encoding;
import org.apache.parquet.format.FieldRepetitionType;
import org.apache.parquet.format.RowGroup;
import org.apache.parquet.format.SchemaElement;
import org.apache.parquet.format.Type;
import org.apache.parquet.format.TypeDefinedOrder;
import org.apache.parquet.format.Util;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

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

  /**
   * Build a leaf (primitive) SchemaElement. Leaves carry a physical type and no children.
   */
  private static SchemaElement leaf(String name, FieldRepetitionType rep) {
    SchemaElement e = new SchemaElement(name);
    e.setType(Type.INT32);
    e.setRepetition_type(rep);
    return e;
  }

  /**
   * Build a group (non-leaf) SchemaElement with the given child count. Groups carry no physical
   * type; a non-null converted type annotates the LIST / MAP wrapper groups.
   */
  private static SchemaElement group(String name, int numChildren, FieldRepetitionType rep,
      ConvertedType converted) {
    SchemaElement e = new SchemaElement(name);
    e.setNum_children(numChildren);
    e.setRepetition_type(rep);
    if (converted != null) {
      e.setConverted_type(converted);
    }
    return e;
  }

  /**
   * Build a footer with four top-level columns covering every nested shape the pruner handles:
   *   id     : INT32
   *   names  : LIST of INT32          (3-level: names -> list -> element)
   *   props  : MAP of INT32 to INT32  (props -> key_value -> {key, value})
   *   nested : STRUCT of x, y (both INT32)
   * The schema is the depth-first flattening parquet uses; the row group carries one column chunk
   * per leaf column (id, element, key, value, x, y).
   */
  private static org.apache.parquet.format.FileMetaData makeNestedFooter(long numRows) {
    SchemaElement root = new SchemaElement("schema");
    root.setNum_children(4);

    List<SchemaElement> schema = new ArrayList<>();
    schema.add(root);
    schema.add(leaf("id", FieldRepetitionType.OPTIONAL));
    schema.add(group("names", 1, FieldRepetitionType.OPTIONAL, ConvertedType.LIST));
    schema.add(group("list", 1, FieldRepetitionType.REPEATED, null));
    schema.add(leaf("element", FieldRepetitionType.OPTIONAL));
    schema.add(group("props", 1, FieldRepetitionType.OPTIONAL, ConvertedType.MAP));
    schema.add(group("key_value", 2, FieldRepetitionType.REPEATED, null));
    schema.add(leaf("key", FieldRepetitionType.REQUIRED));
    schema.add(leaf("value", FieldRepetitionType.OPTIONAL));
    schema.add(group("nested", 2, FieldRepetitionType.OPTIONAL, null));
    schema.add(leaf("x", FieldRepetitionType.OPTIONAL));
    schema.add(leaf("y", FieldRepetitionType.OPTIONAL));

    // One column chunk per leaf column (6 leaves), each with a distinct data page offset.
    List<ColumnChunk> chunks = new ArrayList<>();
    for (int i = 0; i < 6; i++) {
      ColumnMetaData cmd = new ColumnMetaData(
          Type.INT32,
          Collections.singletonList(Encoding.PLAIN),
          Collections.singletonList("c" + i),
          CompressionCodec.UNCOMPRESSED,
          numRows,
          200,
          100,
          100L + i);
      ColumnChunk cc = new ColumnChunk(100L + i);
      cc.setMeta_data(cmd);
      chunks.add(cc);
    }
    RowGroup rg = new RowGroup(chunks, 600, numRows);
    rg.setTotal_compressed_size(600);

    return new org.apache.parquet.format.FileMetaData(1, schema, numRows,
        Collections.singletonList(rg));
  }

  /**
   * Read schema selecting all four top-level columns of makeNestedFooter.
   */
  private static ParquetFooter.StructElement makeNestedReadSchema() {
    return ParquetFooter.StructElement.builder()
        .addChild("id", new ParquetFooter.ValueElement())
        .addChild("names", new ParquetFooter.ListElement(new ParquetFooter.ValueElement()))
        .addChild("props", new ParquetFooter.MapElement(
            new ParquetFooter.ValueElement(), new ParquetFooter.ValueElement()))
        .addChild("nested", ParquetFooter.StructElement.builder()
            .addChild("x", new ParquetFooter.ValueElement())
            .addChild("y", new ParquetFooter.ValueElement())
            .build())
        .build();
  }

  /**
   * Read schema selecting three of the four columns of makeNestedFooter (drops the MAP).
   */
  private static ParquetFooter.StructElement makePrunedReadSchema() {
    return ParquetFooter.StructElement.builder()
        .addChild("id", new ParquetFooter.ValueElement())
        .addChild("names", new ParquetFooter.ListElement(new ParquetFooter.ValueElement()))
        .addChild("nested", ParquetFooter.StructElement.builder()
            .addChild("x", new ParquetFooter.ValueElement())
            .addChild("y", new ParquetFooter.ValueElement())
            .build())
        .build();
  }

  /**
   * Deserialize footer bytes with byte-range filtering and an explicit read schema.
   */
  private static ParquetFooter readFooter(byte[] footerBytes, long partOffset, long partLength,
      ParquetFooter.StructElement schema) throws Exception {
    try (HostMemoryBuffer buffer = HostMemoryBuffer.allocate(footerBytes.length)) {
      buffer.setBytes(0, footerBytes, 0, footerBytes.length);
      return ParquetFooter.readAndFilter(buffer, partOffset, partLength, schema, false);
    }
  }

  /**
   * Build a single-column row group whose column chunk omits inline meta_data, forcing the reader
   * onto the RowGroup.file_offset path (PARQUET-2078). file_offset and total_compressed_size are
   * the cudf-optional fields the reader must read defensively.
   */
  private static RowGroup makeRowGroupNoMetadata(
      long numRows, long fileOffset, long compressedSize) {
    ColumnChunk cc = new ColumnChunk(fileOffset);   // meta_data intentionally left unset
    RowGroup rg = new RowGroup(Collections.singletonList(cc), compressedSize, numRows);
    rg.setFile_offset(fileOffset);
    rg.setTotal_compressed_size(compressedSize);
    return rg;
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

  @Test
  void testColumnPruningNestedSchemas() throws Exception {
    byte[] bytes = serialize(makeNestedFooter(1234));

    // Select all four top-level columns (INT32, LIST, MAP, STRUCT).
    try (ParquetFooter footer = readFooter(bytes, 0, -1, makeNestedReadSchema())) {
      assertEquals(4, footer.getNumColumns());
      assertEquals(1234, footer.getNumRows());
    }

    // Prune away the MAP column; three top-level columns survive.
    try (ParquetFooter footer = readFooter(bytes, 0, -1, makePrunedReadSchema())) {
      assertEquals(3, footer.getNumColumns());
      assertEquals(1234, footer.getNumRows());
    }
  }

  @Test
  void testSerializeThriftFileRoundTripThroughParquetMr() throws Exception {
    byte[] bytes = serialize(makeFooter(
        makeRowGroup(1000, 100, 200),
        makeRowGroup(2000, 400, 200)));
    try (ParquetFooter footer = readFooter(bytes, 0, -1);
         HostMemoryBuffer out = footer.serializeThriftFile()) {
      // Framed layout written by serializeThriftFile: "PAR1" + thrift + 4-byte LE length + "PAR1".
      int total = (int) out.getLength();
      byte[] framed = new byte[total];
      out.getBytes(framed, 0, 0, total);
      assertEquals('P', framed[0]);
      assertEquals('1', framed[3]);
      assertEquals('P', framed[total - 4]);
      assertEquals('1', framed[total - 1]);
      // Re-parse the embedded footer with parquet-mr -- the real cross-library contract.
      org.apache.parquet.format.FileMetaData reparsed =
          Util.readFileMetaData(new ByteArrayInputStream(framed, 4, total - 12));
      assertEquals(3000, reparsed.getNum_rows());
      assertEquals(2, reparsed.getRow_groups().size());
      assertEquals(1000, reparsed.getRow_groups().get(0).getNum_rows());
      assertEquals(2000, reparsed.getRow_groups().get(1).getNum_rows());
    }
  }

  @Test
  void testRepetitionTypeRoundTrip() throws Exception {
    byte[] bytes = serialize(makeFooter(makeRowGroup(1000, 100, 200)));
    try (ParquetFooter footer = readFooter(bytes, 0, -1);
         HostMemoryBuffer out = footer.serializeThriftFile()) {
      int total = (int) out.getLength();
      byte[] framed = new byte[total];
      out.getBytes(framed, 0, 0, total);
      org.apache.parquet.format.FileMetaData reparsed =
          Util.readFileMetaData(new ByteArrayInputStream(framed, 4, total - 12));
      // makeFooter's data column "a" is OPTIONAL INT32; both must survive the cudf read/write.
      SchemaElement col = reparsed.getSchema().get(1);
      assertEquals(COL_NAME, col.getName());
      assertEquals(FieldRepetitionType.OPTIONAL, col.getRepetition_type());
      assertEquals(Type.INT32, col.getType());
    }
  }

  @Test
  void testCorruptFooterThrowsCleanException() throws Exception {
    byte[] bytes = serialize(makeFooter(makeRowGroup(1000, 100, 200)));
    // A truncated thrift stream is unparseable; the reader must surface a clean Java exception
    // (CudfException) rather than crashing the JVM.
    byte[] truncated = Arrays.copyOf(bytes, bytes.length / 2);
    assertThrows(CudfException.class, () -> readFooter(truncated, 0, -1));
  }

  @Test
  void testMetadataAbsentColumnChunk() throws Exception {
    // Two row groups without inline column meta_data; offsets come from RowGroup.file_offset.
    //   RG0: file_offset=4,   compressed=100 -> start=4,   midpoint=54
    //   RG1: file_offset=104, compressed=100 -> start=104, midpoint=154
    org.apache.parquet.format.FileMetaData meta = makeFooter(
        makeRowGroupNoMetadata(1000, 4, 100),
        makeRowGroupNoMetadata(2000, 104, 100));
    byte[] bytes = serialize(meta);

    // Byte range [0, 1000) covers both midpoints, so both row groups survive.
    try (ParquetFooter footer = readFooter(bytes, 0, 1000)) {
      assertArrayEquals(new long[]{0, 1000}, footer.getRowIndexOffsets());
      assertEquals(3000, footer.getNumRows());

      // The reader fabricates default meta_data for the absent chunks; parquet-mr must accept it
      // when the footer is written back out.
      try (HostMemoryBuffer out = footer.serializeThriftFile()) {
        int total = (int) out.getLength();
        byte[] framed = new byte[total];
        out.getBytes(framed, 0, 0, total);
        org.apache.parquet.format.FileMetaData reparsed =
            Util.readFileMetaData(new ByteArrayInputStream(framed, 4, total - 12));
        assertEquals(3000, reparsed.getNum_rows());
        assertEquals(2, reparsed.getRow_groups().size());
      }
    }
  }

  @Test
  void testMapWithoutConvertedTypeThrowsCleanException() throws Exception {
    // A group shaped like a MAP (props -> repeated key_value -> {key, value}) but with no
    // converted_type annotation. The pruner's map handler must reject it with a clean CudfException
    // rather than crashing the JVM.
    SchemaElement root = new SchemaElement("schema");
    root.setNum_children(1);
    List<SchemaElement> schema = Arrays.asList(
        root,
        group("props", 1, FieldRepetitionType.OPTIONAL, null),   // converted_type intentionally unset
        group("key_value", 2, FieldRepetitionType.REPEATED, null),
        leaf("key", FieldRepetitionType.REQUIRED),
        leaf("value", FieldRepetitionType.OPTIONAL));
    List<ColumnChunk> chunks = new ArrayList<>();
    for (int i = 0; i < 2; i++) {   // one chunk per leaf (key, value)
      ColumnMetaData cmd = new ColumnMetaData(Type.INT32,
          Collections.singletonList(Encoding.PLAIN), Collections.singletonList("c" + i),
          CompressionCodec.UNCOMPRESSED, 10, 200, 100, 100L + i);
      ColumnChunk cc = new ColumnChunk(100L + i);
      cc.setMeta_data(cmd);
      chunks.add(cc);
    }
    RowGroup rg = new RowGroup(chunks, 200, 10);
    rg.setTotal_compressed_size(200);
    byte[] bytes = serialize(new org.apache.parquet.format.FileMetaData(
        1, schema, 10, Collections.singletonList(rg)));
    ParquetFooter.StructElement readSchema = ParquetFooter.StructElement.builder()
        .addChild("props", new ParquetFooter.MapElement(
            new ParquetFooter.ValueElement(), new ParquetFooter.ValueElement()))
        .build();
    assertThrows(CudfException.class, () -> readFooter(bytes, 0, -1, readSchema));
  }

  @Test
  void testEmptyFooterNoRowGroups() throws Exception {
    // Zero row groups exercises filter_groups' `num_row_groups > 0` guard: the byte-range filter
    // must produce an empty result (no crash, no rows) rather than indexing row_groups[0].
    byte[] bytes = serialize(makeFooter());   // no row groups
    try (ParquetFooter footer = readFooter(bytes, 0, 1000)) {
      assertArrayEquals(new long[]{}, footer.getRowIndexOffsets());
      assertEquals(0, footer.getNumRows());
    }
  }

  @Test
  void testFileOffsetAbsentFallsBackToDefaultStartIndex() throws Exception {
    // A row group with no inline meta_data AND no file_offset: start_index = file_offset.value_or(0)
    // == 0, which invalid_file_offset() then corrects to the documented first-row-group offset of 4.
    ColumnChunk cc = new ColumnChunk(0L);   // meta_data and file_offset intentionally unset
    RowGroup rg = new RowGroup(Collections.singletonList(cc), 100, 1000);
    rg.setTotal_compressed_size(100);       // file_offset intentionally left unset
    byte[] bytes = serialize(makeFooter(rg));
    // start_index corrected to 4, total_size 100 -> midpoint 54, inside [0, 1000).
    try (ParquetFooter footer = readFooter(bytes, 0, 1000)) {
      assertArrayEquals(new long[]{0}, footer.getRowIndexOffsets());
      assertEquals(1000, footer.getNumRows());
    }
  }

  @Test
  void testTotalCompressedSizeAbsentSumsFromColumnChunks() throws Exception {
    // A row group whose total_compressed_size is unset forces filter_groups to sum the per-column
    // total_compressed_size (60 + 40 = 100) instead. Two INT32 leaf columns carry the sizes.
    SchemaElement root = new SchemaElement("schema");
    root.setNum_children(2);
    ColumnMetaData cmd0 = new ColumnMetaData(Type.INT32,
        Collections.singletonList(Encoding.PLAIN), Collections.singletonList("c0"),
        CompressionCodec.UNCOMPRESSED, 1000, 120, 60, 100);   // total_compressed_size=60, offset=100
    ColumnChunk cc0 = new ColumnChunk(100L);
    cc0.setMeta_data(cmd0);
    ColumnMetaData cmd1 = new ColumnMetaData(Type.INT32,
        Collections.singletonList(Encoding.PLAIN), Collections.singletonList("c1"),
        CompressionCodec.UNCOMPRESSED, 1000, 80, 40, 110);    // total_compressed_size=40, offset=110
    ColumnChunk cc1 = new ColumnChunk(110L);
    cc1.setMeta_data(cmd1);
    RowGroup rg = new RowGroup(Arrays.asList(cc0, cc1), 100, 1000);
    // total_compressed_size intentionally left unset on the row group.
    byte[] bytes = serialize(new org.apache.parquet.format.FileMetaData(
        1, Arrays.asList(root, leaf("c0", FieldRepetitionType.OPTIONAL),
            leaf("c1", FieldRepetitionType.OPTIONAL)), 1000, Collections.singletonList(rg)));
    ParquetFooter.StructElement readSchema = ParquetFooter.StructElement.builder()
        .addChild("c0", new ParquetFooter.ValueElement())
        .addChild("c1", new ParquetFooter.ValueElement())
        .build();
    // start_index = get_offset(columns[0]) = 100; total_size = 60+40 = 100; midpoint = 150.
    // Range [140, 160) contains 150 only if BOTH chunk sizes were summed.
    try (ParquetFooter footer = readFooter(bytes, 140, 20, readSchema)) {
      assertArrayEquals(new long[]{0}, footer.getRowIndexOffsets());
      assertEquals(1000, footer.getNumRows());
    }
  }

  @Test
  void testColumnOrdersSurvivePruning() throws Exception {
    // column_orders (one TYPE_ORDER per leaf column) must be reindexed through the prune map when
    // columns are dropped. makeNestedFooter has 6 leaves; pruning the MAP drops 2, leaving 4.
    org.apache.parquet.format.FileMetaData meta = makeNestedFooter(1234);
    List<ColumnOrder> orders = new ArrayList<>();
    for (int i = 0; i < 6; i++) {
      orders.add(ColumnOrder.TYPE_ORDER(new TypeDefinedOrder()));
    }
    meta.setColumn_orders(orders);
    byte[] bytes = serialize(meta);
    try (ParquetFooter footer = readFooter(bytes, 0, -1, makePrunedReadSchema());
         HostMemoryBuffer out = footer.serializeThriftFile()) {
      assertEquals(3, footer.getNumColumns());
      // Round-trip the pruned footer through parquet-mr; the 4 surviving leaf column_orders remain.
      int total = (int) out.getLength();
      byte[] framed = new byte[total];
      out.getBytes(framed, 0, 0, total);
      org.apache.parquet.format.FileMetaData reparsed =
          Util.readFileMetaData(new ByteArrayInputStream(framed, 4, total - 12));
      assertEquals(4, reparsed.getColumn_orders().size());
    }
  }

  @Test
  void testReadAndFilterToleratesTrailingLengthBytes() throws Exception {
    // spark-rapids hands readAndFilter the thrift footer plus a trailing 4-byte little-endian
    // footer-length word (its "footer + footerLen" slice, PAR1 magic stripped from both ends).
    // The facade parses buffer.getLength() bytes, so it must tolerate that tail, not reject it.
    byte[] footerBytes = serialize(threeRowGroupFooter());

    // Baseline parse of the exact-size footer, with no trailing bytes.
    long[] expectedOffsets;
    long expectedRows;
    int expectedColumns;
    try (ParquetFooter footer = readFooter(footerBytes, 0, -1)) {
      expectedOffsets = footer.getRowIndexOffsets();
      expectedRows = footer.getNumRows();
      expectedColumns = footer.getNumColumns();
    }

    // Append the 4-byte little-endian footer length, reproducing the real footer + footerLen tail.
    int footerLen = footerBytes.length;
    byte[] padded = Arrays.copyOf(footerBytes, footerLen + 4);
    padded[footerLen]     = (byte) footerLen;
    padded[footerLen + 1] = (byte) (footerLen >>> 8);
    padded[footerLen + 2] = (byte) (footerLen >>> 16);
    padded[footerLen + 3] = (byte) (footerLen >>> 24);

    // The trailing length word must not perturb the parse: same row offsets, row count, and
    // column count as the un-padded footer.
    try (ParquetFooter footer = readFooter(padded, 0, -1)) {
      assertArrayEquals(expectedOffsets, footer.getRowIndexOffsets());
      assertEquals(expectedRows, footer.getNumRows());
      assertEquals(expectedColumns, footer.getNumColumns());
    }
  }
}
