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

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.HostColumnVectorCore;
import ai.rapids.cudf.HostColumnVector.*;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;

/**
 * Tests for the Protobuf GPU decoder.
 *
 * Test cases are inspired by Google's protobuf conformance test suite:
 * https://github.com/protocolbuffers/protobuf/tree/main/conformance
 */
public class ProtobufTest {

  // ============================================================================
  // Helper methods for encoding protobuf wire format
  // ============================================================================

  /** Encode a value as a varint (variable-length integer). */
  private static byte[] encodeVarint(long value) {
    long v = value;
    byte[] tmp = new byte[10];
    int idx = 0;
    while ((v & ~0x7FL) != 0) {
      tmp[idx++] = (byte) ((v & 0x7F) | 0x80);
      v >>>= 7;
    }
    tmp[idx++] = (byte) (v & 0x7F);
    byte[] out = new byte[idx];
    System.arraycopy(tmp, 0, out, 0, idx);
    return out;
  }

  /** ZigZag encode a signed 32-bit integer, returning as unsigned long for varint encoding. */
  private static long zigzagEncode32(int n) {
    return Integer.toUnsignedLong((n << 1) ^ (n >> 31));
  }

  /** ZigZag encode a signed 64-bit integer. */
  private static long zigzagEncode64(long n) {
    return (n << 1) ^ (n >> 63);
  }

  /** Encode a 32-bit value in little-endian (fixed32). */
  private static byte[] encodeFixed32(int v) {
    return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array();
  }

  /** Encode a 64-bit value in little-endian (fixed64). */
  private static byte[] encodeFixed64(long v) {
    return ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array();
  }

  /** Encode a float in little-endian (fixed32 wire type). */
  private static byte[] encodeFloat(float f) {
    return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(f).array();
  }

  /** Encode a double in little-endian (fixed64 wire type). */
  private static byte[] encodeDouble(double d) {
    return ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putDouble(d).array();
  }

  /** Create a protobuf tag (field number + wire type). */
  private static byte[] tag(int fieldNumber, int wireType) {
    return encodeVarint(((long) fieldNumber << 3) | wireType);
  }

  // Wire type constants
  private static final int WT_VARINT = 0;
  private static final int WT_64BIT = 1;
  private static final int WT_LEN = 2;
  private static final int WT_SGROUP = 3;
  private static final int WT_EGROUP = 4;
  private static final int WT_32BIT = 5;
  private static final int PROTOBUF_JAVA_RECURSION_LIMIT = 100;

  private static Byte[] box(byte[] bytes) {
    if (bytes == null) return null;
    Byte[] out = new Byte[bytes.length];
    for (int i = 0; i < bytes.length; i++) {
      out[i] = bytes[i];
    }
    return out;
  }

  private static Byte[] concat(Byte[]... parts) {
    int len = 0;
    for (Byte[] p : parts) if (p != null) len += p.length;
    Byte[] out = new Byte[len];
    int pos = 0;
    for (Byte[] p : parts) {
      if (p != null) {
        System.arraycopy(p, 0, out, pos, p.length);
        pos += p.length;
      }
    }
    return out;
  }

  /** Encode a length-delimited byte sequence: varint length prefix followed by the bytes. */
  private static Byte[] encodeBytes(Byte[] bytes) {
    return concat(box(encodeVarint(bytes.length)), bytes);
  }

  private static Byte[] encodeBytes(byte[] bytes) {
    return encodeBytes(box(bytes));
  }

  private static Byte[] encodeString(String value) {
    return encodeBytes(value.getBytes(StandardCharsets.UTF_8));
  }

  /** Encode a length-delimited submessage: varint length prefix followed by the message bytes. */
  private static Byte[] encodeMessage(Byte[] messageBytes) {
    return encodeBytes(messageBytes);
  }

  private static Byte[] wrapInUnknownGroups(Byte[] payload, int depth) {
    Byte[] result = payload;
    for (int i = depth - 1; i >= 0; i--) {
      int fieldNumber = 10 + i;
      result = concat(
          box(tag(fieldNumber, WT_SGROUP)), result, box(tag(fieldNumber, WT_EGROUP)));
    }
    return result;
  }

  private static void assertSingleNullStructRow(ColumnVector actual, String message) {
    try (HostColumnVector hostStruct = actual.copyToHost()) {
      assertEquals(1, actual.getNullCount(), message);
      assertTrue(hostStruct.isNull(0), "Row 0 should be null");
    }
  }

  private static void assertListOffsets(ColumnView list, int... expected) {
    try (ColumnView offsetsView = list.getListOffsetsView();
         ColumnVector offsets = offsetsView.copyToColumnVector();
         HostColumnVector hostOffsets = offsets.copyToHost()) {
      assertEquals(expected.length, hostOffsets.getRowCount(), "Unexpected list offsets count");
      for (int i = 0; i < expected.length; i++) {
        assertEquals(expected[i], hostOffsets.getInt(i), "Unexpected list offset at " + i);
      }
    }
  }

  private static StructData struct(Object... values) {
    return new StructData(values);
  }

  // ============================================================================
  // Basic Type Tests
  // ============================================================================

  @Test
  void decodeVarintAndStringToStruct() {
    // message Msg { int64 id = 1; string name = 2; }
    // Row0: id=100, name="alice"
    Byte[] row0 = concat(
        box(tag(1, WT_VARINT)),
        box(encodeVarint(100)),
        box(tag(2, WT_LEN)),
        box(encodeVarint(5)),
        box("alice".getBytes()));

    // Row1: id=200, name missing
    Byte[] row1 = concat(
        box(tag(1, WT_VARINT)),
        box(encodeVarint(200)));

    // Row2: null input message
    Byte[] row2 = null;

    StructType expectedType = new StructType(
        true,
        new BasicType(true, DType.INT64),
        new BasicType(true, DType.STRING));
    try (Table input = new Table.TestBuilder().column(row0, row1, row2).build();
         ColumnVector expectedStruct = ColumnVector.fromStructs(
             expectedType, struct(100L, "alice"), struct(200L, null), null);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .addField(2, DType.STRING)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void decodeMoreTypes() {
    // message Msg { uint32 u32 = 1; sint64 s64 = 2; fixed32 f32 = 3; bytes b = 4; }
    Byte[] row0 = concat(
        box(tag(1, WT_VARINT)),
        box(encodeVarint(4000000000L)),
        box(tag(2, WT_VARINT)),
        box(encodeVarint(zigzagEncode64(-1234567890123L))),
        box(tag(3, WT_32BIT)),
        box(encodeFixed32(12345)),
        box(tag(4, WT_LEN)),
        box(encodeVarint(3)),
        box(new byte[]{1, 2, 3}));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0}).build();
         ColumnVector expectedU32 = ColumnVector.fromBoxedLongs(4000000000L);
         ColumnVector expectedS64 = ColumnVector.fromBoxedLongs(-1234567890123L);
         ColumnVector expectedF32 = ColumnVector.fromBoxedInts(12345);
         ColumnVector expectedB = ColumnVector.fromLists(
            new ListType(true, new BasicType(true, DType.UINT8)),
             Arrays.asList((byte) 1, (byte) 2, (byte) 3));
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.UINT32)
                 .addField(2, DType.INT64).encoding(Protobuf.ENC_ZIGZAG)
                 .addField(3, DType.INT32).encoding(Protobuf.ENC_FIXED)
                 .addField(4, DType.LIST)
                 .build(),
             true)) {
      try (ColumnVector expectedU32Correct = expectedU32.castTo(DType.UINT32);
           ColumnVector expectedStructCorrect = ColumnVector.makeStruct(
               expectedU32Correct, expectedS64, expectedF32, expectedB)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStructCorrect, actualStruct);
      }
    }
  }

  @Test
  void decodeFloatDoubleAndBool() {
    // message Msg { bool flag = 1; float f32 = 2; double f64 = 3; }
    Byte[] row0 = concat(
        box(tag(1, WT_VARINT)), new Byte[]{(byte)0x01},  // bool=true
        box(tag(2, WT_32BIT)), box(encodeFloat(3.14f)),
        box(tag(3, WT_64BIT)), box(encodeDouble(2.71828)));

    Byte[] row1 = concat(
        box(tag(1, WT_VARINT)), new Byte[]{(byte)0x00},  // bool=false
        box(tag(2, WT_32BIT)), box(encodeFloat(-1.5f)),
        box(tag(3, WT_64BIT)), box(encodeDouble(0.0)));

    try (Table input = new Table.TestBuilder().column(row0, row1).build();
         ColumnVector expectedBool = ColumnVector.fromBoxedBooleans(true, false);
         ColumnVector expectedFloat = ColumnVector.fromBoxedFloats(3.14f, -1.5f);
         ColumnVector expectedDouble = ColumnVector.fromBoxedDoubles(2.71828, 0.0);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedBool, expectedFloat, expectedDouble);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.BOOL8)
                 .addField(2, DType.FLOAT32)
                 .addField(3, DType.FLOAT64)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Schema Projection Tests (new API feature)
  // ============================================================================

  @Test
  void testSchemaProjection() {
    // message Msg { int64 f1 = 1; string f2 = 2; int32 f3 = 3; }
    // Only decode f1 and f3, f2 should be null
    Byte[] row0 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(100)),
        box(tag(2, WT_LEN)), box(encodeVarint(5)), box("hello".getBytes()),
        box(tag(3, WT_VARINT)), box(encodeVarint(42)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0}).build();
         // Expected: f1=100, f3=42 (schema projection: only decode these two)
         ColumnVector expectedF1 = ColumnVector.fromBoxedLongs(100L);
         ColumnVector expectedF3 = ColumnVector.fromBoxedInts(42);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedF1, expectedF3);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)   // f1
                 .addField(3, DType.INT32)   // f3 (f2 skipped -> projection)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testSchemaProjectionDecodeNone() {
    // Decode no fields - all should be null
    Byte[] row0 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(100)),
        box(tag(2, WT_LEN)), box(encodeVarint(5)), box("hello".getBytes()));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0}).build();
         // With no fields in the schema, the GPU returns an empty struct
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .build(),
             true)) {
      assertNotNull(actualStruct);
      assertEquals(DType.STRUCT, actualStruct.getType());
    }
  }

  // ============================================================================
  // Varint Boundary Tests
  // ============================================================================

  @Test
  void testVarintMaxUint64() {
    // Max uint64 = 0xFFFFFFFFFFFFFFFF = 18446744073709551615
    // Encoded as 10 bytes: FF FF FF FF FF FF FF FF FF 01
    Byte[] row = concat(
        box(tag(1, WT_VARINT)),
        new Byte[]{(byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0xFF,
                   (byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0x01});

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.UINT64)
                 .build(),
             true)) {
      try (ColumnVector expectedU64 = ColumnVector.fromBoxedLongs(-1L);  // -1 as unsigned = max
           ColumnVector expectedU64Correct = expectedU64.castTo(DType.UINT64);
           ColumnVector expectedStruct = ColumnVector.makeStruct(expectedU64Correct)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
      }
    }
  }

  // ============================================================================
  // Output shape tests — verify the stub produces correctly typed struct columns
  // ============================================================================

  @Test
  void testEmptySchemaProducesEmptyStruct() {
    Byte[] row = new Byte[]{0x08, 0x01};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder().build(), true)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(1, result.getRowCount());
      assertEquals(0, result.getNumChildren());
    }
  }

  @Test
  void testEmptySchemaStillValidatesMalformedWireData() {
    Byte[] malformed = concat(box(tag(1, WT_VARINT)), new Byte[]{(byte) 0x80});
    ProtobufSchemaDescriptor emptySchema = new ProtobufSchemaDescriptorBuilder().build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{malformed}).build();
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0), emptySchema, false)) {
      assertEquals(0, actual.getNumChildren());
      assertSingleNullStructRow(actual, "Malformed wire data should null an empty-schema row");
    }

    try (Table input = new Table.TestBuilder().column(new Byte[][]{malformed}).build()) {
      ai.rapids.cudf.CudfException error = assertThrows(
          ai.rapids.cudf.CudfException.class,
          () -> {
            try (ColumnVector ignored = Protobuf.decodeToStruct(
                input.getColumn(0), emptySchema, true)) {
            }
          });
      assertTrue(error.getMessage().contains("unable to skip unknown field"));
    }
  }

  @Test
  void testVarintZero() {
    // Zero encoded as single byte: 0x00
    Byte[] row = concat(box(tag(1, WT_VARINT)), new Byte[]{0x00});

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedLongs(0L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testSingleScalarFieldOutputShape() {
    Byte[] row = new Byte[]{0x08, 0x01};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .build(), true)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(1, result.getRowCount());
      assertEquals(1, result.getNumChildren());
      assertEquals(DType.INT64, result.getChildColumnView(0).getType());
    }
  }

  @Test
  void testVarintOverEncodedZero() {
    // Zero over-encoded as 10 bytes (all continuation bits except last)
    // This is valid per protobuf spec - parsers must accept non-canonical varints
    Byte[] row = concat(
        box(tag(1, WT_VARINT)),
        new Byte[]{(byte)0x80, (byte)0x80, (byte)0x80, (byte)0x80, (byte)0x80,
                   (byte)0x80, (byte)0x80, (byte)0x80, (byte)0x80, (byte)0x00});

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedLongs(0L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testVarint10thByteInvalid() {
    // 10th byte with more than 1 significant bit is invalid
    // (uint64 can only hold 64 bits: 9*7=63 bits + 1 bit from 10th byte)
    Byte[] row = concat(
        box(tag(1, WT_VARINT)),
        new Byte[]{(byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0xFF,
                   (byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0x02});  // 0x02 has 2nd bit set

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .build(),
             false)) {
      assertSingleNullStructRow(result, "Overflowing varint should null the struct row");
    }
  }

  // ============================================================================
  // ZigZag Boundary Tests
  // ============================================================================

  @Test
  void testZigzagInt32Min() {
    // int32 min = -2147483648
    // zigzag encoded = 4294967295 = 0xFFFFFFFF
    int minInt32 = Integer.MIN_VALUE;
    Byte[] row = concat(
        box(tag(1, WT_VARINT)),
        box(encodeVarint(zigzagEncode32(minInt32))));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedInts(minInt32);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).encoding(Protobuf.ENC_ZIGZAG)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testZigzagInt32Max() {
    // int32 max = 2147483647
    // zigzag encoded = 4294967294 = 0xFFFFFFFE
    int maxInt32 = Integer.MAX_VALUE;
    Byte[] row = concat(
        box(tag(1, WT_VARINT)),
        box(encodeVarint(zigzagEncode32(maxInt32))));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedInts(maxInt32);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).encoding(Protobuf.ENC_ZIGZAG)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testZigzagInt64Min() {
    // int64 min = -9223372036854775808
    long minInt64 = Long.MIN_VALUE;
    Byte[] row = concat(
        box(tag(1, WT_VARINT)),
        box(encodeVarint(zigzagEncode64(minInt64))));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedLong = ColumnVector.fromBoxedLongs(minInt64);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedLong);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64).encoding(Protobuf.ENC_ZIGZAG)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testZigzagInt64Max() {
    long maxInt64 = Long.MAX_VALUE;
    Byte[] row = concat(
        box(tag(1, WT_VARINT)),
        box(encodeVarint(zigzagEncode64(maxInt64))));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedLong = ColumnVector.fromBoxedLongs(maxInt64);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedLong);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64).encoding(Protobuf.ENC_ZIGZAG)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testZigzagNegativeOne() {
    // -1 zigzag encoded = 1
    Byte[] row = concat(
        box(tag(1, WT_VARINT)),
        box(encodeVarint(zigzagEncode64(-1L))));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedLong = ColumnVector.fromBoxedLongs(-1L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedLong);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64).encoding(Protobuf.ENC_ZIGZAG)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Truncated/Malformed Data Tests
  // ============================================================================

  @Test
  void testMalformedVarint() {
    // Varint that never terminates (all continuation bits set, 11 bytes)
    Byte[] malformed = new Byte[]{(byte)0x08, (byte)0xFF, (byte)0xFF, (byte)0xFF,
                                   (byte)0xFF, (byte)0xFF, (byte)0xFF,
                                   (byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0xFF};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{malformed}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .build(),
             false)) {
      assertSingleNullStructRow(result, "Malformed varint should null the struct row");
    }
  }

  @Test
  void testTruncatedVarint() {
    // Single byte with continuation bit set but no following byte
    Byte[] truncated = concat(box(tag(1, WT_VARINT)), new Byte[]{(byte)0x80});
    try (Table input = new Table.TestBuilder().column(new Byte[][]{truncated}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .build(),
             false)) {
      assertSingleNullStructRow(result, "Truncated varint should null the struct row");
    }
  }

  @Test
  void testTruncatedLengthDelimited() {
    // String field with length=5 but no actual data
    Byte[] truncated = concat(box(tag(2, WT_LEN)), box(encodeVarint(5)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{truncated}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(2, DType.STRING)
                 .build(),
             false)) {
      assertSingleNullStructRow(result,
          "Truncated length-delimited field should null the struct row");
    }
  }

  @Test
  void testTruncatedFixed32() {
    // Fixed32 needs 4 bytes but only 3 provided
    Byte[] truncated = concat(box(tag(1, WT_32BIT)), new Byte[]{0x01, 0x02, 0x03});
    try (Table input = new Table.TestBuilder().column(new Byte[][]{truncated}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).encoding(Protobuf.ENC_FIXED)
                 .build(),
             false)) {
      assertSingleNullStructRow(result, "Truncated fixed32 should null the struct row");
    }
  }

  @Test
  void testTruncatedFixed64() {
    // Fixed64 needs 8 bytes but only 7 provided
    Byte[] truncated = concat(box(tag(1, WT_64BIT)), 
        new Byte[]{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07});
    try (Table input = new Table.TestBuilder().column(new Byte[][]{truncated}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64).encoding(Protobuf.ENC_FIXED)
                 .build(),
             false)) {
      assertSingleNullStructRow(result, "Truncated fixed64 should null the struct row");
    }
  }

  @Test
  void testPartialLengthDelimitedData() {
    // Length says 10 bytes but only 5 provided
    Byte[] partial = concat(
        box(tag(1, WT_LEN)),
        box(encodeVarint(10)),
        box("hello".getBytes()));  // only 5 bytes
    try (Table input = new Table.TestBuilder().column(new Byte[][]{partial}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING)
                 .build(),
             false)) {
      assertSingleNullStructRow(result,
          "Partial length-delimited payload should null the struct row");
    }
  }

  // ============================================================================
  // Wrong Wire Type Tests
  // ============================================================================

  @Test
  void testWrongWireType() {
    // Expect varint (wire type 0) but provide fixed32 (wire type 5)
    Byte[] wrongType = concat(
        box(tag(1, WT_32BIT)),  // wire type 5 instead of 0
        box(encodeFixed32(100)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{wrongType}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .build(),
             false)) {
      assertSingleNullStructRow(result, "Wrong wire type should null the struct row");
    }
  }

  @Test
  void testWrongWireTypeForString() {
    // Expect length-delimited (wire type 2) but provide varint (wire type 0)
    Byte[] wrongType = concat(
        box(tag(1, WT_VARINT)),
        box(encodeVarint(12345)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{wrongType}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING)
                 .build(),
             false)) {
      assertSingleNullStructRow(result, "Wrong wire type for string should null the struct row");
    }
  }

  // ============================================================================
  // Unknown Field Skip Tests
  // ============================================================================

  @Test
  void testSkipUnknownVarintField() {
    // Unknown field 99 with varint, followed by known field 1
    Byte[] row = concat(
        box(tag(99, WT_VARINT)),
        box(encodeVarint(12345)),  // unknown field to skip
        box(tag(1, WT_VARINT)),
        box(encodeVarint(42)));    // known field

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedLongs(42L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testSkipUnknownFixed64Field() {
    // Unknown field 99 with fixed64, followed by known field 1
    Byte[] row = concat(
        box(tag(99, WT_64BIT)),
        box(encodeFixed64(0x123456789ABCDEF0L)),  // unknown field to skip
        box(tag(1, WT_VARINT)),
        box(encodeVarint(42)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedLongs(42L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testSkipUnknownLengthDelimitedField() {
    // Unknown field 99 with length-delimited data, followed by known field 1
    Byte[] row = concat(
        box(tag(99, WT_LEN)),
        box(encodeVarint(5)),
        box("hello".getBytes()),  // unknown field to skip
        box(tag(1, WT_VARINT)),
        box(encodeVarint(42)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedLongs(42L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testSkipUnknownFixed32Field() {
    // Unknown field 99 with fixed32, followed by known field 1
    Byte[] row = concat(
        box(tag(99, WT_32BIT)),
        box(encodeFixed32(12345)),  // unknown field to skip
        box(tag(1, WT_VARINT)),
        box(encodeVarint(42)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedLongs(42L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Last One Wins (Repeated Scalar Field) Tests
  // ============================================================================

  @Test
  void testLastOneWins() {
    // Same field appears multiple times - last value should win
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(100)),
        box(tag(1, WT_VARINT)), box(encodeVarint(200)),
        box(tag(1, WT_VARINT)), box(encodeVarint(300)));  // this should win

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedLongs(300L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testLastOneWinsForString() {
    // Same string field appears multiple times
    Byte[] row = concat(
        box(tag(1, WT_LEN)), box(encodeVarint(5)), box("first".getBytes()),
        box(tag(1, WT_LEN)), box(encodeVarint(6)), box("second".getBytes()),
        box(tag(1, WT_LEN)), box(encodeVarint(4)), box("last".getBytes()));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedStr = ColumnVector.fromStrings("last");
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedStr);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Error Handling Tests
  // ============================================================================

  @Test
  void testFieldNumberZeroInvalid() {
    // Field number 0 is reserved and invalid
    Byte[] invalid = concat(box(tag(0, WT_VARINT)), box(encodeVarint(123)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{invalid}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .build(),
             false)) {
      assertSingleNullStructRow(result, "Field number zero should null the struct row");
    }
  }

  @Test
  void testEmptyMessage() {
    // Empty message should result in null/default values for all fields
    Byte[] empty = new Byte[0];
    try (Table input = new Table.TestBuilder().column(new Byte[][]{empty}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedLongs((Long)null);
         ColumnVector expectedStr = ColumnVector.fromStrings((String)null);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt, expectedStr);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .addField(2, DType.STRING)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Float/Double Special Values Tests
  // ============================================================================

  @Test
  void testFloatSpecialValues() {
    Byte[] rowInf = concat(box(tag(1, WT_32BIT)), box(encodeFloat(Float.POSITIVE_INFINITY)));
    Byte[] rowNegInf = concat(box(tag(1, WT_32BIT)), box(encodeFloat(Float.NEGATIVE_INFINITY)));
    Byte[] rowNaN = concat(box(tag(1, WT_32BIT)), box(encodeFloat(Float.NaN)));
    Byte[] rowMin = concat(box(tag(1, WT_32BIT)), box(encodeFloat(Float.MIN_VALUE)));
    Byte[] rowMax = concat(box(tag(1, WT_32BIT)), box(encodeFloat(Float.MAX_VALUE)));

    try (Table input = new Table.TestBuilder().column(rowInf, rowNegInf, rowNaN, rowMin, rowMax).build();
         ColumnVector expectedFloat = ColumnVector.fromBoxedFloats(
             Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN, 
             Float.MIN_VALUE, Float.MAX_VALUE);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedFloat);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.FLOAT32)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDoubleSpecialValues() {
    Byte[] rowInf = concat(box(tag(1, WT_64BIT)), box(encodeDouble(Double.POSITIVE_INFINITY)));
    Byte[] rowNegInf = concat(box(tag(1, WT_64BIT)), box(encodeDouble(Double.NEGATIVE_INFINITY)));
    Byte[] rowNaN = concat(box(tag(1, WT_64BIT)), box(encodeDouble(Double.NaN)));
    Byte[] rowMin = concat(box(tag(1, WT_64BIT)), box(encodeDouble(Double.MIN_VALUE)));
    Byte[] rowMax = concat(box(tag(1, WT_64BIT)), box(encodeDouble(Double.MAX_VALUE)));

    try (Table input = new Table.TestBuilder().column(rowInf, rowNegInf, rowNaN, rowMin, rowMax).build();
         ColumnVector expectedDouble = ColumnVector.fromBoxedDoubles(
             Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NaN,
             Double.MIN_VALUE, Double.MAX_VALUE);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedDouble);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.FLOAT64)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Enum Tests (enums.as.ints=true semantics)
  // ============================================================================

  @Test
  void testEnumAsInt() {
    // message Msg { enum Color { RED=0; GREEN=1; BLUE=2; } Color c = 1; }
    // c = GREEN (value 1) - encoded as varint
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(1)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedInts(1);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testEnumZeroValue() {
    // Enum with value 0 (first/default enum value)
    // c = RED (value 0)
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(0)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedInts(0);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testEnumUnknownValue() {
    // Protobuf allows unknown enum values - they should still be decoded as integers
    // c = 999 (unknown value not in enum definition)
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(999)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedInts(999);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testEnumNegativeValue() {
    // Negative enum values are valid in protobuf (stored as unsigned varint)
    // c = -1 (represented as 0xFFFFFFFF in protobuf wire format)
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(-1L & 0xFFFFFFFFL)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedInts(-1);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testEnumMultipleFields() {
    // message Msg { enum Status { OK=0; ERROR=1; } Status s1 = 1; int32 count = 2; Status s2 = 3; }
    // s1 = ERROR (1), count = 42, s2 = OK (0)
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),   // s1 = ERROR
        box(tag(2, WT_VARINT)), box(encodeVarint(42)),  // count = 42
        box(tag(3, WT_VARINT)), box(encodeVarint(0)));  // s2 = OK

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedS1 = ColumnVector.fromBoxedInts(1);
         ColumnVector expectedCount = ColumnVector.fromBoxedInts(42);
         ColumnVector expectedS2 = ColumnVector.fromBoxedInts(0);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedS1, expectedCount, expectedS2);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32)
                 .addField(2, DType.INT32)
                 .addField(3, DType.INT32)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testEnumMissingField() {
    // Enum field not present in message - should be null
    Byte[] row = concat(box(tag(2, WT_VARINT)), box(encodeVarint(42)));  // only count field

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedEnum = ColumnVector.fromBoxedInts((Integer) null);
         ColumnVector expectedCount = ColumnVector.fromBoxedInts(42);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedEnum, expectedCount);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32)
                 .addField(2, DType.INT32)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Required Field Tests
  // ============================================================================

  @Test
  void testRequiredFieldPresent() {
    // message Msg { required int64 id = 1; optional string name = 2; }
    // Both fields present - should decode successfully
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(42)),
        box(tag(2, WT_LEN)), box(encodeVarint(5)), box("hello".getBytes()));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedId = ColumnVector.fromBoxedLongs(42L);
         ColumnVector expectedName = ColumnVector.fromStrings("hello");
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedId, expectedName);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64).required()
                 .addField(2, DType.STRING)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testRequiredFieldMissing_Permissive() {
    // Required field missing in permissive mode - should null the whole row without exception
    // message Msg { required int64 id = 1; optional string name = 2; }
    // Only name field present, required id is missing
    Byte[] row = concat(
        box(tag(2, WT_LEN)), box(encodeVarint(5)), box("hello".getBytes()));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64).required()
                 .addField(2, DType.STRING)
                 .build(),
             false)) {  // permissive mode - don't fail on errors
      assertSingleNullStructRow(actualStruct,
          "Missing top-level required field should null the row in PERMISSIVE mode");
    }
  }

  @Test
  void testRequiredFieldMissing_Failfast() {
    // Required field missing in failfast mode - should throw exception
    // message Msg { required int64 id = 1; optional string name = 2; }
    // Only name field present, required id is missing
    Byte[] row = concat(
        box(tag(2, WT_LEN)), box(encodeVarint(5)), box("hello".getBytes()));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.INT64).required()
                .addField(2, DType.STRING)
                .build(),
            true)) {  // failfast mode - should throw
        }
      });
    }
  }

  @Test
  void testMultipleRequiredFields_AllPresent() {
    // message Msg { required int32 a = 1; required int64 b = 2; required string c = 3; }
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(10)),
        box(tag(2, WT_VARINT)), box(encodeVarint(20)),
        box(tag(3, WT_LEN)), box(encodeVarint(3)), box("abc".getBytes()));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedA = ColumnVector.fromBoxedInts(10);
         ColumnVector expectedB = ColumnVector.fromBoxedLongs(20L);
         ColumnVector expectedC = ColumnVector.fromStrings("abc");
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedA, expectedB, expectedC);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).required()
                 .addField(2, DType.INT64).required()
                 .addField(3, DType.STRING).required()
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testMultipleRequiredFields_SomeMissing_Failfast() {
    // message Msg { required int32 a = 1; required int64 b = 2; required string c = 3; }
    // Only field a is present, b and c are missing
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(10)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.INT32).required()
                .addField(2, DType.INT64).required()
                .addField(3, DType.STRING).required()
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testOptionalFieldsOnly_NoValidation() {
    // All fields optional - missing fields should not cause error
    // message Msg { optional int32 a = 1; optional int64 b = 2; }
    Byte[] row = new Byte[0];  // empty message

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedA = ColumnVector.fromBoxedInts((Integer) null);
         ColumnVector expectedB = ColumnVector.fromBoxedLongs((Long) null);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedA, expectedB);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32)
                 .addField(2, DType.INT64)
                 .build(),
             true)) {  // even with failOnErrors=true, should succeed since all fields are optional
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testRequiredFieldMissingInOneOfMultipleRows_Failfast() {
    // Test required field validation across multiple rows
    // Row 0: required field present
    // Row 1: required field missing (should cause error in failfast mode)
    Byte[] row0 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(42)));
    Byte[] row1 = new Byte[0];  // empty - required field missing

    try (Table input = new Table.TestBuilder().column(row0, row1).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.INT64).required()
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testRequiredFieldIgnoresNullInputRow_Failfast() {
    Byte[] row0 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(42)));
    Byte[] row1 = null;

    try (Table input = new Table.TestBuilder().column(row0, row1).build();
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64).required()
                 .build(),
             true);
         ColumnVector idCol = actualStruct.getChildColumnView(0).copyToColumnVector();
         HostColumnVector hostStruct = actualStruct.copyToHost();
         HostColumnVector hostId = idCol.copyToHost()) {
      assertEquals(1, actualStruct.getNullCount(), "Null input row should be null in output struct");
      assertFalse(hostStruct.isNull(0), "Present required field should keep row 0 valid");
      assertTrue(hostStruct.isNull(1), "Null input row should produce null struct row");
      assertEquals(1, idCol.getNullCount(), "The required child value should be null on the null input row");
      assertTrue(hostId.isNull(1),
          "Null input row should produce a null child value, not REQUIRED");
    }
  }

  @Test
  void testRequiredNestedMessageMissing_Failfast() {
    // message Outer { required Inner detail = 1; }
    // message Inner { optional int32 id = 1; }
    // Missing top-level required nested message should fail in FAILFAST mode.
    Byte[] row = new Byte[0];

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.STRUCT).required().down()
                    .addField(1, DType.INT32)
                .up()
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testRequiredNestedMessageMissing_Permissive() {
    Byte[] missing = new Byte[0];
    Byte[] inner = concat(box(tag(1, WT_VARINT)), box(encodeVarint(42)));
    Byte[] present = concat(box(tag(1, WT_LEN)), encodeMessage(inner));
    StructType innerType = new StructType(true, new BasicType(true, DType.INT32));
    StructType outerType = new StructType(true, innerType);

    try (Table input = new Table.TestBuilder().column(missing, present).build();
         ColumnVector expected = ColumnVector.fromStructs(
             outerType, null, struct(struct(42)));
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).required().down()
                     .addField(1, DType.INT32)
                 .up()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  // ============================================================================
  // Default Value Tests (API accepts parameters, CUDA fill not yet implemented)
  // ============================================================================

  @Test
  void testDefaultValueForMissingFields() {
    // Test that missing fields with default values return the defaults
    Byte[] row = new Byte[0];  // empty message

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         // With default values set, missing fields should return the default values
         ColumnVector expectedA = ColumnVector.fromBoxedInts(42);
         ColumnVector expectedB = ColumnVector.fromBoxedLongs(100L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedA, expectedB);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).defaultValue(42)
                 .addField(2, DType.INT64).defaultValue(100)
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultValueFieldPresent_OverridesDefault() {
    // When field is present, use the actual value (not the default)
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(99)),
        box(tag(2, WT_VARINT)), box(encodeVarint(200)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedA = ColumnVector.fromBoxedInts(99);
         ColumnVector expectedB = ColumnVector.fromBoxedLongs(200L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedA, expectedB);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 // Defaults are NOT used here since both fields are present.
                 .addField(1, DType.INT32).defaultValue(42)
                 .addField(2, DType.INT64).defaultValue(100)
                 .build(),
             false)) {
      // Actual values should be used, not defaults
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultIntValue() {
    // optional int32 count = 1 [default = 42];
    // Empty message should return the default value
    Byte[] row = new Byte[0];

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedInts(42);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).defaultValue(42)
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultBoolValue() {
    // optional bool flag = 1 [default = true];
    Byte[] row = new Byte[0];

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedBool = ColumnVector.fromBoxedBooleans(true);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedBool);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.BOOL8).defaultValue(true)
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultFloatValue() {
    // optional double rate = 1 [default = 3.14];
    Byte[] row = new Byte[0];

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedDouble = ColumnVector.fromBoxedDoubles(3.14);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedDouble);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.FLOAT64).defaultValue(3.14)
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultInt64Value() {
    // optional int64 big_num = 1 [default = 9876543210];
    Byte[] row = new Byte[0];

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedLong = ColumnVector.fromBoxedLongs(9876543210L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedLong);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64).defaultValue(9876543210L)
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testMixedDefaultAndNonDefaultFields() {
    // optional int32 a = 1 [default = 42];
    // optional int64 b = 2; (no default)
    // optional bool c = 3 [default = true];
    // Empty message: a=42, b=null, c=true
    Byte[] row = new Byte[0];

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedA = ColumnVector.fromBoxedInts(42);
         ColumnVector expectedB = ColumnVector.fromBoxedLongs((Long) null);  // no default
         ColumnVector expectedC = ColumnVector.fromBoxedBooleans(true);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedA, expectedB, expectedC);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).defaultValue(42)
                 .addField(2, DType.INT64)
                 .addField(3, DType.BOOL8).defaultValue(true)
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultValueWithPartialMessage() {
    // optional int32 a = 1 [default = 42];
    // optional int64 b = 2 [default = 100];
    // Message has only field b set, a should use default
    Byte[] row = concat(
        box(tag(2, WT_VARINT)), box(encodeVarint(999)));  // b = 999

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedA = ColumnVector.fromBoxedInts(42);  // default
         ColumnVector expectedB = ColumnVector.fromBoxedLongs(999L);  // actual value
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedA, expectedB);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).defaultValue(42)
                 .addField(2, DType.INT64).defaultValue(100)
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultStringValue() {
    // optional string name = 1 [default = "hello"];
    // Empty message should return the default string
    Byte[] row = new Byte[0];

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedStr = ColumnVector.fromStrings("hello");
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedStr);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING).defaultValue("hello".getBytes(StandardCharsets.UTF_8))
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultStringValueEmpty() {
    // optional string name = 1 [default = ""];
    // Empty message with empty default string
    Byte[] row = new Byte[0];

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedStr = ColumnVector.fromStrings("");
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedStr);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING).defaultValue(new byte[0])
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultStringValueWithPresent() {
    // optional string name = 1 [default = "default"];
    // Message has actual value, should override default
    Byte[] row = concat(
        box(tag(1, WT_LEN)),
        encodeString("actual"));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedStr = ColumnVector.fromStrings("actual");
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedStr);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING).defaultValue("default".getBytes(StandardCharsets.UTF_8))
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultStringWithMixedFields() {
    // optional int32 count = 1 [default = 42];
    // optional string name = 2 [default = "test"];
    // Empty message should return both defaults
    Byte[] row = new Byte[0];

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedInts(42);
         ColumnVector expectedStr = ColumnVector.fromStrings("test");
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt, expectedStr);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).defaultValue(42)
                 .addField(2, DType.STRING).defaultValue("test".getBytes(StandardCharsets.UTF_8))
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultStringMultipleRows() {
    // optional string name = 1 [default = "default"];
    // Multiple rows: empty, has value, empty
    Byte[] row1 = new Byte[0];  // will use default
    Byte[] row2 = concat(
        box(tag(1, WT_LEN)),
        encodeString("row2val"));
    Byte[] row3 = new Byte[0];  // will use default

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row1, row2, row3}).build();
         ColumnVector expectedStr = ColumnVector.fromStrings("default", "row2val", "default");
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedStr);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING).defaultValue("default".getBytes(StandardCharsets.UTF_8))
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Tests for Nested and Repeated Fields (Phase 1-3 Implementation)
  // ============================================================================

  @Test
  void testUnpackedRepeatedInt32() {
    // Unpacked repeated: same field number appears multiple times
    // message TestMsg { repeated int32 ids = 1; }
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)),
        box(tag(1, WT_VARINT)), box(encodeVarint(3)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedIds = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)),
             Arrays.asList(1, 2, 3));
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedIds);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).repeated()  // ids
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testPackedRepeatedDoubleWithMultipleFields() {
    // Test packed repeated fields with multiple types including edge cases.
    // message WithPackedRepeated {
    //   optional int32 id = 1;
    //   repeated int32 int_values = 2 [packed=true];
    //   repeated double double_values = 3 [packed=true];
    //   repeated bool bool_values = 4 [packed=true];
    // }

    // Row 0: id=42, int_values=[1,-1,100] (12 bytes packed), double_values=[1.5,2.5], bool=[true,false]
    // Row 1: id=7, int_values=15x(-1) (150 bytes packed, 2-byte length varint!), double_values=[3.0,4.0], bool=[true]
    // Row 2: id=0, int_values=[] (field omitted), double_values=[5.0], bool=[] (field omitted)

    // --- Row 0 ---
    byte[] r0IntVarints = concatBytes(encodeVarint(1), encodeVarint(-1L & 0xFFFFFFFFFFFFFFFFL), encodeVarint(100));
    byte[] r0Doubles = concatBytes(encodeDouble(1.5), encodeDouble(2.5));
    byte[] r0Bools = new byte[]{0x01, 0x00};
    Byte[] row0 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(42)),
        box(tag(2, WT_LEN)), encodeBytes(r0IntVarints),
        box(tag(3, WT_LEN)), encodeBytes(r0Doubles),
        box(tag(4, WT_LEN)), encodeBytes(r0Bools));

    // --- Row 1: 15 negative ints => 150 bytes packed (length varint is 2 bytes: 0x96 0x01) ---
    java.io.ByteArrayOutputStream buf1 = new java.io.ByteArrayOutputStream();
    byte[] negOneVarint = encodeVarint(-1L & 0xFFFFFFFFFFFFFFFFL); // 10 bytes
    for (int i = 0; i < 15; i++) {
      buf1.write(negOneVarint, 0, negOneVarint.length);
    }
    byte[] r1IntVarints = buf1.toByteArray(); // 150 bytes
    byte[] r1Doubles = concatBytes(encodeDouble(3.0), encodeDouble(4.0));
    byte[] r1Bools = new byte[]{0x01};
    Byte[] row1 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(7)),
        box(tag(2, WT_LEN)), encodeBytes(r1IntVarints),
        box(tag(3, WT_LEN)), encodeBytes(r1Doubles),
        box(tag(4, WT_LEN)), encodeBytes(r1Bools));

    // --- Row 2: no int_values, no bool_values ---
    byte[] r2Doubles = encodeDouble(5.0);
    Byte[] row2 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(0)),
        box(tag(3, WT_LEN)), encodeBytes(r2Doubles));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1, row2}).build();
         ColumnVector expectedIds = ColumnVector.fromBoxedInts(42, 7, 0);
         ColumnVector expectedInts = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)),
             Arrays.asList(1, -1, 100),
             Collections.nCopies(15, -1),
             Collections.emptyList());
         ColumnVector expectedDoubles = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.FLOAT64)),
             Arrays.asList(1.5, 2.5),
             Arrays.asList(3.0, 4.0),
             Collections.singletonList(5.0));
         ColumnVector expectedBools = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.BOOL8)),
             Arrays.asList(true, false),
             Collections.singletonList(true),
             Collections.emptyList());
         ColumnVector expectedStruct = ColumnVector.makeStruct(
             expectedIds, expectedInts, expectedDoubles, expectedBools);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32)               // id
                 .addField(2, DType.INT32).repeated()    // int_values (packed)
                 .addField(3, DType.FLOAT64).repeated()  // double_values (packed)
                 .addField(4, DType.BOOL8).repeated()    // bool_values (packed)
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  /** Helper: concatenate byte arrays */
  private static byte[] concatBytes(byte[]... arrays) {
    int len = 0;
    for (byte[] a : arrays) len += a.length;
    byte[] out = new byte[len];
    int pos = 0;
    for (byte[] a : arrays) {
      System.arraycopy(a, 0, out, pos, a.length);
      pos += a.length;
    }
    return out;
  }

  @Test
  void testNestedMessage() {
    // message Inner { int32 x = 1; }
    // message Outer { Inner inner = 1; }
    // Outer with inner.x = 42
    Byte[] innerMessage = concat(box(tag(1, WT_VARINT)), box(encodeVarint(42)));
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(innerMessage));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      try (ColumnVector result = Protobuf.decodeToStruct(
          input.getColumn(0),
          new ProtobufSchemaDescriptorBuilder()
              .addField(1, DType.STRUCT).down()    // inner
                  .addField(1, DType.INT32)        // inner.x
              .up()
              .build(),
          false)) {
        assertNotNull(result);
        assertEquals(DType.STRUCT, result.getType());
        try (ColumnVector expectedX = ColumnVector.fromBoxedInts(42);
             ColumnVector expectedInner = ColumnVector.makeStruct(expectedX);
             ColumnVector expectedOuter = ColumnVector.makeStruct(expectedInner)) {
          AssertUtils.assertStructColumnsAreEqual(expectedOuter, result);
        }
      }
    }
  }

  @Test
  void testPackedRepeatedChildInsideRepeatedMessage() {
    // message Item { repeated int32 ids = 1 [packed=true]; optional int32 score = 2; }
    // message Outer { repeated Item items = 1; }
    byte[] item0Ids = concatBytes(encodeVarint(10), encodeVarint(20));
    Byte[] item0 = concat(
        box(tag(1, WT_LEN)),
        encodeBytes(item0Ids),
        box(tag(2, WT_VARINT)),
        box(encodeVarint(7)));
    byte[] item1Ids = concatBytes(encodeVarint(30));
    Byte[] item1 = concat(
        box(tag(1, WT_LEN)),
        encodeBytes(item1Ids),
        box(tag(2, WT_VARINT)),
        box(encodeVarint(9)));
    Byte[] row = concat(
        box(tag(1, WT_LEN)),
        encodeMessage(item0),
        box(tag(1, WT_LEN)),
        encodeMessage(item1));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedItems = ColumnVector.fromLists(
             new ListType(true,
                 new StructType(true,
                     new ListType(true, new BasicType(true, DType.INT32)),
                     new BasicType(true, DType.INT32))),
             Arrays.asList(
                 new StructData(Arrays.asList(10, 20), 7),
                 new StructData(Arrays.asList(30), 9)));
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedItems);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).repeated().down()
                     .addField(1, DType.INT32).repeated()
                     .addField(2, DType.INT32)
                 .up()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testNestedStructChildInsideRepeatedMessage() {
    // message Inner { int32 x = 1; }
    // message Item { Inner inner = 1; }
    // message Outer { repeated Item items = 1; }
    Byte[] inner0 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(7)));
    Byte[] inner1 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(9)));
    Byte[] item0 = concat(box(tag(1, WT_LEN)), encodeMessage(inner0));
    Byte[] item1 = concat(box(tag(1, WT_LEN)), encodeMessage(inner1));
    Byte[] row = concat(
        box(tag(1, WT_LEN)), encodeMessage(item0),
        box(tag(1, WT_LEN)), encodeMessage(item1));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedItems = ColumnVector.fromLists(
             new ListType(true,
                 new StructType(true,
                     new StructType(true, new BasicType(true, DType.INT32)))),
             Arrays.asList(struct(struct(7)), struct(struct(9))));
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedItems);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).repeated().down()
                     .addField(1, DType.STRUCT).down()
                         .addField(1, DType.INT32)
                     .up()
                 .up()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testRepeatedMessageInsideNestedMessage() {
    Byte[] item0 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(10)),
        box(tag(2, WT_LEN)), encodeString("a"));
    Byte[] item1 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(20)),
        box(tag(2, WT_LEN)), encodeString("b"));
    Byte[] parent = concat(
        box(tag(1, WT_LEN)), encodeMessage(item0),
        box(tag(1, WT_LEN)), encodeMessage(item1));
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(parent));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedItems = ColumnVector.fromLists(
             new ListType(true,
                 new StructType(true,
                     new BasicType(true, DType.INT32),
                     new BasicType(true, DType.STRING))),
             Arrays.asList(struct(10, "a"), struct(20, "b")));
         ColumnVector expectedParent = ColumnVector.makeStruct(expectedItems);
         ColumnVector expectedOuter = ColumnVector.makeStruct(expectedParent);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.STRUCT).repeated().down()
                         .addField(1, DType.INT32)
                         .addField(2, DType.STRING)
                     .up()
                 .up()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actual);
    }
  }

  @Test
  void testPermissiveRepeatedWrongWireTypeNullsMalformedRow() {
    // message Msg { repeated int32 ids = 1; }
    // Row 0 has one valid element, then a malformed fixed32 occurrence for the same field,
    // then another valid varint. Spark CPU nulls the entire malformed row in PERMISSIVE mode.
    Byte[] row0 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_32BIT)), box(encodeFixed32(77)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)));
    Byte[] row1 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(100)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1}).build();
         ColumnVector expectedStruct = ColumnVector.fromStructs(
             new StructType(true,
                 new ListType(true, new BasicType(true, DType.INT32))),
             null,
             struct(Arrays.asList(100)));
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).repeated()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testTopLevelNestedMessageWrongWireTypeBeforeRepeatedField_Permissive() {
    // message Msg { Inner inner = 1; repeated int32 ids = 2; }
    Byte[] malformed = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(2, WT_VARINT)), box(encodeVarint(7)));
    Byte[] valid = concat(box(tag(2, WT_VARINT)), box(encodeVarint(8)));
    Byte[][] rows = new Byte[][]{malformed, valid};
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).isOutput(false)
        .addField(2, DType.INT32).repeated()
        .build();

    try (Table input = new Table.TestBuilder().column(rows).build();
         ColumnVector expected = ColumnVector.fromStructs(
             new StructType(true,
                 new ListType(true, new BasicType(true, DType.INT32))),
             null,
             struct(Arrays.asList(8)));
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0), schema, false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testTopLevelNestedMessageWrongWireTypeBeforeRepeatedField_Failfast() {
    Byte[] malformed = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(2, WT_VARINT)), box(encodeVarint(7)));
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).isOutput(false)
        .addField(2, DType.INT32).repeated()
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{malformed}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(
            input.getColumn(0), schema, true)) {
        }
      });
    }
  }

  @Test
  void testRepeatedUint32() {
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)),
        box(tag(1, WT_VARINT)), box(encodeVarint(3)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedValues = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.UINT32)),
             Arrays.asList(1, 2, 3));
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedValues);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.UINT32).repeated()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testRepeatedUint64() {
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(11)),
        box(tag(1, WT_VARINT)), box(encodeVarint(22)),
        box(tag(1, WT_VARINT)), box(encodeVarint(33)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedValues = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.UINT64)),
             Arrays.asList(11L, 22L, 33L));
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedValues);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.UINT64).repeated()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testRepeatedMessageChildWrongWireTypeSkipsMismatchedFieldInBothModes() {
    // Spark CPU treats a known child field with a mismatched wire type as unknown.
    Byte[] badItem = concat(box(tag(1, WT_64BIT)), box(encodeFixed64(123L)));
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(badItem));
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).repeated().down()
            .addField(1, DType.INT32)
        .up()
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedItems = ColumnVector.fromLists(
             new ListType(true,
                 new StructType(true, new BasicType(true, DType.INT32))),
             Arrays.asList(struct((Object) null)));
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedItems);
         ColumnVector actualPermissive = Protobuf.decodeToStruct(
             input.getColumn(0), schema, false);
         ColumnVector actualFailfast = Protobuf.decodeToStruct(
             input.getColumn(0), schema, true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualPermissive);
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualFailfast);
    }
  }

  // ============================================================================
  // FAILFAST Mode Tests (failOnErrors = true)
  // ============================================================================

  @Test
  void testMalformedVarint_Failfast() {
    // Varint that never terminates (all continuation bits set)
    Byte[] malformed = new Byte[]{(byte)0x08, (byte)0xFF, (byte)0xFF, (byte)0xFF,
                                   (byte)0xFF, (byte)0xFF, (byte)0xFF,
                                   (byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0xFF};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{malformed}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.INT64)
                .build(),
            true)) {  // failOnErrors = true
        }
      });
    }
  }

  @Test
  void testTruncatedVarint_Failfast() {
    // Single byte with continuation bit set but no following byte
    Byte[] truncated = concat(box(tag(1, WT_VARINT)), new Byte[]{(byte)0x80});
    try (Table input = new Table.TestBuilder().column(new Byte[][]{truncated}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.INT64)
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testTruncatedString_Failfast() {
    // String field with length=5 but no actual data
    Byte[] truncated = concat(box(tag(2, WT_LEN)), box(encodeVarint(5)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{truncated}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(2, DType.STRING)
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testTruncatedFixed32_Failfast() {
    // Fixed32 needs 4 bytes but only 3 provided
    Byte[] truncated = concat(box(tag(1, WT_32BIT)), new Byte[]{0x01, 0x02, 0x03});
    try (Table input = new Table.TestBuilder().column(new Byte[][]{truncated}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.INT32).encoding(Protobuf.ENC_FIXED)
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testTruncatedFixed64_Failfast() {
    // Fixed64 needs 8 bytes but only 5 provided
    Byte[] truncated = concat(box(tag(1, WT_64BIT)), new Byte[]{0x01, 0x02, 0x03, 0x04, 0x05});
    try (Table input = new Table.TestBuilder().column(new Byte[][]{truncated}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.INT64).encoding(Protobuf.ENC_FIXED)
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testWrongWireType_Failfast() {
    // Field 1 with wire type 2 (length-delimited), but we request varint
    Byte[] row = concat(box(tag(1, WT_LEN)), box(encodeVarint(3)), box("abc".getBytes()));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.INT64)
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testFieldNumberZero_Failfast() {
    // Field number 0 is invalid in protobuf
    Byte[] row = concat(box(tag(0, WT_VARINT)), box(encodeVarint(42)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.INT64)
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testFieldNumberAboveSpecLimit_Failfast() {
    // Protobuf field numbers must be <= 2^29 - 1.
    Byte[] row = concat(box(tag(1 << 29, WT_VARINT)), box(encodeVarint(42)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.INT64)
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testUnknownEndGroupWireTypeNullsMalformedRow() {
    Byte[] row = concat(
        box(tag(5, WT_EGROUP)),
        box(tag(1, WT_VARINT)), box(encodeVarint(42)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .build(),
             false)) {
      assertSingleNullStructRow(actual, "Unknown end-group wire type should null the struct row");
    }
  }

  @Test
  void testMatchingUnknownGroupIsSkipped() {
    Byte[] row = concat(
        box(tag(5, WT_SGROUP)),
        box(tag(7, WT_VARINT)), box(encodeVarint(99)),
        box(tag(5, WT_EGROUP)),
        box(tag(1, WT_VARINT)), box(encodeVarint(42)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedValue = ColumnVector.fromBoxedLongs(42L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedValue);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder().addField(1, DType.INT64).build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actual);
    }
  }

  @Test
  void testMismatchedEndGroupNullsOrFails() {
    Byte[] row = concat(
        box(tag(5, WT_SGROUP)),
        box(tag(7, WT_VARINT)), box(encodeVarint(99)),
        box(tag(6, WT_EGROUP)),
        box(tag(1, WT_VARINT)), box(encodeVarint(42)));
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.INT64)
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actual = Protobuf.decodeToStruct(input.getColumn(0), schema, false)) {
      assertSingleNullStructRow(actual, "Mismatched end-group should null the struct row");
    }
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
        }
      });
    }
  }

  @Test
  void testUnknownGroupAtCpuRecursionLimitIsSkipped() {
    Byte[] row = concat(
        wrapInUnknownGroups(new Byte[0], PROTOBUF_JAVA_RECURSION_LIMIT),
        box(tag(1, WT_VARINT)), box(encodeVarint(42)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedValue = ColumnVector.fromBoxedLongs(42L);
         ColumnVector expected = ColumnVector.makeStruct(expectedValue);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testUnknownGroupBeyondCpuRecursionLimit_Permissive() {
    Byte[] row = wrapInUnknownGroups(new Byte[0], PROTOBUF_JAVA_RECURSION_LIMIT + 1);
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder().build(),
             false)) {
      assertSingleNullStructRow(actual, "Group nesting beyond the CPU limit should null the row");
    }
  }

  @Test
  void testUnknownGroupBeyondCpuRecursionLimit_Failfast() {
    Byte[] row = wrapInUnknownGroups(new Byte[0], PROTOBUF_JAVA_RECURSION_LIMIT + 1);
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(
            input.getColumn(0), new ProtobufSchemaDescriptorBuilder().build(), true)) {
        }
      });
    }
  }

  @Test
  void testUnknownGroupAtCpuRecursionLimitInsideNestedMessageIsSkipped() {
    Byte[] inner = concat(
        wrapInUnknownGroups(new Byte[0], PROTOBUF_JAVA_RECURSION_LIMIT - 1),
        box(tag(1, WT_VARINT)), box(encodeVarint(42)));
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(inner));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedValue = ColumnVector.fromBoxedLongs(42L);
         ColumnVector expectedInner = ColumnVector.makeStruct(expectedValue);
         ColumnVector expectedOuter = ColumnVector.makeStruct(expectedInner);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.INT64)
                 .up()
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actual);
    }
  }

  @Test
  void testUnknownGroupBeyondCpuRecursionLimitInsideNestedMessage_Permissive() {
    Byte[] inner = wrapInUnknownGroups(new Byte[0], PROTOBUF_JAVA_RECURSION_LIMIT);
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(inner));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.INT64)
                 .up()
                 .build(),
             false)) {
      assertSingleNullStructRow(
          actual, "Nested message depth must count toward the protobuf recursion limit");
    }
  }

  @Test
  void testUnknownGroupBeyondCpuRecursionLimitInsideNestedMessage_Failfast() {
    Byte[] inner = wrapInUnknownGroups(new Byte[0], PROTOBUF_JAVA_RECURSION_LIMIT);
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(inner));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.STRUCT).down()
                    .addField(1, DType.INT64)
                .up()
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testValidDataDoesNotThrow_Failfast() {
    // Valid protobuf should not throw even with failOnErrors = true
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(42)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .build(),
             true)) {
      try (ColumnVector expected = ColumnVector.fromBoxedLongs(42L);
           ColumnVector expectedStruct = ColumnVector.makeStruct(expected)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStruct, result);
      }
    }
  }

  // ============================================================================
  // Performance Benchmark Tests (Multi-field)
  // ============================================================================

  @Test
  void testMultiFieldPerformance() {
    // Test with 6 fields to verify fused kernel efficiency
    // message Msg { bool f1=1; int32 f2=2; int64 f3=3; float f4=4; double f5=5; string f6=6; }
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), new Byte[]{0x01},
        box(tag(2, WT_VARINT)), box(encodeVarint(12345)),
        box(tag(3, WT_VARINT)), box(encodeVarint(9876543210L)),
        box(tag(4, WT_32BIT)), box(encodeFloat(3.14f)),
        box(tag(5, WT_64BIT)), box(encodeDouble(2.71828)),
        box(tag(6, WT_LEN)), box(encodeVarint(5)), box("hello".getBytes()));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.BOOL8)
                 .addField(2, DType.INT32)
                 .addField(3, DType.INT64)
                 .addField(4, DType.FLOAT32)
                 .addField(5, DType.FLOAT64)
                 .addField(6, DType.STRING)
                 .build(),
             true)) {
      try (ColumnVector expectedBool = ColumnVector.fromBoxedBooleans(true);
           ColumnVector expectedInt = ColumnVector.fromBoxedInts(12345);
           ColumnVector expectedLong = ColumnVector.fromBoxedLongs(9876543210L);
           ColumnVector expectedFloat = ColumnVector.fromBoxedFloats(3.14f);
           ColumnVector expectedDouble = ColumnVector.fromBoxedDoubles(2.71828);
           ColumnVector expectedString = ColumnVector.fromStrings("hello");
           ColumnVector expectedStruct = ColumnVector.makeStruct(
               expectedBool, expectedInt, expectedLong, expectedFloat, expectedDouble, expectedString)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
      }
    }
  }

  // ============================================================================
  // Enum Validation Tests
  // ============================================================================

  @Test
  void testEnumAsStringValidValue() {
    // enum Color { RED=0; GREEN=1; BLUE=2; }
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(1)));  // GREEN

    byte[][][] enumNames = new byte[][][]{
        new byte[][]{
            "RED".getBytes(StandardCharsets.UTF_8),
            "GREEN".getBytes(StandardCharsets.UTF_8),
            "BLUE".getBytes(StandardCharsets.UTF_8)
        }
    };
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedField = ColumnVector.fromStrings("GREEN");
         ColumnVector expected = ColumnVector.makeStruct(expectedField);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING).enumMetadata(new int[]{0, 1, 2}, enumNames[0])
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testEnumAsStringUnknownValueReturnsNullRow() {
    // Unknown enum value should null the entire struct row (PERMISSIVE behavior).
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(999)));

    byte[][][] enumNames = new byte[][][]{
        new byte[][]{
            "RED".getBytes(StandardCharsets.UTF_8),
            "GREEN".getBytes(StandardCharsets.UTF_8),
            "BLUE".getBytes(StandardCharsets.UTF_8)
        }
    };
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expected = ColumnVector.fromStructs(
             new StructType(true, new BasicType(true, DType.STRING)), (StructData) null);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING).enumMetadata(new int[]{0, 1, 2}, enumNames[0])
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testEnumAsStringUnknownValue_Failfast() {
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(999)));
    byte[][] enumNames = new byte[][]{
        "RED".getBytes(StandardCharsets.UTF_8),
        "GREEN".getBytes(StandardCharsets.UTF_8),
        "BLUE".getBytes(StandardCharsets.UTF_8)
    };

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.STRING).enumMetadata(new int[]{0, 1, 2}, enumNames)
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testRequiredEnumAsStringUnknownValueReportsMissingRequired_Failfast() {
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(999)));
    byte[][] enumNames = new byte[][]{
        "RED".getBytes(StandardCharsets.UTF_8),
        "GREEN".getBytes(StandardCharsets.UTF_8),
        "BLUE".getBytes(StandardCharsets.UTF_8)
    };

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      ai.rapids.cudf.CudfException error = assertThrows(
          ai.rapids.cudf.CudfException.class,
          () -> {
            try (ColumnVector ignored = Protobuf.decodeToStruct(
                input.getColumn(0),
                new ProtobufSchemaDescriptorBuilder()
                    .addField(1, DType.STRING).required()
                        .enumMetadata(new int[]{0, 1, 2}, enumNames)
                    .build(),
                true)) {
            }
          });
      assertTrue(error.getMessage().contains("missing required field"));
    }
  }

  @Test
  void testEnumAsStringMixedValidAndUnknown() {
    Byte[] row0 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(0)));    // RED
    Byte[] row1 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(999)));  // unknown
    Byte[] row2 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(2)));    // BLUE

    byte[][] enumNames = new byte[][]{
        "RED".getBytes(StandardCharsets.UTF_8),
        "GREEN".getBytes(StandardCharsets.UTF_8),
        "BLUE".getBytes(StandardCharsets.UTF_8)
    };
    try (Table input = new Table.TestBuilder().column(row0, row1, row2).build();
         ColumnVector expected = ColumnVector.fromStructs(
             new StructType(true, new BasicType(true, DType.STRING)),
             struct("RED"), null, struct("BLUE"));
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING).enumMetadata(new int[]{0, 1, 2}, enumNames)
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testEnumValidValue() {
    // enum Color { RED=0; GREEN=1; BLUE=2; }
    // message Msg { Color color = 1; }
    // Test with valid enum value (GREEN = 1)
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(1)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedColor = ColumnVector.fromBoxedInts(1);  // GREEN
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedColor);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).enumValidValues(new int[]{0, 1, 2})
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testEnumUnknownValueReturnsNullRow() {
    // enum Color { RED=0; GREEN=1; BLUE=2; }
    // message Msg { Color color = 1; }
    // Test with unknown enum value (999 is not defined)
    // The entire struct row should be null (matching Spark CPU PERMISSIVE mode)
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(999)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expected = ColumnVector.fromStructs(
             new StructType(true, new BasicType(true, DType.INT32)), (StructData) null);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).enumValidValues(new int[]{0, 1, 2})
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actualStruct);
    }
  }

  @Test
  void testEnumUnknownValue_Failfast() {
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),
        box(tag(2, WT_VARINT)), box(encodeVarint(42)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.INT32).enumValidValues(new int[]{0, 1, 2})
                .addField(2, DType.INT32)
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testEnumMixedValidAndUnknown() {
    // Test multiple rows with mix of valid and unknown enum values
    // Rows with unknown enum values should have null struct (not just null field)
    Byte[] row0 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(0)));    // RED (valid) -> struct valid
    Byte[] row1 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(999)));  // unknown -> struct null
    Byte[] row2 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(2)));    // BLUE (valid) -> struct valid
    Byte[] row3 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(-1)));   // negative (unknown) -> struct null

    try (Table input = new Table.TestBuilder().column(row0, row1, row2, row3).build();
         ColumnVector expected = ColumnVector.fromStructs(
             new StructType(true, new BasicType(true, DType.INT32)),
             struct(0), null, struct(2), null);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).enumValidValues(new int[]{0, 1, 2})
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actualStruct);
    }
  }

  @Test
  void testTopLevelEnumAnyUnknownOccurrenceInvalidatesRow() {
    Byte[] validThenUnknown = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),
        box(tag(2, WT_VARINT)), box(encodeVarint(20)));
    Byte[] unknownThenValid = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)),
        box(tag(2, WT_VARINT)), box(encodeVarint(30)));
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.INT32).enumValidValues(new int[]{0, 1, 2})
        .addField(2, DType.INT32)
        .build();
    StructType outputType = new StructType(
        true, new BasicType(true, DType.INT32), new BasicType(true, DType.INT32));

    try (Table input = new Table.TestBuilder().column(validThenUnknown, unknownThenValid).build();
         ColumnVector expected = ColumnVector.fromStructs(outputType, null, null);
         ColumnVector actual = Protobuf.decodeToStruct(input.getColumn(0), schema, false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
        }
      });
    }

    ProtobufSchemaDescriptor requiredSchema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.INT32).required().enumValidValues(new int[]{0, 1, 2})
        .addField(2, DType.INT32)
        .build();
    try (Table input = new Table.TestBuilder().column(validThenUnknown, unknownThenValid).build()) {
      ai.rapids.cudf.CudfException error = assertThrows(
          ai.rapids.cudf.CudfException.class,
          () -> {
            try (ColumnVector ignored = Protobuf.decodeToStruct(
                input.getColumn(0), requiredSchema, true)) {
            }
          });
      assertTrue(error.getMessage().contains("unknown enum value"));
    }
  }

  @Test
  void testMalformedWirePrecedesDeferredUnknownRootEnum_Failfast() {
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),
        box(tag(2, WT_LEN)), new Byte[]{(byte) 0x80});
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.INT32).enumValidValues(new int[]{0, 1, 2})
        .addField(2, DType.STRING)
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      ai.rapids.cudf.CudfException error = assertThrows(
          ai.rapids.cudf.CudfException.class,
          () -> {
            try (ColumnVector ignored = Protobuf.decodeToStruct(
                input.getColumn(0), schema, true)) {
            }
          });
      assertTrue(error.getMessage().contains("invalid or truncated varint"));
      assertFalse(error.getMessage().contains("unknown enum value"));
    }
  }

  @Test
  void testEnumWithOtherFields_NullsEntireRow() {
    // message Msg { Color color = 1; int32 count = 2; }
    // Test that unknown enum value nulls the ENTIRE struct row (not just the enum field)
    // This matches Spark CPU PERMISSIVE mode behavior
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),  // unknown enum value
        box(tag(2, WT_VARINT)), box(encodeVarint(42)));  // count = 42

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expected = ColumnVector.fromStructs(
             new StructType(
                 true,
                 new BasicType(true, DType.INT32),
                 new BasicType(true, DType.INT32)),
             (StructData) null);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).enumValidValues(new int[]{0, 1, 2})
                 .addField(2, DType.INT32)
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actualStruct);
    }
  }

  @Test
  void testRepeatedEnumUnknownValueReturnsNullRow() {
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)),
        box(tag(2, WT_VARINT)), box(encodeVarint(42)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).repeated().enumValidValues(new int[]{0, 1, 2})
                 .addField(2, DType.INT32)
                 .build(),
             false)) {
      assertSingleNullStructRow(
          actual, "Unknown top-level repeated enum should null the row in PERMISSIVE mode");
    }
  }

  @Test
  void testRepeatedEnumValidValues() {
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedValues = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)), Arrays.asList(1, 2));
         ColumnVector expected = ColumnVector.makeStruct(expectedValues);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).repeated().enumValidValues(new int[]{0, 1, 2})
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testRepeatedEnumUnknownValue_Failfast() {
    byte[] packedValues = concatBytes(encodeVarint(1), encodeVarint(999), encodeVarint(2));
    Byte[] row = concat(
        box(tag(1, WT_LEN)), encodeBytes(packedValues),
        box(tag(2, WT_VARINT)), box(encodeVarint(42)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.INT32).repeated().enumValidValues(new int[]{0, 1, 2})
                .addField(2, DType.INT32)
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testRepeatedStructEnumInvalidKeepsTopLevelRowValid() {
    // enum Color { RED=0; GREEN=1; BLUE=2; }
    // message Item { Color color = 1; }
    // message Msg { repeated Item items = 1; }
    Byte[] item00 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(0)));    // valid
    Byte[] item01 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(999)));  // invalid
    Byte[] row0 = concat(
        box(tag(1, WT_LEN)), encodeMessage(item00),
        box(tag(1, WT_LEN)), encodeMessage(item01));
    Byte[] item10 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(1)));    // valid
    Byte[] row1 = concat(
        box(tag(1, WT_LEN)), encodeMessage(item10));

    try (Table input = new Table.TestBuilder().column(row0, row1).build();
         ColumnVector expectedItems = ColumnVector.fromLists(
             new ListType(true, new StructType(true, new BasicType(true, DType.INT32))),
             Arrays.asList(struct(0), struct((Object) null)),
             Collections.singletonList(struct(1)));
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedItems);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).repeated().down()
                     .addField(1, DType.INT32).enumValidValues(new int[]{0, 1, 2})
                 .up()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testRepeatedStructEnumInvalidKeepsSiblingFieldsVisible_Failfast() {
    // enum Color { RED=0; GREEN=1; BLUE=2; }
    // message Item { Color color = 1; int32 count = 2; }
    // message Msg { repeated Item items = 1; }
    Byte[] item00 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(0)),
        box(tag(2, WT_VARINT)), box(encodeVarint(10)));
    Byte[] item01 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),
        box(tag(2, WT_VARINT)), box(encodeVarint(20)));
    Byte[] row0 = concat(
        box(tag(1, WT_LEN)), encodeMessage(item00),
        box(tag(1, WT_LEN)), encodeMessage(item01));
    Byte[] item10 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(2, WT_VARINT)), box(encodeVarint(30)));
    Byte[] row1 = concat(
        box(tag(1, WT_LEN)), encodeMessage(item10));

    try (Table input = new Table.TestBuilder().column(row0, row1).build();
         ColumnVector expectedItems = ColumnVector.fromLists(
             new ListType(true,
                 new StructType(true,
                     new BasicType(true, DType.INT32),
                     new BasicType(true, DType.INT32))),
             Arrays.asList(struct(0, 10), struct(null, 20)),
             Collections.singletonList(struct(1, 30)));
         ColumnVector expected = ColumnVector.makeStruct(expectedItems);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).repeated().down()
                     .addField(1, DType.INT32).enumValidValues(new int[]{0, 1, 2})
                     .addField(2, DType.INT32)
                 .up()
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testEnumMissingFieldDoesNotNullRow() {
    // Missing enum field should return null for the field, but NOT null the entire row
    // Only unknown enum values (present but invalid) trigger row-level null
    Byte[] row = new Byte[0];  // empty message

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedColor = ColumnVector.fromBoxedInts((Integer) null);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedColor);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).enumValidValues(new int[]{0, 1, 2})
                 .build(),
             true)) {
      // Struct row should be valid (not null), only the field is null
      assertEquals(0, actualStruct.getNullCount(), "Struct row should NOT be null for missing field");
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testNestedEnumInvalidKeepsRowAndSiblingFieldsInPermissiveMode() {
    // message WithNestedEnum {
    //   optional int32 id = 1;
    //   optional Detail detail = 2;
    //   optional string name = 3;
    // }
    // message Detail {
    //   enum Status { UNKNOWN = 0; OK = 1; BAD = 2; }
    //   optional Status status = 1;
    //   optional int32 count = 2;
    // }
    // Invalid enum value inside a nested struct: only the enum field becomes null.
    Byte[] detail = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),
        box(tag(2, WT_VARINT)), box(encodeVarint(20)));
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(2)),
        box(tag(2, WT_LEN)), encodeMessage(detail),
        box(tag(3, WT_LEN)), encodeString("bad"));

    byte[][] enumNames = new byte[][]{
        "UNKNOWN".getBytes(StandardCharsets.UTF_8),
        "OK".getBytes(StandardCharsets.UTF_8),
        "BAD".getBytes(StandardCharsets.UTF_8)
    };

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedId = ColumnVector.fromBoxedInts(2);
         ColumnVector expectedStatus = ColumnVector.fromStrings((String) null);
         ColumnVector expectedCount = ColumnVector.fromBoxedInts(20);
         ColumnVector expectedDetail =
             ColumnVector.makeStruct(expectedStatus, expectedCount);
         ColumnVector expectedName = ColumnVector.fromStrings("bad");
         ColumnVector expected =
             ColumnVector.makeStruct(expectedId, expectedDetail, expectedName);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32)
                 .addField(2, DType.STRUCT).down()
                     .addField(1, DType.STRING).enumMetadata(
                         new int[]{0, 1, 2}, enumNames)
                     .addField(2, DType.INT32)
                 .up()
                 .addField(3, DType.STRING)
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testMalformedNestedEnumPermissiveNullsWholeRow() {
    // message WithNestedEnum {
    //   optional int32 id = 1;
    //   optional Detail detail = 2;
    //   optional string name = 3;
    // }
    // message Detail {
    //   enum Status { UNKNOWN = 0; OK = 1; BAD = 2; }
    //   optional Status status = 1;
    //   optional int32 count = 2;
    // }
    //
    // The nested message length is intentionally truncated to 4 bytes. Spark CPU treats this as a
    // malformed row in PERMISSIVE mode and returns a null struct row rather than partial data.
    Byte[] rowValid = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(2, WT_LEN)), box(encodeVarint(4)),
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(2, WT_VARINT)), box(encodeVarint(10)),
        box(tag(3, WT_LEN)), box(encodeVarint(2)), box("ok".getBytes()));
    Byte[] rowInvalid = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(2)),
        box(tag(2, WT_LEN)), box(encodeVarint(4)),
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),
        box(tag(2, WT_VARINT)), box(encodeVarint(20)),
        box(tag(3, WT_LEN)), box(encodeVarint(3)), box("bad".getBytes()));

    StructType detailType = new StructType(
        true,
        new BasicType(true, DType.INT32),
        new BasicType(true, DType.INT32));
    StructType outerType = new StructType(
        true,
        new BasicType(true, DType.INT32),
        detailType,
        new BasicType(true, DType.STRING));
    try (Table input = new Table.TestBuilder().column(rowValid, rowInvalid).build();
         ColumnVector expected = ColumnVector.fromStructs(
             outerType, struct(1, struct(1, 10), "ok"), null);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32)
                 .addField(2, DType.STRUCT).down()
                     .addField(1, DType.INT32).enumValidValues(new int[]{0, 1, 2})
                     .addField(2, DType.INT32)
                 .up()
                 .addField(3, DType.STRING)
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testEnumValidWithOtherFields() {
    // message Msg { Color color = 1; int32 count = 2; }
    // Test that valid enum value works correctly with other fields
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),    // GREEN (valid)
        box(tag(2, WT_VARINT)), box(encodeVarint(42)));  // count = 42

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedColor = ColumnVector.fromBoxedInts(1);
         ColumnVector expectedCount = ColumnVector.fromBoxedInts(42);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedColor, expectedCount);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).enumValidValues(new int[]{0, 1, 2})
                 .addField(2, DType.INT32)
                 .build(),
             false)) {
      // Struct row should be valid with correct values
      assertEquals(0, actualStruct.getNullCount(), "Struct row should be valid");
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Repeated Enum-as-String Tests
  // ============================================================================

  @Test
  void testRepeatedEnumAsString() {
    // repeated Color colors = 1; with Color { RED=0; GREEN=1; BLUE=2; }
    // Row with three occurrences: RED, BLUE, GREEN
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(0)),   // RED
        box(tag(1, WT_VARINT)), box(encodeVarint(2)),   // BLUE
        box(tag(1, WT_VARINT)), box(encodeVarint(1)));  // GREEN

    byte[][][] enumNames = new byte[][][]{
        new byte[][]{
            "RED".getBytes(StandardCharsets.UTF_8),
            "GREEN".getBytes(StandardCharsets.UTF_8),
            "BLUE".getBytes(StandardCharsets.UTF_8)
        }
    };
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedColors = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.STRING)),
             Arrays.asList("RED", "BLUE", "GREEN"));
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedColors);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING).repeated().enumMetadata(new int[]{0, 1, 2}, enumNames[0])
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testRepeatedEnumAsStringUnknownValue_Failfast() {
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)));
    byte[][] enumNames = new byte[][]{
        "UNKNOWN".getBytes(StandardCharsets.UTF_8),
        "FOO".getBytes(StandardCharsets.UTF_8),
        "BAR".getBytes(StandardCharsets.UTF_8)
    };

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.STRING).repeated()
                    .enumMetadata(new int[]{0, 1, 2}, enumNames)
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testRepeatedEnumAsStringUnknownValueReturnsNullRow() {
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)));
    byte[][] enumNames = new byte[][]{
        "UNKNOWN".getBytes(StandardCharsets.UTF_8),
        "FOO".getBytes(StandardCharsets.UTF_8),
        "BAR".getBytes(StandardCharsets.UTF_8)
    };

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING).repeated()
                     .enumMetadata(new int[]{0, 1, 2}, enumNames)
                 .build(),
             false)) {
      assertSingleNullStructRow(
          actual, "Unknown top-level repeated enum should null the row in PERMISSIVE mode");
    }
  }

  @Test
  void testRepeatedMessageChildEnumAsString() {
    // message Item { optional Priority priority = 1; }
    // message Outer { repeated Item items = 1; }
    // enum Priority { UNKNOWN=0; FOO=1; BAR=2; }
    Byte[] item0 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(1)));  // FOO
    Byte[] item1 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(2)));  // BAR
    Byte[] row = concat(
        box(tag(1, WT_LEN)), encodeMessage(item0),
        box(tag(1, WT_LEN)), encodeMessage(item1));

    byte[][] enumNames = new byte[][]{
        "UNKNOWN".getBytes(StandardCharsets.UTF_8),
        "FOO".getBytes(StandardCharsets.UTF_8),
        "BAR".getBytes(StandardCharsets.UTF_8)
    };

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedItems = ColumnVector.fromLists(
             new ListType(true, new StructType(true, new BasicType(true, DType.STRING))),
             Arrays.asList(struct("FOO"), struct("BAR")));
         ColumnVector expected = ColumnVector.makeStruct(expectedItems);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).repeated().down()
                     .addField(1, DType.STRING).enumMetadata(
                         new int[]{0, 1, 2}, enumNames)
                 .up()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testRepeatedMessageChildEnumAsStringInvalidKeepsRowValid_Failfast() {
    Byte[] item0 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(1)));    // FOO
    Byte[] item1 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(999)));  // invalid
    Byte[] row = concat(
        box(tag(1, WT_LEN)), encodeMessage(item0),
        box(tag(1, WT_LEN)), encodeMessage(item1));

    byte[][] enumNames = new byte[][]{
        "UNKNOWN".getBytes(StandardCharsets.UTF_8),
        "FOO".getBytes(StandardCharsets.UTF_8),
        "BAR".getBytes(StandardCharsets.UTF_8)
    };

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedItems = ColumnVector.fromLists(
             new ListType(true, new StructType(true, new BasicType(true, DType.STRING))),
             Arrays.asList(struct("FOO"), struct((Object) null)));
         ColumnVector expected = ColumnVector.makeStruct(expectedItems);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).repeated().down()
                     .addField(1, DType.STRING).enumMetadata(
                         new int[]{0, 1, 2}, enumNames)
                 .up()
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  // ============================================================================
  // Edge case and boundary tests
  // ============================================================================

  @Test
  void testPackedFixedMisaligned_Failfast() {
    byte[] packedData = new byte[]{0x01, 0x02, 0x03, 0x04, 0x05};
    Byte[] row = concat(
        box(tag(1, WT_LEN)),
        encodeBytes(packedData));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(RuntimeException.class, () -> {
        try (ColumnVector result = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.INT32).repeated().encoding(Protobuf.ENC_FIXED)
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testPackedFixedMisaligned64_Failfast() {
    byte[] packedData = new byte[]{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09};
    Byte[] row = concat(
        box(tag(1, WT_LEN)),
        encodeBytes(packedData));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(RuntimeException.class, () -> {
        try (ColumnVector result = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.INT64).repeated().encoding(Protobuf.ENC_FIXED)
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testPackedFixedMisalignedPermissive() {
    // Spark CPU nulls the malformed row in PERMISSIVE mode; a following well-formed row in the
    // same batch must still decode normally.
    byte[] badPackedData = new byte[]{0x01, 0x02, 0x03, 0x04, 0x05};
    Byte[] row0 = concat(
        box(tag(1, WT_LEN)),
        encodeBytes(badPackedData));
    Byte[] row1 = concat(
        box(tag(1, WT_32BIT)), box(encodeFixed32(42)),
        box(tag(1, WT_32BIT)), box(encodeFixed32(99)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1}).build();
         ColumnVector expectedStruct = ColumnVector.fromStructs(
             new StructType(true,
                 new ListType(true, new BasicType(true, DType.INT32))),
             null,
             struct(Arrays.asList(42, 99)));
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).repeated().encoding(Protobuf.ENC_FIXED)
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testPackedFixedMisaligned64Permissive() {
    // Spark CPU nulls the malformed row in PERMISSIVE mode.
    byte[] badPackedData = new byte[]{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09};
    Byte[] row0 = concat(
        box(tag(1, WT_LEN)),
        encodeBytes(badPackedData));
    Byte[] row1 = concat(
        box(tag(1, WT_64BIT)), box(encodeFixed64(7L)),
        box(tag(1, WT_64BIT)), box(encodeFixed64(11L)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1}).build();
         ColumnVector expectedStruct = ColumnVector.fromStructs(
             new StructType(true,
                 new ListType(true, new BasicType(true, DType.INT64))),
             null,
             struct(Arrays.asList(7L, 11L)));
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64).repeated().encoding(Protobuf.ENC_FIXED)
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testMalformedZeroCountRepeatedFieldBeforeLaterField_Permissive() {
    Byte[] invalidRow = concat(
        box(tag(1, WT_LEN)), encodeBytes(new byte[]{(byte) 0x80}),
        box(tag(2, WT_VARINT)), box(encodeVarint(11)));
    Byte[] validRow = concat(box(tag(2, WT_VARINT)), box(encodeVarint(22)));
    StructType expectedType = new StructType(
        true,
        new ListType(true, new BasicType(true, DType.INT32)),
        new ListType(true, new BasicType(true, DType.INT32)));

    try (Table input = new Table.TestBuilder().column(invalidRow, validRow).build();
         ColumnVector expected = ColumnVector.fromStructs(
             expectedType,
             null,
             struct(Collections.emptyList(), Collections.singletonList(22)));
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).repeated()
                 .addField(2, DType.INT32).repeated()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testLargeRepeatedField() throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    Integer[] expectedValues = new Integer[100000];
    for (int i = 0; i < 100000; i++) {
      baos.write(tag(1, WT_VARINT));
      baos.write(encodeVarint(i));
      expectedValues[i] = i;
    }
    Byte[] row = box(baos.toByteArray());

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedIds = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)),
             Arrays.asList(expectedValues));
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedIds);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).repeated()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testMultiFieldOutputShape() {
    Byte[] row = new Byte[]{0x08, 0x01};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .addField(2, DType.STRING)
                 .addField(3, DType.FLOAT32)
                 .build(),
             true)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(1, result.getRowCount());
      assertEquals(3, result.getNumChildren());
      assertEquals(DType.INT64, result.getChildColumnView(0).getType());
      assertEquals(DType.STRING, result.getChildColumnView(1).getType());
      assertEquals(DType.FLOAT32, result.getChildColumnView(2).getType());
    }
  }

  @Test
  void testMultipleRowsOutputShape() {
    Byte[] row0 = new Byte[]{0x08, 0x01};
    Byte[] row1 = new Byte[]{0x08, 0x02};
    Byte[] row2 = new Byte[]{0x08, 0x03};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1, row2}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .build(), true)) {
      assertEquals(3, result.getRowCount());
      assertEquals(1, result.getNumChildren());
    }
  }

  // ============================================================================
  // Null input handling
  // ============================================================================

  @Test
  void testNullInputRowProducesNullStructRow() {
    Byte[] row0 = new Byte[]{0x08, 0x01};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, null}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .build(), true)) {
      assertEquals(2, result.getRowCount());
      try (HostColumnVector hcv = result.copyToHost()) {
        assertFalse(hcv.isNull(0), "Row 0 should not be null");
        assertTrue(hcv.isNull(1), "Row 1 (null input) should be null in output struct");
      }
    }
  }

  @Test
  void testMixedPackedUnpacked() {
    byte[] packedContent = concatBytes(encodeVarint(30), encodeVarint(40));
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(10)),
        box(tag(1, WT_VARINT)), box(encodeVarint(20)),
        box(tag(1, WT_LEN)), encodeBytes(packedContent));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedValues = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)),
             Arrays.asList(10, 20, 30, 40));
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedValues);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).repeated()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testAllNullInputRows() {
    try (Table input = new Table.TestBuilder().column(new Byte[][]{null, null, null}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .addField(2, DType.STRING)
                 .build(), true)) {
      assertEquals(3, result.getRowCount());
      assertEquals(2, result.getNumChildren());
      try (HostColumnVector hcv = result.copyToHost()) {
        for (int row = 0; row < 3; row++) {
          assertTrue(hcv.isNull(row), "Row " + row + " should be null");
        }
      }
    }
  }

  @Test
  void testLargeFieldNumber() {
    int maxFieldNumber = (1 << 29) - 1;
    Byte[] row = concat(
        box(tag(maxFieldNumber, WT_VARINT)),
        box(encodeVarint(42)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedValue = ColumnVector.fromBoxedInts(42);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedValue);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(maxFieldNumber, DType.INT32)
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Empty-row (0 rows) handling
  // ============================================================================

  @Test
  void testZeroRowInput() {
    try (Table input = new Table.TestBuilder().column(new Byte[][]{}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .addField(2, DType.STRING)
                 .build(), true)) {
      assertEquals(0, result.getRowCount());
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(2, result.getNumChildren());
      assertEquals(DType.INT64, result.getChildColumnView(0).getType());
      assertEquals(DType.STRING, result.getChildColumnView(1).getType());
    }
  }

  // ============================================================================
  // Nested schema shape tests (verifies correct column types without decode)
  // ============================================================================

  @Test
  void testNestedMessageOutputShape() {
    // Schema: message Outer { int32 a = 1; Inner b = 2; } message Inner { int32 x = 1; }
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.INT32)            // a
        .addField(2, DType.STRUCT).down()    // b
            .addField(1, DType.INT32)        // x
        .up()
        .build();

    Byte[] row = new Byte[]{0x08, 0x01};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(1, result.getRowCount());
      assertEquals(2, result.getNumChildren());
      assertEquals(DType.INT32, result.getChildColumnView(0).getType());
      assertEquals(DType.STRUCT, result.getChildColumnView(1).getType());
      assertEquals(1, result.getChildColumnView(1).getNumChildren());
      assertEquals(DType.INT32, result.getChildColumnView(1).getChildColumnView(0).getType());
    }
  }

  @Test
  void testDeepNesting9Levels() {
    verifyDeepNesting(9);
  }

  @Test
  void testDeepNesting10Levels() {
    verifyDeepNesting(10);
  }

  // Build a chain of (levels-1) STRUCT fields followed by one INT32 leaf, so the
  // deepest field sits at depth (levels-1). Both 9 and 10 are within MAX_NESTING_DEPTH=10.
  private void verifyDeepNesting(int levels) {
    ProtobufSchemaDescriptorBuilder builder = new ProtobufSchemaDescriptorBuilder();
    for (int i = 0; i < levels; i++) {
      DType type = (i < levels - 1) ? DType.STRUCT : DType.INT32;
      builder.addField(1, type);
      if (i > 0) {
        builder.parent(i - 1);
      }
    }
    ProtobufSchemaDescriptor schema = builder.build();
    Byte[] row = new Byte[0];
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0), schema, false)) {
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(1, result.getRowCount());
    }
  }

  @Test
  void testRepeatedFieldOutputShape() {
    // Schema: message Msg { repeated int32 values = 1; }
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.INT32).repeated()
        .build();

    Byte[] row = new Byte[]{0x08, 0x01};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(1, result.getRowCount());
      assertEquals(1, result.getNumChildren());
      assertEquals(DType.LIST, result.getChildColumnView(0).getType());
    }
  }

  @Test
  void testEmptyPackedRepeated() {
    Byte[] row = concat(
        box(tag(1, WT_LEN)),
        box(encodeVarint(0)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).repeated()
                 .build(),
             false)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
    }
  }

  @Test
  void testNestedMessageInt32Child() {
    // message Inner { int32 x = 1; }
    // message Outer { Inner inner = 1; }
    Byte[] innerMessage = concat(box(tag(1, WT_VARINT)), box(encodeVarint(42)));
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(innerMessage));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.INT32)
                 .up()
                 .build(),
             false)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
      try (ColumnVector expectedX = ColumnVector.fromBoxedInts(42);
           ColumnVector expectedInner = ColumnVector.makeStruct(expectedX);
           ColumnVector expectedOuter = ColumnVector.makeStruct(expectedInner)) {
        AssertUtils.assertStructColumnsAreEqual(expectedOuter, result);
      }
    }
  }

  @Test
  void testNestedMessageDuplicateFieldTags_LastOneWins() {
    // message Inner { int32 x = 1; }
    // message Outer { Inner inner = 1; }
    Byte[] innerMessage = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)));
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(innerMessage));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.INT32)
                 .up()
                 .build(),
             false)) {
      try (ColumnVector expectedX = ColumnVector.fromBoxedInts(2);
           ColumnVector expectedInner = ColumnVector.makeStruct(expectedX);
           ColumnVector expectedOuter = ColumnVector.makeStruct(expectedInner)) {
        AssertUtils.assertStructColumnsAreEqual(expectedOuter, result);
      }
    }
  }

  @Test
  void testDuplicateSingularMessageOccurrencesMergeInBothModes() {
    Byte[] firstFragment = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(3, WT_VARINT)), box(encodeVarint(10)));
    Byte[] secondFragment = concat(
        box(tag(2, WT_VARINT)), box(encodeVarint(2)),
        box(tag(3, WT_VARINT)), box(encodeVarint(20)));
    Byte[] sameFieldFirst = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(2, WT_VARINT)), box(encodeVarint(9)));
    Byte[] sameFieldSecond = concat(box(tag(1, WT_VARINT)), box(encodeVarint(2)));
    Byte[] singleFragment = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(7)),
        box(tag(2, WT_VARINT)), box(encodeVarint(8)));
    Byte[][] rows = new Byte[][]{
        concat(
            box(tag(1, WT_LEN)), encodeMessage(firstFragment),
            box(tag(1, WT_LEN)), encodeMessage(secondFragment)),
        concat(
            box(tag(1, WT_LEN)), encodeMessage(sameFieldFirst),
            box(tag(1, WT_LEN)), encodeMessage(sameFieldSecond)),
        concat(box(tag(1, WT_LEN)), encodeMessage(singleFragment)),
        new Byte[]{},
        null};
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).down()
            .addField(1, DType.INT32).required()
            .addField(2, DType.INT32).required()
            .addField(3, DType.INT32).repeated()
        .up()
        .build();

    StructType childType = new StructType(true,
        new BasicType(true, DType.INT32),
        new BasicType(true, DType.INT32),
        new ListType(true, new BasicType(true, DType.INT32)));
    try (Table input = new Table.TestBuilder().column(rows).build();
         ColumnVector expected = ColumnVector.fromStructs(
             new StructType(true, childType),
             struct(struct(1, 2, Arrays.asList(10, 20))),
             struct(struct(2, 9, Collections.emptyList())),
             struct(struct(7, 8, Collections.emptyList())),
             struct((Object) null),
             (StructData) null);
         ColumnVector actualPermissive = Protobuf.decodeToStruct(
             input.getColumn(0), schema, false);
         ColumnVector actualFailfast = Protobuf.decodeToStruct(
             input.getColumn(0), schema, true)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actualPermissive);
      AssertUtils.assertStructColumnsAreEqual(expected, actualFailfast);
    }
  }

  @Test
  void testDuplicateSingularMessageSlowPathPreservesPresenceAndNullsInBothModes() {
    Byte[] value1 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(1)));
    Byte[] value2 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(2)));
    Byte[][] rows = new Byte[][]{
        concat(
            box(tag(1, WT_LEN)), encodeMessage(value1),
            box(tag(1, WT_LEN)), encodeMessage(value2)),
        concat(box(tag(1, WT_LEN)), encodeMessage(new Byte[]{})),
        concat(
            box(tag(1, WT_LEN)), encodeMessage(new Byte[]{}),
            box(tag(1, WT_LEN)), encodeMessage(new Byte[]{})),
        new Byte[]{},
        null};
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).down()
            .addField(1, DType.INT32)
        .up()
        .build();
    StructType childType = new StructType(true, new BasicType(true, DType.INT32));

    try (Table input = new Table.TestBuilder().column(rows).build();
         ColumnVector expected = ColumnVector.fromStructs(
             new StructType(true, childType),
             struct(struct(2)),
             struct(struct((Object) null)),
             struct(struct((Object) null)),
             struct((Object) null),
             (StructData) null);
         ColumnVector actualPermissive = Protobuf.decodeToStruct(
             input.getColumn(0), schema, false);
         ColumnVector actualFailfast = Protobuf.decodeToStruct(
             input.getColumn(0), schema, true)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actualPermissive);
      AssertUtils.assertStructColumnsAreEqual(expected, actualFailfast);
    }
  }

  @Test
  void testDuplicateSingularMessageOccurrencesMergeRecursivelyInBothModes() {
    Byte[] childFirst = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(3, WT_VARINT)), box(encodeVarint(10)));
    Byte[] childSecond = concat(
        box(tag(2, WT_VARINT)), box(encodeVarint(2)),
        box(tag(3, WT_VARINT)), box(encodeVarint(20)));
    Byte[] parentFirst = concat(box(tag(1, WT_LEN)), encodeMessage(childFirst));
    Byte[] parentSecond = concat(box(tag(1, WT_LEN)), encodeMessage(childSecond));
    Byte[] row = concat(
        box(tag(1, WT_LEN)), encodeMessage(parentFirst),
        box(tag(1, WT_LEN)), encodeMessage(parentSecond));
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).down()
            .addField(1, DType.STRUCT).down()
                .addField(1, DType.INT32)
                .addField(2, DType.INT32)
                .addField(3, DType.INT32).repeated()
            .up()
        .up()
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedChild = ColumnVector.fromStructs(
             new StructType(true,
                 new BasicType(true, DType.INT32),
                 new BasicType(true, DType.INT32),
                 new ListType(true, new BasicType(true, DType.INT32))),
             struct(1, 2, Arrays.asList(10, 20)));
         ColumnVector expectedParent = ColumnVector.makeStruct(expectedChild);
         ColumnVector expected = ColumnVector.makeStruct(expectedParent);
         ColumnVector actualPermissive = Protobuf.decodeToStruct(
             input.getColumn(0), schema, false);
         ColumnVector actualFailfast = Protobuf.decodeToStruct(
             input.getColumn(0), schema, true)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actualPermissive);
      AssertUtils.assertStructColumnsAreEqual(expected, actualFailfast);
    }
  }

  @Test
  void testDuplicateSingularMessageOccurrencesInsideRepeatedParentMergeInBothModes() {
    Byte[] childFirst = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(3, WT_VARINT)), box(encodeVarint(10)));
    Byte[] childSecond = concat(
        box(tag(2, WT_VARINT)), box(encodeVarint(2)),
        box(tag(3, WT_VARINT)), box(encodeVarint(20)));
    Byte[] item0 = concat(
        box(tag(1, WT_LEN)), encodeMessage(childFirst),
        box(tag(1, WT_LEN)), encodeMessage(childSecond));
    Byte[] childFirst1 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(30)),
        box(tag(3, WT_VARINT)), box(encodeVarint(300)));
    Byte[] childSecond1 = concat(
        box(tag(2, WT_VARINT)), box(encodeVarint(40)),
        box(tag(3, WT_VARINT)), box(encodeVarint(400)));
    Byte[] item1 = concat(
        box(tag(1, WT_LEN)), encodeMessage(childFirst1),
        box(tag(1, WT_LEN)), encodeMessage(childSecond1));
    Byte[] childFirst2 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(50)),
        box(tag(3, WT_VARINT)), box(encodeVarint(500)));
    Byte[] childSecond2 = concat(
        box(tag(2, WT_VARINT)), box(encodeVarint(60)),
        box(tag(3, WT_VARINT)), box(encodeVarint(600)));
    Byte[] item2 = concat(
        box(tag(1, WT_LEN)), encodeMessage(childFirst2),
        box(tag(1, WT_LEN)), encodeMessage(childSecond2));
    Byte[][] rows = new Byte[][]{
        concat(
            box(tag(1, WT_LEN)), encodeMessage(item0),
            box(tag(1, WT_LEN)), encodeMessage(item1)),
        concat(box(tag(1, WT_LEN)), encodeMessage(item2))};
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).repeated().down()
            .addField(1, DType.STRUCT).down()
                .addField(1, DType.INT32)
                .addField(2, DType.INT32)
                .addField(3, DType.INT32).repeated()
            .up()
        .up()
        .build();

    StructType childType = new StructType(true,
        new BasicType(true, DType.INT32),
        new BasicType(true, DType.INT32),
        new ListType(true, new BasicType(true, DType.INT32)));
    try (Table input = new Table.TestBuilder().column(rows).build();
         ColumnVector expectedItems = ColumnVector.fromLists(
             new ListType(true, new StructType(true, childType)),
             Arrays.asList(
                 struct(struct(1, 2, Arrays.asList(10, 20))),
                 struct(struct(30, 40, Arrays.asList(300, 400)))),
             Collections.singletonList(
                 struct(struct(50, 60, Arrays.asList(500, 600)))));
         ColumnVector expected = ColumnVector.makeStruct(expectedItems);
         ColumnVector actualPermissive = Protobuf.decodeToStruct(
             input.getColumn(0), schema, false);
         ColumnVector actualFailfast = Protobuf.decodeToStruct(
             input.getColumn(0), schema, true)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actualPermissive);
      AssertUtils.assertStructColumnsAreEqual(expected, actualFailfast);
    }
  }

  @Test
  void testSlicedInputPreservesDuplicateSingularMessageMergeInsideRepeatedParent() {
    Byte[] child0First = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(3, WT_VARINT)), box(encodeVarint(10)));
    Byte[] child0Second = concat(
        box(tag(2, WT_VARINT)), box(encodeVarint(2)),
        box(tag(3, WT_VARINT)), box(encodeVarint(20)));
    Byte[] child1First = concat(box(tag(1, WT_VARINT)), box(encodeVarint(3)));
    Byte[] child1Second = concat(box(tag(2, WT_VARINT)), box(encodeVarint(4)));
    Byte[] child2First = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(5)),
        box(tag(3, WT_VARINT)), box(encodeVarint(50)));
    Byte[] child2Second = concat(
        box(tag(2, WT_VARINT)), box(encodeVarint(6)),
        box(tag(3, WT_VARINT)), box(encodeVarint(60)));
    Byte[] item0 = concat(
        box(tag(1, WT_LEN)), encodeMessage(child0First),
        box(tag(1, WT_LEN)), encodeMessage(child0Second));
    Byte[] item1 = concat(
        box(tag(1, WT_LEN)), encodeMessage(child1First),
        box(tag(1, WT_LEN)), encodeMessage(child1Second));
    Byte[] item2 = concat(
        box(tag(1, WT_LEN)), encodeMessage(child2First),
        box(tag(1, WT_LEN)), encodeMessage(child2Second));
    Byte[] firstSlicedRow = concat(
        box(tag(1, WT_LEN)), encodeMessage(item0),
        box(tag(1, WT_LEN)), encodeMessage(item1));
    Byte[] secondSlicedRow = concat(box(tag(1, WT_LEN)), encodeMessage(item2));
    Byte[] sentinel = concat(box(tag(99, WT_VARINT)), box(encodeVarint(7)));
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).repeated().down()
            .addField(1, DType.STRUCT).down()
                .addField(1, DType.INT32)
                .addField(2, DType.INT32)
                .addField(3, DType.INT32).repeated()
            .up()
        .up()
        .build();
    StructType childType = new StructType(true,
        new BasicType(true, DType.INT32),
        new BasicType(true, DType.INT32),
        new ListType(true, new BasicType(true, DType.INT32)));

    try (Table input = new Table.TestBuilder()
             .column(new Byte[][]{sentinel, firstSlicedRow, secondSlicedRow, sentinel})
             .build();
         ColumnVector expectedItems = ColumnVector.fromLists(
             new ListType(true, new StructType(true, childType)),
             Arrays.asList(
                 struct(struct(1, 2, Arrays.asList(10, 20))),
                 struct(struct(3, 4, Collections.emptyList()))),
             Collections.singletonList(
                 struct(struct(5, 6, Arrays.asList(50, 60)))));
         ColumnVector expected = ColumnVector.makeStruct(expectedItems)) {
      ColumnView[] views = input.getColumn(0).splitAsViews(1, 3);
      try (ColumnVector actualPermissive = Protobuf.decodeToStruct(views[1], schema, false);
           ColumnVector actualFailfast = Protobuf.decodeToStruct(views[1], schema, true)) {
        AssertUtils.assertStructColumnsAreEqual(expected, actualPermissive);
        AssertUtils.assertStructColumnsAreEqual(expected, actualFailfast);
      } finally {
        for (ColumnView view : views) {
          view.close();
        }
      }
    }
  }

  @Test
  void testSlicedNullableInputMapsMalformedRootScalarRow_Permissive() {
    Byte[] sentinel = concat(box(tag(99, WT_VARINT)), box(encodeVarint(7)));
    Byte[] malformed = concat(box(tag(1, WT_VARINT)), new Byte[]{(byte) 0x80});
    Byte[] valid = concat(box(tag(1, WT_VARINT)), box(encodeVarint(42)));
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.INT32)
        .build();
    StructType outputType = new StructType(true, new BasicType(true, DType.INT32));

    try (Table input = new Table.TestBuilder()
             .column(new Byte[][]{sentinel, null, malformed, valid, sentinel})
             .build();
         ColumnVector expected = ColumnVector.fromStructs(
             outputType, (StructData) null, (StructData) null, struct(42))) {
      ColumnView[] views = input.getColumn(0).splitAsViews(1, 4);
      try (ColumnVector actual = Protobuf.decodeToStruct(views[1], schema, false)) {
        AssertUtils.assertStructColumnsAreEqual(expected, actual);
      } finally {
        for (ColumnView view : views) {
          view.close();
        }
      }
    }
  }

  @Test
  void testMalformedSingularMessageFragmentBoundary_Permissive() {
    Byte[] truncatedVarint = concat(box(tag(1, WT_VARINT)), new Byte[]{(byte) 0x80});
    Byte[] validUnknownFixed64 = concat(
        box(tag(3, WT_64BIT)),
        box(encodeFixed64(0x0108010801080108L)));
    Byte[] malformedFirst = concat(
        box(tag(1, WT_LEN)), encodeMessage(truncatedVarint),
        box(tag(1, WT_LEN)), encodeMessage(validUnknownFixed64));
    Byte[] malformedLast = concat(
        box(tag(1, WT_LEN)), encodeMessage(validUnknownFixed64),
        box(tag(1, WT_LEN)), encodeMessage(truncatedVarint));
    Byte[] validFirst = concat(box(tag(1, WT_VARINT)), box(encodeVarint(1)));
    Byte[] validSecond = concat(box(tag(1, WT_VARINT)), box(encodeVarint(2)));
    Byte[] wrongWireBeforeDuplicates = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(7)),
        box(tag(1, WT_LEN)), encodeMessage(validFirst),
        box(tag(1, WT_LEN)), encodeMessage(validSecond));
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).down()
            .addField(1, DType.INT32)
        .up()
        .build();

    try (Table input = new Table.TestBuilder().column(
             new Byte[][]{malformedFirst, malformedLast, wrongWireBeforeDuplicates}).build();
         ColumnVector expected = ColumnVector.fromStructs(
             new StructType(true,
                 new StructType(true, new BasicType(true, DType.INT32))),
             (StructData) null,
             (StructData) null,
             (StructData) null);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0), schema, false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testMalformedSingularMessageFragmentBoundary_Failfast() {
    Byte[] truncatedVarint = concat(box(tag(1, WT_VARINT)), new Byte[]{(byte) 0x80});
    Byte[] validUnknownFixed64 = concat(
        box(tag(3, WT_64BIT)),
        box(encodeFixed64(0x0108010801080108L)));
    Byte[] row = concat(
        box(tag(1, WT_LEN)), encodeMessage(validUnknownFixed64),
        box(tag(1, WT_LEN)), encodeMessage(truncatedVarint));
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).down()
            .addField(1, DType.INT32)
        .up()
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(
            input.getColumn(0), schema, true)) {
        }
      });
    }
  }

  @Test
  void testWrongWireBeforeDuplicateSingularMessageOccurrences_Failfast() {
    Byte[] firstFragment = concat(box(tag(1, WT_VARINT)), box(encodeVarint(1)));
    Byte[] secondFragment = concat(box(tag(1, WT_VARINT)), box(encodeVarint(2)));
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(7)),
        box(tag(1, WT_LEN)), encodeMessage(firstFragment),
        box(tag(1, WT_LEN)), encodeMessage(secondFragment));
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).down()
            .addField(1, DType.INT32)
        .up()
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(
            input.getColumn(0), schema, true)) {
        }
      });
    }
  }

  @Test
  void testNestedMessageMultipleScalarChildren() {
    // message Inner { int32 a = 1; int64 b = 2; bool c = 3; float d = 4; }
    // message Outer { Inner inner = 1; }
    // This exercises every scalar wire type a nested child can use — varint (int32/int64/bool)
    // and fixed32 (float) — across two rows including negatives and zeros. fixed64/string/bytes
    // children share the same per-type extraction paths, covered by their own top-level tests.
    Byte[][] rows = new Byte[][]{
        concat(box(tag(1, WT_LEN)), encodeMessage(concat(
            box(tag(1, WT_VARINT)), box(encodeVarint(7)),
            box(tag(2, WT_VARINT)), box(encodeVarint(123456789012L)),
            box(tag(3, WT_VARINT)), box(encodeVarint(1)),
            box(tag(4, WT_32BIT)), box(encodeFloat(3.5f))))),
        concat(box(tag(1, WT_LEN)), encodeMessage(concat(
            box(tag(1, WT_VARINT)), box(encodeVarint(-1)),
            box(tag(2, WT_VARINT)), box(encodeVarint(0)),
            box(tag(3, WT_VARINT)), box(encodeVarint(0)),
            box(tag(4, WT_32BIT)), box(encodeFloat(-0.25f)))))};

    try (Table input = new Table.TestBuilder().column(rows).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()    // Inner inner = 1
                     .addField(1, DType.INT32)        //   int32 a = 1
                     .addField(2, DType.INT64)        //   int64 b = 2
                     .addField(3, DType.BOOL8)        //   bool  c = 3
                     .addField(4, DType.FLOAT32)      //   float d = 4
                 .up()
                 .build(),
             false)) {
        assertNotNull(result);
        try (ColumnVector expA = ColumnVector.fromBoxedInts(7, -1);
             ColumnVector expB = ColumnVector.fromBoxedLongs(123456789012L, 0L);
             ColumnVector expC = ColumnVector.fromBoxedBooleans(true, false);
             ColumnVector expD = ColumnVector.fromBoxedFloats(3.5f, -0.25f);
             ColumnVector expInner = ColumnVector.makeStruct(expA, expB, expC, expD);
             ColumnVector expOuter = ColumnVector.makeStruct(expInner)) {
          AssertUtils.assertStructColumnsAreEqual(expOuter, result);
        }
    }
  }

  @Test
  void testNestedMessageVarlenChildren() {
    // message Inner { string name = 1; bytes payload = 2; }
    // message Outer { Inner inner = 1; }
    Byte[] innerMessage = concat(
        box(tag(1, WT_LEN)), encodeString("alice"),
        box(tag(2, WT_LEN)), encodeBytes(new byte[]{1, 2, 3}));
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(innerMessage));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedName = ColumnVector.fromStrings("alice");
         ColumnVector expectedPayload = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.UINT8)),
             Arrays.asList((byte) 1, (byte) 2, (byte) 3));
         ColumnVector expectedInner = ColumnVector.makeStruct(expectedName, expectedPayload);
         ColumnVector expectedOuter = ColumnVector.makeStruct(expectedInner);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.STRING)
                     .addField(2, DType.LIST)
                 .up()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actual);
    }
  }

  @Test
  void testNestedMessageStringDefault() {
    // message Inner { optional string name = 1 [default = "missing"]; }
    // message Outer { Inner inner = 1; }
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(new Byte[]{}));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedName = ColumnVector.fromStrings("missing");
         ColumnVector expectedInner = ColumnVector.makeStruct(expectedName);
         ColumnVector expectedOuter = ColumnVector.makeStruct(expectedInner);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.STRING).defaultValue("missing".getBytes())
                 .up()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actual);
    }
  }

  @Test
  void testDeepNestedMessageDepth3() {
    // message Inner  { int32 a = 1; string b = 2; bool c = 3; }
    // message Middle { Inner inner = 1; int64 m = 2; }
    // message Outer  { Middle middle = 1; float score = 2; }
    Byte[] inner = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(7)),
        box(tag(2, WT_LEN)), encodeString("abc"),
        box(tag(3, WT_VARINT)), box(encodeVarint(1)));
    Byte[] middle = concat(
        box(tag(1, WT_LEN)), encodeMessage(inner),
        box(tag(2, WT_VARINT)), box(encodeVarint(123)));
    Byte[] row = concat(
        box(tag(1, WT_LEN)), encodeMessage(middle),
        box(tag(2, WT_32BIT)), box(encodeFloat(1.25f)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedA = ColumnVector.fromBoxedInts(7);
         ColumnVector expectedB = ColumnVector.fromStrings("abc");
         ColumnVector expectedC = ColumnVector.fromBoxedBooleans(true);
         ColumnVector expectedInner = ColumnVector.makeStruct(expectedA, expectedB, expectedC);
         ColumnVector expectedM = ColumnVector.fromBoxedLongs(123L);
         ColumnVector expectedMiddle = ColumnVector.makeStruct(expectedInner, expectedM);
         ColumnVector expectedScore = ColumnVector.fromBoxedFloats(1.25f);
         ColumnVector expectedOuter = ColumnVector.makeStruct(expectedMiddle, expectedScore);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.STRUCT).down()
                         .addField(1, DType.INT32)
                         .addField(2, DType.STRING)
                         .addField(3, DType.BOOL8)
                     .up()
                     .addField(2, DType.INT64)
                 .up()
                 .addField(2, DType.FLOAT32)
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actual);
    }
  }

  @Test
  void testPackedRepeatedInsideNestedMessage() {
    // message Inner { repeated int32 ids = 1 [packed=true]; }
    // message Outer { Inner inner = 1; }
    byte[] packedIds = concatBytes(encodeVarint(10), encodeVarint(20), encodeVarint(30));
    Byte[] inner = concat(
        box(tag(1, WT_LEN)), encodeBytes(packedIds));
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(inner));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedIds = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)),
             Arrays.asList(10, 20, 30));
         ColumnVector expectedInner = ColumnVector.makeStruct(expectedIds);
         ColumnVector expectedOuter = ColumnVector.makeStruct(expectedInner);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.INT32).repeated()
                 .up()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actual);
    }
  }

  @Test
  void testPackedRepeatedDoubleInsideNestedMessage() {
    // message Inner { repeated double values = 1 [packed=true]; }
    // message Outer { Inner inner = 1; }
    Byte[] inner0 = concat(
        box(tag(1, WT_LEN)), encodeBytes(concatBytes(encodeDouble(1.5), encodeDouble(-2.25))));
    Byte[] inner1 = concat(
        box(tag(1, WT_LEN)), encodeBytes(encodeDouble(3.75)));
    Byte[][] rows = new Byte[][]{
        concat(box(tag(1, WT_LEN)), encodeMessage(inner0)),
        concat(box(tag(1, WT_LEN)), encodeMessage(inner1))
    };

    try (Table input = new Table.TestBuilder().column(rows).build();
         ColumnVector expectedValues = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.FLOAT64)),
             Arrays.asList(1.5, -2.25),
             Arrays.asList(3.75));
         ColumnVector expectedInner = ColumnVector.makeStruct(expectedValues);
         ColumnVector expectedOuter = ColumnVector.makeStruct(expectedInner);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.FLOAT64).repeated()
                 .up()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actual);
    }
  }

  @Test
  void testNestedRepeatedEnumAsString() {
    // message Inner { repeated Priority priority = 1 [packed=true]; }
    // message Outer { Inner inner = 1; }
    // enum Priority { UNKNOWN=0; FOO=1; BAR=2; }
    byte[] packedPriorities = concatBytes(encodeVarint(0), encodeVarint(2), encodeVarint(1));
    Byte[] inner = concat(
        box(tag(1, WT_LEN)), encodeBytes(packedPriorities));
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(inner));
    byte[][] enumNames = new byte[][]{
        "UNKNOWN".getBytes(StandardCharsets.UTF_8),
        "FOO".getBytes(StandardCharsets.UTF_8),
        "BAR".getBytes(StandardCharsets.UTF_8)
    };

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedPriorities = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.STRING)),
             Arrays.asList("UNKNOWN", "BAR", "FOO"));
         ColumnVector expectedInner = ColumnVector.makeStruct(expectedPriorities);
         ColumnVector expectedOuter = ColumnVector.makeStruct(expectedInner);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.STRING).repeated()
                         .enumMetadata(new int[]{0, 1, 2}, enumNames)
                 .up()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actual);
    }
  }

  @Test
  void testNestedRepeatedScalarEmptyAndAbsentParent() {
    // message Inner { repeated int32 ids = 1 [packed=true]; }
    // message Outer { Inner inner = 1; }
    Byte[][] rows = new Byte[][]{
        concat(box(tag(1, WT_LEN)), encodeMessage(new Byte[]{})),
        new Byte[]{}
    };

    StructType innerType = new StructType(
        true, new ListType(true, new BasicType(true, DType.INT32)));
    try (Table input = new Table.TestBuilder().column(rows).build();
         ColumnVector expectedInner = ColumnVector.fromStructs(
             innerType, struct(Collections.emptyList()), null);
         ColumnVector expectedOuter = ColumnVector.makeStruct(expectedInner);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.INT32).repeated()
                 .up()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actual);
      assertListOffsets(actual.getChildColumnView(0).getChildColumnView(0), 0, 0, 0);
    }
  }

  @Test
  void testNestedRepeatedScalarNullWhenImmediateParentAbsent() {
    // message Inner { repeated int32 ids = 1 [packed=true]; }
    // message Middle { Inner inner = 1; }
    // message Outer { Middle middle = 1; }
    byte[] packedIds = concatBytes(encodeVarint(7), encodeVarint(8));
    Byte[] inner = concat(
        box(tag(1, WT_LEN)), encodeBytes(packedIds));
    Byte[] middleWithInner = concat(box(tag(1, WT_LEN)), encodeMessage(inner));
    Byte[] middleWithoutInner = new Byte[]{};
    Byte[][] rows = new Byte[][]{
        concat(box(tag(1, WT_LEN)), encodeMessage(middleWithInner)),
        concat(box(tag(1, WT_LEN)), encodeMessage(middleWithoutInner))
    };

    StructType innerType = new StructType(
        true, new ListType(true, new BasicType(true, DType.INT32)));
    try (Table input = new Table.TestBuilder().column(rows).build();
         ColumnVector expectedInner = ColumnVector.fromStructs(
             innerType, struct(Arrays.asList(7, 8)), null);
         ColumnVector expectedMiddle = ColumnVector.makeStruct(expectedInner);
         ColumnVector expectedOuter = ColumnVector.makeStruct(expectedMiddle);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.STRUCT).down()
                         .addField(1, DType.INT32).repeated()
                     .up()
                 .up()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actual);
      assertListOffsets(
          actual.getChildColumnView(0).getChildColumnView(0).getChildColumnView(0), 0, 2, 2);
    }
  }

  @Test
  void testNestedRepeatedEnumAsStringUnknownValueIsDroppedInBothModes() {
    // message Inner { repeated Priority priority = 1 [packed=true]; }
    // message Outer { Inner inner = 1; }
    // enum Priority { UNKNOWN=0; FOO=1; BAR=2; }
    byte[] validPriorities = concatBytes(encodeVarint(1), encodeVarint(2));
    byte[] invalidPriorities = concatBytes(encodeVarint(1), encodeVarint(999));
    Byte[][] rows = new Byte[][]{
        concat(box(tag(1, WT_LEN)), encodeMessage(concat(
            box(tag(1, WT_LEN)), encodeBytes(validPriorities)))),
        concat(box(tag(1, WT_LEN)), encodeMessage(concat(
            box(tag(1, WT_LEN)), encodeBytes(invalidPriorities))))
    };
    byte[][] enumNames = new byte[][]{
        "UNKNOWN".getBytes(StandardCharsets.UTF_8),
        "FOO".getBytes(StandardCharsets.UTF_8),
        "BAR".getBytes(StandardCharsets.UTF_8)
    };
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).down()
            .addField(1, DType.STRING).repeated()
                .enumMetadata(new int[]{0, 1, 2}, enumNames)
        .up()
        .build();

    try (Table input = new Table.TestBuilder().column(rows).build();
         ColumnVector expectedPriorities = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.STRING)),
             Arrays.asList("FOO", "BAR"),
             Collections.singletonList("FOO"));
         ColumnVector expectedInner = ColumnVector.makeStruct(expectedPriorities);
         ColumnVector expectedOuter = ColumnVector.makeStruct(expectedInner);
         ColumnVector actualPermissive = Protobuf.decodeToStruct(
             input.getColumn(0), schema, false);
         ColumnVector actualFailfast = Protobuf.decodeToStruct(
             input.getColumn(0), schema, true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actualPermissive);
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actualFailfast);
    }
  }

  @Test
  void testNestedRepeatedNumericEnumUnknownValueIsDropped() {
    Byte[] inner = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)));
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(inner));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedValues = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)), Arrays.asList(1, 2));
         ColumnVector expectedInner = ColumnVector.makeStruct(expectedValues);
         ColumnVector expectedOuter = ColumnVector.makeStruct(expectedInner);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.INT32).repeated().enumValidValues(new int[]{0, 1, 2})
                 .up()
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actual);
    }
  }

  @Test
  void testNestedRepeatedStringAndBytes() {
    // message Inner { repeated string name = 1; repeated bytes payload = 2; }
    // message Outer { Inner inner = 1; }
    byte[] p0 = new byte[]{0x01, 0x02};
    byte[] p1 = new byte[]{0x03};
    Byte[] inner = concat(
        box(tag(1, WT_LEN)), encodeString("alpha"),
        box(tag(1, WT_LEN)), encodeString("beta"),
        box(tag(2, WT_LEN)), encodeBytes(p0),
        box(tag(2, WT_LEN)), encodeBytes(p1));
    Byte[][] rows = new Byte[][]{
        concat(box(tag(1, WT_LEN)), encodeMessage(inner)),
        concat(box(tag(1, WT_LEN)), encodeMessage(new Byte[]{}))
    };

    try (Table input = new Table.TestBuilder().column(rows).build();
         ColumnVector expectedNames = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.STRING)),
             Arrays.asList("alpha", "beta"),
             Collections.emptyList());
         ColumnVector expectedPayloads = ColumnVector.fromLists(
             new ListType(true, new ListType(true, new BasicType(true, DType.UINT8))),
             Arrays.asList(
                 Arrays.asList((byte) 0x01, (byte) 0x02),
                 Collections.singletonList((byte) 0x03)),
             Collections.emptyList());
         ColumnVector expectedInner = ColumnVector.makeStruct(expectedNames, expectedPayloads);
         ColumnVector expectedOuter = ColumnVector.makeStruct(expectedInner);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.STRING).repeated()
                     .addField(2, DType.LIST).repeated()
                 .up()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actual);
    }
  }

  @Test
  void testNestedEnumAsStringInvalidKeepsSiblingFieldsVisible_Failfast() {
    // message Outer { int32 id = 1; Inner inner = 2; string name = 3; }
    // message Inner { enum Status { UNKNOWN=0; OK=1; BAD=2; } Status status = 1;
    //                 int32 count = 2; }
    Byte[] innerValid = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(2, WT_VARINT)), box(encodeVarint(10)));
    Byte[] innerInvalid = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),
        box(tag(2, WT_VARINT)), box(encodeVarint(20)));
    Byte[][] rows = new Byte[][]{
        concat(
            box(tag(1, WT_VARINT)), box(encodeVarint(1)),
            box(tag(2, WT_LEN)), encodeMessage(innerValid),
            box(tag(3, WT_LEN)), encodeString("ok")),
        concat(
            box(tag(1, WT_VARINT)), box(encodeVarint(2)),
            box(tag(2, WT_LEN)), encodeMessage(innerInvalid),
            box(tag(3, WT_LEN)), encodeString("bad"))};
    byte[][] enumNames = new byte[][]{
        "UNKNOWN".getBytes(), "OK".getBytes(), "BAD".getBytes()};

    try (Table input = new Table.TestBuilder().column(rows).build();
         ColumnVector expectedId = ColumnVector.fromBoxedInts(1, 2);
         ColumnVector expectedStatus = ColumnVector.fromStrings("OK", null);
         ColumnVector expectedCount = ColumnVector.fromBoxedInts(10, 20);
         ColumnVector expectedInner = ColumnVector.makeStruct(expectedStatus, expectedCount);
         ColumnVector expectedName = ColumnVector.fromStrings("ok", "bad");
         ColumnVector expectedOuter = ColumnVector.makeStruct(
             expectedId, expectedInner, expectedName);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32)
                 .addField(2, DType.STRUCT).down()
                     .addField(1, DType.STRING)
                         .enumMetadata(new int[]{0, 1, 2}, enumNames)
                     .addField(2, DType.INT32)
                 .up()
                 .addField(3, DType.STRING)
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actual);
    }
  }

  @Test
  void testNestedNumericEnumUnknownValueUsesExplicitDefaultInBothModes() {
    Byte[] inner = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),
        box(tag(2, WT_VARINT)), box(encodeVarint(20)));
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(inner));
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).down()
            .addField(1, DType.INT32).defaultValue(2)
                .enumValidValues(new int[]{0, 1, 2})
            .addField(2, DType.INT32)
        .up()
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedPriority = ColumnVector.fromBoxedInts(2);
         ColumnVector expectedCount = ColumnVector.fromBoxedInts(20);
         ColumnVector expectedInner = ColumnVector.makeStruct(expectedPriority, expectedCount);
         ColumnVector expectedOuter = ColumnVector.makeStruct(expectedInner);
         ColumnVector actualPermissive =
             Protobuf.decodeToStruct(input.getColumn(0), schema, false);
         ColumnVector actualFailfast = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actualPermissive);
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actualFailfast);
    }
  }

  @Test
  void testNestedEnumIgnoresUnknownOccurrencesWhenRecognizedValueExists() {
    Byte[] validThenUnknown = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),
        box(tag(2, WT_VARINT)), box(encodeVarint(1)),
        box(tag(2, WT_VARINT)), box(encodeVarint(999)),
        box(tag(3, WT_VARINT)), box(encodeVarint(20)));
    Byte[] unknownThenValid = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)),
        box(tag(2, WT_VARINT)), box(encodeVarint(999)),
        box(tag(2, WT_VARINT)), box(encodeVarint(2)),
        box(tag(3, WT_VARINT)), box(encodeVarint(30)));
    Byte[][] rows = new Byte[][]{
        concat(box(tag(1, WT_LEN)), encodeMessage(validThenUnknown)),
        concat(box(tag(1, WT_LEN)), encodeMessage(unknownThenValid))
    };

    try (Table input = new Table.TestBuilder().column(rows).build();
         ColumnVector expectedDefaulted = ColumnVector.fromBoxedInts(1, 2);
         ColumnVector expectedRequired = ColumnVector.fromBoxedInts(1, 2);
         ColumnVector expectedCount = ColumnVector.fromBoxedInts(20, 30);
         ColumnVector expectedInner =
             ColumnVector.makeStruct(expectedDefaulted, expectedRequired, expectedCount);
         ColumnVector expectedOuter = ColumnVector.makeStruct(expectedInner);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.INT32).defaultValue(2)
                         .enumValidValues(new int[]{0, 1, 2})
                     .addField(2, DType.INT32).required()
                         .enumValidValues(new int[]{0, 1, 2})
                     .addField(3, DType.INT32)
                 .up()
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actual);
    }
  }

  @Test
  void testRepeatedMessageChildEnumUnknownValueUsesExplicitDefault() {
    Byte[] validItem = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(2, WT_VARINT)), box(encodeVarint(10)));
    Byte[] invalidItem = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),
        box(tag(2, WT_VARINT)), box(encodeVarint(20)));
    Byte[] row = concat(
        box(tag(1, WT_LEN)), encodeMessage(validItem),
        box(tag(1, WT_LEN)), encodeMessage(invalidItem));
    byte[][] enumNames = new byte[][]{
        "UNKNOWN".getBytes(StandardCharsets.UTF_8),
        "FOO".getBytes(StandardCharsets.UTF_8),
        "BAR".getBytes(StandardCharsets.UTF_8)
    };

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedItems = ColumnVector.fromLists(
             new ListType(true,
                 new StructType(true,
                     new BasicType(true, DType.STRING),
                     new BasicType(true, DType.INT32))),
             Arrays.asList(struct("FOO", 10), struct("BAR", 20)));
         ColumnVector expected = ColumnVector.makeStruct(expectedItems);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).repeated().down()
                     .addField(1, DType.STRING).defaultValue(2)
                         .enumMetadata(new int[]{0, 1, 2}, enumNames)
                     .addField(2, DType.INT32)
                 .up()
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testRequiredFieldInsideNestedMessageMissing_Failfast() {
    // message Outer { Inner inner = 1; }
    // message Inner { required int32 id = 1; optional string note = 2; }
    Byte[] inner = concat(box(tag(2, WT_LEN)), encodeString("oops"));
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(inner));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.STRUCT).down()
                    .addField(1, DType.INT32).required()
                    .addField(2, DType.STRING)
                .up()
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testRequiredFieldInsideNestedMessageMissing_Permissive() {
    // message Outer { Inner inner = 1; string name = 2; }
    // message Inner { required int32 id = 1; optional string note = 2; }
    Byte[] inner = concat(box(tag(2, WT_LEN)), encodeString("oops"));
    Byte[] row = concat(
        box(tag(1, WT_LEN)), encodeMessage(inner),
        box(tag(2, WT_LEN)), encodeString("outside"));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.INT32).required()
                     .addField(2, DType.STRING)
                 .up()
                 .addField(2, DType.STRING)
                 .build(),
             false)) {
      assertSingleNullStructRow(actual,
          "Missing nested required field should null the outer row in PERMISSIVE mode");
    }
  }

  @Test
  void testRequiredNumericEnumInsideNestedMessageUnknownInvalidatesRoot() {
    Byte[] inner = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),
        box(tag(2, WT_VARINT)), box(encodeVarint(20)));
    Byte[] row = concat(
        box(tag(1, WT_LEN)), encodeMessage(inner),
        box(tag(2, WT_LEN)), encodeString("outside"));
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).down()
            .addField(1, DType.INT32).required().enumValidValues(new int[]{0, 1, 2})
            .addField(2, DType.INT32)
        .up()
        .addField(2, DType.STRING)
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actual = Protobuf.decodeToStruct(input.getColumn(0), schema, false)) {
      assertSingleNullStructRow(
          actual, "Unknown nested required enum should null the root in PERMISSIVE mode");
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
        }
      });
    }
  }

  @Test
  void testRequiredEnumInsideRepeatedMessageUnknownInvalidatesRoot() {
    Byte[] validItem = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(2, WT_VARINT)), box(encodeVarint(10)));
    Byte[] invalidItem = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),
        box(tag(2, WT_VARINT)), box(encodeVarint(20)));
    Byte[] row = concat(
        box(tag(1, WT_LEN)), encodeMessage(validItem),
        box(tag(1, WT_LEN)), encodeMessage(invalidItem));
    byte[][] enumNames = new byte[][]{
        "UNKNOWN".getBytes(StandardCharsets.UTF_8),
        "FOO".getBytes(StandardCharsets.UTF_8),
        "BAR".getBytes(StandardCharsets.UTF_8)
    };
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).repeated().down()
            .addField(1, DType.STRING).required()
                .enumMetadata(new int[]{0, 1, 2}, enumNames)
            .addField(2, DType.INT32)
        .up()
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actual = Protobuf.decodeToStruct(input.getColumn(0), schema, false)) {
      assertSingleNullStructRow(
          actual, "Unknown required enum in a repeated message should null the root");
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
        }
      });
    }
  }

  @Test
  void testAbsentNestedParentSkipsRequiredChildCheck_Failfast() {
    // message Outer { optional Inner inner = 1; }
    // message Inner { required int32 id = 1; }
    Byte[] row = new Byte[0];

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.INT32).required()
                 .up()
                 .build(),
             true);
         ColumnVector inner = actual.getChildColumnView(0).copyToColumnVector();
         HostColumnVector hostOuter = actual.copyToHost();
         HostColumnVector hostInner = inner.copyToHost()) {
      assertEquals(0, actual.getNullCount(), "Outer row should remain valid");
      assertFalse(hostOuter.isNull(0), "Top-level row should not be null");
      assertEquals(1, inner.getNullCount(), "Absent nested parent should stay null");
      assertTrue(hostInner.isNull(0), "Missing optional nested struct should skip child required");
    }
  }

  @Test
  void testAbsentNestedMessage_ProducesNull() {
    // Outer message present, but the nested Inner field is missing from the wire.
    Byte[] row = new Byte[]{};

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.INT32).defaultValue(99)
                 .up()
                 .build(),
             false);
         ColumnVector inner = result.getChildColumnView(0).copyToColumnVector();
         ColumnVector innerX = inner.getChildColumnView(0).copyToColumnVector();
         HostColumnVector hostInner = inner.copyToHost();
         HostColumnVector hostX = innerX.copyToHost()) {
      assertNotNull(result);
      assertTrue(hostInner.isNull(0), "Inner struct should be null when parent field absent");
      assertTrue(hostX.isNull(0), "Inner x should inherit nullability from absent parent");
    }
  }

  @Test
  void testZeroLengthNestedMessage_ChildIsNull() {
    // Outer carries the nested tag but with an empty (length-0) Inner.
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(new Byte[]{}));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.INT32)
                 .up()
                 .build(),
             false);
         ColumnVector inner = result.getChildColumnView(0).copyToColumnVector();
         ColumnVector innerX = inner.getChildColumnView(0).copyToColumnVector();
         HostColumnVector hostInner = inner.copyToHost();
         HostColumnVector hostX = innerX.copyToHost()) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
      assertFalse(hostInner.isNull(0), "Inner struct should be present (length=0 nested)");
      assertTrue(hostX.isNull(0), "Inner x should be null since the field is absent");
    }
  }

  @Test
  void testChildlessNestedMessage_IsPresent() {
    // message Empty {}
    // message Outer { Empty inner = 1; }
    // Row 0: present-but-empty Inner; row 1: Inner field absent entirely.
    Byte[][] rows = new Byte[][]{
        concat(box(tag(1, WT_LEN)), encodeMessage(new Byte[]{})),
        new Byte[]{}};

    try (Table input = new Table.TestBuilder().column(rows).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT)  // childless Empty message
                 .build(),
             false);
         ColumnVector inner = result.getChildColumnView(0).copyToColumnVector();
         HostColumnVector hostInner = inner.copyToHost()) {
      assertFalse(hostInner.isNull(0), "Present empty message should produce a valid STRUCT<>");
      assertTrue(hostInner.isNull(1), "Missing empty message should produce a null STRUCT<>");
    }
  }

  @Test
  void testMalformedChildlessNestedMessage_Failfast() {
    // message Empty {}
    // message Outer { Empty inner = 1; }
    // The childless Inner body contains an unknown field with a truncated varint value.
    Byte[] malformedInner = concat(box(tag(1, WT_VARINT)), new Byte[]{(byte) 0x80});
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(malformedInner));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.STRUCT)
                .build(),
            true)) {
        }
      });
    }
  }

  @Test
  void testMalformedChildlessNestedMessage_PermissiveReturnsNull() {
    // message Empty {}
    // message Outer { Empty inner = 1; }
    // The childless Inner body contains an unknown field with a truncated varint value.
    Byte[] malformedInner = concat(box(tag(1, WT_VARINT)), new Byte[]{(byte) 0x80});
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(malformedInner));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT)
                 .build(),
             false);
         ColumnVector inner = result.getChildColumnView(0).copyToColumnVector();
         HostColumnVector hostResult = result.copyToHost();
         HostColumnVector hostInner = inner.copyToHost()) {
      assertEquals(1, result.getNullCount(), "Malformed childless nested message should null row");
      assertTrue(hostResult.isNull(0), "Malformed row should be null in permissive mode");
      assertEquals(1, inner.getNullCount(), "Child null should reflect the top-level null row");
      assertTrue(hostInner.isNull(0), "Childless nested struct should be null for malformed row");
    }
  }

  @Test
  void testChildlessNestedMessageWithWellFormedUnknownField_IsPresent() {
    // message Empty {}
    // message Outer { Empty inner = 1; }
    // Unknown fields inside a childless message are still scanned and skipped.
    Byte[] innerWithUnknown = concat(box(tag(7, WT_VARINT)), box(encodeVarint(123)));
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(innerWithUnknown));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT)
                 .build(),
             true);
         ColumnVector inner = result.getChildColumnView(0).copyToColumnVector();
         HostColumnVector hostInner = inner.copyToHost()) {
      assertEquals(0, result.getNullCount(), "Well-formed unknown child field should pass");
      assertEquals(0, inner.getNullCount(), "Present childless message should stay valid");
      assertFalse(hostInner.isNull(0), "Present childless message should produce a valid STRUCT<>");
    }
  }

  @Test
  void testRecursiveChildlessNestedMessage_PreservesPresence() {
    // message Empty {}
    // message Middle { Empty empty = 1; }
    // message Outer { Middle middle = 1; }
    Byte[] middleWithEmpty = concat(box(tag(1, WT_LEN)), encodeMessage(new Byte[]{}));
    Byte[][] rows = new Byte[][]{
        concat(box(tag(1, WT_LEN)), encodeMessage(middleWithEmpty)),
        concat(box(tag(1, WT_LEN)), encodeMessage(new Byte[]{})),
        new Byte[]{}};

    try (Table input = new Table.TestBuilder().column(rows).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.STRUCT)
                 .up()
                 .build(),
             false);
         ColumnVector middle = result.getChildColumnView(0).copyToColumnVector();
         ColumnVector empty = middle.getChildColumnView(0).copyToColumnVector();
         HostColumnVector hostMiddle = middle.copyToHost();
         HostColumnVector hostEmpty = empty.copyToHost()) {
      assertFalse(hostMiddle.isNull(0), "Present middle should produce a valid STRUCT");
      assertFalse(hostEmpty.isNull(0), "Present empty child should produce a valid STRUCT<>");
      assertFalse(hostMiddle.isNull(1), "Present middle should stay valid when empty is absent");
      assertTrue(hostEmpty.isNull(1), "Missing empty child should produce a null STRUCT<>");
      assertTrue(hostMiddle.isNull(2), "Missing middle should produce a null STRUCT");
      assertTrue(hostEmpty.isNull(2), "Empty child should inherit missing middle nullability");
    }
  }

  @Test
  void testNestedSingularWrongWireTypeSkipsMismatchedOccurrenceInBothModes() {
    // Spark CPU treats nested known fields with a mismatched wire type as unknown fields.
    Byte[] wrongOnly = concat(
        box(tag(1, WT_32BIT)), box(encodeFixed32(77)),
        box(tag(2, WT_VARINT)), box(encodeVarint(42)));
    Byte[] wrongThenValid = concat(
        box(tag(1, WT_32BIT)), box(encodeFixed32(88)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)),
        box(tag(2, WT_VARINT)), box(encodeVarint(43)));
    Byte[][] rows = new Byte[][]{
        concat(box(tag(1, WT_LEN)), encodeMessage(wrongOnly)),
        concat(box(tag(1, WT_LEN)), encodeMessage(wrongThenValid))};
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).down()
            .addField(1, DType.INT32)
            .addField(2, DType.INT32)
        .up()
        .build();
    StructType innerType = new StructType(
        true, new BasicType(true, DType.INT32), new BasicType(true, DType.INT32));

    try (Table input = new Table.TestBuilder().column(rows).build();
         ColumnVector expectedInner = ColumnVector.fromStructs(
             innerType, struct((Object) null, 42), struct(2, 43));
         ColumnVector expectedOuter = ColumnVector.makeStruct(expectedInner);
         ColumnVector actualPermissive = Protobuf.decodeToStruct(
             input.getColumn(0), schema, false);
         ColumnVector actualFailfast = Protobuf.decodeToStruct(
             input.getColumn(0), schema, true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actualPermissive);
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actualFailfast);
    }
  }

  @Test
  void testNestedRepeatedWrongWireTypeSkipsMismatchedOccurrence_Failfast() {
    // message Inner { repeated int32 x = 1; }
    // message Outer { Inner inner = 1; }
    Byte[] innerMessage = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_32BIT)), box(encodeFixed32(77)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)));
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(innerMessage));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedIds = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)),
             Arrays.asList(1, 2));
         ColumnVector expectedInner = ColumnVector.makeStruct(expectedIds);
         ColumnVector expectedOuter = ColumnVector.makeStruct(expectedInner);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.INT32).repeated()
                 .up()
                 .build(),
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actual);
    }
  }

  @Test
  void testNestedRepeatedWrongWireTypeSkipsMismatchedOccurrence_Permissive() {
    // message Inner { repeated int32 x = 1; }
    // message Outer { Inner inner = 1; }
    Byte[] inner0 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_32BIT)), box(encodeFixed32(77)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)));
    Byte[] inner1 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(100)));
    Byte[][] rows = new Byte[][]{
        concat(box(tag(1, WT_LEN)), encodeMessage(inner0)),
        concat(box(tag(1, WT_LEN)), encodeMessage(inner1))};

    try (Table input = new Table.TestBuilder().column(rows).build();
         ColumnVector expectedIds = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)),
             Arrays.asList(1, 2),
             Arrays.asList(100));
         ColumnVector expectedInner = ColumnVector.makeStruct(expectedIds);
         ColumnVector expectedOuter = ColumnVector.makeStruct(expectedInner);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.INT32).repeated()
                 .up()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedOuter, actual);
    }
  }

  @Test
  void testNestedMalformedZeroCountRepeatedFieldBeforeLaterField_Permissive() {
    Byte[] invalidInner = concat(
        box(tag(1, WT_LEN)), encodeBytes(new byte[]{(byte) 0x80}),
        box(tag(2, WT_VARINT)), box(encodeVarint(11)));
    Byte[] validInner = concat(box(tag(2, WT_VARINT)), box(encodeVarint(22)));
    Byte[][] rows = new Byte[][]{
        concat(box(tag(1, WT_LEN)), encodeMessage(invalidInner)),
        concat(box(tag(1, WT_LEN)), encodeMessage(validInner))};
    StructType innerType = new StructType(
        true,
        new ListType(true, new BasicType(true, DType.INT32)),
        new ListType(true, new BasicType(true, DType.INT32)));
    StructType outerType = new StructType(true, innerType);

    try (Table input = new Table.TestBuilder().column(rows).build();
         ColumnVector expected = ColumnVector.fromStructs(
             outerType,
             null,
             struct(struct(Collections.emptyList(), Collections.singletonList(22))));
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).down()
                     .addField(1, DType.INT32).repeated()
                     .addField(2, DType.INT32).repeated()
                 .up()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testZeroRowNestedSchemaShape() {
    // 0 rows with nested schema — verify correct type hierarchy.
    // message Outer { int32 a = 1; Inner b = 2; }  message Inner { int32 x = 1; }
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.INT32)            // a
        .addField(2, DType.STRUCT).down()    // b
            .addField(1, DType.INT32)        // x
        .up()
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
      assertEquals(0, result.getRowCount());
      assertEquals(2, result.getNumChildren());
      assertEquals(DType.INT32, result.getChildColumnView(0).getType());
      assertEquals(DType.STRUCT, result.getChildColumnView(1).getType());
      assertEquals(1, result.getChildColumnView(1).getNumChildren());
      assertEquals(DType.INT32, result.getChildColumnView(1).getChildColumnView(0).getType());
    }
  }

  @Test
  void testZeroRowNestedRepeatedScalarShape() {
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).down()
            .addField(1, DType.INT32).repeated()
        .up()
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
      ColumnView nested = result.getChildColumnView(0);
      ColumnView values = nested.getChildColumnView(0);
      assertEquals(0, result.getRowCount());
      assertEquals(DType.STRUCT, nested.getType());
      assertEquals(DType.LIST, values.getType());
      assertEquals(DType.INT32, values.getChildColumnView(0).getType());
    }
  }

  @Test
  void testZeroRowRepeatedMessageShape() {
    // 0 rows with repeated message schema: repeated Inner inner = 1; message Inner { int32 x = 1; }
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).repeated().down()
            .addField(1, DType.INT32)
        .up()
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
      assertEquals(0, result.getRowCount());
      assertEquals(1, result.getNumChildren());
      assertEquals(DType.LIST, result.getChildColumnView(0).getType());
    }
  }

  @Test
  void testZeroRowRepeatedScalarShape() {
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.INT32).repeated()
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
      assertEquals(0, result.getRowCount());
      assertEquals(1, result.getNumChildren());
      assertEquals(DType.LIST, result.getChildColumnView(0).getType());
    }
  }

  // ============================================================================
  // Input validation tests
  // ============================================================================

  @Test
  void testNullBinaryInputThrows() {
    assertThrows(IllegalArgumentException.class, () ->
        Protobuf.decodeToStruct(null,
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.INT64)
                .build(), true));
  }

  @Test
  void testNullSchemaThrows() {
    Byte[] row = new Byte[]{0x08, 0x01};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(IllegalArgumentException.class, () ->
          Protobuf.decodeToStruct(input.getColumn(0), null, true));
    }
  }

  @Test
  void testRepeatedString() {
    // Exercises the build_repeated_string_column non-enum path (CUB DeviceMemcpy::Batched
    // copy + length-extraction), which the existing testRepeatedEnumAsString does not cover.
    byte[] s1 = "hello".getBytes(StandardCharsets.UTF_8);
    byte[] s2 = "world".getBytes(StandardCharsets.UTF_8);
    byte[] s3 = "foo".getBytes(StandardCharsets.UTF_8);
    Byte[] row0 = concat(
        box(tag(1, WT_LEN)), encodeBytes(s1),
        box(tag(1, WT_LEN)), encodeBytes(s2));
    Byte[] row1 = concat(
        box(tag(1, WT_LEN)), encodeBytes(s3));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1}).build();
         ColumnVector expectedValues = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.STRING)),
             Arrays.asList("hello", "world"),
             Collections.singletonList("foo"));
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedValues);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING).repeated()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testRepeatedBytes() {
    // Exercises build_repeated_string_column with is_bytes=true (BYTES dispatched as
    // LIST<UINT8>), which testRepeatedString does not cover.
    byte[] b1 = new byte[]{0x00, 0x01, 0x02};
    byte[] b2 = new byte[]{0x7f, (byte) 0xff};
    byte[] b3 = new byte[]{0x10};
    Byte[] row0 = concat(
        box(tag(1, WT_LEN)), encodeBytes(b1),
        box(tag(1, WT_LEN)), encodeBytes(b2));
    Byte[] row1 = concat(
        box(tag(1, WT_LEN)), encodeBytes(b3));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1}).build();
         ColumnVector expectedValues = ColumnVector.fromLists(
             new ListType(true, new ListType(true, new BasicType(true, DType.UINT8))),
             Arrays.asList(
                 Arrays.asList((byte) 0x00, (byte) 0x01, (byte) 0x02),
                 Arrays.asList((byte) 0x7f, (byte) 0xff)),
             Collections.singletonList(Collections.singletonList((byte) 0x10)));
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedValues);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.LIST).repeated()  // repeated bytes -> LIST<UINT8> element
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
      try (ColumnView outerList = actualStruct.getChildColumnView(0);
           ColumnView innerList = outerList.getChildColumnView(0)) {
        assertListOffsets(outerList, 0, 2, 3);
        assertListOffsets(innerList, 0, b1.length, b1.length + b2.length,
            b1.length + b2.length + b3.length);
      }
    }
  }

  @Test
  void testRepeatedSint32() {
    // sint32 zigzag encoding: zigzag(-1) = 1, zigzag(-2) = 3, zigzag(3) = 6.
    // Verifies the extract_varint_kernel<T, true, repeated_location_provider> instantiation.
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1L)),
        box(tag(1, WT_VARINT)), box(encodeVarint(3L)),
        box(tag(1, WT_VARINT)), box(encodeVarint(6L)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedValues = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)),
             Arrays.asList(-1, -2, 3));
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedValues);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).repeated().encoding(Protobuf.ENC_ZIGZAG)
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testNullInputRowProducesNullListForRepeatedField() {
    // Verifies make_list_column_with_input_nulls propagates the input null mask to the
    // output LIST column; previously only exercised by scalar (non-LIST) schemas.
    Byte[] row0 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(7)),
        box(tag(1, WT_VARINT)), box(encodeVarint(8)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, null}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).repeated()
                 .build(),
             false)) {
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(2, result.getRowCount());
      try (ColumnView listCol = result.getChildColumnView(0);
           HostColumnVector hList = listCol.copyToHost()) {
        assertEquals(DType.LIST, hList.getType());
        assertFalse(hList.isNull(0), "row 0 should have a valid list");
        assertTrue(hList.isNull(1), "null input row 1 should produce a null list");
      }
    }
  }

  @Test
  void testSchemaWithTooManyRepeatedFields() {
    int n = 33;
    ProtobufSchemaDescriptorBuilder builder = new ProtobufSchemaDescriptorBuilder();
    for (int i = 0; i < n; i++) {
      builder.addField(i + 1, DType.INT32).repeated();
    }
    ProtobufSchemaDescriptor schema = builder.build();
    Byte[][] rows = new Byte[][]{new Byte[0]};
    try (Table input = new Table.TestBuilder().column(rows).build()) {
      assertThrows(RuntimeException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
        }
      });
    }
  }

  @Test
  void testNestedSchemaWithTooManyRepeatedFields() {
    int n = 33;
    ProtobufSchemaDescriptorBuilder builder = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).down();
    for (int i = 0; i < n; i++) {
      builder.addField(i + 1, DType.INT32).repeated();
    }
    ProtobufSchemaDescriptor schema = builder.up().build();
    Byte[][] rows = new Byte[][]{new Byte[0]};
    try (Table input = new Table.TestBuilder().column(rows).build()) {
      assertThrows(RuntimeException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
        }
      });
    }
  }

  @Test
  void testHiddenFieldDoesNotAppearInOutput() {
    // message Msg { int32 a = 1; int32 b = 2; } — both present in the wire, b is hidden.
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(7)),
        box(tag(2, WT_VARINT)), box(encodeVarint(11)));

    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.INT32)
        .addField(2, DType.INT32).isOutput(false)  // hide b
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedA = ColumnVector.fromBoxedInts(7);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedA);
         ColumnVector actualStruct = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testHiddenRepeatedMessageAbsentDoesNotFail() {
    // message Msg { int32 a = 1; repeated Inner hidden = 2; }
    // message Inner { int32 x = 1; }
    // Wire data omits hidden; it should be validated/scanned, then dropped from the output.
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(7)));

    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.INT32)
        .addField(2, DType.STRUCT).repeated().isOutput(false).down()
            .addField(1, DType.INT32).isOutput(false)
        .up()
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedA = ColumnVector.fromBoxedInts(7);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedA);
         ColumnVector actualStruct = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testHiddenRepeatedMessageMissingRequiredFieldStillValidates_Failfast() {
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(7)),
        box(tag(2, WT_LEN)), encodeMessage(new Byte[0]));

    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.INT32)
        .addField(2, DType.STRUCT).repeated().isOutput(false).down()
            .addField(1, DType.INT32).required().isOutput(false)
        .up()
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
        }
      });
    }
  }

  @Test
  void testHiddenRepeatedMessageMissingRequiredFieldPermissiveNullsTopRow() {
    Byte[] invalidRow = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(7)),
        box(tag(2, WT_LEN)), encodeMessage(new Byte[0]),
        box(tag(2, WT_LEN)), encodeMessage(new Byte[0]));
    Byte[] validRow = concat(box(tag(1, WT_VARINT)), box(encodeVarint(9)));

    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.INT32)
        .addField(2, DType.STRUCT).repeated().isOutput(false).down()
            .addField(1, DType.INT32).required().isOutput(false)
        .up()
        .build();
    StructType expectedType = new StructType(true, new BasicType(true, DType.INT32));

    try (Table input = new Table.TestBuilder().column(invalidRow, validRow).build();
         ColumnVector expected = ColumnVector.fromStructs(expectedType, null, struct(9));
         ColumnVector actual = Protobuf.decodeToStruct(input.getColumn(0), schema, false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testHiddenRequiredFieldStillValidates_Failfast() {
    // message Msg { int32 a = 1; int32 b = 2 [required]; } — b is hidden but required;
    // wire data omits b. In failfast mode the missing required field must still throw.
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(5)));

    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.INT32)
        .addField(2, DType.INT32).required().isOutput(false)  // hidden but required
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(RuntimeException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
          // unreachable: required b is missing, must throw even though hidden
        }
      });
    }
  }

  @Test
  void testAllFieldsHiddenProducesEmptyStruct() {
    // message Msg { int32 a = 1; int32 b = 2; } — both present on the wire but both hidden.
    // The result is a STRUCT with no children, still carrying the correct row count.
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(7)),
        box(tag(2, WT_VARINT)), box(encodeVarint(11)));

    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.INT32).isOutput(false)
        .addField(2, DType.INT32).isOutput(false)
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(0, result.getNumChildren());
      assertEquals(1, result.getRowCount());
    }
  }

  @Test
  void testTopLevelRepeatedMessageWithSimpleChildrenAcrossRows() {
    // message Item { int32 id = 1; string name = 2; bytes payload = 3; }
    // message Outer { repeated Item items = 1; }
    Byte[] item0 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(7)),
        box(tag(2, WT_LEN)), encodeString("a"),
        box(tag(3, WT_LEN)), encodeBytes(new byte[]{0x01, 0x02}));
    Byte[] item1 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(8)),
        box(tag(2, WT_LEN)), encodeString("b"),
        box(tag(3, WT_LEN)), encodeBytes(new byte[0]));
    Byte[] item2 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(9)));
    Byte[] row0 = concat(
        box(tag(1, WT_LEN)), encodeMessage(item0),
        box(tag(1, WT_LEN)), encodeMessage(item1));
    Byte[] row1 = new Byte[0];
    Byte[] row2 = concat(box(tag(1, WT_LEN)), encodeMessage(item2));

    ListType itemsType = new ListType(true,
        new StructType(true,
            new BasicType(true, DType.INT32),
            new BasicType(true, DType.STRING),
            new ListType(true, new BasicType(true, DType.UINT8))));
    StructType outputType = new StructType(true, itemsType);
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).repeated().down()
            .addField(1, DType.INT32)
            .addField(2, DType.STRING)
            .addField(3, DType.LIST)
        .up()
        .build();

    try (Table input = new Table.TestBuilder()
             .column(new Byte[][]{row0, row1, row2, null})
             .build();
         ColumnVector expected = ColumnVector.fromStructs(
             outputType,
             struct(Arrays.asList(
                 struct(7, "a", Arrays.asList((byte) 0x01, (byte) 0x02)),
                 struct(8, "b", Collections.emptyList()))),
             struct(Collections.emptyList()),
             struct(Collections.singletonList(struct(9, null, null))),
             null);
         ColumnVector actual = Protobuf.decodeToStruct(input.getColumn(0), schema, false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testZeroLengthRepeatedMessageElementPreservesPresence() {
    // A present zero-length message is one non-null struct element with an absent child.
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(new Byte[0]));
    ListType itemsType = new ListType(true,
        new StructType(true, new BasicType(true, DType.INT32)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedItems = ColumnVector.fromLists(
             itemsType, Collections.singletonList(struct((Object) null)));
         ColumnVector expected = ColumnVector.makeStruct(expectedItems);
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRUCT).repeated().down()
                     .addField(1, DType.INT32)
                 .up()
                 .build(),
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testMalformedRepeatedMessageElementNullsOnlyOwningTopRow_Permissive() {
    Byte[] valid0 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(7)));
    Byte[] malformed = concat(box(tag(1, WT_VARINT)), new Byte[]{(byte) 0x80});
    Byte[] valid1 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(9)));
    Byte[] row0 = concat(
        box(tag(1, WT_LEN)), encodeMessage(valid0),
        box(tag(1, WT_LEN)), encodeMessage(malformed));
    Byte[] row1 = concat(box(tag(1, WT_LEN)), encodeMessage(valid1));
    ListType itemsType = new ListType(true,
        new StructType(true, new BasicType(true, DType.INT32)));
    StructType outputType = new StructType(true, itemsType);
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).repeated().down()
            .addField(1, DType.INT32)
        .up()
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1}).build();
         ColumnVector expected = ColumnVector.fromStructs(
             outputType,
             null,
             struct(Collections.singletonList(struct(9))));
         ColumnVector actual = Protobuf.decodeToStruct(input.getColumn(0), schema, false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testMalformedRepeatedMessageElement_Failfast() {
    Byte[] valid = concat(box(tag(1, WT_VARINT)), box(encodeVarint(7)));
    Byte[] malformed = concat(box(tag(1, WT_VARINT)), new Byte[]{(byte) 0x80});
    Byte[] row = concat(
        box(tag(1, WT_LEN)), encodeMessage(valid),
        box(tag(1, WT_LEN)), encodeMessage(malformed));
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).repeated().down()
            .addField(1, DType.INT32)
        .up()
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
        }
      });
    }
  }

  @Test
  void testVisibleRequiredFieldInsideRepeatedMessageMissing_Permissive() {
    // The second element in row 0 omits required id=1; row 1 must remain valid.
    Byte[] valid0 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(2, WT_VARINT)), box(encodeVarint(10)));
    Byte[] missingRequired = concat(
        box(tag(2, WT_VARINT)), box(encodeVarint(20)));
    Byte[] valid1 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(2)),
        box(tag(2, WT_VARINT)), box(encodeVarint(30)));
    Byte[] row0 = concat(
        box(tag(1, WT_LEN)), encodeMessage(valid0),
        box(tag(1, WT_LEN)), encodeMessage(missingRequired));
    Byte[] row1 = concat(box(tag(1, WT_LEN)), encodeMessage(valid1));
    ListType itemsType = new ListType(true,
        new StructType(true,
            new BasicType(true, DType.INT32),
            new BasicType(true, DType.INT32)));
    StructType outputType = new StructType(true, itemsType);
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).repeated().down()
            .addField(1, DType.INT32).required()
            .addField(2, DType.INT32)
        .up()
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1}).build();
         ColumnVector expected = ColumnVector.fromStructs(
             outputType,
             null,
             struct(Collections.singletonList(struct(2, 30))));
         ColumnVector actual = Protobuf.decodeToStruct(input.getColumn(0), schema, false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testVisibleRequiredFieldInsideRepeatedMessageMissing_Failfast() {
    Byte[] valid = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(2, WT_VARINT)), box(encodeVarint(10)));
    Byte[] missingRequired = concat(
        box(tag(2, WT_VARINT)), box(encodeVarint(20)));
    Byte[] row = concat(
        box(tag(1, WT_LEN)), encodeMessage(valid),
        box(tag(1, WT_LEN)), encodeMessage(missingRequired));
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).repeated().down()
            .addField(1, DType.INT32).required()
            .addField(2, DType.INT32)
        .up()
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
        }
      });
    }
  }

  @Test
  void testRepeatedMessageInsideRepeatedMessageWithMultipleRows() {
    Byte[] child0 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(2, WT_LEN)), encodeString("a"));
    Byte[] child1 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(2)),
        box(tag(2, WT_LEN)), encodeString("b"));
    Byte[] child2 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(3)),
        box(tag(2, WT_LEN)), encodeString("c"));
    Byte[] parent0 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(10)),
        box(tag(2, WT_LEN)), encodeMessage(child0),
        box(tag(2, WT_LEN)), encodeMessage(child1));
    Byte[] parent1 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(20)));
    Byte[] parent2 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(30)),
        box(tag(2, WT_LEN)), encodeMessage(child2));
    Byte[] row0 = concat(
        box(tag(1, WT_LEN)), encodeMessage(parent0),
        box(tag(1, WT_LEN)), encodeMessage(parent1));
    Byte[] row1 = concat(box(tag(1, WT_LEN)), encodeMessage(parent2));
    Byte[] row2 = new Byte[0];
    ListType parentsType = new ListType(true,
        new StructType(true,
            new BasicType(true, DType.INT32),
            new ListType(true,
                new StructType(true,
                    new BasicType(true, DType.INT32),
                    new BasicType(true, DType.STRING)))));
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).repeated().down()
            .addField(1, DType.INT32)
            .addField(2, DType.STRUCT).repeated().down()
                .addField(1, DType.INT32)
                .addField(2, DType.STRING)
            .up()
        .up()
        .build();

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1, row2}).build();
         ColumnVector expectedParents = ColumnVector.fromLists(
             parentsType,
             Arrays.asList(
                 struct(10, Arrays.asList(struct(1, "a"), struct(2, "b"))),
                 struct(20, Collections.emptyList())),
             Collections.singletonList(
                 struct(30, Collections.singletonList(struct(3, "c")))),
             Collections.emptyList());
         ColumnVector expected = ColumnVector.makeStruct(expectedParents);
         ColumnVector actual = Protobuf.decodeToStruct(input.getColumn(0), schema, false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testSlicedInputPreservesRepeatedMessageInsideRepeatedMessage() {
    Byte[] child0 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(1)));
    Byte[] child1 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(2)));
    Byte[] child2 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(3)));
    Byte[] parent0 = concat(
        box(tag(1, WT_LEN)), encodeMessage(child0),
        box(tag(1, WT_LEN)), encodeMessage(child1));
    Byte[] parent1 = concat(box(tag(1, WT_LEN)), encodeMessage(child2));
    Byte[] row0 = concat(box(tag(1, WT_LEN)), encodeMessage(parent0));
    Byte[] row1 = concat(box(tag(1, WT_LEN)), encodeMessage(parent1));
    Byte[] sentinel = concat(box(tag(99, WT_VARINT)), box(encodeVarint(7)));
    ListType parentsType = new ListType(true,
        new StructType(true,
            new ListType(true,
                new StructType(true, new BasicType(true, DType.INT32)))));
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).repeated().down()
            .addField(1, DType.STRUCT).repeated().down()
                .addField(1, DType.INT32)
            .up()
        .up()
        .build();

    try (Table input = new Table.TestBuilder()
             .column(new Byte[][]{sentinel, row0, row1, sentinel})
             .build();
         ColumnVector expectedParents = ColumnVector.fromLists(
             parentsType,
             Collections.singletonList(
                 struct(Arrays.asList(struct(1), struct(2)))),
             Collections.singletonList(
                 struct(Collections.singletonList(struct(3)))));
         ColumnVector expected = ColumnVector.makeStruct(expectedParents)) {
      ColumnView[] views = input.getColumn(0).splitAsViews(1, 3);
      try (ColumnVector actual = Protobuf.decodeToStruct(views[1], schema, false)) {
        AssertUtils.assertStructColumnsAreEqual(expected, actual);
      } finally {
        for (ColumnView view : views) {
          view.close();
        }
      }
    }
  }
}
