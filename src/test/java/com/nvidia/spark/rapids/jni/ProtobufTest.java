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
  private static final int WT_32BIT = 5;

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

  private static void assertSingleNullStructRow(ColumnVector actual, String message) {
    try (HostColumnVector hostStruct = actual.copyToHost()) {
      assertEquals(1, actual.getNullCount(), message);
      assertTrue(hostStruct.isNull(0), "Row 0 should be null");
    }
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

    try (Table input = new Table.TestBuilder().column(row0, row1, row2).build();
         ColumnVector expectedId = ColumnVector.fromBoxedLongs(100L, 200L, null);
         ColumnVector expectedName = ColumnVector.fromStrings("alice", null, null);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedId, expectedName);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT64)
                 .addField(2, DType.STRING)
                 .build(),
             true)) {
      // Row 2 is a null input row — the output struct row should also be null.
      assertEquals(3, actualStruct.getRowCount());
      assertEquals(1, actualStruct.getNullCount());
      try (HostColumnVector hcv = actualStruct.copyToHost()) {
        assertFalse(hcv.isNull(0));
        assertFalse(hcv.isNull(1));
        assertTrue(hcv.isNull(2), "Null input row should produce null struct row");
      }
      // Children for non-null rows should still match
      try (ColumnVector childId = actualStruct.getChildColumnView(0).copyToColumnVector();
           HostColumnVector hostId = childId.copyToHost();
           ColumnVector childName = actualStruct.getChildColumnView(1).copyToColumnVector();
           HostColumnVector hostName = childName.copyToHost()) {
        assertEquals(100L, hostId.getLong(0));
        assertEquals(200L, hostId.getLong(1));
        assertEquals("alice", hostName.getJavaString(0));
        assertTrue(hostName.isNull(1));
      }
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
      try (ColumnVector expected = ColumnVector.fromBoxedLongs((Long)null);
           ColumnVector expectedStruct = ColumnVector.makeStruct(expected)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStruct, result);
      }
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
  void testFailOnErrorsTrue() {
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
             true)) {
        }
      });
    }
  }

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
  void testRequiredFieldWithMultipleRows() {
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
          "Null input row should produce a null child value, not protobuf_error::REQUIRED");
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
                 .addField(1, DType.STRING).defaultValue("hello".getBytes(java.nio.charset.StandardCharsets.UTF_8))
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
    byte[] strBytesRaw = "actual".getBytes(java.nio.charset.StandardCharsets.UTF_8);
    Byte[] row = concat(
        box(tag(1, WT_LEN)),
        box(encodeVarint(strBytesRaw.length)),
        box(strBytesRaw));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedStr = ColumnVector.fromStrings("actual");
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedStr);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING).defaultValue("default".getBytes(java.nio.charset.StandardCharsets.UTF_8))
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
                 .addField(2, DType.STRING).defaultValue("test".getBytes(java.nio.charset.StandardCharsets.UTF_8))
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
    byte[] strBytesRaw = "row2val".getBytes(java.nio.charset.StandardCharsets.UTF_8);
    Byte[] row2 = concat(
        box(tag(1, WT_LEN)),
        box(encodeVarint(strBytesRaw.length)),
        box(strBytesRaw));
    Byte[] row3 = new Byte[0];  // will use default

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row1, row2, row3}).build();
         ColumnVector expectedStr = ColumnVector.fromStrings("default", "row2val", "default");
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedStr);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING).defaultValue("default".getBytes(java.nio.charset.StandardCharsets.UTF_8))
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

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      try (ColumnVector result = Protobuf.decodeToStruct(
          input.getColumn(0),
          new ProtobufSchemaDescriptorBuilder()
              .addField(1, DType.INT32).repeated()  // ids
              .build(),
          false)) {
        // Result should be STRUCT<ids: LIST<INT32>> containing [1, 2, 3]
        assertNotNull(result);
        assertEquals(DType.STRUCT, result.getType());
        try (ColumnVector listCol = result.getChildColumnView(0).copyToColumnVector();
             ColumnVector children = listCol.getChildColumnView(0).copyToColumnVector();
             HostColumnVector hostChildren = children.copyToHost()) {
          assertEquals(DType.LIST, listCol.getType());
          assertEquals(DType.INT32, children.getType());
          assertEquals(3, children.getRowCount());
          assertEquals(1, hostChildren.getInt(0));
          assertEquals(2, hostChildren.getInt(1));
          assertEquals(3, hostChildren.getInt(2));
        }
      }
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

    // Helper to build packed int data (varints)
    java.io.ByteArrayOutputStream intBuf = new java.io.ByteArrayOutputStream();

    // Row 0: id=42, int_values=[1,-1,100] (12 bytes packed), double_values=[1.5,2.5], bool=[true,false]
    // Row 1: id=7, int_values=15x(-1) (150 bytes packed, 2-byte length varint!), double_values=[3.0,4.0], bool=[true]
    // Row 2: id=0, int_values=[] (field omitted), double_values=[5.0], bool=[] (field omitted)

    // --- Row 0 ---
    byte[] r0IntVarints = concatBytes(encodeVarint(1), encodeVarint(-1L & 0xFFFFFFFFFFFFFFFFL), encodeVarint(100));
    byte[] r0Doubles = concatBytes(encodeDouble(1.5), encodeDouble(2.5));
    byte[] r0Bools = new byte[]{0x01, 0x00};
    Byte[] row0 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(42)),
        box(tag(2, WT_LEN)), box(encodeVarint(r0IntVarints.length)), box(r0IntVarints),
        box(tag(3, WT_LEN)), box(encodeVarint(r0Doubles.length)), box(r0Doubles),
        box(tag(4, WT_LEN)), box(encodeVarint(r0Bools.length)), box(r0Bools));

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
        box(tag(2, WT_LEN)), box(encodeVarint(r1IntVarints.length)), box(r1IntVarints),
        box(tag(3, WT_LEN)), box(encodeVarint(r1Doubles.length)), box(r1Doubles),
        box(tag(4, WT_LEN)), box(encodeVarint(r1Bools.length)), box(r1Bools));

    // --- Row 2: no int_values, no bool_values ---
    byte[] r2Doubles = encodeDouble(5.0);
    Byte[] row2 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(0)),
        box(tag(3, WT_LEN)), box(encodeVarint(r2Doubles.length)), box(r2Doubles));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1, row2}).build()) {
      try (ColumnVector result = Protobuf.decodeToStruct(
          input.getColumn(0),
          new ProtobufSchemaDescriptorBuilder()
              .addField(1, DType.INT32)               // id
              .addField(2, DType.INT32).repeated()    // int_values (packed)
              .addField(3, DType.FLOAT64).repeated()  // double_values (packed)
              .addField(4, DType.BOOL8).repeated()    // bool_values (packed)
              .build(),
          false)) {
        assertNotNull(result);
        assertEquals(DType.STRUCT, result.getType());
        assertEquals(3, result.getRowCount());

        // Focus of this test is the multi-field packed-double layout (cross-row count
        // accumulation, child-column construction across mixed packed/unpacked rows); the
        // exact decoded values are spot-checked only on the double column. Per-element-type
        // round-trips are covered separately by testUnpackedRepeatedInt32, testRepeatedUint32,
        // testRepeatedUint64, testRepeatedString, testRepeatedBytes, and
        // testRepeatedEnumAsString.
        try (ColumnVector doubleListCol = result.getChildColumnView(2).copyToColumnVector()) {
          assertEquals(DType.LIST, doubleListCol.getType());
          try (ColumnVector doubleChildren = doubleListCol.getChildColumnView(0).copyToColumnVector()) {
            assertEquals(DType.FLOAT64, doubleChildren.getType());
            assertEquals(5, doubleChildren.getRowCount(),
                "Total packed doubles across 3 rows should be 5, got " + doubleChildren.getRowCount());
            try (HostColumnVector hd = doubleChildren.copyToHost()) {
              assertEquals(1.5, hd.getDouble(0), 1e-10);
              assertEquals(2.5, hd.getDouble(1), 1e-10);
              assertEquals(3.0, hd.getDouble(2), 1e-10);
              assertEquals(4.0, hd.getDouble(3), 1e-10);
              assertEquals(5.0, hd.getDouble(4), 1e-10);
            }
          }
        }
      }
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
      }
    }
  }

  @Test
  void testPermissiveRepeatedWrongWireTypeDoesNotCorruptFollowingRow() {
    // message Msg { repeated int32 ids = 1; }
    // Row 0 has one valid element, then a malformed fixed32 occurrence for the same field,
    // then another valid varint that must be ignored once the row is marked malformed.
    // Row 1 must keep its own slot and not be overwritten by row 0's trailing occurrence.
    Byte[] row0 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_32BIT)), box(encodeFixed32(77)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)));
    Byte[] row1 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(100)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1}).build();
         ColumnVector expectedIds = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)),
             Arrays.asList(1),
             Arrays.asList(100));
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
  void testRepeatedUint32() {
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)),
        box(tag(1, WT_VARINT)), box(encodeVarint(3)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.UINT32).repeated()
                 .build(),
             false)) {
      try (ColumnVector list = result.getChildColumnView(0).copyToColumnVector();
           ColumnVector vals = list.getChildColumnView(0).copyToColumnVector();
           HostColumnVector hostVals = vals.copyToHost()) {
        assertEquals(DType.UINT32, vals.getType());
        assertEquals(3, vals.getRowCount());
        assertEquals(1, Integer.toUnsignedLong(hostVals.getInt(0)));
        assertEquals(2, Integer.toUnsignedLong(hostVals.getInt(1)));
        assertEquals(3, Integer.toUnsignedLong(hostVals.getInt(2)));
      }
    }
  }

  @Test
  void testRepeatedUint64() {
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(11)),
        box(tag(1, WT_VARINT)), box(encodeVarint(22)),
        box(tag(1, WT_VARINT)), box(encodeVarint(33)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.UINT64).repeated()
                 .build(),
             false)) {
      try (ColumnVector list = result.getChildColumnView(0).copyToColumnVector();
           ColumnVector vals = list.getChildColumnView(0).copyToColumnVector();
           HostColumnVector hostVals = vals.copyToHost()) {
        assertEquals(DType.UINT64, vals.getType());
        assertEquals(3, vals.getRowCount());
        assertEquals(11L, hostVals.getLong(0));
        assertEquals(22L, hostVals.getLong(1));
        assertEquals(33L, hostVals.getLong(2));
      }
    }
  }

  // ============================================================================
  // FAILFAST Mode Tests (failOnErrors = true)
  // ============================================================================

  @Test
  void testFailfastMalformedVarint() {
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
  void testFailfastTruncatedVarint() {
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
  void testFailfastTruncatedString() {
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
  void testFailfastTruncatedFixed32() {
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
  void testFailfastTruncatedFixed64() {
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
  void testFailfastWrongWireType() {
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
  void testFailfastFieldNumberZero() {
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
  void testFailfastFieldNumberAboveSpecLimit() {
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
        box(tag(5, 4)),
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
  void testFailfastValidDataDoesNotThrow() {
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

    byte[][][] enumNames = new byte[][][] {
        new byte[][] {
            "RED".getBytes(java.nio.charset.StandardCharsets.UTF_8),
            "GREEN".getBytes(java.nio.charset.StandardCharsets.UTF_8),
            "BLUE".getBytes(java.nio.charset.StandardCharsets.UTF_8)
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

    byte[][][] enumNames = new byte[][][] {
        new byte[][] {
            "RED".getBytes(java.nio.charset.StandardCharsets.UTF_8),
            "GREEN".getBytes(java.nio.charset.StandardCharsets.UTF_8),
            "BLUE".getBytes(java.nio.charset.StandardCharsets.UTF_8)
        }
    };
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING).enumMetadata(new int[]{0, 1, 2}, enumNames[0])
                 .build(),
             false);
         HostColumnVector hostStruct = actual.copyToHost()) {
      assertEquals(1, actual.getNullCount(), "Struct row should be null for unknown enum value");
      assertTrue(hostStruct.isNull(0), "Row 0 should be null");
    }
  }

  @Test
  void testEnumAsStringMixedValidAndUnknown() {
    Byte[] row0 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(0)));    // RED
    Byte[] row1 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(999)));  // unknown
    Byte[] row2 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(2)));    // BLUE

    byte[][][] enumNames = new byte[][][] {
        new byte[][] {
            "RED".getBytes(java.nio.charset.StandardCharsets.UTF_8),
            "GREEN".getBytes(java.nio.charset.StandardCharsets.UTF_8),
            "BLUE".getBytes(java.nio.charset.StandardCharsets.UTF_8)
        }
    };
    try (Table input = new Table.TestBuilder().column(row0, row1, row2).build();
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING).enumMetadata(new int[]{0, 1, 2}, enumNames[0])
                 .build(),
             false);
         HostColumnVector hostStruct = actual.copyToHost()) {
      assertEquals(1, actual.getNullCount(), "Only the unknown enum row should be null");
      assertTrue(!hostStruct.isNull(0), "Row 0 should be valid");
      assertTrue(hostStruct.isNull(1), "Row 1 should be null");
      assertTrue(!hostStruct.isNull(2), "Row 2 should be valid");
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
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).enumValidValues(new int[]{0, 1, 2})
                 .build(),
             false);
         HostColumnVector hostStruct = actualStruct.copyToHost()) {
      // The struct itself should be null (not just the field)
      assertEquals(1, actualStruct.getNullCount(), "Struct row should be null for unknown enum");
      assertTrue(hostStruct.isNull(0), "Row 0 should be null");
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
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).enumValidValues(new int[]{0, 1, 2})
                 .build(),
             false);
         HostColumnVector hostStruct = actualStruct.copyToHost()) {
      // Check struct-level nulls
      assertEquals(2, actualStruct.getNullCount(), "Should have 2 null rows (rows 1 and 3)");
      assertTrue(!hostStruct.isNull(0), "Row 0 should be valid");
      assertTrue(hostStruct.isNull(1), "Row 1 should be null (unknown enum 999)");
      assertTrue(!hostStruct.isNull(2), "Row 2 should be valid");
      assertTrue(hostStruct.isNull(3), "Row 3 should be null (unknown enum -1)");
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
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).enumValidValues(new int[]{0, 1, 2})
                 .addField(2, DType.INT32)
                 .build(),
             false);
         HostColumnVector hostStruct = actualStruct.copyToHost()) {
      // The entire struct row should be null
      assertEquals(1, actualStruct.getNullCount(), "Struct row should be null");
      assertTrue(hostStruct.isNull(0), "Row 0 should be null due to unknown enum");
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
             false)) {
      // Struct row should be valid (not null), only the field is null
      assertEquals(0, actualStruct.getNullCount(), "Struct row should NOT be null for missing field");
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
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

    byte[][][] enumNames = new byte[][][] {
        new byte[][] {
            "RED".getBytes(java.nio.charset.StandardCharsets.UTF_8),
            "GREEN".getBytes(java.nio.charset.StandardCharsets.UTF_8),
            "BLUE".getBytes(java.nio.charset.StandardCharsets.UTF_8)
        }
    };
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING).repeated().enumMetadata(new int[]{0, 1, 2}, enumNames[0])
                 .build(),
             false)) {
      assertNotNull(actual);
      assertEquals(DType.STRUCT, actual.getType());
      assertEquals(1, actual.getNumChildren());
      try (ColumnView listCol = actual.getChildColumnView(0)) {
        assertEquals(DType.LIST, listCol.getType());
        try (ColumnView strChild = listCol.getChildColumnView(0);
             HostColumnVector hostStrs = strChild.copyToHost()) {
          assertEquals(3, hostStrs.getRowCount());
          assertEquals("RED", hostStrs.getJavaString(0));
          assertEquals("BLUE", hostStrs.getJavaString(1));
          assertEquals("GREEN", hostStrs.getJavaString(2));
        }
      }
    }
  }

  // ============================================================================
  // Edge case and boundary tests
  // ============================================================================

  @Test
  void testPackedFixedMisaligned() {
    byte[] packedData = new byte[]{0x01, 0x02, 0x03, 0x04, 0x05};
    Byte[] row = concat(
        box(tag(1, WT_LEN)),
        box(encodeVarint(packedData.length)),
        box(packedData));

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
  void testPackedFixedMisaligned64() {
    byte[] packedData = new byte[]{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09};
    Byte[] row = concat(
        box(tag(1, WT_LEN)),
        box(encodeVarint(packedData.length)),
        box(packedData));

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
    // PERMISSIVE counterpart of testPackedFixedMisaligned: instead of throwing on the
    // misaligned packed payload, the malformed row should produce an empty list (the
    // error fires inside walk_repeated_element before any occurrence is counted) and
    // a following well-formed row in the same batch must still decode normally.
    byte[] badPackedData = new byte[]{0x01, 0x02, 0x03, 0x04, 0x05};
    Byte[] row0 = concat(
        box(tag(1, WT_LEN)),
        box(encodeVarint(badPackedData.length)),
        box(badPackedData));
    Byte[] row1 = concat(
        box(tag(1, WT_32BIT)), box(encodeFixed32(42)),
        box(tag(1, WT_32BIT)), box(encodeFixed32(99)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1}).build();
         ColumnVector expectedIds = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)),
             Arrays.asList(),
             Arrays.asList(42, 99));
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedIds);
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
    // PERMISSIVE counterpart of testPackedFixedMisaligned64.
    byte[] badPackedData = new byte[]{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09};
    Byte[] row0 = concat(
        box(tag(1, WT_LEN)),
        box(encodeVarint(badPackedData.length)),
        box(badPackedData));
    Byte[] row1 = concat(
        box(tag(1, WT_64BIT)), box(encodeFixed64(7L)),
        box(tag(1, WT_64BIT)), box(encodeFixed64(11L)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1}).build();
         ColumnVector expectedIds = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT64)),
             Arrays.asList(),
             Arrays.asList(7L, 11L));
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedIds);
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
  void testLargeRepeatedField() throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    for (int i = 0; i < 100000; i++) {
      baos.write(tag(1, WT_VARINT));
      baos.write(encodeVarint(i));
    }
    Byte[] row = box(baos.toByteArray());

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).repeated()
                 .build(),
             false)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
      try (ColumnVector list = result.getChildColumnView(0).copyToColumnVector();
           ColumnVector vals = list.getChildColumnView(0).copyToColumnVector();
           HostColumnVector hostVals = vals.copyToHost()) {
        assertEquals(DType.LIST, list.getType());
        assertEquals(100000, vals.getRowCount(), "All 100K elements should be decoded");
        assertEquals(0, hostVals.getInt(0));
        assertEquals(50000, hostVals.getInt(50000));
        assertEquals(99999, hostVals.getInt(99999));
      }
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
        box(tag(1, WT_LEN)), box(encodeVarint(packedContent.length)), box(packedContent));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).repeated()
                 .build(),
             false)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
      try (ColumnVector list = result.getChildColumnView(0).copyToColumnVector();
           ColumnVector vals = list.getChildColumnView(0).copyToColumnVector();
           HostColumnVector hostVals = vals.copyToHost()) {
        assertEquals(4, vals.getRowCount());
        assertEquals(10, hostVals.getInt(0));
        assertEquals(20, hostVals.getInt(1));
        assertEquals(30, hostVals.getInt(2));
        assertEquals(40, hostVals.getInt(3));
      }
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
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(maxFieldNumber, DType.INT32)
                 .build(),
             false)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
      try (ColumnVector child = result.getChildColumnView(0).copyToColumnVector();
           HostColumnVector hostChild = child.copyToHost()) {
        assertEquals(42, hostChild.getInt(0));
      }
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
  void testNestedEnumAsStringInvalidKeepsSiblingFieldsVisible() {
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
         ColumnVector actual = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32)
                 .addField(2, DType.STRUCT).down()
                     .addField(1, DType.STRING).encoding(Protobuf.ENC_ENUM_STRING)
                         .enumMetadata(new int[]{0, 1, 2}, enumNames)
                     .addField(2, DType.INT32)
                 .up()
                 .addField(3, DType.STRING)
                 .build(),
             false);
         ColumnVector inner = actual.getChildColumnView(1).copyToColumnVector();
         ColumnVector status = inner.getChildColumnView(0).copyToColumnVector();
         ColumnVector count = inner.getChildColumnView(1).copyToColumnVector();
         HostColumnVector hostOuter = actual.copyToHost();
         HostColumnVector hostInner = inner.copyToHost();
         HostColumnVector hostStatus = status.copyToHost();
         HostColumnVector hostCount = count.copyToHost()) {
      assertEquals(0, actual.getNullCount(), "Invalid nested enum should not null outer rows");
      assertFalse(hostOuter.isNull(0));
      assertFalse(hostOuter.isNull(1));
      assertEquals(0, inner.getNullCount(), "Nested struct rows should stay present");
      assertFalse(hostInner.isNull(1));
      assertEquals("OK", hostStatus.getJavaString(0));
      assertTrue(hostStatus.isNull(1), "Unknown nested enum should null only the enum field");
      assertEquals(20, hostCount.getInt(1));
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
  void testMalformedChildlessNestedMessage_FailsFast() {
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
  void testNestedRepeatedWrongWireType_FailsFast() {
    // message Inner { repeated int32 x = 1; }
    // message Outer { Inner inner = 1; }
    Byte[] innerMessage = concat(box(tag(1, WT_32BIT)), box(encodeFixed32(7)));
    Byte[] row = concat(box(tag(1, WT_LEN)), encodeMessage(innerMessage));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = Protobuf.decodeToStruct(
            input.getColumn(0),
            new ProtobufSchemaDescriptorBuilder()
                .addField(1, DType.STRUCT).down()
                    .addField(1, DType.INT32).repeated()
                .up()
                .build(),
            true)) {
        }
      });
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
    byte[] s1 = "hello".getBytes(java.nio.charset.StandardCharsets.UTF_8);
    byte[] s2 = "world".getBytes(java.nio.charset.StandardCharsets.UTF_8);
    byte[] s3 = "foo".getBytes(java.nio.charset.StandardCharsets.UTF_8);
    Byte[] row0 = concat(
        box(tag(1, WT_LEN)), box(encodeVarint(s1.length)), box(s1),
        box(tag(1, WT_LEN)), box(encodeVarint(s2.length)), box(s2));
    Byte[] row1 = concat(
        box(tag(1, WT_LEN)), box(encodeVarint(s3.length)), box(s3));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.STRING).repeated()
                 .build(),
             false)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(2, result.getRowCount());
      try (ColumnView listCol = result.getChildColumnView(0)) {
        assertEquals(DType.LIST, listCol.getType());
        try (ColumnView strChild = listCol.getChildColumnView(0);
             HostColumnVector hcv = strChild.copyToHost()) {
          assertEquals(3, hcv.getRowCount());
          assertEquals("hello", hcv.getJavaString(0));
          assertEquals("world", hcv.getJavaString(1));
          assertEquals("foo", hcv.getJavaString(2));
        }
      }
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
        box(tag(1, WT_LEN)), box(encodeVarint(b1.length)), box(b1),
        box(tag(1, WT_LEN)), box(encodeVarint(b2.length)), box(b2));
    Byte[] row1 = concat(
        box(tag(1, WT_LEN)), box(encodeVarint(b3.length)), box(b3));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1}).build();
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.LIST).repeated()  // repeated bytes -> LIST<UINT8> element
                 .build(),
             false)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(2, result.getRowCount());
      // result.child(0) is LIST<LIST<UINT8>>; the outer offsets are [0, 2, 3], the inner
      // child holds the concatenated bytes [b1, b2, b3].
      try (ColumnView outerList = result.getChildColumnView(0);
           ColumnView innerList = outerList.getChildColumnView(0);
           ColumnView bytesChild = innerList.getChildColumnView(0);
           HostColumnVector hOuterOff = outerList.getListOffsetsView().copyToColumnVector().copyToHost();
           HostColumnVector hInnerOff = innerList.getListOffsetsView().copyToColumnVector().copyToHost();
           HostColumnVector hBytes    = bytesChild.copyToHost()) {
        assertEquals(DType.LIST, outerList.getType());
        assertEquals(DType.LIST, innerList.getType());
        assertEquals(DType.UINT8, bytesChild.getType());

        assertEquals(0, hOuterOff.getInt(0));
        assertEquals(2, hOuterOff.getInt(1));
        assertEquals(3, hOuterOff.getInt(2));

        // Three byte-string elements total
        assertEquals(0, hInnerOff.getInt(0));
        assertEquals(b1.length, hInnerOff.getInt(1));
        assertEquals(b1.length + b2.length, hInnerOff.getInt(2));
        assertEquals(b1.length + b2.length + b3.length, hInnerOff.getInt(3));

        byte[] expected = new byte[b1.length + b2.length + b3.length];
        System.arraycopy(b1, 0, expected, 0, b1.length);
        System.arraycopy(b2, 0, expected, b1.length, b2.length);
        System.arraycopy(b3, 0, expected, b1.length + b2.length, b3.length);
        for (int i = 0; i < expected.length; i++) {
          assertEquals(expected[i], hBytes.getByte(i), "byte mismatch at " + i);
        }
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
         ColumnVector result = Protobuf.decodeToStruct(
             input.getColumn(0),
             new ProtobufSchemaDescriptorBuilder()
                 .addField(1, DType.INT32).repeated().encoding(Protobuf.ENC_ZIGZAG)
                 .build(),
             false);
         ColumnView listCol = result.getChildColumnView(0);
         ColumnView vals = listCol.getChildColumnView(0);
         HostColumnVector hcv = vals.copyToHost()) {
      assertEquals(3, hcv.getRowCount());
      assertEquals(-1, hcv.getInt(0));
      assertEquals(-2, hcv.getInt(1));
      assertEquals(3, hcv.getInt(2));
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
    // Hits protobuf_error::SCHEMA_TOO_LARGE: the scan_all_repeated_occurrences_kernel stack-array
    // guard rejects schemas with more than 32 top-level repeated fields.
    int n = 33;
    ProtobufSchemaDescriptorBuilder builder = new ProtobufSchemaDescriptorBuilder();
    for (int i = 0; i < n; i++) {
      builder.addField(i + 1, DType.INT32).repeated();
    }
    ProtobufSchemaDescriptor schema = builder.build();

    // The MAX_STACK_FIELDS guard in scan_all_repeated_occurrences_kernel only fires when
    // every field actually has occurrences (zero-count fields are filtered out before the
    // kernel launch). Encode one occurrence per field so num_scan_fields == n.
    java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
    try {
      for (int i = 0; i < n; i++) {
        baos.write(tag(i + 1, WT_VARINT));
        baos.write(encodeVarint(1));
      }
    } catch (java.io.IOException e) {
      throw new RuntimeException(e);
    }
    Byte[] row = box(baos.toByteArray());
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(RuntimeException.class, () -> {
        try (ColumnVector ignored = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
          // unreachable: should throw
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
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(1, result.getNumChildren());
      try (ColumnVector childA = result.getChildColumnView(0).copyToColumnVector();
           ColumnVector expectedA = ColumnVector.fromBoxedInts(7)) {
        AssertUtils.assertColumnsAreEqual(expectedA, childA);
      }
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
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(1, result.getNumChildren());
      try (ColumnVector childA = result.getChildColumnView(0).copyToColumnVector();
           ColumnVector expectedA = ColumnVector.fromBoxedInts(7)) {
        AssertUtils.assertColumnsAreEqual(expectedA, childA);
      }
    }
  }

  @Test
  void testHiddenRequiredFieldStillValidates() {
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
}
