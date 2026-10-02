/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.*;
import com.nvidia.spark.rapids.jni.Arms;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Isolated;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FilterInputStream;
import java.io.IOException;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static java.lang.Math.toIntExact;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static java.util.Collections.singletonList;
import static org.junit.jupiter.api.Assertions.*;

@Isolated("Read failure tests temporarily replace the default host memory allocator")
public class KudoSerializerTest extends CudfTestBase {
  private static final Logger log = LoggerFactory.getLogger(KudoSerializerTest.class);

  static final long RMM_POOL_SIZE_LARGE = 10L * 1024 * 1024 * 1024;

  public KudoSerializerTest() {
    super(RmmAllocationMode.POOL, RMM_POOL_SIZE_LARGE);
  }

  @Test
  public void testSerializeAndDeserializeTable() {
    try(Table expected = buildTestTable()) {
      int rowCount = toIntExact(expected.getRowCount());
      for (int sliceSize = rowCount; sliceSize >= 1; sliceSize--) {
        List<TableSlice> tableSlices = new ArrayList<>();
        for (int startRow = 0; startRow < rowCount; startRow += sliceSize) {
          tableSlices.add(new TableSlice(startRow, Math.min(sliceSize, rowCount - startRow), expected));
        }

        checkMergeTable(expected, tableSlices);
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  @Test
  public void testReadIOExceptionClosesBuffer() throws IOException {
    checkReadFailureClosesBuffer(new IOException("read"));
  }

  @Test
  public void testReadOutOfMemoryErrorClosesBuffer() throws IOException {
    checkReadFailureClosesBuffer(new OutOfMemoryError("read"));
  }

  private void checkReadFailureClosesBuffer(Throwable failure) throws IOException {
    KudoTableHeader header = new KudoTableHeader(0, 1, 0, 0, 4, 1, new byte[]{0});
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    header.writeTo(new DataOutputStreamWriter(new DataOutputStream(bytes)));
    FilterInputStream input = new FilterInputStream(new ByteArrayInputStream(bytes.toByteArray())) {
      @Override
      public int read(byte[] bytes, int offset, int length) throws IOException {
        if (in.available() == 0) {
          if (failure instanceof Error) {
            throw (Error) failure;
          }
          throw (IOException) failure;
        }
        return super.read(bytes, offset, length);
      }
    };

    HostMemoryAllocator originalAllocator = DefaultHostMemoryAllocator.get();
    HostMemoryBuffer[] allocated = new HostMemoryBuffer[1];
    DefaultHostMemoryAllocator.set(new DefaultHostMemoryAllocator() {
      @Override
      public HostMemoryBuffer allocate(long size, boolean preferPinned) {
        allocated[0] = originalAllocator.allocate(size, preferPinned);
        return allocated[0];
      }
    });
    try {
      assertSame(failure, assertThrows(failure.getClass(), () -> KudoTable.from(input)));
      assertNotNull(allocated[0]);
      assertEquals(0, allocated[0].getRefCount());
    } finally {
      DefaultHostMemoryAllocator.set(originalAllocator);
      if (allocated[0] != null && allocated[0].getRefCount() > 0) {
        allocated[0].close();
      }
    }
  }

  @Test
  public void testRowCountOnly() throws Exception {
    OpenByteArrayOutputStream out = new OpenByteArrayOutputStream();
    long bytesWritten = KudoSerializer.writeRowCountToStream(out, 5);
    assertEquals(28, bytesWritten);

    ByteArrayInputStream in = new ByteArrayInputStream(out.toByteArray());
    KudoTableHeader header = KudoTableHeader.readFrom(new DataInputStream(in)).get();

    assertEquals(0, header.getNumColumns());
    assertEquals(0, header.getOffset());
    assertEquals(5, header.getNumRows());
    assertEquals(0, header.getValidityBufferLen());
    assertEquals(0, header.getOffsetBufferLen());
    assertEquals(0, header.getTotalDataLen());
  }

  @Test
  public void testSerializeAndDeserializeEmptyStructTable() {
    try(Table expected = buildEmptyStructTable()) {
      int rowCount = toIntExact(expected.getRowCount());
      for (int sliceSize = rowCount; sliceSize >= 1; sliceSize--) {
        List<TableSlice> tableSlices = new ArrayList<>();
        for (int startRow = 0; startRow < rowCount; startRow += sliceSize) {
          tableSlices.add(new TableSlice(startRow, Math.min(sliceSize, rowCount - startRow), expected));
        }

        checkMergeTable(expected, tableSlices);
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  @Test
  public void testWriteSimple() throws Exception {
    KudoSerializer serializer = new KudoSerializer(buildSimpleTestSchema());

    try (Table t = buildSimpleTable()) {
      OpenByteArrayOutputStream out = new OpenByteArrayOutputStream();
      long bytesWritten = serializer.writeToStreamWithMetrics(t, out, 0, 4).getWrittenBytes();

      assertEquals(172, bytesWritten);

      ByteArrayInputStream in = new ByteArrayInputStream(out.toByteArray());

      KudoTableHeader header = KudoTableHeader.readFrom(new DataInputStream(in)).get();
      assertEquals(7, header.getNumColumns());
      assertEquals(0, header.getOffset());
      assertEquals(4, header.getNumRows());
      assertEquals(7, header.getValidityBufferLen());
      assertEquals(40, header.getOffsetBufferLen());
      assertEquals(143, header.getTotalDataLen());

      // First integer column has no validity buffer
      assertFalse(header.hasValidityBuffer(0));
      for (int i = 1; i < 7; i++) {
        assertTrue(header.hasValidityBuffer(i));
      }
    }
  }

  @Test
  public void testMergeTableWithDifferentValidity() {
    Arms.withResource(new ArrayList<Table>(), tables -> {
      Table table1 = new Table.TestBuilder()
          .column(-83182L, 5822L, 3389L, 7384L, 7297L)
          .column(-2.06, -2.14, 8.04, 1.16, -1.0)
          .build();
      tables.add(table1);

      Table table2 = new Table.TestBuilder()
          .column(-47L, null, -83L, -166L, -220L, 470L, 619L, 803L, 661L)
          .column(-6.08, 1.6, 1.78, -8.01, 1.22, 1.43, 2.13, -1.65, null)
          .build();
      tables.add(table2);

      Table table3 = new Table.TestBuilder()
          .column(8722L, 8733L)
          .column(2.51, 0.0)
          .build();
      tables.add(table3);


      Table expected = new Table.TestBuilder()
          .column(7384L, 7297L, 803L, 661L, 8733L)
          .column(1.16, -1.0, -1.65, null, 0.0)
          .build();
      tables.add(expected);

      checkMergeTable(expected, asList(
          new TableSlice(3, 2, table1),
          new TableSlice(7, 2, table2),
          new TableSlice(1, 1, table3)));
      return null;
    });
  }

  @Test
  public void testMergeString() {
      Arms.withResource(new ArrayList<Table>(), tables -> {
                  Table table1 = new Table.TestBuilder()
                          .column("A", "B", "C", "D", null, "TESTING", "1", "2", "3", "4",
                                  "5", "6", "7", null, "9", "10", "11", "12", "13", null, "15")
                          .build();
                  tables.add(table1);

                  Table table2 = new Table.TestBuilder()
                          .column("A", "A", "C", "C", "E", "TESTING", "1", "2", "3", "4", "5",
                                  "6", "7", "", "9", "10", "11", "12", "13", "", "15")
                          .build();
                  tables.add(table2);

                  Table expected = new Table.TestBuilder()
                          .column("C", "D", null, "TESTING", "1", "2", "3", "4",
                                  "5", "6", "7", null, "9", "C", "E", "TESTING", "1", "2")
                          .build();
                  tables.add(expected);

                  checkMergeTable(expected, asList(
                          new TableSlice(2, 13, table1),
                          new TableSlice(3, 5, table2)));

                  return null;
              }
      );
  }

  @Test
  public void testMergeList() {
    Arms.withResource(new ArrayList<Table>(), tables -> {
      Table table1 = new Table.TestBuilder()
          .column(-881L, 482L, 660L, 896L, -129L, -108L, -428L, 0L, 617L, 782L)
          .column(integers(665), integers(-267), integers(398), integers(-314),
              integers(-370), integers(181), integers(665, 544), integers(222), integers(-587),
              integers(544))
          .build();
      tables.add(table1);

      Table table2 = new Table.TestBuilder()
          .column(-881L, 482L, 660L, 896L, 122L, 241L, 281L, 680L, 783L, null)
          .column(integers(-370), integers(398), integers(-587, 398), integers(-314),
              integers(307), integers(-397, -633), integers(-314, 307), integers(-633), integers(-397),
              integers(181, -919, -175))
          .build();
      tables.add(table2);

      Table expected = new Table.TestBuilder()
          .column(896L, -129L, -108L, -428L, 0L, 617L, 782L, 482L, 660L, 896L, 122L, 241L,
              281L, 680L, 783L, null)
          .column(integers(-314), integers(-370), integers(181), integers(665, 544), integers(222),
              integers(-587), integers(544), integers(398), integers(-587, 398), integers(-314),
              integers(307), integers(-397, -633), integers(-314, 307), integers(-633), integers(-397),
              integers(181, -919, -175))
          .build();
      tables.add(expected);

      checkMergeTable(expected, asList(
          new TableSlice(3, 7, table1),
          new TableSlice(1, 9, table2)));

      return null;
    });
  }

  @Test
  public void testMergeComplexStructList() {
    Arms.withResource(new ArrayList<Table>(), tables -> {
      HostColumnVector.ListType listMapType = new HostColumnVector.ListType(true,
              new HostColumnVector.ListType(true,
                      new HostColumnVector.StructType(true,
                              new HostColumnVector.BasicType(false, DType.STRING),
                              new HostColumnVector.BasicType(true, DType.STRING))));

      Table table = new Table.TestBuilder()
              .column(listMapType, asList(asList(struct("k1", "v1"), struct("k2", "v2")),
                              singletonList(struct("k3", "v3"))),
                      null,
                      singletonList(asList(struct("k14", "v14"), struct("k15", "v15"))),
                      null,
                      asList(null, null, null),
                      asList(singletonList(struct("k22", null)), singletonList(struct("k23", null))),
                      null, null,
                      null)
              .build();
      tables.add(table);

      checkMergeTable(table, asList(
              new TableSlice(0, 3, table),
              new TableSlice(3, 3, table),
              new TableSlice(6, 3, table))
      );
      return null;
    });
  }


  @Test
  public void testSerializeValidity() {
    Arms.withResource(new ArrayList<Table>(), tables -> {
      List<Integer> col1 = new ArrayList<>(512);
      col1.add(null);
      col1.add(null);
      col1.addAll(IntStream.range(2, 512).boxed().collect(Collectors.toList()));

      Table table1 = new Table.TestBuilder()
          .column(col1.toArray(new Integer[0]))
          .build();
      tables.add(table1);

      Table table2 = new Table.TestBuilder()
          .column(509, 510, 511)
          .build();
      tables.add(table2);

      checkMergeTable(table2, asList(new TableSlice(509, 3, table1)));
      return null;
    });
  }

  @Test
  public void testByteArrayOutputStreamWriter() throws Exception {
    ByteArrayOutputStream bout = new ByteArrayOutputStream(32);
    DataWriter writer = new ByteArrayOutputStreamWriter(bout);

    writer.writeInt(0x12345678);

    byte[] testByteArr1 = new byte[2097];
    ThreadLocalRandom.current().nextBytes(testByteArr1);
    writer.write(testByteArr1, 0, testByteArr1.length);

    byte[] testByteArr2 = new byte[7896];
    ThreadLocalRandom.current().nextBytes(testByteArr2);
    try(HostMemoryBuffer buffer = HostMemoryBuffer.allocate(testByteArr2.length)) {
      buffer.setBytes(0, testByteArr2, 0, testByteArr2.length);
      writer.copyDataFrom(buffer, 0, testByteArr2.length);
    }

    byte[] expected = new byte[4 + testByteArr1.length + testByteArr2.length];
    expected[0] = 0x12;
    expected[1] = 0x34;
    expected[2] = 0x56;
    expected[3] = 0x78;
    System.arraycopy(testByteArr1, 0, expected, 4, testByteArr1.length);
    System.arraycopy(testByteArr2, 0, expected, 4 + testByteArr1.length,
        testByteArr2.length);

    assertArrayEquals(expected, bout.toByteArray());
  }


  private static Schema buildSimpleTestSchema() {
    Schema.Builder builder = Schema.builder();

    builder.addColumn(DType.INT32, "a");
    builder.addColumn(DType.STRING, "b");
    Schema.Builder listBuilder = builder.addColumn(DType.LIST, "c");
    listBuilder.addColumn(DType.INT32, "c1");

    Schema.Builder structBuilder = builder.addColumn(DType.STRUCT, "d");
    structBuilder.addColumn(DType.INT8, "d1");
    structBuilder.addColumn(DType.INT64, "d2");

    return builder.build();
  }

  static Table buildSimpleTable() {
    HostColumnVector.StructType st = new HostColumnVector.StructType(
        true,
        new HostColumnVector.BasicType(true, DType.INT8),
        new HostColumnVector.BasicType(true, DType.INT64)
    );
    return new Table.TestBuilder()
        .column(1, 2, 3, 4)
        .column("1", "12", null, "45")
        .column(new Integer[]{1, null, 3}, new Integer[]{4, 5, 6}, null, new Integer[]{7, 8, 9})
        .column(st, new HostColumnVector.StructData((byte) 1, 11L),
            new HostColumnVector.StructData((byte) 2, null), null,
            new HostColumnVector.StructData((byte) 3, 33L))
        .build();
  }

  static Table buildEmptyStructTable() {
    HostColumnVector.StructType st = new HostColumnVector.StructType(true);
    return new Table.TestBuilder()
        .column(st,
            struct(), null, null, struct(), null, null, struct(), struct(),
            null, struct(), struct(), null, struct(), struct(), null, null,
            struct(), null, null, struct(), null, null, struct(), struct(),
            null, struct(), struct(), null, struct(), struct(), null, null,
            struct(), struct(), null, struct(), null, null, struct(), struct(),
            null, struct(), struct(), null, struct(), null, null, null,
            struct(), null, null, struct(), null, struct(), struct(), null,
            null, struct(), struct(), null, struct(), struct(), null, null,
            struct(), null, null, struct(), null, null, struct(), struct(),
            null, struct(), struct(), null, struct(), struct(), null, null,
            struct(), null, null, null, null, null, struct(), struct(),
            null, struct(), struct(), null, struct(), struct(), null, null,
            struct())
        .build();
  }

  static Table buildTestTable() {
    HostColumnVector.ListType listMapType = new HostColumnVector.ListType(true,
        new HostColumnVector.ListType(true,
            new HostColumnVector.StructType(true,
                new HostColumnVector.BasicType(false, DType.STRING),
                new HostColumnVector.BasicType(true, DType.STRING))));
    HostColumnVector.ListType mapStructType = new HostColumnVector.ListType(true,
        new HostColumnVector.StructType(true,
            new HostColumnVector.BasicType(false, DType.STRING),
            new HostColumnVector.BasicType(false, DType.STRING)));
    HostColumnVector.StructType structType = new HostColumnVector.StructType(true,
        new HostColumnVector.BasicType(true, DType.INT32),
        new HostColumnVector.BasicType(false, DType.FLOAT32));
    HostColumnVector.ListType listDateType = new HostColumnVector.ListType(true,
        new HostColumnVector.StructType(false,
            new HostColumnVector.BasicType(false, DType.INT32),
            new HostColumnVector.BasicType(true, DType.INT32)));

    return new Table.TestBuilder()
        .column(100, 202, 3003, 40004, 5, -60, 1, null, 3, null, 5, null, 7, null, 9, null, 11, null, 13, null, 15)
        .column(true, true, false, false, true, null, true, true, null, false, false, null, true,
            true, null, false, false, null, true, true, null)
        .column((byte)1, (byte)2, null, (byte)4, (byte)5,(byte)6,(byte)1,(byte)2,(byte)3, null,(byte)5, (byte)6,
            (byte) 7, null,(byte) 9,(byte) 10,(byte) 11, null,(byte) 13,(byte) 14,(byte) 15)
        .column((short)6, (short)5, (short)4, null, (short)2, (short)1,
            (short)1, (short)2, (short)3, null, (short)5, (short)6, (short)7, null, (short)9,
            (short)10, null, (short)12, (short)13, (short)14, null)
        .column(1L, null, 1001L, 50L, -2000L, null, 1L, 2L, 3L, 4L, null, 6L,
            7L, 8L, 9L, null, 11L, 12L, 13L, 14L, null)
        .column(10.1f, 20f, -1f, 3.1415f, -60f, null, 1f, 2f, 3f, 4f, 5f, null, 7f, 8f, 9f, 10f, 11f, null, 13f, 14f, 15f)
        .column(10.1f, 20f, -2f, 3.1415f, -60f, -50f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f)
        .column(10.1, 20.0, 33.1, 3.1415, -60.5, null, 1d, 2.0, 3.0, 4.0, 5.0,
            6.0, null, 8.0, 9.0, 10.0, 11.0, 12.0, null, 14.0, 15.0)
        .column((Float)null, null, null, null, null, null, null, null, null, null,
            null, null, null, null, null, null, null, null, null, null, null)
        .timestampDayColumn(99, 100, 101, 102, 103, 104, 1, 2, 3, 4, 5, 6, 7, null, 9, 10, 11, 12, 13, null, 15)
        .timestampMillisecondsColumn(9L, 1006L, 101L, 5092L, null, 88L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, null, 10L, 11L, 12L, 13L, 14L, 15L)
        .timestampSecondsColumn(1L, null, 3L, 4L, 5L, 6L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, null, 11L, 12L, 13L, 14L, 15L)
        .decimal32Column(-3, 100, 202, 3003, 40004, 5, -60, 1, null, 3,
            null, 5, null, 7, null, 9, null, 11, null, 13, null, 15)
        .decimal64Column(-8, 1L, null, 1001L, 50L, -2000L, null, 1L, 2L, 3L,
            4L, null, 6L, 7L, 8L, 9L, null, 11L, 12L, 13L, 14L, null)
        .decimal128Column(-2, RoundingMode.UNNECESSARY, new BigInteger("1"), null, new BigInteger("1001"),
            new BigInteger("50"), new BigInteger("-2000"), null, new BigInteger("1"), new BigInteger("2"),
            new BigInteger("3"), new BigInteger("4"), null, new BigInteger("6"), new BigInteger("7"),
            new BigInteger("8"), new BigInteger("9"), null, new BigInteger("11"), new BigInteger("12"),
            new BigInteger("13"), new BigInteger("14"), null)
        .column("A", "B", "C", "D", null, "TESTING", "1", "2", "3", "4",
            "5", "6", "7", null, "9", "10", "11", "12", "13", null, "15")
        .column("A", "A", "C", "C", "E", "TESTING", "1", "2", "3", "4", "5",
            "6", "7", "", "9", "10", "11", "12", "13", "", "15")
        .column("", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
            "", "", "", "", "", "")
        .column("", null, "", "", null, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "")
        .column((String)null, null, null, null, null, null, null, null, null, null,
            null, null, null, null, null, null, null, null, null, null, null)
        .column(mapStructType, structs(struct("1", "2")), structs(struct("3", "4")), null, null,
            structs(struct("key", "value"), struct("a", "b")), null, null,
            structs(struct("3", "4"), struct("1", "2")), structs(),
            structs(null, struct("foo", "bar")),
            structs(null, null, null), null, null, null, null, null, null, null, null, null,
            structs(struct("the", "end")))
        .column(structType, struct(1, 1f), null, struct(2, 3f),
            null, struct(8, 7f), struct(0, 0f), null,
            null, struct(-1, -1f), struct(-100, -100f),
            struct(Integer.MAX_VALUE, Float.MAX_VALUE), null,
            null, null,
            null, null,
            null, null,
            null, null,
            struct(Integer.MIN_VALUE, Float.MIN_VALUE))
        .column(integers(1, 2), null, integers(3, 4, null, 5, null), null, null, integers(6, 7, 8),
            integers(null, null, null), integers(1, 2, 3), integers(4, 5, 6), integers(7, 8, 9),
            integers(10, 11, 12), integers((Integer)null), integers(14, null),
            integers(14, 15, null, 16, 17, 18), integers(19, 20, 21), integers(22, 23, 24),
            integers(25, 26, 27), integers(28, 29, 30), integers(31, 32, 33), null,
            integers(37, 38, 39))
        .column(integers(), integers(), integers(), integers(), integers(), integers(), integers(),
            integers(), integers(), integers(), integers(), integers(), integers(), integers(),
            integers(), integers(), integers(), integers(), integers(), integers(), integers())
        .column(integers(null, null), integers(null, null, null, null), integers(),
            integers(null, null, null), integers(), integers(null, null, null, null, null),
            integers((Integer)null), integers(null, null, null), integers(null, null),
            integers(null, null, null, null), integers(null, null, null, null, null), integers(),
            integers(null, null, null, null), integers(null, null, null), integers(null, null),
            integers(null, null, null), integers(null, null), integers((Integer)null),
            integers((Integer)null), integers(null, null),
            integers(null, null, null, null, null))
        .column((Integer)null, null, null, null, null, null, null, null, null, null,
            null, null, null, null, null, null, null, null, null, null, null)
        .column(strings("1", "2", "3"), strings("4"), strings("5"), strings("6, 7"),
            strings("", "9", null), strings("11"), strings(""), strings(null, null),
            strings("15", null), null, null, strings("18", "19", "20"), null, strings("22"),
            strings("23", ""), null, null, null, null, strings(), strings("the end"))
        .column(strings(), strings(), strings(), strings(), strings(), strings(), strings(),
            strings(), strings(), strings(), strings(), strings(), strings(), strings(), strings(),
            strings(), strings(), strings(), strings(), strings(), strings())
        .column(strings(null, null), strings(null, null, null, null), strings(),
            strings(null, null, null), strings(), strings(null, null, null, null, null),
            strings((String)null), strings(null, null, null), strings(null, null),
            strings(null, null, null, null), strings(null, null, null, null, null), strings(),
            strings(null, null, null, null), strings(null, null, null), strings(null, null),
            strings(null, null, null), strings(null, null), strings((String)null),
            strings((String)null), strings(null, null),
            strings(null, null, null, null, null))
        .column((String)null, null, null, null, null, null, null, null, null, null,
            null, null, null, null, null, null, null, null, null, null, null)
        .column(listMapType, asList(asList(struct("k1", "v1"), struct("k2", "v2")),
                singletonList(struct("k3", "v3"))),
            asList(asList(struct("k4", "v4"), struct("k5", "v5"),
                struct("k6", "v6")), singletonList(struct("k7", "v7"))),
            null, null, null, asList(asList(struct("k8", "v8"), struct("k9", "v9")),
                asList(struct("k10", "v10"), struct("k11", "v11"), struct("k12", "v12"),
                    struct("k13", "v13"))),
            singletonList(asList(struct("k14", "v14"), struct("k15", "v15"))), null, null, null, null,
            asList(asList(struct("k16", "v16"), struct("k17", "v17")),
                singletonList(struct("k18", "v18"))),
            asList(asList(struct("k19", "v19"), struct("k20", "v20")),
                singletonList(struct("k21", "v21"))),
            asList(singletonList(struct("k22", "v22")), singletonList(struct("k23", "v23"))),
            asList(null, null, null),
            asList(singletonList(struct("k22", null)), singletonList(struct("k23", null))),
            null, null, null, null, null)
        .column(listDateType, asList(struct(-210, 293), struct(-719, 205), struct(-509, 183),
                struct(174, 122), struct(647, 683)),
            asList(struct(311, 992), struct(-169, 482), struct(166, 525)),
            asList(struct(156, 197), struct(926, 134), struct(747, 312), struct(293, 801)),
            asList(struct(647, null), struct(293, 387)), emptyList(),
            null, emptyList(), null,
            asList(struct(-210, 293), struct(-719, 205), struct(-509, 183), struct(174, 122),
                struct(647, 683)),
            asList(struct(311, 992), struct(-169, 482), struct(166, 525)),
            asList(struct(156, 197), struct(926, 134), struct(747, 312), struct(293, 801)),
            asList(struct(647, null), struct(293, 387)), emptyList(), null,
            emptyList(), null,
            singletonList(struct(778, 765)), asList(struct(7, 87), struct(8, 96)),
            asList(struct(9, 56), struct(10, 532), struct(11, 456)), null, emptyList())
        .build();
  }

  // When expected is null, it means we are not expecting some error to happen when merging the tables.
  private static void checkMergeTable(Table expected, List<TableSlice> tableSlices) {
    try {
      KudoSerializer serializer = new KudoSerializer(schemaOf(tableSlices.get(0).getBaseTable()));

      List<OpenByteArrayOutputStream> outputStreams = new ArrayList<>();
      for (TableSlice slice : tableSlices) {
        OpenByteArrayOutputStream bout = new OpenByteArrayOutputStream();
        serializer.writeToStreamWithMetrics(slice.getBaseTable(), bout, slice.getStartRow(), slice.getNumRows());
        bout.flush();
        outputStreams.add(bout);
      }

      Arms.withResource(new ArrayList<KudoTable>(tableSlices.size()), kudoTables -> {
        try {
          for (int i = 0; i < tableSlices.size(); i++) {
            ByteArrayInputStream bin = new ByteArrayInputStream(outputStreams.get(i).toByteArray());
            kudoTables.add(KudoTable.from(bin).get());
          }

          if (expected != null) {
            long rows = kudoTables.stream().mapToLong(t -> t.getHeader().getNumRows()).sum();
            assertEquals(expected.getRowCount(), rows);
          }

          try (Table merged = serializer.mergeToTable(kudoTables.toArray(new KudoTable[0]))) {
            if (expected != null){
              assertEquals(expected.getRowCount(), merged.getRowCount());

              AssertUtils.assertTablesAreEqual(expected, merged);
            }
          }
        } catch (Exception e) {
          throw new RuntimeException(e);
        }

        return null;
      });
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  private static Integer[] integers(Integer... values) {
    return values;
  }

  public static HostColumnVector.StructData struct(Object... values) {
    return new HostColumnVector.StructData(values);
  }

  public static List<HostColumnVector.StructData> structs(HostColumnVector.StructData... values) {
    return asList(values);
  }

  public static String[] strings(String... values) {
    return values;
  }

  public static Schema schemaOf(Table t) {
    Schema.Builder builder = Schema.builder();

    for (int i = 0; i < t.getNumberOfColumns(); i++) {
      ColumnVector cv = t.getColumn(i);
      addToSchema(cv, "col_" + i + "_", builder);
    }

    return builder.build();
  }

  private static void addToSchema(ColumnView cv, String namePrefix, Schema.Builder builder) {
    toSchemaInner(cv, 0, namePrefix, builder);
  }

  private static int toSchemaInner(ColumnView cv, int idx, String namePrefix,
                                   Schema.Builder builder) {
    String name = namePrefix + idx;

    Schema.Builder thisBuilder = builder.addColumn(cv.getType(), name);
    int lastIdx = idx;
    for (int i = 0; i < cv.getNumChildren(); i++) {
      lastIdx = toSchemaInner(cv.getChildColumnView(i), lastIdx + 1, namePrefix,
          thisBuilder);
    }

    return lastIdx;
  }

  private static class TableSlice {
    private final int startRow;
    private final int numRows;
    private final Table baseTable;

    private TableSlice(int startRow, int numRows, Table baseTable) {
      this.startRow = startRow;
      this.numRows = numRows;
      this.baseTable = baseTable;
    }

    public int getStartRow() {
      return startRow;
    }

    public int getNumRows() {
      return numRows;
    }

    public Table getBaseTable() {
      return baseTable;
    }
  }

  @Test
  public void testMergeWithDumpPath() {
    File tempFile = null;
    try(CloseableArray<Table> tables = CloseableArray.wrap(new Table[2])) {
      //Create a temporary file for dumping
      tempFile = File.createTempFile("kudo_dump_test", ".bin");
      tempFile.deleteOnExit();

      String dumpPath = tempFile.getAbsolutePath();


      Table table1 = new Table.TestBuilder()
          .column(1, 2, 3, 4)
          .column("a", "b", "c", "d")
          .build();
      tables.set(0, table1);
      
      Table table2 = new Table.TestBuilder()
          .column(5, 6, 7, 8)
          .column("e", "f", "g", "h")
          .build();
      tables.set(1, table2);

      // Create KudoSerializer with table1's schema
      KudoSerializer serializer = new KudoSerializer(schemaOf(table1));
      
      // Serialize both tables
      ByteArrayOutputStream bout = new ByteArrayOutputStream();
      serializer.writeToStreamWithMetrics(table1, bout, 0, (int)table1.getRowCount());
      
      // Serialize table2 using same serializer - this will create an inconsistent state
      serializer.writeToStreamWithMetrics(table2, bout, 0, (int)table2.getRowCount());
      bout.flush();
      
      ByteArrayInputStream bin = new ByteArrayInputStream(bout.toByteArray());

      try(CloseableArray<KudoTable> kudoTables  = CloseableArray.wrap(new KudoTable[2])) {
        // Read the KudoTables from the stream
        kudoTables.set(0, KudoTable.from(bin).get());
        kudoTables.set(1, KudoTable.from(bin).get());

        // merge the two tables and dump the result to the temp file
        Supplier<OutputStream> outputStreamSupplier = () -> {
          try {
            return new FileOutputStream(dumpPath);
          } catch (Exception e) {
            throw new RuntimeException(e);
          }
        };
        MergeOptions options = new MergeOptions(DumpOption.Always, outputStreamSupplier, dumpPath);
        serializer.mergeOnHost(kudoTables.getArray(), options).close();

        // Verify dump file exists and has content
        assertTrue(tempFile.exists(), "Dump file should exist");
        assertTrue(tempFile.length() > 0, "Dump file should not be empty");

        // Basic check that file contains schema info
        byte[] fileContent = java.nio.file.Files.readAllBytes(tempFile.toPath());
        String contentStart = new String(fileContent, 0, Math.min(100, fileContent.length));
        assertTrue(contentStart.contains("col_0_0") || contentStart.contains("Schema"),
            "Dump file should contain schema information");  // Intentionally left blank to ensure proper closing of KudoTables
      }

    } catch (Exception e) {
      fail("Test failed with exception: " + e.getMessage());
    } finally {
      // Cleanup
      if (tempFile != null && tempFile.exists()) {
        tempFile.delete();
      }
    }
  }

  // This test ensures that we can serialize and deserialize tables whose total size
  // exceeds Integer.MAX_VALUE
  @Test
  public void testSerializeAndDeserializeLargeTable() {
    try(Table t1 = buildLargeTestTable();
        Table t2 = buildLargeTestTable();
        Table expected = Table.concatenate(t1, t2)) {
          assertTrue(expected.getDeviceMemorySize() > Integer.MAX_VALUE,
              "Expected table size should exceed Integer.MAX_VALUE");
          final int sliceSize = 50000000;

          int rowCount1 = Math.toIntExact(t1.getRowCount());
          int rowCount2 = Math.toIntExact(t2.getRowCount());

          List<TableSlice> tableSlices = new ArrayList<>();
          for (int startRow = 0; startRow < rowCount1; startRow += sliceSize) {
            tableSlices.add(new TableSlice(startRow, Math.min(sliceSize, rowCount1 - startRow), t1));
          }

          for (int startRow = 0; startRow < rowCount2; startRow += sliceSize) {
            tableSlices.add(new TableSlice(startRow, Math.min(sliceSize, rowCount2 - startRow), t2));
          }

          checkMergeTable(expected, tableSlices);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  @Test
  public void testRowCountAndNullCountOverflow() {
    // Create three tables where both total row count AND null count exceed Integer.MAX_VALUE
    // Each table will have significant null counts that sum to > Integer.MAX_VALUE

    final int nonNullCount1 = 5000;
    final int nullCount1 = Integer.MAX_VALUE / 3 + 50000000;

    final int nonNullCount2 = 3000; //
    final int nullCount2 = Integer.MAX_VALUE / 3 - 50000000;

    final int nonNullCount3 = 4000;
    final int nullCount3 = Integer.MAX_VALUE / 3 + 10000000;


    try(Table t1 = buildSingleLargeByteTable(nonNullCount1, nullCount1);
        Table t2 = buildSingleLargeByteTable(nonNullCount2, nullCount2);
        Table t3 = buildSingleLargeByteTable(nonNullCount3, nullCount3)) {


        final int sliceSize = 50000000;

        int rowCount1 = Math.toIntExact(t1.getRowCount());
        int rowCount2 = Math.toIntExact(t2.getRowCount());
        int rowCount3 = Math.toIntExact(t3.getRowCount());

        List<TableSlice> tableSlices = new ArrayList<>();

        // Add slices from first table
        for (int startRow = 0; startRow < rowCount1; startRow += sliceSize) {
          tableSlices.add(new TableSlice(startRow, Math.min(sliceSize, rowCount1 - startRow), t1));
        }

        // Add slices from second table
        for (int startRow = 0; startRow < rowCount2; startRow += sliceSize) {
          tableSlices.add(new TableSlice(startRow, Math.min(sliceSize, rowCount2 - startRow), t2));
        }

        // Add slices from third table
        for (int startRow = 0; startRow < rowCount3; startRow += sliceSize) {
          tableSlices.add(new TableSlice(startRow, Math.min(sliceSize, rowCount3 - startRow), t3));
        }

        assertThrows(RuntimeException.class, () -> {
          try {
            checkMergeTable(null, tableSlices);
          } catch (RuntimeException e) {
            throw e.getCause().getCause();
          }
        });
    }
  }

  @Test
  public void testOffsetOverflow() {
    // Create tables with string columns where total string data size exceeds Integer.MAX_VALUE
    // This should trigger an offset overflow and throw IllegalStateException

    final int stringSize = 100000;
    final int rowsPerTable = 15000;

    try (Table t1 = buildLargeStringTable(stringSize, rowsPerTable);
         Table t2 = buildLargeStringTable(stringSize, rowsPerTable)) {

        final int sliceSize = 50000;

        int rowCount1 = Math.toIntExact(t1.getRowCount());
        int rowCount2 = Math.toIntExact(t2.getRowCount());

        List<TableSlice> tableSlices = new ArrayList<>();

        // Add slices from first table
        for (int startRow = 0; startRow < rowCount1; startRow += sliceSize) {
          tableSlices.add(new TableSlice(startRow, Math.min(sliceSize, rowCount1 - startRow), t1));
        }

        // Add slices from second table
        for (int startRow = 0; startRow < rowCount2; startRow += sliceSize) {
          tableSlices.add(new TableSlice(startRow, Math.min(sliceSize, rowCount2 - startRow), t2));
        }

        assertThrows(IllegalArgumentException.class, () -> {
          try {
            checkMergeTable(null, tableSlices);
          } catch (RuntimeException e) {
            throw e.getCause().getCause();
          }
        });
      }

  }

  @Test
  public void testLargeOffsetBuffer() {
    // This test ensures proper handling of large offset buffers where:
    // 1. Offset buffer length > Integer.MAX_VALUE
    //    For LIST columns, offsets are 4 bytes each: (rowCount+1) * 4 bytes
    //    Need rowCount > 536,870,911 to exceed Integer.MAX_VALUE
    // 2. Multiple table slices for each original table (to test slice merging)
    //
    // Strategy: Create 2 tables with LIST<BYTE> column
    // - Each table: 269M rows (total 538M rows)
    // - LIST<BYTE> column: 1 byte per list element
    //   * Per table: last offset = 269M * 1 = 269M < Integer.MAX_VALUE ✓ (valid column)
    //   * Concatenated: offset buffer = (538M + 1) * 4 = 2.152GB > Integer.MAX_VALUE ✓
    //
    // Memory usage (concatenated table):
    // - List data: 538M * 1 byte = 538MB
    // - List offsets: (538M + 1) * 4 = 2.152GB
    // - Total: ~2.7GB < 3.2GB ✓

    final int rowsPerTable = 269_000_000;
    final int listSize = 1; // number of bytes per list

    try (Table t1 = buildTableWithByteList(listSize, rowsPerTable);
         Table t2 = buildTableWithByteList(listSize, rowsPerTable)) {

      // Slice each table into multiple slices (requirement 2)
      final int sliceSize = 50_000_000; // 50M rows per slice -> 6 slices per table
      List<TableSlice> tableSlices = new ArrayList<>();

      // Slice first table into multiple parts
      int rowCount1 = Math.toIntExact(t1.getRowCount());
      for (int startRow = 0; startRow < rowCount1; startRow += sliceSize) {
        tableSlices.add(new TableSlice(startRow, Math.min(sliceSize, rowCount1 - startRow), t1));
      }

      // Slice second table into multiple parts
      int rowCount2 = Math.toIntExact(t2.getRowCount());
      for (int startRow = 0; startRow < rowCount2; startRow += sliceSize) {
        tableSlices.add(new TableSlice(startRow, Math.min(sliceSize, rowCount2 - startRow), t2));
      }

      // Create expected result
      try (Table expected = Table.concatenate(t1, t2)) {
        // Verify requirements are met:
        long totalRows = expected.getRowCount();
        long offsetBufferSize = (totalRows + 1) * 4L; // INT32 offsets for LIST column

        assertTrue(offsetBufferSize > Integer.MAX_VALUE,
            "Offset buffer should exceed Integer.MAX_VALUE: " + offsetBufferSize);
        assertTrue(tableSlices.size() > 2,
            "Should have multiple slices per table: " + tableSlices.size());

        checkMergeTable(expected, tableSlices);
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  @Test
  public void testLargeDataBuffer() {
    // This test ensures proper handling of large data buffers where:
    // 1. Data buffer length > Integer.MAX_VALUE
    //    For LONG columns, data is 8 bytes per value: rowCount * 8 bytes
    //    Need rowCount > 268,435,455 to exceed Integer.MAX_VALUE
    // 2. Multiple table slices for each original table (to test slice merging)
    //
    // Strategy: Create 2 tables with LONG column
    // - Each table: 135M rows (total 270M rows)
    // - LONG column: 8 bytes per value
    //   * Per table: data = 135M * 8 = 1.08GB < Integer.MAX_VALUE ✓ (valid column)
    //   * Concatenated: data buffer = 270M * 8 = 2.16GB > Integer.MAX_VALUE ✓
    //
    // Memory usage (concatenated table):
    // - Long data: 270M * 8 = 2.16GB < 3.2GB ✓

    final int rowsPerTable = 135_000_000;

    try (Table t1 = buildTableWithLong(rowsPerTable);
         Table t2 = buildTableWithLong(rowsPerTable)) {

      // Slice each table into multiple slices (requirement 2)
      final int sliceSize = 25_000_000; // 25M rows per slice -> 6 slices per table
      List<TableSlice> tableSlices = new ArrayList<>();

      // Slice first table into multiple parts
      int rowCount1 = Math.toIntExact(t1.getRowCount());
      for (int startRow = 0; startRow < rowCount1; startRow += sliceSize) {
        tableSlices.add(new TableSlice(startRow, Math.min(sliceSize, rowCount1 - startRow), t1));
      }

      // Slice second table into multiple parts
      int rowCount2 = Math.toIntExact(t2.getRowCount());
      for (int startRow = 0; startRow < rowCount2; startRow += sliceSize) {
        tableSlices.add(new TableSlice(startRow, Math.min(sliceSize, rowCount2 - startRow), t2));
      }

      // Create expected result
      try (Table expected = Table.concatenate(t1, t2)) {
        // Verify requirements are met:
        long totalRows = expected.getRowCount();
        long dataBufferSize = totalRows * 8L; // LONG data

        assertTrue(dataBufferSize > Integer.MAX_VALUE,
            "Data buffer should exceed Integer.MAX_VALUE: " + dataBufferSize);
        assertTrue(tableSlices.size() > 2,
            "Should have multiple slices per table: " + tableSlices.size());

        checkMergeTable(expected, tableSlices);
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }


  static Table buildLargeTestTable() {
    List<ColumnVector> allCols = new ArrayList<>();
    List<ColumnVector> tableCols = new ArrayList<>();

    final int nonNullCount = Integer.MAX_VALUE / 128 - 21; // to avoid OOM
    final int nullCount = 11; // to add nulls
    final int totalCount = nonNullCount + nullCount;

    try {
      ((Runnable) () -> {
        try (Scalar v1 = Scalar.fromInt(100);
             ColumnVector cv1 = ColumnVector.fromScalar(v1, nonNullCount);
             ColumnVector cv2 = ColumnVector.fromBoxedInts(new Integer[nullCount])) { // nulls
          ColumnVector cv = ColumnVector.concatenate(cv1, cv2);
          tableCols.add(cv);
          allCols.add(cv);
        }
      }).run();

      ((Runnable) () -> {
        try (Scalar v2 = Scalar.fromLong(100L);
             ColumnVector cv1 = ColumnVector.fromScalar(v2, nonNullCount);
             ColumnVector cv2 = ColumnVector.fromBoxedLongs(new Long[nullCount])) { // nulls
          ColumnVector cv = ColumnVector.concatenate(cv1, cv2);
          tableCols.add(cv);
          allCols.add(cv);
        }
      }).run();

      ((Runnable) () -> {
        try (Scalar v2 = Scalar.fromDouble(100.0);
             ColumnVector cv1 = ColumnVector.fromScalar(v2, nonNullCount);
             ColumnVector cv2 = ColumnVector.fromBoxedDoubles(new Double[nullCount])) { // nulls
          ColumnVector cv = ColumnVector.concatenate(cv1, cv2);
          tableCols.add(cv);
          allCols.add(cv);
        }
      }).run();

      ((Runnable) () -> {
        try (Scalar v3 = Scalar.fromString("test_string");
             ColumnVector cv1 = ColumnVector.fromScalar(v3, nonNullCount);
             ColumnVector cv2 = ColumnVector.fromStrings(new String[nullCount])) { // nulls
          ColumnVector cv = ColumnVector.concatenate(cv1, cv2);
          tableCols.add(cv);
          allCols.add(cv);
        }
      }).run();

      // List<List<Integer>>
      ((Runnable) () -> {
        ColumnVector integerList;
        try (ColumnVector cv1 = ColumnVector.fromBoxedInts(new Integer[4 * nonNullCount]);
             ColumnVector cv2 = ColumnVector.fromBoxedInts(new Integer[4 * nullCount])) { // nulls
          integerList = ColumnVector.concatenate(cv1, cv2);
          allCols.add(integerList);
        }

        ColumnVector list1;
        try (Scalar zero = Scalar.fromInt(0);
             Scalar two = Scalar.fromInt(2);
            ColumnVector offsets = ColumnVector.sequence(zero, two, totalCount * 2 + 1)) { // nulls
          list1 = integerList.makeListFromOffsets(totalCount * 2, offsets);
          allCols.add(list1);
        }

        ColumnVector list;
        try (Scalar zero = Scalar.fromInt(0);
             Scalar two = Scalar.fromInt(2);
             ColumnVector offsets = ColumnVector.sequence(zero, two, totalCount + 1)) { // nulls
          list = list1.makeListFromOffsets(totalCount, offsets);
          tableCols.add(list);
          allCols.add(list);
        }
      }).run();

      // Struct
      ((Runnable) () -> {
        // Create int32 child column
        ColumnVector int32Child;
        try (Scalar v1 = Scalar.fromInt(42);
             ColumnVector cv1 = ColumnVector.fromScalar(v1, nonNullCount);
             ColumnVector cv2 = ColumnVector.fromBoxedInts(new Integer[nullCount])) { // nulls
          int32Child = ColumnVector.concatenate(cv1, cv2);
          allCols.add(int32Child);
        }

        // Create float32 child column
        ColumnVector float32Child;
        try (Scalar v2 = Scalar.fromFloat(3.14f);
             ColumnVector cv1 = ColumnVector.fromScalar(v2, nonNullCount);
             ColumnVector cv2 = ColumnVector.fromBoxedFloats(new Float[nullCount])) { // nulls
          float32Child = ColumnVector.concatenate(cv1, cv2);
          allCols.add(float32Child);
        }

        // Create struct column from child columns
        ColumnVector structColumn = ColumnVector.makeStruct(int32Child, float32Child);
        tableCols.add(structColumn);
        allCols.add(structColumn);
      }).run();

      return new Table(tableCols.toArray(new ColumnVector[0]));
    } finally {
      for (ColumnVector cv : allCols) {
        cv.close();
      }
    }
  }

  static Table buildSingleLargeByteTable(int nonNullCount, int nullCount) {
    List<ColumnVector> allCols = new ArrayList<>();
    List<ColumnVector> tableCols = new ArrayList<>();


    try {
      // Create a single double column
      ColumnVector doubleColumn;
      try (Scalar v1 = Scalar.fromByte((byte) 123);
           ColumnVector cv1 = ColumnVector.fromScalar(v1, nonNullCount);
           ColumnVector cv2 = ColumnVector.fromBoxedBytes(new Byte[nullCount])) { // nulls
        doubleColumn = ColumnVector.concatenate(cv1, cv2);
        tableCols.add(doubleColumn);
        allCols.add(doubleColumn);
      }

      return new Table(tableCols.toArray(new ColumnVector[0]));
    } finally {
      for (ColumnVector cv : allCols) {
        cv.close();
      }
    }
  }

  static Table buildLargeStringTable(int stringSize, int rowCount) {
    List<ColumnVector> allCols = new ArrayList<>();
    List<ColumnVector> tableCols = new ArrayList<>();

    try {
      // Create a large string by repeating characters
      StringBuilder sb = new StringBuilder(stringSize);
      for (int i = 0; i < stringSize; i++) {
        sb.append((char)('A' + (i % 26))); // Cycle through A-Z
      }
      String largeString = sb.toString();
      
      // Create string array with the large string repeated
      String[] stringArray = new String[rowCount];
      Arrays.fill(stringArray, largeString);
      
      // Create the string column
      ColumnVector stringColumn = ColumnVector.fromStrings(stringArray);
      tableCols.add(stringColumn);
      allCols.add(stringColumn);

      return new Table(tableCols.toArray(new ColumnVector[0]));
    } finally {
      for (ColumnVector cv : allCols) {
        cv.close();
      }
    }
  }

  static Table buildTableWithByteList(int listSize, int rowCount) {
    List<ColumnVector> allCols = new ArrayList<>();
    List<ColumnVector> tableCols = new ArrayList<>();

    try {
      // Create child data (BYTE column) - total size = rowCount * listSize
      ColumnVector childData;
      try (Scalar byteScalar = Scalar.fromByte((byte) 42)) {
        childData = ColumnVector.fromScalar(byteScalar, rowCount * listSize);
        allCols.add(childData);
      }

      // Create offsets for LIST column - each list has 'listSize' elements
      ColumnVector offsets;
      try (Scalar zero = Scalar.fromInt(0);
           Scalar step = Scalar.fromInt(listSize)) {
        offsets = ColumnVector.sequence(zero, step, rowCount + 1);
        allCols.add(offsets);
      }

      // Create LIST column from child data and offsets
      ColumnVector listColumn = childData.makeListFromOffsets(rowCount, offsets);
      tableCols.add(listColumn);
      allCols.add(listColumn);

      return new Table(tableCols.toArray(new ColumnVector[0]));
    } finally {
      for (ColumnVector cv : allCols) {
        cv.close();
      }
    }
  }

  static Table buildTableWithLong(int rowCount) {
    List<ColumnVector> allCols = new ArrayList<>();
    List<ColumnVector> tableCols = new ArrayList<>();

    try {
      // Create a LONG column
      ColumnVector longColumn;
      try (Scalar longScalar = Scalar.fromLong(123456789L)) {
        longColumn = ColumnVector.fromScalar(longScalar, rowCount);
        tableCols.add(longColumn);
        allCols.add(longColumn);
      }

      return new Table(tableCols.toArray(new ColumnVector[0]));
    } finally {
      for (ColumnVector cv : allCols) {
        cv.close();
      }
    }
  }
}
