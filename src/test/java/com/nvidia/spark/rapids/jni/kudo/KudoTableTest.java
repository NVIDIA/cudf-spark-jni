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

package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.DefaultHostMemoryAllocator;
import ai.rapids.cudf.HostMemoryAllocator;
import ai.rapids.cudf.HostMemoryBuffer;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.FilterInputStream;
import java.io.IOException;
import java.util.stream.Stream;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Isolated;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.*;

@Isolated("Changes the default host memory allocator")
class KudoTableTest {
  private HostMemoryAllocator originalAllocator;
  private HostMemoryBuffer allocated;

  @BeforeEach
  void trackAllocations() {
    originalAllocator = DefaultHostMemoryAllocator.get();
    DefaultHostMemoryAllocator.set(new DefaultHostMemoryAllocator() {
      @Override
      public HostMemoryBuffer allocate(long bytes, boolean preferPinned) {
        allocated = originalAllocator.allocate(bytes, preferPinned);
        return allocated;
      }
    });
  }

  @AfterEach
  void restoreAllocator() {
    DefaultHostMemoryAllocator.set(originalAllocator);
    if (allocated != null && allocated.getRefCount() > 0) {
      allocated.close();
    }
  }

  private byte[] serialize(KudoTableHeader header, byte... body) throws IOException {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    header.writeTo(new DataOutputStreamWriter(new DataOutputStream(bytes)));
    bytes.write(body);
    return bytes.toByteArray();
  }

  private KudoTableHeader bodyHeader() {
    return new KudoTableHeader(0, 1, 0, 0, 4, 1, new byte[]{0});
  }

  @Test
  void emptyStreamDoesNotAllocate() throws IOException {
    assertFalse(KudoTable.from(new ByteArrayInputStream(new byte[0])).isPresent());
    assertNull(allocated);
  }

  @Test
  void rowCountOnlyDoesNotAllocate() throws Exception {
    KudoTableHeader header = new KudoTableHeader(0, 5, 0, 0, 0, 0, new byte[0]);
    try (KudoTable table = KudoTable.from(new ByteArrayInputStream(serialize(header))).get()) {
      assertEquals(5, table.getHeader().getNumRows());
      assertNull(table.getBuffer());
      assertNull(allocated);
    }
  }

  @Test
  void successfulReadTransfersBufferToTable() throws Exception {
    byte[] body = {1, 2, 3, 4};
    byte[] input = serialize(bodyHeader(), body);
    try (KudoTable table = KudoTable.from(new ByteArrayInputStream(input)).get()) {
      assertSame(allocated, table.getBuffer());
      assertEquals(1, allocated.getRefCount());
      assertEquals(1, table.getHeader().getNumRows());
      for (int i = 0; i < body.length; i++) {
        assertEquals(body[i], allocated.getByte(i));
      }
    }
    assertEquals(0, allocated.getRefCount());
  }

  @Test
  void truncatedBodyClosesBufferAndThrowsEOFException() throws IOException {
    byte[] input = serialize(bodyHeader(), (byte) 1, (byte) 2);
    assertThrows(EOFException.class, () -> KudoTable.from(new ByteArrayInputStream(input)));
    assertNotNull(allocated);
    assertEquals(0, allocated.getRefCount());
  }

  static Stream<Throwable> readFailures() {
    return Stream.of(new IOException("read"), new AssertionError("read"),
        new OutOfMemoryError("read"));
  }

  @ParameterizedTest
  @MethodSource("readFailures")
  void bodyReadFailureClosesBufferAndPreservesFailure(Throwable failure) throws IOException {
    byte[] header = serialize(bodyHeader());
    FilterInputStream input = new FilterInputStream(new ByteArrayInputStream(header)) {
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
    assertSame(failure, assertThrows(failure.getClass(), () -> KudoTable.from(input)));
    assertNotNull(allocated);
    assertEquals(0, allocated.getRefCount());
  }
}
