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

import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Tests reads with KvikIO in libcudf's default mode, ON, where every read goes through POSIX and
 * device reads are staged in host memory. NativeFileSourceGdsTest covers AUTO mode.
 */
public class NativeFileSourceTest {

  @TempDir
  Path tempDir;

  private Path writeFile(int length) throws IOException {
    byte[] data = new byte[length];
    for (int i = 0; i < length; i++) {
      data[i] = (byte) i;
    }
    return Files.write(tempDir.resolve("data.bin"), data);
  }

  // Three ranges read out of file order, the last one ending at the end of the 1000-byte file.
  private static final long[] FILE_OFFSETS = {100, 0, 900};
  private static final long[] LENGTHS = {200, 50, 100};
  private static final long[] DST_OFFSETS = {0, 300, 200};
  private static final int DST_LENGTH = 350;

  private static void assertRangesRead(HostMemoryBuffer dst) {
    for (int r = 0; r < LENGTHS.length; r++) {
      for (int i = 0; i < LENGTHS[r]; i++) {
        assertEquals((byte) (FILE_OFFSETS[r] + i), dst.getByte(DST_OFFSETS[r] + i),
            "range " + r + " byte " + i);
      }
    }
  }

  @Test
  void readToHostReadsEachRange() throws IOException {
    Path file = writeFile(1000);
    try (NativeFileSource source = NativeFileSource.open(file.toString());
         HostMemoryBuffer dst = HostMemoryBuffer.allocate(DST_LENGTH)) {
      assertEquals(1000, source.size());
      source.readToHost(FILE_OFFSETS, LENGTHS, dst, DST_OFFSETS);
      assertRangesRead(dst);
    }
  }

  // KvikIO is in its POSIX compatibility mode in this JVM, so this reads through host memory and
  // copies to the device; NativeFileSourceGdsTest covers reads through cuFile.
  @Test
  void readToDeviceReadsEachRange() throws IOException {
    Path file = writeFile(1000);
    try (NativeFileSource source = NativeFileSource.open(file.toString());
         DeviceMemoryBuffer dst = DeviceMemoryBuffer.allocate(DST_LENGTH);
         HostMemoryBuffer host = HostMemoryBuffer.allocate(DST_LENGTH)) {
      source.readToDevice(FILE_OFFSETS, LENGTHS, dst, DST_OFFSETS, Cuda.DEFAULT_STREAM);
      host.copyFromDeviceBuffer(dst);
      assertRangesRead(host);
    }
  }

  @Test
  void emptyRangesReadNothing() throws IOException {
    Path file = writeFile(100);
    try (NativeFileSource source = NativeFileSource.open(file.toString());
         HostMemoryBuffer dst = HostMemoryBuffer.allocate(10)) {
      dst.setMemory(0, 10, (byte) -1);
      source.readToHost(new long[0], new long[0], dst, new long[0]);
      source.readToHost(new long[] {100}, new long[] {0}, dst, new long[] {10});
      for (int i = 0; i < 10; i++) {
        assertEquals((byte) -1, dst.getByte(i), "byte " + i);
      }
    }
  }

  @Test
  void rangePastTheEndOfTheFileFailsTheBatchBeforeReading() throws IOException {
    Path file = writeFile(100);
    try (NativeFileSource source = NativeFileSource.open(file.toString());
         HostMemoryBuffer dst = HostMemoryBuffer.allocate(60)) {
      dst.setMemory(0, 60, (byte) -1);
      assertThrows(CudfException.class, () -> source.readToHost(
          new long[] {0, 60}, new long[] {10, 50}, dst, new long[] {0, 10}));
      for (int i = 0; i < 60; i++) {
        assertEquals((byte) -1, dst.getByte(i), "byte " + i);
      }
    }
  }

  /**
   * Open the file, then shrink it. The source keeps the size it read first, so a range past the
   * new end passes validation and then fails in the read itself.
   */
  private static NativeFileSource openThenTruncate(Path file, long newLength) throws IOException {
    NativeFileSource source = NativeFileSource.open(file.toString());
    try {
      long size = source.size();
      try (RandomAccessFile raf = new RandomAccessFile(file.toFile(), "rw")) {
        raf.setLength(newLength);
      }
      assertEquals(size, source.size());
      return source;
    } catch (Throwable t) {
      source.close();
      throw t;
    }
  }

  private static void assertBytesRead(HostMemoryBuffer dst, long dstOffset, int fileOffset,
      int length) {
    for (int i = 0; i < length; i++) {
      assertEquals((byte) (fileOffset + i), dst.getByte(dstOffset + i), "byte " + i);
    }
  }

  // Range 0 is inside the truncated file and range 1 is past its end.
  private static final long[] TRUNCATED_FILE_OFFSETS = {0, 60};
  private static final long[] TRUNCATED_LENGTHS = {10, 40};
  private static final long[] TRUNCATED_DST_OFFSETS = {0, 10};

  @Test
  void failedHostReadThrows() throws IOException {
    Path file = writeFile(100);
    try (NativeFileSource source = openThenTruncate(file, 50);
         HostMemoryBuffer dst = HostMemoryBuffer.allocate(50)) {
      assertThrows(CudfException.class, () -> source.readToHost(
          TRUNCATED_FILE_OFFSETS, TRUNCATED_LENGTHS, dst, TRUNCATED_DST_OFFSETS));
      assertBytesRead(dst, 0, 0, 10);
      source.readToHost(new long[] {20}, new long[] {10}, dst, new long[] {0});
      assertBytesRead(dst, 0, 20, 10);
    }
  }

  @Test
  void failedDeviceReadThrows() throws IOException {
    Path file = writeFile(100);
    try (NativeFileSource source = openThenTruncate(file, 50);
         DeviceMemoryBuffer dst = DeviceMemoryBuffer.allocate(50);
         HostMemoryBuffer host = HostMemoryBuffer.allocate(50)) {
      assertThrows(CudfException.class, () -> source.readToDevice(
          TRUNCATED_FILE_OFFSETS, TRUNCATED_LENGTHS, dst, TRUNCATED_DST_OFFSETS,
          Cuda.DEFAULT_STREAM));
      host.copyFromDeviceBuffer(dst);
      assertBytesRead(host, 0, 0, 10);
      source.readToDevice(new long[] {20}, new long[] {10}, dst, new long[] {0},
          Cuda.DEFAULT_STREAM);
      host.copyFromDeviceBuffer(dst);
      assertBytesRead(host, 0, 20, 10);
    }
  }

  @Test
  void negativeFileOffsetThrows() throws IOException {
    Path file = writeFile(100);
    try (NativeFileSource source = NativeFileSource.open(file.toString());
         HostMemoryBuffer dst = HostMemoryBuffer.allocate(10)) {
      assertThrows(CudfException.class,
          () -> source.readToHost(new long[] {-1}, new long[] {10}, dst, new long[] {0}));
    }
  }

  @Test
  void rangeOutsideTheBufferThrows() throws IOException {
    Path file = writeFile(100);
    try (NativeFileSource source = NativeFileSource.open(file.toString());
         DeviceMemoryBuffer dst = DeviceMemoryBuffer.allocate(10)) {
      assertThrows(IllegalArgumentException.class, () -> source.readToDevice(
          new long[] {0}, new long[] {10}, dst, new long[] {5}, Cuda.DEFAULT_STREAM));
    }
  }

  @Test
  void mismatchedRangeArraysThrow() throws IOException {
    Path file = writeFile(100);
    try (NativeFileSource source = NativeFileSource.open(file.toString());
         HostMemoryBuffer dst = HostMemoryBuffer.allocate(10)) {
      assertThrows(IllegalArgumentException.class,
          () -> source.readToHost(new long[] {0, 5}, new long[] {5}, dst, new long[] {0}));
    }
  }

  @Test
  void defaultCompatModeIsNotDirect() throws IOException {
    // libcudf reads the mode once per process, so an override in the environment can't be undone.
    String mode = System.getenv("KVIKIO_COMPAT_MODE");
    if (mode != null && !mode.equals("ON")) {
      return;
    }
    Path file = writeFile(100);
    try (NativeFileSource source = NativeFileSource.open(file.toString())) {
      assertFalse(source.isDeviceReadDirect(1 << 20));
    }
  }

  @Test
  void setUpKvikioRejectsNoThreads() {
    assertThrows(IllegalArgumentException.class,
        () -> NativeFileSource.setUpKvikio(NativeFileSource.CompatMode.ON, 0));
  }

  @Test
  void closedSourceThrows() throws IOException {
    Path file = writeFile(100);
    NativeFileSource source = NativeFileSource.open(file.toString());
    source.close();
    assertThrows(IllegalStateException.class, source::size);
    assertThrows(IllegalStateException.class, source::close);
  }
}
