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
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

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

  @Test
  void deviceReadReadsTheRange() throws IOException {
    Path file = writeFile(1000);
    try (NativeFileSource source = NativeFileSource.open(file.toString());
         DeviceMemoryBuffer dst = DeviceMemoryBuffer.allocate(300);
         HostMemoryBuffer host = HostMemoryBuffer.allocate(200)) {
      assertEquals(1000, source.size());
      assertEquals(200, source.deviceRead(100, dst, 50, 200, Cuda.DEFAULT_STREAM));
      try (DeviceMemoryBuffer read = dst.slice(50, 200)) {
        host.copyFromDeviceBuffer(read);
      }
      for (int i = 0; i < 200; i++) {
        assertEquals((byte) (100 + i), host.getByte(i), "byte " + i);
      }
    }
  }

  @Test
  void deviceReadStopsAtTheEndOfTheFile() throws IOException {
    Path file = writeFile(100);
    try (NativeFileSource source = NativeFileSource.open(file.toString());
         DeviceMemoryBuffer dst = DeviceMemoryBuffer.allocate(100)) {
      assertEquals(40, source.deviceRead(60, dst, 0, 100, Cuda.DEFAULT_STREAM));
    }
  }

  @Test
  void deviceReadRejectsARangeOutsideTheBuffer() throws IOException {
    Path file = writeFile(100);
    try (NativeFileSource source = NativeFileSource.open(file.toString());
         DeviceMemoryBuffer dst = DeviceMemoryBuffer.allocate(10)) {
      assertThrows(IllegalArgumentException.class,
          () -> source.deviceRead(0, dst, 5, 10, Cuda.DEFAULT_STREAM));
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
  void closedSourceThrows() throws IOException {
    Path file = writeFile(100);
    NativeFileSource source = NativeFileSource.open(file.toString());
    source.close();
    source.close();
    assertThrows(IllegalStateException.class, source::size);
  }
}
