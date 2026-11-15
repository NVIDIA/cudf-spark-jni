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
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * Reads with KvikIO in AUTO mode, which uses cuFile where it works and POSIX otherwise, so these
 * pass whether or not GDS is supported here. Runs in its own JVM, since the mode is fixed by the
 * first cuDF IO in the process.
 */
public class NativeFileSourceGdsTest {
  private static final int FILE_SIZE = 9 << 20;

  // Aligned and unaligned offsets and lengths, a range large enough to be split across KvikIO's
  // threads, and a range ending at the end of the file.
  private static final long[] FILE_OFFSETS = {(1 << 20) + 3, 0, 4097, FILE_SIZE - 777};
  private static final long[] LENGTHS = {5 << 20, 4096, 10000, 777};

  @TempDir
  static Path tempDir;

  private static byte[] data;
  private static Path file;

  @BeforeAll
  static void setUp() throws IOException {
    NativeFileSource.setUpKvikio(NativeFileSource.CompatMode.AUTO, 4);
    data = new byte[FILE_SIZE];
    new Random(42).nextBytes(data);
    file = Files.write(tempDir.resolve("data.bin"), data);
  }

  /** Back-to-back destination offsets starting at 1, so device addresses are unaligned. */
  private static long[] dstOffsets() {
    long[] dstOffsets = new long[LENGTHS.length];
    long offset = 1;
    for (int r = 0; r < LENGTHS.length; r++) {
      dstOffsets[r] = offset;
      offset += LENGTHS[r];
    }
    return dstOffsets;
  }

  private static long dstLength(long[] dstOffsets) {
    int last = LENGTHS.length - 1;
    return dstOffsets[last] + LENGTHS[last];
  }

  private static void assertRangesRead(HostMemoryBuffer dst, long[] dstOffsets) {
    for (int r = 0; r < LENGTHS.length; r++) {
      byte[] actual = new byte[(int) LENGTHS[r]];
      dst.getBytes(actual, 0, dstOffsets[r], LENGTHS[r]);
      int start = (int) FILE_OFFSETS[r];
      assertArrayEquals(Arrays.copyOfRange(data, start, start + actual.length), actual,
          "range " + r);
    }
  }

  private static void readToDeviceAndCheck() {
    long[] dstOffsets = dstOffsets();
    long dstLength = dstLength(dstOffsets);
    try (NativeFileSource source = NativeFileSource.open(file.toString());
         DeviceMemoryBuffer dst = DeviceMemoryBuffer.allocate(dstLength);
         HostMemoryBuffer host = HostMemoryBuffer.allocate(dstLength)) {
      source.readToDevice(FILE_OFFSETS, LENGTHS, dst, dstOffsets, Cuda.DEFAULT_STREAM);
      host.copyFromDeviceBuffer(dst);
      assertRangesRead(host, dstOffsets);
    }
  }

  @Test
  void readToDeviceReadsEachRange() {
    readToDeviceAndCheck();
  }

  @Test
  void readToHostReadsEachRange() {
    long[] dstOffsets = dstOffsets();
    try (NativeFileSource source = NativeFileSource.open(file.toString());
         HostMemoryBuffer dst = HostMemoryBuffer.allocate(dstLength(dstOffsets))) {
      source.readToHost(FILE_OFFSETS, LENGTHS, dst, dstOffsets);
      assertRangesRead(dst, dstOffsets);
    }
  }

  @Test
  void concurrentReadsToDeviceOnSeparateSources() throws Exception {
    ExecutorService pool = Executors.newFixedThreadPool(4);
    try {
      List<Future<?>> reads = new ArrayList<>();
      for (int i = 0; i < 4; i++) {
        reads.add(pool.submit(NativeFileSourceGdsTest::readToDeviceAndCheck));
      }
      for (Future<?> read : reads) {
        read.get();
      }
    } finally {
      pool.shutdown();
    }
  }
}
