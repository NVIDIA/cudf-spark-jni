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

import static org.junit.jupiter.api.Assertions.assertEquals;

public class PackedHostToDeviceCopyTest {

  private static HostMemoryBuffer filled(int length, int seed) {
    HostMemoryBuffer buffer = HostMemoryBuffer.allocate(length);
    for (int i = 0; i < length; i++) {
      buffer.setByte(i, (byte) (seed + i));
    }
    return buffer;
  }

  private static void assertCopied(HostMemoryBuffer[] expected, DeviceMemoryBuffer[] actual) {
    assertEquals(expected.length, actual.length);
    for (int i = 0; i < expected.length; i++) {
      assertEquals(expected[i].getLength(), actual[i].getLength(), "length of buffer " + i);
      if (actual[i].getLength() == 0) {
        continue;
      }
      try (HostMemoryBuffer back = HostMemoryBuffer.allocate(actual[i].getLength())) {
        back.copyFromDeviceBuffer(actual[i]);
        for (long j = 0; j < back.getLength(); j++) {
          assertEquals(expected[i].getByte(j), back.getByte(j), "buffer " + i + " byte " + j);
        }
      }
    }
  }

  private static final int UNIT = 1 << 18;

  private static void assertPackLengths(long[] expected, DeviceMemoryBuffer[] packs) {
    assertEquals(expected.length, packs.length, "pack count");
    for (int p = 0; p < expected.length; p++) {
      assertEquals(expected[p], packs[p].getLength(), "length of pack " + p);
    }
  }

  // Buffers of 10 and 15 units adjacent in host memory, 7 units from another allocation, an empty
  // one, and 10 units past a gap, so that the copy covers both merged and separate runs. Only
  // waitOnEvent() orders the read-back after a side-stream copy; the buffers are large so a missing
  // wait likely shows. The unit is a multiple of 8, so the packs carry no padding.
  private static void checkMixedLayout(Cuda.Stream stream, long maxPackBytes,
      long... expectedPackUnits) {
    long[] expectedPackLengths = new long[expectedPackUnits.length];
    for (int p = 0; p < expectedPackUnits.length; p++) {
      expectedPackLengths[p] = expectedPackUnits[p] * UNIT;
    }
    try (HostMemoryBuffer whole = filled(64 * UNIT, 0);
         HostMemoryBuffer other = filled(7 * UNIT, 100);
         HostMemoryBuffer first = whole.slice(0, 10 * UNIT);
         HostMemoryBuffer adjacent = whole.slice(10 * UNIT, 15 * UNIT);
         HostMemoryBuffer empty = whole.slice(30 * UNIT, 0);
         HostMemoryBuffer pastGap = whole.slice(40 * UNIT, 10 * UNIT)) {
      HostMemoryBuffer[] hostBuffers = {first, adjacent, other, empty, pastGap};
      try (PackedHostToDeviceCopy copy =
               PackedHostToDeviceCopy.copyAsync(hostBuffers, stream, maxPackBytes)) {
        copy.waitOnEvent();
        assertPackLengths(expectedPackLengths, copy.getPacks());
        assertCopied(hostBuffers, copy.getBuffers());
      }
    }
  }

  @Test
  void testMixedLayoutOnDefaultStream() {
    checkMixedLayout(Cuda.DEFAULT_STREAM, Long.MAX_VALUE, 42);
  }

  @Test
  void testMixedLayoutOnSideStream() {
    try (Cuda.Stream stream = new Cuda.Stream(true)) {
      checkMixedLayout(stream, Long.MAX_VALUE, 42);
    }
  }

  // A cap of 12 units puts the host-adjacent 10- and 15-unit buffers in separate packs, so they
  // must not be merged into one copy, and the 15-unit buffer exceeds the cap on its own. The empty
  // buffer joins the 7-unit buffer's pack.
  @Test
  void testMixedLayoutSplitIntoPacks() {
    try (Cuda.Stream stream = new Cuda.Stream(true)) {
      checkMixedLayout(stream, 12L * UNIT, 10, 15, 7, 10);
    }
  }

  // A cap of 25 units is filled exactly by the 10- and 15-unit buffers, which share a pack.
  @Test
  void testMixedLayoutPackFilledToCap() {
    try (Cuda.Stream stream = new Cuda.Stream(true)) {
      checkMixedLayout(stream, 25L * UNIT, 25, 17);
    }
  }

  @Test
  void testMixedLayoutOneBufferPerPack() {
    checkMixedLayout(Cuda.DEFAULT_STREAM, 0, 10, 15, 7, 10);
  }

  // Each pack is padded on its own. The empty buffer after the 3-byte one, which already exceeds
  // the cap, joins its pack rather than starting one. The 3- and 5-byte buffers are adjacent in
  // host memory but land in different packs.
  @Test
  void testPacksPaddedSeparately() {
    try (HostMemoryBuffer whole = filled(16, 0);
         HostMemoryBuffer three = whole.slice(0, 3);
         HostMemoryBuffer empty = whole.slice(3, 0);
         HostMemoryBuffer five = whole.slice(3, 5)) {
      HostMemoryBuffer[] hostBuffers = {three, empty, five};
      try (PackedHostToDeviceCopy copy =
               PackedHostToDeviceCopy.copyAsync(hostBuffers, Cuda.DEFAULT_STREAM, 2)) {
        assertPackLengths(new long[]{8, 8}, copy.getPacks());
        assertCopied(hostBuffers, copy.getBuffers());
      }
    }
  }

  @Test
  void testNoBuffers() {
    try (PackedHostToDeviceCopy copy = PackedHostToDeviceCopy.copyAsync(
             new HostMemoryBuffer[0], Cuda.DEFAULT_STREAM, Long.MAX_VALUE)) {
      copy.waitOnEvent();
      assertEquals(0, copy.getBuffers().length);
    }
  }

  @Test
  void testOnlyEmptyBuffers() {
    try (HostMemoryBuffer whole = filled(8, 0);
         HostMemoryBuffer a = whole.slice(0, 0);
         HostMemoryBuffer b = whole.slice(4, 0)) {
      HostMemoryBuffer[] hostBuffers = {a, b};
      try (PackedHostToDeviceCopy copy = PackedHostToDeviceCopy.copyAsync(
               hostBuffers, Cuda.DEFAULT_STREAM, Long.MAX_VALUE)) {
        copy.waitOnEvent();
        assertCopied(hostBuffers, copy.getBuffers());
      }
    }
  }

  @Test
  void testCloseTwice() {
    try (Cuda.Stream stream = new Cuda.Stream(true);
         HostMemoryBuffer data = filled(16, 7)) {
      PackedHostToDeviceCopy copy =
          PackedHostToDeviceCopy.copyAsync(new HostMemoryBuffer[]{data}, stream, Long.MAX_VALUE);
      copy.close();
      copy.close();
    }
  }
}
