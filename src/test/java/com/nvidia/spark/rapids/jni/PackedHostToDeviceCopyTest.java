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

  // Two buffers adjacent in host memory, one from another allocation, an empty one, and one past
  // a gap, so that the copy covers both merged and separate runs.
  private static void checkMixedLayout(boolean onSideStream) {
    try (HostMemoryBuffer whole = filled(64, 0);
         HostMemoryBuffer other = filled(7, 100);
         HostMemoryBuffer first = whole.slice(0, 10);
         HostMemoryBuffer adjacent = whole.slice(10, 15);
         HostMemoryBuffer empty = whole.slice(30, 0);
         HostMemoryBuffer pastGap = whole.slice(40, 10)) {
      HostMemoryBuffer[] hostBuffers = {first, adjacent, other, empty, pastGap};
      try (PackedHostToDeviceCopy copy =
               PackedHostToDeviceCopy.copyAsync(hostBuffers, onSideStream)) {
        copy.waitOnEvent();
        Cuda.deviceSynchronize();
        assertCopied(hostBuffers, copy.getBuffers());
      }
    }
  }

  @Test
  void testMixedLayoutOnDefaultStream() {
    checkMixedLayout(false);
  }

  @Test
  void testMixedLayoutOnSideStream() {
    checkMixedLayout(true);
  }

  @Test
  void testNoBuffers() {
    try (PackedHostToDeviceCopy copy =
             PackedHostToDeviceCopy.copyAsync(new HostMemoryBuffer[0], true)) {
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
      try (PackedHostToDeviceCopy copy = PackedHostToDeviceCopy.copyAsync(hostBuffers, true)) {
        copy.waitOnEvent();
        assertCopied(hostBuffers, copy.getBuffers());
      }
    }
  }

  @Test
  void testCloseTwice() {
    try (HostMemoryBuffer data = filled(16, 7)) {
      PackedHostToDeviceCopy copy =
          PackedHostToDeviceCopy.copyAsync(new HostMemoryBuffer[]{data}, true);
      copy.close();
      copy.close();
    }
  }
}
