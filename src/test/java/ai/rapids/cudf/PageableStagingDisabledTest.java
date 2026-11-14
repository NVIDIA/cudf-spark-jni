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
package ai.rapids.cudf;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PageableStagingDisabledTest {
  private static final String PREFIX = "ai.rapids.cudf.pageable.staged.copy.";
  private static final long PINNED_BYTES = 1024 * 1024;
  private final Map<String, String> properties = new HashMap<>();

  @BeforeEach
  public void setup() {
    for (String suffix : new String[] {"enabled", "disabled", "h2d.disabled", "d2h.disabled",
        "registered.buffers", "async.d2h", "async.d2h.slots", "threads", "chunk.bytes"}) {
      String key = PREFIX + suffix;
      properties.put(key, System.getProperty(key));
      System.clearProperty(key);
    }
    System.setProperty(PREFIX + "threads", "1");
    System.setProperty(PREFIX + "chunk.bytes", "4096");
    System.setProperty(PREFIX + "async.d2h", "true");
    System.setProperty(PREFIX + "async.d2h.slots", "1");
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 64L * 1024 * 1024);
    PinnedMemoryPool.initialize(PINNED_BYTES);
  }

  @AfterEach
  public void teardown() {
    try {
      PageableMemoryPool.shutdown();
    } finally {
      HostMemoryBuffer.setDeviceToHostCopyHandler(null);
      try {
        PinnedMemoryPool.shutdown();
        Rmm.shutdown();
      } finally {
        properties.forEach((key, value) -> {
          if (value == null) {
            System.clearProperty(key);
          } else {
            System.setProperty(key, value);
          }
        });
      }
    }
  }

  @Test
  public void defaultInitializationLeavesPinnedPoolAvailable() throws Exception {
    checkPoolOnly(false);
  }

  @Test
  public void registeredBuffersAndAsyncOptionsDoNotOptIn() throws Exception {
    checkPoolOnly(true);
  }

  @Test
  public void explicitFalseDisablesStaging() throws Exception {
    System.setProperty(PREFIX + "enabled", "false");
    checkPoolOnly(true);
  }

  @Test
  public void globalDisableOverridesOptIn() throws Exception {
    System.setProperty(PREFIX + "enabled", "true");
    System.setProperty(PREFIX + "disabled", "true");
    checkPoolOnly(true);
  }

  @Test
  public void bothDirectionsDisabledNeedNoStagingResources() throws Exception {
    System.setProperty(PREFIX + "enabled", "true");
    System.setProperty(PREFIX + "h2d.disabled", "true");
    System.setProperty(PREFIX + "d2h.disabled", "true");
    checkPoolOnly(false);
  }

  @Test
  public void enabledThenDisabledInitializationDoesNotReuseResources() throws Exception {
    System.setProperty(PREFIX + "enabled", "true");
    System.setProperty(PREFIX + "registered.buffers", "true");
    PageableMemoryPool.initialize(8192, 1);
    assertTrue(PageableMemoryPool.isDeviceToHostStagingEnabled());
    assertTrue(PageableMemoryPool.isHostToDeviceStagingEnabled());
    assertNotNull(resource("stagedCopyExecutor"));
    assertNotNull(resource("asyncD2HPoller"));
    PageableMemoryPool.shutdown();
    System.clearProperty(PREFIX + "enabled");
    checkPoolOnly(true);
  }

  @Test
  public void h2dOnlyDoesNotStartD2HPoller() throws Exception {
    System.setProperty(PREFIX + "enabled", "true");
    System.setProperty(PREFIX + "d2h.disabled", "true");
    System.setProperty(PREFIX + "registered.buffers", "true");
    PageableMemoryPool.initialize(8192, 1);
    assertTrue(PageableMemoryPool.isInitialized());
    assertNotNull(resource("stagedCopyExecutor"));
    assertNull(resource("asyncD2HPoller"));
    assertNull(resource("asyncD2HSlots"));
    assertFalse(PageableMemoryPool.isDeviceToHostStagingEnabled());
    assertTrue(PageableMemoryPool.isHostToDeviceStagingEnabled());
  }

  private void checkPoolOnly(boolean registered) throws Exception {
    System.setProperty(PREFIX + "registered.buffers", Boolean.toString(registered));
    AtomicInteger fallbackCalls = new AtomicInteger();
    HostMemoryBuffer.setDeviceToHostCopyHandler((dst, dstOffset, src, srcOffset, length, stream) -> {
      fallbackCalls.incrementAndGet();
      return false;
    });
    PageableMemoryPool.initialize(8192, 1);
    assertTrue(PageableMemoryPool.isInitialized());
    assertFalse(PageableMemoryPool.isDeviceToHostStagingEnabled());
    assertFalse(PageableMemoryPool.isHostToDeviceStagingEnabled());
    for (String field : new String[] {"stagedCopyExecutor", "stagedCopyBuffers",
        "registeredStagedCopyBuffers", "asyncD2HPoller", "asyncD2HSlots",
        "freeAsyncD2HSlots", "pendingAsyncD2H"}) {
      assertNull(resource(field), field + " must not exist in pool-only mode");
    }
    // The entire pinned pool remains available: no hidden staging reservation.
    try (HostMemoryBuffer pinned = PinnedMemoryPool.tryAllocate(PINNED_BYTES)) {
      assertNotNull(pinned);
    }
    byte[] expected = {1, 2, 3, 4};
    try (HostMemoryBuffer input = HostMemoryBuffer.allocateRaw(4);
         HostMemoryBuffer output = PageableMemoryPool.tryAllocateForDeviceToHost(4);
         DeviceMemoryBuffer device = DeviceMemoryBuffer.allocate(4);
         Cuda.Stream stream = new Cuda.Stream(true)) {
      assertNotNull(output);
      input.setBytes(0, expected, 0, 4);
      device.copyFromHostBuffer(input);
      output.copyFromDeviceBufferAsync(device, stream);
      stream.sync();
      byte[] actual = new byte[4];
      output.getBytes(actual, 0, 0, 4);
      assertArrayEquals(expected, actual);
      assertEquals(1, fallbackCalls.get(), "disabled initialization replaced the handler");
      assertFalse(PageableMemoryPool.copyFromDeviceBuffer(output, 0, device, 0, 4, stream));
      assertThrows(IllegalStateException.class, () ->
          PageableMemoryPool.copyDeviceToPageableHost(output, 0, device, 0, 4, stream));
      assertThrows(IllegalStateException.class, () ->
          PageableMemoryPool.copyPageableHostToDevice(device, 0, output, 0, 4, stream));
    }
    PageableMemoryPool.shutdown();
    // Disabled teardown must not unregister a handler it did not install.
    try (HostMemoryBuffer host = HostMemoryBuffer.allocateRaw(4);
         DeviceMemoryBuffer device = DeviceMemoryBuffer.allocate(4)) {
      device.copyFromHostBuffer(host);
      host.copyFromDeviceBuffer(device);
      assertEquals(2, fallbackCalls.get());
    }
  }

  private static Object resource(String name) throws Exception {
    Field field = PageableMemoryPool.class.getDeclaredField(name);
    field.setAccessible(true);
    return field.get(null);
  }
}
