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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PageableMemoryPoolTest {
  @AfterEach
  public void tearDown() {
    PageableMemoryPool.shutdown();
  }

  @Test
  public void allocationAndExhaustion() {
    assertFalse(PageableMemoryPool.isInitialized());
    assertNull(PageableMemoryPool.tryAllocate(1));
    PageableMemoryPool.initialize(1024 * 1024, 2);
    assertTrue(PageableMemoryPool.isInitialized());
    assertEquals(1024 * 1024, PageableMemoryPool.getTotalPoolSizeBytes());
    try (HostMemoryBuffer buffer = PageableMemoryPool.tryAllocate(1024)) {
      assertNotNull(buffer);
      assertNull(PageableMemoryPool.tryAllocate(2 * 1024 * 1024));
    }
  }

  @Test
  public void numaBindInitialization() {
    // Exercises the numaBind=true JNI path end to end; binding is best-effort, so no
    // multi-node host is required — pool construction and allocation must just succeed.
    PageableMemoryPool.initialize(1024 * 1024, 2, true);
    assertTrue(PageableMemoryPool.isInitialized());
    try (HostMemoryBuffer buffer = PageableMemoryPool.tryAllocate(1024)) {
      assertNotNull(buffer);
    }
  }

  @Test
  public void rejectsInvalidAllocationSize() {
    assertThrows(IllegalArgumentException.class, () -> PageableMemoryPool.tryAllocate(0));
    assertThrows(IllegalArgumentException.class, () -> PageableMemoryPool.tryAllocate(-1));
  }
}
