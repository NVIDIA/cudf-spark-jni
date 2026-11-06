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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantReadWriteLock;

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
  public void numaBindExplicitDeviceInitialization() {
    PageableMemoryPool.initialize(1024 * 1024, 2, true, Cuda.getDevice());
    assertTrue(PageableMemoryPool.isInitialized());
    try (HostMemoryBuffer buffer = PageableMemoryPool.tryAllocate(1024)) {
      assertNotNull(buffer);
    }
  }

  @Test
  public void allocationStrategies() {
    for (PageableMemoryPool.AllocationStrategy strategy :
        PageableMemoryPool.AllocationStrategy.values()) {
      PageableMemoryPool.initialize(1024 * 1024, 2, false, -1, strategy, 4, false);
      try (HostMemoryBuffer buffer = PageableMemoryPool.tryAllocate(128 * 1024)) {
        assertNotNull(buffer, strategy.toString());
      }
      PageableMemoryPool.shutdown();
    }
  }

  @Test
  public void arenaCapacityAndInvalidOptions() {
    assertThrows(IllegalArgumentException.class,
        () -> PageableMemoryPool.initialize(1024 * 1024, 2, false, -1,
            PageableMemoryPool.AllocationStrategy.ARENA, 0, false));
    PageableMemoryPool.initialize(1024 * 1024, 2, false, -1,
        PageableMemoryPool.AllocationStrategy.ARENA, 4, false);
    assertTrue(PageableMemoryPool.isInitialized());
    assertNull(PageableMemoryPool.tryAllocate(512 * 1024));
  }

  @Test
  public void rejectsInvalidAllocationSize() {
    assertThrows(IllegalArgumentException.class, () -> PageableMemoryPool.tryAllocate(0));
    assertThrows(IllegalArgumentException.class, () -> PageableMemoryPool.tryAllocate(-1));
  }

  @Test
  public void freeAfterShutdownThrowsIllegalState() {
    PageableMemoryPool.initialize(1024 * 1024, 1);
    HostMemoryBuffer buffer = PageableMemoryPool.tryAllocate(1024);
    assertNotNull(buffer);
    PageableMemoryPool.shutdown();
    // Deterministic regression for the free-after-shutdown IllegalStateException
    // branch: the cleaner's free path must fail loudly, not silently corrupt.
    assertThrows(IllegalStateException.class, buffer::close);
  }

  @Test
  public void concurrentAllocationThroughLifecycleGate() throws Exception {
    PageableMemoryPool.initialize(4 * 1024 * 1024, 2);
    final int workerCount = 8;
    final CountDownLatch start = new CountDownLatch(1);
    final AtomicBoolean stop = new AtomicBoolean(false);
    final AtomicInteger successfulAllocs = new AtomicInteger(0);
    final List<Throwable> unexpected = Collections.synchronizedList(new ArrayList<>());
    List<Thread> workers = new ArrayList<>();
    for (int i = 0; i < workerCount; i++) {
      Thread worker = new Thread(() -> {
        try {
          start.await();
          while (!stop.get()) {
            HostMemoryBuffer buffer = null;
            try {
              buffer = PageableMemoryPool.tryAllocate(4096);
            } catch (IllegalStateException expectedAfterClose) {
              // alloc raced close/shutdown — allowed by the documented contract
            }
            if (buffer == null) {
              continue;  // uninitialized or exhausted mid-teardown
            }
            successfulAllocs.incrementAndGet();
            try {
              buffer.close();
            } catch (IllegalStateException expectedAfterClose) {
              // free raced shutdown — allowed by the documented contract
            }
          }
        } catch (Throwable t) {
          unexpected.add(t);
        }
      });
      workers.add(worker);
      worker.start();
    }
    start.countDown();
    // Wait until at least one worker has used the allocation path before shutdown.
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
    while (successfulAllocs.get() == 0 && System.nanoTime() < deadline) {
      Thread.yield();
    }
    assertTrue(successfulAllocs.get() > 0, "workers never allocated through the gate");
    stop.set(true);
    // The write gate must wait for in-flight alloc/free (read locks) before releasing
    // the native pool — no use-after-free, no unexpected worker exceptions.
    PageableMemoryPool.shutdown();
    for (Thread worker : workers) {
      worker.join();
    }
    assertTrue(unexpected.isEmpty(), () -> "unexpected exceptions: " + unexpected);
  }

  @Test
  public void shutdownWaitsForInFlightAllocation() throws Exception {
    PageableMemoryPool.initialize(4 * 1024 * 1024, 2);
    assertTrue(PageableMemoryPool.isInitialized());
    Field singletonField = PageableMemoryPool.class.getDeclaredField("singleton_");
    singletonField.setAccessible(true);
    Object pool = singletonField.get(null);
    Field gateField = PageableMemoryPool.class.getDeclaredField("lifecycleLock");
    gateField.setAccessible(true);
    ReentrantReadWriteLock gate = (ReentrantReadWriteLock) gateField.get(pool);

    CountDownLatch operationEntered = new CountDownLatch(1);
    CountDownLatch releaseOperation = new CountDownLatch(1);
    CountDownLatch shutdownFinished = new CountDownLatch(1);
    AtomicReference<Throwable> failure = new AtomicReference<>();
    Thread operation = new Thread(() -> {
      gate.readLock().lock();
      try {
        operationEntered.countDown();
        releaseOperation.await();
        try (HostMemoryBuffer buffer = PageableMemoryPool.tryAllocate(4096)) {
          assertNotNull(buffer);
        }
      } catch (Throwable t) {
        failure.set(t);
      } finally {
        gate.readLock().unlock();
      }
    });
    Thread shutdown = new Thread(() -> {
      try {
        PageableMemoryPool.shutdown();
      } catch (Throwable t) {
        failure.set(t);
      } finally {
        shutdownFinished.countDown();
      }
    });
    operation.start();
    try {
      assertTrue(operationEntered.await(10, TimeUnit.SECONDS));
      shutdown.start();
      long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
      while (!gate.hasQueuedThread(shutdown) && System.nanoTime() < deadline) {
        Thread.yield();
      }
      assertTrue(gate.hasQueuedThread(shutdown), "shutdown did not wait for the read gate");
      assertFalse(shutdownFinished.await(100, TimeUnit.MILLISECONDS));
    } finally {
      releaseOperation.countDown();
      operation.join(10000);
      if (shutdown.isAlive()) {
        shutdown.join(10000);
      }
    }
    assertFalse(operation.isAlive());
    assertFalse(shutdown.isAlive());
    assertNull(failure.get(), () -> "unexpected failure: " + failure.get());
    assertTrue(shutdownFinished.getCount() == 0);
  }
}
