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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

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
  public void rejectsInvalidAllocationSize() {
    assertThrows(IllegalArgumentException.class, () -> PageableMemoryPool.tryAllocate(0));
    assertThrows(IllegalArgumentException.class, () -> PageableMemoryPool.tryAllocate(-1));
  }

  @Test
  public void interruptedCopyWaitDrainsEveryWorkerBeforeReturning() throws Exception {
    CountDownLatch waiting = new CountDownLatch(1);
    CountDownLatch sawInterrupt = new CountDownLatch(1);
    CompletableFuture<Void> first = new CompletableFuture<Void>() {
      @Override
      public Void get() throws InterruptedException, ExecutionException {
        waiting.countDown();
        try {
          return super.get();
        } catch (InterruptedException e) {
          sawInterrupt.countDown();
          throw e;
        }
      }
    };
    CountDownLatch waitingForSecond = new CountDownLatch(1);
    CompletableFuture<Void> second = new CompletableFuture<Void>() {
      @Override
      public Void get() throws InterruptedException, ExecutionException {
        waitingForSecond.countDown();
        return super.get();
      }
    };
    AtomicBoolean interruptedOnReturn = new AtomicBoolean(false);
    AtomicBoolean returned = new AtomicBoolean(false);
    List<Throwable> failures = Collections.synchronizedList(new ArrayList<>());
    Thread caller = new Thread(() -> {
      try {
        PageableMemoryPool.waitForStagedCopies(Arrays.<Future<?>>asList(first, second));
      } catch (RuntimeException expected) {
        failures.add(expected.getCause());
      } finally {
        interruptedOnReturn.set(Thread.currentThread().isInterrupted());
        returned.set(true);
      }
    });
    caller.start();
    assertTrue(waiting.await(5, TimeUnit.SECONDS));
    caller.interrupt();
    assertTrue(sawInterrupt.await(5, TimeUnit.SECONDS));
    first.complete(null);
    assertTrue(waitingForSecond.await(5, TimeUnit.SECONDS));
    assertFalse(returned.get(), "caller returned before the second worker finished");
    second.complete(null);
    caller.join(TimeUnit.SECONDS.toMillis(5));
    assertFalse(caller.isAlive());
    assertTrue(interruptedOnReturn.get());
    assertEquals(1, failures.size());
    assertTrue(failures.get(0) instanceof InterruptedException);
  }

  @Test
  public void failedCopyWaitDrainsOtherWorkersBeforeReturning() throws Exception {
    CompletableFuture<Void> failed = new CompletableFuture<>();
    CompletableFuture<Void> pending = new CompletableFuture<>();
    CountDownLatch waitingForPending = new CountDownLatch(1);
    CompletableFuture<Void> observedPending = new CompletableFuture<Void>() {
      @Override
      public Void get() throws InterruptedException, ExecutionException {
        waitingForPending.countDown();
        return pending.get();
      }
    };
    AtomicBoolean returned = new AtomicBoolean(false);
    Thread caller = new Thread(() -> {
      try {
        PageableMemoryPool.waitForStagedCopies(
            Arrays.<Future<?>>asList(failed, observedPending));
      } catch (RuntimeException expected) {
        returned.set(true);
      }
    });
    failed.completeExceptionally(new IllegalStateException("injected worker failure"));
    caller.start();
    assertTrue(waitingForPending.await(5, TimeUnit.SECONDS));
    assertFalse(returned.get(), "caller returned while another worker was active");
    pending.complete(null);
    caller.join(TimeUnit.SECONDS.toMillis(5));
    assertFalse(caller.isAlive());
    assertTrue(returned.get());
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
  public void concurrentAllocationAndShutdown() throws Exception {
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
    // Start shutdown after allocation activity has begun. This is a concurrency
    // smoke test; it does not force a particular allocation/shutdown interleaving.
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
    while (successfulAllocs.get() == 0 && System.nanoTime() < deadline) {
      Thread.yield();
    }
    assertTrue(successfulAllocs.get() > 0, "workers never allocated");
    stop.set(true);
    // Synchronized allocation/free and close serialize access to the native pool.
    PageableMemoryPool.shutdown();
    for (Thread worker : workers) {
      worker.join();
    }
    assertTrue(unexpected.isEmpty(), () -> "unexpected exceptions: " + unexpected);
  }
}
