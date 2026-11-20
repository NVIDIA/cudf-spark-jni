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
import ai.rapids.cudf.PageableMemoryPool;
import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmAllocationMode;

import java.lang.reflect.Field;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PageableAsyncD2HLifecycleTest {
  @Test
  public void interruptedSlotWaitLeavesCapacityIntact() throws Exception {
    configureAsyncPool(1);
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 64L * 1024 * 1024);
    ArrayBlockingQueue<Object> free = null;
    Object held = null;
    Thread caller = null;
    try {
      PageableMemoryPool.initialize(8192, 1);
      assertTrue(PageableMemoryPool.isInitialized());
      @SuppressWarnings("unchecked")
      ArrayBlockingQueue<Object> slots =
          (ArrayBlockingQueue<Object>) field("freeAsyncD2HSlots").get(null);
      free = slots;
      held = free.poll();
      assertNotNull(held);
      List<Throwable> failures = Collections.synchronizedList(new ArrayList<>());
      CountDownLatch started = new CountDownLatch(1);
      try (DeviceMemoryBuffer device = DeviceMemoryBuffer.allocate(4096);
           HostMemoryBuffer host = PageableMemoryPool.allocateRawForDeviceToHost(4096)) {
        caller = new Thread(() -> {
          started.countDown();
          try {
            PageableMemoryPool.copyDeviceToPageableHost(host, 0, device, 0, 4096,
                Cuda.DEFAULT_STREAM);
          } catch (Throwable t) {
            failures.add(t);
          }
        }, "interrupted D2H slot waiter");
        caller.start();
        assertTrue(started.await(5, TimeUnit.SECONDS));
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (caller.getState() != Thread.State.TIMED_WAITING &&
            System.nanoTime() < deadline) {
          Thread.yield();
        }
        assertEquals(Thread.State.TIMED_WAITING, caller.getState());
        caller.interrupt();
        caller.join(TimeUnit.SECONDS.toMillis(5));
        assertFalse(caller.isAlive());
        assertEquals(1, failures.size());
        assertTrue(hasCause(failures.get(0), InterruptedException.class));
        free.add(held);
        held = null;
        PageableMemoryPool.copyDeviceToPageableHost(host, 0, device, 0, 4096,
            Cuda.DEFAULT_STREAM);
        assertEquals(1, free.size());
      }
    } finally {
      if (held != null && free != null) {
        free.add(held);
      }
      if (caller != null) {
        caller.interrupt();
        caller.join(TimeUnit.SECONDS.toMillis(5));
      }
      PageableMemoryPool.shutdown();
      Rmm.shutdown();
      clearAsyncPoolConfig();
    }
  }

  @Test
  public void failedEventQueryAndSynchronizationsQuarantineSlot() throws Exception {
    configureAsyncPool(1);
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 64L * 1024 * 1024);
    Field query = field("asyncD2HEventQueryOverride");
    Field streamSync = field("beforeAsyncD2HStreamSync");
    Field deviceSync = field("beforeAsyncD2HDeviceSync");
    try {
      PageableMemoryPool.initialize(8192, 1);
      assertTrue(PageableMemoryPool.isInitialized());
      query.set(null, (BooleanSupplier) () -> {
        throw new IllegalStateException("injected event query failure");
      });
      streamSync.set(null, (Runnable) () -> {
        throw new IllegalStateException("injected stream sync failure");
      });
      deviceSync.set(null, (Runnable) () -> {
        throw new IllegalStateException("injected device sync failure");
      });
      ArrayBlockingQueue<?> free = (ArrayBlockingQueue<?>) field("freeAsyncD2HSlots").get(null);
      try (DeviceMemoryBuffer device = DeviceMemoryBuffer.allocate(4096);
           HostMemoryBuffer host = PageableMemoryPool.allocateRawForDeviceToHost(4096)) {
        RuntimeException failure = assertThrows(RuntimeException.class,
            () -> PageableMemoryPool.copyDeviceToPageableHost(host, 0, device, 0, 4096,
                Cuda.DEFAULT_STREAM));
        assertEquals(2, failure.getCause().getSuppressed().length);
        assertEquals(0, free.size(), "uncertain DMA returned an unsafe slot");
        assertThrows(IllegalStateException.class,
            () -> PageableMemoryPool.copyDeviceToPageableHost(host, 0, device, 0, 4096,
                Cuda.DEFAULT_STREAM));
      }
    } finally {
      query.set(null, null);
      streamSync.set(null, null);
      deviceSync.set(null, null);
      PageableMemoryPool.shutdown();
      Rmm.shutdown();
      clearAsyncPoolConfig();
    }
  }

  @Test
  public void shutdownWaitsForPendingCudaCopy() throws Exception {
    configureAsyncPool(1);
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 64L * 1024 * 1024);
    Field query = field("asyncD2HEventQueryOverride");
    CountDownLatch polling = new CountDownLatch(1);
    CountDownLatch release = new CountDownLatch(1);
    Thread caller = null;
    Thread shutdown = null;
    try {
      PageableMemoryPool.initialize(8192, 1);
      assertTrue(PageableMemoryPool.isInitialized());
      query.set(null, (BooleanSupplier) () -> {
        polling.countDown();
        try {
          release.await();
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new RuntimeException(e);
        }
        return false;
      });
      List<Throwable> failures = Collections.synchronizedList(new ArrayList<>());
      try (DeviceMemoryBuffer device = DeviceMemoryBuffer.allocate(4096);
           HostMemoryBuffer host = PageableMemoryPool.allocateRawForDeviceToHost(4096)) {
        caller = new Thread(() -> {
          try {
            PageableMemoryPool.copyDeviceToPageableHost(host, 0, device, 0, 4096,
                Cuda.DEFAULT_STREAM);
          } catch (Throwable t) {
            failures.add(t);
          }
        }, "pending CUDA D2H caller");
        caller.start();
        assertTrue(polling.await(5, TimeUnit.SECONDS));
        shutdown = new Thread(() -> {
          try {
            PageableMemoryPool.shutdown();
          } catch (Throwable t) {
            failures.add(t);
          }
        }, "pending CUDA D2H shutdown");
        shutdown.start();
        assertTrue(shutdown.isAlive());
        query.set(null, null);
        release.countDown();
        caller.join(TimeUnit.SECONDS.toMillis(5));
        shutdown.join(TimeUnit.SECONDS.toMillis(5));
        assertFalse(caller.isAlive());
        assertFalse(shutdown.isAlive());
        assertTrue(failures.isEmpty(), () -> "copy failures: " + failures);
      }
    } finally {
      query.set(null, null);
      release.countDown();
      if (caller != null) {
        caller.join(TimeUnit.SECONDS.toMillis(5));
      }
      if (shutdown != null) {
        shutdown.join(TimeUnit.SECONDS.toMillis(5));
      }
      PageableMemoryPool.shutdown();
      Rmm.shutdown();
      clearAsyncPoolConfig();
    }
  }

  @Test
  public void eventQueryFailureDrainsAndReturnsSlot() throws Exception {
    configureAsyncPool(1);
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 64L * 1024 * 1024);
    Field queryField = field("asyncD2HEventQueryOverride");
    try {
      PageableMemoryPool.initialize(8192, 1);
      assertTrue(PageableMemoryPool.isInitialized());
      AtomicInteger queries = new AtomicInteger();
      queryField.set(null, (BooleanSupplier) () -> {
        queries.incrementAndGet();
        throw new IllegalStateException("injected CUDA event query failure");
      });
      ArrayBlockingQueue<?> free = (ArrayBlockingQueue<?>) field("freeAsyncD2HSlots").get(null);
      try (DeviceMemoryBuffer device = DeviceMemoryBuffer.allocate(4096);
           HostMemoryBuffer host = PageableMemoryPool.allocateRawForDeviceToHost(4096)) {
        RuntimeException failure = assertThrows(RuntimeException.class,
            () -> PageableMemoryPool.copyDeviceToPageableHost(host, 0, device, 0, 4096,
                Cuda.DEFAULT_STREAM));
        assertTrue(hasCause(failure, IllegalStateException.class));
        assertTrue(queries.get() > 0);
        assertEquals(1, free.size(), "failed query did not return the drained slot");
        assertThrows(IllegalStateException.class,
            () -> PageableMemoryPool.copyDeviceToPageableHost(host, 0, device, 0, 4096,
                Cuda.DEFAULT_STREAM));
      }
    } finally {
      queryField.set(null, null);
      PageableMemoryPool.shutdown();
      Rmm.shutdown();
      clearAsyncPoolConfig();
    }
  }

  @Test
  public void slotAllocationFailureCleansPartialInitializationAndCanRetry() throws Exception {
    configureAsyncPool(2);
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 64L * 1024 * 1024);
    Field allocationField = field("beforeRegisteredStagingAllocation");
    try {
      AtomicInteger allocations = new AtomicInteger();
      allocationField.set(null, (Runnable) () -> {
        if (allocations.incrementAndGet() == 3) {
          throw new IllegalStateException("injected registered allocation failure");
        }
      });
      assertThrows(IllegalStateException.class, () -> PageableMemoryPool.initialize(8192, 1));
      assertEquals(3, allocations.get());
      assertEquals(null, field("stagedCopyExecutor").get(null));
      assertEquals(null, field("asyncD2HSlots").get(null));
      allocationField.set(null, null);
      PageableMemoryPool.initialize(8192, 1);
      assertTrue(PageableMemoryPool.isInitialized());
      assertEquals(2, ((List<?>) field("asyncD2HSlots").get(null)).size());
    } finally {
      allocationField.set(null, null);
      PageableMemoryPool.shutdown();
      Rmm.shutdown();
      clearAsyncPoolConfig();
    }
  }

  private static Field field(String name) throws NoSuchFieldException {
    Field result = PageableMemoryPool.class.getDeclaredField(name);
    result.setAccessible(true);
    return result;
  }

  private static void configureAsyncPool(int slots) {
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.enabled", "true");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.threads", "1");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.chunk.bytes", "4096");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.registered.buffers", "true");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h", "true");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h.slots",
        Integer.toString(slots));
  }

  private static void clearAsyncPoolConfig() {
    System.clearProperty("ai.rapids.cudf.pageable.staged.copy.enabled");
    System.clearProperty("ai.rapids.cudf.pageable.staged.copy.threads");
    System.clearProperty("ai.rapids.cudf.pageable.staged.copy.chunk.bytes");
    System.clearProperty("ai.rapids.cudf.pageable.staged.copy.registered.buffers");
    System.clearProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h");
    System.clearProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h.slots");
  }

  @Test
  public void moreCallersThanSlotsDrainAndRecoverCapacity() throws Exception {
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.enabled", "true");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.threads", "1");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.chunk.bytes", "4096");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.registered.buffers", "true");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h", "true");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h.slots", "1");
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 64L * 1024 * 1024);
    CountDownLatch releaseWorker = new CountDownLatch(1);
    CountDownLatch start = new CountDownLatch(1);
    List<Thread> callers = new ArrayList<>();
    ArrayBlockingQueue<Object> free = null;
    Object heldSlot = null;
    try {
      PageableMemoryPool.initialize(64 * 1024, 1);
      assertTrue(PageableMemoryPool.isInitialized());
      Field executorField = PageableMemoryPool.class.getDeclaredField("stagedCopyExecutor");
      executorField.setAccessible(true);
      ThreadPoolExecutor workers = (ThreadPoolExecutor) executorField.get(null);
      Field freeField = PageableMemoryPool.class.getDeclaredField("freeAsyncD2HSlots");
      freeField.setAccessible(true);
      @SuppressWarnings("unchecked")
      ArrayBlockingQueue<Object> slots = (ArrayBlockingQueue<Object>) freeField.get(null);
      free = slots;
      heldSlot = free.poll();
      assertNotNull(heldSlot);
      Field allSlotsField = PageableMemoryPool.class.getDeclaredField("asyncD2HSlots");
      allSlotsField.setAccessible(true);
      assertEquals(1, ((List<?>) allSlotsField.get(null)).size());
      // Registration is fixed at initialization: one worker buffer plus one async slot.
      assertEquals(1, ((long[]) field("registeredStagedCopyBuffers").get(null)).length);
      CountDownLatch workerBlocked = new CountDownLatch(1);
      workers.submit(() -> {
        workerBlocked.countDown();
        try {
          releaseWorker.await();
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
        }
      });
      assertTrue(workerBlocked.await(5, TimeUnit.SECONDS));

      byte[] pattern = new byte[2048];
      for (int i = 0; i < pattern.length; i++) {
        pattern[i] = (byte) (i * 13 + 5);
      }
      CountDownLatch ready = new CountDownLatch(6);
      List<Throwable> failures = Collections.synchronizedList(new ArrayList<>());
      try (HostMemoryBuffer input = HostMemoryBuffer.allocateRaw(pattern.length);
           DeviceMemoryBuffer device = DeviceMemoryBuffer.allocate(pattern.length)) {
        input.setBytes(0, pattern, 0, pattern.length);
        device.copyFromHostBuffer(input);
        for (int i = 0; i < 4; i++) {
          Thread caller = new Thread(() -> {
            try (HostMemoryBuffer destination =
                     PageableMemoryPool.allocateRawForDeviceToHost(pattern.length)) {
              ready.countDown();
              start.await();
              PageableMemoryPool.copyDeviceToPageableHost(destination, 0, device, 0,
                  pattern.length, Cuda.DEFAULT_STREAM);
              byte[] actual = new byte[pattern.length];
              destination.getBytes(actual, 0, 0, actual.length);
              assertArrayEquals(pattern, actual);
            } catch (Throwable t) {
              failures.add(t);
            }
          }, "contended D2H caller " + i);
          callers.add(caller);
          caller.start();
        }
        for (int i = 0; i < 2; i++) {
          Thread caller = new Thread(() -> {
            try (HostMemoryBuffer source = PageableMemoryPool.tryAllocate(pattern.length);
                 DeviceMemoryBuffer destination = DeviceMemoryBuffer.allocate(pattern.length);
                 HostMemoryBuffer output = HostMemoryBuffer.allocateRaw(pattern.length)) {
              assertNotNull(source);
              source.setBytes(0, pattern, 0, pattern.length);
              ready.countDown();
              start.await();
              PageableMemoryPool.copyPageableHostToDevice(destination, 0, source, 0,
                  pattern.length, Cuda.DEFAULT_STREAM);
              output.copyFromDeviceBuffer(destination);
              byte[] actual = new byte[pattern.length];
              output.getBytes(actual, 0, 0, actual.length);
              assertArrayEquals(pattern, actual);
            } catch (Throwable t) {
              failures.add(t);
            }
          }, "contended H2D caller " + i);
          callers.add(caller);
          caller.start();
        }
        assertTrue(ready.await(5, TimeUnit.SECONDS));
        start.countDown();
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while ((!allWaiting(callers.subList(0, 4)) || workers.getQueue().size() < 2) &&
            System.nanoTime() < deadline) {
          Thread.yield();
        }
        assertTrue(allWaiting(callers.subList(0, 4)), "D2H callers did not wait for the slot");
        assertTrue(workers.getQueue().size() >= 2, "H2D callers did not queue");
        assertEquals(0, free.size());
        free.add(heldSlot);
        heldSlot = null;
        releaseWorker.countDown();
        for (Thread caller : callers) {
          caller.join(TimeUnit.SECONDS.toMillis(5));
          assertFalse(caller.isAlive(), () -> "caller remained blocked: " + caller.getName());
        }
        assertTrue(failures.isEmpty(), () -> "copy failures: " + failures);
        assertEquals(1, free.size(), "async bounce capacity was not recovered");
        try (HostMemoryBuffer recovery =
                 PageableMemoryPool.allocateRawForDeviceToHost(pattern.length)) {
          PageableMemoryPool.copyDeviceToPageableHost(recovery, 0, device, 0,
              pattern.length, Cuda.DEFAULT_STREAM);
          byte[] actual = new byte[pattern.length];
          recovery.getBytes(actual, 0, 0, actual.length);
          assertArrayEquals(pattern, actual);
        }
        assertEquals(1, free.size());
      }
    } finally {
      if (heldSlot != null && free != null) {
        free.add(heldSlot);
      }
      start.countDown();
      releaseWorker.countDown();
      for (Thread caller : callers) {
        caller.join(TimeUnit.SECONDS.toMillis(5));
      }
      PageableMemoryPool.shutdown();
      Rmm.shutdown();
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.enabled");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.threads");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.chunk.bytes");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.registered.buffers");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h.slots");
    }
  }

  private static boolean allWaiting(List<Thread> callers) {
    for (Thread caller : callers) {
      if (caller.getState() != Thread.State.TIMED_WAITING) {
        return false;
      }
    }
    return true;
  }

  @Test
  public void rejectedHostCopyReturnsSlotAndFailsCaller() throws Exception {
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.enabled", "true");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.threads", "1");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.chunk.bytes", "4096");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.registered.buffers", "true");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h", "true");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h.slots", "1");
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 64L * 1024 * 1024);
    Field executorField = PageableMemoryPool.class.getDeclaredField("stagedCopyExecutor");
    executorField.setAccessible(true);
    ExecutorService original = null;
    try {
      PageableMemoryPool.initialize(8192, 1);
      assertTrue(PageableMemoryPool.isInitialized());
      original = (ExecutorService) executorField.get(null);
      ExecutorService delegate = original;
      ExecutorService rejecting = (ExecutorService) Proxy.newProxyInstance(
          ExecutorService.class.getClassLoader(), new Class<?>[] {ExecutorService.class},
          (proxy, method, args) -> {
            if ("execute".equals(method.getName())) {
              throw new RejectedExecutionException("injected async D2H host-copy rejection");
            }
            return method.invoke(delegate, args);
          });
      executorField.set(null, rejecting);
      Field freeField = PageableMemoryPool.class.getDeclaredField("freeAsyncD2HSlots");
      freeField.setAccessible(true);
      ArrayBlockingQueue<?> free = (ArrayBlockingQueue<?>) freeField.get(null);
      try (DeviceMemoryBuffer device = DeviceMemoryBuffer.allocate(4096);
           HostMemoryBuffer host = PageableMemoryPool.allocateRawForDeviceToHost(4096)) {
        RuntimeException failure = assertThrows(RuntimeException.class,
            () -> PageableMemoryPool.copyDeviceToPageableHost(host, 0, device, 0, 4096,
                Cuda.DEFAULT_STREAM));
        assertTrue(hasCause(failure, RejectedExecutionException.class),
            () -> "unexpected failure: " + failure);
        assertEquals(1, free.size(), "rejected host copy leaked its bounce slot");
        assertThrows(IllegalStateException.class,
            () -> PageableMemoryPool.copyDeviceToPageableHost(host, 0, device, 0, 4096,
                Cuda.DEFAULT_STREAM));
      }
    } finally {
      if (original != null) {
        executorField.set(null, original);
      }
      PageableMemoryPool.shutdown();
      Rmm.shutdown();
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.enabled");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.threads");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.chunk.bytes");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.registered.buffers");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h.slots");
    }
  }

  private static boolean hasCause(Throwable failure, Class<? extends Throwable> type) {
    for (Throwable cause = failure; cause != null; cause = cause.getCause()) {
      if (type.isInstance(cause)) {
        return true;
      }
    }
    return false;
  }
}
