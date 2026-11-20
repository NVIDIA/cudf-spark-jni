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
import ai.rapids.cudf.PinnedMemoryPool;
import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmAllocationMode;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PageableSpillCopyTest {
  @Test
  public void rejectedLaterChunkDrainsAcceptedChunk() throws Exception {
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.enabled", "true");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.threads", "1");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.chunk.bytes", "4096");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.registered.buffers", "true");
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 64L * 1024 * 1024);
    CountDownLatch releaseWorker = new CountDownLatch(1);
    Thread copier = null;
    java.lang.reflect.Field field = PageableMemoryPool.class.getDeclaredField(
        "stagedCopyExecutor");
    field.setAccessible(true);
    ExecutorService original = null;
    try {
      PageableMemoryPool.initialize(8192, 1);
      PageableMemoryPool.isInitialized();
      original = (ExecutorService) field.get(null);
      CountDownLatch workerBlocked = new CountDownLatch(1);
      original.submit(() -> {
        workerBlocked.countDown();
        try {
          releaseWorker.await();
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
        }
      });
      assertTrue(workerBlocked.await(5, TimeUnit.SECONDS));
      AtomicInteger submissions = new AtomicInteger();
      CountDownLatch accepted = new CountDownLatch(1);
      ExecutorService delegate = original;
      ExecutorService rejecting = (ExecutorService) java.lang.reflect.Proxy.newProxyInstance(
          ExecutorService.class.getClassLoader(), new Class<?>[] {ExecutorService.class},
          (proxy, method, args) -> {
            if ("submit".equals(method.getName()) && submissions.incrementAndGet() == 2) {
              throw new RejectedExecutionException("injected second chunk rejection");
            }
            Object result = method.invoke(delegate, args);
            if ("submit".equals(method.getName())) {
              accepted.countDown();
            }
            return result;
          });
      field.set(null, rejecting);
      AtomicReference<Throwable> failure = new AtomicReference<>();
      try (DeviceMemoryBuffer device = DeviceMemoryBuffer.allocate(8192);
           HostMemoryBuffer host = PageableMemoryPool.allocateRawForDeviceToHost(8192)) {
        copier = new Thread(() -> {
          try {
            PageableMemoryPool.copyDeviceToPageableHost(host, 0, device, 0, 8192,
                Cuda.DEFAULT_STREAM);
          } catch (Throwable t) {
            failure.set(t);
          }
        });
        copier.start();
        assertTrue(accepted.await(5, TimeUnit.SECONDS));
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (submissions.get() < 2 && System.nanoTime() < deadline) {
          Thread.yield();
        }
        assertEquals(2, submissions.get());
        assertTrue(copier.isAlive(), "copy returned before accepted chunk finished");
        releaseWorker.countDown();
        copier.join(TimeUnit.SECONDS.toMillis(5));
        assertFalse(copier.isAlive());
        assertTrue(failure.get() instanceof RejectedExecutionException,
            () -> "unexpected failure: " + failure.get());
      }
    } finally {
      releaseWorker.countDown();
      if (copier != null) {
        copier.join(TimeUnit.SECONDS.toMillis(5));
      }
      if (original != null) {
        field.set(null, original);
      }
      PageableMemoryPool.shutdown();
      Rmm.shutdown();
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.enabled");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.threads");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.chunk.bytes");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.registered.buffers");
    }
  }

  @Test
  public void shutdownDrainsQueuedDeviceToHostCopy() throws Exception {
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.enabled", "true");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.threads", "1");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.chunk.bytes", "4096");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.registered.buffers", "true");
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 64L * 1024 * 1024);
    CountDownLatch releaseWorker = new CountDownLatch(1);
    AtomicReference<Throwable> copyFailure = new AtomicReference<>();
    AtomicReference<Throwable> shutdownFailure = new AtomicReference<>();
    Thread copier = null;
    Thread shutdown = null;
    try {
      PageableMemoryPool.initialize(8192, 1);
      PageableMemoryPool.isInitialized();
      java.lang.reflect.Field field = PageableMemoryPool.class.getDeclaredField(
          "stagedCopyExecutor");
      field.setAccessible(true);
      ThreadPoolExecutor workers = (ThreadPoolExecutor) field.get(null);
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
      try (DeviceMemoryBuffer device = DeviceMemoryBuffer.allocate(4096);
           HostMemoryBuffer host = PageableMemoryPool.allocateRawForDeviceToHost(4096)) {
        copier = new Thread(() -> {
          try {
            PageableMemoryPool.copyDeviceToPageableHost(host, 0, device, 0, 4096,
                Cuda.DEFAULT_STREAM);
          } catch (Throwable t) {
            copyFailure.set(t);
          }
        });
        copier.start();
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (workers.getQueue().isEmpty() && System.nanoTime() < deadline) {
          Thread.yield();
        }
        assertFalse(workers.getQueue().isEmpty(), "copy was not queued");
        shutdown = new Thread(() -> {
          try {
            PageableMemoryPool.shutdown();
          } catch (Throwable t) {
            shutdownFailure.set(t);
          }
        });
        shutdown.start();
        java.lang.reflect.Field lifecycleField = PageableMemoryPool.class.getDeclaredField(
            "stagedCopyLifecycleLock");
        lifecycleField.setAccessible(true);
        ReentrantReadWriteLock lifecycle =
            (ReentrantReadWriteLock) lifecycleField.get(null);
        deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (!lifecycle.hasQueuedThread(shutdown) && System.nanoTime() < deadline) {
          Thread.yield();
        }
        assertTrue(lifecycle.hasQueuedThread(shutdown), "shutdown did not wait for copy lease");
        assertTrue(shutdown.isAlive(), "shutdown returned while copy was queued");
        releaseWorker.countDown();
        copier.join(TimeUnit.SECONDS.toMillis(5));
        shutdown.join(TimeUnit.SECONDS.toMillis(5));
        assertFalse(copier.isAlive());
        assertFalse(shutdown.isAlive());
        assertNull(copyFailure.get());
        assertNull(shutdownFailure.get());
      }
    } finally {
      releaseWorker.countDown();
      if (copier != null) {
        copier.join(TimeUnit.SECONDS.toMillis(5));
      }
      if (shutdown != null) {
        shutdown.join(TimeUnit.SECONDS.toMillis(5));
      }
      PageableMemoryPool.shutdown();
      Rmm.shutdown();
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.enabled");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.threads");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.chunk.bytes");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.registered.buffers");
    }
  }

  @Test
  public void roundTripWithOffsetsAndMultipleChunks() {
    roundTrip(false, 2, false);
  }

  @Test
  public void asyncRoundTripWithOffsetsAndMultipleChunks() {
    roundTrip(true, 2, true);
  }

  @Test
  public void asyncRoundTripWithMoreSlotsThanWorkers() {
    roundTrip(true, 4, true);
  }

  private void roundTrip(boolean asyncD2H, int slots, boolean registeredBuffers) {
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h",
        Boolean.toString(asyncD2H));
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h.slots",
        Integer.toString(slots));
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.enabled", "true");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.threads", "2");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.chunk.bytes", "1048576");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.registered.buffers",
        Boolean.toString(registeredBuffers));

    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 64L * 1024 * 1024);
    try {
      if (!registeredBuffers) {
        PinnedMemoryPool.initialize(4L * 1024 * 1024);
      }
      PageableMemoryPool.initialize(8L * 1024 * 1024, 2);
      int length = 1024 * 1024 + 17;
      byte[] expected = new byte[length];
      Arrays.fill(expected, (byte) 0x5a);
      try (HostMemoryBuffer input = HostMemoryBuffer.allocateRaw(length);
           DeviceMemoryBuffer device = DeviceMemoryBuffer.allocate(length);
           DeviceMemoryBuffer restored = DeviceMemoryBuffer.allocate(length);
           HostMemoryBuffer output = HostMemoryBuffer.allocateRaw(length)) {
        input.setBytes(0, expected, 0, length);
        device.copyFromHostBuffer(input);
        try (HostMemoryBuffer staged =
                 PageableMemoryPool.tryAllocateForDeviceToHost(length)) {
          assertNotNull(staged);
          staged.copyFromDeviceBufferAsync(device, Cuda.DEFAULT_STREAM);
          byte[] automatic = new byte[length];
          staged.getBytes(automatic, 0, 0, length);
          assertArrayEquals(expected, automatic);
        }
        PageableMemoryPool.copyDeviceToPageableHost(output, 0, device, 0, length,
            Cuda.DEFAULT_STREAM);
        byte[] rawExplicit = new byte[length];
        output.getBytes(rawExplicit, 0, 0, length);
        assertArrayEquals(expected, rawExplicit);
        try (HostMemoryBuffer raw = PageableMemoryPool.allocateRawForDeviceToHost(length)) {
          assertFalse(PageableMemoryPool.isPageablePoolBuffer(raw));
          raw.copyFromDeviceBufferAsync(device, Cuda.DEFAULT_STREAM);
          byte[] rawAutomatic = new byte[length];
          raw.getBytes(rawAutomatic, 0, 0, length);
          assertArrayEquals(expected, rawAutomatic);
        }
        try (HostMemoryBuffer spill = PageableMemoryPool.tryAllocate(length + 32)) {
          assertNotNull(spill);
          PageableMemoryPool.copyDeviceToPageableHost(spill, 16, device, 0, length,
              Cuda.DEFAULT_STREAM);
          PageableMemoryPool.copyPageableHostToDevice(restored, 0, spill, 16, length,
              Cuda.DEFAULT_STREAM);
        }
        output.copyFromDeviceBuffer(restored);
        byte[] actual = new byte[length];
        output.getBytes(actual, 0, 0, length);
        assertArrayEquals(expected, actual);
      }
    } finally {
      PageableMemoryPool.shutdown();
      if (!registeredBuffers) {
        PinnedMemoryPool.shutdown();
      }
      Rmm.shutdown();
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.enabled");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.threads");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.chunk.bytes");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.registered.buffers");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h.slots");
    }
  }
}
