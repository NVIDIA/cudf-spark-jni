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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PageableStagedCopyConcurrencyTest {
  @Test
  public void mixedDirectionsShareWorkerAndRespectIndependentStreams() throws Exception {
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.enabled", "true");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.threads", "1");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.chunk.bytes", "4096");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.registered.buffers", "true");
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 64L * 1024 * 1024);
    CountDownLatch releaseWorker = new CountDownLatch(1);
    Thread h2dCaller = null;
    Thread d2hCaller = null;
    try {
      PageableMemoryPool.initialize(64 * 1024, 1);
      assertTrue(PageableMemoryPool.isInitialized());
      Field executorField = PageableMemoryPool.class.getDeclaredField("stagedCopyExecutor");
      executorField.setAccessible(true);
      ThreadPoolExecutor workers = (ThreadPoolExecutor) executorField.get(null);
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

      int length = 2 * 4096 + 17;
      int h2dSrcOffset = 7;
      int h2dDstOffset = 19;
      int d2hSrcOffset = 9;
      int d2hDstOffset = 13;
      byte[] h2dPattern = new byte[length];
      byte[] d2hPattern = new byte[length];
      for (int i = 0; i < length; i++) {
        h2dPattern[i] = (byte) (i * 31 + 7);
        d2hPattern[i] = (byte) (i * 17 + 11);
      }
      byte[] h2dSource = new byte[length + h2dSrcOffset + 3];
      Arrays.fill(h2dSource, (byte) 0x55);
      System.arraycopy(h2dPattern, 0, h2dSource, h2dSrcOffset, length);
      byte[] expectedH2D = new byte[length + h2dDstOffset + 5];
      Arrays.fill(expectedH2D, (byte) 0x44);
      System.arraycopy(h2dPattern, 0, expectedH2D, h2dDstOffset, length);
      byte[] expectedD2H = new byte[length + d2hDstOffset + 5];
      Arrays.fill(expectedD2H, (byte) 0x33);
      System.arraycopy(d2hPattern, 0, expectedD2H, d2hDstOffset, length);
      byte[] initialD2H = new byte[expectedD2H.length];
      Arrays.fill(initialD2H, (byte) 0x33);
      byte[] seedBytes = new byte[length + d2hSrcOffset];
      System.arraycopy(d2hPattern, 0, seedBytes, d2hSrcOffset, length);

      CountDownLatch start = new CountDownLatch(1);
      List<Throwable> failures = Collections.synchronizedList(new ArrayList<>());
      AtomicReference<byte[]> d2hResult = new AtomicReference<>();
      try (HostMemoryBuffer seedHost = HostMemoryBuffer.allocateRaw(seedBytes.length);
           DeviceMemoryBuffer seed = DeviceMemoryBuffer.allocate(seedBytes.length);
           DeviceMemoryBuffer h2dDestination = DeviceMemoryBuffer.allocate(expectedH2D.length);
           DeviceMemoryBuffer observed = DeviceMemoryBuffer.allocate(expectedH2D.length);
           HostMemoryBuffer output = HostMemoryBuffer.allocateRaw(expectedH2D.length)) {
        seedHost.setBytes(0, seedBytes, 0, seedBytes.length);
        seed.copyFromHostBuffer(seedHost);
        Cuda.memset(h2dDestination.getAddress(), (byte) 0x44, expectedH2D.length);

        h2dCaller = new Thread(() -> {
          try (Cuda.Stream callerStream = new Cuda.Stream(true);
               HostMemoryBuffer source = PageableMemoryPool.tryAllocate(h2dSource.length)) {
            assertNotNull(source);
            source.setBytes(0, h2dSource, 0, h2dSource.length);
            start.await();
            PageableMemoryPool.copyPageableHostToDevice(h2dDestination, h2dDstOffset,
                source, h2dSrcOffset, length, callerStream);
          } catch (Throwable t) {
            failures.add(t);
          }
        }, "staged H2D caller");
        d2hCaller = new Thread(() -> {
          try (Cuda.Stream callerStream = new Cuda.Stream(true);
               DeviceMemoryBuffer source = DeviceMemoryBuffer.allocate(seedBytes.length);
               HostMemoryBuffer destination =
                   PageableMemoryPool.allocateRawForDeviceToHost(expectedD2H.length)) {
            destination.setBytes(0, initialD2H, 0, initialD2H.length);
            // Producer work is queued on this caller's stream before the staged copy.
            source.copyFromDeviceBufferAsync(0, seed, 0, seedBytes.length, callerStream);
            start.await();
            PageableMemoryPool.copyDeviceToPageableHost(destination, d2hDstOffset,
                source, d2hSrcOffset, length, callerStream);
            byte[] result = new byte[expectedD2H.length];
            destination.getBytes(result, 0, 0, result.length);
            d2hResult.set(result);
          } catch (Throwable t) {
            failures.add(t);
          }
        }, "staged D2H caller");
        h2dCaller.start();
        d2hCaller.start();
        start.countDown();
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (workers.getQueue().size() < 2 && System.nanoTime() < deadline) {
          Thread.yield();
        }
        assertTrue(workers.getQueue().size() >= 2, "both copy directions did not queue");
        releaseWorker.countDown();
        h2dCaller.join(TimeUnit.SECONDS.toMillis(5));
        d2hCaller.join(TimeUnit.SECONDS.toMillis(5));
        assertFalse(h2dCaller.isAlive());
        assertFalse(d2hCaller.isAlive());
        assertTrue(failures.isEmpty(), () -> "copy failures: " + failures);
        assertArrayEquals(expectedD2H, d2hResult.get());
        // Consume on a separate stream after H2D has returned and its source has closed.
        try (Cuda.Stream consumerStream = new Cuda.Stream(true)) {
          observed.copyFromDeviceBufferAsync(0, h2dDestination, 0,
              expectedH2D.length, consumerStream);
          consumerStream.sync();
        }
        output.copyFromDeviceBuffer(observed);
        byte[] actualH2D = new byte[expectedH2D.length];
        output.getBytes(actualH2D, 0, 0, actualH2D.length);
        assertArrayEquals(expectedH2D, actualH2D);
      }
    } finally {
      releaseWorker.countDown();
      if (h2dCaller != null) {
        h2dCaller.join(TimeUnit.SECONDS.toMillis(5));
      }
      if (d2hCaller != null) {
        d2hCaller.join(TimeUnit.SECONDS.toMillis(5));
      }
      PageableMemoryPool.shutdown();
      Rmm.shutdown();
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.enabled");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.threads");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.chunk.bytes");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.registered.buffers");
    }
  }
}
