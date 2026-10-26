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
import ai.rapids.cudf.NativeDepsLoader;

import java.util.Arrays;

/**
 * Host buffers staged on the device in packed allocations, with one device buffer viewing each
 * host buffer's slice of them.
 * <p>
 * The buffers are packed back to back in the order given, starting a new allocation whenever the
 * next buffer would take the current one past a size cap; a buffer larger than the cap gets an
 * allocation of its own. Buffers adjacent in host memory and in the same allocation move in a
 * single copy. Each allocation's size is rounded up to a multiple of 8 bytes, as libcudf does for
 * the column chunk buffers it reads into.
 * <p>
 * The copies are only queued, and they read the host buffers asynchronously: the host buffers must
 * stay open until the copies complete, which closing this waits for. The allocations are made and
 * freed on the copy stream. On a stream other than the default stream the copies can overlap work
 * on the default stream; call {@link #waitOnEvent()} before the default stream reads the device
 * buffers. Closing keeps the allocations from being reused until the default stream finishes the
 * work queued on it so far. The default stream is per thread, so every call must come from the
 * thread that made the copy.
 */
public final class PackedHostToDeviceCopy implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  // Each allocation's size is rounded up to a multiple of this, as libcudf does for the column
  // chunk buffers it reads into. Matches libcudf's BUFFER_PADDING_MULTIPLE.
  private static final long PADDING_MULTIPLE = 8;

  private final DeviceMemoryBuffer[] packs;
  private final DeviceMemoryBuffer[] buffers;
  private final Cuda.Stream stream;
  // Recorded after the copies; null when nothing was copied.
  private Cuda.Event event = null;
  private boolean closed = false;

  private PackedHostToDeviceCopy(DeviceMemoryBuffer[] packs, DeviceMemoryBuffer[] buffers,
      Cuda.Stream stream) {
    this.packs = packs;
    this.buffers = buffers;
    this.stream = stream;
  }

  /**
   * Queue the copies of the host buffers to packed device allocations.
   *
   * @param hostBuffers the buffers to copy, which must stay open until the copies complete
   * @param stream the stream to queue the copies on. Another stream than
   *               {@link Cuda#DEFAULT_STREAM} lets them overlap later work on the default stream
   *               (e.g. decode), at the cost of cross-stream synchronization. It must not be one
   *               cuDF's stream pool could hand that later work, and must stay open until the
   *               result is closed.
   * @param maxPackBytes the most bytes packed into one allocation, unless a single buffer is
   *                     larger. Every allocation is made before any copy is queued, so if one
   *                     fails nothing is in flight and the call can be retried with a smaller cap.
   * @return the staged copy, which the caller must close
   */
  public static PackedHostToDeviceCopy copyAsync(HostMemoryBuffer[] hostBuffers,
      Cuda.Stream stream, long maxPackBytes) {
    // Gather the host addresses and lengths.
    int count = hostBuffers.length;
    long[] addrs = new long[count];
    long[] lens = new long[count];
    long total = 0;
    for (int i = 0; i < count; i++) {
      addrs[i] = hostBuffers[i].getAddress();
      lens[i] = hostBuffers[i].getLength();
      total += lens[i];
    }

    // Assign each buffer a pack and an offset in it; there are at most as many packs as buffers.
    int[] packOf = new int[count];
    long[] offsets = new long[count];
    long[] packSizes = new long[count];
    int[] packCounts = new int[count];
    int packCount = 0;
    for (int i = 0; i < count; i++) {
      // An empty buffer never starts a pack, so no pack is left holding only empty buffers.
      if (packCount == 0 || (lens[i] > 0 && packSizes[packCount - 1] > 0 &&
          packSizes[packCount - 1] + lens[i] > maxPackBytes)) {
        packCount++;
      }
      int p = packCount - 1;
      packOf[i] = p;
      offsets[i] = packSizes[p];
      packSizes[p] += lens[i];
      packCounts[p]++;
    }

    // Nothing to copy: hand back empty device buffers.
    DeviceMemoryBuffer[] packs = new DeviceMemoryBuffer[packCount];
    DeviceMemoryBuffer[] buffers = new DeviceMemoryBuffer[count];
    if (total == 0) {
      return Arms.closeIfException(new PackedHostToDeviceCopy(packs, buffers, stream), copy -> {
        for (int i = 0; i < buffers.length; i++) {
          buffers[i] = DeviceMemoryBuffer.allocate(0);
        }
        return copy;
      });
    }

    return Arms.closeIfException(new PackedHostToDeviceCopy(packs, buffers, stream), copy -> {
      // Allocate every pack before queueing any copy, so a failed allocation leaves nothing in
      // flight.
      long[] dstAddrs = new long[packs.length];
      long[] dstLens = new long[packs.length];
      for (int p = 0; p < packs.length; p++) {
        long size = (packSizes[p] + PADDING_MULTIPLE - 1) / PADDING_MULTIPLE * PADDING_MULTIPLE;
        packs[p] = DeviceMemoryBuffer.allocate(size, stream);
        dstAddrs[p] = packs[p].getAddress();
        dstLens[p] = packs[p].getLength();
      }

      // Slice out each buffer's view of its pack.
      for (int i = 0; i < count; i++) {
        buffers[i] = packs[packOf[i]].slice(offsets[i], lens[i]);
      }

      // Queue the copies and record the event after them. The event is made first, so failing to
      // make it leaves nothing in flight.
      copy.event = new Cuda.Event();
      copyAsync(addrs, lens, dstAddrs, dstLens, Arrays.copyOf(packCounts, packs.length),
          stream.getStream());
      copy.event.record(stream);
      return copy;
    });
  }

  /**
   * @return one device buffer per host buffer, in the order given; owned by this object, so the
   *         caller must not close them
   */
  public DeviceMemoryBuffer[] getBuffers() {
    return buffers;
  }

  // Visible for testing.
  DeviceMemoryBuffer[] getPacks() {
    return packs;
  }

  /**
   * Make the default stream wait for the copies before running any work queued on it afterwards.
   * Does not block the calling thread.
   */
  public void waitOnEvent() {
    if (event != null) {
      Cuda.DEFAULT_STREAM.waitOn(event);
    }
  }

  /** Wait for the copies to finish, then release the device allocations. */
  @Override
  public synchronized void close() {
    // Close only once.
    if (closed) {
      return;
    }
    closed = true;

    // Each step below catches its own failure, so none can skip the rest of the cleanup.
    Throwable failure = null;

    // Wait for the copies, which may still be reading the host buffers and writing the packs.
    if (event != null) {
      try {
        event.sync();
      } catch (Throwable t) {
        failure = t;
      }
    }

    // The packs are freed on the copy stream, so order the frees behind the default stream's reads
    // of them. Without the event the caller never had the buffers to read.
    if (event != null && stream != Cuda.DEFAULT_STREAM) {
      try {
        joinStreams(Cuda.DEFAULT_STREAM.getStream(), stream.getStream());
      } catch (Throwable t) {
        if (failure == null) {
          failure = t;
        } else {
          failure.addSuppressed(t);
        }
      }
    }

    // Free the slices, then the packs they view.
    for (DeviceMemoryBuffer buffer : buffers) {
      failure = closeCollecting(buffer, failure);
    }
    for (DeviceMemoryBuffer pack : packs) {
      failure = closeCollecting(pack, failure);
    }

    // Release the event.
    if (event != null) {
      try {
        event.close();
      } catch (Throwable t) {
        if (failure == null) {
          failure = t;
        } else {
          failure.addSuppressed(t);
        }
      }
    }

    // Rethrow the first failure, with any later ones suppressed into it.
    if (failure instanceof Error) {
      throw (Error) failure;
    }
    if (failure != null) {
      throw (RuntimeException) failure;
    }
  }

  private static Throwable closeCollecting(DeviceMemoryBuffer buffer, Throwable failure) {
    if (buffer == null) {
      return failure;
    }
    try {
      buffer.close();
    } catch (Throwable t) {
      if (failure == null) {
        return t;
      }
      failure.addSuppressed(t);
    }
    return failure;
  }

  // Destination d takes the next dstCounts[d] host buffers, packed back to back.
  private static native void copyAsync(long[] addrs, long[] lens, long[] dstAddrs,
      long[] dstLens, int[] dstCounts, long stream);

  // Make stream wait for the work queued on waitedOn so far.
  private static native void joinStreams(long waitedOn, long stream);
}
