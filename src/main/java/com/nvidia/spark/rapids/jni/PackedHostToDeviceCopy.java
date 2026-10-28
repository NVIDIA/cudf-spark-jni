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
 * stay open until the copies complete, which closing this waits for. The allocations are made on
 * the copy stream, and freed on the stream passed to {@link #waitOnEvent} if it was called, else on
 * the copy stream. The copies record the event the caller passes in.
 */
public final class PackedHostToDeviceCopy implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private final DeviceMemoryBuffer[] buffers;
  private final Cuda.Stream stream;
  private final Cuda.Event event;
  // What the native copy returned, holding each pack's rmm::device_buffer. Non-null only if the
  // native copy queued every copy successfully; if it throws, it has already waited for any copies
  // it queued and freed the packs.
  private long[] layout = null;
  // The packs wrapped so far; null until the native copy returns.
  private DeviceMemoryBuffer[] packs = null;
  private boolean recorded = false;
  private boolean closed = false;

  private PackedHostToDeviceCopy(int bufferCount, Cuda.Stream stream, Cuda.Event event) {
    this.buffers = new DeviceMemoryBuffer[bufferCount];
    this.stream = stream;
    this.event = event;
  }

  /**
   * Queue the copies of the host buffers to packed device allocations.
   *
   * @param hostBuffers the buffers to copy, which must stay open until the copies complete
   * @param stream the stream to allocate on and queue the copies on. It must not be one cuDF's
   *               stream pool could hand later work, and must stay open until the result is
   *               closed.
   * @param maxPackBytes the most bytes packed into one allocation, unless a single buffer is
   *                     larger. Every allocation is made before any copy is queued, so if one
   *                     fails nothing is in flight and the call can be retried with a smaller cap.
   * @param event the event recorded after the copies. It must stay open until the result is
   *              closed, and must not be passed to another copy before then.
   * @return the staged copy, which the caller must close
   */
  public static PackedHostToDeviceCopy copyAsync(HostMemoryBuffer[] hostBuffers,
      Cuda.Stream stream, long maxPackBytes, Cuda.Event event) {
    int count = hostBuffers.length;
    long[] addrs = new long[count];
    long[] lens = new long[count];
    for (int i = 0; i < count; i++) {
      addrs[i] = hostBuffers[i].getAddress();
      lens[i] = hostBuffers[i].getLength();
    }

    // Made before the copy, so that a failure from here on closes it, which waits for the copies
    // and frees the packs.
    return Arms.closeIfException(new PackedHostToDeviceCopy(count, stream, event), copy -> {
      long[] layout = copyAsync(addrs, lens, maxPackBytes, stream.getStream());
      copy.layout = layout;
      int packCount = (int) layout[0];
      int base = 1 + 3 * packCount;
      copy.packs = new DeviceMemoryBuffer[packCount];
      for (int p = 0; p < packCount; p++) {
        copy.packs[p] = DeviceMemoryBuffer.fromRmm(layout[2 + 3 * p], layout[3 + 3 * p],
            layout[1 + 3 * p]);
      }
      for (int i = 0; i < count; i++) {
        copy.buffers[i] = copy.packs[(int) layout[base + 2 * i]].slice(
            layout[base + 2 * i + 1], lens[i]);
      }
      event.record(stream);
      copy.recorded = true;
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
   * Make the stream that reads the device buffers wait for the copies, and free the allocations
   * on it, behind those reads. Does not block the calling thread.
   *
   * @param decodeStream the stream that reads the device buffers
   */
  public synchronized void waitOnEvent(Cuda.Stream decodeStream) {
    checkOpen();
    decodeStream.waitOn(event);
    setFreeStream(layout, decodeStream.getStream());
  }

  /**
   * Make a stream that does not read the device buffers wait for the copies, e.g. to queue a
   * later copy behind them. Does not block the calling thread.
   *
   * @param waiting the stream that waits
   */
  public synchronized void makeStreamWait(Cuda.Stream waiting) {
    checkOpen();
    waiting.waitOn(event);
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

    // A null layout means the native copy threw, after finishing any copies and freeing the packs.
    // Otherwise the copies were queued successfully, and may still be reading the host buffers, so
    // wait for them.
    if (layout == null) {
      return;
    }
    try {
      // Syncing the stream would also wait for work queued on it after the copies, such as a
      // decode, so wait on the event, which marks just the copies. If the event was never recorded
      // it doesn't mark them, so wait on the whole stream instead.
      if (recorded) {
        event.sync();
      } else {
        stream.sync();
      }
    } catch (Throwable t) {
      failure = t;
    }

    // Free the slices, then the packs they view, freeing any pack not yet wrapped by its handle.
    for (DeviceMemoryBuffer buffer : buffers) {
      failure = closeCollecting(buffer, failure);
    }
    for (int p = 0; p < (int) layout[0]; p++) {
      if (packs != null && packs[p] != null) {
        failure = closeCollecting(packs[p], failure);
      } else {
        try {
          freePack(layout[1 + 3 * p]);
        } catch (Throwable t) {
          failure = addFailure(failure, t);
        }
      }
    }

    rethrow(failure);
  }

  private void checkOpen() {
    if (closed) {
      throw new IllegalStateException("copy is closed");
    }
  }

  // Keep the first failure, with any later ones suppressed into it.
  private static Throwable addFailure(Throwable failure, Throwable t) {
    if (failure == null) {
      return t;
    }
    failure.addSuppressed(t);
    return failure;
  }

  private static Throwable closeCollecting(AutoCloseable resource, Throwable failure) {
    if (resource == null) {
      return failure;
    }
    try {
      resource.close();
    } catch (Throwable t) {
      return addFailure(failure, t);
    }
    return failure;
  }

  private static void rethrow(Throwable failure) {
    if (failure instanceof Error) {
      throw (Error) failure;
    }
    if (failure instanceof RuntimeException) {
      throw (RuntimeException) failure;
    }
    if (failure != null) {
      throw new RuntimeException(failure);
    }
  }

  // Returns the pack count, then each pack's rmm::device_buffer handle, address and length, then
  // each host buffer's pack and offset in it.
  private static native long[] copyAsync(long[] addrs, long[] lens, long maxPackBytes,
      long stream);

  // Free each pack in the layout copyAsync returned on stream when it is closed.
  private static native void setFreeStream(long[] layout, long stream);

  // Free a pack copyAsync returned that was never wrapped.
  private static native void freePack(long handle);
}
