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

import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.NativeDepsLoader;

/**
 * Host buffers staged on the device as one packed allocation, with one device buffer viewing each
 * host buffer's slice of it.
 * <p>
 * The buffers are packed back to back in the order given, and buffers adjacent in host memory move
 * in a single copy. The allocation is padded past the last buffer, because the Parquet decode
 * kernels read beyond the end of the column chunk data.
 * <p>
 * The copies are only queued, and they read the host buffers asynchronously: the host buffers must
 * stay open until this is closed, which waits for the copies to finish. On a side stream the
 * copies are queued behind everything already on the default stream, and can overlap work queued
 * on the default stream afterwards; call {@link #waitOnEvent()} before the default stream
 * reads the device buffers. The default stream is per thread, so every call must come from the
 * thread that made the copy.
 */
public final class PackedHostToDeviceCopy implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private final DeviceMemoryBuffer packed;
  private final DeviceMemoryBuffer[] buffers;
  private long event;
  private boolean closed = false;

  private PackedHostToDeviceCopy(DeviceMemoryBuffer packed, DeviceMemoryBuffer[] buffers,
      long event) {
    this.packed = packed;
    this.buffers = buffers;
    this.event = event;
  }

  /**
   * Queue the copies of the host buffers to one packed device allocation.
   *
   * @param hostBuffers the buffers to copy, which must stay open until the result is closed
   * @param onSideStream whether to queue the copies on a stream forked from the default stream
   *                     rather than on the default stream itself
   * @return the staged copy, which the caller must close
   */
  public static PackedHostToDeviceCopy copyAsync(HostMemoryBuffer[] hostBuffers,
      boolean onSideStream) {
    long[] addrs = new long[hostBuffers.length];
    long[] lens = new long[hostBuffers.length];
    for (int i = 0; i < hostBuffers.length; i++) {
      addrs[i] = hostBuffers[i].getAddress();
      lens[i] = hostBuffers[i].getLength();
    }
    // [buffer address, allocation size, rmm buffer handle, event handle]
    long[] result = copyAsync(addrs, lens, onSideStream);
    DeviceMemoryBuffer[] buffers = new DeviceMemoryBuffer[hostBuffers.length];
    if (result[2] == 0) {
      try {
        for (int i = 0; i < buffers.length; i++) {
          buffers[i] = DeviceMemoryBuffer.allocate(0);
        }
      } catch (Throwable t) {
        closeAll(buffers);
        throw t;
      }
      return new PackedHostToDeviceCopy(null, buffers, 0);
    }
    DeviceMemoryBuffer packed = null;
    try {
      packed = DeviceMemoryBuffer.fromRmm(result[0], result[1], result[2]);
      long offset = 0;
      for (int i = 0; i < buffers.length; i++) {
        buffers[i] = packed.slice(offset, lens[i]);
        offset += lens[i];
      }
    } catch (Throwable t) {
      // The copies may still be writing the allocation, so wait for them before it is freed.
      try {
        finish(result[3]);
      } finally {
        closeAll(buffers);
        if (packed != null) {
          packed.close();
        }
      }
      throw t;
    }
    return new PackedHostToDeviceCopy(packed, buffers, result[3]);
  }

  /**
   * @return one device buffer per host buffer, in the order given; owned by this object, so the
   *         caller must not close them
   */
  public DeviceMemoryBuffer[] getBuffers() {
    return buffers;
  }

  /**
   * Make the default stream wait for the copies before running any work queued on it afterwards.
   * Does not block the calling thread.
   */
  public void waitOnEvent() {
    if (event != 0) {
      waitOnEvent(event);
    }
  }

  /** Wait for the copies to finish, then release the device allocation. */
  @Override
  public synchronized void close() {
    if (closed) {
      return;
    }
    closed = true;
    try {
      if (event != 0) {
        long toFinish = event;
        event = 0;
        finish(toFinish);
      }
    } finally {
      closeAll(buffers);
      if (packed != null) {
        packed.close();
      }
    }
  }

  private static void closeAll(DeviceMemoryBuffer[] buffers) {
    for (int i = 0; i < buffers.length; i++) {
      if (buffers[i] != null) {
        buffers[i].close();
        buffers[i] = null;
      }
    }
  }

  private static native long[] copyAsync(long[] addrs, long[] lens, boolean onSideStream);

  private static native void waitOnEvent(long event);

  private static native void finish(long event);
}
