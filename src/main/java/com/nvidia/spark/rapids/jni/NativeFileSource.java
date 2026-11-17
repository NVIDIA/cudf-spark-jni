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
import ai.rapids.cudf.MemoryBuffer;
import ai.rapids.cudf.NativeDepsLoader;

/**
 * A local file opened through libcudf's datasource, which reads it with KvikIO: with cuFile when
 * KvikIO opened the file in GDS mode, else through host memory. cuFile reads directly to the device
 * with GPUDirect Storage unless it falls back to its own compatibility mode, which stages through
 * host memory and is not reported through this class.
 */
public final class NativeFileSource implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * KvikIO's compatibility mode, which decides whether files are read with cuFile. cuFile has its
   * own compatibility mode and may still stage a read through host memory. The order must match
   * libcudf's kvikio_integration::compat_mode.
   */
  public enum CompatMode {
    /** Always POSIX, libcudf's default. */
    ON,
    /** Always cuFile; opening a file cuFile cannot read fails. */
    OFF,
    /** cuFile where it works for a file, else POSIX. */
    AUTO
  }

  private final long handle;
  // The owner's reference, released by close, plus one per call using the handle. The last
  // release frees the handle.
  private int refCount = 1;

  private NativeFileSource(long handle) {
    this.handle = handle;
  }

  /** One call's use of the handle, which keeps it open until the use is closed. */
  private final class NativeFileSourceHandle implements AutoCloseable {
    final long handle = NativeFileSource.this.handle;

    @Override
    public void close() {
      releaseHandle();
    }
  }

  /**
   * Set KvikIO's compatibility mode and thread pool size for the process, in place of the
   * KVIKIO_COMPAT_MODE and KVIKIO_NTHREADS environment variables. libcudf sets up KvikIO once, so
   * this must be called before any other cuDF IO in the process; a later call has no effect.
   *
   * @param mode the compatibility mode
   * @param numThreads the number of threads KvikIO splits each read across, at least 1
   */
  public static void setUpKvikio(CompatMode mode, int numThreads) {
    if (numThreads < 1) {
      throw new IllegalArgumentException("KvikIO needs at least 1 thread, got " + numThreads);
    }
    setUpKvikioNative(mode.ordinal(), numThreads);
  }

  /**
   * Open a local file.
   *
   * @param path the file's path, without a URI scheme
   * @return the source, which the caller must close
   */
  public static NativeFileSource open(String path) {
    return new NativeFileSource(openNative(path));
  }

  /** @return the file's size in bytes */
  public long size() {
    try (NativeFileSourceHandle h = acquireHandle()) {
      return size(h.handle);
    }
  }

  /**
   * @param size the number of bytes a device read would read
   * @return whether such a read is issued directly to the device through cuFile, without KvikIO
   *         staging it in host memory. cuFile may still stage it if it falls back to its own
   *         compatibility mode.
   */
  public boolean isDeviceReadDirect(long size) {
    try (NativeFileSourceHandle h = acquireHandle()) {
      return isDeviceReadDirect(h.handle, size);
    }
  }

  /**
   * Read byte ranges from the file into host memory. The reads run concurrently, and this blocks
   * until all of them finish, even when one fails. Range {@code i} reads {@code lengths[i]} bytes
   * at {@code fileOffsets[i]} into {@code dst} at {@code dstOffsets[i]}.
   *
   * @param fileOffsets the file offset of each range
   * @param lengths the length of each range; every range must lie within the file
   * @param dst the buffer to read into
   * @param dstOffsets the offset in {@code dst} of each range
   */
  public void readToHost(long[] fileOffsets, long[] lengths, HostMemoryBuffer dst,
      long[] dstOffsets) {
    long[] dstAddresses = dstAddresses(fileOffsets, lengths, dst, dstOffsets);
    try (NativeFileSourceHandle h = acquireHandle()) {
      readToHost(h.handle, fileOffsets, lengths, dstAddresses);
    }
  }

  /**
   * Read byte ranges from the file into device memory. The reads run concurrently, and this blocks
   * until all of them finish, even when one fails. Range {@code i} reads {@code lengths[i]} bytes
   * at {@code fileOffsets[i]} into {@code dst} at {@code dstOffsets[i]}.
   *
   * @param fileOffsets the file offset of each range
   * @param lengths the length of each range; every range must lie within the file
   * @param dst the buffer to read into
   * @param dstOffsets the offset in {@code dst} of each range
   * @param stream the stream ordering the reads; it is synchronized before they start
   */
  public void readToDevice(long[] fileOffsets, long[] lengths, DeviceMemoryBuffer dst,
      long[] dstOffsets, Cuda.Stream stream) {
    long[] dstAddresses = dstAddresses(fileOffsets, lengths, dst, dstOffsets);
    try (NativeFileSourceHandle h = acquireHandle()) {
      readToDevice(h.handle, fileOffsets, lengths, dstAddresses, stream.getStream());
    }
  }

  private static long[] dstAddresses(long[] fileOffsets, long[] lengths, MemoryBuffer dst,
      long[] dstOffsets) {
    if (fileOffsets.length != lengths.length || dstOffsets.length != lengths.length) {
      throw new IllegalArgumentException("got " + fileOffsets.length + " file offsets, " +
          lengths.length + " lengths and " + dstOffsets.length + " destination offsets");
    }
    long[] dstAddresses = new long[lengths.length];
    for (int i = 0; i < lengths.length; i++) {
      if (dstOffsets[i] < 0 || lengths[i] < 0 || dstOffsets[i] > dst.getLength() - lengths[i]) {
        throw new IllegalArgumentException("read of " + lengths[i] + " bytes at offset " +
            dstOffsets[i] + " does not fit a buffer of " + dst.getLength() + " bytes");
      }
      dstAddresses[i] = dst.getAddress() + dstOffsets[i];
    }
    return dstAddresses;
  }

  @Override
  public synchronized void close() {
    releaseHandle();
  }

  private synchronized NativeFileSourceHandle acquireHandle() {
    if (refCount == 0) {
      throw new IllegalStateException("file source is closed");
    }
    refCount++;
    return new NativeFileSourceHandle();
  }

  private synchronized void releaseHandle() {
    if (refCount == 0) {
      throw new IllegalStateException("Close called too many times");
    }
    refCount--;
    if (refCount == 0) {
      close(handle);
    }
  }

  private static native void setUpKvikioNative(int mode, int numThreads);
  private static native long openNative(String path);

  private static native long size(long handle);

  private static native boolean isDeviceReadDirect(long handle, long size);

  private static native void readToHost(long handle, long[] fileOffsets, long[] lengths,
      long[] dstAddresses);

  private static native void readToDevice(long handle, long[] fileOffsets, long[] lengths,
      long[] dstAddresses, long stream);

  private static native void close(long handle);
}
