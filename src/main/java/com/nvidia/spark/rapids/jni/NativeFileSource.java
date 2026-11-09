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
import ai.rapids.cudf.NativeDepsLoader;

/**
 * A local file opened through libcudf's datasource, which reads it with KvikIO: with cuFile when
 * KvikIO opened the file in GDS mode, else through host memory. cuFile reads directly to the device
 * with GPUDirect Storage unless it falls back to its own compatibility mode, which stages through
 * host memory and is not observable.
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

  private long handle;

  private NativeFileSource(long handle) {
    this.handle = handle;
  }

  /**
   * Set KvikIO's compatibility mode for the process. libcudf reads the mode once, when it first
   * sets up KvikIO, so this must be called before any other cuDF IO in the process; a later call
   * has no effect.
   *
   * @param mode the compatibility mode
   */
  public static void setUpKvikio(CompatMode mode) {
    setUpKvikioNative(mode.ordinal());
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
    return size(getHandle());
  }

  /**
   * @param size the number of bytes a device read would read
   * @return whether such a read is issued directly to the device through cuFile, without KvikIO
   *         staging it in host memory. cuFile may still stage it if it falls back to its own
   *         compatibility mode.
   */
  public boolean isDeviceReadDirect(long size) {
    return isDeviceReadDirect(getHandle(), size);
  }

  /**
   * Read bytes from the file into device memory, blocking until the read completes.
   *
   * @param fileOffset the file offset to read from
   * @param dst the buffer to read into
   * @param dstOffset the offset in {@code dst} to read to
   * @param length the number of bytes to read
   * @param stream the stream ordering the read; it is synchronized before the read starts
   * @return the number of bytes read, fewer than {@code length} at the end of the file
   */
  public long deviceRead(long fileOffset, DeviceMemoryBuffer dst, long dstOffset, long length,
      Cuda.Stream stream) {
    if (dstOffset < 0 || length < 0 || dstOffset + length > dst.getLength()) {
      throw new IllegalArgumentException("read of " + length + " bytes at offset " + dstOffset +
          " does not fit a buffer of " + dst.getLength() + " bytes");
    }
    return deviceRead(getHandle(), fileOffset, length, dst.getAddress() + dstOffset,
        stream.getStream());
  }

  @Override
  public synchronized void close() {
    if (handle != 0) {
      close(handle);
      handle = 0;
    }
  }

  private synchronized long getHandle() {
    if (handle == 0) {
      throw new IllegalStateException("file source is closed");
    }
    return handle;
  }

  private static native void setUpKvikioNative(int mode);
  private static native long openNative(String path);

  private static native long size(long handle);

  private static native boolean isDeviceReadDirect(long handle, long size);

  private static native long deviceRead(long handle, long fileOffset, long length, long dstAddress,
      long stream);

  private static native void close(long handle);
}
