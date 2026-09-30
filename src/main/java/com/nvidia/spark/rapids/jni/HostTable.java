/*
 *  Copyright (c) 2024, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;

/**
 * Represents a cudf Table but in host memory instead of device memory.
 * Table is tracked in native code as a host_table_view.
 */
public class HostTable implements AutoCloseable {

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private long nativeTableView;
  private HostMemoryBuffer hostBuffer;

  /**
   * Copies a device table to a host table asynchronously.
   * NOTE: The caller must synchronize on the stream before examining the data on the host.
   * @param table device table to copy
   * @param stream stream to use for the copy
   * @return host table
   */
  public static HostTable fromTableAsync(Table table, Cuda.Stream stream) {
    long size = bufferSize(table.getNativeView(), stream.getStream());
    long tableHandle = 0;
    HostMemoryBuffer hostBuffer = HostMemoryBuffer.allocate(size);
    try {
      tableHandle = copyFromTableAsync(table.getNativeView(),
          hostBuffer.getAddress(), hostBuffer.getLength(), stream.getStream());
    } catch (Throwable t) {
      try {
        hostBuffer.close();
      } catch (Throwable t2) {
        t.addSuppressed(t2);
      }
      throw t;
    }
    return new HostTable(tableHandle, hostBuffer);
  }

  /**
   * Copies a device table to a host table synchronously.
   * @param table device table to copy
   * @param stream stream to use for the copy
   * @return host table
   */
  public static HostTable fromTable(Table table, Cuda.Stream stream) {
    HostTable hostTable = fromTableAsync(table, stream);
    stream.sync();
    return hostTable;
  }

  /**
   * Copies a device table to a host table synchronously on the default stream.
   * @param table device table to copy
   * @return host table
   */
  public static HostTable fromTable(Table table) {
    return fromTable(table, Cuda.DEFAULT_STREAM);
  }

  private HostTable(long tableHandle, HostMemoryBuffer hostBuffer) {
    this.nativeTableView = tableHandle;
    this.hostBuffer = hostBuffer;
  }

  /**
   * Gets the address of the host_table_view for this host table.
   * NOTE: This is only valid as long as the HostTable instance is valid.
   */
  public long getNativeTableView() {
    return nativeTableView;
  }

  /**
   * Gets the host memory buffer containing the data for this host table.
   */
  public HostMemoryBuffer getHostBuffer() {
    return hostBuffer;
  }

  /**
   * Copies the host table to a device table asynchronously.
   * NOTE: The caller must synchronize on the stream before closing this instance,
   * or the copy could still be in-flight when the host memory is invalidated or reused.
   * @param stream stream to use for the copy
   * @return device table
   */
  public Table toTableAsync(Cuda.Stream stream) {
    long size = hostBuffer.getLength();
    Table table = null;
    try (DeviceMemoryBuffer devBuffer = DeviceMemoryBuffer.allocate(size, stream)) {
      devBuffer.copyFromHostBufferAsync(hostBuffer, stream);
      long hostToDevPtrOffset = devBuffer.getAddress() - hostBuffer.getAddress();
      long[] columnViewHandles = toDeviceColumnViews(nativeTableView, hostToDevPtrOffset);
      ColumnVector[] columns = new ColumnVector[columnViewHandles.length];
      boolean done = false;
      try {
        for (int i = 0; i < columnViewHandles.length; i++) {
          columns[i] = ColumnVector.fromViewWithContiguousAllocation(columnViewHandles[i], devBuffer);
          columnViewHandles[i] = 0;
        }
        table = new Table(columns);
        // Need to synchronize before returning to ensure host copy completed, otherwise caller may
        // free and reuse the host buffer before device copy completes.
        stream.sync();
        done = true;
      } finally {
        // always close columns because Table incremented refcounts
        for (ColumnVector c : columns) {
          if (c != null) {
            c.close();
          }
        }
        if (!done) {
          for (long viewHandle : columnViewHandles) {
            if (viewHandle != 0) {
              freeDeviceColumnView(viewHandle);
            }
          }
        }
      }
    }
    return table;
  }

  /**
   * Copies the host table to a device table synchronously.
   * @param stream stream to use for the copy
   * @return device table
   */
  public Table toTable(Cuda.Stream stream) {
    Table table = toTableAsync(stream);
    stream.sync();
    return table;
  }

  /**
   * Copies the host table to a device table synchronously on the default stream.
   * @return device table
   */
  public Table toTable() {
    return toTable(Cuda.DEFAULT_STREAM);
  }

  @Override
  public void close() {
    try {
      freeHostTable(nativeTableView);
    } finally {
      nativeTableView = 0;
      hostBuffer.close();
      hostBuffer = null;
    }
  }

  private static native long bufferSize(long tableHandle, long stream);

  private static native long copyFromTableAsync(long tableHandle, long hostAddress, long hostSize,
                                                long stream);

  private static native long[] toDeviceColumnViews(long tableHandle, long hostToDevPtrOffset);

  private static native void freeDeviceColumnView(long columnHandle);

  private static native void freeHostTable(long tableHandle);
}
