/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

import ai.rapids.cudf.NativeDepsLoader;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;

/** Profiler that collects CUDA and NVTX events for the current process. */
public class Profiler {
  private static final long DEFAULT_WRITE_BUFFER_SIZE = 1024 * 1024;
  private static final int DEFAULT_FLUSH_PERIOD_MILLIS = 0;
  private static final boolean DEFAULT_ALLOC_ASYNC_CAPTURING = false;
  private static DataWriter writer = null;

  /**
   * Initialize the profiler in a standby state. The start method must be called after this
   * to start collecting profiling data.
   * @param config profiler configuration
   */
  public static void init(DataWriter w, Config config) {
    if (writer == null) {
      File libPath;
      try {
        libPath = NativeDepsLoader.loadNativeDep("profilerjni", true);
      } catch (IOException e) {
        throw new RuntimeException("Error loading profiler library", e);
      }
      nativeInit(libPath.getAbsolutePath(), w, config.writeBufferSize, config.flushPeriodMillis,
          config.allocAsyncCapturing);
      writer = w;
    } else {
      throw new IllegalStateException("Already initialized");
    }
  }

  /**
   * Deprecated. Use init(Config) instead.
   * Initialize the profiler in a standby state. The start method must be called after this
   * to start collecting profiling data.
   * @param w data writer for writing profiling data
   * @param writeBufferSize size of host memory buffer to use for collecting profiling data.
   *                        Recommended to be between 1-8 MB in size to balance callback
   *                        overhead with latency.
   * @param flushPeriodMillis time period in milliseconds to explicitly flush collected
   *                          profiling data to the writer. A value <= 0 will disable explicit
   *                          flushing.
   */
  public static void init(DataWriter w, long writeBufferSize, int flushPeriodMillis) {
    Config config = new Config.Builder()
        .withWriteBufferSize(writeBufferSize)
        .withFlushPeriodMillis(flushPeriodMillis)
        .build();
    init(w, config);
  }

  /**
   * Shutdown the profiling session. Flushes collected profiling data to the writer and
   * closes the writer.
   */
  public static void shutdown() {
    if (writer != null) {
      nativeShutdown();
      try {
        writer.close();
      } catch (Exception e) {
        throw new RuntimeException("Error closing writer", e);
      } finally {
        writer = null;
      }
    }
  }

  /**
   * Start collecting profiling data. Safe to call if profiling data is already being collected.
   */
  public static void start() {
    if (writer != null) {
      nativeStart();
    } else {
      throw new IllegalStateException("Profiler not initialized");
    }
  }

  /**
   * Stop collecting profiling data. Safe to call if the profiler is initialized but not
   * actively collecting data.
   */
  public static void stop() {
    if (writer != null) {
      nativeStop();
    } else {
      throw new IllegalStateException("Profiler not initialized");
    }
  }

  private static native void nativeInit(String libPath, DataWriter writer,
                                        long writeBufferSize, int flushPeriodMillis,
                                        boolean allocAsyncCapturing);

  private static native void nativeStart();

  private static native void nativeStop();

  private static native void nativeShutdown();

  /** Interface for profiler data writers */
  public interface DataWriter extends AutoCloseable {
    /**
     * Called by the profiler to write a block of profiling data. Profiling data is written
     * in a size-prefixed flatbuffer format. See profiler.fbs for the schema.
     * @param data profiling data to be written
     */
    void write(ByteBuffer data);
  }

  /** Profiler configuration class. **/
  public static class Config {
    private final long writeBufferSize;
    private final int flushPeriodMillis;
    private final boolean allocAsyncCapturing;

    Config(Builder builder) {
      this.writeBufferSize = builder.writeBufferSize;
      this.flushPeriodMillis = builder.flushPeriodMillis;
      this.allocAsyncCapturing = builder.allocAsyncCapturing;
    }

    /** Builder interface for profiler configuration. **/
    public static class Builder {
      private long writeBufferSize = DEFAULT_WRITE_BUFFER_SIZE;
      private int flushPeriodMillis = DEFAULT_FLUSH_PERIOD_MILLIS;
      private boolean allocAsyncCapturing = DEFAULT_ALLOC_ASYNC_CAPTURING;

      /**
       * Configure the size of the host memory buffer used for collecting profiling data.
       * Recommended to be between 1 to 8 MB in size to balance callback overhead with
       * latency.
       * @param writeBufferSize size of buffer in bytes
       */
      public Builder withWriteBufferSize(long writeBufferSize) {
        this.writeBufferSize = writeBufferSize;
        return this;
      }

      /**
       * Configure the time period to explicitly flush collected profiling data to the writer.
       * @param flushPeriodMillis time period in milliseconds. A value <= 0 will disable explicit
       *                          flushing.
       */
      public Builder withFlushPeriodMillis(int flushPeriodMillis) {
        this.flushPeriodMillis = flushPeriodMillis;
        return this;
      }

      /**
       * Configure whether async allocation and free events are captured by the profiler.
       * @param allocAsyncCapturing true if async allocation and free events should be captured,
       *                            false otherwise.
       */
      public Builder withAllocAsyncCapturing(boolean allocAsyncCapturing) {
        this.allocAsyncCapturing = allocAsyncCapturing;
        return this;
      }

      /** Build a profiler configuration object. */
      public Config build() {
        return new Config(this);
      }
    }
  }
}
