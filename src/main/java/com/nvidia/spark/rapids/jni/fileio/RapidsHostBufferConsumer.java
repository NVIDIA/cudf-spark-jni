/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.jni.fileio;

import ai.rapids.cudf.HostMemoryBuffer;

import java.io.IOException;

/**
 * Consumes encoded file data directly from cuDF host buffers.
 *
 * <p>Ownership of {@code buffer} transfers to the consumer as soon as
 * {@link #handleBuffer(HostMemoryBuffer, long)} is called, including when that method throws.
 * The caller must not close or access the buffer after the call.</p>
 */
public interface RapidsHostBufferConsumer extends AutoCloseable {
  /**
   * Accept ownership of an encoded host buffer.
   *
   * @param buffer buffer whose ownership transfers to this consumer
   * @param len number of valid bytes starting at offset zero
   * @throws IOException if the data cannot be accepted
   */
  void handleBuffer(HostMemoryBuffer buffer, long len) throws IOException;

  /** Complete the output and release all retained buffers. */
  @Override
  void close() throws IOException;

  /** Abort incomplete output and release all retained buffers. */
  void abort() throws IOException;

  /** @return number of encoded bytes accepted by this consumer */
  default long getBytesWritten() {
    return 0L;
  }

  /** @return nanoseconds spent waiting for upload capacity */
  default long getWaitTimeNanos() {
    return 0L;
  }

  /** @return cumulative nanoseconds spent in storage requests */
  default long getRequestTimeNanos() {
    return 0L;
  }

  /** @return number of request-body replay attempts */
  default long getRetryCount() {
    return 0L;
  }

  /** @return number of storage requests that ultimately failed */
  default long getFailureCount() {
    return 0L;
  }

  /** @return peak bytes retained from host buffers */
  default long getPeakRetainedBytes() {
    return 0L;
  }
}
