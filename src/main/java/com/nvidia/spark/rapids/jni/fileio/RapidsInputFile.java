/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

import java.io.EOFException;
import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;

/**
 * Represents an input file that can be read from.
 * <br/>
 * The implementation of this interface should be thread-safe.
 */
public interface RapidsInputFile {
  int DEFAULT_READ_VECTORED_COPY_BUFFER_SIZE = 8 * 1024 * 1024;

  /**
   * Get the path of this input file.
   * @return the file path string
   */
  default String path() {
    throw new UnsupportedOperationException("path is not supported");
  }

  /**
   * Get the length of the file in bytes.
   * @return the length of the file in bytes
   * @throws IOException if an I/O error occurs while getting the length
   */
  long getLength() throws IOException;

  /**
   * Get the last modification time of the file in milliseconds since epoch.
   * @return an OptionalLong containing the last modification time, or empty if not available
   * @throws IOException if an I/O error occurs while getting the modification time
   */
  default OptionalLong getLastModificationTime() throws IOException {
    return OptionalLong.empty();
  }

  /**
   * Reads data from the input file into the provided output buffer using vectored read.
   *
   * <p>The output buffer will not be closed by this method. It is the caller's responsibility
   * to close it.</p>
   *
   * @param output the buffer to read data into
   * @param copyRanges a list of copy ranges specifying the input offsets, lengths, and output
   *                   offsets
   * @throws IOException if an I/O error occurs during reading
   */
  default void readVectored(HostMemoryBuffer output, List<CopyRange> copyRanges)
      throws IOException {
    Objects.requireNonNull(output, "output can't be null");
    Objects.requireNonNull(copyRanges, "copyRanges can't be null");
    if (copyRanges.isEmpty()) {
      return;
    }
    RapidsInputFileUtils.readVectoredUsingCopyBuffer(
        this, output, copyRanges, DEFAULT_READ_VECTORED_COPY_BUFFER_SIZE);
  }

  /**
   * Reads data from {@code inputFile} into {@code output} using a caller-supplied copy buffer.
   *
   * @param inputFile the file to read from
   * @param output the destination buffer
   * @param copyRanges input ranges and output offsets
   * @param copyBuffer reusable copy buffer
   * @throws IOException if an I/O error occurs during reading
   */
  static void readVectoredUsingCopyBuffer(
      RapidsInputFile inputFile,
      HostMemoryBuffer output,
      List<CopyRange> copyRanges,
      byte[] copyBuffer) throws IOException {
    RapidsInputFileUtils.readVectoredUsingCopyBuffer(inputFile, output, copyRanges, copyBuffer);
  }

  /**
   * Reads the last {@code length} bytes of the input file into the provided output buffer.
   *
   * <p>The output buffer will not be closed by this method. It is the caller's responsibility
   * to close it.</p>
   *
   * <p>Data is written starting at offset 0 of the output buffer. The output buffer must have
   * capacity for at least {@code length} bytes.</p>
   *
   * <p>This default implementation computes the tail offset using {@link #getLength()} before
   * calling {@link #open()}. If the file is truncated between those calls, the read may still
   * fail with an I/O error.</p>
   *
   * @param length the number of bytes to read from the tail
   * @param output the buffer to read data into
   * @throws IOException if an I/O error occurs during reading
   */
  default void readTail(long length, HostMemoryBuffer output) throws IOException {
    Objects.requireNonNull(output, "output can't be null");
    if (length < 0) {
      throw new IllegalArgumentException("length must be non-negative");
    }
    if (length == 0) {
      return;
    }

    long fileLength = getLength();
    if (length > fileLength) {
      throw new EOFException(
          "Cannot read tail of length " + length + " from file of length " + fileLength);
    }

    try (SeekableInputStream input = open()) {
      input.seek(fileLength - length);
      output.copyFromStream(0, input, length);
    }
  }

  /**
   * Open the file for reading.
   * @return a {@link SeekableInputStream } to read from the file
   * @throws IOException if an I/O error occurs while opening the file
   */
  SeekableInputStream open() throws IOException;

  /**
   * Describes a range of bytes to copy from the input file into an output buffer.
   */
  final class CopyRange {
    private final long inputOffset;
    private final long length;
    private final long outputOffset;

    public CopyRange(long inputOffset, long length, long outputOffset) {
      if (inputOffset < 0) {
        throw new IllegalArgumentException("inputOffset must be non-negative");
      }
      if (length <= 0) {
        throw new IllegalArgumentException("length must be positive");
      }
      if (outputOffset < 0) {
        throw new IllegalArgumentException("outputOffset must be non-negative");
      }
      this.inputOffset = inputOffset;
      this.length = length;
      this.outputOffset = outputOffset;
    }

    public long getInputOffset() {
      return inputOffset;
    }

    public long getLength() {
      return length;
    }

    public long getOutputOffset() {
      return outputOffset;
    }
  }
}
