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

import java.io.EOFException;
import java.io.IOException;
import java.util.List;
import java.util.Objects;

final class RapidsInputFileUtils {
  private RapidsInputFileUtils() {
  }

  static void readVectoredUsingCopyBuffer(
      RapidsInputFile inputFile,
      HostMemoryBuffer output,
      List<RapidsInputFile.CopyRange> copyRanges,
      int copyBufferSize) throws IOException {
    validateReadVectoredArgs(inputFile, output, copyRanges, copyBufferSize);
    if (copyRanges.isEmpty()) {
      return;
    }

    byte[] copyBuffer = new byte[getCopyBufferAllocationSize(copyRanges, copyBufferSize)];
    try (SeekableInputStream input = inputFile.open()) {
      readVectored(input, output, copyRanges, copyBuffer);
    }
  }

  static void readVectoredUsingCopyBuffer(
      RapidsInputFile inputFile,
      HostMemoryBuffer output,
      List<RapidsInputFile.CopyRange> copyRanges,
      byte[] copyBuffer) throws IOException {
    validateReadVectoredArgs(inputFile, output, copyRanges, copyBuffer);
    if (copyRanges.isEmpty()) {
      return;
    }

    try (SeekableInputStream input = inputFile.open()) {
      readVectored(input, output, copyRanges, copyBuffer);
    }
  }

  private static void readVectored(
      SeekableInputStream input,
      HostMemoryBuffer output,
      List<RapidsInputFile.CopyRange> copyRanges,
      byte[] copyBuffer) throws IOException {
    for (RapidsInputFile.CopyRange copyRange : copyRanges) {
      if (input.getPos() != copyRange.getInputOffset()) {
        input.seek(copyRange.getInputOffset());
      }
      long outputOffset = copyRange.getOutputOffset();
      long bytesLeft = copyRange.getLength();
      while (bytesLeft > 0) {
        int readLength = (int) Math.min(bytesLeft, copyBuffer.length);
        readFully(input, copyBuffer, readLength);
        output.setBytes(outputOffset, copyBuffer, 0, readLength);
        outputOffset += readLength;
        bytesLeft -= readLength;
      }
    }
  }

  private static void validateReadVectoredArgs(
      RapidsInputFile inputFile,
      HostMemoryBuffer output,
      List<RapidsInputFile.CopyRange> copyRanges,
      int copyBufferSize) {
    Objects.requireNonNull(inputFile, "inputFile can't be null");
    if (copyBufferSize <= 0) {
      throw new IllegalArgumentException("copyBufferSize must be positive");
    }
    validateReadVectoredArgs(inputFile, output, copyRanges);
  }

  private static void validateReadVectoredArgs(
      RapidsInputFile inputFile,
      HostMemoryBuffer output,
      List<RapidsInputFile.CopyRange> copyRanges,
      byte[] copyBuffer) {
    Objects.requireNonNull(inputFile, "inputFile can't be null");
    Objects.requireNonNull(copyBuffer, "copyBuffer can't be null");
    if (copyBuffer.length == 0) {
      throw new IllegalArgumentException("copyBuffer must not be empty");
    }
    validateReadVectoredArgs(inputFile, output, copyRanges);
  }

  private static void validateReadVectoredArgs(
      RapidsInputFile inputFile,
      HostMemoryBuffer output,
      List<RapidsInputFile.CopyRange> copyRanges) {
    Objects.requireNonNull(inputFile, "inputFile can't be null");
    Objects.requireNonNull(output, "output can't be null");
    Objects.requireNonNull(copyRanges, "copyRanges can't be null");

    long outputLength = output.getLength();
    for (RapidsInputFile.CopyRange copyRange : copyRanges) {
      Objects.requireNonNull(copyRange, "copyRange can't be null");
      long end = copyRange.getOutputOffset() + copyRange.getLength();
      if (end < 0 || end > outputLength) {
        throw new IllegalArgumentException(
            "Output buffer length " + outputLength + " is smaller than requested end " + end);
      }
    }
  }

  private static int getCopyBufferAllocationSize(
      List<RapidsInputFile.CopyRange> copyRanges,
      int copyBufferSize) {
    long maxRangeLength = 0;
    for (RapidsInputFile.CopyRange copyRange : copyRanges) {
      maxRangeLength = Math.max(maxRangeLength, copyRange.getLength());
      if (maxRangeLength >= copyBufferSize) {
        return copyBufferSize;
      }
    }
    return (int) Math.min(copyBufferSize, maxRangeLength);
  }

  private static void readFully(SeekableInputStream input, byte[] copyBuffer, int readLength)
      throws IOException {
    int totalRead = 0;
    while (totalRead < readLength) {
      int amountRead = input.read(copyBuffer, totalRead, readLength - totalRead);
      if (amountRead < 0) {
        throw new EOFException(
            "Unexpected end of stream, expected " + (readLength - totalRead) + " more bytes");
      }
      if (amountRead == 0) {
        throw new IOException(
            "No progress reading stream, expected " + (readLength - totalRead) + " more bytes");
      }
      totalRead += amountRead;
    }
  }
}
