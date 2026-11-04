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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.CompressionType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.ORCWriterOptions;
import ai.rapids.cudf.ParquetChunkedReader;
import ai.rapids.cudf.ParquetWriterOptions;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.TableWriter;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.File;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class NvcompCompressionTest {
  private static final int ROW_COUNT = 1_000_000;
  private static final long CHUNK_SIZE_BYTES = 240_000;

  @TempDir
  Path tempDir;

  @ParameterizedTest
  @ValueSource(strings = {"orc", "parquet"})
  void zstdCompressedRead(String format) {
    // Keep enough nonconstant data to exercise nvCOMP decompression, rather than an
    // uncompressed small-file path. Its pinned pool must share a compatible CCCL ABI.
    int[] values = new int[ROW_COUNT];
    for (int i = 0; i < values.length; ++i) {
      values[i] = (i * 123457) ^ (i / 7);
    }
    File file = tempDir.resolve("compressed." + format).toFile();
    try (ColumnVector column = ColumnVector.fromInts(values);
         Table expected = new Table(column)) {
      if (format.equals("orc")) {
        try (TableWriter writer = Table.writeORCChunked(ORCWriterOptions.builder()
            .withColumns(false, "value").withCompressionType(CompressionType.ZSTD).build(),
            file)) {
          writer.write(expected);
        }
        try (Table actual = Table.readORC(file)) {
          assertEquals(values.length, assertValues(values, 0, actual));
        }
      } else {
        try (TableWriter writer = Table.writeParquetChunked(ParquetWriterOptions.builder()
            .withColumns(false, "value").withCompressionType(CompressionType.ZSTD).build(),
            file)) {
          writer.write(expected);
        }
        int rows = 0;
        int chunks = 0;
        try (ParquetChunkedReader reader = new ParquetChunkedReader(CHUNK_SIZE_BYTES, file)) {
          while (reader.hasNext()) {
            try (Table chunk = reader.readChunk()) {
              rows = assertValues(values, rows, chunk);
              ++chunks;
            }
          }
        }
        assertEquals(values.length, rows);
        assertTrue(chunks > 1);
      }
    }
  }

  private static int assertValues(int[] values, int offset, Table table) {
    assertEquals(1, table.getNumberOfColumns());
    try (HostColumnVector actual = table.getColumn(0).copyToHost()) {
      assertTrue(offset + actual.getRowCount() <= values.length);
      for (int i = 0; i < actual.getRowCount(); ++i) {
        assertEquals(values[offset++], actual.getInt(i));
      }
    }
    return offset;
  }
}
