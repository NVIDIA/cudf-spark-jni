/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;

public class Aggregation64UtilsTest {
  @Test
  public void extractInt32Chunks() {
    try (ColumnVector cv = ColumnVector.fromBoxedLongs(null,
        0x7FFFFFFFFFFFFFFFL,
        0L,
        0x123456789abcdef0L,
        0x8000000000000000L,
        null);
         ColumnVector chunk1 = Aggregation64Utils.extractInt32Chunk(cv, DType.UINT32, 0);
         ColumnVector chunk2 = Aggregation64Utils.extractInt32Chunk(cv, DType.INT32, 1);
         Table actualChunks = new Table(chunk1, chunk2);
         ColumnVector expectedChunk1 = ColumnVector.fromBoxedUnsignedInts(
             null, 0xFFFFFFFF, 0, 0x9abcdef0, 0, null);
         ColumnVector expectedChunk2 = ColumnVector.fromBoxedInts(
             null, 0x7FFFFFFF, 0, 0x12345678, 0x80000000, null);
         Table expectedChunks = new Table(expectedChunk1, expectedChunk2)) {
      AssertUtils.assertTablesAreEqual(expectedChunks, actualChunks);
    }
  }

  @Test
  public void testCombineInt64SumChunks() {
    try (ColumnVector chunks0 = ColumnVector.fromBoxedUnsignedLongs(
        null, 0L, 1L, 0L, 0L, 0x12345678L, 0x123456789L, 0x1234567812345678L, 0xfedcba9876543210L);
         ColumnVector chunks1 = ColumnVector.fromBoxedLongs(
             null, 0L, 0xFFFFFFFFL, 0x100000000L, 0x80000000L, 0x55667788L, 0x01234567L, 0x66554434L, -0x42042043L);
         Table chunksTable = new Table(chunks0, chunks1);
         Table actual = Aggregation64Utils.combineInt64SumChunks(chunksTable, DType.INT64);
         ColumnVector expectedOverflows = ColumnVector.fromBoxedBooleans(
             null, false, true, true, true, false, false, false, true);
         ColumnVector expectedValues = ColumnVector.fromBoxedLongs(
             null,
             0L,
             -4294967295L,
             0L,
             -9223372036854775808L,
             6153737367153038968L,
             81985531793467273L,
             8685643420191184504L,
             -4838948107761470960L);
         Table expected = new Table(expectedOverflows, expectedValues)) {
      AssertUtils.assertTablesAreEqual(expected, actual);
    }
  }
}
