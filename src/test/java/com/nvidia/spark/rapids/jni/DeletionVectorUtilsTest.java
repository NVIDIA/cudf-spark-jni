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

import ai.rapids.cudf.DeletionVector;
import ai.rapids.cudf.HostMemoryBuffer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class DeletionVectorUtilsTest {
  @Test
  void testEmptyDeletionVector() {
    try (HostMemoryBuffer serializedBitmap = HostMemoryBuffer.allocate(8)) {
      serializedBitmap.setLong(0, 0);
      DeletionVector.DeletionVectorInfo info = new DeletionVector.DeletionVectorInfo(
          serializedBitmap, false, new long[] {1000}, new int[] {100});
      DeletionVector.DeletionVectorInfo retentionInfo = new DeletionVector.DeletionVectorInfo(
          serializedBitmap, true, new long[] {1000}, new int[] {100});

      assertEquals(0, DeletionVectorUtils.computeNumDeletedRows(info, 25));
      assertEquals(100, DeletionVectorUtils.computeNumDeletedRows(retentionInfo, 25));
    }
  }

  @Test
  void testInvalidChunkSize() {
    try (HostMemoryBuffer serializedBitmap = HostMemoryBuffer.allocate(8)) {
      serializedBitmap.setLong(0, 0);
      DeletionVector.DeletionVectorInfo info = new DeletionVector.DeletionVectorInfo(
          serializedBitmap, false, new long[] {0}, new int[] {1});

      assertThrows(IllegalArgumentException.class,
          () -> DeletionVectorUtils.computeNumDeletedRows(info, 0));
    }
  }
}
