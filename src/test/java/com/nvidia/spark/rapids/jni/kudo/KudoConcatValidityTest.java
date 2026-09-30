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

package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.HostMemoryBuffer;
import com.nvidia.spark.rapids.jni.Arms;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class KudoConcatValidityTest {
    private static final int SEED = 7863832;

    private static Random getRandom() {
        return new Random(SEED);
    }

    private static HostMemoryBuffer fillValidityBuffer(int startRow, boolean[] values) {
      SliceInfo sliceInfo = new SliceInfo(startRow, values.length);
      int bufferSize = (int) KudoSerializer.padForHostAlignment(sliceInfo.getValidityBufferInfo().getBufferLength());
      int startBit = sliceInfo.getValidityBufferInfo().getBeginBit();
      return Arms.closeIfException(HostMemoryBuffer.allocate(bufferSize), buffer -> {
        for (int i = 0; i < bufferSize; i++) {
          buffer.setByte(i, (byte) 0x00);
        }
        for (int i = 0; i < values.length; i++) {
          if (values[i]) {
            int index = startBit + i;
            int byteIdx = index / 8;
            int bitIdx = index % 8;
            byte b = buffer.getByte(byteIdx);
            b |= (byte) (1 << bitIdx);
            buffer.setByte(byteIdx, b);
          }
        }
        return buffer;
      });
    }

    private static boolean[] getValidityBuffer(HostMemoryBuffer buffer, int len) {
      boolean[] arr = new boolean[len];
      for (int i = 0; i < len; i++) {
        int byteIdx = i / 8;
        int bitIdx = i % 8;
        arr[i] = (buffer.getByte(byteIdx) & (1 << bitIdx) & 0xFF) != 0;
      }
      return arr;
    }

    //When srcBitIdx < destBitIdx, srcIntBufLen = 1
    @Test
    public void testConcatValidityCase1() {
        Random random = getRandom();
        int accuArrLen = 0;
        // Be careful with startRow, they are carefully designed to cover all test cases.
        try (HostMemoryBuffer dest = HostMemoryBuffer.allocate(4096)) {
            KudoTableMerger.ValidityBufferMerger merger = new KudoTableMerger.ValidityBufferMerger(dest, 0, new int[64], new int[64]);
            // When srcBitIdx == destBitIdx
            ValidityConcatArray arr1 = new ValidityConcatArray(0, 29, random, "Array 1", accuArrLen);
            arr1.appendToDest(merger);
            accuArrLen += arr1.array.length;

            // Now destBitIdx = 29

            // srcBitIdx < destBitIdx, srcIntBufLen = 1
            ValidityConcatArray arr2 = new ValidityConcatArray(7, 27, random, "Array 2", accuArrLen);
            arr2.appendToDest(merger);
            accuArrLen += arr2.array.length;

            boolean[] result = getValidityBuffer(dest, accuArrLen);
            arr1.verifyData(result);
            arr2.verifyData(result);
        }
    }

    //When srcBitIdx < destBitIdx, srcIntBufLen > 1, last leftRowCount < 0
    @Test
    public void testConcatValidityCase2() {
        Random random = getRandom();
        int accuArrLen = 0;
        // Be careful with startRow, they are carefully designed to cover all test cases.
        try (HostMemoryBuffer dest = HostMemoryBuffer.allocate(4096)) {
            KudoTableMerger.ValidityBufferMerger merger = new KudoTableMerger.ValidityBufferMerger(dest, 0, new int[64], new int[64]);
            // When srcBitIdx == destBitIdx
            ValidityConcatArray arr1 = new ValidityConcatArray(0, 29, random, "Array 1", accuArrLen);
            arr1.appendToDest(merger);
            accuArrLen += arr1.array.length;

            // Now destBitIdx = 29

            // srcBitIdx < destBitIdx, srcIntBufLen > 1
            ValidityConcatArray arr2 = new ValidityConcatArray(7, 127, random, "Array 2", accuArrLen);
            arr2.appendToDest(merger);
            accuArrLen += arr2.array.length;

            boolean[] result = getValidityBuffer(dest, accuArrLen);
            arr1.verifyData(result);
            arr2.verifyData(result);
        }
    }

    //When srcBitIdx < destBitIdx, srcIntBufLen > 1, last leftRowCount > 0
    @Test
    public void testConcatValidityCase3() {
        Random random = getRandom();
        int accuArrLen = 0;
        // Be careful with startRow, they are carefully designed to cover all test cases.
        try (HostMemoryBuffer dest = HostMemoryBuffer.allocate(4096)) {
            KudoTableMerger.ValidityBufferMerger merger = new KudoTableMerger.ValidityBufferMerger(dest, 0, new int[64], new int[64]);
            // When srcBitIdx == destBitIdx
            ValidityConcatArray arr1 = new ValidityConcatArray(0, 29, random, "Array 1", accuArrLen);
            arr1.appendToDest(merger);
            accuArrLen += arr1.array.length;

            // Now destBitIdx = 29

            // srcBitIdx < destBitIdx, srcIntBufLen > 1
            ValidityConcatArray arr2 = new ValidityConcatArray(7, 133, random, "Array 2", accuArrLen);
            arr2.appendToDest(merger);
            accuArrLen += arr2.array.length;

            boolean[] result = getValidityBuffer(dest, accuArrLen);
            arr1.verifyData(result);
            arr2.verifyData(result);
        }
    }

    // When srcBitIdx == destBitIdx, srcIntBufLen == 1
    @Test
    public void testConcatValidityCase4() {
        Random random = getRandom();
        int accuArrLen = 0;
        // Be careful with startRow, they are carefully designed to cover all test cases.
        try (HostMemoryBuffer dest = HostMemoryBuffer.allocate(4096)) {
            KudoTableMerger.ValidityBufferMerger merger = new KudoTableMerger.ValidityBufferMerger(dest, 0, new int[64], new int[64]);
            // When srcBitIdx == destBitIdx
            ValidityConcatArray arr1 = new ValidityConcatArray(0, 29, random, "Array 1", accuArrLen);
            arr1.appendToDest(merger);
            accuArrLen += arr1.array.length;


            boolean[] result = getValidityBuffer(dest, accuArrLen);
            arr1.verifyData(result);
        }
    }

    // When srcBitIdx == destBitIdx, srcIntBufLen > 1
    @Test
    public void testConcatValidityCase5() {
        Random random = getRandom();
        int accuArrLen = 0;
        // Be careful with startRow, they are carefully designed to cover all test cases.
        try (HostMemoryBuffer dest = HostMemoryBuffer.allocate(4096)) {
            KudoTableMerger.ValidityBufferMerger merger = new KudoTableMerger.ValidityBufferMerger(dest, 0, new int[64], new int[64]);
            // When srcBitIdx == destBitIdx
            ValidityConcatArray arr1 = new ValidityConcatArray(0, 29, random, "Array 1", accuArrLen);
            arr1.appendToDest(merger);
            accuArrLen += arr1.array.length;

            // destBitIdx = 29
            ValidityConcatArray arr2 = new ValidityConcatArray(29, 105, random, "Array 2",
                    accuArrLen);
            arr2.appendToDest(merger);
            accuArrLen += arr2.array.length;


            boolean[] result = getValidityBuffer(dest, accuArrLen);
            arr1.verifyData(result);
            arr2.verifyData(result);
        }
    }

    // When srcBitIdx > destBitIdx, srcIntBufLen = 1
    @Test
    public void testConcatValidityCase6() {
        Random random = getRandom();
        int accuArrLen = 0;
        // Be careful with startRow, they are carefully designed to cover all test cases.
        try (HostMemoryBuffer dest = HostMemoryBuffer.allocate(4096)) {
            KudoTableMerger.ValidityBufferMerger merger = new KudoTableMerger.ValidityBufferMerger(dest, 0, new int[64], new int[64]);
            // When srcBitIdx == destBitIdx
            ValidityConcatArray arr1 = new ValidityConcatArray(0, 14, random, "Array 1",
                    accuArrLen);
            arr1.appendToDest(merger);
            accuArrLen += arr1.array.length;

            // destBitIdx = 14
            ValidityConcatArray arr2 = new ValidityConcatArray(17, 9, random, "Array 2",
                    accuArrLen);
            arr2.appendToDest(merger);
            accuArrLen += arr2.array.length;


            boolean[] result = getValidityBuffer(dest, accuArrLen);
            arr1.verifyData(result);
            arr2.verifyData(result);
        }
    }

    // When srcBitIdx > destBitIdx, srcIntBufLen > 1, last leftRowCount > 0
    @Test
    public void testConcatValidityCase7() {
        Random random = getRandom();
        int accuArrLen = 0;
        // Be careful with startRow, they are carefully designed to cover all test cases.
        try (HostMemoryBuffer dest = HostMemoryBuffer.allocate(4096)) {
            KudoTableMerger.ValidityBufferMerger merger = new KudoTableMerger.ValidityBufferMerger(dest, 0, new int[64], new int[64]);
            // When srcBitIdx == destBitIdx
            ValidityConcatArray arr1 = new ValidityConcatArray(0, 14, random, "Array 1",
                    accuArrLen);
            arr1.appendToDest(merger);
            accuArrLen += arr1.array.length;

            // destBitIdx = 14
            ValidityConcatArray arr2 = new ValidityConcatArray(17, 87, random, "Array 2",
                    accuArrLen);
            arr2.appendToDest(merger);
            accuArrLen += arr2.array.length;


            boolean[] result = getValidityBuffer(dest, accuArrLen);
            arr1.verifyData(result);
            arr2.verifyData(result);
        }
    }

    // When srcBitIdx > destBitIdx, srcIntBufLen > 1, last leftRowCount < 0
    @Test
    public void testConcatValidityCase8() {
        Random random = getRandom();
        int accuArrLen = 0;
        // Be careful with startRow, they are carefully designed to cover all test cases.
        try (HostMemoryBuffer dest = HostMemoryBuffer.allocate(4096)) {
            KudoTableMerger.ValidityBufferMerger merger = new KudoTableMerger.ValidityBufferMerger(dest, 0, new int[64], new int[64]);
            // When srcBitIdx == destBitIdx
            ValidityConcatArray arr1 = new ValidityConcatArray(0, 8, random, "Array 1",
                    accuArrLen);
            arr1.appendToDest(merger);
            accuArrLen += arr1.array.length;

            // destBitIdx = 8
            ValidityConcatArray arr2 = new ValidityConcatArray(12, 85, random, "Array 2",
                    accuArrLen);
            arr2.appendToDest(merger);
            accuArrLen += arr2.array.length;


            boolean[] result = getValidityBuffer(dest, accuArrLen);
            arr1.verifyData(result);
            arr2.verifyData(result);
        }
    }

    @Test
    public void testConcatValidity() {
        Random random = getRandom();
        int accuArrLen = 0;
        // Be careful with startRow, they are carefully designed to cover all test cases.
        try (HostMemoryBuffer dest = HostMemoryBuffer.allocate(4096)) {
            KudoTableMerger.ValidityBufferMerger merger = new KudoTableMerger.ValidityBufferMerger(dest, 0, new int[64], new int[64]);
            // Second case when srcBitIdx > destBitIdx
            ValidityConcatArray arr1 = new ValidityConcatArray(3, 129, random, "Array 1", accuArrLen);
            arr1.appendToDest(merger);
            accuArrLen += arr1.array.length;

            // Second case when srcBitIdx > destBitIdx
            ValidityConcatArray arr2 = new ValidityConcatArray(7, 79, random, "Array 2", accuArrLen);
            arr2.appendToDest(merger);
            accuArrLen += arr2.array.length;


            // Append all validity
            ValidityConcatArray arr3 = new ValidityConcatArray(-1, 129, null, "Array 3", accuArrLen);
            arr3.appendToDest(merger);
            accuArrLen += arr3.array.length;

            // First case when srcBitIdx < destBitIdx
            ValidityConcatArray arr4 = new ValidityConcatArray(3, 70, random, "Array 4", accuArrLen);
            arr4.appendToDest(merger);
            accuArrLen += arr4.array.length;

            // First case when srcBitIdx < destBitIdx
            ValidityConcatArray arr5 = new ValidityConcatArray(3, 62, random, "Array 5", accuArrLen);
            arr5.appendToDest(merger);
            accuArrLen += arr5.array.length;

            // Third cas when srcBitIdx == destBitIdx
            ValidityConcatArray arr6 = new ValidityConcatArray(21, 79, random, "Array 5", accuArrLen);
            arr6.appendToDest(merger);
            accuArrLen += arr6.array.length;


            boolean[] result = getValidityBuffer(dest, accuArrLen);
            arr1.verifyData(result);
            arr2.verifyData(result);
            arr3.verifyData(result);
            arr4.verifyData(result);
            arr5.verifyData(result);
        }
    }

    @Test
    public void testConcatValidityWithEmpty() {
      Random random = getRandom();
      int accuArrLen = 0;
      // Be careful with startRow, they are carefully designed to cover all test cases.
      try (HostMemoryBuffer dest = HostMemoryBuffer.allocate(64)) {
        KudoTableMerger.ValidityBufferMerger merger = new KudoTableMerger.ValidityBufferMerger(dest, 0, new int[64], new int[64]);

        ValidityConcatArray arr1 = new ValidityConcatArray(3, 512, random, "Array 1", accuArrLen);
        arr1.appendToDest(merger);
        accuArrLen += arr1.array.length;


        // Should not throw
        merger.appendAllValid(0);

        try (HostMemoryBuffer src = HostMemoryBuffer.allocate(32)) {
          // Should not throw
          merger.copyValidityBuffer(src, 0, new SliceInfo(0, 0));
        }

        boolean[] result = getValidityBuffer(dest, accuArrLen);
        arr1.verifyData(result);
      }
    }



    private static class ValidityConcatArray {
      private final int startRow;
      private final int nullCount;
      private final String name;
      private final boolean[] array;
      private final boolean allValid;
      private final int resultStart;

      public ValidityConcatArray(int startRow, int numRow, Random random, String name,
                                 int resultStart) {
        this.startRow = startRow;
        this.array = new boolean[numRow];
        this.resultStart = resultStart;
        if (random == null) {
          this.allValid = true;
          Arrays.fill(array, true);
          this.nullCount = 0;
        } else {
          this.allValid = false;
          int nullCount = 0;
          for (int i = 0; i < numRow; i++) {
            array[i] = random.nextBoolean();
            if (!array[i]) {
              nullCount++;
            }
          }
          this.nullCount = nullCount;
        }

        this.name = name;
      }

      public void appendToDest(KudoTableMerger.ValidityBufferMerger merger) {
        assertEquals(this.resultStart, merger.getTotalRowCount(), name + " result start not match");
        if (allValid) {
          merger.appendAllValid(this.array.length);
        } else {
          try (HostMemoryBuffer src = fillValidityBuffer(this.startRow, array)) {
            int nullCount = merger.copyValidityBuffer(src, 0, new SliceInfo(this.startRow, array.length));
            assertEquals(this.nullCount, nullCount, name + " null count not match");
          }
        }
      }

      public void verifyData(boolean[] result) {
        for (int i = 0; i < array.length; i++) {
          int index = i;
          assertEquals(this.array[i], result[this.resultStart + i], () -> name + " index " + index + " value not match");
        }
      }
    }
}
