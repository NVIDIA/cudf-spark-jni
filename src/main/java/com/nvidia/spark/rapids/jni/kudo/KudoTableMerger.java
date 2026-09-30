/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import ai.rapids.cudf.BufferType;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Schema;
import com.nvidia.spark.rapids.jni.Arms;
import com.nvidia.spark.rapids.jni.schema.SimpleSchemaVisitor;
import com.nvidia.spark.rapids.jni.schema.Visitors;

import java.util.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static com.nvidia.spark.rapids.jni.Preconditions.ensure;
import static com.nvidia.spark.rapids.jni.kudo.ColumnOffsetInfo.INVALID_OFFSET;
import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.*;
import static java.lang.Math.*;
import static java.util.Objects.requireNonNull;

/**
 * This class is used to merge multiple KudoTables into a single contiguous buffer, e.g. {@link KudoHostMergeResult},
 * which could be easily converted to a {@link ai.rapids.cudf.ContiguousTable}.
 */
class KudoTableMerger implements SimpleSchemaVisitor {
  private static final Logger LOG = LoggerFactory.getLogger(KudoTableMerger.class);
  // Number of 1s in a byte
  private static final int[] ONES = new int[1024];
  private static final SliceInfo EMPTY_SLICE = new SliceInfo(0, 0);

  static {
    Arrays.fill(ONES, 0xFFFFFFFF);
  }

  private final KudoTable[] kudoTables;
  private final ColumnOffsetInfo[] columnOffsets;
  private final long[] rowCounts;
  private final HostMemoryBuffer buffer;
  private final ColumnViewInfo[] colViewInfoList;
  private final long[] validityOffsets;
  private final long[] offsetOffsets;
  private final long[] dataOffsets;
  private final Deque<SliceInfo>[] sliceInfos;

  private final SliceInfo[] sliceInfoBuf;
  // A temp buffer for computing
  private final int[] inputBuf;
  private final int[] outputBuf;

  private int curColIdx = 0;
  private KudoHostMergeResult result;

  public KudoTableMerger(KudoTable[] tables, HostMemoryBuffer buffer,
                         ColumnOffsetInfo[] columnOffsets,
                         long[] rowCounts) {
    this.kudoTables = requireNonNull(tables, "tables can't be null");
    requireNonNull(buffer, "buffer can't be null!");
    ensure(columnOffsets != null, "column offsets cannot be null");
    ensure(columnOffsets.length !=0, "column offsets cannot be empty");
    this.columnOffsets = columnOffsets;
    this.rowCounts = rowCounts;
    this.buffer = buffer;
    this.inputBuf = new int[1024];
    this.outputBuf = new int[1024];
    this.colViewInfoList = new ColumnViewInfo[columnOffsets.length];

    this.validityOffsets = new long[kudoTables.length];
    this.offsetOffsets = new long[kudoTables.length];
    this.dataOffsets = new long[kudoTables.length];

    for (int i = 0; i < kudoTables.length; i++) {
      validityOffsets[i] = kudoTables[i].getHeader().startOffsetOf(BufferType.VALIDITY);
      offsetOffsets[i] = kudoTables[i].getHeader().startOffsetOf(BufferType.OFFSET);
      dataOffsets[i] = kudoTables[i].getHeader().startOffsetOf(BufferType.DATA);
    }

    sliceInfos = new Deque[kudoTables.length];
    for (int i = 0; i < sliceInfos.length; i++) {
      sliceInfos[i] = new ArrayDeque<>(8);
      KudoTableHeader header = kudoTables[i].getHeader();
      sliceInfos[i].addLast(new SliceInfo(header.getOffset(), header.getNumRows()));
    }

    sliceInfoBuf = new SliceInfo[kudoTables.length];
  }

  @Override
  public void visitTopSchema(Schema schema) {
    result = new KudoHostMergeResult(schema, buffer, colViewInfoList);
  }

  @Override
  public void preVisitStruct(Schema structType) {
    ColumnOffsetInfo offsetInfo = getCurColumnOffsets();
    long nullCount = deserializeValidityBuffer(offsetInfo);
    long totalRowCount = rowCounts[curColIdx];
    colViewInfoList[curColIdx] = new ColumnViewInfo(structType.getType(),
        offsetInfo, nullCount, totalRowCount);


    for (int i=0; i<kudoTables.length; i++) {
      KudoTableHeader header = kudoTables[i].getHeader();
      SliceInfo sliceInfo = sliceInfoOf(i);
      if (header.hasValidityBuffer(curColIdx) && sliceInfo.getRowCount() > 0) {
        validityOffsets[i] += sliceInfo.getValidityBufferInfo().getBufferLength();
      }
    }
    curColIdx++;
  }

  public void visitStruct(Schema structType) {
    // Noop
  }

  @Override
  public void preVisitList(Schema listType) {
    ColumnOffsetInfo offsetInfo = getCurColumnOffsets();
    long nullCount = deserializeValidityBuffer(offsetInfo);
    long totalRowCount = rowCounts[curColIdx];
    deserializeOffsetBuffer(offsetInfo);

    colViewInfoList[curColIdx] = new ColumnViewInfo(listType.getType(),
        offsetInfo, nullCount, totalRowCount);

    for (int i=0; i<kudoTables.length; i++) {
      KudoTableHeader header = kudoTables[i].getHeader();
      SliceInfo sliceInfo = sliceInfoOf(i);
      if (header.hasValidityBuffer(curColIdx) && sliceInfo.getRowCount() > 0) {
        validityOffsets[i] += sliceInfo.getValidityBufferInfo().getBufferLength();
      }
      if (sliceInfo.getRowCount() > 0) {
        offsetOffsets[i] += (sliceInfo.getRowCount() + 1L) * Integer.BYTES;
      }
      sliceInfos[i].addLast(sliceInfoBuf[i]);
    }
    curColIdx++;
  }

  @Override
  public void visitList(Schema listType) {
    for (int i = 0; i < kudoTables.length; i++) {
      sliceInfos[i].removeLast();
    }
  }

  @Override
  public void visit(Schema primitiveType) {
    ColumnOffsetInfo offsetInfo = getCurColumnOffsets();
    long nullCount = deserializeValidityBuffer(offsetInfo);
    long totalRowCount = rowCounts[curColIdx];
    if (primitiveType.getType().hasOffsets()) {
      deserializeOffsetBuffer(offsetInfo);
      deserializeDataBuffer(offsetInfo, OptionalInt.empty());
    } else {
      deserializeDataBuffer(offsetInfo, OptionalInt.of(primitiveType.getType().getSizeInBytes()));
    }

    colViewInfoList[curColIdx] = new ColumnViewInfo(primitiveType.getType(),
        offsetInfo, nullCount, totalRowCount);

    if (primitiveType.getType().hasOffsets()) {
      for (int i=0; i<kudoTables.length; i++) {
        KudoTableHeader header = kudoTables[i].getHeader();
        SliceInfo sliceInfo = sliceInfoOf(i);
        if (header.hasValidityBuffer(curColIdx) && sliceInfo.getRowCount() > 0) {
          validityOffsets[i] += sliceInfo.getValidityBufferInfo().getBufferLength();
        }
        if (sliceInfo.getRowCount() > 0) {
          offsetOffsets[i] += (sliceInfo.getRowCount() + 1L) * Integer.BYTES;
          dataOffsets[i] += sliceInfoBuf[i].getRowCount();
        }
      }
    } else {
      for (int i=0; i<kudoTables.length; i++) {
        KudoTableHeader header = kudoTables[i].getHeader();
        SliceInfo sliceInfo = sliceInfoOf(i);
        if (header.hasValidityBuffer(curColIdx) && sliceInfo.getRowCount() > 0) {
          validityOffsets[i] += sliceInfo.getValidityBufferInfo().getBufferLength();
        }
        if (sliceInfo.getRowCount() > 0) {
          dataOffsets[i] += primitiveType.getType().getSizeInBytes() *
              (long) sliceInfo.getRowCount();
        }
      }
    }
    curColIdx++;
  }

  private long deserializeValidityBuffer(ColumnOffsetInfo curColOffset) {
    if (curColOffset.getValidity() != INVALID_OFFSET) {
      long offset = curColOffset.getValidity();

      ValidityBufferMerger merger = new ValidityBufferMerger(buffer, offset, inputBuf, outputBuf);
      for (int tableIdx = 0; tableIdx < kudoTables.length; tableIdx += 1) {
        SliceInfo sliceInfo = sliceInfoOf(tableIdx);
        long validityOffset = validityOffsets[tableIdx];
        if (kudoTables[tableIdx].getHeader().hasValidityBuffer(curColIdx)) {
          merger.copyValidityBuffer(kudoTables[tableIdx].getBuffer(), toIntExact(validityOffset), sliceInfo);
        } else {
          merger.appendAllValid(sliceInfo.getRowCount());
        }
      }
      return merger.getTotalNullCount();
    } else {
      return 0;
    }
  }

  private void deserializeOffsetBuffer(ColumnOffsetInfo curColOffset) {
    Arrays.fill(sliceInfoBuf, EMPTY_SLICE);

    if (curColOffset.getOffset() == INVALID_OFFSET) {
      return;
    }

    long outputOffset = curColOffset.getOffset();
    HostMemoryBuffer offsetBuf = buffer;

    int accumulatedDataLen = 0;

    for (int tableIdx = 0; tableIdx < kudoTables.length; tableIdx += 1) {
      SliceInfo sliceInfo = sliceInfoOf(tableIdx);
      if (sliceInfo.getRowCount() <= 0) {
        sliceInfoBuf[tableIdx] = EMPTY_SLICE;
        continue;
      }

      int rowCnt = sliceInfo.getRowCount();

      int firstOffset = offsetOf(tableIdx, 0);
      int lastOffset = offsetOf(tableIdx, rowCnt);
      long inputOffset = offsetOffsets[tableIdx];

      if (firstOffset < 0 || lastOffset < firstOffset) {
        if (KUDO_SANITY_CHECK) {
          int[] offsetValues = new int[rowCnt];
          for (int i = 0; i < rowCnt; i++) {
            offsetValues[i] = offsetOf(tableIdx, i);
          }
          LOG.error("Invalid offset values: [{}], table index: {}, row count: {}, " +
                  "first offset: {}, last offset: {}, kudo table header: {}",
              Arrays.toString(offsetValues), tableIdx, rowCnt, firstOffset, lastOffset,
              kudoTables[tableIdx].getHeader());
        }
        throw new IllegalArgumentException("Invalid kudo offset buffer content, first offset: "
            + firstOffset + ", last offset: " + lastOffset);
      }

      while (rowCnt > 0) {
        int arrLen = min(rowCnt, min(inputBuf.length, outputBuf.length));
        kudoTables[tableIdx].getBuffer().getInts(inputBuf, 0, inputOffset, arrLen);

        boolean isValid = true;
        for (int i = 0; i < arrLen; i++) {
          outputBuf[i] = inputBuf[i] - firstOffset + accumulatedDataLen;
          isValid = isValid && (outputBuf[i] >= 0);
        }

        if (!isValid) {
          if (KUDO_SANITY_CHECK) {
            int[] offsetValues = new int[sliceInfo.getRowCount()];
            for (int i = 0; i < sliceInfo.getRowCount(); i++) {
              offsetValues[i] = offsetOf(tableIdx, i);
            }
            LOG.error("Negative output offset found, invalid offset values: [{}], " +
                    "table index: {}, row count: {}, kudo table header: {}",
                Arrays.toString(offsetValues), tableIdx, sliceInfo.getRowCount(),
                kudoTables[tableIdx].getHeader());
          }
          throw new IllegalArgumentException("Invalid kudo offset buffer content: " +
              "negative output offset found");
        }

        offsetBuf.setInts(outputOffset, outputBuf, 0, arrLen);
        rowCnt -= arrLen;
        inputOffset += arrLen * (long) Integer.BYTES;
        outputOffset += arrLen * (long) Integer.BYTES;
      }

      sliceInfoBuf[tableIdx] = new SliceInfo(firstOffset, lastOffset - firstOffset);
      long newAccumulatedDataLen = accumulatedDataLen + (long)(lastOffset - firstOffset);
      accumulatedDataLen = toIntExact(newAccumulatedDataLen);
    }

    offsetBuf.setInt(outputOffset, accumulatedDataLen);
  }

  private void deserializeDataBuffer(ColumnOffsetInfo curColOffset, OptionalInt sizeInBytes) {
    if (curColOffset.getData() != INVALID_OFFSET && curColOffset.getDataBufferLen() > 0) {
      long offset = curColOffset.getData();

      if (sizeInBytes.isPresent()) {
        // Fixed size type
        int elementSize = sizeInBytes.getAsInt();

        long start = offset;
        for (int tableIdx = 0; tableIdx < kudoTables.length; tableIdx += 1) {
          SliceInfo sliceInfo = sliceInfoOf(tableIdx);
          if (sliceInfo.getRowCount() > 0) {
            long thisDataLen = elementSize * (long) sliceInfo.getRowCount();
            copyDataBuffer(buffer, start, tableIdx, thisDataLen);
            start += thisDataLen;
          }
        }
      } else {
        // String type
        long start = offset;
        for (int tableIdx = 0; tableIdx < kudoTables.length; tableIdx += 1) {
          int thisDataLen = sliceInfoBuf[tableIdx].getRowCount();
          copyDataBuffer(buffer, start, tableIdx, thisDataLen);
          start += thisDataLen;
        }
      }
    }
  }


  private ColumnOffsetInfo getCurColumnOffsets() {
    return columnOffsets[curColIdx];
  }

  private SliceInfo sliceInfoOf(int tableIdx) {
    return sliceInfos[tableIdx].getLast();
  }

  private int offsetOf(int tableIdx, long rowIdx) {
    long startOffset = offsetOffsets[tableIdx];
    return kudoTables[tableIdx].getBuffer().getInt(startOffset + rowIdx * Integer.BYTES);
  }

  private void copyDataBuffer(HostMemoryBuffer dst, long dstOffset, int tableIdx, long dataLen) {
    long startOffset = dataOffsets[tableIdx];
    dst.copyFromHostBuffer(dstOffset, kudoTables[tableIdx].getBuffer(), startOffset, dataLen);
  }

  static KudoHostMergeResult merge(Schema schema, MergedInfoCalc mergedInfo) {
    KudoTable[] serializedTables = mergedInfo.getTables();
    return Arms.closeIfException(HostMemoryBuffer.allocate(mergedInfo.getTotalDataLen(), true),
        buffer -> {
          KudoTableMerger merger = new KudoTableMerger(serializedTables, buffer, mergedInfo.getColumnOffsets(),
                  mergedInfo.getRowCount());
          Visitors.visitSchema(schema, merger);
          return merger.result;
        });
  }

  /**
   * A helper class to merge validity buffers of multiple tables into a single validity buffer.
   * <br/>
   * Visible for testing.
   */
  static class ValidityBufferMerger {
    private final HostMemoryBuffer dest;
    private final long destOffset;
    private final int[] inputBuf;
    private final int[] outputBuf;

    private long totalNullCount = 0;
    private long totalRowCount = 0;

    ValidityBufferMerger(HostMemoryBuffer dest, long destOffset, int[] inputBuf, int[] outputBuf) {
      this.dest = dest;
      this.destOffset = destOffset;
      this.inputBuf = inputBuf;
      this.outputBuf = outputBuf;
    }

    long getTotalNullCount() {
      return totalNullCount;
    }

    long getTotalRowCount() {
      return totalRowCount;
    }

    /**
     * Copy source validity buffer to the destination buffer.
     * <br/>
     * The algorithm copy source validity buffer into destination buffer, and it improves efficiency
     * with following key points:
     *
     * <ul>
     *   <li> It processed this buffer integer by integer, and uses {@link Integer#bitCount(int)} to
     *   count null values in each integer.
     *   </li>
     *   <li> It uses an intermediate int array to avoid
     *   {@link HostMemoryBuffer#getInt(long)} method calls in for loop, which makes the for loop quite efficient.
     *   </li>
     * </ul>
     *
     *
     * @param src The memory buffer of source kudo table.
     * @param srcOffset The offset of validity buffer in the source buffer.
     * @param sliceInfo The slice info of the source kudo table.
     * @return Number of null values in the validity buffer.
     */
    int copyValidityBuffer(HostMemoryBuffer src, int srcOffset,
                            SliceInfo sliceInfo) {
      if (sliceInfo.getRowCount() <= 0) {
        return 0;
      }

      int curDestBitIdx = toIntExact(totalRowCount % 32);
      int curSrcBitIdx = sliceInfo.getValidityBufferInfo().getBeginBit();
      int nullCount;

      if (curSrcBitIdx < curDestBitIdx) {
        // First case of this algorithm, in which we always need to merge remained bits of previous
        // integer when copying it to destination buffer.
        nullCount = copySourceCaseOne(src, srcOffset, sliceInfo);
      } else if (curSrcBitIdx > curDestBitIdx) {
        // Second case of this algorithm, in which we always need to borrow bits from next integer
        // when copying it to destination buffer.
        nullCount = copySourceCaseTwo(src, srcOffset, sliceInfo);
      } else {
        // Third case of this algorithm, in which we can directly copy source buffer to destination
        // buffer, except some special handling of first integer.
        nullCount = copySourceCaseThree(src, srcOffset, sliceInfo);
      }

      totalRowCount += sliceInfo.getRowCount();
      totalNullCount += nullCount;

      return nullCount;
    }

    /**
     * Append {@code numRows} valid bits to the destination buffer.
     * @param numRows Number of rows to append.
     */
    void appendAllValid(int numRows) {
      if (numRows <= 0) {
        return;
      }
      long curDestIntIdx = destOffset + (totalRowCount / 32) * 4;
      int curDestBitIdx = toIntExact(totalRowCount % 32);

      // First output
      int firstOutput = dest.getInt(curDestIntIdx);
      firstOutput |= -(1 << curDestBitIdx);
      dest.setInt(curDestIntIdx, firstOutput);

      int leftRowCount = max(0, numRows - (32 - curDestBitIdx));

      curDestIntIdx += 4;
      while (leftRowCount > 0) {
        int curArrLen = min(leftRowCount / 32, ONES.length);
        if (curArrLen == 0) {
          dest.setInt(curDestIntIdx, 0xFFFFFFFF);
          leftRowCount = 0;
        } else {
          dest.setInts(curDestIntIdx, ONES, 0, curArrLen);
          leftRowCount = max(0, leftRowCount - 32 * curArrLen);
          curDestIntIdx += curArrLen * 4;
        }
      }

      totalRowCount += numRows;
    }

    private int copySourceCaseOne(HostMemoryBuffer src, int srcOffset,
                                  SliceInfo sliceInfo) {
      int nullCount = 0;
      int leftRowCount = sliceInfo.getRowCount();

      long curDestIntIdx = destOffset + (totalRowCount / 32) * 4;
      int curDestBitIdx = toIntExact(totalRowCount % 32);

      int srcIntBufLen = (sliceInfo.getValidityBufferInfo().getBufferLength() + 3) / 4;
      int curSrcIntIdx = srcOffset;
      int curSrcBitIdx = sliceInfo.getValidityBufferInfo().getBeginBit();

      int rshift = curDestBitIdx - curSrcBitIdx;
      // process first element
      int outputMask = (1 << curDestBitIdx) - 1;
      int destOutput = dest.getInt(curDestIntIdx) & outputMask;

      int rawInput = getIntSafe(src, curSrcIntIdx);
      int input = (rawInput >>> curSrcBitIdx) << curDestBitIdx;
      destOutput = input | destOutput;
      dest.setInt(curDestIntIdx, destOutput);

      if (srcIntBufLen == 1) {
        int leftRem = 32 - curSrcBitIdx - leftRowCount;
        assert leftRem >= 0;
        nullCount += leftRowCount - Integer.bitCount((rawInput >>> curSrcBitIdx) << (curSrcBitIdx + leftRem));
        if ((leftRowCount + curDestBitIdx) > 32) {
          curDestIntIdx += 4;
          input = rawInput >>> (curSrcBitIdx + 32 - curDestBitIdx);
          dest.setInt(curDestIntIdx, input);
        }
        assert nullCount >= 0;
        return nullCount;
      }

      nullCount += 32 - curDestBitIdx - Integer.bitCount(input & ~outputMask);
      leftRowCount -= (32 - curDestBitIdx);
      int lastValue = rawInput >>> (32 - rshift);
      int lastOutput = 0;

      curSrcIntIdx += 4;
      curDestIntIdx += 4;
      while (leftRowCount > 0) {
        int curArrLen = min(min(inputBuf.length, outputBuf.length), srcIntBufLen - (curSrcIntIdx - srcOffset) / 4);
        if (curArrLen <= 0) {
          dest.setInt(curDestIntIdx, lastValue);
          nullCount += leftRowCount - Integer.bitCount(lastValue & ((1 << leftRowCount) - 1));
          leftRowCount = 0;
          break;
        }

        getIntsSafe(src, inputBuf, 0, curSrcIntIdx, curArrLen);

        for (int i=0; i<curArrLen; i++) {
          outputBuf[i] = (inputBuf[i] << rshift) | lastValue;
          lastValue = inputBuf[i] >>> (32 - rshift);
          nullCount += 32 - Integer.bitCount(outputBuf[i]);
          leftRowCount -= 32;
        }

        lastOutput = outputBuf[curArrLen - 1];
        dest.setInts(curDestIntIdx, outputBuf, 0, curArrLen);
        curSrcIntIdx += curArrLen * 4;
        curDestIntIdx += curArrLen * 4L;
      }

      if (leftRowCount < 0) {
        nullCount -= -leftRowCount - Integer.bitCount(lastOutput >>> (32 + leftRowCount));
      }
      assert nullCount >= 0;

      return nullCount;
    }

    private int copySourceCaseTwo(HostMemoryBuffer src, int srcOffset,
                                  SliceInfo sliceInfo) {
      int leftRowCount = sliceInfo.getRowCount();
      int nullCount = 0;

      long curDestIntIdx = destOffset + (totalRowCount / 32) * 4;
      int curDestBitIdx = toIntExact(totalRowCount % 32);

      int srcIntBufLen = (sliceInfo.getValidityBufferInfo().getBufferLength() + 3) / 4;
      int curSrcIntIdx = srcOffset;
      int curSrcBitIdx = sliceInfo.getValidityBufferInfo().getBeginBit();

      int rshift = curSrcBitIdx - curDestBitIdx;
      // process first element
      int destMask = (1 << curDestBitIdx) - 1;
      int destOutput = dest.getInt(curDestIntIdx) & destMask;

      int input = getIntSafe(src, curSrcIntIdx);
      if (srcIntBufLen == 1) {
        int leftRem = 32 - curSrcBitIdx - leftRowCount;
        assert leftRem >= 0;
        int inputMask = -(1 << curSrcBitIdx);
        nullCount += leftRowCount - Integer.bitCount( (input & inputMask) << leftRem);
        destOutput |= (input >>> curSrcBitIdx) << curDestBitIdx;
        dest.setInt(curDestIntIdx, destOutput);
        assert nullCount >= 0;
        return nullCount;
      }

      nullCount -= curDestBitIdx - Integer.bitCount(destOutput);
      leftRowCount += curDestBitIdx;
      int lastValue = destOutput | ((input >>> curSrcBitIdx) << curDestBitIdx);
      int lastOutput = 0;
      curSrcIntIdx += 4;

      while (leftRowCount > 0) {
        int curArrLen = min(min(inputBuf.length, outputBuf.length), srcIntBufLen - (curSrcIntIdx - srcOffset) / 4);
        if (curArrLen <= 0) {
          nullCount += leftRowCount - Integer.bitCount(lastValue & ((1 << leftRowCount) - 1));
          dest.setInt(curDestIntIdx, lastValue);
          leftRowCount = 0;
          break;
        }
        getIntsSafe(src, inputBuf, 0, curSrcIntIdx, curArrLen);
        for (int i=0; i<curArrLen; i++) {
          outputBuf[i] = (inputBuf[i] << (32 - rshift)) | lastValue;
          nullCount += 32 - Integer.bitCount(outputBuf[i]);
          leftRowCount -= 32;
          lastValue = inputBuf[i] >>> rshift;
        }
        lastOutput = outputBuf[curArrLen - 1];
        dest.setInts(curDestIntIdx, outputBuf, 0, curArrLen);
        curSrcIntIdx += curArrLen * 4;
        curDestIntIdx += curArrLen * 4L;
      }

      if (leftRowCount < 0) {
        nullCount -= -leftRowCount - Integer.bitCount(lastOutput >>> (32 + leftRowCount));
      }
      assert nullCount >= 0;
      return nullCount;
    }

    private static int getIntAsBytes(HostMemoryBuffer src, long offset, long length) {
      // We need to build up the int ourselves and pad it as needed
      int ret = 0;
      for (int at = 0; at < 4 && (offset + at) < length ; at++) {
        int b = src.getByte(offset + at) & 0xFF;
        ret |= b << (at * 8);
      }
      return ret;
    }

    private static int getIntSafe(HostMemoryBuffer src, long offset) {
      long length = src.getLength();
      if (offset + 4 < length) {
        return src.getInt(offset);
      } else {
        return getIntAsBytes(src, offset, length);
      }
    }

    private static void getIntsSafe(HostMemoryBuffer src, int[] dst, long dstIndex, long srcOffset, int count) {
      long length = src.getLength();
      if (srcOffset + (4L * count) < length) {
        src.getInts(dst, dstIndex, srcOffset, count);
      } else {
        // Read as much as we can the fast way
        int fastCount = (int)((length - srcOffset) / 4);
        src.getInts(dst, dstIndex, srcOffset, fastCount);
        // Then read the rest slowly...
        for (int index = fastCount; index < count; index++) {
          dst[index + (int)dstIndex] = getIntAsBytes(src, srcOffset + (index * 4L), length);
        }
      }
    }

    private int copySourceCaseThree(HostMemoryBuffer src, int srcOffset,
                                    SliceInfo sliceInfo) {
      int leftRowCount = sliceInfo.getRowCount();
      int nullCount = 0;

      long curDestIntIdx = destOffset + (totalRowCount / 32) * 4;
      int curDestBitIdx = toIntExact(totalRowCount % 32);

      int srcIntBufLen = (sliceInfo.getValidityBufferInfo().getBufferLength() + 3) / 4;
      int curSrcIntIdx = srcOffset;
      int curSrcBitIdx = sliceInfo.getValidityBufferInfo().getBeginBit();

      // Process first element
      int mask = (1 << curDestBitIdx) - 1;
      int firstInput = getIntSafe(src, curSrcIntIdx);
      int destOutput = dest.getInt(curDestIntIdx);
      destOutput = (firstInput & ~mask) | (destOutput & mask);
      dest.setInt(curDestIntIdx, destOutput);

      if (srcIntBufLen == 1) {
        int leftRem = 32 - curSrcBitIdx - leftRowCount;
        assert leftRem >= 0;
        nullCount += leftRowCount - Integer.bitCount((firstInput & ~mask) << leftRem);
        assert nullCount >= 0;
        return nullCount;
      }

      nullCount += 32 - curSrcBitIdx - Integer.bitCount((firstInput & ~mask));
      leftRowCount -= 32 - curSrcBitIdx;

      curSrcIntIdx += 4;
      curDestIntIdx += 4;
      int lastOutput = 0;
      while (leftRowCount > 0) {
        int curArrLen = min(min(inputBuf.length, outputBuf.length), srcIntBufLen - (curSrcIntIdx - srcOffset) / 4);
        assert curArrLen > 0;
        getIntsSafe(src, inputBuf, 0, curSrcIntIdx, curArrLen);
        for (int i=0; i<curArrLen; i++) {
          nullCount += 32 - Integer.bitCount(inputBuf[i]);
          leftRowCount -= 32;
        }
        dest.setInts(curDestIntIdx, inputBuf, 0, curArrLen);
        lastOutput = inputBuf[curArrLen - 1];
        curSrcIntIdx += curArrLen * 4;
        curDestIntIdx += curArrLen * 4L;
      }

      if (leftRowCount < 0) {
        nullCount -= -leftRowCount - Integer.bitCount(lastOutput >>> (32 + leftRowCount));
      }
      assert nullCount >= 0;

      return nullCount;
    }
  }
}
