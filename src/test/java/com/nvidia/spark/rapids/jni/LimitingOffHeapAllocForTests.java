/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

import ai.rapids.cudf.HostMemoryBuffer;

import java.util.Optional;

/**
 * This provides a way to allocate and deallocate off heap buffers using the RmmSpark APIs for
 * retry on allocations.
 */
public class LimitingOffHeapAllocForTests {
  private static long limit;
  private static long amountAllocated = 0;
  public static synchronized void setLimit(long limit) {
    LimitingOffHeapAllocForTests.limit = limit;
    if (amountAllocated > 0) {
      throw new IllegalStateException("PREVIOUS TEST LEAKED MEMORY!!!");
    }
  }

  private static Optional<HostMemoryBuffer> allocInternal(long amount, boolean blocking) {
    Optional<HostMemoryBuffer> ret = Optional.empty();
    boolean wasOom = true;
    boolean isRecursive = RmmSpark.preCpuAlloc(amount, blocking);
    try {
      synchronized (LimitingOffHeapAllocForTests.class) {
        if (amountAllocated + amount <= limit) {
          amountAllocated += amount;
          wasOom = false;
          HostMemoryBuffer buff = HostMemoryBuffer.allocate(amount);
          final long ptr = buff.getAddress();
          buff.setEventHandler(refCount -> {
            if (refCount == 0) {
              synchronized (LimitingOffHeapAllocForTests.class) {
                amountAllocated -= amount;
              }
              RmmSpark.cpuDeallocate(ptr, amount);
            }
          });
          ret = Optional.of(buff);
        }
      }
    } finally {
      if (ret.isPresent()) {
        RmmSpark.postCpuAllocSuccess(ret.get().getAddress(), amount, blocking, isRecursive);
      } else {
        RmmSpark.postCpuAllocFailed(wasOom, blocking, isRecursive);
      }
    }
    return ret;
  }

  /**
   * Do a non-blocking allocation
   * @param amount the amount to allocate
   * @return the allocated buffer or not.
   */
  public static Optional<HostMemoryBuffer> tryAlloc(long amount) {
    return allocInternal(amount, false);
  }

  /**
   * Do a blocking allocation
   * @param amount the amount to allocate
   * @return the allocated buffer
   */
  public static HostMemoryBuffer alloc(long amount) {
    Optional<HostMemoryBuffer> ret = Optional.empty();
    while (!ret.isPresent()) {
      ret = allocInternal(amount, true);
    }
    return ret.get();
  }
}