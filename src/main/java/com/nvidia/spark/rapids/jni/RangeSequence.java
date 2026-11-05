/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ai.rapids.cudf.NativeDepsLoader;

import java.util.Objects;

/** Generates flat INT64 columns from host-side sequence ranges. */
public final class RangeSequence {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private RangeSequence() {}

  /**
   * Expands each {@code (starts[i], sizes[i])} range in input order, with a step of one.
   * For example, starts {@code [100, 200]} and sizes {@code [3, 2]} produce
   * {@code [100, 101, 102, 200, 201]}. Empty ranges are ignored; starts need not be sorted.
   *
   * <p>The CPU validates ranges and computes cumulative ends.
   * The GPU writes the output directly into the final column, using O(number of ranges)
   * temporary device memory without materializing and concatenating separate sequences.
   * A single range needs no device metadata allocation. Empty output launches no kernel.
   * The call may synchronize the CUDA stream. Metadata uploads for multiple ranges complete
   * before their host staging buffers can be released. Output generation may still be pending
   * on return.
   *
   * @param starts first value of each sequence; must have the same length as sizes
   * @param sizes nonnegative number of values in each sequence
   * @return a non-nullable INT64 column owned by the caller, which must close it
   * @throws NullPointerException if either input array is null
   * @throws ai.rapids.cudf.CudfException if lengths differ, a size is negative, or a sequence
   *     value overflows INT64
   * @throws ai.rapids.cudf.CudfColumnSizeOverflowException if the total size exceeds Integer.MAX_VALUE
   */
  public static ColumnVector generate(long[] starts, int[] sizes) {
    Objects.requireNonNull(starts, "starts");
    Objects.requireNonNull(sizes, "sizes");
    return new ColumnVector(generateNative(starts, sizes));
  }

  private static native long generateNative(long[] starts, int[] sizes);
}
