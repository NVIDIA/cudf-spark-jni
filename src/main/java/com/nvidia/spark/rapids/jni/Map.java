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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.NativeDepsLoader;

/**
 * APIs for map column operations.
 */
public class Map {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Sort entries for each map in map column according to the keys of each map.
   * Note:
   *   The keys of map MUST not be null.
   *   Assume that maps do not have duplicate keys.
   *   Do not normalize/sort the nested maps in `KEY` column; This means
   *   Only consider the first level LIST(STRUCT(KEY, VALUE)) as map type.
   *
   * @param cv           Input map column, should in LIST(STRUCT(KEY, VALUE))
   *                     type.
   * @param isDescending True if sort in descending order, false if sort in
   *                     ascending order
   * @return Sorted map according to the sort order of the key column in map.
   * @throws CudfException If the input column is not a LIST(STRUCT(KEY, VALUE))
   *                       column or the keys contain nulls.
   */
  public static ColumnVector sort(ColumnView cv, boolean isDescending) {
    assert (cv.getType().equals(DType.LIST));
    long r = sort(cv.getNativeView(), isDescending);
    return new ColumnVector(r);
  }

  private static native long sort(long handle, boolean isDescending) throws CudfException;
}

