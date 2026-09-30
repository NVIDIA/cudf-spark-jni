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

import ai.rapids.cudf.*;

import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

import java.util.Arrays;
import java.util.List;

public class MapTest {

  @Test
  void sort() {
    // Map is List<Struct<KEY, VALUE>>
    List<HostColumnVector.StructData> map1 = Arrays.asList(
        new HostColumnVector.StructData(Arrays.asList(5, 2)),
        new HostColumnVector.StructData(Arrays.asList(4, 1)));
    List<HostColumnVector.StructData> map2 = Arrays.asList(
        new HostColumnVector.StructData(Arrays.asList(2, 1)),
        new HostColumnVector.StructData(Arrays.asList(4, 3)));

    List<HostColumnVector.StructData> sorted_map1 = Arrays.asList(
        new HostColumnVector.StructData(Arrays.asList(4, 1)),
        new HostColumnVector.StructData(Arrays.asList(5, 2)));
    List<HostColumnVector.StructData> sorted_map2 = map2;

    HostColumnVector.StructType structType = new HostColumnVector.StructType(true,
        Arrays.asList(new HostColumnVector.BasicType(true, DType.INT32),
            new HostColumnVector.BasicType(true, DType.INT32)));
    try (ColumnVector cv = ColumnVector.fromLists(
        new HostColumnVector.ListType(true, structType), map1, map2);
        ColumnVector res = Map.sort(cv, false);
        ColumnVector expected = ColumnVector.fromLists(
            new HostColumnVector.ListType(true, structType), sorted_map1, sorted_map2)) {

      assertColumnsAreEqual(expected, res);
    }
  }

}

