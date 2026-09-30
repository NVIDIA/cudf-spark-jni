/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
import ai.rapids.cudf.Scalar;
import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class RegexRewriteUtilsTest {

  @Test
  void testLiteralRangePattern() {
    int d = 3;
    try (ColumnVector inputCv = ColumnVector.fromStrings(
        "abc123", "aabc123", "aabc12", "abc1232", "aabc1232");
        Scalar pattern = Scalar.fromString("abc");
        ColumnVector expected = ColumnVector.fromBooleans(true, true, false, true, true);
        ColumnVector actual = RegexRewriteUtils.literalRangePattern(inputCv, pattern, d, 48, 57)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testLiteralRangePatternChinese() {
    int d = 2;
    try (ColumnVector inputCv = ColumnVector.fromStrings(
        "数据砖块", "火花-急流英伟达", "英伟达Nvidia", "火花-急流");
        Scalar pattern = Scalar.fromString("英");
        ColumnVector expected = ColumnVector.fromBooleans(false, true, true, false);
        ColumnVector actual = RegexRewriteUtils.literalRangePattern(inputCv, pattern, d, 19968, 40869)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

}
