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

import ai.rapids.cudf.*;

public class RegexRewriteUtils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

/**
 * @brief Check if input string contains regex pattern `literal[start-end]{len,}`, which means
 * a literal string followed by a range of characters in the range of start to end, with at least
 * len characters.
 *
 * @param input Column of strings to check for literal.
 * @param literal UTF-8 encoded string to check in strings column.
 * @param len Minimum number of characters to check after the literal.
 * @param start Minimum UTF-8 codepoint value to check for in the range.
 * @param end Maximum UTF-8 codepoint value to check for in the range.
 * @return ColumnVector of booleans where true indicates the string contains the pattern.
 */
  public static ColumnVector literalRangePattern(ColumnVector input, Scalar literal, int len, int start, int end) {
    assert(input.getType().equals(DType.STRING)) : "column must be a String";
    return new ColumnVector(literalRangePattern(input.getNativeView(), literal.getScalarHandle(), len, start, end));
  }

  private static native long literalRangePattern(long input, long literal, int len, int start, int end);
}
