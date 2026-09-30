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

public class NumberConverter {

  /**
   *
   * Convert numbers in a string column between different number bases. If toBase>0 the result
   * is unsigned, otherwise it is signed.
   * First trim the space characters (ASCII 32).
   * Return null if len(trim_ascii_32(str)) == 0.
   * Return all nulls if `from_base` or `to_base` is not in range [2, 36].
   *
   * e.g.:
   * convert('11', 2, 10) = '3'
   * convert('F', 16, 10) = '15'
   * convert('17', 10, 16) = '11'
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return the string column contains numbers with `to_base` base
   */
  public static ColumnVector convertCvCvCv(
    ColumnVector input, ColumnVector fromBase, ColumnVector toBase) {

    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (fromBase.getType().equals(DType.INT32)) : "From base must be Integers";
    assert (toBase.getType().equals(DType.INT32)) : "To base must be Integers";

    return new ColumnVector(convert(
      input.getNativeView(), true,
      fromBase.getNativeView(), true,
      toBase.getNativeView(), true));
  }

  /**
   *
   * Convert numbers in a string column between different number bases. If toBase>0 the result
   * is unsigned, otherwise it is signed.
   * First trim the space characters (ASCII 32).
   * Return null if len(trim_ascii_32(str)) == 0.
   * Return all nulls if `from_base` or `to_base` is not in range [2, 36].
   *
   * e.g.:
   * convert('11', 2, 10) = '3'
   * convert('F', 16, 10) = '15'
   * convert('17', 10, 16) = '11'
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return the string column contains numbers with `to_base` base
   */
  public static ColumnVector convertCvCvS(ColumnVector input, ColumnVector fromBase, int toBase) {

    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (fromBase.getType().equals(DType.INT32)) : "From base must be Integers";

    return new ColumnVector(convert(
      input.getNativeView(), true,
      fromBase.getNativeView(), true,
      toBase, false));
  }

  /**
   *
   * Convert numbers in a string column between different number bases. If toBase>0 the result
   * is unsigned, otherwise it is signed.
   * First trim the space characters (ASCII 32).
   * Return null if len(trim_ascii_32(str)) == 0.
   * Return all nulls if `from_base` or `to_base` is not in range [2, 36].
   *
   * e.g.:
   * convert('11', 2, 10) = '3'
   * convert('F', 16, 10) = '15'
   * convert('17', 10, 16) = '11'
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return the string column contains numbers with `to_base` base
   */
  public static ColumnVector convertCvSCv(ColumnVector input, int fromBase, ColumnVector toBase) {

    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (toBase.getType().equals(DType.INT32)) : "To base must be Integers";

    return new ColumnVector(convert(
      input.getNativeView(), true,
      fromBase, false,
      toBase.getNativeView(), true));
  }

  /**
   *
   * Convert numbers in a string column between different number bases. If toBase>0 the result
   * is unsigned, otherwise it is signed.
   * First trim the space characters (ASCII 32).
   * Return null if len(trim_ascii_32(str)) == 0.
   * Return all nulls if `from_base` or `to_base` is not in range [2, 36].
   *
   * e.g.:
   * convert('11', 2, 10) = '3'
   * convert('F', 16, 10) = '15'
   * convert('17', 10, 16) = '11'
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return the string column contains numbers with `to_base` base
   */
  public static ColumnVector convertCvSS(ColumnVector input, int fromBase, int toBase) {

    assert (input.getType().equals(DType.STRING)) : "Input must be strings";

    return new ColumnVector(convert(
      input.getNativeView(), true,
      fromBase, false,
      toBase, false));
  }

  /**
   *
   * Convert numbers in a string column between different number bases. If toBase>0 the result
   * is unsigned, otherwise it is signed.
   * First trim the space characters (ASCII 32).
   * Return null if len(trim_ascii_32(str)) == 0.
   * Return all nulls if `from_base` or `to_base` is not in range [2, 36].
   *
   * e.g.:
   * convert('11', 2, 10) = '3'
   * convert('F', 16, 10) = '15'
   * convert('17', 10, 16) = '11'
   *
   * @param input    the input string scalar
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return the string column contains numbers with `to_base` base
   */
  public static ColumnVector convertSCvCv(
    Scalar input, ColumnVector fromBase, ColumnVector toBase) {
    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (fromBase.getType().equals(DType.INT32)) : "From base must be Integers";
    assert (toBase.getType().equals(DType.INT32)) : "To base must be Integers";
    return new ColumnVector(convert(
      input.getScalarHandle(), false,
      fromBase.getNativeView(), true,
      toBase.getNativeView(), true));
  }

  /**
   *
   * Convert numbers in a string column between different number bases. If toBase>0 the result
   * is unsigned, otherwise it is signed.
   * First trim the space characters (ASCII 32).
   * Return null if len(trim_ascii_32(str)) == 0.
   * Return all nulls if `from_base` or `to_base` is not in range [2, 36].
   *
   * e.g.:
   * convert('11', 2, 10) = '3'
   * convert('F', 16, 10) = '15'
   * convert('17', 10, 16) = '11'
   *
   * @param input    the input string scalar
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return the string column contains numbers with `to_base` base
   */
  public static ColumnVector convertSCvS(Scalar input, ColumnVector fromBase, int toBase) {
    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (fromBase.getType().equals(DType.INT32)) : "From base must be Integers";
    return new ColumnVector(convert(
      input.getScalarHandle(), false,
      fromBase.getNativeView(), true,
      toBase, false));
  }

  /**
   *
   * Convert numbers in a string column between different number bases. If toBase>0 the result
   * is unsigned, otherwise it is signed.
   * First trim the space characters (ASCII 32).
   * Return null if len(trim_ascii_32(str)) == 0.
   * Return all nulls if `from_base` or `to_base` is not in range [2, 36].
   *
   * e.g.:
   * convert('11', 2, 10) = '3'
   * convert('F', 16, 10) = '15'
   * convert('17', 10, 16) = '11'
   *
   * @param input    the input string scalar
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return the string column contains numbers with `to_base` base
   */
  public static ColumnVector convertSSCv(Scalar input, int fromBase, ColumnVector toBase) {
    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (toBase.getType().equals(DType.INT32)) : "To base must be Integers";
    return new ColumnVector(convert(
      input.getScalarHandle(), false,
      fromBase, false,
      toBase.getNativeView(), true));
  }

  /**
   *
   * Check if overflow occurs for converting numbers(in string column) between different
   * number bases. This is for the checking when it's ANSI mode. For more details,
   * please refer to the convert function.
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return If overflow occurs, return true; otherwise, return false.
   */
  public static boolean isConvertOverflowCvCvCv(
    ColumnVector input, ColumnVector fromBase, ColumnVector toBase) {
    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (fromBase.getType().equals(DType.INT32)) : "From base must be Integers";
    assert (toBase.getType().equals(DType.INT32)) : "To base must be Integers";
    return isConvertOverflow(
      input.getNativeView(), true,
      fromBase.getNativeView(), true,
      toBase.getNativeView(), true);
  }

  /**
   *
   * Check if overflow occurs for converting numbers(in string column) between different
   * number bases. This is for the checking when it's ANSI mode. For more details,
   * please refer to the convert function.
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return If overflow occurs, return true; otherwise, return false.
   */
  public static boolean isConvertOverflowCvCvS(
    ColumnVector input, ColumnVector fromBase, int toBase) {

    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (fromBase.getType().equals(DType.INT32)) : "From base must be Integers";

    return isConvertOverflow(
      input.getNativeView(), true,
      fromBase.getNativeView(), true,
      toBase, false);
  }

  /**
   *
   * Check if overflow occurs for converting numbers(in string column) between different
   * number bases. This is for the checking when it's ANSI mode. For more details,
   * please refer to the convert function.
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return If overflow occurs, return true; otherwise, return false.
   */
  public static boolean isConvertOverflowCvSCv(
    ColumnVector input, int fromBase, ColumnVector toBase) {

    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (toBase.getType().equals(DType.INT32)) : "To base must be Integers";

    return isConvertOverflow(
      input.getNativeView(), true,
      fromBase, false,
      toBase.getNativeView(), true);
  }

  /**
   *
   * Check if overflow occurs for converting numbers(in string column) between different
   * number bases. This is for the checking when it's ANSI mode. For more details,
   * please refer to the convert function.
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return If overflow occurs, return true; otherwise, return false.
   */
  public static boolean isConvertOverflowCvSS(ColumnVector input, int fromBase, int toBase) {

    assert (input.getType().equals(DType.STRING)) : "Input must be strings";

    return isConvertOverflow(
      input.getNativeView(), true,
      fromBase, false,
      toBase, false);
  }

  /**
   *
   * Check if overflow occurs for converting numbers(in string column) between different
   * number bases. This is for the checking when it's ANSI mode. For more details,
   * please refer to the convert function.
   *
   * @param input    the input string scalar
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return If overflow occurs, return true; otherwise, return false.
   */
  public static boolean isConvertOverflowSCvCv(
    Scalar input, ColumnVector fromBase, ColumnVector toBase) {

    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (fromBase.getType().equals(DType.INT32)) : "From base must be Integers";
    assert (toBase.getType().equals(DType.INT32)) : "To base must be Integers";

    return isConvertOverflow(
      input.getScalarHandle(), false,
      fromBase.getNativeView(), true,
      toBase.getNativeView(),true);
  }

  /**
   *
   * Check if overflow occurs for converting numbers(in string column) between different
   * number bases. This is for the checking when it's ANSI mode. For more details,
   * please refer to the convert function.
   *
   * @param input    the input string scalar
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return If overflow occurs, return true; otherwise, return false.
   */
  public static boolean isConvertOverflowSCvS(Scalar input, ColumnVector fromBase, int toBase) {

    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (fromBase.getType().equals(DType.INT32)) : "From base must be Integers";

    return isConvertOverflow(
      input.getScalarHandle(), false,
      fromBase.getNativeView(), true,
      toBase, false);
  }

  /**
   *
   * Check if overflow occurs for converting numbers(in string column) between different
   * number bases. This is for the checking when it's ANSI mode. For more details,
   * please refer to the convert function.
   *
   * @param input    the input string scalar
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return If overflow occurs, return true; otherwise, return false.
   */
  public static boolean isConvertOverflowSSCv(Scalar input, int fromBase, ColumnVector toBase) {
    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (toBase.getType().equals(DType.INT32)) : "To base must be Integers";
    return isConvertOverflow(
      input.getScalarHandle(), false,
      fromBase, false,
      toBase.getNativeView(), true);
  }

  private static native long convert(
    long input, boolean isInputCv,
    long fromBase, boolean isFromCv,
    long toBase, boolean isToCv);

  private static native boolean isConvertOverflow(
    long input, boolean isInputCv,
    long fromBase, boolean isFromCv,
    long toBase, boolean isToCv);

}
