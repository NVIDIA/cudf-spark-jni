/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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
import ai.rapids.cudf.Scalar;

public class Arithmetic {

  /**
   * Computes multiplication on two ColumnVectors.
   * If the types of the two vectors do not match, an IllegalArgumentException is
   * thrown. If the row counts of the two inputs do not match, an
   * IllegalArgumentException is thrown. If there is any overflow
   * in ANSI mode, an ExceptionWithRowIndex is thrown.
   * Only supports Spark data types: byte, short, integer, long, float32 and float64.
   * 
   * E.g.:
   * Integer.MAX_VALUE * 2 = -2 in regular mode, the result is wrong(overflow occurs).
   * Integer.MAX_VALUE * 2 = null in try mode
   * Integer.MAX_VALUE * 2 = throws exception in ansi mode
   * 
   * @param left       left input
   * @param right      right input
   * @param isAnsiMode is ANSI mode enabled
   * @param isTryMode  if true, set null when overflow occurs
   * @return a new ColumnVector containing the result of multiplying
   *
   * @throws ExceptionWithRowIndex if has any overflow in ANSI mode
   */
  public static ColumnVector multiply(ColumnVector left, ColumnVector right, boolean isAnsiMode,
      boolean isTryMode) {
    return new ColumnVector(multiply(
        left.getNativeView(), true, right.getNativeView(), true, isAnsiMode,
        isTryMode));
  }

  /**
   * Computes multiplication on two ColumnVectors.
   * If the types of the two vectors do not match, an IllegalArgumentException is
   * thrown. If the row counts of the two inputs do not match, an
   * IllegalArgumentException is thrown. If there is any overflow
   * in ANSI mode, an ExceptionWithRowIndex is thrown.
   * Only supports Spark data types: byte, short, integer, long, float32 and float64.
   * 
   * E.g.:
   * Integer.MAX_VALUE * 2 = -2 in regular mode, the result is wrong(overflow occurs).
   * Integer.MAX_VALUE * 2 = null in try mode
   * Integer.MAX_VALUE * 2 = throws exception in ansi mode
   * 
   * @param left       left input
   * @param right      right input
   * @param isAnsiMode is ANSI mode enabled
   * @param isTryMode  if true, set null when overflow occurs
   * @return a new ColumnVector containing the result of multiplying
   *
   * @throws ExceptionWithRowIndex if has any overflow in ANSI mode
   */
  public static ColumnVector multiply(ColumnVector left, Scalar right, boolean isAnsiMode,
      boolean isTryMode) {
    return new ColumnVector(multiply(
        left.getNativeView(), true, right.getScalarHandle(), false, isAnsiMode,
        isTryMode));
  }

  /**
   * Computes multiplication on two ColumnVectors.
   * If the types of the two vectors do not match, an IllegalArgumentException is
   * thrown. If the row counts of the two inputs do not match, an
   * IllegalArgumentException is thrown. If there is any overflow
   * in ANSI mode, an ExceptionWithRowIndex is thrown.
   * Only supports Spark data types: byte, short, integer, long, float32 and float64.
   * 
   * E.g.:
   * Integer.MAX_VALUE * 2 = -2 in regular mode, the result is wrong(overflow occurs).
   * Integer.MAX_VALUE * 2 = null in try mode
   * Integer.MAX_VALUE * 2 = throws exception in ansi mode
   * 
   * @param left       left input
   * @param right      right input
   * @param isAnsiMode is ANSI mode enabled
   * @param isTryMode  if true, set null when overflow occurs
   * @return a new ColumnVector containing the result of multiplying
   *
   * @throws ExceptionWithRowIndex if has any overflow in ANSI mode
   */
  public static ColumnVector multiply(Scalar left, ColumnVector right, boolean isAnsiMode,
      boolean isTryMode) {
    return new ColumnVector(multiply(
        left.getScalarHandle(), false, right.getNativeView(), true, isAnsiMode,
        isTryMode));
  }

  /**
   * Rounds all the values in a column to the specified number of decimal places.
   *
   * @param input         Column of values to be rounded
   * @param decimalPlaces Number of decimal places to round to. If negative, this
   *                      specifies the number of positions to the left of the decimal point.
   * @param mode          Rounding method(either HALF_UP or HALF_EVEN)
   * @return a new ColumnVector with rounded values.
   */
  public static ColumnVector round(ColumnView input, int decimalPlaces, RoundMode mode) {
    return new ColumnVector(round(input.getNativeView(), decimalPlaces, mode.nativeId, false));
  }

  /**
   * Rounds all the values in a column to the specified number of decimal places with
   * optional ANSI mode overflow checking.
   *
   * For integral types with negative decimalPlaces:
   * - ANSI mode: Throws ExceptionWithRowIndex if rounding would cause overflow (the rounded
   *   value exceeds the bounds of the data type). The exception includes the first row index
   *   where overflow would occur.
   * - Non-ANSI mode: Allows overflow to wrap naturally (standard integer overflow behavior).
   *
   * For non-integral types or positive decimal_places, always delegates to the original round() function.
   *
   * Examples:
   *   - round(127, -2) for ByteType = 100 (OK)
   *   - round(125, -1) for ByteType = 130, which overflows (throws in ANSI mode, wraps in non-ANSI)
   *
   * @param input         Column of values to be rounded
   * @param decimalPlaces Number of decimal places to round to. If negative, this
   *                      specifies the number of positions to the left of the decimal point.
   * @param mode          Rounding method (either HALF_UP or HALF_EVEN)
   * @param isAnsiMode    If true, throws exception when overflow would occur for integral types;
   *                      if false, allows overflow wrapping
   * @return a new ColumnVector with rounded values.
   *
   * @throws ExceptionWithRowIndex if ANSI mode is enabled and overflow would occur
   */
  public static ColumnVector round(ColumnView input, int decimalPlaces, RoundMode mode,
      boolean isAnsiMode) {
    return new ColumnVector(round(input.getNativeView(), decimalPlaces, mode.nativeId, isAnsiMode));
  }

  /**
   * Rounds all the values in a column with decimal places = 0. Default number of decimal places
   * to round to is 0.
   *
   * @param input Column of values to be rounded
   * @param round Rounding method(either HALF_UP or HALF_EVEN)
   * @return a new ColumnVector with rounded values.
   */
  public static ColumnVector round(ColumnView input, RoundMode round) {
    return round(input, 0, round);
  }

  /**
   * Rounds all the values in a column to the specified number of decimal places with HALF_UP
   * (default) as Rounding method.
   *
   * @param input         Column of values to be rounded
   * @param decimalPlaces Number of decimal places to round to. If negative, this
   *                      specifies the number of positions to the left of the decimal point.
   * @return a new ColumnVector with rounded values.
   */
  public static ColumnVector round(ColumnView input, int decimalPlaces) {
    return round(input, decimalPlaces, RoundMode.HALF_UP);
  }

  /**
   * Rounds all the values in a column with these default values:
   * decimalPlaces = 0
   * Rounding method = RoundMode.HALF_UP
   *
   * @param input Column of values to be rounded
   * @return a new ColumnVector with rounded values.
   */
  public static ColumnVector round(ColumnView input) {
    return round(input, 0, RoundMode.HALF_UP);
  }

  private static native long multiply(long leftHandle, boolean isLeftCv, long rightHandle,
      boolean isRightCv, boolean isAnsiMode, boolean isTryMode);

  private static native long round(long nativeHandle, int decimalPlaces, int roundingMethod,
      boolean isAnsiMode);
}
