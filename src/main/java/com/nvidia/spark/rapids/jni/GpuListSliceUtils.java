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

public class GpuListSliceUtils {
    static{
        NativeDepsLoader.loadNativeDeps();
    }

    /**
     * @brief Slices each row of a lists column according to the requested `start` and `length`.
     *
     * The indices cannot be zero; they start at 1, or from the end if negative (the value of -1
     * refers to the last element in the list). If any index in start is outside [-n, n] (where n
     * is the number of elements in that row), the result for that row is an empty list.
     *
     * If length is zero, the result for that row is an empty list. If there are not enough elements
     * from the specified start to the end of the target list, the number of elements in the result
     * list will be less than the specified length.
     *
     * Null handling: For each row, if corresponding input is null, the result row will be null.
     *
     * @code{.pseudo}
     * input_column = [
     *   [1, 2, 3, 4],
     *   [5, 6, 7],
     *   [8, 9]
     * ]
     *
     * start = 2, length = 2
     *
     * result = [
     *   [2, 3],
     *   [6, 7],
     *   [9]
     * ]
     * @endcode
     *
     * @param cv    the column of lists to slice
     * @param start the integer offset at which to begin slicing
     * @param length the integer length of elements to include in the slice
     * @param checkStartLength whether to check the validity of start and length, when set to false,
     * the caller is responsible for ensuring the validity of start and length, otherwise the
     * behavior is undefined if there are any invalid values
     * @return a new {@code ColumnVector} containing the sliced lists
     */
    public static ColumnVector listSlice(ColumnView cv, int start, int length) {
        return listSlice(cv, start, length, true);
    }

    public static ColumnVector listSlice(ColumnView cv, int start, int length, boolean checkStartLength) {
        return new ColumnVector(listSliceIntInt(cv.getNativeView(), start, length, checkStartLength));
    }

    /**
     * @brief Slices each row of a lists column according to the requested `start` and `length`.
     *
     * The indices cannot be zero; they start at 1, or from the end if negative (the value of -1
     * refers to the last element in the list). If any index in start is outside [-n, n] (where n
     * is the number of elements in that row), the result for that row is an empty list.
     *
     * If length is zero, the result for that row is an empty list. If there are not enough elements
     * from the specified start to the end of the target list, the number of elements in the result
     * list will be less than the specified length.
     *
     * Null handling: For each row, if either corresponding input or length element is null, the
     * result row will be null.
     *
     * @code{.pseudo}
     * input_column = [
     *   [1, 2, 3, 4],
     *   [5, 6, 7],
     *   [8, 9]
     * ]
     *
     * start = -2, length = [3, 2, 1]
     *
     * result = [
     *   [3, 4],
     *   [6, 7],
     *   [8]
     * ]
     * @endcode
     *
     * @param cv    the column of lists to slice
     * @param start the integer offset at which to begin slicing
     * @param length the column view specifying the lengths for each list slice
     * @param checkStartLength whether to check the validity of start and length, when set to false,
     * the caller is responsible for ensuring the validity of start and length, otherwise the
     * behavior is undefined if there are any invalid values
     * @return a new {@code ColumnVector} containing the sliced lists
     */
    public static ColumnVector listSlice(ColumnView cv, int start, ColumnView length) {
        return listSlice(cv, start, length, true);
    }

    public static ColumnVector listSlice(ColumnView cv, int start, ColumnView length, boolean checkStartLength) {
        return new ColumnVector(listSliceIntCol(cv.getNativeView(), start, length.getNativeView(), checkStartLength));
    }

    /**
     * @brief Slices each row of a lists column according to the requested `start` and `length`.
     *
     * The indices cannot be zero; they start at 1, or from the end if negative (the value of -1
     * refers to the last element in the list). If any index in start is outside [-n, n] (where n
     * is the number of elements in that row), the result for that row is an empty list.
     *
     * If length is zero, the result for that row is an empty list. If there are not enough elements
     * from the specified start to the end of the target list, the number of elements in the result
     * list will be less than the specified length.
     *
     * Null handling: For each row, if either corresponding input or start index is null, the result
     * row will be null.
     *
     * @code{.pseudo}
     * input_column = [
     *   [1, 2, 3, 4],
     *   [5, 6, 7],
     *   [8, 9]
     * ]
     *
     * start = [2, 1, -1], length = 2
     *
     * result = [
     *   [2, 3],
     *   [5, 6],
     *   [9]
     * ]
     * @endcode
     *
     * @param cv    the column of lists to slice
     * @param start the column view specifying the start offsets for each list slice
     * @param length the integer length of elements to include in the slice
     * @param checkStartLength whether to check the validity of start and length, when set to false,
     * the caller is responsible for ensuring the validity of start and length, otherwise the
     * behavior is undefined if there are any invalid values
     * @return a new {@code ColumnVector} containing the sliced lists
     */
    public static ColumnVector listSlice(ColumnView cv, ColumnView start, int length) {
        return listSlice(cv, start, length, true);
    }

    public static ColumnVector listSlice(ColumnView cv, ColumnView start, int length, boolean checkStartLength) {
        return new ColumnVector(listSliceColInt(cv.getNativeView(), start.getNativeView(), length, checkStartLength));
    }

    /**
     * @brief Slices each row of a lists column according to the requested `start` and `length`.
     *
     * The indices cannot be zero; they start at 1, or from the end if negative (the value of -1
     * refers to the last element in the list). If any index in start is outside [-n, n] (where n
     * is the number of elements in that row), the result for that row is an empty list.
     *
     * If length is zero, the result for that row is an empty list. If there are not enough elements
     * from the specified start to the end of the target list, the number of elements in the result
     * list will be less than the specified length.
     *
     * Null handling: For each row, if any corresponding input, start index, or length element is
     * null, the result row will be null.
     *
     * @code{.pseudo}
     * input_column = [
     *   [1, 2, 3],
     *   [4, null, 5],
     *   null,
     *   [],
     *   [null],
     *   [6, 7, 8],
     *   [9, 10]
     * ]
     *
     * start = [1, -2, 2, 3, -10, -3, null], length = [0, 2, 2, null, 4, 10, 1]
     *
     * result = [
     *   [],
     *   [null, 5],
     *   null,
     *   null,
     *   [],
     *   [6, 7, 8],
     *   null
     * ]
     * @endcode
     *
     * @param cv    the column of lists to slice
     * @param start the column view specifying the start offsets for each list slice
     * @param length the column view specifying the lengths for each list slice
     * @param checkStartLength whether to check the validity of start and length, when set to false,
     * the caller is responsible for ensuring the validity of start and length, otherwise the
     * behavior is undefined if there are any invalid values
     * @return a new {@code ColumnVector} containing the sliced lists
     */
    public static ColumnVector listSlice(ColumnView cv, ColumnView start, ColumnView length) {
        return listSlice(cv, start, length, true);
    }

    public static ColumnVector listSlice(ColumnView cv, ColumnView start, ColumnView length, boolean checkStartLength) {
        return new ColumnVector(listSliceColCol(cv.getNativeView(), start.getNativeView(), length.getNativeView(), checkStartLength));
    }

    private static native long listSliceIntInt(long listColumnView, int start, int length, boolean check_start_length);

    private static native long listSliceIntCol(long listColumnView, int start, long lengthColumnView, boolean check_start_length);

    private static native long listSliceColInt(long listColumnView, long startColumnView, int length, boolean check_start_length);

    private static native long listSliceColCol(long listColumnView, long startColumnView, long lengthColumnView, boolean check_start_length);
}
