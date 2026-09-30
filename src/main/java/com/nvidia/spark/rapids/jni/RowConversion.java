/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

/** Utility class for converting between column major and row major data */
public class RowConversion {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * For details about how this method functions refer to
   * {@link #convertToRowsFixedWidthOptimized()}.
   *
   * The only thing different between this method and {@link #convertToRowsFixedWidthOptimized()}
   * is that this can handle rougly 250M columns while {@link #convertToRowsFixedWidthOptimized()}
   * can only handle columns less than 100
   */
  public static ColumnVector[] convertToRows(Table table) {
    long[] ptrs = convertToRows(table.getNativeView());
    ColumnVector[] ret = new ColumnVector[ptrs.length];
    for (int i = 0; i < ptrs.length; i++) {
      ret[i] = new ColumnVector(ptrs[i]);
    }
    return ret;
  }

  /**
   * Convert this table of columns into a row major format that is useful for interacting with other
   * systems that do row major processing of the data. Currently only fixed-width column types are
   * supported.
   * <p/>
   * The output is one or more ColumnVectors that are lists of bytes. A ColumnVector that is a
   * list of bytes can have at most 2GB of data stored in it. Multiple ColumnVectors are returned
   * if not all of the data can fit in a single one.
   * <p/>
   * Each row in the returned ColumnVector array corresponds to a row in the input table. The rows
   * will be in the same order as the input Table. The first ColumnVector in the array will hold
   * the first N rows followed by the second ColumnVector and so on.  The following illustrates
   * this and also shows some of the internal structure that will be explained later.
   * <p/><pre>
   * result[0]:
   *  | row 0 | validity for row 0 | padding |
   *  ...
   *  | row N | validity for row N | padding |
   *  result[1]:
   *  |row N+1 | validity for row N+1 | padding |
   *  ...
   * </pre>
   *
   * The format of each row is similar in layout to a C struct where each column will have padding
   * in front of it to align it properly. Each row has padding inserted at the end so the next row
   * is aligned to a 64-bit boundary. This is so that the first column will always start at the
   * beginning (first byte) of the list of bytes and each row has a consistent layout for fixed
   * width types.
   * <p/>
   * Validity bytes are added to the end of the row. There will be one byte for each 8 columns in a
   * row. Because the validity is byte aligned there is no padding between it and the last column
   * in the row.
   * <p/>
   * For example a table consisting of the following columns A, B, C with the corresponding types
   * <p/><pre>
   *   | A - BOOL8 (8-bit) | B - INT16 (16-bit) | C - DURATION_DAYS (32-bit) |
   * </pre>
   * <p/>
   *  Will have a layout that looks like
   *  <p/><pre>
   *  | A_0 | P | B_0 | B_1 | C_0 | C_1 | C_2 | C_3 | V0 | P | P | P | P | P | P | P |
   * </pre>
   * <p/>
   * In this P corresponds to a byte of padding, [LETTER]_[NUMBER] represents the NUMBER
   * byte of the corresponding LETTER column, and V[NUMBER] is a validity byte for the `NUMBER * 8`
   * to `(NUMBER + 1) * 8` columns.
   * <p/>
   * The order of the columns will not be changed, but to reduce the total amount of padding it is
   * recommended to order the columns in the following way.
   * <p/>
   * <ol>
   *  <li>64-bit columns</li>
   *  <li>32-bit columns</li>
   *  <li>16-bit columns</li>
   *  <li>8-bit columns</li>
   * </ol>
   * <p/>
   * This way padding is only inserted at the end of a row to make the next column 64-bit aligned.
   * So for the example above if the columns were ordered C, B, A the layout would be.
   * <pre>
   * | C_0 | C_1 | C_2 | C_3 | B_0 | B_1 | A_0 | V0 |
   * </pre>
   * This would have reduced the overall size of the data transferred by half.
   * <p/>
   * One of the main motivations for doing a row conversion on the GPU is to avoid cache problems
   * when walking through columnar data on the CPU in a row wise manner. If you are not transferring
   * very many columns it is likely to be more efficient to just pull back the columns and walk
   * through them. This is especially true of a single column of fixed width data. The extra
   * padding will slow down the transfer and looking at only a handful of buffers is not likely to
   * cause cache issues.
   * <p/>
   * There are some limits on the size of a single row.  If the row is larger than 1KB this will
   * throw an exception.
   */
  public static ColumnVector[] convertToRowsFixedWidthOptimized(Table table) {
    long[] ptrs = convertToRowsFixedWidthOptimized(table.getNativeView());
    ColumnVector[] ret = new ColumnVector[ptrs.length];
    for (int i = 0; i < ptrs.length; i++) {
      ret[i] = new ColumnVector(ptrs[i]);
    }
    return ret;
  }

  /**
   * Convert a column of list of bytes that is formatted like the output from `convertToRows`
   * and convert it back to a table.
   *
   * NOTE: This method doesn't support nested types
   *
   * @param vec the row data to process.
   * @param schema the types of each column.
   * @return the parsed table.
   */
  public static Table convertFromRows(ColumnView vec, DType ... schema) {
    int[] types = new int[schema.length];
    int[] scale = new int[schema.length];
    for (int i = 0; i < schema.length; i++) {
      types[i] = schema[i].getTypeId().getNativeId();
      scale[i] = schema[i].getScale();

    }
    return new Table(convertFromRows(vec.getNativeView(), types, scale));
  }

  /**
   * Convert a column of list of bytes that is formatted like the output from `convertToRows`
   * and convert it back to a table.
   *
   * NOTE: This method doesn't support nested types
   *
   * @param vec the row data to process.
   * @param schema the types of each column.
   * @return the parsed table.
   */
  public static Table convertFromRowsFixedWidthOptimized(ColumnView vec, DType ... schema) {
    int[] types = new int[schema.length];
    int[] scale = new int[schema.length];
    for (int i = 0; i < schema.length; i++) {
      types[i] = schema[i].getTypeId().getNativeId();
      scale[i] = schema[i].getScale();

    }
    return new Table(convertFromRowsFixedWidthOptimized(vec.getNativeView(), types, scale));
  }

  private static native long[] convertToRows(long nativeHandle);
  private static native long[] convertToRowsFixedWidthOptimized(long nativeHandle);

  private static native long[] convertFromRows(long nativeColumnView, int[] types, int[] scale);
  private static native long[] convertFromRowsFixedWidthOptimized(long nativeColumnView, int[] types, int[] scale);
}
