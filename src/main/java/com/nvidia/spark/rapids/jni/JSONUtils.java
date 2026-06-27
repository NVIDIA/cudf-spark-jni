/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

import java.util.List;

public class JSONUtils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public static final int MAX_PATH_DEPTH = getMaxJSONPathDepth();

  public enum PathInstructionType {
    WILDCARD,
    INDEX,
    NAMED
  }

  public static class PathInstructionJni {
    // type: byte, name: String, index: int
    private final byte type;
    private final String name;
    private final int index;

    public PathInstructionJni(PathInstructionType type, String name, long index) {
      this.type = (byte) type.ordinal();
      this.name = name;
      if (index > Integer.MAX_VALUE) {
        throw new IllegalArgumentException("index is too large " + index);
      }
      this.index = (int) index;
    }

    public PathInstructionJni(PathInstructionType type, String name, int index) {
      this.type = (byte) type.ordinal();
      this.name = name;
      this.index = index;
    }
  }

  /**
   * Extract a JSON path from a JSON column. The path is processed in a Spark compatible way.
   * @param input the string column containing JSON
   * @param pathInstructions the instructions for the path processing
   * @return the result of processing the path
   */
  public static ColumnVector getJsonObject(ColumnVector input, PathInstructionJni[] pathInstructions) {
    assert (input.getType().equals(DType.STRING)) : "Input must be of STRING type";
    int numTotalInstructions = pathInstructions.length;
    byte[] typeNums = new byte[numTotalInstructions];
    String[] names = new String[numTotalInstructions];
    int[] indexes = new int[numTotalInstructions];

    for (int i = 0; i < pathInstructions.length; i++) {
      PathInstructionJni current = pathInstructions[i];
      typeNums[i] = current.type;
      names[i] = current.name;
      indexes[i] = current.index;
    }
    return new ColumnVector(getJsonObject(input.getNativeView(), typeNums, names, indexes));
  }

  /**
   * Extract multiple JSON paths from a JSON column. The paths are processed in a Spark
   * compatible way.
   * @param input the string column containing JSON
   * @param paths the instructions for multiple paths
   * @return the result of processing each path in the order that they were passed in
   */
  public static ColumnVector[] getJsonObjectMultiplePaths(ColumnVector input,
                                                          List<List<PathInstructionJni>> paths) {
    return getJsonObjectMultiplePaths(input, paths, -1, -1);
  }

  /**
   * Extract multiple JSON paths from a JSON column. The paths are processed in a Spark
   * compatible way.
   * @param input the string column containing JSON
   * @param paths the instructions for multiple paths
   * @param memoryBudgetBytes a budget that is used to limit the amount of memory
   *                          that is used when processing the paths. This is a soft limit.
   *                          A value <= 0 disables this and all paths will be processed in parallel.
   * @param parallelOverride Set a maximum number of paths to be processed in parallel. The memory
   *                         budget can limit how many paths can be processed in parallel. This overrides
   *                         that automatically calculated value with a set value for benchmarking purposes.
   *                         A value <= 0 disables this.
   * @return the result of processing each path in the order that they were passed in
   */
  public static ColumnVector[] getJsonObjectMultiplePaths(ColumnVector input,
                                                          List<List<PathInstructionJni>> paths,
                                                          long memoryBudgetBytes,
                                                          int parallelOverride) {
    assert (input.getType().equals(DType.STRING)) : "Input must be of STRING type";
    int[] pathOffsets = new int[paths.size() + 1];
    int offset = 0;
    for (int i = 0; i < paths.size(); i++) {
      pathOffsets[i] = offset;
      offset += paths.get(i).size();
    }
    pathOffsets[paths.size()] = offset;
    int numTotalInstructions = offset;
    byte[] typeNums = new byte[numTotalInstructions];
    String[] names = new String[numTotalInstructions];
    int[] indexes = new int[numTotalInstructions];

    for (int i = 0; i < paths.size(); i++) {
      for (int j = 0; j < paths.get(i).size(); j++) {
        PathInstructionJni current = paths.get(i).get(j);
        typeNums[pathOffsets[i] + j] = current.type;
        names[pathOffsets[i] + j] = current.name;
        indexes[pathOffsets[i] + j] = current.index;
      }
    }
    long[] ptrs = getJsonObjectMultiplePaths(input.getNativeView(), typeNums,
        names, indexes, pathOffsets, memoryBudgetBytes, parallelOverride);
    ColumnVector[] ret = new ColumnVector[ptrs.length];
    for (int i = 0; i < ptrs.length; i++) {
      ret[i] = new ColumnVector(ptrs[i]);
    }
    return ret;
  }


  /**
   * The type of the map values produced by {@link #extractRawMapFromJsonString(ColumnView,
   * JSONOptions, MapValueType)}, selecting how each JSON value is interpreted.
   */
  public enum MapValueType {
    /** Map values are scalar strings, producing a {@code List<Struct<String,String>>} column. */
    STRING,
    /**
     * Map values are JSON arrays of strings, producing a
     * {@code List<Struct<String,List<String>>>} column.
     */
    ARRAY_OF_STRING
  }

  /**
   * Extract key-value pairs for each output map from the given json strings. These key-value are
   * copied directly as substrings of the input without any type conversion.
   * <p/>
   * Since there is not any validity check, the output of this function may be different from
   * what generated by Spark's `from_json` function. Situations that can lead to
   * different/incorrect outputs may include:<br>
   * - The value in the input json string is invalid, such as 'abc' value for an integer key.<br>
   * - The value string can be non-clean format for floating-point type, such as '1.00000'.
   * <p/>
   * The output of these situations should all be NULL or a value '1.0', respectively. However, this
   * function will just simply copy the input value strings to the output.
   *
   * @param input The input strings column in which each row specifies a json object
   * @param opts The options for parsing JSON strings
   * @return A map column (i.e., a column of type {@code List<Struct<String,String>>}) in
   *         which the key-value pairs are extracted directly from the input json strings
   * @deprecated Retained for backward compatibility. Use {@link #extractRawMapFromJsonString(ColumnView, JSONOptions, MapValueType)}
   *             with {@link MapValueType#STRING}, or {@link MapValueType#ARRAY_OF_STRING} for the
   *             array-valued variant that produces {@code List<Struct<String,List<String>>>}.
   */
  @Deprecated
  public static ColumnVector extractRawMapFromJsonString(ColumnView input, JSONOptions opts) {
    return extractRawMapFromJsonString(input, opts, MapValueType.STRING);
  }

  /**
   * Extract key-value pairs for each output map from the given json strings. The keys and values
   * are copied directly as substrings of the input without any type conversion.
   * <p/>
   * The {@code valueType} parameter selects how each JSON value is interpreted and therefore the
   * shape of the output map column:<br>
   * - {@link MapValueType#STRING}: values are scalar strings, producing a column of type
   *   {@code List<Struct<String,String>>}.<br>
   * - {@link MapValueType#ARRAY_OF_STRING}: values are JSON arrays of strings, producing a column of
   *   type {@code List<Struct<String,List<String>>>} (targets Spark's `from_json` for the schema
   *   {@code MapType[StringType, ArrayType[StringType]]}).
   * <p/>
   * For {@link MapValueType#ARRAY_OF_STRING}, each value is expected to be a JSON array. A value
   * that is the JSON {@code null} literal yields a null inner list while keeping the map row,
   * whereas a value that is non-null but not an array (a scalar or an object) is treated as a bad
   * record and nullifies the entire output map row, matching Spark's `from_json`. Array elements
   * are copied as raw de-quoted substrings, and a literal {@code null} element produces a null
   * element.
   * <p/>
   * Beyond that structural handling, keys and values are copied verbatim with no type or format
   * validation, so the output of this function may be different from
   * what generated by Spark's `from_json` function. Situations that can lead to
   * different/incorrect outputs may include:<br>
   * - The value in the input json string is invalid, such as 'abc' value for an integer key.<br>
   * - The value string can be non-clean format for floating-point type, such as '1.00000'.
   * <p/>
   * The output of these situations should all be NULL or a value '1.0', respectively. However, this
   * function will just simply copy the input value strings to the output.
   *
   * @param input The input strings column in which each row specifies a json object
   * @param opts The options for parsing JSON strings
   * @param valueType The interpretation of the map values, selecting the output column shape
   * @return A map column whose value type is determined by {@code valueType}, in which the keys and
   *         values are extracted directly from the input json strings
   */
  public static ColumnVector extractRawMapFromJsonString(ColumnView input, JSONOptions opts,
                                                         MapValueType valueType) {
    assert (input.getType().equals(DType.STRING)) : "Input must be of STRING type";
    switch (valueType) {
      case STRING:
        return new ColumnVector(extractRawMapFromJsonString(input.getNativeView(),
            opts.isNormalizeSingleQuotes(),
            opts.leadingZerosAllowed(),
            opts.nonNumericNumbersAllowed(),
            opts.unquotedControlChars()));
      case ARRAY_OF_STRING:
        return new ColumnVector(extractRawMapArrayFromJsonString(input.getNativeView(),
            opts.isNormalizeSingleQuotes(),
            opts.leadingZerosAllowed(),
            opts.nonNumericNumbersAllowed(),
            opts.unquotedControlChars()));
      default:
        throw new IllegalArgumentException("Unsupported map value type: " + valueType);
    }
  }

  /**
   * Parse a JSON string into a struct column following by the given data schema.
   * <p/>
   * Many JSON options in the given {@code opts} parameter are ignored from passing down to the
   * native code. That is because these options are hard-coded with the same values in both the
   * plugin code and native code. Specifically:<br>
   * - {@code RecoverWithNull: true}<br>
   * - {@code MixedTypesAsStrings: true}<br>
   * - {@code NormalizeWhitespace: true}<br>
   * - {@code KeepQuotes: true}<br>
   * - {@code StrictValidation: true}<br>
   * - {@code Experimental: true}
   *
   * @param input The input strings column in which each row specifies a json object
   * @param schema The schema of the output struct column
   * @param opts The options for parsing JSON strings
   * @param isUSLocale Whether the current local is US locale, used when converting strings to
   *        decimal types
   * @return A struct column in which each row is parsed from the corresponding json string
   */
  public static ColumnVector fromJSONToStructs(ColumnView input, Schema schema, JSONOptions opts,
                                               boolean isUSLocale) {
    assert (input.getType().equals(DType.STRING)) : "Input must be of STRING type";
    return new ColumnVector(fromJSONToStructs(input.getNativeView(),
        schema.getFlattenedColumnNames(),
        schema.getFlattenedNumChildren(),
        schema.getFlattenedTypeIds(),
        schema.getFlattenedTypeScales(),
        schema.getFlattenedDecimalPrecisions(),
        opts.isNormalizeSingleQuotes(),
        opts.leadingZerosAllowed(),
        opts.nonNumericNumbersAllowed(),
        opts.unquotedControlChars(),
        isUSLocale));
  }

  /**
   * Convert from a strings column to a column with the desired type given by a data schema.
   *
   * @param input The input strings column
   * @param schema The schema of the output column
   * @param allowedNonNumericNumbers Whether non-numeric numbers are allowed, used when converting
   *        strings to float types
   * @param isUSLocale Whether the current local is US locale, used when converting strings to
   *        decimal types
   * @return A column with the desired data type
   */
  public static ColumnVector convertFromStrings(ColumnView input, Schema schema,
                                                boolean allowedNonNumericNumbers,
                                                boolean isUSLocale) {
    assert (input.getType().equals(DType.STRING)) : "Input must be of STRING type";
    return new ColumnVector(convertFromStrings(input.getNativeView(),
        schema.getFlattenedNumChildren(),
        schema.getFlattenedTypeIds(),
        schema.getFlattenedTypeScales(),
        schema.getFlattenedDecimalPrecisions(),
        allowedNonNumericNumbers,
        isUSLocale));
  }

  /**
   * Remove quotes from each string in the given strings column.
   * <p/>
   * If `nullifyIfNotQuoted` is true, an input string that is not quoted will result in a null.
   * Otherwise, the output will be the same as the unquoted input.
   *
   * @param input The input strings column
   * @param nullifyIfNotQuoted Whether to output a null row if the input string is not quoted
   * @return A strings column in which quotes are removed from all strings
   */
  public static ColumnVector removeQuotes(ColumnView input, boolean nullifyIfNotQuoted) {
    assert (input.getType().equals(DType.STRING)) : "Input must be of STRING type";
    return new ColumnVector(removeQuotes(input.getNativeView(), nullifyIfNotQuoted));
  }

  private static native int getMaxJSONPathDepth();

  private static native long getJsonObject(long input,
                                           byte[] typeNums,
                                           String[] names,
                                           int[] indexes);

  private static native long[] getJsonObjectMultiplePaths(long input,
                                                          byte[] typeNums,
                                                          String[] names,
                                                          int[] indexes,
                                                          int[] pathOffsets,
                                                          long memoryBudgetBytes,
                                                          int parallelOverride);

  private static native long extractRawMapFromJsonString(long input,
                                                         boolean normalizeSingleQuotes,
                                                         boolean leadingZerosAllowed,
                                                         boolean nonNumericNumbersAllowed,
                                                         boolean unquotedControlChars);

  private static native long extractRawMapArrayFromJsonString(long input,
                                                              boolean normalizeSingleQuotes,
                                                              boolean leadingZerosAllowed,
                                                              boolean nonNumericNumbersAllowed,
                                                              boolean unquotedControlChars);

  private static native long fromJSONToStructs(long input,
                                               String[] names,
                                               int[] numChildren,
                                               int[] typeIds,
                                               int[] typeScales,
                                               int[] typePrecision,
                                               boolean normalizeSingleQuotes,
                                               boolean leadingZerosAllowed,
                                               boolean nonNumericNumbersAllowed,
                                               boolean unquotedControlChars,
                                               boolean isUSLocale);

  private static native long convertFromStrings(long input,
                                                int[] numChildren,
                                                int[] typeIds,
                                                int[] typeScales,
                                                int[] typePrecision,
                                                boolean nonNumericNumbersAllowed,
                                                boolean isUSLocale);

  private static native long removeQuotes(long input, boolean nullifyIfNotQuoted);
}
