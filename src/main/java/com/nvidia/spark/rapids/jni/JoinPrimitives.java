/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.GatherMap;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.ast.CompiledExpression;

/**
 * Join primitive operations for composable join implementations.
 * <p>
 * This class provides low-level join primitives that can be composed to build
 * various join operations. These primitives allow for flexible join strategies
 * and optimization at higher levels.
 * </p>
 */
public class JoinPrimitives {

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Helper to convert gather map data from JNI to GatherMap array
   */
  private static GatherMap[] gatherMapsFromJNI(long[] gatherMapData) {
    long bufferSize = gatherMapData[0];
    long leftAddr = gatherMapData[1];
    long leftHandle = gatherMapData[2];
    long rightAddr = gatherMapData[3];
    long rightHandle = gatherMapData[4];
    GatherMap[] maps = new GatherMap[2];
    maps[0] = new GatherMap(DeviceMemoryBuffer.fromRmm(leftAddr, bufferSize, leftHandle));
    maps[1] = new GatherMap(DeviceMemoryBuffer.fromRmm(rightAddr, bufferSize, rightHandle));
    return maps;
  }

  /**
   * Helper to convert single gather map data from JNI to GatherMap
   */
  private static GatherMap gatherMapFromJNI(long bufferAddr, long bufferSize, long bufferHandle) {
    return new GatherMap(DeviceMemoryBuffer.fromRmm(bufferAddr, bufferSize, bufferHandle));
  }

  // =============================================================================
  // BASIC EQUALITY JOINS (Sort-Merge and Hash)
  // =============================================================================

  /**
   * Perform an inner join using sort-merge algorithm.
   * <p>
   * Returns gather maps for matching rows. Does not optimize by swapping tables
   * based on size - that optimization should be done at a higher level if desired.
   * </p>
   *
   * @param leftKeys The left table for equality comparison
   * @param rightKeys The right table for equality comparison
   * @param isRightSorted Whether the right table is pre-sorted
   * @param compareNullsEqual Whether nulls in equality keys should be considered equal
   * @return An array of two GatherMaps: [left_map, right_map]
   */
  public static GatherMap[] sortMergeInnerJoin(Table leftKeys,
                                               Table rightKeys,
                                               boolean isRightSorted,
                                               boolean compareNullsEqual) {
    long[] result = nativeSortMergeInnerJoin(
      leftKeys.getNativeView(),
      rightKeys.getNativeView(),
      isRightSorted,
      compareNullsEqual);

    return gatherMapsFromJNI(result);
  }

  /**
   * Perform an inner join using sort-merge algorithm.
   * <p>
   * Use {@link #sortMergeInnerJoin(Table, Table, boolean, boolean)} instead. This
   * overload remains only so existing callers keep compiling, and will be removed
   * once they have moved over.
   * </p>
   *
   * @param leftKeys The left table for equality comparison
   * @param rightKeys The right table for equality comparison
   * @param isLeftSorted Ignored. The underlying cudf API no longer uses it.
   * @param isRightSorted Whether the right table is pre-sorted
   * @param compareNullsEqual Whether nulls in equality keys should be considered equal
   * @return An array of two GatherMaps: [left_map, right_map]
   */
  public static GatherMap[] sortMergeInnerJoin(Table leftKeys,
                                               Table rightKeys,
                                               boolean isLeftSorted,
                                               boolean isRightSorted,
                                               boolean compareNullsEqual) {
    return sortMergeInnerJoin(leftKeys, rightKeys, isRightSorted, compareNullsEqual);
  }

  /**
   * Perform an inner join using hash join algorithm.
   * <p>
   * Returns gather maps for matching rows. Does not optimize by swapping tables
   * based on size - that optimization should be done at a higher level if desired.
   * </p>
   *
   * @param leftKeys The left table for equality comparison
   * @param rightKeys The right table for equality comparison
   * @param compareNullsEqual Whether nulls in equality keys should be considered equal
   * @return An array of two GatherMaps: [left_map, right_map]
   */
  public static GatherMap[] hashInnerJoin(Table leftKeys,
                                          Table rightKeys,
                                          boolean compareNullsEqual) {
    long[] result = nativeHashInnerJoin(
      leftKeys.getNativeView(),
      rightKeys.getNativeView(),
      compareNullsEqual);
    
    return gatherMapsFromJNI(result);
  }

  // =============================================================================
  // AST FILTERING
  // =============================================================================

  /**
   * Filter gather maps using an AST conditional expression.
   * <p>
   * Takes existing gather maps and filters them by evaluating the AST expression
   * on the corresponding rows from the left and right tables. Only pairs where
   * the expression evaluates to true are kept.
   * </p>
   * <p>
   * <b>NOTE:</b> The input gather maps are not modified or closed by this method.
   * The caller is responsible for closing them when no longer needed.
   * </p>
   *
   * @param leftGatherMap Input gather map for left table
   * @param rightGatherMap Input gather map for right table (must be same size as leftGatherMap)
   * @param leftTable The left table for conditional expression evaluation
   * @param rightTable The right table for conditional expression evaluation
   * @param condition The compiled conditional expression (AST) to evaluate (must return boolean)
   * @return An array of two filtered GatherMaps: [left_map, right_map]
   */
  public static GatherMap[] filterGatherMapsByAST(GatherMap leftGatherMap,
                                                   GatherMap rightGatherMap,
                                                   Table leftTable,
                                                   Table rightTable,
                                                   CompiledExpression condition) {
    long[] result = nativeFilterGatherMapsByAST(
      leftGatherMap.getBufferAddress(),
      leftGatherMap.getBufferLength(),
      rightGatherMap.getBufferAddress(),
      rightGatherMap.getBufferLength(),
      leftTable.getNativeView(),
      rightTable.getNativeView(),
      condition.getNativeHandle());
    
    return gatherMapsFromJNI(result);
  }

  // =============================================================================
  // MAKE OUTER JOINS
  // =============================================================================

  /**
   * Convert inner join gather maps to left outer join gather maps.
   * <p>
   * Takes gather maps from an inner join and adds entries for unmatched left rows.
   * Unmatched left rows will have right indices set to INT32_MIN (sentinel value for null).
   * </p>
   * <p>
   * <b>NOTE:</b> The input gather maps are not modified or closed by this method.
   * The caller is responsible for closing them when no longer needed.
   * </p>
   *
   * @param leftGatherMap Inner join gather map for left table
   * @param rightGatherMap Inner join gather map for right table (must be same size as leftGatherMap)
   * @param leftTableSize Number of rows in the left table
   * @param rightTableSize Number of rows in the right table
   * @return An array of two GatherMaps: [left_map, right_map] for left outer join
   */
  public static GatherMap[] makeLeftOuter(GatherMap leftGatherMap,
                                         GatherMap rightGatherMap,
                                         int leftTableSize,
                                         int rightTableSize) {
    long[] result = nativeMakeLeftOuter(
      leftGatherMap.getBufferAddress(),
      leftGatherMap.getBufferLength(),
      rightGatherMap.getBufferAddress(),
      rightGatherMap.getBufferLength(),
      leftTableSize,
      rightTableSize);
    
    return gatherMapsFromJNI(result);
  }


  /**
   * Convert inner join gather maps to full outer join gather maps.
   * <p>
   * Takes gather maps from an inner join and adds entries for both unmatched left and right rows.
   * Unmatched left rows will have right indices set to INT32_MIN (sentinel value for null).
   * Unmatched right rows will have left indices set to INT32_MIN (sentinel value for null).
   * </p>
   * <p>
   * <b>NOTE:</b> The input gather maps are not modified or closed by this method.
   * The caller is responsible for closing them when no longer needed.
   * </p>
   *
   * @param leftGatherMap Inner join gather map for left table
   * @param rightGatherMap Inner join gather map for right table (must be same size as leftGatherMap)
   * @param leftTableSize Number of rows in the left table
   * @param rightTableSize Number of rows in the right table
   * @return An array of two GatherMaps: [left_map, right_map] for full outer join
   */
  public static GatherMap[] makeFullOuter(GatherMap leftGatherMap,
                                          GatherMap rightGatherMap,
                                          int leftTableSize,
                                          int rightTableSize) {
    long[] result = nativeMakeFullOuter(
      leftGatherMap.getBufferAddress(),
      leftGatherMap.getBufferLength(),
      rightGatherMap.getBufferAddress(),
      rightGatherMap.getBufferLength(),
      leftTableSize,
      rightTableSize);
    
    return gatherMapsFromJNI(result);
  }

  // =============================================================================
  // MAKE SEMI/ANTI JOINS
  // =============================================================================

  /**
   * Convert inner join gather maps to semi join result.
   * <p>
   * Takes the gather map from an inner join and returns unique indices.
   * Each row appears at most once in the result. The right gather map is not needed
   * since semi join only cares about which rows have matches.
   * </p>
   * <p>
   * <b>NOTE:</b> The input gather map is not modified or closed by this method.
   * The caller is responsible for closing it when no longer needed.
   * </p>
   *
   * @param gatherMap Inner join gather map
   * @param tableSize Number of rows in the table that the indices are from
   * @return A GatherMap of unique indices
   */
  public static GatherMap makeSemi(GatherMap gatherMap, int tableSize) {
    long[] result = nativeMakeSemi(
      gatherMap.getBufferAddress(),
      gatherMap.getBufferLength(),
      tableSize);
    
    return gatherMapFromJNI(result[1], result[0], result[2]);
  }

  /**
   * Convert semi join result to anti join result.
   * <p>
   * Takes a gather map of matched indices and returns indices of unmatched rows.
   * This is the complement of the semi join.
   * </p>
   * <p>
   * <b>NOTE:</b> The input gather map is not modified or closed by this method.
   * The caller is responsible for closing it when no longer needed.
   * </p>
   *
   * @param gatherMap Semi join result (gather map of matched indices)
   * @param tableSize Number of rows in the table that the indices are from
   * @return A GatherMap of unmatched indices
   */
  public static GatherMap makeAnti(GatherMap gatherMap,
                                   int tableSize) {
    long[] result = nativeMakeAnti(
      gatherMap.getBufferAddress(),
      gatherMap.getBufferLength(),
      tableSize);
    
    return gatherMapFromJNI(result[1], result[0], result[2]);
  }

  // =============================================================================
  // PARTITIONED JOIN SUPPORT
  // =============================================================================

  /**
   * Get boolean column indicating which rows were matched.
   * <p>
   * For partitioned joins, returns a boolean column where true indicates the row
   * at that index was matched. This allows combining results from multiple partitions
   * by OR-ing the boolean columns together.
   * </p>
   * <p>
   * <b>NOTE:</b> The input gather map is not modified or closed by this method.
   * The caller is responsible for closing it when no longer needed.
   * </p>
   *
   * @param gatherMap Gather map from a join operation
   * @param tableSize Total number of rows in the source table
   * @return Boolean ColumnVector where true indicates row was matched
   */
  public static ColumnVector getMatchedRows(GatherMap gatherMap,
                                            int tableSize) {
    return new ColumnVector(nativeGetMatchedRows(
      gatherMap.getBufferAddress(),
      gatherMap.getBufferLength(),
      tableSize));
  }

  // =============================================================================
  // NATIVE METHOD DECLARATIONS
  // =============================================================================

  private static native long[] nativeSortMergeInnerJoin(
    long leftKeys,
    long rightKeys,
    boolean isRightSorted,
    boolean compareNullsEqual);

  private static native long[] nativeHashInnerJoin(
    long leftKeys,
    long rightKeys,
    boolean compareNullsEqual);

  private static native long[] nativeFilterGatherMapsByAST(
    long leftBufferAddress,
    long leftBufferLength,
    long rightBufferAddress,
    long rightBufferLength,
    long leftTable,
    long rightTable,
    long condition);

  private static native long[] nativeMakeLeftOuter(
    long leftBufferAddress,
    long leftBufferLength,
    long rightBufferAddress,
    long rightBufferLength,
    int leftTableSize,
    int rightTableSize);

  private static native long[] nativeMakeFullOuter(
    long leftBufferAddress,
    long leftBufferLength,
    long rightBufferAddress,
    long rightBufferLength,
    int leftTableSize,
    int rightTableSize);

  private static native long[] nativeMakeSemi(
    long leftBufferAddress,
    long leftBufferLength,
    int leftTableSize);

  private static native long[] nativeMakeAnti(
    long leftBufferAddress,
    long leftBufferLength,
    int leftTableSize);

  private static native long nativeGetMatchedRows(
    long bufferAddress,
    long bufferLength,
    int tableSize);
}

