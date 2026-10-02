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
import ai.rapids.cudf.GatherMap;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.ast.BinaryOperation;
import ai.rapids.cudf.ast.BinaryOperator;
import ai.rapids.cudf.ast.ColumnReference;
import ai.rapids.cudf.ast.CompiledExpression;
import ai.rapids.cudf.ast.TableReference;
import org.junit.jupiter.api.Test;

import java.util.AbstractMap;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

public class JoinPrimitivesTest {

  /**
   * Sentinel value used in gather maps to indicate null/unmatched rows.
   * This must match the sentinel value used in the C++ join primitives implementation.
   * This is needed because the spark rapids fixup code assumes this.
   */
  private static final int GATHER_MAP_SENTINEL = -2147483648; // INT32_MIN

  // =============================================================================
  // HELPER METHODS FOR ROBUST TESTING
  // =============================================================================

  /**
   * Converts a gather map index to an Integer value or null if it's the sentinel.
   * @param index The gather map index
   * @param tableSize The size of the table being gathered from (unused with 
   * sentinel approach, but kept for future extensibility)
   * @return Integer value if valid index, null if sentinel value
   */
  private static Integer gatherMapIndexToValue(int index, long tableSize) {
    return (index == GATHER_MAP_SENTINEL) ? null : Integer.valueOf(index);
  }

  /**
   * Converts a single gather map to a Set of Integer values (nulls for OOB indices).
   * @param gatherMap The gather map column vector
   * @param tableSize The size of the table being gathered from
   * @return Set of Integer values (null represents OOB/unmatched)
   */
  private static Set<Integer> gatherMapToSet(HostColumnVector gatherMap, long tableSize) {
    Set<Integer> result = new HashSet<>();
    for (int i = 0; i < gatherMap.getRowCount(); i++) {
      result.add(gatherMapIndexToValue(gatherMap.getInt(i), tableSize));
    }
    return result;
  }

  /**
   * Converts a pair of gather maps to a Set of Map.Entry pairs.
   * @param leftMap The left gather map
   * @param rightMap The right gather map
   * @param leftTableSize The size of the left table
   * @param rightTableSize The size of the right table
   * @return Set of (left, right) pairs (nulls represent OOB/unmatched)
   */
  private static Set<Map.Entry<Integer, Integer>> gatherMapPairToSet(
      HostColumnVector leftMap, HostColumnVector rightMap, long leftTableSize, long rightTableSize) {
    assertEquals(leftMap.getRowCount(), rightMap.getRowCount(), 
        "Left and right gather maps must have same row count");
    
    Set<Map.Entry<Integer, Integer>> result = new HashSet<>();
    for (int i = 0; i < leftMap.getRowCount(); i++) {
      Integer leftVal = gatherMapIndexToValue(leftMap.getInt(i), leftTableSize);
      Integer rightVal = gatherMapIndexToValue(rightMap.getInt(i), rightTableSize);
      result.add(new AbstractMap.SimpleEntry<>(leftVal, rightVal));
    }
    return result;
  }

  /**
   * Asserts that two sets of gather map indices are equal, with detailed error message.
   */
  private static void assertGatherMapSetEquals(
      Set<Integer> expected, Set<Integer> actual, String message) {
    if (!expected.equals(actual)) {
      Set<Integer> missing = new HashSet<>(expected);
      missing.removeAll(actual);
      Set<Integer> extra = new HashSet<>(actual);
      extra.removeAll(expected);
      
      String errorMsg = message + "\n" +
          "Expected size: " + expected.size() + ", Actual size: " + actual.size() + "\n" +
          "Missing indices: " + missing + "\n" +
          "Extra indices: " + extra;
      fail(errorMsg);
    }
  }

  /**
   * Asserts that two sets of gather map pairs are equal, with detailed error message.
   */
  private static void assertGatherMapPairSetEquals(
      Set<Map.Entry<Integer, Integer>> expected, 
      Set<Map.Entry<Integer, Integer>> actual, 
      String message) {
    if (!expected.equals(actual)) {
      Set<Map.Entry<Integer, Integer>> missing = new HashSet<>(expected);
      missing.removeAll(actual);
      Set<Map.Entry<Integer, Integer>> extra = new HashSet<>(actual);
      extra.removeAll(expected);
      
      String errorMsg = message + "\n" +
          "Expected size: " + expected.size() + ", Actual size: " + actual.size() + "\n" +
          "Missing pairs: " + missing + "\n" +
          "Extra pairs: " + extra;
      fail(errorMsg);
    }
  }

  /**
   * Helper to create a set of expected gather map pairs.
   */
  @SafeVarargs
  private static Set<Map.Entry<Integer, Integer>> pairSet(Map.Entry<Integer, Integer>... pairs) {
    return new HashSet<>(Arrays.asList(pairs));
  }

  /**
   * Helper to create a Map.Entry pair.
   */
  private static Map.Entry<Integer, Integer> pair(Integer left, Integer right) {
    return new AbstractMap.SimpleEntry<>(left, right);
  }

  /**
   * High-level assertion for paired gather maps (e.g., inner, outer joins).
   * Copies gather maps to host, converts to set, and asserts equality.
   * 
   * @param gatherMaps The GPU gather map pair to validate
   * @param leftTableSize Size of the left table
   * @param rightTableSize Size of the right table
   * @param expected Expected set of (left, right) index pairs
   * @param message Error message prefix
   */
  private static void assertGatherMapPairs(
      GatherMap[] gatherMaps,
      long leftTableSize,
      long rightTableSize,
      Set<Map.Entry<Integer, Integer>> expected,
      String message) {
    
    long expectedSize = expected.size();
    int leftRowCount = (int)gatherMaps[0].getRowCount();
    int rightRowCount = (int)gatherMaps[1].getRowCount();
    assertEquals(expectedSize, leftRowCount, message + ": left gather map size mismatch (" + 
      leftRowCount + " != " + expectedSize + ")");
    assertEquals(expectedSize, rightRowCount, message + ": right gather map size mismatch (" + 
      rightRowCount + " != " + expectedSize + ")");

    try (HostColumnVector leftHost = gatherMaps[0].toColumnView(0, leftRowCount).copyToHost();
         HostColumnVector rightHost = gatherMaps[1].toColumnView(0, rightRowCount).copyToHost()) {
      
      Set<Map.Entry<Integer, Integer>> actual = gatherMapPairToSet(leftHost, rightHost,
        leftTableSize, rightTableSize);
      assertGatherMapPairSetEquals(expected, actual, message);
    }
  }

  /**
   * High-level assertion for single gather maps (e.g., semi, anti joins).
   * Copies gather map to host, converts to set, and asserts equality.
   * 
   * @param gatherMap The GPU gather map to validate
   * @param tableSize Size of the table being gathered from
   * @param expected Expected set of indices
   * @param message Error message prefix
   */
  private static void assertGatherMapIndices(
      GatherMap gatherMap,
      long tableSize,
      Set<Integer> expected,
      String message) {
    
    long expectedSize = expected.size();
    int rowCount = (int)gatherMap.getRowCount();
    assertEquals(expectedSize, rowCount, message + ": gather map size mismatch (" + 
      rowCount + " != " + expectedSize + ")");

    try (HostColumnVector host = gatherMap.toColumnView(0, rowCount).copyToHost()) {
      Set<Integer> actual = gatherMapToSet(host, tableSize);
      assertGatherMapSetEquals(expected, actual, message);
    }
  }

  // =============================================================================
  // BASIC EQUALITY JOINS (Sort-Merge and Hash)
  // =============================================================================

  @Test
  void testSortMergeInnerJoin() {
    // Left:  {0, 1, 2, 3}
    // Right: {1, 2, 4}
    // Expected matches: (1,0) and (2,1) - left/right indices

    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2, 3);
         ColumnVector rightKeys = ColumnVector.fromInts(1, 2, 4);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();
      int rightSize = (int) rightTable.getRowCount();

      GatherMap[] gatherMaps = JoinPrimitives.sortMergeInnerJoin(
          leftTable, rightTable, false, true);

      try {
        Set<Map.Entry<Integer, Integer>> expected = pairSet(pair(1, 0), pair(2, 1));
        assertGatherMapPairs(gatherMaps, leftSize, rightSize, expected, 
            "Sort merge inner join");
      } finally {
        for (GatherMap gm : gatherMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testHashInnerJoin() {
    // Same test as sort merge to verify hash join works correctly
    // Left:  {0, 1, 2, 3}
    // Right: {1, 2, 4}
    // Expected matches: (1,0) and (2,1) - left/right indices

    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2, 3);
         ColumnVector rightKeys = ColumnVector.fromInts(1, 2, 4);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();
      int rightSize = (int) rightTable.getRowCount();

      GatherMap[] gatherMaps = JoinPrimitives.hashInnerJoin(
          leftTable, rightTable, true);

      try {
        Set<Map.Entry<Integer, Integer>> expected = pairSet(pair(1, 0), pair(2, 1));
        assertGatherMapPairs(gatherMaps, leftSize, rightSize, expected, 
            "Hash inner join");
      } finally {
        for (GatherMap gm : gatherMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testSortMergeInnerJoinNullsUnequal() {
    // Verify compareNullsEqual=false prevents null==null matches while matching real values

    try (ColumnVector leftKeys = ColumnVector.fromBoxedInts(null, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromBoxedInts(null, 1, 3);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();
      int rightSize = (int) rightTable.getRowCount();

      GatherMap[] gatherMaps = JoinPrimitives.sortMergeInnerJoin(
          leftTable, rightTable, false, false);

      try {
        Set<Map.Entry<Integer, Integer>> expected = pairSet(pair(1, 1));
        assertGatherMapPairs(gatherMaps, leftSize, rightSize, expected,
            "Sort merge join with compareNullsEqual=false");
      } finally {
        for (GatherMap gm : gatherMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testHashInnerJoinNullsUnequal() {
    // Verify compareNullsEqual=false prevents null==null matches while matching real values

    try (ColumnVector leftKeys = ColumnVector.fromBoxedInts(null, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromBoxedInts(null, 1, 3);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();
      int rightSize = (int) rightTable.getRowCount();

      GatherMap[] gatherMaps = JoinPrimitives.hashInnerJoin(
          leftTable, rightTable, false);

      try {
        Set<Map.Entry<Integer, Integer>> expected = pairSet(pair(1, 1));
        assertGatherMapPairs(gatherMaps, leftSize, rightSize, expected,
            "Hash join with compareNullsEqual=false");
      } finally {
        for (GatherMap gm : gatherMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testInnerJoinEmpty() {
    // Test with no matches
    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1);
         ColumnVector rightKeys = ColumnVector.fromInts(2, 3);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();
      int rightSize = (int) rightTable.getRowCount();

      GatherMap[] gatherMaps = JoinPrimitives.sortMergeInnerJoin(
          leftTable, rightTable, false, true);

      try {
        Set<Map.Entry<Integer, Integer>> expected = pairSet(); // Empty set
        assertGatherMapPairs(gatherMaps, leftSize, rightSize, expected, 
            "Inner join with no matches");
      } finally {
        for (GatherMap gm : gatherMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  // =============================================================================
  // AST FILTERING
  // =============================================================================

  @Test
  void testFilterGatherMapsByAST() {
    // Start with an equality join, then filter by AST
    // Keys: {1, 2, 3} join {1, 2, 3} -> all match
    // Left data: {10, 20, 30}
    // Right data: {15, 25, 5}
    // Condition: left > right
    // Expected: Only (2,2) passes (30>5)

    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftKeys = ColumnVector.fromInts(1, 2, 3);
         ColumnVector rightKeys = ColumnVector.fromInts(1, 2, 3);
         Table leftKeyTable = new Table(leftKeys);
         Table rightKeyTable = new Table(rightKeys);
         ColumnVector leftData = ColumnVector.fromInts(10, 20, 30);
         ColumnVector rightData = ColumnVector.fromInts(15, 25, 5);
         Table leftTable = new Table(leftData);
         Table rightTable = new Table(rightData);
         CompiledExpression condition = expr.compile()) {

      int leftSize = (int) leftTable.getRowCount();
      int rightSize = (int) rightTable.getRowCount();

      // First get gather maps from equality join
      GatherMap[] equalityMaps = JoinPrimitives.sortMergeInnerJoin(
          leftKeyTable, rightKeyTable, false, true);

      try {
        // Then filter by AST
        GatherMap[] filteredMaps = JoinPrimitives.filterGatherMapsByAST(
            equalityMaps[0], equalityMaps[1], leftTable, rightTable, condition);

        try {
          Set<Map.Entry<Integer, Integer>> expected = pairSet(pair(2, 2));
          assertGatherMapPairs(filteredMaps, leftSize, rightSize, expected, 
              "AST filtered join (left > right)");
        } finally {
          for (GatherMap gm : filteredMaps) {
            if (gm != null) gm.close();
          }
        }
      } finally {
        for (GatherMap gm : equalityMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  // =============================================================================
  // MAKE OUTER JOINS
  // =============================================================================

  @Test
  void testMakeLeftOuter() {
    // Left: {0, 1, 2, 3}, Right: {1, 2, 4}
    // Inner join result: (1,0), (2,1)
    // Expected left outer: all left rows, with unmatched having null right indices

    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2, 3);
         ColumnVector rightKeys = ColumnVector.fromInts(1, 2, 4);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();
      int rightSize = (int) rightTable.getRowCount();

      // Get inner join result
      GatherMap[] innerMaps = JoinPrimitives.sortMergeInnerJoin(
          leftTable, rightTable, false, true);

      try {
        // Make it left outer
        GatherMap[] outerMaps = JoinPrimitives.makeLeftOuter(
            innerMaps[0], innerMaps[1], leftSize, rightSize);

        try {
          // Expected: (0,null), (1,0), (2,1), (3,null)
          Set<Map.Entry<Integer, Integer>> expected = pairSet(
              pair(0, null), pair(1, 0), pair(2, 1), pair(3, null));
          assertGatherMapPairs(outerMaps, leftSize, rightSize, expected, 
              "Left outer join");
        } finally {
          for (GatherMap gm : outerMaps) {
            if (gm != null) gm.close();
          }
        }
      } finally {
        for (GatherMap gm : innerMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testMakeRightOuterBySwapping() {
    // Test that we can achieve right outer by swapping left/right in makeLeftOuter
    // Left: {0, 1, 2, 3}, Right: {1, 2, 4}
    // Inner join result: (1,0), (2,1)
    // For right outer: swap to get all right rows

    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2, 3);
         ColumnVector rightKeys = ColumnVector.fromInts(1, 2, 4);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();
      int rightSize = (int) rightTable.getRowCount();

      // Get inner join result
      GatherMap[] innerMaps = JoinPrimitives.sortMergeInnerJoin(
          leftTable, rightTable, false, true);

      try {
        // Make it right outer by swapping: makeLeftOuter(right, left, rightSize, leftSize)
        GatherMap[] outerMaps = JoinPrimitives.makeLeftOuter(
            innerMaps[1], innerMaps[0], rightSize, leftSize);

        try {
          // Expected: (0,1), (1,2), (2,null) - right/left indices
          Set<Map.Entry<Integer, Integer>> expected = pairSet(
              pair(0, 1), pair(1, 2), pair(2, null));
          assertGatherMapPairs(outerMaps, rightSize, leftSize, expected, 
              "Right outer join (via swap)");
        } finally {
          for (GatherMap gm : outerMaps) {
            if (gm != null) gm.close();
          }
        }
      } finally {
        for (GatherMap gm : innerMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testMakeFullOuter() {
    // Left: {1, 3}, Right: {1, 5}
    // Inner join result: (0,0) matching key 1
    // Expected full outer: matched (0,0) + unmatched left (1,null) + unmatched right (null,1)

    try (ColumnVector leftKeys = ColumnVector.fromInts(1, 3);
         ColumnVector rightKeys = ColumnVector.fromInts(1, 5);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();
      int rightSize = (int) rightTable.getRowCount();

      // Get inner join result
      GatherMap[] innerMaps = JoinPrimitives.sortMergeInnerJoin(
          leftTable, rightTable, false, true);

      try {
        // Make it full outer
        GatherMap[] outerMaps = JoinPrimitives.makeFullOuter(
            innerMaps[0], innerMaps[1], leftSize, rightSize);

        try {
          // Expected: (0,0), (1,null), (null,1)
          Set<Map.Entry<Integer, Integer>> expected = pairSet(
              pair(0, 0), pair(1, null), pair(null, 1));
          assertGatherMapPairs(outerMaps, leftSize, rightSize, expected, 
              "Full outer join");
        } finally {
          for (GatherMap gm : outerMaps) {
            if (gm != null) gm.close();
          }
        }
      } finally {
        for (GatherMap gm : innerMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  // =============================================================================
  // MAKE SEMI/ANTI JOINS
  // =============================================================================

  @Test
  void testMakeSemi() {
    // Left: {1, 1, 2, 3}, Right: {1, 2}
    // Inner join result will have duplicates for key 1
    // Expected semi result: {0, 1, 2} (unique left indices that match)

    try (ColumnVector leftKeys = ColumnVector.fromInts(1, 1, 2, 3);
         ColumnVector rightKeys = ColumnVector.fromInts(1, 2);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();

      // Get inner join result (will have duplicates)
      GatherMap[] innerMaps = JoinPrimitives.sortMergeInnerJoin(
          leftTable, rightTable, false, true);

      try {
        // Make it semi (only needs left gather map)
        try (GatherMap semiMap = JoinPrimitives.makeSemi(innerMaps[0], leftSize)) {
          Set<Integer> expected = new HashSet<>(Arrays.asList(0, 1, 2));
          assertGatherMapIndices(semiMap, leftSize, expected, "Semi join");
        }
      } finally {
        for (GatherMap gm : innerMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testMakeAnti() {
    // Left: {0, 1, 2, 3}, Right: {1, 2}
    // Semi join result will have matched indices: {1, 2}
    // Expected anti result: {0, 3} (unmatched left indices)

    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2, 3);
         ColumnVector rightKeys = ColumnVector.fromInts(1, 2);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();

      // Get inner join result
      GatherMap[] innerMaps = JoinPrimitives.sortMergeInnerJoin(
          leftTable, rightTable, false, true);

      try {
        // Make it semi first (only needs left gather map)
        try (GatherMap semiMap = JoinPrimitives.makeSemi(innerMaps[0], leftSize)) {
          // Then make it anti
          try (GatherMap antiMap = JoinPrimitives.makeAnti(semiMap, leftSize)) {
            Set<Integer> expected = new HashSet<>(Arrays.asList(0, 3));
            assertGatherMapIndices(antiMap, leftSize, expected, "Anti join");
          }
        }
      } finally {
        for (GatherMap gm : innerMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  // =============================================================================
  // PARTITIONED JOIN SUPPORT
  // =============================================================================

  @Test
  void testGetMatchedRows() {
    // Left: {1, 2, 3, 4, 5, 6, 7}, Right: {2, 4, 4, 6}
    // Join will match indices: {1, 3, 3, 5}
    // Expected boolean column: {false, true, false, true, false, true, false}

    try (ColumnVector leftKeys = ColumnVector.fromInts(1, 2, 3, 4, 5, 6, 7);
         ColumnVector rightKeys = ColumnVector.fromInts(2, 4, 4, 6);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftTableSize = (int)leftTable.getRowCount();

      // Get inner join result (will have duplicates)
      GatherMap[] innerMaps = JoinPrimitives.sortMergeInnerJoin(
          leftTable, rightTable, false, true);

      try {
        // Get matched rows boolean column for left side
        try (ColumnVector matchedRows = JoinPrimitives.getMatchedRows(innerMaps[0], leftTableSize)) {
          assertEquals(leftTableSize, matchedRows.getRowCount(), 
              "Matched rows column should have same length as table");

          try (HostColumnVector matchedHost = matchedRows.copyToHost()) {
            assertFalse(matchedHost.getBoolean(0));  // 1 doesn't match
            assertTrue(matchedHost.getBoolean(1));   // 2 matches
            assertFalse(matchedHost.getBoolean(2));  // 3 doesn't match
            assertTrue(matchedHost.getBoolean(3));   // 4 matches
            assertFalse(matchedHost.getBoolean(4));  // 5 doesn't match
            assertTrue(matchedHost.getBoolean(5));   // 6 matches
            assertFalse(matchedHost.getBoolean(6));  // 7 doesn't match
          }
        }
      } finally {
        for (GatherMap gm : innerMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testGetMatchedRowsWithSentinelIndices() {
    // Ensure gather maps containing sentinel entries are ignored when marking matches

    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromInts(0);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();
      int rightSize = (int) rightTable.getRowCount();

      GatherMap[] innerMaps = JoinPrimitives.sortMergeInnerJoin(
          leftTable, rightTable, false, true);

      try {
        GatherMap[] outerMaps = JoinPrimitives.makeLeftOuter(
            innerMaps[0], innerMaps[1], leftSize, rightSize);

        try {
          try (ColumnVector matched = JoinPrimitives.getMatchedRows(outerMaps[1], rightSize)) {
            assertEquals(rightSize, matched.getRowCount(),
                "Matched rows column should have same length as table");

            try (HostColumnVector host = matched.copyToHost()) {
              assertTrue(host.getBoolean(0));
            }
          }
        } finally {
          for (GatherMap gm : outerMaps) {
            if (gm != null) gm.close();
          }
        }
      } finally {
        for (GatherMap gm : innerMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  // =============================================================================
  // INTEGRATION TESTS - Composing Primitives
  // =============================================================================

  @Test
  void testComposeInnerJoinWithAST() {
    // Demonstrate composing sort merge join + AST filter to achieve
    // the same result as MixedSortMergeJoin.innerJoin
    
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftKeys = ColumnVector.fromInts(1, 2);
         ColumnVector rightKeys = ColumnVector.fromInts(1, 2);
         Table leftKeyTable = new Table(leftKeys);
         Table rightKeyTable = new Table(rightKeys);
         ColumnVector leftData = ColumnVector.fromInts(10, 20);
         ColumnVector rightData = ColumnVector.fromInts(5, 25);
         Table leftDataTable = new Table(leftData);
         Table rightDataTable = new Table(rightData);
         CompiledExpression condition = expr.compile()) {

      int leftSize = (int) leftDataTable.getRowCount();
      int rightSize = (int) rightDataTable.getRowCount();

      // Step 1: Equality join
      GatherMap[] equalityMaps = JoinPrimitives.sortMergeInnerJoin(
          leftKeyTable, rightKeyTable, false, true);

      try {
        // Step 2: Filter by AST
        GatherMap[] filteredMaps = JoinPrimitives.filterGatherMapsByAST(
            equalityMaps[0], equalityMaps[1], leftDataTable, rightDataTable, condition);

        try {
          // Expected: only (0,0) passes (1==1 and 10>5)
          Set<Map.Entry<Integer, Integer>> expected = pairSet(pair(0, 0));
          assertGatherMapPairs(filteredMaps, leftSize, rightSize, expected, 
              "Composed inner join with AST");
        } finally {
          for (GatherMap gm : filteredMaps) {
            if (gm != null) gm.close();
          }
        }
      } finally {
        for (GatherMap gm : equalityMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testComposeLeftJoinWithAST() {
    // Demonstrate composing: sort merge + AST filter + make left outer
    
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftKeys = ColumnVector.fromInts(1, 2, 3);
         ColumnVector rightKeys = ColumnVector.fromInts(1, 2);
         Table leftKeyTable = new Table(leftKeys);
         Table rightKeyTable = new Table(rightKeys);
         ColumnVector leftData = ColumnVector.fromInts(10, 20, 30);
         ColumnVector rightData = ColumnVector.fromInts(5, 25);
         Table leftDataTable = new Table(leftData);
         Table rightDataTable = new Table(rightData);
         CompiledExpression condition = expr.compile()) {

      int leftSize = (int) leftDataTable.getRowCount();
      int rightSize = (int) rightDataTable.getRowCount();

      // Step 1: Equality join
      GatherMap[] equalityMaps = JoinPrimitives.sortMergeInnerJoin(
          leftKeyTable, rightKeyTable, false, true);

      try {
        // Step 2: Filter by AST
        GatherMap[] filteredMaps = JoinPrimitives.filterGatherMapsByAST(
            equalityMaps[0], equalityMaps[1], leftDataTable, rightDataTable, condition);

        try {
          // Step 3: Make left outer
          GatherMap[] leftOuterMaps = JoinPrimitives.makeLeftOuter(
              filteredMaps[0], filteredMaps[1], leftSize, rightSize);

          try {
            // Expected: (0,0) matches from AST, plus (1,null) and (2,null) unmatched left rows
            Set<Map.Entry<Integer, Integer>> expected = pairSet(
                pair(0, 0),
                pair(1, null),
                pair(2, null));
            assertGatherMapPairs(leftOuterMaps, leftSize, rightSize, expected, 
                "Composed left join with AST");
          } finally {
            for (GatherMap gm : leftOuterMaps) {
              if (gm != null) gm.close();
            }
          }
        } finally {
          for (GatherMap gm : filteredMaps) {
            if (gm != null) gm.close();
          }
        }
      } finally {
        for (GatherMap gm : equalityMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  // =============================================================================
  // EMPTY GATHER MAP TESTS
  // =============================================================================

  @Test
  void testSortMergeInnerJoinEmpty() {
    // No matching keys - should produce empty gather maps
    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromInts(5, 6, 7);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();
      int rightSize = (int) rightTable.getRowCount();

      GatherMap[] gatherMaps = JoinPrimitives.sortMergeInnerJoin(
          leftTable, rightTable, false, true);

      try {
        Set<Map.Entry<Integer, Integer>> expected = pairSet(); // Empty set
        assertGatherMapPairs(gatherMaps, leftSize, rightSize, expected, 
            "Sort merge inner join with no matches");
      } finally {
        for (GatherMap gm : gatherMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testHashInnerJoinEmpty() {
    // No matching keys - should produce empty gather maps
    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromInts(5, 6, 7);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();
      int rightSize = (int) rightTable.getRowCount();

      GatherMap[] gatherMaps = JoinPrimitives.hashInnerJoin(
          leftTable, rightTable, true);

      try {
        Set<Map.Entry<Integer, Integer>> expected = pairSet(); // Empty set
        assertGatherMapPairs(gatherMaps, leftSize, rightSize, expected, 
            "Hash inner join with no matches");
      } finally {
        for (GatherMap gm : gatherMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testFilterGatherMapsByASTEmpty() {
    // Start with empty gather maps and filter - should stay empty
    BinaryOperation expr = new BinaryOperation(BinaryOperator.EQUAL,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromInts(5, 6, 7);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys);
         CompiledExpression condition = expr.compile()) {

      int leftSize = (int) leftTable.getRowCount();
      int rightSize = (int) rightTable.getRowCount();

      // First get empty gather maps
      GatherMap[] emptyMaps = JoinPrimitives.hashInnerJoin(
          leftTable, rightTable, true);

      try {
        // Filter the empty maps
        GatherMap[] filteredMaps = JoinPrimitives.filterGatherMapsByAST(
            emptyMaps[0], emptyMaps[1], leftTable, rightTable, condition);

        try {
          Set<Map.Entry<Integer, Integer>> expected = pairSet(); // Empty set
          assertGatherMapPairs(filteredMaps, leftSize, rightSize, expected, 
              "AST filter on empty gather maps");
        } finally {
          for (GatherMap gm : filteredMaps) {
            if (gm != null) gm.close();
          }
        }
      } finally {
        for (GatherMap gm : emptyMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testMakeLeftOuterFromEmpty() {
    // Start with empty inner join, make left outer - should get all left rows
    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromInts(5, 6, 7);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();
      int rightSize = (int) rightTable.getRowCount();

      GatherMap[] emptyMaps = JoinPrimitives.hashInnerJoin(
          leftTable, rightTable, true);

      try {
        GatherMap[] leftOuterMaps = JoinPrimitives.makeLeftOuter(
            emptyMaps[0], emptyMaps[1], leftSize, rightSize);

        try {
          // Expected: all left rows with null right indices (0,null), (1,null), (2,null)
          Set<Map.Entry<Integer, Integer>> expected = pairSet(
              pair(0, null), pair(1, null), pair(2, null));
          assertGatherMapPairs(leftOuterMaps, leftSize, rightSize, expected, 
              "Left outer from empty inner join");
        } finally {
          for (GatherMap gm : leftOuterMaps) {
            if (gm != null) gm.close();
          }
        }
      } finally {
        for (GatherMap gm : emptyMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testMakeFullOuterFromEmpty() {
    // Start with empty inner join, make full outer - should get all rows from both sides
    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1);
         ColumnVector rightKeys = ColumnVector.fromInts(5, 6, 7);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();
      int rightSize = (int) rightTable.getRowCount();

      GatherMap[] emptyMaps = JoinPrimitives.hashInnerJoin(
          leftTable, rightTable, true);

      try {
        GatherMap[] fullOuterMaps = JoinPrimitives.makeFullOuter(
            emptyMaps[0], emptyMaps[1], leftSize, rightSize);

        try {
          // Expected: (0,null), (1,null), (null,0), (null,1), (null,2)
          Set<Map.Entry<Integer, Integer>> expected = pairSet(
              pair(0, null), pair(1, null), 
              pair(null, 0), pair(null, 1), pair(null, 2));
          assertGatherMapPairs(fullOuterMaps, leftSize, rightSize, expected, 
              "Full outer from empty inner join");
        } finally {
          for (GatherMap gm : fullOuterMaps) {
            if (gm != null) gm.close();
          }
        }
      } finally {
        for (GatherMap gm : emptyMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testMakeSemiFromEmpty() {
    // Start with empty inner join, make semi - should return empty
    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromInts(5, 6, 7);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();

      GatherMap[] emptyMaps = JoinPrimitives.hashInnerJoin(
          leftTable, rightTable, true);

      try {
        try (GatherMap semiMap = JoinPrimitives.makeSemi(emptyMaps[0], leftSize)) {
          // No matches, so semi should be empty
          Set<Integer> expected = new HashSet<>();
          assertGatherMapIndices(semiMap, leftSize, expected, "Semi from empty inner join");
        }
      } finally {
        for (GatherMap gm : emptyMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testMakeAntiFromEmpty() {
    // Start with empty inner join, make anti - should return all left rows
    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromInts(5, 6, 7);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();

      GatherMap[] emptyMaps = JoinPrimitives.hashInnerJoin(
          leftTable, rightTable, true);

      try {
        try (GatherMap antiMap = JoinPrimitives.makeAnti(emptyMaps[0], leftSize)) {
          // No matches, so anti should return all left rows
          Set<Integer> expected = new HashSet<>(Arrays.asList(0, 1, 2));
          assertGatherMapIndices(antiMap, leftSize, expected, "Anti from empty inner join");
        }
      } finally {
        for (GatherMap gm : emptyMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testGetMatchedRowsEmpty() {
    // Empty gather map - should return all false
    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromInts(5, 6, 7);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftTableSize = (int)leftTable.getRowCount();

      GatherMap[] emptyMaps = JoinPrimitives.hashInnerJoin(
          leftTable, rightTable, true);

      try {
        try (ColumnVector matched = JoinPrimitives.getMatchedRows(emptyMaps[0], leftTableSize)) {
          assertEquals(leftTableSize, matched.getRowCount(), 
              "Matched rows column should have same length as table");
          
          try (HostColumnVector matchedHost = matched.copyToHost()) {
            // All should be false (no matches)
            for (int i = 0; i < leftTableSize; i++) {
              assertFalse(matchedHost.getBoolean(i), 
                  "Row " + i + " should be false (unmatched)");
            }
          }
        }
      } finally {
        for (GatherMap gm : emptyMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testGetMatchedRowsWithDuplicates() {
    // Test that duplicate indices in gather map still only mark each row once
    // Left: {1, 1, 2, 3}, Right: {1, 2}
    // Join will produce gather map with duplicates: left indices {0, 1, 2}
    // (both row 0 and row 1 in left match with row 0 in right due to key=1)
    // Expected matched rows: {true, true, true, false}

    try (ColumnVector leftKeys = ColumnVector.fromInts(1, 1, 2, 3);
         ColumnVector rightKeys = ColumnVector.fromInts(1, 2);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftTableSize = (int)leftTable.getRowCount();

      // Get inner join result (will have duplicates for key 1)
      GatherMap[] innerMaps = JoinPrimitives.sortMergeInnerJoin(
          leftTable, rightTable, false, true);

      try {
        // Get matched rows boolean column for left side
        try (ColumnVector matchedRows = JoinPrimitives.getMatchedRows(innerMaps[0], leftTableSize)) {
          assertEquals(leftTableSize, matchedRows.getRowCount(), 
              "Matched rows column should have same length as table");

          try (HostColumnVector matchedHost = matchedRows.copyToHost()) {
            assertTrue(matchedHost.getBoolean(0));   // 1 matches (first occurrence)
            assertTrue(matchedHost.getBoolean(1));   // 1 matches (second occurrence)
            assertTrue(matchedHost.getBoolean(2));   // 2 matches
            assertFalse(matchedHost.getBoolean(3));  // 3 doesn't match
          }
        }
      } finally {
        for (GatherMap gm : innerMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testMakeSemiWithSentinelValues() {
    // Test that makeSemi properly handles sentinel values (bounds checking)
    // When making left outer, unmatched right indices get sentinel values
    // If we accidentally pass right gather map to makeSemi, it should handle sentinels gracefully

    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromInts(0);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();
      int rightSize = (int) rightTable.getRowCount();

      GatherMap[] innerMaps = JoinPrimitives.sortMergeInnerJoin(
          leftTable, rightTable, false, true);

      try {
        GatherMap[] outerMaps = JoinPrimitives.makeLeftOuter(
            innerMaps[0], innerMaps[1], leftSize, rightSize);

        try {
          // outerMaps[1] (right indices) will contain sentinel values for unmatched rows
          // This tests that makeSemi handles out-of-bounds sentinel values correctly
          try (GatherMap semiResult = JoinPrimitives.makeSemi(outerMaps[1], rightSize)) {
            // Should only include valid right index 0, ignoring sentinel values
            assertEquals(1, semiResult.getRowCount(), 
                "Semi join should only include valid (non-sentinel) indices");
            
            try (HostColumnVector semiHost = semiResult.toColumnView(0, 1).copyToHost()) {
              assertEquals(0, semiHost.getInt(0), "Should have right index 0");
            }
          }
        } finally {
          for (GatherMap gm : outerMaps) {
            if (gm != null) gm.close();
          }
        }
      } finally {
        for (GatherMap gm : innerMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testMakeAntiWithSentinelValues() {
    // Test that makeAnti properly handles sentinel values (bounds checking)
    // Similar to testMakeSemiWithSentinelValues but for anti join

    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromInts(0);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      int leftSize = (int) leftTable.getRowCount();
      int rightSize = (int) rightTable.getRowCount();

      GatherMap[] innerMaps = JoinPrimitives.sortMergeInnerJoin(
          leftTable, rightTable, false, true);

      try {
        GatherMap[] outerMaps = JoinPrimitives.makeLeftOuter(
            innerMaps[0], innerMaps[1], leftSize, rightSize);

        try {
          // outerMaps[1] (right indices) will contain sentinel values for unmatched rows
          // This tests that makeAnti handles out-of-bounds sentinel values correctly
          try (GatherMap antiResult = JoinPrimitives.makeAnti(outerMaps[1], rightSize)) {
            // Should return empty since right index 0 is matched (ignoring sentinels)
            assertEquals(0, antiResult.getRowCount(), 
                "Anti join should return empty when all valid indices are matched");
          }
        } finally {
          for (GatherMap gm : outerMaps) {
            if (gm != null) gm.close();
          }
        }
      } finally {
        for (GatherMap gm : innerMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }
}

