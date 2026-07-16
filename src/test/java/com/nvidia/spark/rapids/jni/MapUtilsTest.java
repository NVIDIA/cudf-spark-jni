/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

import ai.rapids.cudf.CloseableArray;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for {@link MapUtils#isValidMap} and {@link MapUtils#mapFromEntries} —
 * null struct entry and null key handling.
 *
 * Column schema: LIST(STRUCT(INT32 key, INT32 value))
 *
 * API contract under test:
 *  - {@code isValidMap} returns {@code true} when every row is already a valid map row
 *    (no null struct entries, optionally no null keys depending on {@code throwOnNullKey}).
 *  - {@code mapFromEntries} ALWAYS returns a non-null column.  When {@code isValidMap} would
 *    return {@code true}, the returned column is a deep copy of the input.
 */
public class MapUtilsTest {

  // Common schema: LIST(STRUCT<key:INT32, value:INT32>)
  private static final HostColumnVector.StructType STRUCT_TYPE =
      new HostColumnVector.StructType(true,
          Arrays.asList(
              new HostColumnVector.BasicType(true, DType.INT32),   // key (nullable)
              new HostColumnVector.BasicType(true, DType.INT32))); // value (nullable)
  private static final HostColumnVector.ListType LIST_TYPE =
      new HostColumnVector.ListType(true, STRUCT_TYPE);

  // Helpers
  private static HostColumnVector.StructData entry(Integer key, Integer value) {
    return new HostColumnVector.StructData(Arrays.asList(key, value));
  }

  // --------------------------------------------------------------------------
  // isValidMap == true cases — every row already a valid map row.
  // mapFromEntries on the same input returns a deep copy equal to input.
  // --------------------------------------------------------------------------

  @Test
  void noNullsIsValidMap() {
    List<HostColumnVector.StructData> row0 = Arrays.asList(entry(1, 10), entry(2, 20));
    List<HostColumnVector.StructData> row1 = Arrays.asList(entry(3, 30));
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, row0, row1);
         ColumnVector result   = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, row0, row1)) {
      assertTrue(MapUtils.isValidMap(input, true));
      assertNotNull(result);
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void nullKeyInValidStructIsValidMapWhenPolicyAllows() {
    // [{null_key, 10}] with throwOnNullKey=false  →  the row is valid map row
    List<HostColumnVector.StructData> row0 = Arrays.asList(entry(null, 10));
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, row0);
         ColumnVector result   = MapUtils.mapFromEntries(input, false);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, row0)) {
      assertTrue(MapUtils.isValidMap(input, false));
      assertNotNull(result);
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void emptyListRowIsValidMap() {
    // Row 0: []          —  no entries → no null struct, no null key
    // Row 1: [{1, 10}]   —  valid
    List<HostColumnVector.StructData> row0 = Arrays.asList();
    List<HostColumnVector.StructData> row1 = Arrays.asList(entry(1, 10));
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, row0, row1);
         ColumnVector result   = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, row0, row1)) {
      assertTrue(MapUtils.isValidMap(input, true));
      assertNotNull(result);
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void zeroRowInputIsTriviallyValid() {
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE);
         ColumnVector result   = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE)) {
      assertTrue(MapUtils.isValidMap(input, true));
      assertNotNull(result);
      assertColumnsAreEqual(expected, result);
    }
  }

  // --------------------------------------------------------------------------
  // Throw cases — both isValidMap and mapFromEntries throw on null key under throw policy.
  // --------------------------------------------------------------------------

  @Test
  void nullKeyInValidStructThrows() {
    // [{null_key, 10}] with throwOnNullKey=true  →  must throw
    List<HostColumnVector.StructData> row0 = Arrays.asList(entry(null, 10));
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, row0)) {
      assertThrows(CudfException.class, () -> MapUtils.isValidMap(input, true));
      assertThrows(CudfException.class, () -> MapUtils.mapFromEntries(input, true));
    }
  }

  @Test
  void nullStructRowDoesNotSuppressThrowInOtherRow() {
    // Row 0: [null_struct]     →  STATE_NULL  (would mask)
    // Row 1: [{null_key, 20}]  →  STATE_NULL_KEY → throw under policy=true
    List<HostColumnVector.StructData> row0 =
        Arrays.asList((HostColumnVector.StructData) null);
    List<HostColumnVector.StructData> row1 = Arrays.asList(entry(null, 20));
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, row0, row1)) {
      assertThrows(CudfException.class, () -> MapUtils.isValidMap(input, true));
      assertThrows(CudfException.class, () -> MapUtils.mapFromEntries(input, true));
    }
  }

  // --------------------------------------------------------------------------
  // isValidMap == false cases — at least one row has a null struct entry.
  // mapFromEntries on the same input masks the null-struct rows.
  // --------------------------------------------------------------------------

  @Test
  void nullStructEntryMasksRow() {
    // Row 0: [null_struct, {2,20}]  →  output row 0 = null
    // Row 1: [{3,30}]               →  unchanged
    List<HostColumnVector.StructData> row0 = Arrays.asList(null, entry(2, 20));
    List<HostColumnVector.StructData> row1 = Arrays.asList(entry(3, 30));
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, row0, row1);
         ColumnVector result = MapUtils.mapFromEntries(input, true)) {
      assertFalse(MapUtils.isValidMap(input, true));
      assertNotNull(result);
      try (ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, row1)) {
        assertColumnsAreEqual(expected, result);
      }
    }
  }

  @Test
  void allNullStructEntriesAllRowsNull() {
    List<HostColumnVector.StructData> row0 = Arrays.asList((HostColumnVector.StructData) null);
    List<HostColumnVector.StructData> row1 =
        Arrays.asList((HostColumnVector.StructData) null, (HostColumnVector.StructData) null);
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, row0, row1);
         ColumnVector result   = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, null)) {
      assertFalse(MapUtils.isValidMap(input, true));
      assertNotNull(result);
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void mixedNullStructAndNullKeyReturnsNullNotThrow() {
    // Row 0: [null_struct, {null_key, 20}] — CPU short-circuits on the null struct entry.
    List<HostColumnVector.StructData> row0 = Arrays.asList(null, entry(null, 20));
    try (ColumnVector input  = ColumnVector.fromLists(LIST_TYPE, row0);
         ColumnVector result = MapUtils.mapFromEntries(input, true)) {
      assertFalse(MapUtils.isValidMap(input, true));
      assertNotNull(result);
      try (ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, (List<?>) null)) {
        assertColumnsAreEqual(expected, result);
      }
    }
  }

  @Test
  void outerNullRowPreservedAsNull() {
    List<HostColumnVector.StructData> row1 = Arrays.asList(entry(1, 10));
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, null, row1);
         ColumnVector result   = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, row1)) {
      assertFalse(MapUtils.isValidMap(input, true));
      assertNotNull(result);
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void allOuterNullRowsRemainNull() {
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, null, null);
         ColumnVector result   = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, null)) {
      assertFalse(MapUtils.isValidMap(input, true));
      assertNotNull(result);
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void allOuterNullRowsRemainNullNoThrowPolicy() {
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, null, null);
         ColumnVector result   = MapUtils.mapFromEntries(input, false);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, null)) {
      assertFalse(MapUtils.isValidMap(input, false));
      assertNotNull(result);
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void singleOuterNullRowRemainNull() {
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, (List<?>) null);
         ColumnVector result   = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, (List<?>) null)) {
      assertFalse(MapUtils.isValidMap(input, true));
      assertNotNull(result);
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void outerNullRowAndNullStructEntryRowBothNull() {
    // Row 0: null outer row                →  STATE_NULL
    // Row 1: [null_struct, {2,20}]         →  STATE_NULL
    // Row 2: [{3,30}]                      →  STATE_VALID
    List<HostColumnVector.StructData> row1 = Arrays.asList(null, entry(2, 20));
    List<HostColumnVector.StructData> row2 = Arrays.asList(entry(3, 30));
    try (ColumnVector input  = ColumnVector.fromLists(LIST_TYPE, null, row1, row2);
         ColumnVector result = MapUtils.mapFromEntries(input, true)) {
      assertFalse(MapUtils.isValidMap(input, true));
      assertNotNull(result);
      try (ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, null, row2)) {
        assertColumnsAreEqual(expected, result);
      }
    }
  }

  // --------------------------------------------------------------------------
  // Sliced input is rejected by both APIs.
  // --------------------------------------------------------------------------

  @Test
  void slicedInputThrows() {
    List<HostColumnVector.StructData> row0 = Arrays.asList(entry(1, 10));
    List<HostColumnVector.StructData> row1 = Arrays.asList(entry(2, 20));
    List<HostColumnVector.StructData> row2 = Arrays.asList(entry(3, 30));
    try (ColumnVector full = ColumnVector.fromLists(LIST_TYPE, row0, row1, row2);
         CloseableArray<ColumnView> views = CloseableArray.wrap(full.splitAsViews(1))) {
      ColumnView sliced = views.get(1);  // Exercise the nonzero-offset view.
      assertThrows(CudfException.class, () -> MapUtils.isValidMap(sliced, true));
      assertThrows(CudfException.class, () -> MapUtils.mapFromEntries(sliced, true));
    }
  }

  // --------------------------------------------------------------------------
  // Slow path opt-out coverage and bitmask boundary stress.
  // --------------------------------------------------------------------------

  @Test
  void slowPathNullKeyNoThrowWhenPolicyAllows() {
    // Row 0 has a null struct entry; row 1 has a null key in a valid struct.
    // With throwOnNullKey=false, row 1 is treated as valid, and isValidMap returns false
    // only because of row 0's null struct entry.  mapFromEntries masks row 0 to null and
    // returns row 1 unchanged.
    List<HostColumnVector.StructData> row0 = Arrays.asList(null, entry(1, 10));
    List<HostColumnVector.StructData> row1 = Arrays.asList(entry(null, 20));
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, row0, row1);
         ColumnVector result   = MapUtils.mapFromEntries(input, false);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, row1)) {
      assertFalse(MapUtils.isValidMap(input, false));
      assertNotNull(result);
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  @SuppressWarnings("unchecked")
  void slowPathAcrossMultipleBitmaskWords() {
    // 70 rows: rows 0, 33, 65 contain a null struct entry — each must become null in the
    // output.  Crosses the 32-row warp boundary and the 64-row bitmask-word boundary to
    // guard against bit-alignment regressions in valid_if + the gather-map kernel.
    final int numRows = 70;
    List<HostColumnVector.StructData>[] rows         = new List[numRows];
    List<HostColumnVector.StructData>[] expectedRows = new List[numRows];
    for (int i = 0; i < numRows; i++) {
      if (i == 0 || i == 33 || i == 65) {
        rows[i]         = Arrays.asList(null, entry(i, i * 10));
        expectedRows[i] = null;
      } else {
        rows[i]         = Arrays.asList(entry(i, i * 10));
        expectedRows[i] = rows[i];
      }
    }
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, rows);
         ColumnVector result   = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, expectedRows)) {
      assertFalse(MapUtils.isValidMap(input, true));
      assertNotNull(result);
      assertColumnsAreEqual(expected, result);
    }
  }
}
