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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.NativeDepsLoader;

/**
 * Utility APIs for map column operations that require Spark-specific semantics
 * not available in the standard cuDF Java bindings.
 */
public class MapUtils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Returns {@code true} when every row of {@code input} is already a valid Spark map row that
   * can be used as-is (i.e. {@link #mapFromEntries} would not need to mask any row).
   *
   * <p>The check matches the semantics of {@link #mapFromEntries}:
   * <ul>
   *   <li>A row containing any null struct entry is NOT a valid map row → returns {@code false}.</li>
   *   <li>A row whose valid struct has a null key:
   *     <ul>
   *       <li>{@code throwOnNullKey == true}  — throws {@link RuntimeException}.</li>
   *       <li>{@code throwOnNullKey == false} — the row is treated as valid.</li>
   *     </ul>
   *   </li>
   *   <li>An empty input is trivially valid → returns {@code true}.</li>
   * </ul>
   *
   * <p>Intended for callers that want to skip a deep copy when the input is already
   * structurally a map: a Spark {@code MAP<K,V>} and a cuDF {@code LIST<STRUCT<K,V>>} share the
   * same physical layout, so when this method returns {@code true} the caller can reinterpret
   * {@code input} as the result (typically via {@code input.incRefCount()}) — no copy required.
   *
   * <p><b>Sliced input is not supported.</b>  Materialize the slice before calling.
   *
   * @param input          Input LIST(STRUCT(KEY, VALUE)) column.  Must not be sliced.
   * @param throwOnNullKey When {@code true}, throw if any valid-struct entry has a null key.
   * @return {@code true} when every row of {@code input} is a valid map row, {@code false}
   *         otherwise.
   * @throws RuntimeException if {@code throwOnNullKey} is true and any row (with no null struct
   *         entry) contains a null key inside a valid struct.
   * @throws RuntimeException if {@code input} or its struct/key child is sliced.
   */
  public static boolean isValidMap(ColumnView input, boolean throwOnNullKey) {
    return isValidMap(input.getNativeView(), throwOnNullKey);
  }

  /**
   * Converts a LIST(STRUCT(KEY, VALUE)) column to a map column following Spark semantics.
   *
   * <p>Spark semantics for {@code map_from_entries}:
   * <ul>
   *   <li>If a row's array contains any null struct entry (the whole struct is null), the output
   *       row is null — even if another entry in that same row has a null key inside a valid
   *       struct.</li>
   *   <li>If a row's array contains no null struct entry but a valid struct's key is null,
   *       behavior depends on {@code throwOnNullKey}:
   *       <ul>
   *         <li>{@code true}  — throws a {@link RuntimeException}.</li>
   *         <li>{@code false} — returns the row as-is (caller handles dedup policy).</li>
   *       </ul>
   *   </li>
   * </ul>
   *
   * <p>This method ALWAYS returns a non-null {@link ColumnVector}.  When every row would be
   * returned unchanged (the case where {@link #isValidMap} would return {@code true}), it
   * returns a deep copy of {@code input}.  Callers that want to avoid that deep copy should
   * call {@link #isValidMap} first and reinterpret {@code input} via {@code incRefCount()} when
   * it returns {@code true}.
   *
   * <p><b>Sliced input is not supported.</b>  Materialize the slice before calling.
   *
   * <p>Duplicate-key deduplication is intentionally left to the caller so that the EXCEPTION
   * and LAST_WIN policies can be applied after this function returns.
   *
   * @param input          Input LIST(STRUCT(KEY, VALUE)) column.  Must not be sliced.
   * @param throwOnNullKey When {@code true}, throw if any valid-struct entry has a null key.
   * @return A new column with rows containing null struct entries replaced with null outer
   *         rows.  Never {@code null}.
   * @throws RuntimeException if {@code throwOnNullKey} is true and any row (with no null struct
   *         entry) contains a null key inside a valid struct.
   * @throws RuntimeException if {@code input} or its struct/key child is sliced.
   */
  public static ColumnVector mapFromEntries(ColumnView input, boolean throwOnNullKey) {
    return new ColumnVector(mapFromEntries(input.getNativeView(), throwOnNullKey));
  }

  private static native boolean isValidMap(long inputHandle, boolean throwOnNullKey);

  private static native long mapFromEntries(long inputHandle, boolean throwOnNullKey);
}
