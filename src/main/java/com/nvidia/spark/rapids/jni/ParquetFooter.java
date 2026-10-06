/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.
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

import java.util.ArrayList;
import java.util.Locale;

/**
 * Represents a footer for a parquet file that can be parsed using native code.
 */
public class ParquetFooter implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Base element for all types in a parquet schema.
   */
  public static abstract class SchemaElement {}

  private static class ElementWithName {
    final String name;
    final SchemaElement element;

    public ElementWithName(String name, SchemaElement element) {
      this.name = name;
      this.element = element;
    }
  }

  public static class StructElement extends SchemaElement {
    public static StructBuilder builder() {
      return new StructBuilder();
    }

    private final ElementWithName[] children;
    private StructElement(ElementWithName[] children) {
      this.children = children;
    }
  }

  public static class StructBuilder {
    ArrayList<ElementWithName> children = new ArrayList<>();

    StructBuilder() {
      // Empty
    }

    public StructBuilder addChild(String name, SchemaElement child) {
      children.add(new ElementWithName(name, child));
      return this;
    }

    public StructElement build() {
      return new StructElement(children.toArray(new ElementWithName[0]));
    }
  }

  public static class ValueElement extends SchemaElement {
    public ValueElement() {}
  }

  public static class ListElement extends SchemaElement {
    private final SchemaElement item;
    public ListElement(SchemaElement item) {
      this.item = item;
    }
  }

  public static class MapElement extends SchemaElement {
    private final SchemaElement key;
    private final SchemaElement value;
    public MapElement(SchemaElement key, SchemaElement value) {
      this.key = key;
      this.value = value;
    }
  }

  private long nativeHandle;

  private ParquetFooter(long handle) {
    nativeHandle = handle;
  }

  private void checkClosed() {
    Preconditions.ensure(nativeHandle != 0, "ParquetFooter has already been closed");
  }

  /**
   * Write the filtered footer back out in a format that is compatible with a parquet
   * footer file. This will include the MAGIC PAR1 at the beginning and end and also the
   * length of the footer just before the PAR1 at the end.
   */
  public HostMemoryBuffer serializeThriftFile(HostMemoryAllocator hostMemoryAllocator) {
    checkClosed();
    return serializeThriftFile(nativeHandle, hostMemoryAllocator);
  }

  public HostMemoryBuffer serializeThriftFile() {
    return serializeThriftFile(DefaultHostMemoryAllocator.get());
  }

  /**
   * Get the number of rows in the footer after filtering.
   */
  public long getNumRows() {
    checkClosed();
    return getNumRows(nativeHandle);
  }

  /**
   * Get the number of top level columns in the footer after filtering.
   */
  public int getNumColumns() {
    checkClosed();
    return getNumColumns(nativeHandle);
  }

  /**
   * Returns the file-global row index offset for each row group in the filtered footer.
   * Offsets are cumulative row counts computed from ALL row groups in the original file
   * before byte-range filtering, so they remain correct even when the footer contains
   * only a subset of the file's row groups.
   *
   * @return array of length equal to the number of row groups in this (filtered) footer,
   *         where element i is the file-global row index of the first row in row group i.
   */
  public long[] getRowIndexOffsets() {
    checkClosed();
    return getRowIndexOffsets(nativeHandle);
  }

  @Override
  public void close() throws Exception {
    if (nativeHandle != 0) {
      close(nativeHandle);
      nativeHandle = 0;
    }
  }

  /**
   * Recursive helper function to flatten a SchemaElement, so it can more efficiently be passed
   * through JNI.
   */
  private static void depthFirstNamesHelper(SchemaElement se, String name, boolean makeLowerCase,
      ArrayList<String> names, ArrayList<Integer> numChildren, ArrayList<Integer> tags) {
    if (makeLowerCase) {
      name = name.toLowerCase(Locale.ROOT);
    }

    if (se instanceof ValueElement) {
      names.add(name);
      numChildren.add(0);
      tags.add(0);
    } else if (se instanceof StructElement) {
      StructElement st = (StructElement) se;
      names.add(name);
      numChildren.add(st.children.length);
      tags.add(1);
      for (ElementWithName child : st.children) {
        depthFirstNamesHelper(child.element, child.name, makeLowerCase, names, numChildren, tags);
      }
    } else if (se instanceof ListElement) {
      ListElement le = (ListElement) se;
      // This follows the conventions of newer parquet. This is just here as a bridge to the new
      // API and code.
      names.add(name);
      numChildren.add(1);
      tags.add(2);
      depthFirstNamesHelper(le.item, "element", makeLowerCase, names, numChildren, tags);
    } else if (se instanceof MapElement) {
      MapElement me = (MapElement) se;
      // This follows the conventions of newer parquet. This is just here as a bridge to the new
      // API and code.
      names.add(name);
      numChildren.add(2);
      tags.add(3);
      depthFirstNamesHelper(me.key, "key", makeLowerCase, names, numChildren, tags);
      depthFirstNamesHelper(me.value, "value", makeLowerCase, names, numChildren, tags);
    } else {
      throw new UnsupportedOperationException(se + " is not a supported schema element type");
    }
  }

  /**
   * Flatten a SchemaElement, so it can more efficiently be passed through JNI.
   */
  private static void depthFirstNames(StructElement schema, boolean makeLowerCase,
      ArrayList<String> names, ArrayList<Integer> numChildren, ArrayList<Integer> tags) {
    // Initialize them with a quick length for non-nested values
    for (ElementWithName se: schema.children) {
      depthFirstNamesHelper(se.element, se.name, makeLowerCase, names, numChildren, tags);
    }
  }

  /**
     * Read a parquet thrift footer from a buffer and filter it like the java code would. The buffer
   * should only include the thrift footer itself. This includes filtering out row groups that do
   * not fall within the partition and pruning columns that are not needed.
   * @param buffer the buffer to parse the footer out from.
   * @param partOffset for a split the start of the split
   * @param partLength the length of the split
   * @param schema a stripped down schema so the code can verify that the types match what is
   *               expected. The java code does this too.
   * @param ignoreCase should case be ignored when matching column names. If this is true then
   *                   names should be converted to lower case before being passed to this.
   * @return a reference to the parsed footer.
   */
  public static ParquetFooter readAndFilter(HostMemoryBuffer buffer,
      long partOffset, long partLength, StructElement schema, boolean ignoreCase) {
    int parentNumChildren = schema.children.length;
    ArrayList<String> names = new ArrayList<>();
    ArrayList<Integer> numChildren = new ArrayList<>();
    ArrayList<Integer> tags = new ArrayList<>();

    depthFirstNames(schema, ignoreCase, names, numChildren, tags);
    return new ParquetFooter(
        readAndFilter
            (buffer.getAddress(), buffer.getLength(),
                partOffset, partLength,
                names.toArray(new String[0]),
                numChildren.stream().mapToInt(i -> i).toArray(),
                tags.stream().mapToInt(i -> i).toArray(),
                parentNumChildren,
                ignoreCase));
  }

  // Native APIS

  private static native long readAndFilter(long address, long length,
      long partOffset, long partLength,
      String[] names,
      int[] numChildren,
      int[] tags,
      int parentNumChildren,
      boolean ignoreCase) throws CudfException;

  private static native void close(long nativeHandle);

  private static native long getNumRows(long nativeHandle);

  private static native int getNumColumns(long nativeHandle);

  private static native HostMemoryBuffer serializeThriftFile(long nativeHandle,
    HostMemoryAllocator hostMemoryAllocator);

  private static native long[] getRowIndexOffsets(long nativeHandle);
}
