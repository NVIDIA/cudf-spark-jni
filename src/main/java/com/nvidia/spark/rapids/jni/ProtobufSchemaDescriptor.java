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

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Immutable descriptor for a flattened protobuf schema, grouping the parallel arrays
 * that describe field structure, types, defaults, and enum metadata.
 *
 * <p>Use this class instead of passing 15+ individual arrays through the JNI boundary.
 * Validation is performed once in the constructor (and again on deserialization).
 *
 * <p>All arrays provided to the constructor are defensively copied to guarantee immutability.
 * During deserialization, {@code defaultReadObject()} reconstructs a fresh object graph and
 * {@link #readObject(java.io.ObjectInputStream)} re-validates the schema invariants before the
 * instance becomes visible. Package-private field access from {@link Protobuf} is therefore safe
 * because constructor callers cannot retain mutable aliases into the stored arrays.
 */
public final class ProtobufSchemaDescriptor implements java.io.Serializable {
  private static final long serialVersionUID = 1L;
  private static final int MAX_FIELD_NUMBER = (1 << 29) - 1;
  private static final int MAX_NESTING_DEPTH = 10;
  private static final int STRUCT_TYPE_ID = ai.rapids.cudf.DType.STRUCT.getTypeId().getNativeId();
  private static final int STRING_TYPE_ID = ai.rapids.cudf.DType.STRING.getTypeId().getNativeId();
  private static final int LIST_TYPE_ID = ai.rapids.cudf.DType.LIST.getTypeId().getNativeId();
  private static final int BOOL8_TYPE_ID = ai.rapids.cudf.DType.BOOL8.getTypeId().getNativeId();
  private static final int INT32_TYPE_ID = ai.rapids.cudf.DType.INT32.getTypeId().getNativeId();
  private static final int UINT32_TYPE_ID = ai.rapids.cudf.DType.UINT32.getTypeId().getNativeId();
  private static final int INT64_TYPE_ID = ai.rapids.cudf.DType.INT64.getTypeId().getNativeId();
  private static final int UINT64_TYPE_ID = ai.rapids.cudf.DType.UINT64.getTypeId().getNativeId();
  private static final int FLOAT32_TYPE_ID = ai.rapids.cudf.DType.FLOAT32.getTypeId().getNativeId();
  private static final int FLOAT64_TYPE_ID = ai.rapids.cudf.DType.FLOAT64.getTypeId().getNativeId();

  final int[] fieldNumbers;
  final int[] parentIndices;
  final int[] depthLevels;
  final int[] wireTypes;
  final int[] outputTypeIds;
  final int[] encodings;
  final boolean[] isRepeated;
  final boolean[] isRequired;
  final boolean[] hasDefaultValue;
  // Not final: instances serialized before isOutput existed deserialize it as null, and
  // readObject() backfills it via allOutput() for backward compatibility.
  boolean[] isOutput;
  final long[] defaultInts;
  final double[] defaultFloats;
  final boolean[] defaultBools;
  final byte[][] defaultStrings;
  final int[][] enumValidValues;
  final byte[][][] enumNames;

  /**
   * Back-compat constructor: every field is treated as visible in the output.
   *
   * @throws IllegalArgumentException if any array is null, arrays have mismatched lengths,
   *         field numbers are out of range, or encoding values are invalid.
   */
  public ProtobufSchemaDescriptor(
      int[] fieldNumbers,
      int[] parentIndices,
      int[] depthLevels,
      int[] wireTypes,
      int[] outputTypeIds,
      int[] encodings,
      boolean[] isRepeated,
      boolean[] isRequired,
      boolean[] hasDefaultValue,
      long[] defaultInts,
      double[] defaultFloats,
      boolean[] defaultBools,
      byte[][] defaultStrings,
      int[][] enumValidValues,
      byte[][][] enumNames) {
    this(fieldNumbers, parentIndices, depthLevels, wireTypes, outputTypeIds,
        encodings, isRepeated, isRequired, hasDefaultValue,
        fieldNumbers == null ? null : allOutput(fieldNumbers.length),
        defaultInts, defaultFloats, defaultBools, defaultStrings, enumValidValues, enumNames);
  }

  /**
   * @param isOutput per-field flag; if {@code false}, the field is decoded for validation but
   *        dropped from the returned struct. Hidden fields must agree with their parent — a
   *        visible STRUCT cannot have hidden children and vice versa.
   * @throws IllegalArgumentException if any array is null, arrays have mismatched lengths,
   *         field numbers are out of range, or encoding values are invalid.
   */
  public ProtobufSchemaDescriptor(
      int[] fieldNumbers,
      int[] parentIndices,
      int[] depthLevels,
      int[] wireTypes,
      int[] outputTypeIds,
      int[] encodings,
      boolean[] isRepeated,
      boolean[] isRequired,
      boolean[] hasDefaultValue,
      boolean[] isOutput,
      long[] defaultInts,
      double[] defaultFloats,
      boolean[] defaultBools,
      byte[][] defaultStrings,
      int[][] enumValidValues,
      byte[][][] enumNames) {

    validate(fieldNumbers, parentIndices, depthLevels, wireTypes, outputTypeIds,
        encodings, isRepeated, isRequired, hasDefaultValue, isOutput, defaultInts,
        defaultFloats, defaultBools, defaultStrings, enumValidValues, enumNames);

    this.fieldNumbers = fieldNumbers.clone();
    this.parentIndices = parentIndices.clone();
    this.depthLevels = depthLevels.clone();
    this.wireTypes = wireTypes.clone();
    this.outputTypeIds = outputTypeIds.clone();
    this.encodings = encodings.clone();
    this.isRepeated = isRepeated.clone();
    this.isRequired = isRequired.clone();
    this.hasDefaultValue = hasDefaultValue.clone();
    this.isOutput = isOutput.clone();
    this.defaultInts = defaultInts.clone();
    this.defaultFloats = defaultFloats.clone();
    this.defaultBools = defaultBools.clone();
    this.defaultStrings = deepCopy(defaultStrings);
    this.enumValidValues = deepCopy(enumValidValues);
    this.enumNames = deepCopy(enumNames);
  }

  public int numFields() { return fieldNumbers.length; }

  private void readObject(java.io.ObjectInputStream in)
      throws java.io.IOException, ClassNotFoundException {
    // defaultReadObject() reconstructs new array objects from the serialized stream; we do not
    // receive caller-owned array aliases here. Re-run validate() so deserialization cannot bypass
    // the constructor's schema invariants.
    in.defaultReadObject();
    // Backward compatibility: streams written before isOutput existed carry no such field, so
    // defaultReadObject() leaves it null. Treat a legacy descriptor as "all fields output",
    // mirroring the back-compat constructor. A fully malformed stream (no fieldNumbers) still
    // fails the non-null check in validate() below.
    if (isOutput == null && fieldNumbers != null) {
      isOutput = allOutput(fieldNumbers.length);
    }
    try {
      validate(fieldNumbers, parentIndices, depthLevels, wireTypes, outputTypeIds,
          encodings, isRepeated, isRequired, hasDefaultValue, isOutput, defaultInts,
          defaultFloats, defaultBools, defaultStrings, enumValidValues, enumNames);
    } catch (IllegalArgumentException e) {
      java.io.InvalidObjectException ioe = new java.io.InvalidObjectException(e.getMessage());
      ioe.initCause(e);
      throw ioe;
    }
  }

  private static byte[][] deepCopy(byte[][] src) {
    byte[][] dst = new byte[src.length][];
    for (int i = 0; i < src.length; i++) {
      dst[i] = src[i] != null ? src[i].clone() : null;
    }
    return dst;
  }

  private static int[][] deepCopy(int[][] src) {
    int[][] dst = new int[src.length][];
    for (int i = 0; i < src.length; i++) {
      dst[i] = src[i] != null ? src[i].clone() : null;
    }
    return dst;
  }

  private static byte[][][] deepCopy(byte[][][] src) {
    byte[][][] dst = new byte[src.length][][];
    for (int i = 0; i < src.length; i++) {
      if (src[i] == null) continue;
      dst[i] = new byte[src[i].length][];
      for (int j = 0; j < src[i].length; j++) {
        dst[i][j] = src[i][j] != null ? src[i][j].clone() : null;
      }
    }
    return dst;
  }

  private static boolean[] allOutput(int length) {
    boolean[] ret = new boolean[length];
    Arrays.fill(ret, true);
    return ret;
  }

  private static void validate(
      int[] fieldNumbers, int[] parentIndices, int[] depthLevels,
      int[] wireTypes, int[] outputTypeIds, int[] encodings,
      boolean[] isRepeated, boolean[] isRequired, boolean[] hasDefaultValue,
      boolean[] isOutput,
      long[] defaultInts, double[] defaultFloats, boolean[] defaultBools,
      byte[][] defaultStrings, int[][] enumValidValues, byte[][][] enumNames) {

    if (fieldNumbers == null || parentIndices == null || depthLevels == null ||
        wireTypes == null || outputTypeIds == null || encodings == null ||
        isRepeated == null || isRequired == null || hasDefaultValue == null ||
        isOutput == null || defaultInts == null || defaultFloats == null ||
        defaultBools == null || defaultStrings == null || enumValidValues == null ||
        enumNames == null) {
      throw new IllegalArgumentException("All schema arrays must be non-null");
    }

    int n = fieldNumbers.length;
    if (parentIndices.length != n || depthLevels.length != n ||
        wireTypes.length != n || outputTypeIds.length != n ||
        encodings.length != n || isRepeated.length != n ||
        isRequired.length != n || hasDefaultValue.length != n ||
        isOutput.length != n ||
        defaultInts.length != n || defaultFloats.length != n ||
        defaultBools.length != n || defaultStrings.length != n ||
        enumValidValues.length != n || enumNames.length != n) {
      throw new IllegalArgumentException("All schema arrays must have the same length");
    }

    Set<Long> seenFieldNumbers = new HashSet<>();
    for (int i = 0; i < n; i++) {
      validateFieldRange(i, fieldNumbers[i], depthLevels[i]);
      validateParentChild(i, parentIndices[i], depthLevels, outputTypeIds, isOutput);
      validateUniqueFieldKey(i, parentIndices[i], fieldNumbers[i], seenFieldNumbers);
      validateWireTypeAndEncoding(i, wireTypes[i], outputTypeIds[i], encodings[i]);
      validateFieldFlags(i, isRepeated[i], isRequired[i], hasDefaultValue[i], outputTypeIds[i]);
      validateEnumMetadata(i, encodings[i], enumValidValues[i], enumNames[i],
          hasDefaultValue[i], defaultInts[i]);
    }
  }

  private static void validateFieldRange(int index, int fieldNumber, int depth) {
    if (fieldNumber <= 0 || fieldNumber > MAX_FIELD_NUMBER) {
      throw new IllegalArgumentException(
          "Invalid field number at index " + index + ": " + fieldNumber +
          " (must be 1-" + MAX_FIELD_NUMBER + ")");
    }
    if (depth < 0 || depth >= MAX_NESTING_DEPTH) {
      throw new IllegalArgumentException(
          "Invalid depth at index " + index + ": " + depth +
          " (must be 0-" + (MAX_NESTING_DEPTH - 1) + ")");
    }
  }

  private static void validateParentChild(int index, int parentIndex,
                                           int[] depthLevels, int[] outputTypeIds,
                                           boolean[] isOutput) {
    if (parentIndex < -1 || parentIndex >= index) {
      throw new IllegalArgumentException(
          "Invalid parent index at index " + index + ": " + parentIndex +
          " (must be -1 or a prior index < " + index + ")");
    }
    if (parentIndex == -1) {
      if (depthLevels[index] != 0) {
        throw new IllegalArgumentException(
            "Top-level field at index " + index + " must have depth 0, got " +
            depthLevels[index]);
      }
    } else {
      if (outputTypeIds[parentIndex] != STRUCT_TYPE_ID) {
        throw new IllegalArgumentException(
            "Parent at index " + parentIndex + " for field " + index +
            " must be STRUCT, got type id " + outputTypeIds[parentIndex]);
      }
      // A field and its parent must share an output flag: a hidden STRUCT cannot expose
      // visible children, and a visible STRUCT cannot hide individual children.
      if (isOutput[index] != isOutput[parentIndex]) {
        throw new IllegalArgumentException(
            "Nested field at index " + index + " must use the same output flag as parent " +
            parentIndex);
      }
      if (depthLevels[index] != depthLevels[parentIndex] + 1) {
        throw new IllegalArgumentException(
            "Field at index " + index + " depth (" + depthLevels[index] +
            ") must be parent depth (" + depthLevels[parentIndex] + ") + 1");
      }
    }
  }

  private static void validateUniqueFieldKey(int index, int parentIndex,
                                              int fieldNumber, Set<Long> seen) {
    long fieldKey = (((long) parentIndex) << 32) | (fieldNumber & 0xFFFFFFFFL);
    if (!seen.add(fieldKey)) {
      throw new IllegalArgumentException(
          "Duplicate field number " + fieldNumber +
          " under parent index " + parentIndex + " at schema index " + index);
    }
  }

  private static void validateWireTypeAndEncoding(int index, int wireType,
                                                   int outputTypeId, int encoding) {
    if (wireType != Protobuf.WT_VARINT && wireType != Protobuf.WT_64BIT &&
        wireType != Protobuf.WT_LEN && wireType != Protobuf.WT_32BIT) {
      throw new IllegalArgumentException(
          "Invalid wire type at index " + index + ": " + wireType +
          " (must be one of {0, 1, 2, 5})");
    }
    if (encoding < Protobuf.ENC_DEFAULT || encoding > Protobuf.ENC_ENUM_STRING) {
      throw new IllegalArgumentException(
          "Invalid encoding at index " + index + ": " + encoding);
    }
    if (!isEncodingCompatible(wireType, outputTypeId, encoding)) {
      throw new IllegalArgumentException(
          "Incompatible wire type / output type / encoding at index " + index +
          ": wireType=" + wireType + ", outputTypeId=" + outputTypeId +
          ", encoding=" + encoding);
    }
  }

  private static void validateFieldFlags(int index, boolean repeated, boolean required,
                                          boolean hasDefault, int outputTypeId) {
    if (repeated && required) {
      throw new IllegalArgumentException(
          "Field at index " + index + " cannot be both repeated and required");
    }
    if (repeated && hasDefault) {
      throw new IllegalArgumentException(
          "Repeated field at index " + index + " cannot carry a default value");
    }
    if (hasDefault && (outputTypeId == STRUCT_TYPE_ID || outputTypeId == LIST_TYPE_ID)) {
      throw new IllegalArgumentException(
          "STRUCT/LIST field at index " + index + " cannot carry a default value");
    }
  }

  private static void validateEnumMetadata(int index, int encoding,
                                            int[] validValues, byte[][] names,
                                            boolean hasDefault, long defaultValue) {
    if (encoding == Protobuf.ENC_ENUM_STRING &&
        (validValues == null || validValues.length == 0 ||
         names == null || names.length == 0)) {
      throw new IllegalArgumentException(
          "Enum-as-string field at index " + index +
          " must provide non-empty enumValidValues and enumNames");
    }
    if (validValues != null) {
      for (int j = 1; j < validValues.length; j++) {
        if (validValues[j] <= validValues[j - 1]) {
          throw new IllegalArgumentException(
              "enumValidValues[" + index + "] must be strictly sorted in ascending order " +
              "(binary search requires unique values), but found " + validValues[j - 1] +
              " followed by " + validValues[j]);
        }
      }
      if (names != null && names.length != validValues.length) {
        throw new IllegalArgumentException(
            "enumNames[" + index + "].length (" + names.length + ") must equal " +
            "enumValidValues[" + index + "].length (" + validValues.length + ")");
      }
      if (hasDefault && validValues.length > 0 &&
          (defaultValue < Integer.MIN_VALUE || defaultValue > Integer.MAX_VALUE ||
           Arrays.binarySearch(validValues, (int) defaultValue) < 0)) {
        throw new IllegalArgumentException(
            "Enum default at index " + index + " must be present in enumValidValues");
      }
    } else if (names != null) {
      throw new IllegalArgumentException(
          "enumNames[" + index + "] is non-null but enumValidValues[" + index + "] is null; " +
          "both must be provided together for enum-as-string fields");
    }
  }

  private static boolean isEncodingCompatible(int wireType, int outputTypeId, int encoding) {
    switch (encoding) {
      case Protobuf.ENC_DEFAULT:
        return isDefaultEncodingCompatible(wireType, outputTypeId);
      case Protobuf.ENC_FIXED:
        return isFixedEncodingCompatible(wireType, outputTypeId);
      case Protobuf.ENC_ZIGZAG:
        return wireType == Protobuf.WT_VARINT &&
            (outputTypeId == INT32_TYPE_ID || outputTypeId == INT64_TYPE_ID);
      case Protobuf.ENC_ENUM_STRING:
        return wireType == Protobuf.WT_VARINT && outputTypeId == STRING_TYPE_ID;
      default:
        return false;
    }
  }

  private static boolean isDefaultEncodingCompatible(int wireType, int outputTypeId) {
    if (outputTypeId == BOOL8_TYPE_ID || outputTypeId == INT32_TYPE_ID ||
        outputTypeId == UINT32_TYPE_ID || outputTypeId == INT64_TYPE_ID ||
        outputTypeId == UINT64_TYPE_ID) {
      return wireType == Protobuf.WT_VARINT;
    }
    if (outputTypeId == FLOAT32_TYPE_ID) {
      return wireType == Protobuf.WT_32BIT;
    }
    if (outputTypeId == FLOAT64_TYPE_ID) {
      return wireType == Protobuf.WT_64BIT;
    }
    if (outputTypeId == STRING_TYPE_ID || outputTypeId == LIST_TYPE_ID ||
        outputTypeId == STRUCT_TYPE_ID) {
      return wireType == Protobuf.WT_LEN;
    }
    return false;
  }

  private static boolean isFixedEncodingCompatible(int wireType, int outputTypeId) {
    if (outputTypeId == INT32_TYPE_ID || outputTypeId == UINT32_TYPE_ID ||
        outputTypeId == FLOAT32_TYPE_ID) {
      return wireType == Protobuf.WT_32BIT;
    }
    if (outputTypeId == INT64_TYPE_ID || outputTypeId == UINT64_TYPE_ID ||
        outputTypeId == FLOAT64_TYPE_ID) {
      return wireType == Protobuf.WT_64BIT;
    }
    return false;
  }
}
