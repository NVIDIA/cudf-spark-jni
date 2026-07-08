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
 * GPU protobuf decoding utilities.
 *
 * This API uses a multi-pass approach for efficient decoding:
 * <ul>
 *   <li>Pass 1: Scan all messages, count nested elements and repeated field occurrences</li>
 *   <li>Pass 2: Prefix sum to compute output offsets for arrays and nested structs</li>
 *   <li>Pass 3: Extract data using pre-computed offsets</li>
 *   <li>Pass 4: Build nested column structure</li>
 * </ul>
 *
 * The schema is represented as a flattened array of field descriptors with parent-child
 * relationships. Top-level fields have parentIndices == -1 and depthLevels == 0.
 * For pure scalar schemas, all fields are top-level with isRepeated == false.
 *
 * Supported protobuf field types include:
 * <ul>
 *   <li>VARINT: {@code int32}, {@code int64}, {@code uint32}, {@code uint64}, {@code bool}</li>
 *   <li>ZIGZAG VARINT (encoding=2): {@code sint32}, {@code sint64}</li>
 *   <li>FIXED32 (encoding=1): {@code fixed32}, {@code sfixed32}, {@code float}</li>
 *   <li>FIXED64 (encoding=1): {@code fixed64}, {@code sfixed64}, {@code double}</li>
 *   <li>LENGTH_DELIMITED: {@code string}, {@code bytes}, nested {@code message}</li>
 *   <li>Nested messages and repeated fields</li>
 * </ul>
 *
 * <p>In permissive mode ({@code failOnErrors=false}), if decoding encounters a row-local parse
 * error from which it cannot safely recover its cursor position (for example, an unexpected wire
 * type or malformed varint), scanning for that row stops at the error position. Fields that appear
 * later in the same message are therefore treated as "not found" and follow the normal
 * missing-field semantics (nulls or defaults, depending on the schema metadata).
 */
public class Protobuf {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public static final int ENC_DEFAULT = 0;
  public static final int ENC_FIXED   = 1;
  public static final int ENC_ZIGZAG  = 2;
  public static final int ENC_ENUM_STRING = 3;

  // Wire type constants
  public static final int WT_VARINT = 0;
  public static final int WT_64BIT  = 1;
  public static final int WT_LEN    = 2;
  public static final int WT_32BIT  = 5;

  /**
   * Decode protobuf messages into a STRUCT column.
   *
   * @param binaryInput column of type LIST&lt;INT8/UINT8&gt; where each row is one protobuf message.
   * @param schema descriptor containing flattened schema arrays (field numbers, types, defaults, etc.)
   * @param failOnErrors if true, throw an exception on malformed protobuf messages. If false,
   *                     malformed rows are handled permissively; when a row-local parse error
   *                     prevents safe resynchronization, later fields in that same row are treated
   *                     as absent rather than continuing from an uncertain cursor position.
   * @return a cudf STRUCT column with nested structure.
   */
  public static ColumnVector decodeToStruct(ColumnView binaryInput,
                                            ProtobufSchemaDescriptor schema,
                                            boolean failOnErrors) {
    if (binaryInput == null) {
      throw new IllegalArgumentException("binaryInput must not be null");
    }
    if (schema == null) {
      throw new IllegalArgumentException("schema must not be null");
    }
    long handle = decodeToStruct(binaryInput.getNativeView(),
        schema.fieldNumbers, schema.parentIndices, schema.depthLevels,
        schema.wireTypes, schema.outputTypeIds, schema.encodings,
        schema.isRepeated, schema.isRequired, schema.hasDefaultValue,
        schema.isOutput,
        schema.defaultInts, schema.defaultFloats, schema.defaultBools,
        schema.defaultStrings, schema.enumValidValues, schema.enumNames, failOnErrors);
    return new ColumnVector(handle);
  }

  private static native long decodeToStruct(long binaryInputView,
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
                                            byte[][][] enumNames,
                                            boolean failOnErrors);
}
