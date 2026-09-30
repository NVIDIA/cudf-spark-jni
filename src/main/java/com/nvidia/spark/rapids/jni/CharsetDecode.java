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

import ai.rapids.cudf.*;

public class CharsetDecode {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /** Charset constant for GBK encoding */
  public static final int GBK = 0;

  /** Error action constants mirroring {@link java.nio.charset.CodingErrorAction}. */
  public static final int REPLACE = 0;
  public static final int REPORT  = 1;

  /**
   * Thrown when {@link #decode(ColumnView, int, int)} is invoked with {@link #REPORT} and the
   * input contains a malformed or unmappable byte sequence. Callers are responsible for
   * translating this into a Spark {@code MALFORMED_CHARACTER_CODING} error.
   */
  public static final class MalformedInputException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    public MalformedInputException(String message) {
      super(message);
    }
  }

  /**
   * Decode a binary column from the specified charset encoding to a UTF-8 strings column.
   *
   * Invalid or unmappable byte sequences are replaced with U+FFFD.
   *
   * @param cv Binary column (LIST of UINT8) to decode
   * @param charset Charset identifier (use constants like {@link #GBK})
   * @return UTF-8 string column
   */
  public static ColumnVector decode(ColumnView cv, int charset) {
    return decode(cv, charset, REPLACE);
  }

  /**
   * Decode a binary column with a configurable action on malformed/unmappable bytes.
   *
   * @param cv Binary column (LIST of UINT8) to decode
   * @param charset Charset identifier (use constants like {@link #GBK})
   * @param errorAction {@link #REPLACE} or {@link #REPORT}
   * @return UTF-8 string column (only when no malformed bytes are found, for REPORT mode)
   * @throws MalformedInputException in REPORT mode if malformed/unmappable bytes are present
   * @throws IllegalArgumentException if {@code charset} or {@code errorAction} is not a recognized value
   */
  public static ColumnVector decode(ColumnView cv, int charset, int errorAction) {
    if (charset != GBK) {
      throw new IllegalArgumentException("Unsupported charset: " + charset);
    }
    if (errorAction != REPLACE && errorAction != REPORT) {
      throw new IllegalArgumentException("Unsupported errorAction: " + errorAction);
    }
    return new ColumnVector(decodeNative(cv.getNativeView(), charset, errorAction));
  }

  private static native long decodeNative(long nativeColumnView, int charset, int errorAction)
      throws CudfException, MalformedInputException;
}
