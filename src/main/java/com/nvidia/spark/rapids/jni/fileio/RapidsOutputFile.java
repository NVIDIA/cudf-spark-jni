/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.jni.fileio;

import java.io.IOException;
import java.io.OutputStream;

/**
 * Represents an output file that can be written to.
 * <br/>
 * The implementation of this interface should be thread-safe.
 */
public interface RapidsOutputFile {
  /**
   * Create the file and return a {@link RapidsOutputStream} for writing.
   * @param overwrite Whether an existing file should be overwritten.
   * @return a {@link RapidsOutputStream} to write to the file
   * @throws IOException if an I/O error occurs while creating the file
   */
  RapidsOutputStream create(boolean overwrite) throws IOException;

  /**
   * Create the file and return a {@link RapidsOutputStream} for writing.
   * This is equivalent to invoking {@link #create(boolean)} with {@code false}.
   * @return a {@link RapidsOutputStream} to write to the file
   * @throws IOException if an I/O error occurs while creating the file
   */
  default RapidsOutputStream create() throws IOException {
    return create(false);
  }

  /**
   * Get the absolute path of the file as a String.
   * @return the absolute path of the file
   */
  String getPath();
}

