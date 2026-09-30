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

package com.nvidia.spark.rapids.jni;

public class Version {
  private final int platformOrdinal;
  private final int major;
  private final int minor;
  private final int patch;

  public Version(SparkPlatformType platform, int major, int minor, int patch) {
    this.platformOrdinal = platform.ordinal();
    this.major = major;
    this.minor = minor;
    this.patch = patch;
  }

  /**
   * Note: this is used in the JNI code and kernel code, so it must match the
   * enum SparkPlatformType in com.nvidia.spark.rapids.jni.SparkPlatformType.
   */
  public int getPlatformOrdinal() {
    return platformOrdinal;
  }

  public int getMajor() {
    return major;
  }

  public int getMinor() {
    return minor;
  }

  public int getPatch() {
    return patch;
  }
}
