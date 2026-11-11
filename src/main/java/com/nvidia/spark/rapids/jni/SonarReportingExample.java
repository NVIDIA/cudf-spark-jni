/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * Test-only, unused example for validating Sonar pull request reporting. Do not merge.
 */
final class SonarReportingExample {
  private SonarReportingExample() {
  }

  static boolean intentionallyCompareWithSelf(int value) {
    // Intentionally triggers java:S1764: identical operands in a comparison.
    return value == value;
  }

  static int intentionalHighSeverityFinding() {
    // Intentionally triggers javabugs:S3518 (High); this unused example must not be called.
    int zero = 0;
    return 1 / zero;
  }

  static int secondIntentionalHighSeverityFinding() {
    // A second unused division-by-zero finding validates the exact High count.
    int zero = 0;
    return 42 / zero;
  }
}
