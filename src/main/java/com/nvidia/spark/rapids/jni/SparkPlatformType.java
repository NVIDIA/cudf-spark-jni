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

/**
 * Enum representing the platform.
 * NOTE: MUST keep sync with version.hpp
 * The ordinal values are used to represent the platform in JNI calls.
 */
public enum SparkPlatformType {
  // ordinal 0 is vanilla Spark, Will translate to spark_platform_type::VANILLA_SPARK
  VANILLA_SPARK,

  // ordinal 1 is Databricks, Will translate to spark_platform_type::DATABRICKS
  DATABRICKS,

  // ordinal 2 is Cloudera, Will translate to spark_platform_type::CLOUDERA
  CLOUDERA,

  // ordinal 3 is Unknown. Will translate to spark_platform_type::UNKNOWN
  // Maybe customized Spark distributions.
  UNKNOWN;
}
