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

#pragma once

namespace spark_rapids_jni {

/**
 * @brief Enum class representing different Spark platform types.
 * The values must match the ordinal values defined in SparkPlatformType.java.
 * - VANILLA_SPARK: Represents the standard Apache Spark platform.
 * - DATABRICKS: Represents the Databricks platform.
 * - CLOUDERA: Represents the Cloudera platform.
 * - UNKNOWN: Represents any other platform not specifically defined.
 * - NUM_PLATFORMS: Represents the total number of platforms defined.
 */
enum class spark_platform_type { VANILLA_SPARK = 0, DATABRICKS, CLOUDERA, UNKNOWN, NUM_PLATFORMS };

class spark_system {
 public:
  /**
   * @brief Constructor to initialize the spark system with platform type and version.
   * NOTE: The `platform_ordinal` MUST keep sync with SparkPlatformType.java
   * @param platform_ordinal The platform ordinal value.
   * @param major Major version number.
   * @param minor Minor version number.
   * @param patch Patch version number.
   */
  spark_system(int platform_ordinal, int major_, int minor_, int patch_)
    : platform_type{static_cast<spark_platform_type>(platform_ordinal)},
      major{major_},
      minor{minor_},
      patch{patch_}
  {
  }

  bool is_vanilla_spark() const { return platform_type == spark_platform_type::VANILLA_SPARK; }
  bool is_databricks() const { return platform_type == spark_platform_type::DATABRICKS; }

  bool is_version_eq(int major_, int minor_, int patch_) const
  {
    return major == major_ && minor == minor_ && patch == patch_;
  }

  bool is_version_ge(int major_, int minor_, int patch_) const
  {
    return (major > major_) || (major == major_ && minor > minor_) ||
           (major == major_ && minor == minor_ && patch >= patch_);
  }

  bool is_vanilla_320() const { return is_vanilla_spark() && is_version_eq(3, 2, 0); }

  bool is_vanilla_400_or_later() const { return is_vanilla_spark() && is_version_ge(4, 0, 0); }

  bool is_databricks_14_3_or_later() const { return is_databricks() && is_version_ge(14, 3, 0); }

 private:
  spark_platform_type platform_type;
  int major, minor, patch;
};

}  // namespace spark_rapids_jni
