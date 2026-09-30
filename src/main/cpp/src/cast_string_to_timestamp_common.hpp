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

#include <cudf/types.hpp>

namespace spark_rapids_jni {

/**
 * Timezone type
 * Used in casting string with timezone to timestamp
 */
enum class TZ_TYPE : uint8_t {

  // Not specified timezone in the string, indicate to use the default timezone.
  NOT_SPECIFIED = 0,

  // Fixed offset timezone
  // String starts with UT/GMT/UTC/[+-], and it's valid.
  // E.g: +08:00, +08, +1:02:30, -010203, GMT+8, UTC+8:00, UT+8
  // E.g: +01:2:03
  FIXED_TZ = 1,

  // Not FIXED_TZ, it's a valid timezone string.
  // E.g.: java.time.ZoneId.SHORT_IDS: CTT
  // E.g.: Region-based timezone: America/Los_Angeles
  OTHER_TZ = 2,

  // Invalid timezone.
  // String starts with UT/GMT/UTC/[+-], but it's invalid.
  // E.g: UTC+19:00, GMT+19:00, max offset is 18 hours
  // E.g: GMT+01:2:03, +01:2:03, special case
  // E.g: non-exist-timezone
  INVALID_TZ = 3
};

}  // namespace spark_rapids_jni
