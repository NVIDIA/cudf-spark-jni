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
 * @brief The number of bits that is required for a HLLPP register value.
 *
 * This number is determined by the maximum number of leading binary zeros a
 * hashcode can produce. This is equal to the number of bits the hashcode
 * returns. The current implementation uses a 64-bit hashcode, this means 6-bits
 * are (at most) needed to store the number of leading zeros.
 */
constexpr int REGISTER_VALUE_BITS = 6;

/**
 * @brief The number of registers that can be stored in a single long.
 * It's 64 / 6 = 10.
 */
constexpr int REGISTERS_PER_LONG = 64 / REGISTER_VALUE_BITS;

}  // namespace spark_rapids_jni
