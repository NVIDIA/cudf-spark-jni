/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cupti.h>

#include <cstdint>
#include <string>

namespace spark_rapids_jni::profiler {

std::string activity_kind_to_string(CUpti_ActivityKind kind);

void print_cupti_buffer(uint8_t* buffer, size_t valid_size);

}  // namespace spark_rapids_jni::profiler
