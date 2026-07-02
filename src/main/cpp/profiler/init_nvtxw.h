/*
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.txt for license information.
 */

#pragma once

#include "NvtxwEvents.h"
#include "nvtxw3.h"

#include <filesystem>
#include <optional>
#include <string>

namespace spark_rapids_jni::profiler {

bool create_nvtxw_stream(nvtxwInterfaceCore_t const* nvtxwInterface,
                         nvtxwSessionHandle_t const& session,
                         std::string const& name,
                         std::string const& domain,
                         nvtxwStreamHandle_t& stream);

int initialize_nvtxw(std::ifstream& in,
                     std::string const& outPath,
                     void*& nvtxwModuleHandle,
                     nvtxwInterfaceCore_t*& nvtxwInterface,
                     nvtxwSessionHandle_t& session,
                     nvtxwStreamHandle_t& stream,
                     std::optional<std::filesystem::path> const& nvtxw_backend_path = std::nullopt);

}  // namespace spark_rapids_jni::profiler
