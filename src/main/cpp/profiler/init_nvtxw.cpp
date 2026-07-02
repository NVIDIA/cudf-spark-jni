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

#include "init_nvtxw.h"

#include <cxxabi.h>

#include <cerrno>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace spark_rapids_jni::profiler {

bool create_nvtxw_stream(const nvtxwInterfaceCore_t* nvtxwInterface,
                         const nvtxwSessionHandle_t& session,
                         const std::string& name,
                         const std::string& domain,
                         nvtxwStreamHandle_t& stream)
{
  nvtxwResultCode_t result           = NVTXW3_RESULT_SUCCESS;
  nvtxwStreamAttributes_t streamAttr = {sizeof(nvtxwStreamAttributes_t),
                                        name.c_str(),
                                        domain.c_str(),
                                        "",
                                        NVTXW3_STREAM_ORDER_INTERLEAVING_NONE,
                                        NVTXW3_STREAM_ORDERING_TYPE_UNKNOWN,
                                        NVTXW3_STREAM_ORDERING_SKID_NONE,
                                        0};
  result                             = nvtxwInterface->StreamOpen(&stream, session, &streamAttr);
  if (result != NVTXW3_RESULT_SUCCESS) {
    fprintf(stderr, "StreamOpen failed with code %d\n", (int)result);
    return false;
  }
  if (!stream.opaque) {
    fprintf(stderr, "StreamOpen returned null stream handle!\n");
    return false;
  }
  return true;
}

/// outName: basename of output nsys-rep, without .nsys-rep extension
int initialize_nvtxw(std::ifstream& in,
                     const std::string& outPath,
                     void*& nvtxwModuleHandle,
                     nvtxwInterfaceCore_t*& nvtxwInterface,
                     nvtxwSessionHandle_t& session,
                     nvtxwStreamHandle_t& stream,
                     const std::optional<std::filesystem::path>& nvtxw_backend_path)
{
  nvtxwResultCode_t result = NVTXW3_RESULT_SUCCESS;
  int errorCode            = 0;

  // Get the basename for session name (needed by NVTXW API)
  std::filesystem::path p(outPath);
  std::string sessionName = p.stem().string();

  // initialize
  static const char soNameDefault[] = "libNvtxwBackend.so";
  const char* soName                = soNameDefault;
  // First check if a path was provided via command line option
  if (nvtxw_backend_path) {
    soName = nvtxw_backend_path->c_str();
  } else {
    // Otherwise, check the environment variable
    const char* backend_env = getenv("NVTXW_BACKEND");
    if (backend_env) { soName = backend_env; }
  }
  nvtxwGetInterface_t getInterfaceFunc = nullptr;
  result                               = nvtxwInitialize(
    NVTXW3_INIT_MODE_LIBRARY_FILENAME, soName, &getInterfaceFunc, &nvtxwModuleHandle);
  if (result != NVTXW3_RESULT_SUCCESS) {
    fprintf(stderr, "nvtxwInitialize failed with code %d\n", (int)result);
    if (result == NVTXW3_RESULT_LIBRARY_NOT_FOUND) {
      fprintf(stderr, "Failed to find NVTXW backend library: %s\n", soName);
      fprintf(stderr, "Please specify the path using one of these methods:\n");
      fprintf(stderr, "1. Command line option: --nvtxw-backend=PATH\n");
      fprintf(stderr,
              "2. Environment variable: export NVTXW_BACKEND=/path/to/libNvtxwBackend.so\n");
      fprintf(stderr, "Typically found in the Nsight Systems installation directory, e.g.:\n");
      fprintf(stderr, "/opt/nvidia/nsight-systems/VERSION/host-linux-x64/libNvtxwBackend.so\n");
    }
    return 1;
  }
  if (!getInterfaceFunc) {
    fprintf(stderr, "nvtxwInitialize returned null nvtxwGetInterface_t!\n");
    return 1;
  }

  const void* interfaceVoid;
  result = getInterfaceFunc(NVTXW3_INTERFACE_ID_CORE_V1, &interfaceVoid);
  if (result != NVTXW3_RESULT_SUCCESS) {
    fprintf(stderr, "getInterfaceFunc failed with code %d\n", (int)result);
    return 1;
  }
  if (!interfaceVoid) {
    fprintf(stderr, "getInterfaceFunc returned null nvtxwInterface pointer!\n");
    return 1;
  }
  nvtxwInterface = reinterpret_cast<nvtxwInterfaceCore_t*>((void*)interfaceVoid);

  // session begin
  char* sessionConfig                  = nullptr;
  nvtxwSessionAttributes_t sessionAttr = {
    sizeof(nvtxwSessionAttributes_t), sessionName.c_str(), sessionConfig};
  result = nvtxwInterface->SessionBegin(&session, &sessionAttr);
  free(sessionConfig);
  if (result != NVTXW3_RESULT_SUCCESS) {
    fprintf(stderr, "SessionBegin failed with code %d\n", (int)result);
    return 1;
  }
  if (!session.opaque) {
    fprintf(stderr, "SessionBegin returned null session handle!\n");
    return 1;
  }

  // stream open
  std::string streamName("CUPTI");
  std::string domainName("CUPTI");
  bool valid = create_nvtxw_stream(nvtxwInterface, session, streamName, domainName, stream);
  if (!valid) {
    errorCode |= 1;
    return errorCode;
  }
  // schema register
  result = nvtxwInterface->SchemaRegister(stream, NvidiaNvtxw::GetNameSchemaAttr());
  if (result != NVTXW3_RESULT_SUCCESS) {
    fprintf(stderr, "SchemaRegister failed for 'nameSchema' with code %d\n", (int)result);
    errorCode |= 2;
  }
  result = nvtxwInterface->SchemaRegister(stream, NvidiaNvtxw::GetNvtxRangePushPopSchemaAttr());
  if (result != NVTXW3_RESULT_SUCCESS) {
    fprintf(
      stderr, "SchemaRegister failed with 'nvtxRangePushPopSchema' with code %d\n", (int)result);
    errorCode |= 2;
  }
  result = nvtxwInterface->SchemaRegister(stream, NvidiaNvtxw::GetCuptiApiSchemaAttr());
  if (result != NVTXW3_RESULT_SUCCESS) {
    fprintf(stderr, "SchemaRegister failed with 'cuptiApiSchema' with code %d\n", (int)result);
    errorCode |= 2;
  }
  result = nvtxwInterface->SchemaRegister(stream, NvidiaNvtxw::GetCuptiDeviceSchemaAttr());
  if (result != NVTXW3_RESULT_SUCCESS) {
    fprintf(stderr, "SchemaRegister failed with 'cuptiDeviceSchema' with code %d\n", (int)result);
    errorCode |= 2;
  }
  result = nvtxwInterface->SchemaRegister(stream, NvidiaNvtxw::GetCuptiKernelSchemaAttr());
  if (result != NVTXW3_RESULT_SUCCESS) {
    fprintf(stderr, "SchemaRegister failed with 'cuptiKernelSchema' with code %d\n", (int)result);
    errorCode |= 2;
  }
  result = nvtxwInterface->SchemaRegister(stream, NvidiaNvtxw::GetCuptiMemcpySchemaAttr());
  if (result != NVTXW3_RESULT_SUCCESS) {
    fprintf(stderr, "SchemaRegister failed with 'cuptiMemcpySchema' with code %d\n", (int)result);
    errorCode |= 2;
  }
  result = nvtxwInterface->SchemaRegister(stream, NvidiaNvtxw::GetCuptiMemsetSchemaAttr());
  if (result != NVTXW3_RESULT_SUCCESS) {
    fprintf(stderr, "SchemaRegister failed with 'cuptiMemsetSchema' with code %d\n", (int)result);
    errorCode |= 2;
  }
  result = nvtxwInterface->SchemaRegister(stream, NvidiaNvtxw::GetCuptiOverheadSchemaAttr());
  if (result != NVTXW3_RESULT_SUCCESS) {
    fprintf(stderr, "SchemaRegister failed with 'cuptiOverheadSchema' with code %d\n", (int)result);
    errorCode |= 2;
  }
  return errorCode;
}

}  // namespace spark_rapids_jni::profiler
