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

#include "profiler_debug.hpp"

#include <iostream>
#include <sstream>

namespace spark_rapids_jni::profiler {

namespace {

std::string marker_flags_to_string(CUpti_ActivityFlag flags)
{
  std::string s("");
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS) { s += "INSTANTANEOUS "; }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_START) { s += "START "; }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_END) { s += "END "; }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE) { s += "SYNCACQUIRE "; }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_SUCCESS) { s += "SYNCACQUIRESUCCESS "; }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_FAILED) { s += "SYNCACQUIREFAILED "; }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_SYNC_RELEASE) { s += "SYNCRELEASE "; }
  return s;
}

std::string activity_object_kind_to_string(CUpti_ActivityObjectKind kind)
{
  switch (kind) {
    case CUPTI_ACTIVITY_OBJECT_PROCESS: return "PROCESS";
    case CUPTI_ACTIVITY_OBJECT_THREAD: return "THREAD";
    case CUPTI_ACTIVITY_OBJECT_DEVICE: return "DEVICE";
    case CUPTI_ACTIVITY_OBJECT_CONTEXT: return "CONTEXT";
    case CUPTI_ACTIVITY_OBJECT_STREAM: return "STREAM";
    case CUPTI_ACTIVITY_OBJECT_UNKNOWN: return "UNKNOWN";
    default: {
      std::ostringstream oss;
      oss << "UNRECOGNIZED(" << kind << ")";
      return oss.str();
    }
  }
}

}  // anonymous namespace

std::string activity_kind_to_string(CUpti_ActivityKind kind)
{
  switch (kind) {
    case CUPTI_ACTIVITY_KIND_MEMCPY: return "CUPTI_ACTIVITY_KIND_MEMCPY";
    case CUPTI_ACTIVITY_KIND_MEMSET: return "CUPTI_ACTIVITY_KIND_MEMSET";
    case CUPTI_ACTIVITY_KIND_KERNEL: return "CUPTI_ACTIVITY_KIND_KERNEL";
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: return "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL";
    case CUPTI_ACTIVITY_KIND_DRIVER: return "CPUTI_ACTIVITY_KIND_DRIVER";
    case CUPTI_ACTIVITY_KIND_RUNTIME: return "CUPTI_ACTIVITY_KIND_RUNTIME";
    case CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API: return "CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API";
    case CUPTI_ACTIVITY_KIND_EVENT: return "CUPTI_ACTIVITY_KIND_EVENT";
    case CUPTI_ACTIVITY_KIND_METRIC: return "CUPTI_ACTIVITY_KIND_METRIC";
    case CUPTI_ACTIVITY_KIND_DEVICE: return "CUPTI_ACTIVITY_KIND_DEVICE";
    case CUPTI_ACTIVITY_KIND_CONTEXT: return "CUPTI_ACTIVITY_KIND_CONTEXT";
    case CUPTI_ACTIVITY_KIND_NAME: return "CUPTI_ACTIVITY_KIND_NAME";
    case CUPTI_ACTIVITY_KIND_MARKER: return "CUPTI_ACTIVITY_KIND_MARKER";
    case CUPTI_ACTIVITY_KIND_MARKER_DATA: return "CUPTI_ACTIVITY_KIND_MARKER_DATA";
    case CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR: return "CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR";
    case CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS: return "CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS";
    case CUPTI_ACTIVITY_KIND_BRANCH: return "CUPTI_ACTIVITY_KIND_BRANCH";
    case CUPTI_ACTIVITY_KIND_OVERHEAD: return "CUPTI_ACTIVITY_KIND_OVERHEAD";
    case CUPTI_ACTIVITY_KIND_CDP_KERNEL: return "CUPTI_ACTIVITY_KIND_CDP_KERNEL";
    case CUPTI_ACTIVITY_KIND_PREEMPTION: return "CUPTI_ACTIVITY_KIND_PREEMPTION";
    case CUPTI_ACTIVITY_KIND_ENVIRONMENT: return "CUPTI_ACTIVITY_KIND_ENVIRONMENT";
    case CUPTI_ACTIVITY_KIND_EVENT_INSTANCE: return "CUPTI_ACTIVITY_KIND_EVENT_INSTANCE";
    case CUPTI_ACTIVITY_KIND_MEMCPY2: return "CUPTI_ACTIVITY_KIND_MEMCPY2";
    case CUPTI_ACTIVITY_KIND_METRIC_INSTANCE: return "CUPTI_ACTIVITY_KIND_METRIC_INSTANCE";
    case CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION:
      return "CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION";
    case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER:
      return "CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER";
    case CUPTI_ACTIVITY_KIND_FUNCTION: return "CUPTI_ACTIVITY_KIND_FUNCTION";
    case CUPTI_ACTIVITY_KIND_MODULE: return "CUPTI_ACTIVITY_KIND_MODULE";
    case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE: return "CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE";
    case CUPTI_ACTIVITY_KIND_SHARED_ACCESS: return "CUPTI_ACTIVITY_KIND_SHARED_ACCESS";
    case CUPTI_ACTIVITY_KIND_PC_SAMPLING: return "CUPTI_ACTIVITY_KIND_PC_SAMPLING";
    case CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO:
      return "CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO";
    case CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION:
      return "CUPTI_ACTIVITY_KIND_INSTRUCTION_CORRELATION";
    case CUPTI_ACTIVITY_KIND_OPENACC_DATA: return "CUPTI_ACTIVITY_KIND_OPENACC_DATA";
    case CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH: return "CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH";
    case CUPTI_ACTIVITY_KIND_OPENACC_OTHER: return "CUPTI_ACTIVITY_KIND_OPENACC_OTHER";
    case CUPTI_ACTIVITY_KIND_CUDA_EVENT: return "CUPTI_ACTIVITY_KIND_CUDA_EVENT";
    case CUPTI_ACTIVITY_KIND_STREAM: return "CUPTI_ACTIVITY_KIND_STREAM";
    case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: return "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION";
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
      return "CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION";
    case CUPTI_ACTIVITY_KIND_NVLINK: return "CUPTI_ACTIVITY_KIND_NVLINK";
    case CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT: return "CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT";
    case CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE:
      return "CUPTI_ACTIVITY_KIND_INSTANTANEOUS_EVENT_INSTANCE";
    case CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC:
      return "CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC";
    case CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE:
      return "CUPTI_ACTIVITY_KIND_INSTANTANEOUS_METRIC_INSTANCE";
    case CUPTI_ACTIVITY_KIND_MEMORY: return "CUPTI_ACTIVITY_KIND_MEMORY";
    case CUPTI_ACTIVITY_KIND_PCIE: return "CUPTI_ACTIVITY_KIND_PCIE";
    case CUPTI_ACTIVITY_KIND_OPENMP: return "CUPTI_ACTIVITY_KIND_OPENMP";
    case CUPTI_ACTIVITY_KIND_MEMORY2: return "CUPTI_ACTIVITY_KIND_MEMORY2";
    case CUPTI_ACTIVITY_KIND_MEMORY_POOL: return "CUPTI_ACTIVITY_KIND_MEMORY_POOL";
    case CUPTI_ACTIVITY_KIND_GRAPH_TRACE: return "CUPTI_ACTIVITY_KIND_GRAPH_TRACE";
    case CUPTI_ACTIVITY_KIND_JIT: return "CUPTI_ACTIVITY_KIND_JIT";
    default: {
      std::ostringstream oss;
      oss << "UNRECOGNIZED(" << kind << ")";
      return oss.str();
    }
  }
}

void print_cupti_buffer(uint8_t* buffer, size_t valid_size)
{
  if (valid_size > 0) {
    std::cerr << "PROFILER: CUPTI buffer size: " << valid_size << std::endl;
    CUpti_Activity* record_ptr = nullptr;
    auto rc                    = cuptiActivityGetNextRecord(buffer, valid_size, &record_ptr);
    while (rc == CUPTI_SUCCESS) {
      std::cerr << "RECORD: " << activity_kind_to_string(record_ptr->kind) << std::endl;
      switch (record_ptr->kind) {
        case CUPTI_ACTIVITY_KIND_DRIVER: {
          auto api_record  = reinterpret_cast<CUpti_ActivityAPI const*>(record_ptr);
          char const* name = nullptr;
          cuptiGetCallbackName(CUPTI_CB_DOMAIN_DRIVER_API, api_record->cbid, &name);
          name = name ? name : "NULL";
          std::cerr << "  NAME: " << name << " THREAD: " << api_record->threadId << std::endl;
          break;
        }
        case CUPTI_ACTIVITY_KIND_DEVICE: {
          auto device_record = reinterpret_cast<CUpti_ActivityDevice4 const*>(record_ptr);
          char const* name   = device_record->name != nullptr ? device_record->name : "NULL";
          std::cerr << "  " << activity_kind_to_string(device_record->kind) << " " << name
                    << std::endl;
          break;
        }
        case CUPTI_ACTIVITY_KIND_RUNTIME: {
          auto api_record  = reinterpret_cast<CUpti_ActivityAPI const*>(record_ptr);
          char const* name = nullptr;
          cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, api_record->cbid, &name);
          name = name ? name : "NULL";
          std::cerr << "  NAME: " << name << " THREAD: " << api_record->threadId << std::endl;
          break;
        }
        case CUPTI_ACTIVITY_KIND_MARKER: {
          auto marker_record = reinterpret_cast<CUpti_ActivityMarker2 const*>(record_ptr);
          std::cerr << "  FLAGS: " << marker_flags_to_string(marker_record->flags)
                    << " ID: " << marker_record->id
                    << " OBJECTKIND: " << activity_object_kind_to_string(marker_record->objectKind)
                    << " NAME: " << std::string(marker_record->name ? marker_record->name : "NULL")
                    << " DOMAIN: "
                    << std::string(marker_record->domain ? marker_record->domain : "NULL")
                    << std::endl;
          break;
        }
        case CUPTI_ACTIVITY_KIND_MARKER_DATA: {
          auto marker_record = reinterpret_cast<CUpti_ActivityMarkerData const*>(record_ptr);
          std::cerr << "  FLAGS: " << marker_flags_to_string(marker_record->flags)
                    << " ID: " << marker_record->id << " COLOR: " << marker_record->color
                    << " COLOR FLAG: " << marker_record->flags
                    << " CATEGORY: " << marker_record->category
                    << " DATA KIND: " << marker_record->payloadKind
                    << " DATA: " << marker_record->payload.metricValueUint64 << "/"
                    << marker_record->payload.metricValueDouble << std::endl;
          break;
        }
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
          auto kernel_record = reinterpret_cast<CUpti_ActivityKernel8 const*>(record_ptr);
          std::cerr << "  NAME: " << kernel_record->name << std::endl;
        }
        default: break;
      }
      rc = cuptiActivityGetNextRecord(buffer, valid_size, &record_ptr);
    }
  }
}

}  // namespace spark_rapids_jni::profiler
