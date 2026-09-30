/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include "profiler_serializer.hpp"

#include "profiler_debug.hpp"
#include "profiler_generated.h"
#include "spark_rapids_jni_version.h"

#include <cupti.h>

#include <iostream>

namespace spark_rapids_jni::profiler {

namespace {

constexpr uint32_t PROFILE_VERSION = 1;

flatbuffers::Offset<ActivityObjectId> add_object_id(flatbuffers::FlatBufferBuilder& fbb,
                                                    CUpti_ActivityObjectKind kind,
                                                    CUpti_ActivityObjectKindId const& object_id)
{
  switch (kind) {
    case CUPTI_ACTIVITY_OBJECT_PROCESS:
    case CUPTI_ACTIVITY_OBJECT_THREAD: {
      ActivityObjectIdBuilder aoib(fbb);
      aoib.add_process_id(object_id.pt.processId);
      if (kind == CUPTI_ACTIVITY_OBJECT_THREAD) { aoib.add_thread_id(object_id.pt.threadId); }
      return aoib.Finish();
    }
    case CUPTI_ACTIVITY_OBJECT_DEVICE:
    case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    case CUPTI_ACTIVITY_OBJECT_STREAM: {
      ActivityObjectIdBuilder aoib(fbb);
      aoib.add_device_id(object_id.dcs.deviceId);
      if (kind == CUPTI_ACTIVITY_OBJECT_CONTEXT || kind == CUPTI_ACTIVITY_OBJECT_STREAM) {
        aoib.add_context_id(object_id.dcs.contextId);
        if (kind == CUPTI_ACTIVITY_OBJECT_STREAM) { aoib.add_stream_id(object_id.dcs.streamId); }
      }
      return aoib.Finish();
    }
    default:
      std::cerr << "PROFILER: Unrecognized object kind: " << kind << std::endl;
      return flatbuffers::Offset<ActivityObjectId>();
  }
}

MarkerFlags marker_flags_to_fb(CUpti_ActivityFlag flags)
{
  uint8_t result = 0;
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_INSTANTANEOUS) { result |= MarkerFlags_Instantaneous; }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_START) { result |= MarkerFlags_Start; }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_END) { result |= MarkerFlags_End; }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE) { result |= MarkerFlags_SyncAcquire; }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_SUCCESS) {
    result |= MarkerFlags_SyncAcquireSuccess;
  }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_SYNC_ACQUIRE_FAILED) {
    result |= MarkerFlags_SyncAcquireFailed;
  }
  if (flags & CUPTI_ACTIVITY_FLAG_MARKER_SYNC_RELEASE) { result |= MarkerFlags_SyncRelease; }
  return static_cast<MarkerFlags>(result);
}

ChannelType to_channel_type(CUpti_ChannelType t)
{
  switch (t) {
    case CUPTI_CHANNEL_TYPE_INVALID: return ChannelType_Invalid;
    case CUPTI_CHANNEL_TYPE_COMPUTE: return ChannelType_Compute;
    case CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY: return ChannelType_AsyncMemcpy;
    default:
      std::cerr << "PROFILER: Unrecognized channel type: " << t << std::endl;
      return ChannelType_Invalid;
  }
}

LaunchType to_launch_type(uint8_t t)
{
  switch (t) {
    case CUPTI_ACTIVITY_LAUNCH_TYPE_REGULAR: return LaunchType_Regular;
    case CUPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_SINGLE_DEVICE:
      return LaunchType_CooperativeSingleDevice;
    case CUPTI_ACTIVITY_LAUNCH_TYPE_COOPERATIVE_MULTI_DEVICE:
      return LaunchType_CooperativeMultiDevice;
    default:
      std::cerr << "PROFILER: Unrecognized launch type: " << t << std::endl;
      return LaunchType_Regular;
  }
}

MemcpyFlags to_memcpy_flags(uint32_t flags)
{
  uint8_t result = 0;
  if (flags & CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC) { result |= MemcpyFlags_Async; }
  return static_cast<MemcpyFlags>(result);
}

MemcpyKind to_memcpy_kind(uint8_t k)
{
  switch (k) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN: return MemcpyKind_Unknown;
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return MemcpyKind_HtoD;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return MemcpyKind_DtoH;
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA: return MemcpyKind_HtoA;
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH: return MemcpyKind_AtoH;
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA: return MemcpyKind_AtoA;
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD: return MemcpyKind_AtoD;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA: return MemcpyKind_DtoA;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: return MemcpyKind_DtoD;
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: return MemcpyKind_HtoH;
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP: return MemcpyKind_PtoP;
    default:
      std::cerr << "PROFILER: Unrecognized memcpy kind: " << k << std::endl;
      return MemcpyKind_Unknown;
  }
}

MemoryKind to_memory_kind(uint8_t k)
{
  switch (k) {
    case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN: return MemoryKind_Unknown;
    case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE: return MemoryKind_Pageable;
    case CUPTI_ACTIVITY_MEMORY_KIND_PINNED: return MemoryKind_Pinned;
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE: return MemoryKind_Device;
    case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY: return MemoryKind_Array;
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED: return MemoryKind_Managed;
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC: return MemoryKind_DeviceStatic;
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC: return MemoryKind_ManagedStatic;
    default:
      std::cerr << "PROFILER: Unrecognized memory kind: " << k << std::endl;
      return MemoryKind_Unknown;
  }
}

MemsetFlags to_memset_flags(uint32_t flags)
{
  uint8_t result = 0;
  if (flags & CUPTI_ACTIVITY_FLAG_MEMSET_ASYNC) { result |= MemsetFlags_Async; }
  return static_cast<MemsetFlags>(result);
}

OverheadKind to_overhead_kind(CUpti_ActivityOverheadKind k)
{
  switch (k) {
    case CUPTI_ACTIVITY_OVERHEAD_UNKNOWN: return OverheadKind_Unknown;
    case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER: return OverheadKind_DriverCompiler;
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH: return OverheadKind_CUptiBufferFlush;
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION: return OverheadKind_CUptiInstrumentation;
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE: return OverheadKind_CUptiResource;
    default:
      std::cerr << "PROFILER: Unrecognized overhead kind: " << k << std::endl;
      return OverheadKind_Unknown;
  }
}

PartitionedGlobalCacheConfig to_partitioned_global_cache_config(
  CUpti_ActivityPartitionedGlobalCacheConfig c)
{
  switch (c) {
    case CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_UNKNOWN:
      return PartitionedGlobalCacheConfig_Unknown;
    case CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_NOT_SUPPORTED:
      return PartitionedGlobalCacheConfig_NotSupported;
    case CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_OFF:
      return PartitionedGlobalCacheConfig_Off;
    case CUPTI_ACTIVITY_PARTITIONED_GLOBAL_CACHE_CONFIG_ON: return PartitionedGlobalCacheConfig_On;
    default:
      std::cerr << "PROFILER: Unrecognized partitioned global cache config: " << c << std::endl;
      return PartitionedGlobalCacheConfig_Unknown;
  }
}

ShmemLimitConfig to_shmem_limit_config(CUpti_FuncShmemLimitConfig c)
{
  switch (c) {
    case CUPTI_FUNC_SHMEM_LIMIT_DEFAULT: return ShmemLimitConfig_Default;
    case CUPTI_FUNC_SHMEM_LIMIT_OPTIN: return ShmemLimitConfig_Optin;
    default:
      std::cerr << "PROFILER: Unrecognized shmem limit config: " << c << std::endl;
      return ShmemLimitConfig_Default;
  }
}

}  // anonymous namespace

profiler_serializer::profiler_serializer(
  JNIEnv* env, jobject writer, size_t buffer_size, size_t flush_threshold, bool capture_allocs)
  : env_(env),
    j_writer_(writer),
    flush_threshold_(flush_threshold),
    fbb_(buffer_size),
    capture_allocs_(capture_allocs)
{
  auto writer_class = env->GetObjectClass(writer);
  if (!writer_class) { throw std::runtime_error("Failed to locate class of data writer"); }
  j_write_method_ = env->GetMethodID(writer_class, "write", "(Ljava/nio/ByteBuffer;)V");
  if (!j_write_method_) { throw std::runtime_error("Failed to locate data writer write method"); }
  write_profile_header();
}

void profiler_serializer::write_profile_header()
{
  auto writer_version = fbb_.CreateString(spark_rapids_jni::Version);
  auto magic          = fbb_.CreateString("spark-rapids profile");
  auto header         = CreateProfileHeader(fbb_, magic, PROFILE_VERSION, writer_version);
  fbb_.FinishSizePrefixed(header);
  write_current_fb();
}

void profiler_serializer::process_cupti_buffer(uint8_t* buffer, size_t valid_size)
{
  report_num_dropped_records();
  if (valid_size > 0) {
    CUpti_Activity* record_ptr = nullptr;
    auto rc                    = cuptiActivityGetNextRecord(buffer, valid_size, &record_ptr);
    while (rc == CUPTI_SUCCESS) {
      switch (record_ptr->kind) {
        case CUPTI_ACTIVITY_KIND_DEVICE: {
          auto device_record = reinterpret_cast<CUpti_ActivityDevice4 const*>(record_ptr);
          process_device_activity(device_record);
          break;
        }
        case CUPTI_ACTIVITY_KIND_DRIVER:
        case CUPTI_ACTIVITY_KIND_RUNTIME: {
          auto api_record = reinterpret_cast<CUpti_ActivityAPI const*>(record_ptr);
          process_api_activity(api_record);
          break;
        }
        case CUPTI_ACTIVITY_KIND_MARKER: {
          auto marker = reinterpret_cast<CUpti_ActivityMarker2 const*>(record_ptr);
          process_marker_activity(marker);
          break;
        }
        case CUPTI_ACTIVITY_KIND_MARKER_DATA: {
          auto marker = reinterpret_cast<CUpti_ActivityMarkerData const*>(record_ptr);
          process_marker_data(marker);
          break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
          auto r = reinterpret_cast<CUpti_ActivityMemcpy5 const*>(record_ptr);
          process_memcpy(r);
          break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET: {
          auto r = reinterpret_cast<CUpti_ActivityMemset4 const*>(record_ptr);
          process_memset(r);
          break;
        }
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
          auto r = reinterpret_cast<CUpti_ActivityKernel8 const*>(record_ptr);
          process_kernel(r);
          break;
        }
        case CUPTI_ACTIVITY_KIND_OVERHEAD: {
          auto r = reinterpret_cast<CUpti_ActivityOverhead const*>(record_ptr);
          process_overhead(r);
          break;
        }
        default:
          std::cerr << "PROFILER: Ignoring activity record "
                    << activity_kind_to_string(record_ptr->kind) << std::endl;
          break;
      }
      if (fbb_.GetSize() >= flush_threshold_) { flush(); }
      rc = cuptiActivityGetNextRecord(buffer, valid_size, &record_ptr);
    }
  }
}

void profiler_serializer::flush()
{
  if (fbb_.GetSize() > 0) {
    using flatbuffers::Offset;
    using flatbuffers::Vector;
    Offset<Vector<Offset<ApiActivity>>> api_vec;
    Offset<Vector<Offset<DeviceActivity>>> device_vec;
    Offset<Vector<Offset<DroppedRecords>>> dropped_vec;
    Offset<Vector<Offset<KernelActivity>>> kernel_vec;
    Offset<Vector<Offset<MarkerActivity>>> marker_vec;
    Offset<Vector<Offset<MarkerData>>> marker_data_vec;
    Offset<Vector<Offset<MemcpyActivity>>> memcpy_vec;
    Offset<Vector<Offset<MemsetActivity>>> memset_vec;
    Offset<Vector<Offset<OverheadActivity>>> overhead_vec;
    if (api_offsets_.size() > 0) { api_vec = fbb_.CreateVector(api_offsets_); }
    if (device_offsets_.size() > 0) { device_vec = fbb_.CreateVector(device_offsets_); }
    if (dropped_offsets_.size() > 0) { dropped_vec = fbb_.CreateVector(dropped_offsets_); }
    if (kernel_offsets_.size() > 0) { kernel_vec = fbb_.CreateVector(kernel_offsets_); }
    if (marker_offsets_.size() > 0) { marker_vec = fbb_.CreateVector(marker_offsets_); }
    if (marker_data_offsets_.size() > 0) {
      marker_data_vec = fbb_.CreateVector(marker_data_offsets_);
    }
    if (memcpy_offsets_.size() > 0) { memcpy_vec = fbb_.CreateVector(memcpy_offsets_); }
    if (memset_offsets_.size() > 0) { memset_vec = fbb_.CreateVector(memset_offsets_); }
    if (overhead_offsets_.size() > 0) { overhead_vec = fbb_.CreateVector(overhead_offsets_); }
    ActivityRecordsBuilder arb(fbb_);
    arb.add_api(api_vec);
    arb.add_device(device_vec);
    arb.add_dropped(dropped_vec);
    arb.add_kernel(kernel_vec);
    arb.add_marker(marker_vec);
    arb.add_marker_data(marker_data_vec);
    arb.add_memcpy(memcpy_vec);
    arb.add_memset(memset_vec);
    arb.add_overhead(overhead_vec);
    auto r = arb.Finish();
    fbb_.FinishSizePrefixed(r);
    write_current_fb();
  }
}

void profiler_serializer::process_api_activity(CUpti_ActivityAPI const* r)
{
  if (r->start == 0 || r->end == 0) {
    // Ignore records with bad timestamps
    return;
  }
  auto api_kind = ApiKind_Runtime;
  if (r->kind == CUPTI_ACTIVITY_KIND_DRIVER) {
    api_kind = ApiKind_Driver;
  } else if (r->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
    // skip some very common and uninteresting APIs to reduce the profile size
    switch (r->cbid) {
      case CUPTI_RUNTIME_TRACE_CBID_cudaGetDevice_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaGetLastError_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaPeekAtLastError_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetAttribute_v5000: return;
      case CUPTI_RUNTIME_TRACE_CBID_cudaMallocAsync_v11020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMallocAsync_ptsz_v11020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMallocFromPoolAsync_v11020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMallocFromPoolAsync_ptsz_v11020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_v11020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_ptsz_v11020:
        if (capture_allocs_) { break; }
        return;
      default: break;
    }
  } else {
    std::cerr << "PROFILER: Ignoring API activity record kind: " << activity_kind_to_string(r->kind)
              << std::endl;
    return;
  }
  ApiActivityBuilder aab(fbb_);
  aab.add_kind(api_kind);
  aab.add_cbid(r->cbid);
  aab.add_start(r->start);
  aab.add_end(r->end);
  aab.add_process_id(r->processId);
  aab.add_thread_id(r->threadId);
  aab.add_correlation_id(r->correlationId);
  aab.add_return_value(r->returnValue);
  api_offsets_.push_back(aab.Finish());
}

void profiler_serializer::process_device_activity(CUpti_ActivityDevice4 const* r)
{
  auto has_name = r->name != nullptr;
  flatbuffers::Offset<flatbuffers::String> name;
  if (has_name) { name = fbb_.CreateSharedString(r->name); }
  DeviceActivityBuilder dab(fbb_);
  dab.add_global_memory_bandwidth(r->globalMemoryBandwidth);
  dab.add_global_memory_size(r->globalMemorySize);
  dab.add_constant_memory_size(r->constantMemorySize);
  dab.add_l2_cache_size(r->l2CacheSize);
  dab.add_num_threads_per_warp(r->numThreadsPerWarp);
  dab.add_core_clock_rate(r->coreClockRate);
  dab.add_num_memcpy_engines(r->numMemcpyEngines);
  dab.add_num_multiprocessors(r->numMultiprocessors);
  dab.add_max_ipc(r->maxIPC);
  dab.add_max_warps_per_multiprocessor(r->maxWarpsPerMultiprocessor);
  dab.add_max_blocks_per_multiprocessor(r->maxBlocksPerMultiprocessor);
  dab.add_max_shared_memory_per_multiprocessor(r->maxSharedMemoryPerMultiprocessor);
  dab.add_max_registers_per_multiprocessor(r->maxRegistersPerMultiprocessor);
  dab.add_max_registers_per_block(r->maxRegistersPerBlock);
  dab.add_max_shared_memory_per_block(r->maxSharedMemoryPerBlock);
  dab.add_max_threads_per_block(r->maxThreadsPerBlock);
  dab.add_max_block_dim_x(r->maxBlockDimX);
  dab.add_max_block_dim_y(r->maxBlockDimY);
  dab.add_max_block_dim_z(r->maxBlockDimZ);
  dab.add_max_grid_dim_x(r->maxGridDimX);
  dab.add_max_grid_dim_y(r->maxGridDimY);
  dab.add_max_grid_dim_z(r->maxGridDimZ);
  dab.add_compute_capability_major(r->computeCapabilityMajor);
  dab.add_compute_capability_minor(r->computeCapabilityMinor);
  dab.add_id(r->id);
  dab.add_ecc_enabled(r->eccEnabled);
  dab.add_name(name);
  device_offsets_.push_back(dab.Finish());
}

void profiler_serializer::process_dropped_records(size_t num_dropped)
{
  auto dropped = CreateDroppedRecords(fbb_, num_dropped);
  dropped_offsets_.push_back(dropped);
}

void profiler_serializer::process_kernel(CUpti_ActivityKernel8 const* r)
{
  if (r->start == 0 || r->end == 0) {
    // Ignore records with invalid timestamps
    return;
  }
  auto has_name = r->name != nullptr;
  flatbuffers::Offset<flatbuffers::String> name;
  if (has_name) { name = fbb_.CreateSharedString(r->name); }
  KernelActivityBuilder kab(fbb_);
  kab.add_requested(r->cacheConfig.config.requested);
  kab.add_executed(r->cacheConfig.config.executed);
  kab.add_shared_memory_config(r->sharedMemoryConfig);
  kab.add_registers_per_thread(r->registersPerThread);
  kab.add_partitioned_global_cache_requested(
    to_partitioned_global_cache_config(r->partitionedGlobalCacheRequested));
  kab.add_partitioned_global_cache_executed(
    to_partitioned_global_cache_config(r->partitionedGlobalCacheExecuted));
  kab.add_start(r->start);
  kab.add_end(r->end);
  kab.add_completed(r->completed);
  kab.add_device_id(r->deviceId);
  kab.add_context_id(r->contextId);
  kab.add_stream_id(r->streamId);
  kab.add_grid_x(r->gridX);
  kab.add_grid_y(r->gridY);
  kab.add_grid_z(r->gridZ);
  kab.add_block_x(r->blockX);
  kab.add_block_y(r->blockY);
  kab.add_block_z(r->blockZ);
  kab.add_static_shared_memory(r->staticSharedMemory);
  kab.add_dynamic_shared_memory(r->dynamicSharedMemory);
  kab.add_local_memory_per_thread(r->localMemoryPerThread);
  kab.add_local_memory_total(r->localMemoryTotal);
  kab.add_correlation_id(r->correlationId);
  kab.add_grid_id(r->gridId);
  kab.add_name(name);
  kab.add_queued(r->queued);
  kab.add_submitted(r->submitted);
  kab.add_launch_type(to_launch_type(r->launchType));
  kab.add_is_shared_memory_carveout_requested(r->isSharedMemoryCarveoutRequested);
  kab.add_shared_memory_carveout_requested(r->sharedMemoryCarveoutRequested);
  kab.add_shared_memory_executed(r->sharedMemoryExecuted);
  kab.add_graph_node_id(r->graphNodeId);
  kab.add_shmem_limit_config(to_shmem_limit_config(r->shmemLimitConfig));
  kab.add_graph_id(r->graphId);
  kab.add_channel_id(r->channelID);
  kab.add_channel_type(to_channel_type(r->channelType));
  kab.add_cluster_x(r->clusterX);
  kab.add_cluster_y(r->clusterY);
  kab.add_cluster_z(r->clusterZ);
  kab.add_cluster_scheduling_policy(r->clusterSchedulingPolicy);
  kab.add_local_memory_total_v2(r->localMemoryTotal_v2);
  kernel_offsets_.push_back(kab.Finish());
}

void profiler_serializer::process_marker_activity(CUpti_ActivityMarker2 const* r)
{
  if (r->timestamp == 0) {
    // Ignore records with invalid timestamps
    return;
  }
  auto object_id  = add_object_id(fbb_, r->objectKind, r->objectId);
  auto has_name   = r->name != nullptr;
  auto has_domain = r->domain != nullptr;
  flatbuffers::Offset<flatbuffers::String> name;
  flatbuffers::Offset<flatbuffers::String> domain;
  if (has_name) { name = fbb_.CreateSharedString(r->name); }
  if (has_domain) { domain = fbb_.CreateSharedString(r->domain); }
  MarkerActivityBuilder mab(fbb_);
  mab.add_flags(marker_flags_to_fb(r->flags));
  mab.add_timestamp(r->timestamp);
  mab.add_id(r->id);
  mab.add_object_id(object_id);
  mab.add_name(name);
  mab.add_domain(domain);
  marker_offsets_.push_back(mab.Finish());
}

void profiler_serializer::process_marker_data(CUpti_ActivityMarkerData const* r)
{
  if (r->flags == 0 && r->color == 0 && r->category == 0) {
    // Ignore uninteresting marker data records
    return;
  }
  MarkerDataBuilder mdb(fbb_);
  mdb.add_flags(marker_flags_to_fb(r->flags));
  mdb.add_id(r->id);
  mdb.add_color(r->color);
  mdb.add_category(r->category);
  marker_data_offsets_.push_back(mdb.Finish());
}

void profiler_serializer::process_memcpy(CUpti_ActivityMemcpy5 const* r)
{
  if (r->start == 0 || r->end == 0) {
    // Ignore records with invalid timestamps
    return;
  }
  MemcpyActivityBuilder mab(fbb_);
  mab.add_copy_kind(to_memcpy_kind(r->copyKind));
  mab.add_src_kind(to_memory_kind(r->srcKind));
  mab.add_dst_kind(to_memory_kind(r->dstKind));
  mab.add_flags(to_memcpy_flags(r->flags));
  mab.add_bytes(r->bytes);
  mab.add_start(r->start);
  mab.add_end(r->end);
  mab.add_device_id(r->deviceId);
  mab.add_context_id(r->contextId);
  mab.add_stream_id(r->streamId);
  mab.add_correlation_id(r->correlationId);
  mab.add_runtime_correlation_id(r->runtimeCorrelationId);
  mab.add_graph_node_id(r->graphNodeId);
  mab.add_graph_id(r->graphId);
  mab.add_channel_id(r->channelID);
  mab.add_channel_type(to_channel_type(r->channelType));
  memcpy_offsets_.push_back(mab.Finish());
}

void profiler_serializer::process_memset(CUpti_ActivityMemset4 const* r)
{
  if (r->start == 0 || r->end == 0) {
    // Ignore records with invalid timestamps
    return;
  }
  MemsetActivityBuilder mab(fbb_);
  mab.add_value(r->value);
  mab.add_bytes(r->bytes);
  mab.add_start(r->start);
  mab.add_end(r->end);
  mab.add_device_id(r->deviceId);
  mab.add_context_id(r->contextId);
  mab.add_stream_id(r->streamId);
  mab.add_correlation_id(r->correlationId);
  mab.add_flags(to_memset_flags(r->flags));
  mab.add_memory_kind(to_memory_kind(r->memoryKind));
  mab.add_graph_node_id(r->graphNodeId);
  mab.add_graph_id(r->graphId);
  mab.add_channel_id(r->channelID);
  mab.add_channel_type(to_channel_type(r->channelType));
  memset_offsets_.push_back(mab.Finish());
}

void profiler_serializer::process_overhead(CUpti_ActivityOverhead const* r)
{
  if (r->start == 0 || r->end == 0) {
    // Ignore records with invalid timestamps
    return;
  }
  auto object_id = add_object_id(fbb_, r->objectKind, r->objectId);
  OverheadActivityBuilder oab(fbb_);
  oab.add_overhead_kind(to_overhead_kind(r->overheadKind));
  oab.add_object_id(object_id);
  oab.add_start(r->start);
  oab.add_end(r->end);
  overhead_offsets_.push_back(oab.Finish());
}

// Query CUPTI for dropped records, and if any, record in the current activity record
void profiler_serializer::report_num_dropped_records()
{
  size_t num_dropped = 0;
  auto rc            = cuptiActivityGetNumDroppedRecords(NULL, 0, &num_dropped);
  if (rc == CUPTI_SUCCESS && num_dropped > 0) { process_dropped_records(num_dropped); }
}

// Write out the current flatbuffer and reset state for the next flatbuffer.
void profiler_serializer::write_current_fb()
{
  auto fb_size = fbb_.GetSize();
  if (fb_size > 0) {
    auto fb          = fbb_.GetBufferPointer();
    auto bytebuf_obj = env_->NewDirectByteBuffer(fb, fb_size);
    if (bytebuf_obj != nullptr) {
      env_->CallVoidMethod(j_writer_, j_write_method_, bytebuf_obj);
    } else {
      std::cerr << "PROFILER: Unable to create ByteBuffer for writer" << std::endl;
    }
  }
  fbb_.Clear();
  api_offsets_.clear();
  device_offsets_.clear();
  dropped_offsets_.clear();
  kernel_offsets_.clear();
  marker_offsets_.clear();
  marker_data_offsets_.clear();
  memcpy_offsets_.clear();
  memset_offsets_.clear();
  overhead_offsets_.clear();
}

}  // namespace spark_rapids_jni::profiler
