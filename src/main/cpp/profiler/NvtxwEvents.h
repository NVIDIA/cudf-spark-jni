/*
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
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

#include <nvtx3/nvToolsExtPayload.h>

#include <stdint.h>

namespace NvidiaNvtxw {

namespace PayloadSchemaId {
static constexpr uint64_t nameId              = 0xffffff00;
static constexpr uint64_t nvtxRangePushPopId  = 0xffffff01;
static constexpr uint64_t cuptiApiId          = 0xffffff02;
static constexpr uint64_t cuptiMemcpyId       = 0xffffff03;
static constexpr uint64_t cuptiMemsetId       = 0xffffff04;
static constexpr uint64_t cuptiDeviceId       = 0xffffff05;
static constexpr uint64_t cuptiKernelId       = 0xffffff06;
static constexpr uint64_t cuptiOverheadId     = 0xffffff07;
static constexpr uint64_t nvtxRangeStartEndId = 0xffffff08;
};  // namespace PayloadSchemaId

const nvtxPayloadSchemaAttr_t* GetNameSchemaAttr();

struct nvtxRangeEvent {
  uint64_t time_start;
  uint64_t time_stop;
  const char* name;
  uint32_t process_id;
  uint32_t thread_id;
  uint32_t color;
};
const nvtxPayloadSchemaAttr_t* GetNvtxRangePushPopSchemaAttr();
const nvtxPayloadSchemaAttr_t* GetNvtxRangeStartEndSchemaAttr();
struct cuptiApiEvent {
  uint64_t time_start;
  uint64_t time_stop;
  uint32_t kind;
  uint32_t cbid;
  uint32_t process_id;
  uint32_t thread_id;
  uint32_t correlation_id;
  uint32_t return_value;
};
const nvtxPayloadSchemaAttr_t* GetCuptiApiSchemaAttr();
struct cuptiDevice {
  uint64_t global_memory_bandwidth;
  uint64_t global_memory_size;
  uint32_t constant_memory_size;
  uint32_t l2_cache_size;
  uint32_t num_threads_per_warp;
  uint32_t core_clock_rate;
  uint32_t num_memcpy_engines;
  uint32_t num_multiprocessors;
  uint32_t max_ipc;
  uint32_t max_warps_per_multiprocessor;
  uint32_t max_blocks_per_multiprocessor;
  uint32_t max_shared_memory_per_multiprocessor;
  uint32_t max_registers_per_multiprocessor;
  uint32_t max_registers_per_block;
  uint32_t max_shared_memory_per_block;
  uint32_t max_threads_per_block;
  uint32_t max_block_dim_x;
  uint32_t max_block_dim_y;
  uint32_t max_block_dim_z;
  uint32_t max_grid_dim_x;
  uint32_t max_grid_dim_y;
  uint32_t max_grid_dim_z;
  uint32_t compute_capability_major;
  uint32_t compute_capability_minor;
  uint32_t id;
  uint32_t ecc_enabled;
  const char* name;
};
const nvtxPayloadSchemaAttr_t* GetCuptiDeviceSchemaAttr();
struct cuptiKernelEvent {
  uint64_t time_start;
  uint64_t time_stop;
  uint64_t completed;
  uint64_t grid_id;
  uint64_t queued;
  uint64_t submitted;
  uint64_t graph_node_id;
  uint64_t local_memory_total_v2;
  const char* name;
  uint32_t device_id;
  uint32_t context_id;
  uint32_t stream_id;
  uint32_t process_id;
  uint32_t grid_x;
  uint32_t grid_y;
  uint32_t grid_z;
  uint32_t block_x;
  uint32_t block_y;
  uint32_t block_z;
  uint32_t static_shared_memory;
  uint32_t dynamic_shared_memory;
  uint32_t local_memory_per_thread;
  uint32_t local_memory_total;
  uint32_t correlation_id;
  uint32_t shared_memory_executed;
  uint32_t graph_id;
  uint32_t channel_id;
  uint32_t cluster_x;
  uint32_t cluster_y;
  uint32_t cluster_z;
  uint32_t cluster_scheduling_policy;
  uint16_t registers_per_thread;
  uint8_t requested;
  uint8_t executed;
  uint8_t shared_memory_config;
  uint8_t partitioned_global_cache_requested;
  uint8_t partitioned_global_cache_executed;
  uint8_t launch_type;
  uint8_t is_shared_memory_carveout_requested;
  uint8_t shared_memory_carveout_requested;
  uint8_t shmem_limit_config;
  uint8_t channel_type;
};
const nvtxPayloadSchemaAttr_t* GetCuptiKernelSchemaAttr();

struct cuptiMemcpyEvent {
  uint64_t time_start;
  uint64_t time_stop;
  uint64_t bytes;
  uint64_t graph_node_id;
  uint32_t device_id;
  uint32_t context_id;
  uint32_t stream_id;
  uint32_t process_id;
  uint32_t correlation_id;
  uint32_t runtime_correlation_id;
  uint32_t graph_id;
  uint32_t channel_id;
  uint8_t channelType;
  uint8_t copy_kind;
  uint8_t src_kind;
  uint8_t dst_kind;
};
const nvtxPayloadSchemaAttr_t* GetCuptiMemcpySchemaAttr();

struct cuptiMemsetEvent {
  uint64_t time_start;
  uint64_t time_stop;
  uint64_t bytes;
  uint64_t graph_node_id;
  uint32_t device_id;
  uint32_t context_id;
  uint32_t stream_id;
  uint32_t process_id;
  uint32_t correlation_id;
  uint32_t graph_id;
  uint32_t channel_id;
  uint32_t value;
  uint8_t channelType;
  uint8_t mem_kind;
  uint8_t flags;
};
const nvtxPayloadSchemaAttr_t* GetCuptiMemsetSchemaAttr();
struct cuptiOverheadEvent {
  uint64_t time_start;
  uint64_t time_stop;
  uint32_t process_id;
  uint32_t thread_id;
  uint8_t overhead_kind;
};
const nvtxPayloadSchemaAttr_t* GetCuptiOverheadSchemaAttr();

}  // namespace NvidiaNvtxw
