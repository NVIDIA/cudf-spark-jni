/*
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "NvtxwEvents.h"

#include <type_traits>

namespace NvidiaNvtxw {

#define PAYLOAD_ENTRY_SIMPLE(flags, type, name) \
  {(flags), (type), (name), nullptr, 0, 0, nullptr, nullptr}

// The C string containing the event's name must be provided in a special way.
static const nvtxPayloadSchemaEntry_t nameSchema[] = {PAYLOAD_ENTRY_SIMPLE(
  NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE | NVTX_PAYLOAD_ENTRY_FLAG_ARRAY_ZERO_TERMINATED,
  NVTX_PAYLOAD_ENTRY_TYPE_CSTRING,
  "name")};
static const nvtxPayloadSchemaAttr_t nameSchemaAttr{
  /*.fieldMask = */
  NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_FLAGS |
    NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
    NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_SCHEMA_ID,
  /*.name = */
  nullptr,
  /*.type = */
  NVTX_PAYLOAD_SCHEMA_TYPE_DYNAMIC,
  /*.flags = */
  NVTX_PAYLOAD_SCHEMA_FLAG_REFERENCED,
  /*.entries = */
  nameSchema,
  /*.numEntries = */
  std::extent<decltype(nameSchema)>::value,
  /*.payloadStaticSize = */
  0,
  /*.packAlign = */
  0,
  /*.schemaId = */
  NvidiaNvtxw::PayloadSchemaId::nameId,
  /*.extension = */
  nullptr};

static const nvtxPayloadSchemaEntry_t nvtxRangeSchema[] = {
  PAYLOAD_ENTRY_SIMPLE(NVTX_PAYLOAD_ENTRY_FLAG_RANGE_BEGIN | NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE,
                       NVTX_PAYLOAD_ENTRY_TYPE_UINT64,
                       "time_start"),
  PAYLOAD_ENTRY_SIMPLE(NVTX_PAYLOAD_ENTRY_FLAG_RANGE_END | NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE,
                       NVTX_PAYLOAD_ENTRY_TYPE_UINT64,
                       "time_stop"),
  PAYLOAD_ENTRY_SIMPLE(NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE | NVTX_PAYLOAD_ENTRY_FLAG_POINTER,
                       NVTX_PAYLOAD_ENTRY_TYPE_CSTRING,
                       "name"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_PID_UINT32, "process_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_TID_UINT32, "thread_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_COLOR_ARGB, "color"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "push_pop")};
// TimeBase = Relative
static const nvtxPayloadSchemaAttr_t nvtxRangePushPopSchemaAttr = {
  NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE |
    NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
    NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_SCHEMA_ID,
  "NVTX Range Push Pop Event",
  NVTX_PAYLOAD_SCHEMA_TYPE_STATIC,
  NVTX_PAYLOAD_SCHEMA_FLAG_NONE,
  nvtxRangeSchema,
  std::extent<decltype(nvtxRangeSchema)>::value,
  sizeof(struct NvidiaNvtxw::nvtxRangeEvent),
  0,
  NvidiaNvtxw::PayloadSchemaId::nvtxRangePushPopId,
  nullptr};
// TimeBase = Relative
static const nvtxPayloadSchemaAttr_t nvtxRangeStartEndSchemaAttr = {
  NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE |
    NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
    NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_SCHEMA_ID,
  "NVTX Range Start End Event",
  NVTX_PAYLOAD_SCHEMA_TYPE_STATIC,
  NVTX_PAYLOAD_SCHEMA_FLAG_NONE,
  nvtxRangeSchema,
  std::extent<decltype(nvtxRangeSchema)>::value,
  sizeof(struct NvidiaNvtxw::nvtxRangeEvent),
  0,
  NvidiaNvtxw::PayloadSchemaId::nvtxRangeStartEndId,
  nullptr};

static const nvtxPayloadSchemaEntry_t cuptiApiSchema[] = {
  PAYLOAD_ENTRY_SIMPLE(NVTX_PAYLOAD_ENTRY_FLAG_RANGE_BEGIN | NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE,
                       NVTX_PAYLOAD_ENTRY_TYPE_UINT64,
                       "time_start"),
  PAYLOAD_ENTRY_SIMPLE(NVTX_PAYLOAD_ENTRY_FLAG_RANGE_END | NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE,
                       NVTX_PAYLOAD_ENTRY_TYPE_UINT64,
                       "time_stop"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "kind"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "cbid"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_PID_UINT32, "process_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_TID_UINT32, "thread_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "correlation_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "return_value")};
static const nvtxPayloadSchemaAttr_t cuptiApiSchemaAttr = {
  NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE |
    NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
    NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_SCHEMA_ID,
  "CUPTI API Activity",
  NVTX_PAYLOAD_SCHEMA_TYPE_STATIC,
  NVTX_PAYLOAD_SCHEMA_FLAG_NONE,
  cuptiApiSchema,
  std::extent<decltype(cuptiApiSchema)>::value,
  sizeof(struct NvidiaNvtxw::cuptiApiEvent),
  0,
  NvidiaNvtxw::PayloadSchemaId::cuptiApiId,
  nullptr};
static const nvtxPayloadSchemaEntry_t cuptiDeviceSchema[] = {
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT64, "global_memory_bandwidth"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT64, "global_memory_size"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "constant_memory_size"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "l2_cache_size"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "num_threads_per_warp"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "core_clock_rate"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "num_memcpy_engines"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "num_multiprocessors"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "max_ipc"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "max_warps_per_multiprocessor"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "max_blocks_per_multiprocessor"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "max_shared_memory_per_multiprocessor"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "max_registers_per_multiprocessor"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "max_registers_per_block"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "max_shared_memory_per_block"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "max_threads_per_block"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "max_block_dim_x"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "max_block_dim_y"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "max_block_dim_z"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "max_grid_dim_x"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "max_grid_dim_y"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "max_grid_dim_z"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "compute_capability_major"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "compute_capability_minor"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "ecc_enabled"),
  PAYLOAD_ENTRY_SIMPLE(NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE | NVTX_PAYLOAD_ENTRY_FLAG_POINTER,
                       NVTX_PAYLOAD_ENTRY_TYPE_CSTRING,
                       "name")};
static const nvtxPayloadSchemaAttr_t cuptiDeviceSchemaAttr = {
  NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE |
    NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
    NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_SCHEMA_ID,
  "CUPTI Device",
  NVTX_PAYLOAD_SCHEMA_TYPE_STATIC,
  NVTX_PAYLOAD_SCHEMA_FLAG_NONE,
  cuptiDeviceSchema,
  std::extent<decltype(cuptiDeviceSchema)>::value,
  sizeof(struct NvidiaNvtxw::cuptiDevice),
  0,
  NvidiaNvtxw::PayloadSchemaId::cuptiDeviceId,
  nullptr};
static const nvtxPayloadSchemaEntry_t cuptiKernelSchema[] = {
  PAYLOAD_ENTRY_SIMPLE(NVTX_PAYLOAD_ENTRY_FLAG_RANGE_BEGIN | NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE,
                       NVTX_PAYLOAD_ENTRY_TYPE_UINT64,
                       "time_start"),
  PAYLOAD_ENTRY_SIMPLE(NVTX_PAYLOAD_ENTRY_FLAG_RANGE_END | NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE,
                       NVTX_PAYLOAD_ENTRY_TYPE_UINT64,
                       "time_stop"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT64, "completed"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT64, "grid_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT64, "queued"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT64, "submitted"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT64, "graph_node_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT64, "local_memory_total_v2"),
  PAYLOAD_ENTRY_SIMPLE(NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE | NVTX_PAYLOAD_ENTRY_FLAG_POINTER,
                       NVTX_PAYLOAD_ENTRY_TYPE_CSTRING,
                       "name"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "device_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "context_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "stream_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_PID_UINT32, "process_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "grid_x"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "grid_y"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "grid_z"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "block_x"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "block_y"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "block_z"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "static_shared_memory"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "dynamic_shared_memory"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "local_memory_per_thread"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "local_memory_total"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "correlation_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "shared_memory_executed"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "graph_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "channel_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "cluster_x"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "cluster_y"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "cluster_z"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "cluster_scheduling_policy"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT16, "registers_per_thread"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "requested"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "executed"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "shared_memory_config"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "partitioned_global_cache_requested"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "partitioned_global_cache_executed"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "launch_type"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "is_shared_memory_carveout_requested"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "shared_memory_carveout_requested"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "shmem_limit_config"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "channel_type")};
static const nvtxPayloadSchemaAttr_t cuptiKernelSchemaAttr = {
  NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE |
    NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
    NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_SCHEMA_ID,
  "CUPTI Kernel",
  NVTX_PAYLOAD_SCHEMA_TYPE_STATIC,
  NVTX_PAYLOAD_SCHEMA_FLAG_NONE,
  cuptiKernelSchema,
  std::extent<decltype(cuptiKernelSchema)>::value,
  sizeof(struct NvidiaNvtxw::cuptiKernelEvent),
  0,
  NvidiaNvtxw::PayloadSchemaId::cuptiKernelId,
  nullptr};
static const nvtxPayloadSchemaEntry_t cuptiMemcpySchema[] = {
  PAYLOAD_ENTRY_SIMPLE(NVTX_PAYLOAD_ENTRY_FLAG_RANGE_BEGIN | NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE,
                       NVTX_PAYLOAD_ENTRY_TYPE_UINT64,
                       "time_start"),
  PAYLOAD_ENTRY_SIMPLE(NVTX_PAYLOAD_ENTRY_FLAG_RANGE_END | NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE,
                       NVTX_PAYLOAD_ENTRY_TYPE_UINT64,
                       "time_stop"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT64, "bytes"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT64, "graph_node_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "device_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "context_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "stream_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_PID_UINT32, "process_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "correlation_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "runtime_correlation_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "graph_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "channel_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "channel_type"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "copy_kind"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "src_kind"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "dst_kind")};
static const nvtxPayloadSchemaAttr_t cuptiMemcpySchemaAttr = {
  NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE |
    NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
    NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_SCHEMA_ID,
  "CUPTI Memcpy",
  NVTX_PAYLOAD_SCHEMA_TYPE_STATIC,
  NVTX_PAYLOAD_SCHEMA_FLAG_NONE,
  cuptiMemcpySchema,
  std::extent<decltype(cuptiMemcpySchema)>::value,
  sizeof(struct NvidiaNvtxw::cuptiMemcpyEvent),
  0,
  NvidiaNvtxw::PayloadSchemaId::cuptiMemcpyId,
  nullptr};
static const nvtxPayloadSchemaEntry_t cuptiMemsetSchema[] = {
  PAYLOAD_ENTRY_SIMPLE(NVTX_PAYLOAD_ENTRY_FLAG_RANGE_BEGIN | NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE,
                       NVTX_PAYLOAD_ENTRY_TYPE_UINT64,
                       "time_start"),
  PAYLOAD_ENTRY_SIMPLE(NVTX_PAYLOAD_ENTRY_FLAG_RANGE_END | NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE,
                       NVTX_PAYLOAD_ENTRY_TYPE_UINT64,
                       "time_stop"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT64, "bytes"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT64, "graph_node_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "device_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "context_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "stream_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_PID_UINT32, "process_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "correlation_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "graph_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "channel_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT32, "value"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "channel_type"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "mem_kind"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "flags")};
static const nvtxPayloadSchemaAttr_t cuptiMemsetSchemaAttr = {
  NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE |
    NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
    NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_SCHEMA_ID,
  "CUPTI Memset",
  NVTX_PAYLOAD_SCHEMA_TYPE_STATIC,
  NVTX_PAYLOAD_SCHEMA_FLAG_NONE,
  cuptiMemsetSchema,
  std::extent<decltype(cuptiMemsetSchema)>::value,
  sizeof(struct NvidiaNvtxw::cuptiMemsetEvent),
  0,
  NvidiaNvtxw::PayloadSchemaId::cuptiMemsetId,
  nullptr};
static const nvtxPayloadSchemaEntry_t cuptiOverheadSchema[] = {
  PAYLOAD_ENTRY_SIMPLE(NVTX_PAYLOAD_ENTRY_FLAG_RANGE_BEGIN | NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE,
                       NVTX_PAYLOAD_ENTRY_TYPE_UINT64,
                       "time_start"),
  PAYLOAD_ENTRY_SIMPLE(NVTX_PAYLOAD_ENTRY_FLAG_RANGE_END | NVTX_PAYLOAD_ENTRY_FLAG_EVENT_MESSAGE,
                       NVTX_PAYLOAD_ENTRY_TYPE_UINT64,
                       "time_stop"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_PID_UINT32, "process_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_TID_UINT32, "thread_id"),
  PAYLOAD_ENTRY_SIMPLE(0, NVTX_PAYLOAD_ENTRY_TYPE_UINT8, "overhead_kind"),
};
static const nvtxPayloadSchemaAttr_t cuptiOverheadSchemaAttr = {
  NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NAME | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE |
    NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
    NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_SCHEMA_ID,
  "CUPTI Overhead",
  NVTX_PAYLOAD_SCHEMA_TYPE_STATIC,
  NVTX_PAYLOAD_SCHEMA_FLAG_NONE,
  cuptiOverheadSchema,
  std::extent<decltype(cuptiOverheadSchema)>::value,
  sizeof(struct NvidiaNvtxw::cuptiOverheadEvent),
  0,
  NvidiaNvtxw::PayloadSchemaId::cuptiOverheadId,
  nullptr};
#undef PAYLOAD_ENTRY_SIMPLE

const nvtxPayloadSchemaAttr_t* GetNameSchemaAttr() { return &nameSchemaAttr; }
const nvtxPayloadSchemaAttr_t* GetNvtxRangePushPopSchemaAttr()
{
  return &nvtxRangePushPopSchemaAttr;
}
const nvtxPayloadSchemaAttr_t* GetNvtxRangeStartEndSchemaAttr()
{
  return &nvtxRangeStartEndSchemaAttr;
}
const nvtxPayloadSchemaAttr_t* GetCuptiApiSchemaAttr() { return &cuptiApiSchemaAttr; }
const nvtxPayloadSchemaAttr_t* GetCuptiDeviceSchemaAttr() { return &cuptiDeviceSchemaAttr; }
const nvtxPayloadSchemaAttr_t* GetCuptiKernelSchemaAttr() { return &cuptiKernelSchemaAttr; }
const nvtxPayloadSchemaAttr_t* GetCuptiMemcpySchemaAttr() { return &cuptiMemcpySchemaAttr; }
const nvtxPayloadSchemaAttr_t* GetCuptiMemsetSchemaAttr() { return &cuptiMemsetSchemaAttr; }
const nvtxPayloadSchemaAttr_t* GetCuptiOverheadSchemaAttr() { return &cuptiOverheadSchemaAttr; }
}  // namespace NvidiaNvtxw