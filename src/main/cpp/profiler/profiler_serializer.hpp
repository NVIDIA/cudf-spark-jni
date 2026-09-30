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

#include "profiler_generated.h"

#include <cupti.h>
#include <flatbuffers/flatbuffers.h>
#include <jni.h>

#include <cstdint>
#include <vector>

namespace spark_rapids_jni::profiler {

// Serializes profile data as flatbuffers
struct profiler_serializer {
  profiler_serializer(
    JNIEnv* env, jobject writer, size_t buffer_size, size_t flush_threshold, bool capture_allocs);
  void process_cupti_buffer(uint8_t* buffer, size_t valid_size);
  void flush();

 private:
  void write_profile_header();
  void process_api_activity(CUpti_ActivityAPI const*);
  void process_device_activity(CUpti_ActivityDevice4 const*);
  void process_dropped_records(size_t num_dropped);
  void process_marker_activity(CUpti_ActivityMarker2 const*);
  void process_marker_data(CUpti_ActivityMarkerData const*);
  void process_memcpy(CUpti_ActivityMemcpy5 const*);
  void process_memset(CUpti_ActivityMemset4 const*);
  void process_kernel(CUpti_ActivityKernel8 const*);
  void process_overhead(CUpti_ActivityOverhead const*);
  void report_num_dropped_records();
  void write_current_fb();

  JNIEnv* env_;
  jmethodID j_write_method_;
  jobject j_writer_;
  size_t flush_threshold_;
  bool capture_allocs_;
  flatbuffers::FlatBufferBuilder fbb_;
  std::vector<flatbuffers::Offset<ApiActivity>> api_offsets_;
  std::vector<flatbuffers::Offset<DeviceActivity>> device_offsets_;
  std::vector<flatbuffers::Offset<DroppedRecords>> dropped_offsets_;
  std::vector<flatbuffers::Offset<KernelActivity>> kernel_offsets_;
  std::vector<flatbuffers::Offset<MarkerActivity>> marker_offsets_;
  std::vector<flatbuffers::Offset<MarkerData>> marker_data_offsets_;
  std::vector<flatbuffers::Offset<MemcpyActivity>> memcpy_offsets_;
  std::vector<flatbuffers::Offset<MemsetActivity>> memset_offsets_;
  std::vector<flatbuffers::Offset<OverheadActivity>> overhead_offsets_;
};

}  // namespace spark_rapids_jni::profiler
