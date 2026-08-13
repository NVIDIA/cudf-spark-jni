/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

#include "cudf_jni_apis.hpp"
#include "host_table_view.hpp"

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/aligned.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using spark_rapids_jni::host_column_view;
using spark_rapids_jni::host_table_view;

// Padding sizes to 64-byte for compatibility with Arrow
std::size_t pad_size(std::size_t size) { return rmm::align_up(size, 64); }

// Determine the size of buffer needed to hold just the data portion of this column.
// This does not include validity, offsets, or any child columns.
std::size_t get_data_size(cudf::column_view const& c, cudaStream_t stream)
{
  auto dtype = c.type();
  if (cudf::is_fixed_width(dtype)) {
    return cudf::size_of(dtype) * c.size();
  } else if (dtype.id() == cudf::type_id::STRING) {
    auto scv = cudf::strings_column_view(c);
    return scv.chars_size(stream);
  } else {
    throw std::runtime_error(std::string("unexpected data type: ") +
                             std::to_string(static_cast<int>(dtype.id())));
  }
}

// Determine the size of buffer needed to hold all of the data for a column.
// This includes validity, data, offsets, and child columns.
std::size_t column_size(cudf::column_view const& c, cudaStream_t stream)
{
  std::size_t size = 0;
  if (c.data<uint8_t>() != nullptr) { size += pad_size(get_data_size(c, stream)); }
  if (c.has_nulls()) { size += cudf::bitmask_allocation_size_bytes(c.size()); }
  return std::accumulate(c.child_begin(),
                         c.child_end(),
                         size,
                         [stream](std::size_t sum, cudf::column_view const& child) {
                           return sum + column_size(child, stream);
                         });
}

// Determine the size of buffer needed to hold all of the data for a table.
std::size_t host_buffer_size(cudf::table_view const& t, cudaStream_t stream)
{
  std::size_t s = 0;
  return std::accumulate(
    t.begin(), t.end(), s, [stream](std::size_t sum, cudf::column_view const& c) {
      return sum + column_size(c, stream);
    });
}

uint8_t* copy_to_host_async(
  void const* src, uint8_t* dest, std::size_t size, uint8_t const* dest_end, cudaStream_t stream)
{
  if (dest + size > dest_end) { throw std::runtime_error("buffer overflow"); }
  CUDF_CUDA_TRY(cudaMemcpyAsync(dest, src, size, cudaMemcpyDefault, stream));
  return dest + size;
}

uint8_t* build_host_column_view_async(cudf::column_view const& dev_col,
                                      uint8_t* bp,
                                      uint8_t const* bp_end,
                                      cudaStream_t stream,
                                      std::vector<host_column_view>& host_cols)
{
  void const* host_data = nullptr;
  void const* dev_data  = dev_col.data<uint8_t>();
  if (dev_data != nullptr) {
    host_data          = bp;
    auto data_size     = get_data_size(dev_col, stream);
    auto padded_bp_end = bp + pad_size(data_size);
    bp                 = copy_to_host_async(dev_data, bp, data_size, bp_end, stream);
    while (bp != padded_bp_end) {
      *bp++ = 0;
    }
  }
  cudf::bitmask_type const* host_null_mask = nullptr;
  if (dev_col.has_nulls()) {
    host_null_mask = std::bit_cast<cudf::bitmask_type const*>(bp);
    auto mask_size = cudf::bitmask_allocation_size_bytes(dev_col.size());
    bp             = copy_to_host_async(dev_col.null_mask(), bp, mask_size, bp_end, stream);
  }
  std::vector<host_column_view> children;
  children.reserve(dev_col.num_children());
  std::for_each(dev_col.child_begin(), dev_col.child_end(), [&](cudf::column_view const& child) {
    bp = build_host_column_view_async(child, bp, bp_end, stream, children);
  });
  host_cols.push_back(host_column_view(
    dev_col.type(), dev_col.size(), host_data, host_null_mask, dev_col.null_count(), children));
  return bp;
}

std::unique_ptr<host_table_view> to_host_table_async(cudf::table_view const& dev_table,
                                                     uint8_t* buffer,
                                                     std::size_t buffer_size,
                                                     cudaStream_t stream)
{
  uint8_t* bp               = buffer;
  uint8_t const* buffer_end = buffer + buffer_size;
  std::vector<host_column_view> cols;
  cols.reserve(dev_table.num_columns());
  std::for_each(dev_table.begin(), dev_table.end(), [&](cudf::column_view const& dev_col) {
    bp = build_host_column_view_async(dev_col, bp, buffer_end, stream, cols);
  });
  return std::make_unique<host_table_view>(cols);
}

cudf::column_view to_device_column(host_column_view const& host_col, jlong host_to_dev_offset)
{
  auto data = host_col.data<uint8_t>();
  if (data != nullptr) { data += host_to_dev_offset; }
  auto mask = host_col.null_mask();
  if (mask != nullptr) { mask += host_to_dev_offset / sizeof(*mask); }
  std::vector<cudf::column_view> children;
  std::transform(host_col.child_begin(),
                 host_col.child_end(),
                 std::back_inserter(children),
                 [host_to_dev_offset](host_column_view const& c) {
                   return to_device_column(c, host_to_dev_offset);
                 });
  return cudf::column_view(
    host_col.type(), host_col.size(), data, mask, host_col.null_count(), 0, children);
}

std::vector<std::unique_ptr<cudf::column_view>> to_device_column_views(
  host_table_view const& host_table, jlong host_to_dev_offset)
{
  std::vector<std::unique_ptr<cudf::column_view>> cv_ptrs;
  cv_ptrs.reserve(host_table.num_columns());
  std::transform(
    host_table.begin(),
    host_table.end(),
    std::back_inserter(cv_ptrs),
    [host_to_dev_offset](host_column_view const& host_col) {
      return std::make_unique<cudf::column_view>(to_device_column(host_col, host_to_dev_offset));
    });
  return cv_ptrs;
}

}  // anonymous namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_HostTable_bufferSize(JNIEnv* env,
                                                                              jclass,
                                                                              jlong table_handle,
                                                                              jlong jstream)
{
  JNI_NULL_CHECK(env, table_handle, "table is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto t      = std::bit_cast<cudf::table_view const*>(table_handle);
    auto stream = std::bit_cast<cudaStream_t>(jstream);
    return static_cast<jlong>(host_buffer_size(*t, stream));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_HostTable_copyFromTableAsync(
  JNIEnv* env, jclass, jlong table_handle, jlong host_address, jlong host_size, jlong jstream)
{
  JNI_NULL_CHECK(env, table_handle, "table is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto table           = std::bit_cast<cudf::table_view const*>(table_handle);
    auto buffer          = std::bit_cast<uint8_t*>(host_address);
    auto buffer_size     = static_cast<std::size_t>(host_size);
    auto stream          = std::bit_cast<cudaStream_t>(jstream);
    auto host_table_view = to_host_table_async(*table, buffer, buffer_size, stream);
    return std::bit_cast<jlong>(host_table_view.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_HostTable_toDeviceColumnViews(
  JNIEnv* env, jclass, jlong table_handle, jlong host_to_dev_offset)
{
  JNI_NULL_CHECK(env, table_handle, "table is null", nullptr);
  JNI_ARG_CHECK(
    env, host_to_dev_offset % sizeof(cudf::bitmask_type) == 0, "invalid offset", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto host_table       = std::bit_cast<spark_rapids_jni::host_table_view const*>(table_handle);
    auto column_view_ptrs = to_device_column_views(*host_table, host_to_dev_offset);
    cudf::jni::native_jlongArray handles(env, static_cast<int>(column_view_ptrs.size()));
    std::transform(
      column_view_ptrs.begin(),
      column_view_ptrs.end(),
      handles.begin(),
      [](std::unique_ptr<cudf::column_view>& p) { return cudf::jni::release_as_jlong(p); });
    return handles.get_jArray();
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_HostTable_freeDeviceColumnView(
  JNIEnv* env, jclass, jlong dev_column_view_handle)
{
  JNI_NULL_CHECK(env, dev_column_view_handle, "view is null", );
  JNI_TRY { delete std::bit_cast<cudf::column_view*>(dev_column_view_handle); }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_HostTable_freeHostTable(JNIEnv* env,
                                                                                jclass,
                                                                                jlong table_handle)
{
  JNI_NULL_CHECK(env, table_handle, "table is null", );
  JNI_TRY { delete std::bit_cast<host_table_view*>(table_handle); }
  JNI_CATCH(env, );
}

}  // extern "C"
