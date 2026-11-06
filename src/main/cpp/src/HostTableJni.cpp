/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "jni_utils.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/aligned.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using spark_rapids_jni::host_column_view;
using spark_rapids_jni::host_table_view;

// Padding sizes to 64-byte for compatibility with Arrow
std::size_t pad_size(std::size_t size) { return rmm::align_up(size, 64); }

// Reads the values at positions `first` and `last` of an INT32 or INT64 offsets column, counted
// from the start of its data. Synchronizes `stream`.
std::pair<int64_t, int64_t> read_offsets(cudf::column_view const& offsets,
                                         cudf::size_type first,
                                         cudf::size_type last,
                                         cudaStream_t stream)
{
  auto read = [&]<typename T>() {
    T values[2];
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(&values[0], offsets.head<T>() + first, sizeof(T), cudaMemcpyDefault, stream));
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(&values[1], offsets.head<T>() + last, sizeof(T), cudaMemcpyDefault, stream));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream));
    return std::pair<int64_t, int64_t>{values[0], values[1]};
  };
  if (offsets.type().id() == cudf::type_id::INT64) { return read.template operator()<int64_t>(); }
  CUDF_EXPECTS(offsets.type().id() == cudf::type_id::INT32, "Unexpected offsets type");
  return read.template operator()<int32_t>();
}

// Places columns in a host buffer: for each column its data (padded to 64 bytes), then its null
// mask if it has nulls, then its children, depth first. Without a buffer it only measures the
// size; with one it enqueues the copies on `stream`. Both make the same decisions, so the measured
// size is the size the copy needs.
//
// Host columns always have offset 0, and only the rows a column covers are copied, straight from
// the device buffers they are in. This matters for slices, whose offset or row count leaves part
// of the buffers or children outside the column:
//  - fixed-width data and null masks are copied from the first row; a null mask whose first row is
//    not at a word boundary is first shifted into a temporary device buffer;
//  - STRING and LIST offsets are rebased to start at 0, in a temporary device column when they do
//    not already, and only the chars or child rows they cover are copied;
//  - STRUCT children are taken over the parent's rows.
// Temporary device memory is freed on `stream` when the layout is destroyed, after the copies that
// read it have been enqueued.
class host_layout {
 public:
  // Measures the size only.
  explicit host_layout(cudaStream_t stream) : _measuring{true}, _stream{stream} {}

  // Copies into `buffer`, which holds `capacity` bytes. `buffer` may be null when `capacity` is 0.
  host_layout(uint8_t* buffer, std::size_t capacity, cudaStream_t stream)
    : _measuring{false}, _buffer{buffer}, _capacity{capacity}, _stream{stream}
  {
  }

  [[nodiscard]] std::size_t size() const { return _size; }

  // Adds a column. Returns its host view, or an empty view when measuring.
  host_column_view add(cudf::column_view const& col) { return add(col, col.offset(), col.size()); }

 private:
  [[nodiscard]] bool measuring() const { return _measuring; }

  // Reserves `bytes` of the buffer. Returns where they start, or nullptr when measuring.
  uint8_t* reserve(std::size_t bytes)
  {
    auto const start = _size;
    _size += bytes;
    if (measuring()) { return nullptr; }
    if (_size > _capacity) { throw std::runtime_error("buffer overflow"); }
    return _buffer + start;
  }

  // Copies `bytes` from device memory at `src`, padded with zeros to 64 bytes.
  void const* add_data(void const* src, std::size_t bytes)
  {
    auto const padded = pad_size(bytes);
    auto const dst    = reserve(padded);
    if (measuring()) { return nullptr; }
    if (bytes > 0) { CUDF_CUDA_TRY(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, _stream)); }
    std::fill(dst + bytes, dst + padded, 0);
    return dst;
  }

  // Copies bits [begin, begin + rows) of `mask` into a mask that starts at bit 0.
  cudf::bitmask_type const* add_mask(cudf::bitmask_type const* mask,
                                     cudf::size_type begin,
                                     cudf::size_type rows)
  {
    auto const bytes = cudf::bitmask_allocation_size_bytes(rows);
    auto const dst   = reserve(bytes);
    if (measuring()) { return nullptr; }
    auto src = mask + cudf::word_index(begin);
    if (cudf::intra_word_index(begin) != 0) {
      _masks.push_back(cudf::copy_bitmask(mask, begin, begin + rows, _stream));
      src = _masks.back().data();
    }
    // Copy only the words the rows need: the source may end right after them.
    auto const used = cudf::num_bitmask_words(rows) * sizeof(cudf::bitmask_type);
    CUDF_CUDA_TRY(cudaMemcpyAsync(dst, src, used, cudaMemcpyDefault, _stream));
    std::fill(dst + used, dst + bytes, 0);
    return std::bit_cast<cudf::bitmask_type const*>(dst);
  }

  // Copies the `rows + 1` offsets at positions [first, first + rows] of `offsets`, counted from
  // the start of its data, rebased so that the first is 0. `first_value` is the value at `first`.
  host_column_view add_offsets(cudf::column_view const& offsets,
                               cudf::size_type first,
                               cudf::size_type rows,
                               int64_t first_value)
  {
    auto const type  = offsets.type();
    auto const count = rows + 1;
    auto const bytes = cudf::size_of(type) * count;
    if (measuring()) {
      add_data(nullptr, bytes);
      return {};
    }
    void const* src = offsets.head<uint8_t>() + cudf::size_of(type) * first;
    if (first_value != 0) {
      auto const slice = cudf::column_view(type, count, offsets.head(), nullptr, 0, first);
      std::unique_ptr<cudf::scalar> base;
      if (type.id() == cudf::type_id::INT64) {
        base = std::make_unique<cudf::numeric_scalar<int64_t>>(first_value, true, _stream);
      } else {
        base = std::make_unique<cudf::numeric_scalar<int32_t>>(
          static_cast<int32_t>(first_value), true, _stream);
      }
      _columns.push_back(
        cudf::binary_operation(slice, *base, cudf::binary_operator::SUB, type, _stream));
      src = _columns.back()->view().head();
    }
    return host_column_view(type, count, add_data(src, bytes), nullptr, 0);
  }

  // Adds rows [begin, begin + rows) of `col`, where `begin` counts from the start of the column's
  // buffers, so it includes `col.offset()`.
  host_column_view add(cudf::column_view const& col, cudf::size_type begin, cudf::size_type rows)
  {
    auto const id       = col.type().id();
    bool const own_rows = begin == col.offset() && rows == col.size();
    cudf::size_type const nulls =
      !col.nullable() ? 0
      : own_rows      ? col.null_count()
                      : cudf::null_count(col.null_mask(), begin, begin + rows, _stream);

    // STRING and LIST rows cover a range of offsets values. A LIST whose offsets child holds
    // exactly its rows' offsets keeps them as they are, with its child column whole; otherwise the
    // range is read so that only the chars or child rows it covers are copied.
    bool const has_offsets = (id == cudf::type_id::STRING || id == cudf::type_id::LIST) &&
                             col.num_children() > 0 && col.child(0).size() > 0;
    bool const list_as_is = id == cudf::type_id::LIST && has_offsets && begin == 0 &&
                            col.child(0).offset() == 0 && col.child(0).size() == rows + 1;
    bool const read_range = has_offsets && !list_as_is;
    int64_t first_value   = 0;
    int64_t last_value    = 0;
    if (read_range) {
      auto const& offsets = col.child(0);
      std::tie(first_value, last_value) =
        read_offsets(offsets, offsets.offset() + begin, offsets.offset() + begin + rows, _stream);
    }

    void const* data = nullptr;
    if (col.head() != nullptr) {
      if (cudf::is_fixed_width(col.type())) {
        auto const width = cudf::size_of(col.type());
        data             = add_data(col.head<uint8_t>() + width * begin, width * rows);
      } else if (id == cudf::type_id::STRING) {
        // The offsets values index the chars from the start of the parent's data.
        data = add_data(col.head<uint8_t>() + first_value, last_value - first_value);
      } else {
        throw std::runtime_error(std::string("unexpected data type: ") +
                                 std::to_string(static_cast<int>(id)));
      }
    }

    auto const mask = nulls > 0 ? add_mask(col.null_mask(), begin, rows) : nullptr;

    std::vector<host_column_view> children;
    children.reserve(col.num_children());
    if (read_range) {
      auto const& offsets = col.child(0);
      children.push_back(add_offsets(offsets, offsets.offset() + begin, rows, first_value));
      if (id == cudf::type_id::LIST) {
        auto const& child = col.child(cudf::lists_column_view::child_column_index);
        children.push_back(add(child,
                               child.offset() + static_cast<cudf::size_type>(first_value),
                               static_cast<cudf::size_type>(last_value - first_value)));
      }
    } else if (id == cudf::type_id::STRUCT) {
      // A STRUCT's offset and row count apply to its children.
      std::for_each(col.child_begin(), col.child_end(), [&](cudf::column_view const& child) {
        children.push_back(add(child, child.offset() + begin, rows));
      });
    } else {
      std::for_each(col.child_begin(), col.child_end(), [&](cudf::column_view const& child) {
        children.push_back(add(child));
      });
    }

    if (measuring()) { return {}; }
    return host_column_view(col.type(), rows, data, mask, nulls, children);
  }

  bool _measuring;
  uint8_t* _buffer{};
  std::size_t _capacity{};
  std::size_t _size{};
  cudaStream_t _stream;
  std::vector<std::unique_ptr<cudf::column>> _columns;
  std::vector<cuda::device_buffer<cudf::bitmask_type>> _masks;
};

// Determine the size of buffer needed to hold all of the data for a table.
std::size_t host_buffer_size(cudf::table_view const& t, cudaStream_t stream)
{
  host_layout layout{stream};
  std::for_each(t.begin(), t.end(), [&](cudf::column_view const& c) { layout.add(c); });
  return layout.size();
}

// Copies a table into `buffer` on `stream`. If this throws, no copy into `buffer` is still pending,
// so the caller may free it.
std::unique_ptr<host_table_view> to_host_table_async(cudf::table_view const& dev_table,
                                                     uint8_t* buffer,
                                                     std::size_t buffer_size,
                                                     cudaStream_t stream)
{
  host_layout layout{buffer, buffer_size, stream};
  std::vector<host_column_view> cols;
  cols.reserve(dev_table.num_columns());
  try {
    std::for_each(dev_table.begin(), dev_table.end(), [&](cudf::column_view const& dev_col) {
      cols.push_back(layout.add(dev_col));
    });
  } catch (...) {
    // Copies into `buffer` may already be queued: temporaries for later columns are allocated
    // between copies, and the allocation can fail. Wait for the copies before rethrowing. An error
    // from the wait itself is dropped in favor of the original one.
    static_cast<void>(cudaStreamSynchronize(stream));
    throw;
  }
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
    auto table       = std::bit_cast<cudf::table_view const*>(table_handle);
    auto buffer      = std::bit_cast<uint8_t*>(host_address);
    auto buffer_size = static_cast<std::size_t>(host_size);
    auto stream      = std::bit_cast<cudaStream_t>(jstream);
    // Temporary device memory for sliced columns is freed on `stream` after the copy is enqueued.
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
  JNI_TRY { cudf::jni::safe_delete<cudf::column_view>(dev_column_view_handle); }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_HostTable_freeHostTable(JNIEnv* env,
                                                                                jclass,
                                                                                jlong table_handle)
{
  JNI_NULL_CHECK(env, table_handle, "table is null", );
  JNI_TRY { cudf::jni::safe_delete<host_table_view>(table_handle); }
  JNI_CATCH(env, );
}

}  // extern "C"
