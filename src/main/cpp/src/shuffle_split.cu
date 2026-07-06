/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

#include "nvtx_ranges.hpp"
#include "shuffle_split.hpp"
#include "shuffle_split_detail.hpp"
#include "utilities/iterator.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda/functional>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <cstddef>
#include <numeric>

namespace spark_rapids_jni {

using namespace cudf;
using namespace spark_rapids_jni::detail;

namespace {

/**
 * @brief Struct which contains information on a source buffer.
 *
 * The definition of "buffer" used throughout this module is a component piece of a
 * cudf column. So for example, a fixed-width column with validity would have 2 associated
 * buffers : the data itself and the validity buffer.  contiguous_split operates by breaking
 * each column up into it's individual components and copying each one as a separate kernel
 * block.
 */
struct src_buf_info {
  src_buf_info(cudf::type_id _type,
               int _offset_stack_pos,
               int _parent_offsets_index,
               uint8_t const* _data,
               buffer_type _btype,
               size_type _column_offset,
               size_type _column_index)
    : type(_type),
      offset_stack_pos(_offset_stack_pos),
      parent_offsets_index(_parent_offsets_index),
      data(_data),
      btype(_btype),
      column_offset(_column_offset),
      column_index(_column_index)
  {
  }

  cudf::type_id type;
  int offset_stack_pos;      // position in the offset stack buffer
  int parent_offsets_index;  // immediate parent that has offsets, or -1 if none
  uint8_t const* data;
  buffer_type btype;
  size_type column_offset;  // offset in the case of a sliced column
  size_type column_index;   // (flattened) column index
};

/**
 * @brief Struct which contains information on a destination buffer.
 *
 * Similar to src_buf_info, dst_buf_info contains information on a destination buffer we
 * are going to copy to.  If we have N input buffers (which come from X columns), and
 * M partitions, then we have N*M destination buffers.
 */
struct dst_buf_info {
  size_t buf_size;  // total size of buffer in bytes
  buffer_type type;

  int src_buf_index;
  size_t src_offset;
  size_t dst_offset;
};

// The block of functions below are all related:
//
// compute_offset_stack_size()
// count_src_bufs()
// setup_source_buf_info()
// build_output_columns()
//
// Critically, they all traverse the hierarchy of source columns and their children
// in a specific order to guarantee they produce various outputs in a consistent
// way.
//
// So please be careful if you change the way in which these functions and
// functors traverse the hierarchy.

/**
 * @brief Returns whether or not the specified type is a column that contains offsets.
 */
bool is_offset_type(type_id id) { return (id == type_id::STRING or id == type_id::LIST); }

/**
 * @brief Whether or not nullability should be included in the data for this column.
 *
 * The edge case we're catching here is a column that is nullable, but has no rows. CPU
 * kud0 optimizes that case out. In the future if we also wanted to ignore columns has are
 * nullable but have no nulls, we would add that logic here.
 *
 * A second edge case to be aware of is that while a column may have a > 0 size, and therefore
 * is eligible to have nulls included, an individual partition may still be of size 0. In that case
 * the null vector is not included for that partition, even though this function will return true
 * for the column as a whole.
 */
bool include_nulls_for_column(column_view const& col) { return col.nullable() && col.size() > 0; }

/**
 * @brief Compute total device memory stack size needed to process nested
 * offsets per-output buffer.
 *
 * When determining the range of rows to be copied for each output buffer
 * we have to recursively apply the stack of offsets from our parent columns
 * (lists or strings).  We want to do this computation on the gpu because offsets
 * are stored in device memory.  However we don't want to do recursion on the gpu, so
 * each destination buffer gets a "stack" of space to work with equal in size to
 * it's offset nesting depth.  This function computes the total size of all of those
 * stacks.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param offset_depth Current offset nesting depth
 *
 * @returns Total offset stack size needed for this range of columns.
 */
template <typename InputIter>
size_t compute_offset_stack_size(InputIter begin, InputIter end, int offset_depth = 0)
{
  return std::accumulate(begin, end, 0, [offset_depth](auto stack_size, column_view const& col) {
    auto const num_buffers = 1 + (include_nulls_for_column(col) ? 1 : 0);
    return stack_size + (offset_depth * num_buffers) +
           compute_offset_stack_size(
             col.child_begin(), col.child_end(), offset_depth + is_offset_type(col.type().id()));
  });
}

/**
 * @brief A count of the three fundamental types of buffers. validity, offsets and data
 */
struct src_buf_count {
  size_t validity_buf_count;
  size_t offset_buf_count;
  size_t data_buf_count;
};

/**
 * @brief Count the total number of source buffers, broken down by type (validity, offset, data)
 * we will be copying from.
 *
 * This count includes buffers for all input columns. For example a
 * fixed-width column with validity would be 2 buffers (validity, data).
 * A string column with validity would be 3 buffers (validity, offssets, data (chars))
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 *
 * @returns total number of source buffer per type for this range of columns
 */
template <typename InputIter>
src_buf_count count_src_bufs(InputIter begin, InputIter end)
{
  auto src_buf_count_add = [](src_buf_count const& lhs, src_buf_count const& rhs) -> src_buf_count {
    return src_buf_count{lhs.validity_buf_count + rhs.validity_buf_count,
                         lhs.offset_buf_count + rhs.offset_buf_count,
                         lhs.data_buf_count + rhs.data_buf_count};
  };

  auto buf_iter = thrust::make_transform_iterator(begin, [&](column_view const& col) {
    auto const type = col.type().id();
    // for lists and strings, account for their offset child here instead of recursively
    auto const has_offsets_child =
      type == cudf::type_id::LIST || (type == cudf::type_id::STRING && col.num_children() > 0);
    src_buf_count const counts{
      static_cast<size_t>(include_nulls_for_column(col)),
      static_cast<size_t>(has_offsets_child),
      size_t{
        1}};  // this is 1 for all types because even lists and structs have stubs for data buffers

    auto child_counts = [&]() {
      // strings don't need to recurse. we count their offsets and data right here.
      switch (type) {
        case cudf::type_id::STRING: return src_buf_count{0, 0, 0};

        // offset for lists child is accounted for here
        case cudf::type_id::LIST: {
          auto data_child = col.child_begin() + cudf::lists_column_view::child_column_index;
          return count_src_bufs(data_child, std::next(data_child));
        }

        default: break;
      }
      return count_src_bufs(col.child_begin(), col.child_end());
    };
    return src_buf_count_add(counts, child_counts());
  });
  return std::accumulate(
    buf_iter, buf_iter + std::distance(begin, end), src_buf_count{0, 0, 0}, src_buf_count_add);
}

/**
 * @brief Computes source buffer information for the copy process.
 *
 * For each input column to be split we need to know several pieces of information
 * in the copy kernel.  This function traverses the input columns and prepares this
 * information for the gpu.
 *
 * This function is called recursively in the case of nested types.
 *
 * @param begin Beginning of input columns
 * @param end End of input columns
 * @param head Beginning of source buffer info array
 * @param validity_cur[in, out] Current validity source buffer info to be read from
 * @param offset_cur[in, out] Current offset source buffer info to be read from
 * @param data_cur[in, out] Current data source buffer info to be read from
 * @param offset_stack_pos[in, out] Integer representing our current offset nesting depth
 * @param col_index[in, out] Integer representing out current (flattened) column index
 * (how many list or string levels deep we are)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param parent_offset_index Index into src_buf_info output array indicating our nearest
 * containing list parent. -1 if we have no list parent
 * @param offset_depth Current offset nesting depth (how many list levels deep we are)
 *
 * @returns next src_buf_output after processing this range of input columns
 */
// setup source buf info
template <typename InputIter>
void setup_source_buf_info(InputIter begin,
                           InputIter end,
                           src_buf_info* head,
                           src_buf_info*& validity_cur,
                           src_buf_info*& offset_cur,
                           src_buf_info*& data_cur,
                           int& offset_stack_pos,
                           int& col_index,
                           rmm::cuda_stream_view stream,
                           int parent_offset_index = -1,
                           int offset_depth        = 0);

/**
 * @brief Functor that builds source buffer information based on input columns.
 *
 * Called by setup_source_buf_info to build information for a single source column.  This function
 * will recursively call setup_source_buf_info in the case of nested types.
 */
struct buf_info_functor {
  src_buf_info* head;

  template <typename T>
  void operator()(column_view const& col,
                  int& col_index,
                  src_buf_info*& validity_cur,
                  src_buf_info*& offset_cur,
                  src_buf_info*& data_cur,
                  int& offset_stack_pos,
                  int parent_offset_index,
                  int offset_depth,
                  rmm::cuda_stream_view)
  {
    if (include_nulls_for_column(col)) {
      add_null_buffer(
        col, validity_cur, offset_stack_pos, parent_offset_index, offset_depth, col_index);
    }

    // info for the data buffer
    *data_cur = src_buf_info(col.type().id(),
                             offset_stack_pos,
                             parent_offset_index,
                             col.head<uint8_t>(),
                             buffer_type::DATA,
                             col.offset(),
                             col_index);
    data_cur++;

    offset_stack_pos += offset_depth;
    col_index++;
  }

  template <typename T, typename... Args>
  std::enable_if_t<std::is_same_v<T, cudf::dictionary32>, void> operator()(Args&&...)
  { CUDF_FAIL("Unsupported type"); }

 private:
  void add_null_buffer(column_view const& col,
                       src_buf_info*& validity_cur,
                       int& offset_stack_pos,
                       int parent_offset_index,
                       int offset_depth,
                       int col_index)
  {
    // info for the validity buffer
    *validity_cur = src_buf_info(type_id::INT32,
                                 offset_stack_pos,
                                 parent_offset_index,
                                 reinterpret_cast<uint8_t const*>(col.null_mask()),
                                 buffer_type::VALIDITY,
                                 col.offset(),
                                 col_index);
    validity_cur++;

    offset_stack_pos += offset_depth;
  }
};

template <>
void buf_info_functor::operator()<cudf::string_view>(column_view const& col,
                                                     int& col_index,
                                                     src_buf_info*& validity_cur,
                                                     src_buf_info*& offset_cur,
                                                     src_buf_info*& data_cur,
                                                     int& offset_stack_pos,
                                                     int parent_offset_index,
                                                     int offset_depth,
                                                     rmm::cuda_stream_view)
{
  if (include_nulls_for_column(col)) {
    add_null_buffer(
      col, validity_cur, offset_stack_pos, parent_offset_index, offset_depth, col_index);
  }

  // the way strings are arranged, the strings column itself contains char data, but our child
  // offsets column contains the offsets. So our parent_offset_index is actually our child.

  // string columns don't necessarily have children if they are empty
  auto const has_offsets_child = col.num_children() > 0;

  // string columns contain the underlying chars data.
  *data_cur = src_buf_info(type_id::STRING,
                           offset_stack_pos,
                           // if I have offsets, it's index will be the current offset buffer,
                           // otherwise it's whatever my parent's was.
                           has_offsets_child ? (offset_cur - head) : parent_offset_index,
                           col.head<uint8_t>(),
                           buffer_type::DATA,
                           col.offset(),
                           col_index);
  data_cur++;
  // if I have offsets, I need to include that in the stack size
  offset_stack_pos += has_offsets_child ? offset_depth + 1 : offset_depth;

  if (has_offsets_child) {
    CUDF_EXPECTS(col.num_children() == 1, "Encountered malformed string column");
    strings_column_view scv(col);

    // info for the offsets buffer
    CUDF_EXPECTS(not scv.offsets().nullable(), "Encountered nullable string offsets column");
    *offset_cur = src_buf_info(
      type_id::INT32,
      offset_stack_pos,
      parent_offset_index,
      // note: offsets can be null in the case where the string column
      // has been created with empty_like().
      reinterpret_cast<uint8_t const*>(scv.offsets().begin<cudf::id_to_type<type_id::INT32>>()),
      buffer_type::OFFSETS,
      col.offset(),
      col_index);

    offset_cur++;
    offset_stack_pos += offset_depth;
  }

  col_index++;
}

template <>
void buf_info_functor::operator()<cudf::list_view>(column_view const& col,
                                                   int& col_index,
                                                   src_buf_info*& validity_cur,
                                                   src_buf_info*& offset_cur,
                                                   src_buf_info*& data_cur,
                                                   int& offset_stack_pos,
                                                   int parent_offset_index,
                                                   int offset_depth,
                                                   rmm::cuda_stream_view stream)
{
  lists_column_view lcv(col);

  if (include_nulls_for_column(col)) {
    add_null_buffer(
      col, validity_cur, offset_stack_pos, parent_offset_index, offset_depth, col_index);
  }

  // list columns hold no actual data, but we need to keep a record
  // of it so we know it's size when we are constructing the output columns
  *data_cur = src_buf_info(type_id::LIST,
                           offset_stack_pos,
                           parent_offset_index,
                           nullptr,
                           buffer_type::DATA,
                           col.offset(),
                           col_index);
  data_cur++;
  offset_stack_pos += offset_depth;

  CUDF_EXPECTS(col.num_children() == 2, "Encountered malformed list column");

  // info for the offsets buffer
  *offset_cur = src_buf_info(
    type_id::INT32,
    offset_stack_pos,
    parent_offset_index,
    // note: offsets can be null in the case where the lists column
    // has been created with empty_like().
    reinterpret_cast<uint8_t const*>(lcv.offsets().begin<cudf::id_to_type<type_id::INT32>>()),
    buffer_type::OFFSETS,
    col.offset(),
    col_index);

  // since we are crossing an offset boundary, calculate our new depth and parent offset index.
  parent_offset_index = offset_cur - head;
  offset_cur++;
  offset_stack_pos += offset_depth;
  offset_depth++;
  col_index++;

  auto child_col = col.child_begin() + lists_column_view::child_column_index;
  setup_source_buf_info(child_col,
                        std::next(child_col),
                        head,
                        validity_cur,
                        offset_cur,
                        data_cur,
                        offset_stack_pos,
                        col_index,
                        stream,
                        parent_offset_index,
                        offset_depth);
}

template <>
void buf_info_functor::operator()<cudf::struct_view>(column_view const& col,
                                                     int& col_index,
                                                     src_buf_info*& validity_cur,
                                                     src_buf_info*& offset_cur,
                                                     src_buf_info*& data_cur,
                                                     int& offset_stack_pos,
                                                     int parent_offset_index,
                                                     int offset_depth,
                                                     rmm::cuda_stream_view stream)
{
  if (include_nulls_for_column(col)) {
    add_null_buffer(
      col, validity_cur, offset_stack_pos, parent_offset_index, offset_depth, col_index);
  }

  // struct columns hold no actual data, but we need to keep a record
  // of it so we know it's size when we are constructing the output columns
  *data_cur = src_buf_info(type_id::STRUCT,
                           offset_stack_pos,
                           parent_offset_index,
                           nullptr,
                           buffer_type::DATA,
                           col.offset(),
                           col_index);
  data_cur++;
  offset_stack_pos += offset_depth;
  col_index++;

  // recurse on children
  cudf::structs_column_view scv(col);
  std::vector<column_view> sliced_children;
  sliced_children.reserve(scv.num_children());
  std::transform(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(scv.num_children()),
    std::back_inserter(sliced_children),
    [&scv, &stream](size_type child_index) { return scv.get_sliced_child(child_index, stream); });
  setup_source_buf_info(sliced_children.begin(),
                        sliced_children.end(),
                        head,
                        validity_cur,
                        offset_cur,
                        data_cur,
                        offset_stack_pos,
                        col_index,
                        stream,
                        parent_offset_index,
                        offset_depth);
}

template <typename InputIter>
void setup_source_buf_info(InputIter begin,
                           InputIter end,
                           src_buf_info* head,
                           src_buf_info*& validity_cur,
                           src_buf_info*& offset_cur,
                           src_buf_info*& data_cur,
                           int& offset_stack_pos,
                           int& col_index,
                           rmm::cuda_stream_view stream,
                           int parent_offset_index,
                           int offset_depth)
{
  std::for_each(begin, end, [&](column_view const& col) {
    cudf::type_dispatcher(col.type(),
                          buf_info_functor{head},
                          col,
                          col_index,
                          validity_cur,
                          offset_cur,
                          data_cur,
                          offset_stack_pos,
                          parent_offset_index,
                          offset_depth,
                          stream);
  });
}

/**
 * @brief Output iterator for writing values to the dst_offset field of the
 * dst_buf_info struct
 */
struct dst_offset_output_iterator {
  dst_buf_info* c;
  using value_type        = size_t;
  using difference_type   = size_t;
  using pointer           = size_t*;
  using reference         = size_t&;
  using iterator_category = thrust::output_device_iterator_tag;

  dst_offset_output_iterator operator+ __host__ __device__(int i) { return {c + i}; }

  void operator++ __host__ __device__() { c++; }

  reference operator[] __device__(int i) { return dereference(c + i); }
  reference operator* __device__() { return dereference(c); }

 private:
  reference __device__ dereference(dst_buf_info* c) { return c->dst_offset; }
};

/**
 * @brief Count the total number of flattened columns in a list of input columns.
 *
 * @param begin First column in the range
 * @param end Last column in the range
 * @param depth Current depth in the column hierarchy
 *
 * @return A pair containing the total number of columns and the maximum depth of the hierarchy.
 */
template <typename InputIter>
std::pair<size_t, size_t> count_flattened_columns(InputIter begin, InputIter end, int depth = 0)
{
  auto child_count = [&](column_view const& col, int depth) -> std::pair<size_t, size_t> {
    if (col.type().id() == cudf::type_id::STRUCT) {
      return count_flattened_columns(col.child_begin(), col.child_end(), depth + 1);
    } else if (col.type().id() == cudf::type_id::LIST) {
      cudf::lists_column_view lcv(col);
      std::vector<cudf::column_view> children({lcv.child()});
      return count_flattened_columns(children.begin(), children.end(), depth + 1);
    }
    return {size_t{0}, depth};
  };

  size_t col_count = 0;
  size_t max_depth = 0;
  std::for_each(begin, end, [&](column_view const& col) {
    auto const cc = child_count(col, depth);
    col_count += (1 + cc.first);
    max_depth = std::max(max_depth, cc.second);
  });

  return {col_count, max_depth};
}

/**
 * @brief Sizes of each of the data sections in a partition.
 *
 * Sizes include padding.
 */
struct partition_size_info {
  size_t validity_size;
  size_t offset_size;
  size_t data_size;
};

/**
 * @brief Kernel that packs the per-partition metadata for the output.
 *
 * @param out_buffer The output buffer for the entire split operation
 * @param out_buffer_offsets Per-partition offsets into the output buffer
 * @param num_partitions The number of partitions
 * @param columns_per_partition The number of flattened columns per partition
 * @param split_indices Per-partition row split indices
 * @param flattened_col_inst_has_validity Per-column bool on whether each column contains a validity
 * vector, per-partition
 * @param partition_size_info Per-partition size information for each buffer type
 *
 */
__global__ void pack_per_partition_metadata_kernel(uint8_t* out_buffer,
                                                   size_t const* out_buffer_offsets,
                                                   size_t num_partitions,
                                                   size_t columns_per_partition,
                                                   size_type const* split_indices,
                                                   bool const* flattened_col_inst_has_validity,
                                                   partition_size_info const* partition_sizes)
{
  constexpr uint32_t magic = 0x4b554430;

  int const tid = threadIdx.x + (blockIdx.x * blockDim.x);
  auto const threads_per_partition =
    cudf::util::round_up_safe(columns_per_partition, static_cast<size_t>(cudf::detail::warp_size));
  auto const partition_index = tid / threads_per_partition;
  if (partition_index >= num_partitions) { return; }
  auto const col_index = tid % threads_per_partition;

  // start of the metadata buffer for this partition
  uint8_t* buf_start        = out_buffer + out_buffer_offsets[partition_index];
  partition_header* pheader = reinterpret_cast<partition_header*>(buf_start);

  // first thread in each partition stores constant stuff
  if (col_index == 0) {
    pheader->magic_number = cudf::hashing::detail::swap_endian(magic);

    pheader->row_index =
      cudf::hashing::detail::swap_endian(static_cast<uint32_t>(split_indices[partition_index]));

    // it is possible to get in here with no columns -or- no rows.
    auto const partition_num_rows =
      col_index < columns_per_partition
        ? split_indices[partition_index + 1] - split_indices[partition_index]
        : 0;
    pheader->num_rows =
      cudf::hashing::detail::swap_endian(static_cast<uint32_t>(partition_num_rows));

    auto const& psize = partition_sizes[partition_index];
    pheader->validity_size =
      cudf::hashing::detail::swap_endian(static_cast<uint32_t>(psize.validity_size));
    pheader->offset_size =
      cudf::hashing::detail::swap_endian(static_cast<uint32_t>(psize.offset_size));
    pheader->total_size = cudf::hashing::detail::swap_endian(
      static_cast<uint32_t>(psize.validity_size + psize.offset_size + psize.data_size));
    pheader->num_flattened_columns =
      cudf::hashing::detail::swap_endian(static_cast<uint32_t>(columns_per_partition));
  }

  bitmask_type* has_validity =
    reinterpret_cast<bitmask_type*>(buf_start + sizeof(partition_header));

  // store has-validity bits. note that the kudo format only aligns to byte boundaries at the end of
  // the validity section, but we are doing this before anything further is written and we are
  // guaranteed that the overall buffer is padded out to >= 4 bytes.
  auto const col_instance_index = col_index + (partition_index * columns_per_partition);
  bitmask_type mask             = __ballot_sync(
    0xffffffff,
    col_index < columns_per_partition ? flattened_col_inst_has_validity[col_instance_index] : 0);
  if ((col_index % cudf::detail::warp_size == 0) && col_index < columns_per_partition) {
    has_validity[col_index / cudf::detail::warp_size] = mask;
  }
}

/**
 * @brief Copy data from source buffers into the output.
 *
 * Uses Cub batched memcpy
 *
 * @param src_bufs Source buffer pointers
 * @param dst_buf Pointer to the output buffer
 * @param num_copies The number of copy operations to perform
 * @param d_dst_buf_info num_copies sized array of buffer information
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 */
void split_copy(src_buf_info const* src_bufs,
                uint8_t* dst_buf,
                size_t num_bufs,
                dst_buf_info const* d_dst_buf_info,
                rmm::cuda_stream_view stream)
{
  auto input_iter = spark_rapids_jni::util::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<void*>([src_bufs, d_dst_buf_info] __device__(size_t i) {
      auto const& cinfo = d_dst_buf_info[i];
      return reinterpret_cast<void*>(
        const_cast<uint8_t*>(src_bufs[cinfo.src_buf_index].data + cinfo.src_offset));
    }));
  auto output_iter = spark_rapids_jni::util::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<void*>([dst_buf, d_dst_buf_info] __device__(size_t i) {
      auto const& cinfo = d_dst_buf_info[i];
      return reinterpret_cast<void*>(dst_buf + cinfo.dst_offset);
    }));
  auto size_iter = spark_rapids_jni::util::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<size_t>([d_dst_buf_info] __device__(size_t i) {
      auto const& cinfo = d_dst_buf_info[i];
      return cinfo.buf_size;
    }));

  size_t temp_storage_bytes = 0;  // Initialized on the next line, not that the compiler would know.
  cub::DeviceMemcpy::Batched(
    nullptr, temp_storage_bytes, input_iter, output_iter, size_iter, num_bufs, stream);
  rmm::device_buffer temp_storage(
    temp_storage_bytes, stream, cudf::get_current_device_resource_ref());
  cub::DeviceMemcpy::Batched(
    temp_storage.data(), temp_storage_bytes, input_iter, output_iter, size_iter, num_bufs, stream);
}

/**
 * @brief Fill in column metadata info for a list of input columns.
 *
 * @param meta Output metadata
 * @param begin Start of the range of columns
 * @param end End of the range of columns
 *
 */
template <typename InputIter>
void populate_column_data(shuffle_split_metadata& meta, InputIter begin, InputIter end)
{
  std::for_each(begin, end, [&meta](column_view const& col) {
    switch (col.type().id()) {
      case cudf::type_id::STRUCT:
        meta.col_info.push_back({col.type().id(), col.num_children()});
        populate_column_data(meta, col.child_begin(), col.child_end());
        break;

      case cudf::type_id::LIST: {
        meta.col_info.push_back({col.type().id(), 1});
        cudf::lists_column_view lcv(col);
        std::vector<cudf::column_view> children({lcv.child()});
        populate_column_data(meta, children.begin(), children.end());
      } break;

      case cudf::type_id::DECIMAL32:
      case cudf::type_id::DECIMAL64:
      case cudf::type_id::DECIMAL128:
        meta.col_info.push_back({col.type().id(), col.type().scale()});
        break;

      default: meta.col_info.push_back({col.type().id(), 0}); break;
    }
  });
}

/**
 * @brief Create the shuffle_split_metadata struct for the split operation
 *
 * @param input The input table
 * @param total_flattened_columns The total number of flattened columns in the input
 *
 * @return The final shuffle_split_metadata struct
 */
shuffle_split_metadata compute_metadata(cudf::table_view const& input,
                                        size_t total_flattened_columns)
{
  // compute the metadata
  shuffle_split_metadata ret;
  ret.col_info.reserve(total_flattened_columns);
  populate_column_data(ret, input.begin(), input.end());
  return ret;
}

};  // anonymous namespace

/**
 * @copydoc spark_rapids_jni::shuffle_split
 */
std::pair<shuffle_split_result, shuffle_split_metadata> shuffle_split(
  cudf::table_view const& input,
  std::vector<size_type> const& splits,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  // empty inputs
  CUDF_EXPECTS(input.num_columns() != 0, "Encountered input with no columns.");
  if (input.num_rows() == 0) {
    rmm::device_uvector<size_t> empty_offsets(1, stream, mr);
    thrust::fill(rmm::exec_policy(stream), empty_offsets.begin(), empty_offsets.end(), 0);
    return {shuffle_split_result{std::make_unique<rmm::device_buffer>(0, stream, mr),
                                 std::move(empty_offsets)},
            shuffle_split_metadata{compute_metadata(input, 0)}};
  }
  if (splits.size() > 0) {
    CUDF_EXPECTS(splits.back() <= input.column(0).size(),
                 "splits can't exceed size of input columns");
  }
  {
    size_type begin = 0;
    for (size_t i = 0; i < splits.size(); i++) {
      size_type end = splits[i];
      CUDF_EXPECTS(begin >= 0, "Starting index cannot be negative.");
      CUDF_EXPECTS(end >= begin, "End index cannot be smaller than the starting index.");
      CUDF_EXPECTS(end <= input.column(0).size(), "Slice range out of bounds.");
      begin = end;
    }
  }

  auto temp_mr = cudf::get_current_device_resource_ref();

  size_t const num_partitions   = splits.size() + 1;
  size_t const num_root_columns = input.num_columns();

  // compute # of source buffers (column data, validity, children), # of partitions
  // and total # of buffers
  src_buf_count const num_src_bufs_by_type = count_src_bufs(input.begin(), input.end());
  size_t const num_src_bufs                = num_src_bufs_by_type.validity_buf_count +
                                             num_src_bufs_by_type.offset_buf_count +
                                             num_src_bufs_by_type.data_buf_count;
  size_t const num_bufs                    = num_src_bufs * num_partitions;
  auto const bufs_per_partition            = num_src_bufs;

  // packed block of memory 1. split indices and src_buf_info structs
  size_t const indices_size =
    cudf::util::round_up_safe((num_partitions + 1) * sizeof(size_type), split_align);
  size_t const src_buf_info_size =
    cudf::util::round_up_safe(num_src_bufs * sizeof(src_buf_info), split_align);
  // host-side
  std::vector<uint8_t> h_indices_and_source_info(indices_size + src_buf_info_size);
  size_type* h_indices = reinterpret_cast<size_type*>(h_indices_and_source_info.data());
  src_buf_info* h_src_buf_head =
    reinterpret_cast<src_buf_info*>(h_indices_and_source_info.data() + indices_size);
  src_buf_info* h_validity_buf_info = h_src_buf_head;
  src_buf_info* h_offset_buf_info   = h_validity_buf_info + num_src_bufs_by_type.validity_buf_count;
  src_buf_info* h_data_buf_info     = h_offset_buf_info + num_src_bufs_by_type.offset_buf_count;
  // device-side
  // gpu-only : stack space needed for nested list offset calculation
  int const offset_stack_partition_size = compute_offset_stack_size(input.begin(), input.end());
  size_t const offset_stack_size = offset_stack_partition_size * num_partitions * sizeof(size_type);
  rmm::device_buffer d_indices_and_source_info(indices_size + src_buf_info_size + offset_stack_size,
                                               stream,
                                               rmm::mr::get_current_device_resource_ref());
  auto* d_indices              = reinterpret_cast<size_type*>(d_indices_and_source_info.data());
  src_buf_info* d_src_buf_info = reinterpret_cast<src_buf_info*>(
    reinterpret_cast<uint8_t*>(d_indices_and_source_info.data()) + indices_size);
  size_type* d_offset_stack =
    reinterpret_cast<size_type*>(reinterpret_cast<uint8_t*>(d_indices_and_source_info.data()) +
                                 indices_size + src_buf_info_size);

  // compute splits -> indices.
  h_indices[0]              = 0;
  h_indices[num_partitions] = input.column(0).size();
  std::copy(splits.begin(), splits.end(), std::next(h_indices));

  // setup source buf info
  auto const total_flattened_columns = count_flattened_columns(input.begin(), input.end()).first;
  int offset_stack_pos               = 0;
  int col_index                      = 0;
  setup_source_buf_info(input.begin(),
                        input.end(),
                        h_src_buf_head,
                        h_validity_buf_info,
                        h_offset_buf_info,
                        h_data_buf_info,
                        offset_stack_pos,
                        col_index,
                        stream);

  // HtoD indices and source buf info to device
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_indices, h_indices, indices_size + src_buf_info_size, cudaMemcpyDefault, stream.value()));

  // packed block of memory 2. partition buffer sizes, dst_buf_info structs and per-partition
  // has-validity buffer
  size_t const partition_sizes_size =
    cudf::util::round_up_safe(num_partitions * sizeof(partition_size_info) * 2, split_align);
  size_t const dst_buf_info_size =
    cudf::util::round_up_safe(num_bufs * sizeof(dst_buf_info), split_align);
  size_t const partition_has_validity_size =
    cudf::util::round_up_safe(total_flattened_columns * num_partitions * sizeof(bool), split_align);
  // device-side
  rmm::device_buffer d_buf_sizes_and_dst_info(
    partition_sizes_size + dst_buf_info_size + partition_has_validity_size, stream, temp_mr);
  partition_size_info* d_partition_sizes_unpadded =
    reinterpret_cast<partition_size_info*>(d_buf_sizes_and_dst_info.data());
  partition_size_info* d_partition_sizes = d_partition_sizes_unpadded + num_partitions;
  dst_buf_info* d_dst_buf_info           = reinterpret_cast<dst_buf_info*>(
    static_cast<uint8_t*>(d_buf_sizes_and_dst_info.data()) + partition_sizes_size);
  bool* d_flattened_col_inst_has_validity =
    reinterpret_cast<bool*>(static_cast<uint8_t*>(d_buf_sizes_and_dst_info.data()) +
                            partition_sizes_size + dst_buf_info_size);

  // this has to be a separate allocation because it gets returned.
  rmm::device_uvector<size_t> d_partition_offsets(num_partitions + 1, stream, mr);

  // set all has-validity bools to false for all column instances by default
  thrust::fill_n(rmm::exec_policy_nosync(stream),
                 d_flattened_col_inst_has_validity,
                 total_flattened_columns * num_partitions,
                 false);

  // compute sizes of each buffer in each partition, including alignment.
  thrust::for_each(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(num_bufs),
    [d_dst_buf_info,
     bufs_per_partition,
     d_indices,
     d_src_buf_info,
     d_offset_stack,
     offset_stack_partition_size,
     d_flattened_col_inst_has_validity,
     total_flattened_columns] __device__(size_t t) {
      int const partition_index = t / bufs_per_partition;
      int const src_buf_index   = t % bufs_per_partition;
      int const dst_buf_index   = (bufs_per_partition * partition_index) + src_buf_index;
      auto const& src_info      = d_src_buf_info[src_buf_index];

      // apply nested offsets (for lists and string columns).
      //
      // We can't just use the incoming row indices to figure out where to read from in a
      // nested list situation.  We have to apply offsets every time we cross a boundary
      // (list or string).  This loop applies those offsets so that our incoming row_index_start
      // and row_index_end get transformed to our final values.
      //
      int const stack_pos =
        src_info.offset_stack_pos + (partition_index * offset_stack_partition_size);
      size_type* offset_stack  = &d_offset_stack[stack_pos];
      int parent_offsets_index = src_info.parent_offsets_index;
      int stack_size           = 0;
      int root_column_offset   = src_info.column_offset;
      while (parent_offsets_index >= 0) {
        offset_stack[stack_size++] = parent_offsets_index;
        root_column_offset         = d_src_buf_info[parent_offsets_index].column_offset;
        parent_offsets_index       = d_src_buf_info[parent_offsets_index].parent_offsets_index;
      }
      // make sure to include the -column- offset on the root column in our calculation.
      int row_start = d_indices[partition_index] + root_column_offset;
      int row_end   = d_indices[partition_index + 1] + root_column_offset;
      while (stack_size > 0) {
        stack_size--;
        auto const offsets =
          reinterpret_cast<size_type const*>(d_src_buf_info[offset_stack[stack_size]].data);
        // this case can happen when you have empty string or list columns constructed with
        // empty_like()
        if (offsets != nullptr) {
          row_start = offsets[row_start];
          row_end   = offsets[row_end];
        }
      }

      // final element indices and row count
      size_t const src_element_index =
        src_info.btype == buffer_type::VALIDITY ? row_start / 8 : row_start;
      int const num_rows = row_end - row_start;
      // # of rows isn't necessarily the same as # of elements to be copied.
      auto const num_elements = [&]() {
        if ((src_info.btype == buffer_type::OFFSETS) && (src_info.data != nullptr) &&
            (num_rows > 0)) {
          return num_rows + 1;
        } else if (src_info.btype == buffer_type::VALIDITY) {
          // edge case: we are going to be copying at the nearest byte boundary, so
          // if we are at row 2 and we need 8 rows, we are actually copying 10 rows of info.
          return num_rows > 0 ? (num_rows + (row_start % 8) + 7) / 8 : 0;
        }
        return num_rows;
      }();

      // if this is a validity buffer with a > number of rows, flag this
      // column as including a validity buffer for this partition
      if (src_info.btype == buffer_type::VALIDITY && num_rows > 0) {
        d_flattened_col_inst_has_validity[src_info.column_index +
                                          (partition_index * total_flattened_columns)] = true;
      }

      int const element_size =
        src_info.btype == buffer_type::VALIDITY
          ? 1
          : cudf::type_dispatcher(data_type{src_info.type}, size_of_helper{});
      size_t const bytes = static_cast<size_t>(num_elements) * static_cast<size_t>(element_size);

      d_dst_buf_info[dst_buf_index] =
        dst_buf_info{bytes, src_info.btype, src_buf_index, src_element_index * element_size, 0};
    });

  // compute per-partition metadata size
  auto const per_partition_metadata_size =
    compute_per_partition_metadata_size(total_flattened_columns);

  auto partition_keys = spark_rapids_jni::util::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<size_t>([bufs_per_partition] __device__(size_t buf_index) {
      return buf_index / bufs_per_partition;
    }));

  // - compute: unpadded section sizes (validity, offsets, data)
  auto buf_sizes_by_type = spark_rapids_jni::util::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<partition_size_info>([d_dst_buf_info] __device__(int index) {
      switch (d_dst_buf_info[index].type) {
        case buffer_type::VALIDITY:
          return partition_size_info{d_dst_buf_info[index].buf_size, 0, 0};
        case buffer_type::OFFSETS: return partition_size_info{0, d_dst_buf_info[index].buf_size, 0};
        case buffer_type::DATA: return partition_size_info{0, 0, d_dst_buf_info[index].buf_size};
        default: break;
      }
      return partition_size_info{0, 0, 0};
    }));
  auto buf_size_reduce = cuda::proclaim_return_type<partition_size_info>(
    [] __device__(partition_size_info const& lhs, partition_size_info const& rhs) {
      auto const validity_size = lhs.validity_size + rhs.validity_size;
      auto const offset_size   = lhs.offset_size + rhs.offset_size;
      auto const data_size     = lhs.data_size + rhs.data_size;
      return partition_size_info{validity_size, offset_size, data_size};
    });
  thrust::reduce_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                        partition_keys,
                        partition_keys + num_bufs,
                        buf_sizes_by_type,
                        thrust::make_discard_iterator(),
                        d_partition_sizes_unpadded,
                        cuda::std::equal_to<size_t>{},  // key equality check
                        buf_size_reduce);

  // - compute: padded section sizes
  thrust::transform(
    rmm::exec_policy_nosync(stream, temp_mr),
    d_partition_sizes_unpadded,
    d_partition_sizes_unpadded + num_partitions,
    d_partition_sizes,
    cuda::proclaim_return_type<partition_size_info>(
      [per_partition_metadata_size] __device__(partition_size_info const& pinfo) {
        auto const validity_begin = per_partition_metadata_size;
        auto const offsets_begin =
          cudf::util::round_up_safe(validity_begin + pinfo.validity_size, validity_pad);
        auto const data_begin =
          cudf::util::round_up_safe(offsets_begin + pinfo.offset_size, offset_pad);
        auto const data_end = cudf::util::round_up_safe(data_begin + pinfo.data_size, data_pad);

        return partition_size_info{
          offsets_begin - validity_begin, data_begin - offsets_begin, data_end - data_begin};
      }));

  // - compute partition start offsets and total output buffer size overall
  auto partition_size_iter = spark_rapids_jni::util::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<size_t>(
      [num_partitions, d_partition_sizes, per_partition_metadata_size] __device__(size_t i) {
        auto const& pinfo = d_partition_sizes[i];
        return i >= num_partitions ? 0
                                   : per_partition_metadata_size + pinfo.validity_size +
                                       pinfo.offset_size + pinfo.data_size;
      }));
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                         partition_size_iter,
                         partition_size_iter + num_partitions + 1,
                         d_partition_offsets.begin());

  size_t dst_buf_total_size;
  cudaMemcpyAsync(&dst_buf_total_size,
                  d_partition_offsets.begin() + num_partitions,
                  sizeof(size_t),
                  cudaMemcpyDeviceToHost,
                  stream);

  // generate destination offsets for each of the source copies, by partition, by section.
  auto buf_sizes = spark_rapids_jni::util::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<size_t>([d_dst_buf_info] __device__(size_t i) {
      return d_dst_buf_info[i].buf_size;
    }));
  thrust::exclusive_scan_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                                partition_keys,
                                partition_keys + num_bufs,
                                buf_sizes,
                                dst_offset_output_iterator{d_dst_buf_info});
  auto iter = thrust::make_counting_iterator(0);
  thrust::for_each(
    rmm::exec_policy_nosync(stream, temp_mr),
    iter,
    iter + num_bufs,
    [per_partition_metadata_size,
     bufs_per_partition,
     d_dst_buf_info,
     d_partition_sizes,
     d_partition_sizes_unpadded,
     d_partition_offsets = d_partition_offsets.begin()] __device__(size_type i) {
      auto const partition_index = i / bufs_per_partition;
      // number of bytes from the start of the (validity|offsets|data) section, not counting
      // padding between sections. we need this because the incoming dst_offset values are
      // computed this way.
      auto const unpadded_buffer_offset = cuda::proclaim_return_type<size_t>([&] __device__() {
        auto const& unpadded_ps = d_partition_sizes_unpadded[partition_index];
        switch (d_dst_buf_info[i].type) {
          case buffer_type::OFFSETS: return unpadded_ps.validity_size;
          case buffer_type::DATA: return unpadded_ps.validity_size + unpadded_ps.offset_size;
          default: return size_t{0};
        }
      })();
      // number of bytes from the beginning of the partition
      auto const section_offset = cuda::proclaim_return_type<size_t>([&] __device__() {
        auto const& ps = d_partition_sizes[partition_index];
        switch (d_dst_buf_info[i].type) {
          case buffer_type::OFFSETS: return per_partition_metadata_size + ps.validity_size;
          case buffer_type::DATA:
            return per_partition_metadata_size + ps.validity_size + ps.offset_size;
          // validity
          default: return per_partition_metadata_size;
        }
      })();
      d_dst_buf_info[i].dst_offset =
        d_partition_offsets[partition_index] +  // offset to the entire partition
        section_offset +                        // partition-relative offset to our section start
        (d_dst_buf_info[i].dst_offset -
         unpadded_buffer_offset);  // unpadded section-relative offset to the start of our buffer
    });

  // allocate output buffer
  stream.synchronize();  // for dst_buf_total_size from above
  rmm::device_buffer dst_buf(dst_buf_total_size, stream, mr);

  // pack per-partition data. one thread per (flattened) column.
  size_type const thread_count_per_partition = cudf::util::round_up_safe(
    total_flattened_columns, static_cast<size_t>(cudf::detail::warp_size));
  cudf::detail::grid_1d const grid{
    thread_count_per_partition * static_cast<size_type>(num_partitions), 128};
  pack_per_partition_metadata_kernel<<<grid.num_blocks,
                                       grid.num_threads_per_block,
                                       0,
                                       stream.value()>>>(reinterpret_cast<uint8_t*>(dst_buf.data()),
                                                         d_partition_offsets.data(),
                                                         num_partitions,
                                                         total_flattened_columns,
                                                         d_indices,
                                                         d_flattened_col_inst_has_validity,
                                                         d_partition_sizes);

  // perform the copy.
  split_copy(
    d_src_buf_info, reinterpret_cast<uint8_t*>(dst_buf.data()), num_bufs, d_dst_buf_info, stream);

  // do this before the synchronize to take advantage of any gpu time we can overlap with (this
  // function only uses the cpu).
  auto metadata = compute_metadata(input, total_flattened_columns);

  stream.synchronize();
  return {shuffle_split_result{std::make_unique<rmm::device_buffer>(std::move(dst_buf)),
                               std::move(d_partition_offsets)},
          std::move(metadata)};
}

}  // namespace spark_rapids_jni
