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
#include <cudf/detail/utilities/batched_memset.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda/functional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <thrust/binary_search.h>
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

/*
 * Code block for computing assemble_column_info structures for columns and column instances.
 * Key function: assemble_build_column_info
 *
 */
namespace {

#define OUTPUT_ITERATOR(__name, __T, __field_name)                                       \
  template <typename __T>                                                                \
  struct __name##generic_output_iter {                                                   \
    __T* c;                                                                              \
    using value_type        = decltype(__T::__field_name);                               \
    using difference_type   = size_t;                                                    \
    using pointer           = decltype(__T::__field_name)*;                              \
    using reference         = decltype(__T::__field_name)&;                              \
    using iterator_category = thrust::output_device_iterator_tag;                        \
                                                                                         \
    __name##generic_output_iter operator+ __host__ __device__(int i) { return {c + i}; } \
                                                                                         \
    __name##generic_output_iter& operator++ __host__ __device__()                        \
    {                                                                                    \
      c++;                                                                               \
      return *this;                                                                      \
    }                                                                                    \
                                                                                         \
    reference operator[] __device__(int i) { return dereference(c + i); }                \
    reference operator* __device__() { return dereference(c); }                          \
                                                                                         \
   private:                                                                              \
    reference __device__ dereference(__T* c) { return c->__field_name; }                 \
  };                                                                                     \
  using __name = __name##generic_output_iter<__T>

/**
 * @brief Struct which contains information about columns and column instances.
 *
 * Used for both columns (one per output column) and column instances (one column per
 * output column * the number of partitions)
 */
struct assemble_column_info {
  cudf::type_id type;
  bool has_validity;
  size_type num_rows, num_chars;
  size_type valid_count;
  size_type num_children;

  // only valid for column instances
  size_type row_index, src_row_index;
  size_type char_index;
  size_type child_num_rows, child_src_row_index;
};
OUTPUT_ITERATOR(assemble_column_info_has_validity_output_iter, assemble_column_info, has_validity);

/**
 * @brief Information for columns that are children of offset-based parents, or are
 * offset-based themselves. Used for computing row counts and indices.
 */
struct offset_column_info {
  int col_index;  // index in shuffle split metadata column array
  int parent;     // index of the parent for this column, or -1 if it is a 'root' offset column
};

/**
 * @brief Helper function for computing root offset columns and branching depth.
 */
int compute_offset_column_info_traverse(cudf::host_span<shuffle_split_col_data const> cols,
                                        int col_index,
                                        int offset_info_parent,
                                        std::vector<offset_column_info>& out)
{
  auto const& col      = cols[col_index];
  bool const is_struct = col.type == cudf::type_id::STRUCT;
  bool const is_list   = col.type == cudf::type_id::LIST;
  bool const is_string = col.type == cudf::type_id::STRING;

  // add any column that affects row counts, or has a parent that does
  if (is_list || is_string || offset_info_parent >= 0) {
    out.push_back({col_index, offset_info_parent});
  }
  // list columns update the offset parent
  if (is_list) { offset_info_parent = col_index; }

  // recurse through structs and lists
  col_index++;
  if (is_struct || is_list) {
    auto const num_children = col.num_children();
    for (auto idx = 0; idx < num_children; idx++) {
      col_index = compute_offset_column_info_traverse(cols, col_index, offset_info_parent, out);
    }
  }
  return col_index;
}

/**
 * @brief Computes the set of column indices for all root columns that contain offsets, and their
 * children.
 *
 * We need to do some extra processing on all columns that contain offsets, so we build a list
 * of all the 'root' columns which contain offsets and their children.
 * 'root' does not necessarily imply a column at the top of the table
 * hierarchy, it means any offset-based column that does not have any offset-based columns above it
 * in the hierarchy. An example of this would be a list column inside of a struct.
 *
 * Children of offset columns have their parent offset-column index stored so that we can look up
 * our updated row count/index.
 *
 * @param metadata Metadata for the input
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr User provided resource used for allocating the returned device memory
 *
 * @returns A vector containing the offset_column_info for all root offset columns, and their
 * children.
 */
rmm::device_uvector<offset_column_info> compute_offset_column_info(
  shuffle_split_metadata const& metadata,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<offset_column_info> offset_info;
  offset_info.reserve(metadata.col_info.size());
  int col_index = 0;
  while (col_index < static_cast<int>(metadata.col_info.size())) {
    col_index = compute_offset_column_info_traverse(metadata.col_info, col_index, -1, offset_info);
  }
  return cudf::detail::make_device_uvector_async(offset_info, stream, mr);
}

/**
 * @brief Kernel which computes the row counts for all column instances that are the children
 * of one or more offset columns.
 *
 * The number of rows in the partition header is only applicable to the root columns of the
 * input. As soon as we have a column that contains offsets (strings or lists), the row counts
 * change for all children. This kernel computes these row counts for all child column instances.
 *
 * @param offset_columns Offset column information
 * @param column_metadata Input column metadata
 * @param column_instances The column instance structs to be filled in with row counts.
 * @param partitions The input buffer
 * @param partition_offsets Per-partition offsets into the input buffer
 * @param per_partition_metadata_size Size of the header for each partition
 *
 */
__global__ void compute_offset_child_row_counts(
  cudf::device_span<offset_column_info> offset_columns,
  cudf::device_span<shuffle_split_col_data const> column_metadata,
  cudf::device_span<assemble_column_info> column_instances,
  cudf::device_span<uint8_t const> partitions,
  cudf::device_span<size_t const> partition_offsets,
  size_t per_partition_metadata_size)
{
  if (threadIdx.x != 0) { return; }
  auto const partition_index            = blockIdx.x;
  partition_header const* const pheader = reinterpret_cast<partition_header const*>(
    partitions.data() + partition_offsets[partition_index]);
  size_t const offsets_begin = partition_offsets[partition_index] + per_partition_metadata_size +
                               cudf::hashing::detail::swap_endian(pheader->validity_size);
  size_type const* offsets = reinterpret_cast<size_type const*>(partitions.data() + offsets_begin);

  // walk all of the offset-based columns and their children for this partition and apply offsets to
  // shift row counts and src row index.
  auto const base_num_rows      = cudf::hashing::detail::swap_endian(pheader->num_rows);
  auto const base_src_row_index = cudf::hashing::detail::swap_endian(pheader->row_index);
  auto const base_col_index     = column_metadata.size() * partition_index;
  for (auto idx = 0; idx < offset_columns.size(); idx++) {
    auto& offset_info = offset_columns[idx];

    auto const col_index          = offset_info.col_index;
    auto const col_instance_index = col_index + base_col_index;
    auto const& meta              = column_metadata[col_index];
    auto& col_inst                = column_instances[col_instance_index];

    bool const is_list   = meta.type == cudf::type_id::LIST;
    bool const is_string = meta.type == cudf::type_id::STRING;

    auto const [num_rows, src_row_index] = [&] __device__() {
      // if I'm a root column, use the base partition row info
      // otherwise use the row info computed for me by my parent
      return offset_info.parent < 0 || base_num_rows == 0
               ? cuda::std::pair<size_type, size_type>{base_num_rows, base_src_row_index}
               : cuda::std::pair<size_type, size_type>{
                   column_instances[offset_info.parent + base_col_index].child_num_rows,
                   column_instances[offset_info.parent + base_col_index].child_src_row_index};
    }();
    col_inst.num_rows      = num_rows;
    col_inst.src_row_index = src_row_index;

    // strings also compute their char count
    if (is_string) {
      if (num_rows > 0) {
        col_inst.num_chars = offsets[num_rows] - offsets[0];
        offsets += (num_rows + 1);
      } else {
        col_inst.num_chars = 0;
      }
    }
    // lists change row counts and src row indices
    else if (is_list) {
      if (num_rows > 0) {
        col_inst.child_src_row_index = offsets[0];
        col_inst.child_num_rows      = offsets[num_rows] - offsets[0];
        offsets += (num_rows + 1);
      } else {
        col_inst.child_src_row_index = src_row_index;
        col_inst.child_num_rows      = 0;
      }
    }
  }
}

/**
 * @brief Generate assemble_column_info structures for columns and column_instances
 *
 * We need various pieces of information at the column level (corresponding to the
 * actual input schema) and the column instance level (per-column, per-partition), such
 * as type, number of children, row counts, etc. This function generates that data from
 * the input buffer.
 *
 * @param shuffle_split_metadata Global metadata about the partition buffer
 * @param partitions The partition buffer. One large blob of bytes.
 * @param partition_offsets Per-partition offets into the buffer
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr User provided resource used for allocating the returned device memory
 *
 * @return per-column info, per-column-instance info and the total per-partition metadata size
 */
std::tuple<rmm::device_uvector<assemble_column_info>,
           rmm::device_uvector<assemble_column_info>,
           size_t>
assemble_build_column_info(shuffle_split_metadata const& h_global_metadata,
                           cudf::device_span<uint8_t const> partitions,
                           cudf::device_span<size_t const> partition_offsets,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  auto temp_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<shuffle_split_col_data> global_metadata =
    cudf::detail::make_device_uvector_async(h_global_metadata.col_info, stream, temp_mr);

  // "columns" here means the number of flattened columns in the entire source table, not just the
  // number of columns at the top level
  auto const num_columns          = global_metadata.size();
  size_type const num_partitions  = partition_offsets.size() - 1;
  auto const num_column_instances = num_columns * num_partitions;

  // return values
  rmm::device_uvector<assemble_column_info> column_info(num_columns, stream, mr);
  rmm::device_uvector<assemble_column_info> column_instance_info(num_column_instances, stream, mr);

  // compute per-partition metadata size
  auto const per_partition_metadata_size =
    compute_per_partition_metadata_size(h_global_metadata.col_info.size());

  // generate per-column data ------------------------------------------------------

  // compute has-validity
  // note that we are iterating vertically -> horizontally here, with each column's individual piece
  // per partition first.
  auto col_has_validity = cuda::proclaim_return_type<bool>(
    [] __device__(bitmask_type const* const has_validity_buf, int col_index) -> bool {
      return has_validity_buf[col_index / 32] & (1 << (col_index % 32)) ? 1 : 0;
    });
  auto column_keys = spark_rapids_jni::util::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<size_type>([num_partitions] __device__(size_type i) {
      return i / num_partitions;
    }));
  auto has_validity_values = spark_rapids_jni::util::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<bool>([num_partitions,
                                      partitions        = partitions.data(),
                                      partition_offsets = partition_offsets.begin(),
                                      col_has_validity] __device__(int i) -> bool {
      auto const partition_index                 = i % num_partitions;
      bitmask_type const* const has_validity_buf = reinterpret_cast<bitmask_type const*>(
        partitions + partition_offsets[partition_index] + sizeof(partition_header));
      auto const col_index = i / num_partitions;

      return col_has_validity(has_validity_buf, col_index);
    }));
  thrust::reduce_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                        column_keys,
                        column_keys + num_column_instances,
                        has_validity_values,
                        thrust::make_discard_iterator(),
                        assemble_column_info_has_validity_output_iter{column_info.begin()},
                        cuda::std::equal_to<size_type>{},
                        cuda::std::logical_or<bool>{});

  // compute everything else except row count (which will be done later after we have computed
  // column instance information)
  auto iter = thrust::make_counting_iterator(0);
  thrust::for_each(rmm::exec_policy_nosync(stream, temp_mr),
                   iter,
                   iter + num_columns,
                   [column_info     = column_info.begin(),
                    global_metadata = global_metadata.begin()] __device__(size_type col_index) {
                     auto const& metadata = global_metadata[col_index];
                     auto& cinfo          = column_info[col_index];

                     cinfo.type         = metadata.type;
                     cinfo.valid_count  = 0;
                     cinfo.num_children = metadata.num_children();
                   });

  // generate per-column-instance data ------------------------------------------------------

  // has-validity, type, # of children, row count for non-offset child columns
  thrust::for_each(
    rmm::exec_policy_nosync(stream, temp_mr),
    iter,
    iter + num_column_instances,
    [column_instance_info = column_instance_info.begin(),
     global_metadata      = global_metadata.begin(),
     partitions           = partitions.data(),
     partition_offsets    = partition_offsets.begin(),
     num_columns,
     per_partition_metadata_size,
     col_has_validity] __device__(size_type i) {
      auto const partition_index    = i / num_columns;
      auto const col_index          = i % num_columns;
      auto const col_instance_index = (partition_index * num_columns) + col_index;

      auto const& metadata = global_metadata[col_index];
      auto& cinstance_info = column_instance_info[col_instance_index];

      uint8_t const* const buf_start =
        reinterpret_cast<uint8_t const*>(partitions + partition_offsets[partition_index]);
      partition_header const* const pheader = reinterpret_cast<partition_header const*>(buf_start);

      bitmask_type const* const has_validity_buf =
        reinterpret_cast<bitmask_type const*>(buf_start + sizeof(partition_header));
      cinstance_info.has_validity = col_has_validity(has_validity_buf, col_index);

      cinstance_info.type         = metadata.type;
      cinstance_info.valid_count  = 0;
      cinstance_info.num_chars    = 0;
      cinstance_info.num_children = metadata.num_children();

      // note that these will be incorrect for any columns that are children of offset columns.
      // those values will be fixed up below.
      cinstance_info.num_rows      = cudf::hashing::detail::swap_endian(pheader->num_rows);
      cinstance_info.src_row_index = cudf::hashing::detail::swap_endian(pheader->row_index);
    });

  // reconstruct row counts for columns and columns instances  ------------------------------

  // compute row counts for offset-based column instances.
  // TODO: the kudo format forces us to be less parallel here than we could be. maybe find a way
  // around that which doesn't grow size very much.

  {
    auto offset_column_info = compute_offset_column_info(h_global_metadata, stream, mr);

    // parallelize by partition.
    // unfortunately, there's no way to parallelize this at the column level. we don't know where
    // the offsets start in the partition buffer for any given column, so we have to march through
    // each partition linearly. to fix this, we'd have to change the kudo format in a way that would
    // increase it's size. I'm doing this as a kernel instead of through thrust so that I can
    // guarantee each partition is being marched by a seperate block to avoid thread divergence.
    compute_offset_child_row_counts<<<num_partitions, 32, 0, stream.value()>>>(
      offset_column_info,
      global_metadata,
      column_instance_info,
      partitions,
      partition_offsets,
      per_partition_metadata_size);
  }

  // returns column instance from index, in the order of 0->num_partitions, 0->num_columns
  auto col_instance_vertical = cuda::proclaim_return_type<assemble_column_info&>(
    [num_partitions, num_columns, column_instance_info = column_instance_info.begin()] __device__(
      int i) -> assemble_column_info& {
      auto const partition_index = i % num_partitions;
      auto const col_inst_index  = (partition_index * num_columns) + (i / num_partitions);
      return column_instance_info[col_inst_index];
    });

  // compute row indices per column instance
  auto col_inst_row_index = spark_rapids_jni::util::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<int&>([col_instance_vertical] __device__(int i) -> int& {
      return col_instance_vertical(i).row_index;
    }));
  auto col_inst_row_count = spark_rapids_jni::util::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<size_type>([col_instance_vertical] __device__(int i) {
      return col_instance_vertical(i).num_rows;
    }));
  thrust::exclusive_scan_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                                column_keys,
                                column_keys + num_column_instances,
                                col_inst_row_count,
                                col_inst_row_index);

  // compute char indices per column instance
  auto col_inst_char_index = spark_rapids_jni::util::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<int&>([col_instance_vertical] __device__(int i) -> int& {
      return col_instance_vertical(i).char_index;
    }));
  auto col_inst_char_count = spark_rapids_jni::util::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<size_type>([col_instance_vertical] __device__(int i) {
      return col_instance_vertical(i).num_chars;
    }));
  thrust::exclusive_scan_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                                column_keys,
                                column_keys + num_column_instances,
                                col_inst_char_count,
                                col_inst_char_index);

  // compute row counts and char counts. because we already know per-instance indices and counts,
  // this can be done without a reduction
  thrust::for_each(rmm::exec_policy_nosync(stream, temp_mr),
                   iter,
                   iter + num_columns,
                   [col_inst_begin       = (num_partitions - 1) * num_columns,
                    column_info          = column_info.begin(),
                    column_instance_info = column_instance_info.begin()] __device__(int i) {
                     auto const& last_col_inst = column_instance_info[col_inst_begin + i];
                     column_info[i].num_rows   = last_col_inst.row_index + last_col_inst.num_rows;
                     column_info[i].num_chars  = last_col_inst.char_index + last_col_inst.num_chars;
                   });

  return {std::move(column_info), std::move(column_instance_info), per_partition_metadata_size};
}

}  // namespace

/*
 * Code block for generating output buffers and the associated copy batch information
 * needed to fill them
 * Key function: assemble_build_buffers
 *
 */
namespace {

/**
 * @brief Compute size of a bitmask in bytes, including padding
 *
 * @param number_of_bits Number of bits in the bitmask
 * @param pad Padding for the final buffer size
 *
 * @return Size of the required allocation in bytes
 */
__host__ __device__ constexpr size_t bitmask_allocation_size_bytes(size_type number_of_bits,
                                                                   int pad)
{
  return cudf::util::round_up_safe((number_of_bits + 7) / 8, pad);
}

/**
 * @brief Functor that fills in source buffer sizes (validity, offsets, data) per column.
 *
 * This returns the size of the buffer -without- padding. Just the size of
 * the raw bytes containing the actual data.
 *
 * Note that there is an odd edge case here. We may have a partition that contains 8 rows, which
 * would translate to 1 byte of validity data. However, we copy from byte boundaries. So if this
 * partition started at row index 2, we would need to account for those extra 2 bits that will be
 * included in the buffer. So the source buffer will be 2 bytes (10 bits) even though the we are
 * only using 8 bits of that data in the output. Therefore, ths size returned from this functor is
 * only valid for -source- data.
 *
 */
struct assemble_src_buffer_size_functor {
  int const src_row_index;

  template <typename T, typename OutputIter, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  __device__ void operator()(assemble_column_info const& col,
                             OutputIter validity_out,
                             OutputIter offsets_out,
                             OutputIter data_out)
  {
    // validity
    *validity_out = validity_size(col);

    // no offsets for fixed width types
    *offsets_out = 0;

    // data
    *data_out = cudf::type_dispatcher(data_type{col.type}, size_of_helper{}) * col.num_rows;
  }

  template <typename T,
            typename OutputIter,
            CUDF_ENABLE_IF(cuda::std::is_same_v<T, cudf::list_view>)>
  __device__ void operator()(assemble_column_info const& col,
                             OutputIter validity_out,
                             OutputIter offsets_out,
                             OutputIter data_out)
  {
    // validity
    *validity_out = validity_size(col);

    // offsets
    *offsets_out = offsets_size(col);

    // no data for lists
    *data_out = 0;
  }

  template <typename T,
            typename OutputIter,
            CUDF_ENABLE_IF(cuda::std::is_same_v<T, cudf::struct_view>)>
  __device__ void operator()(assemble_column_info const& col,
                             OutputIter validity_out,
                             OutputIter offsets_out,
                             OutputIter data_out)
  {
    // validity
    *validity_out = validity_size(col);

    // no offsets or data for structs
    *offsets_out = 0;
    *data_out    = 0;
  }

  template <typename T,
            typename OutputIter,
            CUDF_ENABLE_IF(cuda::std::is_same_v<T, cudf::string_view>)>
  __device__ void operator()(assemble_column_info const& col,
                             OutputIter validity_out,
                             OutputIter offsets_out,
                             OutputIter data_out)
  {
    // validity
    *validity_out = validity_size(col);

    // chars
    *data_out = sizeof(int8_t) * col.num_chars;

    // offsets
    *offsets_out = offsets_size(col);
  }

  template <typename T,
            typename OutputIter,
            CUDF_ENABLE_IF(!cuda::std::is_same_v<T, cudf::struct_view> &&
                           !cuda::std::is_same_v<T, cudf::list_view> &&
                           !cuda::std::is_same_v<T, cudf::string_view> &&
                           !cudf::is_fixed_width<T>())>
  __device__ void operator()(assemble_column_info const& col,
                             OutputIter validity_out,
                             OutputIter offsets_out,
                             OutputIter data_out)
  {
  }

  __device__ size_t validity_size(assemble_column_info const& col) const
  {
    // if we have no validity, or no rows, include nothing.
    return col.has_validity ?
                            // handle the validity edge case from the function header
             (col.num_rows > 0
                ? bitmask_allocation_size_bytes(col.num_rows + (src_row_index % 8), 1)
                : 0)
                            : 0;
  }

  __device__ size_t offsets_size(assemble_column_info const& col) const
  {
    // if we have no rows, don't even generate the blank offset
    return col.num_rows > 0 ? (sizeof(size_type) * (col.num_rows + 1)) : 0;
  }
};

/**
 * @brief Utility function which expands a range of sizes, and invokes a function on
 * each element in all of the groups, providing group and subgroup indices, returning
 * the generated list of values.
 *
 * As an example, imagine we had the input
 * [2, 0, 5, 1]
 *
 * transform_expand will invoke `op` 8 times, with the values
 *
 * (0, 0) (0, 1),  (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),   (3, 0)
 *
 * @param first Beginning of the input range
 * @param end End of the input range
 * @param op Function to be invoked on each expanded value. This function should return
 * a single value that gets collected into the overall result returned from transform_expand
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr User provided resource used for allocating the returned device memory
 *
 * @return The resulting device_uvector containing the output from `op`
 */
template <typename SizeIterator, typename GroupFunction>
rmm::device_uvector<std::invoke_result_t<GroupFunction>> transform_expand(
  SizeIterator first,
  SizeIterator last,
  GroupFunction op,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto temp_mr = cudf::get_current_device_resource_ref();

  auto value_count  = std::distance(first, last);
  auto size_wrapper = spark_rapids_jni::util::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<size_t>([value_count, first] __device__(size_t i) {
      return i >= value_count ? 0 : first[i];
    }));
  rmm::device_uvector<size_t> group_offsets(value_count + 1, stream, temp_mr);
  thrust::exclusive_scan(rmm::exec_policy(stream, temp_mr),
                         size_wrapper,
                         size_wrapper + group_offsets.size(),
                         group_offsets.begin());
  size_t total_size = group_offsets.back_element(stream);  // note memcpy and device sync

  using OutputType = std::invoke_result_t<GroupFunction>;
  rmm::device_uvector<OutputType> result(total_size, stream, mr);
  auto iter = thrust::make_counting_iterator(0);
  thrust::transform(rmm::exec_policy(stream, temp_mr),
                    iter,
                    iter + total_size,
                    result.begin(),
                    cuda::proclaim_return_type<OutputType>(
                      [op,
                       group_offsets_begin = group_offsets.begin(),
                       group_offsets_end   = group_offsets.end()] __device__(size_t i) {
                        auto const index =
                          thrust::upper_bound(
                            thrust::seq, group_offsets_begin, group_offsets_end, i) -
                          group_offsets_begin;
                        auto const group_index = i < group_offsets_begin[index] ? index - 1 : index;
                        auto const intra_group_index = i - group_offsets_begin[group_index];
                        return op(group_index, intra_group_index);
                      }));

  return result;
}

/**
 * @brief Functor to calculate buffer sizes for the shared buffer approach.
 */
struct shared_buffer_size_functor {
  template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  void operator()(assemble_column_info const& col,
                  size_t& validity_size,
                  size_t& offsets_size,
                  size_t& data_size)
  {
    // validity
    validity_size = get_output_validity_size(col);

    // no offsets for fixed width types
    offsets_size = 0;

    // data
    data_size = cudf::util::round_up_safe(
      cudf::type_dispatcher(data_type{col.type}, size_of_helper{}) * col.num_rows, split_align);
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::list_view>)>
  void operator()(assemble_column_info const& col,
                  size_t& validity_size,
                  size_t& offsets_size,
                  size_t& data_size)
  {
    // validity
    validity_size = get_output_validity_size(col);

    // offsets
    offsets_size = get_output_offsets_size(col);

    // no data for lists
    data_size = 0;
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::struct_view>)>
  void operator()(assemble_column_info const& col,
                  size_t& validity_size,
                  size_t& offsets_size,
                  size_t& data_size)
  {
    // validity
    validity_size = get_output_validity_size(col);

    // no offsets or data for structs
    offsets_size = 0;
    data_size    = 0;
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::string_view>)>
  void operator()(assemble_column_info const& col,
                  size_t& validity_size,
                  size_t& offsets_size,
                  size_t& data_size)
  {
    // validity
    validity_size = get_output_validity_size(col);

    // chars
    data_size = cudf::util::round_up_safe(static_cast<size_t>(col.num_chars), split_align);

    // offsets
    offsets_size = get_output_offsets_size(col);
  }

  template <typename T,
            CUDF_ENABLE_IF(!std::is_same_v<T, cudf::struct_view> &&
                           !std::is_same_v<T, cudf::list_view> &&
                           !std::is_same_v<T, cudf::string_view> && !cudf::is_fixed_width<T>())>
  void operator()(assemble_column_info const& col,
                  size_t& validity_size,
                  size_t& offsets_size,
                  size_t& data_size)
  {
    CUDF_FAIL("Unsupported type in shared_buffer_size_functor");
  }

 private:
  /**
   * @brief Helper function to calculate validity buffer size for a column
   */
  size_t get_output_validity_size(assemble_column_info const& col) const
  {
    return col.has_validity
             ? cudf::util::round_up_safe(bitmask_allocation_size_bytes(col.num_rows, split_align),
                                         split_align)
             : 0;
  }

  /**
   * @brief Helper function to calculate offsets buffer size for a column
   */
  size_t get_output_offsets_size(assemble_column_info const& col) const
  {
    return cudf::util::round_up_safe(sizeof(size_type) * (col.num_rows + 1), split_align);
  }
};

// The size that shuffle_assemble uses internally as the GPU unit of work.
// For the validity and offset copy steps, the work is broken up into
// (bytes / desired_assemble_batch_size) kernel blocks
constexpr std::size_t desired_assemble_batch_size = 1 * 1024 * 1024;

/**
 * @brief Return the number of batches the specified number of bytes should be split into.
 */
__host__ __device__ constexpr size_t size_to_batch_count(size_t bytes)
{
  return cudf::util::round_up_unsafe(bytes, desired_assemble_batch_size) /
         desired_assemble_batch_size;
}

/**
 * @brief Information on a copy batch.
 */
struct assemble_batch {
  __device__ assemble_batch(uint8_t const* _src,
                            uint8_t* _dst,
                            size_t _size,
                            buffer_type _btype,
                            int _value_shift,
                            int _src_bit_shift,
                            int _dst_bit_shift,
                            size_type _validity_row_count,
                            size_type* _valid_count)
    : src(_src),
      dst(_dst),
      size(_size),
      btype(_btype),
      value_shift(_value_shift),
      src_bit_shift(_src_bit_shift),
      dst_bit_shift(_dst_bit_shift),
      validity_row_count(_validity_row_count),
      valid_count(_valid_count)
  {
  }

  uint8_t const* src;
  uint8_t* dst;
  size_t size;  // bytes
  buffer_type btype;
  int value_shift;               // amount to shift values down by (for offset buffers)
  int src_bit_shift;             // source bit (right) shift. easy way to think about this is
                                 // 'the number of rows at the beginning of the buffer to ignore'.
                                 // we need to ignore them because the split-copy happens at
                                 // byte boundaries, not bit/row boundaries. so we may have
                                 // irrelevant rows at the very beginning.
  int dst_bit_shift;             // dest bit (left) shift
  size_type validity_row_count;  // only valid for validity buffers
  size_type* valid_count;        // (output) validity count for this block of work
};

/**
 * @brief Generate the shared buffer allocation with buffer slices and copy batches.
 *
 * This is the key optimization: instead of O(n) allocations, we do one allocation
 * and create slices/views into it.
 *
 * @param column_info Per-column information
 * @param h_column_info Host memory per-column information
 * @param column_instance_info Per-column-instance information
 * @param partitions The partition buffer
 * @param partition_offsets Per-partition offsets into the partition buffer
 * @param per_partition_metadata_size Per-partition metadata header size
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr User provided resource used for allocating the returned device memory
 *
 * @return Shuffle assemble result with buffer slices and copy batches
 */
std::pair<shuffle_assemble_result, rmm::device_uvector<assemble_batch>> assemble_build_buffers(
  cudf::device_span<assemble_column_info> column_info,
  cudf::host_span<assemble_column_info const> h_column_info,
  cudf::device_span<assemble_column_info const> const& column_instance_info,
  cudf::device_span<uint8_t const> partitions,
  cudf::device_span<size_t const> partition_offsets,
  size_t per_partition_metadata_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto temp_mr                    = cudf::get_current_device_resource_ref();
  auto const num_columns          = column_info.size();
  auto const num_column_instances = column_instance_info.size();
  auto const num_partitions       = partition_offsets.size() - 1;

  // to simplify things, we will reserve 3 buffers for each column. validity, data, offsets. not
  // every column will use all of them, so those buffers will remain unallocated/zero size.
  size_t const num_dst_buffers       = num_columns * 3;
  size_t const num_src_buffers       = num_dst_buffers * num_partitions;
  size_t const buffers_per_partition = num_dst_buffers;

  // *** The shared buffer optimization ***
  // Calculate total memory needed for all buffers combined
  std::vector<size_t> validity_sizes(num_columns);
  std::vector<size_t> offsets_sizes(num_columns);
  std::vector<size_t> data_sizes(num_columns);

  for (size_t idx = 0; idx < num_columns; idx++) {
    cudf::type_dispatcher(cudf::data_type{h_column_info[idx].type},
                          shared_buffer_size_functor{},
                          h_column_info[idx],
                          validity_sizes[idx],
                          offsets_sizes[idx],
                          data_sizes[idx]);
  }

  // Calculate buffer offsets and total size using exclusive scan
  std::vector<size_t> all_sizes;
  all_sizes.reserve(num_dst_buffers);
  for (size_t col = 0; col < num_columns; col++) {
    all_sizes.push_back(validity_sizes[col]);
    all_sizes.push_back(offsets_sizes[col]);
    all_sizes.push_back(data_sizes[col]);
  }

  std::vector<size_t> buffer_offsets(num_dst_buffers);
  std::exclusive_scan(all_sizes.begin(), all_sizes.end(), buffer_offsets.begin(), size_t{0});
  size_t total_size = buffer_offsets.back() + all_sizes.back();

  // The shared buffer allocation! This is the core of the optimization
  rmm::device_buffer shared_buffer(total_size, stream, mr);
  uint8_t* base_ptr = static_cast<uint8_t*>(shared_buffer.data());

  // Create buffer slices that point into the shared buffer allocation.
  // Each slice represents a contiguous region of the shared buffer for one column's
  // validity, offsets, or data buffer. Previously we allocated each buffer individually
  // (O(n) allocations), but now we use one shared allocation with slices/views into it
  // for better memory management and reduced allocation overhead.
  std::vector<buffer_slice> buffer_slices(num_dst_buffers);

  // Collect validity buffers that need zero-initialization
  std::vector<cudf::device_span<cudf::bitmask_type>> validity_spans_to_zero;

  for (size_t i = 0; i < num_dst_buffers; i++) {
    size_t col_idx  = i / 3;
    size_t buf_type = i % 3;  // 0=validity, 1=offsets, 2=data
    size_t size     = (buf_type == 0)   ? validity_sizes[col_idx]
                      : (buf_type == 1) ? offsets_sizes[col_idx]
                                        : data_sizes[col_idx];

    buffer_slices[i] = buffer_slice(base_ptr + buffer_offsets[i], size, buffer_offsets[i]);

    // Collect validity buffers that need zero-initialization
    if (buf_type == 0 && size > 0 && h_column_info[col_idx].has_validity) {
      // Convert byte size to element count for bitmask_type (uint32_t) span
      size_t num_elements = size / sizeof(cudf::bitmask_type);
      validity_spans_to_zero.emplace_back(
        reinterpret_cast<cudf::bitmask_type*>(buffer_slices[i].data), num_elements);
    }
  }

  // Batch memset all validity buffers at once for better performance with many columns
  if (!validity_spans_to_zero.empty()) {
    cudf::host_span<cudf::device_span<cudf::bitmask_type> const> host_spans(
      validity_spans_to_zero.data(), validity_spans_to_zero.size());
    cudf::detail::batched_memset<cudf::bitmask_type>(host_spans, cudf::bitmask_type{0}, stream);
  }

  // Create device buffer of pointers to slices for copy operations
  std::vector<uint8_t*> h_dst_buffers(buffer_slices.size());
  std::transform(buffer_slices.begin(),
                 buffer_slices.end(),
                 h_dst_buffers.begin(),
                 [](const buffer_slice& slice) { return slice.data; });
  auto dst_buffers = cudf::detail::make_device_uvector_async(h_dst_buffers, stream, temp_mr);

  // compute:
  // - unpadded sizes of the source buffers
  // - offsets into the incoming partition data where each source buffer starts
  // - offsets into the output buffer data where each source buffer is copied into

  // ordered by the same as the incoming partition buffers (all validity buffers, all offset
  // buffers, all data buffers) so for 4 columns and 2 partitions, the ordering would be: vvvv oooo
  // dddd | vvvv oooo dddd
  rmm::device_uvector<size_t> src_sizes_unpadded(num_src_buffers, stream, mr);
  // these are per-source-buffer absolute offsets into the incoming partition buffer
  rmm::device_uvector<size_t> src_offsets(num_src_buffers, stream, mr);
  // arranged in destination buffer order, by partition
  // so for 4 columns and 2 partitions, the ordering would be:
  // vv oo dd vv oo dd vv oo dd vv oo dd  vp0/vp1, op0/op1, dp0/dp1, etc
  // 0  0  0  1  1  1  2  2  2  3  3  3   col_index
  rmm::device_uvector<size_t> dst_offsets(num_src_buffers, stream, mr);

  // mapping functions to try and manage the slipperiness of the various orderings.
  //
  // for 4 columns, 2 partitions
  // source buffer ordering: vvvv oooo dddd    | vvvv oooo dddd
  // dst buffer ordering:    vod vod vod vod   (the dst buffers are not partitioned. they are the
  // final column output buffers) dst offset ordering:    vv oo dd vv oo dd | vv oo dd vv oo dd
  auto src_buf_to_type = cuda::proclaim_return_type<size_t>(
    [buffers_per_partition, num_columns] __device__(size_t src_buf_index) {
      return (src_buf_index % buffers_per_partition) /
             num_columns;  // 0, 1, 2 (validity, offsets, data)
    });
  auto src_buf_to_dst_buf = cuda::proclaim_return_type<size_t>(
    [buffers_per_partition, num_columns, src_buf_to_type] __device__(size_t src_buf_index) {
      auto const col_index    = src_buf_index % num_columns;
      auto const buffer_index = src_buf_to_type(src_buf_index);
      return (col_index * 3) + buffer_index;
    });
  auto col_buffer_inst_to_dst_offset = cuda::proclaim_return_type<size_t>(
    [num_columns, num_partitions] __device__(
      size_t partition_index, size_t col_index, size_t buffer_index) {
      return (col_index * num_partitions * 3) + (buffer_index * num_partitions) + partition_index;
    });
  auto src_buf_to_dst_offset = cuda::proclaim_return_type<size_t>(
    [num_columns, buffers_per_partition, src_buf_to_type, col_buffer_inst_to_dst_offset] __device__(
      size_t src_buf_index) {
      auto const partition_index = src_buf_index / buffers_per_partition;
      auto const col_index       = src_buf_index % num_columns;
      auto const buffer_index    = src_buf_to_type(src_buf_index);
      return col_buffer_inst_to_dst_offset(partition_index, col_index, buffer_index);
    });
  auto dst_offset_to_src_buf = cuda::proclaim_return_type<size_t>(
    [num_partitions, num_columns, buffers_per_partition] __device__(size_t dst_offset_index) {
      auto const partition_index = dst_offset_index % num_partitions;
      auto const col_index       = dst_offset_index / (num_partitions * 3);
      auto const buffer_index    = (dst_offset_index / num_partitions) % 3;
      return (partition_index * buffers_per_partition) + col_index + (buffer_index * num_columns);
    });

  {
    // generate unpadded sizes of the source buffers
    auto const num_column_instances = column_instance_info.size();
    auto iter                       = thrust::make_counting_iterator(0);
    thrust::for_each(
      rmm::exec_policy(stream, temp_mr),
      iter,
      iter + num_column_instances,
      [buffers_per_partition,
       num_columns,
       column_instance_info = column_instance_info.begin(),
       src_sizes_unpadded   = src_sizes_unpadded.begin(),
       partition_offsets    = partition_offsets.begin(),
       partitions           = partitions.data()] __device__(size_type i) {
        auto const partition_index    = i / num_columns;
        auto const col_index          = i % num_columns;
        auto const col_instance_index = (partition_index * num_columns) + col_index;

        auto const& cinfo_instance    = column_instance_info[col_instance_index];
        auto const validity_buf_index = (partition_index * buffers_per_partition) + col_index;
        auto const offset_buf_index =
          (partition_index * buffers_per_partition) + num_columns + col_index;
        auto const data_buf_index =
          (partition_index * buffers_per_partition) + (num_columns * 2) + col_index;
        cudf::type_dispatcher(cudf::data_type{cinfo_instance.type},
                              assemble_src_buffer_size_functor{cinfo_instance.src_row_index},
                              cinfo_instance,
                              &src_sizes_unpadded[validity_buf_index],
                              &src_sizes_unpadded[offset_buf_index],
                              &src_sizes_unpadded[data_buf_index]);
      });

    // scan to source offsets, by partition, by section
    // so for 2 partitions with 2 columns, we would generate
    //      validity  offsets  data
    // P0:  0 A       0 B      0 C
    // P1   0 D       0 E      0 E
    auto section_keys = spark_rapids_jni::util::make_counting_transform_iterator(
      0, cuda::proclaim_return_type<size_t>([num_columns] __device__(size_t i) {
        return (i / num_columns);
      }));
    thrust::exclusive_scan_by_key(rmm::exec_policy(stream, temp_mr),
                                  section_keys,
                                  section_keys + num_src_buffers,
                                  src_sizes_unpadded.begin(),
                                  src_offsets.begin());

    // adjust the source offsets:
    // - add metadata offset
    // - add partition offset
    thrust::for_each(
      rmm::exec_policy(stream, temp_mr),
      iter,
      iter + num_column_instances,
      [num_columns,
       buffers_per_partition,
       column_instance_info = column_instance_info.begin(),
       src_offsets          = src_offsets.begin(),
       partition_offsets    = partition_offsets.begin(),
       partitions           = partitions.data(),
       per_partition_metadata_size] __device__(size_type i) {
        auto const partition_index  = i / num_columns;
        auto const partition_offset = partition_offsets[partition_index];
        auto const col_index        = i % num_columns;

        partition_header const* const pheader =
          reinterpret_cast<partition_header const*>(partitions + partition_offset);

        auto const validity_buf_index = (partition_index * buffers_per_partition) + col_index;
        auto const offset_buf_index   = validity_buf_index + num_columns;
        auto const data_buf_index     = offset_buf_index + num_columns;

        auto const validity_section_begin = partition_offset + per_partition_metadata_size;
        src_offsets[validity_buf_index] += validity_section_begin;

        auto const validity_size = cudf::hashing::detail::swap_endian(pheader->validity_size);
        auto const offset_size   = cudf::hashing::detail::swap_endian(pheader->offset_size);

        auto const offset_section_begin = validity_section_begin + validity_size;
        src_offsets[offset_buf_index] += offset_section_begin;

        auto const data_section_begin = offset_section_begin + offset_size;
        src_offsets[data_buf_index] += data_section_begin;
      });

    // compute: destination buffer offsets. see note above about ordering of dst_offsets.
    // Note: we're wasting a little work here as the work for validity has to be redone later.
    {
      auto dst_buf_key = spark_rapids_jni::util::make_counting_transform_iterator(
        0, cuda::proclaim_return_type<size_t>([num_partitions] __device__(size_t i) {
          return i / num_partitions;
        }));
      auto size_iter = spark_rapids_jni::util::make_counting_transform_iterator(
        0,
        cuda::proclaim_return_type<size_t>([src_sizes_unpadded = src_sizes_unpadded.begin(),
                                            num_partitions,
                                            dst_offset_to_src_buf] __device__(size_t i) {
          auto const src_buf_index     = dst_offset_to_src_buf(i);
          auto const buffer_index      = (i / num_partitions) % 3;
          bool const is_offsets_buffer = buffer_index == 1;

          // there is a mismatch between input and output sizes when it comes to offset buffers.
          // Each partition contains num_rows+1 offsets, however as we reassembly them, we only
          // consume num_rows offsets from each partition (except for the last one). So adjust our
          // side accordingly
          return src_sizes_unpadded[src_buf_index] -
                 ((is_offsets_buffer && src_sizes_unpadded[src_buf_index] > 0) ? 4 : 0);
        }));
      thrust::exclusive_scan_by_key(rmm::exec_policy(stream, temp_mr),
                                    dst_buf_key,
                                    dst_buf_key + num_src_buffers,
                                    size_iter,
                                    dst_offsets.begin());
    }

    // for validity, we need to do a little more work. our destination positions are defined by bit
    // position, not byte position. so round down into the nearest starting bitmask word. note that
    // this implies we will potentially be writing our leading bits into the same word as another
    // copy is writing it's trailing bits, so atomics will be necessary.
    thrust::for_each(
      rmm::exec_policy(stream, temp_mr),
      iter,
      iter + num_column_instances,
      [column_info          = column_info.begin(),
       column_instance_info = column_instance_info.begin(),
       num_columns,
       num_partitions,
       col_buffer_inst_to_dst_offset,
       dst_offsets = dst_offsets.begin()] __device__(size_t i) {
        auto const partition_index = i / num_columns;
        auto const col_index       = i % num_columns;
        auto const& cinfo          = column_info[col_index];
        if (cinfo.has_validity) {
          // for 4 columns and 2 partitions, the ordering of offsets is:
          // vv oo dd vv oo dd vv oo dd vv oo dd  vp0/vp1, op0/op1, dp0/dp1, etc
          auto const dst_offset_index = col_buffer_inst_to_dst_offset(
            partition_index, col_index, static_cast<int>(buffer_type::VALIDITY));
          auto const col_instance_index = (partition_index * num_columns) + col_index;
          dst_offsets[dst_offset_index] =
            (column_instance_info[col_instance_index].row_index / 32) * sizeof(bitmask_type);
        }
      });
  }

  // generate copy batches ------------------------------------
  //
  // - validity and offsets will be copied by custom kernels, so we will subdivide them them into
  // batches of 1 MB
  // - data is copied by cub, so we will do no subdivision of the batches under the assumption that
  // cub will make it's
  //   own smart internal decisions
  auto is_non_nullable_col_instance = cuda::proclaim_return_type<bool>(
    [] __device__(assemble_column_info const& cinfo, assemble_column_info const& cinfo_inst) {
      return cinfo.has_validity && !cinfo_inst.has_validity;
    });
  auto batch_src_buf_size = cuda::proclaim_return_type<size_t>(
    [src_buf_to_type,
     src_sizes_unpadded = src_sizes_unpadded.begin(),
     num_columns,
     buffers_per_partition,
     column_info,
     column_instance_info,
     is_non_nullable_col_instance] __device__(int src_buf_index) {
      // - handle an edge case with validity. if a particular column instance that does not contain
      //   validity is part of an overall column that does, it will have no src validity size. but
      //   we do need to fill it's rows in with all valids. So we have to create some artificial
      //   batches that will do a memset.
      if (static_cast<buffer_type>(src_buf_to_type(src_buf_index)) == buffer_type::VALIDITY) {
        auto const partition_index    = src_buf_index / buffers_per_partition;
        auto const col_index          = src_buf_index % num_columns;
        auto const col_instance_index = (partition_index * num_columns) + col_index;
        auto& cinfo                   = column_info[col_index];
        auto const& cinfo_inst        = column_instance_info[col_instance_index];
        if (is_non_nullable_col_instance(cinfo, cinfo_inst)) {
          return bitmask_allocation_size_bytes(cinfo_inst.num_rows, 1);
        }
      }
      return src_sizes_unpadded[src_buf_index];
    });
  auto batch_count_iter = spark_rapids_jni::util::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<size_t>(
      [batch_src_buf_size, src_buf_to_type] __device__(size_t src_buf_index) {
        // data buffers are copied via cub batched memcpy so we will not pre-batch things and will
        // return one full sized batch.
        if (static_cast<buffer_type>(src_buf_to_type(src_buf_index)) == buffer_type::DATA) {
          return size_t{1};
        }
        return size_to_batch_count(batch_src_buf_size(src_buf_index));
      }));
  auto copy_batches = transform_expand(
    batch_count_iter,
    batch_count_iter + src_sizes_unpadded.size(),
    cuda::proclaim_return_type<assemble_batch>(
      [dst_buffers       = dst_buffers.begin(),
       dst_offsets       = dst_offsets.begin(),
       partitions        = partitions.data(),
       partition_offsets = partition_offsets.data(),
       buffers_per_partition,
       num_partitions,
       src_offsets                 = src_offsets.begin(),
       desired_assemble_batch_size = desired_assemble_batch_size,
       column_info                 = column_info.begin(),
       column_instance_info        = column_instance_info.begin(),
       num_columns,
       src_buf_to_dst_buf,
       src_buf_to_dst_offset,
       src_buf_to_type,
       batch_src_buf_size,
       is_non_nullable_col_instance] __device__(size_t src_buf_index, size_t batch_index) {
        auto const batch_offset    = batch_index * desired_assemble_batch_size;
        auto const partition_index = src_buf_index / buffers_per_partition;

        auto const col_index          = src_buf_index % num_columns;
        auto const col_instance_index = (partition_index * num_columns) + col_index;
        auto& cinfo                   = column_info[col_index];
        auto const& cinfo_inst        = column_instance_info[col_instance_index];

        auto const src_offset = src_offsets[src_buf_index];

        buffer_type const btype = static_cast<buffer_type>(src_buf_to_type(src_buf_index));

        auto const dst_buf_index    = src_buf_to_dst_buf(src_buf_index);
        auto const dst_offset_index = src_buf_to_dst_offset(src_buf_index);
        auto const dst_offset       = dst_offsets[dst_offset_index];

        size_t const batch_src_size = batch_src_buf_size(src_buf_index);

        auto const bytes = [&] __device__() {
          switch (btype) {
            // validity gets batched
            case buffer_type::VALIDITY:
              return cuda::std::min(batch_src_size - batch_offset, desired_assemble_batch_size);

            // for offsets, all source buffers have an extra offset per partition (the terminating
            // offset for that partition) that we need to ignore, except in the case of the final
            // partition.
            case buffer_type::OFFSETS: {
              if (batch_src_size == 0) { return size_t{0}; }
              bool const end_of_buffer =
                (batch_offset + desired_assemble_batch_size) >= batch_src_size;
              if (!end_of_buffer) {
                return desired_assemble_batch_size;
              } else {
                // if we are the last partition to be copied, include the terminating offset.
                // there is an edge case to catch here. we may be the last partition that has rows
                // to be copied, but not the actual -last- partition.  For example, if the last 3
                // partitions had the row counts:  5, 0, 0 The final two partitions have no offset
                // data in them at all, so they can't provide the terminating offset. instead, the
                // partition containing 5 rows is the "last" partition, so it must do the
                // termination. to identify if we are this "last" partition, we check to see if our
                // final row index == the final row index of the true last partition.
                auto const last_partition_index = num_partitions - 1;
                auto const last_col_inst_index  = (last_partition_index * num_columns) + col_index;
                auto const& last_cinfo_inst     = column_instance_info[last_col_inst_index];
                auto const last_row_index = last_cinfo_inst.row_index + last_cinfo_inst.num_rows;
                bool const is_terminating_partition =
                  cinfo_inst.row_index + cinfo_inst.num_rows == last_row_index;

                auto const size =
                  cuda::std::min(batch_src_size - batch_offset, desired_assemble_batch_size);
                return is_terminating_partition ? size : size - 4;
              }
            }
            default: break;
          }

          // data copies go through the cub batched memcopy, so just do the whole thing in one shot.
          return batch_src_size;
        }();

        auto const validity_rows_per_batch  = desired_assemble_batch_size * 8;
        auto const validity_batch_row_index = (batch_index * validity_rows_per_batch);
        auto const validity_row_count =
          min(cinfo_inst.num_rows - validity_batch_row_index, validity_rows_per_batch);

        auto const dst_bit_shift = (cinfo_inst.row_index) % 32;
        // since the initial split copy is done on simple byte boundaries, the first bit we want to
        // copy may not be the first bit in the source buffer. so we need to shift right by these
        // leading bits. for example, the partition may start at row 3. but in that case, we will
        // have started copying from byte 0. so we have to shift right 3 rows.
        // Important: we need to know the -source- row index to do this, as that ultimately is what
        // affects what is in the incoming buffer. the destination row index is irrelevant.
        auto const src_bit_shift = cinfo_inst.src_row_index % 8;

        // transform the incoming raw offsets into final offsets
        int const offset_shift = [&] __device__() {
          if (btype != buffer_type::OFFSETS || batch_src_size == 0) { return 0; }

          auto const root_partition_offset =
            (reinterpret_cast<size_type const*>(partitions + src_offset))[0];
          // subtract the first offset value in the buffer, then add the row/char index
          if (cinfo.type == cudf::type_id::STRING) {
            return cinfo_inst.char_index - root_partition_offset;
          } else if (cinfo.type == cudf::type_id::LIST) {
            auto const& child_inst = column_instance_info[col_instance_index + 1];
            return child_inst.row_index - root_partition_offset;
          }
          // not an offset based column
          return 0;
        }();

        // for validity buffers that are going to end up being all-valid memsets, pass nullptr.
        auto src = is_non_nullable_col_instance(cinfo, cinfo_inst) && btype == buffer_type::VALIDITY
                     ? nullptr
                     : partitions + src_offset + batch_offset;

        return assemble_batch{src,
                              dst_buffers[dst_buf_index] + dst_offset + batch_offset,
                              bytes,
                              btype,
                              offset_shift,
                              src_bit_shift,
                              dst_bit_shift,
                              static_cast<size_type>(validity_row_count),
                              &cinfo.valid_count};
      }),
    stream,
    mr);

  // Return shuffle assemble result with slices and copy batches (column_views will be populated
  // later)
  shuffle_assemble_result result_buffers(std::move(shared_buffer), std::move(buffer_slices));
  return {std::move(result_buffers), std::move(copy_batches)};
}

}  // namespace

/*
 * Code block for copying validity, offset and data buffers.
 * Key function: assemble_copy_data
 *
 */
namespace {

/**
 * @brief Copy validity batches.
 *
 * This kernel assumes misaligned source buffers and an aligned (4 byte) destination buffer.
 */
template <int block_size>
__global__ void copy_validity(cudf::device_span<assemble_batch> batches)
{
  int batch_index = blockIdx.x;
  auto& batch     = batches[batch_index];
  if (batch.size <= 0 || (batch.btype != buffer_type::VALIDITY)) { return; }

  __shared__ bitmask_type prev_word[block_size];

  // how many leading bytes we have. that is, how many bytes will be read by the initial read, which
  // accounts for misaligned source buffers.
  int const leading_bytes = (4 - (reinterpret_cast<uint64_t>(batch.src) % 4));
  int remaining_rows      = batch.validity_row_count;

  // - if the address is misaligned, load byte-by-byte and only store up to that many bits/rows off
  // instead of a full word. Note: src_bit_shift effectively means "how many rows at the beginning
  // should we ignore", so we subtract that from the amount of rows that are actually in the bytes
  // that we've read.
  int rows_in_batch =
    min(remaining_rows,
        leading_bytes != 4 ? (leading_bytes * 8) - batch.src_bit_shift : 32 - batch.src_bit_shift);

  // safely load a word, handling alignment and allocation boundaries
  auto load_word = cuda::proclaim_return_type<bitmask_type>(
    [] __device__(void const* const _src, int num_bits, int leading_bytes) {
      // if src is null, this is coming from a column instance that has no validity, so we will
      // just fake it by returning all set bits
      if (_src == nullptr) { return static_cast<bitmask_type>((1 << num_bits) - 1); }

      uint8_t const* const src_b = reinterpret_cast<uint8_t const*>(_src);
      if (num_bits > 24) {
        // if we are aligned we can do a single read. this should be the most common case. only
        // at the ends of the range of bits to copy will we hit the non-aligned cases.
        return leading_bytes == 4
                 ? (reinterpret_cast<bitmask_type const*>(_src))[0]
                 : static_cast<bitmask_type>(src_b[0]) | static_cast<bitmask_type>(src_b[1] << 8) |
                     static_cast<bitmask_type>(src_b[2] << 16) |
                     static_cast<bitmask_type>(src_b[3] << 24);
      } else if (num_bits > 16) {
        return static_cast<bitmask_type>(src_b[0]) | static_cast<bitmask_type>(src_b[1] << 8) |
               static_cast<bitmask_type>(src_b[2] << 16);
      } else if (num_bits > 8) {
        return static_cast<bitmask_type>(src_b[0]) | static_cast<bitmask_type>(src_b[1] << 8);
      }
      return static_cast<bitmask_type>(src_b[0]);
    });

  size_type valid_count = 0;
  // thread 0 does all the work for the leading (unaligned) bytes in the source
  if (threadIdx.x == 0) {
    // note that the first row doesn't necessarily start at the byte boundary, so we need to add
    // the src_bit_shift to get the true number of bits to read.
    size_type const bits_to_load         = remaining_rows + batch.src_bit_shift;
    bitmask_type word                    = load_word(batch.src, bits_to_load, leading_bytes);
    bitmask_type const relevant_row_mask = ((1 << rows_in_batch) - 1);

    // shift and mask the incoming word so that bit 0 is the first row we're going to store.
    word = (word >> batch.src_bit_shift) & relevant_row_mask;

    // any bits that are not being stored in the current dest word get overflowed to the next copy
    prev_word[0] = word >> (32 - batch.dst_bit_shift);
    // shift to the final destination bit position.
    word <<= batch.dst_bit_shift;

    // use an atomic because we could be overlapping with another copy
    valid_count += __popc(word);
    atomicOr(reinterpret_cast<bitmask_type*>(batch.dst), word);
  }
  remaining_rows -= rows_in_batch;
  if (remaining_rows == 0) {
    if (threadIdx.x == 0) {
      // any overflow bits from the first word. the example case here is
      // 12 bits of data.  dst_bit_shift is 22, so we end up with 2 extra bits
      // (12 + 22 = 34) overflowing from the first batch.
      if (prev_word[0] != 0) {
        atomicOr(reinterpret_cast<bitmask_type*>(batch.dst) + 1, prev_word[0]);
        valid_count += __popc(prev_word[0]);
      }
      atomicAdd(batch.valid_count, valid_count);
    }
    return;
  }

  // src and dst pointers. src will be word-aligned now
  auto src = reinterpret_cast<bitmask_type const*>(
    batch.src == nullptr ? nullptr : batch.src + leading_bytes);
  auto dst = reinterpret_cast<bitmask_type*>(batch.dst);

  // how many words we visited in the initial batch. note that even if we only wrote 2 bits, it is
  // possible we skipped past the first word.  For example if our dst_bit_shift was 30 (that is, we
  // were starting to write at row index 30), 1 of the 2 bits would go in word 0, and 1 of them
  // would go in word 1. So our leading number of words is 1.
  auto const num_leading_words = ((rows_in_batch + batch.dst_bit_shift) / 32);
  auto remaining_words         = (remaining_rows + 31) / 32;

  // compute a new bit_shift.
  // - src_bit_shift is now irrelevant because we have skipped past any leading irrelevant bits in
  // the input
  // - the amount we have to dst shift is simply incremented by the number of rows we've processed.
  auto const bit_shift = (batch.dst_bit_shift + rows_in_batch) % 32;

  // we can still be anywhere in the destination buffer, so any stores to either the first or last
  // destination words (where other copies may be happening at the same time) need to use atomicOr.
  auto const last_word_index = num_leading_words + ((((remaining_rows + bit_shift) + 31) / 32) - 1);
  auto store_word            = [last_word_index, dst] __device__(int i, uint32_t val) {
    if (i == 0 || i == last_word_index) {
      atomicOr(dst + i, val);
    } else {
      dst[i] = val;
    }
    return __popc(val);
  };

  // copy the remaining words
  auto const rows_per_batch = blockDim.x * 32;
  int src_word_index        = threadIdx.x;
  int dst_word_index        = threadIdx.x + num_leading_words;
  int words_in_batch;
  bitmask_type cur, prev;
  do {
    words_in_batch = min(block_size, remaining_words);
    rows_in_batch  = min(remaining_rows, rows_per_batch);

    __syncthreads();
    if (threadIdx.x < words_in_batch) {
      // load current word, strip down to exactly the number of rows this thread is dealing with
      auto const thread_num_rows           = min(remaining_rows - (threadIdx.x * 32), 32);
      bitmask_type const relevant_row_mask = ((1 << thread_num_rows) - 1);
      cur = (load_word(src == nullptr ? nullptr : &src[src_word_index], thread_num_rows, 4) &
             relevant_row_mask);

      // bounce our trailing bits off shared memory. for example, if bit_shift is
      // 27, we are only storing the first 5 bits at the top of the current destination. The
      // trailing 27 bits become the -leading- 27 bits for the next word.
      //
      // dst_bit_shift = 27;
      // src:      00000000000000000000000000011111
      //
      // dst[0] =  11111xxxxxxxxxxxxxxxxxxxxxxxxxxx
      // dst[1] =  xxxx0000000000000000000000000000
      //
      prev = cur >> (32 - bit_shift);
      // the trailing bits in the last thread goes to the 0th thread for the next iteration of the
      // loop, so don't write it until the end of the loop, otherwise we'll inadvertently blow away
      // the 0th thread's correct value
      if (threadIdx.x < words_in_batch - 1) { prev_word[threadIdx.x + 1] = prev; }
    }
    __syncthreads();
    if (threadIdx.x < words_in_batch) {
      // construct final word from cur leading bits and prev trailing bits
      auto const word = (cur << bit_shift) | prev_word[threadIdx.x];
      valid_count += store_word(dst_word_index, word);
    }
    __syncthreads();
    // store the final trailing bits at the beginning for the next iteration
    if (threadIdx.x == words_in_batch - 1) { prev_word[0] = prev; }
    src_word_index += words_in_batch;
    dst_word_index += words_in_batch;
    remaining_words -= words_in_batch;
    remaining_rows -= rows_in_batch;
  } while (remaining_words > 0);

  __syncthreads();

  // final trailing bits, if any
  if (threadIdx.x == 0 && dst_word_index == last_word_index) {
    valid_count += store_word(dst_word_index, prev_word[0]);
  }

  // add the valid count for the entire block to the count for the entire buffer.
  using block_reduce = cub::BlockReduce<cudf::size_type, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;
  valid_count = block_reduce(temp_storage).Sum(valid_count);
  if (threadIdx.x == 0) { atomicAdd(batch.valid_count, valid_count); }
}

/**
 * @brief Copy offset batches.
 *
 * This kernel assumes aligned source buffer and destination buffers. It adjusts offsets
 * as necessary to handle stitching together disparate partitions
 */
template <int block_size>
__global__ void copy_offsets(cudf::device_span<assemble_batch> batches)
{
  int batch_index = blockIdx.x;
  auto& batch     = batches[batch_index];
  if ((batch.size <= 0) || (batch.btype != buffer_type::OFFSETS)) { return; }

  auto const offset_shift = batch.value_shift;
  auto num_offsets =
    batch.size / sizeof(cudf::size_type);  // TODO, someday handle long string columns
  auto offset_index = threadIdx.x;

  size_type const* const src = reinterpret_cast<size_type const*>(batch.src);
  size_type* dst             = reinterpret_cast<size_type*>(batch.dst);
  while (offset_index < num_offsets) {
    dst[offset_index] = src[offset_index] + offset_shift;
    offset_index += blockDim.x;
  }
}

/**
 * @brief Copy all batches into the final destination buffers.
 *
 * Invokes three separate copy kernels. Cub batched memcpy for data buffers, and custom kernels for
 * validity and offset copying. The validity and offset kernels both need to manipulate the data
 * during the copy.
 *
 * @param batches The set of all copy batches to be performed
 * @param column_info Per-column information
 * @param h_column_info Host-side per-column information. The valid_count field of this struct is
 * updated by this function
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void assemble_copy(cudf::device_span<assemble_batch> batches,
                   cudf::device_span<assemble_column_info const> column_info,
                   cudf::host_span<assemble_column_info> h_column_info,
                   rmm::cuda_stream_view stream)
{
  // TODO: it might make sense to launch these three copies on separate streams. It is likely that
  // the validity and offset copies will be much smaller than the data copies and could
  // theoretically not cause the gpu to be saturated.

  // main data copy. everything except validity and offsets
  {
    auto input_iter = spark_rapids_jni::util::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<void*>(
        [batches = batches.begin(), num_columns = column_info.size()] __device__(size_t i) {
          return batches[i].btype == buffer_type::DATA
                   ? reinterpret_cast<void*>(const_cast<uint8_t*>(batches[i].src))
                   : nullptr;
        }));
    auto output_iter = spark_rapids_jni::util::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<void*>(
        [batches = batches.begin(), num_columns = column_info.size()] __device__(size_t i) {
          return batches[i].btype == buffer_type::DATA
                   ? reinterpret_cast<void*>(const_cast<uint8_t*>(batches[i].dst))
                   : nullptr;
        }));
    auto size_iter = spark_rapids_jni::util::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<size_t>(
        [batches = batches.begin(), num_columns = column_info.size()] __device__(size_t i) {
          return batches[i].btype == buffer_type::DATA ? batches[i].size : 0;
        }));

    size_t temp_storage_bytes{0};
    cub::DeviceMemcpy::Batched(
      nullptr, temp_storage_bytes, input_iter, output_iter, size_iter, batches.size(), stream);
    rmm::device_buffer temp_storage(
      temp_storage_bytes, stream, cudf::get_current_device_resource_ref());
    cub::DeviceMemcpy::Batched(temp_storage.data(),
                               temp_storage_bytes,
                               input_iter,
                               output_iter,
                               size_iter,
                               batches.size(),
                               stream);
  }

  // copy validity
  constexpr int copy_validity_block_size = 128;
  copy_validity<copy_validity_block_size>
    <<<batches.size(), copy_validity_block_size, 0, stream.value()>>>(batches);

  // copy offsets
  constexpr int copy_offsets_block_size = 128;
  copy_offsets<copy_offsets_block_size>
    <<<batches.size(), copy_offsets_block_size, 0, stream.value()>>>(batches);

  // we have to sync because the build_table step will need the cpu-side valid_count in
  // h_column_info when constructing the columns.
  cudaMemcpyAsync(h_column_info.data(),
                  column_info.data(),
                  column_info.size() * sizeof(assemble_column_info),
                  cudaMemcpyDeviceToHost,
                  stream);
  stream.synchronize();
}

}  // namespace

/*
 * Code block for assembling the final shuffle assemble result
 * Key function: build_table
 *
 */
namespace {

// create mock assemble_column_info for empty columns
std::vector<assemble_column_info> create_empty_assemble_data(
  cudf::host_span<shuffle_split_col_data const> col_info)
{
  std::vector<assemble_column_info> result;
  result.reserve(col_info.size());

  for (size_t i = 0; i < col_info.size(); i++) {
    auto const& col = col_info[i];
    assemble_column_info info{};
    info.type         = col.type;
    info.has_validity = true;  // Assume columns have validity for consistency
    info.num_rows     = 0;
    info.valid_count  = 0;  // No rows, so valid_count = 0
    info.num_children = col.num_children();
    // Other fields are not used for empty columns
    result.push_back(info);
  }

  return result;
}

// calculate buffer sizes for empty columns
std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
calculate_empty_buffer_sizes(cudf::host_span<shuffle_split_col_data const> col_info,
                             cudf::host_span<assemble_column_info const> assemble_data)
{
  std::vector<size_t> validity_sizes(col_info.size());
  std::vector<size_t> offsets_sizes(col_info.size());
  std::vector<size_t> data_sizes(col_info.size());

  for (size_t i = 0; i < col_info.size(); i++) {
    cudf::type_dispatcher(cudf::data_type{col_info[i].type},
                          shared_buffer_size_functor{},
                          assemble_data[i],
                          validity_sizes[i],
                          offsets_sizes[i],
                          data_sizes[i]);
  }

  return {validity_sizes, offsets_sizes, data_sizes};
}

// initialize buffers for empty columns
void initialize_empty_buffers(uint8_t* buffer_base, size_t total_size, rmm::cuda_stream_view stream)
{
  // Initialize all buffers to 0, which is correct for:
  // - Validity: all bits 0 (but since num_rows=0, no bits matter)
  // - Offsets: 0 (correct for empty offsets)
  // - Data: 0 (empty data)
  cudaMemsetAsync(buffer_base, 0, total_size, stream);
}

// create buffer slices for empty columns
std::vector<buffer_slice> create_empty_buffer_slices(
  cudf::host_span<shuffle_split_col_data const> col_info,
  cudf::host_span<assemble_column_info const> assemble_data,
  uint8_t* buffer_base)
{
  auto [validity_sizes, offsets_sizes, data_sizes] =
    calculate_empty_buffer_sizes(col_info, assemble_data);

  size_t num_dst_buffers = col_info.size() * 3;
  std::vector<size_t> all_sizes;
  all_sizes.reserve(num_dst_buffers);
  for (size_t col = 0; col < col_info.size(); col++) {
    all_sizes.push_back(validity_sizes[col]);
    all_sizes.push_back(offsets_sizes[col]);
    all_sizes.push_back(data_sizes[col]);
  }

  std::vector<size_t> buffer_offsets(num_dst_buffers);
  std::exclusive_scan(all_sizes.begin(), all_sizes.end(), buffer_offsets.begin(), size_t{0});

  std::vector<buffer_slice> buffer_slices(num_dst_buffers);
  for (size_t i = 0; i < num_dst_buffers; i++) {
    buffer_slices[i] =
      buffer_slice(buffer_base + buffer_offsets[i], all_sizes[i], buffer_offsets[i]);
  }

  return buffer_slices;
}

/**
 * @brief Functor that generates column_view objects from buffer slices, handling nested types
 */
struct assemble_column_view_functor {
  cudf::host_span<shuffle_split_col_data const> column_meta;
  cudf::host_span<assemble_column_info const> assemble_data;
  shuffle_assemble_result const& assemble_result;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  std::pair<size_t, std::unique_ptr<cudf::column_view>> operator()(size_t col_index) const
  {
    auto const& col      = assemble_data[col_index];
    auto const& meta_col = column_meta[col_index];

    // Calculate buffer slice indices
    size_t validity_buffer_idx = col_index * 3;
    size_t data_buffer_idx     = col_index * 3 + 2;

    // Get buffer slices
    auto const& validity_slice = assemble_result.buffer_slices[validity_buffer_idx];
    auto const& data_slice     = assemble_result.buffer_slices[data_buffer_idx];

    // Create cudf::data_type with proper scale for fixed-point types
    cudf::type_id type = static_cast<cudf::type_id>(col.type);
    int32_t scale =
      spark_rapids_jni::is_fixed_point(cudf::data_type{col.type}) ? meta_col.scale() : 0;
    cudf::data_type dtype{type, scale};

    cudf::size_type null_count = col.has_validity ? (col.num_rows - col.valid_count) : 0;

    auto column_view = std::make_unique<cudf::column_view>(
      dtype,
      static_cast<cudf::size_type>(col.num_rows),
      data_slice.size > 0 ? static_cast<void const*>(data_slice.data) : nullptr,
      col.has_validity ? reinterpret_cast<cudf::bitmask_type const*>(validity_slice.data) : nullptr,
      null_count);

    return {col_index + 1, std::move(column_view)};
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::struct_view>)>
  std::pair<size_t, std::unique_ptr<cudf::column_view>> operator()(size_t col_index) const
  {
    auto const& col = assemble_data[col_index];

    // Calculate buffer slice indices (only validity for structs)
    size_t validity_buffer_idx = col_index * 3;
    auto const& validity_slice = assemble_result.buffer_slices[validity_buffer_idx];

    // Build children recursively
    std::vector<cudf::column_view> children;
    children.reserve(col.num_children);
    auto next = col_index + 1;
    for (cudf::size_type i = 0; i < col.num_children; i++) {
      auto [next_idx, child_view] = cudf::type_dispatcher(
        cudf::data_type{assemble_data[next].type},
        assemble_column_view_functor{column_meta, assemble_data, assemble_result, stream, mr},
        next);
      children.push_back(*child_view);  // Copy the column_view (not owning)
      next = next_idx;
    }

    cudf::size_type null_count = col.has_validity ? (col.num_rows - col.valid_count) : 0;

    auto column_view = std::make_unique<cudf::column_view>(
      cudf::data_type{cudf::type_id::STRUCT},
      static_cast<cudf::size_type>(col.num_rows),
      nullptr,  // structs have no data buffer
      col.has_validity ? reinterpret_cast<cudf::bitmask_type const*>(validity_slice.data) : nullptr,
      null_count,
      0,        // offset
      children  // pass children to constructor
    );

    return {next, std::move(column_view)};
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::string_view>)>
  std::pair<size_t, std::unique_ptr<cudf::column_view>> operator()(size_t col_index) const
  {
    auto const& col = assemble_data[col_index];

    // Calculate buffer slice indices
    size_t validity_buffer_idx = col_index * 3;
    size_t offsets_buffer_idx  = col_index * 3 + 1;
    size_t data_buffer_idx     = col_index * 3 + 2;

    // Get buffer slices
    auto const& validity_slice = assemble_result.buffer_slices[validity_buffer_idx];
    auto const& offsets_slice  = assemble_result.buffer_slices[offsets_buffer_idx];
    auto const& data_slice     = assemble_result.buffer_slices[data_buffer_idx];

    cudf::size_type null_count = col.has_validity ? (col.num_rows - col.valid_count) : 0;

    // For strings: chars data goes in parent's data buffer, offsets as single child column
    std::vector<cudf::column_view> children;
    // Always include offsets child column, even for empty strings
    children.emplace_back(cudf::data_type{cudf::type_id::INT32},
                          static_cast<cudf::size_type>(col.num_rows + 1),
                          static_cast<void const*>(offsets_slice.data),
                          nullptr,  // offsets never have nulls
                          0);

    auto column_view = std::make_unique<cudf::column_view>(
      cudf::data_type{cudf::type_id::STRING},
      static_cast<cudf::size_type>(col.num_rows),
      static_cast<void const*>(data_slice.data),  // chars data in parent's buffer
      col.has_validity ? reinterpret_cast<cudf::bitmask_type const*>(validity_slice.data) : nullptr,
      null_count,
      0,  // offset
      children);

    return {col_index + 1, std::move(column_view)};
  }

  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, cudf::list_view>)>
  std::pair<size_t, std::unique_ptr<cudf::column_view>> operator()(size_t col_index) const
  {
    auto const& col = assemble_data[col_index];

    // Calculate buffer slice indices
    size_t validity_buffer_idx = col_index * 3;
    size_t offsets_buffer_idx  = col_index * 3 + 1;
    auto const& validity_slice = assemble_result.buffer_slices[validity_buffer_idx];
    auto const& offsets_slice  = assemble_result.buffer_slices[offsets_buffer_idx];

    // Build the child column recursively
    std::vector<cudf::column_view> children;
    auto next                   = col_index + 1;
    auto [next_idx, child_view] = cudf::type_dispatcher(
      cudf::data_type{assemble_data[next].type},
      assemble_column_view_functor{column_meta, assemble_data, assemble_result, stream, mr},
      next);
    children.push_back(*child_view);

    cudf::size_type null_count = col.has_validity ? (col.num_rows - col.valid_count) : 0;

    // Create offsets child column
    children.insert(children.begin(),
                    cudf::column_view(cudf::data_type{cudf::type_id::INT32},
                                      static_cast<cudf::size_type>(col.num_rows + 1),
                                      static_cast<void const*>(offsets_slice.data),
                                      nullptr,  // offsets never have nulls
                                      0));

    auto column_view = std::make_unique<cudf::column_view>(
      cudf::data_type{cudf::type_id::LIST},
      static_cast<cudf::size_type>(col.num_rows),
      nullptr,  // lists have no data buffer at top level
      col.has_validity ? reinterpret_cast<cudf::bitmask_type const*>(validity_slice.data) : nullptr,
      null_count,
      0,  // offset
      children);

    return {next_idx, std::move(column_view)};
  }

  template <typename T,
            CUDF_ENABLE_IF(!cudf::is_fixed_width<T>() and !std::is_same_v<T, cudf::struct_view> and
                           !std::is_same_v<T, cudf::string_view> and
                           !std::is_same_v<T, cudf::list_view>)>
  std::pair<size_t, std::unique_ptr<cudf::column_view>> operator()(size_t col_index) const
  {
    CUDF_FAIL("Unsupported type in assemble_column_view_functor");
  }
};

// assemble all the column_views and populate the final shuffle assemble result
void build_table(cudf::host_span<shuffle_split_col_data const> column_meta,
                 cudf::host_span<assemble_column_info const> assemble_data,
                 shuffle_assemble_result& assemble_result,
                 rmm::cuda_stream_view stream,
                 rmm::device_async_resource_ref mr)
{
  // create native cudf::column_view objects pointing to slices in the shared buffer
  std::vector<std::unique_ptr<cudf::column_view>> column_views;
  column_views.reserve(assemble_data.size());

  size_t col_index = 0;
  while (col_index < assemble_data.size()) {
    auto [next_idx, column_view] = cudf::type_dispatcher(
      cudf::data_type{assemble_data[col_index].type},
      assemble_column_view_functor{column_meta, assemble_data, assemble_result, stream, mr},
      col_index);
    column_views.push_back(std::move(column_view));
    col_index = next_idx;
  }

  // update the result with our column_views
  assemble_result.column_views = std::move(column_views);
}

}  // namespace

/**
 * @copydoc spark_rapids_jni::shuffle_assemble
 */
shuffle_assemble_result shuffle_assemble(shuffle_split_metadata const& metadata,
                                         cudf::device_span<uint8_t const> partitions,
                                         cudf::device_span<size_t const> _partition_offsets,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  auto temp_mr = cudf::get_current_device_resource_ref();

  // trim off trailing empty partitions to handle ambiguities with copying offsets
  CUDF_EXPECTS(_partition_offsets.size() > 0,
               "Encountered an invalid offset buffer in shuffle_assemble");
  auto iter = thrust::make_transform_iterator(
    thrust::counting_iterator(size_t{0}),
    cuda::proclaim_return_type<size_t>(
      [partitions, _partition_offsets] __device__(size_t pindex) -> size_t {
        partition_header const* const pheader =
          reinterpret_cast<partition_header const*>(partitions.data() + _partition_offsets[pindex]);
        return cudf::hashing::detail::swap_endian(pheader->num_rows) > 0 ? pindex + 1 : 0;
      }));
  size_t const num_partitions_raw = thrust::reduce(rmm::exec_policy(stream, temp_mr),
                                                   iter,
                                                   iter + (_partition_offsets.size() - 1),
                                                   size_t{0},
                                                   cuda::maximum<size_t>{});
  size_t const num_partitions     = num_partitions_raw == 0 ? 1 : num_partitions_raw;

  // if the input is empty, allocate minimal buffer and create column_views
  if (num_partitions_raw == 0) {
    auto empty_assemble_data = create_empty_assemble_data(metadata.col_info);

    // Calculate total buffer size needed
    auto [validity_sizes, offsets_sizes, data_sizes] =
      calculate_empty_buffer_sizes(metadata.col_info, empty_assemble_data);
    size_t total_size = 0;
    for (size_t i = 0; i < validity_sizes.size(); i++) {
      total_size += validity_sizes[i] + offsets_sizes[i] + data_sizes[i];
    }

    // Allocate minimal buffer
    rmm::device_buffer shared_buffer(total_size, stream, mr);
    uint8_t* buffer_base = static_cast<uint8_t*>(shared_buffer.data());

    // Initialize buffers appropriately for empty columns
    initialize_empty_buffers(buffer_base, total_size, stream);

    // Create buffer slices
    std::vector<buffer_slice> buffer_slices =
      create_empty_buffer_slices(metadata.col_info, empty_assemble_data, buffer_base);

    shuffle_assemble_result result(std::move(shared_buffer), std::move(buffer_slices));

    // Use build_table to create column_views
    build_table(metadata.col_info, empty_assemble_data, result, stream, mr);

    return result;
  }

  cudf::device_span<size_t const> partition_offsets{_partition_offsets.data(), num_partitions + 1};

  // generate the info structs representing the flattened column hierarchy. the total number of
  // assembled rows, null counts, etc
  auto [column_info, column_instance_info, per_partition_metadata_size] =
    assemble_build_column_info(metadata, partitions, partition_offsets, stream, temp_mr);
  auto h_column_info = cudf::detail::make_std_vector(column_info, stream);

  // generate the shared buffer allocation with buffer slices and copy batches
  auto [assemble_result, batches] = assemble_build_buffers(column_info,
                                                           h_column_info,
                                                           column_instance_info,
                                                           partitions,
                                                           partition_offsets,
                                                           per_partition_metadata_size,
                                                           stream,
                                                           mr);

  // copy the data. also updates valid_count in column_info
  assemble_copy(batches, column_info, h_column_info, stream);

  // the gpu is synchronized at this point. a hypothetical optimization here would be to
  // not synchronize at the end of assemble_copy, which would mean the valid_count field
  // in h_column_info would not be updated yet, and poke those values specifically after
  // build_table completes.  If we are dealing with very large schemas, it is possible that
  // there's a decent chunk of time spend on the cpu in build_table that could be overlapped
  // with the the gpu work being done in assemble_copy.

  // build the final shuffle assemble result while the gpu is performing the copy
  build_table(metadata.col_info, h_column_info, assemble_result, stream, mr);
  return std::move(assemble_result);
}

}  // namespace spark_rapids_jni
