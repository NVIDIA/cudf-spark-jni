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
#include "hash/hash.hpp"
#include "hyper_log_log_plus_plus.hpp"
#include "hyper_log_log_plus_plus_const.hpp"
#include "utilities/iterator.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/detail/hyperloglog/finalizer.cuh>
#include <cuda/atomic>
#include <cuda/std/__algorithm/min.h>  // TODO #include <cuda/std/algorithm> once available
#include <cuda/std/bit>
#include <cuda/stream>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace spark_rapids_jni {

namespace {

/**
 * @brief The max precision that will leverage shared memory to do the reduction. If the precision
 * is bigger than this value, then leverage global memory to do the reduction to avoid the
 * limitation of shared memory.
 */
constexpr int MAX_SHARED_MEM_PRECISION = 12;

/**
 * @brief The seed used for the XXHash64 hash function.
 * It's consistent with Spark
 */
constexpr int64_t SEED = 42L;

/**
 * @brief 6 binary MASK bits: 111-111
 */
constexpr uint64_t MASK = (1L << REGISTER_VALUE_BITS) - 1L;

/**
 * @brief The maximum precision that can be used for the HLLPP algorithm.
 * If input precision is bigger than 18, then use 18.
 */
constexpr int MAX_PRECISION = 18;

/**
 * @brief Get register value from a long which contains 10 register values,
 * each register value in long is 6 bits.
 */
__device__ inline int get_register_value(int64_t const ten_registers, int reg_idx)
{
  auto const shift_bits = REGISTER_VALUE_BITS * reg_idx;
  auto const shift_mask = MASK << shift_bits;
  auto const v          = (ten_registers & shift_mask) >> shift_bits;
  return static_cast<int>(v);
}

/**
 * @brief Computes HLLPP sketches(register values) from hash values and
 * partially merge the sketches.
 *
 * Tried to use `reduce_by_key`, but it uses too much of memory, so give up using `reduce_by_key`.
 * More details:
 * `reduce_by_key` uses num_rows_input intermidate cache:
 * https://github.com/NVIDIA/thrust/blob/2.1.0/thrust/system/detail/generic/reduce_by_key.inl#L112
 * // scan the values by flag
 * thrust::detail::temporary_array<ValueType,ExecutionPolicy>
 * scanned_values(exec, n);
 * Each sketch contains multiple integers, by default 512 integers(precision is
 * 9), num_rows_input * 512 is huge.
 *
 * This function uses a differrent approach to use less intermidate cache.
 * It uses 2 phase merges: partial merge and final merge
 *
 * This function splits input into multiple segments with each segment has
 * num_hashs_per_thread items. The input is sorted by group labels, each segment
 * contains one or more consecutive groups. Each thread handles one segment with
 * num_hashs_per_thread items in it:
 * - Scan all the items in the segment, update the max value.
 * - Output max value into registers_output_cache for the previous group when
 * meets a new group.
 * - Output max value into registers_thread_cache when reach the last item in
 * the segment.
 *
 * In this way, we can save memory usage, cache less intermidate sketches
 * (num_hashs / num_hashs_per_thread) sketches.
 * num_threads = div_round_up(num_hashs, num_hashs_per_thread).
 *
 * Note: Must exclude null hash values from computing HLLPP sketches.
 *
 * e.g.: num_registers_per_sketch = 512 and num_hashs_per_thread = 4;
 *
 * Input is hashs, compute and get pair: register index -> register value
 *
 *   reg_index, reg_value, group_label
 * [
 * ---------- segment 0 begin --------------------------
 *    (0,            1),          g0
 *    (0,            2),          g0
 * // meets new group g1, save result for group g0 into registers_output_cache
 *    (1,            1),          g1
 * // outputs result at segemnt end for this thread to registers_thread_cache
 *    (1,            9),          g1
 * ---------- segment 1 begin --------------------------
 *    (1,            1),          g1
 *    (1,            1),          g1
 *    (1,            5),          g1
 * // outputs result at segemnt end for this thread to registers_thread_cache
 *    (1,            1),          g1
 * ---------- segment 2 begin --------------------------
 *    (1,            1),          g1
 *    (1,            1),          g1
 *    (1,            8),          g1
 * // outputs result at segemnt end for this thread to registers_thread_cache
 * // assumes meets new group when at the end, save to registers_output_cache
 *    (1,            1),          g1
 * ]
 * Output e.g.:
 *
 * group_labels_thread_cache:
 * [
 *   g1
 *   g1
 *   g1
 * ]
 * Has num_threads rows.
 *
 * registers_thread_cache:
 * [
 *    512 values: [0, 9, 0, ... ] // register values for group 1
 *    512 values: [0, 5, 0, ... ] // register values for group 1
 *    512 values: [0, 8, 0, ... ] // register values for group 1
 * ]
 * Has num_threads rows, each row is corresponding to
 * `group_labels_thread_cache`
 *
 * registers_output_cache:
 * [
 *    512 values: [2, 0, 0, ... ] // register values for group 0
 *    512 values: [0, 8, 0, ... ] // register values for group 1
 * ]
 * Has num_groups rows.
 *
 * The next kernel will merge the registers_output_cache and
 * registers_thread_cache and get the final result.
 */
template <int block_size, int num_hashs_per_thread>
__launch_bounds__(block_size) CUDF_KERNEL void partial_group_sketches_from_hashs_kernel(
  cudf::column_device_view hashs,
  cudf::device_span<cudf::size_type const> group_labels,
  int64_t const precision,                          // num of bits for register addressing, e.g.: 9
  int* const registers_output_cache,                // num is num_groups * num_registers_per_sketch
  int* const registers_thread_cache,                // num is num_threads * num_registers_per_sketch
  cudf::size_type* const group_labels_thread_cache  // save the group labels for each thread
)
{
  auto const tid          = cudf::detail::grid_1d::global_thread_id();
  int64_t const num_hashs = hashs.size();
  if (tid * num_hashs_per_thread >= hashs.size()) { return; }

  // 2^precision = num_registers_per_sketch
  int64_t num_registers_per_sketch = 1L << precision;
  // e.g.: integer in binary: 1 0000 0000
  uint64_t const w_padding = 1ULL << (precision - 1);
  // e.g.: 64 - 9 = 55
  int const idx_shift = 64 - precision;

  auto const hash_first = tid * num_hashs_per_thread;
  auto const hash_end   = cuda::std::min((tid + 1) * num_hashs_per_thread, num_hashs);

  // init sketches for each thread
  int* const sketch_ptr = registers_thread_cache + tid * num_registers_per_sketch;
  memset(sketch_ptr, 0, num_registers_per_sketch * sizeof(int));

  cudf::size_type prev_group = group_labels[hash_first];
  for (auto hash_idx = hash_first; hash_idx < hash_end; hash_idx++) {
    cudf::size_type curr_group = group_labels[hash_idx];

    int reg_idx = 0;  // init value for null hash
    int reg_v   = 0;  // init value for null hash
    if (!hashs.is_null(hash_idx)) {
      // cast to unsigned, then >> will shift without preserve the sign bit.
      uint64_t const hash = static_cast<uint64_t>(hashs.element<int64_t>(hash_idx));
      reg_idx             = hash >> idx_shift;
      // get the leading zeros
      reg_v = static_cast<int>(cuda::std::countl_zero((hash << precision) | w_padding) + 1ULL);
    }

    if (curr_group == prev_group) {
      // still in the same group, update the max value
      if (reg_v > sketch_ptr[reg_idx]) { sketch_ptr[reg_idx] = reg_v; }
    } else {
      // meets new group, save output for the previous group and reset
      memcpy(registers_output_cache + prev_group * num_registers_per_sketch,
             sketch_ptr,
             num_registers_per_sketch * sizeof(int));
      memset(sketch_ptr, 0, num_registers_per_sketch * sizeof(int));

      // save the result for current group
      sketch_ptr[reg_idx] = reg_v;
    }

    if (hash_idx == hash_end - 1) {
      // meets the last hash in the segment
      if (hash_idx == num_hashs - 1) {
        // meets the last segment, special logic: assume meets new group
        memcpy(registers_output_cache + curr_group * num_registers_per_sketch,
               sketch_ptr,
               num_registers_per_sketch * sizeof(int));
      } else if (curr_group != group_labels[hash_idx + 1]) {
        // not the last segment, probe one item forward.
        // meets a new group by checking the next item in the next segment
        memcpy(registers_output_cache + curr_group * num_registers_per_sketch,
               sketch_ptr,
               num_registers_per_sketch * sizeof(int));
      }
    }

    prev_group = curr_group;
  }

  // save the group label for this thread
  group_labels_thread_cache[tid] = group_labels[hash_end - 1];
}

/*
 * @brief Merge registers_thread_cache into registers_output_cache, both of them
 * are produced in the above kernel. Merge sketches vertically.
 *
 * For each register index, starts a thread to merge registers in
 * registers_thread_cache to registers_output_cache. num_threads =
 * num_registers_per_sketch.
 *
 * Input e.g.:
 *
 * group_labels_thread_cache:
 * [
 *   g0
 *   g0
 *   g1
 *   ...
 *   gN
 * ]
 * Has num_threads rows.
 *
 * registers_thread_cache:
 * [
 *    r0_g0, r1_g0, r2_g0, r3_g0, ... , r511_g0 // register values for group 0
 *    r0_g0, r1_g0, r2_g0, r3_g0, ... , r511_g0 // register values for group 0
 *    r0_g1, r1_g1, r2_g1, r3_g1, ... , r511_g1 // register values for group 1
 *    ...
 *    r0_gN, r1_gN, r2_gN, r3_gN, ... , r511_gN // register values for group N
 * ]
 * Has num_threads rows, each row is corresponding to
 * `group_labels_thread_cache`
 *
 * registers_output_cache:
 * [
 *    r0_g0, r1_g0, r2_g0, r3_g0, ... , r511_g0 // register values for group 0
 *    r0_g1, r1_g1, r2_g1, r3_g1, ... , r511_g1 // register values for group 1
 *    ...
 *    r0_gN, r1_gN, r2_gN, r3_gN, ... , r511_gN // register values for group N
 * ]
 * registers_output_cache has num_groups rows.
 *
 * For each thread, scan from the first register to the last register, find the
 * max value in the same group, and then update to registers_output_cache
 */
template <int block_size>
__launch_bounds__(block_size) CUDF_KERNEL
  void merge_sketches_vertically(int64_t num_sketches,
                                 int64_t num_registers_per_sketch,
                                 int* const registers_output_cache,
                                 int const* const registers_thread_cache,
                                 cudf::size_type const* const group_labels_thread_cache)
{
  __shared__ int8_t shared_data[block_size];
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_registers_per_sketch) { return; }
  int shared_idx = tid % block_size;

  // register idx is tid
  shared_data[shared_idx] = static_cast<int8_t>(0);
  int prev_group          = group_labels_thread_cache[0];
  for (auto i = 0; i < num_sketches; i++) {
    int curr_group = group_labels_thread_cache[i];
    int8_t curr_reg_v =
      static_cast<int8_t>(registers_thread_cache[i * num_registers_per_sketch + tid]);
    if (curr_group == prev_group) {
      if (curr_reg_v > shared_data[shared_idx]) { shared_data[shared_idx] = curr_reg_v; }
    } else {
      // meets a new group, store the result for previous group
      int64_t result_reg_idx = prev_group * num_registers_per_sketch + tid;
      int result_curr_reg_v  = registers_output_cache[result_reg_idx];
      if (shared_data[shared_idx] > result_curr_reg_v) {
        registers_output_cache[result_reg_idx] = shared_data[shared_idx];
      }

      shared_data[shared_idx] = curr_reg_v;
    }
    prev_group = curr_group;
  }

  // handles the last register in this thread
  int64_t reg_idx = prev_group * num_registers_per_sketch + tid;
  int curr_reg_v  = registers_output_cache[reg_idx];
  if (shared_data[shared_idx] > curr_reg_v) {
    registers_output_cache[reg_idx] = shared_data[shared_idx];
  }
}

/**
 * @brief Compact register values, compact 10 registers values
 * (each register value is 6 bits) into a long.
 * This is consistent with Spark.
 * Output: long columns which will be composed into a struct column
 *
 * Number of threads is num_groups * num_long_cols.
 *
 * e.g., num_registers_per_sketch is 512(precision is 9):
 * Input:
 * registers_output_cache:
 * [
 *    r0_g0, r1_g0, r2_g0, r3_g0, ... , r511_g0 // register values for group 0
 *    r0_g1, r1_g1, r2_g1, r3_g1, ... , r511_g1 // register values for group 1
 *    ...
 *    r0_gN, r1_gN, r2_gN, r3_gN, ... , r511_gN // register values for group N
 * ]
 * Has num_groups rows.
 *
 * Output:
 * 52 long columns
 *
 * e.g.: r0 to r9 integers are all: 00000000-00000000-00000000-00100001, tailing
 * 6 bits: 100-001 Compact to one long is:
 * 100001-100001-100001-100001-100001-100001-100001-100001-100001-100001
 */
template <int block_size>
__launch_bounds__(block_size) CUDF_KERNEL
  void compact_kernel(int64_t const num_groups,
                      int64_t const num_registers_per_sketch,
                      cudf::device_span<int64_t*> sketches_output,
                      // num_groups * num_registers_per_sketch integers
                      cudf::device_span<int> registers_output_cache)
{
  int64_t const tid           = cudf::detail::grid_1d::global_thread_id();
  int64_t const num_long_cols = num_registers_per_sketch / REGISTERS_PER_LONG + 1;
  if (tid >= num_groups * num_long_cols) { return; }

  int64_t const group_idx = tid / num_long_cols;
  int64_t const long_idx  = tid % num_long_cols;

  int64_t const reg_begin_idx =
    group_idx * num_registers_per_sketch + long_idx * REGISTERS_PER_LONG;
  int64_t num_regs = REGISTERS_PER_LONG;
  if (long_idx == num_long_cols - 1) { num_regs = num_registers_per_sketch % REGISTERS_PER_LONG; }

  int64_t ten_registers = 0;
  for (auto i = 0; i < num_regs; i++) {
    int64_t reg_v = registers_output_cache[reg_begin_idx + i];
    int64_t tmp   = reg_v << (REGISTER_VALUE_BITS * i);
    ten_registers |= tmp;
  }

  sketches_output[long_idx][group_idx] = ten_registers;
}

std::unique_ptr<cudf::column> group_hllpp(cudf::column_view const& input,
                                          int64_t const num_groups,
                                          cudf::device_span<cudf::size_type const> group_labels,
                                          int64_t const precision,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  int64_t num_registers_per_sketch   = 1 << precision;
  constexpr int64_t block_size       = 256;
  constexpr int num_hashs_per_thread = 256;  // handles 256 items per thread
  int64_t num_threads_partial_kernel =
    cudf::util::div_rounding_up_safe(input.size(), num_hashs_per_thread);

  auto const default_mr = cudf::get_current_device_resource_ref();
  auto sketches_output =
    rmm::device_uvector<int32_t>(num_groups * num_registers_per_sketch, stream, default_mr);

  {  // add this block to release `registers_thread_cache` and
    // `group_labels_thread_cache`
    auto registers_thread_cache = rmm::device_uvector<int32_t>(
      num_threads_partial_kernel * num_registers_per_sketch, stream, default_mr);
    auto group_labels_thread_cache =
      rmm::device_uvector<int32_t>(num_threads_partial_kernel, stream, default_mr);

    {  // add this block to release `hash_col`
      // 1. compute all the hashs
      auto input_table_view = cudf::table_view{{input}};
      auto hash_col         = xxhash64(input_table_view, SEED, stream, default_mr);
      hash_col->set_null_mask(cudf::copy_bitmask(input, stream, default_mr), input.null_count());
      auto d_hashs = cudf::column_device_view::create(hash_col->view(), stream, default_mr);

      // 2. execute partial group by
      int64_t num_blocks_p1 =
        cudf::util::div_rounding_up_safe(num_threads_partial_kernel, block_size);
      partial_group_sketches_from_hashs_kernel<block_size, num_hashs_per_thread>
        <<<num_blocks_p1, block_size, 0, stream.get()>>>(*d_hashs,
                                                         group_labels,
                                                         precision,
                                                         sketches_output.begin(),
                                                         registers_thread_cache.begin(),
                                                         group_labels_thread_cache.begin());
    }
    // 3. merge the intermidate result
    auto num_merge_threads = num_registers_per_sketch;
    auto num_merge_blocks  = cudf::util::div_rounding_up_safe(num_merge_threads, block_size);
    merge_sketches_vertically<block_size>
      <<<num_merge_blocks, block_size, block_size, stream.get()>>>(
        num_threads_partial_kernel,  // num_sketches
        num_registers_per_sketch,
        sketches_output.begin(),
        registers_thread_cache.begin(),
        group_labels_thread_cache.begin());
  }

  // 4. create output columns
  auto num_long_cols      = num_registers_per_sketch / REGISTERS_PER_LONG + 1;
  auto const results_iter = spark_rapids_jni::util::make_counting_transform_iterator(0, [&](int i) {
    return cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64}, num_groups, cudf::mask_state::UNALLOCATED, stream, mr);
  });
  auto children =
    std::vector<std::unique_ptr<cudf::column>>(results_iter, results_iter + num_long_cols);

  auto host_results_pointer_iter =
    thrust::make_transform_iterator(children.begin(), [](auto const& results_column) {
      return results_column->mutable_view().template data<int64_t>();
    });
  auto host_results_pointers =
    std::vector<int64_t*>(host_results_pointer_iter, host_results_pointer_iter + children.size());
  auto d_results = cudf::detail::make_device_uvector(host_results_pointers, stream, default_mr);

  auto result = cudf::make_structs_column(num_groups,
                                          std::move(children),
                                          0,                     // null count
                                          rmm::device_buffer{},  // null mask
                                          stream);

  // 5. compact sketches
  auto num_phase3_threads = num_groups * num_long_cols;
  auto num_phase3_blocks  = cudf::util::div_rounding_up_safe(num_phase3_threads, block_size);
  compact_kernel<block_size><<<num_phase3_blocks, block_size, 0, stream.get()>>>(
    num_groups, num_registers_per_sketch, d_results, sketches_output);

  return result;
}

/**
 * @brief Partial groups sketches in long columns, similar to
 * `partial_group_sketches_from_hashs_kernel` It split longs into segments with
 * each has `num_longs_per_threads` elements e.g.: num_registers_per_sketch =
 * 512. Each sketch uses 52 (512 / 10 + 1) longs.
 *
 * Input:
 *           col_0  col_1      col_51
 * sketch_0: long,  long, ..., long
 * sketch_1: long,  long, ..., long
 * sketch_2: long,  long, ..., long
 *
 * num_threads = 52 * div_round_up(num_sketches_input, num_longs_per_threads)
 * Each thread scans and merge num_longs_per_threads longs,
 * and output the max register value when meets a new group.
 * For the last long in a thread, outputs the result into
 * `registers_thread_cache`.
 *
 * By split inputs into segments like `partial_group_sketches_from_hashs_kernel`
 * and do partial merge, it will use less memory. Then the kernel
 * merge_sketches_vertically can be used to merge the intermidate results:
 * registers_output_cache, registers_thread_cache
 */
template <int block_size, int num_longs_per_threads>
__launch_bounds__(block_size) CUDF_KERNEL
  void partial_group_long_sketches_kernel(cudf::device_span<int64_t const*> sketches_input,
                                          int64_t const num_sketches_input,
                                          int64_t const num_threads_per_col,
                                          int64_t const num_registers_per_sketch,
                                          int64_t const num_groups,
                                          cudf::device_span<cudf::size_type const> group_labels,
                                          // num_groups * num_registers_per_sketch integers
                                          int* const registers_output_cache,
                                          // num_threads * num_registers_per_sketch integers
                                          int* const registers_thread_cache,
                                          // num_threads integers
                                          cudf::size_type* const group_labels_thread_cache)
{
  auto const tid           = cudf::detail::grid_1d::global_thread_id();
  auto const num_long_cols = sketches_input.size();
  if (tid >= num_threads_per_col * num_long_cols) { return; }

  auto const long_idx            = tid / num_threads_per_col;
  auto const thread_idx_in_cols  = tid % num_threads_per_col;
  int64_t const* const longs_ptr = sketches_input[long_idx];

  int* const registers_thread_ptr =
    registers_thread_cache + thread_idx_in_cols * num_registers_per_sketch;

  auto const sketch_first = thread_idx_in_cols * num_longs_per_threads;
  auto const sketch_end = cuda::std::min(sketch_first + num_longs_per_threads, num_sketches_input);

  int num_regs = REGISTERS_PER_LONG;
  if (long_idx == num_long_cols - 1) { num_regs = num_registers_per_sketch % REGISTERS_PER_LONG; }

  for (auto i = 0; i < num_regs; i++) {
    cudf::size_type prev_group = group_labels[sketch_first];
    int max_reg_v              = 0;
    int reg_idx_in_sketch      = long_idx * REGISTERS_PER_LONG + i;
    for (auto sketch_idx = sketch_first; sketch_idx < sketch_end; sketch_idx++) {
      cudf::size_type curr_group = group_labels[sketch_idx];
      int curr_reg_v             = get_register_value(longs_ptr[sketch_idx], i);
      if (curr_group == prev_group) {
        // still in the same group, update the max value
        if (curr_reg_v > max_reg_v) { max_reg_v = curr_reg_v; }
      } else {
        // meets new group, save output for the previous group
        int64_t output_idx_prev = num_registers_per_sketch * prev_group + reg_idx_in_sketch;
        registers_output_cache[output_idx_prev] = max_reg_v;

        // reset
        max_reg_v = curr_reg_v;
      }

      if (sketch_idx == sketch_end - 1) {
        // last item in the segment
        int64_t output_idx_curr = num_registers_per_sketch * curr_group + reg_idx_in_sketch;
        if (sketch_idx == num_sketches_input - 1) {
          // last segment
          registers_output_cache[output_idx_curr] = max_reg_v;
          max_reg_v                               = curr_reg_v;
        } else {
          if (curr_group != group_labels[sketch_idx + 1]) {
            // look the first item in the next segment
            registers_output_cache[output_idx_curr] = max_reg_v;
            max_reg_v                               = curr_reg_v;
          }
        }
      }

      prev_group = curr_group;
    }

    // For each thread, output current max value
    registers_thread_ptr[reg_idx_in_sketch] = max_reg_v;
  }

  if (long_idx == 0) {
    group_labels_thread_cache[thread_idx_in_cols] = group_labels[sketch_end - 1];
  }
}

/**
 * @brief Merge for struct<long, ..., long> column. Each long contains 10
 * register values. Merge all rows in the same group.
 */
std::unique_ptr<cudf::column> group_merge_hllpp(
  cudf::column_view const& hll_input,  // struct<long, ..., long> column
  int64_t const num_groups,
  cudf::device_span<cudf::size_type const> group_labels,
  int64_t const precision,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  int64_t num_registers_per_sketch        = 1 << precision;
  int64_t const num_sketches              = hll_input.size();
  int64_t const num_long_cols             = num_registers_per_sketch / REGISTERS_PER_LONG + 1;
  constexpr int64_t num_longs_per_threads = 256;
  constexpr int64_t block_size            = 256;

  int64_t num_threads_per_col_phase1 =
    cudf::util::div_rounding_up_safe(num_sketches, num_longs_per_threads);
  int64_t num_threads_phase1 = num_threads_per_col_phase1 * num_long_cols;
  int64_t num_blocks         = cudf::util::div_rounding_up_safe(num_threads_phase1, block_size);
  auto const default_mr      = cudf::get_current_device_resource_ref();
  auto registers_output_cache =
    rmm::device_uvector<int32_t>(num_registers_per_sketch * num_groups, stream, default_mr);
  {
    auto registers_thread_cache = rmm::device_uvector<int32_t>(
      num_registers_per_sketch * num_threads_phase1, stream, default_mr);
    auto group_labels_thread_cache =
      rmm::device_uvector<int32_t>(num_threads_per_col_phase1, stream, default_mr);

    cudf::structs_column_view scv(hll_input);
    auto const input_iter = spark_rapids_jni::util::make_counting_transform_iterator(
      0, [&](int i) { return scv.get_sliced_child(i, stream).begin<int64_t>(); });
    auto input_cols = std::vector<int64_t const*>(input_iter, input_iter + num_long_cols);
    auto d_inputs   = cudf::detail::make_device_uvector(input_cols, stream, default_mr);
    // 1st kernel: partially group
    partial_group_long_sketches_kernel<block_size, num_longs_per_threads>
      <<<num_blocks, block_size, 0, stream.get()>>>(d_inputs,
                                                    num_sketches,
                                                    num_threads_per_col_phase1,
                                                    num_registers_per_sketch,
                                                    num_groups,
                                                    group_labels,
                                                    registers_output_cache.begin(),
                                                    registers_thread_cache.begin(),
                                                    group_labels_thread_cache.begin());
    auto const num_phase2_threads = num_registers_per_sketch;
    auto const num_phase2_blocks = cudf::util::div_rounding_up_safe(num_phase2_threads, block_size);
    // 2nd kernel: vertical merge
    merge_sketches_vertically<block_size>
      <<<num_phase2_blocks, block_size, block_size, stream.get()>>>(
        num_threads_per_col_phase1,  // num_sketches
        num_registers_per_sketch,
        registers_output_cache.begin(),
        registers_thread_cache.begin(),
        group_labels_thread_cache.begin());
  }

  // create output columns
  auto const results_iter = spark_rapids_jni::util::make_counting_transform_iterator(0, [&](int i) {
    return cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64}, num_groups, cudf::mask_state::UNALLOCATED, stream, mr);
  });
  auto results =
    std::vector<std::unique_ptr<cudf::column>>(results_iter, results_iter + num_long_cols);

  auto host_results_pointer_iter =
    thrust::make_transform_iterator(results.begin(), [](auto const& results_column) {
      return results_column->mutable_view().template data<int64_t>();
    });
  auto host_results_pointers =
    std::vector<int64_t*>(host_results_pointer_iter, host_results_pointer_iter + results.size());
  auto d_sketches_output =
    cudf::detail::make_device_uvector(host_results_pointers, stream, default_mr);

  // 3rd kernel: compact
  auto num_phase3_threads = num_groups * num_long_cols;
  auto num_phase3_blocks  = cudf::util::div_rounding_up_safe(num_phase3_threads, block_size);
  compact_kernel<block_size><<<num_phase3_blocks, block_size, 0, stream.get()>>>(
    num_groups, num_registers_per_sketch, d_sketches_output, registers_output_cache);

  return make_structs_column(num_groups, std::move(results), 0, rmm::device_buffer{});
}

/**
 * @brief Launch only 1 block, uses max 1M(2^18 *sizeof(int)) shared memory.
 * For each hash, get a pair: (register index, register value).
 * Use shared memory to speedup the fetch max atomic operation.
 */
template <int block_size>
__launch_bounds__(block_size) CUDF_KERNEL
  void reduce_hllpp_kernel(cudf::column_device_view hashs,
                           cudf::device_span<int64_t*> output,
                           int precision,
                           int32_t* mem_cache)
{
  extern __shared__ int32_t shared_mem_cache[];
  int32_t* cache = mem_cache != nullptr ? mem_cache : shared_mem_cache;

  auto const tid                          = cudf::detail::grid_1d::global_thread_id();
  auto const num_hashs                    = hashs.size();
  uint64_t const num_registers_per_sketch = 1L << precision;
  int const idx_shift                     = 64 - precision;
  uint64_t const w_padding                = 1ULL << (precision - 1);

  // init tmp data
  for (int i = tid; i < num_registers_per_sketch; i += block_size) {
    cache[i] = 0;
  }
  __syncthreads();

  // update max reg value for the reg index
  for (int i = tid; i < num_hashs; i += block_size) {
    int reg_idx = 0;  // init value for null hash
    int reg_v   = 0;  // init value for null hash
    if (!hashs.is_null(i)) {
      // cast to unsigned, then >> will shift without preserve the sign bit.
      uint64_t const hash = static_cast<uint64_t>(hashs.element<int64_t>(i));
      reg_idx             = hash >> idx_shift;
      // get the leading zeros
      reg_v = static_cast<int>(cuda::std::countl_zero((hash << precision) | w_padding) + 1ULL);
    }

    cuda::atomic_ref<int32_t, cuda::thread_scope_block> register_ref(cache[reg_idx]);
    register_ref.fetch_max(reg_v, cuda::memory_order_relaxed);
  }
  __syncthreads();

  // compact from register values (int array) to long array
  // each long holds 10 integers, note reg value < 64 which means the bits from
  // 7 to highest are all 0.
  for (int i = tid; i * REGISTERS_PER_LONG < num_registers_per_sketch; i += block_size) {
    int start = i * REGISTERS_PER_LONG;
    int end   = (i + 1) * REGISTERS_PER_LONG;
    if (end > num_registers_per_sketch) { end = num_registers_per_sketch; }

    int64_t ret = 0;
    for (int j = 0; j < end - start; j++) {
      int shift   = j * REGISTER_VALUE_BITS;
      int64_t reg = cache[start + j];
      ret |= (reg << shift);
    }

    output[i][0] = ret;
  }
}

std::unique_ptr<cudf::scalar> reduce_hllpp(cudf::column_view const& input,
                                           int64_t const precision,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  int64_t num_registers_per_sketch = 1L << precision;
  // 1. compute all the hashs
  auto input_table_view = cudf::table_view{{input}};
  auto const default_mr = cudf::get_current_device_resource_ref();
  auto hash_col         = xxhash64(input_table_view, SEED, stream, default_mr);
  hash_col->set_null_mask(cudf::copy_bitmask(input, stream, default_mr), input.null_count());
  auto d_hashs = cudf::column_device_view::create(hash_col->view(), stream, default_mr);

  // 2. generate long columns, the size of each long column is 1
  auto num_long_cols      = num_registers_per_sketch / REGISTERS_PER_LONG + 1;
  auto const results_iter = spark_rapids_jni::util::make_counting_transform_iterator(0, [&](int i) {
    return cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                     1 /**num_groups*/,
                                     cudf::mask_state::UNALLOCATED,
                                     stream,
                                     mr);
  });
  auto children =
    std::vector<std::unique_ptr<cudf::column>>(results_iter, results_iter + num_long_cols);

  auto host_results_pointer_iter =
    thrust::make_transform_iterator(children.begin(), [](auto const& results_column) {
      return results_column->mutable_view().template data<int64_t>();
    });
  auto host_results_pointers =
    std::vector<int64_t*>(host_results_pointer_iter, host_results_pointer_iter + children.size());
  auto d_results = cudf::detail::make_device_uvector(host_results_pointers, stream, default_mr);

  // 2. reduce and generate compacted long values
  constexpr int64_t block_size = 256;
  if (precision <= MAX_SHARED_MEM_PRECISION) {
    // use shared memory, max shared memory is 2^12 * 4 = 16M
    auto shared_mem_size = num_registers_per_sketch * sizeof(int32_t);
    reduce_hllpp_kernel<block_size>
      <<<1, block_size, shared_mem_size, stream.get()>>>(*d_hashs, d_results, precision, nullptr);
  } else {
    // use global memory because shared memory may be not enough
    auto mem_cache = rmm::device_uvector<int32_t>(num_registers_per_sketch, stream, default_mr);
    reduce_hllpp_kernel<block_size>
      <<<1, block_size, 0, stream.get()>>>(*d_hashs, d_results, precision, mem_cache.data());
  }

  // 3. create struct scalar
  return std::make_unique<cudf::struct_scalar>(cudf::table{std::move(children)}, true, stream, mr);
}

template <int block_size>
__launch_bounds__(block_size) CUDF_KERNEL
  void reduce_merge_hll_kernel_vertically(cudf::device_span<int64_t const*> sketch_longs,
                                          cudf::size_type num_sketches,
                                          int num_registers_per_sketch,
                                          int* const output)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_registers_per_sketch) { return; }
  auto long_idx        = tid / REGISTERS_PER_LONG;
  auto reg_idx_in_long = tid % REGISTERS_PER_LONG;
  int max              = 0;
  for (auto row_idx = 0; row_idx < num_sketches; row_idx++) {
    int reg_v = get_register_value(sketch_longs[long_idx][row_idx], reg_idx_in_long);
    if (reg_v > max) { max = reg_v; }
  }
  output[tid] = max;
}

std::unique_ptr<cudf::scalar> reduce_merge_hllpp(cudf::column_view const& input,
                                                 int64_t const precision,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  // create device input
  int64_t num_registers_per_sketch = 1 << precision;
  auto num_long_cols               = num_registers_per_sketch / REGISTERS_PER_LONG + 1;
  cudf::structs_column_view scv(input);
  auto const input_iter = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [&](int i) { return scv.get_sliced_child(i, stream).begin<int64_t>(); });
  auto input_cols       = std::vector<int64_t const*>(input_iter, input_iter + num_long_cols);
  auto const default_mr = cudf::get_current_device_resource_ref();
  auto d_inputs         = cudf::detail::make_device_uvector(input_cols, stream, default_mr);

  // create one row output
  auto const results_iter = spark_rapids_jni::util::make_counting_transform_iterator(0, [&](int i) {
    return cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                     1 /** num_rows */,
                                     cudf::mask_state::UNALLOCATED,
                                     stream,
                                     mr);
  });
  auto children =
    std::vector<std::unique_ptr<cudf::column>>(results_iter, results_iter + num_long_cols);

  auto host_results_pointer_iter =
    thrust::make_transform_iterator(children.begin(), [](auto const& results_column) {
      return results_column->mutable_view().template data<int64_t>();
    });
  auto host_results_pointers =
    std::vector<int64_t*>(host_results_pointer_iter, host_results_pointer_iter + children.size());
  auto d_results = cudf::detail::make_device_uvector(host_results_pointers, stream, default_mr);

  // execute merge kernel
  auto num_threads             = num_registers_per_sketch;
  constexpr int64_t block_size = 256;
  auto num_blocks              = cudf::util::div_rounding_up_safe(num_threads, block_size);
  auto output_cache = rmm::device_uvector<int32_t>(num_registers_per_sketch, stream, default_mr);
  reduce_merge_hll_kernel_vertically<block_size><<<num_blocks, block_size, 0, stream.get()>>>(
    d_inputs, input.size(), num_registers_per_sketch, output_cache.begin());

  // compact to longs
  auto const num_compact_threads = num_long_cols;
  auto const num_compact_blocks = cudf::util::div_rounding_up_safe(num_compact_threads, block_size);
  compact_kernel<block_size><<<num_compact_blocks, block_size, 0, stream.get()>>>(
    1 /** num_groups **/, num_registers_per_sketch, d_results, output_cache);

  // create scalar
  return std::make_unique<cudf::struct_scalar>(cudf::table{std::move(children)}, true, stream, mr);
}

struct estimate_fn {
  cudf::device_span<int64_t const*> sketches;
  int64_t* out;
  int precision;

  __device__ void operator()(cudf::size_type const idx) const
  {
    auto const num_regs = 1 << precision;
    double sum          = 0;
    int zeroes          = 0;

    for (auto reg_idx = 0; reg_idx < num_regs; ++reg_idx) {
      // each long contains 10 register values
      int long_col_idx    = reg_idx / REGISTERS_PER_LONG;
      int reg_idx_in_long = reg_idx % REGISTERS_PER_LONG;
      int reg             = get_register_value(sketches[long_col_idx][idx], reg_idx_in_long);
      sum += double{1} / static_cast<double>(1ull << reg);
      zeroes += reg == 0;
    }

    auto const finalize = cuco::hyperloglog_ns::detail::finalizer(precision);
    out[idx]            = finalize(sum, zeroes);
  }
};

}  // end anonymous namespace

std::unique_ptr<cudf::column> group_hyper_log_log_plus_plus(
  cudf::column_view const& input,
  int64_t const num_groups,
  cudf::device_span<cudf::size_type const> group_labels,
  int64_t const precision,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(precision >= 4, "HyperLogLogPlusPlus requires precision bigger than 4.");
  auto adjust_precision = precision > MAX_PRECISION ? MAX_PRECISION : precision;
  return group_hllpp(input, num_groups, group_labels, adjust_precision, stream, mr);
}

std::unique_ptr<cudf::column> group_merge_hyper_log_log_plus_plus(
  cudf::column_view const& input,
  int64_t const num_groups,
  cudf::device_span<cudf::size_type const> group_labels,
  int64_t const precision,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(precision >= 4, "HyperLogLogPlusPlus requires precision bigger than 4.");
  CUDF_EXPECTS(input.type().id() == cudf::type_id::STRUCT,
               "HyperLogLogPlusPlus buffer type must be a STRUCT of long columns.");
  for (auto i = 0; i < input.num_children(); i++) {
    CUDF_EXPECTS(input.child(i).type().id() == cudf::type_id::INT64,
                 "HyperLogLogPlusPlus buffer type must be a STRUCT of long columns.");
  }
  auto adjust_precision   = precision > MAX_PRECISION ? MAX_PRECISION : precision;
  auto expected_num_longs = (1 << adjust_precision) / REGISTERS_PER_LONG + 1;
  CUDF_EXPECTS(input.num_children() == expected_num_longs,
               "The num of long columns in input is incorrect.");
  return group_merge_hllpp(input, num_groups, group_labels, adjust_precision, stream, mr);
}

std::unique_ptr<cudf::scalar> reduce_hyper_log_log_plus_plus(cudf::column_view const& input,
                                                             int64_t const precision,
                                                             cuda::stream_ref stream,
                                                             rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(precision >= 4, "HyperLogLogPlusPlus requires precision bigger than 4.");
  auto adjust_precision = precision > MAX_PRECISION ? MAX_PRECISION : precision;
  return reduce_hllpp(input, adjust_precision, stream, mr);
}

std::unique_ptr<cudf::scalar> reduce_merge_hyper_log_log_plus_plus(
  cudf::column_view const& input,
  int64_t const precision,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(precision >= 4, "HyperLogLogPlusPlus requires precision bigger than 4.");
  CUDF_EXPECTS(input.type().id() == cudf::type_id::STRUCT,
               "HyperLogLogPlusPlus buffer type must be a STRUCT of long columns.");
  for (auto i = 0; i < input.num_children(); i++) {
    CUDF_EXPECTS(input.child(i).type().id() == cudf::type_id::INT64,
                 "HyperLogLogPlusPlus buffer type must be a STRUCT of long columns.");
  }
  auto adjust_precision   = precision > MAX_PRECISION ? MAX_PRECISION : precision;
  auto expected_num_longs = (1 << adjust_precision) / REGISTERS_PER_LONG + 1;
  CUDF_EXPECTS(input.num_children() == expected_num_longs,
               "The num of long columns in input is incorrect.");
  return reduce_merge_hllpp(input, adjust_precision, stream, mr);
}

std::unique_ptr<cudf::column> estimate_from_hll_sketches(cudf::column_view const& input,
                                                         int precision,
                                                         cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(precision >= 4, "HyperLogLogPlusPlus requires precision bigger than 4.");
  CUDF_EXPECTS(input.type().id() == cudf::type_id::STRUCT,
               "HyperLogLogPlusPlus buffer type must be a STRUCT of long columns.");
  for (auto i = 0; i < input.num_children(); i++) {
    CUDF_EXPECTS(input.child(i).type().id() == cudf::type_id::INT64,
                 "HyperLogLogPlusPlus buffer type must be a STRUCT of long columns.");
  }
  auto const input_iter = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [&](int i) { return input.child(i).begin<int64_t>(); });
  auto const h_input_ptrs =
    std::vector<int64_t const*>(input_iter, input_iter + input.num_children());
  auto const default_mr = cudf::get_current_device_resource_ref();
  auto d_inputs         = cudf::detail::make_device_uvector(h_input_ptrs, stream, default_mr);
  auto result           = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, input.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  // evaluate from struct<long, ..., long>
  thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     thrust::make_counting_iterator(0),
                     input.size(),
                     estimate_fn{d_inputs, result->mutable_view().data<int64_t>(), precision});
  return result;
}

}  // namespace spark_rapids_jni
