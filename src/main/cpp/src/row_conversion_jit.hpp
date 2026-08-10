/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>
#include <vector>

namespace spark_rapids_jni {
namespace detail {

/**
 * @brief Launch a schema-specialized JIT gather kernel for the from-rows direction if available
 *
 * The kernel is compiled at runtime (NVRTC through librtcx, both already statically linked into
 * this binary) for the exact column layout: every column offset, element width and the row stride
 * are folded into the instruction stream, removing the per-element layout loads and the width
 * switch of the generic kernel. Only single-band fixed-width layouts are supported; the caller
 * must have verified that the tile march produced exactly one column band. Compiled kernels are
 * memoized in a bounded process-global cache keyed by layout and device environment. The env var
 * `SPARK_RAPIDS_ROWCONV_JIT=0` is the kill switch.
 *
 * The generated kernel addresses rows as `input + row * row_stride` with no batch adjustment, so
 * the caller must pass a single row batch starting at row 0. Nothing in the signature enforces
 * this: a multi-batch from-rows caller must launch the generic kernel instead.
 *
 * @param column_sizes per-column element widths in bytes
 * @param column_starts per-column offsets within a row plus the trailing validity offset
 * @param row_stride padded JCUDF row pitch in bytes
 * @param num_tiles number of row tiles (grid size); one thread block processes one tile
 * @param dev_tile_infos device array of `tile_info` (layout asserted at the call site)
 * @param input_data device pointer to the packed JCUDF row data
 * @param dev_output_data device array of per-column output base pointers
 * @param stream CUDA stream for the launch
 * @return true when the specialized kernel was launched; false when the JIT is disabled, the
 * layout is unsupported, or compilation failed — the caller must then launch the generic kernel
 */
bool try_jit_copy_from_rows(std::vector<cudf::size_type> const& column_sizes,
                            std::vector<cudf::size_type> const& column_starts,
                            cudf::size_type row_stride,
                            int num_tiles,
                            void const* dev_tile_infos,
                            int8_t const* input_data,
                            int8_t** dev_output_data,
                            rmm::cuda_stream_view stream);

/**
 * @brief Launch a schema-specialized JIT kernel for the to-rows direction if available
 *
 * Same contract as `try_jit_copy_from_rows` for the opposite direction: the generated kernel
 * stages one tile in shared memory with a fully unrolled column sweep and copies rows out with a
 * compile-time row width. Validity and variable-width data are handled by the unchanged incumbent
 * kernels.
 *
 * @param column_sizes per-column element widths in bytes
 * @param column_starts per-column offsets within a row plus the trailing validity offset
 * @param row_stride padded JCUDF row pitch in bytes
 * @param num_tiles number of row tiles (grid size); one thread block processes one tile
 * @param dev_tile_infos device array of `tile_info` (layout asserted at the call site)
 * @param shmem_bytes dynamic shared-memory size the tiles were shaped against
 * @param dev_input_data device array of per-column input base pointers
 * @param dev_output_data device array of per-batch output buffer pointers
 * @param dev_batch_row_boundaries device array of batch starting row numbers
 * @param stream CUDA stream for the launch
 * @return true when the specialized kernel was launched; false otherwise (see above)
 */
bool try_jit_copy_to_rows(std::vector<cudf::size_type> const& column_sizes,
                          std::vector<cudf::size_type> const& column_starts,
                          cudf::size_type row_stride,
                          int num_tiles,
                          void const* dev_tile_infos,
                          int shmem_bytes,
                          int8_t const** dev_input_data,
                          int8_t** dev_output_data,
                          cudf::size_type const* dev_batch_row_boundaries,
                          rmm::cuda_stream_view stream);

}  // namespace detail
}  // namespace spark_rapids_jni
