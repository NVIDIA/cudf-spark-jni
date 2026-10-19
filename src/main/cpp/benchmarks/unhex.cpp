/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/generate_input.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cast_string.hpp>
#include <nvbench/nvbench.cuh>

namespace {

/**
 * @brief Generate `num_rows` hex strings of `length` characters on the GPU.
 *
 * Random strings of `length / 2` bytes are hex-encoded, so `length` must be even.
 */
std::unique_ptr<cudf::column> make_hex_strings(cudf::size_type num_rows, int32_t length)
{
  data_profile const profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::UNIFORM, length / 2, length / 2)
      .no_validity()
      .cardinality(0);
  auto const bytes = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  return spark_rapids_jni::bytes_to_hex(cudf::strings_column_view{bytes->view()},
                                        cudf::get_default_stream());
}

void unhex(nvbench::state& state)
{
  auto const num_rows        = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const string_length   = static_cast<int32_t>(state.get_int64("string_length"));
  auto const long_row_length = static_cast<int32_t>(state.get_int64("long_row_length"));
  auto const invalid_percent = state.get_int64("invalid_percent");
  if (static_cast<int64_t>(num_rows) * string_length > (int64_t{1} << 30)) {
    state.skip("input exceeds 1 GiB of characters");
    return;
  }

  auto input = make_hex_strings(num_rows, string_length);
  if (invalid_percent > 0) {
    // Random bytes are almost never valid hex, so each selected row decodes to null.
    data_profile const profile =
      data_profile_builder()
        .distribution(cudf::type_id::STRING, distribution_id::UNIFORM, string_length, string_length)
        .no_validity()
        .cardinality(0);
    auto const invalid = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
    auto const select_invalid = create_random_column(
      cudf::type_id::BOOL8,
      row_count{num_rows},
      data_profile_builder().bool_probability_true(invalid_percent / 100.0).no_validity());
    input = cudf::copy_if_else(invalid->view(), input->view(), select_invalid->view());
  }
  if (long_row_length > 0) {
    // Prepend one long row to measure skewed batches.
    auto const long_row = make_hex_strings(1, long_row_length);
    input               = cudf::concatenate(std::vector{long_row->view(), input->view()});
  }

  auto const input_view = cudf::strings_column_view{input->view()};
  auto const stream     = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_element_count(input_view.size(), "Rows");
  state.add_global_memory_reads<nvbench::int8_t>(input_view.chars_size(stream));
  state.add_global_memory_writes<nvbench::int8_t>(input_view.chars_size(stream) / 2);
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = spark_rapids_jni::hex_to_bytes(input_view, stream);
  });
}

}  // namespace

NVBENCH_BENCH(unhex)
  .set_name("Unhex")
  .add_int64_axis("num_rows", {1, 4'096, 1'048'576, 8'388'608})
  .add_int64_axis("string_length", {16, 512, 1'048'576})
  .add_int64_axis("long_row_length", {0, 1'048'576})
  .add_int64_axis("invalid_percent", {0, 1});
