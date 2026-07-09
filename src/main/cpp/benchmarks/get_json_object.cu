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

#include "common/generate_input.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/io/json.hpp>
#include <cudf/io/types.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <get_json_object.hpp>
#include <nvbench/nvbench.cuh>

// #define DEBUG_PRINT

#ifdef DEBUG_PRINT

#include <cudf/utilities/type_dispatcher.hpp>

#include <iostream>

namespace {

// Copy from `cudf/cpp/tests/utilities/column_utilities.cu`.
struct strings_to_host_fn {
  template <typename OffsetType,
            std::enable_if_t<std::is_same_v<OffsetType, int32_t> ||
                             std::is_same_v<OffsetType, int64_t>>* = nullptr>
  void operator()(std::vector<std::string>& host_data,
                  char const* chars,
                  cudf::column_view const& offsets,
                  rmm::cuda_stream_view stream)
  {
    auto const h_offsets = cudf::detail::make_std_vector_sync(
      cudf::device_span<OffsetType const>(offsets.data<OffsetType>(), offsets.size()), stream);
    // build std::string vector from chars and offsets
    std::transform(std::begin(h_offsets),
                   std::end(h_offsets) - 1,
                   std::begin(h_offsets) + 1,
                   host_data.begin(),
                   [&](auto start, auto end) { return std::string(chars + start, end - start); });
  }

  template <typename OffsetType,
            std::enable_if_t<!std::is_same_v<OffsetType, int32_t> &&
                             !std::is_same_v<OffsetType, int64_t>>* = nullptr>
  void operator()(std::vector<std::string>&,
                  char const*,
                  cudf::column_view const&,
                  rmm::cuda_stream_view)
  {
    CUDF_FAIL("invalid offsets type");
  }
};

template <typename CV>
std::vector<std::string> to_host_strings(CV const& c)
{
  std::vector<std::string> host_strs(c.size());
  auto stream        = cudf::get_default_stream();
  auto const scv     = cudf::strings_column_view(c);
  auto const h_chars = cudf::detail::make_std_vector_sync<char>(
    cudf::device_span<char const>(scv.chars_begin(stream), scv.chars_size(stream)), stream);
  auto const offsets =
    cudf::slice(scv.offsets(), {scv.offset(), scv.offset() + scv.size() + 1}).front();
  cudf::type_dispatcher(
    offsets.type(), strings_to_host_fn{}, host_strs, h_chars.data(), offsets, stream);
  return host_strs;
}

}  // namespace
#endif  // #ifdef DEBUG_PRINT

constexpr auto list_depth = 2;
constexpr auto min_width  = 10;
constexpr auto max_width  = 10;

auto generate_input(std::size_t size_bytes, cudf::size_type max_depth)
{
  data_profile const table_profile =
    data_profile_builder()
      .no_validity()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width)
      .distribution(cudf::type_id::LIST, distribution_id::NORMAL, min_width, max_width)
      .list_depth(list_depth)
      .list_type(cudf::type_id::STRING)
      .struct_depth(max_depth > list_depth ? max_depth - list_depth : 1)
      .struct_types(std::vector<cudf::type_id>{cudf::type_id::LIST});

  auto const input_table = create_random_table(
    std::vector<cudf::type_id>{cudf::type_id::INT32, cudf::type_id::STRING, cudf::type_id::STRUCT},
    table_size_bytes{size_bytes},
    table_profile);

  std::vector<char> buffer;
  cudf::io::sink_info sink(&buffer);
  cudf::io::table_metadata mt{{{"int32"}, {"string"}, {"struct"}}};
  auto write_opts =
    cudf::io::json_writer_options::builder(sink, input_table->view()).lines(true).metadata(mt);
  cudf::io::write_json(write_opts);

  // Split one JSON string into separate JSON objects.
  auto const json_str = std::string{buffer.begin(), buffer.end()};
  auto const json_col = cudf::test::strings_column_wrapper{{json_str}};
  auto split_strs =
    cudf::strings::split_record(cudf::strings_column_view{json_col}, cudf::string_scalar("\n"))
      ->release();

  // Note that split_strs is a list of strings thus we need to extract the strings column.
  auto& json_strings = split_strs.children[cudf::lists_column_view::child_column_index];

#ifdef DEBUG_PRINT
  {
    auto const strs = to_host_strings(json_strings->view());
    std::cout << "First input row: \n" << strs.front() << std::endl;
  }
#endif  // #ifdef DEBUG_PRINT
  return std::move(json_strings);
}

void BM_get_json_object(nvbench::state& state)
{
  auto const size_bytes = static_cast<cudf::size_type>(state.get_int64("size_bytes"));
  auto const max_depth  = static_cast<cudf::size_type>(state.get_int64("max_depth"));

  auto const json_strings = generate_input(size_bytes, max_depth);

  using path_instruction_type = spark_rapids_jni::path_instruction_type;
  std::vector<std::tuple<path_instruction_type, std::string, int32_t>> instructions;
  instructions.emplace_back(path_instruction_type::NAMED, "struct", -1);
  for (int i = 0; i < max_depth - list_depth; ++i) {
    instructions.emplace_back(path_instruction_type::NAMED, "0", -1);
  }

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // Can also verify at https://jsonpath.com/.
    [[maybe_unused]] auto const output = spark_rapids_jni::get_json_object(
      cudf::strings_column_view{json_strings->view()}, instructions);

#ifdef DEBUG_PRINT
    {
      auto const strs = to_host_strings(output->view());
      std::cout << "First output row: \n" << strs.front() << std::endl << std::endl << std::endl;
    }
#endif  // #ifdef DEBUG_PRINT
  });
  state.add_global_memory_reads<nvbench::int8_t>(size_bytes);
}

NVBENCH_BENCH(BM_get_json_object)
  .set_name("get_json_object")
  .add_int64_axis("size_bytes", {1'000'000, 10'000'000, 100'000'000, 1'000'000'000})
  .add_int64_axis("max_depth", {2, 4, 6, 8});
