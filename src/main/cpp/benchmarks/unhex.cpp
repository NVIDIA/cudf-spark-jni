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

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cast_string.hpp>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace {

void unhex(nvbench::state& state)
{
  auto const shape = state.get_int64("shape");
  int64_t num_rows;
  int64_t string_length;
  switch (shape) {
    case 0:
      num_rows      = 1;
      string_length = 1'048'576;
      break;
    case 1:
      num_rows      = 32;
      string_length = 65'536;
      break;
    case 2:
      num_rows      = 1024;
      string_length = 1024;
      break;
    case 3:
      num_rows      = 1'000'000;
      string_length = 16;
      break;
    case 4:
      num_rows      = 64;
      string_length = 16;
      break;
    case 5:
      num_rows      = 8192;
      string_length = 16;
      break;
    case 6:
      num_rows      = 1024;
      string_length = 16;
      break;
    case 7:
      num_rows      = 1024;
      string_length = 16;
      break;
    case 8:
      num_rows      = 2048;
      string_length = 16;
      break;
    case 9:
      num_rows      = 4096;
      string_length = 16;
      break;
    case 10:
      num_rows      = 4096;
      string_length = 16;
      break;
    case 11:
      num_rows      = 8193;
      string_length = 16;
      break;
    case 12:
      num_rows      = 16'384;
      string_length = 16;
      break;
    case 13:
      num_rows      = 1'000'000;
      string_length = 16;
      break;
    case 14:
      num_rows      = 8193;
      string_length = 16;
      break;
    case 15:
      num_rows      = 2049;
      string_length = 16;
      break;
    case 16:
      num_rows      = 4096;
      string_length = 16;
      break;
    case 17:
      num_rows      = 4096;
      string_length = 16;
      break;
    case 18:
      num_rows      = 20'001;
      string_length = 16;
      break;
    case 19:
      num_rows      = 20'001;
      string_length = 16;
      break;
    case 20:
      num_rows      = 1'000'000;
      string_length = 16;
      break;
    case 21:
      num_rows      = 20'001;
      string_length = 16;
      break;
    case 22:
      num_rows      = 20'001;
      string_length = 16;
      break;
    case 23:
      num_rows      = 4096;
      string_length = 16;
      break;
    case 24:
      num_rows      = 4096;
      string_length = 16;
      break;
    case 25:
      num_rows      = 20'001;
      string_length = 16;
      break;
    case 26:
      num_rows      = 20'001;
      string_length = 16;
      break;
    case 27:
      num_rows      = 20'001;
      string_length = 16;
      break;
    case 28:
      num_rows      = 20'001;
      string_length = 16;
      break;
    case 29:
      num_rows      = 20'001;
      string_length = 512;
      break;
    case 30:
      num_rows      = 16'384;
      string_length = 512;
      break;
    case 31:
      num_rows      = 16'385;
      string_length = 512;
      break;
    case 32:
      num_rows      = 20'001;
      string_length = 16;
      break;
    case 33:
      num_rows      = 8192;
      string_length = 16;
      break;
    case 34:
      num_rows      = 8192;
      string_length = 16;
      break;
    case 35:
      num_rows      = 8'388'608;
      string_length = 16;
      break;
    case 36:
      num_rows      = 8'388'608;
      string_length = 16;
      break;
    default:
      num_rows      = 1024;
      string_length = 16;
      break;
  }
  std::vector<std::string> values(num_rows, std::string(string_length, 'A'));
  if (shape == 6 || shape == 10 || shape == 11 || shape == 12 || shape == 13 || shape == 20 ||
      shape == 36) {
    values.front() = std::string(1'048'576, 'A');
  }
  if (shape == 16 || shape == 17) { values.front() = std::string(shape == 16 ? 1023 : 1024, 'A'); }
  if (shape == 18 || shape == 19) { values.front() = std::string(shape == 18 ? 1249 : 1250, 'A'); }
  if (shape == 20) { values.front().back() = 'G'; }
  if (shape == 21 || shape == 22 || shape == 27 || shape == 28) {
    auto const long_row_count = shape == 21 ? 1249 : shape == 22 ? 1250 : shape == 27 ? 4999 : 5000;
    for (int64_t row = 0; row < long_row_count; ++row) {
      values[row * (long_row_count > 1250 ? 4 : 16)] = std::string(512, 'A');
    }
  }
  if (shape == 23 || shape == 24) { values.back() = std::string(shape == 23 ? 255 : 256, 'A'); }
  if (shape == 25 || shape == 26) { values.back() = std::string(shape == 25 ? 311 : 312, 'A'); }
  if (shape == 33 || shape == 34) { values.back() = std::string(shape == 33 ? 255 : 256, 'A'); }
  std::vector<bool> validity(num_rows, true);
  if (shape == 32) {
    for (int64_t row = 0; row < num_rows; ++row) {
      values[row] = std::string((row * 37) % 96, "0123456789ABCDEF"[row % 16]);
      if (row % 23 == 0 && !values[row].empty()) { values[row].back() = 'G'; }
      if (row % 17 == 0) { validity[row] = false; }
    }
    values.back() = std::string(2048, 'a');
  }
  if (shape == 14) {
    for (int64_t row = 0; row < num_rows; row += 4) {
      validity[row] = false;
      if (row + 1 < num_rows) { values[row + 1].back() = 'G'; }
    }
  }
  state.add_element_count(num_rows, "Rows");
  auto const [shortest, longest] =
    std::minmax_element(values.begin(), values.end(), [](auto const& left, auto const& right) {
      return left.size() < right.size();
    });
  state.add_summary("MinStringLength").set_float64("value", shortest->size());
  state.add_summary("MaxStringLength").set_float64("value", longest->size());
  state.add_summary("NullRows")
    .set_float64("value", std::count(validity.begin(), validity.end(), false));
  state.add_summary("InvalidRows")
    .set_float64("value", std::count_if(values.begin(), values.end(), [](auto const& value) {
                   return value.find('G') != std::string::npos;
                 }));
  auto const input =
    cudf::test::strings_column_wrapper(values.begin(), values.end(), validity.begin());
  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = spark_rapids_jni::hex_to_bytes(cudf::strings_column_view{input}, stream);
  });
}

}  // namespace

NVBENCH_BENCH(unhex).set_name("Unhex").add_int64_axis(
  "shape", {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36});
