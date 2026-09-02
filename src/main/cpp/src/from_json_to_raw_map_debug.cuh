/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.
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

// #define DEBUG_FROM_JSON

#ifdef DEBUG_FROM_JSON

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/detail/tokenize_json.hpp>
#include <cudf/strings/strings_column_view.hpp>

//
#include <rmm/device_uvector.hpp>

#include <cuda/stream>

//
#include <sstream>

namespace spark_rapids_jni {

using namespace cudf::io::json;

// Convert the token value into string name, for debugging purpose.
std::string token_to_string(PdaTokenT const token_type)
{
  switch (token_type) {
    case token_t::StructBegin: return "StructBegin";
    case token_t::StructEnd: return "StructEnd";
    case token_t::ListBegin: return "ListBegin";
    case token_t::ListEnd: return "ListEnd";
    case token_t::StructMemberBegin: return "StructMemberBegin";
    case token_t::StructMemberEnd: return "StructMemberEnd";
    case token_t::FieldNameBegin: return "FieldNameBegin";
    case token_t::FieldNameEnd: return "FieldNameEnd";
    case token_t::StringBegin: return "StringBegin";
    case token_t::StringEnd: return "StringEnd";
    case token_t::ValueBegin: return "ValueBegin";
    case token_t::ValueEnd: return "ValueEnd";
    case token_t::ErrorBegin: return "ErrorBegin";
    default: return "Unknown";
  }
}

// Print the content of the input device vector.
template <typename T, typename U = int>
void print_debug(rmm::device_uvector<T> const& input,
                 std::string const& name,
                 std::string const& separator,
                 cuda::stream_ref stream)
{
  auto const h_input = cudf::detail::make_host_vector_sync(
    cudf::device_span<T const>{input.data(), input.size()}, stream);
  std::stringstream ss;
  ss << name << ":\n";
  for (size_t i = 0; i < h_input.size(); ++i) {
    ss << static_cast<U>(h_input[i]);
    if (separator.size() > 0 && i + 1 < h_input.size()) { ss << separator; }
  }
  std::cerr << ss.str() << std::endl;
}

// Print the content of the input map given by a device vector.
template <typename T, typename U = int>
void print_map_debug(rmm::device_uvector<T> const& input,
                     std::string const& name,
                     cuda::stream_ref stream)
{
  auto const h_input = cudf::detail::make_host_vector_sync(
    cudf::device_span<T const>{input.data(), input.size()}, stream);
  std::stringstream ss;
  ss << name << ":\n";
  for (size_t i = 0; i < h_input.size(); ++i) {
    ss << i << " => " << static_cast<U>(h_input[i]) << "\n";
  }
  std::cerr << ss.str() << std::endl;
}

// Print the content of the input pairs given by a device vector.
template <typename T, typename U = int>
void print_pair_debug(rmm::device_uvector<T> const& input,
                      std::string const& name,
                      cuda::stream_ref stream)
{
  auto const h_input = cudf::detail::make_host_vector_sync(
    cudf::device_span<T const>{input.data(), input.size()}, stream);
  std::stringstream ss;
  ss << name << ":\n";
  for (size_t i = 0; i < h_input.size(); ++i) {
    ss << "[ " << static_cast<int>(h_input[i].first) << ", " << static_cast<int>(h_input[i].second)
       << " ]\n";
  }
  std::cerr << ss.str() << std::endl;
}

// Print the final output map data (Spark's MapType, i.e., List<Struct<String,String>>).
void print_output_spark_map(std::unique_ptr<cudf::column> const& list_offsets,
                            std::unique_ptr<cudf::column> const& extracted_keys,
                            std::unique_ptr<cudf::column> const& extracted_values,
                            cuda::stream_ref stream)
{
  if (extracted_keys->size() == 0) {
    std::cerr << "Extract keys-values are all empty.\n" << std::endl;
    return;
  }

  auto const keys_sv      = cudf::strings_column_view{extracted_keys->view()};
  auto const values_sv    = cudf::strings_column_view{extracted_values->view()};
  auto const keys_offsets = extracted_keys->child(cudf::strings_column_view::offsets_column_index);
  auto const values_offsets =
    extracted_values->child(cudf::strings_column_view::offsets_column_index);

  auto const h_extracted_keys_child = cudf::detail::make_host_vector_sync(
    cudf::device_span<char const>{keys_sv.chars_begin(stream),
                                  static_cast<size_t>(keys_sv.chars_size(stream))},
    stream);
  auto const h_extracted_keys_offsets = cudf::detail::make_host_vector_sync(
    cudf::device_span<int const>{keys_offsets.view().data<int>(),
                                 static_cast<size_t>(keys_offsets.size())},
    stream);

  auto const h_extracted_values_child = cudf::detail::make_host_vector_sync(
    cudf::device_span<char const>{values_sv.chars_begin(stream),
                                  static_cast<size_t>(values_sv.chars_size(stream))},
    stream);
  auto const h_extracted_values_offsets = cudf::detail::make_host_vector_sync(
    cudf::device_span<int const>{values_offsets.view().data<int>(),
                                 static_cast<size_t>(values_offsets.size())},
    stream);

  auto const h_list_offsets = cudf::detail::make_host_vector_sync(
    cudf::device_span<cudf::size_type const>{list_offsets->view().begin<cudf::size_type>(),
                                             static_cast<std::size_t>(list_offsets->size())},
    stream);
  CUDF_EXPECTS(h_list_offsets.back() == extracted_keys->size(),
               "Invalid list offsets computation.");

  std::stringstream ss;
  ss << "Extract keys-values:\n";

  for (size_t i = 0; i + 1 < h_list_offsets.size(); ++i) {
    ss << "List " << i << ": [" << h_list_offsets[i] << ", " << h_list_offsets[i + 1] << "]\n";
    for (cudf::size_type string_idx = h_list_offsets[i]; string_idx < h_list_offsets[i + 1];
         ++string_idx) {
      {
        auto const string_begin = h_extracted_keys_offsets[string_idx];
        auto const string_end   = h_extracted_keys_offsets[string_idx + 1];
        auto const size         = string_end - string_begin;
        auto const ptr          = &h_extracted_keys_child[string_begin];
        ss << "\t\"" << std::string(ptr, size) << "\" : ";
      }
      {
        auto const string_begin = h_extracted_values_offsets[string_idx];
        auto const string_end   = h_extracted_values_offsets[string_idx + 1];
        auto const size         = string_end - string_begin;
        auto const ptr          = &h_extracted_values_child[string_begin];
        ss << "\"" << std::string(ptr, size) << "\"\n";
      }
    }
  }
  std::cerr << ss.str() << std::endl;
}

}  // namespace spark_rapids_jni

#endif  // DEBUG_FROM_JSON
