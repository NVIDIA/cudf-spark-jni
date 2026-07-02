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
#include "get_json_object.hpp"
#include "json_utils.hpp"

#include <cudf/strings/strings_column_view.hpp>

#include <vector>

using path_instruction_type = spark_rapids_jni::path_instruction_type;

extern "C" {

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_JSONUtils_getMaxJSONPathDepth(JNIEnv* env,
                                                                                      jclass)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    return spark_rapids_jni::MAX_JSON_PATH_DEPTH;
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_JSONUtils_getJsonObject(JNIEnv* env,
                                                         jclass,
                                                         jlong input_column,
                                                         jbyteArray j_type_nums,
                                                         jobjectArray j_names,
                                                         jintArray j_indexes)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  JNI_NULL_CHECK(env, j_type_nums, "j_type_nums is null", 0);
  JNI_NULL_CHECK(env, j_names, "j_names is null", 0);
  JNI_NULL_CHECK(env, j_indexes, "j_indexes is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const n_column_view      = reinterpret_cast<cudf::column_view const*>(input_column);
    auto const n_strings_col_view = cudf::strings_column_view{*n_column_view};

    std::vector<std::tuple<path_instruction_type, std::string, int32_t>> instructions;

    auto const type_nums = cudf::jni::native_jbyteArray(env, j_type_nums).to_vector();
    auto const names     = cudf::jni::native_jstringArray(env, j_names);
    auto const indexes   = cudf::jni::native_jintArray(env, j_indexes).to_vector();
    int size             = type_nums.size();
    if (names.size() != size || indexes.size() != static_cast<std::size_t>(size) ||
        type_nums.size() != static_cast<std::size_t>(size)) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "wrong number of entries passed in", 0);
    }

    for (int i = 0; i < size; i++) {
      path_instruction_type instruction_type = static_cast<path_instruction_type>(type_nums[i]);
      char const* name_str                   = names[i].get();
      jlong index                            = indexes[i];
      instructions.emplace_back(instruction_type, name_str, index);
    }

    return cudf::jni::release_as_jlong(
      spark_rapids_jni::get_json_object(n_strings_col_view, instructions));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_JSONUtils_getJsonObjectMultiplePaths(JNIEnv* env,
                                                                      jclass,
                                                                      jlong j_input,
                                                                      jbyteArray j_type_nums,
                                                                      jobjectArray j_names,
                                                                      jintArray j_indexes,
                                                                      jintArray j_path_offsets,
                                                                      jlong memory_budget_bytes,
                                                                      jint parallel_override)
{
  JNI_NULL_CHECK(env, j_input, "j_input column is null", 0);
  JNI_NULL_CHECK(env, j_type_nums, "j_type_nums is null", 0);
  JNI_NULL_CHECK(env, j_names, "j_names is null", 0);
  JNI_NULL_CHECK(env, j_indexes, "j_indexes is null", 0);
  JNI_NULL_CHECK(env, j_path_offsets, "j_path_offsets is null", 0);

  using path_type = std::vector<std::tuple<path_instruction_type, std::string, int32_t>>;

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const path_offsets = cudf::jni::native_jintArray(env, j_path_offsets).to_vector();
    CUDF_EXPECTS(path_offsets.size() > 1, "Invalid path offsets.");
    auto const type_nums = cudf::jni::native_jbyteArray(env, j_type_nums).to_vector();
    auto const names     = cudf::jni::native_jstringArray(env, j_names);
    auto const indexes   = cudf::jni::native_jintArray(env, j_indexes).to_vector();
    auto const num_paths = path_offsets.size() - 1;
    std::vector<path_type> paths(num_paths);
    auto const num_entries = path_offsets[num_paths];

    if (num_entries < 0 || names.size() != num_entries ||
        indexes.size() != static_cast<std::size_t>(num_entries) ||
        type_nums.size() != static_cast<std::size_t>(num_entries)) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "wrong number of entries passed in", 0);
    }

    for (std::size_t i = 0; i < num_paths; ++i) {
      auto const path_size = path_offsets[i + 1] - path_offsets[i];
      auto path            = path_type{};
      path.reserve(path_size);
      for (int j = path_offsets[i]; j < path_offsets[i + 1]; ++j) {
        path_instruction_type instruction_type = static_cast<path_instruction_type>(type_nums[j]);
        char const* name_str                   = names[j].get();
        jlong index                            = indexes[j];
        path.emplace_back(instruction_type, name_str, index);
      }

      paths[i] = std::move(path);
    }

    auto const input_cv = reinterpret_cast<cudf::column_view const*>(j_input);
    auto output         = spark_rapids_jni::get_json_object_multiple_paths(
      cudf::strings_column_view{*input_cv}, paths, memory_budget_bytes, parallel_override);

    auto out_handles = cudf::jni::native_jlongArray(env, output.size());
    std::transform(output.begin(), output.end(), out_handles.begin(), [](auto& col) {
      return cudf::jni::release_as_jlong(col);
    });
    return out_handles.get_jArray();
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_JSONUtils_extractRawMapFromJsonString(
  JNIEnv* env,
  jclass,
  jlong j_input,
  jboolean normalize_single_quotes,
  jboolean allow_leading_zeros,
  jboolean allow_nonnumeric_numbers,
  jboolean allow_unquoted_control)
{
  JNI_NULL_CHECK(env, j_input, "j_input is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input_cv = reinterpret_cast<cudf::column_view const*>(j_input);
    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::from_json_to_raw_map(
        cudf::strings_column_view{*input_cv},
        spark_rapids_jni::json_parse_options{
          .normalize_single_quotes  = static_cast<bool>(normalize_single_quotes),
          .allow_leading_zeros      = static_cast<bool>(allow_leading_zeros),
          .allow_nonnumeric_numbers = static_cast<bool>(allow_nonnumeric_numbers),
          .allow_unquoted_control   = static_cast<bool>(allow_unquoted_control)})
        .release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_JSONUtils_extractRawMapArrayFromJsonString(
  JNIEnv* env,
  jclass,
  jlong j_input,
  jboolean normalize_single_quotes,
  jboolean allow_leading_zeros,
  jboolean allow_nonnumeric_numbers,
  jboolean allow_unquoted_control)
{
  JNI_NULL_CHECK(env, j_input, "j_input is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input_cv = reinterpret_cast<cudf::column_view const*>(j_input);
    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::from_json_to_raw_map_array_values(
        cudf::strings_column_view{*input_cv},
        spark_rapids_jni::json_parse_options{
          .normalize_single_quotes  = static_cast<bool>(normalize_single_quotes),
          .allow_leading_zeros      = static_cast<bool>(allow_leading_zeros),
          .allow_nonnumeric_numbers = static_cast<bool>(allow_nonnumeric_numbers),
          .allow_unquoted_control   = static_cast<bool>(allow_unquoted_control)})
        .release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_JSONUtils_fromJSONToStructs(JNIEnv* env,
                                                             jclass,
                                                             jlong j_input,
                                                             jobjectArray j_col_names,
                                                             jintArray j_num_children,
                                                             jintArray j_types,
                                                             jintArray j_scales,
                                                             jintArray j_precisions,
                                                             jboolean normalize_single_quotes,
                                                             jboolean allow_leading_zeros,
                                                             jboolean allow_nonnumeric_numbers,
                                                             jboolean allow_unquoted_control,
                                                             jboolean is_us_locale)
{
  JNI_NULL_CHECK(env, j_input, "j_input is null", 0);
  JNI_NULL_CHECK(env, j_col_names, "j_col_names is null", 0);
  JNI_NULL_CHECK(env, j_num_children, "j_num_children is null", 0);
  JNI_NULL_CHECK(env, j_types, "j_types is null", 0);
  JNI_NULL_CHECK(env, j_scales, "j_scales is null", 0);
  JNI_NULL_CHECK(env, j_precisions, "j_precisions is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const input_cv     = reinterpret_cast<cudf::column_view const*>(j_input);
    auto const col_names    = cudf::jni::native_jstringArray(env, j_col_names).as_cpp_vector();
    auto const num_children = cudf::jni::native_jintArray(env, j_num_children).to_vector();
    auto const types        = cudf::jni::native_jintArray(env, j_types).to_vector();
    auto const scales       = cudf::jni::native_jintArray(env, j_scales).to_vector();
    auto const precisions   = cudf::jni::native_jintArray(env, j_precisions).to_vector();

    CUDF_EXPECTS(col_names.size() > 0, "Invalid schema data: col_names.");
    CUDF_EXPECTS(col_names.size() == num_children.size(), "Invalid schema data: num_children.");
    CUDF_EXPECTS(col_names.size() == types.size(), "Invalid schema data: types.");
    CUDF_EXPECTS(col_names.size() == scales.size(), "Invalid schema data: scales.");
    CUDF_EXPECTS(col_names.size() == precisions.size(), "Invalid schema data: precisions.");

    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::from_json_to_structs(cudf::strings_column_view{*input_cv},
                                             col_names,
                                             num_children,
                                             types,
                                             scales,
                                             precisions,
                                             normalize_single_quotes,
                                             allow_leading_zeros,
                                             allow_nonnumeric_numbers,
                                             allow_unquoted_control,
                                             is_us_locale)
        .release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_JSONUtils_convertFromStrings(JNIEnv* env,
                                                              jclass,
                                                              jlong j_input,
                                                              jintArray j_num_children,
                                                              jintArray j_types,
                                                              jintArray j_scales,
                                                              jintArray j_precisions,
                                                              jboolean allow_nonnumeric_numbers,
                                                              jboolean is_us_locale)
{
  JNI_NULL_CHECK(env, j_input, "j_input is null", 0);
  JNI_NULL_CHECK(env, j_num_children, "j_num_children is null", 0);
  JNI_NULL_CHECK(env, j_types, "j_types is null", 0);
  JNI_NULL_CHECK(env, j_scales, "j_scales is null", 0);
  JNI_NULL_CHECK(env, j_precisions, "j_precisions is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const input_cv     = reinterpret_cast<cudf::column_view const*>(j_input);
    auto const num_children = cudf::jni::native_jintArray(env, j_num_children).to_vector();
    auto const types        = cudf::jni::native_jintArray(env, j_types).to_vector();
    auto const scales       = cudf::jni::native_jintArray(env, j_scales).to_vector();
    auto const precisions   = cudf::jni::native_jintArray(env, j_precisions).to_vector();

    CUDF_EXPECTS(num_children.size() > 0, "Invalid schema data: num_children.");
    CUDF_EXPECTS(num_children.size() == types.size(), "Invalid schema data: types.");
    CUDF_EXPECTS(num_children.size() == scales.size(), "Invalid schema data: scales.");
    CUDF_EXPECTS(num_children.size() == precisions.size(), "Invalid schema data: precisions.");

    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::convert_from_strings(cudf::strings_column_view{*input_cv},
                                             num_children,
                                             types,
                                             scales,
                                             precisions,
                                             allow_nonnumeric_numbers,
                                             is_us_locale)
        .release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_JSONUtils_removeQuotes(
  JNIEnv* env, jclass, jlong j_input, jboolean nullify_if_not_quoted)
{
  JNI_NULL_CHECK(env, j_input, "j_input is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input_cv = reinterpret_cast<cudf::column_view const*>(j_input);
    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::remove_quotes(cudf::strings_column_view{*input_cv}, nullify_if_not_quoted)
        .release());
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
