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

#include "cudf_jni_apis.hpp"
#include "protobuf/protobuf.hpp"

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace {

/**
 * Convert a Java Object[] of primitive arrays into a vector-of-vectors.
 * @tparam CppT   Element type in the output vectors (e.g. host_vector<uint8_t>,
 * host_vector<int32_t>).
 * @param convert Per-element callback: (JNIEnv*, jobject) -> std::vector<CppT>.
 *                Must return an empty vector on null input.  Returns std::nullopt on JNI error.
 */
template <typename CppT, typename ConvertFn>
std::vector<CppT> jni_array_of_arrays_to_vectors(JNIEnv* env,
                                                 jobjectArray arr,
                                                 int num_elements,
                                                 ConvertFn convert)
{
  std::vector<CppT> result;
  result.reserve(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    jobject elem = env->GetObjectArrayElement(arr, i);
    if (env->ExceptionCheck()) { return {}; }
    auto vec = convert(env, elem);
    if (elem != nullptr) { env->DeleteLocalRef(elem); }
    if (env->ExceptionCheck()) { return {}; }
    result.push_back(std::move(vec));
  }
  return result;
}

cudf::detail::host_vector<uint8_t> jni_byte_array_to_vector(JNIEnv* env, jobject obj)
{
  if (obj == nullptr) {
    return cudf::detail::make_host_vector<uint8_t>(0, cudf::get_default_stream());
  }
  auto byte_arr = static_cast<jbyteArray>(obj);
  jsize len     = env->GetArrayLength(byte_arr);
  jbyte* bytes  = env->GetByteArrayElements(byte_arr, nullptr);
  if (bytes == nullptr) {
    return cudf::detail::make_host_vector<uint8_t>(0, cudf::get_default_stream());
  }
  auto vec = cudf::detail::make_host_vector<uint8_t>(len, cudf::get_default_stream());
  std::copy(
    reinterpret_cast<uint8_t*>(bytes), reinterpret_cast<uint8_t*>(bytes) + len, vec.begin());
  env->ReleaseByteArrayElements(byte_arr, bytes, JNI_ABORT);
  return vec;
}

cudf::detail::host_vector<int32_t> jni_int_array_to_vector(JNIEnv* env, jobject obj)
{
  if (obj == nullptr) {
    return cudf::detail::make_host_vector<int32_t>(0, cudf::get_default_stream());
  }
  auto int_arr = static_cast<jintArray>(obj);
  jsize len    = env->GetArrayLength(int_arr);
  jint* ints   = env->GetIntArrayElements(int_arr, nullptr);
  if (ints == nullptr) {
    return cudf::detail::make_host_vector<int32_t>(0, cudf::get_default_stream());
  }
  auto vec = cudf::detail::make_host_vector<int32_t>(len, cudf::get_default_stream());
  std::copy(ints, ints + len, vec.begin());
  env->ReleaseIntArrayElements(int_arr, ints, JNI_ABORT);
  return vec;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_Protobuf_decodeToStruct(JNIEnv* env,
                                                         jclass,
                                                         jlong binary_input_view,
                                                         jintArray field_numbers,
                                                         jintArray parent_indices,
                                                         jintArray depth_levels,
                                                         jintArray wire_types,
                                                         jintArray output_type_ids,
                                                         jintArray encodings,
                                                         jbooleanArray is_repeated,
                                                         jbooleanArray is_required,
                                                         jbooleanArray has_default_value,
                                                         jbooleanArray is_output,
                                                         jlongArray default_ints,
                                                         jdoubleArray default_floats,
                                                         jbooleanArray default_bools,
                                                         jobjectArray default_strings,
                                                         jobjectArray enum_valid_values,
                                                         jobjectArray enum_names,
                                                         jboolean fail_on_errors)
{
  auto const all_inputs_valid = binary_input_view && field_numbers && parent_indices &&
                                depth_levels && wire_types && output_type_ids && encodings &&
                                is_repeated && is_required && has_default_value && is_output &&
                                default_ints && default_floats && default_bools &&
                                default_strings && enum_valid_values && enum_names;
  JNI_NULL_CHECK(env, all_inputs_valid, "one or more input arrays are null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const* input = reinterpret_cast<cudf::column_view const*>(binary_input_view);

    cudf::jni::native_jintArray n_field_numbers(env, field_numbers);
    cudf::jni::native_jintArray n_parent_indices(env, parent_indices);
    cudf::jni::native_jintArray n_depth_levels(env, depth_levels);
    cudf::jni::native_jintArray n_wire_types(env, wire_types);
    cudf::jni::native_jintArray n_output_type_ids(env, output_type_ids);
    cudf::jni::native_jintArray n_encodings(env, encodings);
    cudf::jni::native_jbooleanArray n_is_repeated(env, is_repeated);
    cudf::jni::native_jbooleanArray n_is_required(env, is_required);
    cudf::jni::native_jbooleanArray n_has_default(env, has_default_value);
    cudf::jni::native_jbooleanArray n_is_output(env, is_output);
    cudf::jni::native_jlongArray n_default_ints(env, default_ints);
    cudf::jni::native_jdoubleArray n_default_floats(env, default_floats);
    cudf::jni::native_jbooleanArray n_default_bools(env, default_bools);

    int num_fields = n_field_numbers.size();

    // Validate array sizes
    if (n_parent_indices.size() != num_fields || n_depth_levels.size() != num_fields ||
        n_wire_types.size() != num_fields || n_output_type_ids.size() != num_fields ||
        n_encodings.size() != num_fields || n_is_repeated.size() != num_fields ||
        n_is_required.size() != num_fields || n_has_default.size() != num_fields ||
        n_is_output.size() != num_fields || n_default_ints.size() != num_fields ||
        n_default_floats.size() != num_fields || n_default_bools.size() != num_fields ||
        env->GetArrayLength(default_strings) != num_fields ||
        env->GetArrayLength(enum_valid_values) != num_fields ||
        env->GetArrayLength(enum_names) != num_fields) {
      JNI_THROW_NEW(env,
                    cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                    "All field arrays must have the same length",
                    0);
    }

    // Build schema descriptors
    std::vector<spark_rapids_jni::protobuf::nested_field_descriptor> schema;
    schema.reserve(num_fields);
    for (int i = 0; i < num_fields; ++i) {
      schema.push_back({n_field_numbers[i],
                        n_parent_indices[i],
                        n_depth_levels[i],
                        static_cast<spark_rapids_jni::protobuf::proto_wire_type>(n_wire_types[i]),
                        static_cast<cudf::type_id>(n_output_type_ids[i]),
                        static_cast<spark_rapids_jni::protobuf::proto_encoding>(n_encodings[i]),
                        n_is_repeated[i] != 0,
                        n_is_required[i] != 0,
                        n_has_default[i] != 0});
    }

    // Convert boolean arrays
    std::vector<bool> default_bool_values;
    default_bool_values.reserve(num_fields);
    std::vector<bool> output_fields;
    output_fields.reserve(num_fields);
    for (int i = 0; i < num_fields; ++i) {
      default_bool_values.push_back(n_default_bools[i] != 0);
      output_fields.push_back(n_is_output[i] != 0);
    }

    // Convert default values
    std::vector<int64_t> default_int_values(n_default_ints.begin(), n_default_ints.end());
    std::vector<double> default_float_values(n_default_floats.begin(), n_default_floats.end());

    auto default_string_values = jni_array_of_arrays_to_vectors<cudf::detail::host_vector<uint8_t>>(
      env, default_strings, num_fields, jni_byte_array_to_vector);
    if (env->ExceptionCheck()) { return 0; }

    auto enum_values = jni_array_of_arrays_to_vectors<cudf::detail::host_vector<int32_t>>(
      env, enum_valid_values, num_fields, jni_int_array_to_vector);
    if (env->ExceptionCheck()) { return 0; }

    auto enum_name_values =
      jni_array_of_arrays_to_vectors<std::vector<cudf::detail::host_vector<uint8_t>>>(
        env,
        enum_names,
        num_fields,
        [](JNIEnv* e, jobject obj) -> std::vector<cudf::detail::host_vector<uint8_t>> {
          if (obj == nullptr) { return {}; }
          auto inner_arr = static_cast<jobjectArray>(obj);
          jsize num      = e->GetArrayLength(inner_arr);
          return jni_array_of_arrays_to_vectors<cudf::detail::host_vector<uint8_t>>(
            e, inner_arr, num, jni_byte_array_to_vector);
        });
    if (env->ExceptionCheck()) { return 0; }

    spark_rapids_jni::protobuf::protobuf_decode_context context{std::move(schema),
                                                                std::move(default_int_values),
                                                                std::move(default_float_values),
                                                                std::move(default_bool_values),
                                                                std::move(default_string_values),
                                                                std::move(enum_values),
                                                                std::move(enum_name_values),
                                                                static_cast<bool>(fail_on_errors),
                                                                std::move(output_fields)};

    auto result = spark_rapids_jni::protobuf::decode_protobuf_to_struct(
      *input, context, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

    return cudf::jni::release_as_jlong(result);
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
