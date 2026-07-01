#=============================================================================
# Copyright (c) 2022-2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

add_library(spark_rapids_jni_datagen STATIC common/generate_input.cu)
target_compile_features(spark_rapids_jni_datagen PUBLIC cxx_std_20 cuda_std_20)

target_compile_options(
  spark_rapids_jni_datagen PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                      "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)

target_link_libraries(
  spark_rapids_jni_datagen PUBLIC cudf::cudf nvtx3::nvtx3-cpp
)

target_include_directories(
  spark_rapids_jni_datagen
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>" "$<BUILD_INTERFACE:${CUDF_DIR}/cpp>"
         "$<BUILD_INTERFACE:${CUDF_DIR}/cpp/include>"
)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureBench CMAKE_BENCH_NAME)
    add_executable(${CMAKE_BENCH_NAME} ${ARGN})
    target_compile_options(${CMAKE_BENCH_NAME}
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")

    target_include_directories(${CMAKE_BENCH_NAME}
            PRIVATE "$<BUILD_INTERFACE:${CUDF_DIR}/cpp>"
                    "$<BUILD_INTERFACE:${CUDF_DIR}/cpp/include>"
                    "$<BUILD_INTERFACE:${SPARK_RAPIDS_JNI_SOURCE_DIR}>"
                    "$<BUILD_INTERFACE:${SPARK_RAPIDS_JNI_SOURCE_DIR}/src>")
    set_target_properties(${CMAKE_BENCH_NAME}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/benchmarks>"
                   INSTALL_RPATH "\$ORIGIN/../../../lib"
                   CXX_STANDARD 20
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS ON
                   CUDA_STANDARD 20
                   CUDA_STANDARD_REQUIRED ON
        )
    target_link_libraries(${CMAKE_BENCH_NAME} nvbench::main spark_rapids_jni_datagen ${CUDF_BENCHMARK_COMMON}
                                              cudf::cudf spark_rapids_jni Threads::Threads cudf::cudftestutil)
    install(
        TARGETS ${CMAKE_BENCH_NAME}
        COMPONENT testing
        DESTINATION bin/benchmarks/spark-rapids-jni
        EXCLUDE_FROM_ALL
    )
endfunction(ConfigureBench)

###################################################################################################
### benchmark sources #############################################################################
###################################################################################################

ConfigureBench(ROW_CONVERSION_BENCH
    row_conversion.cpp)

ConfigureBench(STRING_TO_FLOAT_BENCH
  cast_string_to_float.cpp)

ConfigureBench(LONG_TO_BINARY_STRING_BENCH
  cast_long_to_binary_string.cpp)

ConfigureBench(BLOOM_FILTER_BENCH
  bloom_filter.cu)

ConfigureBench(GET_JSON_OBJECT_BENCH
  get_json_object.cu)

ConfigureBench(PARSE_URI_BENCH
  parse_uri.cpp)

ConfigureBench(PROTOBUF_DECODE_BENCH
  protobuf_decode.cu)
