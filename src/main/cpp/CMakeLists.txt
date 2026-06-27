# =============================================================================
# Copyright (c) 2022-2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

# CUDF_DIR is set from the environment variable, allowing to specify
# a custom cudf repository other than `thirdparty/cudf`.
set(CUDF_DIR
  "$ENV{CUDF_ROOT}"
  CACHE STRING "path to cudf repository"
)
set(CUDF_INSTALL_DIR
  "$ENV{CUDF_INSTALL_DIR}"
  CACHE STRING "path to libcudf install root"
)
set(CUDFJNI_BUILD_DIR
  "$ENV{CUDFJNI_BUILD_DIR}"
  CACHE STRING "path to libcudfjni build root"
)

# libcudf's kvikio dependency requires this to be set when statically linking CUDA runtime
set(CUDA_STATIC_RUNTIME ON)

include("${CUDF_DIR}/cmake/rapids_config.cmake")
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)
rapids_cpm_init()

rapids_cuda_init_architectures(SPARK_RAPIDS_JNI)

project(
  SPARK_RAPIDS_JNI
  VERSION 26.08.00
  LANGUAGES CXX CUDA
)

# We use no C++20 modules; disable module scanning so CMake emits no ninja dyndep phase. Otherwise
# its dyndep refresh trips ninja's RefreshDyndepDependents assertion on incremental rebuilds after
# a structural source change. Mirrors libcudf (rapidsai/cudf#19192).
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# ##################################################################################################
# * build options ---------------------------------------------------------------------------------

option(CUDF_USE_PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" ON)
option(USE_GDS "Build with GPUDirect Storage (GDS)/cuFile support" OFF)
option(BUILD_TESTS "Configure CMake to build tests" OFF)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)
option(BUILD_FAULTINJ "Configure CMake to build fault injection" ON)
option(BUILD_PROFILER "Configure CMake to build profiler" ON)

message(
  VERBOSE "SPARK_RAPIDS_JNI: Build with per-thread default stream:
            ${CUDF_USE_PER_THREAD_DEFAULT_STREAM}"
)
message(VERBOSE "SPARK_RAPIDS_JNI: Configure CMake to build tests: ${BUILD_TESTS}")
message(VERBOSE "SPARK_RAPIDS_JNI: Configure CMake to build (nvbench) benchmarks: ${BUILD_BENCHMARKS}")
message(VERBOSE "SPARK_RAPIDS_JNI: Configure CMake to build fault injection: ${BUILD_FAULTINJ}")

set(SPARK_RAPIDS_JNI_BUILD_TESTS ${BUILD_TESTS})
set(SPARK_RAPIDS_JNI_BUILD_BENCHMARKS ${BUILD_BENCHMARKS})
set(SPARK_RAPIDS_JNI_BUILD_FAULTINJ ${BUILD_FAULTINJ})
if(NOT SPARK_RAPIDS_JNI_GENERATED_INCLUDE_DIR)
  set(SPARK_RAPIDS_JNI_GENERATED_INCLUDE_DIR ${SPARK_RAPIDS_JNI_BINARY_DIR}/generated/include)
endif()
if(NOT SPARK_RAPIDS_JNI_GENERATED_SOURCE_DIR)
  set(SPARK_RAPIDS_JNI_GENERATED_SOURCE_DIR ${SPARK_RAPIDS_JNI_BINARY_DIR}/generated/src)
endif()

# Set RMM logging level
set(RMM_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level."
)
set_property(
  CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF"
)
message(VERBOSE "SPARK_RAPIDS_JNI: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

# ##################################################################################################
# * compiler options ------------------------------------------------------------------------------

rapids_find_package(CUDAToolkit REQUIRED)

# Set a default build type if none was specified
rapids_cmake_build_type("Release")

# ##################################################################################################
# * dependencies ----------------------------------------------------------------------------------

# version header
find_package(Git REQUIRED)
execute_process(COMMAND
  "${GIT_EXECUTABLE}" describe --abbrev=40 --always --dirty --long
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE SPARK_RAPIDS_JNI_COMMIT_DETAILS
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
configure_file(
  src/spark_rapids_jni_version.cpp.in
  "${SPARK_RAPIDS_JNI_GENERATED_SOURCE_DIR}/spark_rapids_jni_version.cpp"
  @ONLY
)

# Path to search for the pre-installed packages.
set(CMAKE_PREFIX_PATH ${CUDF_INSTALL_DIR} "${CUDF_INSTALL_DIR}/lib64/cmake")

# Reuse cached nvtx package that is already available after installing libcudf.
find_package(nvtx3 REQUIRED)

if(NOT TARGET spdlog::spdlog_header_only)
  message(STATUS "Fetch package spdlog from source.")
  include(${CMAKE_SOURCE_DIR}/cmake/get_spdlog.cmake)
endif()

# JNI
find_package(JNI REQUIRED)
if(JNI_FOUND)
  message(STATUS "JDK with JNI in ${JNI_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "JDK with JNI not found, please check your settings.")
endif()

# cudf
if(BUILD_TESTS)
  include(${rapids-cmake-dir}/cpm/gtest.cmake)
  rapids_cpm_gtest(BUILD_STATIC)
  rapids_find_package(cudf REQUIRED COMPONENTS testing)
else()
  rapids_find_package(cudf REQUIRED)
endif()

# cuCollections
# As of rapidsai/cudf PR #22277 (commit 5a375a3417ef70c55564bd0c4c6255a2da5dca49), libcudf
# wraps its `cuco::cuco` link with `$<BUILD_LOCAL_INTERFACE:...>`, so the cuco headers are no
# longer transitively exposed to downstream consumers via `find_package(cudf)`. Several
# spark-rapids-jni sources (e.g. the hash kernels) include cudf detail headers that pull in
# `<cuco/hash_functions.cuh>`, so we need to fetch cuco ourselves.
include(${rapids-cmake-dir}/cpm/cuco.cmake)
rapids_cpm_cuco()

# Get the compilation flags and definitions from cudf.
get_target_property(CUDF_CXX_FLAGS cudf::cudf CUDF_CXX_FLAGS)
get_target_property(CUDF_CUDA_FLAGS cudf::cudf CUDF_CUDA_FLAGS)
get_target_property(CUDF_CXX_DEFINITIONS cudf::cudf CUDF_CXX_DEFINITIONS)
get_target_property(CUDF_CUDA_DEFINITIONS cudf::cudf CUDF_CUDA_DEFINITIONS)

# cudfjni
find_library(CUDFJNI_LIB "libcudfjni.a" REQUIRED NO_DEFAULT_PATH
  HINTS "${CUDFJNI_BUILD_DIR}"
)

# arrow
find_library(ARROW_LIB "libarrow.a" REQUIRED NO_DEFAULT_PATH
  HINTS "${CUDFJNI_BUILD_DIR}/_deps/arrow-build/release/"
)

# parquet
find_library(PARQUET_LIB "libparquet.a" REQUIRED NO_DEFAULT_PATH
  HINTS "${CUDFJNI_BUILD_DIR}/_deps/arrow-build/release/"
)

# Internal parquet headers
set (GENERATED_PARQUET_INCLUDE
    "${CUDFJNI_BUILD_DIR}/_deps/arrow-src/cpp/src/"
    CACHE STRING "generated parquet thrift headers"
)

# thrift
find_library(THRIFT_LIB "libthrift.a" REQUIRED NO_DEFAULT_PATH
    HINTS "${CUDFJNI_BUILD_DIR}/lib/"
)

set(CUDFJNI_INCLUDE_DIRS
  "${CUDF_DIR}/java/src/main/native/include"
  "${CUDF_DIR}/java/src/main/native/src"
  "${GENERATED_PARQUET_INCLUDE}"
  "${CUDFJNI_BUILD_DIR}/_deps/thrift-build/"
  "${CUDFJNI_BUILD_DIR}/_deps/thrift-src/lib/cpp/src/"
)

# ##################################################################################################
# * library targets -------------------------------------------------------------------------------

add_library(
  spark_rapids_jni SHARED
  # Files starting with uppercase
  src/Aggregation64Utils.cpp
  src/ArithmeticJni.cpp
  src/BloomFilterJni.cpp
  src/CaseWhenJni.cpp
  src/CastStringJni.cpp
  src/CharsetDecodeJni.cpp
  src/DateTimeUtilsJni.cpp
  src/DecimalUtilsJni.cpp
  src/DeviceAttrJni.cpp
  src/GpuTimeZoneDBJni.cpp
  src/HistogramJni.cpp
  src/HostTableJni.cpp
  src/HyperLogLogPlusPlusHostUDFJni.cpp
  src/JSONUtilsJni.cpp
  src/KudoGpuSerializerJni.cpp
  src/ListSliceJni.cpp
  src/MapJni.cpp
  src/MapUtilsJni.cpp
  src/MapZipWithUtilsJNI.cpp
  src/JoinPrimitivesJni.cpp
  src/NativeParquetJni.cpp
  src/NumberConverterJni.cpp
  src/ParseURIJni.cpp
  src/ProtobufJni.cpp
  src/RegexRewriteUtilsJni.cpp
  src/RowConversionJni.cpp
  src/SparkResourceAdaptorJni.cpp
  src/StringUtilsJni.cpp
  src/SubStringIndexJni.cpp
  src/TaskPriorityJni.cpp
  src/ZOrderJni.cpp
  src/iceberg/IcebergBucketJni.cpp
  src/iceberg/IcebergDateTimeUtilJni.cpp
  src/iceberg/IcebergTruncateJni.cpp
  # Files starting with lowercase
  src/aggregation64_utils.cu
  src/bloom_filter.cu
  src/case_when.cu
  src/cast_decimal_to_string.cu
  src/cast_float_to_string.cu
  src/cast_long_to_binary_string.cu
  src/cast_string.cu
  src/cast_string_to_datetime.cu
  src/cast_string_to_float.cu
  src/charset_decode.cu
  src/datetime_rebase.cu
  src/datetime_truncate.cu
  src/decimal_utils.cu
  src/exception_with_row_index_utilities.cu
  src/find_in_set.cu
  src/format_float.cu
  src/from_json_to_raw_map.cu
  src/from_json_to_structs.cu
  src/get_json_object.cu
  src/hex.cu
  src/hash/HashJni.cpp
  src/hash/hive_hash.cu
  src/hash/murmur_hash.cu
  src/hash/sha.cpp
  src/hash/xxhash64.cu
  src/histogram.cu
  src/hyper_log_log_plus_plus.cu
  src/hyper_log_log_plus_plus_host_udf.cu
  src/iceberg/iceberg_bucket.cu
  src/iceberg/iceberg_datetime_util.cu
  src/iceberg/iceberg_truncate.cu
  src/json_utils.cu
  src/join_primitives.cu
  src/list_slice.cu
  src/map.cu
  src/map_utils.cu
  src/map_zip_with_utils.cu
  src/multiply.cu
  src/number_converter.cu
  src/parse_timestamp_with_format.cu
  src/parse_uri.cu
  src/protobuf/protobuf.cu
  src/protobuf/protobuf_builders.cu
  src/protobuf/protobuf_kernels.cu
  src/regex_rewrite_utils.cu
  src/row_conversion.cu
  src/round_float.cu
  src/shuffle_assemble.cu
  src/shuffle_split.cu
  src/substring_index.cu
  src/timezones.cu
  src/utilities.cu
  src/uuid.cu
  src/zorder.cu
)

set_target_properties(
  spark_rapids_jni
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
             CUDA_RUNTIME_LIBRARY Static
)

target_compile_options(
  spark_rapids_jni PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                           "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)

# Specify include paths for the current target and dependents
target_include_directories(
  spark_rapids_jni
  PRIVATE "${SPARK_RAPIDS_JNI_SOURCE_DIR}/src"
          "${JNI_INCLUDE_DIRS}"
          "${CUDFJNI_INCLUDE_DIRS}"
          "${CUDAToolkit_INCLUDE_DIRS}"
)

target_compile_definitions(
  spark_rapids_jni PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_DEFINITIONS}>"
                          "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_DEFINITIONS}>>"
)

# Per-thread default stream
if(CUDF_USE_PER_THREAD_DEFAULT_STREAM)
  target_compile_definitions(
    spark_rapids_jni PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM  CUDF_USE_PER_THREAD_DEFAULT_STREAM
  )
endif()

target_link_libraries(
  spark_rapids_jni PRIVATE
  -Wl,--whole-archive
    ${CUDFJNI_LIB}
    cudf::cudf
    nvtx3::nvtx3-cpp
  -Wl,--no-whole-archive
    cuco::cuco
    spdlog::spdlog_header_only
    ${ARROW_LIB}
    ${PARQUET_LIB}
    ${THRIFT_LIB}
)
rapids_cuda_set_runtime(spark_rapids_jni USE_STATIC ON)
set_target_properties(spark_rapids_jni PROPERTIES LINK_LANGUAGE "CXX")
# For backwards-compatibility with the cudf Java bindings and RAPIDS accelerated UDFs,
# all of the code is built into libcudf.so that is statically linked to the CUDA runtime library.
# libcudfjni.so is a stub library dynamically-linked to libcudf.so.
set_target_properties(spark_rapids_jni PROPERTIES OUTPUT_NAME "cudf")


add_library(cudfjnistub SHARED src/emptyfile.cpp)
set_target_properties(cudfjnistub PROPERTIES OUTPUT_NAME "cudfjni")
target_link_libraries(cudfjnistub -Wl,--no-as-needed $<TARGET_FILE:spark_rapids_jni> -Wl,--as-needed)
add_dependencies(cudfjnistub spark_rapids_jni)

# ##################################################################################################
# * NVML JNI ---------------------------------------------------------------------------------------

add_library(nvmljni SHARED src/NVMLJni.cpp)
set_target_properties(
  nvmljni
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_include_directories(
  nvmljni
  PRIVATE "${JNI_INCLUDE_DIRS}"
          "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(nvmljni nvidia-ml)

# ##################################################################################################
# * cuFile JNI -------------------------------------------------------------------------------------

if(USE_GDS)
  find_library(CUFILEJNI_LIB "libcufilejni.a" REQUIRED NO_DEFAULT_PATH
    HINTS "${CUDFJNI_BUILD_DIR}"
  )
  add_library(cufilejni SHARED src/emptyfile.cpp)
  set_target_properties(
    cufilejni
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               # set target compile options
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
               CUDA_RUNTIME_LIBRARY Static
  )
  target_link_libraries(
    cufilejni PRIVATE
    -Wl,--whole-archive
      ${CUFILEJNI_LIB}
    -Wl,--no-whole-archive
    spark_rapids_jni
    CUDA::cuFile_static
  )
  rapids_cuda_set_runtime(cufilejni USE_STATIC ON)
endif()

# ##################################################################################################
# * add tests -------------------------------------------------------------------------------------

if(SPARK_RAPIDS_JNI_BUILD_TESTS)
  # include CTest module -- automatically calls enable_testing()
  include(CTest)
  add_subdirectory(tests)
endif()

# ##################################################################################################
# * add benchmarks --------------------------------------------------------------------------------

if(SPARK_RAPIDS_JNI_BUILD_BENCHMARKS)
  # Find or install NVBench
  include(${rapids-cmake-dir}/cpm/nvbench.cmake)
  rapids_cpm_nvbench()
  add_subdirectory(benchmarks)
endif()

if(SPARK_RAPIDS_JNI_BUILD_FAULTINJ)
  add_subdirectory(faultinj)
endif()

if(BUILD_PROFILER)
  add_subdirectory(profiler)
endif()
