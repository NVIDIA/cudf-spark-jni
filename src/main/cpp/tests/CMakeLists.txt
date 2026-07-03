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

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

# We need to include the source code defined in cudftestutil_impl, but it is an interface library
# that does not compile, we need to define an object library for the compilation. By doing so, we
# only compile such source files once instead of doing so for every test executable.
add_library(spark_rapids_jni_test_common OBJECT)

target_compile_options(spark_rapids_jni_test_common
                PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
target_link_libraries(
    spark_rapids_jni_test_common
    PUBLIC cudf::cudftestutil GTest::gmock GTest::gmock_main GTest::gtest GTest::gtest_main
    PRIVATE cudf::cudftestutil_impl
  )

function(ConfigureTest CMAKE_TEST_NAME)
    add_executable(${CMAKE_TEST_NAME} ${ARGN})
    target_compile_options(${CMAKE_TEST_NAME}
                PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
    target_include_directories(${CMAKE_TEST_NAME}
                PRIVATE "$<BUILD_INTERFACE:${SPARK_RAPIDS_JNI_SOURCE_DIR}>"
                        "$<BUILD_INTERFACE:${SPARK_RAPIDS_JNI_SOURCE_DIR}/src>")
    set_target_properties(
        ${CMAKE_TEST_NAME}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/gtests>"
                   INSTALL_RPATH "\$ORIGIN/../../../lib"
                   CXX_STANDARD 20
                   CXX_STANDARD_REQUIRED ON
                   # For std:: support of __int128_t. Can be removed once using cuda::std
                   CXX_EXTENSIONS ON
                   CUDA_STANDARD 20
                   CUDA_STANDARD_REQUIRED ON
    )
    target_compile_definitions(
        ${CMAKE_TEST_NAME} PRIVATE THRUST_FORCE_32_BIT_OFFSET_TYPE=1 CCCL_AVOID_SORT_UNROLL=1
    )
    target_link_libraries(${CMAKE_TEST_NAME} PRIVATE spark_rapids_jni_test_common
                                                     spark_rapids_jni)
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
    install(
        TARGETS ${CMAKE_TEST_NAME}
        COMPONENT testing
        DESTINATION bin/gtests
        EXCLUDE_FROM_ALL
    )
endfunction(ConfigureTest)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

ConfigureTest(CAST_STRING
    cast_string.cpp)

ConfigureTest(CAST_DECIMAL_TO_STRING
    cast_decimal_to_string.cpp)

ConfigureTest(EXCEPTION_WITH_ROW_INDEX
    exception_with_row_index.cpp)

ConfigureTest(FORMAT_FLOAT
    format_float.cpp)

ConfigureTest(CAST_FLOAT_TO_STRING
    cast_float_to_string.cpp)

ConfigureTest(CAST_LONG_TO_BINARY_STRING
    cast_long_to_binary_string.cpp)

ConfigureTest(DATETIME_REBASE
    datetime_rebase.cpp)

ConfigureTest(ROW_CONVERSION
    row_conversion.cpp)

ConfigureTest(HASH
    hash.cpp)

ConfigureTest(BLOOM_FILTER
    bloom_filter.cu)

ConfigureTest(TIMEZONES
    timezones.cpp)

ConfigureTest(UTILITIES
    utilities.cpp)

ConfigureTest(PARSE_URI
    parse_uri.cpp)

ConfigureTest(PROTOBUF_HELPERS
    protobuf_helpers.cu)

ConfigureTest(LIST_SLICE
    list_slice.cpp)

ConfigureTest(MAP_ZIP_WITH
    map_zip_with_utils.cpp)

ConfigureTest(MAP_UTILS
    map_utils.cpp)

ConfigureTest(SUBSTRING_INDEX
    substring_index.cpp)

ConfigureTest(SHUFFLE_SPLIT
    shuffle_split.cu)
    
ConfigureTest(HLLPP
    hyper_log_log_plus_plus.cu)

ConfigureTest(CONV
    number_converter.cpp)

ConfigureTest(MULTIPLY
    multiply.cpp)

ConfigureTest(FROM_JSON
    from_json.cpp)
