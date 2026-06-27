# =============================================================================
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

include(../cmake/get_flatbuffers.cmake)

# ##################################################################################################
# * flatbuffer generation---------------------------------------------------------------------------

set(SPARK_RAPIDS_JNI_FBS_DIR "${SPARK_RAPIDS_JNI_SOURCE_DIR}/../fbs")
add_custom_command(
  OUTPUT ${SPARK_RAPIDS_JNI_GENERATED_INCLUDE_DIR}/profiler_generated.h
  DEPENDS "${SPARK_RAPIDS_JNI_FBS_DIR}/profiler.fbs"
  WORKING_DIRECTORY "${SPARK_RAPIDS_JNI_FBS_DIR}"
  VERBATIM
  COMMAND ${CMAKE_COMMAND} -E make_directory "${SPARK_RAPIDS_JNI_GENERATED_INCLUDE_DIR}"
  COMMAND
    $<TARGET_FILE:flatbuffers::flatc> --cpp -o "${SPARK_RAPIDS_JNI_GENERATED_INCLUDE_DIR}" profiler.fbs
  COMMENT "Generating profiler flatbuffer code"
)

# ##################################################################################################
# * profiler JNI -----------------------------------------------------------------------------------

add_library(profilerjni SHARED
  ProfilerJni.cpp
  profiler_debug.cpp
  profiler_serializer.cpp
  "${SPARK_RAPIDS_JNI_GENERATED_SOURCE_DIR}/spark_rapids_jni_version.cpp"
  "${SPARK_RAPIDS_JNI_GENERATED_INCLUDE_DIR}/profiler_generated.h"
)

set_target_properties(
  profilerjni
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_VISIBILITY_PRESET "hidden"
             VISIBILITY_INLINES_HIDDEN TRUE
)

target_include_directories(
  profilerjni
  PRIVATE "${JNI_INCLUDE_DIRS}"
          "${CUDAToolkit_INCLUDE_DIRS}"
          "${SPARK_RAPIDS_JNI_GENERATED_INCLUDE_DIR}"
          "${SPARK_RAPIDS_JNI_SOURCE_DIR}/src"
)

spark_rapids_jni_add_strict_warning_flags(profilerjni)

find_library(CUPTI_LIBRARY_PATH cupti_static PATHS
  "/usr/local/cuda/lib64"
  "/usr/local/cuda/extras/CUPTI/lib64"
)

target_link_libraries(profilerjni
  PRIVATE ${CUPTI_LIBRARY_PATH} nvtx3::nvtx3-cpp flatbuffers::flatbuffers
)

file(READ "${SPARK_RAPIDS_JNI_FBS_DIR}/profiler.fbs" SPARK_RAPIDS_JNI_PROFILER_SCHEMA)
configure_file(
  profiler_schema.cpp.in
  "${SPARK_RAPIDS_JNI_GENERATED_SOURCE_DIR}/profiler_schema.cpp"
  @ONLY
)

add_executable(spark_rapids_profile_converter
  spark_rapids_profile_converter.cpp
  init_nvtxw.cpp
  nvtxw3.cpp
  NvtxwEvents.cpp
  "${SPARK_RAPIDS_JNI_GENERATED_SOURCE_DIR}/profiler_schema.cpp"
  "${SPARK_RAPIDS_JNI_GENERATED_SOURCE_DIR}/spark_rapids_jni_version.cpp"
  "${SPARK_RAPIDS_JNI_GENERATED_INCLUDE_DIR}/profiler_generated.h"
)

target_include_directories(
  spark_rapids_profile_converter
  PRIVATE
  "${CUDAToolkit_INCLUDE_DIRS}"
  "${SPARK_RAPIDS_JNI_SOURCE_DIR}/profiler"
  "${SPARK_RAPIDS_JNI_SOURCE_DIR}/src"
  "${SPARK_RAPIDS_JNI_GENERATED_INCLUDE_DIR}"
)

spark_rapids_jni_add_strict_warning_flags(spark_rapids_profile_converter)

target_link_libraries(spark_rapids_profile_converter
  "${CUPTI_LIBRARY_PATH}"
  flatbuffers::flatbuffers
  dl
  pthread
  rt
  -static-libstdc++
  -static-libgcc)
