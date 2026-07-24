#
# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
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
#

if(PROJECT_NAME STREQUAL arrow)
  if(DEFINED ENV{ARROW_THRIFT_MIRROR_URL} AND NOT DEFINED ENV{ARROW_THRIFT_URL})
    file(STRINGS "${PROJECT_SOURCE_DIR}/thirdparty/versions.txt" arrow_thrift_version
      REGEX "^ARROW_THRIFT_BUILD_VERSION=")
    string(REPLACE "ARROW_THRIFT_BUILD_VERSION=" "" arrow_thrift_version "${arrow_thrift_version}")
    set(ENV{ARROW_THRIFT_URL}
      "$ENV{ARROW_THRIFT_MIRROR_URL}/thrift/${arrow_thrift_version}/thrift-${arrow_thrift_version}.tar.gz")
  endif()
  return()
endif()

string(TOLOWER "${CUDF_DEPENDENCY_PIN_MODE}" CUDF_DEPENDENCY_PIN_MODE)
if(NOT (CUDF_DEPENDENCY_PIN_MODE STREQUAL pinned OR
        CUDF_DEPENDENCY_PIN_MODE STREQUAL latest))
  message(FATAL_ERROR "The CUDF_DEPENDENCY_PIN_MODE variable must be set to either `pinned` or `latest`.")
 endif()

function(set_rapids_cmake_pin_sha1)
  set(rapids-cmake-sha "${rapids-cmake-sha}" PARENT_SCOPE)

  message(STATUS "Pinning rapids-cmake SHA1 to ${rapids-cmake-sha}")
endfunction()

# We need to set the rapids-cmake SHA1 before any CMake code in libcudf is executed when
# we are in pin mode. Otherwise we will use the latest rapids-cmake version since that
# is what cudf does via `fetch_rapids.cmake`
if(CUDF_DEPENDENCY_PIN_MODE STREQUAL pinned)
  # Extract the rapids sha1 from the file
  file(READ "${CMAKE_CURRENT_LIST_DIR}/rapids-cmake.sha" rapids-cmake-sha)
  string(STRIP rapids-cmake-sha "${rapids-cmake-sha}")
  string(REPLACE "\n" "" rapids-cmake-sha "${rapids-cmake-sha}")
  set(rapids-cmake-sha "${rapids-cmake-sha}" CACHE STRING "rapids-cmake sha to use" FORCE)
  message(STATUS "Pinning rapids-cmake SHA1 [${rapids-cmake-sha}]")
else()
  set(rapids-cmake-fetch-via-git "ON" CACHE STRING "Make sure rapids-cmake is cloned so we can get SHA value" FORCE)
endif()

# We need to use a project() call hook, since rapids-cmake cpm_init()
# can't be called from a `-C` CMake file
set(CMAKE_PROJECT_TOP_LEVEL_INCLUDES "${CMAKE_CURRENT_LIST_DIR}/add_dependency_pins.cmake" CACHE FILEPATH "" )
set(CMAKE_PROJECT_arrow_INCLUDE "${CMAKE_CURRENT_LIST_FILE}" CACHE FILEPATH "")
