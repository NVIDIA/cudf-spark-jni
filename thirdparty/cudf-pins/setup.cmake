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

# Redirect Arrow's bundled Thrift download to an optional mirror without hard-coding its
# pinned version or overriding an explicit ARROW_THRIFT_URL.
if(PROJECT_NAME STREQUAL arrow)
  if(DEFINED ENV{ARROW_THRIFT_MIRROR_URL} AND NOT DEFINED ENV{ARROW_THRIFT_URL})
    file(STRINGS "${PROJECT_SOURCE_DIR}/thirdparty/versions.txt" arrow_thrift_version
      REGEX "^ARROW_THRIFT_BUILD_VERSION=")
    string(REPLACE "ARROW_THRIFT_BUILD_VERSION=" "" arrow_thrift_version "${arrow_thrift_version}")
    set(ENV{ARROW_THRIFT_URL}
      "$ENV{ARROW_THRIFT_MIRROR_URL}/thrift/${arrow_thrift_version}/thrift-${arrow_thrift_version}.tar.gz")
  endif()
  if(NOT "$ENV{JNI_DOWNLOAD_CACHE}" STREQUAL "")
    include("${CMAKE_CURRENT_LIST_DIR}/download_cache.cmake")
    # Read the checked-out (and possibly patched) Arrow pins, not a second
    # version table. Explicit caller URLs and the Thrift mirror above win.
    file(STRINGS "${PROJECT_SOURCE_DIR}/thirdparty/versions.txt" archive_versions
      REGEX "^ARROW_(RAPIDJSON|THRIFT|XSIMD)_BUILD_(VERSION|SHA256_CHECKSUM)=")
    foreach(entry IN LISTS archive_versions)
      if(entry MATCHES "^([A-Z0-9_]+)=([A-Za-z0-9_.-]+)$")
        set(${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
      endif()
    endforeach()
    set(archive_RAPIDJSON "https://github.com/miloyip/rapidjson/archive/${ARROW_RAPIDJSON_BUILD_VERSION}.tar.gz")
    set(archive_THRIFT "https://www.apache.org/dyn/closer.lua/thrift/${ARROW_THRIFT_BUILD_VERSION}/thrift-${ARROW_THRIFT_BUILD_VERSION}.tar.gz?action=download")
    set(archive_XSIMD "https://github.com/xtensor-stack/xsimd/archive/${ARROW_XSIMD_BUILD_VERSION}.tar.gz")
    foreach(dependency RAPIDJSON THRIFT XSIMD)
      if(NOT DEFINED ENV{ARROW_${dependency}_URL} AND
          ARROW_${dependency}_BUILD_VERSION AND ARROW_${dependency}_BUILD_SHA256_CHECKSUM)
        jni_cached_archive("${dependency}.tar.gz" "${archive_${dependency}}"
          "${ARROW_${dependency}_BUILD_SHA256_CHECKSUM}" cached_archive)
        if(cached_archive)
          set(ENV{ARROW_${dependency}_URL} "${cached_archive}")
        endif()
      endif()
    endforeach()
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
# Remove only our previous URL so reconfiguration honors new pins, opt-out and
# latest mode. A URL explicitly changed by the caller is left untouched.
if(DEFINED JNI_CACHED_RAPIDS_URL AND rapids-cmake-url STREQUAL JNI_CACHED_RAPIDS_URL)
  unset(rapids-cmake-url CACHE)
  unset(rapids-cmake-value-to-clone CACHE)
endif()
unset(JNI_CACHED_RAPIDS_URL CACHE)
if(CUDF_DEPENDENCY_PIN_MODE STREQUAL pinned)
  # Extract the rapids sha1 from the file
  file(READ "${CMAKE_CURRENT_LIST_DIR}/rapids-cmake.sha" rapids-cmake-sha)
  string(STRIP rapids-cmake-sha "${rapids-cmake-sha}")
  string(REPLACE "\n" "" rapids-cmake-sha "${rapids-cmake-sha}")
  set(rapids-cmake-sha "${rapids-cmake-sha}" CACHE STRING "rapids-cmake sha to use" FORCE)
  message(STATUS "Pinning rapids-cmake SHA1 [${rapids-cmake-sha}]")
  if(NOT "$ENV{JNI_DOWNLOAD_CACHE}" STREQUAL "" AND
      NOT rapids-cmake-url AND NOT rapids-cmake-repo AND NOT rapids-cmake-fetch-via-git)
    include("${CMAKE_CURRENT_LIST_DIR}/download_cache.cmake")
    jni_cached_archive("rapids-cmake.zip"
      "https://github.com/rapidsai/rapids-cmake/archive/${rapids-cmake-sha}.zip"
      "" cached_archive)
    if(cached_archive)
      # Only cache variables survive this -C preload script. Use RAPIDS' URL
      # override; its own FetchContent still extracts/configures in this build.
      set(rapids-cmake-url "${cached_archive}" CACHE STRING "RAPIDS bootstrap archive")
      set(rapids-cmake-value-to-clone "" CACHE STRING "No suffix for a local archive" FORCE)
      set(JNI_CACHED_RAPIDS_URL "${cached_archive}" CACHE INTERNAL "JNI-owned URL" FORCE)
    endif()
  endif()
else()
  set(rapids-cmake-fetch-via-git "ON" CACHE STRING "Make sure rapids-cmake is cloned so we can get SHA value" FORCE)
endif()

# We need to use a project() call hook, since rapids-cmake cpm_init()
# can't be called from a `-C` CMake file
set(CMAKE_PROJECT_TOP_LEVEL_INCLUDES "${CMAKE_CURRENT_LIST_DIR}/add_dependency_pins.cmake" CACHE FILEPATH "" )
set(CMAKE_PROJECT_arrow_INCLUDE "${CMAKE_CURRENT_LIST_FILE}" CACHE FILEPATH "")
