#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

# This hook runs after Arrow's project() call, when its dependency metadata is available but before
# ThirdpartyToolchain.cmake resolves the Thrift download URL.
if(DEFINED ENV{ARROW_THRIFT_URL} AND NOT "$ENV{ARROW_THRIFT_URL}" STREQUAL "")
  return()
endif()

if(NOT DEFINED ENV{ARROW_THRIFT_MIRROR_URL} OR "$ENV{ARROW_THRIFT_MIRROR_URL}" STREQUAL "")
  return()
endif()

set(_arrow_versions_file "${PROJECT_SOURCE_DIR}/thirdparty/versions.txt")
if(NOT EXISTS "${_arrow_versions_file}")
  message(FATAL_ERROR "Arrow dependency versions file not found: ${_arrow_versions_file}")
endif()

file(
  STRINGS "${_arrow_versions_file}" _arrow_thrift_version_entries
  REGEX "^ARROW_THRIFT_BUILD_VERSION="
)
list(LENGTH _arrow_thrift_version_entries _arrow_thrift_version_entry_count)
if(NOT _arrow_thrift_version_entry_count EQUAL 1)
  message(
    FATAL_ERROR
      "Expected one ARROW_THRIFT_BUILD_VERSION entry in ${_arrow_versions_file}, "
      "found ${_arrow_thrift_version_entry_count}"
  )
endif()

list(GET _arrow_thrift_version_entries 0 _arrow_thrift_version_entry)
string(
  REGEX REPLACE "^ARROW_THRIFT_BUILD_VERSION=" "" _arrow_thrift_version
                "${_arrow_thrift_version_entry}"
)
if(NOT _arrow_thrift_version MATCHES "^[0-9A-Za-z._+-]+$")
  message(FATAL_ERROR "Invalid ARROW_THRIFT_BUILD_VERSION in ${_arrow_versions_file}")
endif()

set(_arrow_thrift_mirror_url "$ENV{ARROW_THRIFT_MIRROR_URL}")
string(REGEX REPLACE "/+$" "" _arrow_thrift_mirror_url "${_arrow_thrift_mirror_url}")
if(_arrow_thrift_mirror_url STREQUAL "")
  message(FATAL_ERROR "ARROW_THRIFT_MIRROR_URL must contain a non-empty URL")
endif()

set(
  ENV{ARROW_THRIFT_URL}
  "${_arrow_thrift_mirror_url}/thrift/${_arrow_thrift_version}/thrift-${_arrow_thrift_version}.tar.gz"
)
message(STATUS "Using the configured mirror for Arrow Thrift ${_arrow_thrift_version}")
