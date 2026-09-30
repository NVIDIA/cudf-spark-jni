#
# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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

if(NOT DEFINED rapids-cmake-dir)
  include(../cudf/cmake/rapids_config.cmake)
endif()

include(rapids-cpm)
rapids_cpm_init()

function(add_override_if_requested)
  if(CUDF_DEPENDENCY_PIN_MODE STREQUAL pinned)
    include(${rapids-cmake-dir}/cpm/package_override.cmake)
    rapids_cpm_package_override(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/versions.json)

    message(STATUS "Pinning CUDF dependencies to values found in ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/versions.json")
  else()
    include(${rapids-cmake-dir}/cpm/generate_pinned_versions.cmake)
    rapids_cpm_generate_pinned_versions(OUTPUT ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/versions.json)

    message(STATUS "Building with latest CUDF dependencies (saving pinned versions to ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/versions.json)")
  endif()
endfunction()
add_override_if_requested()
