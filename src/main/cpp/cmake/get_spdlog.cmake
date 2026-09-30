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

function(find_and_configure_spdlog)
  set(spdlog_version 1.14.1)
  rapids_cpm_find(
    spdlog "${spdlog_version}"
    GLOBAL_TARGETS spdlog::spdlog_header_only
    CPM_ARGS
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG "v${spdlog_version}"
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL ON
    OPTIONS "SPDLOG_INSTALL OFF" "SPDLOG_USE_STD_FORMAT ON" "CMAKE_POSITION_INDEPENDENT_CODE ON"
      "SPDLOG_FMT_EXTERNAL OFF" "SPDLOG_FMT_EXTERNAL_HO OFF" 
  )
endfunction()

find_and_configure_spdlog()
