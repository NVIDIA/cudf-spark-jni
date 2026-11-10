# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Cache only archive bytes. Upstream still extracts, patches and configures them
# in this build's workspace; no generated CMake state is reused.
function(jni_cached_archive name url sha256 output)
  set(${output} "" PARENT_SCOPE)
  if("$ENV{JNI_DOWNLOAD_CACHE}" STREQUAL "")
    return()
  endif()
  string(SHA256 key "${url}|${sha256}")
  set(archive "$ENV{JNI_DOWNLOAD_CACHE}/${key}/${name}")
  set(auth_args)
  foreach(option NETRC NETRC_FILE TLS_CAINFO)
    if(DEFINED CMAKE_${option})
      list(APPEND auth_args "-DCMAKE_${option}=${CMAKE_${option}}")
    endif()
  endforeach()
  # Isolate optional-cache I/O errors (including disk/permission errors) from
  # configuration. Never print libcurl's potentially credential-bearing errors.
  execute_process(
    COMMAND "${CMAKE_COMMAND}" "-Darchive=${archive}" "-Durl=${url}"
      "-Dsha256=${sha256}" ${auth_args} -P "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
    RESULT_VARIABLE status OUTPUT_VARIABLE result ERROR_QUIET TIMEOUT 90)
  if(status STREQUAL "0")
    string(STRIP "${result}" result)
    message(STATUS "JNI download cache ${result}: ${name}")
    set(${output} "${archive}" PARENT_SCOPE)
  else()
    message(STATUS "JNI download cache unavailable: ${name}; using upstream download")
  endif()
endfunction()

# Private subprocess entry point. A lock plus atomic rename excludes partial
# downloads. Checksums supplied by Arrow take priority over our saved digest.
if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
  cmake_minimum_required(VERSION 3.25)
  get_filename_component(directory "${archive}" DIRECTORY)
  file(MAKE_DIRECTORY "${directory}")
  file(LOCK "${directory}/lock" GUARD PROCESS TIMEOUT 5)
  set(expected "${sha256}")
  if(NOT expected AND EXISTS "${archive}.sha256")
    file(READ "${archive}.sha256" expected)
  endif()
  if(EXISTS "${archive}" AND expected)
    file(SHA256 "${archive}" actual)
    if(actual STREQUAL expected)
      execute_process(COMMAND "${CMAKE_COMMAND}" -E echo hit)
      return()
    endif()
  endif()

  file(DOWNLOAD "${url}" "${archive}.part" STATUS status
    TIMEOUT 60 INACTIVITY_TIMEOUT 15 TLS_VERIFY ON)
  list(GET status 0 code)
  if(NOT code EQUAL 0)
    file(REMOVE "${archive}.part")
    message(FATAL_ERROR "Archive download failed")
  endif()
  file(SHA256 "${archive}.part" actual)
  if(sha256 AND NOT actual STREQUAL sha256)
    file(REMOVE "${archive}.part")
    message(FATAL_ERROR "Archive checksum mismatch")
  endif()
  # RAPIDS bootstrap supplies a commit pin but no archive checksum. Reject
  # truncated/error documents before recording a digest for subsequent reuse.
  execute_process(COMMAND "${CMAKE_COMMAND}" -E tar tf "${archive}.part"
    RESULT_VARIABLE status OUTPUT_QUIET ERROR_QUIET)
  if(NOT status EQUAL 0)
    file(REMOVE "${archive}.part")
    message(FATAL_ERROR "Invalid archive")
  endif()
  file(WRITE "${archive}.sha256" "${actual}")
  file(RENAME "${archive}.part" "${archive}")
  execute_process(COMMAND "${CMAKE_COMMAND}" -E echo stored)
endif()
