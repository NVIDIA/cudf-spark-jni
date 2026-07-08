#!/bin/bash
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

#
# Script to build native code in cudf and spark-rapids-jni
#

set -e

if [[ $FROM_MAVEN == "true" ]]; then
  echo "Building native libraries. To rerun outside Maven enter the build environment via

$ ./build/run-in-docker

then run

$ REUSE_ENV=true $0
"
fi

# Disable items on arm64 due to missing dependencies in the CUDA toolkit
if [ "$(uname -m)" == "aarch64" ]; then
 USE_GDS="OFF" # cuFile RDMA libraries are missing
 BUILD_FAULTINJ="OFF" # libcupti_static.a is missing
fi

# Environment variables to control the build
PROJECT_BASE_DIR=${PROJECT_BASE_DIR:-$(realpath $(dirname $0)/..)}
PROJECT_BUILD_DIR=${PROJECT_BUILD_DIR:-$PROJECT_BASE_DIR/target}
if [[ "$REUSE_ENV" != "true" ]]; then
  echo "
BUILD_BENCHMARKS=${BUILD_BENCHMARKS:-ON}
BUILD_CUDF_BENCHMARKS=${BUILD_CUDF_BENCHMARKS:-OFF}
BUILD_CUDF_TESTS=${BUILD_CUDF_TESTS:-OFF}
BUILD_FAULTINJ=${BUILD_FAULTINJ:-ON}
BUILD_PROFILER=${BUILD_PROFILER:-ON}
BUILD_TESTS=${BUILD_TESTS:-ON}
CMAKE_EXPORT_COMPILE_COMMANDS=${CMAKE_EXPORT_COMPILE_COMMANDS:-ON}
export CMAKE_GENERATOR=${CMAKE_GENERATOR:-Ninja}
CPP_PARALLEL_LEVEL=${CPP_PARALLEL_LEVEL:-10}
CUDF_BUILD_TYPE=${CUDF_BUILD_TYPE:-Release}
CUDF_PATH=${CUDF_PATH:-$PROJECT_BASE_DIR/thirdparty/cudf}
CUDF_PIN_PATH=${CUDF_PIN_PATH:-$PROJECT_BASE_DIR/thirdparty/cudf-pins}
CUDF_USE_PER_THREAD_DEFAULT_STREAM=${CUDF_USE_PER_THREAD_DEFAULT_STREAM:-ON}
GPU_ARCHS=${GPU_ARCHS:-DEPRECATED}
CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES:-RAPIDS}
LIBCUDF_BUILD_CONFIGURE=${LIBCUDF_BUILD_CONFIGURE:-false}
LIBCUDF_BUILD_PATH=${LIBCUDF_BUILD_PATH:-$PROJECT_BUILD_DIR/libcudf/cmake-build}
LIBCUDF_DEPENDENCY_MODE=${LIBCUDF_DEPENDENCY_MODE:-pinned}
LIBCUDF_INSTALL_PATH=${LIBCUDF_INSTALL_PATH:-$PROJECT_BUILD_DIR/libcudf-install}
LIBCUDFJNI_BUILD_PATH=${LIBCUDFJNI_BUILD_PATH:-$PROJECT_BUILD_DIR/libcudfjni}
SPARK_JNI_BUILD_PATH=${SPARK_JNI_BUILD_PATH:-$PROJECT_BUILD_DIR/jni/cmake-build}
RMM_LOGGING_LEVEL=${RMM_LOGGING_LEVEL:-OFF}
USE_GDS=${USE_GDS:-OFF}
LIBCUDF_CONFIGURE_ONLY=${LIBCUDF_CONFIGURE_ONLY:-OFF}" > "$PROJECT_BUILD_DIR/buildcpp-env.sh"
fi

source "$PROJECT_BUILD_DIR/buildcpp-env.sh"

if [[ "$GPU_ARCHS" != "DEPRECATED" ]]; then
    CMAKE_CUDA_ARCHITECTURES="$GPU_ARCHS"    
    echo "==========================================================================================
WARNING: CMAKE_CUDA_ARCHITECTURES is overridden by GPU_ARCHS.
         GPU_ARCHS is deprecated. Please use CMAKE_CUDA_ARCHITECTURES instead.
=========================================================================================="
fi

#
# Function to create symlink to compile_commands.json for IDE/clangd discovery
# (similar to NVBenchClangdCompileInfo.cmake)
#
create_compile_commands_symlink() {
  local build_dir=$1
  local source_dir=$2
  local compile_commands_file="$build_dir/compile_commands.json"
  local compile_commands_link="$source_dir/compile_commands.json"
  
  echo "Creating symlink from $compile_commands_link to $compile_commands_file..."
  ln -sf "$compile_commands_file" "$compile_commands_link"
}

#
# libcudf build
#
mkdir -p "$LIBCUDF_INSTALL_PATH" "$LIBCUDF_BUILD_PATH"
cd "$LIBCUDF_BUILD_PATH"

# Skip explicit cudf cmake configuration if it appears it has already configured
if [[ $LIBCUDF_BUILD_CONFIGURE == true || ! -f $LIBCUDF_BUILD_PATH/CMakeCache.txt ]]; then
  echo "Configuring cudf native libs"
  cmake "$CUDF_PATH/cpp" \
    -DBUILD_BENCHMARKS="$BUILD_CUDF_BENCHMARKS" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS="$CMAKE_EXPORT_COMPILE_COMMANDS" \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTS="$BUILD_CUDF_TESTS" \
    -DCMAKE_BUILD_TYPE="$CUDF_BUILD_TYPE" \
    -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
    -DCMAKE_INSTALL_PREFIX="$LIBCUDF_INSTALL_PATH" \
    -DCUDF_DEPENDENCY_PIN_MODE="$LIBCUDF_DEPENDENCY_MODE" \
    -DCUDA_STATIC_CUFILE=ON \
    -DCUDA_STATIC_RUNTIME=ON \
    -DCUDF_USE_PER_THREAD_DEFAULT_STREAM="$CUDF_USE_PER_THREAD_DEFAULT_STREAM" \
    -DCUDF_KVIKIO_REMOTE_IO=OFF \
    -DCUDF_LARGE_STRINGS_DISABLED=ON \
    -DCUDF_BUILD_STATIC_DEPS=FORCE \
    -DLIBCUDF_LOGGING_LEVEL="$RMM_LOGGING_LEVEL" \
    -DRMM_LOGGING_LEVEL="$RMM_LOGGING_LEVEL" \
    -C="$CUDF_PIN_PATH/setup.cmake"
fi
if [[ $LIBCUDF_CONFIGURE_ONLY == ON ]]; then # submodule-sync.sh phase 1 call this script with LIBCUDF_CONFIGURE_ONLY=ON
  echo "Skip build..."
  exit 0
fi
echo "Building cudf native libs"
cmake --build "$LIBCUDF_BUILD_PATH" --target install "-j$CPP_PARALLEL_LEVEL"

#
# libcudfjni build
#
mkdir -p "$LIBCUDFJNI_BUILD_PATH"
cd "$LIBCUDFJNI_BUILD_PATH"
echo "Configuring cudfjni native libs"
CUDF_INSTALL_DIR="$LIBCUDF_INSTALL_PATH" cmake \
  "$CUDF_PATH/java/src/main/native" \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_EXPORT_COMPILE_COMMANDS="$CMAKE_EXPORT_COMPILE_COMMANDS" \
  -DCUDA_STATIC_CUFILE=ON \
  -DCUDA_STATIC_RUNTIME=ON \
  -DCUDF_DEPENDENCY_PIN_MODE=pinned \
  -DCUDF_JNI_LIBCUDF_STATIC=ON \
  -DCUDF_USE_PER_THREAD_DEFAULT_STREAM="$CUDF_USE_PER_THREAD_DEFAULT_STREAM" \
  -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
  -DRMM_LOGGING_LEVEL="$RMM_LOGGING_LEVEL" \
  -DUSE_GDS="$USE_GDS" \
  -C="$CUDF_PIN_PATH/setup.cmake"

create_compile_commands_symlink "$LIBCUDFJNI_BUILD_PATH" "$CUDF_PATH/java/src/main/native"

echo "Building cudfjni native libs"
cmake --build "$LIBCUDFJNI_BUILD_PATH" "-j$CPP_PARALLEL_LEVEL"

#
# sparkjni build
#
mkdir -p "$SPARK_JNI_BUILD_PATH"
cd "$SPARK_JNI_BUILD_PATH"
echo "Configuring spark-rapids-jni native libs"
CUDF_ROOT="$CUDF_PATH" \
  CUDF_INSTALL_DIR="$LIBCUDF_INSTALL_PATH" \
  CUDFJNI_BUILD_DIR="$LIBCUDFJNI_BUILD_PATH" \
  cmake \
    "$PROJECT_BASE_DIR/src/main/cpp" \
    -DBUILD_BENCHMARKS="$BUILD_BENCHMARKS" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS="$CMAKE_EXPORT_COMPILE_COMMANDS" \
    -DBUILD_FAULTINJ="$BUILD_FAULTINJ" \
    -DBUILD_PROFILER="$BUILD_PROFILER" \
    -DBUILD_TESTS="$BUILD_TESTS" \
    -DCUDF_DEPENDENCY_PIN_MODE=pinned \
    -DCUDF_USE_PER_THREAD_DEFAULT_STREAM="$CUDF_USE_PER_THREAD_DEFAULT_STREAM" \
    -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
    -DRMM_LOGGING_LEVEL="$RMM_LOGGING_LEVEL" \
    -DUSE_GDS="$USE_GDS" \
    -C="$CUDF_PIN_PATH/setup.cmake"

create_compile_commands_symlink "$SPARK_JNI_BUILD_PATH" "$PROJECT_BASE_DIR/src/main/cpp"

echo "Building spark-rapids-jni native libs"
cmake --build "$SPARK_JNI_BUILD_PATH" "-j$CPP_PARALLEL_LEVEL"
