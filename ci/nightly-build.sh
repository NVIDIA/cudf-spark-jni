#!/bin/bash
#
# Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
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

set -ex

nvidia-smi

git submodule update --init --recursive

MVN_SETTINGS=${MVN_SETTINGS:-"ci/settings.xml"}
MVN="mvn -Dmaven.wagon.http.retryHandler.count=3 -B -s $MVN_SETTINGS"
# cuda12
CUDA_VER=${CUDA_VER:-cuda`nvcc --version | sed -n 's/^.*release \([0-9]\+\)\..*$/\1/p'`}
PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
USE_GDS=${USE_GDS:-ON}
USE_SANITIZER=${USE_SANITIZER:-ON}
BUILD_FAULTINJ=${BUILD_FAULTINJ:-ON}
BUILD_PROFILER=${BUILD_PROFILER:-ON}
ARM64=${ARM64:-false}
artifact_suffix="${CUDA_VER}"

profiles="source-javadoc"
if [ "${ARM64}" == "true" ]; then
  profiles="${profiles},arm64"
  USE_GDS="OFF"
  # libcupti_static.a linked by cufaultinj and the profiler, does not exist in the arm64 CUDA toolkit
  BUILD_FAULTINJ="OFF"
  BUILD_PROFILER="OFF"
  artifact_suffix="${artifact_suffix}-arm64"
fi

# disable the profiler and cufaultinj, since there are issues linking
# against libcupti_static.a in CUDA 13
if [ "${CUDA_VER}" == "cuda13" ]; then
  BUILD_FAULTINJ="OFF"
  BUILD_PROFILER="OFF"
  # Disable sanitizer https://github.com/NVIDIA/spark-rapids-jni/issues/4127
  USE_SANITIZER="OFF"
fi

${MVN} clean package ${MVN_MIRROR}  \
  -P${profiles} \
  -DCPP_PARALLEL_LEVEL=${PARALLEL_LEVEL} \
  -Dlibcudf.build.configure=true \
  -DUSE_GDS=${USE_GDS} -Dtest=*,!CuFileTest,!CudaFatalTest,!ColumnViewNonEmptyNullsTest,!NativeDepsLoaderTest \
  -DBUILD_TESTS=ON \
  -DBUILD_BENCHMARKS=ON \
  -DBUILD_FAULTINJ=${BUILD_FAULTINJ} \
  -DBUILD_PROFILER=${BUILD_PROFILER} \
  -Dcuda.version=$CUDA_VER \
  -DUSE_SANITIZER=${USE_SANITIZER}

build_name=$(${MVN} help:evaluate -Dexpression=project.build.finalName -q -DforceStdout)
. ci/check-cuda-dependencies.sh "target/${build_name}-${artifact_suffix}.jar"
