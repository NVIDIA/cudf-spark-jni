#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reuse the Jenkins workload with public dependencies and an immutable local image.
set -euo pipefail
[[ "${CLASSIFIER:?CUDA classifier required}" == cuda12 || "$CLASSIFIER" == cuda13 ]]
[[ "${JNI_IMAGE:?Verified toolchain image required}" =~ ^sha256:[0-9a-f]{64}$ ]]
parallel_level=${PARALLEL_LEVEL:-$(nproc)}
[[ "$parallel_level" =~ ^[1-9][0-9]*$ ]]
# Keep the existing premerge ceiling, with 4 GiB per compiler on smaller workers.
if (( parallel_level > 18 )); then parallel_level=18; fi
memory_jobs=$(awk '/MemAvailable:/ {n=int($2 / 4194304); print (n > 0 ? n : 1)}' /proc/meminfo)
if [[ -n "$memory_jobs" ]] && (( parallel_level > memory_jobs )); then
  parallel_level=$memory_jobs
fi
home_dir=${RUNNER_TEMP:?}/jni-home
mkdir -p "$home_dir/.m2/repository" "$home_dir/.ccache" "$home_dir/.cache/cpm" target
echo "JNI $CLASSIFIER: parallel_level=$parallel_level; Maven, ccache and CPM caches enabled"

docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -v "$home_dir:/home/ci" -v /etc/passwd:/etc/passwd:ro \
  -w /workspace -e HOME=/home/ci -e WORKSPACE=/workspace \
  -e "CUDA_VER=$CLASSIFIER" \
  -e MAVEN_OPTS='-Duser.home=/home/ci -Dmaven.repo.local=/home/ci/.m2/repository' \
  -e MVN_SETTINGS=/workspace/ci/settings-public.xml \
  -e "PARALLEL_LEVEL=$parallel_level" \
  -e CCACHE_DIR=/home/ci/.ccache -e CCACHE_BASEDIR=/workspace -e CCACHE_MAXSIZE=10G \
  -e CPM_SOURCE_CACHE=/home/ci/.cache/cpm \
  -e CMAKE_C_COMPILER_LAUNCHER=ccache -e CMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -e CMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  --entrypoint /bin/bash "$JNI_IMAGE" -c '
    set -eo pipefail
    source build/env.sh
    # Maven user properties must reach verify and every help:evaluate invocation.
    # MAVEN_OPTS system properties do not consistently override POM interpolation.
    mkdir -p .mvn
    printf "\n-Dcuda.version=%s\n" "$CUDA_VER" >> .mvn/maven.config
    ccache --zero-stats
    trap "ccache --show-stats || :" EXIT
    ${sclCMD} "ci/premerge-build.sh"
    bash ci/fuzz-test.sh
  ' 2>&1 | tee target/gha-premerge.log
