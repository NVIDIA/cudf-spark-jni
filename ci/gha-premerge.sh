#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reuse the Jenkins workload with public dependencies and an immutable local image.
set -euo pipefail
# This script receives the job token only while preparing its private netrc.
set +x
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
auth_dir=
container_started=false

finish() {
  status=$?
  set +e
  printf '%s\n' "$status" > target/gha-exit-code.txt || :
  # A timed-out Docker client can leave the container running on the GPU.
  stopped=true
  if [[ -s target/gha-container.cid ]]; then
    container_id=$(< target/gha-container.cid)
    if ! timeout -k 5s 20s docker rm -f "$container_id" > target/gha-cleanup.log 2>&1; then
      # --rm normally already removed it. A daemon error is not proof of removal.
      if ! containers=$(timeout -k 5s 10s docker ps -aq --no-trunc --filter "id=$container_id") ||
          [[ -n "$containers" ]]; then
        stopped=false
      fi
    fi
  elif [[ "$container_started" == true ]]; then
    stopped=false
  fi
  if [[ "$stopped" == true && -n "$auth_dir" ]] &&
      python3 ci/gha-download-auth.py "$auth_dir/netrc" --stage target/gha-evidence; then
    [[ -z "${GITHUB_OUTPUT:-}" ]] || echo 'publish-safe=true' >> "$GITHUB_OUTPUT"
  else
    echo '::warning::Could not safely collect diagnostics; evidence and cache uploads are disabled.'
  fi
  [[ -z "$auth_dir" ]] || rm -rf -- "$auth_dir"
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Only approved PR workers receive this read-only, short-lived token. The file is
# outside caches/workspace and mounted read-only; approved build code can read it.
# No PAT, publishing credential or token environment variable enters the container.
auth_dir=$(mktemp -d "$RUNNER_TEMP/jni-download-auth.XXXXXXXX")
(umask 077; : > "$auth_dir/netrc")
if [[ -n "${GITHUB_DOWNLOAD_TOKEN:-}" ]]; then
  printf 'machine github.com login x-access-token password %s\n' \
    "$GITHUB_DOWNLOAD_TOKEN" > "$auth_dir/netrc"
  encoded=$(printf 'x-access-token:%s' "$GITHUB_DOWNLOAD_TOKEN" | base64 | tr -d '\n')
  printf '::add-mask::%s\n' "$encoded"
  unset encoded
fi
unset GITHUB_DOWNLOAD_TOKEN

printf 'starting container\n' > target/gha-phase.txt || :
[[ "${JNI_WORKLOAD_DEADLINE:?Worker deadline required}" =~ ^[0-9]+$ ]]
remaining=$((JNI_WORKLOAD_DEADLINE - $(date +%s)))
if (( remaining <= 0 )); then
  echo "Workload budget exhausted during setup; collecting available evidence."
  exit 124
fi
echo "Workload budget remaining: $remaining seconds"

container_started=true
set +e
# Variables in the command below are expanded inside the workload container.
# shellcheck disable=SC2016
timeout -k 30s "${remaining}s" docker run --rm --init --gpus all \
  --cidfile "$PWD/target/gha-container.cid" --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -v "$home_dir:/home/ci" -v /etc/passwd:/etc/passwd:ro \
  -v "$auth_dir/netrc:/run/secrets/github.netrc:ro" \
  -w /workspace -e HOME=/home/ci -e WORKSPACE=/workspace \
  -e "CMAKE_NETRC=${CMAKE_NETRC:-OPTIONAL}" -e CMAKE_NETRC_FILE=/run/secrets/github.netrc \
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
    finish_container() {
      status=$?
      timeout -k 5s 10s ccache --show-stats > target/gha-ccache.log 2>&1 || :
      exit "$status"
    }
    trap finish_container EXIT

    # Best-effort identity/version diagnostics, not a new readiness gate.
    timeout -k 5s 10s nvidia-smi --query-gpu=name,uuid,driver_version --format=csv \
      > target/gha-gpu.txt 2>&1 || :
    cat target/gha-gpu.txt || :
    timeout -k 5s 10s nvcc --version || :
    timeout -k 5s 10s java -version || :

    printf "build and Java/native tests\n" > target/gha-phase.txt || :
    echo "=== JNI build and tests ==="
    ${sclCMD} "ci/premerge-build.sh"
    printf "fuzz test\n" > target/gha-phase.txt || :
    echo "=== JNI fuzz test ==="
    bash ci/fuzz-test.sh
    printf "complete\n" > target/gha-phase.txt || :
  ' 2>&1 | python3 ci/gha-download-auth.py "$auth_dir/netrc" | tee target/gha-premerge.log
results=("${PIPESTATUS[@]}")
# Keep the real workload failure even if filtering or tee also failed.
if (( results[0] != 0 )); then exit "${results[0]}"; fi
if (( results[1] != 0 || results[2] != 0 )); then exit 1; fi
