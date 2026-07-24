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
CUDF_REUSE_FORCE=${CUDF_REUSE_FORCE:-false}
CUDF_REUSE_PREBUILT=${CUDF_REUSE_PREBUILT:-false}
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

# Symlink compile_commands.json into the source tree for IDE/clangd discovery.
create_compile_commands_symlink() {
  local build_dir=$1
  local source_dir=$2
  local compile_commands_file="$build_dir/compile_commands.json"
  local compile_commands_link="$source_dir/compile_commands.json"

  echo "Creating symlink from $compile_commands_link to $compile_commands_file..."
  ln -sf "$compile_commands_file" "$compile_commands_link"
}

# One nonce per run, stamped into both manifests: equal nonces prove the install/libcudfjni pair
# came from one seed run. Used over cudf_sha, which is "unknown" for a git-less source.
SEED_NONCE="$(cat /proc/sys/kernel/random/uuid 2>/dev/null || uuidgen)"

# Skew-guard rationale: enforce a dimension (a cmp. key, diffed on reuse) iff phase 3 can diverge
# from the prebuilt silently (ODR/ABI). Enforced keys and why:
#   pins_sha256  - the 4 tracked cudf-pins files (hashed by name); phase 3 fetches these deps.
#   ptds         - public compile define; mixing per-thread/legacy splits stream semantics.
#   cuda_archs   - device code for different real archs can't link/run.
#   use_gds      - gates the cuFile paths and the libcufilejni.a link.
#   rmm_logging  - macro baked into the rmm/cudf headers phase 3 recompiles.
#   build_type   - the prebuilt libcudf's CMAKE_BUILD_TYPE.
#   dep_mode     - reuse needs "pinned"; "latest" drifts daily.
#   cuda_toolkit - device ABI + statically linked CUDA runtime.
# Not enforced: cudf source/commit fails loudly (phase-3 link or runtime UnsatisfiedLinkError), so
# only warned; nvcomp + Arrow/Parquet are consumed wholesale. seed_nonce/cudf_sha/toolchain/date
# are RECORD-only.
cudf_fingerprint_compare_lines() {
  local pins_sha cuda_toolkit
  # Hash only the digest column (cut -f1): sha256sum's raw output embeds absolute paths that
  # differ between seed and consumer trees, faking skew on identical pins.
  pins_sha="$(set -o pipefail; sha256sum \
    "$CUDF_PIN_PATH/versions.json" \
    "$CUDF_PIN_PATH/setup.cmake" \
    "$CUDF_PIN_PATH/add_dependency_pins.cmake" \
    "$CUDF_PIN_PATH/rapids-cmake.sha" | cut -d' ' -f1 | sha256sum | cut -d' ' -f1)" \
    || { echo "ERROR: cannot hash the tracked cudf-pins files under $CUDF_PIN_PATH" >&2; exit 1; }
  cuda_toolkit="${CUDA_VERSION:-}"
  if [[ -z "$cuda_toolkit" ]]; then
    cuda_toolkit="$(set -o pipefail; nvcc --version | grep -o 'release [0-9][0-9.]*' | head -1)" \
      || { echo "ERROR: cannot determine CUDA toolkit (no CUDA_VERSION, no nvcc)" >&2; exit 1; }
    cuda_toolkit="${cuda_toolkit#release }"
  fi
  echo "cmp.pins_sha256=$pins_sha"
  echo "cmp.ptds=$CUDF_USE_PER_THREAD_DEFAULT_STREAM"
  echo "cmp.cuda_archs=$CMAKE_CUDA_ARCHITECTURES"
  echo "cmp.use_gds=$USE_GDS"
  echo "cmp.rmm_logging=$RMM_LOGGING_LEVEL"
  echo "cmp.build_type=$CUDF_BUILD_TYPE"
  echo "cmp.dep_mode=$LIBCUDF_DEPENDENCY_MODE"
  echo "cmp.cuda_toolkit=$cuda_toolkit"
}

# Write one manifest: the cmp. lines plus RECORD-only provenance. Single-shot overwrite (">"), so
# a rebuild replaces it; appending would duplicate lines and break the reuse check's single greps.
cudf_fingerprint_stamp() {
  local manifest="$1" content cudf_sha gcc_ver nvcc_ver
  cudf_sha="$(git -C "$CUDF_PATH" rev-parse HEAD 2>/dev/null)" || cudf_sha=unknown
  gcc_ver="$(gcc -dumpfullversion 2>/dev/null)" || gcc_ver=unknown
  nvcc_ver="$(set -o pipefail; nvcc --version 2>/dev/null | grep -o 'V[0-9][0-9.]*' | head -1)" \
    || nvcc_ver=unknown
  content="$(
    cudf_fingerprint_compare_lines
    echo "seed_nonce=$SEED_NONCE"
    echo "cudf_sha=$cudf_sha"
    echo "gcc=$gcc_ver"
    echo "nvcc=$nvcc_ver"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  )" || { echo "ERROR: fingerprint generation failed for $manifest" >&2; exit 1; }
  printf '%s\n' "$content" > "$manifest"
}

# Reuse validation: fail fast here (not at phase-3 link) if the prebuilt trees are incomplete for
# this build config or fingerprint-skewed vs this worktree.
if [[ "$CUDF_REUSE_PREBUILT" == "true" ]]; then
  LIBCUDF_A="$LIBCUDF_INSTALL_PATH/lib64/libcudf.a"
  [[ -f "$LIBCUDF_A" ]] || LIBCUDF_A="$LIBCUDF_INSTALL_PATH/lib/libcudf.a"
  LIBDIR="$(dirname "$LIBCUDF_A")"
  [[ -f "$LIBCUDF_A" ]] \
    || { echo "ERROR: no prebuilt lib{64,}/libcudf.a under $LIBCUDF_INSTALL_PATH" >&2; exit 1; }
  [[ -f "$LIBCUDFJNI_BUILD_PATH/libcudfjni.a" ]] \
    || { echo "ERROR: no prebuilt libcudfjni.a at $LIBCUDFJNI_BUILD_PATH" >&2; exit 1; }
  [[ -f "$LIBCUDFJNI_BUILD_PATH/_deps/arrow-build/release/libarrow.a" ]] \
    || { echo "ERROR: incomplete libcudfjni _deps (arrow)" >&2; exit 1; }
  # jar packaging copies libnvcomp*.so from here (pom.xml copy-native-libs) but does not fail if
  # absent, so a trimmed prebuilt would silently ship a jar without nvcomp.
  [[ -f "$LIBCUDFJNI_BUILD_PATH/libnvcomp.so" ]] \
    || { echo "ERROR: prebuilt libcudfjni missing libnvcomp.so (jar packaging)" >&2; exit 1; }
  if [[ "$BUILD_TESTS" == "ON" || "$BUILD_BENCHMARKS" == "ON" ]]; then
    [[ -d "$LIBCUDF_INSTALL_PATH/src/cudftestutil" ]] \
      || { echo "ERROR: test/bench reuse needs installed testutil (src/cudftestutil); re-seed" >&2
           exit 1; }
    [[ -f "$LIBDIR/libcudftest_default_stream.a" ]] \
      || { echo "ERROR: test/bench reuse needs cudftest_default_stream; re-seed with testutil" >&2
           exit 1; }
  fi
  [[ "$LIBCUDF_DEPENDENCY_MODE" == "pinned" ]] \
    || { echo "ERROR: reuse requires pinned deps; got $LIBCUDF_DEPENDENCY_MODE" >&2; exit 1; }
  m1="$LIBCUDF_INSTALL_PATH/cudf-prebuilt-fingerprint.txt"
  m2="$LIBCUDFJNI_BUILD_PATH/cudf-prebuilt-fingerprint.txt"
  want="$(cudf_fingerprint_compare_lines)"
  want="$(LC_ALL=C sort <<<"$want")" # capture, THEN sort (set -e without pipefail)
  for m in "$m1" "$m2"; do
    [[ -f "$m" ]] || { echo "ERROR: prebuilt not stamped ($m); re-seed" >&2; exit 1; }
    if ! diff <(grep '^cmp\.' "$m" | LC_ALL=C sort) <(printf '%s\n' "$want"); then
      if [[ "$CUDF_REUSE_FORCE" == "true" ]]; then
        echo "WARNING: cudf skew in $m overridden" >&2
      else
        echo "ERROR: prebuilt does not match this worktree (skew in $m)." \
          "Re-seed or -Dcudf.reuse.force=true." >&2
        exit 1
      fi
    fi
  done
  # Mismatched-pair guard: both trees must come from ONE seed run, else libcudfjni.a was built
  # against different cudf headers than libcudf.a (silent ODR). Compare seed_nonce, not cudf_sha:
  # a git-less source makes cudf_sha "unknown" on both, passing vacuously. The "|| exit" on each
  # grep is required under set -e: a bare n=$(grep) matching nothing aborts silently; treat a
  # manifest without seed_nonce as corrupt.
  n1="$(grep '^seed_nonce=' "$m1")" \
    || { echo "ERROR: $m1 missing seed_nonce (corrupt/old manifest); re-seed" >&2; exit 1; }
  n2="$(grep '^seed_nonce=' "$m2")" \
    || { echo "ERROR: $m2 missing seed_nonce (corrupt/old manifest); re-seed" >&2; exit 1; }
  if [[ "$n1" != "$n2" ]]; then
    if [[ "$CUDF_REUSE_FORCE" == "true" ]]; then
      echo "WARNING: install/libcudfjni from different seed runs ($n1 vs $n2)" >&2
    else
      echo "ERROR: install ($n1) and libcudfjni ($n2) prebuilt from different seed runs" \
        "(mismatched pair). Re-seed both, or -Dcudf.reuse.force=true." >&2
      exit 1
    fi
  fi
  # Non-enforcing cudf-source skew warning (enforce jni pins, not cudf): if CUDF_PATH's commit
  # differs from the prebuilt's cudf_sha, the native/JNI surface may diverge but fails loudly
  # later (phase-3 link or a runtime UnsatisfiedLinkError). Warn early; never block.
  consumer_sha="$(git -C "$CUDF_PATH" rev-parse HEAD 2>/dev/null || echo unknown)"
  prebuilt_sha="$(grep '^cudf_sha=' "$m1" | head -1)"
  prebuilt_sha="${prebuilt_sha#cudf_sha=}"
  if [[ "$consumer_sha" != unknown && -n "$prebuilt_sha" && "$prebuilt_sha" != unknown \
        && "$consumer_sha" != "$prebuilt_sha" ]]; then
    echo "WARNING: cudf.path HEAD ($consumer_sha) != prebuilt cudf ($prebuilt_sha);" \
      "native/JNI skew may fail at phase-3 link or test runtime." >&2
  fi
fi

#
# libcudf build
#
if [[ "$CUDF_REUSE_PREBUILT" == "true" ]]; then
  # Reuse also bypasses the legacy LIBCUDF_CONFIGURE_ONLY early-exit: no cudf build to configure.
  echo "Skipping libcudf build; reusing $LIBCUDF_INSTALL_PATH"
else
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
  # submodule-sync.sh phase 1 calls this script with LIBCUDF_CONFIGURE_ONLY=ON
  if [[ $LIBCUDF_CONFIGURE_ONLY == ON ]]; then
    echo "Skip build..."
    exit 0
  fi
  echo "Building cudf native libs"
  cmake --build "$LIBCUDF_BUILD_PATH" --target install "-j$CPP_PARALLEL_LEVEL"
  cudf_fingerprint_stamp "$LIBCUDF_INSTALL_PATH/cudf-prebuilt-fingerprint.txt"
fi

#
# libcudfjni build
#
if [[ "$CUDF_REUSE_PREBUILT" == "true" ]]; then
  echo "Skipping libcudfjni build; reusing $LIBCUDFJNI_BUILD_PATH"
else
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
  cudf_fingerprint_stamp "$LIBCUDFJNI_BUILD_PATH/cudf-prebuilt-fingerprint.txt"
fi

#
# sparkjni build
#
# Persisted phase-3 build dirs pin stale trees: CMake never re-runs a find_library/find_package
# whose cache holds a valid path, so on a CUDF_DIR/CUDF_INSTALL_DIR/CUDFJNI_BUILD_DIR change the
# derived entries (CUDFJNI_LIB, ARROW_LIB, find_package dirs, ...) keep resolving the OLD trees.
# Wipe the build dir on a trio mismatch so the configure re-derives them; no cache means no wipe.
SPARK_JNI_CACHE="$SPARK_JNI_BUILD_PATH/CMakeCache.txt"
if [[ -f "$SPARK_JNI_CACHE" ]]; then
  cached_cudf_dir="$(grep '^CUDF_DIR:' "$SPARK_JNI_CACHE" | cut -d= -f2-)" || cached_cudf_dir=""
  cached_install_dir="$(grep '^CUDF_INSTALL_DIR:' "$SPARK_JNI_CACHE" | cut -d= -f2-)" \
    || cached_install_dir=""
  cached_cudfjni_dir="$(grep '^CUDFJNI_BUILD_DIR:' "$SPARK_JNI_CACHE" | cut -d= -f2-)" \
    || cached_cudfjni_dir=""
  if [[ "$cached_cudf_dir" != "$CUDF_PATH" || "$cached_install_dir" != "$LIBCUDF_INSTALL_PATH" \
        || "$cached_cudfjni_dir" != "$LIBCUDFJNI_BUILD_PATH" ]]; then
    echo "cudf tree paths changed since the last cudf-spark-jni configure;" \
      "wiping $SPARK_JNI_BUILD_PATH"
    rm -rf "$SPARK_JNI_BUILD_PATH"
  fi
fi
mkdir -p "$SPARK_JNI_BUILD_PATH"
cd "$SPARK_JNI_BUILD_PATH"
echo "Configuring spark-rapids-jni native libs"
# The trio -D pre-creates the env-seeded set(CACHE) entries (which never overwrite an existing
# value), so the current paths win over a persisted CMakeCache.txt.
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
    -DCUDF_DIR="$CUDF_PATH" \
    -DCUDF_INSTALL_DIR="$LIBCUDF_INSTALL_PATH" \
    -DCUDFJNI_BUILD_DIR="$LIBCUDFJNI_BUILD_PATH" \
    -DCUDF_DEPENDENCY_PIN_MODE=pinned \
    -DCUDF_USE_PER_THREAD_DEFAULT_STREAM="$CUDF_USE_PER_THREAD_DEFAULT_STREAM" \
    -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
    -DRMM_LOGGING_LEVEL="$RMM_LOGGING_LEVEL" \
    -DUSE_GDS="$USE_GDS" \
    -C="$CUDF_PIN_PATH/setup.cmake"

create_compile_commands_symlink "$SPARK_JNI_BUILD_PATH" "$PROJECT_BASE_DIR/src/main/cpp"

echo "Building spark-rapids-jni native libs"
cmake --build "$SPARK_JNI_BUILD_PATH" "-j$CPP_PARALLEL_LEVEL"
