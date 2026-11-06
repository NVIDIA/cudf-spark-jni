#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Compile only: validate the shared JNI build toolchain without allocating a GPU.
set -euo pipefail
image=${1:?Image reference required}
platform=${2:?Platform required}
cuda=${3:?CUDA version required}
fingerprint=${4:?Expected recipe fingerprint required}
[[ "$fingerprint" =~ ^recipe-[0-9a-f]{64}$ ]]
case "$platform" in
  linux/amd64|linux/arm64) arch=${platform#linux/} ;;
  *) echo "Unsupported smoke platform: $platform" >&2; exit 1 ;;
esac
if ! image_id=$(docker image inspect "$image" | jq -er --arg expected "$fingerprint" --arg arch "$arch" '
  .[0] | select(.Os == "linux" and .Architecture == $arch and
    .Config.Labels["org.opencontainers.image.version"] == $expected) |
  .Id | select(type == "string" and test("^sha256:[0-9a-f]{64}$"))'); then
  echo "::error::Unable to verify expected toolchain recipe $fingerprint and platform $platform for $image" >&2
  exit 1
fi
echo "Verified toolchain recipe $fingerprint on $platform for $image_id"
docker run --rm -i --platform "$platform" --network none \
  -e "EXPECTED_CUDA_VERSION=$cuda" --entrypoint /bin/bash "$image_id" -s <<'SMOKE'
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
gcc --version
g++ --version
cmake --version
ccache --version
ninja --version
mvn --version
java -version
javac -version
python -c 'import requests'
[[ "$CUDA_VERSION" == "$EXPECTED_CUDA_VERSION" ]]
nvcc --version
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
printf '#include <boost/filesystem.hpp>\nint main() { return boost::filesystem::exists(".") ? 0 : 1; }\n' > smoke.cpp
g++ smoke.cpp -L/usr/local/lib -Wl,-rpath,/usr/local/lib -lboost_filesystem -lboost_system -o smoke
# Boost's own shared dependencies also live outside the system loader paths.
LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ./smoke
printf '__global__ void smoke(int *value) { atomicAdd(value, 1); }\n' > smoke.cu
nvcc -c smoke.cu -o smoke.o
printf 'public class Smoke { public static void main(String[] args) {} }\n' > Smoke.java
javac Smoke.java
java Smoke
SMOKE
