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

# JNI-owned metadata only; shared actions resolve toolchains and build/publish images.
set -euo pipefail
operation=${1:-}
image_name=${2:-}
case "$operation" in
  version|list|describe|matrix) ;;
  *) echo "Usage: $0 version | list | describe <catalog-entry> | matrix" >&2; exit 2 ;;
esac
cd "$(dirname "${BASH_SOURCE[0]}")/.."
version=$(python3 - <<'PY'
import xml.etree.ElementTree as ET
version = ET.parse("pom.xml").getroot().findtext("{http://maven.apache.org/POM/4.0.0}version")
if not version:
    raise SystemExit("pom.xml must declare a literal project version")
print(version)
PY
)
[[ "$version" =~ ^[0-9]{2}\.[0-9]{2}\.[0-9]+(-[A-Za-z0-9.-]+)?$ ]]
release_line=${version:0:5}
if [[ "$operation" == version ]]; then
  jq -cn --arg version "$version" --arg line "$release_line" \
    '{version: $version, "release-line": $line}'
  exit 0
fi

catalog=.github/image-builds.json
: "${CI_TOOLCHAINS:?Pass the selected resolve-toolchain-config output}"
if [[ "$operation" == matrix ]]; then
  names=$(jq -ce 'select(type == "object") | keys | select(length > 0)' "$catalog")
  families=$(jq -ce '[.[].toolchain] | unique' "$catalog")
  jq -e --argjson families "$families" 'type == "object" and keys == $families and
    all(to_entries[]; (.key | test("^cuda[0-9]+$")) and
      (.value | type == "string" and test("^[0-9]+\\.[0-9]+\\.[0-9]+$")))' \
    <<< "$CI_TOOLCHAINS" >/dev/null
  configs='[]'
  while IFS= read -r name; do
    config=$(bash ci/image-config.sh describe "$name")
    configs=$(jq -c --argjson config "$config" '. + [$config]' <<< "$configs")
  done < <(jq -r '.[]' <<< "$names")
  # Build each shared tag once; incompatible recipes must not compete for a tag.
  jq -ce 'group_by(.image + ":" + .tag) | map(
    if ([.[].version] | unique | length) == 1 then .[0]
    else error("Conflicting recipes for " + .[0].image + ":" + .[0].tag) end) |
    select(length > 0 and length <= 256)' <<< "$configs"
  exit 0
fi

if [[ "$operation" == list ]]; then
  jq -ce 'select(type == "object") | keys | select(length > 0)' "$catalog"
  exit 0
fi
[[ "$image_name" =~ ^[a-z0-9]+([._-][a-z0-9]+)*$ ]]
config=$(jq -ce --arg name "$image_name" '.[$name] | select(type == "object")' "$catalog")
toolchain=$(jq -er '.toolchain | select(type == "string")' <<< "$config")
if ! cuda=$(jq -er --arg name "$toolchain" '.[$name] | select(type == "string")' <<< "$CI_TOOLCHAINS"); then
  echo "Missing $toolchain in selected toolchains for project $version" >&2
  exit 1
fi
if [[ ! "$cuda" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ || "$toolchain" != "cuda${cuda%%.*}" ]]; then
  echo "Invalid CUDA toolchain $toolchain=$cuda for project $version" >&2
  exit 1
fi
if jq -e '."build-args" | has("CUDA_VERSION")' <<< "$config" >/dev/null; then
  echo "Set CUDA_VERSION only in ci/toolchains.json, not $catalog" >&2
  exit 1
fi
os_release=$(jq -er '."build-args".OS_RELEASE' <<< "$config")
[[ "$os_release" =~ ^[0-9]+$ ]]
platform=$(jq -er '.platform' <<< "$config")
case "$platform" in
  linux/amd64) arch=amd64; cmake_arch=x86_64 ;;
  linux/arm64) arch=arm64; cmake_arch=aarch64 ;;
  *) echo "Unsupported image platform: $platform" >&2; exit 1 ;;
esac
if ! jq -e --arg platform "$platform" --arg cmake_arch "$cmake_arch" \
  '."build-args".TARGETPLATFORM == $platform and ."build-args".CMAKE_ARCH == $cmake_arch' \
  <<< "$config" >/dev/null; then
  echo "TARGETPLATFORM and CMAKE_ARCH must match image platform $platform" >&2
  exit 1
fi
parallel=${PARALLEL_LEVEL:-$(getconf _NPROCESSORS_ONLN)}
[[ "$parallel" =~ ^[1-9][0-9]*$ ]]
(( parallel <= 8 )) || parallel=8
tag_prefix=${IMAGE_TAG_PREFIX:-$release_line}
if [[ "$tag_prefix" != default && ! "$tag_prefix" =~ ^[0-9]{2}\.[0-9]{2}$ ]]; then
  echo "Image tag prefix must be default or a major.minor project version" >&2
  exit 1
fi
tag="${tag_prefix}-cuda${cuda}-rockylinux${os_release}-${arch}"
config=$(jq -c --arg cuda "$cuda" --arg parallel "$parallel" \
  '."build-args" += {CUDA_VERSION: $cuda, PARALLEL_LEVEL: $parallel} |
   ."cache-from" |= map(gsub("\\$\\{CUDA_VERSION\\}"; $cuda)) |
   ."cache-to" |= map(gsub("\\$\\{CUDA_VERSION\\}"; $cuda))' <<< "$config")
dockerfile=$(jq -er '.dockerfile' <<< "$config")
checksum=$(sha256sum "$dockerfile" | cut -d ' ' -f 1)
# This toolchain has no COPY/ADD inputs. Parallelism changes build speed, not the recipe.
recipe=$(jq -cS --arg dockerfile "$checksum" \
  '{dockerfile: $dockerfile, platform, "build-args": (."build-args" | del(.PARALLEL_LEVEL))}' \
  <<< "$config")
fingerprint="recipe-$(printf '%s' "$recipe" | sha256sum | cut -d ' ' -f 1)"
jq -c --arg fingerprint "$fingerprint" --arg tag "$tag" --arg cuda "$cuda" \
  '. + {version: $fingerprint, tag: $tag, "smoke-args": [$cuda, $fingerprint]}' <<< "$config"
