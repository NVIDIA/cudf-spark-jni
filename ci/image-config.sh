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
  *) echo "Usage: $0 version | list | describe <catalog-entry> | matrix {all|selected}" >&2; exit 2 ;;
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
if [[ "$operation" == matrix ]]; then
  case "$image_name" in
    all)
      requirements=$(jq -ce '
        select(type == "object" and length > 0 and has("default") and
          all(keys[]; . == "default" or test("^[0-9]{2}\\.[0-9]{2}$"))) |
        [.[]] | unique' ci/toolchains.json)
      ;;
    selected) requirements=$(jq -cn --argjson selected "${CI_TOOLCHAINS:?Selected toolchains required}" '[$selected]') ;;
    *) echo "Matrix mode must be all or selected" >&2; exit 2 ;;
  esac
  names=$(jq -ce 'select(type == "object") | keys | select(length > 0)' "$catalog")
  families=$(jq -ce '[.[].toolchain] | unique' "$catalog")
  configs='[]'
  while IFS= read -r selected; do
    jq -e --argjson families "$families" 'type == "object" and keys == $families and
      all(to_entries[]; (.key | test("^cuda[0-9]+$")) and
        (.value | type == "string" and test("^[0-9]+\\.[0-9]+\\.[0-9]+$")))' \
      <<< "$selected" >/dev/null
    while IFS= read -r name; do
      config=$(CI_TOOLCHAINS="$selected" bash ci/image-config.sh describe "$name")
      configs=$(jq -c --argjson config "$config" '. + [$config]' <<< "$configs")
    done < <(jq -r '.[]' <<< "$names")
  done < <(jq -c '.[]' <<< "$requirements")
  # Build each shared tag once; incompatible recipes must not compete for a tag.
  jq -ce 'group_by(.image + ":" + .tag) | map(
    if ([.[].version] | unique | length) == 1 then .[0]
    else error("Conflicting recipes for " + .[0].image + ":" + .[0].tag) end) |
    select(length > 0 and length <= 256)' <<< "$configs"
  exit 0
fi

: "${CI_TOOLCHAINS:?Pass the selected resolve-toolchain-config output}"
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
[[ "$(jq -er '.platform' <<< "$config")" == linux/amd64 ]]
parallel=${PARALLEL_LEVEL:-$(getconf _NPROCESSORS_ONLN)}
[[ "$parallel" =~ ^[1-9][0-9]*$ ]]
(( parallel <= 8 )) || parallel=8
tag="cuda${cuda}-rockylinux${os_release}-amd64"
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
