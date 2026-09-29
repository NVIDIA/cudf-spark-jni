#!/bin/bash
#
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
#

set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
retry="$repo_root/ci/retry-git-network.sh"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/retry-git-network-test-XXXXXX")
trap 'rm -rf "$tmp"' EXIT

make_fake_command() {
  local name=$1
  local failures=$2
  local message=$3
  cat > "$tmp/$name" <<EOF
#!/bin/bash
count_file="$tmp/$name.count"
count=0
[[ -f "\$count_file" ]] && count=\$(<"\$count_file")
count=\$((count + 1))
echo "\$count" > "\$count_file"
if ((count <= $failures)); then
  echo '$message' >&2
  exit 128
fi
echo success
EOF
  chmod +x "$tmp/$name"
}

assert_count() {
  local name=$1
  local expected=$2
  local actual
  actual=$(<"$tmp/$name.count")
  [[ "$actual" == "$expected" ]] || {
    echo "$name ran $actual times, expected $expected" >&2
    exit 1
  }
}

make_fake_command transient_then_success 2 "fatal: unable to access 'https://github.com/NVIDIA/cudf/': Failed to connect to github.com port 443: Connection timed out"
GIT_NETWORK_RETRY_ATTEMPTS=3 GIT_NETWORK_RETRY_DELAY_SECONDS=0   "$retry" "$tmp/transient_then_success"
assert_count transient_then_success 3

make_fake_command http_503_then_success 1 "error: RPC failed; HTTP 503 curl 22 The requested URL returned error: 503"
GIT_NETWORK_RETRY_ATTEMPTS=2 GIT_NETWORK_RETRY_DELAY_SECONDS=0 \
  "$retry" "$tmp/http_503_then_success"
assert_count http_503_then_success 2

make_fake_command http_502_then_success 1 "fatal: unable to access 'https://github.com/NVIDIA/cudf/': The requested URL returned error: 502"
GIT_NETWORK_RETRY_ATTEMPTS=2 GIT_NETWORK_RETRY_DELAY_SECONDS=0 \
  "$retry" "$tmp/http_502_then_success"
assert_count http_502_then_success 2

make_fake_command persistent_transient 5 "fatal: unable to access 'https://github.com/NVIDIA/cudf/': Could not resolve host: github.com"
if GIT_NETWORK_RETRY_ATTEMPTS=2 GIT_NETWORK_RETRY_DELAY_SECONDS=0     "$retry" "$tmp/persistent_transient"; then
  echo "persistent transient failure unexpectedly succeeded" >&2
  exit 1
fi
assert_count persistent_transient 2

make_fake_command merge_conflict 1 "fatal: refusing to merge unrelated histories"
if GIT_NETWORK_RETRY_ATTEMPTS=3 GIT_NETWORK_RETRY_DELAY_SECONDS=0     "$retry" "$tmp/merge_conflict"; then
  echo "deterministic merge failure unexpectedly succeeded" >&2
  exit 1
fi
assert_count merge_conflict 1

make_fake_command auth_failure 1 "fatal: Authentication failed for 'https://github.com/NVIDIA/cudf/'"
if GIT_NETWORK_RETRY_ATTEMPTS=3 GIT_NETWORK_RETRY_DELAY_SECONDS=0     "$retry" "$tmp/auth_failure"; then
  echo "authentication failure unexpectedly succeeded" >&2
  exit 1
fi
assert_count auth_failure 1

echo "retry-git-network tests passed"
