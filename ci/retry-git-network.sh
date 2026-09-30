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

set -uo pipefail

max_attempts=${GIT_NETWORK_RETRY_ATTEMPTS:-3}
delay_seconds=${GIT_NETWORK_RETRY_DELAY_SECONDS:-10}

if [[ ! $max_attempts =~ ^[0-9]+$ ]]; then
  echo "GIT_NETWORK_RETRY_ATTEMPTS must be a decimal integer" >&2
  exit 2
fi
if [[ ! $delay_seconds =~ ^[0-9]+$ ]]; then
  echo "GIT_NETWORK_RETRY_DELAY_SECONDS must be a decimal integer" >&2
  exit 2
fi

# Force decimal interpretation so values such as 08 are valid instead of being
# parsed as invalid octal by Bash arithmetic.
max_attempts=$((10#$max_attempts))
delay_seconds=$((10#$delay_seconds))

if ((max_attempts < 1)); then
  echo "GIT_NETWORK_RETRY_ATTEMPTS must be at least 1" >&2
  exit 2
fi

is_permanent_git_failure() {
  grep -Eiq \
    'authentication failed|permission denied|repository not found|could not read username|refusing to merge unrelated histories|automatic merge failed|merge conflict'
}

is_transient_git_transport_failure() {
  grep -Eiq \
    'connection timed out|failed to connect .*timed out|operation timed out|could not resolve host|temporary failure in name resolution|connection reset by peer|remote end hung up unexpectedly|rpc failed;.*curl (18|28|35|52|56)|http (408|429|5[0-9][0-9])|requested url returned error: (408|429|5[0-9][0-9])|gnutls recv error|tls.*(timeout|connection.*(closed|reset))'
}

attempt=1
while ((attempt <= max_attempts)); do
  log_file=$(mktemp "${TMPDIR:-/tmp}/git-network-retry-XXXXXX")
  "$@" 2>&1 | tee "$log_file"
  status=${PIPESTATUS[0]}
  if ((status == 0)); then
    rm -f "$log_file"
    exit 0
  fi

  if is_permanent_git_failure < "$log_file"; then
    echo "Git command failed with a permanent error; not retrying." >&2
    rm -f "$log_file"
    exit "$status"
  fi

  if ! is_transient_git_transport_failure < "$log_file"; then
    echo "Git command failed with a non-transient error; not retrying." >&2
    rm -f "$log_file"
    exit "$status"
  fi

  rm -f "$log_file"
  if ((attempt == max_attempts)); then
    echo "Git command still failed after ${max_attempts} attempt(s)." >&2
    exit "$status"
  fi

  echo "Transient Git transport failure on attempt ${attempt}/${max_attempts}; retrying in ${delay_seconds}s..." >&2
  sleep "$delay_seconds"
  ((attempt += 1))
done
