#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run only on a disposable hosted runner: Maven can execute PR-controlled code.
set -euo pipefail

cd "$GITHUB_WORKSPACE/source"
[[ "$(git rev-parse HEAD)" = "$APPROVED_SHA" ]]
[[ -z "${SONAR_TOKEN:-}" ]]
[[ -z "${GITHUB_RUNNER_ADMIN_TOKEN:-}" ]]

# Reuse JNI's public Maven routing. Skip the native build and GPU tests, as in
# the existing Sonar job; build-helper still compiles the pinned cuDF Java sources.
mvn --batch-mode --no-transfer-progress \
  --settings "$GITHUB_WORKSPACE/workflow/.github/scripts/gpu-ci/${MVN_SETTINGS_FILE:-settings.xml}" \
  -DskipTests=true -Dmaven.antrun.skip=true test-compile

if ! find target/classes -type f -name '*.class' -print -quit | grep -q .; then
  echo 'Java compilation produced no class files for Sonar.' >&2
  exit 1
fi
