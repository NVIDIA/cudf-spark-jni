#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

set -euo pipefail
test "$(git rev-parse HEAD)" = "$APPROVED_SHA"
test -z "${GITHUB_RUNNER_ADMIN_TOKEN:-}"
: "${SONAR_HOST_URL:?Set the existing SONAR_HOST_URL repository secret}"
: "${SONAR_TOKEN:?Set the existing SONAR_TOKEN repository secret}"
test -n "$(find target/classes -type f -name '*.class' -print -quit)"
rm -f .scannerwork/report-task.txt

sonar-scanner \
  -Dsonar.ci.autoconfig.disabled=true \
  "-Dsonar.projectKey=$SONAR_PROJECT_KEY" \
  "-Dsonar.projectVersion=$PROJECT_VERSION" \
  "-Dsonar.scm.revision=$APPROVED_SHA" \
  "-Dsonar.pullrequest.key=$PR_NUMBER" \
  "-Dsonar.pullrequest.branch=$PR_BRANCH" \
  "-Dsonar.pullrequest.base=$PR_BASE" \
  -Dsonar.java.binaries=target/classes \
  -Dsonar.sources=src/main/java,src/main/cpp/src \
  -Dsonar.qualitygate.wait=true \
  -Dsonar.qualitygate.timeout=600
