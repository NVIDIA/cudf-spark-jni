#!/bin/bash
#
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# NOTE:
#     this script is for jenkins only, and should not be used for local development
#     run with ci/Dockerfile in jenkins:
#         source build/env.sh
#         ${sclCMD} ci/submodule-sync.sh
#     Optional prepare/validate modes split preparation from validation and publication.
#     Pass the same temporary state-file path as the second argument to both modes.
#     prepare writes 'noop' or the prepared commit; the caller cleans up the file.
# SUBMODULE_SYNC_PHASES=1

set -ex

phase=${1:-all}
case "$phase" in
  all|prepare|validate) ;;
  *) echo "Usage: $0 [prepare|validate <state-file>]" >&2; exit 2 ;;
esac
if [[ $phase == all ]]; then
  state_file=$(mktemp "${TMPDIR:-/tmp}/submodule-sync-$(date +%s)-XXXXXX")
  trap 'rm -f "$state_file"' EXIT
else
  state_file=${2:?Pass the same temporary state-file path to prepare and validate}
fi

OWNER=${OWNER:-"NVIDIA"}
REPO=${REPO:-"cudf-spark-jni"}
PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
REPO_LOC="github.com/${OWNER}/${REPO}.git"
INTERMEDIATE_HEAD=bot-submodule-sync-${REF}
MVN_SETTINGS=${MVN_SETTINGS:-"ci/settings.xml"}
MVN="mvn -Dmaven.wagon.http.retryHandler.count=3 -B -s $MVN_SETTINGS"

release_line() {
  local version=$1
  if [[ $version =~ ^([0-9]{2})\.([0-9]{1,2})(\.[0-9]+)?(-[[:alnum:]][[:alnum:].-]*)?$ ]]; then
    local year=$((10#${BASH_REMATCH[1]}))
    local month=$((10#${BASH_REMATCH[2]}))
    if ((month >= 1 && month <= 12)); then
      printf '%02d.%02d\n' "$year" "$month"
      return
    fi
  fi
  return 1
}

if [[ $phase != validate ]]; then
  : > "$state_file"
  export GIT_AUTHOR_NAME="cudf-spark automation"
  export GIT_COMMITTER_NAME="cudf-spark automation"
  export GIT_AUTHOR_EMAIL="70000568+nvauto@users.noreply.github.com"
  export GIT_COMMITTER_EMAIL="70000568+nvauto@users.noreply.github.com"
  git submodule update --init --recursive

  # try cleanup remote first if no open PR for HEAD targeting BASE
  $WORKSPACE/.github/workflows/action-helper/python/cleanup-bot-branch \
    --owner=${OWNER} --repo=${REPO} --head=${INTERMEDIATE_HEAD} --base=${REF} --token=${GIT_TOKEN} || true

  remote_head=$(git ls-remote --heads origin ${INTERMEDIATE_HEAD})
  if [[ -z $remote_head ]]; then
    git checkout -b ${INTERMEDIATE_HEAD} origin/${REF}
  else
    git fetch origin ${INTERMEDIATE_HEAD} ${REF}
    git checkout -b ${INTERMEDIATE_HEAD} origin/${INTERMEDIATE_HEAD}
    git merge origin/${REF}
  fi

  # Check both remote-branch and explicit-tag sync targets before updating any pins.
  if [ -n "$CUDF_TAG" ]; then
    git -C thirdparty/cudf checkout tags/$CUDF_TAG
  else
    git submodule update --remote --merge
  fi
  cudf_version='<missing>'
  if [[ -r thirdparty/cudf/VERSION ]]; then
    cudf_version=$(< thirdparty/cudf/VERSION)
  fi
  if ! jni_version=$(${MVN} help:evaluate ${MVN_MIRROR} \
      -Dexpression=project.version -q -DforceStdout -Dstyle.color=never); then
    echo "Cannot read JNI project.version (cuDF VERSION: ${cudf_version})." >&2
    exit 1
  fi
  if ! cudf_release=$(release_line "$cudf_version") ||
      ! jni_release=$(release_line "$jni_version") ||
      [[ $cudf_release != "$jni_release" ]]; then
    echo "Incompatible or invalid release versions: cuDF='${cudf_version}', JNI='${jni_version}'." >&2
    echo "If cuDF has started a new release, create JNI's release/YY.MM branch for its current version (if missing)," >&2
    echo "then bump JNI main to cuDF's new release version. For release-branch syncs, select a matching cuDF branch/tag." >&2
    exit 1
  fi
  cudf_sha=$(git -C thirdparty/cudf rev-parse HEAD)

  echo "Configure libcudf only to update pinned versions..."
  # Configure without patches. A failure must not be treated as an unchanged pin set.
  # Native architecture is validation-only; nightly/premerge retain the POM's RAPIDS default.
  ${MVN} antrun:run@buildcpp ${MVN_MIRROR} \
    -DCPP_PARALLEL_LEVEL=${PARALLEL_LEVEL} \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -Dlibcudf.build.configure=true \
    -Dlibcudf.dependency.mode=latest \
    -DUSE_GDS=ON \
    -DBUILD_TESTS=ON \
    -DUSE_SANITIZER=ON \
    -DLIBCUDF_CONFIGURE_ONLY=ON

  LIBCUDF_BUILD_PATH=$(${MVN} help:evaluate ${MVN_MIRROR} -Dexpression=libcudf.build.path -q -DforceStdout)
  # Extract the rapids-cmake sha1 that we need to pin too
  rapids_cmake_sha=$(git -C "${LIBCUDF_BUILD_PATH}/_deps/rapids-cmake-src/" rev-parse HEAD)
  echo "Update rapids-cmake pinned SHA1 to ${rapids_cmake_sha}"
  echo "${rapids_cmake_sha}" > thirdparty/cudf-pins/rapids-cmake.sha

  git add thirdparty/cudf thirdparty/cudf-pins
  if ! git diff --cached --quiet; then
    git commit -s -m "Update cudf ${cudf_sha} and pinned versions"
  fi
  # Include pending bot-branch updates, even when this run produced no new commit.
  if git diff --quiet "origin/${REF}" HEAD -- thirdparty/cudf thirdparty/cudf-pins; then
    echo noop > "$state_file"
    echo "No submodule or pin changes. Exit early..."
    exit 0
  fi
  git rev-parse HEAD > "$state_file"
  if [[ $phase == prepare ]]; then
    exit 0
  fi
fi

# The handoff is a commit ID, not executable shell state, and lives outside Maven's target dir.
if [[ ! -s $state_file ]] ||
    [[ $(cat "$state_file") != "$(git rev-parse HEAD)" ]] ||
    [[ $(git branch --show-current) != "$INTERMEDIATE_HEAD" ]] ||
    ! git diff --quiet HEAD --ignore-submodules=untracked; then
  echo "Missing preparation or checkout changed; run submodule-sync.sh prepare first." >&2
  exit 1
fi
rm -f "$state_file"

sha=$(git rev-parse HEAD)
cudf_sha=$(git -C thirdparty/cudf rev-parse HEAD)

echo "Test against ${cudf_sha}..."
ccache -s || true
set +e
# phase 3:
# now build and test everything with the patches in place
${MVN} clean verify ${MVN_MIRROR} \
  -DCPP_PARALLEL_LEVEL=${PARALLEL_LEVEL} \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -Dlibcudf.build.configure=true \
  -DUSE_GDS=ON -Dtest=*,!CuFileTest,!CudaFatalTest,!ColumnViewNonEmptyNullsTest,!NativeDepsLoaderTest,!PackagedJarOriginCheck \
  -DBUILD_TESTS=ON \
  -DUSE_SANITIZER=ON
verify_status=$?
set -e
ccache -s || true

test_pass="False"
if [[ "${verify_status}" == "0" ]]; then
  echo "Test passed, will try merge the change"
  test_pass="True"
else
  echo "Test failed, will update the result"
fi

build_name=$(${MVN} help:evaluate ${MVN_MIRROR} -Dexpression=project.build.finalName -q -DforceStdout)
cuda_version=$(${MVN} help:evaluate ${MVN_MIRROR} -Dexpression=cuda.version -q -DforceStdout)
bash ci/check-cuda-dependencies.sh "target/${build_name}-${cuda_version}.jar"

# push the intermediate branch and create PR against REF
# if test passed, it will try auto-merge the PR
# if test failed, it will only comment the test result in the PR
git push https://${GIT_USER}:${GIT_TOKEN}@${REPO_LOC} ${INTERMEDIATE_HEAD}
sleep 30 # sleep for a while to avoid inconsistent sha between HEAD branch and GitHub REST API
$WORKSPACE/.github/workflows/action-helper/python/submodule-sync \
  --owner=${OWNER} \
  --repo=${REPO} \
  --head=${INTERMEDIATE_HEAD} \
  --base=${REF} \
  --sha=${sha} \
  --cudf_sha=${cudf_sha} \
  --token=${GIT_TOKEN} \
  --passed=${test_pass} \
  --delete_head=True

exit $verify_status # always exit return code of mvn verify at the end
