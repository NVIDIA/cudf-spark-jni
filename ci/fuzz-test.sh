#!/bin/bash
#
# Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
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
#     run fuzz test after build
#     required jars: jni, jni-tests, slf4j-api

set -ex

WORKSPACE=${WORKSPACE:-$PWD}
M2DIR=${M2DIR:-"$HOME/.m2"}

MVN_SETTINGS=${MVN_SETTINGS:-"$WORKSPACE/ci/settings.xml"}
SLF4J_VER=$(mvn -s "$MVN_SETTINGS" help:evaluate -Dexpression=slf4j.version -q -DforceStdout)
CLASSPATH=${CLASSPATH:-"$WORKSPACE/target/*:$M2DIR/repository/org/slf4j/slf4j-api/$SLF4J_VER/slf4j-api-$SLF4J_VER.jar"}

java -cp "$CLASSPATH" \
  com.nvidia.spark.rapids.jni.RmmSparkMonteCarlo \
  --taskMaxMiB=2048 --gpuMiB=3072 --skewed --allocMode=ASYNC
