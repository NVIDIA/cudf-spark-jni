#!/bin/bash
#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

# common script to help check if packaged *.so files have dynamical link to CUDA Runtime

set -exo pipefail

jar_path=$1
tmp_path=/tmp/"jni-$(date "+%Y%m%d%H%M%S")"
unzip -j "${jar_path}" "*64/Linux/*.so" -d "${tmp_path}"

if unzip -Z -1 "${jar_path}" | grep -Eq '^[^/]+/Linux/libcudf\.so\.chunks\.properties$'; then
    m2_dir=${M2DIR:-"$HOME/.m2"}
    mvn_settings=${MVN_SETTINGS:-"${WORKSPACE:-$PWD}/ci/settings.xml"}
    zstd_jni_ver=$(mvn -s "$mvn_settings" help:evaluate -Dexpression=zstd-jni.version -q -DforceStdout)
    zstd_jni_jar="$m2_dir/repository/com/github/luben/zstd-jni/$zstd_jni_ver/zstd-jni-$zstd_jni_ver.jar"
    java -cp "$jar_path:$zstd_jni_jar" ai.rapids.cudf.NativeDepUtil extract cudf "$tmp_path/libcudf.so"
fi

if [[ ! -f "$tmp_path/libcudf.so" ]]; then
    echo "libcudf.so is missing from $jar_path"
    exit 1
fi

find "$tmp_path" -type f -name "*.so" | while read -r so_file; do
    # Check if *.so file has a dynamic link to CUDA Runtime
    if objdump -p "$so_file" | grep NEEDED | grep -qi cudart; then
        echo "Dynamic link to CUDA Runtime found in $so_file..."
        ldd "$so_file"
        exit 1
    else
        echo "No dynamic link to CUDA Runtime found in $so_file"
    fi
done
rm -rf "$tmp_path"
