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
