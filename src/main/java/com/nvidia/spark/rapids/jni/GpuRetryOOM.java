/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.jni;

/**
 * A special version of an out of memory error that indicates we ran out of GPU memory, but should
 * roll back to a point when all memory for the task is spillable and then retry the operation.
 */
public class GpuRetryOOM extends GpuOOM {
  public GpuRetryOOM() {
    super();
  }

  public GpuRetryOOM(String message) {
    super(message);
  }
}
