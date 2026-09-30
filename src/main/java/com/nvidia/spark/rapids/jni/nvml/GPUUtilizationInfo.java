/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
package com.nvidia.spark.rapids.jni.nvml;

/**
 * GPU utilization information from nvmlDeviceGetUtilizationRates()
 */
public class GPUUtilizationInfo {
    public int gpuUtilization;          // GPU utilization percentage (0-100)
    public int memoryUtilization;       // Memory utilization percentage (0-100)

    public GPUUtilizationInfo() {}

    public GPUUtilizationInfo(int gpuUtilization, int memoryUtilization) {
        this.gpuUtilization = gpuUtilization;
        this.memoryUtilization = memoryUtilization;
    }

    public GPUUtilizationInfo(GPUUtilizationInfo other) {
        this.gpuUtilization = other.gpuUtilization;
        this.memoryUtilization = other.memoryUtilization;
    }

    @Override
    public String toString() {
        return String.format("GPU: %d%%, Memory: %d%%", gpuUtilization, memoryUtilization);
    }
}
