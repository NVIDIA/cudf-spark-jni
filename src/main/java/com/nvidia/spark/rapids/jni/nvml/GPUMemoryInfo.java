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
 * GPU memory information from nvmlDeviceGetMemoryInfo()
 */
public class GPUMemoryInfo {
    public long memoryUsedMB;           // Used memory in MB
    public long memoryTotalMB;          // Total memory in MB
    public long memoryFreeMB;           // Free memory in MB

    public GPUMemoryInfo() {}

    public GPUMemoryInfo(long memoryUsedMB, long memoryTotalMB, long memoryFreeMB) {
        this.memoryUsedMB = memoryUsedMB;
        this.memoryTotalMB = memoryTotalMB;
        this.memoryFreeMB = memoryFreeMB;
    }

    public GPUMemoryInfo(GPUMemoryInfo other) {
        this.memoryUsedMB = other.memoryUsedMB;
        this.memoryTotalMB = other.memoryTotalMB;
        this.memoryFreeMB = other.memoryFreeMB;
    }

    /**
     * Get memory utilization as percentage based on used/total
     */
    public double getUtilizationPercent() {
        if (memoryTotalMB == 0) return 0.0;
        return (double) memoryUsedMB / memoryTotalMB * 100.0;
    }

    @Override
    public String toString() {
        return String.format("%dMB/%dMB (%.1f%% used)", memoryUsedMB, memoryTotalMB, getUtilizationPercent());
    }
}
