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
 * GPU hardware information from various NVML calls
 * (nvmlDeviceGetNumGpuCores, nvmlDeviceGetPerformanceState, nvmlDeviceGetFanSpeed)
 */
public class GPUHardwareInfo {
    public int streamingMultiprocessors; // Number of SMs
    public int performanceState;        // Performance state (P-state)
    public int fanSpeedPercent;         // Fan speed percentage

    public GPUHardwareInfo() {}

    public GPUHardwareInfo(int streamingMultiprocessors, int performanceState, int fanSpeedPercent) {
        this.streamingMultiprocessors = streamingMultiprocessors;
        this.performanceState = performanceState;
        this.fanSpeedPercent = fanSpeedPercent;
    }

    public GPUHardwareInfo(GPUHardwareInfo other) {
        this.streamingMultiprocessors = other.streamingMultiprocessors;
        this.performanceState = other.performanceState;
        this.fanSpeedPercent = other.fanSpeedPercent;
    }

    /**
     * Get performance state description
     */
    public String getPerformanceStateDescription() {
        switch (performanceState) {
            case 0: return "P0 (Maximum Performance)";
            case 1: return "P1 (High Performance)";
            case 2: return "P2 (Moderate Performance)";
            case 3: return "P3 (Battery Optimized)";
            case 8: return "P8 (Minimum Performance)";
            case 12: return "P12 (Lowest Performance)";
            default: return "P" + performanceState;
        }
    }

    @Override
    public String toString() {
        return String.format("%d SMs, %s, Fan: %d%%", 
                           streamingMultiprocessors, getPerformanceStateDescription(), fanSpeedPercent);
    }
}
