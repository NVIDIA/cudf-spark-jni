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
 * GPU power information from nvmlDeviceGetPowerUsage() and nvmlDeviceGetPowerManagementLimit()
 */
public class GPUPowerInfo {
    public int powerUsageW;             // Current power usage in Watts
    public int powerLimitW;             // Current power limit in Watts

    public GPUPowerInfo() {}

    public GPUPowerInfo(int powerUsageW, int powerLimitW) {
        this.powerUsageW = powerUsageW;
        this.powerLimitW = powerLimitW;
    }

    public GPUPowerInfo(GPUPowerInfo other) {
        this.powerUsageW = other.powerUsageW;
        this.powerLimitW = other.powerLimitW;
    }

    /**
     * Get power utilization as percentage of current limit
     */
    public double getUtilizationPercent() {
        if (powerLimitW == 0) return 0.0;
        return (double) powerUsageW / powerLimitW * 100.0;
    }

    @Override
    public String toString() {
        return String.format("%dW/%dW (%.1f%% of limit)", powerUsageW, powerLimitW, getUtilizationPercent());
    }
}
