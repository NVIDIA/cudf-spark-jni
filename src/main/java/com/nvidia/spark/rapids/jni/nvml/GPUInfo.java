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
 * Comprehensive GPU information collected via NVML
 * Contains all major GPU metrics for detailed monitoring and analysis
 *
 * This class provides fine-grained access via nested info objects,
 * with each object corresponding to a specific NVML API call group.
 */
public class GPUInfo {

    // Nested info objects for fine-grained API access
    public GPUDeviceInfo deviceInfo;
    public GPUUtilizationInfo utilizationInfo;
    public GPUMemoryInfo memoryInfo;
    public GPUTemperatureInfo temperatureInfo;
    public GPUPowerInfo powerInfo;
    public GPUClockInfo clockInfo;
    public GPUHardwareInfo hardwareInfo;
    public GPUPCIeInfo pcieInfo;
    public GPUECCInfo eccInfo;

    // Timestamp when this info was collected
    public long timestampMs;

    public GPUInfo() {
        this.timestampMs = System.currentTimeMillis();
        // Initialize nested objects
        this.deviceInfo = new GPUDeviceInfo();
        this.utilizationInfo = new GPUUtilizationInfo();
        this.memoryInfo = new GPUMemoryInfo();
        this.temperatureInfo = new GPUTemperatureInfo();
        this.powerInfo = new GPUPowerInfo();
        this.clockInfo = new GPUClockInfo();
        this.hardwareInfo = new GPUHardwareInfo();
        this.pcieInfo = new GPUPCIeInfo();
        this.eccInfo = new GPUECCInfo();
    }

    /**
     * Constructor that takes nested info objects
     */
    public GPUInfo(GPUDeviceInfo deviceInfo, GPUUtilizationInfo utilizationInfo,
                   GPUMemoryInfo memoryInfo, GPUTemperatureInfo temperatureInfo,
                   GPUPowerInfo powerInfo, GPUClockInfo clockInfo,
                   GPUHardwareInfo hardwareInfo, GPUPCIeInfo pcieInfo, GPUECCInfo eccInfo) {
        this.timestampMs = System.currentTimeMillis();

        this.deviceInfo = deviceInfo;
        this.utilizationInfo = utilizationInfo;
        this.memoryInfo = memoryInfo;
        this.temperatureInfo = temperatureInfo;
        this.powerInfo = powerInfo;
        this.clockInfo = clockInfo;
        this.hardwareInfo = hardwareInfo;
        this.pcieInfo = pcieInfo;
        this.eccInfo = eccInfo;
    }

    /**
     * Copy constructor for creating snapshots
     */
    public GPUInfo(GPUInfo other) {
        this.timestampMs = System.currentTimeMillis();

        // Delegate to nested objects' copy constructors
        this.deviceInfo = (other.deviceInfo != null) ? new GPUDeviceInfo(other.deviceInfo) : null;
        this.utilizationInfo = (other.utilizationInfo != null) ? new GPUUtilizationInfo(other.utilizationInfo) : null;
        this.memoryInfo = (other.memoryInfo != null) ? new GPUMemoryInfo(other.memoryInfo) : null;
        this.temperatureInfo = (other.temperatureInfo != null) ? new GPUTemperatureInfo(other.temperatureInfo) : null;
        this.powerInfo = (other.powerInfo != null) ? new GPUPowerInfo(other.powerInfo) : null;
        this.clockInfo = (other.clockInfo != null) ? new GPUClockInfo(other.clockInfo) : null;
        this.hardwareInfo = (other.hardwareInfo != null) ? new GPUHardwareInfo(other.hardwareInfo) : null;
        this.pcieInfo = (other.pcieInfo != null) ? new GPUPCIeInfo(other.pcieInfo) : null;
        this.eccInfo = (other.eccInfo != null) ? new GPUECCInfo(other.eccInfo) : null;
    }


    /**
     * Format basic GPU info for display
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        if (deviceInfo != null) {
            sb.append(deviceInfo.toString());
        } else {
            sb.append("GPU_Unknown");
        }

        if (utilizationInfo != null) {
            sb.append(": ").append(utilizationInfo.toString());
        }

        if (memoryInfo != null) {
            sb.append(", ").append(memoryInfo.toString());
        }

        if (temperatureInfo != null) {
            sb.append(", ").append(temperatureInfo.toString());
        }

        if (powerInfo != null) {
            sb.append(", ").append(powerInfo.toString());
        }

        if (clockInfo != null) {
            sb.append(", ").append(clockInfo.toString());
        }

        return sb.toString();
    }

    /**
     * Get detailed multi-line format with NVML API annotations
     */
    public String toDetailedString() {
        StringBuilder sb = new StringBuilder();

        if (deviceInfo != null) {
            sb.append(deviceInfo.toString()).append("\n");
        } else {
            sb.append("GPU_Unknown: N/A\n");
        }

        if (utilizationInfo != null) {
            sb.append("  Utilization: ").append(utilizationInfo.toString()).append(" [nvmlDeviceGetUtilizationRates()]\n");
        }

        if (memoryInfo != null) {
            sb.append("  Memory: ").append(memoryInfo.toString()).append(" [nvmlDeviceGetMemoryInfo()]\n");
        }

        if (temperatureInfo != null) {
            sb.append("  Temperature: ").append(temperatureInfo.toString()).append(" [nvmlDeviceGetTemperature()]\n");
        }

        if (powerInfo != null) {
            sb.append("  Power: ").append(powerInfo.toString()).append(" [nvmlDeviceGetPowerUsage/Limit]\n");
        }

        if (clockInfo != null) {
            sb.append("  Clocks: ").append(clockInfo.toString()).append(" [nvmlDeviceGetClockInfo()]\n");
        }

        if (hardwareInfo != null) {
            sb.append("  Hardware: ").append(hardwareInfo.toString()).append(" [nvmlDeviceGetNumGpuCores/PerformanceState/FanSpeed]\n");
        }

        if (pcieInfo != null) {
            sb.append("  PCIe: ").append(pcieInfo.toString()).append(" [nvmlDeviceGetCurrPcieLinkGeneration/Width]\n");
        }

        if (eccInfo != null) {
            sb.append("  ECC: ").append(eccInfo.toString()).append(" [nvmlDeviceGetTotalEccErrors]\n");
        }

        return sb.toString();
    }

    /**
     * Get compact single-line format for logs
     */
    public String toCompactString() {
        StringBuilder sb = new StringBuilder();

        if (deviceInfo != null && deviceInfo.name != null) {
            sb.append(deviceInfo.name);
        } else {
            sb.append("GPU_Unknown");
        }

        sb.append(":");

        if (utilizationInfo != null) {
            sb.append(" ").append(utilizationInfo.gpuUtilization).append("%/").append(utilizationInfo.memoryUtilization).append("%");
        }

        if (memoryInfo != null) {
            sb.append(" ").append(memoryInfo.memoryUsedMB).append("MB");
        }

        if (temperatureInfo != null) {
            sb.append(" ").append(temperatureInfo.temperatureGpu).append("°C");
        }

        if (powerInfo != null) {
            sb.append(" ").append(powerInfo.powerUsageW).append("W");
        }

        if (clockInfo != null) {
            sb.append(" ").append(clockInfo.graphicsClockMHz).append("MHz");
        }

        return sb.toString();
    }
}
