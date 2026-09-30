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
 * GPU clock information from nvmlDeviceGetClockInfo() calls
 */
public class GPUClockInfo {
    public int graphicsClockMHz;        // Graphics clock in MHz
    public int memoryClockMHz;          // Memory clock in MHz
    public int smClockMHz;              // SM clock in MHz

    public GPUClockInfo() {}

    public GPUClockInfo(int graphicsClockMHz, int memoryClockMHz, int smClockMHz) {
        this.graphicsClockMHz = graphicsClockMHz;
        this.memoryClockMHz = memoryClockMHz;
        this.smClockMHz = smClockMHz;
    }

    public GPUClockInfo(GPUClockInfo other) {
        this.graphicsClockMHz = other.graphicsClockMHz;
        this.memoryClockMHz = other.memoryClockMHz;
        this.smClockMHz = other.smClockMHz;
    }

    @Override
    public String toString() {
        return String.format("Graphics: %dMHz, Memory: %dMHz, SM: %dMHz", 
                           graphicsClockMHz, memoryClockMHz, smClockMHz);
    }
}
