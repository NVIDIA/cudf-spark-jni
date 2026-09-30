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
 * GPU PCIe information from nvmlDeviceGetCurrPcieLinkGeneration() and nvmlDeviceGetCurrPcieLinkWidth()
 */
public class GPUPCIeInfo {
    public int pcieLinkGeneration;      // PCIe generation (1, 2, 3, 4, etc.)
    public int pcieLinkWidth;           // PCIe width (x1, x4, x8, x16)

    public GPUPCIeInfo() {}

    public GPUPCIeInfo(int pcieLinkGeneration, int pcieLinkWidth) {
        this.pcieLinkGeneration = pcieLinkGeneration;
        this.pcieLinkWidth = pcieLinkWidth;
    }

    public GPUPCIeInfo(GPUPCIeInfo other) {
        this.pcieLinkGeneration = other.pcieLinkGeneration;
        this.pcieLinkWidth = other.pcieLinkWidth;
    }

    /**
     * Get PCIe generation description
     */
    public String getDescription() {
        return "PCIe Gen" + pcieLinkGeneration + " x" + pcieLinkWidth;
    }

    @Override
    public String toString() {
        return getDescription();
    }
}
