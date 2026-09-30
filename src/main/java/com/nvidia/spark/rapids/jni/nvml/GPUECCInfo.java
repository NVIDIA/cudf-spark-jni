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
 * GPU ECC error information from nvmlDeviceGetTotalEccErrors()
 */
public class GPUECCInfo {
    public long eccSingleBitErrors;     // Single bit ECC errors
    public long eccDoubleBitErrors;     // Double bit ECC errors

    public GPUECCInfo() {}

    public GPUECCInfo(long eccSingleBitErrors, long eccDoubleBitErrors) {
        this.eccSingleBitErrors = eccSingleBitErrors;
        this.eccDoubleBitErrors = eccDoubleBitErrors;
    }

    public GPUECCInfo(GPUECCInfo other) {
        this.eccSingleBitErrors = other.eccSingleBitErrors;
        this.eccDoubleBitErrors = other.eccDoubleBitErrors;
    }

    /**
     * Check if ECC is supported and has errors
     */
    public boolean hasErrors() {
        return eccSingleBitErrors > 0 || eccDoubleBitErrors > 0;
    }

    /**
     * Get total ECC errors
     */
    public long getTotalErrors() {
        return eccSingleBitErrors + eccDoubleBitErrors;
    }

    @Override
    public String toString() {
        if (!hasErrors()) {
            return "No ECC errors";
        }
        return String.format("%d single-bit, %d double-bit errors", eccSingleBitErrors, eccDoubleBitErrors);
    }
}
