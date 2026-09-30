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
 * Generic result wrapper for NVML operations that contains both the success code
 * and the actual data. This allows users to check if NVML operations succeeded
 * and get the specific error code if they failed.
 *
 * @param <T> The type of data returned by the NVML operation
 */
public class NVMLResult<T> {
    int returnCode;  // NVML return code (0 = NVML_SUCCESS) - package-private for JNI
    T data;          // The actual data, may be null if operation failed - package-private for JNI

    /**
     * Create a new NVMLResult with default values (used by JNI)
     */
    public NVMLResult() {
        this.returnCode = -1; // Default to error state
        this.data = null;
    }

    /**
     * Create a new NVMLResult
     * @param returnCode The NVML return code (0 = success)
     * @param data The data object, may be null on failure
     */
    public NVMLResult(int returnCode, T data) {
        this.returnCode = returnCode;
        this.data = data;
    }

    /**
     * @return true if the NVML operation succeeded (returnCode == 0)
     */
    public boolean isSuccess() {
        return returnCode == 0;
    }

    /**
     * @return The NVML return code (0 = NVML_SUCCESS)
     */
    public int getReturnCode() {
        return returnCode;
    }

    /**
     * @return The NVML return code as an enum value
     */
    public NVMLReturnCode getReturnCodeEnum() {
        return NVMLReturnCode.fromValue(returnCode);
    }

    /**
     * @return The data object, may be null if the operation failed
     */
    public T getData() {
        return data;
    }

    /**
     * Get the data object if successful, otherwise throw an exception
     * @return The data object
     * @throws NVMLException if the operation failed
     */
    public T getDataOrThrow() throws NVMLException {
        if (!isSuccess()) {
            throw new NVMLException(returnCode);
        }
        return data;
    }

    @Override
    public String toString() {
        return String.format("NVMLResult{success=%s, returnCode=%s(%d), data=%s}",
                           isSuccess(), getReturnCodeEnum(), returnCode, data);
    }
}
