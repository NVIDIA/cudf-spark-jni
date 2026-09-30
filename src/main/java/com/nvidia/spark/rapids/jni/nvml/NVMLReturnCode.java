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
 * NVML return codes mapped to meaningful enum values.
 * These correspond to the nvmlReturn_t values from the NVML library.
 */
public enum NVMLReturnCode {
    /** The operation was successful */
    SUCCESS(0),

    /** NVML was not first initialized with nvmlInit() */
    ERROR_UNINITIALIZED(1),

    /** A supplied argument is invalid */
    ERROR_INVALID_ARGUMENT(2),

    /** The requested operation is not available on target device */
    ERROR_NOT_SUPPORTED(3),

    /** The current user does not have permission for operation */
    ERROR_NO_PERMISSION(4),

    /** Deprecated: Multiple initializations are now allowed through ref counting */
    ERROR_ALREADY_INITIALIZED(5),

    /** A query to find an object was unsuccessful */
    ERROR_NOT_FOUND(6),

    /** An input argument is not large enough */
    ERROR_INSUFFICIENT_SIZE(7),

    /** A device's external power cables are not properly attached */
    ERROR_INSUFFICIENT_POWER(8),

    /** NVIDIA driver is not loaded */
    ERROR_DRIVER_NOT_LOADED(9),

    /** User provided timeout passed */
    ERROR_TIMEOUT(10),

    /** NVIDIA Kernel detected an interrupt issue with a GPU */
    ERROR_IRQ_ISSUE(11),

    /** NVML Shared Library couldn't be found or loaded */
    ERROR_LIBRARY_NOT_FOUND(12),

    /** Local version of NVML doesn't implement this function */
    ERROR_FUNCTION_NOT_FOUND(13),

    /** infoROM is corrupted */
    ERROR_CORRUPTED_INFOROM(14),

    /** The GPU has fallen off the bus or has otherwise become inaccessible */
    ERROR_GPU_IS_LOST(15),

    /** The GPU requires a reset before it can be used again */
    ERROR_RESET_REQUIRED(16),

    /** The GPU control device has been blocked by the operating system/cgroups */
    ERROR_OPERATING_SYSTEM(17),

    /** RM detects a driver/library version mismatch */
    ERROR_LIB_RM_VERSION_MISMATCH(18),

    /** An operation cannot be performed because the GPU is currently in use */
    ERROR_IN_USE(19),

    /** Insufficient memory */
    ERROR_MEMORY(20),

    /** No data */
    ERROR_NO_DATA(21),

    /** The requested vgpu operation is not available on target device, because ECC is enabled */
    ERROR_VGPU_ECC_NOT_SUPPORTED(22),

    /** Ran out of critical resources, other than memory */
    ERROR_INSUFFICIENT_RESOURCES(23),

    /** Ran out of critical resources, other than memory */
    ERROR_FREQ_NOT_SUPPORTED(24),

    /** The provided version is invalid/unsupported */
    ERROR_ARGUMENT_VERSION_MISMATCH(25),

    /** The requested functionality has been deprecated */
    ERROR_DEPRECATED(26),

    /** The system is not ready for the request */
    ERROR_NOT_READY(27),

    /** No GPUs were found */
    ERROR_GPU_NOT_FOUND(28),

    /** Resource not in correct state to perform requested operation */
    ERROR_INVALID_STATE(29),

    /** An internal driver error occurred */
    ERROR_UNKNOWN(999);

    private final int value;

    NVMLReturnCode(int value) {
        this.value = value;
    }

    /**
     * @return The integer value of this return code
     */
    public int getValue() {
        return value;
    }

    /**
     * Get the NVMLReturnCode enum for the given integer value
     * @param value The integer return code value
     * @return The corresponding enum value, or ERROR_UNKNOWN if not found
     */
    public static NVMLReturnCode fromValue(int value) {
        switch (value) {
            case 0: return SUCCESS;
            case 1: return ERROR_UNINITIALIZED;
            case 2: return ERROR_INVALID_ARGUMENT;
            case 3: return ERROR_NOT_SUPPORTED;
            case 4: return ERROR_NO_PERMISSION;
            case 5: return ERROR_ALREADY_INITIALIZED;
            case 6: return ERROR_NOT_FOUND;
            case 7: return ERROR_INSUFFICIENT_SIZE;
            case 8: return ERROR_INSUFFICIENT_POWER;
            case 9: return ERROR_DRIVER_NOT_LOADED;
            case 10: return ERROR_TIMEOUT;
            case 11: return ERROR_IRQ_ISSUE;
            case 12: return ERROR_LIBRARY_NOT_FOUND;
            case 13: return ERROR_FUNCTION_NOT_FOUND;
            case 14: return ERROR_CORRUPTED_INFOROM;
            case 15: return ERROR_GPU_IS_LOST;
            case 16: return ERROR_RESET_REQUIRED;
            case 17: return ERROR_OPERATING_SYSTEM;
            case 18: return ERROR_LIB_RM_VERSION_MISMATCH;
            case 19: return ERROR_IN_USE;
            case 20: return ERROR_MEMORY;
            case 21: return ERROR_NO_DATA;
            case 22: return ERROR_VGPU_ECC_NOT_SUPPORTED;
            case 23: return ERROR_INSUFFICIENT_RESOURCES;
            case 24: return ERROR_FREQ_NOT_SUPPORTED;
            case 25: return ERROR_ARGUMENT_VERSION_MISMATCH;
            case 26: return ERROR_DEPRECATED;
            case 27: return ERROR_NOT_READY;
            case 28: return ERROR_GPU_NOT_FOUND;
            case 29: return ERROR_INVALID_STATE;
            default: return ERROR_UNKNOWN;
        }
    }
}
