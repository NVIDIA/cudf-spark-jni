/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.jni;

import java.util.function.Supplier;

/**
 * This class contains utility methods for checking preconditions.
 */
public class Preconditions {
    /**
     * Check if the condition is true, otherwise throw an IllegalStateException with the given message.
     */
    public static void ensure(boolean condition, String message) {
        if (!condition) {
            throw new IllegalStateException(message);
        }
    }

    /**
     * Check if the condition is true, otherwise throw an IllegalStateException with the given message supplier.
     */
    public static void ensure(boolean condition, Supplier<String> messageSupplier) {
        if (!condition) {
            throw new IllegalStateException(messageSupplier.get());
        }
    }

    /**
     * Check if the value is non-negative, otherwise throw an IllegalArgumentException with the given message.
     * @param value the value to check
     * @param name the name of the value
     * @return the value if it is non-negative
     * @throws IllegalArgumentException if the value is negative
     */
    public static int ensureNonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be non-negative, but was " + value);
        }
        return value;
    }

    /**
     * Check if the value is non-negative, otherwise throw an IllegalArgumentException with the given message.
     * @param value the value to check
     * @param name the name of the value
     * @return the value if it is non-negative
     * @throws IllegalArgumentException if the value is negative
     */
    public static long ensureNonNegative(long value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be non-negative, but was " + value);
        }
        return value;
    }
}
