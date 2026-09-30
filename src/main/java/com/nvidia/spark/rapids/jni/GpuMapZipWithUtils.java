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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.*;

import java.util.Map;

public class GpuMapZipWithUtils {
    static{
        NativeDepsLoader.loadNativeDeps();
    }
    /**
     * @brief Zip two lists columns row-wise to create key-value pairs
     *
     * The map_zip function combines two lists columns row-wise to create key-value pairs,
     * similar to Spark SQL's map_zip_with function. It takes two input
     * columns where each row contains lists of key-value pairs, merges
     * them based on matching keys, and produces a result where each key
     * maps to a tuple containing the corresponding values from both
     * input columns (with NULL values for missing keys).
     *
     * @code{.pseudo}
     * col1 = [
     *   [(1,100), (2, 200), (3, 300), (4, 400)],
     *   [(5,500), (6,600), (7,700)],
     * ]
     * col2 = [
     *   [(2,20), (4,40), (8,80)],
     *   [(9,90), (6,60), (10,100)],
     * ]
     *
     * result = [
     *   [(1, (100, NULL)), (2, (200,20)), (3, (300, NULL)), (4, (400,40)), (8, (NULL, 80))],
     *   [(5, (500, NULL)), (6, (600, 60)), (7, (700, NULL)), (9, (NULL,90)), (10, (NULL, 100))],
     * ]
     * @endcode
     *
     * @param col1 The first lists column containing key-value pairs
     * @param col2 The second lists column containing key-value pairs
     * @param stream CUDA stream for asynchronous execution (default: default stream)
     * @param mr Memory resource for device memory allocation (default: current device resource)
     *
     * @return A unique pointer to the column of zipped maps
     *
     * @note Both input columns must have the same number of rows.
     *
     * @note The function preserves the null mask and validity of the input columns.
     */
    public static ColumnVector mapZip(ColumnView input1, ColumnView input2) {
        return new ColumnVector(mapZip(input1.getNativeView(), input2.getNativeView()));
    }

    private static native long mapZip(long input1, long input2);
}