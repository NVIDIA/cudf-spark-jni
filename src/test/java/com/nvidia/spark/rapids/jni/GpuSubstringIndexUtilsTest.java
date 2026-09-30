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

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Scalar;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

public class GpuSubstringIndexUtilsTest {
    @Test
    void gpuSubstringIndexTest(){
        Table.TestBuilder tb = new Table.TestBuilder();
        tb.column( "www.apache.org");
        tb.column("www.apache");
        tb.column("www");
        tb.column("");
        tb.column("org");
        tb.column("apache.org");
        tb.column("www.apache.org");
        tb.column("");
        tb.column("大千世界大");
        tb.column("www||apache");

        try(Table expected = tb.build()){
            Table.TestBuilder tb2 = new Table.TestBuilder();
            tb2.column("www.apache.org");
            tb2.column("www.apache.org");
            tb2.column("www.apache.org");
            tb2.column("www.apache.org");
            tb2.column("www.apache.org");
            tb2.column("www.apache.org");
            tb2.column("www.apache.org");
            tb2.column("");
            tb2.column("大千世界大千世界");
            tb2.column("www||apache||org");

            try (Scalar dotScalar = Scalar.fromString(".");
                 Scalar cnChar = Scalar.fromString("千");
                 Scalar verticalBar = Scalar.fromString("||")) {
                Scalar[] delimiterArray = new Scalar[] { dotScalar, dotScalar, dotScalar, dotScalar, dotScalar,
                        dotScalar, dotScalar, dotScalar, cnChar, verticalBar };
                int[] countArray = new int[] { 3, 2, 1, 0, -1, -2, -3, -2, 2, 2 };
                List<ColumnVector> result = new ArrayList<>();
                try (Table origTable = tb2.build()) {
                    for (int i = 0; i < origTable.getNumberOfColumns(); i++) {
                        ColumnVector string_col = origTable.getColumn(i);
                        result.add(GpuSubstringIndexUtils.substringIndex(string_col, delimiterArray[i], countArray[i]));
                    }
                    try (Table result_tbl = new Table(
                            result.toArray(new ColumnVector[result.size()]))) {
                        AssertUtils.assertTablesAreEqual(expected, result_tbl);
                    }
                } finally {
                    result.forEach(ColumnVector::close);
                }
            }
        }
    }
}
