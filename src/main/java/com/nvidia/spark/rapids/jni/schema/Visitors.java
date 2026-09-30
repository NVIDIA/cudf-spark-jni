/*
 *
 *  Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package com.nvidia.spark.rapids.jni.schema;

import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.HostColumnVectorCore;
import ai.rapids.cudf.Schema;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.util.Objects.requireNonNull;

/**
 * A utility class for visiting a schema or a list of host columns.
 */
public class Visitors {
    /**
     * Visiting a schema in post order. For more details, see {@link SchemaVisitor}.
     *
     * @param schema the schema to visit
     * @param visitor the visitor to use
     * @param <T> Return type when visiting intermediate nodes. See {@link SchemaVisitor}
     * @param <P> Return type when previsiting a list. See {@link SchemaVisitor}
     * @param <R> Return type after processing all children values. See {@link SchemaVisitor}
     * @return the result of visiting the schema
     */
    public static <T, P, R> R visitSchema(Schema schema, SchemaVisitor<T, P, R> visitor) {
        requireNonNull(schema, "schema cannot be null");
        requireNonNull(visitor, "visitor cannot be null");

        List<T> childrenResult = IntStream.range(0, schema.getNumChildren())
                .mapToObj(i -> visitSchemaInner(schema.getChild(i), visitor))
                .collect(Collectors.toList());

        return visitor.visitTopSchema(schema, childrenResult);
    }

    public static void visitSchema(Schema schema, SimpleSchemaVisitor visitor) {
        requireNonNull(schema, "schema cannot be null");
        requireNonNull(visitor, "visitor cannot be null");

        for (int i = 0; i < schema.getNumChildren(); i++) {
            visitSchemaInner(schema.getChild(i), visitor);
        }
        visitor.visitTopSchema(schema);
    }

    private static void visitSchemaInner(Schema schema, SimpleSchemaVisitor visitor) {
        switch (schema.getType().getTypeId()) {
            case STRUCT:
                visitor.preVisitStruct(schema);
                for (int i=0; i<schema.getNumChildren(); i++) {
                    visitSchemaInner(schema.getChild(i), visitor);
                }
                visitor.visitStruct(schema);
                break;
            case LIST:
                visitor.preVisitList(schema);
                visitSchemaInner(schema.getChild(0), visitor);
                visitor.visitList(schema);
                break;
            default:
                visitor.visit(schema);
                break;
        }
    }

    private static <T, P, R> T visitSchemaInner(Schema schema, SchemaVisitor<T, P, R> visitor) {
        switch (schema.getType().getTypeId()) {
            case STRUCT: {
                P preVisitResult = visitor.preVisitStruct(schema);
                List<T> children = IntStream.range(0, schema.getNumChildren())
                    .mapToObj(childIdx -> visitSchemaInner(schema.getChild(childIdx), visitor))
                    .collect(Collectors.toList());
                return visitor.visitStruct(schema, preVisitResult, children);
            }
            case LIST: {
                P preVisitResult = visitor.preVisitList(schema);
                T childResult = visitSchemaInner(schema.getChild(0), visitor);
                return visitor.visitList(schema, preVisitResult, childResult);
            }
            default:
                return visitor.visit(schema);
        }
    }


    /**
     * Visiting a list of host columns in post order. For more details, see {@link HostColumnsVisitor}.
     *
     * @param cols the list of host columns to visit
     * @param visitor the visitor to use
     */
    public static void visitColumns(HostColumnVector[] cols,
                                        HostColumnsVisitor visitor) {
        requireNonNull(cols, "cols cannot be null");
        requireNonNull(visitor, "visitor cannot be null");

        for (HostColumnVector col : cols) {
            visitColumn(col, visitor);
        }
        visitor.done();
    }

    private static void visitColumn(HostColumnVectorCore col, HostColumnsVisitor visitor) {
        switch (col.getType().getTypeId()) {
            case STRUCT:
                visitor.preVisitStruct(col);
                for (int i=0; i<col.getNumChildren(); i++) {
                    visitColumn(col.getChildColumnView(i), visitor);
                }
                visitor.visitStruct(col);
                return;
            case LIST:
                visitor.preVisitList(col);
                visitColumn(col.getChildColumnView(0), visitor);
                visitor.visitList(col);
                return;
            default:
                visitor.visit(col);
        }
    }
}
