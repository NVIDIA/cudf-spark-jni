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

import ai.rapids.cudf.Schema;

import java.util.List;

/**
 * A post order visitor for schemas.
 *
 * <h1>Flattened Schema</h1>
 *
 * A flattened schema is a schema where all fields with nested types are flattened into an array of fields. For example,
 * for a schema with following fields:
 *
 * <ul>
 *    <li> A: <code>struct { int a1; long a2} </code> </li>
 *    <li> B: <code>list { int b1} </code> </li>
 *    <li> C: <code>string </code> </li>
 *    <li> D: <code>long </code> </li>
 * </ul>
 *
 * The flattened schema will be:
 *
 * <ul>
 *   <li> A: <code>struct</code> </li>
 *   <li> A.a1: <code>int</code> </li>
 *   <li> A.a2: <code>long</code> </li>
 *   <li> B: <code>list</code> </li>
 *   <li> B.b1: <code>int</code> </li>
 *   <li> C: <code>string</code> </li>
 *   <li> D: <code>long</code> </li>
 * </ul>
 *
 * <h1>Example</h1>
 *
 * <p>
 * This visitor visits each filed in the flattened schema in post order. For example, if our schema consists of three
 * fields A, B, and C with following fields:
 * <ul>
 *    <li> A: <code>struct { int a1; long a2} </code> </li>
 *    <li> B: <code>list { int b1} </code> </li>
 *    <li> C: <code>string </code> </li>
 * </ul>
 *
 * The order of visiting will be:
 * <ol>
 *     <li> Visit primitive field a1 </li>
 *     <li> Visit primitive field a2</li>
 *     <li> Visit struct field A, with results from a1 and a2</li>
 *     <li> Previsit list field B</li>
 *     <li> Visit primitive field b1</li>
 *     <li> Visit list field B with results from b1 and previsit result. </li>
 *     <li> Visit primitive field C</li>
 *     <li> Visit top schema with results from fields A, B, and C</li>
 * </ol>
 *
 * </p>
 *
 * @param <T> Return type when visiting intermediate nodes.
 * @param <P> Return type after visiting a list schema before visiting its child.
 * @param <R> Return type after processing all children values.
 */
public interface SchemaVisitor<T, P, R> {
    /**
     * Visit the top level schema.
     * @param schema the top level schema to visit
     * @param children the results of visiting the children
     * @return the result of visiting the top level schema
     */
    R visitTopSchema(Schema schema, List<T> children);

    /**
     * Visit a struct schema before actually visiting its children.
     * @param structType the struct schema to visit
     * @return the result of visiting the struct schema
     */
    P preVisitStruct(Schema structType);

    /**
     * Visit a struct schema.
     * @param structType the struct schema to visit
     * @param preVisitResult the result of visiting the struct schema before visiting its children
     * @param children the results of visiting the children
     * @return the result of visiting the struct schema
     */
    T visitStruct(Schema structType, P preVisitResult, List<T> children);

    /**
     * Visit a list schema before actually visiting its child.
     * @param listType the list schema to visit
     * @return the result of visiting the list schema
     */
    P preVisitList(Schema listType);

    /**
     * Visit a list schema after visiting its child.
     * @param listType the list schema to visit
     * @param preVisitResult the result of visiting the list schema before visiting its child
     * @param childResult the result of visiting the child
     * @return the result of visiting the list schema
     */
    T visitList(Schema listType, P preVisitResult, T childResult);

    /**
     * Visit a primitive type.
     * @param primitiveType the primitive type to visit
     * @return the result of visiting the primitive type
     */
    T visit(Schema primitiveType);
}
