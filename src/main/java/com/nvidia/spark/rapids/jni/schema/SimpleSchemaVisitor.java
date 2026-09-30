/*
 *
 *  Copyright (c) 2025, NVIDIA CORPORATION.
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

/**
 * A schema visitor similar to {@link SchemaVisitor} but with a simplified interface, please refer
 * to {@link SchemaVisitor} for more details.
 * <br/>
 * This interface removed generic types and return values to simplify the interface, which could
 * avoid unnecessary allocation during visiting schema.
 */
public interface SimpleSchemaVisitor {
    /**
     * Visit the top level schema.
     * @param schema the top level schema to visit
     */
    void visitTopSchema(Schema schema);

    /**
     * Visit a struct schema before actually visiting its children.
     * @param structType the struct schema to visit
     */
    void preVisitStruct(Schema structType);

    /**
     * Visit a struct schema.
     * @param structType the struct schema to visit
     */
    void visitStruct(Schema structType);

    /**
     * Visit a list schema before actually visiting its child.
     * @param listType the list schema to visit
     */
    void preVisitList(Schema listType);

    /**
     * Visit a list schema after visiting its child.
     * @param listType the list schema to visit
     */
    void visitList(Schema listType);

    /**
     * Visit a primitive type.
     * @param primitiveType the primitive type to visit
     */
    void visit(Schema primitiveType);
}
