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

import java.lang.reflect.Field;
import java.util.Objects;

/**
 * Utilities for working with Fields by reflection.
 * Similar to org.apache.commons.lang3.reflect.FieldUtils
 */
public class FieldUtils {

  /**
   * Reads a field value via reflection from a target object.
   * @param target target object to read from
   * @param fieldName name of the field to read
   * @return the field value
   * @throws IllegalStateException if the field cannot be read for any reason
   */
  public static Object readField(Object target, String fieldName) {
    return readField(target, fieldName, true);
  }

  /**
   * Reads a field value via reflection from a target object.
   * @param target target object to read from
   * @param fieldName name of the field to read
   * @param forceAccess if true, will call setAccessible(true) on the field
   * @return the field value
   * @throws IllegalStateException if the field cannot be read for any reason
   */
  public static Object readField(Object target, String fieldName, boolean forceAccess) {
    Objects.requireNonNull(target);
    try {
      Class<?> cls = target.getClass();
      Field field = cls.getDeclaredField(fieldName);
      if (forceAccess) {
        field.setAccessible(true);
      }
      return field.get(target);
    } catch (Exception e) {
      throw new IllegalStateException("Unable to read field " + fieldName + " on " + target, e);
    }
  }
}
