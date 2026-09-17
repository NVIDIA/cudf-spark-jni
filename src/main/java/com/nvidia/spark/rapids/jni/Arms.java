/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.function.Function;

/**
 * This class contains utility methods for automatic resource management.
 */
public class Arms {
    @FunctionalInterface
    public interface ThrowingFunction<R, T, E extends Exception> {
        T apply(R resource) throws E;
    }

    /**
     * This method close the resource if an exception is thrown while executing the function.
     */
    public static <R extends AutoCloseable, T> T closeIfException(R resource, Function<R, T> function) {
        try {
            return function.apply(resource);
        } catch (Throwable e) {
            closeAndCollect(resource, e);
            throw e;
        }
    }

    /**
     * Like closeIfException, but preserves checked exceptions thrown by the function.
     */
    public static <R extends AutoCloseable, T, E extends Exception> T closeIfExceptionChecked(
        R resource, ThrowingFunction<R, T, E> function) throws E {
        try {
            return function.apply(resource);
        } catch (Throwable e) {
            closeAndCollect(resource, e);
            throw e;
        }
    }

    /**
     * This method safely closes all the resources.
     * <p>
     * This method will iterate through all the resources and closes them. If any exception happened during the
     * traversal, exception will be captured and rethrown after all resources closed. Errors and runtime
     * exceptions are rethrown unchanged; checked exceptions are wrapped in RuntimeException.
     * </p>
     */
    public static <R extends AutoCloseable> void closeAll(Iterator<R> resources) {
        rethrowUnchecked(closeAll(resources, null));
    }

    private static Throwable closeAll(Iterator<? extends AutoCloseable> resources, Throwable primary) {
        while (resources.hasNext()) {
            try {
                primary = closeAndCollect(resources.next(), primary);
            } catch (Throwable e) {
                primary = collectFailure(primary, e);
            }
        }
        return primary;
    }

    private static Throwable closeAndCollect(AutoCloseable resource, Throwable primary) {
        if (resource != null) {
            try {
                resource.close();
            } catch (Throwable e) {
                return collectFailure(primary, e);
            }
        }
        return primary;
    }

    private static Throwable collectFailure(Throwable primary, Throwable failure) {
        if (primary == null) {
            return failure;
        }
        if (primary != failure) {
            primary.addSuppressed(failure);
        }
        return primary;
    }

    private static void rethrowUnchecked(Throwable failure) {
        if (failure instanceof Error) {
            throw (Error) failure;
        }
        if (failure instanceof RuntimeException) {
            throw (RuntimeException) failure;
        }
        if (failure != null) {
            throw new RuntimeException(failure);
        }
    }


    /**
     * This method safely closes all the resources. See {@link #closeAll(Iterator)} for more details.
     */
    public static <R extends AutoCloseable> void closeAll(R... resources) {
        closeAll(Arrays.asList(resources));
    }

    /**
     * This method safely closes the resources. See {@link #closeAll(Iterator)} for more details.
     */
    public static <R extends AutoCloseable> void closeAll(Collection<R> resources) {
        closeAll(resources.iterator());
    }

    /**
     * This method safely closes the resources after applying the function.
     * If the function fails, cleanup failures are suppressed on the original failure.
     * <br/>
     * See {@link #closeAll(Iterator)} for more details.
     */
    public static <R extends AutoCloseable, C extends Collection<R>, V> V withResource(
        C resource, Function<C, V> function) {
        V result;
        try {
            result = function.apply(resource);
        } catch (Throwable primary) {
            try {
                closeAll(resource.iterator(), primary);
            } catch (Throwable e) {
                collectFailure(primary, e);
            }
            throw primary;
        }
        closeAll(resource);
        return result;
    }
}
