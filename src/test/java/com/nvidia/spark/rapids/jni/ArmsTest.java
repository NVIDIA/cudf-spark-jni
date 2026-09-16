/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.*;

class ArmsTest {
  private static class RecordingResource implements AutoCloseable {
    private final Throwable failure;
    private int closeCount;

    RecordingResource(Throwable failure) {
      this.failure = failure;
    }

    @Override
    public void close() throws Exception {
      closeCount++;
      if (failure instanceof Error) {
        throw (Error) failure;
      }
      if (failure != null) {
        throw (Exception) failure;
      }
    }
  }

  static Stream<Throwable> failures() {
    return Stream.of(new IllegalStateException("body"), new AssertionError("body"),
        new OutOfMemoryError("body"));
  }

  private static void failUnchecked(Throwable failure) {
    if (failure instanceof Error) {
      throw (Error) failure;
    }
    throw (RuntimeException) failure;
  }

  @Test
  void closeIfExceptionTransfersOwnershipOnSuccess() throws IOException {
    RecordingResource resource = new RecordingResource(null);
    assertSame(resource, Arms.closeIfException(resource, r -> r));
    assertSame(resource, Arms.<RecordingResource, RecordingResource, IOException>
        closeIfExceptionChecked(resource, r -> r));
    assertEquals(0, resource.closeCount);
  }

  @ParameterizedTest
  @MethodSource("failures")
  void closeIfExceptionPreservesBodyFailure(Throwable failure) {
    Error closeFailure = new AssertionError("close");
    RecordingResource resource = new RecordingResource(closeFailure);
    Throwable actual = assertThrows(failure.getClass(), () ->
        Arms.closeIfException(resource, r -> {
          failUnchecked(failure);
          return null;
        }));
    assertSame(failure, actual);
    assertArrayEquals(new Throwable[]{closeFailure}, actual.getSuppressed());
    assertEquals(1, resource.closeCount);
  }

  @Test
  void checkedCallbackPreservesIOException() {
    IOException failure = new IOException("read");
    Error closeFailure = new OutOfMemoryError("close");
    RecordingResource resource = new RecordingResource(closeFailure);
    IOException actual = assertThrows(IOException.class, () ->
        Arms.closeIfExceptionChecked(resource, r -> { throw failure; }));
    assertSame(failure, actual);
    assertArrayEquals(new Throwable[]{closeFailure}, actual.getSuppressed());
    assertEquals(1, resource.closeCount);
  }

  @ParameterizedTest
  @MethodSource("failures")
  void checkedCallbackAlsoCleansUpUncheckedFailures(Throwable failure) {
    RecordingResource resource = new RecordingResource(null);
    assertSame(failure, assertThrows(failure.getClass(), () ->
        Arms.closeIfExceptionChecked(resource, r -> {
          failUnchecked(failure);
          return null;
        })));
    assertEquals(1, resource.closeCount);
  }

  @ParameterizedTest
  @MethodSource("failures")
  void closeAllContinuesAfterUncheckedFailure(Throwable failure) {
    IOException secondFailure = new IOException("second close");
    RecordingResource first = new RecordingResource(failure);
    RecordingResource second = new RecordingResource(secondFailure);
    RecordingResource last = new RecordingResource(null);
    Throwable actual = assertThrows(failure.getClass(), () ->
        Arms.closeAll(first, null, second, last));
    assertSame(failure, actual);
    assertArrayEquals(new Throwable[]{secondFailure}, actual.getSuppressed());
    assertEquals(1, first.closeCount);
    assertEquals(1, second.closeCount);
    assertEquals(1, last.closeCount);
  }

  @Test
  void closeAllWrapsOnlyCheckedFailure() {
    IOException failure = new IOException("close");
    Error secondFailure = new AssertionError("second close");
    RecordingResource first = new RecordingResource(failure);
    RecordingResource second = new RecordingResource(secondFailure);
    RuntimeException actual = assertThrows(RuntimeException.class, () ->
        Arms.closeAll(Arrays.asList(first, second).iterator()));
    assertSame(failure, actual.getCause());
    assertArrayEquals(new Throwable[]{secondFailure}, failure.getSuppressed());
    assertEquals(1, first.closeCount);
    assertEquals(1, second.closeCount);
  }

  @Test
  void withResourceClosesResourcesAddedByCallback() {
    RecordingResource resource = new RecordingResource(null);
    Object expected = new Object();
    Object actual = Arms.withResource(new ArrayList<RecordingResource>(), resources -> {
      resources.add(resource);
      return expected;
    });
    assertSame(expected, actual);
    assertEquals(1, resource.closeCount);
  }

  @ParameterizedTest
  @MethodSource("failures")
  void withResourcePreservesBodyFailure(Throwable failure) {
    Error firstCloseFailure = new AssertionError("first close");
    IOException secondCloseFailure = new IOException("second close");
    RecordingResource first = new RecordingResource(firstCloseFailure);
    RecordingResource second = new RecordingResource(secondCloseFailure);
    RecordingResource last = new RecordingResource(null);
    Throwable actual = assertThrows(failure.getClass(), () ->
        Arms.withResource(Arrays.asList(first, null, second, last), resources -> {
          failUnchecked(failure);
          return null;
        }));
    assertSame(failure, actual);
    assertArrayEquals(new Throwable[]{firstCloseFailure, secondCloseFailure},
        actual.getSuppressed());
    assertEquals(1, first.closeCount);
    assertEquals(1, second.closeCount);
    assertEquals(1, last.closeCount);
  }

  @Test
  void withResourcePreservesBodyFailureWhenIteratorCreationFails() {
    IllegalStateException failure = new IllegalStateException("body");
    OutOfMemoryError iteratorFailure = new OutOfMemoryError("iterator");
    ArrayList<RecordingResource> resources = new ArrayList<RecordingResource>() {
      @Override
      public Iterator<RecordingResource> iterator() {
        throw iteratorFailure;
      }
    };

    IllegalStateException actual = assertThrows(IllegalStateException.class, () ->
        Arms.withResource(resources, r -> { throw failure; }));
    assertSame(failure, actual);
    assertArrayEquals(new Throwable[]{iteratorFailure}, actual.getSuppressed());
  }

  @Test
  void withResourceReportsCleanupFailureAfterSuccess() {
    Error failure = new AssertionError("close");
    RecordingResource first = new RecordingResource(failure);
    RecordingResource last = new RecordingResource(null);
    assertSame(failure, assertThrows(AssertionError.class, () ->
        Arms.withResource(Arrays.asList(first, last), resources -> "success")));
    assertEquals(1, first.closeCount);
    assertEquals(1, last.closeCount);
  }

  @Test
  void repeatedFailureDoesNotPreventCleanup() {
    Error failure = new AssertionError("shared");
    RecordingResource first = new RecordingResource(failure);
    RecordingResource second = new RecordingResource(failure);
    RecordingResource last = new RecordingResource(null);
    assertSame(failure, assertThrows(AssertionError.class, () ->
        Arms.closeAll(first, second, last)));
    assertEquals(0, failure.getSuppressed().length);
    assertEquals(1, first.closeCount);
    assertEquals(1, second.closeCount);
    assertEquals(1, last.closeCount);
  }

  @Test
  void bodyAndCloseCanThrowSameFailure() {
    Error failure = new AssertionError("shared");
    RecordingResource resource = new RecordingResource(failure);
    assertSame(failure, assertThrows(AssertionError.class, () ->
        Arms.closeIfException(resource, r -> { throw failure; })));
    assertEquals(0, failure.getSuppressed().length);
    assertEquals(1, resource.closeCount);
  }

  @Test
  void nullAndEmptyResources() {
    assertEquals("success", Arms.closeIfException(null, r -> "success"));
    Error failure = new AssertionError("body");
    assertSame(failure, assertThrows(AssertionError.class, () ->
        Arms.closeIfException(null, r -> { throw failure; })));
    Arms.closeAll(Collections.<AutoCloseable>emptyList());
    Arms.closeAll((AutoCloseable) null);
    assertEquals("success", Arms.withResource(Collections.<AutoCloseable>emptyList(),
        resources -> "success"));
  }
}
