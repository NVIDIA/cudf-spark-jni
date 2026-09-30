/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

import ai.rapids.cudf.NativeDepsLoader;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies the fatal native error handling path for SparkResourceAdaptor deallocation.
 *
 * The native test hook intentionally reaches a path that calls std::terminate(), so this test
 * cannot invoke it in the JUnit JVM. Instead, the test launches a child JVM running the nested
 * Child class' main method. Only the child loads the native library, invokes the hook, and is
 * expected to terminate after logging the deallocation failure. The parent JUnit test asserts that
 * the child exited with a nonzero status and that the expected failure log was emitted.
 */
public class RmmSparkDeallocationFailureTest {
  private static final String[] CHILD_ARGS = {"--trigger-deallocation-failure"};
  private static final String EXPECTED_LOG =
      "deallocate failed; terminating: injected deallocate failure";

  @Test
  public void testDeallocateFailureLogsAndTerminates() throws Exception {
    Process process = new ProcessBuilder(childCommand()).redirectErrorStream(true).start();
    boolean exited = process.waitFor(30, TimeUnit.SECONDS);
    if (!exited) {
      process.destroyForcibly();
      process.waitFor();
    }

    String output = readFully(process.getInputStream());
    assertTrue(exited, "child JVM did not exit; output:\n" + output);
    assertNotEquals(0, process.exitValue(), "child JVM unexpectedly succeeded; output:\n" + output);
    assertTrue(output.contains(EXPECTED_LOG), "expected log line missing; output:\n" + output);
  }

  private static List<String> childCommand() {
    List<String> command = new ArrayList<>();
    command.add(javaExecutable());

    String libraryPath = System.getProperty("java.library.path");
    if (libraryPath != null && !libraryPath.isEmpty()) {
      command.add("-Djava.library.path=" + libraryPath);
    }

    command.add("-cp");
    command.add(testClassPath());
    command.add(Child.class.getName());
    command.addAll(Arrays.asList(CHILD_ARGS));
    return command;
  }

  private static String javaExecutable() {
    return Paths.get(System.getProperty("java.home"), "bin", "java").toString();
  }

  private static String testClassPath() {
    String classPath = System.getProperty("surefire.test.class.path");
    if (classPath != null && !classPath.isEmpty()) {
      return classPath;
    }
    return System.getProperty("java.class.path");
  }

  private static String readFully(InputStream stream) throws IOException {
    ByteArrayOutputStream output = new ByteArrayOutputStream();
    byte[] buffer = new byte[8192];
    int amountRead;
    while ((amountRead = stream.read(buffer)) != -1) {
      output.write(buffer, 0, amountRead);
    }
    return output.toString("UTF-8");
  }

  public static class Child {
    static {
      NativeDepsLoader.loadNativeDeps();
    }

    public static void main(String[] args) {
      if (!Arrays.equals(CHILD_ARGS, args)) {
        throw new IllegalArgumentException("unexpected child arguments");
      }
      triggerDeallocationFailureForTesting();
    }

    private static native void triggerDeallocationFailureForTesting();
  }
}
