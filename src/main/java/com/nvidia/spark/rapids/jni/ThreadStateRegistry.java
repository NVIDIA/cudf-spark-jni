/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ConcurrentHashMap;
import java.util.ArrayList;

/**
 * This is used to allow us to map a native thread id to a java thread so we can look at the
 * state from a java perspective.
 */
public class ThreadStateRegistry {
  private static final Logger LOG = LoggerFactory.getLogger(ThreadStateRegistry.class);

  private static final ConcurrentHashMap<Long, Thread> knownThreads = new ConcurrentHashMap<>();
  
  private static Boolean printStackTraceCausingThreadBlocked = false;

  public static void enablePrintStackTraceCausingThreadBlocked() {
    printStackTraceCausingThreadBlocked = true;
  }

  public static void disablePrintStackTraceCausingThreadBlocked() {
    printStackTraceCausingThreadBlocked = false;
  }

  public static void addThread(long nativeId, Thread t) {
    knownThreads.put(nativeId, t);
  }

  // Typically called from JNI
  public static void removeThread(long threadId) {
    knownThreads.remove(threadId);
  }

  public static long[] blockedThreadIds() {
    ArrayList<Long> blockedThreadIds = new ArrayList<>();
    knownThreads.forEach((nativeId, thread) -> {
      if (isThreadBlocked(nativeId)) {
        blockedThreadIds.add(nativeId);
      }
    });
    return blockedThreadIds.stream().mapToLong(Long::longValue).toArray();
  }

  private static boolean isThreadBlocked(long nativeId) {
    Thread t = knownThreads.get(nativeId);
    if (t == null || !t.isAlive()) {
      // Dead is as good as blocked. This is mostly for tests, not so much for
      // production
      if (printStackTraceCausingThreadBlocked) {
        LOG.info("Thread with native ID {} is null or not alive, printing stack trace:", nativeId);
        if (t != null) {
          LOG.info("Thread {} stack trace:", t.getName());
          for (StackTraceElement element : t.getStackTrace()) {
            LOG.info("  at {}", element);
          }
        }
      }
      return true;
    }
    Thread.State state = t.getState();
    switch (state) {
      case BLOCKED:
        // fall through
      case WAITING:
        // fall through
      case TIMED_WAITING:
        if (printStackTraceCausingThreadBlocked) {
          LOG.info("Thread {} (native ID: {}) is blocked in state {}, printing stack trace:", 
                   t.getName(), nativeId, state);
          for (StackTraceElement element : t.getStackTrace()) {
            LOG.info("  at {}", element);
          }
        }
        return true;
      case TERMINATED:
        // Technically there is a race with `!t.isAlive` check above, and dead is as good as
        // blocked.
        if (printStackTraceCausingThreadBlocked) {
          LOG.info("Thread {} (native ID: {}) is terminated, printing stack trace:", 
                   t.getName(), nativeId);
          for (StackTraceElement element : t.getStackTrace()) {
            LOG.info("  at {}", element);
          }
        }
        return true;
      default:
        return false;
    }
  }
}
