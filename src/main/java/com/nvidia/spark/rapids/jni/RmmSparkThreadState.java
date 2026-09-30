/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

/**
 * The state of a given thread according to RmmSpark. This is intended really for debugging and
 * testing only.
 */
public enum RmmSparkThreadState {
  UNKNOWN(-1), // thread is not associated with anything...
  THREAD_RUNNING(0), // task thread running normally
  THREAD_ALLOC(1), // task thread in the middle of doing an allocation
  THREAD_ALLOC_FREE(2), // task thread in the middle of doing an allocation and a free happened
  THREAD_BLOCKED(3), // task thread that is temporarily blocked
  THREAD_BUFN_THROW(4), // task thread that should throw an exception to roll back before blocking
  THREAD_BUFN_WAIT(5), // task thread that threw an exception to roll back and now should
  // block the next time alloc is called
  THREAD_BUFN(6), // task thread that is blocked until higher priority tasks start to succeed
  THREAD_SPLIT_THROW(7), // task thread that should throw an exception to split input and retry
  THREAD_REMOVE_THROW(8); // task thread that is being removed and needs to throw an exception

  private final int nativeId;

  RmmSparkThreadState(int nativeId) {
    this.nativeId = nativeId;
  }

  static RmmSparkThreadState fromNativeId(int nativeId) {
    for (RmmSparkThreadState ts : RmmSparkThreadState.values()) {
      if (ts.nativeId == nativeId) {
        return ts;
      }
    }
    throw new IllegalArgumentException("Could not find an ID for " + nativeId);
  }
}
