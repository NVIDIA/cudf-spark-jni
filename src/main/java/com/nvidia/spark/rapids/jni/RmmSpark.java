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

import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmDeviceMemoryResource;
import ai.rapids.cudf.RmmEventHandler;
import ai.rapids.cudf.RmmEventHandlerResourceAdaptor;
import ai.rapids.cudf.RmmException;
import ai.rapids.cudf.RmmTrackingResourceAdaptor;

/**
 * Initialize RMM in ways that are specific to Spark.
 * 
 * Because of the close ties of this class with Rmm.class, we are going to use
 * locks provided by Rmm class here, instead of our own locking.
 * 
 * Rmm.writeLock:
 * - Only used when setting or clearing the event handler. 
 * - This makes sure we have Rmm.class exclusively, and also protected `sra`
 *   in this class.
 * 
 * Rmm.readLock:
 * - Used for all other apis.
 */
public class RmmSpark {

  /**
   * Enumeration of OOM injection types for testing purposes.
   */
  public enum OomInjectionType {
    CPU_OR_GPU,
    CPU,
    GPU;
  }

  private static volatile SparkResourceAdaptor sra = null;

  /**
   * Set the event handler in a way that Spark wants it. For now this is the same as RMM, but in
   * the future it is likely to change.
   */
  public static void setEventHandler(RmmEventHandler handler) throws RmmException {
    setEventHandler(handler, null, false);
  }

  /**
   * Set the event handler in a way that Spark wants it. For now this is the same as RMM, but in
   * the future it is likely to change.
   * @param handler the handler to set
   * @param logLocation the location where you want spark state transitions. Alloc and free logging
   *                    is handled separately when setting up RMM. "stderr" or "stdout" are treated
   *                    as `std::cerr` and `std::cout` respectively in native code. Anything else
   *                    is treated as a file.
   */
  public static void setEventHandler(RmmEventHandler handler, String logLocation) throws RmmException {
    setEventHandler(handler, logLocation, false);
  }

  /**
   * Set the event handler in a way that Spark wants it. For now this is the same as RMM, but in
   * the future it is likely to change.
   * @param handler the handler to set
   * @param logLocation the location where you want spark state transitions. Alloc and free logging
   *                    is handled separately when setting up RMM. "stderr" or "stdout" are treated
   *                    as `std::cerr` and `std::cout` respectively in native code. Anything else
   *                    is treated as a file.
   * @param enableDebug if true, enable debug callbacks (onAllocated, onDeallocated) in the handler.
   *                    Note: enabling debug mode may impact performance.
   */
  public static void setEventHandler(RmmEventHandler handler, String logLocation,
      boolean enableDebug) throws RmmException {
    // synchronize with RMM not RmmSpark to stay in sync with Rmm itself.
    Rmm.writeLock.lock();
    try {
      // RmmException constructor is not public, so we have to use a different exception
      if (!Rmm.isInitialized()) {
        throw new RuntimeException("RMM has not been initialized");
      }
      RmmDeviceMemoryResource deviceResource = Rmm.getCurrentDeviceResource();
      if (deviceResource instanceof RmmEventHandlerResourceAdaptor ||
          deviceResource instanceof SparkResourceAdaptor) {
        throw new RuntimeException("Another event handler is already set");
      }
      RmmTrackingResourceAdaptor<RmmDeviceMemoryResource> tracker = Rmm.getTracker();
      if (tracker == null) {
        // This is just to be safe it should always be true if this is initialized.
        throw new RuntimeException("A tracker must be set for the event handler to work");
      }
      RmmEventHandlerResourceAdaptor<RmmDeviceMemoryResource> eventHandler =
          new RmmEventHandlerResourceAdaptor<>(deviceResource, tracker, handler, enableDebug);
      SparkResourceAdaptor localSra = null;
      try {
        SparkResourceAdaptor.initializeLogger(logLocation);
        localSra = new SparkResourceAdaptor(eventHandler);
        Rmm.setCurrentDeviceResource(localSra, deviceResource, false);
        sra = localSra; // Only assign to static field after successful setup
      } catch (Throwable t) {
        // Clean up resources if anything fails before successful setup
        if (localSra != null) {
          try {
            localSra.releaseWrapped();
          } catch (Throwable t2) {
            t.addSuppressed(t2);
          }
        }
        try {
          eventHandler.releaseWrapped();
        } catch (Throwable t2) {
          t.addSuppressed(t2);
        }
        throw t;
      }
    } finally {
      Rmm.writeLock.unlock();
    }
  }

  /**
   * Clears the active RMM event handler and anything else that is needed for spark to behave
   * properly.
   */
  public static void clearEventHandler() throws RmmException {
    // synchronize with RMM not RmmSpark to stay in sync with Rmm itself.
    Rmm.writeLock.lock();
    try {
      RmmDeviceMemoryResource deviceResource = Rmm.getCurrentDeviceResource();
      if (deviceResource instanceof SparkResourceAdaptor) {
        SparkResourceAdaptor sra = (SparkResourceAdaptor) deviceResource;
        RmmEventHandlerResourceAdaptor<RmmDeviceMemoryResource> event = sra.getWrapped();
        boolean success = false;
        try {
          Rmm.setCurrentDeviceResource(event.getWrapped(), sra, false);
          success = true;
        } finally {
          if (success) {
            RmmSpark.sra = null;
            sra.releaseWrapped();
            event.releaseWrapped();
          }
          SparkResourceAdaptor.shutdownLogger();
        }
      }
    } finally {
      Rmm.writeLock.unlock();
    }
  }

  /**
   * Get the id of the current thread as used by RmmSpark.
   */
  public static long getCurrentThreadId() {
    return SparkResourceAdaptor.getCurrentThreadId();
  }

  /**
   * Indicate that a given thread is dedicated to a specific task. This thread can be part of a
   * thread pool, but if it blocks it can never transitively block another active task.
   * @param threadId the thread ID to use
   * @param taskId the task ID this thread is working on.
   */
  public static void startDedicatedTaskThread(long threadId, long taskId, Thread thread) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        ThreadStateRegistry.addThread(threadId, thread);
        sra.startDedicatedTaskThread(threadId, taskId);
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * Indicate that the current thread is dedicated to a specific task. This thread can be part of
   * a thread pool, but if this blocks it can never transitively block another active task.
   * @param taskId the task ID this thread is working on.
   */
  public static void currentThreadIsDedicatedToTask(long taskId) {
    startDedicatedTaskThread(getCurrentThreadId(), taskId, Thread.currentThread());
  }

  /**
   * A shuffle thread has started to work on some tasks.
   * @param threadId the thread ID (not java thread id).
   * @param thread the java thread
   * @param taskIds the IDs of tasks that this is starting work on.
   */
  public static void shuffleThreadWorkingTasks(long threadId, Thread thread, long[] taskIds) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        ThreadStateRegistry.addThread(threadId, thread);
        sra.poolThreadWorkingOnTasks(true, threadId, taskIds);
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * The current thread is a shuffle thread and has started to work on some tasks.
   * @param taskIds the IDs of the tasks that this is starting work on.
   */
  public static void shuffleThreadWorkingOnTasks(long[] taskIds) {
    shuffleThreadWorkingTasks(getCurrentThreadId(), Thread.currentThread(), taskIds);
  }

  public static boolean isThreadWorkingOnTaskAsPoolThread() {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        return sra.isThreadWorkingOnTaskAsPoolThread(getCurrentThreadId());
      }
    } finally {
      Rmm.readLock.unlock();
    }
    return false;
  }

  /**
   * The current thread which is in a thread pool that could transitively block other tasks has
   * started to work on a task.
   * @param taskId the ID of the task that this is starting work on.
   */
  public static void poolThreadWorkingOnTask(long taskId) {
    long threadId = getCurrentThreadId();
    Thread thread = Thread.currentThread();
    long[] taskIds = new long[]{taskId};
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        ThreadStateRegistry.addThread(threadId, thread);
        sra.poolThreadWorkingOnTasks(false, threadId, taskIds);
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * A thread in a thread pool that could transitively block other tasks has finished work
   * on some tasks.
   * @param threadId the thread ID (not java thread id).
   * @param taskIds the IDs of the tasks that are done.
   */
  public static void poolThreadFinishedForTasks(long threadId, long[] taskIds) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        sra.poolThreadFinishedForTasks(threadId, taskIds);
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * A shuffle thread has finished work on some tasks.
   * @param threadId the thread ID (not java thread id).
   * @param taskIds the IDs of the tasks that are done.
   */
  private static void shuffleThreadFinishedForTasks(long threadId, long[] taskIds) {
    poolThreadFinishedForTasks(threadId, taskIds);
  }

  /**
   * The current thread which is in a thread pool that could transitively block other tasks
   * has finished work on some tasks.
   * @param taskIds the IDs of the tasks that are done.
   */
  public static void poolThreadFinishedForTasks(long[] taskIds) {
    poolThreadFinishedForTasks(getCurrentThreadId(), taskIds);
  }

  /**
   * The current shuffle thread has finished work on some tasks.
   * @param taskIds the IDs of the tasks that are done.
   */
  public static void shuffleThreadFinishedForTasks(long[] taskIds) {
    shuffleThreadFinishedForTasks(getCurrentThreadId(), taskIds);
  }

  /**
   * The current thread which is in a thread pool that could transitively block other tasks
   * has finished work on a task.
   * @param taskId the ID of the task that is done.
   */
  public static void poolThreadFinishedForTask(long taskId) {
    poolThreadFinishedForTasks(getCurrentThreadId(), new long[]{taskId});
  }

  /**
   * Indicate that a retry block has started for a given thread.
   * @param threadId the id of the thread, not the java ID.
   */
  public static void startRetryBlock(long threadId) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        sra.startRetryBlock(threadId);
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * Indicate that the current thread is entering a retry block.
   */
  public static void currentThreadStartRetryBlock() {
    startRetryBlock(getCurrentThreadId());
  }

  /**
   * Indicate that a retry block has ended for a given thread.
   * @param threadId the id of the thread, not the java ID.
   */
  public static void endRetryBlock(long threadId) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        sra.endRetryBlock(threadId);
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * Indicate that the current thread is exiting a retry block.
   */
  public static void currentThreadEndRetryBlock() {
    endRetryBlock(getCurrentThreadId());
  }

  private static void checkAndBreakDeadlocks() {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        sra.checkAndBreakDeadlocks();
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * Remove the given thread ID from being associated with a given task
   * @param threadId the ID of the thread that is no longer a part of a task or shuffle
   *                 (not java thread id).
   */
  public static void removeDedicatedThreadAssociation(long threadId, long taskId) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        sra.removeThreadAssociation(threadId, taskId);
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * Remove the current thread from being associated with the given task.
   */
  public static void removeCurrentDedicatedThreadAssociation(long taskId) {
    removeDedicatedThreadAssociation(getCurrentThreadId(), taskId);
  }

  /**
   * Remove all task associations for a given thread. This is intended to be used as a part
   * of tests when a thread is shutting down, or for a pool thread when it is fully done.
   * Dedicated task thread typically are cleaned when the task itself completes.
   * @param threadId the id of the thread to clean up
   */
  public static void removeAllThreadAssociation(long threadId) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        sra.removeThreadAssociation(threadId, -1);
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * Remove all task associations for the current thread. This is intended to be used as a part
   * of tests when a thread is shutting down, or for a pool thread when it is fully done.
   * Dedicated task thread typically are cleaned when the task itself completes.
   */
  public static void removeAllCurrentThreadAssociation() {
    removeAllThreadAssociation(getCurrentThreadId());
  }

  /**
   * Indicate that a given task is done and if there are any threads still associated with it
   * then they should also be removed.
   * @param taskId the ID of the task that has completed.
   */
  public static void taskDone(long taskId) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        sra.taskDone(taskId);
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * A dedicated task thread is about to submit work to a pool that could transitively block it.
   * @param threadId the ID of the thread that is about to submit the work.
   */
  public static void submittingToPool(long threadId) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        sra.submittingToPool(threadId);
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * The current thread is about to submit work to a thread pool that might transitively block
   * this thread. This thread must be a dedicated task thread.
   */
  public static void submittingToPool() {
    submittingToPool(getCurrentThreadId());
  }

  /**
   * A dedicated task thread is about to wait on work done on a pool that could transitively
   * block it.
   * @param threadId the ID of the thread that is about to wait.
   */
  public static void waitingOnPool(long threadId) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        sra.waitingOnPool(threadId);
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * The current thread is about to wait on work done on a thread pool that might transitively block
   * this thread. This thread must be a dedicated task thread.
   */
  public static void waitingOnPool() {
    waitingOnPool(getCurrentThreadId());
  }

  /**
   * A dedicated task thread is done waiting on a pool, either for a result or after submitting
   * something to the pool.
   * @param threadId the ID of the thread that is done.
   */
  public static void doneWaitingOnPool(long threadId) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        sra.doneWaitingOnPool(threadId);
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * The current thread is done waiting on a pool either for a result or after submitting something
   * to the pool. This thread must be a dedicated task thread.
   */
  public static void doneWaitingOnPool() {
    doneWaitingOnPool(getCurrentThreadId());
  }

  /**
   * This should be called as a part of handling any GpuRetryOOM or GpuSplitAndRetryOOM exception.
   * The order should be something like.
   * <ol>
   *   <li>Catch Exception</li>
   *   <li>Mark any GPU input as spillable, (should have already had contig split called on it)</li>
   *   <li>call blockUntilReady</li>
   *   <li>split the input data if GpuSplitAndRetryOOM</li>
   *   <li>retry processing with the data</li>
   * </ol>
   * This should be a NOOP if the thread is not in a state where it would need to block. Note
   * that any call to alloc could also block in the same way as a precaution in case this is
   * not followed and a task attempts to retry som processing either because it is old code or
   * in error.
   */
  public static void blockThreadUntilReady() {
    SparkResourceAdaptor local;
    Rmm.readLock.lock();
    try {
      local = sra;
    } finally {
      Rmm.readLock.unlock();
    }
    // Technically there is a race here, but because this can block we cannot hold the Rmm
    // lock while doing this, or we can deadlock. So we are going to rely on Rmm shutting down
    // or being reconfigured to be rare.
    if (local != null && local.isOpen()) {
      local.blockThreadUntilReady();
    }
  }

  /**
   * Force the thread with the given ID to throw a GpuRetryOOM or CpuRetryOOM on their next
   * allocation attempt, depending on the type of allocation being done.
   * @param threadId the ID of the thread to throw the exception (not java thread id).
   */
  public static void forceRetryOOM(long threadId) {
    forceRetryOOM(threadId, 1);
  }

  /**
   * Force the thread with the given ID to throw a GpuRetryOOM or CpuRetryOOM on their next
   * allocation attempt, depending on the type of allocation being done.
   * @param threadId the ID of the thread to throw the exception (not java thread id).
   * @param numOOMs the number of times the *RetryOOM should be thrown
   * @param oomMode the ordinal corresponding to OomInjectionType to filter allocations
   * @param skipCount how many matching allocations to skip
   */
  public static void forceRetryOOM(long threadId, int numOOMs, int oomMode, int skipCount) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        sra.forceRetryOOM(threadId, numOOMs, oomMode, skipCount);
      } else {
        throw new IllegalStateException("RMM has not been configured for OOM injection");
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  public static void forceRetryOOM(long threadId, int numOOMs) {
    forceRetryOOM(threadId, numOOMs, OomInjectionType.CPU_OR_GPU.ordinal(), 0);
  }

  /**
   * Force the thread with the given ID to throw a GpuSplitAndRetryOOM of CpuSplitAndRetryOOM
   * on their next allocation attempt, depending on the allocation being done.
   * @param threadId the ID of the thread to throw the exception (not java thread id).
   */
  public static void forceSplitAndRetryOOM(long threadId) {
    forceSplitAndRetryOOM(threadId, 1);
  }

  /**
   * Force the thread with the given ID to throw a GpuSplitAndRetryOOM or CpuSplitAndRetryOOm
   * on their next allocation attempt, depending on the allocation being done.
   * @param threadId the ID of the thread to throw the exception (not java thread id).
   * @param numOOMs the number of times the *SplitAndRetryOOM should be thrown
   * @param oomMode the ordinal corresponding to OomInjectionType to filter allocations
   * @param skipCount how many matching allocations to skip
   */
  public static void forceSplitAndRetryOOM(long threadId, int numOOMs, int oomMode, int skipCount) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        sra.forceSplitAndRetryOOM(threadId, numOOMs, oomMode, skipCount);
      } else {
        throw new IllegalStateException("RMM has not been configured for OOM injection");
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  public static void forceSplitAndRetryOOM(long threadId, int numOOMs) {
    forceSplitAndRetryOOM(threadId, numOOMs, OomInjectionType.CPU_OR_GPU.ordinal(), 0);
  }

  /**
   * Force the thread with the given ID to throw a CudfException on their next allocation attempt.
   * This is to simulate a cuDF exception being thrown from a kernel and test retry handling code.
   * @param threadId the ID of the thread to throw the exception (not java thread id).
   */
  public static void forceCudfException(long threadId) {
    forceCudfException(threadId, 1);
  }

  /**
   * Force the thread with the given ID to throw a CudfException on their next allocation attempt.
   * This is to simulate a cuDF exception being thrown from a kernel and test retry handling code.
   * @param threadId the ID of the thread to throw the exception (not java thread id).
   * @param numTimes the number of times the CudfException should be thrown
   */
  public static void forceCudfException(long threadId, int numTimes) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        sra.forceCudfException(threadId, numTimes);
      } else {
        throw new IllegalStateException("RMM has not been configured for OOM injection");
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  public static RmmSparkThreadState getStateOf(long threadId) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        return sra.getStateOf(threadId);
      } else {
        // sra is not set so the thread is by definition unknown to it.
        return RmmSparkThreadState.UNKNOWN;
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * Function used to remove task metrics.
   * 
   * Use this after a task is complete, and metrics have been reported to
   * clean up any metric related state
   * 
   * @param taskId the id of the task to remove metrics for.
   */
  public static void removeTaskMetrics(long taskId) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
         sra.removeTaskMetrics(taskId);
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * Get the number of retry exceptions that were thrown and reset the metric.
   * @param taskId the id of the task to get the metric for.
   * @return the number of times it was thrown or 0 if in the UNKNOWN state.
   */
  public static int getAndResetNumRetryThrow(long taskId) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        return sra.getAndResetNumRetryThrow(taskId);
      } else {
        // sra is not set so the value is by definition 0
        return 0;
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * Get the number of split and retry exceptions that were thrown and reset the metric.
   * @param taskId the id of the task to get the metric for.
   * @return the number of times it was thrown or 0 if in the UNKNOWN state.
   */
  public static int getAndResetNumSplitRetryThrow(long taskId) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        return sra.getAndResetNumSplitRetryThrow(taskId);
      } else {
        // sra is not set so the value is by definition 0
        return 0;
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * Get how long, in nanoseconds, that the task was blocked for
   * @param taskId the id of the task to get the metric for.
   * @return the time the task was blocked or 0 if in the UNKNOWN state.
   */
  public static long getAndResetBlockTimeNs(long taskId) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        return sra.getAndResetBlockTime(taskId);
      } else {
        // sra is not set so the value is by definition 0
        return 0;
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * Get how long, in nanoseconds, that this task lost in computation time due to retries.
   * @param taskId the id of the task to get the metric for.
   * @return the time the task did computation that was lost.
   */
  public static long getAndResetComputeTimeLostToRetryNs(long taskId) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        return sra.getAndResetComputeTimeLostToRetry(taskId);
      } else {
        // sra is not set so the value is by definition 0
        return 0;
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  public static long getTotalBlockedOrLostTime(long taskId) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        return sra.getTotalBlockedOrLostTime(taskId);
      } else {
        // sra is not set so the value is by definition 0
        return 0;
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * Get the max device memory footprint, in bytes, that this task had allocated over its lifetime
   * @param taskId the id of the task to get the metric for.
   * @return the max device memory footprint.
   */
  public static long getAndResetGpuMaxMemoryAllocated(long taskId) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        return sra.getAndResetGpuMaxMemoryAllocated(taskId);
      } else {
        // sra is not set so the value is by definition 0
        return 0;
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  public static long getMaxGpuTaskMemory(long taskId) {
    Rmm.readLock.lock();
    try {
      if (sra != null && sra.isOpen()) {
        return sra.getMaxGpuTaskMemory(taskId);
      } else {
        // sra is not set so the value is by definition 0
        return 0;
      }
    } finally {
      Rmm.readLock.unlock();
    }
  }

  /**
   * Called before doing an allocation on the CPU. This could throw an injected exception to help
   * with testing.
   * @param amount the amount of memory being requested
   * @param blocking is this for a blocking allocate or a non-blocking one.
   * @return a boolean that indicates if the allocation is recursive. Note that recursive
   * allocations on the CPU are only allowed with non-blocking allocations. This must be passed
   * back into the post allocations calls.
   */
  public static boolean preCpuAlloc(long amount, boolean blocking) {
    SparkResourceAdaptor local;
    Rmm.readLock.lock();
    try {
      local = sra;
    } finally {
      Rmm.readLock.unlock();
    }
    if (local != null && local.isOpen()) {
      return local.preCpuAlloc(amount, blocking);
    } else {
      return false;
    }
  }

  /**
   * The allocation that was going to be done succeeded.
   * @param ptr a pointer to the memory that was allocated.
   * @param amount the amount of memory that was allocated.
   * @param blocking is this for a blocking allocate or a non-blocking one.
   * @param wasRecursive the boolean that was returned from `preCpuAlloc`.
   */
  public static void postCpuAllocSuccess(long ptr, long amount, boolean blocking,
                                         boolean wasRecursive) {
    SparkResourceAdaptor local;
    Rmm.readLock.lock();
    try {
      local = sra;
    } finally {
      Rmm.readLock.unlock();
    }
    if (local != null && local.isOpen()) {
      local.postCpuAllocSuccess(ptr, amount, blocking, wasRecursive);
    }
  }

  /**
   * The allocation failed, and spilling didn't save it.
   * @param wasOom was the failure caused by an OOM or something else.
   * @param blocking is this for a blocking allocate or a non-blocking one.
   * @param wasRecursive the boolean that was returned from `preCpuAlloc`.
   * @return true if the allocation should be retried else false if the state machine
   * thinks that a retry would not help.
   */
  public static boolean postCpuAllocFailed(boolean wasOom, boolean blocking, boolean wasRecursive) {
    SparkResourceAdaptor local;
    Rmm.readLock.lock();
    try {
      local = sra;
    } finally {
      Rmm.readLock.unlock();
    }
    if (local != null && local.isOpen()) {
      return local.postCpuAllocFailed(wasOom, blocking, wasRecursive);
    } else {
      return false;
    }
  }

  /**
   * Some CPU memory was freed.
   * @param ptr a pointer to the memory being deallocated.
   * @param amount the amount that was made available.
   */
  public static void cpuDeallocate(long ptr, long amount) {
    SparkResourceAdaptor local;
    Rmm.readLock.lock();
    try {
      local = sra;
    } finally {
      Rmm.readLock.unlock();
    }
    if (local != null && local.isOpen()) {
      local.cpuDeallocate(ptr, amount);
    }
  }

  public static void spillRangeStart() {
    SparkResourceAdaptor local;
    Rmm.readLock.lock();
    try {
      local = sra;
    } finally {
      Rmm.readLock.unlock();
    }
    if (local != null && local.isOpen()) {
      local.spillRangeStart();
    }
  }

  public static void spillRangeDone() {
    SparkResourceAdaptor local;
    Rmm.readLock.lock();
    try {
      local = sra;
    } finally {
      Rmm.readLock.unlock();
    }
    if (local != null && local.isOpen()) {
      local.spillRangeDone();
    }
  }

}
