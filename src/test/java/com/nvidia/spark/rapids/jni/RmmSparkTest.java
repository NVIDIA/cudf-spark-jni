/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.MemoryBuffer;
import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmAllocationMode;
import ai.rapids.cudf.RmmCudaMemoryResource;
import ai.rapids.cudf.RmmDeviceMemoryResource;
import ai.rapids.cudf.RmmEventHandler;
import ai.rapids.cudf.RmmLimitingResourceAdaptor;
import ai.rapids.cudf.RmmTrackingResourceAdaptor;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.fail;

public class RmmSparkTest {
  private final static long ALIGNMENT = 256;
  private static final AtomicLong TID = new AtomicLong(1);

  private long getNextTid() {
    return TID.getAndIncrement();
  }

  @BeforeEach
  public void setup() {
    if (Rmm.isInitialized()) {
      Rmm.shutdown();
    }
  }

  @AfterEach
  public void teardown() {
    if (Rmm.isInitialized()) {
      Rmm.shutdown();
    }
  }

  public interface TaskThreadOp<T> {
    T doIt();
  }

  public static class TaskThread extends Thread {
    private final String name;
    private final boolean isForPool;
    private long threadId = -1;
    private long taskId = -1;

    public TaskThread(String name, long taskId) {
      this(name, false);
      this.taskId = taskId;
    }

    public TaskThread(String name, boolean isForPool) {
      super(name);
      this.name = name;
      this.isForPool = isForPool;
    }

    public synchronized long getThreadId() {
      return threadId;
    }

    public long getTaskId() {
      return taskId;
    }

    public boolean isThreadForPool() {
      return isForPool;
    }

    private LinkedBlockingQueue<TaskThreadOp> queue = new LinkedBlockingQueue<>();

    public void initialize() throws ExecutionException, InterruptedException, TimeoutException {
      setDaemon(true);
      start();
      Future<Void> waitForStart = doIt(new TaskThreadOp<Void>() {
        @Override
        public Void doIt() {
          if (!isForPool) {
            RmmSpark.currentThreadIsDedicatedToTask(taskId);
            // Make sure the order they are initialized corresponds to the priority
            TaskPriority.getTaskPriority(taskId);
          }
          return null;
        }

        @Override
        public String toString() {
          return "INIT TASK " + name + " " + (isForPool ? "POOL" : ("TASK " + taskId));
        }
      });
      System.err.println("WAITING FOR STARTUP (" + name + ")");
      waitForStart.get(1000, TimeUnit.MILLISECONDS);
      System.err.println("THREAD IS READY TO GO (" + name + ")");
    }

    public void pollForState(RmmSparkThreadState state, long l, TimeUnit tu) throws TimeoutException, InterruptedException {
      long start = System.nanoTime();
      long timeoutAfter = start + tu.toNanos(l);
      RmmSparkThreadState currentState = null;
      while (System.nanoTime() <= timeoutAfter) {
        currentState = RmmSpark.getStateOf(threadId);
        if (currentState == state) {
          return;
        }
        // Yes we are essentially doing a busy wait...
        Thread.sleep(10);
      }
      throw new TimeoutException(name + " WAITING FOR STATE " + state + " BUT STATE IS " + currentState);
    }

    private static class TaskThreadDoneOp implements TaskThreadOp<Void>, Future<Object> {
      private TaskThread wrapped;

      TaskThreadDoneOp(TaskThread td) {
        wrapped = td;
      }

      @Override
      public String toString() {
        return "TASK DONE";
      }

      @Override
      public Void doIt() {
        if (!wrapped.isThreadForPool()) {
          long tid = wrapped.getTaskId();
          RmmSpark.removeAllCurrentThreadAssociation();
          RmmSpark.taskDone(tid);
          TaskPriority.taskDone(tid);
        }
        return null;
      }

      @Override
      public boolean cancel(boolean b) {
        return false;
      }

      @Override
      public boolean isCancelled() {
        return false;
      }

      @Override
      public boolean isDone() {
        return !wrapped.isAlive();
      }

      @Override
      public Object get() throws InterruptedException, ExecutionException {
        throw new RuntimeException("FUTURE NEEDS A TIMEOUT. THIS IS A TEST!");
      }

      @Override
      public Object get(long l, TimeUnit timeUnit) throws InterruptedException, ExecutionException, TimeoutException {
        System.err.println("WAITING FOR THREAD DONE " + l + " " + timeUnit);
        wrapped.join(timeUnit.toMillis(l));
        return null;
      }
    }

    public Future done() {
      TaskThreadDoneOp op = new TaskThreadDoneOp(this);
      queue.offer(op);
      return op;
    }

    private static class TaskThreadTrackingOp<T> implements TaskThreadOp<T>, Future<T> {
      private final TaskThreadOp<T> wrapped;
      private boolean done = false;
      private Throwable t = null;
      private T ret = null;


      @Override
      public String toString() {
        return wrapped.toString();
      }

      TaskThreadTrackingOp(TaskThreadOp<T> td) {
        wrapped = td;
      }

      @Override
      public T doIt() {
        try {
          T tmp = wrapped.doIt();
          synchronized (this) {
            ret = tmp;
            return ret;
          }
        } catch (Throwable t) {
          synchronized (this) {
            this.t = t;
          }
          return null;
        } finally {
          synchronized (this) {
            done = true;
            this.notifyAll();
          }
        }
      }

      @Override
      public boolean cancel(boolean b) {
        return false;
      }

      @Override
      public boolean isCancelled() {
        return false;
      }

      @Override
      public synchronized boolean isDone() {
        return done;
      }

      @Override
      public synchronized T get() throws InterruptedException, ExecutionException {
        throw new RuntimeException("This is a test you should always have timeouts...");
      }

      @Override
      public synchronized T get(long l, TimeUnit timeUnit) throws InterruptedException, ExecutionException, TimeoutException {
        if (!done) {
          System.err.println("WAITING " + l + " " + timeUnit + " FOR '" + wrapped + "'");
          wait(timeUnit.toMillis(l));
          if (!done) {
            throw new TimeoutException();
          }
        }
        if (t != null) {
          throw new ExecutionException(t);
        }
        return ret;
      }
    }

    public <T> Future<T> doIt(TaskThreadOp<T> op) {
      if (!isAlive()) {
        throw new IllegalStateException("Thread is already done...");
      }
      TaskThreadTrackingOp<T> tracking = new TaskThreadTrackingOp<>(op);
      queue.offer(tracking);
      return tracking;
    }

    public Future<Void> blockUntilReady() {
      return doIt(new TaskThreadOp<Void>() {
        @Override
        public Void doIt() {
          RmmSpark.blockThreadUntilReady();
          return null;
        }

        @Override
        public String toString() {
          return "BLOCK UNTIL THREAD IS READY";
        }
      });
    }

    @Override
    public void run() {
      try {
        synchronized (this) {
          threadId = RmmSpark.getCurrentThreadId();
        }
        System.err.println("INSIDE THREAD RUNNING (" + name + ")");
        while (true) {
          // Because of how our deadlock detection code works we don't want to
          // block this thread, so we do this in a busy loop. It is not ideal,
          // but works, and is more accurate to what the Spark is likely to do
          TaskThreadOp op = queue.poll();
          // null is returned from the queue if it is empty
          if (op != null) {
            System.err.println("GOT '" + op + "' ON " + name);
            if (op instanceof TaskThreadDoneOp) {
              op.doIt();
              return;
            }
            op.doIt();
            System.err.println("'" + op + "' FINISHED ON " + name);
          }
        }
      } catch (Throwable t) {
        System.err.println("THROWABLE CAUGHT IN " + name);
        t.printStackTrace(System.err);
      } finally {
        System.err.println("THREAD EXITING " + name);
      }
    }
  }

  @Test
  public void testBasicInitAndTeardown() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 512 * 1024 * 1024);
    RmmSpark.setEventHandler(new BaseRmmEventHandler(), "stderr");
  }

  @Test
  public void testInsertOOMsGpu() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 512 * 1024 * 1024);
    RmmSpark.setEventHandler(new BaseRmmEventHandler(), "stderr");
    long threadId = RmmSpark.getCurrentThreadId();
    long taskid = getNextTid();
    Thread t = Thread.currentThread();
    assertEquals(RmmSparkThreadState.UNKNOWN, RmmSpark.getStateOf(threadId));
    assertEquals(0, RmmSpark.getAndResetNumRetryThrow(taskid));
    assertEquals(0, RmmSpark.getAndResetNumSplitRetryThrow(taskid));
    assertEquals(0, RmmSpark.getAndResetComputeTimeLostToRetryNs(taskid));
    assertEquals(0, RmmSpark.getAndResetGpuMaxMemoryAllocated(taskid));
    RmmSpark.startDedicatedTaskThread(threadId, taskid, t);
    assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
    try {
      RmmSpark.startRetryBlock(threadId);
      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));

      try {
        Thread.sleep(1); // Just in case we run on a really fast system in the future where
        // all of this is sub-nanosecond...
      } catch (InterruptedException e) {
        // Ignored
      }
      // Force an exception
      RmmSpark.forceRetryOOM(threadId);
      // No change in the state after a force
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
      assertThrows(GpuRetryOOM.class, () -> Rmm.alloc(100).close());
      assert(RmmSpark.getAndResetComputeTimeLostToRetryNs(taskid) > 0);

      // Verify that injecting OOM does not cause the block to actually happen or
      // the state to change
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
      assertEquals(1, RmmSpark.getAndResetNumRetryThrow(taskid));
      assertEquals(0, RmmSpark.getAndResetNumSplitRetryThrow(taskid));
      assertEquals(100, RmmSpark.getAndResetGpuMaxMemoryAllocated(taskid));
      RmmSpark.blockThreadUntilReady();

      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));

      // Force another exception
      RmmSpark.forceSplitAndRetryOOM(threadId);
      // No change in state after force
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
      assertThrows(GpuSplitAndRetryOOM.class, () -> Rmm.alloc(100).close());
      assertEquals(0, RmmSpark.getAndResetNumRetryThrow(taskid));
      assertEquals(1, RmmSpark.getAndResetNumSplitRetryThrow(taskid));
      assertEquals(100, RmmSpark.getAndResetGpuMaxMemoryAllocated(taskid));

      // Verify that injecting OOM does not cause the block to actually happen
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
      RmmSpark.blockThreadUntilReady();

      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
    } finally {
      RmmSpark.taskDone(taskid);
      TaskPriority.taskDone(taskid);
    }
    assertEquals(RmmSparkThreadState.UNKNOWN, RmmSpark.getStateOf(threadId));
  }

  @Test
  public void testInsertOOMsCpu() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 512 * 1024 * 1024);
    RmmSpark.setEventHandler(new BaseRmmEventHandler(), "stderr");
    LimitingOffHeapAllocForTests.setLimit(512 * 1024 * 1024);
    long threadId = RmmSpark.getCurrentThreadId();
    long taskid = getNextTid();
    Thread t = Thread.currentThread();
    assertEquals(RmmSparkThreadState.UNKNOWN, RmmSpark.getStateOf(threadId));
    assertEquals(0, RmmSpark.getAndResetNumRetryThrow(taskid));
    assertEquals(0, RmmSpark.getAndResetNumSplitRetryThrow(taskid));
    assertEquals(0, RmmSpark.getAndResetComputeTimeLostToRetryNs(taskid));
    RmmSpark.startDedicatedTaskThread(threadId, taskid, t);
    assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
    try {
      RmmSpark.startRetryBlock(threadId);
      // Allocate something small and verify that it works...
      LimitingOffHeapAllocForTests.alloc(100).close();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));

      try {
        Thread.sleep(1); // Just in case we run on a really fast system in the future where
        // all of this is sub-nanosecond...
      } catch (InterruptedException e) {
        // Ignored
      }
      // Force an exception
      RmmSpark.forceRetryOOM(threadId);
      // No change in the state after a force
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
      assertThrows(CpuRetryOOM.class, () -> LimitingOffHeapAllocForTests.alloc(100).close());
      assert(RmmSpark.getAndResetComputeTimeLostToRetryNs(taskid) > 0);

      // Verify that injecting OOM does not cause the block to actually happen or
      // the state to change
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
      assertEquals(1, RmmSpark.getAndResetNumRetryThrow(taskid));
      assertEquals(0, RmmSpark.getAndResetNumSplitRetryThrow(taskid));
      RmmSpark.blockThreadUntilReady();

      // Allocate something small and verify that it works...
      LimitingOffHeapAllocForTests.alloc(100).close();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));

      // Force another exception
      RmmSpark.forceSplitAndRetryOOM(threadId);
      // No change in state after force
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
      assertThrows(CpuSplitAndRetryOOM.class, () -> LimitingOffHeapAllocForTests.alloc(100).close());
      assertEquals(0, RmmSpark.getAndResetNumRetryThrow(taskid));
      assertEquals(1, RmmSpark.getAndResetNumSplitRetryThrow(taskid));

      // Verify that injecting OOM does not cause the block to actually happen
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
      RmmSpark.blockThreadUntilReady();

      // Allocate something small and verify that it works...
      LimitingOffHeapAllocForTests.alloc(100).close();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
    } finally {
      RmmSpark.taskDone(taskid);
      TaskPriority.taskDone(taskid);
    }
    assertEquals(RmmSparkThreadState.UNKNOWN, RmmSpark.getStateOf(threadId));
  }

  @Test
  public void testReentrantAssociateThread() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 512 * 1024 * 1024);
    RmmSpark.setEventHandler(new BaseRmmEventHandler(), "stderr");
    long threadId = 100;
    long taskId = getNextTid();
    long[] taskIds = new long[] {taskId};
    Thread t = Thread.currentThread();
    try {
      RmmSpark.startDedicatedTaskThread(threadId, taskId, t);
      RmmSpark.startDedicatedTaskThread(threadId, taskId, t);
      RmmSpark.removeDedicatedThreadAssociation(threadId, taskId);
      // Not removing twice because we don't have to match up the counts so it fits with how
      // the GPU semaphore is used.
      RmmSpark.shuffleThreadWorkingTasks(threadId, t, taskIds);
      RmmSpark.shuffleThreadWorkingTasks(threadId, t, taskIds);
      RmmSpark.removeDedicatedThreadAssociation(threadId, taskId);
      RmmSpark.removeDedicatedThreadAssociation(threadId, taskId);
    } finally {
      RmmSpark.taskDone(taskId);
      TaskPriority.taskDone(taskId);
    }
  }

  @Test
  public void testAssociateThread() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 512 * 1024 * 1024);
    RmmSpark.setEventHandler(new BaseRmmEventHandler(), "stderr");
    long threadIdOne = 200;
    long threadIdTwo = 300;
    long taskId = getNextTid();
    long otherTaskId = getNextTid();
    long[] taskIds = new long[] {taskId, otherTaskId};
    Thread t = Thread.currentThread();
    try {
      RmmSpark.startDedicatedTaskThread(threadIdOne, taskId, t);
      assertThrows(CudfException.class, () -> RmmSpark.shuffleThreadWorkingTasks(threadIdOne, t, taskIds));
      // There can be races when a thread goes from one task to another, so we just make it safe to do.
      RmmSpark.startDedicatedTaskThread(threadIdOne, otherTaskId, t);

      RmmSpark.shuffleThreadWorkingTasks(threadIdTwo, t, taskIds);
      assertThrows(CudfException.class, () -> RmmSpark.startDedicatedTaskThread(threadIdTwo, otherTaskId, t));
      // Remove the association
      RmmSpark.removeDedicatedThreadAssociation(threadIdTwo, taskId);
      RmmSpark.removeDedicatedThreadAssociation(threadIdTwo, otherTaskId);
      // Add in a new association
      RmmSpark.startDedicatedTaskThread(threadIdTwo, taskId, t);
    } finally {
      RmmSpark.taskDone(taskId);
      RmmSpark.taskDone(otherTaskId);
      TaskPriority.taskDone(taskId);
      TaskPriority.taskDone(otherTaskId);
    }
  }


  static abstract class AllocOnAnotherThread implements AutoCloseable {
    final TaskThread thread;
    final long size;
    final long taskId;
    MemoryBuffer b = null;
    Future<Void> fb;
    Future<Void> fc = null;

    public AllocOnAnotherThread(TaskThread thread, long size) {
      this.thread = thread;
      this.size = size;
      this.taskId = -1;
      fb = thread.doIt(new TaskThreadOp<Void>() {
        @Override
        public Void doIt() {
          doAlloc();
          return null;
        }

        @Override
        public String toString() {
          return "ALLOC(" + size + ")";
        }
      });
    }

    public AllocOnAnotherThread(TaskThread thread, long size, long taskId) {
      this.thread = thread;
      this.size = size;
      this.taskId = taskId;
      fb = thread.doIt(new TaskThreadOp<Void>() {
        @Override
        public Void doIt() {
          RmmSpark.shuffleThreadWorkingOnTasks(new long[]{taskId});
          doAlloc();
          return null;
        }

        @Override
        public String toString() {
          return "ALLOC(" + size + ")";
        }
      });
    }

    public void waitForAlloc() throws ExecutionException, InterruptedException, TimeoutException {
      fb.get(1000, TimeUnit.MILLISECONDS);
    }

    public void freeOnThread() {
      if (fc != null) {
        throw new IllegalStateException("free called multiple times");
      }

      fc = thread.doIt(new TaskThreadOp<Void>() {
        @Override
        public Void doIt() {
          close();
          return null;
        }

        @Override
        public String toString() {
          return "FREE(" + size + ")";
        }
      });
    }

    public void waitForFree() throws ExecutionException, InterruptedException, TimeoutException {
      if (fc == null) {
        freeOnThread();
      }
      fc.get(1000, TimeUnit.MILLISECONDS);
    }

    public void freeAndWait() throws ExecutionException, InterruptedException, TimeoutException {
      waitForFree();
    }

    abstract protected Void doAlloc();

    @Override
    public synchronized void close() {
      if (b != null) {
        try {
          b.close();
          b = null;
        } finally {
          if (this.taskId > 0) {
            RmmSpark.poolThreadFinishedForTasks(thread.threadId, new long[]{taskId});
          }
        }
      }
    }
  }

  public static class GpuAllocOnAnotherThread extends AllocOnAnotherThread {

    public GpuAllocOnAnotherThread(TaskThread thread, long size) {
      super(thread, size);
    }

    public GpuAllocOnAnotherThread(TaskThread thread, long size, long taskId) {
      super(thread, size, taskId);
    }

    @Override
    protected Void doAlloc() {
      DeviceMemoryBuffer tmp = Rmm.alloc(size);
      synchronized (this) {
        b = tmp;
      }
      return null;
    }
  }

  public static class CpuAllocOnAnotherThread extends AllocOnAnotherThread {

    public CpuAllocOnAnotherThread(TaskThread thread, long size) {
      super(thread, size);
    }

    public CpuAllocOnAnotherThread(TaskThread thread, long size, long taskId) {
      super(thread, size, taskId);
    }

    @Override
    protected Void doAlloc() {
      HostMemoryBuffer tmp = LimitingOffHeapAllocForTests.alloc(size);
      synchronized (this) {
        b = tmp;
      }
      return null;
    }
  }

  void setupRmmForTestingWithLimits(long maxAllocSize) {
    setupRmmForTestingWithLimits(maxAllocSize, new BaseRmmEventHandler());
  }

  void setupRmmForTestingWithLimits(long maxAllocSize, RmmEventHandler eventHandler) {
    // Rmm.initialize is not going to limit allocations without a pool, so we
    // need to set it up ourselves.
    RmmDeviceMemoryResource resource = null;
    boolean succeeded = false;
    try {
      resource = new RmmCudaMemoryResource();
      resource = new RmmLimitingResourceAdaptor<>(resource, maxAllocSize, ALIGNMENT);
      resource = new RmmTrackingResourceAdaptor<>(resource, ALIGNMENT);
      Rmm.setCurrentDeviceResource(resource, null, false);
      succeeded = true;
    } finally {
      if (!succeeded && resource != null) {
        resource.close();
      }
    }
    RmmSpark.setEventHandler(eventHandler, "stderr");
  }

  @Test
  public void testNonBlockingCpuAlloc() {
    // We are not going to use the GPU here, but this sets it all up for us.
    setupRmmForTestingWithLimits(10 * 1024 * 1024);
    // We are just going to pretend that we are doing an allocations
    long taskId = getNextTid();
    long threadId = RmmSpark.getCurrentThreadId();
    RmmSpark.currentThreadIsDedicatedToTask(taskId);
    try {
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
      boolean wasRecursive = RmmSpark.preCpuAlloc(100, false);
      assertEquals(RmmSparkThreadState.THREAD_ALLOC, RmmSpark.getStateOf(threadId));
      long address;
      try (HostMemoryBuffer buffer = HostMemoryBuffer.allocate(100)) {
        address = buffer.getAddress();
        RmmSpark.postCpuAllocSuccess(address, 100, false, wasRecursive);
        assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
      }
      RmmSpark.cpuDeallocate(address, 100);
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
    } finally {
      RmmSpark.removeDedicatedThreadAssociation(threadId, taskId);
      RmmSpark.taskDone(taskId);
      TaskPriority.taskDone(taskId);
    }
  }

  @Test
  public void testNonBlockingCpuAllocFailedOOM() {
    // We are not going to use the GPU here, but this sets it all up for us.
    setupRmmForTestingWithLimits(10 * 1024 * 1024);
    // We are just going to pretend that we are doing an allocations
    long taskId = getNextTid();
    long threadId = RmmSpark.getCurrentThreadId();
    RmmSpark.currentThreadIsDedicatedToTask(taskId);
    assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
    try {
      boolean wasRecursive = RmmSpark.preCpuAlloc(100, false);
      assertEquals(RmmSparkThreadState.THREAD_ALLOC, RmmSpark.getStateOf(threadId));
      // TODO put this on a background thread so we can time out if it blocks.
      RmmSpark.postCpuAllocFailed(true, false, wasRecursive);
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
    } finally {
      RmmSpark.removeDedicatedThreadAssociation(threadId, taskId);
      RmmSpark.taskDone(taskId);
      TaskPriority.taskDone(taskId);
    }
  }

  @Test
  public void testBasicBlocking() throws ExecutionException, InterruptedException, TimeoutException {
    // 10 MiB
    setupRmmForTestingWithLimits(10 * 1024 * 1024);
    TaskThread taskOne = new TaskThread("TEST THREAD ONE", getNextTid());
    TaskThread taskTwo = new TaskThread("TEST THREAD TWO", getNextTid());
    taskOne.initialize();
    taskTwo.initialize();
    try {
      long tOneId = taskOne.getThreadId();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(tOneId));

      long tTwoId = taskTwo.getThreadId();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(tTwoId));

      try (AllocOnAnotherThread firstOne = new GpuAllocOnAnotherThread(taskOne, 5 * 1024 * 1024)) {
        firstOne.waitForAlloc();
        // This one should block
        try (AllocOnAnotherThread secondOne = new GpuAllocOnAnotherThread(taskTwo, 6 * 1024 * 1024)) {
          taskTwo.pollForState(RmmSparkThreadState.THREAD_BLOCKED, 1000, TimeUnit.MILLISECONDS);
          // Free the first allocation to wake up the second task...
          firstOne.freeAndWait();
          secondOne.waitForAlloc();
          secondOne.freeAndWait();
        }
      }
    } finally {
      taskOne.done();
      taskTwo.done();
    }
  }

  @Test
  public void testBasicCpuBlocking() throws ExecutionException, InterruptedException, TimeoutException {
    // 10 MiB
    setupRmmForTestingWithLimits(10 * 1024 * 1024);
    LimitingOffHeapAllocForTests.setLimit(10 * 1024 * 1024);
    TaskThread taskOne = new TaskThread("TEST THREAD ONE", getNextTid());
    TaskThread taskTwo = new TaskThread("TEST THREAD TWO", getNextTid());
    taskOne.initialize();
    taskTwo.initialize();
    try {
      long tOneId = taskOne.getThreadId();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(tOneId));

      long tTwoId = taskTwo.getThreadId();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(tTwoId));

      try (AllocOnAnotherThread firstOne = new CpuAllocOnAnotherThread(taskOne, 5 * 1024 * 1024)) {
        firstOne.waitForAlloc();
        // This one should block
        try (AllocOnAnotherThread secondOne = new CpuAllocOnAnotherThread(taskTwo, 6 * 1024 * 1024)) {
          taskTwo.pollForState(RmmSparkThreadState.THREAD_BLOCKED, 1000, TimeUnit.MILLISECONDS);
          // Free the first allocation to wake up the second task...
          firstOne.freeAndWait();
          secondOne.waitForAlloc();
          secondOne.freeAndWait();
        }
      }

    } finally {
      taskOne.done();
      taskTwo.done();
    }
  }

  @Test
  public void testBasicMixedBlocking() throws ExecutionException, InterruptedException, TimeoutException {
    final long MB = 1024 * 1024;
    setupRmmForTestingWithLimits(10 * MB);
    LimitingOffHeapAllocForTests.setLimit(10 * MB);
    TaskThread taskOne = new TaskThread("TEST THREAD ONE", getNextTid());
    TaskThread taskTwo = new TaskThread("TEST THREAD TWO", getNextTid());
    TaskThread taskThree = new TaskThread("TEST THREAD THREE", getNextTid());
    TaskThread taskFour = new TaskThread("TEST THREAD FOUR", getNextTid());
    taskOne.initialize();
    taskTwo.initialize();
    taskThree.initialize();
    taskFour.initialize();

    final long FIVE_MB = 5 * MB;
    final long SIX_MB = 6 * MB;
    try {
      long tOneId = taskOne.getThreadId();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(tOneId));

      long tTwoId = taskTwo.getThreadId();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(tTwoId));

      long tThreeId = taskThree.getThreadId();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(tThreeId));

      long tFourId = taskFour.getThreadId();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(tFourId));

      try (AllocOnAnotherThread firstGpuAlloc = new GpuAllocOnAnotherThread(taskOne, FIVE_MB)) {
        firstGpuAlloc.waitForAlloc();

        try (AllocOnAnotherThread firstCpuAlloc = new CpuAllocOnAnotherThread(taskTwo, FIVE_MB)) {
          firstCpuAlloc.waitForAlloc();

          // Blocking GPU Alloc
          try (AllocOnAnotherThread secondGpuAlloc = new GpuAllocOnAnotherThread(taskThree, SIX_MB)) {
            taskThree.pollForState(RmmSparkThreadState.THREAD_BLOCKED, 1000, TimeUnit.MILLISECONDS);

            // Blocking CPU Alloc
            try (AllocOnAnotherThread secondCpuAlloc = new CpuAllocOnAnotherThread(taskFour, SIX_MB)) {
              taskFour.pollForState(RmmSparkThreadState.THREAD_BLOCKED, 1000, TimeUnit.MILLISECONDS);

              // We want to make sure that the order of wakeup corresponds to the location of the data that was released
              // Not necessarily the priority of the task/thread.
              firstCpuAlloc.freeAndWait();
              secondCpuAlloc.waitForAlloc();
              secondCpuAlloc.freeAndWait();
            }

            // Now do the GPU frees
            firstGpuAlloc.freeAndWait();
            secondGpuAlloc.waitForAlloc();
            secondGpuAlloc.freeAndWait();
          }
          // Do one more alloc after freeing on same task to show the max allocation metric is unimpacted
          try (AllocOnAnotherThread secondGpuAlloc = new GpuAllocOnAnotherThread(taskThree, FIVE_MB)) {
            secondGpuAlloc.waitForAlloc();
            secondGpuAlloc.freeAndWait();
          }
        }
      }
    } finally {
      taskOne.done();
      assertEquals(FIVE_MB, RmmSpark.getAndResetGpuMaxMemoryAllocated(taskOne.getTaskId()));
      taskTwo.done();
      assertEquals(0, RmmSpark.getAndResetGpuMaxMemoryAllocated(taskTwo.getTaskId()));
      taskThree.done();
      assertEquals(SIX_MB, RmmSpark.getAndResetGpuMaxMemoryAllocated(taskThree.getTaskId()));
      taskFour.done();
      assertEquals(0, RmmSpark.getAndResetGpuMaxMemoryAllocated(taskFour.getTaskId()));
    }
  }

  @Test
  public void testShuffleBlocking() throws ExecutionException, InterruptedException, TimeoutException {
    // 10 MiB
    setupRmmForTestingWithLimits(10 * 1024 * 1024);
    TaskThread shuffleOne = new TaskThread("TEST THREAD SHUFFLE", true);
    TaskThread taskOne = new TaskThread("TEST THREAD ONE", getNextTid());
    TaskThread taskTwo = new TaskThread("TEST THREAD TWO", getNextTid());

    shuffleOne.initialize();
    taskOne.initialize();
    taskTwo.initialize();
    try {
      long sOneId = shuffleOne.getThreadId();
      // It is not in a running state until it has something to do.

      long tOneId = taskOne.getThreadId();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(tOneId));

      long tTwoId = taskTwo.getThreadId();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(tTwoId));

      try (AllocOnAnotherThread firstOne = new GpuAllocOnAnotherThread(taskOne, 5 * 1024 * 1024)) {
        firstOne.waitForAlloc();
        // This one should block
        try (AllocOnAnotherThread secondOne = new GpuAllocOnAnotherThread(taskTwo, 6 * 1024 * 1024)) {
          taskTwo.pollForState(RmmSparkThreadState.THREAD_BLOCKED, 1000, TimeUnit.MILLISECONDS);
          // Make sure that shuffle has higher priority than tasks...
          try (AllocOnAnotherThread thirdOne = new GpuAllocOnAnotherThread(shuffleOne, 6 * 1024 * 1024, 2)) {
            shuffleOne.pollForState(RmmSparkThreadState.THREAD_BLOCKED, 1000, TimeUnit.MILLISECONDS);
            // But taskOne is not blocked, so there will be no retry until it is blocked, or else
            // it is making progress
            taskOne.doIt((TaskThreadOp<Void>) () -> {
              try {
                Thread.sleep(200);
              } catch (InterruptedException e) {
                throw new RuntimeException(e);
              }
              return null;
            });

            try {
              secondOne.waitForAlloc();
              fail("SHOULD HAVE THROWN...");
            } catch (ExecutionException ee) {
              assert (ee.getCause() instanceof GpuRetryOOM);
            }
            secondOne.freeAndWait();

            // Free the first allocation to wake up the shuffle thread, but not the second task yet...
            firstOne.freeAndWait();

            thirdOne.waitForAlloc();
            thirdOne.freeAndWait();
          }
        }
      }
    } finally {
      shuffleOne.done();
      taskOne.done();
      taskTwo.done();
    }
  }


  @Test
  public void testShuffleBlockingCpu() throws ExecutionException, InterruptedException, TimeoutException {
    // 10 MiB
    setupRmmForTestingWithLimits(10 * 1024 * 1024);
    LimitingOffHeapAllocForTests.setLimit(10 * 1024 * 1024);
    TaskThread shuffleOne = new TaskThread("TEST THREAD SHUFFLE", true);
    TaskThread taskOne = new TaskThread("TEST THREAD ONE", getNextTid());
    TaskThread taskTwo = new TaskThread("TEST THREAD TWO", getNextTid());

    shuffleOne.initialize();
    taskOne.initialize();
    taskTwo.initialize();
    try {
      long sOneId = shuffleOne.getThreadId();
      // It is not in a running state until it has something to do.

      long tOneId = taskOne.getThreadId();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(tOneId));

      long tTwoId = taskTwo.getThreadId();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(tTwoId));

      try (AllocOnAnotherThread firstOne = new CpuAllocOnAnotherThread(taskOne, 5 * 1024 * 1024)) {
        firstOne.waitForAlloc();
        // This one should block
        try (AllocOnAnotherThread secondOne = new CpuAllocOnAnotherThread(taskTwo, 6 * 1024 * 1024)) {
          taskTwo.pollForState(RmmSparkThreadState.THREAD_BLOCKED, 1000, TimeUnit.MILLISECONDS);
          // Make sure that shuffle has higher priority than tasks...
          try (AllocOnAnotherThread thirdOne = new CpuAllocOnAnotherThread(shuffleOne, 6 * 1024 * 1024,
              taskTwo.getTaskId())) {
            shuffleOne.pollForState(RmmSparkThreadState.THREAD_BLOCKED, 1000, TimeUnit.MILLISECONDS);
            // But taskOne is not blocked, so there will be no retry until it is blocked, or else
            // it is making progress
            taskOne.doIt((TaskThreadOp<Void>) () -> {
              try {
                Thread.sleep(200);
              } catch (InterruptedException e) {
                throw new RuntimeException(e);
              }
              return null;
            });

            try {
              secondOne.waitForAlloc();
              fail("SHOULD HAVE THROWN...");
            } catch (ExecutionException ee) {
              assert (ee.getCause() instanceof CpuRetryOOM);
            }
            secondOne.freeAndWait();

            // Free the first allocation to wake up the shuffle thread, but not the second task yet...
            firstOne.freeAndWait();

            thirdOne.waitForAlloc();
            thirdOne.freeAndWait();
          }
        }
      }
    } finally {
      shuffleOne.done();
      taskOne.done();
      taskTwo.done();
    }
  }

  @Test
  public void testBasicBUFN() throws ExecutionException, InterruptedException, TimeoutException {
    // 10 MiB
    setupRmmForTestingWithLimits(10 * 1024 * 1024);
    // Task priority is assigned in FIFO order, we want task 2 to have the highest priority
    TaskThread taskTwo = new TaskThread("TEST THREAD TWO", getNextTid());
    taskTwo.initialize();
    TaskThread taskThree = new TaskThread("TEST THREAD ONE", getNextTid());
    taskThree.initialize();
    try {
      long tThreeId = taskThree.getThreadId();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(tThreeId));

      long tTwoId = taskTwo.getThreadId();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(tTwoId));

      try (AllocOnAnotherThread allocThreeOne = new GpuAllocOnAnotherThread(taskThree, 5 * 1024 * 1024)) {
        allocThreeOne.waitForAlloc();
        try (AllocOnAnotherThread allocTwoOne = new GpuAllocOnAnotherThread(taskTwo, 3 * 1024 * 1024)) {
          allocTwoOne.waitForAlloc();

          try (AllocOnAnotherThread allocTwoTwo = new GpuAllocOnAnotherThread(taskTwo, 3 * 1024 * 1024)) {
            taskTwo.pollForState(RmmSparkThreadState.THREAD_BLOCKED, 1000, TimeUnit.MILLISECONDS);

            try (AllocOnAnotherThread allocThreeTwo = new GpuAllocOnAnotherThread(taskThree, 4 * 1024 * 1024)) {
              // This one should be able to allocate because there is not enough memory, but
              // now all the threads would be blocked, so the lowest priority thread is going to
              // become BUFN
              taskThree.pollForState(RmmSparkThreadState.THREAD_BUFN_WAIT, 1000, TimeUnit.MILLISECONDS);
              try {
                allocThreeTwo.waitForAlloc();
                fail("ALLOC AFTER BUFN SHOULD HAVE THROWN...");
              } catch (ExecutionException ee) {
                assert(ee.getCause() instanceof GpuRetryOOM);
              }
              // allocOneTwo cannot be freed, nothing was allocated because it threw an exception.
              allocThreeOne.freeAndWait();
              Future<Void> f = taskThree.blockUntilReady();
              taskThree.pollForState(RmmSparkThreadState.THREAD_BUFN, 1000, TimeUnit.MILLISECONDS);

              // taskOne should only wake up after we finish task 2
              // Task two is now able to alloc
              allocTwoTwo.freeAndWait();
              allocTwoOne.freeAndWait();
              // Task two has freed things, but is still not done, so task one will stay blocked...
              taskTwo.pollForState(RmmSparkThreadState.THREAD_RUNNING, 1000, TimeUnit.MILLISECONDS);
              taskThree.pollForState(RmmSparkThreadState.THREAD_BUFN, 1000, TimeUnit.MILLISECONDS);

              taskTwo.done().get(1000, TimeUnit.MILLISECONDS);
              // Now that task two is done see if task one is running again...
              taskThree.pollForState(RmmSparkThreadState.THREAD_RUNNING, 1000, TimeUnit.MILLISECONDS);
              // Now we could finish trying our allocations, but this is good enough...
            }
          }
        }
      }
    } finally {
      taskThree.done();
      taskTwo.done();
    }
  }

  @Test
  public void testBasicBUFNCpu() throws ExecutionException, InterruptedException, TimeoutException {
    // 10 MiB
    setupRmmForTestingWithLimits(10 * 1024 * 1024);
    LimitingOffHeapAllocForTests.setLimit(10 * 1024 * 1024);
    // Task priority is assigned in FIFO order, we want task 2 to have the highest priority
    TaskThread taskTwo = new TaskThread("TEST THREAD TWO", getNextTid());
    taskTwo.initialize();
    TaskThread taskThree = new TaskThread("TEST THREAD ONE", getNextTid());
    taskThree.initialize();
    try {
      long tThreeId = taskThree.getThreadId();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(tThreeId));

      long tTwoId = taskTwo.getThreadId();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(tTwoId));

      try (AllocOnAnotherThread allocThreeOne = new CpuAllocOnAnotherThread(taskThree, 5 * 1024 * 1024)) {
        allocThreeOne.waitForAlloc();
        try (AllocOnAnotherThread allocTwoOne = new CpuAllocOnAnotherThread(taskTwo, 3 * 1024 * 1024)) {
          allocTwoOne.waitForAlloc();

          try (AllocOnAnotherThread allocTwoTwo = new CpuAllocOnAnotherThread(taskTwo, 3 * 1024 * 1024)) {
            taskTwo.pollForState(RmmSparkThreadState.THREAD_BLOCKED, 1000, TimeUnit.MILLISECONDS);

            try (AllocOnAnotherThread allocThreeTwo = new CpuAllocOnAnotherThread(taskThree, 4 * 1024 * 1024)) {
              // This one should be able to allocate because there is not enough memory, but
              // now all the threads would be blocked, so the lowest priority thread is going to
              // become BUFN
              taskThree.pollForState(RmmSparkThreadState.THREAD_BUFN_WAIT, 1000, TimeUnit.MILLISECONDS);
              try {
                allocThreeTwo.waitForAlloc();
                fail("ALLOC AFTER BUFN SHOULD HAVE THROWN...");
              } catch (ExecutionException ee) {
                assert(ee.getCause() instanceof CpuRetryOOM);
              }
              // allocOneTwo cannot be freed, nothing was allocated because it threw an exception.
              allocThreeOne.freeAndWait();
              Future<Void> f = taskThree.blockUntilReady();
              taskThree.pollForState(RmmSparkThreadState.THREAD_BUFN, 1000, TimeUnit.MILLISECONDS);

              // taskOne should only wake up after we finish task 2
              // Task two is now able to alloc
              allocTwoTwo.freeAndWait();
              allocTwoOne.freeAndWait();
              // Task two has freed things, but is still not done, so task one will stay blocked...
              taskTwo.pollForState(RmmSparkThreadState.THREAD_RUNNING, 1000, TimeUnit.MILLISECONDS);
              taskThree.pollForState(RmmSparkThreadState.THREAD_BUFN, 1000, TimeUnit.MILLISECONDS);

              taskTwo.done().get(1000, TimeUnit.MILLISECONDS);
              // Now that task two is done see if task one is running again...
              taskThree.pollForState(RmmSparkThreadState.THREAD_RUNNING, 1000, TimeUnit.MILLISECONDS);
              // Now we could finish trying our allocations, but this is good enough...
            }
          }
        }
      }
    } finally {
      taskThree.done();
      taskTwo.done();
    }
  }

  @Test
  public void testBUFNSplitAndRetrySingleThread() throws ExecutionException, InterruptedException, TimeoutException {
    // We are doing ths one single threaded.
    // 10 MiB
    setupRmmForTestingWithLimits(10 * 1024 * 1024);

    TaskThread taskOne = new TaskThread("TEST THREAD ZERO", getNextTid());
    taskOne.initialize();
    try {
      long threadId = taskOne.getThreadId();
      assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
      try (AllocOnAnotherThread one = new GpuAllocOnAnotherThread(taskOne, 5 * 1024 * 1024)) {
        one.waitForAlloc();
        try (AllocOnAnotherThread two = new GpuAllocOnAnotherThread(taskOne, 6 * 1024 * 1024)) {
          two.waitForAlloc();
          fail("Expect that allocating more memory than is allowed would fail");
        } catch (ExecutionException oom) {
          assert oom.getCause() instanceof GpuRetryOOM : oom.toString();
        }
        try {
          taskOne.blockUntilReady().get(1000, TimeUnit.MILLISECONDS);
          fail("Expect split and retry after all tasks blocked.");
        } catch (ExecutionException oom) {
          assert oom.getCause() instanceof GpuSplitAndRetryOOM : oom.toString();
        }
        assertEquals(RmmSparkThreadState.THREAD_RUNNING, RmmSpark.getStateOf(threadId));
        // Now we try to allocate with half the data.
        try (AllocOnAnotherThread secondTry = new GpuAllocOnAnotherThread(taskOne, 3 * 1024 * 1024)) {
          secondTry.waitForAlloc();
        }
      }
    } finally {
      taskOne.done();
    }
  }

  @Test
  public void testInsertMultipleOOMs() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 10 * 1024 * 1024);
    RmmSpark.setEventHandler(new BaseRmmEventHandler(), "stderr");
    long threadId = RmmSpark.getCurrentThreadId();
    long taskId = getNextTid();
    Thread t = Thread.currentThread();
    RmmSpark.startDedicatedTaskThread(threadId, taskId, t);
    try {
      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();

      // Force an exception
      int numRetryOOMs = 3;
      RmmSpark.forceRetryOOM(threadId, numRetryOOMs);
      for (int i = 0; i < numRetryOOMs; i++) {
        assertThrows(GpuRetryOOM.class, () -> Rmm.alloc(100).close());
        // Verify that injecting OOM does not cause the block to actually happen
        RmmSpark.blockThreadUntilReady();
      }

      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();

      // Force another exception
      int numSplitAndRetryOOMs = 5;
      RmmSpark.forceSplitAndRetryOOM(threadId, numSplitAndRetryOOMs);
      for (int i = 0; i < numSplitAndRetryOOMs; i++) {
        assertThrows(GpuSplitAndRetryOOM.class, () -> Rmm.alloc(100).close());
        // Verify that injecting OOM does not cause the block to actually happen
        RmmSpark.blockThreadUntilReady();
      }

      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();
    } finally {
      RmmSpark.removeDedicatedThreadAssociation(threadId, taskId);
      RmmSpark.taskDone(taskId);
      TaskPriority.taskDone(taskId);
    }
  }

  @Test
  public void testCudfException() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 10 * 1024 * 1024);
    RmmSpark.setEventHandler(new BaseRmmEventHandler(), "stderr");
    long threadId = RmmSpark.getCurrentThreadId();
    long taskId = getNextTid();
    Thread t = Thread.currentThread();
    RmmSpark.startDedicatedTaskThread(threadId, taskId, t);
    try {
      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();

      // Force an exception
      int numCudfExceptions = 3;
      RmmSpark.forceCudfException(threadId, numCudfExceptions);
      for (int i = 0; i < numCudfExceptions; i++) {
        assertThrows(CudfException.class, () -> Rmm.alloc(100).close());
        // Verify that injecting OOM does not cause the block to actually happen
        RmmSpark.blockThreadUntilReady();
      }

      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();
    } finally {
      RmmSpark.removeDedicatedThreadAssociation(threadId, taskId);
      RmmSpark.taskDone(taskId);
      TaskPriority.taskDone(taskId);
    }
  }

  @Test
  public void retryWatchdog() {
    // 10 MiB
    setupRmmForTestingWithLimits(10 * 1024 * 1024);
    long threadId = RmmSpark.getCurrentThreadId();
    long taskId = getNextTid();
    long numRetries = 0;
    Thread t = Thread.currentThread();
    RmmSpark.startDedicatedTaskThread(threadId, taskId, t);
    long startTime = System.nanoTime();
    try (DeviceMemoryBuffer filler = Rmm.alloc(9 * 1024 * 1024)) {
      while (numRetries < 10000) {
        try {
          Rmm.alloc(2 * 1024 * 1024).close();
          fail("overallocation should have failed");
        } catch (GpuRetryOOM room) {
          numRetries++;
          try {
            RmmSpark.blockThreadUntilReady();
          } catch (GpuSplitAndRetryOOM sroom) {
            numRetries++;
          }
        } catch (GpuSplitAndRetryOOM sroom) {
          fail("retry should be thrown before split and retry...");
        }
      }
      fail("retried too many times " + numRetries);
    } catch (OutOfMemoryError oom) {
      // The 500 is hard coded in the code below
      assertEquals(500, numRetries);
    } finally {
      RmmSpark.removeDedicatedThreadAssociation(threadId, taskId);
      RmmSpark.taskDone(taskId);
      TaskPriority.taskDone(taskId);
    }
    long endTime = System.nanoTime();
    System.err.println("Took " + (endTime - startTime) + "ns to retry 500 times...");
  }
  
  //
  // These next two tests deal with a special case where allocations (and allocation failures)
  // could happen during spill handling.
  //
  // When we spill we may need to invoke cuDF code that creates memory, specifically to
  // pack previously unpacked memory into a single contiguous buffer (cudf::chunked_pack).
  // This operation, although it makes use of an auxiliary memory resource, still deals with
  // cuDF apis that could, at any time, allocate small amounts of memory in the default memory
  // resource. As such, allocations and allocation failures could happen, which cause us
  // to recursively enter the state machine in SparkResourceAdaptorJni.
  //
  @Test
  public void testAllocationDuringSpill() {
    // Create a handler that allocates 1 byte from the handler (it should succeed)
    AllocatingRmmEventHandler rmmEventHandler = new AllocatingRmmEventHandler(1);
    // 10 MiB
    setupRmmForTestingWithLimits(10 * 1024 * 1024, rmmEventHandler);
    long threadId = RmmSpark.getCurrentThreadId();
    long taskId = getNextTid();
    Thread t = Thread.currentThread();
    RmmSpark.startDedicatedTaskThread(threadId, taskId, t);
    assertThrows(GpuOOM.class, () -> {
      try (DeviceMemoryBuffer filler = Rmm.alloc(9 * 1024 * 1024)) {
        try (DeviceMemoryBuffer shouldFail = Rmm.alloc(2 * 1024 * 1024)) {}
        fail("overallocation should have failed");
      } finally {
        RmmSpark.removeDedicatedThreadAssociation(threadId, taskId);
        RmmSpark.taskDone(taskId);
        TaskPriority.taskDone(taskId);
      }
    });
    // We retry the failed allocation for the last thread before going into
    // the BUFN state. So we have 22 allocations instead of the expected 11
    assertEquals(22, rmmEventHandler.getAllocationCount());
  }

  @Test
  public void testAllocationFailedDuringSpill() {
    // Create a handler that allocates 2MB from the handler (it should fail)
    AllocatingRmmEventHandler rmmEventHandler = new AllocatingRmmEventHandler(2L*1024*1024);
    // 10 MiB
    setupRmmForTestingWithLimits(10 * 1024 * 1024, rmmEventHandler);
    long threadId = RmmSpark.getCurrentThreadId();
    long taskId = getNextTid();
    Thread t = Thread.currentThread();
    RmmSpark.startDedicatedTaskThread(threadId, taskId, t);
    assertThrows(GpuOOM.class, () -> {
      try (DeviceMemoryBuffer filler = Rmm.alloc(9 * 1024 * 1024)) {
        try (DeviceMemoryBuffer shouldFail = Rmm.alloc(2 * 1024 * 1024)) {}
        fail("overallocation should have failed");
      } finally {
        RmmSpark.removeDedicatedThreadAssociation(threadId, taskId);
        RmmSpark.taskDone(taskId);
        TaskPriority.taskDone(taskId);
      }
    });
    assertEquals(0, rmmEventHandler.getAllocationCount());
  }

  private static class BaseRmmEventHandler implements RmmEventHandler {
    @Override
    public long[] getAllocThresholds() {
      return null;
    }

    @Override
    public long[] getDeallocThresholds() {
      return null;
    }

    @Override
    public void onAllocThreshold(long totalAllocSize) {
    }

    @Override
    public void onDeallocThreshold(long totalAllocSize) {
    }

    @Override
    public boolean onAllocFailure(long sizeRequested, int retryCount) {
      // This is just a test for now, no spilling...
      return false;
    }
  }

  private static class AllocatingRmmEventHandler extends BaseRmmEventHandler {
    // if true, we are still in the onAllocFailure callback (recursive call)
    boolean stillHandlingAllocFailure = false;

    int allocationCount;

    long allocSize;

    public int getAllocationCount() {
      return allocationCount;
    }

    public AllocatingRmmEventHandler(long allocSize) {
      this.allocSize = allocSize;
    }

    @Override
    public boolean onAllocFailure(long sizeRequested, int retryCount) {
      // Catch java.lang.OutOfMemory since we could gt this exception during `Rmm.alloc`.
      // Catch all throwables because any other exception is not handled gracefully from callers
      // but if we do see such exceptions make sure we call `fail` so we get a test failure.
      try {
        if (stillHandlingAllocFailure) {
          // detected a loop
          stillHandlingAllocFailure = false;
          return false;
        } else {
          stillHandlingAllocFailure = true;
          try (DeviceMemoryBuffer dmb = Rmm.alloc(allocSize)) { // try to allocate one byte, and free
            allocationCount++;
            stillHandlingAllocFailure = false;
          }
          // allow retries up to 10 times
          return retryCount < 10;
        }
      } catch (java.lang.OutOfMemoryError e) {
        // return false here, this allocation failure handling failed with 
        // java.lang.OutOfMemory from `RmmJni`
        return false;
      } catch (Throwable t) {
        fail("unexpected exception in onAllocFailure", t);
        return false;
      }
    }

  }
}