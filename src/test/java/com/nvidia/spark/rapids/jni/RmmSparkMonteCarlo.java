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

import ai.rapids.cudf.*;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;

public class RmmSparkMonteCarlo {
  public static AtomicLong numSplitAndRetry = new AtomicLong(0);
  public static AtomicLong numRetry = new AtomicLong(0);

  private static int parsePosInt(String input) {
    int value = Integer.parseInt(input);
    if (value <= 0) {
      throw new RuntimeException(value + " is not positive");
    }
    return value;
  }

  private static long parsePosLong(String input) {
    long value = Long.parseLong(input);
    if (value <= 0) {
      throw new RuntimeException(value + " is not positive");
    }
    return value;
  }

  public static void main(String [] args) throws InterruptedException {
    // Run a simple monte carlo simulation to try and see the performance impact of retry
    // on random situations.
    boolean useSparkRmm = true;
    int numIterations = 500;
    long numTasks = 12;
    int parallelism = 4;
    long seed = System.nanoTime();
    long gpuMemoryMiB = 1024;
    long taskMaxMiB = 300;
    int allocMode = RmmAllocationMode.CUDA_ASYNC;
    int taskRetry = 3;
    int maxTaskAllocs = 50;
    int maxTaskSleep = 100;
    boolean logging = true;
    boolean isSkewed = false;
    boolean useTemplate = false;
    double skewAmount = 2.0;
    double templateChangeAmount = 0.05;
    int shuffleThreads = 0;
    boolean debugOoms = false;

    for (String arg: args) {
      if (arg.equals("--baseline")) {
        useSparkRmm = false;
      } else if (arg.equals("--debugOOMs")) {
        debugOoms = true;
      } else if (arg.startsWith("--iter=")) {
        numIterations = parsePosInt(arg.substring(7));
      } else if (arg.startsWith("--numTasks=")) {
        numTasks = parsePosLong(arg.substring(11));
      } else if (arg.startsWith("--parallel=")) {
        parallelism = parsePosInt(arg.substring(11));
      } else if (arg.startsWith("--seed=")) {
        seed = Long.parseLong(arg.substring(7));
      } else if (arg.startsWith("--gpuMiB=")) {
        gpuMemoryMiB = parsePosLong(arg.substring(9));
      } else if (arg.startsWith("--taskMaxMiB=")) {
        taskMaxMiB = parsePosLong(arg.substring(13));
      } else if (arg.startsWith("--taskRetry=")) {
        taskRetry = parsePosInt(arg.substring(12));
      } else if (arg.startsWith("--maxTaskAllocs=")) {
        maxTaskAllocs = parsePosInt(arg.substring(16));
      } else if (arg.startsWith("--maxTaskSleep=")) {
        maxTaskSleep = Integer.parseInt(arg.substring(15));
      } else if (arg.startsWith("--shuffleThreads=")) {
        shuffleThreads = parsePosInt(arg.substring(17));
      } else if (arg.startsWith("--allocMode=")) {
        String mode = arg.substring(12);
        if (mode.equalsIgnoreCase("POOL")) {
          allocMode = RmmAllocationMode.POOL | RmmAllocationMode.CUDA_DEFAULT;
        } else if (mode.equalsIgnoreCase("ASYNC")) {
          allocMode = RmmAllocationMode.CUDA_ASYNC;
        } else if (mode.equalsIgnoreCase("ARENA")) {
          allocMode = RmmAllocationMode.ARENA | RmmAllocationMode.CUDA_DEFAULT;
        } else if (mode.equalsIgnoreCase("CUDA")) {
          allocMode = RmmAllocationMode.CUDA_DEFAULT;
        } else {
          throw new IllegalArgumentException("Unknown RMM allocation mode " + mode);
        }
      } else if (arg.equals("--noLog")) {
        logging = false;
      } else if (arg.startsWith("--skewAmount=")) {
        skewAmount = Math.abs(Double.parseDouble(arg.substring(13)));
      } else if (arg.startsWith("--templateChangeAmount=")) {
        templateChangeAmount = Math.abs(Double.parseDouble(arg.substring(23)));
      } else if (arg.equals("--skewed")) {
        isSkewed = true;
        useTemplate = true;
      } else if (arg.equals("--useTemplate")) {
        useTemplate = true;
      } else if (arg.equals("--help")) {
        System.out.println("RMM Spark Monte Carlo Simulation");
        System.out.println("--baseline\trun without RmmSpark for a baseline");
        System.out.println("--debugOOMs\tprint debug messages on OutOfMemoryError");
        System.out.println("--help\tprint this message");
        System.out.println("--iter=<NUM>\tnumber of iterations to do for the simulation");
        System.out.println("--parallel=<NUM>\tnumber of tasks that can run in parallel on the GPU");
        System.out.println("--seed=<NUM>\tthe random seed to use for the test");
        System.out.println("--gpuMiB=<NUM>\tlimit on the GPUs memory to use for testing");
        System.out.println("--taskMaxMiB=<NUM>\tmaximum amount of memory a regular task may have allocated");
        System.out.println("--allocMode=<MODE>\tthe RMM allocation mode to use POOL, ASYNC, ARENA, CUDA");
        System.out.println("--taskRetry=<NUM>\tmaximum number of times to retry a task before failing the situation");
        System.out.println("--maxTaskAllocs=<NUM>\tmaximum number of allocations a task can make");
        System.out.println("--maxTaskSleep=<NUM>\tmaximum amount of time a task can sleep for (sim processing)");
        System.out.println("--noLog\tdisable logging");
        System.out.println("--skewed\tgenerate templated tasks and skew one of them by skewAmount");
        System.out.println("--skewAmount=<NUM>\tthe amount to multiply the skewed allocations by");
        System.out.println("--useTemplate\tif all of the tasks should be the same, but change by +/- templateChangeAmount as a multiplier");
        System.out.println("--templateChangeAmount=<NUM>\tA multiplication factor to change the template task by when making new tasks (as a multiplier)");
        System.out.println("--shuffleThreads=<NUM>\tThe number of threads to use to simulate UCX shuffle");
        System.exit(0);
      } else {
        throw new IllegalArgumentException("Unexpected argument " + arg +
            " use --help for allowed args");
      }
    }

    System.out.println("Running simulations with");
    System.out.println("useSparkRmm: " + useSparkRmm);
    System.out.println("numIterations " + numIterations);
    System.out.println("numTasks " + numTasks);
    System.out.println("parallelism " + parallelism);
    System.out.println("seed " + seed);
    System.out.println("gpuMemoryMiB " + gpuMemoryMiB);
    System.out.println("taskMaxMiB " + taskMaxMiB);
    System.out.println("allocMode " + allocMode);
    System.out.println("taskRetry " + taskRetry);
    System.out.println("logging " + logging);
    System.out.println("maxTaskAllocs " + maxTaskAllocs);
    System.out.println("maxTaskSleep " + maxTaskSleep);
    System.out.println("skewed " + isSkewed);
    System.out.println("skewAmount " + skewAmount);
    System.out.println("templated " + useTemplate);
    System.out.println("templateChangeAmount " + templateChangeAmount);
    System.out.println("shuffleThreads " + shuffleThreads);

    List<Situation> situations = generateSituations(seed, numIterations, numTasks,
        taskMaxMiB, maxTaskAllocs, maxTaskSleep,
        isSkewed, skewAmount, useTemplate, templateChangeAmount);
    SituationRunner runner = new SituationRunner(parallelism, taskRetry, shuffleThreads, debugOoms);
    setupRmm(allocMode, gpuMemoryMiB, useSparkRmm, logging);
    int result = runner.run(situations);
    runner.finish();
    System.exit(result);
  }

  public static void setupRmm(int allocationMode, long limitMiB, boolean useSparkRmm,
      boolean enableLogging) {
    long limitBytes = limitMiB * 1024 * 1024;
    Rmm.LogConf rmmLog = null;
    if (enableLogging) {
     rmmLog = Rmm.logTo(new File("./monte.rmm.log"));
    }
    if (allocationMode == RmmAllocationMode.CUDA_DEFAULT) {
      // We want to limit the total size, but Rmm will not do that by default...
      // Setup Rmm is a simple way
      RmmDeviceMemoryResource resource = null;
      boolean succeeded = false;
      try {
        resource = new RmmCudaMemoryResource();
        resource = new RmmLimitingResourceAdaptor<>(resource, limitBytes, 256);
        if (enableLogging) {
          resource = new RmmLoggingResourceAdaptor<>(resource, rmmLog, true);
        }
        resource = new RmmTrackingResourceAdaptor<>(resource, 256);
        Rmm.setCurrentDeviceResource(resource, null, false);
        succeeded = true;
      } finally {
        if (!succeeded && resource != null) {
          resource.close();
        }
      }
    } else {
      Rmm.initialize(allocationMode, rmmLog, limitBytes);
    }
    boolean needsSync = (allocationMode & RmmAllocationMode.CUDA_ASYNC) == RmmAllocationMode.CUDA_ASYNC;
    if (useSparkRmm) {
      if (enableLogging) {
        RmmSpark.setEventHandler(new TestRmmEventHandler(needsSync), "./monte.state.log");
      } else {
        RmmSpark.setEventHandler(new TestRmmEventHandler(needsSync), null);
      }
      System.out.println("RMMSpark is initialized");
    } else {
      Rmm.setEventHandler(new TestRmmEventHandler(needsSync));
      System.out.println("RMM is initialized");
    }
  }

  private static class TestRmmEventHandler implements RmmEventHandler {
    private final boolean needsSync;

    public TestRmmEventHandler(boolean needsSync) {
      this.needsSync = needsSync;
    }
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
      if (needsSync && retryCount <= 0) {
        Cuda.DEFAULT_STREAM.sync();
        return true;
      }
      return false;
    }
  }

  public static class TaskRunnerThread extends Thread {
    private final CyclicBarrier barrier;
    private final SituationRunner runner;
    private final int taskRetry;
    private final ExecutorService shuffle;
    Situation currentSit = null;

    volatile boolean hadOtherFailures = false;

    volatile boolean done = false;

    public TaskRunnerThread(CyclicBarrier barrier, SituationRunner runner, int taskRetry,
        ExecutorService shuffle) {
      this.barrier = barrier;
      this.runner = runner;
      this.taskRetry = taskRetry;
      this.shuffle = shuffle;
    }

    public void finish() {
      this.done = true;
      synchronized (this) {
        notifyAll();
      }
    }

    public synchronized void setSit(Situation sit) {
      currentSit = sit;
      notifyAll();
    }

    synchronized void waitForSitToBeSet() {
      try {
        while (currentSit == null && !done) {
          wait(1000);
        }
      } catch (InterruptedException e) {
        throw new RuntimeException(e);
      }
    }

    synchronized Task getNextTask() {
      if (currentSit != null) {
        return currentSit.nextTask();
      } else {
        return null;
      }
    }

    private synchronized void retryTask(Task t) {
      currentSit.retryTask(t);
    }

    public void run() {
      try {
        while (!done) { // situations loop
          waitForSitToBeSet();
          Task t = getNextTask();
          long timeLost = 0;
          while (t != null) { // task loop
            Task backup = t.cloneForRetry();
            long start = System.nanoTime();
            boolean success = false;
            try {
              t.run(shuffle);
              success = true;
            } catch (OutOfMemoryError oom) {
              timeLost += System.nanoTime() - start;
              if (runner.debugOoms) {
                System.err.println("OOM for task: " + t.taskId +
                    " and thread: " + RmmSpark.getCurrentThreadId() + " " + oom);
                oom.printStackTrace(System.err);
              }
              // ignored
            }
            timeLost += t.getTimeLost();
            Cuda.DEFAULT_STREAM.sync();
            if (!success) {
              long stopTime = System.nanoTime() + 50;
              while (System.nanoTime() < stopTime) {
                Thread.yield();
              }
              if (backup.retryCount > taskRetry) {
                runner.setSitFailed();
              } else {
                // Going to retry
                retryTask(backup);
              }
            }
            long end = System.nanoTime();
            runner.updateTaskStats(success, end - start, timeLost);
            t = getNextTask();
          }
          try { // situation is done so wait for all others too
            synchronized (this) {
              currentSit = null;
            }
            barrier.await();
          } catch (InterruptedException | BrokenBarrierException e) {
            throw new RuntimeException(e);
          }
        }
      } catch (Throwable e) {
        System.err.println("ERROR: TID: " + RmmSpark.getCurrentThreadId() + " " + e);
        e.printStackTrace(System.err);
        hadOtherFailures = true;
      }
    }

    public boolean hadOtherFailures() {
      return hadOtherFailures;
    }
  }

  static class ShuffleThreadFactory implements ThreadFactory {
    static final AtomicLong idGen = new AtomicLong(0);
    @Override
    public Thread newThread(Runnable runnable) {
      long id = idGen.getAndIncrement();
      Thread t = new Thread(runnable);
      t.setDaemon(true);
      t.setName("SHUFFLE-THREAD-" + id);
      return t;
    }
  }

  public static class SituationRunner {
    final TaskRunnerThread[] threads;
    public final boolean debugOoms;
    private ExecutorService shuffle;
    final CyclicBarrier barrier;
    volatile boolean sitIsDone = false;

    // Stats
    volatile int failedSits;
    volatile int successSits;
    volatile int numTasks;
    volatile int failedTasks;
    volatile int successTasks;
    volatile long totalTaskTime;
    volatile long totalTimeLost;
    volatile boolean sitFailed;
    volatile boolean didThisSitFail = false;

    public SituationRunner(int parallelism, int taskRetry, int shuffleThreads, boolean debugOoms) {
      this.debugOoms = debugOoms;
      Object notify = this;
      if (shuffleThreads > 0) {
        shuffle = java.util.concurrent.Executors.newFixedThreadPool(shuffleThreads,
            new ShuffleThreadFactory());
      } else {
        shuffle = null;
      }
      threads = new TaskRunnerThread[parallelism];
      barrier = new CyclicBarrier(parallelism, () -> {
        synchronized (notify) {
          sitIsDone = true;
          if (sitFailed) {
            failedSits++;
            sitFailed = false;
            didThisSitFail = true;
          } else {
            successSits++;
          }
          notify.notifyAll();
        }
      });

      for (int i = 0; i < parallelism; i++) {
        threads[i] = new TaskRunnerThread(barrier, this, taskRetry, shuffle);
        threads[i].setDaemon(true);
        threads[i].start();
      }
    }

    public int run(List<Situation> situations) throws InterruptedException {
      numSplitAndRetry.set(0);
      numRetry.set(0);
      synchronized (this) {
        failedSits = 0;
        successSits = 0;
        numTasks = 0;
        failedTasks = 0;
        successTasks = 0;
        totalTaskTime = 0;
        totalTimeLost = 0;
      }
      int numSits = 0;
      long totalSitTime = 0;
      for (Situation sit: situations) {
        numSits++;
        long start = System.nanoTime();
        for (TaskRunnerThread t : threads) {
          t.setSit(sit);
        }
        synchronized (this) {
          while (!sitIsDone) {
            wait();
          }
          sitIsDone = false;
        }
        long end = System.nanoTime();
        if (didThisSitFail) {
          System.out.print("f");
          didThisSitFail = false;
        } else {
          System.out.print(".");
        }
        if (numSits % 100 == 0 || numSits == situations.size()) {
          System.out.println();
        }
        System.out.flush();
        totalSitTime += (end - start);
      }

      synchronized (this) {
        System.out.println("Monte Carlo sim finished....");
        System.out.println("Situations: " + numSits + " total, " + successSits + " successful, " +
            failedSits + " failed. " + asTimeStr(totalSitTime));
        System.out.println("Tasks: " + numTasks + " total, " + successTasks + " successful, " +
            failedTasks + " failed. " + asTimeStr(totalTaskTime) + " taskTime " +
            asTimeStr(totalTimeLost) + " lost task computation");
        System.out.println("Exceptions: " + numSplitAndRetry.get() + " splits, " +
            numRetry.get() + " retries.");
      }
      if (failedSits > 0) {
        return -1;
      } else {
        boolean unexpectedFailures = false;
        for (TaskRunnerThread t : threads) {
          unexpectedFailures = t.hadOtherFailures() || unexpectedFailures;
        }
        if (unexpectedFailures) {
          return -2;
        }
        return 0;
      }
    }

    private static String asTimeStr(long timeNs) {
      long justms = TimeUnit.NANOSECONDS.toMillis(timeNs);

      long hours = TimeUnit.NANOSECONDS.toHours(timeNs);
      long hoursInNanos = TimeUnit.HOURS.toNanos(hours);
      timeNs = timeNs - hoursInNanos;
      long mins = TimeUnit.NANOSECONDS.toMinutes(timeNs);
      long minsInNanos = TimeUnit.MINUTES.toNanos(mins);
      timeNs = timeNs - minsInNanos;
      long secs = TimeUnit.NANOSECONDS.toSeconds(timeNs);
      long secsInNanos = TimeUnit.SECONDS.toNanos(secs);
      long ns = timeNs - secsInNanos;
      return String.format("%1$02d:%2$02d:%3$02d.%4$09d", hours, mins, secs, ns) +
          " or " + justms + " ms";
    }

    public void finish() {
      for (TaskRunnerThread t : threads) {
        t.finish();
      }
    }

    public synchronized void updateTaskStats(boolean success, long timeNs, long timeLost) {
      if (success) {
        successTasks++;
      } else {
        failedTasks++;
      }
      numTasks++;
      totalTaskTime += timeNs;
      totalTimeLost += timeLost;
    }

    public synchronized void setSitFailed() {
      sitFailed = true;
    }
  }

  interface MemoryOp {
    default void doIt(DeviceMemoryBuffer[] buffers, long taskId) {
      long threadId = RmmSpark.getCurrentThreadId();
      RmmSpark.shuffleThreadWorkingOnTasks(new long[]{taskId});
      RmmSpark.startRetryBlock(threadId);
      try {
        int tries = 0;
        while (tries < 100 && tries >= 0) {
          try {
            if (tries > 0) {
              RmmSpark.blockThreadUntilReady();
            }
            tries++;
            doIt(buffers);
            tries = -1;
          } catch (GpuRetryOOM oom) {
            // Don't need to clear the buffers, because there is only one buffer.
            numRetry.incrementAndGet();
          } catch (CpuRetryOOM oom) {
            // Don't need to clear the buffers, because there is only one buffer.
            numRetry.incrementAndGet();
          }
        }
        if (tries >= 100) {
          throw new OutOfMemoryError("Could not make shuffle work after " + tries + " tries");
        }
      } finally {
        RmmSpark.endRetryBlock(threadId);
        RmmSpark.poolThreadFinishedForTask(taskId);
      }
    }

    void doIt(DeviceMemoryBuffer[] buffers);

    MemoryOp[] split();

    MemoryOp randomMod(Random r, double templateChangeAmount);

    MemoryOp makeSkewed(double skewAmount);
  }

  public static class AllocOp implements MemoryOp {
    private static AtomicLong idgen = new AtomicLong(0);

    public final int offset;
    private final long size;
    private final long sleepTime;

    private final long id;

    public AllocOp(int offset, long size, long sleepTime) {
      this.offset = offset;
      this.size = size;
      this.sleepTime = sleepTime;
      this.id = idgen.getAndIncrement();
    }

    private AllocOp(int offset, long size, long sleepTime, long id) {
      this.offset = offset;
      this.size = size;
      this.sleepTime = sleepTime;
      this.id = id;
    }
    @Override
    public String toString() {
      return "ALLOC[" + offset + "] " + size + " SLEEP " + sleepTime;
    }

    @Override
    public void doIt(DeviceMemoryBuffer[] buffers) {
      buffers[offset] = Rmm.alloc(size);
      if (sleepTime > 0) {
        long stopTime = System.nanoTime() + sleepTime;
        while (System.nanoTime() < stopTime) {
          Thread.yield();
        }
      }
    }

    @Override
    public MemoryOp[] split() {
      // 256 is the alignment so going smaller does not help, but we might need it because of
      // skew/etc

      MemoryOp[] ret = new MemoryOp[2];
      // If we get smaller than 256, the alignment size, it does not matter
      if (size < 256) {
        // We cannot split anymore, but it is unlikely to actually happen this way, so we
        // are just going to keep it the same.
        ret[0] = this;
        ret[1] = this;
      } else {
        ret[0] = new AllocOp(offset, size / 2, sleepTime, id);
        ret[1] = new AllocOp(offset, size - (size / 2), sleepTime, id);
      }
      return ret;
    }

    @Override
    public MemoryOp randomMod(Random r, double templateChangeAmount) {
      double proposedSizeMult = (1.0 - (r.nextDouble() * 2.0)) * templateChangeAmount;
      long newSize = (long)(proposedSizeMult * size);
      if (newSize <= 0) {
        newSize = 1;
      }
      return new AllocOp(offset, newSize, sleepTime);
    }

    @Override
    public MemoryOp makeSkewed(double skewAmount) {
      return new AllocOp(offset, (long)(size * skewAmount), sleepTime);
    }
  }

  public static class FreeOp implements MemoryOp {
    private final int offset;

    public FreeOp(int offset) {
      this.offset = offset;
    }

    @Override
    public String toString() {
      return "FREE[" + offset + "]";
    }

    @Override
    public void doIt(DeviceMemoryBuffer[]  buffers) {
      DeviceMemoryBuffer buf = buffers[offset];
      if (buf != null) {
        buf.close();
      }
      buffers[offset] = null;
    }

    @Override
    public MemoryOp[] split() {
      MemoryOp[] ret = new MemoryOp[2];
      ret[0] = this;
      ret[1] = this;
      return ret;
    }

    @Override
    public MemoryOp randomMod(Random r, double templateChangeAmount) {
      // No need to change anything here...
      return this;
    }

    @Override
    public MemoryOp makeSkewed(double skewAmount) {
      // No need to change anything here...
      return this;
    }
  }

  public static class TaskOpSet {
    DeviceMemoryBuffer[] buffers;
    ArrayList<MemoryOp> operations;
    final int numBuffers;
    long allocatedBeforeError = 0;

    private TaskOpSet(int numBuffers, ArrayList<MemoryOp> operations) {
      this.numBuffers = numBuffers;
      this.operations = operations;
    }

    public TaskOpSet(Random r, long taskMaxMiB,
        int maxTaskAllocs, int maxTaskSleep) {
      long maxSleepTimeNano = TimeUnit.MILLISECONDS.toNanos(maxTaskSleep);
      long totalSleepTimeNano = 0;
      if (maxSleepTimeNano > 0) {
        totalSleepTimeNano = Math.abs(r.nextLong() % maxSleepTimeNano);
      }
      long maxBytes = taskMaxMiB * 1024 * 1024;
      long totalAllocated = 0;
      int numAllocOps = 1 + r.nextInt(maxTaskAllocs);
      numBuffers = numAllocOps;
      int numOps = numAllocOps * 2; // Alloc + corresponding free
      operations = new ArrayList<>(numOps);
      LinkedList<AllocOp> outstandingAllocOps = new LinkedList<>();
      double[] sleepWeights = new double[numAllocOps];
      double totalSleepWeight = 0;
      for (int i = 0; i < sleepWeights.length; i++) {
        sleepWeights[i] = 0.00001 + r.nextDouble();
        totalSleepWeight += sleepWeights[i];
      }

      int allocOpNum = 0;
      for (int i = 0; i < numOps; i++) {
        long maxAllocAmount = maxBytes - totalAllocated;
        // We favor allocs over frees, unless we have to
        if (allocOpNum < numAllocOps && maxAllocAmount > 0 &&
            (outstandingAllocOps.size() <= 0 || r.nextDouble() > 0.40)) {
          long size = 1 + Math.abs(r.nextLong() % maxAllocAmount);
          // We want the sleeps to be very small because we are not simulating
          // the time, and generally they will be. In the future we can make this
          // configurable.
          long sleepTime = (long)(totalSleepTimeNano * sleepWeights[allocOpNum]/totalSleepWeight);
          AllocOp ao = new AllocOp(allocOpNum, size, sleepTime);
          operations.add(ao);
          outstandingAllocOps.add(ao);
          allocOpNum++;
          totalAllocated += size;
        } else {
          AllocOp ao = outstandingAllocOps.remove(r.nextInt(outstandingAllocOps.size()));
          operations.add(new FreeOp(ao.offset));
          totalAllocated -= ao.size;
        }
      }
    }

    public TaskOpSet[] split() {
      TaskOpSet[] ret = new TaskOpSet[2];
      ArrayList<MemoryOp> aOperations = new ArrayList<>(operations.size());
      ArrayList<MemoryOp> bOperations = new ArrayList<>(operations.size());
      for (int i = 0; i < operations.size(); i++) {
        MemoryOp[] splitOp = operations.get(i).split();
        aOperations.add(splitOp[0]);
        bOperations.add(splitOp[1]);
      }
      ret[0] = new TaskOpSet(numBuffers, aOperations);
      ret[1] = new TaskOpSet(numBuffers, bOperations);
      return ret;
    }

    private void cleanBuffers() {
      for (int i = 0; i < buffers.length; i++) {
        DeviceMemoryBuffer buff = buffers[i];
        if (buff != null) {
          allocatedBeforeError += buff.getLength();
          buff.close();
        }
        buffers[i] = null;
      }
    }

    public void run(ExecutorService shuffle, long taskId) {
      buffers = new DeviceMemoryBuffer[numBuffers];
      allocatedBeforeError = 0;
      boolean isForShuffle = shuffle != null;
      boolean done = false;
      while(!done) {
        try {
          for (MemoryOp op: operations) {
            if (isForShuffle) {
              try {
                RmmSpark.submittingToPool();
                Future<?> f = shuffle.submit(() -> op.doIt(buffers, taskId));
                RmmSpark.doneWaitingOnPool();
                RmmSpark.waitingOnPool();
                f.get(1000, TimeUnit.SECONDS);
              } finally {
                isForShuffle = false;
                RmmSpark.doneWaitingOnPool();
              }
            } else {
              op.doIt(buffers);
            }
          }
          done = true;
        } catch (GpuRetryOOM room) {
          numRetry.incrementAndGet();
          cleanBuffers();
          RmmSpark.blockThreadUntilReady();
        } catch (CpuRetryOOM room) {
          numRetry.incrementAndGet();
          cleanBuffers();
          RmmSpark.blockThreadUntilReady();
        } catch (ExecutionException ee) {
          OutOfMemoryError oom = new OutOfMemoryError("Came From Shuffle");
          oom.addSuppressed(ee);
          throw oom;
        } catch (InterruptedException | TimeoutException e) {
          throw new RuntimeException(e);
        } finally {
          cleanBuffers();
        }
      }
    }

    @Override
    public String toString() {
      return operations.toString();
    }

    public long getAllocatedBeforeError() {
      return allocatedBeforeError;
    }

    public TaskOpSet randomMod(Random r, double templateChangeAmount) {
      ArrayList<MemoryOp> newOps = new ArrayList<>(operations.size());
      for (MemoryOp op: operations) {
        newOps.add(op.randomMod(r, templateChangeAmount));
      }
      return new TaskOpSet(numBuffers, newOps);
    }

    public TaskOpSet makeSkewed(double skewAmount) {
      ArrayList<MemoryOp> newOps = new ArrayList<>(operations.size());
      for (MemoryOp op: operations) {
        newOps.add(op.makeSkewed(skewAmount));
      }
      return new TaskOpSet(numBuffers, newOps);
    }
  }

  public static class Task {
    static AtomicLong idGen = new AtomicLong(0);
    public final long taskId = idGen.getAndIncrement();
    public final int retryCount;
    LinkedList<TaskOpSet> toDo = new LinkedList<>();

    long timeLost = 0;

    public Task(Random r, long taskMaxMiB, int maxTaskAllocs, int maxTaskSleep) {
      toDo.add(new TaskOpSet(r, taskMaxMiB, maxTaskAllocs, maxTaskSleep));
      retryCount = 0;
    }

    private Task(LinkedList<TaskOpSet> toDo, int retryCount) {
      this.toDo.addAll(toDo);
      this.retryCount = retryCount;
    }

    public Task cloneForRetry() {
      LinkedList<TaskOpSet> cloned = (LinkedList<TaskOpSet>) toDo.clone();
      return new Task(cloned, retryCount + 1);
    }

    public long getTimeLost() {
      return timeLost;
    }

    public void run(ExecutorService shuffle) {
      Thread.currentThread().setName("TASK RUNNER FOR " + taskId);
      RmmSpark.currentThreadIsDedicatedToTask(taskId);
      try {
        RmmSpark.currentThreadStartRetryBlock();
        while (!toDo.isEmpty()) {
          TaskOpSet tos = toDo.pollFirst();
          try {
            tos.run(shuffle, taskId);
          } catch (GpuSplitAndRetryOOM soom) {
            TaskOpSet[] split = tos.split();
            toDo.push(split[1]);
            toDo.push(split[0]);
            numSplitAndRetry.incrementAndGet();
          } catch (CpuSplitAndRetryOOM soom) {
            TaskOpSet[] split = tos.split();
            toDo.push(split[1]);
            toDo.push(split[0]);
            numSplitAndRetry.incrementAndGet();
          }
        }
      } finally {
        RmmSpark.currentThreadEndRetryBlock();
        timeLost += RmmSpark.getAndResetComputeTimeLostToRetryNs(taskId);
        RmmSpark.taskDone(taskId);
      }
    }

    @Override
    public String toString() {
      return "task: " + toDo;
    }

    public Task randomMod(Random r, double templateChangeAmount) {
      LinkedList<TaskOpSet> changed = new LinkedList<>();
      for (TaskOpSet orig: toDo) {
        changed.add(orig.randomMod(r, templateChangeAmount));
      }
      return new Task(changed, 0);
    }

    public Task makeSkewed(double skewAmount) {
      LinkedList<TaskOpSet> changed = new LinkedList<>();
      for (TaskOpSet orig: toDo) {
        changed.add(orig.makeSkewed(skewAmount));
      }
      return new Task(changed, 0);
    }
  }

  public static class Situation {
    LinkedList<Task> tasks = new LinkedList<>();

    public Situation(Random r, long numTasks, long taskMaxMiB,
        int maxTaskAllocs, int maxTaskSleep,
        boolean isSkewed, double skewAmount, boolean useTemplate, double templateChangeAmount) {
      if (useTemplate) {
        Task template = new Task(r, taskMaxMiB, maxTaskAllocs, maxTaskSleep);
        tasks.add(template);
        for (int i = 1; i < numTasks; i++) {
          tasks.add(template.randomMod(r, templateChangeAmount));
        }
      } else {
        for (int i = 0; i < numTasks; i++) {
          tasks.add(new Task(r, taskMaxMiB, maxTaskAllocs, maxTaskSleep));
        }
      }

      if (isSkewed) {
        int skewIndex = r.nextInt(tasks.size());
        Task skewed = tasks.get(skewIndex).makeSkewed(skewAmount);
        tasks.set(skewIndex, skewed);
      }
    }

    public synchronized Task nextTask() {
      return tasks.pollFirst();
    }

    public synchronized void retryTask(Task t) {
      tasks.add(t);
    }

    @Override
    public String toString() {
      return "Sit: " + tasks.size();
    }
  }

  private static List<Situation> generateSituations(long seed, int numIterations, long numTasks,
      long taskMaxMiB, int maxTaskAllocs, int maxTaskSleep,
      boolean isSkewed, double skewAmount, boolean useTemplate, double templateChangeAmount) {
    ArrayList<Situation> ret = new ArrayList<>(numIterations);
    long start = System.nanoTime();
    System.out.println("Generating " + numIterations + " test situations...");

    Random r = new Random(seed);
    for (int i = 0; i < numIterations; i++) {
      ret.add(new Situation(r, numTasks, taskMaxMiB, maxTaskAllocs, maxTaskSleep,
          isSkewed, skewAmount, useTemplate, templateChangeAmount));
    }

    long end = System.nanoTime();
    long diff = TimeUnit.MILLISECONDS.convert(end - start, TimeUnit.NANOSECONDS);
    System.out.println("Took " + diff + " milliseconds to generate " + numIterations);
    return ret;
  }
}
