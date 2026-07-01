/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * JNI interface to a rmm::pool_memory_resource backed by a host memory resource that
 * pre-touches each allocated page in parallel. Intended as a fallback when the pinned
 * pool is exhausted: a pre-touched pageable destination reaches a much higher fraction
 * of pinned DtoH bandwidth than freshly-malloc'd pageable, because the page faults are
 * paid at pool growth time rather than at DtoH copy time.
 */
public final class PageableMemoryPool implements AutoCloseable {
  private static final Logger log = LoggerFactory.getLogger(PageableMemoryPool.class);

  // These static fields should only ever be accessed when class-synchronized.
  // Do NOT use singleton_ directly!  Use the getSingleton accessor instead.
  private static volatile PageableMemoryPool singleton_ = null;
  private static Future<PageableMemoryPool> initFuture = null;
  private long poolHandle;
  private long poolSize;

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static native long newPageablePoolMemoryResource(long initSize, long maxSize,
                                                           int pretouchThreads);

  private static native void releasePageablePoolMemoryResource(long poolPtr);

  private static native long allocFromPageablePool(long poolPtr, long size);

  private static native void freeFromPageablePool(long poolPtr, long ptr, long size);

  private static final class PageableHostBufferCleaner extends MemoryBuffer.MemoryBufferCleaner {
    private long address;
    private final long origLength;

    PageableHostBufferCleaner(long address, long length) {
      this.address = address;
      this.origLength = length;
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      long origAddress = 0;
      if (address != -1) {
        origAddress = address;
        try {
          PageableMemoryPool.freeInternal(address, origLength);
        } finally {
          // Always mark the resource as freed even if an exception is thrown.
          // We cannot know how far it progressed before the exception, and
          // therefore it is unsafe to retry.
          address = -1;
        }
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("A PAGEABLE HOST BUFFER WAS LEAKED (ID: " + id + " "
            + Long.toHexString(origAddress) + ")");
        logRefCountDebug("Leaked pageable host buffer");
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return address == -1;
    }
  }

  private static PageableMemoryPool getSingleton() {
    if (singleton_ == null) {
      if (initFuture == null) {
        return null;
      }
      synchronized (PageableMemoryPool.class) {
        if (singleton_ == null) {
          try {
            singleton_ = initFuture.get();
          } catch (Exception e) {
            throw new RuntimeException("Error initializing pageable memory pool", e);
          }
          initFuture = null;
        }
      }
    }
    return singleton_;
  }

  private static void freeInternal(long address, long origLength) {
    Objects.requireNonNull(getSingleton()).free(address, origLength);
  }

  /**
   * Initialize the pool. The backing buffer is allocated and pre-touched (each system page
   * written to force the kernel to map physical pages) using the given thread count.
   * Pre-touching parallelizes the page-fault work so it amortizes to a few hundred ms
   * for multi-GB pools instead of several seconds serially.
   *
   * @param poolSize        size of the pool in bytes
   * @param pretouchThreads number of worker threads to use for the parallel pre-touch
   */
  public static synchronized void initialize(long poolSize, int pretouchThreads) {
    if (isInitialized()) {
      throw new IllegalStateException("Can only initialize the pageable pool once.");
    }
    ExecutorService initService = Executors.newSingleThreadExecutor(runnable -> {
      Thread t = new Thread(runnable, "pageable pool init");
      t.setDaemon(true);
      return t;
    });
    initFuture = initService.submit(() -> new PageableMemoryPool(poolSize, pretouchThreads));
    initService.shutdown();
  }

  /**
   * Check if the pool has been initialized or not.
   */
  public static boolean isInitialized() {
    return getSingleton() != null;
  }

  /**
   * Shut down the pool, nulling out our reference. Any allocation or free in flight
   * will fail after this.
   */
  public static synchronized void shutdown() {
    PageableMemoryPool pool = getSingleton();
    if (pool != null) {
      pool.close();
    }
    initFuture = null;
    singleton_ = null;
  }

  /**
   * Factory method to create a pageable host memory buffer.
   *
   * @param bytes size in bytes to allocate
   * @return newly created buffer, or null if the pool is uninitialized or exhausted
   *         (caller should fall back to a regular malloc'd buffer)
   */
  public static HostMemoryBuffer tryAllocate(long bytes) {
    HostMemoryBuffer result = null;
    PageableMemoryPool pool = getSingleton();
    if (pool != null) {
      result = pool.tryAllocateInternal(bytes);
    }
    return result;
  }

  /**
   * Get the number of bytes that the pageable memory pool was allocated with.
   */
  public static long getTotalPoolSizeBytes() {
    PageableMemoryPool pool = getSingleton();
    if (pool != null) {
      return pool.poolSize;
    }
    return 0;
  }

  private PageableMemoryPool(long poolSize, int pretouchThreads) {
    this.poolHandle = newPageablePoolMemoryResource(poolSize, poolSize, pretouchThreads);
    this.poolSize = poolSize;
  }

  @Override
  public void close() {
    releasePageablePoolMemoryResource(this.poolHandle);
    this.poolHandle = -1;
  }

  /**
   * Attempts to allocate from the pageable pool. Returns null rather than throwing if
   * the pool is exhausted, so callers can fall back gracefully.
   */
  private synchronized HostMemoryBuffer tryAllocateInternal(long bytes) {
    long allocated = allocFromPageablePool(this.poolHandle, bytes);
    if (allocated == -1) {
      return null;
    } else {
      return new HostMemoryBuffer(allocated, bytes,
              new PageableHostBufferCleaner(allocated, bytes));
    }
  }

  private synchronized void free(long address, long size) {
    freeFromPageablePool(this.poolHandle, address, size);
  }
}
