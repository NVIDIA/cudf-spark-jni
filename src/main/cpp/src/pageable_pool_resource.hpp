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

#pragma once

#include "indexed_free_list.hpp"
#include "numa_placement.hpp"

#include <rmm/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/detail/coalescing_free_list.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/cmath>
#include <cuda/memory_resource>

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

namespace spark_rapids_jni {

#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23  // kernel 5.14+; defined for older build headers
#endif

inline bool is_power_of_two(std::size_t value) noexcept
{
  return value != 0 && (value & (value - 1)) == 0;
}

struct pageable_pool_exhausted : std::exception {
  [[nodiscard]] char const* what() const noexcept override { return "pageable pool exhausted"; }
};

namespace detail {

// Returns the kernel page size (cached); throws rmm::logic_error on failure.
inline std::size_t system_page_size()
{
  static std::size_t const page_size = [] {
    long const size = ::sysconf(_SC_PAGESIZE);
    RMM_EXPECTS(size > 0, "sysconf(_SC_PAGESIZE) failed", rmm::logic_error);
    return static_cast<std::size_t>(size);
  }();
  return page_size;
}

// The pool's fixed allocation granularity (matches rmm::pool_memory_resource).
inline std::size_t pool_align_up(std::size_t bytes) noexcept
{
  return rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

// Faults in every page of [base, base + round-up-to-page bytes) with up to `threads`
// worker threads. Workers bind to the NUMA node's CPUs when bound; page faults are
// served via madvise(MADV_POPULATE_WRITE) when available (kernel 5.14+) and via a
// manual per-page write otherwise.
inline void pretouch_parallel(
  void* base, std::size_t bytes, int threads, int numa_node, bool populate_write)
{
  std::size_t const page_size  = system_page_size();
  std::size_t const page_count = bytes / page_size + (bytes % page_size != 0);
  if (page_count == 0) { return; }
  std::size_t const requested_threads = static_cast<std::size_t>(std::max(1, threads));
  std::size_t const hardware_threads  = std::thread::hardware_concurrency();
  std::size_t const worker_count      = std::min(
    {requested_threads, page_count, hardware_threads == 0 ? requested_threads : hardware_threads});
  // Resolve the node's CPU list once so pre-touch faults (and thus page placement
  // latency) happen on CPUs local to the bound node.
  bool bind_cpus = false;
  cpu_set_t cpus{};
  if (numa_node >= 0) { bind_cpus = node_cpu_set(numa_node, &cpus); }
  // -1/1/0 = unsupported/known-good/unknown; one sticky probe avoids syscall spam.
  static std::atomic<int> populate_state{0};
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  std::atomic<bool> start_workers{false};
  auto* const pages     = static_cast<char volatile*>(base);
  std::size_t next_page = 0;
  try {
    for (std::size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
      std::size_t const worker_pages =
        page_count / worker_count + (worker_index < page_count % worker_count ? 1 : 0);
      std::size_t const first_page = next_page;
      next_page += worker_pages;
      workers.emplace_back([pages,
                            first_page,
                            worker_pages,
                            page_size,
                            &start_workers,
                            bind_cpus,
                            cpus,
                            populate_write]() {
        if (bind_cpus) {
          // Best-effort: sched_setaffinity fails harmlessly when the node's CPUs
          // fall outside this process's cpuset (e.g. container restrictions).
          ::sched_setaffinity(0, sizeof(cpu_set_t), &cpus);
        }
        // Finish creating thread stacks before concurrent page faults contend for mmap_lock.
        start_workers.wait(false);
        auto const chunk     = pages + first_page * page_size;
        auto const chunk_len = worker_pages * page_size;
        if (populate_write) {
          int const state = populate_state.load(std::memory_order_relaxed);
          if (state == 0) {
            if (::madvise(const_cast<char*>(chunk), chunk_len, MADV_POPULATE_WRITE) == 0) {
              populate_state.store(1, std::memory_order_relaxed);
              return;
            }
            populate_state.store(-1, std::memory_order_relaxed);  // old kernel: fall back
          } else if (state == 1) {
            if (::madvise(const_cast<char*>(chunk), chunk_len, MADV_POPULATE_WRITE) == 0) {
              return;
            }
          }
        }
        for (std::size_t page = 0; page < worker_pages; ++page) {
          pages[(first_page + page) * page_size] = 0;
        }
      });
    }
    start_workers.store(true);
    start_workers.notify_all();
    for (auto& worker : workers)
      worker.join();
  } catch (...) {
    start_workers.store(true);
    start_workers.notify_all();
    for (auto& worker : workers) {
      if (worker.joinable()) { worker.join(); }
    }
    throw;
  }
}

/**
 * @brief Allocates a page-aligned backing buffer from the upstream and prepares it:
 *        MADV_NOHUGEPAGE (optional), mbind to the GPU-local NUMA node (optional),
 *        parallel pre-touch. Exception-safe: the backing is released before rethrowing.
 */
inline void* prepare_backing_region(
  cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>& upstream,
  std::size_t size,
  int pretouch_threads,
  int numa_node,
  bool populate_write,
  bool thp_deny,
  bool memset_pretouch)
{
  void* const base = upstream.allocate_sync(size, system_page_size());
  try {
    // Deny THP BEFORE the pre-touch faults pages in, so no huge page is promoted.
    if (thp_deny) { ::madvise(base, size, MADV_NOHUGEPAGE); }
    // Bind BEFORE the pre-touch faults the pages in, so placement follows the policy.
    if (numa_node >= 0) { bind_memory_to_numa_node(base, size, numa_node); }
    if (memset_pretouch) {
      // Compare a single contiguous write with the page-at-a-time strategies.
      std::memset(base, 0, size);
    } else {
      pretouch_parallel(base, size, pretouch_threads, numa_node, populate_write);
    }
  } catch (...) {
    upstream.deallocate_sync(base, size, system_page_size());
    throw;
  }
  return base;
}

inline void release_backing_region(
  cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>& upstream,
  void* base,
  std::size_t size)
{
  upstream.deallocate_sync(base, size, system_page_size());
}

}  // namespace detail

// ---------------------------------------------------------------------------
// pageable_memory_resource
//
// Thin CCCL-conforming upstream resource that allocates pageable host memory
// via posix_memalign / free. Satisfies cuda::mr::synchronous_resource_with
// <cuda::mr::host_accessible>.
// ---------------------------------------------------------------------------
struct pageable_memory_resource : cuda::mr::memory_resource_base<pageable_memory_resource> {
  [[nodiscard]] void* allocate_sync(
    std::size_t bytes, std::size_t alignment = cuda::mr::default_cuda_malloc_host_alignment)
  {
    if (bytes == 0) { return nullptr; }
    void* ptr = nullptr;
    // posix_memalign requires alignment to be a power of two and a multiple of sizeof(void*).
    std::size_t const a = std::max(alignment, sizeof(void*));
    RMM_EXPECTS(is_power_of_two(a), "alignment must be a power of two", rmm::logic_error);
    if (::posix_memalign(&ptr, a, bytes) != 0 || ptr == nullptr) {
      RMM_FAIL("posix_memalign failed", rmm::out_of_memory);
    }
    return ptr;
  }

  void deallocate_sync(
    void* ptr,
    [[maybe_unused]] std::size_t bytes,
    [[maybe_unused]] std::size_t alignment = cuda::mr::default_cuda_malloc_host_alignment) noexcept
  {
    ::free(ptr);
  }

  bool operator==(pageable_memory_resource const&) const noexcept { return true; }

  friend void get_property(pageable_memory_resource const&, cuda::mr::host_accessible) noexcept
  {
    // The presence of this tag-only function declares host accessibility; there is no work to do.
  }
};

static_assert(
  cuda::mr::synchronous_resource_with<pageable_memory_resource, cuda::mr::host_accessible>);

// ---------------------------------------------------------------------------
// pageable_pool_resource_t
//
// Coalescing best-fit suballocator over a single backing buffer allocated from
// an any_resource<host_accessible> upstream. Satisfies
// cuda::mr::synchronous_resource_with<cuda::mr::host_accessible>.
//
// No stream ordering: pageable memory does not support async operations —
// cudaMemcpyAsync with pageable memory is synchronous on the CPU side.
//
// Pre-touching all backing pages at construction time amortizes first-touch
// page-fault cost, so DtoH copies into pool-allocated buffers reach near-pinned
// bandwidth rather than incurring per-page faults at copy time.
//
// Fragmentation: known limitation inherited from the underlying coalescing free
// list — same behavior as rmm::pool_memory_resource. Unlike the pinned pool, this
// pool is not growable; supporting growth would require multiple backing buffers
// plus per-buffer lifetime, pretouch, and coalescing-boundary bookkeeping.
//
// Thread safety: allocate_sync/deallocate_sync are thread-safe (internally
// mutex-guarded). The constructor performs internal multi-threaded pre-touch.
//
// FreeList: rmm::mr::detail::coalescing_free_list (std::list, O(F) best-fit/insert) or
// detail::indexed_free_list ((size,address)+(address) ordered indexes, O(log F)) —
// identical allocation policy, different lookup cost.
// ---------------------------------------------------------------------------
template <typename FreeList = rmm::mr::detail::coalescing_free_list>
class pageable_pool_resource_t
  : public cuda::mr::memory_resource_base<pageable_pool_resource_t<FreeList>> {
 public:
  /**
   * @brief Construct the pool: allocate the backing buffer via upstream,
   *        pre-touch all pages in parallel, then register the entire region
   *        as a single free block.
   *
   * @param upstream       Host-accessible upstream resource for the backing buffer.
   * @param size           Total pool size in bytes.
   * @param pretouch_threads Number of threads used to fault in backing pages.
   * @param numa_node      NUMA node to place the backing buffer on (-1 disables
   *                       binding; binding is best-effort and silently skipped on
   *                       failure). The intended value is the GPU-local node so
   *                       DtoH copies avoid cross-socket traffic.
   * @param use_remainder_cache  When true, the tail remainder of a split is served
   *                       directly to the next allocation instead of being re-inserted
   *                       into the free list (skips the O(F) best-fit scan on the hot
   *                       path; slightly deviates from strict best-fit order).
   * @param populate_write Use madvise(MADV_POPULATE_WRITE) to fault pages in (kernel
   *                       5.14+; falls back to the manual byte-touch loop otherwise).
   *                       Faulting via madvise amortizes syscall overhead vs one trap
   *                       per page; placement follows the same mbind policy either way.
   * @param thp_deny       When true, madvise(MADV_NOHUGEPAGE) the backing buffer so
   *                       transparent huge pages are not promoted during the pool's
   *                       lifetime (THP promotion was observed to slow repeated DtoH).
   * @param memset_pretouch When true, pre-touch with one memset of the full backing
   *                       buffer instead of the parallel page-at-a-time methods.
   * @throws rmm::out_of_memory if the backing allocation fails.
   */
  pageable_pool_resource_t(cuda::mr::any_synchronous_resource<cuda::mr::host_accessible> upstream,
                           std::size_t size,
                           int pretouch_threads,
                           int numa_node            = -1,
                           bool use_remainder_cache = false,
                           bool populate_write      = true,
                           bool thp_deny            = false,
                           bool memset_pretouch     = false)
    : upstream_(std::move(upstream)),
      pool_size_(size),
      base_(detail::prepare_backing_region(
        upstream_, size, pretouch_threads, numa_node, populate_write, thp_deny, memset_pretouch)),
      use_remainder_cache_(use_remainder_cache)
  {
    try {
      // is_head=false: all sub-blocks live within one contiguous upstream
      // allocation and may coalesce freely across their boundaries.
      free_list_.insert(rmm::mr::detail::block{static_cast<char*>(base_), pool_size_, false});
    } catch (...) {
      detail::release_backing_region(upstream_, base_, pool_size_);
      throw;
    }
  }

  ~pageable_pool_resource_t() { detail::release_backing_region(upstream_, base_, pool_size_); }

  pageable_pool_resource_t(pageable_pool_resource_t const&)            = delete;
  pageable_pool_resource_t& operator=(pageable_pool_resource_t const&) = delete;
  pageable_pool_resource_t(pageable_pool_resource_t&&)                 = delete;
  pageable_pool_resource_t& operator=(pageable_pool_resource_t&&)      = delete;

  /**
   * @brief Allocate @p bytes from the pool (best-fit).
   *
   * Like rmm::mr::pool_memory_resource, this pool uses a fixed allocation
   * granularity and does not honor larger per-call alignment requests. Returned
   * pointers are at least rmm::CUDA_ALLOCATION_ALIGNMENT aligned.
   *
   * Throws pageable_pool_exhausted if no free block is large enough.
   */
  [[nodiscard]] void* allocate_sync(std::size_t bytes,
                                    std::size_t = cuda::mr::default_cuda_malloc_host_alignment)
  {
    if (bytes == 0) { return nullptr; }
    if (bytes > pool_size_) { throw pageable_pool_exhausted{}; }
    bytes = detail::pool_align_up(bytes);
    if (bytes == 0 || bytes > pool_size_) { throw pageable_pool_exhausted{}; }
    std::lock_guard<std::mutex> lock(mtx_);
    if (use_remainder_cache_) {
      // The cached split tail is the most-recently carved free range; serving it
      // directly skips the O(F) best-fit scan. Flush it back to the free list when it
      // cannot serve the request so best-fit still sees it.
      if (cached_remainder_.pointer() != nullptr && cached_remainder_.size() >= bytes) {
        void* cached = cached_remainder_.pointer();
        // An exact-fit serve leaves nothing cached — never store a zero-size block.
        auto const leftover = cached_remainder_.size() - bytes;
        cached_remainder_ =
          leftover > 0 ? rmm::mr::detail::block{static_cast<char*>(cached) + bytes, leftover, false}
                       : rmm::mr::detail::block{};
        return cached;
      }
      if (cached_remainder_.pointer() != nullptr && cached_remainder_.size() > 0) {
        free_list_.insert(cached_remainder_);
        cached_remainder_ = rmm::mr::detail::block{};
      }
    }
    auto blk = free_list_.get_block(bytes);
    if (!blk.is_valid()) { throw pageable_pool_exhausted{}; }
    if (blk.size() > bytes) {
      rmm::mr::detail::block const remainder{blk.pointer() + bytes, blk.size() - bytes, false};
      // Only cache small tails: a large remainder stays address-ordered in the free
      // list so later large allocations can still find and coalesce it.
      if (use_remainder_cache_ && remainder.size() <= kRemainderCacheMax) {
        cached_remainder_ = remainder;
      } else {
        free_list_.insert(remainder);
      }
    }
    return blk.pointer();
  }

  /**
   * @brief Return a previously-allocated block to the pool, coalescing with
   *        adjacent free blocks.
   */
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t = cuda::mr::default_cuda_malloc_host_alignment)
  {
    if (bytes == 0) { return; }
    std::lock_guard<std::mutex> lock(mtx_);
    if (use_remainder_cache_ && cached_remainder_.pointer() != nullptr) {
      // Return the cached tail to the free list BEFORE coalescing the freed block, so
      // a free adjacent to the cached tail still merges correctly.
      free_list_.insert(cached_remainder_);
      cached_remainder_ = rmm::mr::detail::block{};
    }
    free_list_.insert(
      rmm::mr::detail::block{static_cast<char*>(ptr), detail::pool_align_up(bytes), false});
  }

  bool operator==(pageable_pool_resource_t const& other) const noexcept { return this == &other; }

  friend void get_property(pageable_pool_resource_t const&, cuda::mr::host_accessible) noexcept {}

  std::size_t pool_size() const noexcept { return pool_size_; }

 private:
  cuda::mr::any_synchronous_resource<cuda::mr::host_accessible> upstream_;
  std::size_t pool_size_;
  void* base_;
  std::mutex mtx_;
  FreeList free_list_;
  // Remainder cache: the most recently carved split tail, held OUT of the free
  // list so the next allocation can reuse it without an O(F) scan. Not address-stable
  // across frees — deallocate_sync flushes it before coalescing.
  static constexpr std::size_t kRemainderCacheMax = 4 * 1024 * 1024;
  rmm::mr::detail::block cached_remainder_{};
  bool use_remainder_cache_;
};

static_assert(cuda::mr::synchronous_resource_with<
              pageable_pool_resource_t<rmm::mr::detail::coalescing_free_list>,
              cuda::mr::host_accessible>);
static_assert(
  cuda::mr::synchronous_resource_with<pageable_pool_resource_t<detail::indexed_free_list>,
                                      cuda::mr::host_accessible>);

// Default pool: the PR's coalescing free list (policy-identical to rmm::pool_memory_resource).
using pageable_pool_resource = pageable_pool_resource_t<rmm::mr::detail::coalescing_free_list>;

}  // namespace spark_rapids_jni
