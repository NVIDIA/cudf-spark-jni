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

#include <rmm/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/detail/coalescing_free_list.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace spark_rapids_jni {

inline bool is_power_of_two(std::size_t value) noexcept
{
  return value != 0 && (value & (value - 1)) == 0;
}

struct pageable_pool_exhausted : std::exception {
  [[nodiscard]] char const* what() const noexcept override { return "pageable pool exhausted"; }
};

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
// pageable_pool_resource
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
// ---------------------------------------------------------------------------
class pageable_pool_resource : public cuda::mr::memory_resource_base<pageable_pool_resource> {
 public:
  /**
   * @brief Construct the pool: allocate the backing buffer via upstream,
   *        pre-touch all pages in parallel, then register the entire region
   *        as a single free block.
   *
   * @param upstream       Host-accessible upstream resource for the backing buffer.
   * @param size           Total pool size in bytes.
   * @param pretouch_threads Number of threads used to fault in backing pages.
   * @throws rmm::out_of_memory if the backing allocation fails.
   */
  pageable_pool_resource(cuda::mr::any_synchronous_resource<cuda::mr::host_accessible> upstream,
                         std::size_t size,
                         int pretouch_threads)
    : upstream_(std::move(upstream)),
      pool_size_(size),
      base_(upstream_.allocate_sync(size, system_page_size()))
  {
    try {
      pretouch_parallel(base_, pool_size_, pretouch_threads);
      // is_head=false: all sub-blocks live within one contiguous upstream
      // allocation and may coalesce freely across their boundaries.
      free_list_.insert(rmm::mr::detail::block{static_cast<char*>(base_), pool_size_, false});
    } catch (...) {
      upstream_.deallocate_sync(base_, pool_size_, system_page_size());
      throw;
    }
  }

  ~pageable_pool_resource() { upstream_.deallocate_sync(base_, pool_size_, system_page_size()); }

  pageable_pool_resource(pageable_pool_resource const&)            = delete;
  pageable_pool_resource& operator=(pageable_pool_resource const&) = delete;
  pageable_pool_resource(pageable_pool_resource&&)                 = delete;
  pageable_pool_resource& operator=(pageable_pool_resource&&)      = delete;

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
    bytes = align_up(bytes);
    if (bytes == 0 || bytes > pool_size_) { throw pageable_pool_exhausted{}; }
    std::lock_guard<std::mutex> lock(mtx_);
    auto blk = free_list_.get_block(bytes);
    if (!blk.is_valid()) { throw pageable_pool_exhausted{}; }
    if (blk.size() > bytes) {
      free_list_.insert(rmm::mr::detail::block{blk.pointer() + bytes, blk.size() - bytes, false});
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
    free_list_.insert(rmm::mr::detail::block{static_cast<char*>(ptr), align_up(bytes), false});
  }

  bool operator==(pageable_pool_resource const& other) const noexcept { return this == &other; }

  friend void get_property(pageable_pool_resource const&, cuda::mr::host_accessible) noexcept
  {
    // The presence of this tag-only function declares host accessibility; there is no work to do.
  }

  std::size_t pool_size() const noexcept { return pool_size_; }

 private:
  static void pretouch_parallel(void* base, std::size_t bytes, int threads)
  {
    std::size_t const page_size  = system_page_size();
    std::size_t const page_count = bytes / page_size + (bytes % page_size != 0);
    if (page_count == 0) { return; }
    std::size_t const requested_threads = static_cast<std::size_t>(std::max(1, threads));
    std::size_t const hardware_threads  = std::thread::hardware_concurrency();
    std::size_t const worker_count =
      std::min({requested_threads,
                page_count,
                hardware_threads == 0 ? requested_threads : hardware_threads});
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
        workers.emplace_back([pages, first_page, worker_pages, page_size, &start_workers]() {
          // Finish creating thread stacks before concurrent page faults contend for mmap_lock.
          start_workers.wait(false);
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

  static std::size_t system_page_size()
  {
    static std::size_t const page_size = [] {
      long const size = ::sysconf(_SC_PAGESIZE);
      RMM_EXPECTS(size > 0, "sysconf(_SC_PAGESIZE) failed", rmm::logic_error);
      return static_cast<std::size_t>(size);
    }();
    return page_size;
  }

  static std::size_t align_up(std::size_t bytes) noexcept
  {
    return rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }

  cuda::mr::any_synchronous_resource<cuda::mr::host_accessible> upstream_;
  std::size_t pool_size_;
  void* base_;
  std::mutex mtx_;
  rmm::mr::detail::coalescing_free_list free_list_;
};

static_assert(
  cuda::mr::synchronous_resource_with<pageable_pool_resource, cuda::mr::host_accessible>);

}  // namespace spark_rapids_jni
