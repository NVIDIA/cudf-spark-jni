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

#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/detail/coalescing_free_list.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <unistd.h>

#include <algorithm>
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

  friend void get_property(pageable_memory_resource const&, cuda::mr::host_accessible) noexcept {}
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
   * Rounds up to default_cuda_malloc_host_alignment. Throws pageable_pool_exhausted if
   * no free block is large enough.
   */
  [[nodiscard]] void* allocate_sync(
    std::size_t bytes,
    [[maybe_unused]] std::size_t alignment = cuda::mr::default_cuda_malloc_host_alignment)
  {
    if (bytes == 0) { return nullptr; }
    bytes = align_up(bytes);
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
  void deallocate_sync(
    void* ptr,
    std::size_t bytes,
    [[maybe_unused]] std::size_t alignment = cuda::mr::default_cuda_malloc_host_alignment)
  {
    if (bytes == 0) { return; }
    std::lock_guard<std::mutex> lock(mtx_);
    free_list_.insert(rmm::mr::detail::block{static_cast<char*>(ptr), align_up(bytes), false});
  }

  bool operator==(pageable_pool_resource const& other) const noexcept { return this == &other; }

  friend void get_property(pageable_pool_resource const&, cuda::mr::host_accessible) noexcept {}

  std::size_t pool_size() const noexcept { return pool_size_; }

 private:
  static void pretouch_parallel(void* base, std::size_t bytes, int threads)
  {
    std::size_t const page_size = system_page_size();
    int const n                 = std::max(1, threads);
    std::vector<std::thread> ts;
    ts.reserve(n);
    std::size_t per = (bytes + n - 1) / static_cast<std::size_t>(n);
    per             = rmm::align_up(per, page_size);
    try {
      for (int i = 0; i < n; ++i) {
        std::size_t off = static_cast<std::size_t>(i) * per;
        if (off >= bytes) break;
        std::size_t end = std::min(off + per, bytes);
        ts.emplace_back([base, off, end, page_size]() {
          // volatile prevents the compiler from eliding these writes — the write's
          // only purpose is to fault in the page.
          auto* c = static_cast<volatile char*>(base);
          for (std::size_t k = off; k < end; k += page_size)
            c[k] = 0;
        });
      }
      for (auto& t : ts)
        t.join();
    } catch (...) {
      for (auto& t : ts) {
        if (t.joinable()) { t.join(); }
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
    constexpr std::size_t a = cuda::mr::default_cuda_malloc_host_alignment;
    return (bytes + a - 1) & ~(a - 1);
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
