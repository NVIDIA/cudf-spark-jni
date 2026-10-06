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

#include "pageable_pool_resource.hpp"

#include <cuda/memory_resource>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace spark_rapids_jni {

// ---------------------------------------------------------------------------
// pageable_arena_pool_resource_t
//
// Host-only arena pool: K contiguous equal-capacity arenas carved from ONE
// pre-touched backing buffer, each with its own free list + mutex. Allocation
// round-robins across arenas (spreading concurrent threads over disjoint locks);
// frees route to the owning arena by address.
//
// POLICY CHANGE vs pageable_pool_resource (this is the trade under test): capacity
// is per-arena — a request larger than any single arena's free space fails with
// pageable_pool_exhausted even when the pool has aggregate headroom, and free
// blocks never coalesce across arena boundaries (is_head marks every arena start).
//
// Thread safety: allocate_sync/deallocate_sync are thread-safe; each arena has its
// own mutex so distinct arenas operate fully in parallel.
// ---------------------------------------------------------------------------
template <typename FreeList = rmm::mr::detail::coalescing_free_list>
class pageable_arena_pool_resource_t
  : public cuda::mr::memory_resource_base<pageable_arena_pool_resource_t<FreeList>> {
 public:
  pageable_arena_pool_resource_t(
    cuda::mr::any_synchronous_resource<cuda::mr::host_accessible> upstream,
    std::size_t size,
    int pretouch_threads,
    int numa_node       = -1,
    bool populate_write = true,
    bool thp_deny       = false,
    int arena_count     = 8)
    : upstream_(std::move(upstream)),
      pool_size_(size),
      base_(detail::prepare_backing_region(
        upstream_, size, pretouch_threads, numa_node, populate_write, thp_deny))
  {
    try {
      // Each arena needs at least one aligned-granularity unit of capacity.
      std::size_t const min_cap = detail::pool_align_up(1);
      auto count                = static_cast<std::size_t>(std::max(1, arena_count));
      if (count > pool_size_ / min_cap) { count = std::max<std::size_t>(1, pool_size_ / min_cap); }
      arena_count_    = count;
      arena_capacity_ = detail::pool_align_up(pool_size_ / arena_count_);
      if (arena_capacity_ == 0) {
        arena_capacity_ = pool_size_;  // single-arena degenerate case
        arena_count_    = 1;
      }
      arenas_.reserve(arena_count_);
      for (std::size_t i = 0; i < arena_count_; ++i) {
        arenas_.emplace_back();
        char* const arena_begin = static_cast<char*>(base_) + i * arena_capacity_;
        std::size_t const cap =
          (i == arena_count_ - 1) ? (pool_size_ - i * arena_capacity_) : arena_capacity_;
        // is_head on every arena start (except the pool head) blocks coalescing
        // across the boundary — block::is_contiguous_before refuses head blocks.
        arenas_[i].list.insert(rmm::mr::detail::block{arena_begin, cap, /*is_head=*/i > 0});
      }
    } catch (...) {
      detail::release_backing_region(upstream_, base_, pool_size_);
      throw;
    }
  }

  ~pageable_arena_pool_resource_t()
  {
    detail::release_backing_region(upstream_, base_, pool_size_);
  }

  pageable_arena_pool_resource_t(pageable_arena_pool_resource_t const&)            = delete;
  pageable_arena_pool_resource_t& operator=(pageable_arena_pool_resource_t const&) = delete;
  pageable_arena_pool_resource_t(pageable_arena_pool_resource_t&&)                 = delete;
  pageable_arena_pool_resource_t& operator=(pageable_arena_pool_resource_t&&)      = delete;

  [[nodiscard]] void* allocate_sync(std::size_t bytes,
                                    std::size_t = cuda::mr::default_cuda_malloc_host_alignment)
  {
    if (bytes == 0) { return nullptr; }
    bytes = detail::pool_align_up(bytes);
    // Largest arena capacity: every arena has arena_capacity_ except the last, which
    // holds the remainder (>= arena_capacity_ when pool_size_ % arena_count_ != 0).
    std::size_t const last_cap = pool_size_ - (arena_count_ - 1) * arena_capacity_;
    if (bytes > last_cap) { throw pageable_pool_exhausted{}; }
    std::size_t const start = next_arena_.fetch_add(1) % arena_count_;
    for (std::size_t k = 0; k < arena_count_; ++k) {
      auto const idx = (start + k) % arena_count_;
      auto& arena    = arenas_[idx];
      std::lock_guard<std::mutex> lock(arena.mtx);
      auto blk = arena.list.get_block(bytes);
      if (!blk.is_valid()) { continue; }
      if (blk.size() > bytes) {
        // Split remainder stays inside the owning arena (interior block).
        arena.list.insert(rmm::mr::detail::block{blk.pointer() + bytes, blk.size() - bytes, false});
      }
      return blk.pointer();
    }
    throw pageable_pool_exhausted{};
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t = cuda::mr::default_cuda_malloc_host_alignment)
  {
    if (bytes == 0) { return; }
    auto const pos        = static_cast<char*>(ptr) - static_cast<char*>(base_);
    std::size_t const idx = std::min(pos / arena_capacity_, arena_count_ - 1);
    // A block starting exactly at an arena boundary (past the pool head) is that
    // arena's head: is_head prevents coalescing into the previous arena.
    bool const is_boundary_head =
      (idx > 0) && (static_cast<char*>(ptr) == static_cast<char*>(base_) + idx * arena_capacity_);
    std::lock_guard<std::mutex> lock(arenas_[idx].mtx);
    arenas_[idx].list.insert(rmm::mr::detail::block{
      static_cast<char*>(ptr), detail::pool_align_up(bytes), is_boundary_head});
  }

  bool operator==(pageable_arena_pool_resource_t const& other) const noexcept
  {
    return this == &other;
  }

  friend void get_property(pageable_arena_pool_resource_t const&,
                           cuda::mr::host_accessible) noexcept
  {
  }

  std::size_t pool_size() const noexcept { return pool_size_; }

 private:
  struct arena {
    FreeList list;
    std::mutex mtx;
  };

  cuda::mr::any_synchronous_resource<cuda::mr::host_accessible> upstream_;
  std::size_t pool_size_;
  void* base_;
  std::atomic<std::uint64_t> next_arena_{0};
  std::size_t arena_count_{1};
  std::size_t arena_capacity_{0};
  std::vector<arena> arenas_;
};

static_assert(cuda::mr::synchronous_resource_with<
              pageable_arena_pool_resource_t<rmm::mr::detail::coalescing_free_list>,
              cuda::mr::host_accessible>);
static_assert(
  cuda::mr::synchronous_resource_with<pageable_arena_pool_resource_t<detail::indexed_free_list>,
                                      cuda::mr::host_accessible>);

using pageable_arena_pool_resource = pageable_arena_pool_resource_t<detail::indexed_free_list>;

}  // namespace spark_rapids_jni
