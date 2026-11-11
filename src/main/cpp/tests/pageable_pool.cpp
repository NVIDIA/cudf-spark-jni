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

#include "pageable_pool_resource.hpp"

#include <rmm/mr/detail/coalescing_free_list.hpp>

#include <gtest/gtest.h>
#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <vector>

using namespace spark_rapids_jni;

namespace {

constexpr std::size_t kPoolSize = 4 * 1024 * 1024;  // 4 MiB

pageable_pool_resource make_pool(std::size_t size = kPoolSize,
                                 int threads      = 1,
                                 bool use_indexed = true)
{
  // Explicit any_resource construction is required because the converting constructor
  // of any_resource is not implicit in this CCCL version.
  return pageable_pool_resource(
    cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>(pageable_memory_resource{}),
    size,
    threads,
    use_indexed);
}

testing::AssertionResult all_eq(unsigned char const* data,
                                std::size_t length,
                                unsigned char expected)
{
  for (std::size_t i = 0; i < length; ++i) {
    if (data[i] != expected) {
      return testing::AssertionFailure()
             << "mismatch at index " << i << " expected " << static_cast<int>(expected)
             << " actual " << static_cast<int>(data[i]);
    }
  }
  return testing::AssertionSuccess();
}

}  // namespace

// ---------------------------------------------------------------------------
// Basic alloc / dealloc
// ---------------------------------------------------------------------------
TEST(PageablePool, AllocReturnsNonNull)
{
  auto pool = make_pool();
  void* p   = pool.allocate_sync(1024);
  ASSERT_NE(p, nullptr);
  pool.deallocate_sync(p, 1024);
}

TEST(PageablePool, AllocFillAndFree)
{
  auto pool = make_pool();
  void* p   = pool.allocate_sync(1024);
  ASSERT_NE(p, nullptr);
  std::memset(p, 0xAB, 1024);
  auto* bytes = static_cast<unsigned char*>(p);
  EXPECT_TRUE(all_eq(bytes, 1024, 0xAB));
  pool.deallocate_sync(p, 1024);
}

TEST(PageablePool, ZeroByteAllocationDoesNotConsumePool)
{
  auto pool = make_pool();
  void* p   = pool.allocate_sync(0);
  EXPECT_EQ(p, nullptr);
  pool.deallocate_sync(p, 0);

  void* full = pool.allocate_sync(kPoolSize);
  ASSERT_NE(full, nullptr);
  pool.deallocate_sync(full, kPoolSize);
}

TEST(PageablePool, PretouchCoversPartialPageWithExcessThreads)
{
  auto const page_size = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
  auto const bytes     = page_size + rmm::CUDA_ALLOCATION_ALIGNMENT;
  auto pool            = make_pool(bytes, std::numeric_limits<int>::max());
  void* p              = pool.allocate_sync(bytes);
  ASSERT_NE(p, nullptr);
  for (std::size_t page = 0; page < 2; ++page) {
    unsigned char resident = 0;
    ASSERT_EQ(::mincore(static_cast<char*>(p) + page * page_size, page_size, &resident), 0);
    EXPECT_NE(resident & 1, 0) << "page " << page << " was not pre-touched";
  }
  pool.deallocate_sync(p, bytes);
}

TEST(PageablePool, OversizedAllocationDoesNotCorruptPool)
{
  auto pool = make_pool();
  EXPECT_THROW(
    { [[maybe_unused]] void* _ = pool.allocate_sync(SIZE_MAX); }, pageable_pool_exhausted);
  void* full = pool.allocate_sync(kPoolSize);
  ASSERT_NE(full, nullptr);
  pool.deallocate_sync(full, kPoolSize);
}

// ---------------------------------------------------------------------------
// Alignment
// ---------------------------------------------------------------------------
TEST(PageablePool, PointerIsAligned)
{
  auto pool                   = make_pool();
  constexpr std::size_t align = rmm::CUDA_ALLOCATION_ALIGNMENT;
  void* p                     = pool.allocate_sync(1);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % align, 0u);
  pool.deallocate_sync(p, 1);
}

// ---------------------------------------------------------------------------
// Coalescing: alloc two halves, free both, alloc a full-pool block
// ---------------------------------------------------------------------------
TEST(PageablePool, CoalescingAllowsFullReuse)
{
  auto pool = make_pool();
  void* a   = pool.allocate_sync(kPoolSize / 2);
  void* b   = pool.allocate_sync(kPoolSize / 2);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);

  // Free in reverse order to exercise coalescing in both directions.
  pool.deallocate_sync(b, kPoolSize / 2);
  pool.deallocate_sync(a, kPoolSize / 2);

  // After coalescing the two half-blocks should merge into one block large
  // enough for a near-full-pool allocation.
  constexpr std::size_t align = cuda::mr::default_cuda_malloc_host_alignment;
  void* c                     = pool.allocate_sync(kPoolSize - align);
  ASSERT_NE(c, nullptr);
  pool.deallocate_sync(c, kPoolSize - align);
}

TEST(PageablePool, IndexedFreeListPreservesBestFitAndCoalescing)
{
  // Both implementations must choose the same block when equally sized holes
  // exist, then recover a full span after adjacent blocks are returned.
  alignas(rmm::CUDA_ALLOCATION_ALIGNMENT) char backing[4096];
  rmm::mr::detail::coalescing_free_list original;
  detail::indexed_free_list indexed;
  auto insert_both = [&](std::size_t offset, std::size_t size) {
    rmm::mr::detail::block const b{backing + offset, size, false};
    original.insert(b);
    indexed.insert(b);
  };
  insert_both(0, 1024);
  insert_both(2048, 1024);
  auto const first_original = original.get_block(512);
  auto const first_indexed  = indexed.get_block(512);
  ASSERT_TRUE(first_original.is_valid());
  EXPECT_EQ(first_original.pointer(), first_indexed.pointer());
  auto const second_original = original.get_block(512);
  auto const second_indexed  = indexed.get_block(512);
  ASSERT_TRUE(second_original.is_valid());
  EXPECT_EQ(second_original.pointer(), second_indexed.pointer());
  insert_both(0, 1024);
  insert_both(1024, 1024);
  insert_both(2048, 1024);
  auto const merged_original = original.get_block(3072);
  auto const merged_indexed  = indexed.get_block(3072);
  ASSERT_TRUE(merged_original.is_valid());
  EXPECT_EQ(merged_original.pointer(), merged_indexed.pointer());
}

TEST(PageablePool, IndexedFreeListChoosesBestFitAmongUnequalHoles)
{
  alignas(rmm::CUDA_ALLOCATION_ALIGNMENT) char backing[8192];
  rmm::mr::detail::coalescing_free_list original;
  detail::indexed_free_list indexed;
  for (auto const& [offset, size] :
       {std::pair<std::size_t, std::size_t>{0, 512}, {1024, 2048}, {4096, 1024}}) {
    rmm::mr::detail::block const b{backing + offset, size, false};
    original.insert(b);
    indexed.insert(b);
  }

  auto const first_original = original.get_block(900);
  auto const first_indexed  = indexed.get_block(900);
  ASSERT_TRUE(first_original.is_valid());
  EXPECT_EQ(backing + 4096, first_original.pointer());
  EXPECT_EQ(first_original.pointer(), first_indexed.pointer());
  EXPECT_EQ(first_original.size(), first_indexed.size());

  auto const second_original = original.get_block(1500);
  auto const second_indexed  = indexed.get_block(1500);
  ASSERT_TRUE(second_original.is_valid());
  EXPECT_EQ(backing + 1024, second_original.pointer());
  EXPECT_EQ(second_original.pointer(), second_indexed.pointer());
  EXPECT_EQ(second_original.size(), second_indexed.size());
}

TEST(PageablePool, IndexedFreeListCoalescesBothNeighbors)
{
  alignas(rmm::CUDA_ALLOCATION_ALIGNMENT) char backing[4096];
  rmm::mr::detail::coalescing_free_list original;
  detail::indexed_free_list indexed;
  for (auto const offset : {std::size_t{0}, std::size_t{2048}, std::size_t{1024}}) {
    rmm::mr::detail::block const b{backing + offset, 1024, false};
    original.insert(b);
    indexed.insert(b);
  }

  auto const original_block = original.get_block(3072);
  auto const indexed_block  = indexed.get_block(3072);
  ASSERT_TRUE(original_block.is_valid());
  EXPECT_EQ(backing, original_block.pointer());
  EXPECT_EQ(original_block.pointer(), indexed_block.pointer());
  EXPECT_EQ(original_block.size(), indexed_block.size());
}

TEST(PageablePool, IndexedFreeListCoalescesEveryFourBlockOrder)
{
  alignas(rmm::CUDA_ALLOCATION_ALIGNMENT) char backing[4096];
  std::array<std::size_t, 4> order{0, 1, 2, 3};
  do {
    rmm::mr::detail::coalescing_free_list original;
    detail::indexed_free_list indexed;
    for (auto const index : order) {
      rmm::mr::detail::block const b{backing + index * 1024, 1024, false};
      original.insert(b);
      indexed.insert(b);
    }
    auto const expected = original.get_block(4096);
    auto const actual   = indexed.get_block(4096);
    ASSERT_TRUE(expected.is_valid());
    ASSERT_TRUE(actual.is_valid());
    EXPECT_EQ(expected.pointer(), actual.pointer());
    EXPECT_EQ(expected.size(), actual.size());
    EXPECT_TRUE(indexed.is_empty());
  } while (std::next_permutation(order.begin(), order.end()));
}

TEST(PageablePool, BothFreeListStrategiesRecoverFullPool)
{
  for (bool use_indexed : {true, false}) {
    auto pool    = make_pool(kPoolSize, 1, use_indexed);
    void* first  = pool.allocate_sync(kPoolSize / 4);
    void* second = pool.allocate_sync(kPoolSize / 4);
    void* third  = pool.allocate_sync(kPoolSize / 2);
    EXPECT_THROW(
      { [[maybe_unused]] void* ignored = pool.allocate_sync(1); }, pageable_pool_exhausted);
    pool.deallocate_sync(second, kPoolSize / 4);
    pool.deallocate_sync(first, kPoolSize / 4);
    pool.deallocate_sync(third, kPoolSize / 2);
    void* whole = pool.allocate_sync(kPoolSize);
    EXPECT_EQ(first, whole);
    pool.deallocate_sync(whole, kPoolSize);
  }
}

TEST(PageablePool, VariedAllocationsPreserveLiveDataAndRecoverPool)
{
  struct allocation {
    unsigned char* data;
    std::size_t size;
    unsigned char pattern;
  };
  for (bool use_indexed : {true, false}) {
    auto pool = make_pool(kPoolSize, 1, use_indexed);
    std::vector<allocation> live;
    std::mt19937 rng{0x5eed};
    for (int step = 0; step < 3000; ++step) {
      if (!live.empty() && (live.size() >= 96 || rng() % 3 == 0)) {
        auto const index = rng() % live.size();
        auto const a     = live[index];
        EXPECT_TRUE(all_eq(a.data, a.size, a.pattern)) << "step " << step;
        pool.deallocate_sync(a.data, a.size);
        live.erase(live.begin() + index);
      } else {
        auto const size = static_cast<std::size_t>(1 + rng() % 32768);
        auto* data      = static_cast<unsigned char*>(pool.allocate_sync(size));
        ASSERT_NE(data, nullptr);
        auto const begin = reinterpret_cast<std::uintptr_t>(data);
        auto const end   = begin + size;
        for (auto const& a : live) {
          auto const other_begin = reinterpret_cast<std::uintptr_t>(a.data);
          auto const other_end   = other_begin + a.size;
          EXPECT_TRUE(end <= other_begin || other_end <= begin) << "step " << step;
        }
        auto const pattern = static_cast<unsigned char>(step % 251 + 1);
        std::memset(data, pattern, size);
        live.push_back({data, size, pattern});
      }
      if (step % 97 == 0) {
        for (auto const& a : live) {
          EXPECT_TRUE(all_eq(a.data, a.size, a.pattern)) << "step " << step;
        }
      }
    }
    for (auto const& a : live) {
      EXPECT_TRUE(all_eq(a.data, a.size, a.pattern));
      pool.deallocate_sync(a.data, a.size);
    }
    void* whole = pool.allocate_sync(kPoolSize);
    ASSERT_NE(whole, nullptr);
    pool.deallocate_sync(whole, kPoolSize);
  }
}

// ---------------------------------------------------------------------------
// OOM: exhausting the pool throws pageable_pool_exhausted
// ---------------------------------------------------------------------------
TEST(PageablePool, OOMThrows)
{
  auto pool = make_pool();
  void* p   = pool.allocate_sync(kPoolSize);
  ASSERT_NE(p, nullptr);
  // Suppress -Werror=unused-result for the nodiscard alloc inside EXPECT_THROW.
  EXPECT_THROW({ [[maybe_unused]] void* _ = pool.allocate_sync(1); }, pageable_pool_exhausted);
  pool.deallocate_sync(p, kPoolSize);
}

// ---------------------------------------------------------------------------
// Multiple allocations sum to pool size
// ---------------------------------------------------------------------------
TEST(PageablePool, MultipleAllocsExhaustPool)
{
  auto pool                   = make_pool();
  constexpr std::size_t chunk = 256 * 1024;  // 256 KiB
  constexpr int n             = static_cast<int>(kPoolSize / chunk);

  std::vector<void*> ptrs(n);
  for (int i = 0; i < n; ++i) {
    ptrs[i] = pool.allocate_sync(chunk);
    ASSERT_NE(ptrs[i], nullptr) << "i=" << i;
  }
  EXPECT_THROW({ [[maybe_unused]] void* _ = pool.allocate_sync(chunk); }, pageable_pool_exhausted);
  for (int i = 0; i < n; ++i) {
    pool.deallocate_sync(ptrs[i], chunk);
  }
}

// ---------------------------------------------------------------------------
// Thread safety: concurrent allocs and frees must not corrupt the pool
// ---------------------------------------------------------------------------
TEST(PageablePool, ConcurrentAllocFree)
{
  auto pool = make_pool(16 * 1024 * 1024, 4);

  constexpr int kThreads       = 8;
  constexpr int kOpsPerThread  = 100;
  constexpr std::size_t kChunk = 4096;

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&pool]() {
      for (int i = 0; i < kOpsPerThread; ++i) {
        void* p = nullptr;
        try {
          p = pool.allocate_sync(kChunk);
        } catch (pageable_pool_exhausted const&) {
          continue;  // Pool transiently full — ok under contention.
        }
        std::memset(p, 0x55, kChunk);
        pool.deallocate_sync(p, kChunk);
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }

  // After all threads finish the full pool must be reclaimable.
  void* final_alloc = pool.allocate_sync(16 * 1024 * 1024);
  ASSERT_NE(final_alloc, nullptr);
  pool.deallocate_sync(final_alloc, 16 * 1024 * 1024);
}

// ---------------------------------------------------------------------------
// CCCL concept: static_asserts in the header verify this at compile time.
// The upstream resource is type-erased via any_resource<host_accessible>.
// ---------------------------------------------------------------------------
TEST(PageablePool, CCCLConcept)
{
  static_assert(
    cuda::mr::synchronous_resource_with<pageable_memory_resource, cuda::mr::host_accessible>);
  static_assert(
    cuda::mr::synchronous_resource_with<pageable_pool_resource, cuda::mr::host_accessible>);

  // Verify the upstream any_resource round-trips correctly: allocate from the
  // pool, which internally calls upstream_.allocate_sync for the backing buffer.
  auto pool = make_pool(1024 * 1024);
  void* p   = pool.allocate_sync(4096);
  ASSERT_NE(p, nullptr);
  pool.deallocate_sync(p, 4096);
}
