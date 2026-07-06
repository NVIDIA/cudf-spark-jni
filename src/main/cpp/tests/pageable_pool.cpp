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

#include <gtest/gtest.h>

#include <cstdint>
#include <thread>
#include <vector>

using namespace spark_rapids_jni;

namespace {

static constexpr std::size_t kPoolSize = 4 * 1024 * 1024;  // 4 MiB

static pageable_pool_resource make_pool(std::size_t size = kPoolSize, int threads = 1)
{
  // Explicit any_resource construction is required because the converting constructor
  // of any_resource is not implicit in this CCCL version.
  return pageable_pool_resource(
    cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>(pageable_memory_resource{}),
    size,
    threads);
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

// ---------------------------------------------------------------------------
// Alignment
// ---------------------------------------------------------------------------
TEST(PageablePool, PointerIsAligned)
{
  auto pool                   = make_pool();
  constexpr std::size_t align = cuda::mr::default_cuda_malloc_host_alignment;
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
