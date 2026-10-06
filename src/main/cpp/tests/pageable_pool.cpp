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

#include "pageable_pool_resource.hpp"

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

using namespace spark_rapids_jni;

namespace {

constexpr std::size_t kPoolSize = 4 * 1024 * 1024;  // 4 MiB

[[nodiscard]] pageable_pool_resource make_pool(std::size_t size = kPoolSize,
                                               int threads      = 1,
                                               int numa_node    = -1)
{
  // Explicit any_resource construction is required because the converting constructor
  // of any_resource is not implicit in this CCCL version.
  return pageable_pool_resource(
    cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>(pageable_memory_resource{}),
    size,
    threads,
    numa_node);
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
// NUMA: binding the backing buffer places pool pages on the requested node
// ---------------------------------------------------------------------------
TEST(PageablePool, NumaBindPlacesPagesOnRequestedNode)
{
  // The binding is only observable on hosts with more than one NUMA node.
  if (::access("/sys/devices/system/node/node1/cpulist", F_OK) != 0) {
    GTEST_SKIP() << "host has no second NUMA node";
  }
  // Probe that mbind is permitted here (containers may restrict memory policy).
  void* probe = nullptr;
  ASSERT_EQ(::posix_memalign(&probe, 4'096, 4'096), 0);
  bool const can_bind = detail::bind_memory_to_numa_node(probe, 4'096, 0);
  ::free(probe);
  if (!can_bind) { GTEST_SKIP() << "mbind unavailable or restricted"; }

  // Bind to a node DIFFERENT from the current thread's node, so a green test proves
  // the pages moved because of the policy, not because of where the process runs.
  void* current_probe = nullptr;
  ASSERT_EQ(::posix_memalign(&current_probe, 4'096, 4'096), 0);
  std::memset(current_probe, 0, 4'096);  // first-touch
  int const current_node = detail::numa_node_of(current_probe);
  ::free(current_probe);
  ASSERT_GE(current_node, 0) << "could not resolve current NUMA node";
  int const target_node = (current_node == 0) ? 1 : 0;

  auto pool = make_pool(kPoolSize, 1, target_node);
  // Check pages spread across the pool: first, middle, last.
  void* p = pool.allocate_sync(kPoolSize);
  ASSERT_NE(p, nullptr);
  auto* base = static_cast<unsigned char*>(p);
  EXPECT_EQ(detail::numa_node_of(base), target_node);
  EXPECT_EQ(detail::numa_node_of(base + kPoolSize / 2), target_node);
  EXPECT_EQ(detail::numa_node_of(base + kPoolSize - 4'096), target_node);
  pool.deallocate_sync(p, kPoolSize);
}

TEST(PageablePool, NumaHelperArgumentGuards)
{
  auto pool = make_pool();
  void* p   = pool.allocate_sync(1024);
  ASSERT_NE(p, nullptr);
  EXPECT_FALSE(detail::bind_memory_to_numa_node(nullptr, 4'096, 0));  // null base
  EXPECT_FALSE(detail::bind_memory_to_numa_node(p, 0, 0));            // zero bytes
  EXPECT_FALSE(detail::bind_memory_to_numa_node(p, 1024, -1));        // negative node
  EXPECT_FALSE(detail::bind_memory_to_numa_node(p, 1024, 64));        // out of nodemask range
  pool.deallocate_sync(p, 1024);

  cpu_set_t cpus{};
  EXPECT_FALSE(detail::node_cpu_set(-1, &cpus));
}

TEST(PageablePool, GpuNumaNodeDegradesCleanly)
{
  // Whatever the environment (GPU-less sandbox, container without sysfs), the
  // helper must return -1 (unknown) instead of throwing or leaving a pending CUDA error.
  int const node = detail::gpu_numa_node();
  if (node == -1) { EXPECT_EQ(cudaGetLastError(), cudaSuccess); }
}

TEST(PageablePool, ParseCpuList)
{
  char const* tmpdir     = std::getenv("TMPDIR");
  std::string const tmp  = (tmpdir != nullptr && tmpdir[0] != '\0') ? tmpdir : "/tmp";
  std::string const path = tmp + "/pageable_pool_cpulist_test.txt";
  {
    std::ofstream out(path);
    out << "0-3,8\n9\nfoo\n1048\n";  // ranges, single, malformed, above CPU_SETSIZE
  }
  cpu_set_t cpus{};
  ASSERT_TRUE(detail::parse_cpu_list(path.c_str(), &cpus));
  EXPECT_TRUE(CPU_ISSET(0, &cpus));
  EXPECT_TRUE(CPU_ISSET(3, &cpus));
  EXPECT_TRUE(CPU_ISSET(8, &cpus));
  EXPECT_TRUE(CPU_ISSET(9, &cpus));
  EXPECT_FALSE(CPU_ISSET(4, &cpus));
  EXPECT_FALSE(CPU_ISSET(1048, &cpus));
  std::remove(path.c_str());

  cpu_set_t none{};
  EXPECT_FALSE(detail::parse_cpu_list("/nonexistent/cpulist", &none));
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
