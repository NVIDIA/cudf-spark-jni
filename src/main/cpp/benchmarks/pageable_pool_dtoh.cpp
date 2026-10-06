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

// Host-only benchmarks for the pageable pool family: CUDA runtime API calls +
// nvbench, no device code. Each optimization technique gets its own axis:
//   - init bench:    pretouch method (MADV_POPULATE_WRITE vs manual loop)
//   - dtoh bench:    THP policy (default vs MADV_NOHUGEPAGE)
//   - alloc bench:   free-list/allocator variant (list, list+cache, indexed, arena)
//   - fragment bench: O(F) list vs O(log F) indexed under heavy fragmentation
//   - chunked bench: K-way split DtoH copies across private streams

#include "pageable_arena_pool_resource.hpp"
#include "pageable_pool_resource.hpp"

#include <cuda_runtime_api.h>

#include <nvbench/nvbench.cuh>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t kMiB = 1'024ULL * 1'024ULL;
constexpr std::size_t kGiB = 1'024ULL * kMiB;

[[nodiscard]] void* checked_cuda_malloc(std::size_t size)
{
  void* ptr = nullptr;
  if (cudaMalloc(&ptr, size) != cudaSuccess || ptr == nullptr) {
    cudaGetLastError();
    return nullptr;
  }
  return ptr;
}

// --- pre-touch init: method axis -------------------------------------------
void pool_init(nvbench::state& state)
{
  auto const size             = static_cast<std::size_t>(state.get_int64("size"));
  auto const pretouch_threads = static_cast<int>(state.get_int64("pretouch_threads"));
  bool const populate         = state.get_string("method") == "populate";

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch&, auto& timer) {
    timer.start();
    auto pool =
      pageable_pool_resource(cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>(
                               spark_rapids_jni::pageable_memory_resource{}),
                             size,
                             pretouch_threads,
                             /*numa_node=*/-1,
                             /*use_remainder_cache=*/false,
                             populate);
    timer.stop();
  });
}

// --- DtoH bandwidth: THP axis ----------------------------------------------
void dtoh_pool(nvbench::state& state)
{
  auto const size     = static_cast<std::size_t>(state.get_int64("size"));
  bool const thp_deny = state.get_string("thp") == "deny";

  auto pool = pageable_pool_resource(cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>(
                                       spark_rapids_jni::pageable_memory_resource{}),
                                     size,
                                     /*pretouch_threads=*/8,
                                     /*numa_node=*/-1,
                                     /*use_remainder_cache=*/false,
                                     /*populate_write=*/true,
                                     thp_deny);
  void* dst = pool.allocate_sync(size);

  void* src = checked_cuda_malloc(size);
  if (src == nullptr) {
    state.skip("cudaMalloc failed");
    return;
  }
  if (cudaMemset(src, 0xAB, size) != cudaSuccess) {
    cudaGetLastError();
    state.skip("cudaMemset failed");
    cudaFree(src);
    return;
  }

  state.add_global_memory_reads(size);
  state.add_global_memory_writes(size);
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    if (cudaMemcpy(dst, src, size, cudaMemcpyDefault) != cudaSuccess) {
      cudaGetLastError();
      cudaFree(src);
      throw std::runtime_error{"cudaMemcpy DtoH failed"};
    }
  });

  cudaFree(src);
  pool.deallocate_sync(dst, size);
}

// --- DtoH references: fresh malloc + pinned --------------------------------
void dtoh_reference(nvbench::state& state)
{
  auto const size     = static_cast<std::size_t>(state.get_int64("size"));
  auto const dst_type = state.get_string("dst");

  void* src = checked_cuda_malloc(size);
  if (src == nullptr) {
    state.skip("cudaMalloc failed");
    return;
  }
  if (cudaMemset(src, 0xAB, size) != cudaSuccess) {
    cudaGetLastError();
    state.skip("cudaMemset failed");
    cudaFree(src);
    return;
  }

  state.add_global_memory_reads(size);
  state.add_global_memory_writes(size);

  if (dst_type == "fresh_malloc") {
    // Fresh pageable: a new allocation per iteration, like the real fallback path —
    // the DtoH copy pays first-touch page faults every time.
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      void* dst = nullptr;
      if (::posix_memalign(&dst, 4'096, size) != 0) {
        throw std::runtime_error{"posix_memalign failed"};
      }
      if (cudaMemcpy(dst, src, size, cudaMemcpyDefault) != cudaSuccess) {
        cudaGetLastError();
        ::free(dst);
        cudaFree(src);
        throw std::runtime_error{"cudaMemcpy DtoH failed"};
      }
      ::free(dst);
    });
  } else {  // "pinned"
    void* dst = nullptr;
    if (cudaMallocHost(&dst, size) != cudaSuccess || dst == nullptr) {
      cudaGetLastError();
      state.skip("cudaMallocHost failed");
      cudaFree(src);
      return;
    }
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      if (cudaMemcpy(dst, src, size, cudaMemcpyDefault) != cudaSuccess) {
        cudaGetLastError();
        cudaFreeHost(dst);
        cudaFree(src);
        throw std::runtime_error{"cudaMemcpy DtoH failed"};
      }
    });
    cudaFreeHost(dst);
  }

  cudaFree(src);
}

// --- chunked DtoH: K private streams ---------------------------------------
void dtoh_pool_chunked(nvbench::state& state)
{
  auto const size   = static_cast<std::size_t>(state.get_int64("size"));
  auto const chunks = static_cast<int>(state.get_int64("chunks"));

  auto pool = pageable_pool_resource(cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>(
                                       spark_rapids_jni::pageable_memory_resource{}),
                                     size,
                                     /*pretouch_threads=*/8);
  void* dst = pool.allocate_sync(size);

  void* src = checked_cuda_malloc(size);
  if (src == nullptr) {
    state.skip("cudaMalloc failed");
    return;
  }
  if (cudaMemset(src, 0xAB, size) != cudaSuccess) {
    cudaGetLastError();
    state.skip("cudaMemset failed");
    cudaFree(src);
    return;
  }

  std::vector<cudaStream_t> streams(static_cast<std::size_t>(chunks));
  for (auto& stream : streams) {
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      cudaGetLastError();
      state.skip("cudaStreamCreate failed");
      cudaFree(src);
      return;
    }
  }

  state.add_global_memory_reads(size);
  state.add_global_memory_writes(size);
  std::size_t const chunk = size / static_cast<std::size_t>(chunks);
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    for (int k = 0; k < chunks; ++k) {
      if (cudaMemcpyAsync(static_cast<char*>(dst) + k * chunk,
                          static_cast<char*>(src) + k * chunk,
                          chunk,
                          cudaMemcpyDefault,
                          streams[static_cast<std::size_t>(k)]) != cudaSuccess) {
        cudaGetLastError();
        throw std::runtime_error{"cudaMemcpyAsync DtoH failed"};
      }
    }
    for (auto& stream : streams) {
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        cudaGetLastError();
        throw std::runtime_error{"cudaStreamSynchronize failed"};
      }
    }
  });

  for (auto& stream : streams) {
    cudaStreamDestroy(stream);
  }
  cudaFree(src);
  pool.deallocate_sync(dst, size);
}

// --- alloc/free latency + contention: allocator variant axis ----------------
template <typename Resource>
void alloc_free_impl(nvbench::state& state, Resource& pool)
{
  auto const chunk            = static_cast<std::size_t>(state.get_int64("chunk"));
  auto const threads          = static_cast<int>(state.get_int64("threads"));
  constexpr int kOpsPerThread = 64;

  std::atomic<std::uint64_t> round{0};
  std::atomic<int> finished{0};
  std::atomic<bool> stop{false};
  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&pool, &round, &finished, &stop, chunk]() {
      std::uint64_t seen = round.load();
      for (;;) {
        round.wait(seen);
        if (stop.load()) { return; }
        seen = round.load();
        for (int i = 0; i < kOpsPerThread; ++i) {
          try {
            void* p = pool.allocate_sync(chunk);
            pool.deallocate_sync(p, chunk);
          } catch (spark_rapids_jni::pageable_pool_exhausted const&) {
            // transiently full under contention — not a latency signal
          }
        }
        finished.fetch_add(1);
      }
    });
  }

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    finished.store(0);
    round.store(round.load() + 1);
    round.notify_all();
    while (finished.load() < threads) {
      std::this_thread::yield();
    }
  });

  stop.store(true);
  round.store(round.load() + 1);
  round.notify_all();
  for (auto& worker : workers) {
    worker.join();
  }
}

void alloc_free_list(nvbench::state& state)
{
  auto pool = pageable_pool_resource(cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>(
                                       spark_rapids_jni::pageable_memory_resource{}),
                                     4 * kGiB,
                                     /*pretouch_threads=*/8);
  alloc_free_impl(state, pool);
}

void alloc_free_list_cache(nvbench::state& state)
{
  auto pool = pageable_pool_resource(cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>(
                                       spark_rapids_jni::pageable_memory_resource{}),
                                     4 * kGiB,
                                     /*pretouch_threads=*/8,
                                     /*numa_node=*/-1,
                                     /*use_remainder_cache=*/true);
  alloc_free_impl(state, pool);
}

void alloc_free_indexed(nvbench::state& state)
{
  auto pool = pageable_pool_resource_t<spark_rapids_jni::detail::indexed_free_list>(
    cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>(
      spark_rapids_jni::pageable_memory_resource{}),
    4 * kGiB,
    /*pretouch_threads=*/8);
  alloc_free_impl(state, pool);
}

void alloc_free_arena(nvbench::state& state)
{
  auto pool =
    pageable_arena_pool_resource(cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>(
                                   spark_rapids_jni::pageable_memory_resource{}),
                                 4 * kGiB,
                                 /*pretouch_threads=*/8,
                                 /*numa_node=*/-1,
                                 /*populate_write=*/true,
                                 /*thp_deny=*/false,
                                 /*arena_count=*/16);
  alloc_free_impl(state, pool);
}

// --- fragmentation: O(F) list vs O(log F) indexed ---------------------------
template <typename Resource>
void fragmented_alloc_impl(nvbench::state& state, Resource& pool)
{
  // Fill a 4 GiB pool with 1 MiB blocks and free every OTHER one: ~2048 non-adjacent
  // 1 MiB holes (never coalescible — each hole is fenced by held blocks). The timed
  // loop repeatedly best-fit-allocates 256 KiB (fits any hole; every iteration scans
  // all F entries) and frees it back, so the hole set stays stable.
  constexpr std::size_t kHole  = kMiB;
  constexpr std::size_t kBytes = 4 * kGiB;
  std::vector<std::pair<void*, std::size_t>> held;
  for (std::size_t off = 0; off + kHole <= kBytes; off += kHole) {
    void* p = pool.allocate_sync(kHole);
    if (p == nullptr) { throw std::runtime_error{"fragment fill alloc failed"}; }
    if ((off / kHole) % 2 == 0) {
      held.emplace_back(p, kHole);  // keep: fences the hole after it
    } else {
      pool.deallocate_sync(p, kHole);  // becomes a hole
    }
  }
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    for (int i = 0; i < 8; ++i) {
      void* p = pool.allocate_sync(256 * 1'024);
      pool.deallocate_sync(p, 256 * 1'024);
    }
  });
  for (auto& [ptr, sz] : held) {
    pool.deallocate_sync(ptr, sz);
  }
}

void fragment_list(nvbench::state& state)
{
  auto pool = pageable_pool_resource(cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>(
                                       spark_rapids_jni::pageable_memory_resource{}),
                                     4 * kGiB,
                                     /*pretouch_threads=*/8);
  fragmented_alloc_impl(state, pool);
}

void fragment_indexed(nvbench::state& state)
{
  auto pool = pageable_pool_resource_t<spark_rapids_jni::detail::indexed_free_list>(
    cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>(
      spark_rapids_jni::pageable_memory_resource{}),
    4 * kGiB,
    /*pretouch_threads=*/8);
  fragmented_alloc_impl(state, pool);
}

}  // namespace

NVBENCH_BENCH(pool_init)
  .set_name("PageablePool Init")
  .add_int64_axis("size", {kGiB, 4 * kGiB})
  .add_int64_axis("pretouch_threads", {1, 8, 16})
  .add_string_axis("method", {"populate", "manual"});

NVBENCH_BENCH(dtoh_pool)
  .set_name("PageablePool DtoH Copy")
  .add_int64_axis("size", {64 * kMiB, 256 * kMiB, kGiB, 4 * kGiB})
  .add_string_axis("thp", {"default", "deny"});

NVBENCH_BENCH(dtoh_reference)
  .set_name("PageablePool DtoH Reference")
  .add_int64_axis("size", {64 * kMiB, 256 * kMiB, kGiB})
  .add_string_axis("dst", {"fresh_malloc", "pinned"});

NVBENCH_BENCH(dtoh_pool_chunked)
  .set_name("PageablePool DtoH Chunked")
  .add_int64_axis("size", {kGiB})
  .add_int64_axis("chunks", {1, 2, 4, 8});

NVBENCH_BENCH(alloc_free_list)
  .set_name("PageablePool AllocFree List")
  .add_int64_axis("chunk", {64 * 1'024, 64 * kMiB})
  .add_int64_axis("threads", {1, 4, 16, 32});

NVBENCH_BENCH(alloc_free_list_cache)
  .set_name("PageablePool AllocFree ListCache")
  .add_int64_axis("chunk", {64 * 1'024, 64 * kMiB})
  .add_int64_axis("threads", {1, 4, 16, 32});

NVBENCH_BENCH(alloc_free_indexed)
  .set_name("PageablePool AllocFree Indexed")
  .add_int64_axis("chunk", {64 * 1'024, 64 * kMiB})
  .add_int64_axis("threads", {1, 4, 16, 32});

NVBENCH_BENCH(alloc_free_arena)
  .set_name("PageablePool AllocFree Arena")
  .add_int64_axis("chunk", {64 * 1'024, 64 * kMiB})
  .add_int64_axis("threads", {1, 4, 16, 32});

NVBENCH_BENCH(fragment_list).set_name("PageablePool Fragmented List");
NVBENCH_BENCH(fragment_indexed).set_name("PageablePool Fragmented Indexed");
