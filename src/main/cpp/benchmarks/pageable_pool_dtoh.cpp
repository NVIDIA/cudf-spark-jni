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

// Host-only benchmark: CUDA runtime API calls + nvbench, no device code.

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

[[nodiscard]] spark_rapids_jni::pageable_pool_resource make_pool(std::size_t size,
                                                                 int pretouch_threads,
                                                                 int numa_node)
{
  return spark_rapids_jni::pageable_pool_resource(
    cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>(
      spark_rapids_jni::pageable_memory_resource{}),
    size,
    pretouch_threads,
    numa_node);
}

[[nodiscard]] void* checked_cuda_malloc(std::size_t size)
{
  void* ptr = nullptr;
  if (cudaMalloc(&ptr, size) != cudaSuccess || ptr == nullptr) {
    cudaGetLastError();
    return nullptr;
  }
  return ptr;
}

// DtoH copy bandwidth into pool-allocated (pre-touched, optionally NUMA-bound)
// memory. NOTE: the numa=on axis silently degrades to unbound placement when
// the GPU node cannot be resolved (e.g. no sysfs in a container) — cross-check
// with the init bench.
void dtoh_pool(nvbench::state& state)
{
  auto const size             = static_cast<std::size_t>(state.get_int64("size"));
  bool const bind_numa        = state.get_string("numa") == "on";
  auto const pretouch_threads = static_cast<int>(state.get_int64("pretouch_threads"));

  auto pool =
    make_pool(size, pretouch_threads, bind_numa ? spark_rapids_jni::detail::gpu_numa_node() : -1);
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

// Reference DtoH bandwidth into freshly malloc'd pageable and pinned memory.
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
    // Fresh pageable: a new allocation per iteration, like the real fallback
    // path — the DtoH copy pays first-touch page faults every time.
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      void* dst = nullptr;
      if (::posix_memalign(&dst, 4'096, size) != 0) {
        // Allocation failure here is a real bench-host problem, not a skippable
        // config.
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

// Pool construction time = backing allocation + parallel pre-touch. The manual
// timer excludes pool destruction (upstream free), which follows timer.stop().
void pool_init(nvbench::state& state)
{
  auto const size             = static_cast<std::size_t>(state.get_int64("size"));
  auto const pretouch_threads = static_cast<int>(state.get_int64("pretouch_threads"));
  bool const bind_numa        = state.get_string("numa") == "on";
  int const numa_node         = bind_numa ? spark_rapids_jni::detail::gpu_numa_node() : -1;

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch&, auto& timer) {
    timer.start();
    auto pool = make_pool(size, pretouch_threads, numa_node);
    timer.stop();
  });
}

// Allocation/deallocation latency, including contention across Spark tasks.
// Workers are spawned once OUTSIDE the timed region and synchronized per sample
// by generation counter, so per-sample thread spawn/join does not pollute the
// measurement and no worker can run an uncounted extra round.
void alloc_free_latency(nvbench::state& state)
{
  auto const chunk            = static_cast<std::size_t>(state.get_int64("chunk"));
  auto const threads          = static_cast<int>(state.get_int64("threads"));
  constexpr int kOpsPerThread = 64;

  auto pool = make_pool(4 * kGiB, 8, -1);
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
            // Transiently full under contention — not a latency signal.
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

}  // namespace

NVBENCH_BENCH(dtoh_pool)
  .set_name("PageablePool DtoH Copy")
  .add_int64_axis("size", {64 * kMiB, 256 * kMiB, kGiB, 4 * kGiB})
  .add_string_axis("numa", {"off", "on"})
  .add_int64_axis("pretouch_threads", {8});

NVBENCH_BENCH(dtoh_reference)
  .set_name("PageablePool DtoH Reference")
  .add_int64_axis("size", {64 * kMiB, 256 * kMiB, kGiB})
  .add_string_axis("dst", {"fresh_malloc", "pinned"});

NVBENCH_BENCH(pool_init)
  .set_name("PageablePool Init")
  .add_int64_axis("size", {kGiB, 4 * kGiB})
  .add_int64_axis("pretouch_threads", {1, 4, 8, 16})
  .add_string_axis("numa", {"off", "on"});

NVBENCH_BENCH(alloc_free_latency)
  .set_name("PageablePool AllocFree")
  .add_int64_axis("chunk", {64 * 1'024, 4 * kMiB, 64 * kMiB})
  .add_int64_axis("threads", {1, 4, 16, 32});
