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

#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"
#include "nvtx_ranges.hpp"
#include "pageable_pool_resource.hpp"

#include <cuda/memory_resource>

#include <sys/resource.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string_view>

#define CATCH_PAGEABLE_POOL_EXHAUSTED(env, ret_val)                        \
  JNI_CATCH_BEGIN(env, ret_val)                                            \
  catch (spark_rapids_jni::pageable_pool_exhausted const&)                 \
  {                                                                        \
    return -1; /* Pool exhausted — caller falls back to regular malloc. */ \
  }                                                                        \
  CATCH_SPECIAL_EXCEPTION(env, ret_val)                                    \
  CATCH_STD_EXCEPTION(env, ret_val)

namespace {

using clock_type = std::chrono::steady_clock;

std::uint64_t elapsed_ns(clock_type::time_point start)
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now() - start).count();
}

std::uint64_t thread_cpu_ns()
{
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}

struct staged_copy_stats {
  std::atomic<std::uint64_t> calls{0};
  std::atomic<std::uint64_t> bytes{0};
  std::atomic<std::uint64_t> total_ns{0};
  std::atomic<std::uint64_t> ensure_capacity_ns{0};
  std::atomic<std::uint64_t> ensure_stream_ns{0};
  std::atomic<std::uint64_t> stream_wait_ns{0};
  std::atomic<std::uint64_t> gpu_copy_sync_ns{0};
  std::atomic<std::uint64_t> host_memcpy_ns{0};
  std::atomic<std::uint64_t> host_memcpy_cpu_ns{0};
  std::atomic<std::uint64_t> host_memcpy_minor_faults{0};
  std::atomic<std::uint64_t> host_memcpy_major_faults{0};
  std::atomic<std::uint64_t> active_host_memcpy{0};
  std::atomic<std::uint64_t> max_active_host_memcpy{0};
  std::atomic<std::uint64_t> stream_create_calls{0};
  std::atomic<std::uint64_t> max_bytes{0};
};

staged_copy_stats stats;
std::atomic<bool> stats_enabled{false};

void update_max(std::atomic<std::uint64_t>& max_value, std::uint64_t value)
{
  auto old_value = max_value.load(std::memory_order_relaxed);
  while (value > old_value &&
         !max_value.compare_exchange_weak(old_value, value, std::memory_order_relaxed)) {}
}

class thread_local_staging_buffer {
 public:
  ~thread_local_staging_buffer()
  {
    if (stream_ != nullptr) { cudaStreamDestroy(stream_); }
  }

  void set_buffer(void* buffer, std::size_t size)
  {
    buffer_   = buffer;
    capacity_ = size;
    ensure_stream();
  }

  void copy_from_device(void* dst, void const* src, std::size_t size, cudaEvent_t wait_event)
  {
    if (size == 0) { return; }
    if (!stats_enabled.load(std::memory_order_relaxed)) {
      ensure_capacity(size);
      ensure_stream();
      CUDF_CUDA_TRY(cudaStreamWaitEvent(stream_, wait_event, 0));
      {
        nvtx3::scoped_range_in<spark_rapids_jni::srj_domain> range{"stagedD2HGpuCopySync"};
        CUDF_CUDA_TRY(cudaMemcpyAsync(buffer_, src, size, cudaMemcpyDefault, stream_));
        CUDF_CUDA_TRY(cudaStreamSynchronize(stream_));
      }
      {
        nvtx3::scoped_range_in<spark_rapids_jni::srj_domain> range{"stagedD2HHostMemcpy"};
        std::memcpy(dst, buffer_, size);
      }
      return;
    }
    auto const total_start = clock_type::now();
    stats.calls.fetch_add(1, std::memory_order_relaxed);
    stats.bytes.fetch_add(size, std::memory_order_relaxed);
    update_max(stats.max_bytes, size);
    auto phase_start = clock_type::now();
    ensure_capacity(size);
    stats.ensure_capacity_ns.fetch_add(elapsed_ns(phase_start), std::memory_order_relaxed);
    phase_start = clock_type::now();
    ensure_stream();
    stats.ensure_stream_ns.fetch_add(elapsed_ns(phase_start), std::memory_order_relaxed);
    phase_start = clock_type::now();
    CUDF_CUDA_TRY(cudaStreamWaitEvent(stream_, wait_event, 0));
    stats.stream_wait_ns.fetch_add(elapsed_ns(phase_start), std::memory_order_relaxed);
    phase_start = clock_type::now();
    {
      nvtx3::scoped_range_in<spark_rapids_jni::srj_domain> range{"stagedD2HGpuCopySync"};
      CUDF_CUDA_TRY(cudaMemcpyAsync(buffer_, src, size, cudaMemcpyDefault, stream_));
      CUDF_CUDA_TRY(cudaStreamSynchronize(stream_));
    }
    stats.gpu_copy_sync_ns.fetch_add(elapsed_ns(phase_start), std::memory_order_relaxed);
    phase_start       = clock_type::now();
    auto const active = stats.active_host_memcpy.fetch_add(1, std::memory_order_relaxed) + 1;
    update_max(stats.max_active_host_memcpy, active);
    auto const cpu_start = thread_cpu_ns();
    rusage faults_before{};
    getrusage(RUSAGE_THREAD, &faults_before);
    {
      nvtx3::scoped_range_in<spark_rapids_jni::srj_domain> range{"stagedD2HHostMemcpy"};
      std::memcpy(dst, buffer_, size);
    }
    rusage faults_after{};
    getrusage(RUSAGE_THREAD, &faults_after);
    stats.host_memcpy_minor_faults.fetch_add(faults_after.ru_minflt - faults_before.ru_minflt,
                                             std::memory_order_relaxed);
    stats.host_memcpy_major_faults.fetch_add(faults_after.ru_majflt - faults_before.ru_majflt,
                                             std::memory_order_relaxed);
    stats.host_memcpy_cpu_ns.fetch_add(thread_cpu_ns() - cpu_start, std::memory_order_relaxed);
    stats.host_memcpy_ns.fetch_add(elapsed_ns(phase_start), std::memory_order_relaxed);
    stats.active_host_memcpy.fetch_sub(1, std::memory_order_relaxed);
    stats.total_ns.fetch_add(elapsed_ns(total_start), std::memory_order_relaxed);
  }

  void copy_to_device(void* dst, void const* src, std::size_t size, cudaEvent_t wait_event)
  {
    if (size == 0) { return; }
    if (!stats_enabled.load(std::memory_order_relaxed)) {
      ensure_capacity(size);
      ensure_stream();
      CUDF_CUDA_TRY(cudaStreamWaitEvent(stream_, wait_event, 0));
      std::memcpy(buffer_, src, size);
      CUDF_CUDA_TRY(cudaMemcpyAsync(dst, buffer_, size, cudaMemcpyDefault, stream_));
      CUDF_CUDA_TRY(cudaStreamSynchronize(stream_));
      return;
    }
    auto const total_start = clock_type::now();
    stats.calls.fetch_add(1, std::memory_order_relaxed);
    stats.bytes.fetch_add(size, std::memory_order_relaxed);
    update_max(stats.max_bytes, size);
    auto phase_start = clock_type::now();
    ensure_capacity(size);
    stats.ensure_capacity_ns.fetch_add(elapsed_ns(phase_start), std::memory_order_relaxed);
    phase_start = clock_type::now();
    ensure_stream();
    stats.ensure_stream_ns.fetch_add(elapsed_ns(phase_start), std::memory_order_relaxed);
    phase_start = clock_type::now();
    CUDF_CUDA_TRY(cudaStreamWaitEvent(stream_, wait_event, 0));
    stats.stream_wait_ns.fetch_add(elapsed_ns(phase_start), std::memory_order_relaxed);
    phase_start = clock_type::now();
    std::memcpy(buffer_, src, size);
    stats.host_memcpy_ns.fetch_add(elapsed_ns(phase_start), std::memory_order_relaxed);
    phase_start = clock_type::now();
    CUDF_CUDA_TRY(cudaMemcpyAsync(dst, buffer_, size, cudaMemcpyDefault, stream_));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream_));
    stats.gpu_copy_sync_ns.fetch_add(elapsed_ns(phase_start), std::memory_order_relaxed);
    stats.total_ns.fetch_add(elapsed_ns(total_start), std::memory_order_relaxed);
  }

 private:
  void ensure_capacity(std::size_t size)
  {
    if (capacity_ >= size) { return; }
    throw std::runtime_error("pinned staging buffer is too small");
  }

  void ensure_stream()
  {
    if (stream_ == nullptr) {
      CUDF_CUDA_TRY(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
      if (stats_enabled.load(std::memory_order_relaxed)) {
        stats.stream_create_calls.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  void* buffer_         = nullptr;
  std::size_t capacity_ = 0;
  cudaStream_t stream_  = nullptr;
};

thread_local thread_local_staging_buffer staging_buffer;

}  // namespace

extern "C" {

JNIEXPORT void JNICALL
Java_ai_rapids_cudf_PageableMemoryPool_setStagedCopyStatsEnabled(JNIEnv*, jclass, jboolean enabled)
{
  stats_enabled.store(enabled != JNI_FALSE, std::memory_order_relaxed);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_PageableMemoryPool_newPageablePoolMemoryResource(
  JNIEnv* env, jclass, jlong pool_size, jint pretouch_threads)
{
  JNI_TRY
  {
    // Internal comparison switch; the indexed free list remains the default.
    auto const* strategy   = std::getenv("CUDF_SPARK_PAGEABLE_FREE_LIST");
    bool const use_indexed = strategy == nullptr || std::string_view{strategy} != "linear";
    auto pool              = std::make_unique<spark_rapids_jni::pageable_pool_resource>(
      cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>(
        spark_rapids_jni::pageable_memory_resource{}),
      static_cast<std::size_t>(pool_size),
      static_cast<int>(pretouch_threads),
      use_indexed);
    return reinterpret_cast<jlong>(pool.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_PageableMemoryPool_releasePageablePoolMemoryResource(
  JNIEnv* env, jclass, jlong pool_ptr)
{
  JNI_TRY { cudf::jni::safe_delete<spark_rapids_jni::pageable_pool_resource>(pool_ptr); }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_PageableMemoryPool_allocFromPageablePool(JNIEnv* env,
                                                                                     jclass,
                                                                                     jlong pool_ptr,
                                                                                     jlong size)
{
  JNI_TRY
  {
    auto* pool = reinterpret_cast<spark_rapids_jni::pageable_pool_resource*>(pool_ptr);
    void* ret  = pool->allocate_sync(static_cast<std::size_t>(size));
    return reinterpret_cast<jlong>(ret);
  }
  CATCH_PAGEABLE_POOL_EXHAUSTED(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_PageableMemoryPool_freeFromPageablePool(
  JNIEnv* env, jclass, jlong pool_ptr, jlong ptr, jlong size)
{
  JNI_TRY
  {
    auto* pool = reinterpret_cast<spark_rapids_jni::pageable_pool_resource*>(pool_ptr);
    pool->deallocate_sync(reinterpret_cast<void*>(ptr), static_cast<std::size_t>(size));
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_PageableMemoryPool_stagedCopyFromDevice(
  JNIEnv* env, jclass, jlong jdst, jlong jsrc, jlong jsize, jlong jwait_event)
{
  if (jsize == 0) { return; }
  JNI_ARG_CHECK(env, jdst != 0, "dst memory pointer is null", );
  JNI_ARG_CHECK(env, jsrc != 0, "src memory pointer is null", );
  JNI_ARG_CHECK(env, jwait_event != 0, "wait event is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    staging_buffer.copy_from_device(reinterpret_cast<void*>(jdst),
                                    reinterpret_cast<void const*>(jsrc),
                                    static_cast<std::size_t>(jsize),
                                    reinterpret_cast<cudaEvent_t>(jwait_event));
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_PageableMemoryPool_stagedCopyToDevice(
  JNIEnv* env, jclass, jlong jdst, jlong jsrc, jlong jsize, jlong jwait_event)
{
  if (jsize == 0) { return; }
  JNI_ARG_CHECK(env, jdst != 0, "dst memory pointer is null", );
  JNI_ARG_CHECK(env, jsrc != 0, "src memory pointer is null", );
  JNI_ARG_CHECK(env, jwait_event != 0, "wait event is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    staging_buffer.copy_to_device(reinterpret_cast<void*>(jdst),
                                  reinterpret_cast<void const*>(jsrc),
                                  static_cast<std::size_t>(jsize),
                                  reinterpret_cast<cudaEvent_t>(jwait_event));
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_PageableMemoryPool_stagedHostCopy(
  JNIEnv* env, jclass, jlong jdst, jlong jsrc, jlong jsize)
{
  if (jsize == 0) { return; }
  JNI_ARG_CHECK(env, jdst != 0, "dst memory pointer is null", );
  JNI_ARG_CHECK(env, jsrc != 0, "src memory pointer is null", );
  JNI_TRY
  {
    if (!stats_enabled.load(std::memory_order_relaxed)) {
      nvtx3::scoped_range_in<spark_rapids_jni::srj_domain> range{"stagedD2HHostMemcpy"};
      std::memcpy(reinterpret_cast<void*>(jdst),
                  reinterpret_cast<void const*>(jsrc),
                  static_cast<std::size_t>(jsize));
      return;
    }
    auto const start = clock_type::now();
    stats.calls.fetch_add(1, std::memory_order_relaxed);
    stats.bytes.fetch_add(jsize, std::memory_order_relaxed);
    update_max(stats.max_bytes, jsize);
    auto const active = stats.active_host_memcpy.fetch_add(1, std::memory_order_relaxed) + 1;
    update_max(stats.max_active_host_memcpy, active);
    auto const cpu_start = thread_cpu_ns();
    rusage faults_before{};
    getrusage(RUSAGE_THREAD, &faults_before);
    {
      nvtx3::scoped_range_in<spark_rapids_jni::srj_domain> range{"stagedD2HHostMemcpy"};
      std::memcpy(reinterpret_cast<void*>(jdst),
                  reinterpret_cast<void const*>(jsrc),
                  static_cast<std::size_t>(jsize));
    }
    rusage faults_after{};
    getrusage(RUSAGE_THREAD, &faults_after);
    stats.host_memcpy_minor_faults.fetch_add(faults_after.ru_minflt - faults_before.ru_minflt,
                                             std::memory_order_relaxed);
    stats.host_memcpy_major_faults.fetch_add(faults_after.ru_majflt - faults_before.ru_majflt,
                                             std::memory_order_relaxed);
    stats.host_memcpy_cpu_ns.fetch_add(thread_cpu_ns() - cpu_start, std::memory_order_relaxed);
    stats.host_memcpy_ns.fetch_add(elapsed_ns(start), std::memory_order_relaxed);
    stats.active_host_memcpy.fetch_sub(1, std::memory_order_relaxed);
    stats.total_ns.fetch_add(elapsed_ns(start), std::memory_order_relaxed);
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_PageableMemoryPool_setStagingBuffer(JNIEnv* env,
                                                                               jclass,
                                                                               jlong jbuffer,
                                                                               jlong jsize)
{
  JNI_ARG_CHECK(env, jbuffer != 0, "staging buffer pointer is null", );
  JNI_ARG_CHECK(env, jsize > 0, "staging buffer size must be positive", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    staging_buffer.set_buffer(reinterpret_cast<void*>(jbuffer), static_cast<std::size_t>(jsize));
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_PageableMemoryPool_allocateRegisteredStagingBuffer(
  JNIEnv* env, jclass, jlong jsize)
{
  JNI_ARG_CHECK(env, jsize > 0, "staging buffer size must be positive", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* buffer = std::malloc(static_cast<std::size_t>(jsize));
    if (buffer == nullptr) { throw std::bad_alloc{}; }
    std::memset(buffer, 0, static_cast<std::size_t>(jsize));
    auto const status =
      cudaHostRegister(buffer, static_cast<std::size_t>(jsize), cudaHostRegisterDefault);
    if (status != cudaSuccess) {
      std::free(buffer);
      cudf::detail::throw_cuda_error(status, __FILE__, __LINE__);
    }
    return reinterpret_cast<jlong>(buffer);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_PageableMemoryPool_freeRegisteredStagingBuffer(
  JNIEnv* env, jclass, jlong jaddress)
{
  JNI_ARG_CHECK(env, jaddress != 0, "staging buffer pointer is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* buffer = reinterpret_cast<void*>(jaddress);
    CUDF_CUDA_TRY(cudaHostUnregister(buffer));
    std::free(buffer);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jstring JNICALL
Java_ai_rapids_cudf_PageableMemoryPool_getAndResetStagedCopyNativeStats(JNIEnv* env, jclass)
{
  auto exchange = [](std::atomic<std::uint64_t>& value) {
    return value.exchange(0, std::memory_order_relaxed);
  };
  auto const calls                    = exchange(stats.calls);
  auto const bytes                    = exchange(stats.bytes);
  auto const total_ns                 = exchange(stats.total_ns);
  auto const ensure_capacity_ns       = exchange(stats.ensure_capacity_ns);
  auto const ensure_stream_ns         = exchange(stats.ensure_stream_ns);
  auto const stream_wait_ns           = exchange(stats.stream_wait_ns);
  auto const gpu_copy_sync_ns         = exchange(stats.gpu_copy_sync_ns);
  auto const host_memcpy_ns           = exchange(stats.host_memcpy_ns);
  auto const host_memcpy_cpu_ns       = exchange(stats.host_memcpy_cpu_ns);
  auto const host_memcpy_minor_faults = exchange(stats.host_memcpy_minor_faults);
  auto const host_memcpy_major_faults = exchange(stats.host_memcpy_major_faults);
  auto const max_active_host_memcpy   = exchange(stats.max_active_host_memcpy);
  auto const stream_create_calls      = exchange(stats.stream_create_calls);
  auto const max_bytes                = exchange(stats.max_bytes);

  std::ostringstream out;
  out << "calls=" << calls << " bytes=" << bytes << " chunksGBps="
      << (total_ns == 0 ? 0.0 : static_cast<double>(bytes) / static_cast<double>(total_ns))
      << " totalMs=" << (static_cast<double>(total_ns) / 1.0e6)
      << " ensureCapacityMs=" << (static_cast<double>(ensure_capacity_ns) / 1.0e6)
      << " ensureStreamMs=" << (static_cast<double>(ensure_stream_ns) / 1.0e6)
      << " streamWaitMs=" << (static_cast<double>(stream_wait_ns) / 1.0e6)
      << " gpuCopySyncMs=" << (static_cast<double>(gpu_copy_sync_ns) / 1.0e6)
      << " hostMemcpyMs=" << (static_cast<double>(host_memcpy_ns) / 1.0e6)
      << " hostMemcpyCpuMs=" << (static_cast<double>(host_memcpy_cpu_ns) / 1.0e6)
      << " hostMemcpyMinorFaults=" << host_memcpy_minor_faults
      << " hostMemcpyMajorFaults=" << host_memcpy_major_faults
      << " maxActiveHostMemcpy=" << max_active_host_memcpy
      << " streamCreateCalls=" << stream_create_calls << " maxBytes=" << max_bytes;
  return env->NewStringUTF(out.str().c_str());
}

}  // extern "C"
