/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "profiler_generated.h"
#include "profiler_serializer.hpp"
#include "spark_rapids_jni_version.h"

#include <cupti.h>
#include <jni.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <stack>
#include <thread>

// Set this to true to have each CUPTI buffer dumped to stderr as it arrives.
#define PROFILER_DEBUG_LOG_BUFFER 0

#define JNI_EXCEPTION_OCCURRED_CHECK(env, ret_val)    \
  {                                                   \
    if (env->ExceptionOccurred()) { return ret_val; } \
  }

#define JNI_THROW_NEW(env, class_name, message, ret_val) \
  {                                                      \
    jclass ex_class = env->FindClass(class_name);        \
    if (ex_class == NULL) { return ret_val; }            \
    env->ThrowNew(ex_class, message);                    \
    return ret_val;                                      \
  }

#define CATCH_STD_CLASS(env, class_name, ret_val) \
  catch (const std::exception& e) { JNI_THROW_NEW(env, class_name, e.what(), ret_val) }

#define CATCH_STD(env, ret_val) CATCH_STD_CLASS(env, "java/lang/RuntimeException", ret_val)

namespace spark_rapids_jni::profiler {

namespace {

// Encapsulates a buffer of profile data
struct profile_buffer {
  explicit profile_buffer(size_t size) : size_(size), valid_size_(0)
  {
    auto err = posix_memalign(reinterpret_cast<void**>(&data_), ALIGN_BYTES, size_);
    if (err != 0) {
      std::cerr << "PROFILER: Failed to allocate CUPTI buffer: " << strerror(err) << std::endl;
      data_ = nullptr;
      size_ = 0;
    }
  }

  profile_buffer(uint8_t* data, size_t size, size_t valid_size)
    : data_(data), size_(size), valid_size_(valid_size)
  {
  }

  // Disconnects the underlying buffer of memory from the instance.
  // The caller is responsible for freeing the resulting buffer.
  void release(uint8_t** data_ptr_ptr, size_t* size_ptr)
  {
    *data_ptr_ptr = data_;
    *size_ptr     = size_;
    data_         = nullptr;
    size_         = 0;
  }

  ~profile_buffer()
  {
    free(data_);
    data_ = nullptr;
    size_ = 0;
  }

  uint8_t const* data() const { return data_; }
  uint8_t* data() { return data_; }
  size_t size() const { return size_; }
  size_t valid_size() const { return valid_size_; }
  void set_valid_size(size_t size) { valid_size_ = size; }

 private:
  static constexpr size_t ALIGN_BYTES = 8;
  uint8_t* data_;
  size_t size_;
  size_t valid_size_;
};

// Queue of profile buffers that have been filled with profile data.
struct completed_buffer_queue {
  // Gets the next available buffer of profile data, blocking until a buffer is available
  // or the queue is shutdown. If the queue is shutdown, a nullptr is returned.
  std::unique_ptr<profile_buffer> get()
  {
    std::unique_lock lock(lock_);
    cv_.wait(lock, [this] { return shutdown_ || buffers_.size() > 0; });
    if (buffers_.size() > 0) {
      auto result = std::move(buffers_.front());
      buffers_.pop();
      return result;
    }
    return std::unique_ptr<profile_buffer>(nullptr);
  }

  void put(std::unique_ptr<profile_buffer>&& buffer)
  {
    std::unique_lock lock(lock_);
    if (!shutdown_) {
      buffers_.push(std::move(buffer));
      lock.unlock();
      cv_.notify_one();
    }
  }

  void shutdown()
  {
    std::unique_lock lock(lock_);
    shutdown_ = true;
    lock.unlock();
    cv_.notify_one();
  }

 private:
  std::mutex lock_;
  std::condition_variable cv_;
  std::queue<std::unique_ptr<profile_buffer>> buffers_;
  bool shutdown_ = false;
};

// Stack of profile buffers that are ready to be filled with profile data.
struct free_buffer_tracker {
  explicit free_buffer_tracker(size_t size) : buffer_size_(size) {}

  // Returns the next available profile buffer or creates one if none are available.
  std::unique_ptr<profile_buffer> get()
  {
    {
      std::lock_guard lock(lock_);
      if (buffers_.size() > 0) {
        auto result = std::move(buffers_.top());
        buffers_.pop();
        return result;
      }
    }
    return std::make_unique<profile_buffer>(buffer_size_);
  }

  void put(std::unique_ptr<profile_buffer>&& buffer)
  {
    buffer->set_valid_size(0);
    std::lock_guard lock(lock_);
    if (buffers_.size() < NUM_CACHED_BUFFERS) {
      buffers_.push(std::move(buffer));
    } else {
      buffer.reset(nullptr);
    }
  }

 private:
  static constexpr size_t NUM_CACHED_BUFFERS = 2;
  std::mutex lock_;
  std::stack<std::unique_ptr<profile_buffer>> buffers_;
  size_t buffer_size_;
};

void writer_thread_process(JavaVM* vm,
                           jobject j_writer,
                           size_t buffer_size,
                           size_t flush_threshold,
                           bool async_alloc_capture);

struct subscriber_state {
  CUpti_SubscriberHandle subscriber_handle;
  jobject j_writer;
  std::thread writer_thread;
  free_buffer_tracker free_buffers;
  completed_buffer_queue completed_buffers;
  bool has_cupti_callback_errored;
  bool is_shutdown;

  subscriber_state(jobject writer, size_t buffer_size)
    : j_writer(writer),
      free_buffers(buffer_size),
      has_cupti_callback_errored(false),
      is_shutdown(false)
  {
  }
};

// Global variables
subscriber_state* State = nullptr;
uint64_t Flush_period_msec;
std::atomic_uint64_t Last_flush_time_msec;

JavaVM* get_jvm(JNIEnv* env)
{
  JavaVM* vm;
  if (env->GetJavaVM(&vm) != 0) { throw std::runtime_error("Unable to get JavaVM"); }
  return vm;
}

JNIEnv* attach_to_jvm(JavaVM* vm)
{
  JavaVMAttachArgs args;
  args.version = JNI_VERSION_1_6;
  args.name    = const_cast<char*>("profiler writer");
  args.group   = nullptr;
  JNIEnv* env;
  if (vm->AttachCurrentThread(reinterpret_cast<void**>(&env), &args) != JNI_OK) {
    char const* msg = "PROFILER: unable to attach to JVM";
    std::cerr << msg << std::endl;
    throw std::runtime_error(msg);
  }
  return env;
}

char const* get_cupti_error(CUptiResult rc)
{
  char const* err;
  if (cuptiGetResultString(rc, &err) != CUPTI_SUCCESS) { err = "UNKNOWN"; }
  return err;
}

void check_cupti(CUptiResult rc, std::string msg)
{
  if (rc != CUPTI_SUCCESS) { throw std::runtime_error(msg + ": " + get_cupti_error(rc)); }
}

uint64_t timestamp_now()
{
  timespec info;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &info) != 0) {
    static bool have_logged_error = false;
    if (!have_logged_error) {
      std::cerr << "PROFILER: Unable to determine current time!" << std::endl;
      have_logged_error = true;
    }
    // No idea what time it is, so return the last flush time which will effectively
    // disable periodic flushing but avoid pathologic flushing on every kernel launch.
    return Last_flush_time_msec;
  }
  return info.tv_sec * 1e3 + info.tv_nsec / 1e6;
}

void on_driver_launch_exit()
{
  auto now = timestamp_now();
  if (now - Last_flush_time_msec >= Flush_period_msec) {
    auto rc = cuptiActivityFlushAll(0);
    if (rc != CUPTI_SUCCESS) {
      std::cerr << "PROFILER: Error interval flushing records: " << get_cupti_error(rc)
                << std::endl;
    }
    Last_flush_time_msec = now;
  }
}

void domain_driver_callback(CUpti_CallbackId callback_id, CUpti_CallbackData const* cb_data)
{
  if (cb_data->callbackSite == CUPTI_API_ENTER) { return; }

  switch (callback_id) {
    case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch:
    case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunch:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz: on_driver_launch_exit(); break;
    default:
      std::cerr << "PROFILER: Unexpected driver API callback for " << callback_id << std::endl;
      break;
  }
}

void domain_runtime_callback(CUpti_CallbackId callback_id, CUpti_CallbackData const* data_ptr)
{
  switch (callback_id) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020:
      if (data_ptr->callbackSite == CUPTI_API_ENTER) {
        auto rc = cuptiActivityFlushAll(0);
        if (rc != CUPTI_SUCCESS) {
          std::cerr << "PROFILER: Error flushing CUPTI activity on device reset: "
                    << get_cupti_error(rc) << std::endl;
        }
      }
      break;
    default: break;
  }
}

// Invoked by CUPTI when something occurs for which we previously requested a callback.
void CUPTIAPI callback_handler(void*,
                               CUpti_CallbackDomain domain,
                               CUpti_CallbackId callback_id,
                               const void* callback_data_ptr)
{
  auto rc = cuptiGetLastError();
  if (rc != CUPTI_SUCCESS && !State->has_cupti_callback_errored) {
    // State->has_cupti_callback_errored = true;
    std::cerr << "PROFILER: Error handling callback: " << get_cupti_error(rc) << std::endl;
    return;
  }

  auto cb_data = static_cast<CUpti_CallbackData const*>(callback_data_ptr);
  switch (domain) {
    case CUPTI_CB_DOMAIN_DRIVER_API: domain_driver_callback(callback_id, cb_data); break;
    case CUPTI_CB_DOMAIN_RUNTIME_API: domain_runtime_callback(callback_id, cb_data); break;
    default: break;
  }
}

// Invoked by CUPTI when a new buffer is needed to record CUPTI activity events.
void CUPTIAPI buffer_requested_callback(uint8_t** buffer_ptr_ptr,
                                        size_t* size_ptr,
                                        size_t* max_num_records_ptr)
{
  *max_num_records_ptr = 0;
  if (!State->is_shutdown) {
    auto buffer = State->free_buffers.get();
    buffer->release(buffer_ptr_ptr, size_ptr);
  } else {
    *buffer_ptr_ptr = nullptr;
    *size_ptr       = 0;
  }
}

// Invoked by CUPTI when an activity event buffer has completed.
void CUPTIAPI buffer_completed_callback(
  CUcontext, uint32_t, uint8_t* buffer, size_t buffer_size, size_t valid_size)
{
  auto pb = std::make_unique<profile_buffer>(buffer, buffer_size, valid_size);
  if (!State->is_shutdown) { State->completed_buffers.put(std::move(pb)); }
}

// Setup the environment variables for NVTX library injection so we can capture NVTX events.
void setup_nvtx_env(JNIEnv* env, jstring j_lib_path)
{
  auto lib_path = env->GetStringUTFChars(j_lib_path, 0);
  if (lib_path == NULL) { throw std::runtime_error("Error getting library path"); }
  setenv("NVTX_INJECTION64_PATH", lib_path, 1);
  env->ReleaseStringUTFChars(j_lib_path, lib_path);
}

// Main processing loop for the background writer thread
void writer_thread_process(JavaVM* vm,
                           jobject j_writer,
                           size_t buffer_size,
                           size_t flush_threshold,
                           bool async_alloc_capture)
{
  try {
    JNIEnv* env = attach_to_jvm(vm);
    profiler_serializer serializer(
      env, j_writer, buffer_size, flush_threshold, async_alloc_capture);
    auto buffer = State->completed_buffers.get();
    while (buffer) {
      serializer.process_cupti_buffer(buffer->data(), buffer->valid_size());
      State->free_buffers.put(std::move(buffer));
      buffer = State->completed_buffers.get();
    }
    serializer.flush();
  } catch (std::exception const& e) {
    std::cerr << "PROFILER: WRITER THREAD ERROR: " << e.what() << std::endl;
    // no-op process buffers
    auto buffer = State->completed_buffers.get();
    while (buffer) {
      State->free_buffers.put(std::move(buffer));
      buffer = State->completed_buffers.get();
    }
  }
  vm->DetachCurrentThread();
}

// Enable/disable capture of CUPTI activity events
void update_activity_enable(bool enable)
{
  CUpti_ActivityKind const activity_ids[] = {CUPTI_ACTIVITY_KIND_DEVICE,
                                             CUPTI_ACTIVITY_KIND_DRIVER,
                                             CUPTI_ACTIVITY_KIND_RUNTIME,
                                             CUPTI_ACTIVITY_KIND_MEMCPY,
                                             CUPTI_ACTIVITY_KIND_MEMSET,
                                             CUPTI_ACTIVITY_KIND_NAME,
                                             CUPTI_ACTIVITY_KIND_MARKER,
                                             CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
                                             CUPTI_ACTIVITY_KIND_OVERHEAD};
  if (enable) {
    for (CUpti_ActivityKind const id : activity_ids) {
      check_cupti(cuptiActivityEnable(id), "Error enabling device activity");
    }
  } else {
    for (CUpti_ActivityKind const id : activity_ids) {
      check_cupti(cuptiActivityDisable(id), "Error disabling device activity");
    }
    check_cupti(cuptiActivityFlushAll(0), "Error flushing activity records");
  }
}

}  // anonymous namespace

}  // namespace spark_rapids_jni::profiler

extern "C" {

using namespace spark_rapids_jni::profiler;

JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_jni_Profiler_nativeInit(JNIEnv* env,
                                                     jclass,
                                                     jstring j_lib_path,
                                                     jobject j_writer,
                                                     jlong write_buffer_size,
                                                     jint flush_period_msec,
                                                     bool async_alloc_capture)
{
  try {
    setup_nvtx_env(env, j_lib_path);
    // grab a global reference to the writer instance so it isn't garbage collected
    auto writer = static_cast<jobject>(env->NewGlobalRef(j_writer));
    if (!writer) { throw std::runtime_error("Unable to create a global reference to writer"); }
    State                = new subscriber_state(writer, write_buffer_size);
    State->writer_thread = std::thread(writer_thread_process,
                                       get_jvm(env),
                                       writer,
                                       write_buffer_size,
                                       write_buffer_size,
                                       async_alloc_capture);
    auto rc              = cuptiSubscribe(&State->subscriber_handle, callback_handler, nullptr);
    check_cupti(rc, "Error initializing CUPTI");
    rc = cuptiEnableCallback(1,
                             State->subscriber_handle,
                             CUPTI_CB_DOMAIN_RUNTIME_API,
                             CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020);
    if (flush_period_msec > 0) {
      std::cerr << "PROFILER: Flushing activity records every " << flush_period_msec
                << " milliseconds" << std::endl;
      Flush_period_msec    = static_cast<uint64_t>(flush_period_msec);
      Last_flush_time_msec = timestamp_now();
      // CUPTI's periodic flush does not appear to work in this environment. As a workaround,
      // register a callback for all the various ways a GPU kernel gets launched. The callback
      // checks if the flush period has elapsed since we last flushed, and if so, forces a flush.
      CUpti_CallbackId const driver_launch_callback_ids[] = {
        CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch,
        CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz,
        CUPTI_DRIVER_TRACE_CBID_cuLaunch,
        CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel,
        CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz,
        CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice,
        CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid,
        CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync,
        CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
        CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz};
      for (CUpti_CallbackId const id : driver_launch_callback_ids) {
        rc = cuptiEnableCallback(1, State->subscriber_handle, CUPTI_CB_DOMAIN_DRIVER_API, id);
        check_cupti(rc, "Error registering driver launch callbacks");
      }
    }
    check_cupti(rc, "Error enabling device reset callback");
    rc = cuptiActivityRegisterCallbacks(buffer_requested_callback, buffer_completed_callback);
    check_cupti(rc, "Error registering activity buffer callbacks");
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_Profiler_nativeStart(JNIEnv* env, jclass)
{
  try {
    if (State && !State->is_shutdown) { update_activity_enable(true); }
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_Profiler_nativeStop(JNIEnv* env, jclass)
{
  try {
    if (State && !State->is_shutdown) { update_activity_enable(false); }
  }
  CATCH_STD(env, );
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_Profiler_nativeShutdown(JNIEnv* env, jclass)
{
  try {
    if (State && !State->is_shutdown) {
      auto unsub_rc = cuptiUnsubscribe(State->subscriber_handle);
      auto flush_rc = cuptiActivityFlushAll(1);
      State->completed_buffers.shutdown();
      State->writer_thread.join();
      State->is_shutdown = true;
      env->DeleteGlobalRef(State->j_writer);
      // There can be late arrivals of CUPTI activity events and other callbacks, so it's safer
      // and simpler to _not_ delete the State object on shutdown.
      check_cupti(unsub_rc, "Error unsubscribing from CUPTI");
      check_cupti(flush_rc, "Error flushing CUPTI records");
    }
  }
  CATCH_STD(env, );
}

}  // extern "C"

// Extern the CUPTI NVTX initialization APIs. The APIs are thread-safe.
extern "C" CUptiResult CUPTIAPI cuptiNvtxInitialize(void* pfnGetExportTable);
extern "C" CUptiResult CUPTIAPI cuptiNvtxInitialize2(void* pfnGetExportTable);

// Interface that may be called by NVTX to capture NVTX events
extern "C" JNIEXPORT int InitializeInjectionNvtx(void* p)
{
  CUptiResult res = cuptiNvtxInitialize(p);
  return (res == CUPTI_SUCCESS) ? 1 : 0;
}

// Interface that may be called by NVTX to capture NVTX events
extern "C" JNIEXPORT int InitializeInjectionNvtx2(void* p)
{
  CUptiResult res = cuptiNvtxInitialize2(p);
  return (res == CUPTI_SUCCESS) ? 1 : 0;
}
