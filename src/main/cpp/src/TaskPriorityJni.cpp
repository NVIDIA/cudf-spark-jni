/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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
#include "task_priority.hpp"

#include <limits>
#include <mutex>
#include <unordered_map>

namespace {

class task_priority_tracker {
 public:
  long get(long attempt_id)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto const it = attempt_priorities_.find(attempt_id);
    if (it != attempt_priorities_.end()) { return it->second; }

    auto const priority = next_task_priority_--;
    attempt_priorities_.emplace(attempt_id, priority);
    return priority;
  }

  void done(long attempt_id)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    attempt_priorities_.erase(attempt_id);
  }

 private:
  long next_task_priority_ = std::numeric_limits<long>::max() - 1;
  std::mutex mutex_;
  std::unordered_map<long, long> attempt_priorities_;
};

task_priority_tracker& get_task_priority_tracker()
{
  static task_priority_tracker tracker;
  return tracker;
}

}  // namespace

namespace spark_rapids_jni {

long get_task_priority(long attempt_id)
{
  if (attempt_id == -1) {
    // Special case: -1 always gets highest priority
    return std::numeric_limits<long>::max();
  }

  return get_task_priority_tracker().get(attempt_id);
}

void task_done(long attempt_id)
{
  if (attempt_id == -1) {
    return;  // Nothing to do for special case
  }

  get_task_priority_tracker().done(attempt_id);
}

}  // namespace spark_rapids_jni

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_TaskPriority_getTaskPriority(
  JNIEnv* env, jclass, jlong task_attempt_id)
{
  return spark_rapids_jni::get_task_priority(task_attempt_id);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_TaskPriority_taskDone(JNIEnv* env,
                                                                              jclass,
                                                                              jlong task_attempt_id)
{
  spark_rapids_jni::task_done(task_attempt_id);
}
}
