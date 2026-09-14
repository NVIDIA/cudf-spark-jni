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

// Include the implementation to check unpublished state without exposing a test-only JNI API.
#include "ProfilerJni.cpp"

#include <gtest/gtest.h>

#include <string>
#include <tuple>
#include <vector>

namespace {

struct test_state {
  int fail_step         = -1;
  int setup_calls       = 0;
  int global_refs       = 0;
  int unsubscribe_calls = 0;
  bool subscribed       = false;
  std::atomic_int attaches{0};
  std::atomic_int detaches{0};
  std::atomic_int serializer_constructions{0};
  std::string exception;
  JNINativeInterface_ jni_functions{};
  JNIInvokeInterface_ vm_functions{};
  JNIEnv env{&jni_functions};
  JavaVM vm{&vm_functions};
  _jobject writer;
  _jclass exception_class;

  CUptiResult setup_result()
  {
    return setup_calls++ == fail_step ? CUPTI_ERROR_UNKNOWN : CUPTI_SUCCESS;
  }
};

test_state* Test_state;

class ProfilerInitTest : public ::testing::TestWithParam<std::tuple<int, int>> {
 protected:
  test_state state;

  void SetUp() override
  {
    Test_state = &state;
    ASSERT_EQ(State, nullptr);
    state.jni_functions.GetStringUTFChars = [](JNIEnv*, jstring, jboolean*) {
      return "unused-profiler-library";
    };
    state.jni_functions.ReleaseStringUTFChars = [](JNIEnv*, jstring, char const*) {};
    state.jni_functions.NewGlobalRef          = [](JNIEnv*, jobject writer) {
      ++Test_state->global_refs;
      return writer;
    };
    state.jni_functions.DeleteGlobalRef = [](JNIEnv*, jobject) {
      EXPECT_EQ(Test_state->attaches.load(), Test_state->detaches.load());
      --Test_state->global_refs;
    };
    state.jni_functions.GetJavaVM = [](JNIEnv*, JavaVM** vm) -> jint {
      *vm = &Test_state->vm;
      return JNI_OK;
    };
    state.jni_functions.FindClass = [](JNIEnv*, char const*) -> jclass {
      return &Test_state->exception_class;
    };
    state.jni_functions.ThrowNew = [](JNIEnv*, jclass, char const* message) -> jint {
      Test_state->exception = message;
      return JNI_OK;
    };
    state.vm_functions.AttachCurrentThread = [](JavaVM*, void** env, void*) -> jint {
      *env = &Test_state->env;
      ++Test_state->attaches;
      return JNI_OK;
    };
    state.vm_functions.DetachCurrentThread = [](JavaVM*) -> jint {
      ++Test_state->detaches;
      return JNI_OK;
    };
  }

  void TearDown() override
  {
    // Also reclaim a published session if an assertion catches a rollback regression.
    if (State) {
      State->completed_buffers.shutdown();
      if (State->writer_thread.joinable()) { State->writer_thread.join(); }
      delete State;
      State = nullptr;
    }
    Test_state = nullptr;
  }

  void init(int flush_period)
  {
    Java_com_nvidia_spark_rapids_jni_Profiler_nativeInit(
      &state.env, nullptr, nullptr, &state.writer, 1024, flush_period, false);
  }
};

TEST_P(ProfilerInitTest, RollsBackAndAllowsRetry)
{
  auto const [flush_period, fail_step] = GetParam();
  state.fail_step                      = fail_step;
  init(flush_period);

  auto const error = fail_step == 0   ? "Error initializing CUPTI"
                     : fail_step == 1 ? "Error enabling device reset callback"
                     : fail_step == (flush_period > 0 ? 12 : 2)
                       ? "Error registering activity buffer callbacks"
                       : "Error registering driver launch callbacks";
  EXPECT_EQ(state.exception, std::string(error) + ": injected failure");
  EXPECT_EQ(state.setup_calls, fail_step + 1);
  EXPECT_EQ(state.global_refs, 0);
  EXPECT_EQ(state.attaches.load(), 0);
  EXPECT_EQ(state.detaches.load(), 0);
  EXPECT_EQ(state.serializer_constructions.load(), 0);
  EXPECT_FALSE(state.subscribed);
  EXPECT_EQ(state.unsubscribe_calls, fail_step == 0 ? 0 : 1);
  ASSERT_EQ(State, nullptr);

  state.fail_step   = -1;
  state.setup_calls = 0;
  state.exception.clear();
  init(flush_period);
  ASSERT_TRUE(state.exception.empty()) << state.exception;
  ASSERT_NE(State, nullptr);
  EXPECT_EQ(state.setup_calls, flush_period > 0 ? 13 : 3);
  EXPECT_TRUE(state.subscribed);
  EXPECT_EQ(state.global_refs, 1);

  Java_com_nvidia_spark_rapids_jni_Profiler_nativeShutdown(&state.env, nullptr);
  EXPECT_TRUE(state.exception.empty()) << state.exception;
  EXPECT_FALSE(state.subscribed);
  EXPECT_EQ(state.global_refs, 0);
  EXPECT_EQ(state.attaches.load(), 1);
  EXPECT_EQ(state.detaches.load(), 1);
  EXPECT_EQ(state.serializer_constructions.load(), 1);
  EXPECT_EQ(state.unsubscribe_calls, fail_step == 0 ? 1 : 2);
}

std::vector<std::tuple<int, int>> failure_cases()
{
  std::vector<std::tuple<int, int>> cases;
  for (int step = 0; step < 3; ++step) {
    cases.emplace_back(0, step);
  }
  for (int step = 0; step < 13; ++step) {
    cases.emplace_back(1, step);
  }
  return cases;
}

INSTANTIATE_TEST_SUITE_P(CuptiSetupFailures,
                         ProfilerInitTest,
                         ::testing::ValuesIn(failure_cases()));

}  // namespace

namespace spark_rapids_jni::profiler {

// Keep the real writer thread and queue; serialization is outside this test's scope.
profiler_serializer::profiler_serializer(JNIEnv*, jobject writer, size_t, size_t, bool)
{
  EXPECT_EQ(writer, &Test_state->writer);
  ++Test_state->serializer_constructions;
}
void profiler_serializer::process_cupti_buffer(uint8_t*, size_t) {}
void profiler_serializer::flush() {}

}  // namespace spark_rapids_jni::profiler

extern "C" {

CUptiResult CUPTIAPI cuptiSubscribe(CUpti_SubscriberHandle* subscriber, CUpti_CallbackFunc, void*)
{
  if (Test_state->subscribed) { return CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED; }
  auto rc = Test_state->setup_result();
  if (rc == CUPTI_SUCCESS) {
    *subscriber            = reinterpret_cast<CUpti_SubscriberHandle>(Test_state);
    Test_state->subscribed = true;
  }
  return rc;
}

CUptiResult CUPTIAPI cuptiEnableCallback(uint32_t,
                                         CUpti_SubscriberHandle,
                                         CUpti_CallbackDomain,
                                         CUpti_CallbackId)
{
  return Test_state->setup_result();
}

CUptiResult CUPTIAPI cuptiActivityRegisterCallbacks(CUpti_BuffersCallbackRequestFunc request,
                                                    CUpti_BuffersCallbackCompleteFunc complete)
{
  // CUPTI callbacks may run before initialization has committed its state.
  uint8_t* buffer    = nullptr;
  size_t size        = 0;
  size_t max_records = 0;
  request(&buffer, &size, &max_records);
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(size, 0);
  complete(nullptr, 0, buffer, size, 0);
  return Test_state->setup_result();
}

CUptiResult CUPTIAPI cuptiUnsubscribe(CUpti_SubscriberHandle subscriber)
{
  EXPECT_EQ(subscriber, reinterpret_cast<CUpti_SubscriberHandle>(Test_state));
  EXPECT_TRUE(Test_state->subscribed);
  ++Test_state->unsubscribe_calls;
  Test_state->subscribed = false;
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI cuptiGetResultString(CUptiResult, char const** message)
{
  *message = "injected failure";
  return CUPTI_SUCCESS;
}

CUptiResult CUPTIAPI cuptiGetLastError() { return CUPTI_SUCCESS; }
CUptiResult CUPTIAPI cuptiActivityFlushAll(uint32_t) { return CUPTI_SUCCESS; }
CUptiResult CUPTIAPI cuptiActivityEnable(CUpti_ActivityKind) { return CUPTI_SUCCESS; }
CUptiResult CUPTIAPI cuptiActivityDisable(CUpti_ActivityKind) { return CUPTI_SUCCESS; }
CUptiResult CUPTIAPI cuptiNvtxInitialize(void*) { return CUPTI_SUCCESS; }
CUptiResult CUPTIAPI cuptiNvtxInitialize2(void*) { return CUPTI_SUCCESS; }

}  // extern "C"
