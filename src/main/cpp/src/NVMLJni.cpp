/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <jni.h>
#include <nvml.h>
#include <stdio.h>

#include <cstdint>

// NVML JNI implementation with comprehensive GPU metrics for Spark Rapids

#define NVML_CLASS_PATH "com/nvidia/spark/rapids/jni/nvml/"

// Helper functions for individual NVML API groups
namespace {

// C++ struct to hold NVML operation results
struct nvml_result {
  nvmlReturn_t return_code;
  jobject data;

  nvml_result(nvmlReturn_t return_code_, jobject data_) : return_code{return_code_}, data{data_} {}
  nvml_result() : return_code{NVML_ERROR_UNKNOWN}, data{nullptr} {}
};

// Helper function to create nvml_result Java object from C++ nvml_result
jobject create_nvml_result(JNIEnv* env, nvml_result const& cpp_result)
{
  // Create nvml_result object using default constructor
  jclass nvml_result_class = env->FindClass(NVML_CLASS_PATH "NVMLResult");
  if (nvml_result_class == nullptr) return nullptr;

  jmethodID constructor = env->GetMethodID(nvml_result_class, "<init>", "()V");
  if (constructor == nullptr) return nullptr;

  jobject java_nvml_result = env->NewObject(nvml_result_class, constructor);
  if (java_nvml_result == nullptr) return nullptr;

  // Set the return code and data fields directly
  jfieldID return_code_field = env->GetFieldID(nvml_result_class, "returnCode", "I");
  jfieldID data_field        = env->GetFieldID(nvml_result_class, "data", "Ljava/lang/Object;");

  env->SetIntField(java_nvml_result, return_code_field, static_cast<jint>(cpp_result.return_code));
  env->SetObjectField(java_nvml_result, data_field, cpp_result.data);

  return java_nvml_result;
}

jobject create_object(JNIEnv* env, char const* class_name, char const* constructor_signature)
{
  jclass j_class = env->FindClass(class_name);
  if (j_class == nullptr) { return nullptr; }

  jmethodID constructor = env->GetMethodID(j_class, "<init>", constructor_signature);
  if (constructor == nullptr) { return nullptr; }

  return env->NewObject(j_class, constructor);
}

nvml_result populate_device_info(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result{};

  jobject device_info = create_object(env, NVML_CLASS_PATH "GPUDeviceInfo", "()V");
  if (device_info == nullptr) { return result; }

  jclass device_info_class = env->GetObjectClass(device_info);

  jfieldID name_field  = env->GetFieldID(device_info_class, "name", "Ljava/lang/String;");
  jfieldID brand_field = env->GetFieldID(device_info_class, "brand", "Ljava/lang/String;");

  char name[NVML_DEVICE_NAME_BUFFER_SIZE];
  auto return_code = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
  result           = nvml_result{return_code, device_info};

  if (return_code == NVML_SUCCESS) {
    jstring j_name = env->NewStringUTF(name);
    env->SetObjectField(device_info, name_field, j_name);
    env->DeleteLocalRef(j_name);
  }

  nvmlBrandType_t brand_type;
  auto brand_return_code = nvmlDeviceGetBrand(device, &brand_type);
  if (brand_return_code == NVML_SUCCESS) {
    char brand[50];
    snprintf(brand, sizeof(brand), "Brand_%d", static_cast<int>(brand_type));
    jstring j_brand = env->NewStringUTF(brand);
    env->SetObjectField(device_info, brand_field, j_brand);
    env->DeleteLocalRef(j_brand);
  }

  return result;
}

nvml_result populate_utilization_info(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result{};

  jobject utilization_info = create_object(env, NVML_CLASS_PATH "GPUUtilizationInfo", "()V");
  if (utilization_info == nullptr) { return result; }

  jclass utilization_info_class = env->GetObjectClass(utilization_info);

  jfieldID gpu_util_field = env->GetFieldID(utilization_info_class, "gpuUtilization", "I");
  jfieldID mem_util_field = env->GetFieldID(utilization_info_class, "memoryUtilization", "I");

  nvmlUtilization_t utilization;
  auto return_code = nvmlDeviceGetUtilizationRates(device, &utilization);
  result           = nvml_result{return_code, utilization_info};

  if (return_code == NVML_SUCCESS) {
    env->SetIntField(utilization_info, gpu_util_field, static_cast<jint>(utilization.gpu));
    env->SetIntField(utilization_info, mem_util_field, static_cast<jint>(utilization.memory));
  }

  return result;
}

nvml_result populate_memory_info(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result{};

  jobject memory_info = create_object(env, NVML_CLASS_PATH "GPUMemoryInfo", "()V");
  if (memory_info == nullptr) { return result; }

  jclass memory_info_class = env->GetObjectClass(memory_info);

  jfieldID mem_used_field  = env->GetFieldID(memory_info_class, "memoryUsedMB", "J");
  jfieldID mem_total_field = env->GetFieldID(memory_info_class, "memoryTotalMB", "J");
  jfieldID mem_free_field  = env->GetFieldID(memory_info_class, "memoryFreeMB", "J");

  nvmlMemory_t memory;
  auto return_code = nvmlDeviceGetMemoryInfo(device, &memory);
  result           = nvml_result{return_code, memory_info};

  if (return_code == NVML_SUCCESS) {
    env->SetLongField(memory_info, mem_used_field, static_cast<jlong>(memory.used / (1024 * 1024)));
    env->SetLongField(
      memory_info, mem_total_field, static_cast<jlong>(memory.total / (1024 * 1024)));
    env->SetLongField(memory_info, mem_free_field, static_cast<jlong>(memory.free / (1024 * 1024)));
  }

  return result;
}

nvml_result populate_temperature_info(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result{};

  jobject temperature_info = create_object(env, NVML_CLASS_PATH "GPUTemperatureInfo", "()V");
  if (temperature_info == nullptr) { return result; }

  jclass temperature_info_class = env->GetObjectClass(temperature_info);

  jfieldID temp_gpu_field = env->GetFieldID(temperature_info_class, "temperatureGpu", "I");

  unsigned int temp;
  auto return_code = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
  result           = nvml_result{return_code, temperature_info};

  if (return_code == NVML_SUCCESS) {
    env->SetIntField(temperature_info, temp_gpu_field, static_cast<jint>(temp));
  }

  return result;
}

nvml_result populate_power_info(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result{};

  jobject power_info = create_object(env, NVML_CLASS_PATH "GPUPowerInfo", "()V");
  if (power_info == nullptr) { return result; }

  jclass power_info_class = env->GetObjectClass(power_info);

  jfieldID power_usage_field = env->GetFieldID(power_info_class, "powerUsageW", "I");
  jfieldID power_limit_field = env->GetFieldID(power_info_class, "powerLimitW", "I");

  unsigned int power;
  auto return_code = nvmlDeviceGetPowerUsage(device, &power);
  result           = nvml_result{return_code, power_info};

  if (return_code == NVML_SUCCESS) {
    env->SetIntField(power_info, power_usage_field, static_cast<jint>(power / 1000));  // mW to W
  }

  auto limit_return_code = nvmlDeviceGetPowerManagementLimit(device, &power);
  if (limit_return_code == NVML_SUCCESS) {
    env->SetIntField(power_info, power_limit_field, static_cast<jint>(power / 1000));  // mW to W
  }

  return result;
}

nvml_result populate_clock_info(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result{};

  jobject clock_info = create_object(env, NVML_CLASS_PATH "GPUClockInfo", "()V");
  if (clock_info == nullptr) { return result; }

  jclass clock_info_class = env->GetObjectClass(clock_info);

  jfieldID graphics_clock_field = env->GetFieldID(clock_info_class, "graphicsClockMHz", "I");
  jfieldID memory_clock_field   = env->GetFieldID(clock_info_class, "memoryClockMHz", "I");
  jfieldID sm_clock_field       = env->GetFieldID(clock_info_class, "smClockMHz", "I");

  unsigned int clock;
  auto return_code = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clock);
  result           = nvml_result{return_code, clock_info};

  if (return_code == NVML_SUCCESS) {
    env->SetIntField(clock_info, graphics_clock_field, static_cast<jint>(clock));
  }

  auto mem_return_code = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &clock);
  if (mem_return_code == NVML_SUCCESS) {
    env->SetIntField(clock_info, memory_clock_field, static_cast<jint>(clock));
  }

  auto sm_return_code = nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clock);
  if (sm_return_code == NVML_SUCCESS) {
    env->SetIntField(clock_info, sm_clock_field, static_cast<jint>(clock));
  }

  return result;
}

nvml_result populate_hardware_info(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result{};

  jobject hardware_info = create_object(env, NVML_CLASS_PATH "GPUHardwareInfo", "()V");
  if (hardware_info == nullptr) { return result; }

  jclass hardware_info_class = env->GetObjectClass(hardware_info);

  jfieldID sm_count_field = env->GetFieldID(hardware_info_class, "streamingMultiprocessors", "I");
  jfieldID performance_state_field = env->GetFieldID(hardware_info_class, "performanceState", "I");
  jfieldID fan_speed_field         = env->GetFieldID(hardware_info_class, "fanSpeedPercent", "I");

  unsigned int sm_count = 0;
  auto return_code      = nvmlDeviceGetNumGpuCores(device, &sm_count);
  result                = nvml_result{return_code, hardware_info};

  if (return_code == NVML_SUCCESS) {
    env->SetIntField(hardware_info, sm_count_field, static_cast<jint>(sm_count));
  }

  nvmlPstates_t p_state;
  auto pstate_return_code = nvmlDeviceGetPerformanceState(device, &p_state);
  if (pstate_return_code == NVML_SUCCESS) {
    env->SetIntField(hardware_info, performance_state_field, static_cast<jint>(p_state));
  }

  unsigned int fan_speed;
  auto fan_return_code = nvmlDeviceGetFanSpeed(device, &fan_speed);
  if (fan_return_code == NVML_SUCCESS) {
    env->SetIntField(hardware_info, fan_speed_field, static_cast<jint>(fan_speed));
  }

  return result;
}

nvml_result populate_pcie_info(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result{};

  jobject pcie_info = create_object(env, NVML_CLASS_PATH "GPUPCIeInfo", "()V");
  if (pcie_info == nullptr) { return result; }

  jclass pcie_info_class = env->GetObjectClass(pcie_info);

  jfieldID pcie_link_gen_field   = env->GetFieldID(pcie_info_class, "pcieLinkGeneration", "I");
  jfieldID pcie_link_width_field = env->GetFieldID(pcie_info_class, "pcieLinkWidth", "I");

  unsigned int link_gen;
  auto return_code = nvmlDeviceGetCurrPcieLinkGeneration(device, &link_gen);
  result           = nvml_result{return_code, pcie_info};

  if (return_code == NVML_SUCCESS) {
    env->SetIntField(pcie_info, pcie_link_gen_field, static_cast<jint>(link_gen));
  }

  unsigned int link_width;
  auto width_return_code = nvmlDeviceGetCurrPcieLinkWidth(device, &link_width);
  if (width_return_code == NVML_SUCCESS) {
    env->SetIntField(pcie_info, pcie_link_width_field, static_cast<jint>(link_width));
  }

  return result;
}

nvml_result populate_ecc_info(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result{};

  jobject ecc_info = create_object(env, NVML_CLASS_PATH "GPUECCInfo", "()V");
  if (ecc_info == nullptr) { return result; }

  jclass ecc_info_class = env->GetObjectClass(ecc_info);

  jfieldID ecc_single_bit_field = env->GetFieldID(ecc_info_class, "eccSingleBitErrors", "J");
  jfieldID ecc_double_bit_field = env->GetFieldID(ecc_info_class, "eccDoubleBitErrors", "J");

  unsigned long long eccCount;
  auto return_code =
    nvmlDeviceGetTotalEccErrors(device, NVML_SINGLE_BIT_ECC, NVML_VOLATILE_ECC, &eccCount);
  result = nvml_result{return_code, ecc_info};

  if (return_code == NVML_SUCCESS) {
    env->SetLongField(ecc_info, ecc_single_bit_field, static_cast<jlong>(eccCount));
  }

  auto double_bit_return_code =
    nvmlDeviceGetTotalEccErrors(device, NVML_DOUBLE_BIT_ECC, NVML_VOLATILE_ECC, &eccCount);
  if (double_bit_return_code == NVML_SUCCESS) {
    env->SetLongField(ecc_info, ecc_double_bit_field, static_cast<jlong>(eccCount));
  }

  return result;
}

// Helper function to populate GPUInfo object from NVML device handle using individual helpers
nvml_result populate_gpu_info_from_device(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result{};
  result.return_code = NVML_SUCCESS;  // Start with success, track first error

  // Create GPUInfo object
  jobject gpu_info = create_object(env, NVML_CLASS_PATH "GPUInfo", "()V");
  if (gpu_info == nullptr) { return result; }

  result.data           = gpu_info;
  jclass gpu_info_class = env->GetObjectClass(gpu_info);

  // Populate nested info objects using individual helpers
  nvml_result device_result      = populate_device_info(env, device);
  nvml_result utilization_result = populate_utilization_info(env, device);
  nvml_result memory_result      = populate_memory_info(env, device);
  nvml_result temperature_result = populate_temperature_info(env, device);
  nvml_result power_result       = populate_power_info(env, device);
  nvml_result clock_result       = populate_clock_info(env, device);
  nvml_result hardware_result    = populate_hardware_info(env, device);
  nvml_result pcie_result        = populate_pcie_info(env, device);
  nvml_result ecc_result         = populate_ecc_info(env, device);

  // Set nested info objects in GPUInfo
  jfieldID device_info_field =
    env->GetFieldID(gpu_info_class, "deviceInfo", "L" NVML_CLASS_PATH "GPUDeviceInfo;");
  jfieldID utilization_info_field =
    env->GetFieldID(gpu_info_class, "utilizationInfo", "L" NVML_CLASS_PATH "GPUUtilizationInfo;");
  jfieldID memory_info_field =
    env->GetFieldID(gpu_info_class, "memoryInfo", "L" NVML_CLASS_PATH "GPUMemoryInfo;");
  jfieldID temperature_info_field =
    env->GetFieldID(gpu_info_class, "temperatureInfo", "L" NVML_CLASS_PATH "GPUTemperatureInfo;");
  jfieldID power_info_field =
    env->GetFieldID(gpu_info_class, "powerInfo", "L" NVML_CLASS_PATH "GPUPowerInfo;");
  jfieldID clock_info_field =
    env->GetFieldID(gpu_info_class, "clockInfo", "L" NVML_CLASS_PATH "GPUClockInfo;");
  jfieldID hardware_info_field =
    env->GetFieldID(gpu_info_class, "hardwareInfo", "L" NVML_CLASS_PATH "GPUHardwareInfo;");
  jfieldID pcie_info_field =
    env->GetFieldID(gpu_info_class, "pcieInfo", "L" NVML_CLASS_PATH "GPUPCIeInfo;");
  jfieldID ecc_info_field =
    env->GetFieldID(gpu_info_class, "eccInfo", "L" NVML_CLASS_PATH "GPUECCInfo;");

  auto handle_result = [&](nvml_result const& field_result, jfieldID field_id) {
    if (field_result.return_code == NVML_SUCCESS) {
      env->SetObjectField(gpu_info, field_id, field_result.data);
    } else if (result.return_code == NVML_SUCCESS) {
      // Track the first error encountered
      result.return_code = field_result.return_code;
    }
    env->DeleteLocalRef(field_result.data);
  };

  handle_result(device_result, device_info_field);
  handle_result(utilization_result, utilization_info_field);
  handle_result(memory_result, memory_info_field);
  handle_result(temperature_result, temperature_info_field);
  handle_result(power_result, power_info_field);
  handle_result(clock_result, clock_info_field);
  handle_result(hardware_result, hardware_info_field);
  handle_result(pcie_result, pcie_info_field);
  handle_result(ecc_result, ecc_info_field);

  return result;
}

}  // namespace

extern "C" {

JNIEXPORT jboolean JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlInit(JNIEnv* env,
                                                                               jclass cls)
{
  nvmlReturn_t result = nvmlInit();
  return (result == NVML_SUCCESS) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlShutdown(JNIEnv* env,
                                                                               jclass cls)
{ nvmlShutdown(); }

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetDeviceCount(JNIEnv* env,
                                                                                     jclass cls)
{
  unsigned int device_count = 0;
  nvmlReturn_t result       = nvmlDeviceGetCount(&device_count);

  if (result != NVML_SUCCESS) { return -1; }

  return static_cast<jint>(device_count);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetDeviceHandleFromUUID(
  JNIEnv* env, jclass cls, jbyteArray uuid)
{
  if (uuid == nullptr) { return 0; }

  jsize uuid_len = env->GetArrayLength(uuid);
  if (uuid_len != 16) { return 0; }  // UUID should be 16 bytes

  // Get the UUID bytes from Java (raw binary format from cudaDeviceProp.uuid)
  jbyte* uuid_bytes = env->GetByteArrayElements(uuid, nullptr);
  if (uuid_bytes == nullptr) { return 0; }

  // Convert binary UUID to string format: "GPU-XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"
  char uuid_str[64];
  snprintf(uuid_str,
           sizeof(uuid_str),
           "GPU-%02hhx%02hhx%02hhx%02hhx-%02hhx%02hhx-%02hhx%02hhx-%02hhx%02hhx-%02hhx%02hhx%"
           "02hhx%02hhx%02hhx%02hhx",
           uuid_bytes[0],
           uuid_bytes[1],
           uuid_bytes[2],
           uuid_bytes[3],
           uuid_bytes[4],
           uuid_bytes[5],
           uuid_bytes[6],
           uuid_bytes[7],
           uuid_bytes[8],
           uuid_bytes[9],
           uuid_bytes[10],
           uuid_bytes[11],
           uuid_bytes[12],
           uuid_bytes[13],
           uuid_bytes[14],
           uuid_bytes[15]);

  env->ReleaseByteArrayElements(uuid, uuid_bytes, JNI_ABORT);

  // Get device handle by UUID string
  nvmlDevice_t device;
  nvmlReturn_t nvml_error = nvmlDeviceGetHandleByUUID(uuid_str, &device);

  if (nvml_error != NVML_SUCCESS) { return 0; }

  return static_cast<jlong>(
    reinterpret_cast<std::intptr_t>(device));  // Return the handle as a long
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetGPUInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));

  // Use common helper to populate GPUInfo object
  nvml_result result = populate_gpu_info_from_device(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobjectArray JNICALL
Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetAllGPUInfo(JNIEnv* env, jclass cls)
{
  // Get device count
  unsigned int device_count = 0;
  nvmlReturn_t result       = nvmlDeviceGetCount(&device_count);

  if (result != NVML_SUCCESS) {
    // Return empty array on error
    jclass nvml_result_class = env->FindClass(NVML_CLASS_PATH "NVMLResult");
    if (nvml_result_class == nullptr) { return nullptr; }
    return env->NewObjectArray(0, nvml_result_class, nullptr);
  }

  // Create array of nvml_result objects
  jclass nvml_result_class = env->FindClass(NVML_CLASS_PATH "NVMLResult");
  if (nvml_result_class == nullptr) { return nullptr; }

  jobjectArray result_array =
    env->NewObjectArray(static_cast<jsize>(device_count), nvml_result_class, nullptr);
  if (result_array == nullptr) { return nullptr; }

  // Fill array with individual nvml_result objects for each GPU
  for (unsigned int i = 0; i < device_count; i++) {
    nvmlDevice_t device;
    nvmlReturn_t device_result = nvmlDeviceGetHandleByIndex(i, &device);

    nvml_result cpp_result;
    if (device_result == NVML_SUCCESS) {
      // Successfully got device handle, try to populate GPU info
      cpp_result = populate_gpu_info_from_device(env, device);
    } else {
      // Failed to get device handle
      cpp_result = nvml_result{device_result, nullptr};
    }

    // Create Java nvml_result object and add to array
    jobject java_result = create_nvml_result(env, cpp_result);
    if (java_result != nullptr) {
      env->SetObjectArrayElement(result_array, static_cast<jsize>(i), java_result);
      env->DeleteLocalRef(java_result);
    }
  }

  return result_array;
}

// Individual info JNI functions using device handles
JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetDeviceInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  nvml_result result  = populate_device_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetUtilizationInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  nvml_result result  = populate_utilization_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetMemoryInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  nvml_result result  = populate_memory_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetTemperatureInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  nvml_result result  = populate_temperature_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetPowerInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  nvml_result result  = populate_power_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetClockInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  nvml_result result  = populate_clock_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetHardwareInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  nvml_result result  = populate_hardware_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetPCIeInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  nvml_result result  = populate_pcie_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetECCInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  nvml_result result  = populate_ecc_info(env, device);
  return create_nvml_result(env, result);
}

}  // extern "C"
