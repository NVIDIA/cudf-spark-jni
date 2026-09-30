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

package com.nvidia.spark.rapids.jni.nvml;

import ai.rapids.cudf.Cuda;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static com.nvidia.spark.rapids.jni.nvml.NVMLReturnCode.SUCCESS;
import static com.nvidia.spark.rapids.jni.nvml.NVMLReturnCode.ERROR_NOT_SUPPORTED;

/**
 * Test class for NVML JNI wrapper functionality.
 * Tests require a CUDA-capable GPU to run.
 * Tests pass regardless of NVML availability - they only validate NVML functionality when available.
 */
@EnabledIfSystemProperty(named = "ai.rapids.cudf.nvml.test.enabled", matches = "true",
    disabledReason = "NVML tests require GPU and NVML library")
public class NVMLTest {

  private static boolean nvmlAvailable = false;

  @BeforeAll
  public static void setup() {
    // Set CUDA device to 0 before running tests
    Cuda.setDevice(0);

    // Try to initialize NVML - tests should pass regardless of success/failure
    boolean initResult = NVML.initialize();
    nvmlAvailable = NVML.isAvailable();

    System.out.println("NVML initialization result: " + initResult + ", available: " + nvmlAvailable);
  }

  @AfterAll
  public static void teardown() {
    // Shutdown NVML after all tests
    NVML.shutdown();
  }

  @Test
  public void testGetDeviceCount() {
    int deviceCount = NVML.getDeviceCount();
    if (nvmlAvailable) {
      assertTrue(deviceCount > 0, "Device count should be greater than 0 when NVML is available");
      System.out.println("Found " + deviceCount + " GPU device(s)");
    } else {
      // When NVML is not available, getDeviceCount returns 0
      assertEquals(0, deviceCount, "Device count should be 0 when NVML is not available");
    }
  }

  @Test
  public void testGetDeviceInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    NVMLResult<GPUDeviceInfo> result = NVML.getDeviceInfo(uuid);

    assertNotNull(result, "Result should not be null");

    if (nvmlAvailable) {
      // Device info should generally be supported, but accept both SUCCESS and NOT_SUPPORTED
      NVMLReturnCode returnCode = result.getReturnCodeEnum();
      assertTrue(returnCode == SUCCESS || returnCode == ERROR_NOT_SUPPORTED,
                 "Return code should be SUCCESS or NOT_SUPPORTED when NVML is available, got: " + returnCode);

      if (result.isSuccess()) {
        // Only validate data if the operation succeeded
        GPUDeviceInfo deviceInfo = result.getData();
        assertNotNull(deviceInfo, "Device info should not be null");
        assertNotNull(deviceInfo.name, "Device name should not be null");
        assertNotNull(deviceInfo.brand, "Device brand should not be null");
      }
    } else {
      // When NVML is not available, return code should be -1 (unknown error)
      assertEquals(-1, result.getReturnCode(), "Return code should be -1 when NVML is not available");
      assertNull(result.getData(), "Data should be null when NVML is not available");
    }
  }

  @Test
  public void testGetUtilizationInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    NVMLResult<GPUUtilizationInfo> result = NVML.getUtilizationInfo(uuid);

    assertNotNull(result, "Result should not be null");

    if (nvmlAvailable) {
      // Utilization info should generally be supported, but accept both SUCCESS and NOT_SUPPORTED
      NVMLReturnCode returnCode = result.getReturnCodeEnum();
      assertTrue(returnCode == SUCCESS || returnCode == ERROR_NOT_SUPPORTED,
                 "Return code should be SUCCESS or NOT_SUPPORTED when NVML is available, got: " + returnCode);

      if (result.isSuccess()) {
        // Only validate data if the operation succeeded
        GPUUtilizationInfo utilizationInfo = result.getData();
        assertNotNull(utilizationInfo, "Utilization info should not be null");
        assertTrue(utilizationInfo.gpuUtilization >= 0 && utilizationInfo.gpuUtilization <= 100,
            "GPU utilization should be between 0 and 100");
        assertTrue(utilizationInfo.memoryUtilization >= 0 && utilizationInfo.memoryUtilization <= 100,
            "Memory utilization should be between 0 and 100");
      }
    } else {
      // When NVML is not available, return code should be -1
      assertEquals(-1, result.getReturnCode(), "Return code should be -1 when NVML is not available");
      assertNull(result.getData(), "Data should be null when NVML is not available");
    }
  }

  @Test
  public void testGetMemoryInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    NVMLResult<GPUMemoryInfo> result = NVML.getMemoryInfo(uuid);

    assertNotNull(result, "Result should not be null");

    if (nvmlAvailable) {
      // Memory info should generally be supported, but accept both SUCCESS and NOT_SUPPORTED
      NVMLReturnCode returnCode = result.getReturnCodeEnum();
      assertTrue(returnCode == SUCCESS || returnCode == ERROR_NOT_SUPPORTED,
                 "Return code should be SUCCESS or NOT_SUPPORTED when NVML is available, got: " + returnCode);

      if (result.isSuccess()) {
        // Only validate data if the operation succeeded
        GPUMemoryInfo memoryInfo = result.getData();
        assertNotNull(memoryInfo, "Memory info should not be null");
        assertTrue(memoryInfo.memoryTotalMB > 0, "Total memory should be greater than 0");
        assertTrue(memoryInfo.memoryUsedMB >= 0, "Used memory should be non-negative");
        assertTrue(memoryInfo.memoryFreeMB >= 0, "Free memory should be non-negative");
        assertTrue(memoryInfo.memoryUsedMB + memoryInfo.memoryFreeMB <= memoryInfo.memoryTotalMB,
            "Used + Free should not exceed Total memory");
      }
    } else {
      // When NVML is not available, return code should be -1
      assertEquals(-1, result.getReturnCode(), "Return code should be -1 when NVML is not available");
      assertNull(result.getData(), "Data should be null when NVML is not available");
    }
  }

  @Test
  public void testGetTemperatureInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    NVMLResult<GPUTemperatureInfo> result = NVML.getTemperatureInfo(uuid);

    assertNotNull(result, "Result should not be null");

    if (nvmlAvailable) {
      // Temperature info may not be supported on all GPUs, but accept both SUCCESS and NOT_SUPPORTED
      NVMLReturnCode returnCode = result.getReturnCodeEnum();
      assertTrue(returnCode == SUCCESS || returnCode == ERROR_NOT_SUPPORTED,
                 "Return code should be SUCCESS or NOT_SUPPORTED when NVML is available, got: " + returnCode);

      if (result.isSuccess()) {
        // Only validate data if the operation succeeded
        GPUTemperatureInfo temperatureInfo = result.getData();
        assertNotNull(temperatureInfo, "Temperature info should not be null");
        assertTrue(temperatureInfo.temperatureGpu > 0 && temperatureInfo.temperatureGpu < 150,
            "GPU temperature should be in reasonable range (0-150°C)");
      }
    } else {
      // When NVML is not available, return code should be -1
      assertEquals(-1, result.getReturnCode(), "Return code should be -1 when NVML is not available");
      assertNull(result.getData(), "Data should be null when NVML is not available");
    }
  }

  @Test
  public void testGetPowerInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    NVMLResult<GPUPowerInfo> result = NVML.getPowerInfo(uuid);

    assertNotNull(result, "Result should not be null");

    if (nvmlAvailable) {
      // Power info may not be supported on all GPUs, but accept both SUCCESS and NOT_SUPPORTED
      NVMLReturnCode returnCode = result.getReturnCodeEnum();
      assertTrue(returnCode == SUCCESS || returnCode == ERROR_NOT_SUPPORTED,
                 "Return code should be SUCCESS or NOT_SUPPORTED when NVML is available, got: " + returnCode);

      if (result.isSuccess()) {
        // Only validate data if the operation succeeded
        GPUPowerInfo powerInfo = result.getData();
        assertNotNull(powerInfo, "Power info should not be null");
        assertTrue(powerInfo.powerUsageW >= 0, "Power usage should be non-negative");
        assertTrue(powerInfo.powerLimitW > 0, "Power limit should be greater than 0");
        assertTrue(powerInfo.powerUsageW <= powerInfo.powerLimitW * 2,
            "Power usage should be within reasonable range of power limit");
      }
    } else {
      // When NVML is not available, return code should be -1
      assertEquals(-1, result.getReturnCode(), "Return code should be -1 when NVML is not available");
      assertNull(result.getData(), "Data should be null when NVML is not available");
    }
  }

  @Test
  public void testGetClockInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    NVMLResult<GPUClockInfo> result = NVML.getClockInfo(uuid);

    assertNotNull(result, "Result should not be null");

    if (nvmlAvailable) {
      // Clock info may not be supported on all GPUs, but accept both SUCCESS and NOT_SUPPORTED
      NVMLReturnCode returnCode = result.getReturnCodeEnum();
      assertTrue(returnCode == SUCCESS || returnCode == ERROR_NOT_SUPPORTED,
                 "Return code should be SUCCESS or NOT_SUPPORTED when NVML is available, got: " + returnCode);

      if (result.isSuccess()) {
        // Only validate data if the operation succeeded
        GPUClockInfo clockInfo = result.getData();
        assertNotNull(clockInfo, "Clock info should not be null");
        assertTrue(clockInfo.graphicsClockMHz >= 0, "Graphics clock should be non-negative");
        assertTrue(clockInfo.memoryClockMHz >= 0, "Memory clock should be non-negative");
        assertTrue(clockInfo.smClockMHz >= 0, "SM clock should be non-negative");
      }
    } else {
      // When NVML is not available, return code should be -1
      assertEquals(-1, result.getReturnCode(), "Return code should be -1 when NVML is not available");
      assertNull(result.getData(), "Data should be null when NVML is not available");
    }
  }

  @Test
  public void testGetHardwareInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    NVMLResult<GPUHardwareInfo> result = NVML.getHardwareInfo(uuid);

    assertNotNull(result, "Result should not be null");

    if (nvmlAvailable) {
      // Hardware info may not be supported on all GPUs, but accept both SUCCESS and NOT_SUPPORTED
      NVMLReturnCode returnCode = result.getReturnCodeEnum();
      assertTrue(returnCode == SUCCESS || returnCode == ERROR_NOT_SUPPORTED,
                 "Return code should be SUCCESS or NOT_SUPPORTED when NVML is available, got: " + returnCode);

      if (result.isSuccess()) {
        // Only validate data if the operation succeeded
        GPUHardwareInfo hardwareInfo = result.getData();
        assertNotNull(hardwareInfo, "Hardware info should not be null");
        assertTrue(hardwareInfo.streamingMultiprocessors > 0, "SM count should be greater than 0");
        assertTrue(hardwareInfo.performanceState >= 0 && hardwareInfo.performanceState <= 32,
            "Performance state should be in valid range");
        assertTrue(hardwareInfo.fanSpeedPercent >= 0 && hardwareInfo.fanSpeedPercent <= 100,
            "Fan speed should be between 0 and 100");
      }
    } else {
      // When NVML is not available, return code should be -1
      assertEquals(-1, result.getReturnCode(), "Return code should be -1 when NVML is not available");
      assertNull(result.getData(), "Data should be null when NVML is not available");
    }
  }

  @Test
  public void testGetPCIeInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    NVMLResult<GPUPCIeInfo> result = NVML.getPCIeInfo(uuid);

    assertNotNull(result, "Result should not be null");

    if (nvmlAvailable) {
      // PCIe info should generally be supported, but accept both SUCCESS and NOT_SUPPORTED
      NVMLReturnCode returnCode = result.getReturnCodeEnum();
      assertTrue(returnCode == SUCCESS || returnCode == ERROR_NOT_SUPPORTED,
                 "Return code should be SUCCESS or NOT_SUPPORTED when NVML is available, got: " + returnCode);

      if (result.isSuccess()) {
        // Only validate data if the operation succeeded
        GPUPCIeInfo pcieInfo = result.getData();
        assertNotNull(pcieInfo, "PCIe info should not be null");
        assertTrue(pcieInfo.pcieLinkGeneration > 0, "PCIe link generation should be greater than 0");
        assertTrue(pcieInfo.pcieLinkWidth > 0, "PCIe link width should be greater than 0");
      }
    } else {
      // When NVML is not available, return code should be -1
      assertEquals(-1, result.getReturnCode(), "Return code should be -1 when NVML is not available");
      assertNull(result.getData(), "Data should be null when NVML is not available");
    }
  }

  @Test
  public void testGetECCInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    NVMLResult<GPUECCInfo> result = NVML.getECCInfo(uuid);

    assertNotNull(result, "Result should not be null");

    if (nvmlAvailable) {
      // ECC may not be supported on all GPUs, so accept both SUCCESS and NOT_SUPPORTED
      NVMLReturnCode returnCode = result.getReturnCodeEnum();
      assertTrue(returnCode == SUCCESS || returnCode == ERROR_NOT_SUPPORTED,
                 "Return code should be SUCCESS or NOT_SUPPORTED when NVML is available, got: " + returnCode);

      if (result.isSuccess()) {
        // Only validate data if the operation succeeded
        GPUECCInfo eccInfo = result.getData();
        assertNotNull(eccInfo, "ECC info should not be null");
        assertTrue(eccInfo.eccSingleBitErrors >= 0, "Single-bit ECC errors should be non-negative");
        assertTrue(eccInfo.eccDoubleBitErrors >= 0, "Double-bit ECC errors should be non-negative");
      }
    } else {
      // When NVML is not available, return code should be -1
      assertEquals(-1, result.getReturnCode(), "Return code should be -1 when NVML is not available");
      assertNull(result.getData(), "Data should be null when NVML is not available");
    }
  }

  @Test
  public void testGetGPUInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    NVMLResult<GPUInfo> result = NVML.getGPUInfo(uuid);

    assertNotNull(result, "Result should not be null");

    if (nvmlAvailable) {
      // GPU info should generally be supported, but accept both SUCCESS and NOT_SUPPORTED
      NVMLReturnCode returnCode = result.getReturnCodeEnum();
      assertTrue(returnCode == SUCCESS || returnCode == ERROR_NOT_SUPPORTED,
                 "Return code should be SUCCESS or NOT_SUPPORTED when NVML is available, got: " + returnCode);

      if (result.isSuccess()) {
        // Only validate data if the operation succeeded
        GPUInfo gpuInfo = result.getData();
        assertNotNull(gpuInfo, "GPU info should not be null");
        assertNotNull(gpuInfo.deviceInfo, "Device info should not be null");
        assertNotNull(gpuInfo.utilizationInfo, "Utilization info should not be null");
        assertNotNull(gpuInfo.memoryInfo, "Memory info should not be null");
        assertNotNull(gpuInfo.temperatureInfo, "Temperature info should not be null");
        assertNotNull(gpuInfo.powerInfo, "Power info should not be null");
        assertNotNull(gpuInfo.clockInfo, "Clock info should not be null");
        assertNotNull(gpuInfo.hardwareInfo, "Hardware info should not be null");
        assertNotNull(gpuInfo.pcieInfo, "PCIe info should not be null");
        assertNotNull(gpuInfo.eccInfo, "ECC info should not be null");
      }
    } else {
      // When NVML is not available, return code should be -1
      assertEquals(-1, result.getReturnCode(), "Return code should be -1 when NVML is not available");
      assertNull(result.getData(), "Data should be null when NVML is not available");
    }
  }

  @Test
  public void testGetAllGPUInfo() {
    NVMLResult<GPUInfo>[] results = NVML.getAllGPUInfo();

    assertNotNull(results, "All GPU info results array should not be null");

    if (nvmlAvailable) {
      assertTrue(results.length > 0, "Should have at least one GPU result when NVML is available");

      // Extract successful GPUInfo objects
      List<GPUInfo> allGPUInfo = new ArrayList<>();
      for (NVMLResult<GPUInfo> result : results) {
        if (result.isSuccess() && result.getData() != null) {
          allGPUInfo.add(result.getData());
        }
      }

      System.out.println("===== All GPUs Info =====");
      for (int i = 0; i < allGPUInfo.size(); i++) {
        GPUInfo info = allGPUInfo.get(i);
        assertNotNull(info, "GPU info at index " + i + " should not be null");
        assertNotNull(info.deviceInfo, "Device info at index " + i + " should not be null");

        System.out.println("GPU " + i + ": " + info.deviceInfo.name);
        System.out.println("  Memory: " + info.memoryInfo.memoryUsedMB + "/" +
                           info.memoryInfo.memoryTotalMB + " MB");
        System.out.println("  Temperature: " + info.temperatureInfo.temperatureGpu + "°C");
      }
    } else {
      // When NVML is not available, results array should be empty or contain error results
      if (results.length > 0) {
        // If there are results, they should all have return code -1
        for (NVMLResult<GPUInfo> result : results) {
          assertEquals(-1, result.getReturnCode(),
                       "Return code should be -1 when NVML is not available");
          assertNull(result.getData(), "Data should be null when NVML is not available");
        }
      }
    }
  }

  @Test
  public void testUUIDBasedMethods() {
    // Test UUID-based methods using current CUDA device UUID
    byte[] uuid = Cuda.getGpuUuid();
    assertNotNull(uuid, "GPU UUID should not be null");

    // Test each method with relaxed expectations (SUCCESS or NOT_SUPPORTED)
    testMethodWithRelaxedExpectations("Device Info", NVML.getDeviceInfo(uuid));
    testMethodWithRelaxedExpectations("Utilization Info", NVML.getUtilizationInfo(uuid));
    testMethodWithRelaxedExpectations("Memory Info", NVML.getMemoryInfo(uuid));
    testMethodWithRelaxedExpectations("Temperature Info", NVML.getTemperatureInfo(uuid));
    testMethodWithRelaxedExpectations("Power Info", NVML.getPowerInfo(uuid));
    testMethodWithRelaxedExpectations("Clock Info", NVML.getClockInfo(uuid));
    testMethodWithRelaxedExpectations("Hardware Info", NVML.getHardwareInfo(uuid));
    testMethodWithRelaxedExpectations("PCIe Info", NVML.getPCIeInfo(uuid));
    testMethodWithRelaxedExpectations("ECC Info", NVML.getECCInfo(uuid));
    testMethodWithRelaxedExpectations("GPU Info", NVML.getGPUInfo(uuid));
  }

  private void testMethodWithRelaxedExpectations(String methodName, NVMLResult<?> result) {
    assertNotNull(result, methodName + " result should not be null");

    if (nvmlAvailable) {
      NVMLReturnCode returnCode = result.getReturnCodeEnum();
      assertTrue(returnCode == SUCCESS || returnCode == ERROR_NOT_SUPPORTED,
                 methodName + " return code should be SUCCESS or NOT_SUPPORTED when NVML is available, got: " + returnCode);
    } else {
      // When NVML is not available, return code should be -1
      assertEquals(-1, result.getReturnCode(),
                   methodName + " return code should be -1 when NVML is not available");
      assertNull(result.getData(), methodName + " data should be null when NVML is not available");
    }
  }

  /**
   * Helper method to validate NVML result when NVML is available
   */
  private <T> void validateNVMLResultWhenAvailable(NVMLResult<T> result, String methodName) {
    if (nvmlAvailable) {
      NVMLReturnCode returnCode = result.getReturnCodeEnum();
      assertTrue(returnCode == SUCCESS || returnCode == ERROR_NOT_SUPPORTED,
                 methodName + " return code should be SUCCESS or NOT_SUPPORTED when NVML is available, got: " + returnCode);
    } else {
      assertEquals(-1, result.getReturnCode(),
                   methodName + " return code should be -1 when NVML is not available");
      assertNull(result.getData(), methodName + " data should be null when NVML is not available");
    }
  }
}

