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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.ConcurrentHashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * High-level GPU monitoring service that provides continuous monitoring,
 * lifecycle statistics tracking, and callback support for GPU metrics.
 */
public class NVMLMonitor {
    private static final Logger logger = LoggerFactory.getLogger(NVMLMonitor.class);

    // Instance variables for monitoring state
    private final AtomicBoolean monitoring = new AtomicBoolean(false);
    private Thread monitoringThread;
    private final int intervalMs;
    private final boolean verbose;

    // Lifecycle statistics for each GPU
    private final Map<Integer, GPULifecycleStats> lifecycleStats = new ConcurrentHashMap<>();

    // Callback for real-time monitoring
    private volatile MonitoringCallback callback;

    /**
     * Callback interface for real-time GPU monitoring events
     */
    public interface MonitoringCallback {
        void onGPUUpdate(GPUInfo[] gpuInfos, long timestamp);
        void onMonitoringStarted();
        void onMonitoringStopped();
        void onError(String error);
    }

    /**
     * Create a new monitoring service with specified interval and verbosity
     */
    public NVMLMonitor(int intervalMs, boolean verbose) {
        this.intervalMs = intervalMs;
        this.verbose = verbose;
    }

    /**
     * Create a new monitoring service with default settings (1 second interval, not verbose)
     */
    public NVMLMonitor() {
        this(1000, false);
    }

    /**
     * Set callback for monitoring events
     */
    public void setCallback(MonitoringCallback callback) {
        this.callback = callback;
    }

    /**
     * Start continuous GPU monitoring
     */
    public synchronized void startMonitoring() {
        if (monitoring.get()) {
            return; // Already monitoring
        }

        if (!NVML.isAvailable()) {
            logger.error("NVML not available, cannot start monitoring");
            return;
        }

        // Initialize lifecycle stats for all GPUs
        NVMLResult<GPUInfo>[] initialResults = NVML.getAllGPUInfo();
        for (int i = 0; i < initialResults.length; i++) {
            NVMLResult<GPUInfo> result = initialResults[i];
            if (result.getData() != null && result.getData().deviceInfo != null) {
                lifecycleStats.put(i, new GPULifecycleStats(result.getData().deviceInfo.name));
            }
        }

        monitoring.set(true);
        monitoringThread = new Thread(this::monitoringLoop, "NVMLMonitor");
        monitoringThread.setDaemon(true);
        monitoringThread.start();

        if (verbose) {
            logger.info("Started NVML GPU monitoring (interval: " + intervalMs + "ms)");
        }

        if (callback != null) {
            callback.onMonitoringStarted();
        }
    }

    /**
     * Stop continuous GPU monitoring
     */
    public synchronized void stopMonitoring() {
        if (!monitoring.get()) {
            return; // Not monitoring
        }

        monitoring.set(false);
        if (monitoringThread != null) {
            try {
                monitoringThread.join(2000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        if (verbose) {
            logger.info("Stopped NVML GPU monitoring");
        }

        if (callback != null) {
            callback.onMonitoringStopped();
        }
    }

    /**
     * Check if monitoring is currently active
     */
    public boolean isMonitoring() {
        return monitoring.get();
    }

    /**
     * Main monitoring loop (runs in background thread)
     */
    private void monitoringLoop() {
        while (monitoring.get()) {
            try {
                NVMLResult<GPUInfo>[] results = NVML.getAllGPUInfo();

                if (results == null || results.length == 0) {
                    Thread.sleep(intervalMs);
                    continue;
                }

                // Extract successful GPUInfo objects
                List<GPUInfo> successfulInfos = new ArrayList<>();
                for (NVMLResult<GPUInfo> result : results) {
                    // We will still add nulls to the list to maintain the same index for the lifecycle stats
                    // but they will be ignored in the stats update.
                    successfulInfos.add(result.getData());
                }
                GPUInfo[] gpuInfos = successfulInfos.toArray(new GPUInfo[0]);

                if (gpuInfos.length == 0) {
                    Thread.sleep(intervalMs);
                    continue;
                }

                long timestamp = System.currentTimeMillis();

                // Update lifecycle statistics
                for (int i = 0; i < gpuInfos.length; i++) {
                    GPUInfo info = gpuInfos[i];
                    if (info != null && info.deviceInfo != null) {
                        GPULifecycleStats stats = lifecycleStats.get(i);
                        if (stats != null) {
                            stats.addSample(info);
                        }
                    }
                }

                // Verbose output
                if (verbose) {
                    logger.info("=== GPU Status Update ===");
                    for (GPUInfo info : gpuInfos) {
                        logger.info(info.toString());
                    }
                    logger.info("");
                }

                // Callback notification
                if (callback != null) {
                    callback.onGPUUpdate(gpuInfos, timestamp);
                }

                Thread.sleep(intervalMs);

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                String error = "Error in monitoring loop: " + e.getMessage();
                logger.error(error);
                if (callback != null) {
                    callback.onError(error);
                }
            }
        }
    }

    /**
     * Get lifecycle statistics for all GPUs
     */
    public Map<Integer, GPULifecycleStats> getLifecycleStats() {
        return new HashMap<>(lifecycleStats);
    }

    /**
     * Get lifecycle statistics for specific GPU by array index
     */
    public GPULifecycleStats getLifecycleStats(int arrayIndex) {
        return lifecycleStats.get(arrayIndex);
    }

    /**
     * Print comprehensive lifecycle statistics report
     */
    public void printLifecycleReport() {
        printLifecycleReport("NVML GPU MONITORING");
    }

    /**
     * Print comprehensive lifecycle statistics report with custom lifecycle name
     * @param lifecycleName the name of the lifecycle to display in the report
     */
    public void printLifecycleReport(String lifecycleName) {
        if (lifecycleStats.isEmpty()) {
            logger.info("No lifecycle statistics available for: " + lifecycleName);
            return;
        }

        StringBuilder report = new StringBuilder();

        // Header with summary in same line
        List<String> summaries = new ArrayList<>();
        for (GPULifecycleStats stats : lifecycleStats.values()) {
            summaries.add(stats.getSummary());
        }
        report.append(lifecycleName).append(" - LIFECYCLE REPORT: ")
              .append(String.join(" | ", summaries)).append("\n");

        // Detailed report for each GPU
        for (GPULifecycleStats stats : lifecycleStats.values()) {
            report.append("\n").append(stats.generateReport());
        }

        report.append("\n");

        // Output the entire report as one log message to avoid multiple logger prefixes
        logger.info(report.toString());
    }

    /**
     * Get monitoring duration in seconds
     */
    public double getMonitoringDurationSeconds() {
        if (lifecycleStats.isEmpty()) {
            return 0.0;
        }

        return lifecycleStats.values().iterator().next().getMonitoringDurationSeconds();
    }
}
