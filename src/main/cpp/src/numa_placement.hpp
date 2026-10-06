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

#pragma once

#include <cuda_runtime_api.h>

#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace spark_rapids_jni {
namespace detail {

// NUMA helpers below are best-effort placement utilities: every failure degrades
// gracefully to the default (unbound) behavior instead of failing pool creation,
// because DtoH bandwidth loss is a performance concern, not a correctness one.

/**
 * @brief Requests future page faults on [base, base + bytes) to be served
 * preferentially from the given NUMA node via mbind(MPOL_PREFERRED). Pages are
 * faulted afterwards by the pre-touch, so the policy decides their physical
 * placement. MPOL_PREFERRED (rather than MPOL_BIND) keeps the request
 * best-effort: a full target node degrades to cross-socket placement instead
 * of failing the fault. Uses the raw syscall to avoid a libnuma link
 * dependency. The nodemask covers nodes 0..63.
 *
 * @return true when the policy was applied successfully.
 */
inline bool bind_memory_to_numa_node(void* base, std::size_t bytes, int node)
{
  if (base == nullptr || bytes == 0 || node < 0 || node >= 64) { return false; }
  auto const mask = 1UL << node;
  return ::syscall(SYS_mbind, base, bytes, MPOL_PREFERRED, &mask, sizeof(mask) * 8, 0) == 0;
}

/**
 * @brief Parses a kernel cpulist file (format "0-7,32-39") into a CPU set.
 *
 * @return true when at least one CPU was added to the set.
 */
inline bool parse_cpu_list(char const* path, cpu_set_t* out)
{
  FILE* fp = std::fopen(path, "r");
  if (fp == nullptr) { return false; }
  CPU_ZERO(out);
  bool any = false;
  char line[8192];
  while (std::fgets(line, sizeof(line), fp) != nullptr) {
    for (char* tok = std::strtok(line, ",\n"); tok != nullptr; tok = std::strtok(nullptr, ",\n")) {
      unsigned int lo = 0, hi = 0;
      int const matched = std::sscanf(tok, "%u-%u", &lo, &hi);
      if (matched <= 0) { continue; }
      if (matched == 1) { hi = lo; }
      for (unsigned int cpu = lo; cpu <= hi && cpu < CPU_SETSIZE; ++cpu) {
        CPU_SET(static_cast<int>(cpu), out);
        any = true;
      }
    }
  }
  std::fclose(fp);
  return any;
}

/**
 * @brief Reads the CPU list of a NUMA node from sysfs into a CPU set.
 *
 * @return true when the node's CPU list was found and parsed.
 */
inline bool node_cpu_set(int node, cpu_set_t* out)
{
  if (node < 0) { return false; }
  char path[64];
  int const written =
    std::snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
  if (written <= 0 || static_cast<std::size_t>(written) >= sizeof(path)) { return false; }
  return parse_cpu_list(path, out);
}

/**
 * @brief Returns the NUMA node of the host address (get_mempolicy with
 * MPOL_F_NODE|MPOL_F_ADDR), or -1 when it cannot be determined. Useful for tests.
 */
inline int numa_node_of(void const* addr)
{
  int node      = -1;
  long const rc = ::syscall(
    SYS_get_mempolicy, &node, nullptr, 0, const_cast<void*>(addr), MPOL_F_NODE | MPOL_F_ADDR);
  return rc == 0 ? node : -1;
}

/**
 * @brief Returns the NUMA node hosting the current GPU's PCIe root complex via
 * sysfs, so the pageable pool can place its backing buffer local to the device.
 * cudaDeviceGetPCIBusId reports an uppercase bus id (e.g. "0000:1B:00.0") while
 * the sysfs directory is lowercase. Returns -1 when the node is unknown (e.g.
 * sysfs unavailable in a container) — callers then skip binding.
 */
inline int gpu_numa_node()
{
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    cudaGetLastError();  // clear the error state so later CUDA calls are unaffected
    return -1;
  }
  char bus_id[16];  // fits any PCI bus id "0000:1B:00.0" (12 chars) + NUL
  if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device) != cudaSuccess) {
    cudaGetLastError();
    return -1;
  }
  for (char* c = bus_id; *c != '\0'; ++c) {
    *c = static_cast<char>(std::tolower(static_cast<unsigned char>(*c)));
  }
  std::string const path = std::string("/sys/bus/pci/devices/") + bus_id + "/numa_node";
  FILE* fp               = std::fopen(path.c_str(), "r");
  if (fp == nullptr) { return -1; }
  int node = -1;
  if (std::fscanf(fp, "%d", &node) != 1) { node = -1; }
  std::fclose(fp);
  return node;
}

}  // namespace detail
}  // namespace spark_rapids_jni
