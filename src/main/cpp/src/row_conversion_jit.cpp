/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include "row_conversion_jit.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>
#include <rtcx.hpp>
#include <strings.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <future>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace spark_rapids_jni {
namespace detail {

namespace {

using cudf::size_type;

// The generated kernels mirror the generic kernels' launch shape: 1024 threads = 32 warps.
constexpr int JIT_BLOCK_SIZE = 1024;

// Beyond this the generated straight-line body stops paying for its compile time; the bench wide
// case (320 columns) is multi-band and never reaches the JIT anyway.
constexpr std::size_t JIT_MAX_COLUMNS = 512;

// Compiled kernels are never evicted (unloading a module that may still have launches in flight
// is the hazard), so cap insertions instead; schemas past the cap use the generic kernel.
constexpr std::size_t JIT_CACHE_MAX_ENTRIES = 128;

bool jit_enabled()
{
  static bool const enabled = [] {
    char const* env = std::getenv("SPARK_RAPIDS_ROWCONV_JIT");
    if (env == nullptr) { return true; }
    // The kill switch has to be usable during an incident, so tolerate the whitespace an env var
    // picks up on its way through a container spec, accept the common spellings of "no", and say
    // something when the value is none of them rather than silently staying on.
    std::string value{env};
    auto const first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) { return true; }
    value = value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
    for (auto const* word : {"0", "false", "off", "no", "disable", "disabled"}) {
      if (::strcasecmp(value.c_str(), word) == 0) { return false; }
    }
    std::fprintf(stderr,
                 "[spark-rapids-jni] SPARK_RAPIDS_ROWCONV_JIT=\"%s\" is not a "
                 "recognized value; the row-conversion JIT stays enabled\n",
                 env);
    return true;
  }();
  return enabled;
}

struct device_env {
  int ordinal;
  int compute_capability;  // major * 10 + minor
  int driver_version;
  int runtime_version;
};

std::optional<device_env> query_device_env()
{
  device_env env{};
  int major = 0;
  int minor = 0;
  if (cudaGetDevice(&env.ordinal) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, env.ordinal) !=
        cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, env.ordinal) !=
        cudaSuccess ||
      cudaDriverGetVersion(&env.driver_version) != cudaSuccess ||
      cudaRuntimeGetVersion(&env.runtime_version) != cudaSuccess) {
    return std::nullopt;
  }
  env.compute_capability = major * 10 + minor;
  return env;
}

// The generated code copies each element with one full-width typed access, so every start must be
// aligned to the access width (16-byte columns are accessed as two 8-byte halves on the row side).
bool layout_supported(std::vector<size_type> const& column_sizes,
                      std::vector<size_type> const& column_starts,
                      size_type row_stride)
{
  auto const num_columns = column_sizes.size();
  if (num_columns == 0 || num_columns > JIT_MAX_COLUMNS) { return false; }
  if (column_starts.size() != num_columns + 1) { return false; }
  for (std::size_t i = 0; i < num_columns; ++i) {
    auto const size = column_sizes[i];
    if (size != 1 && size != 2 && size != 4 && size != 8 && size != 16) { return false; }
    auto const access_width = size < 8 ? size : 8;
    if (column_starts[i] % access_width != 0) { return false; }
  }
  // The generated kernels address rows as `input + row * row_stride` and copy them eight bytes at
  // a time, so the stride itself has to keep every row 8-byte aligned.
  if (row_stride % 8 != 0) { return false; }
  return column_starts[num_columns - 1] + column_sizes[num_columns - 1] <= row_stride;
}

std::string make_cache_key(char direction,
                           device_env const& env,
                           size_type row_stride,
                           std::vector<size_type> const& column_sizes,
                           std::vector<size_type> const& column_starts)
{
  std::string key;
  key.reserve(32 + (column_sizes.size() + column_starts.size() + 6) * sizeof(int));
  key.append("SRJC6v1");
  key.push_back(direction);
  auto const append_int = [&key](int v) {
    key.append(reinterpret_cast<char const*>(&v), sizeof(v));
  };
  append_int(env.compute_capability);
  append_int(env.driver_version);
  append_int(env.runtime_version);
  append_int(JIT_BLOCK_SIZE);
  append_int(row_stride);
  append_int(static_cast<int>(column_sizes.size()));
  for (auto const v : column_sizes) {
    append_int(v);
  }
  for (auto const v : column_starts) {
    append_int(v);
  }
  return key;
}

char const* type_for_width(size_type width)
{
  switch (width) {
    case 1: return "char";
    case 2: return "short";
    case 4: return "int";
    default: return "long long";
  }
}

// Shared prologue of the generated source. srj_tile mirrors detail::tile_info (asserted at the
// call sites); srj_v16 reproduces the generic kernel's single 16-byte decimal128 store.
constexpr char const* JIT_SOURCE_PROLOGUE =
  "struct srj_tile { int start_col; int start_row; int end_col; int end_row; int batch_number; "
  "};\n"
  "struct alignas(16) srj_v16 { long long lo; long long hi; };\n";

/**
 * @brief Generate the from-rows gather kernel for one layout
 *
 * Mirrors the generic `copy_from_rows` work split — warp w handles columns w, w+32, ... and its
 * lanes stride the tile's rows — but the column sweep is emitted as straight-line code per warp
 * with all offsets, widths and the row stride as immediates.
 */
std::string make_from_rows_source(std::vector<size_type> const& column_sizes,
                                  std::vector<size_type> const& column_starts,
                                  size_type row_stride)
{
  auto const num_columns = static_cast<int>(column_sizes.size());
  std::string src{JIT_SOURCE_PROLOGUE};
  src += std::format(
    "extern \"C\" __global__ void __launch_bounds__({}) srj_rowconv_specialized(\n"
    "  srj_tile const* tiles, char const* __restrict__ input, char* const* output)\n"
    "{{\n"
    "  srj_tile const t = tiles[blockIdx.x];\n"
    "  int const lane   = static_cast<int>(threadIdx.x & 31u);\n"
    "  int const warp   = static_cast<int>(threadIdx.x >> 5u);\n"
    "  int const nrows  = t.end_row - t.start_row + 1;\n"
    "  switch (warp) {{\n",
    JIT_BLOCK_SIZE);
  for (int w = 0; w < 32 && w < num_columns; ++w) {
    src += std::format("    case {}: {{\n", w);
    for (int c = w; c < num_columns; c += 32) {
      src += std::format("      char* __restrict__ o{0} = output[{0}];\n", c);
    }
    src +=
      "      for (int rr = lane; rr < nrows; rr += 32) {\n"
      "        int const row = t.start_row + rr;\n";
    src += std::format("        char const* __restrict__ rp = input + row * {};\n", row_stride);
    for (int c = w; c < num_columns; c += 32) {
      auto const size  = column_sizes[c];
      auto const start = column_starts[c];
      if (size == 1) {
        src += std::format("        o{0}[row] = rp[{1}];\n", c, start);
      } else if (size == 16) {
        // Rows are only 8-byte aligned: two 8-byte reads, one 16-byte store.
        src += std::format(
          "        {{ long long const* p{0} = reinterpret_cast<long long const*>(rp + {1});\n"
          "          *reinterpret_cast<srj_v16*>(o{0} + row * 16) = srj_v16{{p{0}[0], p{0}[1]}}; "
          "}}\n",
          c,
          start);
      } else {
        src += std::format(
          "        *reinterpret_cast<{2}*>(o{0} + row * {3}) = *reinterpret_cast<{2} "
          "const*>(rp + {1});\n",
          c,
          start,
          type_for_width(size),
          size);
      }
    }
    src += "      }\n    } break;\n";
  }
  src += "  }\n}\n";
  return src;
}

/**
 * @brief Generate the to-rows kernel for one layout
 *
 * Phase one mirrors the generic `copy_to_rows` element staging (warp w owns columns w, w+32, ...)
 * into a shared tile with a compile-time padded row width; phase two copies each staged row to
 * global memory with a warp-cooperative loop whose length is the compile-time un-padded row data
 * width. Both kernels drain with plain word stores; the generated one additionally drops the
 * generic kernel's live-column bookkeeping and the block sync that publishes it, keeping only
 * the stage-to-drain barrier.
 */
std::string make_to_rows_source(std::vector<size_type> const& column_sizes,
                                std::vector<size_type> const& column_starts,
                                size_type row_stride)
{
  auto const num_columns = static_cast<int>(column_sizes.size());
  // Un-padded data width of a full-row band, and its shared-memory pitch; see
  // tile_info::get_actual_row_size / get_shared_row_size.
  auto const actual_row_size = column_starts[num_columns - 1] + column_sizes[num_columns - 1];
  auto const shared_row_size = (actual_row_size + 7) & ~7;
  std::string src{JIT_SOURCE_PROLOGUE};
  src += std::format(
    "extern \"C\" __global__ void __launch_bounds__({}) srj_rowconv_specialized(\n"
    "  srj_tile const* tiles, char const* const* input, char* const* output,\n"
    "  int const* batch_row_boundaries)\n"
    "{{\n"
    "  extern __shared__ char smem[];\n"
    "  srj_tile const t = tiles[blockIdx.x];\n"
    "  int const lane   = static_cast<int>(threadIdx.x & 31u);\n"
    "  int const warp   = static_cast<int>(threadIdx.x >> 5u);\n"
    "  int const nrows  = t.end_row - t.start_row + 1;\n"
    "  switch (warp) {{\n",
    JIT_BLOCK_SIZE);
  for (int w = 0; w < 32 && w < num_columns; ++w) {
    src += std::format("    case {}: {{\n", w);
    for (int c = w; c < num_columns; c += 32) {
      src += std::format("      char const* __restrict__ i{0} = input[{0}];\n", c);
    }
    src +=
      "      for (int rr = lane; rr < nrows; rr += 32) {\n"
      "        int const row = t.start_row + rr;\n";
    src += std::format("        char* sp = smem + rr * {};\n", shared_row_size);
    for (int c = w; c < num_columns; c += 32) {
      auto const size  = column_sizes[c];
      auto const start = column_starts[c];
      if (size == 1) {
        src += std::format("        sp[{1}] = i{0}[row];\n", c, start);
      } else if (size == 16) {
        // The shared row pitch is only 8-byte aligned; stage as two 8-byte halves.
        src += std::format(
          "        {{ long long const* p{0} = reinterpret_cast<long long const*>(i{0} + row * "
          "16);\n"
          "          *reinterpret_cast<long long*>(sp + {1}) = p{0}[0];\n"
          "          *reinterpret_cast<long long*>(sp + {2}) = p{0}[1]; }}\n",
          c,
          start,
          start + 8);
      } else {
        src += std::format(
          "        *reinterpret_cast<{2}*>(sp + {1}) = *reinterpret_cast<{2} const*>(i{0} + row "
          "* {3});\n",
          c,
          start,
          type_for_width(size),
          size);
      }
    }
    src += "      }\n    } break;\n";
  }
  src +=
    "  }\n"
    "  __syncthreads();\n"
    "  char* const obuf      = output[t.batch_number];\n"
    "  int const batch_start = t.batch_number == 0 ? 0 : batch_row_boundaries[t.batch_number];\n"
    "  for (int r = warp; r < nrows; r += 32) {\n";
  src += std::format("    char const* src = smem + r * {};\n", shared_row_size);
  src +=
    std::format("    char* dst       = obuf + (t.start_row + r - batch_start) * {};\n", row_stride);
  src += std::format(
    "    for (int i = lane * 8; i + 8 <= {0}; i += 256) {{\n"
    "      *reinterpret_cast<long long*>(dst + i) = *reinterpret_cast<long long const*>(src + "
    "i);\n"
    "    }}\n",
    actual_row_size);
  if (actual_row_size % 8 != 0) {
    auto const tail_base = actual_row_size & ~7;
    src += std::format("    if (lane < {0}) {{ dst[{1} + lane] = src[{1} + lane]; }}\n",
                       actual_row_size % 8,
                       tail_base);
  }
  src += "  }\n}\n";
  return src;
}

struct jit_kernel {
  rtcx::library library;  // keeps the CUlibrary loaded for the kernel's lifetime
  CUkernel kernel;
};

// The preferred-carveout hint mirrors the generic from-rows launch, which requests the maximum L1
// carveout for its strided row re-reads. libcuda is not on the link line (the static CUDA runtime
// and librtcx both load it dynamically), so resolve the setter the same way; it is a performance
// hint only and every failure path simply leaves the default carveout.
void try_set_max_l1_carveout(CUkernel kernel, int device_ordinal)
{
  // Alias the header's own declaration so a driver-side ABI change is a compile error here
  // rather than a silently mistyped call.
  using set_attribute_fn                      = decltype(&cuKernelSetAttribute);
  static set_attribute_fn const set_attribute = [] {
    void* handle = ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (handle == nullptr) { handle = ::dlopen("libcuda.so.1", RTLD_NOW); }
    return handle == nullptr
             ? nullptr
             : reinterpret_cast<set_attribute_fn>(::dlsym(handle, "cuKernelSetAttribute"));
  }();
  if (set_attribute != nullptr) {
    // 0 == cudaSharedmemCarveoutMaxL1 (carveout values are shared-memory percentages).
    static_cast<void>(
      set_attribute(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 0, kernel, device_ordinal));
  }
}

jit_kernel compile_specialized_kernel(std::string const& source,
                                      device_env const& env,
                                      bool prefer_max_l1)
{
  rtcx::initialize();
  std::vector<std::string> option_strings = {
    std::format("--gpu-architecture=sm_{}", env.compute_capability), "-std=c++20", "--dopt=on"};
  // The generated source is header-free; --minimal (NVRTC >= 12.8) skips the builtin header
  // machinery for a faster cold compile.
  if (rtcx::nvrtc_version() >= 12'080) { option_strings.emplace_back("--minimal"); }
  std::vector<char const*> options;
  options.reserve(option_strings.size());
  for (auto const& option : option_strings) {
    options.push_back(option.c_str());
  }

  auto const params = rtcx::compile_params{.name        = "srj_rowconv_specialized",
                                           .source      = source.c_str(),
                                           .options     = options,
                                           .target_type = rtcx::binary_type::CUBIN};
  auto cubin        = rtcx::compile(params);
  auto library      = rtcx::load_library(std::span<std::uint8_t const>{cubin.data(), cubin.size()});
  auto const kernel = library->get_kernel("srj_rowconv_specialized");
  if (prefer_max_l1) { try_set_max_l1_carveout(kernel.get(), env.ordinal); }
  return jit_kernel{std::move(library), kernel.get()};
}

struct jit_registry {
  std::mutex mutex;
  std::unordered_map<std::string, std::shared_future<jit_kernel>> kernels;
};

jit_registry& registry()
{
  // Leaked deliberately: entries hold loaded CUlibrary handles, and unloading them during static
  // destruction (after the driver may already be torn down) is the hazard this avoids.
  static auto* const instance = new jit_registry();
  return *instance;
}

/**
 * @brief Get the compiled kernel for a layout, compiling it on this thread if needed
 *
 * Block-and-compile first-call policy: a cache miss compiles on the calling thread, so the first
 * conversion of a layout pays the full compile latency and every conversion after it runs the
 * specialized kernel. Concurrent first calls for the same layout block on the shared future of
 * the one compiling thread. A failed compile leaves the exception in the shared future,
 * permanently routing that layout to the generic kernel.
 */
std::optional<jit_kernel> acquire_kernel(std::string const& key,
                                         std::string (*make_source)(std::vector<size_type> const&,
                                                                    std::vector<size_type> const&,
                                                                    size_type),
                                         std::vector<size_type> const& column_sizes,
                                         std::vector<size_type> const& column_starts,
                                         size_type row_stride,
                                         device_env const& env,
                                         bool prefer_max_l1)
{
  auto& reg = registry();
  std::shared_future<jit_kernel> fut;
  bool compile_here = false;
  std::promise<jit_kernel> promise;
  {
    std::lock_guard<std::mutex> const lock{reg.mutex};
    auto const it = reg.kernels.find(key);
    if (it != reg.kernels.end()) {
      fut = it->second;
    } else {
      if (reg.kernels.size() >= JIT_CACHE_MAX_ENTRIES) { return std::nullopt; }
      fut = promise.get_future().share();
      reg.kernels.emplace(key, fut);
      compile_here = true;
    }
  }
  if (compile_here) {
    try {
      promise.set_value(compile_specialized_kernel(
        make_source(column_sizes, column_starts, row_stride), env, prefer_max_l1));
    } catch (std::exception const& e) {
      std::fprintf(stderr,
                   "[spark-rapids-jni] row_conversion JIT compile failed; the generic kernel "
                   "will be used for this schema: %s\n",
                   e.what());
      promise.set_exception(std::current_exception());
    }
  }
  try {
    return fut.get();
  } catch (...) {
    return std::nullopt;  // compile failed; the warning was already printed once
  }
}

bool launch_specialized_kernel(jit_kernel const& kernel,
                               int num_tiles,
                               int shmem_bytes,
                               rmm::cuda_stream_view stream,
                               void** params)
{
  try {
    rtcx::kernel_ref{kernel.kernel}.launch({static_cast<std::uint32_t>(num_tiles), 1, 1},
                                           {JIT_BLOCK_SIZE, 1, 1},
                                           static_cast<std::uint32_t>(shmem_bytes),
                                           stream.value(),
                                           params);
  } catch (std::exception const& e) {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
      std::fprintf(stderr,
                   "[spark-rapids-jni] row_conversion JIT kernel launch failed; falling back to "
                   "the generic kernel: %s\n",
                   e.what());
    }
    return false;
  }
  return true;
}

}  // namespace

bool try_jit_copy_from_rows(std::vector<size_type> const& column_sizes,
                            std::vector<size_type> const& column_starts,
                            size_type row_stride,
                            int num_tiles,
                            void const* dev_tile_infos,
                            int8_t const* input_data,
                            int8_t** dev_output_data,
                            rmm::cuda_stream_view stream)
{
  if (!jit_enabled() || num_tiles <= 0 ||
      !layout_supported(column_sizes, column_starts, row_stride)) {
    return false;
  }
  auto const env = query_device_env();
  if (!env.has_value()) { return false; }
  auto const key    = make_cache_key('F', *env, row_stride, column_sizes, column_starts);
  auto const kernel = acquire_kernel(
    key, &make_from_rows_source, column_sizes, column_starts, row_stride, *env, true);
  if (!kernel.has_value()) { return false; }
  void const* params[] = {&dev_tile_infos, &input_data, &dev_output_data};
  return launch_specialized_kernel(*kernel, num_tiles, 0, stream, const_cast<void**>(params));
}

bool try_jit_copy_to_rows(std::vector<size_type> const& column_sizes,
                          std::vector<size_type> const& column_starts,
                          cudf::size_type row_stride,
                          int num_tiles,
                          void const* dev_tile_infos,
                          int shmem_bytes,
                          int8_t const** dev_input_data,
                          int8_t** dev_output_data,
                          cudf::size_type const* dev_batch_row_boundaries,
                          rmm::cuda_stream_view stream)
{
  if (!jit_enabled() || num_tiles <= 0 ||
      !layout_supported(column_sizes, column_starts, row_stride)) {
    return false;
  }
  auto const env = query_device_env();
  if (!env.has_value()) { return false; }
  auto const key = make_cache_key('T', *env, row_stride, column_sizes, column_starts);
  auto const kernel =
    acquire_kernel(key, &make_to_rows_source, column_sizes, column_starts, row_stride, *env, false);
  if (!kernel.has_value()) { return false; }
  void const* params[] = {
    &dev_tile_infos, &dev_input_data, &dev_output_data, &dev_batch_row_boundaries};
  return launch_specialized_kernel(
    *kernel, num_tiles, shmem_bytes, stream, const_cast<void**>(params));
}

}  // namespace detail
}  // namespace spark_rapids_jni
