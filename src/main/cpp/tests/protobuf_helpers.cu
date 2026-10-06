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

#include "protobuf/protobuf.hpp"
#include "protobuf/protobuf_host_helpers.hpp"
#include "protobuf/protobuf_kernels.cuh"

#include <cudf_test/base_fixture.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

class ProtobufHelpersTest : public cudf::test::BaseFixture {};

namespace {

namespace protobuf = spark_rapids_jni::protobuf;

protobuf::protobuf_decode_context make_numeric_enum_context(int64_t default_value)
{
  auto const stream = cudf::get_default_stream();

  std::vector<cudf::detail::host_vector<uint8_t>> default_strings;
  default_strings.emplace_back(cudf::detail::make_pinned_vector_async<uint8_t>(0, stream));

  std::vector<cudf::detail::host_vector<int32_t>> enum_valid_values;
  auto& values =
    enum_valid_values.emplace_back(cudf::detail::make_pinned_vector_async<int32_t>(3, stream));
  std::iota(values.begin(), values.end(), 0);

  std::vector<std::vector<cudf::detail::host_vector<uint8_t>>> enum_names(1);
  return {{{.field_number      = 1,
            .parent_idx        = -1,
            .wire_type         = protobuf::proto_wire_type::VARINT,
            .output_type       = cudf::type_id::INT32,
            .encoding          = protobuf::proto_encoding::DEFAULT,
            .has_default_value = true}},
          {default_value},
          {0.0},
          {false},
          std::move(default_strings),
          std::move(enum_valid_values),
          std::move(enum_names),
          true};
}

}  // namespace

TEST_F(ProtobufHelpersTest, NumericEnumDefaultMustFitInt32)
{
  EXPECT_NO_THROW(make_numeric_enum_context(2));
  EXPECT_THROW(make_numeric_enum_context(int64_t{1} << 42), std::invalid_argument);
}

TEST_F(ProtobufHelpersTest, AtomicFlagBufferIsPaddedAndZeroed)
{
  auto const stream = cudf::get_default_stream();
  for (std::size_t num_rows : {0, 1, 2, 3, 4, 5, 7, 8, 9}) {
    SCOPED_TRACE(num_rows);
    auto flags = protobuf::detail::make_zeroed_atomic_flag_buffer(
      num_rows, stream, cudf::get_current_device_resource_ref());
    EXPECT_EQ(num_rows == 0, flags.is_empty());
    EXPECT_GE(flags.alignment(), alignof(uint32_t));
    EXPECT_EQ(0u, flags.size() % sizeof(uint32_t));
    EXPECT_GE(flags.size(), num_rows);
    EXPECT_LT(flags.size() - num_rows, sizeof(uint32_t));
    if (num_rows == 0) {
      EXPECT_EQ(nullptr, flags.data());
      continue;
    }
    EXPECT_EQ(0u, reinterpret_cast<std::uintptr_t>(flags.data()) % alignof(uint32_t));
    std::vector<uint8_t> bytes(flags.size());
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(bytes.data(), flags.data(), bytes.size(), cudaMemcpyDefault, stream.get()));
    stream.sync();
    EXPECT_EQ(std::vector<uint8_t>(bytes.size(), 0), bytes);
  }
}

TEST_F(ProtobufHelpersTest, NullMaskFromPaddedValidUsesZeroLogicalRows)
{
  cuda::stream_ref stream = cudf::get_default_stream();

  std::array<bool, 1> h_valid{false};
  rmm::device_uvector<bool> valid(h_valid.size(), stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(valid.data(),
                                h_valid.data(),
                                h_valid.size() * sizeof(h_valid[0]),
                                cudaMemcpyDefault,
                                stream.get()));

  auto [mask, null_count] = spark_rapids_jni::protobuf::detail::make_null_mask_from_valid(
    valid, 0, stream, cudf::get_current_device_resource_ref());

  EXPECT_EQ(0u, mask.size());
  EXPECT_EQ(nullptr, mask.data());
  EXPECT_EQ(0, null_count);
}

TEST_F(ProtobufHelpersTest, NullMaskFromPaddedValidIgnoresTail)
{
  cuda::stream_ref stream = cudf::get_default_stream();

  std::array<bool, 3> h_valid{true, false, false};
  rmm::device_uvector<bool> valid(h_valid.size(), stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(valid.data(),
                                h_valid.data(),
                                h_valid.size() * sizeof(h_valid[0]),
                                cudaMemcpyDefault,
                                stream.get()));

  auto [mask, null_count] = spark_rapids_jni::protobuf::detail::make_null_mask_from_valid(
    valid, 2, stream, cudf::get_current_device_resource_ref());

  EXPECT_EQ(cudf::bitmask_allocation_size_bytes(2), mask.size());
  EXPECT_EQ(1, null_count);

  std::vector<cudf::bitmask_type> h_mask(mask.size() / sizeof(cudf::bitmask_type));
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(h_mask.data(), mask.data(), mask.size(), cudaMemcpyDefault, stream.get()));
  stream.sync();

  EXPECT_TRUE(cudf::bit_is_set(h_mask.data(), 0));
  EXPECT_FALSE(cudf::bit_is_set(h_mask.data(), 1));
}

TEST_F(ProtobufHelpersTest, NullMaskFromAllValidRowsIsEmpty)
{
  cuda::stream_ref stream = cudf::get_default_stream();

  std::array<bool, 2> h_valid{true, true};
  rmm::device_uvector<bool> valid(h_valid.size(), stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(valid.data(),
                                h_valid.data(),
                                h_valid.size() * sizeof(h_valid[0]),
                                cudaMemcpyDefault,
                                stream.get()));

  auto [mask, null_count] = spark_rapids_jni::protobuf::detail::make_null_mask_from_valid(
    valid, h_valid.size(), stream, cudf::get_current_device_resource_ref());

  EXPECT_EQ(0u, mask.size());
  EXPECT_EQ(nullptr, mask.data());
  EXPECT_EQ(0, null_count);
}

namespace spark_rapids_jni::protobuf::detail {

std::ostream& operator<<(std::ostream& out, field_location const& location)
{
  return out << "{offset=" << location.offset << ", length=" << location.length << "}";
}

}  // namespace spark_rapids_jni::protobuf::detail

namespace {

namespace protobuf_detail = spark_rapids_jni::protobuf::detail;
using protobuf_detail::field_location;

// Named accessors avoid member-pointer template arguments in NVCC's generated host stubs.
struct input_location_accessor {
  template <typename Provider>
  __device__ field_location operator()(Provider const& provider, int row) const
  {
    return provider.input_location(row);
  }
};

struct row_location_accessor {
  __device__ field_location operator()(protobuf_detail::nested_location_provider const& provider,
                                       int row,
                                       protobuf_detail::protobuf_error* error) const
  {
    return provider.row_location(row, error);
  }
};

template <typename Accessor, typename Provider, typename Output, typename... Args>
CUDF_KERNEL void invoke_provider_kernel(Provider const* providers,
                                        int num_probes,
                                        Output* output,
                                        Args... args)
{
  auto const idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < num_probes) { output[idx] = Accessor{}(providers[idx], args...); }
}

template <typename Accessor, typename Provider, typename... Args>
auto invoke_provider(std::vector<Provider> const& providers, Args... args)
{
  using result_t        = std::invoke_result_t<Accessor, Provider const&, Args...>;
  auto const stream     = cudf::get_default_stream();
  auto const mr         = cudf::get_current_device_resource_ref();
  auto const num_probes = static_cast<int>(providers.size());
  if (num_probes == 0) { return std::vector<result_t>{}; }
  auto const d_providers = cudf::detail::make_device_uvector_async(providers, stream, mr);
  rmm::device_uvector<result_t> d_output(num_probes, stream, mr);
  invoke_provider_kernel<Accessor>
    <<<1, num_probes, 0, stream.get()>>>(d_providers.data(), num_probes, d_output.data(), args...);
  CUDF_CHECK_CUDA(stream.get());
  return cudf::detail::make_std_vector(d_output, stream);
}

struct rebase_probe {
  field_location location;
  int64_t base;
  protobuf_detail::protobuf_error* error;
};

CUDF_KERNEL void rebase_locations_kernel(rebase_probe const* probes,
                                         int num_probes,
                                         field_location* output)
{
  auto const idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= num_probes) return;
  auto const& probe = probes[idx];
  output[idx]       = protobuf_detail::rebase_location(probe.location, probe.base, probe.error);
}

template <typename Actual, typename Expected>
void expect_locations(Actual const& actual, Expected const& expected)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(actual[i], expected[i]);
  }
}

}  // namespace

TEST_F(ProtobufHelpersTest, FieldLocationPresence)
{
  EXPECT_FALSE(field_location::missing().is_present());
  EXPECT_FALSE((field_location{-2, 7}.is_present()));
  EXPECT_TRUE((field_location{0, 0}.is_present()));
  EXPECT_TRUE((field_location{0, 7}.is_present()));
  EXPECT_TRUE((field_location{std::numeric_limits<int32_t>::max(), 0}.is_present()));
}

TEST_F(ProtobufHelpersTest, TopLevelInputLocationsRebaseSlices)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto const offsets =
    cudf::detail::make_device_uvector(std::vector<int32_t>{100, 120}, stream, mr);
  auto const locations = cudf::detail::make_device_uvector(
    std::vector<field_location>{
      {2, 3}, {4, 0}, field_location::missing(), {std::numeric_limits<int32_t>::max(), 0}},
    stream,
    mr);
  std::vector<protobuf_detail::top_level_location_provider> const probes{
    {offsets.data(), 100, locations.data(), 0, 4},
    {offsets.data(), 100, locations.data(), 1, 4},
    {offsets.data(), 100, locations.data(), 2, 4},
    {offsets.data(), 101, locations.data(), 0, 4},
    {offsets.data(), 99, locations.data(), 3, 4}};
  auto const expected = std::to_array<field_location>({{2, 3},
                                                       {4, 0},
                                                       field_location::missing(),
                                                       field_location::missing(),
                                                       field_location::missing()});
  expect_locations(invoke_provider<input_location_accessor>(probes, 0), expected);
}

TEST_F(ProtobufHelpersTest, NestedRowLocationsPreservePresenceAndCheckOverflow)
{
  auto const stream  = cudf::get_default_stream();
  auto const mr      = cudf::get_current_device_resource_ref();
  auto const parents = cudf::detail::make_device_uvector(
    std::vector<field_location>{
      {5, 10}, field_location::missing(), {std::numeric_limits<int32_t>::max(), 0}},
    stream,
    mr);
  auto const children = cudf::detail::make_device_uvector(
    std::vector<field_location>{{2, 3}, {4, 0}, field_location::missing()}, stream, mr);
  std::vector<protobuf_detail::nested_location_provider> const probes{
    {nullptr, 0, parents.data(), children.data(), 0, 3},
    {nullptr, 0, parents.data(), children.data(), 1, 3},
    {nullptr, 0, parents.data(), children.data(), 2, 3},
    {nullptr, 0, parents.data() + 1, children.data(), 0, 3},
    {nullptr, 0, parents.data() + 2, children.data(), 0, 3}};
  auto const expected = std::to_array<field_location>({{7, 3},
                                                       {9, 0},
                                                       field_location::missing(),
                                                       field_location::missing(),
                                                       field_location::missing()});
  expect_locations(invoke_provider<row_location_accessor>(probes, 0, nullptr), expected);
}

TEST_F(ProtobufHelpersTest, NestedInputLocationsRebaseSlices)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto const offsets =
    cudf::detail::make_device_uvector(std::vector<int32_t>{100, 120}, stream, mr);
  auto const parents = cudf::detail::make_device_uvector(
    std::vector<field_location>{{5, 10}, field_location::missing()}, stream, mr);
  auto const children = cudf::detail::make_device_uvector(
    std::vector<field_location>{{2, 3}, {std::numeric_limits<int32_t>::max() - 5, 0}}, stream, mr);
  std::vector<protobuf_detail::nested_location_provider> const probes{
    {offsets.data(), 90, parents.data(), children.data(), 0, 2},
    {offsets.data(), 90, parents.data() + 1, children.data(), 0, 2},
    {offsets.data(), 101, parents.data(), children.data(), 0, 2},
    {offsets.data(), 99, parents.data(), children.data(), 1, 2}};
  auto const expected = std::to_array<field_location>(
    {{17, 3}, field_location::missing(), field_location::missing(), field_location::missing()});
  expect_locations(invoke_provider<input_location_accessor>(probes, 0), expected);
}

TEST_F(ProtobufHelpersTest, OccurrenceInputLocationsRebaseRowsAndParents)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto const offsets =
    cudf::detail::make_device_uvector(std::vector<int32_t>{100, 120}, stream, mr);
  auto const parents = cudf::detail::make_device_uvector(
    std::vector<field_location>{
      {5, 10}, field_location::missing(), {std::numeric_limits<int32_t>::max(), 0}},
    stream,
    mr);
  auto const occurrences = cudf::detail::make_device_uvector(
    std::vector<protobuf_detail::field_occurrence>{{0, 2, 3},
                                                   {0, std::numeric_limits<int32_t>::max(), 0}},
    stream,
    mr);
  protobuf_detail::protobuf_input_view const input{nullptr, 30, offsets.data(), 90, 1};
  auto negative_base        = input;
  negative_base.base_offset = 101;
  std::vector<protobuf_detail::field_occurrence_location_provider> const probes{
    {input, {parents.data(), 1, nullptr}, occurrences.data()},
    {input, {nullptr, 0, nullptr}, occurrences.data()},
    {input, {parents.data() + 1, 1, nullptr}, occurrences.data()},
    {input, {parents.data() + 2, 1, nullptr}, occurrences.data()},
    {negative_base, {parents.data(), 1, nullptr}, occurrences.data()},
    {input, {nullptr, 0, nullptr}, occurrences.data() + 1}};
  auto const expected = std::to_array<field_location>({{17, 3},
                                                       {12, 3},
                                                       field_location::missing(),
                                                       field_location::missing(),
                                                       field_location::missing(),
                                                       field_location::missing()});
  expect_locations(invoke_provider<input_location_accessor>(probes, 0), expected);
}

TEST_F(ProtobufHelpersTest, OccurrenceInputLocationsUseOwningRow)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  std::vector<int32_t> const h_offsets{100, 120, 150};
  std::vector<field_location> const h_parents{{5, 10}, {7, 12}};
  std::vector<protobuf_detail::field_occurrence> const h_occurrences{{1, 2, 3}, {0, 4, 2}};
  auto const offsets     = cudf::detail::make_device_uvector_async(h_offsets, stream, mr);
  auto const parents     = cudf::detail::make_device_uvector_async(h_parents, stream, mr);
  auto const occurrences = cudf::detail::make_device_uvector_async(h_occurrences, stream, mr);
  protobuf_detail::protobuf_input_view const input{nullptr, 50, offsets.data(), 100, 2};
  std::vector<protobuf_detail::field_occurrence_location_provider> const probes{
    {input, {parents.data(), 2, nullptr}, occurrences.data()},
    {input, {nullptr, 0, nullptr}, occurrences.data()}};
  expect_locations(invoke_provider<input_location_accessor>(probes, 0),
                   std::to_array<field_location>({{29, 3}, {22, 3}}));
  expect_locations(invoke_provider<input_location_accessor>(probes, 1),
                   std::to_array<field_location>({{9, 2}, {4, 2}}));
}

TEST_F(ProtobufHelpersTest, RebaseLocationChecksBoundsAndReportsOverflow)
{
  using protobuf_detail::protobuf_error;
  auto const stream     = cudf::get_default_stream();
  auto const mr         = cudf::get_current_device_resource_ref();
  auto const max_offset = std::numeric_limits<int32_t>::max();
  auto const expected   = std::to_array<field_location>({{max_offset, 0},
                                                         field_location::missing(),
                                                         field_location::missing(),
                                                         field_location::missing(),
                                                         field_location::missing()});
  auto errors =
    cudf::detail::make_zeroed_device_uvector_async<protobuf_error>(expected.size(), stream, mr);
  std::vector<rebase_probe> const probes{
    {{max_offset, 0}, 0, errors.data()},
    {{max_offset, 0}, 1, errors.data() + 1},
    {{2, 0}, -1, errors.data() + 2},
    {{1, 0}, std::numeric_limits<int64_t>::max(), errors.data() + 3},
    {field_location::missing(), 0, errors.data() + 4}};
  auto const d_probes = cudf::detail::make_device_uvector_async(probes, stream, mr);
  rmm::device_uvector<field_location> output(probes.size(), stream, mr);
  auto const num_probes = static_cast<int>(probes.size());
  rebase_locations_kernel<<<1, num_probes, 0, stream.get()>>>(
    d_probes.data(), num_probes, output.data());
  CUDF_CHECK_CUDA(stream.get());
  expect_locations(cudf::detail::make_std_vector(output, stream), expected);
  std::vector<protobuf_error> const expected_errors{protobuf_error::NONE,
                                                    protobuf_error::OVERFLOW,
                                                    protobuf_error::OVERFLOW,
                                                    protobuf_error::OVERFLOW,
                                                    protobuf_error::NONE};
  EXPECT_EQ(cudf::detail::make_std_vector(errors, stream), expected_errors);
}

TEST_F(ProtobufHelpersTest, ByteLengthsUseDefaultOnlyForMissingFields)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto const offsets =
    cudf::detail::make_device_uvector(std::vector<int32_t>{0, 10, 20, 20}, stream, mr);
  auto const locations = cudf::detail::make_device_uvector(
    std::vector<field_location>{{0, 3}, field_location::missing(), {0, 0}}, stream, mr);
  protobuf_detail::top_level_location_provider const provider{
    offsets.data(), 0, locations.data(), 0, 1};
  rmm::device_uvector<int32_t> lengths(3, stream, mr);
  protobuf_detail::extract_lengths_kernel<<<1, 3, 0, stream.get()>>>(provider, 3, lengths.data());
  CUDF_CHECK_CUDA(stream.get());
  EXPECT_EQ(cudf::detail::make_std_vector(lengths, stream), (std::vector<int32_t>{3, 0, 0}));
  protobuf_detail::extract_lengths_kernel<<<1, 3, 0, stream.get()>>>(
    provider, 3, lengths.data(), 7);
  CUDF_CHECK_CUDA(stream.get());
  EXPECT_EQ(cudf::detail::make_std_vector(lengths, stream), (std::vector<int32_t>{3, 7, 0}));
}
