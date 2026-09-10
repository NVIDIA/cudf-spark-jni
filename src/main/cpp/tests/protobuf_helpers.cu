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

#include "protobuf/protobuf.hpp"
#include "protobuf/protobuf_kernels.cuh"

#include <cudf_test/base_fixture.hpp>

#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <numeric>
#include <stdexcept>
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

  std::vector enum_valid_values{cudf::detail::make_pinned_vector_async<int32_t>(3, stream)};
  auto& values = enum_valid_values.back();
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
