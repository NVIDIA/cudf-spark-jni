/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.
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

#include "utilities.hpp"

#include "gtest/gtest.h"

#include <cudf_test/base_fixture.hpp>

#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>

class UtilitiesTest : public cudf::test::BaseFixture {};

TEST_F(UtilitiesTest, BitwiseOr)
{
  auto stream = cudf::get_default_stream();

  // 2 buffers
  {
    std::vector<cudf::bitmask_type> a{0x10011001, 0x0000ffff, 0xffff0000, 0x01010101, 0xab000000};
    rmm::device_uvector<cudf::bitmask_type> da(a.size(), stream);
    cudaMemcpyAsync(da.data(), a.data(), sizeof(cudf::bitmask_type) * a.size(), cudaMemcpyDefault);

    std::vector<cudf::bitmask_type> b{0x01100000, 0xffff0000, 0xffff0000, 0xf000000f, 0x000100ab};
    rmm::device_uvector<cudf::bitmask_type> db(b.size(), stream);
    cudaMemcpyAsync(db.data(), b.data(), sizeof(cudf::bitmask_type) * b.size(), cudaMemcpyDefault);

    std::vector<cudf::bitmask_type> expect{
      0x11111001, 0xffffffff, 0xffff0000, 0xf101010f, 0xab0100ab};
    auto d_result = spark_rapids_jni::bitmask_bitwise_or({{da}, {db}}, stream);
    CUDF_EXPECTS(d_result->size() == expect.size() * sizeof(cudf::bitmask_type),
                 "Unexpected output size");
    std::vector<cudf::bitmask_type> result(expect.size());
    cudaMemcpy(result.data(), d_result->data(), d_result->size(), cudaMemcpyDefault);
    CUDF_EXPECTS(std::equal(result.begin(), result.end(), expect.begin()),
                 "Unexpected output size");
  }

  // 4 buffers
  {
    std::vector<cudf::bitmask_type> a{0x10000000, 0x0000f000, 0xf0000000, 0x01000000, 0xa0000000};
    rmm::device_uvector<cudf::bitmask_type> da(a.size(), stream);
    cudaMemcpyAsync(da.data(), a.data(), sizeof(cudf::bitmask_type) * a.size(), cudaMemcpyDefault);

    std::vector<cudf::bitmask_type> b{0x00010000, 0x00000f00, 0x0f000000, 0x00010000, 0x0b000000};
    rmm::device_uvector<cudf::bitmask_type> db(b.size(), stream);
    cudaMemcpyAsync(db.data(), b.data(), sizeof(cudf::bitmask_type) * b.size(), cudaMemcpyDefault);

    std::vector<cudf::bitmask_type> c{0x00001000, 0x000000f0, 0x00f00000, 0x00000100, 0x000000a0};
    rmm::device_uvector<cudf::bitmask_type> dc(c.size(), stream);
    cudaMemcpyAsync(dc.data(), c.data(), sizeof(cudf::bitmask_type) * c.size(), cudaMemcpyDefault);

    std::vector<cudf::bitmask_type> d{0x00000001, 0x0000000f, 0x000f0000, 0x00000001, 0x0000000b};
    rmm::device_uvector<cudf::bitmask_type> dd(c.size(), stream);
    cudaMemcpyAsync(dd.data(), d.data(), sizeof(cudf::bitmask_type) * d.size(), cudaMemcpyDefault);

    std::vector<cudf::bitmask_type> expect{
      0x10011001, 0x0000ffff, 0xffff0000, 0x01010101, 0xab0000ab};
    auto d_result = spark_rapids_jni::bitmask_bitwise_or({{da}, {db}, {dc}, {dd}}, stream);
    CUDF_EXPECTS(d_result->size() == expect.size() * sizeof(cudf::bitmask_type),
                 "Unexpected output size");
    std::vector<cudf::bitmask_type> result(expect.size());
    cudaMemcpy(result.data(), d_result->data(), d_result->size(), cudaMemcpyDefault);
    CUDF_EXPECTS(std::equal(result.begin(), result.end(), expect.begin()),
                 "Results do not match expected");
  }
}

TEST_F(UtilitiesTest, BitwiseOrEmptyInput)
{
  auto stream = cudf::get_default_stream();

  rmm::device_uvector<cudf::bitmask_type> da(0, stream);
  rmm::device_uvector<cudf::bitmask_type> db(0, stream);
  auto result = spark_rapids_jni::bitmask_bitwise_or({{da}, {db}}, stream);
  CUDF_EXPECTS(result->size() == 0, "Expected empty output");
}

TEST_F(UtilitiesTest, BitwiseOrExpectedFailures)
{
  auto stream = cudf::get_default_stream();

  {
    std::vector<cudf::bitmask_type> a{10, 20, 30, 40};
    rmm::device_uvector<cudf::bitmask_type> da(a.size(), stream);
    cudaMemcpyAsync(da.data(), a.data(), sizeof(cudf::bitmask_type) * a.size(), cudaMemcpyDefault);

    std::vector<cudf::bitmask_type> b{50, 60, 70};
    rmm::device_uvector<cudf::bitmask_type> db(b.size(), stream);
    cudaMemcpyAsync(db.data(), b.data(), sizeof(cudf::bitmask_type) * b.size(), cudaMemcpyDefault);

    EXPECT_THROW(spark_rapids_jni::bitmask_bitwise_or({{da}, {db}}, stream), cudf::logic_error);
  }

  {
    EXPECT_THROW(spark_rapids_jni::bitmask_bitwise_or({}, stream), cudf::logic_error);
  }
}
