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

#include "range_sequence.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/memory_resource_utilities.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
struct RangeSequenceTests : cudf::test::BaseFixture {};

auto constexpr min = std::numeric_limits<int64_t>::min();
auto constexpr max = std::numeric_limits<int64_t>::max();

void assert_sequence(std::vector<int64_t> const& starts,
                     std::vector<cudf::size_type> const& sizes,
                     cuda::stream_ref stream)
{
  std::vector<int64_t> expected;
  for (std::size_t range = 0; range < starts.size(); ++range) {
    for (cudf::size_type row = 0; row < sizes[range]; ++row) {
      expected.push_back(starts[range] + row);
    }
  }
  // Only the returned column may use the supplied resource; metadata must use the current one.
  rmm::mr::limiting_resource_adaptor output_mr{
    rmm::mr::cuda_async_memory_resource{},
    rmm::align_up(expected.size() * sizeof(int64_t), rmm::CUDA_ALLOCATION_ALIGNMENT)};
  auto const output = spark_rapids_jni::generate_range_sequence(starts, sizes, stream, output_mr);
  EXPECT_EQ(output->type(), cudf::data_type{cudf::type_id::INT64});
  EXPECT_EQ(
    expected,
    cudf::detail::make_std_vector(
      cudf::device_span<int64_t const>(output->view().data<int64_t>(), output->size()), stream));
  EXPECT_FALSE(output->nullable());
}

TEST_F(RangeSequenceTests, Sequences)
{
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  assert_sequence({max - 2}, {3}, stream);
  assert_sequence({max}, {0}, stream);
  for (auto const ranges : {0, 1, 2, 8, 4'097}) {
    SCOPED_TRACE(ranges);
    std::vector<int64_t> starts(ranges, -1);
    std::vector<cudf::size_type> sizes(ranges, 3);
    if (ranges > 0) {
      starts.front() = max - 2;
      starts.back()  = min;
      sizes.back()   = 2;
    }
    assert_sequence(starts, sizes, stream);
    if (ranges > 1) {
      // Leading, consecutive interior, and trailing empty ranges produce repeated cumulative ends.
      sizes.front() = sizes.back() = sizes[ranges / 2] = sizes[ranges / 2 - 1] = 0;
      assert_sequence(starts, sizes, stream);
    }
  }
}

TEST_F(RangeSequenceTests, InvalidRanges)
{
  auto generate = [](std::vector<int64_t> const& starts,
                     std::vector<cudf::size_type> const& sizes) {
    static_cast<void>(spark_rapids_jni::generate_range_sequence(starts, sizes));
  };
  auto constexpr max_rows = std::numeric_limits<cudf::size_type>::max();
  EXPECT_THROW(generate({0}, {}), std::invalid_argument);
  EXPECT_THROW(generate({}, {1}), std::invalid_argument);
  EXPECT_THROW(generate({0, 0}, {0, -1}), std::invalid_argument);
  EXPECT_THROW(generate({max}, {2}), std::invalid_argument);
  EXPECT_THROW(generate({0, max - 1}, {0, 3}), std::invalid_argument);
  EXPECT_THROW(generate({0, 0}, {max_rows, 1}), std::overflow_error);
}

TEST_F(RangeSequenceTests, MetadataAllocationFailures)
{
  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  std::vector<int64_t> const starts{100, 200};
  std::vector<cudf::size_type> const sizes{3, 2};
  rmm::mr::limiting_resource_adaptor output_mr{rmm::mr::cuda_async_memory_resource{},
                                               rmm::CUDA_ALLOCATION_ALIGNMENT};
  for (std::size_t const successful_allocations : {0, 1}) {
    SCOPED_TRACE(successful_allocations);
    // Fail the first upload, or the second after starts have been uploaded.
    rmm::mr::limiting_resource_adaptor temporary_mr{
      rmm::mr::cuda_async_memory_resource{},
      successful_allocations * rmm::CUDA_ALLOCATION_ALIGNMENT};
    {
      cudf::test::scoped_current_device_resource current_scope{temporary_mr};
      EXPECT_THROW(spark_rapids_jni::generate_range_sequence(starts, sizes, stream, output_mr),
                   rmm::out_of_memory);
      stream.synchronize();
    }
    EXPECT_EQ(temporary_mr.get_allocated_bytes(), 0);
    EXPECT_EQ(output_mr.get_allocated_bytes(), 0);
  }
  assert_sequence(starts, sizes, stream);
}
}  // namespace
