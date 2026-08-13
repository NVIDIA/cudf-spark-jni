/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

#include "shuffle_split.hpp"
#include "test_utilities.hpp"
#include "utilities/iterator.cuh"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/utilities/memory_resource.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda/functional>

struct ShuffleSplitTests : public cudf::test::BaseFixture {};

std::unique_ptr<cudf::table> reshape_table(cudf::table_view const& tbl,
                                           std::vector<int> const& splits,
                                           std::vector<int> const& remaps)
{
  auto split_result = cudf::split(tbl, splits);
  std::vector<cudf::table_view> remapped;
  remapped.reserve(split_result.size());
  std::transform(remaps.begin(), remaps.end(), std::back_inserter(remapped), [&](int i) {
    return split_result[i];
  });
  return cudf::concatenate(remapped);
}

spark_rapids_jni::shuffle_split_result reshape_partitions(
  cudf::device_span<uint8_t const> partitions,
  cudf::device_span<size_t const> partition_offsets,
  std::vector<int> const& remaps,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(remaps.size() == partition_offsets.size() - 1, "Invaid remaps vector size");
  auto temp_mr = cudf::get_current_device_resource_ref();

  // generate new partition offsets
  auto d_remaps = cudf::detail::make_device_uvector_async(remaps, stream, temp_mr);
  rmm::device_uvector<size_t> remapped_offsets(partition_offsets.size(), stream, mr);
  auto const num_partitions = partition_offsets.size() - 1;
  auto remapped_size_iter   = spark_rapids_jni::util::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<size_t>([partition_offsets = partition_offsets.begin(),
                                        remaps            = d_remaps.begin(),
                                        num_partitions] __device__(size_t i) {
      auto const ri = remaps[i];
      return i >= num_partitions ? 0 : partition_offsets[ri + 1] - partition_offsets[ri];
    }));
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         remapped_size_iter,
                         remapped_size_iter + num_partitions + 1,
                         remapped_offsets.begin());

  // swizzle the data
  rmm::device_buffer remapped_partitions(partitions.size(), stream, mr);
  auto input_iter = spark_rapids_jni::util::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<void*>([partitions        = partitions.data(),
                                       partition_offsets = partition_offsets.begin(),
                                       remaps            = d_remaps.begin()] __device__(size_t i) {
      return reinterpret_cast<void*>(const_cast<uint8_t*>(partitions) +
                                     partition_offsets[remaps[i]]);
    }));
  auto size_iter = spark_rapids_jni::util::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<size_t>([partition_offsets = partition_offsets.begin(),
                                        remaps            = d_remaps.begin()] __device__(size_t i) {
      auto const ri = remaps[i];
      return partition_offsets[ri + 1] - partition_offsets[ri];
    }));
  auto output_iter = spark_rapids_jni::util::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<void*>(
      [remapped_partitions = remapped_partitions.data(),
       remapped_offsets    = remapped_offsets.begin()] __device__(size_t i) {
        return reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(remapped_partitions) +
                                       remapped_offsets[i]);
      }));

  size_t temp_storage_bytes = 0;
  cub::DeviceMemcpy::Batched(
    nullptr, temp_storage_bytes, input_iter, output_iter, size_iter, num_partitions, stream);
  rmm::device_buffer temp_storage(
    temp_storage_bytes, stream, cudf::get_current_device_resource_ref());
  cub::DeviceMemcpy::Batched(temp_storage.data(),
                             temp_storage_bytes,
                             input_iter,
                             output_iter,
                             size_iter,
                             num_partitions,
                             stream);

  return {std::make_unique<rmm::device_buffer>(std::move(remapped_partitions)),
          std::move(remapped_offsets)};
}

// Helper function to create a table_view from shuffle_assemble_result
cudf::table_view make_table_view(spark_rapids_jni::shuffle_assemble_result const& result)
{
  std::vector<cudf::column_view> column_views;
  column_views.reserve(result.column_views.size());
  for (auto const& cv_ptr : result.column_views) {
    column_views.push_back(*cv_ptr);
  }
  return cudf::table_view{column_views};
}

auto run_split(cudf::table_view const& tbl,
               std::vector<cudf::size_type> const& splits,
               std::vector<cudf::size_type> const& remaps = {})
{
  auto [split_data, split_metadata] = spark_rapids_jni::shuffle_split(
    tbl, splits, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref());

  // maybe reshape the results
  if (remaps.size() > 0) {
    CUDF_EXPECTS(remaps.size() == splits.size() + 1, "Invalid remap vector size");
    CUDF_EXPECTS(remaps.size() == split_data.offsets.size() - 1, "Invaid remaps vector size");
    auto reshaped_table = reshape_table(tbl, splits, remaps);
    auto reshaped_data  = reshape_partitions(
      {static_cast<uint8_t*>(split_data.partitions->data()), split_data.partitions->size()},
      split_data.offsets,
      remaps,
      cudf::get_default_stream(),
      rmm::mr::get_current_device_resource_ref());

    auto result = spark_rapids_jni::shuffle_assemble(
      split_metadata,
      {static_cast<uint8_t*>(reshaped_data.partitions->data()), reshaped_data.partitions->size()},
      reshaped_data.offsets,
      cudf::get_default_stream(),
      rmm::mr::get_current_device_resource_ref());

    CUDF_TEST_EXPECT_TABLES_EQUAL(*reshaped_table, make_table_view(result));

    return result;
  }

  auto result = spark_rapids_jni::shuffle_assemble(
    split_metadata,
    {static_cast<uint8_t*>(split_data.partitions->data()), split_data.partitions->size()},
    split_data.offsets,
    cudf::get_default_stream(),
    rmm::mr::get_current_device_resource_ref());

  CUDF_TEST_EXPECT_TABLES_EQUAL(tbl, make_table_view(result));
  return result;
}

TEST_F(ShuffleSplitTests, Simple)
{
  cudf::size_type const num_rows = 10000;
  auto iter                      = thrust::make_counting_iterator(0);
  cudf::test::fixed_width_column_wrapper<int> col0(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<float> col1(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<int16_t> col2(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<int8_t> col3(iter, iter + num_rows);

  // 4 columns split once
  {
    cudf::table_view tbl{{static_cast<cudf::column_view>(col0),
                          static_cast<cudf::column_view>(col1),
                          static_cast<cudf::column_view>(col2),
                          static_cast<cudf::column_view>(col3)}};
    run_split(tbl, {10});
  }

  // 4 columns split multiple times
  {
    cudf::table_view tbl{{static_cast<cudf::column_view>(col0),
                          static_cast<cudf::column_view>(col1),
                          static_cast<cudf::column_view>(col2),
                          static_cast<cudf::column_view>(col3)}};
    run_split(tbl, {10, 100, 2756, 7777});
  }
}

TEST_F(ShuffleSplitTests, Strings)
{
  {
    cudf::test::strings_column_wrapper col{
      {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}};
    cudf::table_view tbl{{static_cast<cudf::column_view>(col)}};

    run_split(tbl, {});
    run_split(tbl, {1});
    run_split(tbl, {1, 4});
  }

  {
    cudf::test::strings_column_wrapper col0{
      {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}};
    cudf::test::strings_column_wrapper col1{{"blue",
                                             "green",
                                             "yellow",
                                             "red",
                                             "black",
                                             "white",
                                             "gray",
                                             "aquamarine",
                                             "mauve",
                                             "ultraviolet"}};
    cudf::test::strings_column_wrapper col2{
      {"left", "up", "right", "down", "", "space", "", "delete", "end", "insert"}};
    cudf::test::strings_column_wrapper col3{
      {"a", "b", "c", "de", "fg", "h", "i", "jk", "lmn", "opq"}};
    cudf::table_view tbl{{static_cast<cudf::column_view>(col0),
                          static_cast<cudf::column_view>(col1),
                          static_cast<cudf::column_view>(col2),
                          static_cast<cudf::column_view>(col3)}};

    run_split(tbl, {});
    run_split(tbl, {1});
    run_split(tbl, {1, 4});
  }
}

TEST_F(ShuffleSplitTests, SimpleWithStrings)
{
  cudf::test::fixed_width_column_wrapper<int8_t> col0({0, 0xF0, 0x0F, 0xAA, 0}, {1, 0, 0, 0, 1});
  cudf::test::strings_column_wrapper col1({"0xFF", "", "0x0F", "0xAA", "0x55"}, {0, 1, 0, 0, 0});

  // 2 columns split once
  {
    cudf::table_view tbl{
      {static_cast<cudf::column_view>(col0), static_cast<cudf::column_view>(col1)}};
    run_split(tbl, {});
    run_split(tbl, {1, 3});
    run_split(tbl, {0, 1, 2});
    run_split(tbl, {5});
  }
}

TEST_F(ShuffleSplitTests, Lists)
{
  // list<uint64_t>
  {
    using lcw = cudf::test::lists_column_wrapper<uint64_t>;
    lcw col0{{9, 8},
             {7, 6, 5},
             {},
             {4},
             {3, 2, 1, 0},
             {20, 21, 22, 23, 24},
             {},
             {66, 666},
             {123, 7},
             {100, 101}};

    cudf::table_view tbl{{static_cast<cudf::column_view>(col0)}};

    run_split(tbl, {});
    run_split(tbl, {1});
    run_split(tbl, {1, 4});
  }

  // list<list<uint64_t>
  {
    using lcw = cudf::test::lists_column_wrapper<uint64_t>;
    lcw col0{{{9, 8}, {7, 6, 5}},
             {lcw{}, {4}},
             {{3, 2, 1, 0}, {20, 21, 22, 23, 24}},
             {lcw{}, {66, 666}},
             {{123, 7}, {100, 101}},
             {{1, 2, 4}, {8, 6, 5}}};

    cudf::table_view tbl{{static_cast<cudf::column_view>(col0)}};

    run_split(tbl, {});
    run_split(tbl, {4});
    run_split(tbl, {1, 4});
  }

  // list<string>
  {
    cudf::test::strings_column_wrapper strings0{{"*", "*", "****", "", "*", ""},
                                                {1, 1, 1, 1, 1, 0}};
    cudf::test::fixed_width_column_wrapper<int> offsets0{0, 1, 2, 3, 6};
    auto col0 = cudf::make_lists_column(4, offsets0.release(), strings0.release(), 0, {});

    cudf::test::strings_column_wrapper strings1{{"", "", "", "", "", "", ""},
                                                {0, 0, 0, 0, 0, 0, 0}};
    cudf::test::fixed_width_column_wrapper<int> offsets1{0, 4, 4, 7, 7};
    auto col1 = cudf::make_lists_column(4, offsets1.release(), strings1.release(), 0, {});

    cudf::table_view tbl{{*col0, *col1}};
    run_split(tbl, {});
    run_split(tbl, {1});
    run_split(tbl, {2});
    run_split(tbl, {1, 3});
  }
}

TEST_F(ShuffleSplitTests, Struct)
{
  cudf::size_type const num_rows = 10000;
  auto iter                      = thrust::make_counting_iterator(0);
  cudf::test::fixed_width_column_wrapper<int> col0(iter, iter + num_rows);

  cudf::test::fixed_width_column_wrapper<float> child0(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<int16_t> child1(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<int8_t> child2(iter, iter + num_rows);
  std::vector<std::unique_ptr<cudf::column>> struct_children;
  struct_children.push_back(child0.release());
  struct_children.push_back(child1.release());
  struct_children.push_back(child2.release());
  cudf::test::structs_column_wrapper col1(std::move(struct_children));

  // 4 columns split once
  {
    cudf::table_view tbl{
      {static_cast<cudf::column_view>(col0), static_cast<cudf::column_view>(col1)}};
    run_split(tbl, {10});
  }

  // 4 columns split multiple times
  {
    cudf::table_view tbl{
      {static_cast<cudf::column_view>(col0), static_cast<cudf::column_view>(col1)}};
    run_split(tbl, {10, 100, 2756, 7777});
  }
}

TEST_F(ShuffleSplitTests, Nulls)
{
  cudf::size_type const num_rows = 10000;
  auto iter                      = thrust::make_counting_iterator(0);
  auto validity                  = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](cudf::size_type i) { return i % 3; });

  cudf::test::fixed_width_column_wrapper<int> col(iter, iter + num_rows, validity);
  cudf::table_view tbl{{static_cast<cudf::column_view>(col)}};
  run_split(tbl, {3});
}

TEST_F(ShuffleSplitTests, ShortNulls)
{
  // one full 32 bit validity word
  {
    cudf::test::fixed_width_column_wrapper<int> col(
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
      {1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1,
       0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1});

    cudf::table_view tbl{{static_cast<cudf::column_view>(col)}};

    // try each bit seperately
    for (int idx = 0; idx < static_cast<cudf::column_view>(col).size(); idx++) {
      run_split(tbl, {idx});
    }
    // all bits in the first byte
    run_split(tbl, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    // all bits in the entire word
    run_split(tbl, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});

    // misc
    run_split(tbl, {});
    run_split(tbl, {2, 5, 18, 30});
    run_split(tbl, {8, 16, 24});
    run_split(tbl, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30});
  }

  // less than one word
  {
    cudf::test::fixed_width_column_wrapper<int> col({0, 1, 2, 3, 4, 5, 6}, {1, 1, 1, 0, 0, 0, 0});

    cudf::table_view tbl{{static_cast<cudf::column_view>(col)}};

    for (int idx = 0; idx < static_cast<cudf::column_view>(col).size(); idx++) {
      run_split(tbl, {idx});
    }

    run_split(tbl, {});
    run_split(tbl, {0, 1, 2, 3, 4, 5});
    run_split(tbl, {2, 5});
    run_split(tbl, {0, 2, 4});
  }

  // 34 rows (two input words) split such that the last two rows of validity are written as overflow
  // bits in the copy kernel.
  // - split at row 22, causing the second copy to have 12 bits to write.
  // - the first 10 of those 12 bits goes in the first word, the last 2 bits goes into the second
  // word. this is an
  //   edge case in the copy kernel involving an early-out.
  {
    cudf::test::fixed_width_column_wrapper<int> col(
      {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34},
      {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1,
       1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1});
    cudf::table_view tbl{{col}};
    run_split(tbl, {22});
  }
}

TEST_F(ShuffleSplitTests, PurgeNulls)
{
  // any column with 0 rows in it should purge nullability as part of the split, resulting in
  // non-nullable outputs at assemble side.

  // manually construct a column with a non-null validity buffer to force it to appear nullable
  auto validity_buffer =
    rmm::device_buffer{1, cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref()};
  auto col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_to_id<float>()},
                                            0,
                                            rmm::device_buffer{},
                                            std::move(validity_buffer),
                                            0);
  CUDF_EXPECTS(col->nullable(), "Expected a nullable input column");

  cudf::table_view tbl{{*col}};
  auto result = run_split(tbl, {});
  CUDF_EXPECTS(!make_table_view(result).column(0).nullable(),
               "Got a nullable column when none was expected");
}

TEST_F(ShuffleSplitTests, EmptyOffsets)
{
  // test various cases where there are no offsets in offset-based columns:
  // - string columns that have no offset child at all
  // - string or list columns with no rows
  // both cases should propagate no offset data, instead of just naively sending (num_rows+1)

  // list<string> with empty strings
  cudf::test::strings_column_wrapper strings0{};
  cudf::test::fixed_width_column_wrapper<int> offsets0{0, 0, 0};
  auto col0 = cudf::make_lists_column(2, offsets0.release(), strings0.release(), 0, {});
  cudf::lists_column_view lcv(*col0);
  CUDF_EXPECTS(lcv.child().num_children() == 0, "String column is expected to have no offsets");

  // list<list<int>> with empty inner list
  cudf::test::lists_column_wrapper<int> list0{};
  cudf::test::fixed_width_column_wrapper<int> offsets1{0, 0, 0};
  auto col1 = cudf::make_lists_column(2, offsets1.release(), list0.release(), 0, {});

  // list<struct<int, int>>
  cudf::test::fixed_width_column_wrapper<int> ints0{-210, 311};
  cudf::test::fixed_width_column_wrapper<int> ints1{293, 992};
  std::vector<std::unique_ptr<cudf::column>> inner_children;
  inner_children.push_back(ints0.release());
  inner_children.push_back(ints1.release());
  cudf::test::structs_column_wrapper inner_struct(std::move(inner_children));
  cudf::test::fixed_width_column_wrapper<int> offsets2{0, 1, 2};
  auto col2 = cudf::make_lists_column(2, offsets2.release(), inner_struct.release(), 0, {});

  cudf::table_view tbl{{*col0, *col1, *col2, *col1}};
  auto result = run_split(tbl, {});
}

TEST_F(ShuffleSplitTests, EmptySplits)
{
  cudf::size_type const num_rows = 100;
  auto iter                      = thrust::make_counting_iterator(0);
  cudf::test::fixed_width_column_wrapper<int> col0(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<float> col1(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<int16_t> col2(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<int8_t> col3(iter, iter + num_rows);

  cudf::table_view tbl{{static_cast<cudf::column_view>(col0),
                        static_cast<cudf::column_view>(col1),
                        static_cast<cudf::column_view>(col2),
                        static_cast<cudf::column_view>(col3)}};
  run_split(tbl, {});
}

TEST_F(ShuffleSplitTests, EmptyInputs)
{
  cudf::size_type const num_rows = 0;
  auto iter                      = thrust::make_counting_iterator(0);
  cudf::test::fixed_width_column_wrapper<int> col0(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<float> col1(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<int16_t> col2(iter, iter + num_rows);
  cudf::test::fixed_width_column_wrapper<int8_t> col3(iter, iter + num_rows);

  cudf::table_view tbl{{static_cast<cudf::column_view>(col0),
                        static_cast<cudf::column_view>(col1),
                        static_cast<cudf::column_view>(col2),
                        static_cast<cudf::column_view>(col3)}};
  run_split(tbl, {});
}

TEST_F(ShuffleSplitTests, NestedTypes)
{
  // struct<list, list, string>
  {
    using lcw = cudf::test::lists_column_wrapper<int64_t>;
    lcw col0{{9, 8},
             {7, 6, 5},
             {},
             {4},
             {3, 2, 1, 0},
             {20, 21, 22, 23, 24},
             {},
             {66, 666},
             {123, 7},
             {100, 101}};
    lcw col1{{1, 2, 3},
             {7},
             {99, 100},
             {4, 5, 6},
             {3, 2, 1, 0},
             {20, 21, 22, 23, 24},
             {},
             {66, 666},
             {123, 7},
             {100, 101, -1, -2, -3, -4 - 5, -6}};

    cudf::test::strings_column_wrapper col2{
      {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}};

    std::vector<std::unique_ptr<cudf::column>> struct_children;
    struct_children.push_back(col0.release());
    struct_children.push_back(col1.release());
    struct_children.push_back(col2.release());
    cudf::test::structs_column_wrapper struct_col(std::move(struct_children));

    {
      cudf::table_view tbl{{static_cast<cudf::column_view>(struct_col)}};
      run_split(tbl, {});
      run_split(tbl, {2});
      run_split(tbl, {2, 4});
    }
  }

  // list<struct<list, list>>
  // this test specifically triggers an important case in the code: branching row counts caused
  // by structs
  {
    // struct<list, list>
    using lcw = cudf::test::lists_column_wrapper<int64_t>;
    lcw col0{{9, 8},
             {7, 6, 5},
             {},
             {4},
             {3, 2, 1, 0},
             {20, 21, 22, 23, 24},
             {},
             {66, 666},
             {123, 7},
             {100, 101},
             {1, 1, 1},
             {2},
             {0},
             {2256, 12, 224, 5},
             {9, 9, 9, 9, 9},
             {-1, -2}};
    lcw col1{{1, 2, 3},
             {7},
             {99, 100},
             {4, 5, 6},
             {3, 2, 1, 0},
             {20, 21, 22, 23, 24},
             {1},
             {66, 666},
             {123, 7},
             {100, 101, -1, -2, -3, -4 - 5, -6},
             {},
             {6, 5, 4, 3, 2, 1, 0},
             {-10, 0, 1},
             {0, 0, 0, 0},
             {},
             {0, 1, 0, 1, 0, 1}};
    std::vector<std::unique_ptr<cudf::column>> struct_children;
    struct_children.push_back(col0.release());
    struct_children.push_back(col1.release());
    cudf::test::structs_column_wrapper struct_col(std::move(struct_children));

    // list<struct<list, list>>
    cudf::test::fixed_width_column_wrapper<int> offsets{0, 2, 4, 6, 7, 9, 9, 12, 16};
    auto list_col = cudf::make_lists_column(8, offsets.release(), struct_col.release(), 0, {});

    cudf::table_view tbl{{*list_col}};
    run_split(tbl, {});
    run_split(tbl, {2});
    run_split(tbl, {2, 4});
  }
}

TEST_F(ShuffleSplitTests, Reshaping)
{
  // a key feature of the kud0 format is being able to reassemble arbitrary partitions.
  // so for example, if we had a shuffle_split() call that produced partitions ABCD, we might
  // want to reassemble that as CDAB. or we may have multiple shuffle_split calls producing multiple
  // sets of partitions that need to be stitched together, such as:
  // - shuffle_split()    -> ABCD
  // - shuffle_split()    -> XYZW
  // - shuffle_assemble(AXBYCDWZ)

  // fixed-width
  {
    cudf::test::fixed_width_column_wrapper<int> col0{
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
      {1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0}};
    cudf::table_view tbl{{col0}};
    run_split(tbl, {2, 10}, {2, 0, 1});
  }

  // strings
  {
    cudf::test::strings_column_wrapper col0{
      "abc", "d", "ef", "g", "", "hijklmn", "o", "pqrst", "u", "v", "wy", "xz"};
    cudf::table_view tbl{{col0}};
    run_split(tbl, {2, 10}, {2, 0, 1});
  }

  // lists
  {
    using lcw = cudf::test::lists_column_wrapper<uint64_t>;
    lcw col0{{9, 8},
             {7, 6, 5},
             {},
             {4},
             {3, 2, 1, 0},
             {20, 21, 22, 23, 24},
             {},
             {66, 666},
             {123, 7},
             {100, 101}};

    cudf::table_view tbl{{static_cast<cudf::column_view>(col0)}};
    run_split(tbl, {1, 4}, {2, 0, 1});
  }

  // nested lists
  {
    using lcw = cudf::test::lists_column_wrapper<uint64_t>;
    lcw col0{{{9, 8}, {7, 6, 5}},
             {lcw{}, {4}},
             {{3, 2, 1, 0}, {20, 21, 22, 23, 24}},
             {lcw{}, {66, 666}},
             {{123, 7}, {100, 101}},
             {{1, 2, 4}, {8, 6, 5}}};

    cudf::table_view tbl{{static_cast<cudf::column_view>(col0)}};
    run_split(tbl, {1, 4}, {2, 0, 1});
  }

  // list<struct<list, list>>
  {
    using lcw = cudf::test::lists_column_wrapper<int64_t>;
    lcw col0{{9, 8},
             {7, 6, 5},
             {},
             {4},
             {3, 2, 1, 0},
             {20, 21, 22, 23, 24},
             {},
             {66, 666},
             {123, 7},
             {100, 101},
             {1, 1, 1},
             {2},
             {0},
             {2256, 12, 224, 5},
             {9, 9, 9, 9, 9},
             {-1, -2}};
    lcw col1{{1, 2, 3},
             {7},
             {99, 100},
             {4, 5, 6},
             {3, 2, 1, 0},
             {20, 21, 22, 23, 24},
             {1},
             {66, 666},
             {123, 7},
             {100, 101, -1, -2, -3, -4 - 5, -6},
             {},
             {6, 5, 4, 3, 2, 1, 0},
             {-10, 0, 1},
             {0, 0, 0, 0},
             {},
             {0, 1, 0, 1, 0, 1}};
    std::vector<std::unique_ptr<cudf::column>> struct_children;
    struct_children.push_back(col0.release());
    struct_children.push_back(col1.release());
    cudf::test::structs_column_wrapper struct_col(std::move(struct_children));

    // list<struct<list, list>>
    cudf::test::fixed_width_column_wrapper<int> offsets{0, 2, 4, 6, 7, 9, 9, 12, 16};
    auto list_col = cudf::make_lists_column(8, offsets.release(), struct_col.release(), 0, {});

    cudf::table_view tbl{{*list_col}};
    run_split(tbl, {2, 4}, {2, 0, 1});
  }
}

TEST_F(ShuffleSplitTests, LargeBatchSimple)
{
  // 8 million rows per validity copy batch.
  constexpr size_t rows_per_column = 32 * 1024 * 1024;

  srand(31337);
  auto validity_iter = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](int i) { return rand() % 2 == 0; });
  auto iter = thrust::make_counting_iterator(0);

  cudf::test::fixed_width_column_wrapper<int> col0(iter, iter + rows_per_column, validity_iter);

  cudf::table_view tbl{{col0}};
  run_split(tbl, {});
  run_split(tbl, {0});
  run_split(tbl, {1});
  run_split(tbl, {31});
  run_split(tbl, {32});
  run_split(tbl, {63});
  run_split(tbl, {64});
  run_split(tbl, {rows_per_column - 1});
  run_split(tbl, {rows_per_column - 31});
  run_split(tbl, {rows_per_column - 32});
  run_split(tbl, {rows_per_column - 33});
  run_split(tbl, {8000001, 16000003});
}

TEST_F(ShuffleSplitTests, FixedPoint)
{
  constexpr auto num_rows = 500'000;

  auto vals0 = random_values<int16_t>(num_rows);
  auto vals1 = random_values<int32_t>(num_rows);
  auto vals2 = random_values<int64_t>(num_rows);

  using cudf::test::iterators::no_nulls;
  cudf::test::fixed_point_column_wrapper<numeric::decimal32::rep> col0(
    vals0.begin(), vals0.end(), no_nulls(), numeric::scale_type{5});
  cudf::test::fixed_point_column_wrapper<numeric::decimal64::rep> col1(
    vals1.begin(), vals1.end(), no_nulls(), numeric::scale_type{-5});
  cudf::test::fixed_point_column_wrapper<numeric::decimal128::rep> col2(
    vals2.begin(), vals2.end(), no_nulls(), numeric::scale_type{-6});

  srand(31337);
  auto validity_iter = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](int i) { return rand() % 2 == 0; });
  cudf::test::fixed_point_column_wrapper<numeric::decimal32::rep> col3(
    vals0.begin(), vals0.end(), validity_iter, numeric::scale_type{5});
  cudf::test::fixed_point_column_wrapper<numeric::decimal64::rep> col4(
    vals1.begin(), vals1.end(), validity_iter, numeric::scale_type{-5});
  cudf::test::fixed_point_column_wrapper<numeric::decimal128::rep> col5(
    vals2.begin(), vals2.end(), validity_iter, numeric::scale_type{-6});

  cudf::table_view tbl{{col0, col1, col2, col3, col4, col5}};
  run_split(tbl, {});
  run_split(tbl, {100});
  run_split(tbl, {1000, num_rows - 1000});
}

TEST_F(ShuffleSplitTests, NestedTerminatingEmptyPartition)
{
  // edge case: non-root columns that contain offsets, where the trailing partitions
  // are completely empty. the situation this causes is that there are no offsets for
  // these partitions, so the code that was selecting which copy batch should apply the terminating
  // offset to the destination was incorrect.  It was selecting "the last partition"
  // when it should have been "the last non-empty partition". In the case where there are no rows in
  // any partition, there are no offsets to terminate.

  // empty trailing partitions
  {
    // list<list<struct<string, string>>>
    cudf::test::strings_column_wrapper str0{"k1", "k4", "k8", "k14", "k16", "k19", "k22", "k22"};
    cudf::test::strings_column_wrapper str1{{"v1", "v4", "v8", "v14", "v16", "v19", "v22", ""},
                                            {1, 1, 1, 1, 1, 1, 1, 0}};
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(str0.release());
    children.push_back(str1.release());
    cudf::test::structs_column_wrapper str(std::move(children));

    cudf::test::fixed_width_column_wrapper<int> inner_offsets{0, 1, 2, 3, 4, 5, 6, 7, 8};
    auto inner_list = cudf::make_lists_column(8, inner_offsets.release(), str.release(), 0, {});

    cudf::test::fixed_width_column_wrapper<int> outer_offsets{
      0, 1, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 5, 6, 7, 7, 8, 8, 8};
    std::vector<int> outer_valids{1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0};
    auto outer_validity =
      cudf::test::detail::make_null_mask(outer_valids.begin(), outer_valids.end());
    auto outer_list = cudf::make_lists_column(
      18, outer_offsets.release(), std::move(inner_list), 9, std::move(outer_validity.first));

    cudf::table_view tbl{{*outer_list}};
    run_split(tbl, {2, 4, 6, 8, 10, 12, 14, 16});
  }

  // list<string>
  // the string in this case is completely empty, so none of the partitions for it contain any rows,
  // hence no terminating offset.
  {
    cudf::test::strings_column_wrapper str{};
    cudf::test::fixed_width_column_wrapper<int> offsets{0, 0, 0, 0, 0, 0, 0, 0, 0};
    auto list = cudf::make_lists_column(8, offsets.release(), str.release(), 0, {});
    cudf::table_view tbl{{*list}};
    run_split(tbl, {2, 4, 6});
  }
}

TEST_F(ShuffleSplitTests, EmptyPartitionsWithNulls)
{
  // tests the case where an input column has nulls, but one of the
  // partitions of that column does not (because it has no rows).
  cudf::test::fixed_width_column_wrapper<int> i0{{0, 4, 7}, {0, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int> o0{0, 1, 2, 3, 3, 3, 3};
  std::vector<int> list_valids{1, 1, 1, 0, 1, 1};
  auto list_validity = cudf::test::detail::make_null_mask(list_valids.begin(), list_valids.end());
  auto col0          = cudf::make_lists_column(
    6, o0.release(), i0.release(), list_validity.second, std::move(list_validity.first));

  cudf::table_view tbl{{*col0}};
  // by splitting at row 3, the inner int column will have no rows in the second partition and
  // should therefore not be including nulls in that partition's header.
  run_split(tbl, {3});
}

TEST_F(ShuffleSplitTests, MixedValidity)
{
  // test assembling partitions where some of the column instances contain validity, but others do
  // not.
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  auto run_mixed_null_split =
    [stream, mr](int num_rows, std::vector<int> const& splits, std::vector<bool> const& has_nulls) {
      CUDF_EXPECTS(has_nulls.size() == splits.size() + 1,
                   "Invalid set of splits and has_nulls vectors");

      auto value_iter =
        spark_rapids_jni::util::make_counting_transform_iterator(0, [](int i) { return i; });
      auto valid_iter =
        spark_rapids_jni::util::make_counting_transform_iterator(0, [](int i) { return i % 2; });
      cudf::test::fixed_width_column_wrapper<int> base(value_iter, value_iter + num_rows);

      // make the column with a specified series of nullable partitions. all other partitions will
      // remain non-nullable
      auto split_cols = cudf::split(base, splits);
      std::vector<std::unique_ptr<cudf::column>> partition_cols;
      std::vector<cudf::column_view> partition_views;
      for (size_t idx = 0; idx < split_cols.size(); idx++) {
        auto new_col = std::make_unique<cudf::column>(split_cols[idx]);
        // if this gap has nulls, make this specific column piece nullable and add some nulls
        if (has_nulls[idx]) {
          auto nm = cudf::test::detail::make_null_mask(valid_iter, valid_iter + new_col->size());
          new_col->set_null_mask(std::move(nm.first), nm.second);
        }
        partition_views.push_back(*new_col);
        partition_cols.push_back(std::move(new_col));
      }

      // make the expected table
      auto expected = cudf::concatenate(partition_views, stream, mr);
      cudf::table_view expected_t{{static_cast<cudf::column_view>(*expected)}};

      // make the concatenated shuffle_split partitions
      std::vector<spark_rapids_jni::shuffle_split_output> shuf;
      size_t total_size = 0;
      for (size_t idx = 0; idx < partition_views.size(); idx++) {
        shuf.push_back(spark_rapids_jni::shuffle_split(
          cudf::table_view{{partition_views[idx]}}, {}, stream, mr));
        total_size += shuf.back().result.partitions->size();
      }
      rmm::device_uvector<uint8_t> full{total_size, stream, mr};
      rmm::device_uvector<size_t> full_offsets{partition_views.size() + 1, stream, mr};
      std::vector<size_t> h_full_offsets(partition_views.size() + 1);
      size_t pos = 0;
      for (size_t idx = 0; idx < partition_views.size(); idx++) {
        cudaMemcpy(static_cast<uint8_t*>(full.data()) + pos,
                   shuf[idx].result.partitions->data(),
                   shuf[idx].result.partitions->size(),
                   cudaMemcpyDefault);
        h_full_offsets[idx] = pos;
        pos += shuf[idx].result.partitions->size();
      }
      h_full_offsets[partition_views.size()] = pos;
      cudaMemcpy(full_offsets.data(),
                 h_full_offsets.data(),
                 sizeof(size_t) * h_full_offsets.size(),
                 cudaMemcpyDefault);

      spark_rapids_jni::shuffle_split_metadata md;
      md.col_info.push_back({cudf::type_id::INT32, 0});
      auto result = spark_rapids_jni::shuffle_assemble(md, full, full_offsets, stream, mr);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_t.column(0), *(result.column_views[0]));
    };

  {
    constexpr int num_rows = 256;
    std::vector<int> splits{1};
    std::vector<bool> has_nulls{0, 1};
    run_mixed_null_split(num_rows, splits, has_nulls);
  }
  {
    constexpr int num_rows = 256;
    std::vector<int> splits{1};
    std::vector<bool> has_nulls{1, 0};
    run_mixed_null_split(num_rows, splits, has_nulls);
  }

  {
    constexpr int num_rows = 256;
    std::vector<int> splits{11, 32};
    std::vector<bool> has_nulls{0, 1, 0};
    run_mixed_null_split(num_rows, splits, has_nulls);
  }

  {
    constexpr int num_rows = 1024;
    std::vector<int> splits{256, 512, 768};
    std::vector<bool> has_nulls{1, 0, 1, 0};
    run_mixed_null_split(num_rows, splits, has_nulls);
  }

  {
    constexpr int num_rows = 1024;
    std::vector<int> splits{256, 512, 768};
    std::vector<bool> has_nulls{0, 1, 0, 1};
    run_mixed_null_split(num_rows, splits, has_nulls);
  }

  {
    constexpr int num_rows = 1024;
    std::vector<int> splits{62, 63};
    std::vector<bool> has_nulls{1, 0, 1};
    run_mixed_null_split(num_rows, splits, has_nulls);
  }

  {
    constexpr int num_rows = 1024;
    std::vector<int> splits{62, 63};
    std::vector<bool> has_nulls{0, 1, 0};
    run_mixed_null_split(num_rows, splits, has_nulls);
  }

  {
    constexpr int num_rows = 1024;
    std::vector<int> splits{62, 97};
    std::vector<bool> has_nulls{1, 0, 1};
    run_mixed_null_split(num_rows, splits, has_nulls);
  }

  {
    constexpr int num_rows = 1024;
    std::vector<int> splits{62, 97};
    std::vector<bool> has_nulls{0, 1, 0};
    run_mixed_null_split(num_rows, splits, has_nulls);
  }

  {
    constexpr int num_rows = 1024;
    std::vector<int> splits{62, 500, 768, 901};
    std::vector<bool> has_nulls{0, 1, 1, 0, 0};
    run_mixed_null_split(num_rows, splits, has_nulls);
  }
}
