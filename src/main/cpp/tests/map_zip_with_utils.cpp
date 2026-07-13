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
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/lists/lists_column_view.hpp>

#include <map_zip_with_utils.hpp>

using namespace cudf;
using indices_col = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

constexpr test::debug_output_level verbosity{test::debug_output_level::ALL_ERRORS};

struct MapZipWithUtilsTests : public test::BaseFixture {};

TEST_F(MapZipWithUtilsTests, BasicMapZipTest)
{
  {
    auto keys1 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6};
    auto vals1 = cudf::test::fixed_width_column_wrapper<int32_t>{{48, 27, 25, 31, 351, 351},
                                                                 {1, 1, 1, 1, 1, 1}};
    auto struct_col1 =
      cudf::test::structs_column_wrapper({keys1, vals1}, {1, 1, 1, 1, 1, 1}).release();
    auto expected_unchanged_struct_col1 = cudf::column(*struct_col1);
    auto list_offsets_column =
      cudf::test::fixed_width_column_wrapper<size_type>{0, 2, 3, 5, 6}.release();
    auto num_list_rows = list_offsets_column->size() - 1;
    auto list_col1     = cudf::make_lists_column(
      num_list_rows, std::move(list_offsets_column), std::move(struct_col1), 0, {});

    auto keys2 = cudf::test::fixed_width_column_wrapper<int32_t>{7, 8, 9, 10, 11, 12};
    auto vals2 = cudf::test::fixed_width_column_wrapper<int32_t>{{12, 35, 25, 31, 351, 351},
                                                                 {1, 1, 1, 1, 1, 1}};
    auto struct_col2 =
      cudf::test::structs_column_wrapper({keys2, vals2}, {1, 1, 1, 1, 1, 1}).release();
    auto expected_unchanged_struct_col2 = cudf::column(*struct_col2);
    auto list_offsets_column2 =
      cudf::test::fixed_width_column_wrapper<size_type>{0, 2, 3, 5, 6}.release();
    auto num_list_rows2 = list_offsets_column2->size() - 1;
    auto list_col2      = cudf::make_lists_column(
      num_list_rows2, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto const k =
      cudf::test::fixed_width_column_wrapper<size_type>{1, 2, 7, 8, 3, 9, 4, 5, 10, 11, 6, 12};
    auto const v1 = cudf::test::fixed_width_column_wrapper<size_type>{
      {48, 27, 3, 4, 25, 6, 31, 351, 9, 10, 351, 12}, {1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}};
    auto const v2 = cudf::test::fixed_width_column_wrapper<size_type>{
      {5, 6, 12, 35, 1, 25, 3, 3, 31, 351, 2, 351}, {0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1}};
    auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                             cudf::lists_column_view(*list_col2));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1), verbosity);
  }
}

TEST_F(MapZipWithUtilsTests, NullMapTest)
{
  {
    auto keys1 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6};
    auto vals1 = cudf::test::fixed_width_column_wrapper<int32_t>{{48, 27, 25, 31, 351, 351},
                                                                 {1, 1, 1, 1, 1, 1}};
    auto struct_col1 =
      cudf::test::structs_column_wrapper({keys1, vals1}, {1, 1, 1, 1, 1, 1}).release();
    auto expected_unchanged_struct_col1 = cudf::column(*struct_col1);
    auto list_offsets_column =
      cudf::test::fixed_width_column_wrapper<size_type>{0, 2, 3, 5, 6}.release();
    auto num_list_rows = list_offsets_column->size() - 1;
    auto mask          = cudf::create_null_mask(4, cudf::mask_state::ALL_VALID);
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), 1, 2, false);
    auto list_col1 = cudf::make_lists_column(
      num_list_rows, std::move(list_offsets_column), std::move(struct_col1), 1, std::move(mask));

    auto keys2 = cudf::test::fixed_width_column_wrapper<int32_t>{7, 8, 9, 10, 11, 12};
    auto vals2 = cudf::test::fixed_width_column_wrapper<int32_t>{{12, 35, 25, 31, 351, 351},
                                                                 {1, 1, 1, 1, 1, 1}};
    auto struct_col2 =
      cudf::test::structs_column_wrapper({keys2, vals2}, {1, 1, 1, 1, 1, 1}).release();
    auto expected_unchanged_struct_col2 = cudf::column(*struct_col2);
    auto list_offsets_column2 =
      cudf::test::fixed_width_column_wrapper<size_type>{0, 2, 3, 5, 6}.release();
    auto num_list_rows2 = list_offsets_column2->size() - 1;
    auto list_col2      = cudf::make_lists_column(
      num_list_rows2, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto const k = cudf::test::fixed_width_column_wrapper<size_type>{
      {1, 2, 7, 8, 4, 5, 10, 11, 6, 12}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
    auto const v1 = cudf::test::fixed_width_column_wrapper<size_type>{
      {48, 27, 3, 4, 31, 351, 9, 10, 351, 12}, {1, 1, 0, 0, 1, 1, 0, 0, 1, 0}};
    auto const v2 = cudf::test::fixed_width_column_wrapper<size_type>{
      {5, 6, 12, 35, 3, 3, 31, 351, 2, 351}, {0, 0, 1, 1, 0, 0, 1, 1, 0, 1}};
    auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                             cudf::lists_column_view(*list_col2));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1), verbosity);
    // check to see of results size and list_col1 size are the same
    EXPECT_EQ(cudf::lists_column_view(*list_col1).size(), cudf::lists_column_view(*results).size());
  }
}

TEST_F(MapZipWithUtilsTests, CharKeysTest)
{
  {
    std::initializer_list<std::string> names = {"a", "b", "c", "d", "e", "f"};
    auto keys1 = cudf::test::strings_column_wrapper{names.begin(), names.end()};
    auto vals1 = cudf::test::fixed_width_column_wrapper<int32_t>{{48, 27, 25, 31, 351, 351},
                                                                 {1, 1, 1, 1, 1, 1}};
    auto struct_col1 =
      cudf::test::structs_column_wrapper({keys1, vals1}, {1, 1, 1, 1, 1, 1}).release();
    auto list_offsets_column =
      cudf::test::fixed_width_column_wrapper<size_type>{0, 2, 3, 5, 6}.release();
    auto num_list_rows = list_offsets_column->size() - 1;
    auto list_col1     = cudf::make_lists_column(
      num_list_rows, std::move(list_offsets_column), std::move(struct_col1), 0, {});

    std::initializer_list<std::string> names2 = {"g", "h", "i", "j", "k", "l"};
    auto keys2 = cudf::test::strings_column_wrapper{names2.begin(), names2.end()};
    auto vals2 = cudf::test::fixed_width_column_wrapper<int32_t>{{12, 35, 25, 31, 351, 351},
                                                                 {1, 1, 1, 1, 1, 1}};
    auto struct_col2 =
      cudf::test::structs_column_wrapper({keys2, vals2}, {1, 1, 1, 1, 1, 1}).release();
    auto expected_unchanged_struct_col2 = cudf::column(*struct_col2);
    auto list_offsets_column2 =
      cudf::test::fixed_width_column_wrapper<size_type>{0, 2, 3, 5, 6}.release();
    auto num_list_rows2 = list_offsets_column2->size() - 1;
    auto list_col2      = cudf::make_lists_column(
      num_list_rows2, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto const k = cudf::test::strings_column_wrapper{
      "a", "b", "g", "h", "c", "i", "d", "e", "j", "k", "f", "l"};
    auto const v1 = cudf::test::fixed_width_column_wrapper<size_type>{
      {48, 27, 3, 4, 25, 6, 31, 351, 9, 31, 351, 351}, {1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0}};
    auto const v2 = cudf::test::fixed_width_column_wrapper<size_type>{
      {5, 6, 12, 35, 1, 25, 3, 3, 31, 351, 351, 351}, {0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1}};
    auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                             cudf::lists_column_view(*list_col2));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1), verbosity);
  }
}

TEST_F(MapZipWithUtilsTests, StringKeysTest)
{
  {
    // Test with string keys and multiple rows in the list
    std::initializer_list<std::string> names1 = {
      "apple", "banana", "cherry", "date", "strawberry", "fig", "grape"};
    auto keys1 = cudf::test::strings_column_wrapper{names1.begin(), names1.end()};
    auto vals1 = cudf::test::fixed_width_column_wrapper<int32_t>{
      {100, 200, 300, 400, 500, 600, 700}, {1, 1, 1, 1, 1, 1, 1}};
    auto struct_col1 =
      cudf::test::structs_column_wrapper({keys1, vals1}, {1, 1, 1, 1, 1, 1, 1}).release();
    auto list_offsets_column1 =
      cudf::test::fixed_width_column_wrapper<size_type>{0, 2, 4, 7}.release();
    auto num_list_rows1 = list_offsets_column1->size() - 1;
    auto list_col1      = cudf::make_lists_column(
      num_list_rows1, std::move(list_offsets_column1), std::move(struct_col1), 0, {});

    std::initializer_list<std::string> names2 = {
      "banana", "cherry", "date", "fig", "grape", "honeydew", "kiwi"};
    auto keys2 = cudf::test::strings_column_wrapper{names2.begin(), names2.end()};
    auto vals2 = cudf::test::fixed_width_column_wrapper<int32_t>{
      {250, 350, 450, 650, 750, 850, 950}, {1, 1, 1, 1, 1, 1, 1}};
    auto struct_col2 =
      cudf::test::structs_column_wrapper({keys2, vals2}, {1, 1, 1, 1, 1, 1, 1}).release();
    auto list_offsets_column2 =
      cudf::test::fixed_width_column_wrapper<size_type>{0, 3, 5, 7}.release();
    auto num_list_rows2 = list_offsets_column2->size() - 1;
    auto list_col2      = cudf::make_lists_column(
      num_list_rows2, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto results  = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                              cudf::lists_column_view(*list_col2));
    auto const k  = cudf::test::strings_column_wrapper{"apple",
                                                       "banana",
                                                       "cherry",
                                                       "date",
                                                       "cherry",
                                                       "date",
                                                       "fig",
                                                       "grape",
                                                       "strawberry",
                                                       "fig",
                                                       "grape",
                                                       "honeydew",
                                                       "kiwi"};
    auto const v1 = cudf::test::fixed_width_column_wrapper<int32_t>{
      {100, 200, 0, 0, 300, 400, 0, 0, 500, 600, 700, 0, 0},
      {1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0}};
    auto const v2 = cudf::test::fixed_width_column_wrapper<int32_t>{
      {0, 250, 350, 450, 0, 0, 650, 750, 0, 0, 0, 850, 950},
      {0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1), verbosity);
  }
}

TEST_F(MapZipWithUtilsTests, OneEmptyMapTest)
{
  {
    // Test with one empty map and one non-empty map
    auto keys1       = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
    auto vals1       = cudf::test::fixed_width_column_wrapper<int32_t>{{10, 20, 30}, {1, 1, 1}};
    auto struct_col1 = cudf::test::structs_column_wrapper({keys1, vals1}, {1, 1, 1}).release();
    auto list_offsets_column1 =
      cudf::test::fixed_width_column_wrapper<size_type>{0, 2, 3}.release();
    auto list_col1 =
      cudf::make_lists_column(2, std::move(list_offsets_column1), std::move(struct_col1), 0, {});

    auto keys2       = cudf::test::fixed_width_column_wrapper<int32_t>{};
    auto vals2       = cudf::test::fixed_width_column_wrapper<int32_t>{};
    auto struct_col2 = cudf::test::structs_column_wrapper({keys2, vals2}).release();
    auto list_offsets_column2 =
      cudf::test::fixed_width_column_wrapper<size_type>{0, 0, 0}.release();
    auto list_col2 =
      cudf::make_lists_column(2, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                             cudf::lists_column_view(*list_col2));

    auto const k  = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
    auto const v1 = cudf::test::fixed_width_column_wrapper<int32_t>{{10, 20, 30}, {1, 1, 1}};
    auto const v2 = cudf::test::fixed_width_column_wrapper<int32_t>{{0, 0, 0}, {0, 0, 0}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1), verbosity);
  }
}

TEST_F(MapZipWithUtilsTests, OverlappingKeysTest)
{
  {
    // Test with overlapping keys between maps
    auto keys1 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4};
    auto vals1 =
      cudf::test::fixed_width_column_wrapper<int32_t>{{100, 200, 300, 400}, {1, 1, 1, 1}};
    auto struct_col1 = cudf::test::structs_column_wrapper({keys1, vals1}, {1, 1, 1, 1}).release();
    auto list_offsets_column1 = cudf::test::fixed_width_column_wrapper<size_type>{0, 4}.release();
    auto list_col1 =
      cudf::make_lists_column(1, std::move(list_offsets_column1), std::move(struct_col1), 0, {});

    auto keys2 = cudf::test::fixed_width_column_wrapper<int32_t>{2, 3, 5, 6};
    auto vals2 =
      cudf::test::fixed_width_column_wrapper<int32_t>{{250, 350, 500, 600}, {1, 1, 1, 1}};
    auto struct_col2 = cudf::test::structs_column_wrapper({keys2, vals2}, {1, 1, 1, 1}).release();
    auto list_offsets_column2 = cudf::test::fixed_width_column_wrapper<size_type>{0, 4}.release();
    auto list_col2 =
      cudf::make_lists_column(1, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                             cudf::lists_column_view(*list_col2));

    auto const k  = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6};
    auto const v1 = cudf::test::fixed_width_column_wrapper<int32_t>{{100, 200, 300, 400, 0, 0},
                                                                    {1, 1, 1, 1, 0, 0}};
    auto const v2 = cudf::test::fixed_width_column_wrapper<int32_t>{{0, 250, 350, 0, 500, 600},
                                                                    {0, 1, 1, 0, 1, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1), verbosity);
  }
}

TEST_F(MapZipWithUtilsTests, NonOverlappingKeysTest)
{
  {
    // Test with completely non-overlapping keys
    auto keys1       = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
    auto vals1       = cudf::test::fixed_width_column_wrapper<int32_t>{{10, 20, 30}, {1, 1, 1}};
    auto struct_col1 = cudf::test::structs_column_wrapper({keys1, vals1}, {1, 1, 1}).release();
    auto list_offsets_column1 = cudf::test::fixed_width_column_wrapper<size_type>{0, 3}.release();
    auto list_col1 =
      cudf::make_lists_column(1, std::move(list_offsets_column1), std::move(struct_col1), 0, {});

    auto keys2       = cudf::test::fixed_width_column_wrapper<int32_t>{4, 5, 6};
    auto vals2       = cudf::test::fixed_width_column_wrapper<int32_t>{{40, 50, 60}, {1, 1, 1}};
    auto struct_col2 = cudf::test::structs_column_wrapper({keys2, vals2}, {1, 1, 1}).release();
    auto list_offsets_column2 = cudf::test::fixed_width_column_wrapper<size_type>{0, 3}.release();
    auto list_col2 =
      cudf::make_lists_column(1, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                             cudf::lists_column_view(*list_col2));

    auto const k = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6};
    auto const v1 =
      cudf::test::fixed_width_column_wrapper<int32_t>{{10, 20, 30, 0, 0, 0}, {1, 1, 1, 0, 0, 0}};
    auto const v2 =
      cudf::test::fixed_width_column_wrapper<int32_t>{{0, 0, 0, 40, 50, 60}, {0, 0, 0, 1, 1, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1), verbosity);
  }
}

TEST_F(MapZipWithUtilsTests, MultipleRowsTest)
{
  {
    // Test with multiple rows (multiple maps)
    auto keys1 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5};
    auto vals1 =
      cudf::test::fixed_width_column_wrapper<int32_t>{{10, 20, 30, 40, 50}, {1, 1, 1, 1, 1}};
    auto struct_col1 =
      cudf::test::structs_column_wrapper({keys1, vals1}, {1, 1, 1, 1, 1}).release();
    auto list_offsets_column1 = cudf::test::fixed_width_column_wrapper<size_type>{0, 2, 5}
                                  .release();  // 2 rows: [1,2], [3,4,5]
    auto list_col1 =
      cudf::make_lists_column(2, std::move(list_offsets_column1), std::move(struct_col1), 0, {});

    auto keys2 = cudf::test::fixed_width_column_wrapper<int32_t>{2, 3, 4, 6, 7};
    auto vals2 =
      cudf::test::fixed_width_column_wrapper<int32_t>{{25, 35, 45, 65, 75}, {1, 1, 1, 1, 1}};
    auto struct_col2 =
      cudf::test::structs_column_wrapper({keys2, vals2}, {1, 1, 1, 1, 1}).release();
    auto list_offsets_column2 = cudf::test::fixed_width_column_wrapper<size_type>{0, 3, 5}
                                  .release();  // 2 rows: [2,3,4], [6,7]
    auto list_col2 =
      cudf::make_lists_column(2, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                             cudf::lists_column_view(*list_col2));

    auto const k  = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 3, 4, 5, 6, 7};
    auto const v1 = cudf::test::fixed_width_column_wrapper<int32_t>{
      {10, 20, 0, 0, 30, 40, 50, 0, 0}, {1, 1, 0, 0, 1, 1, 1, 0, 0}};
    auto const v2 = cudf::test::fixed_width_column_wrapper<int32_t>{
      {0, 25, 35, 45, 0, 0, 0, 65, 75}, {0, 1, 1, 1, 0, 0, 0, 1, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1), verbosity);
  }
}

TEST_F(MapZipWithUtilsTests, SingleElementMapsTest)
{
  {
    // Test with single element maps
    auto keys1                = cudf::test::fixed_width_column_wrapper<int32_t>{1};
    auto vals1                = cudf::test::fixed_width_column_wrapper<int32_t>{{100}, {1}};
    auto struct_col1          = cudf::test::structs_column_wrapper({keys1, vals1}, {1}).release();
    auto list_offsets_column1 = cudf::test::fixed_width_column_wrapper<size_type>{0, 1}.release();
    auto list_col1 =
      cudf::make_lists_column(1, std::move(list_offsets_column1), std::move(struct_col1), 0, {});

    auto keys2                = cudf::test::fixed_width_column_wrapper<int32_t>{2};
    auto vals2                = cudf::test::fixed_width_column_wrapper<int32_t>{{200}, {1}};
    auto struct_col2          = cudf::test::structs_column_wrapper({keys2, vals2}, {1}).release();
    auto list_offsets_column2 = cudf::test::fixed_width_column_wrapper<size_type>{0, 1}.release();
    auto list_col2 =
      cudf::make_lists_column(1, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                             cudf::lists_column_view(*list_col2));

    auto const k  = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2};
    auto const v1 = cudf::test::fixed_width_column_wrapper<int32_t>{{100, 0}, {1, 0}};
    auto const v2 = cudf::test::fixed_width_column_wrapper<int32_t>{{0, 200}, {0, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1), verbosity);
  }
}

TEST_F(MapZipWithUtilsTests, IdenticalKeysTest)
{
  {
    // Test with identical keys in both maps
    auto keys1       = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
    auto vals1       = cudf::test::fixed_width_column_wrapper<int32_t>{{100, 200, 300}, {1, 1, 1}};
    auto struct_col1 = cudf::test::structs_column_wrapper({keys1, vals1}, {1, 1, 1}).release();
    auto list_offsets_column1 = cudf::test::fixed_width_column_wrapper<size_type>{0, 3}.release();
    auto list_col1 =
      cudf::make_lists_column(1, std::move(list_offsets_column1), std::move(struct_col1), 0, {});

    auto keys2       = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
    auto vals2       = cudf::test::fixed_width_column_wrapper<int32_t>{{150, 250, 350}, {1, 1, 1}};
    auto struct_col2 = cudf::test::structs_column_wrapper({keys2, vals2}, {1, 1, 1}).release();
    auto list_offsets_column2 = cudf::test::fixed_width_column_wrapper<size_type>{0, 3}.release();
    auto list_col2 =
      cudf::make_lists_column(1, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                             cudf::lists_column_view(*list_col2));

    auto const k  = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
    auto const v1 = cudf::test::fixed_width_column_wrapper<int32_t>{{100, 200, 300}, {1, 1, 1}};
    auto const v2 = cudf::test::fixed_width_column_wrapper<int32_t>{{150, 250, 350}, {1, 1, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1), verbosity);
  }
}

TEST_F(MapZipWithUtilsTests, LargeMapsTest)
{
  {
    // Test with larger maps
    auto keys1 = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto vals1 =
      cudf::test::fixed_width_column_wrapper<int32_t>{{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}};
    auto struct_col1 = cudf::test::structs_column_wrapper({keys1, vals1}).release();
    auto list_offsets_column1 =
      cudf::test::fixed_width_column_wrapper<size_type>{0, 3, 4, 6, 6, 10}.release();
    auto list_col1 =
      cudf::make_lists_column(5, std::move(list_offsets_column1), std::move(struct_col1), 0, {});

    auto keys2 =
      cudf::test::fixed_width_column_wrapper<int32_t>{3, 11, 12, 13, 4, 5, 6, 7, 9, 8, 14, 15};
    auto vals2 = cudf::test::fixed_width_column_wrapper<int32_t>{
      {55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 163, 189}};
    auto struct_col2 = cudf::test::structs_column_wrapper({keys2, vals2}).release();
    auto list_offsets_column2 =
      cudf::test::fixed_width_column_wrapper<size_type>{0, 4, 5, 8, 8, 12}.release();
    auto list_col2 =
      cudf::make_lists_column(5, std::move(list_offsets_column2), std::move(struct_col2), 0, {});

    auto results = spark_rapids_jni::map_zip(cudf::lists_column_view(*list_col1),
                                             cudf::lists_column_view(*list_col2));

    auto const k = cudf::test::fixed_width_column_wrapper<int32_t>{
      1, 2, 3, 11, 12, 13, 4, 5, 6, 7, 7, 8, 9, 10, 14, 15};
    auto const v1 = cudf::test::fixed_width_column_wrapper<int32_t>{
      {10, 20, 30, 0, 0, 0, 40, 50, 60, 0, 70, 80, 90, 100, 0, 0},
      {1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0}};
    auto const v2 = cudf::test::fixed_width_column_wrapper<int32_t>{
      {0, 0, 55, 66, 77, 88, 99, 110, 121, 132, 0, 154, 143, 0, 163, 189},
      {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      k, cudf::lists_column_view(*results).child().child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v1, cudf::lists_column_view(*results).child().child(1).child(0), verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      v2, cudf::lists_column_view(*results).child().child(1).child(1), verbosity);
  }
}