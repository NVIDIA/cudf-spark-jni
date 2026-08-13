/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.
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
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <row_conversion.hpp>
#include <utilities/iterator.cuh>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>

struct ColumnToRowTests : public cudf::test::BaseFixture {};
struct RowToColumnTests : public cudf::test::BaseFixture {};

TEST_F(ColumnToRowTests, Single)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1});
  cudf::table_view in(std::vector<cudf::column_view>{a});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32}};

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*new_rows[i]), schema);
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, SimpleString)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1, 0, 1, 0, -1});
  cudf::test::strings_column_wrapper b(
    {"hello", "world", "this is a really long string to generate a longer row", "dlrow", "olleh"});
  cudf::table_view in(std::vector<cudf::column_view>{a, b});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32}};

  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(new_rows[0]->size(), 5);
}

TEST_F(ColumnToRowTests, DoubleString)
{
  cudf::test::strings_column_wrapper a(
    {"hello", "world", "this is a really long string to generate a longer row", "dlrow", "olleh"});
  cudf::test::fixed_width_column_wrapper<int32_t> b({0, 1, 2, 3, 4});
  cudf::test::strings_column_wrapper c({"world",
                                        "hello",
                                        "this string isn't as long",
                                        "this one isn't so short though when you think about it",
                                        "dlrow"});
  cudf::table_view in(std::vector<cudf::column_view>{a, b, c});

  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(new_rows[0]->size(), 5);
}

TEST_F(ColumnToRowTests, BigStrings)
{
  char const* TEST_STRINGS[] = {
    "These",
    "are",
    "the",
    "test",
    "strings",
    "that",
    "we",
    "have",
    "some are really long",
    "and some are kinda short",
    "They are all over on purpose with different sizes for the strings in order to test the code "
    "on all different lengths of strings",
    "a",
    "good test",
    "is required to produce reasonable confidence that this is working"};
  auto num_generator = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  auto string_generator =
    spark_rapids_jni::util::make_counting_transform_iterator(0, [&](auto i) -> char const* {
      return TEST_STRINGS[rand() % (sizeof(TEST_STRINGS) / sizeof(TEST_STRINGS[0]))];
    });

  auto const num_rows = 50;
  auto const num_cols = 50;
  std::vector<cudf::data_type> schema;

  std::vector<cudf::test::detail::column_wrapper> cols;
  std::vector<cudf::column_view> views;

  for (auto col = 0; col < num_cols; ++col) {
    if (rand() % 2) {
      cols.emplace_back(
        cudf::test::fixed_width_column_wrapper<int32_t>(num_generator, num_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::data_type{cudf::type_id::INT32});
    } else {
      cols.emplace_back(
        cudf::test::strings_column_wrapper(string_generator, string_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::type_id::STRING);
    }
  }

  cudf::table_view in(views);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(new_rows[0]->size(), num_rows);
}

TEST_F(ColumnToRowTests, ManyStrings)
{
  char const* TEST_STRINGS[] = {
    "These",
    "are",
    "the",
    "test",
    "strings",
    "that",
    "we",
    "have",
    "some are really long",
    "and some are kinda short",
    "They are all over on purpose with different sizes for the strings in order to test the code "
    "on all different lengths of strings",
    "a",
    "good test",
    "is required to produce reasonable confidence that this is working",
    "some strings",
    "are split into multiple strings",
    "some strings have all their data",
    "lots of choices of strings and sizes is sure to test the offset calculation code to ensure "
    "that even a really long string ends up in the correct spot for the final destination allowing "
    "for even crazy run-on sentences to be inserted into the data"};
  auto num_generator = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  auto string_generator =
    spark_rapids_jni::util::make_counting_transform_iterator(0, [&](auto i) -> char const* {
      return TEST_STRINGS[rand() % (sizeof(TEST_STRINGS) / sizeof(TEST_STRINGS[0]))];
    });

  auto const num_rows = 1'000'000;
  auto const num_cols = 50;
  std::vector<cudf::data_type> schema;

  std::vector<cudf::test::detail::column_wrapper> cols;
  std::vector<cudf::column_view> views;

  for (auto col = 0; col < num_cols; ++col) {
    if (rand() % 2) {
      cols.emplace_back(
        cudf::test::fixed_width_column_wrapper<int32_t>(num_generator, num_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::data_type{cudf::type_id::INT32});
    } else {
      cols.emplace_back(
        cudf::test::strings_column_wrapper(string_generator, string_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::type_id::STRING);
    }
  }

  cudf::table_view in(views);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(new_rows[0]->size(), num_rows);
}

TEST_F(ColumnToRowTests, Simple)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1, 0, 1});
  cudf::table_view in(std::vector<cudf::column_view>{a});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32}};

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Tall)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  cudf::test::fixed_width_column_wrapper<int32_t> a(r, r + (size_t)4096);
  cudf::table_view in(std::vector<cudf::column_view>{a});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32}};

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Wide)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  for (int i = 0; i < 256; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>({rand()}));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, SingleByteWide)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int8_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  for (int i = 0; i < 256; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int8_t>({rand()}));
    views.push_back(cols.back());

    schema.push_back(cudf::data_type{cudf::type_id::INT8});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Non2Power)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  constexpr auto num_rows = 6 * 1024 + 557;
  for (int i = 0; i < 131; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    for (int j = 0; j < old_tbl->num_columns(); ++j) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(old_tbl->get_column(j), new_tbl->get_column(j));
    }

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Big)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 28 columns of 1 million rows
  constexpr auto num_rows = 1024 * 1024;
  for (int i = 0; i < 28; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    for (int j = 0; j < old_tbl->num_columns(); ++j) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(old_tbl->get_column(j), new_tbl->get_column(j));
    }

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Bigger)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 128 columns of 1 million rows
  constexpr auto num_rows = 1024 * 1024;
  for (int i = 0; i < 128; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    for (int j = 0; j < old_tbl->num_columns(); ++j) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(old_tbl->get_column(j), new_tbl->get_column(j));
    }

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(ColumnToRowTests, Biggest)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 128 columns of 2 million rows
  constexpr auto num_rows = 2 * 1024 * 1024;
  for (int i = 0; i < 128; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  EXPECT_EQ(old_rows.size(), new_rows.size());

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    for (int j = 0; j < old_tbl->num_columns(); ++j) {
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(old_tbl->get_column(j), new_tbl->get_column(j));
    }

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

// Regression test for https://github.com/NVIDIA/spark-rapids/issues/10062.
//
// AcceleratedColumnarToRowIterator packs columns by size descending, so a
// pivot-shaped schema of N INT64 columns plus one INT32 places the INT32
// last. Choosing N so that the INT32 is the column that tips the estimated
// shmem usage in detail::determine_tiles over the per-tile budget is what
// triggers the bug: the tile containing only that column is never emitted,
// leaving the column's output bytes at whatever rmm::device_buffer handed
// back. With the 48 KB default shmem budget and tile_height = 32, 191 Longs
// fit in one tile (1528 * 32 = 48896 <= 49136) and adding the trailing INT32
// tips the estimate to 1536 * 32 = 49152 > 49136.
//
// We check the trailing INT32 directly against the raw JCUDF bytes rather
// than round-tripping through convert_from_rows, which would mask the bug by
// using the same layout code on the read side.
TEST_F(ColumnToRowTests, PivotLikeLayout)
{
  constexpr int num_longs = 191;
  constexpr int num_rows  = 100;

  std::vector<int64_t> long_data(num_rows, 0);
  std::vector<int32_t> int_data(num_rows);
  for (int r = 0; r < num_rows; ++r) {
    int_data[r] = 0x11223344 + r;
  }

  std::vector<cudf::test::fixed_width_column_wrapper<int64_t>> long_cols;
  long_cols.reserve(num_longs);
  for (int i = 0; i < num_longs; ++i) {
    long_cols.emplace_back(long_data.begin(), long_data.end());
  }
  cudf::test::fixed_width_column_wrapper<int32_t> int_col(int_data.begin(), int_data.end());

  std::vector<cudf::column_view> views;
  views.reserve(num_longs + 1);
  for (auto& c : long_cols) {
    views.emplace_back(c);
  }
  views.emplace_back(int_col);

  auto rows = spark_rapids_jni::convert_to_rows(cudf::table_view(views));
  ASSERT_EQ(rows.size(), 1u);

  // JCUDF row layout: [Longs][INT32][validity bits][pad to 8B].
  constexpr std::size_t int_offset     = static_cast<std::size_t>(num_longs) * sizeof(int64_t);
  constexpr std::size_t data_end       = int_offset + sizeof(int32_t);
  constexpr std::size_t validity_bytes = (num_longs + 1 + 7) / 8;
  constexpr std::size_t row_stride     = (data_end + validity_bytes + 7) & ~std::size_t{7};

  auto host_bytes = cudf::test::to_host<int8_t>(cudf::lists_column_view(*rows[0]).child()).first;
  ASSERT_GE(host_bytes.size(), row_stride * num_rows);
  for (int r = 0; r < num_rows; ++r) {
    int32_t actual = 0;
    std::memcpy(&actual, host_bytes.data() + r * row_stride + int_offset, sizeof(int32_t));
    EXPECT_EQ(actual, int_data[r]) << "row " << r;
  }
}

// Regression test for spark-rapids-jni#4586: the default branch of the type-size switch in
// copy_to_rows wrote to the same byte col_size times rather than advancing the offset, so any
// fixed-width column wider than 8 bytes (DECIMAL128 in practice) was silently corrupted.
TEST_F(ColumnToRowTests, Decimal128RoundTrip)
{
  // Include a value with non-zero bytes spread across all 16 positions so a regression that
  // copies the same byte 16 times (the original bug) is detected anywhere in the word, not
  // only in the low bytes. Also include a null to cover the validity-bitmap path.
  auto const wide = (static_cast<__int128_t>(0x0102030405060708LL) << 64) |
                    static_cast<__int128_t>(0x090A0B0C0D0E0F10LL);
  std::vector<__int128_t> vals{static_cast<__int128_t>(12345),
                               static_cast<__int128_t>(-67890),
                               static_cast<__int128_t>(999999999999LL),
                               wide};
  cudf::test::fixed_point_column_wrapper<__int128_t> col(
    vals.begin(), vals.end(), {true, false, true, true}, numeric::scale_type{-2});
  cudf::table_view in({col});
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::DECIMAL128, -2}};

  auto rows = spark_rapids_jni::convert_to_rows(in);
  ASSERT_EQ(rows.size(), 1u);
  auto result = spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*rows[0]), schema);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in, *result);
}

// Regression test for spark-rapids-jni#4590: when a tile boundary falls at a byte offset that
// is not 8-aligned, the old code wrote `round_up_8(actual_size)` bytes from the shared tile to
// global memory. The trailing padding bytes overlapped with the next tile's destination range,
// causing a non-deterministic race between adjacent CUDA blocks. With the fix, the write length
// is the actual data span, so adjacent tiles never touch the same bytes.
//
// The race itself is timing-dependent; this test simply round-trips a wide INT32 schema known
// to produce a multi-tile layout and asserts data integrity over many iterations. With the bug,
// at least some iterations would mismatch on certain GPUs.
TEST_F(ColumnToRowTests, TileBoundaryWideInt32RoundTrip)
{
  // A row of 500 INT32 columns is ~2 KB, which overflows the per-tile shmem budget and
  // forces multiple tiles. The exact tile boundary location depends on shmem_limit_per_tile
  // at runtime, but for the supported budgets it lands somewhere inside the schema.
  constexpr int num_cols = 500;
  constexpr int num_rows = 64;

  auto data_iter = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return static_cast<int32_t>(i * 2654435761u); });

  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  cols.reserve(num_cols);
  std::vector<cudf::column_view> views;
  views.reserve(num_cols);
  std::vector<cudf::data_type> schema;
  schema.reserve(num_cols);
  for (int c = 0; c < num_cols; ++c) {
    cols.emplace_back(data_iter + c * num_rows, data_iter + c * num_rows + num_rows);
    views.emplace_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  // Repeat to give a non-deterministic tile-write race more chances to surface.
  for (int iter = 0; iter < 8; ++iter) {
    auto rows = spark_rapids_jni::convert_to_rows(in);
    ASSERT_EQ(rows.size(), 1u);
    auto result = spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*rows[0]), schema);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in, *result);
  }
}

// Regression test for spark-rapids-jni#4586: nested types (LIST, STRUCT, MAP) and other
// unsupported data types must be rejected at the entry point with a clear exception, rather
// than producing silently corrupted output (which happened when LIST/STRUCT columns reached
// the variable-width path that assumes STRING).
TEST_F(ColumnToRowTests, RejectListColumn)
{
  cudf::test::lists_column_wrapper<int32_t> list_col{{1, 2}, {3}, {4, 5, 6}};
  cudf::table_view in({list_col});
  EXPECT_THROW(spark_rapids_jni::convert_to_rows(in), cudf::logic_error);
}

TEST_F(ColumnToRowTests, RejectStructColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> child_a({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int64_t> child_b({10L, 20L, 30L});
  cudf::test::structs_column_wrapper struct_col({child_a, child_b});
  cudf::table_view in({struct_col});
  EXPECT_THROW(spark_rapids_jni::convert_to_rows(in), cudf::logic_error);
}

// Regression test for spark-rapids-jni#4586: column_view::data<int8_t>() returns
// `head + offset_in_elements` interpreted as bytes, so a sliced column produced a misaligned
// input pointer and could either crash or silently corrupt data. With the entry-point guard
// the caller now gets a clear exception.
TEST_F(ColumnToRowTests, RejectSlicedColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> source({10, 11, 12, 13, 14, 15, 16, 17});
  auto sliced = cudf::slice(static_cast<cudf::column_view>(source), {2, 6})[0];
  cudf::table_view in({sliced});
  EXPECT_THROW(spark_rapids_jni::convert_to_rows(in), cudf::logic_error);
}

TEST_F(ColumnToRowTests, RejectSlicedColumnFixedWidthOptimized)
{
  cudf::test::fixed_width_column_wrapper<int32_t> source({10, 11, 12, 13, 14, 15, 16, 17});
  auto sliced = cudf::slice(static_cast<cudf::column_view>(source), {2, 6})[0];
  cudf::table_view in({sliced});
  EXPECT_THROW(spark_rapids_jni::convert_to_rows_fixed_width_optimized(in), cudf::logic_error);
}

TEST_F(ColumnToRowTests, RejectStringColumnInFixedWidthOptimized)
{
  cudf::test::strings_column_wrapper col({"a", "bb", "ccc"});
  cudf::table_view in({col});
  EXPECT_THROW(spark_rapids_jni::convert_to_rows_fixed_width_optimized(in), cudf::logic_error);
}

TEST_F(RowToColumnTests, RejectUnsupportedSchema)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  cudf::table_view in({col});
  auto rows = spark_rapids_jni::convert_to_rows(in);
  ASSERT_EQ(rows.size(), 1u);

  std::vector<cudf::data_type> list_schema{cudf::data_type{cudf::type_id::LIST}};
  EXPECT_THROW(spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*rows[0]), list_schema),
               cudf::logic_error);

  std::vector<cudf::data_type> string_schema{cudf::data_type{cudf::type_id::STRING}};
  EXPECT_THROW(spark_rapids_jni::convert_from_rows_fixed_width_optimized(
                 cudf::lists_column_view(*rows[0]), string_schema),
               cudf::logic_error);
}

TEST_F(RowToColumnTests, RejectSlicedRowList)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  cudf::table_view in({col});
  auto rows = spark_rapids_jni::convert_to_rows(in);
  ASSERT_EQ(rows.size(), 1u);
  auto sliced = cudf::slice(rows[0]->view(), {1, 3})[0];
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT32}};

  EXPECT_THROW(spark_rapids_jni::convert_from_rows(cudf::lists_column_view(sliced), schema),
               cudf::logic_error);
  EXPECT_THROW(spark_rapids_jni::convert_from_rows_fixed_width_optimized(
                 cudf::lists_column_view(sliced), schema),
               cudf::logic_error);
}

// Regression repro for spark-rapids-jni#4587. Disabled by default because it requires ~2.5 GB
// of free GPU memory to build the input; enable manually with --gtest_also_run_disabled_tests.
//
// With fewer than 32 rows whose cumulative encoded size exceeds 2 GiB, the old
// detail::build_batches would loop forever because round_down_safe(batch_size, 32) returned
// zero and last_row_end never advanced. The fix surfaces the situation as an exception.
TEST_F(ColumnToRowTests, DISABLED_HugeStringRowThrows)
{
  constexpr std::size_t per_col_bytes = 35ULL * 1024 * 1024;
  constexpr int num_rows              = 33;

  std::string const s(per_col_bytes, 'x');
  std::vector<std::string> data(num_rows, s);

  cudf::test::strings_column_wrapper col_a(data.begin(), data.end());
  cudf::test::strings_column_wrapper col_b(data.begin(), data.end());
  cudf::table_view in({col_a, col_b});

  EXPECT_THROW(spark_rapids_jni::convert_to_rows(in), cudf::logic_error);
}

// Regression repro for spark-rapids-jni#4588. Disabled by default because it needs roughly 7 GB of
// free GPU memory.
//
// The old kernel launch passed batch_num_rows (a per-batch count) as the kernel's `num_rows`
// while start_row was an absolute index, so all batches whose start lay past the per-batch
// count silently produced uninitialized output. The fix passes batch_row_offset +
// batch_num_rows as the absolute end bound; this test exercises the multi-batch path.
//
// Sizing: two 20 MiB string columns give an ~40 MiB JCUDF row (string bytes dominate), so
// floor(INT32_MAX / 40 MiB) = 51 >= 32 rows fit a batch and build_batches never trips the
// "fewer than 32 rows" throw (the earlier 33 MiB columns left only 31 rows per batch, throwing
// before the multi-batch path could run). 64 rows * ~40 MiB = ~2.5 GiB > the 2 GiB batch limit,
// splitting into exactly two 32-row batches.
TEST_F(ColumnToRowTests, DISABLED_MultiBatchStringDoesNotSkip)
{
  constexpr std::size_t per_col_bytes = 20ULL * 1024 * 1024;
  constexpr int num_rows              = 64;

  std::vector<std::string> data_a, data_b;
  data_a.reserve(num_rows);
  data_b.reserve(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    data_a.push_back(std::string(per_col_bytes, static_cast<char>('a' + (i % 26))));
    data_b.push_back(std::string(per_col_bytes, static_cast<char>('A' + (i % 26))));
  }

  cudf::test::strings_column_wrapper col_a(data_a.begin(), data_a.end());
  cudf::test::strings_column_wrapper col_b(data_b.begin(), data_b.end());
  cudf::table_view in({col_a, col_b});
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::STRING},
                                      cudf::data_type{cudf::type_id::STRING}};

  auto rows = spark_rapids_jni::convert_to_rows(in);
  ASSERT_GE(rows.size(), 2u) << "Expected multiple batches; if a single batch fits the issue "
                                "cannot be reproduced — increase num_rows.";

  // Reconstruct each batch and compare against the matching row slice of the input.
  std::size_t row_start = 0;
  for (auto& batch : rows) {
    auto result   = spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*batch), schema);
    auto in_slice = cudf::slice(in,
                                {static_cast<cudf::size_type>(row_start),
                                 static_cast<cudf::size_type>(row_start + result->num_rows())})[0];
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in_slice, *result);
    row_start += result->num_rows();
  }
}

// Fixed-width analogue of MultiBatchStringDoesNotSkip: covers the >2 GiB multi-batch path of the
// all-fixed-width general conversion. Disabled by default because it needs roughly 8.5 GB of free
// GPU memory; enable manually with --gtest_also_run_disabled_tests.
//
// 300 INT32 columns give a JCUDF row of 300*4 data + ceil(300/8) validity = 1238 bytes, padded to
// 1240. 2,500,000 rows * 1240 = 3.1 GB of row data, which exceeds the 2 GiB batch limit and must
// split into exactly two batches: the first holds floor(INT32_MAX / 1240) rows rounded down to a
// 32-row multiple (1,731,840), the second holds the remaining 768,160 rows (~0.95 GB).
TEST_F(ColumnToRowTests, DISABLED_MultiBatchFixedWidthRoundTrip)
{
  constexpr int num_cols             = 300;
  constexpr cudf::size_type num_rows = 2'500'000;

  auto data_iter = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return static_cast<int32_t>(i * 2654435761u); });

  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  cols.reserve(num_cols);
  std::vector<cudf::column_view> views;
  views.reserve(num_cols);
  std::vector<cudf::data_type> schema;
  schema.reserve(num_cols);
  for (int c = 0; c < num_cols; ++c) {
    cols.emplace_back(data_iter + c * num_rows, data_iter + c * num_rows + num_rows);
    views.emplace_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto rows = spark_rapids_jni::convert_to_rows(in);
  ASSERT_EQ(rows.size(), 2u) << "Expected exactly two batches; if the split changed, revisit "
                                "num_cols/num_rows so the row data still exceeds 2 GiB.";

  std::size_t row_start = 0;
  for (auto& batch : rows) {
    auto result   = spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*batch), schema);
    auto in_slice = cudf::slice(in,
                                {static_cast<cudf::size_type>(row_start),
                                 static_cast<cudf::size_type>(row_start + result->num_rows())})[0];
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in_slice, *result);
    row_start += result->num_rows();
  }
  EXPECT_EQ(row_start, static_cast<std::size_t>(num_rows));
}

TEST_F(RowToColumnTests, Single)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1});
  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto old_rows = spark_rapids_jni::convert_to_rows(in);
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT32}};
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Simple)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1, 0, 1});
  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT32}};
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Tall)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  cudf::test::fixed_width_column_wrapper<int32_t> a(r, r + (size_t)4096);
  cudf::table_view in(std::vector<cudf::column_view>{a});

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  std::vector<cudf::data_type> schema;
  schema.reserve(in.num_columns());
  for (auto col = in.begin(); col < in.end(); ++col) {
    schema.push_back(col->type());
  }
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Wide)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;

  for (int i = 0; i < 256; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>({i}));  // rand()}));
    views.push_back(cols.back());
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  std::vector<cudf::data_type> schema;
  schema.reserve(in.num_columns());
  for (auto col = in.begin(); col < in.end(); ++col) {
    schema.push_back(col->type());
  }

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, SingleByteWide)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int8_t>> cols;
  std::vector<cudf::column_view> views;

  for (int i = 0; i < 256; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int8_t>({rand()}));
    views.push_back(cols.back());
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  std::vector<cudf::data_type> schema;
  schema.reserve(in.num_columns());
  for (auto col = in.begin(); col < in.end(); ++col) {
    schema.push_back(col->type());
  }
  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, AllTypes)
{
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT64},
                                      cudf::data_type{cudf::type_id::FLOAT64},
                                      cudf::data_type{cudf::type_id::INT8},
                                      cudf::data_type{cudf::type_id::BOOL8},
                                      cudf::data_type{cudf::type_id::FLOAT32},
                                      cudf::data_type{cudf::type_id::INT8},
                                      cudf::data_type{cudf::type_id::INT32},
                                      cudf::data_type{cudf::type_id::INT64}};

  cudf::test::fixed_width_column_wrapper<int64_t> c0({3, 9, 4, 2, 20, 0}, {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<double> c1({5.0, 9.5, 0.9, 7.23, 2.8, 0.0},
                                                    {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<int8_t> c2({5, 1, 0, 2, 7, 0}, {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> c3({true, false, false, true, false, false},
                                                  {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<float> c4({1.0f, 3.5f, 5.9f, 7.1f, 9.8f, 0.0f},
                                                   {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<int8_t> c5({2, 3, 4, 5, 9, 0}, {1, 1, 1, 1, 1, 0});
  cudf::test::fixed_point_column_wrapper<int32_t> c6(
    {-300, 500, 950, 90, 723, 0}, {1, 1, 1, 1, 1, 1, 1, 0}, numeric::scale_type{-2});
  cudf::test::fixed_point_column_wrapper<int64_t> c7(
    {-80, 30, 90, 20, 200, 0}, {1, 1, 1, 1, 1, 1, 0}, numeric::scale_type{-1});

  cudf::table_view in({c0, c1, c2, c3, c4, c5, c6, c7});

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*new_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, AllTypesLarge)
{
  std::vector<cudf::column> cols;
  std::vector<cudf::data_type> schema{};

  // 15 columns of each type with 1 million entries
  constexpr int num_rows{1024 * 1024 * 1};

  std::default_random_engine re;
  std::uniform_real_distribution<double> rand_double(std::numeric_limits<double>::min(),
                                                     std::numeric_limits<double>::max());
  std::uniform_int_distribution<int64_t> rand_int64(std::numeric_limits<int64_t>::min(),
                                                    std::numeric_limits<int64_t>::max());
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [&](auto i) -> int64_t { return rand_int64(re); });
  auto d = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [&](auto i) -> double { return rand_double(re); });

  auto all_valid =
    spark_rapids_jni::util::make_counting_transform_iterator(0, [](auto i) { return 1; });
  auto none_valid =
    spark_rapids_jni::util::make_counting_transform_iterator(0, [](auto i) { return 0; });
  auto most_valid = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) { return rand() % 2 == 0 ? 0 : 1; });
  auto few_valid = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) { return rand() % 13 == 0 ? 1 : 0; });

  for (int i = 0; i < 15; ++i) {
    cols.push_back(*cudf::test::fixed_width_column_wrapper<int8_t>(r, r + num_rows, all_valid)
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::INT8});
  }

  for (int i = 0; i < 15; ++i) {
    cols.push_back(*cudf::test::fixed_width_column_wrapper<int16_t>(r, r + num_rows, few_valid)
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::INT16});
  }

  for (int i = 0; i < 15; ++i) {
    if (i < 5) {
      cols.push_back(*cudf::test::fixed_width_column_wrapper<int32_t>(r, r + num_rows, few_valid)
                        .release()
                        .release());
    } else {
      cols.push_back(*cudf::test::fixed_width_column_wrapper<int32_t>(r, r + num_rows, none_valid)
                        .release()
                        .release());
    }
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }

  for (int i = 0; i < 15; ++i) {
    cols.push_back(*cudf::test::fixed_width_column_wrapper<float>(d, d + num_rows, most_valid)
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::FLOAT32});
  }

  for (int i = 0; i < 15; ++i) {
    cols.push_back(*cudf::test::fixed_width_column_wrapper<double>(d, d + num_rows, most_valid)
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::FLOAT64});
  }

  for (int i = 0; i < 15; ++i) {
    cols.push_back(*cudf::test::fixed_width_column_wrapper<bool>(r, r + num_rows, few_valid)
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::BOOL8});
  }

  for (int i = 0; i < 15; ++i) {
    cols.push_back(
      *cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>(
         r, r + num_rows, all_valid)
         .release()
         .release());
    schema.push_back(cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS});
  }

  for (int i = 0; i < 15; ++i) {
    cols.push_back(
      *cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>(
         r, r + num_rows, most_valid)
         .release()
         .release());
    schema.push_back(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS});
  }

  for (int i = 0; i < 15; ++i) {
    cols.push_back(*cudf::test::fixed_point_column_wrapper<int32_t>(
                      r, r + num_rows, all_valid, numeric::scale_type{-2})
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::DECIMAL32});
  }

  for (int i = 0; i < 15; ++i) {
    cols.push_back(*cudf::test::fixed_point_column_wrapper<int64_t>(
                      r, r + num_rows, most_valid, numeric::scale_type{-1})
                      .release()
                      .release());
    schema.push_back(cudf::data_type{cudf::type_id::DECIMAL64});
  }

  std::vector<cudf::column_view> views(cols.begin(), cols.end());
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*new_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Non2Power)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  constexpr auto num_rows = 6 * 1024 + 557;
  for (int i = 0; i < 131; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Big)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 28 columns of 1 million rows
  constexpr auto num_rows = 1024 * 1024;
  for (int i = 0; i < 28; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Bigger)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 128 columns of 1 million rows
  constexpr auto num_rows = 1024 * 1024;
  for (int i = 0; i < 128; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*old_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, Biggest)
{
  auto r = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  // 128 columns of 2 million rows
  constexpr auto num_rows = 2 * 1024 * 1024;
  for (int i = 0; i < 128; ++i) {
    cols.push_back(cudf::test::fixed_width_column_wrapper<int32_t>(r + num_rows * i,
                                                                   r + num_rows * i + num_rows));
    views.push_back(cols.back());
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  cudf::table_view in(views);

  auto old_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  for (uint i = 0; i < old_rows.size(); ++i) {
    auto old_tbl = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
      cudf::lists_column_view(*old_rows[i]), schema);
    auto new_tbl =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*new_rows[i]), schema);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*old_tbl, *new_tbl);
  }
}

TEST_F(RowToColumnTests, SimpleString)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a({-1, 0, 1, 0, -1});
  cudf::test::strings_column_wrapper b(
    {"hello", "world", "this is a really long string to generate a longer row", "dlrow", "olleh"});
  cudf::table_view in(std::vector<cudf::column_view>{a, b});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::INT32},
                                         cudf::data_type{cudf::type_id::STRING}};

  auto new_rows = spark_rapids_jni::convert_to_rows(in);
  EXPECT_EQ(new_rows.size(), 1);
  for (auto& row : new_rows) {
    auto new_cols = spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*row), schema);
    EXPECT_EQ(row->size(), 5);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in, *new_cols);
  }
}

TEST_F(RowToColumnTests, DoubleString)
{
  cudf::test::strings_column_wrapper a(
    {"hello", "world", "this is a really long string to generate a longer row", "dlrow", "olleh"});
  cudf::test::fixed_width_column_wrapper<int32_t> b({0, 1, 2, 3, 4});
  cudf::test::strings_column_wrapper c({"world",
                                        "hello",
                                        "this string isn't as long",
                                        "this one isn't so short though when you think about it",
                                        "dlrow"});
  cudf::table_view in(std::vector<cudf::column_view>{a, b, c});
  std::vector<cudf::data_type> schema = {cudf::data_type{cudf::type_id::STRING},
                                         cudf::data_type{cudf::type_id::INT32},
                                         cudf::data_type{cudf::type_id::STRING}};

  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  for (uint i = 0; i < new_rows.size(); ++i) {
    auto new_cols =
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*new_rows[i]), schema);

    EXPECT_EQ(new_rows[0]->size(), 5);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in, *new_cols);
  }
}

TEST_F(RowToColumnTests, BigStrings)
{
  char const* TEST_STRINGS[] = {
    "These",
    "are",
    "the",
    "test",
    "strings",
    "that",
    "we",
    "have",
    "some are really long",
    "and some are kinda short",
    "They are all over on purpose with different sizes for the strings in order to test the code "
    "on all different lengths of strings",
    "a",
    "good test",
    "is required to produce reasonable confidence that this is working"};
  auto num_generator = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  auto string_generator =
    spark_rapids_jni::util::make_counting_transform_iterator(0, [&](auto i) -> char const* {
      return TEST_STRINGS[rand() % (sizeof(TEST_STRINGS) / sizeof(TEST_STRINGS[0]))];
    });

  auto const num_rows = 50;
  auto const num_cols = 50;
  std::vector<cudf::data_type> schema;

  std::vector<cudf::test::detail::column_wrapper> cols;
  std::vector<cudf::column_view> views;

  for (auto col = 0; col < num_cols; ++col) {
    if (rand() % 2) {
      cols.emplace_back(
        cudf::test::fixed_width_column_wrapper<int32_t>(num_generator, num_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::data_type{cudf::type_id::INT32});
    } else {
      cols.emplace_back(
        cudf::test::strings_column_wrapper(string_generator, string_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::type_id::STRING);
    }
  }

  cudf::table_view in(views);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  for (auto& i : new_rows) {
    auto new_cols = spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*i), schema);

    auto in_view = cudf::slice(in, {0, new_cols->num_rows()});
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in_view[0], *new_cols);
  }
}

TEST_F(RowToColumnTests, ManyStrings)
{
  // The sizing of this test is very sensitive to the state of the random number generator,
  // i.e., depending on the order of execution, the number of times the largest string is
  // selected will lead to out-of-memory exceptions. Seeding the RNG here helps prevent that.
  srand(1);
  char const* TEST_STRINGS[] = {
    "These",
    "are",
    "the",
    "test",
    "strings",
    "that",
    "we",
    "have",
    "some are really long",
    "and some are kinda short",
    "They are all over on purpose with different sizes for the strings in order to test the code "
    "on all different lengths of strings",
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine "
    "this string is the longest string because it is duplicated more than you can imagine ",
    "a",
    "good test",
    "is required to produce reasonable confidence that this is working",
    "some strings",
    "are split into multiple strings",
    "some strings have all their data",
    "lots of choices of strings and sizes is sure to test the offset calculation code to ensure "
    "that even a really long string ends up in the correct spot for the final destination allowing "
    "for even crazy run-on sentences to be inserted into the data"};
  auto num_generator = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return rand(); });
  auto string_generator =
    spark_rapids_jni::util::make_counting_transform_iterator(0, [&](auto i) -> char const* {
      return TEST_STRINGS[rand() % (sizeof(TEST_STRINGS) / sizeof(TEST_STRINGS[0]))];
    });

  auto const num_rows = 300'000;
  auto const num_cols = 50;
  std::vector<cudf::data_type> schema;

  std::vector<cudf::test::detail::column_wrapper> cols;
  std::vector<cudf::column_view> views;

  for (auto col = 0; col < num_cols; ++col) {
    if (rand() % 2) {
      cols.emplace_back(
        cudf::test::fixed_width_column_wrapper<int32_t>(num_generator, num_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::data_type{cudf::type_id::INT32});
    } else {
      cols.emplace_back(
        cudf::test::strings_column_wrapper(string_generator, string_generator + num_rows));
      views.push_back(cols.back());
      schema.emplace_back(cudf::type_id::STRING);
    }
  }

  cudf::table_view in(views);
  auto new_rows = spark_rapids_jni::convert_to_rows(in);

  for (auto& i : new_rows) {
    auto new_cols = spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*i), schema);

    auto in_view = cudf::slice(in, {0, new_cols->num_rows()});
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in_view[0], *new_cols);
  }
}

// =================================================================================================
// Test-hardening suite: pins the JCUDF byte format and locks every kernel path so that a kernel
// change breaking either is caught by this gtest target alone (round trips are symmetric and
// would miss a format shift that both directions share).
//
// JCUDF row layout: [fixed region][validity][string chars][pad to 8 bytes].
//   - fixed region: columns packed in schema order, each at an offset aligned to its own element
//     size; a string column occupies an 8-byte slot of two uint32 values (offset, length) with
//     4-byte alignment, where `offset` is relative to the row start.
//   - validity: ceil(num_columns/8) bytes directly after the fixed region (byte-aligned); bit c%8
//     of byte c/8 is 1 when column c is valid.
//   - string chars: packed unaligned immediately after the validity bytes.
//   - the row buffer is uninitialized and the kernels never write alignment holes, the tail pad
//     up to the 8-byte row stride, or validity bits past the last column, so byte-exact compares
//     must mask those ranges; a null cell's data bytes are masked too, since only its validity
//     bit is contractual.
// =================================================================================================

namespace {

// Byte-wise masked compare of a JCUDF row buffer (the INT8 child of a rows column) against a
// hand-built expected buffer. A zero mask bit marks a byte/bit the kernels leave undefined.
void expect_masked_bytes_equal(cudf::column_view const& rows_child,
                               std::vector<uint8_t> const& expected,
                               std::vector<uint8_t> const& mask)
{
  auto const actual = cudf::test::to_host<int8_t>(rows_child).first;
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (mask[i] == 0) { continue; }
    EXPECT_EQ(static_cast<uint8_t>(actual[i]) & mask[i], expected[i] & mask[i]) << "byte " << i;
  }
}

std::vector<int32_t> row_offsets_to_host(cudf::column const& rows_column)
{
  auto const host =
    cudf::test::to_host<int32_t>(cudf::lists_column_view(rows_column).offsets()).first;
  return {host.begin(), host.end()};
}

// General-path round trip; also checks per-column null counts, which catches a validity kernel
// writing a plausible-but-wrong mask that the table compare on its own would only see as data.
void expect_general_round_trip(cudf::table_view const& in,
                               std::vector<cudf::data_type> const& schema)
{
  auto rows = spark_rapids_jni::convert_to_rows(in);
  ASSERT_EQ(rows.size(), 1u);
  auto result = spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*rows[0]), schema);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in, *result);
  for (cudf::size_type c = 0; c < in.num_columns(); ++c) {
    EXPECT_EQ(result->get_column(c).null_count(), in.column(c).null_count()) << "column " << c;
  }
}

void expect_fixed_width_optimized_round_trip(cudf::table_view const& in,
                                             std::vector<cudf::data_type> const& schema)
{
  auto rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  ASSERT_EQ(rows.size(), 1u);
  auto result = spark_rapids_jni::convert_from_rows_fixed_width_optimized(
    cudf::lists_column_view(*rows[0]), schema);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in, *result);
  for (cudf::size_type c = 0; c < in.num_columns(); ++c) {
    EXPECT_EQ(result->get_column(c).null_count(), in.column(c).null_count()) << "column " << c;
  }
}

template <typename ElementT, typename SourceT = ElementT>
std::unique_ptr<cudf::column> make_typed_column(std::vector<int64_t> const& values,
                                                std::vector<bool> const& validity,
                                                bool nullable)
{
  if (nullable) {
    return cudf::test::fixed_width_column_wrapper<ElementT, SourceT>(
             values.begin(), values.end(), validity.begin())
      .release();
  }
  return cudf::test::fixed_width_column_wrapper<ElementT, SourceT>(values.begin(), values.end())
    .release();
}

template <typename RepT>
std::unique_ptr<cudf::column> make_decimal_column(std::vector<int64_t> const& values,
                                                  std::vector<bool> const& validity,
                                                  bool nullable,
                                                  int32_t scale)
{
  if (nullable) {
    return cudf::test::fixed_point_column_wrapper<RepT>(
             values.begin(), values.end(), validity.begin(), numeric::scale_type{scale})
      .release();
  }
  return cudf::test::fixed_point_column_wrapper<RepT>(
           values.begin(), values.end(), numeric::scale_type{scale})
    .release();
}

// Deterministic column of `type`: null_fraction 0 allocates no null mask (the absent-mask kernel
// branch), 1.0 yields an all-null column, anything between draws per-row validity.
std::unique_ptr<cudf::column> make_random_column(cudf::data_type type,
                                                 cudf::size_type num_rows,
                                                 double null_fraction,
                                                 std::mt19937& engine)
{
  std::uniform_int_distribution<int64_t> value_dist(std::numeric_limits<int64_t>::min(),
                                                    std::numeric_limits<int64_t>::max());
  std::vector<int64_t> values(num_rows);
  std::generate(values.begin(), values.end(), [&] { return value_dist(engine); });
  bool const nullable = null_fraction > 0.0;
  std::bernoulli_distribution null_dist(null_fraction);
  std::vector<bool> validity(num_rows);
  std::generate(validity.begin(), validity.end(), [&] { return !null_dist(engine); });

  using cudf::type_id;
  switch (type.id()) {
    case type_id::INT8: return make_typed_column<int8_t>(values, validity, nullable);
    case type_id::INT16: return make_typed_column<int16_t>(values, validity, nullable);
    case type_id::INT32: return make_typed_column<int32_t>(values, validity, nullable);
    case type_id::INT64: return make_typed_column<int64_t>(values, validity, nullable);
    case type_id::UINT8: return make_typed_column<uint8_t>(values, validity, nullable);
    case type_id::UINT16: return make_typed_column<uint16_t>(values, validity, nullable);
    case type_id::UINT32: return make_typed_column<uint32_t>(values, validity, nullable);
    case type_id::UINT64: return make_typed_column<uint64_t>(values, validity, nullable);
    case type_id::FLOAT32: return make_typed_column<float>(values, validity, nullable);
    case type_id::FLOAT64: return make_typed_column<double>(values, validity, nullable);
    case type_id::BOOL8: return make_typed_column<bool>(values, validity, nullable);
    case type_id::DECIMAL32:
      return make_decimal_column<int32_t>(values, validity, nullable, type.scale());
    case type_id::DECIMAL64:
      return make_decimal_column<int64_t>(values, validity, nullable, type.scale());
    case type_id::DECIMAL128:
      return make_decimal_column<__int128_t>(values, validity, nullable, type.scale());
    case type_id::TIMESTAMP_DAYS:
      return make_typed_column<cudf::timestamp_D, cudf::timestamp_D::rep>(
        values, validity, nullable);
    case type_id::TIMESTAMP_SECONDS:
      return make_typed_column<cudf::timestamp_s, cudf::timestamp_s::rep>(
        values, validity, nullable);
    case type_id::TIMESTAMP_MILLISECONDS:
      return make_typed_column<cudf::timestamp_ms, cudf::timestamp_ms::rep>(
        values, validity, nullable);
    case type_id::TIMESTAMP_MICROSECONDS:
      return make_typed_column<cudf::timestamp_us, cudf::timestamp_us::rep>(
        values, validity, nullable);
    case type_id::TIMESTAMP_NANOSECONDS:
      return make_typed_column<cudf::timestamp_ns, cudf::timestamp_ns::rep>(
        values, validity, nullable);
    case type_id::DURATION_DAYS:
      return make_typed_column<cudf::duration_D, cudf::duration_D::rep>(values, validity, nullable);
    case type_id::DURATION_SECONDS:
      return make_typed_column<cudf::duration_s, cudf::duration_s::rep>(values, validity, nullable);
    case type_id::DURATION_MILLISECONDS:
      return make_typed_column<cudf::duration_ms, cudf::duration_ms::rep>(
        values, validity, nullable);
    case type_id::DURATION_MICROSECONDS:
      return make_typed_column<cudf::duration_us, cudf::duration_us::rep>(
        values, validity, nullable);
    case type_id::DURATION_NANOSECONDS:
      return make_typed_column<cudf::duration_ns, cudf::duration_ns::rep>(
        values, validity, nullable);
    case type_id::STRING: {
      std::uniform_int_distribution<int> len_dist(0, 32);
      std::uniform_int_distribution<int> char_dist('a', 'z');
      std::vector<std::string> strings(num_rows);
      for (auto& s : strings) {
        s.resize(len_dist(engine));
        for (auto& ch : s) {
          ch = static_cast<char>(char_dist(engine));
        }
      }
      if (nullable) {
        return cudf::test::strings_column_wrapper(strings.begin(), strings.end(), validity.begin())
          .release();
      }
      return cudf::test::strings_column_wrapper(strings.begin(), strings.end()).release();
    }
    default: CUDF_FAIL("unsupported type id in row-conversion test data factory");
  }
}

struct owning_table {
  std::vector<std::unique_ptr<cudf::column>> columns;
  std::vector<cudf::column_view> views;
  cudf::table_view view() const { return cudf::table_view(views); }
};

owning_table make_random_table(std::vector<cudf::data_type> const& schema,
                               cudf::size_type num_rows,
                               double null_fraction,
                               std::mt19937& engine)
{
  owning_table t;
  for (auto const& type : schema) {
    t.columns.push_back(make_random_column(type, num_rows, null_fraction, engine));
    t.views.push_back(t.columns.back()->view());
  }
  return t;
}

// Every fixed-width type id the row conversion supports (decimal entries carry their scale so the
// same vector serves as the convert_from_rows schema).
std::vector<cudf::data_type> const& all_supported_fixed_width_types()
{
  static std::vector<cudf::data_type> const types{
    cudf::data_type{cudf::type_id::INT8},
    cudf::data_type{cudf::type_id::INT16},
    cudf::data_type{cudf::type_id::INT32},
    cudf::data_type{cudf::type_id::INT64},
    cudf::data_type{cudf::type_id::UINT8},
    cudf::data_type{cudf::type_id::UINT16},
    cudf::data_type{cudf::type_id::UINT32},
    cudf::data_type{cudf::type_id::UINT64},
    cudf::data_type{cudf::type_id::FLOAT32},
    cudf::data_type{cudf::type_id::FLOAT64},
    cudf::data_type{cudf::type_id::BOOL8},
    cudf::data_type{cudf::type_id::DECIMAL32, -2},
    cudf::data_type{cudf::type_id::DECIMAL64, -4},
    cudf::data_type{cudf::type_id::DECIMAL128, -6},
    cudf::data_type{cudf::type_id::TIMESTAMP_DAYS},
    cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS},
    cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS},
    cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS},
    cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS},
    cudf::data_type{cudf::type_id::DURATION_DAYS},
    cudf::data_type{cudf::type_id::DURATION_SECONDS},
    cudf::data_type{cudf::type_id::DURATION_MILLISECONDS},
    cudf::data_type{cudf::type_id::DURATION_MICROSECONDS},
    cudf::data_type{cudf::type_id::DURATION_NANOSECONDS}};
  return types;
}

}  // anonymous namespace

// -- Golden-bytes format pinning ------------------------------------------------------------
//
// Schema INT64,INT32,INT16,INT8 packs hole-free at size-aligned offsets:
//   col0 INT64 @ 0..7, col1 INT32 @ 8..11, col2 INT16 @ 12..13, col3 INT8 @ 14,
//   validity byte @ 15 (bit c = column c valid), row size 16 = stride (zero tail pad).
// Bits 4..7 of the validity byte cover no column and are masked as undefined.
TEST_F(ColumnToRowTests, GoldenBytesFixedWidthLayout)
{
  constexpr cudf::size_type num_rows = 5;
  constexpr std::size_t row_stride   = 16;

  std::array<int64_t, num_rows> const c0{
    0x0102030405060708LL, -1, 0x1122334455667788LL, 42, -9'000'000'000LL};
  std::array<int32_t, num_rows> const c1{0x0A0B0C0D, -2, 7, std::numeric_limits<int32_t>::max(), 0};
  std::array<int16_t, num_rows> const c2{
    0x0102, -3, 300, std::numeric_limits<int16_t>::min(), 32767};
  std::array<int8_t, num_rows> const c3{0x7F, -128, 5, -1, 0};
  // One null per column, each in a different row; row 4 is fully valid.
  std::array<std::array<bool, num_rows>, 4> const valid{{{true, true, false, true, true},
                                                         {true, false, true, true, true},
                                                         {false, true, true, true, true},
                                                         {true, true, true, false, true}}};

  cudf::test::fixed_width_column_wrapper<int64_t> w0(c0.begin(), c0.end(), valid[0].begin());
  cudf::test::fixed_width_column_wrapper<int32_t> w1(c1.begin(), c1.end(), valid[1].begin());
  cudf::test::fixed_width_column_wrapper<int16_t> w2(c2.begin(), c2.end(), valid[2].begin());
  cudf::test::fixed_width_column_wrapper<int8_t> w3(c3.begin(), c3.end(), valid[3].begin());
  cudf::table_view in({w0, w1, w2, w3});

  auto rows = spark_rapids_jni::convert_to_rows(in);
  ASSERT_EQ(rows.size(), 1u);
  auto const lists = cudf::lists_column_view(*rows[0]);
  ASSERT_EQ(static_cast<std::size_t>(lists.child().size()), row_stride * num_rows);

  auto const offsets = row_offsets_to_host(*rows[0]);
  ASSERT_EQ(offsets.size(), static_cast<std::size_t>(num_rows) + 1);
  for (cudf::size_type r = 0; r <= num_rows; ++r) {
    EXPECT_EQ(offsets[r], static_cast<int32_t>(r * row_stride)) << "row " << r;
  }

  std::vector<uint8_t> expected(row_stride * num_rows, 0);
  std::vector<uint8_t> mask(row_stride * num_rows, 0xFF);
  // Bit c of the validity byte follows valid[c][row]: rows 0..3 each null one column, row 4 none.
  std::array<uint8_t, num_rows> const validity_bytes{0x0B, 0x0D, 0x0E, 0x07, 0x0F};
  auto put = [&](cudf::size_type row, std::size_t offset, auto value, bool is_valid) {
    std::memcpy(expected.data() + row * row_stride + offset, &value, sizeof(value));
    if (!is_valid) {
      std::fill_n(mask.begin() + row * row_stride + offset, sizeof(value), uint8_t{0});
    }
  };
  for (cudf::size_type r = 0; r < num_rows; ++r) {
    put(r, 0, c0[r], valid[0][r]);
    put(r, 8, c1[r], valid[1][r]);
    put(r, 12, c2[r], valid[2][r]);
    put(r, 14, c3[r], valid[3][r]);
    expected[r * row_stride + 15] = validity_bytes[r];
    mask[r * row_stride + 15]     = 0x0F;
  }
  expect_masked_bytes_equal(lists.child(), expected, mask);

  // The same bytes must reconstruct the source table.
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT64},
                                      cudf::data_type{cudf::type_id::INT32},
                                      cudf::data_type{cudf::type_id::INT16},
                                      cudf::data_type{cudf::type_id::INT8}};
  auto result = spark_rapids_jni::convert_from_rows(lists, schema);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in, *result);
}

// Schema STRING,INT32,STRING pins the variable-width layout:
//   (offset,length) pair0 @ 0..7, INT32 @ 8..11, pair1 @ 12..19, validity byte @ 20 (bits 0..2),
//   chars @ 21..., row size = round_up(21 + chars, 8); the pair offsets are row-relative.
// Rows (all strings valid; the INT32 is null in row 1):
//   ("AB", 0x01020304, "xyz")        chars "ABxyz"        pairs (21,2),(23,3)  size 26 -> 32
//   ("", null, "Q")                  chars "Q"            pairs (21,0),(21,1)  size 22 -> 24
//   ("hello", 0x7FFFFFFF, "")        chars "hello"        pairs (21,5),(26,0)  size 26 -> 32
//   ("0123456789", -2, "wp")         chars "0123456789wp" pairs (21,10),(31,2) size 33 -> 40
// giving the offsets column {0, 32, 56, 88, 128}.
TEST_F(ColumnToRowTests, GoldenBytesStringLayout)
{
  std::vector<std::string> const s0{"AB", "", "hello", "0123456789"};
  std::vector<int32_t> const ints{0x01020304, -999, std::numeric_limits<int32_t>::max(), -2};
  std::vector<bool> const int_valid{true, false, true, true};
  std::vector<std::string> const s1{"xyz", "Q", "", "wp"};

  cudf::test::strings_column_wrapper w0(s0.begin(), s0.end());
  cudf::test::fixed_width_column_wrapper<int32_t> w1(ints.begin(), ints.end(), int_valid.begin());
  cudf::test::strings_column_wrapper w2(s1.begin(), s1.end());
  cudf::table_view in({w0, w1, w2});

  auto rows = spark_rapids_jni::convert_to_rows(in);
  ASSERT_EQ(rows.size(), 1u);
  auto const lists = cudf::lists_column_view(*rows[0]);

  std::vector<int32_t> const expected_offsets{0, 32, 56, 88, 128};
  auto const offsets = row_offsets_to_host(*rows[0]);
  ASSERT_EQ(offsets.size(), expected_offsets.size());
  for (std::size_t r = 0; r < expected_offsets.size(); ++r) {
    EXPECT_EQ(offsets[r], expected_offsets[r]) << "row " << r;
  }
  ASSERT_EQ(lists.child().size(), expected_offsets.back());

  constexpr std::size_t fixed_and_validity = 21;  // 20 bytes of fixed data + 1 validity byte
  std::vector<uint8_t> expected(expected_offsets.back(), 0);
  std::vector<uint8_t> mask(expected_offsets.back(), 0);  // rows have tail pad: opt-in to defined
  std::array<uint8_t, 4> const validity_bytes{0x07, 0x05, 0x07, 0x07};
  for (std::size_t r = 0; r < s0.size(); ++r) {
    std::size_t const base = static_cast<std::size_t>(expected_offsets[r]);
    auto put_u32           = [&](std::size_t offset, uint32_t value) {
      std::memcpy(expected.data() + base + offset, &value, sizeof(value));
      std::fill_n(mask.begin() + base + offset, sizeof(value), uint8_t{0xFF});
    };
    auto const len0 = static_cast<uint32_t>(s0[r].size());
    auto const len1 = static_cast<uint32_t>(s1[r].size());
    put_u32(0, fixed_and_validity);
    put_u32(4, len0);
    put_u32(8, static_cast<uint32_t>(ints[r]));
    put_u32(12, fixed_and_validity + len0);
    put_u32(16, len1);
    if (!int_valid[r]) { std::fill_n(mask.begin() + base + 8, 4, uint8_t{0}); }
    expected[base + 20] = validity_bytes[r];
    mask[base + 20]     = 0x07;
    std::memcpy(expected.data() + base + fixed_and_validity, s0[r].data(), len0);
    std::memcpy(expected.data() + base + fixed_and_validity + len0, s1[r].data(), len1);
    std::fill_n(mask.begin() + base + fixed_and_validity, len0 + len1, uint8_t{0xFF});
  }
  expect_masked_bytes_equal(lists.child(), expected, mask);

  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::STRING},
                                      cudf::data_type{cudf::type_id::INT32},
                                      cudf::data_type{cudf::type_id::STRING}};
  auto result = spark_rapids_jni::convert_from_rows(lists, schema);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in, *result);
}

// DECIMAL128 elements are 16 bytes and dominate alignment. Schema DECIMAL128,INT32,DECIMAL128,
// INT8 packs as: dec @ 0..15, INT32 @ 16..19, 12-byte alignment hole @ 20..31 (masked), dec @
// 32..47, INT8 @ 48, validity byte @ 49 (bits 0..3), row size 50 -> stride 56 (pad masked).
TEST_F(ColumnToRowTests, GoldenBytesDecimal128Layout)
{
  constexpr cudf::size_type num_rows = 3;
  constexpr std::size_t row_stride   = 56;

  auto const wide = (static_cast<__int128_t>(0x0102030405060708LL) << 64) |
                    static_cast<__int128_t>(0x090A0B0C0D0E0F10LL);
  std::array<__int128_t, num_rows> const d0{
    wide, static_cast<__int128_t>(-1), static_cast<__int128_t>(7)};
  std::array<__int128_t, num_rows> const d1{
    static_cast<__int128_t>(-3), wide, static_cast<__int128_t>(0)};
  std::array<int32_t, num_rows> const iv{11, 22, 33};
  std::array<int8_t, num_rows> const bv{1, -2, 3};
  std::array<bool, num_rows> const d0_valid{true, false, true};
  std::array<bool, num_rows> const iv_valid{true, true, false};

  cudf::test::fixed_point_column_wrapper<__int128_t> w0(
    d0.begin(), d0.end(), d0_valid.begin(), numeric::scale_type{-3});
  cudf::test::fixed_width_column_wrapper<int32_t> w1(iv.begin(), iv.end(), iv_valid.begin());
  cudf::test::fixed_point_column_wrapper<__int128_t> w2(
    d1.begin(), d1.end(), numeric::scale_type{-3});
  cudf::test::fixed_width_column_wrapper<int8_t> w3(bv.begin(), bv.end());
  cudf::table_view in({w0, w1, w2, w3});

  auto rows = spark_rapids_jni::convert_to_rows(in);
  ASSERT_EQ(rows.size(), 1u);
  auto const lists = cudf::lists_column_view(*rows[0]);
  ASSERT_EQ(static_cast<std::size_t>(lists.child().size()), row_stride * num_rows);

  std::vector<uint8_t> expected(row_stride * num_rows, 0);
  std::vector<uint8_t> mask(row_stride * num_rows, 0);
  // Columns 2 and 3 carry no null mask, so their bits are always set.
  std::array<uint8_t, num_rows> const validity_bytes{0x0F, 0x0E, 0x0D};
  for (cudf::size_type r = 0; r < num_rows; ++r) {
    std::size_t const base = r * row_stride;
    auto put               = [&](std::size_t offset, auto value, bool is_valid) {
      std::memcpy(expected.data() + base + offset, &value, sizeof(value));
      std::fill_n(
        mask.begin() + base + offset, sizeof(value), is_valid ? uint8_t{0xFF} : uint8_t{0});
    };
    put(0, d0[r], d0_valid[r]);
    put(16, iv[r], iv_valid[r]);
    put(32, d1[r], true);
    put(48, bv[r], true);
    expected[base + 49] = validity_bytes[r];
    mask[base + 49]     = 0x0F;
  }
  expect_masked_bytes_equal(lists.child(), expected, mask);

  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::DECIMAL128, -3},
                                      cudf::data_type{cudf::type_id::INT32},
                                      cudf::data_type{cudf::type_id::DECIMAL128, -3},
                                      cudf::data_type{cudf::type_id::INT8}};
  auto result = spark_rapids_jni::convert_from_rows(lists, schema);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in, *result);
}

// -- Path-equivalence invariant -------------------------------------------------------------
//
// The general and fixed-width-optimized paths must emit byte-identical rows for any all-fixed
// table. The 16-column schema (12 INT32 @ 0..47, INT16 @ 48, INT16 @ 50, INT8 @ 52, INT8 @ 53,
// validity @ 54..55, stride 56) is hole-free with zero tail pad and exactly 16 validity bits, so
// every output byte is defined and the buffers must match exactly.
TEST_F(ColumnToRowTests, PathEquivalenceFixedWidthBytes)
{
  std::vector<cudf::data_type> schema;
  for (int c = 0; c < 12; ++c) {
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  schema.push_back(cudf::data_type{cudf::type_id::INT16});
  schema.push_back(cudf::data_type{cudf::type_id::INT16});
  schema.push_back(cudf::data_type{cudf::type_id::INT8});
  schema.push_back(cudf::data_type{cudf::type_id::INT8});

  for (cudf::size_type num_rows : {1, 100}) {
    std::mt19937 engine(7013 + num_rows);
    auto const table = make_random_table(schema, num_rows, 0.3, engine);
    auto const in    = table.view();

    auto general_rows   = spark_rapids_jni::convert_to_rows(in);
    auto optimized_rows = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
    ASSERT_EQ(general_rows.size(), 1u);
    ASSERT_EQ(optimized_rows.size(), 1u);

    auto const general =
      cudf::test::to_host<int8_t>(cudf::lists_column_view(*general_rows[0]).child()).first;
    auto const optimized =
      cudf::test::to_host<int8_t>(cudf::lists_column_view(*optimized_rows[0]).child()).first;
    ASSERT_EQ(general.size(), static_cast<std::size_t>(56) * num_rows);
    ASSERT_EQ(general.size(), optimized.size());
    EXPECT_EQ(0, std::memcmp(general.data(), optimized.data(), general.size()));

    auto const general_offsets   = row_offsets_to_host(*general_rows[0]);
    auto const optimized_offsets = row_offsets_to_host(*optimized_rows[0]);
    ASSERT_EQ(general_offsets.size(), optimized_offsets.size());
    for (std::size_t i = 0; i < general_offsets.size(); ++i) {
      EXPECT_EQ(general_offsets[i], optimized_offsets[i]) << "offset " << i;
    }

    // Both from-rows paths must reconstruct the source table.
    expect_general_round_trip(in, schema);
    expect_fixed_width_optimized_round_trip(in, schema);
  }
}

// -- Tile-boundary shapes -----------------------------------------------------------------------
//
// determine_tiles closes a column band once round_up(width, 8) bytes times the 32-row tile height
// exceeds the ~48 KB shared-memory budget (49136 bytes): 191 INT64 columns (1528 B) still fit one
// band, 192 (1536 B) and 193 (1544 B) split into two, and 400 (3200 B) need three.
TEST_F(ColumnToRowTests, TileBandEdgeColumnWidths)
{
  constexpr cudf::size_type num_rows = 97;
  for (int num_cols : {191, 192, 193, 320, 400}) {
    std::vector<cudf::data_type> const schema(num_cols, cudf::data_type{cudf::type_id::INT64});
    for (double null_fraction : {0.0, 0.1}) {
      std::mt19937 engine(3100 + num_cols);
      auto const table = make_random_table(schema, num_rows, null_fraction, engine);
      expect_general_round_trip(table.view(), schema);
    }
  }
}

// Row counts around the 32-row tile height and the 64-row edges on a wide mixed schema
// (256 columns, 32-byte type cycle => 1024-byte fixed region in a single tile band).
TEST_F(ColumnToRowTests, TileBoundaryRowCounts)
{
  std::vector<cudf::data_type> const cycle{cudf::data_type{cudf::type_id::INT64},
                                           cudf::data_type{cudf::type_id::INT32},
                                           cudf::data_type{cudf::type_id::INT16},
                                           cudf::data_type{cudf::type_id::INT8},
                                           cudf::data_type{cudf::type_id::FLOAT64},
                                           cudf::data_type{cudf::type_id::FLOAT32},
                                           cudf::data_type{cudf::type_id::BOOL8},
                                           cudf::data_type{cudf::type_id::INT16}};
  std::vector<cudf::data_type> schema;
  for (int c = 0; c < 256; ++c) {
    schema.push_back(cycle[c % cycle.size()]);
  }

  for (cudf::size_type num_rows : {1, 31, 32, 33, 63, 64, 65}) {
    std::mt19937 engine(9100 + num_rows);
    auto const table = make_random_table(schema, num_rows, 0.15, engine);
    expect_general_round_trip(table.view(), schema);
  }
}

// -- Fixed-width-optimized path boundaries --------------------------------------------------------
//
// The optimized path requires at least 32 rows of shared memory per block, so the padded row size
// (8N data + ceil(N/8) validity rounded up to 8 for N INT64 columns) is capped at 48*1024/32 =
// 1536 bytes: N=188 gives 1528, N=189 exactly 1536, and N=190 (1544) must throw in BOTH
// directions.
TEST_F(ColumnToRowTests, FixedWidthOptimizedRowSizeBoundaries)
{
  constexpr cudf::size_type num_rows = 33;
  for (int num_cols : {188, 189}) {
    std::vector<cudf::data_type> const schema(num_cols, cudf::data_type{cudf::type_id::INT64});
    std::mt19937 engine(5200 + num_cols);
    auto const table = make_random_table(schema, num_rows, 0.15, engine);
    expect_fixed_width_optimized_round_trip(table.view(), schema);
    expect_general_round_trip(table.view(), schema);
  }

  std::vector<cudf::data_type> const schema(190, cudf::data_type{cudf::type_id::INT64});
  std::mt19937 engine(5390);
  auto const table = make_random_table(schema, num_rows, 0.15, engine);
  EXPECT_THROW(spark_rapids_jni::convert_to_rows_fixed_width_optimized(table.view()),
               cudf::logic_error);

  // The general path still handles the same table (two tile bands), and its rows are the right
  // stride for the optimized reader, which must reject them for the same size reason.
  auto rows = spark_rapids_jni::convert_to_rows(table.view());
  ASSERT_EQ(rows.size(), 1u);
  EXPECT_THROW(spark_rapids_jni::convert_from_rows_fixed_width_optimized(
                 cudf::lists_column_view(*rows[0]), schema),
               cudf::logic_error);
  expect_general_round_trip(table.view(), schema);
}

// Row counts around the optimized kernels' 32-row row-group stride.
TEST_F(ColumnToRowTests, FixedWidthOptimizedRowCountBoundaries)
{
  std::vector<cudf::data_type> const schema{cudf::data_type{cudf::type_id::INT64},
                                            cudf::data_type{cudf::type_id::INT32},
                                            cudf::data_type{cudf::type_id::INT16},
                                            cudf::data_type{cudf::type_id::INT8}};
  for (cudf::size_type num_rows : {31, 32, 33}) {
    for (double null_fraction : {0.0, 0.3}) {
      std::mt19937 engine(5400 + num_rows);
      auto const table = make_random_table(schema, num_rows, null_fraction, engine);
      expect_fixed_width_optimized_round_trip(table.view(), schema);
      expect_general_round_trip(table.view(), schema);
    }
  }
}

// -- Validity boundaries --------------------------------------------------------------------------
//
// Column counts {7,8,9} cross the one-validity-byte edge; each null pattern exercises a distinct
// kernel branch, including the absent-null-mask fast path. The round-trip helper checks
// per-column null counts after every convert_from_rows.
TEST_F(RowToColumnTests, ValidityColumnCountBoundaries)
{
  constexpr cudf::size_type num_rows = 100;
  auto data_iter                     = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return static_cast<int32_t>(i * 7919 + 13); });

  for (int num_cols : {7, 8, 9}) {
    // patterns: 0 all-null, 1 none-null with an allocated mask, 2 absent mask, 3 alternating,
    // 4 sparse ~1%
    for (int pattern = 0; pattern < 5; ++pattern) {
      std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> cols;
      cols.reserve(num_cols);
      std::vector<cudf::column_view> views;
      std::vector<cudf::data_type> schema;
      for (int c = 0; c < num_cols; ++c) {
        auto const begin = data_iter + c * num_rows;
        auto const end   = begin + num_rows;
        switch (pattern) {
          case 0: {
            auto v =
              spark_rapids_jni::util::make_counting_transform_iterator(0, [](auto) { return 0; });
            cols.emplace_back(begin, end, v);
            break;
          }
          case 1: {
            auto v =
              spark_rapids_jni::util::make_counting_transform_iterator(0, [](auto) { return 1; });
            cols.emplace_back(begin, end, v);
            break;
          }
          case 2: cols.emplace_back(begin, end); break;
          case 3: {
            auto v = spark_rapids_jni::util::make_counting_transform_iterator(
              0, [](auto i) { return i % 2; });
            cols.emplace_back(begin, end, v);
            break;
          }
          default: {
            auto v = spark_rapids_jni::util::make_counting_transform_iterator(
              0, [](auto i) { return i % 97 != 0; });
            cols.emplace_back(begin, end, v);
            break;
          }
        }
        views.emplace_back(cols.back());
        schema.push_back(cudf::data_type{cudf::type_id::INT32});
      }
      cudf::table_view in(views);
      expect_general_round_trip(in, schema);
    }
  }
}

// {215,216,217} sit at the byte-rounded single-column-tile edge (the split threshold is 221),
// {221,433} force 2- and 3-way column-tile splits at the 216-column stride, and 1500 rows cross
// the 1472-row tile height — across the set the validity kernels split tiles in both dimensions.
TEST_F(RowToColumnTests, ValidityWideColumnBoundaries)
{
  constexpr cudf::size_type num_rows = 1500;
  for (int num_cols : {215, 216, 217, 221, 433}) {
    std::vector<cudf::data_type> const schema(num_cols, cudf::data_type{cudf::type_id::INT8});
    for (double null_fraction : {0.0, 0.5, 1.0}) {
      std::mt19937 engine(4200 + num_cols);
      auto const table = make_random_table(schema, num_rows, null_fraction, engine);
      expect_general_round_trip(table.view(), schema);
    }
  }
}

// -- String edge cases ----------------------------------------------------------------------------

TEST_F(RowToColumnTests, StringEdgeCases)
{
  // String columns in the first and last schema positions; empty strings, an all-empty column,
  // an all-null column, a null/empty mix, 1-char strings, and ~4 KB strings. The ~4 KB column
  // pulls the mean string length past the copy-engine switch, so this reaches the string-parallel
  // from-rows kernel.
  constexpr cudf::size_type num_rows = 65;
  std::vector<std::string> first_col(num_rows), one_char(num_rows), big(num_rows),
    last_col(num_rows), null_empty(num_rows), all_empty(num_rows), all_null(num_rows);
  std::vector<bool> first_valid(num_rows), null_empty_valid(num_rows);
  std::vector<bool> const all_null_valid(num_rows, false);
  std::vector<int32_t> ints(num_rows);
  std::vector<int64_t> longs(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    first_col[i]        = (i % 7 == 0) ? std::string{} : "row" + std::to_string(i);
    first_valid[i]      = i % 11 != 0;
    one_char[i]         = std::string(1, static_cast<char>('a' + i % 26));
    big[i]              = std::string(4096, static_cast<char>('A' + i % 26));
    last_col[i]         = "tail_" + std::to_string(i * 31);
    null_empty_valid[i] = i % 2 == 0;
    ints[i]             = i * 3 - 1;
    longs[i]            = static_cast<int64_t>(i) * 1'000'000'007LL;
  }

  cudf::test::strings_column_wrapper w_first(
    first_col.begin(), first_col.end(), first_valid.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> w_int(ints.begin(), ints.end());
  cudf::test::strings_column_wrapper w_all_empty(all_empty.begin(), all_empty.end());
  cudf::test::strings_column_wrapper w_null_empty(
    null_empty.begin(), null_empty.end(), null_empty_valid.begin());
  cudf::test::fixed_width_column_wrapper<int64_t> w_long(longs.begin(), longs.end());
  cudf::test::strings_column_wrapper w_one_char(one_char.begin(), one_char.end());
  cudf::test::strings_column_wrapper w_all_null(
    all_null.begin(), all_null.end(), all_null_valid.begin());
  cudf::test::strings_column_wrapper w_big(big.begin(), big.end());
  cudf::test::strings_column_wrapper w_last(last_col.begin(), last_col.end());

  cudf::table_view in(
    {w_first, w_int, w_all_empty, w_null_empty, w_long, w_one_char, w_all_null, w_big, w_last});
  std::vector<cudf::data_type> const schema{cudf::data_type{cudf::type_id::STRING},
                                            cudf::data_type{cudf::type_id::INT32},
                                            cudf::data_type{cudf::type_id::STRING},
                                            cudf::data_type{cudf::type_id::STRING},
                                            cudf::data_type{cudf::type_id::INT64},
                                            cudf::data_type{cudf::type_id::STRING},
                                            cudf::data_type{cudf::type_id::STRING},
                                            cudf::data_type{cudf::type_id::STRING},
                                            cudf::data_type{cudf::type_id::STRING}};
  expect_general_round_trip(in, schema);
}

// Row counts straddling a from-rows string-copy block boundary (64 = two 32-string blocks).
TEST_F(RowToColumnTests, StringRowCountBoundaries)
{
  std::vector<cudf::data_type> const schema{cudf::data_type{cudf::type_id::STRING},
                                            cudf::data_type{cudf::type_id::INT32},
                                            cudf::data_type{cudf::type_id::STRING}};
  for (cudf::size_type num_rows : {1, 63, 64, 65}) {
    std::mt19937 engine(6400 + num_rows);
    auto const table = make_random_table(schema, num_rows, 0.2, engine);
    expect_general_round_trip(table.view(), schema);
  }
}

// Eight string columns interleaved with eight fixed-width columns.
TEST_F(RowToColumnTests, StringManyColumnsInterleaved)
{
  std::vector<cudf::data_type> schema;
  for (int c = 0; c < 8; ++c) {
    schema.push_back(cudf::data_type{cudf::type_id::STRING});
    schema.push_back(cudf::data_type{cudf::type_id::INT32});
  }
  constexpr cudf::size_type num_rows = 64;
  std::mt19937 engine(6800);
  auto const table = make_random_table(schema, num_rows, 0.25, engine);
  expect_general_round_trip(table.view(), schema);
}

// -- Full fixed-width type sweep ------------------------------------------------------------------

// Every supported fixed-width type id as a homogeneous 3-column table, with and without nulls,
// through BOTH the general and the fixed-width-optimized paths (the optimized kernels hit their
// wider-than-8-byte fallback branch for DECIMAL128).
TEST_F(RowToColumnTests, TypeSweepHomogeneous)
{
  constexpr cudf::size_type num_rows = 67;
  for (auto const& type : all_supported_fixed_width_types()) {
    std::vector<cudf::data_type> const schema(3, type);
    for (double null_fraction : {0.0, 0.35}) {
      std::mt19937 engine(1000 + static_cast<int>(type.id()));
      auto const table = make_random_table(schema, num_rows, null_fraction, engine);
      expect_general_round_trip(table.view(), schema);
      expect_fixed_width_optimized_round_trip(table.view(), schema);
    }
  }
}

// All supported fixed-width types together in one mixed schema.
TEST_F(RowToColumnTests, TypeSweepMixedSchema)
{
  auto const& schema                 = all_supported_fixed_width_types();
  constexpr cudf::size_type num_rows = 129;
  for (double null_fraction : {0.0, 0.3}) {
    std::mt19937 engine(2026);
    auto const table = make_random_table(schema, num_rows, null_fraction, engine);
    expect_general_round_trip(table.view(), schema);
    expect_fixed_width_optimized_round_trip(table.view(), schema);
  }
}

// Multi-band structural coverage. Replaying the tiler over this 9-type cycle: 320 columns force a
// 2-band split and 640 force 3, with band 0 ending at 1521 B -- a boundary that is NOT 8-aligned,
// which is the shape spark-rapids-jni#4590 corrupted. `TileBoundaryWideInt32RoundTrip` cannot reach
// that shape at all: every column there is INT32, so its band boundaries land on multiples of 8 by
// construction.
//
// These do NOT gate that defect, and mutation testing says no fixed-width round trip can. Restoring
// the padded row width leaves all of them green, because a band closes at the first column that
// raises the PADDED row size, which forces the next band to begin at exactly that rounded offset.
// The over-long write therefore stops where the next band starts, and its stray bytes land in JCUDF
// alignment padding that a round trip never reads. Keep these as multi-band coverage; they are not
// a guard on the un-padded write.

namespace {

// A column of the benchmark's `default_cycle` type at `position`, so the schema reproduces the byte
// offsets the tiler sees. Without nulls the column carries NO null mask, which is the `nulls=none`
// branch the crash was reported on and a different code path from an all-true mask.
template <typename T, typename DataIter, typename ValidIter>
cudf::column make_cycle_column(DataIter data,
                               cudf::size_type num_rows,
                               bool with_nulls,
                               ValidIter valid)
{
  // Move out of the owning pointer: dereferencing a raw pointer here would both leak the column
  // and deep-copy its device buffers into the return value.
  auto col = with_nulls
               ? cudf::test::fixed_width_column_wrapper<T>(data, data + num_rows, valid).release()
               : cudf::test::fixed_width_column_wrapper<T>(data, data + num_rows).release();
  return std::move(*col);
}

void multi_band_round_trip(int num_columns, cudf::size_type num_rows, bool with_nulls)
{
  std::vector<cudf::column> cols;
  std::vector<cudf::data_type> schema;
  cols.reserve(num_columns);
  schema.reserve(num_columns);

  auto valid =
    spark_rapids_jni::util::make_counting_transform_iterator(0, [](auto i) { return i % 7 != 0; });

  for (int c = 0; c < num_columns; ++c) {
    auto data = spark_rapids_jni::util::make_counting_transform_iterator(
      c * num_rows, [](auto i) -> int64_t { return static_cast<int64_t>(i * 2654435761u); });
    auto add = [&](auto tag, cudf::type_id id) {
      cols.push_back(make_cycle_column<decltype(tag)>(data, num_rows, with_nulls, valid));
      schema.push_back(cudf::data_type{id});
    };
    switch (c % 9) {
      case 0: add(int8_t{}, cudf::type_id::INT8); break;
      case 1: add(int32_t{}, cudf::type_id::INT32); break;
      case 2: add(int16_t{}, cudf::type_id::INT16); break;
      case 3: add(int64_t{}, cudf::type_id::INT64); break;
      case 4: add(int32_t{}, cudf::type_id::INT32); break;
      case 5: add(bool{}, cudf::type_id::BOOL8); break;
      case 6: add(uint16_t{}, cudf::type_id::UINT16); break;
      case 7: add(uint8_t{}, cudf::type_id::UINT8); break;
      default: add(uint64_t{}, cudf::type_id::UINT64); break;
    }
  }

  std::vector<cudf::column_view> views(cols.begin(), cols.end());
  cudf::table_view in(views);

  // Repeat, for the same reason TileBoundaryWideInt32RoundTrip does: an overlapping tile write is a
  // race between adjacent blocks, so a single round trip can pass on a run where the timing happens
  // not to collide. A misaligned schema without repetition detects nothing reliably.
  for (int iter = 0; iter < 8; ++iter) {
    auto rows = spark_rapids_jni::convert_to_rows(in);
    ASSERT_EQ(rows.size(), 1u);
    auto result = spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*rows[0]), schema);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in, *result);
  }
}

}  // namespace

TEST_F(ColumnToRowTests, MultiBandTileSplitTwoBandsWithNulls)
{
  multi_band_round_trip(320, 65, true);
}

TEST_F(ColumnToRowTests, MultiBandTileSplitThreeBandsWithNulls)
{
  multi_band_round_trip(640, 65, true);
}

// The one band boundary the fixed-width tests above cannot reach: a string or list column occupies
// 8 bytes but is only 4-byte aligned, while every fixed-width type has alignment == size. That
// makes it the sole way a band can begin BEFORE the 8-byte rounding of the previous band's end.
//
// Replaying the tiler: 381 leading INT32 columns leave band 0 ending at 1524 B, and the string at
// column 381 starts at 1524 -- four bytes below round_up(1524, 8). The count is exact rather than
// incidental: 379 and 383 leading columns both close the band 8-aligned.
//
// This does NOT gate spark-rapids-jni#4590 either, and mutation testing says nothing here does.
// Draining the padded width leaves this green too, because `copy_strings_to_rows` launches after
// `copy_to_rows` on the same stream and rewrites that offset/length pair at exactly the overlapped
// address. Kept for the coverage: no other test combines a multi-band split with a string on a band
// edge.
TEST_F(ColumnToRowTests, StringAtMisalignedBandBoundary)
{
  constexpr int leading_int32        = 381;
  constexpr int trailing_int32       = 20;
  constexpr cudf::size_type num_rows = 65;

  char const* const test_strings[] = {
    "", "a", "small", "a considerably longer string value", "medium length"};
  auto const num_strings = static_cast<int>(std::size(test_strings));

  std::vector<cudf::test::detail::column_wrapper> cols;
  std::vector<cudf::column_view> views;
  std::vector<cudf::data_type> schema;

  auto add_int32 = [&](int seed) {
    auto data = spark_rapids_jni::util::make_counting_transform_iterator(
      seed * num_rows, [](auto i) -> int32_t { return static_cast<int32_t>(i * 2654435761u); });
    cols.emplace_back(cudf::test::fixed_width_column_wrapper<int32_t>(data, data + num_rows));
    views.push_back(cols.back());
    schema.emplace_back(cudf::data_type{cudf::type_id::INT32});
  };

  for (int c = 0; c < leading_int32; ++c) {
    add_int32(c);
  }

  // Varied lengths so a corrupted offset/length pair changes the decoded value rather than
  // happening to select an identical string.
  auto strings = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [&](auto i) -> char const* { return test_strings[i % num_strings]; });
  cols.emplace_back(cudf::test::strings_column_wrapper(strings, strings + num_rows));
  views.push_back(cols.back());
  schema.emplace_back(cudf::data_type{cudf::type_id::STRING});

  for (int c = 0; c < trailing_int32; ++c) {
    add_int32(leading_int32 + 1 + c);
  }

  cudf::table_view in(views);

  // Repeat: an overlapping tile write is a race between adjacent blocks, so one round trip can pass
  // on a run where the timing happens not to collide.
  for (int iter = 0; iter < 8; ++iter) {
    auto rows = spark_rapids_jni::convert_to_rows(in);
    ASSERT_EQ(rows.size(), 1u);
    auto result = spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*rows[0]), schema);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in, *result);
  }
}

namespace {

// Restores ROWCONV_VALIDATE_ALL_VALID on scope exit, including when a test body throws -- a leaked
// setting would silently reroute every later test in the binary through the validating path.
class scoped_validate_all_valid {
 public:
  explicit scoped_validate_all_valid(bool enable)
    : _had(std::getenv("ROWCONV_VALIDATE_ALL_VALID") != nullptr)
  {
    if (_had) { _old = std::getenv("ROWCONV_VALIDATE_ALL_VALID"); }
    setenv("ROWCONV_VALIDATE_ALL_VALID", enable ? "1" : "0", 1);
  }
  ~scoped_validate_all_valid()
  {
    if (_had) {
      setenv("ROWCONV_VALIDATE_ALL_VALID", _old.c_str(), 1);
    } else {
      unsetenv("ROWCONV_VALIDATE_ALL_VALID");
    }
  }
  scoped_validate_all_valid(scoped_validate_all_valid const&)            = delete;
  scoped_validate_all_valid& operator=(scoped_validate_all_valid const&) = delete;

 private:
  bool _had;
  std::string _old;
};

// Asserts every column's mask state, which is the ONLY observable that distinguishes the two paths
// (values are identical by construction on null-free input).
void expect_nullability(cudf::table_view const& t, bool expected)
{
  for (cudf::size_type i = 0; i < t.num_columns(); ++i) {
    EXPECT_EQ(t.column(i).nullable(), expected) << "column " << i << " nullability";
    if (!expected) { EXPECT_EQ(t.column(i).null_count(), 0) << "column " << i << " null count"; }
  }
}

}  // namespace

// The core contract for the general path: on genuinely null-free input the all-valid overload
// returns the same VALUES as the incumbent and returns them WITHOUT masks. Both halves are load
// bearing -- values alone would not show the path engaged, masks alone would not show it stayed
// correct.
TEST_F(RowToColumnTests, AllValidGeneralMatchesIncumbent)
{
  constexpr int num_rows = 137;  // Not a multiple of 32, so the final validity word is partial.

  auto seq = spark_rapids_jni::util::make_counting_transform_iterator(
    0, [](auto i) -> int32_t { return static_cast<int32_t>(i * 2654435761u); });
  cudf::test::fixed_width_column_wrapper<int32_t> c0(seq, seq + num_rows);
  cudf::test::fixed_width_column_wrapper<int64_t> c1(seq, seq + num_rows);
  cudf::test::fixed_width_column_wrapper<int8_t> c2(seq, seq + num_rows);
  cudf::table_view in({c0, c1, c2});
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT32},
                                      cudf::data_type{cudf::type_id::INT64},
                                      cudf::data_type{cudf::type_id::INT8}};

  auto rows = spark_rapids_jni::convert_to_rows(in);
  ASSERT_EQ(rows.size(), 1u);
  auto const rows_view = cudf::lists_column_view(*rows[0]);

  auto incumbent = spark_rapids_jni::convert_from_rows(rows_view, schema);
  auto fast = spark_rapids_jni::convert_from_rows(rows_view, schema, std::vector<bool>(3, false));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in, *fast);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*incumbent, *fast);
  expect_nullability(*fast, false);
  expect_nullability(*incumbent, true);
}

// Strings take a separate branch: the general path allocates the per-row offset column with
// `!trust_all_valid`, and the assembled string column takes its mask from a different source than
// the fixed-width columns do. That branch is the least obviously covered by the two tests above,
// which is exactly why it gets its own.
TEST_F(RowToColumnTests, AllValidWithStringsMatchesIncumbent)
{
  cudf::test::fixed_width_column_wrapper<int32_t> c0({1, 2, 3, 4, 5});
  cudf::test::strings_column_wrapper c1({"alpha", "", "gamma", "a much longer string value", "d"});
  cudf::test::fixed_width_column_wrapper<int64_t> c2({10, 20, 30, 40, 50});
  cudf::table_view in({c0, c1, c2});
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT32},
                                      cudf::data_type{cudf::type_id::STRING},
                                      cudf::data_type{cudf::type_id::INT64}};

  auto rows = spark_rapids_jni::convert_to_rows(in);
  ASSERT_EQ(rows.size(), 1u);
  auto const rows_view = cudf::lists_column_view(*rows[0]);

  auto incumbent = spark_rapids_jni::convert_from_rows(rows_view, schema);
  auto fast = spark_rapids_jni::convert_from_rows(rows_view, schema, std::vector<bool>(3, false));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in, *fast);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*incumbent, *fast);
  expect_nullability(*fast, false);
}

// The fast path is ALL-OR-NOTHING per table: `is_all_valid_profile` is a `none_of`, so one true
// entry routes the WHOLE call through the incumbent -- the columns declared non-nullable still come
// back with masks. This is the behaviour that bounds where the optimization applies in production,
// so it is pinned rather than left to the reader of the predicate.
TEST_F(RowToColumnTests, AllValidRequiresEveryColumnNonNullable)
{
  cudf::test::fixed_width_column_wrapper<int32_t> c0({1, 2, 3, 4});
  cudf::test::fixed_width_column_wrapper<int32_t> c1({5, 6, 7, 8}, {true, false, true, true});
  cudf::table_view in({c0, c1});
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT32},
                                      cudf::data_type{cudf::type_id::INT32}};

  auto rows = spark_rapids_jni::convert_to_rows(in);
  ASSERT_EQ(rows.size(), 1u);
  auto const rows_view = cudf::lists_column_view(*rows[0]);

  // Column 0 is genuinely null-free, but column 1 is not: the declaration must disable the fast
  // path for BOTH, and the null in column 1 must survive.
  auto result =
    spark_rapids_jni::convert_from_rows(rows_view, schema, std::vector<bool>{false, true});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(in, *result);
  expect_nullability(*result, true);
  EXPECT_EQ(result->get_column(1).null_count(), 1);
}

// A declaration whose length does not match the schema is caller error, and silently trusting a
// short vector would read past its end inside `none_of`. Pinned on both entry points because they
// validate through the same helper and a future refactor could easily drop one.
TEST_F(RowToColumnTests, AllValidRejectsMismatchedDeclarationSize)
{
  cudf::test::fixed_width_column_wrapper<int32_t> c0({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> c1({4, 5, 6});
  cudf::table_view in({c0, c1});
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT32},
                                      cudf::data_type{cudf::type_id::INT32}};

  auto rows     = spark_rapids_jni::convert_to_rows(in);
  auto rows_opt = spark_rapids_jni::convert_to_rows_fixed_width_optimized(in);
  ASSERT_EQ(rows.size(), 1u);
  ASSERT_EQ(rows_opt.size(), 1u);

  for (auto const& bad : {std::vector<bool>{false}, std::vector<bool>{false, false, false}}) {
    EXPECT_THROW(
      spark_rapids_jni::convert_from_rows(cudf::lists_column_view(*rows[0]), schema, bad),
      cudf::logic_error);
    EXPECT_THROW(spark_rapids_jni::convert_from_rows_fixed_width_optimized(
                   cudf::lists_column_view(*rows_opt[0]), schema, bad),
                 cudf::logic_error);
  }
}

// A false all-valid declaration corrupts silently by design -- the row bytes are read unchanged and
// their validity bytes are simply never consulted, so nulls come back as whatever the underlying
// storage happened to hold. ROWCONV_VALIDATE_ALL_VALID=1 is the debug net for that, and a net that
// is never tested is not a net. The corrupt output itself is NOT asserted on: the values in a null
// slot are unspecified, so any expectation about them would be testing an accident.
TEST_F(RowToColumnTests, AllValidValidatorRejectsFalseDeclaration)
{
  cudf::test::fixed_width_column_wrapper<int32_t> c0({1, 2, 3, 4}, {true, true, false, true});
  cudf::table_view in({c0});
  std::vector<cudf::data_type> schema{cudf::data_type{cudf::type_id::INT32}};

  auto rows = spark_rapids_jni::convert_to_rows(in);
  ASSERT_EQ(rows.size(), 1u);
  auto const rows_view = cudf::lists_column_view(*rows[0]);

  scoped_validate_all_valid guard(true);
  EXPECT_THROW(spark_rapids_jni::convert_from_rows(rows_view, schema, std::vector<bool>{false}),
               cudf::logic_error);
}
