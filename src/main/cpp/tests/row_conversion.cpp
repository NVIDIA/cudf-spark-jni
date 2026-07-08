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

#include <row_conversion.hpp>
#include <utilities/iterator.cuh>

#include <cstddef>
#include <cstring>
#include <limits>
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

// Regression repro for spark-rapids-jni#4588. Disabled by default for the same memory reason as
// HugeStringRowThrows.
//
// The old kernel launch passed batch_num_rows (a per-batch count) as the kernel's `num_rows`
// while start_row was an absolute index, so all batches whose start lay past the per-batch
// count silently produced uninitialized output. The fix passes batch_row_offset +
// batch_num_rows as the absolute end bound; this test exercises the multi-batch path.
TEST_F(ColumnToRowTests, DISABLED_MultiBatchStringDoesNotSkip)
{
  constexpr std::size_t per_col_bytes = 33ULL * 1024 * 1024;
  constexpr int num_rows              = 35;

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
                                "cannot be reproduced — increase per_col_bytes or num_rows.";

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
