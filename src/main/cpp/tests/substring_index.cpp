/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>

#include <substring_index.hpp>

#include <string>
#include <vector>

using namespace cudf;

struct SubstringIndexTests : public test::BaseFixture {};

TEST_F(SubstringIndexTests, ScalarDelimiter)
{
  auto col0 = test::strings_column_wrapper({"www.yahoo.com",
                                            "www.apache..org",
                                            "tennis...com",
                                            "nvidia....com",
                                            "google...........com",
                                            "microsoft...c.....co..m"});

  auto exp_results = test::strings_column_wrapper(
    {"www.yahoo.com", "www.apache.", "tennis..", "nvidia..", "google..", "microsoft.."});

  auto results =
    spark_rapids_jni::substring_index(strings_column_view{col0}, string_scalar("."), 3);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
}

TEST_F(SubstringIndexTests, OverlappingDelimiter)
{
  // Spark's UTF8String.subStringIndex lets matches of a multi-character delimiter overlap.
  auto const input =
    test::strings_column_wrapper({"||||||", "aaaa", "aaaaa", "abababa", "千千千千"});
  auto const view = strings_column_view{input};

  {
    auto const results = spark_rapids_jni::substring_index(view, string_scalar("|||"), 3);
    auto const expected =
      test::strings_column_wrapper({"||", "aaaa", "aaaaa", "abababa", "千千千千"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const results = spark_rapids_jni::substring_index(view, string_scalar("|||"), -4);
    auto const expected =
      test::strings_column_wrapper({"|||", "aaaa", "aaaaa", "abababa", "千千千千"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const results  = spark_rapids_jni::substring_index(view, string_scalar("aa"), 2);
    auto const expected = test::strings_column_wrapper({"||||||", "a", "a", "abababa", "千千千千"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const results  = spark_rapids_jni::substring_index(view, string_scalar("aa"), -2);
    auto const expected = test::strings_column_wrapper({"||||||", "a", "a", "abababa", "千千千千"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const results = spark_rapids_jni::substring_index(view, string_scalar("aba"), 2);
    auto const expected =
      test::strings_column_wrapper({"||||||", "aaaa", "aaaaa", "ab", "千千千千"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const results = spark_rapids_jni::substring_index(view, string_scalar("aba"), -2);
    auto const expected =
      test::strings_column_wrapper({"||||||", "aaaa", "aaaaa", "ba", "千千千千"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const results = spark_rapids_jni::substring_index(view, string_scalar("千千"), 2);
    auto const expected =
      test::strings_column_wrapper({"||||||", "aaaa", "aaaaa", "abababa", "千"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}
