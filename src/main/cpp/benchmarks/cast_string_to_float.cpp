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

#include "common/generate_input.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cast_string.hpp>
#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

// Construct host strings that all satisfy the helper's caller-side gate:
// `digits > 2^53 AND |exp_ten| <= 19`. The mantissa is capped at 19 digits
// (max 9999999999999999999) so the parser's `max_safe_digits == 19` truncation
// is never triggered — otherwise a 20-digit uint64 would lose its trailing
// digit, bump `exp_base` by 1, and push `exp_ten` to 20, missing the gate.
std::vector<std::string> make_helper_triggering_strings(cudf::size_type n_rows)
{
  std::mt19937_64 rng{0xC0FFEEULL};
  // Inclusive upper bound = 10^19 - 1 (the largest 19-digit value).
  std::uniform_int_distribution<uint64_t> digit_dist{(1ULL << 53) + 1, 9999999999999999999ULL};
  std::uniform_int_distribution<int> q_dist{-19, 19};

  std::vector<std::string> out;
  out.reserve(static_cast<size_t>(n_rows));
  for (cudf::size_type i = 0; i < n_rows; ++i) {
    uint64_t const digits = digit_dist(rng);
    int const q           = q_dist(rng);
    std::string s         = std::to_string(digits);
    if (q != 0) {
      s += 'e';
      s += std::to_string(q);
    }
    out.push_back(std::move(s));
  }
  return out;
}

}  // namespace

// FP32 cast — the correctly-rounded helper does not run on this code path
// (the `cuda::std::is_same_v<T, double>` term in the `helper_eligible` gate
// skips it for float). Kept as the original FP32 baseline.
void string_to_float(nvbench::state& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.get_int64("num_rows")};
  auto const float_tbl  = create_random_table({cudf::type_id::FLOAT32}, row_count{n_rows});
  auto const float_col  = float_tbl->get_column(0);
  auto const string_col = cudf::strings::from_floats(float_col.view());

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto rows = spark_rapids_jni::string_to_float(cudf::data_type{cudf::type_id::FLOAT32},
                                                  string_col->view(),
                                                  false,
                                                  cudf::get_default_stream());
  });
}

NVBENCH_BENCH(string_to_float)
  .set_name("Strings to Float Cast (FP32 baseline)")
  .add_int64_axis("num_rows", {1 * 1024 * 1024, 100 * 1024 * 1024});

// FP64 cast where the correctly-rounded helper is NEVER entered. Inputs are
// produced by `cudf::strings::from_floats`, which caps significant digits at
// 10 via `nine_digits = 10^9` in `dissect_value` (cudf
// `convert_floats.cu:134`), so the parsed mantissa is at most ~10^10 — well
// below 2^53 (~9.007e15). The caller's `helper_eligible` predicate is `false`
// for every row and `default_double_path<T>` runs. The trigger rate is
// asserted statically by `from_floats`'s contract (no host-side re-parse
// needed). This is the regression-check baseline: cost on the FP64 path
// WITHOUT the new helper. Input column is non-nullable to match helper_on's
// column nullability and keep the comparison apples-to-apples (the default
// `create_random_table` profile injects nulls that would early out of the
// parser before the default path runs).
void string_to_double_helper_off(nvbench::state& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.get_int64("num_rows")};
  data_profile const profile = data_profile_builder().no_validity();
  auto const float_tbl  = create_random_table({cudf::type_id::FLOAT64}, row_count{n_rows}, profile);
  auto const float_col  = float_tbl->get_column(0);
  auto const string_col = cudf::strings::from_floats(float_col.view());

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto rows = spark_rapids_jni::string_to_float(cudf::data_type{cudf::type_id::FLOAT64},
                                                  string_col->view(),
                                                  false,
                                                  cudf::get_default_stream());
  });
}

NVBENCH_BENCH(string_to_double_helper_off)
  .set_name("Strings to Double Cast (helper OFF — from_floats, digits <= 10^10)")
  .add_int64_axis("num_rows", {1 * 1024 * 1024, 100 * 1024 * 1024});

// FP64 cast where the correctly-rounded helper IS entered for every row.
// Strings are hand-built on the host so the parsed mantissa is always
// `> 2^53` and the explicit exponent is always in `[-19, 19]`, guaranteeing
// the caller's `helper_eligible` predicate is `true` for every row.
// This measures the worst case: every row pays the int128 helper cost.
void string_to_double_helper_on(nvbench::state& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.get_int64("num_rows")};
  auto const host_strings = make_helper_triggering_strings(n_rows);
  cudf::test::strings_column_wrapper string_col(host_strings.begin(), host_strings.end());
  cudf::strings_column_view const scv{static_cast<cudf::column_view>(string_col)};
  auto const stream = cudf::get_default_stream();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto rows = spark_rapids_jni::string_to_float(
      cudf::data_type{cudf::type_id::FLOAT64}, scv, false, stream);
  });
}

// Note: helper_on uses smaller row counts than the other axes because the
// handcrafted inputs average ~20 chars/row; at 100M rows the total chars
// would exceed `INT32_MAX` and overflow the strings column's int32 offsets.
// `from_floats`-based inputs (the other two axes) are ~10 chars/row so 100M
// fits.
NVBENCH_BENCH(string_to_double_helper_on)
  .set_name("Strings to Double Cast (helper ON — handcrafted digits>2^53, |q|<=19)")
  .add_int64_axis("num_rows", {1 * 1024 * 1024, 10 * 1024 * 1024});
