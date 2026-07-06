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

#include "../nvtx_ranges.hpp"
#include "datetime_utils.cuh"
#include "iceberg_datetime_util.hpp"
#include "integer_utils.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/tabulate.h>

namespace spark_rapids_jni {

namespace {
// the year of epoch day 1970-01-01
constexpr int32_t EPOCH_YEAR = 1970;

// the month of epoch day 1970-01-01
constexpr int32_t EPOCH_MONTH = 1;

constexpr int32_t MONTHS_PER_YEAR = 12;

constexpr int64_t MICROS_PER_DAY = 86400L * 1'000'000L;

constexpr int64_t MICROS_PER_HOUR = 3600L * 1'000'000L;

/**
 * @brief Functor to compute year difference between epoch and date
 */
struct years_from_epoch_for_date_fn {
  int32_t const* dates;

  __device__ int32_t operator()(int32_t row_index) const
  {
    int32_t year, month, day;
    spark_rapids_jni::date_time_utils::to_date(dates[row_index], year, month, day);
    return year - EPOCH_YEAR;
  }
};

/**
 * @brief Functor to compute year difference between epoch and ts (in microseconds)
 */
struct years_from_epoch_for_ts_fn {
  int64_t const* tss;

  __device__ int32_t operator()(int32_t row_index) const
  {
    int days = static_cast<int>(integer_utils::floor_div(tss[row_index], MICROS_PER_DAY));
    int32_t year, month, day;
    spark_rapids_jni::date_time_utils::to_date(days, year, month, day);
    return year - EPOCH_YEAR;
  }
};

/**
 * @brief Functor to compute month difference between epoch and date
 */
struct months_from_epoch_for_date_fn {
  int32_t const* dates;

  __device__ int32_t operator()(int32_t row_index) const
  {
    int32_t year, month, day;
    spark_rapids_jni::date_time_utils::to_date(dates[row_index], year, month, day);
    return (year - EPOCH_YEAR) * MONTHS_PER_YEAR + (month - EPOCH_MONTH);
  }
};

/**
 * @brief Functor to compute month difference between epoch and ts
 */
struct months_from_epoch_for_ts_fn {
  int64_t const* tss;

  __device__ int32_t operator()(int32_t row_index) const
  {
    int days = static_cast<int>(integer_utils::floor_div(tss[row_index], MICROS_PER_DAY));
    int32_t year, month, day;
    spark_rapids_jni::date_time_utils::to_date(days, year, month, day);
    return (year - EPOCH_YEAR) * MONTHS_PER_YEAR + (month - EPOCH_MONTH);
  }
};

/**
 * @brief Functor to compute day difference between epoch and date
 */
struct days_from_epoch_for_date_fn {
  int32_t const* dates;

  __device__ int32_t operator()(int32_t row_index) const { return dates[row_index]; }
};

/**
 * @brief Functor to compute day difference between epoch and ts
 */
struct days_from_epoch_for_ts_fn {
  int64_t const* tss;

  __device__ int32_t operator()(int32_t row_index) const
  { return static_cast<int>(integer_utils::floor_div(tss[row_index], MICROS_PER_DAY)); }
};

/**
 * @brief Functor to compute hour difference between epoch and ts
 */
struct hours_from_epoch_for_ts_fn {
  int64_t const* tss;

  __device__ int32_t operator()(int32_t row_index) const
  {
    // can overflow, but Spark does the same
    return static_cast<int>(integer_utils::floor_div(tss[row_index], MICROS_PER_HOUR));
  }
};

std::unique_ptr<cudf::column> compute_years_from_epoch(cudf::column_view const& input,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32}); }

  auto result = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32},
                                              input.size(),
                                              cudf::copy_bitmask(input, stream, mr),
                                              input.null_count(),
                                              stream,
                                              mr);
  if (input.type().id() == cudf::type_id::TIMESTAMP_DAYS) {
    thrust::tabulate(rmm::exec_policy_nosync(stream),
                     result->mutable_view().begin<int32_t>(),
                     result->mutable_view().end<int32_t>(),
                     years_from_epoch_for_date_fn{input.begin<int32_t>()});
    return result;
  } else if (input.type().id() == cudf::type_id::TIMESTAMP_MICROSECONDS) {
    thrust::tabulate(rmm::exec_policy_nosync(stream),
                     result->mutable_view().begin<int32_t>(),
                     result->mutable_view().end<int32_t>(),
                     years_from_epoch_for_ts_fn{input.begin<int64_t>()});
    return result;
  } else {
    CUDF_FAIL("Unsupported type for to_years");
  }
}

std::unique_ptr<cudf::column> compute_months_from_epoch(cudf::column_view const& input,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32}); }

  auto result = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32},
                                              input.size(),
                                              cudf::copy_bitmask(input, stream, mr),
                                              input.null_count(),
                                              stream,
                                              mr);
  if (input.type().id() == cudf::type_id::TIMESTAMP_DAYS) {
    thrust::tabulate(rmm::exec_policy_nosync(stream),
                     result->mutable_view().begin<int32_t>(),
                     result->mutable_view().end<int32_t>(),
                     months_from_epoch_for_date_fn{input.begin<int32_t>()});
    return result;
  } else if (input.type().id() == cudf::type_id::TIMESTAMP_MICROSECONDS) {
    thrust::tabulate(rmm::exec_policy_nosync(stream),
                     result->mutable_view().begin<int32_t>(),
                     result->mutable_view().end<int32_t>(),
                     months_from_epoch_for_ts_fn{input.begin<int64_t>()});
    return result;
  } else {
    CUDF_FAIL("Unsupported type for to_months");
  }
}

std::unique_ptr<cudf::column> compute_days_from_epoch(cudf::column_view const& input,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS});
  }

  auto result = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS},
                                              input.size(),
                                              cudf::copy_bitmask(input, stream, mr),
                                              input.null_count(),
                                              stream,
                                              mr);
  if (input.type().id() == cudf::type_id::TIMESTAMP_DAYS) {
    thrust::tabulate(rmm::exec_policy_nosync(stream),
                     result->mutable_view().begin<int32_t>(),
                     result->mutable_view().end<int32_t>(),
                     days_from_epoch_for_date_fn{input.begin<int32_t>()});
    return result;
  } else if (input.type().id() == cudf::type_id::TIMESTAMP_MICROSECONDS) {
    thrust::tabulate(rmm::exec_policy_nosync(stream),
                     result->mutable_view().begin<int32_t>(),
                     result->mutable_view().end<int32_t>(),
                     days_from_epoch_for_ts_fn{input.begin<int64_t>()});
    return result;
  } else {
    CUDF_FAIL("Unsupported type for to_days");
  }
}

std::unique_ptr<cudf::column> compute_hours_from_epoch(cudf::column_view const& input,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32}); }

  auto result = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32},
                                              input.size(),
                                              cudf::copy_bitmask(input, stream, mr),
                                              input.null_count(),
                                              stream,
                                              mr);

  if (input.type().id() == cudf::type_id::TIMESTAMP_MICROSECONDS) {
    thrust::tabulate(rmm::exec_policy_nosync(stream),
                     result->mutable_view().begin<int32_t>(),
                     result->mutable_view().end<int32_t>(),
                     hours_from_epoch_for_ts_fn{input.begin<int64_t>()});
    return result;
  } else {
    CUDF_FAIL("Unsupported type for to_hours");
  }
}

}  // anonymous namespace

std::unique_ptr<cudf::column> years_from_epoch(cudf::column_view const& input,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return compute_years_from_epoch(input, stream, mr);
}

std::unique_ptr<cudf::column> months_from_epoch(cudf::column_view const& input,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return compute_months_from_epoch(input, stream, mr);
}

std::unique_ptr<cudf::column> days_from_epoch(cudf::column_view const& input,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return compute_days_from_epoch(input, stream, mr);
}

std::unique_ptr<cudf::column> hours_from_epoch(cudf::column_view const& input,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return compute_hours_from_epoch(input, stream, mr);
}

}  // namespace spark_rapids_jni
