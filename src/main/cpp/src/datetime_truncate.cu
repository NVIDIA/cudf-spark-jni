/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

#include "datetime_utils.hpp"
#include "nvtx_ranges.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/functional>
#include <cuda/std/optional>
#include <cuda/std/utility>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include <type_traits>

namespace spark_rapids_jni {

namespace detail {

namespace {
/**
 * @brief The date/time format to perform truncation.
 *
 * The format must match the descriptions in the Apache Spark documentation:
 * - https://spark.apache.org/docs/latest/api/sql/index.html#trunc
 * - https://spark.apache.org/docs/latest/api/sql/index.html#date_trunc
 */
enum class truncation_format : uint8_t {
  YEAR,
  YYYY,
  YY,
  QUARTER,
  MONTH,
  MM,
  MON,
  WEEK,
  DAY,
  DD,
  HOUR,
  MINUTE,
  SECOND,
  MILLISECOND,
  MICROSECOND,
  INVALID
};

// Convert an ASCII character into uppercase.
__host__ __device__ char to_upper(char const c)
{ return ('a' <= c && c <= 'z') ? static_cast<char>(static_cast<unsigned int>(c ^ 0x20)) : c; }

// Parse the truncation format from a string.
__host__ __device__ truncation_format parse_format(char const* fmt_data, cudf::size_type fmt_size)
{
  // This must be kept in sync with the `truncation_format` enum.
  char const* components[] = {"YEAR",
                              "YYYY",
                              "YY",
                              "QUARTER",
                              "MONTH",
                              "MM",
                              "MON",
                              "WEEK",
                              "DAY",
                              "DD",
                              "HOUR",
                              "MINUTE",
                              "SECOND",
                              "MILLISECOND",
                              "MICROSECOND"};
  // Manually calculate sizes of the strings since `strlen` is not available in device code.
  cudf::size_type constexpr comp_sizes[] = {4, 4, 2, 7, 5, 2, 3, 4, 3, 2, 4, 6, 6, 11, 11};
  auto constexpr num_components          = sizeof(components) / sizeof(components[0]);

  for (std::size_t comp_idx = 0; comp_idx < num_components; ++comp_idx) {
    if (fmt_size != comp_sizes[comp_idx]) { continue; }
    auto const* ptr = components[comp_idx];
    bool equal      = true;
    for (cudf::size_type idx = 0; idx < fmt_size; ++idx) {
      if (to_upper(fmt_data[idx]) != ptr[idx]) {
        equal = false;
        break;
      }
    }
    if (equal) { return static_cast<truncation_format>(comp_idx); }
  }
  return truncation_format::INVALID;
}

// Truncate the given month to the first month of the quarter.
__device__ inline uint32_t trunc_quarter_month(uint32_t month)
{
  auto const zero_based_month = month - 1u;
  return (zero_based_month / 3u) * 3u + 1u;
}

// Truncate the given day to the previous Monday.
__device__ inline cuda::std::chrono::sys_days trunc_to_monday(
  cuda::std::chrono::sys_days const days_since_epoch)
{
  // Define our `MONDAY` constant as `cuda::std::chrono::Monday` is not available in device code.
  // [0, 6] => [Sun, Sat]
  auto constexpr MONDAY       = cuda::std::chrono::weekday{1};
  auto const weekday          = cuda::std::chrono::weekday{days_since_epoch};
  auto const days_to_subtract = weekday - MONDAY;  // [-1, 5]

  if (days_to_subtract.count() == 0) { return days_since_epoch; }

  // If the input is a Sunday (weekday == 0), we have `days_to_subtract` negative thus
  // we need to subtract 6 days to get the previous Monday.
  return days_to_subtract.count() > 0 ? days_since_epoch - days_to_subtract
                                      : days_since_epoch - cuda::std::chrono::days{6};
}

template <typename Timestamp>
__device__ inline cuda::std::optional<Timestamp> trunc_date(
  cuda::std::chrono::sys_days const days_since_epoch,
  cuda::std::chrono::year_month_day const ymd,
  truncation_format const fmt)
{
  using namespace cuda::std::chrono;
  switch (fmt) {
    case truncation_format::YEAR:
    case truncation_format::YYYY:
    case truncation_format::YY:
      return Timestamp{sys_days{year_month_day{ymd.year(), month{1}, day{1}}}};
    case truncation_format::QUARTER:
      return Timestamp{sys_days{year_month_day{
        ymd.year(), month{trunc_quarter_month(static_cast<uint32_t>(ymd.month()))}, day{1}}}};
    case truncation_format::MONTH:
    case truncation_format::MM:
    case truncation_format::MON:
      return Timestamp{sys_days{year_month_day{ymd.year(), ymd.month(), day{1}}}};
    case truncation_format::WEEK: return Timestamp{trunc_to_monday(days_since_epoch)};
    default: return cuda::std::nullopt;
  }
}

// FormatDeviceT is either a `column_device_view` or `truncation_format`.
template <typename FormatDeviceT>
struct truncate_date_fn {
  using Timestamp = cudf::timestamp_D;
  static_assert(cuda::std::is_same_v<FormatDeviceT, cudf::column_device_view> ||
                  cuda::std::is_same_v<FormatDeviceT, truncation_format>,
                "FormatDeviceT must be either 'cudf::column_device_view' or 'truncation_format'.");

  cudf::column_device_view datetime;
  FormatDeviceT format;

  __device__ inline cuda::std::pair<Timestamp, bool> operator()(cudf::size_type const idx) const
  {
    auto const datetime_idx = datetime.size() > 1 ? idx : 0;
    if (datetime.is_null(datetime_idx)) { return {Timestamp{}, false}; }
    if constexpr (cuda::std::is_same_v<FormatDeviceT, cudf::column_device_view>) {
      if (format.is_null(idx)) { return {Timestamp{}, false}; }
    }

    truncation_format fmt{};
    if constexpr (cuda::std::is_same_v<FormatDeviceT, cudf::column_device_view>) {
      auto const fmt_str = format.template element<cudf::string_view>(idx);
      fmt                = parse_format(fmt_str.data(), fmt_str.size_bytes());
    } else {
      fmt = format;
    }
    if (fmt == truncation_format::INVALID) { return {Timestamp{}, false}; }

    using namespace cuda::std::chrono;
    auto const ts               = datetime.template element<Timestamp>(datetime_idx);
    auto const days_since_epoch = floor<days>(ts);
    auto const ymd              = year_month_day(days_since_epoch);

    auto const result = trunc_date<Timestamp>(days_since_epoch, ymd, fmt);
    return {result.value_or(Timestamp{}), result.has_value()};
  }
};

// FormatDeviceT is either a `column_device_view` or `truncation_format`.
template <typename FormatDeviceT>
struct truncate_timestamp_fn {
  using Timestamp = cudf::timestamp_us;
  static_assert(cuda::std::is_same_v<FormatDeviceT, cudf::column_device_view> ||
                  cuda::std::is_same_v<FormatDeviceT, truncation_format>,
                "FormatDeviceT must be either 'cudf::column_device_view' or 'truncation_format'.");

  cudf::column_device_view datetime;
  FormatDeviceT format;

  __device__ inline cuda::std::pair<Timestamp, bool> operator()(cudf::size_type const idx) const
  {
    auto const datetime_idx = datetime.size() > 1 ? idx : 0;
    if (datetime.is_null(datetime_idx)) { return {Timestamp{}, false}; }
    if constexpr (cuda::std::is_same_v<FormatDeviceT, cudf::column_device_view>) {
      if (format.is_null(idx)) { return {Timestamp{}, false}; }
    }

    truncation_format fmt{};
    if constexpr (cuda::std::is_same_v<FormatDeviceT, cudf::column_device_view>) {
      auto const fmt_str = format.template element<cudf::string_view>(idx);
      fmt                = parse_format(fmt_str.data(), fmt_str.size_bytes());
    } else {
      fmt = format;
    }
    if (fmt == truncation_format::INVALID) { return {Timestamp{}, false}; }

    auto const ts = datetime.template element<Timestamp>(datetime_idx);

    // No truncation needed for microsecond timestamps.
    if (fmt == truncation_format::MICROSECOND) { return {ts, true}; }

    // The components that are common for both date and timestamp: year, quarter, month, week.
    using namespace cuda::std::chrono;
    auto const days_since_epoch = floor<days>(ts);
    auto const ymd              = year_month_day(days_since_epoch);
    if (auto const try_trunc_date = trunc_date<Timestamp>(days_since_epoch, ymd, fmt);
        try_trunc_date.has_value()) {
      return {try_trunc_date.value(), true};
    }

    auto time_since_midnight = ts - days_since_epoch;
    if (time_since_midnight.count() < 0) { time_since_midnight += days(1); }

    switch (fmt) {
      case truncation_format::DAY:
      case truncation_format::DD: return {Timestamp{sys_days{ymd}}, true};
      case truncation_format::HOUR:
        return {Timestamp{sys_days{ymd} + floor<hours>(time_since_midnight)}, true};
      case truncation_format::MINUTE:
        return {Timestamp{sys_days{ymd} + floor<minutes>(time_since_midnight)}, true};
      case truncation_format::SECOND:
        return {Timestamp{sys_days{ymd} + floor<seconds>(time_since_midnight)}, true};
      case truncation_format::MILLISECOND:
        return {Timestamp{sys_days{ymd} + floor<milliseconds>(time_since_midnight)}, true};
      default: CUDF_UNREACHABLE("Unhandled truncation format.");
    }
  }
};

template <typename Timestamp, typename FormatT>
std::unique_ptr<cudf::column> truncate_datetime(cudf::column_view const& datetime,
                                                FormatT const& format,
                                                cudf::size_type output_size,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  auto output = cudf::make_fixed_width_column(
    datetime.type(), output_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto validity = rmm::device_uvector<bool>(output_size, stream);

  auto const input_it = thrust::make_counting_iterator(0);
  auto const output_it =
    thrust::make_zip_iterator(output->mutable_view().template begin<Timestamp>(), validity.begin());
  auto const do_transform = [&](auto trunc_fn) {
    thrust::transform(rmm::exec_policy_nosync(stream),
                      input_it,
                      input_it + output_size,
                      output_it,
                      std::move(trunc_fn));
  };

  using FormatDeviceT = std::conditional_t<std::is_same_v<FormatT, cudf::column_view>,
                                           cudf::column_device_view,
                                           truncation_format>;
  using TransformFunc = std::conditional_t<std::is_same_v<Timestamp, cudf::timestamp_D>,
                                           truncate_date_fn<FormatDeviceT>,
                                           truncate_timestamp_fn<FormatDeviceT>>;

  auto const d_datetime_ptr = cudf::column_device_view::create(datetime, stream);
  if constexpr (std::is_same_v<FormatT, cudf::column_view>) {
    auto const d_format_ptr = cudf::column_device_view::create(format, stream);
    do_transform(TransformFunc{*d_datetime_ptr, *d_format_ptr});
  } else {
    auto const fmt = parse_format(format.data(), static_cast<cudf::size_type>(format.size()));
    if (fmt == truncation_format::INVALID) {
      return cudf::make_fixed_width_column(
        datetime.type(), output_size, cudf::mask_state::ALL_NULL, stream, mr);
    }
    do_transform(TransformFunc{*d_datetime_ptr, fmt});
  }

  auto [null_mask, null_count] =
    cudf::bools_to_mask(cudf::device_span<bool const>(validity), stream, mr);
  output->set_null_mask(
    null_count > 0 ? std::move(*null_mask.release()) : rmm::device_buffer{0, stream, mr},
    null_count);
  return output;
}

template <typename FormatT>
std::unique_ptr<cudf::column> truncate_dispatcher(cudf::column_view const& datetime,
                                                  FormatT const& format,
                                                  cudf::size_type output_size,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  if (datetime.type().id() == cudf::type_id::TIMESTAMP_DAYS) {
    return truncate_datetime<cudf::timestamp_D, FormatT>(datetime, format, output_size, stream, mr);
  }
  return truncate_datetime<cudf::timestamp_us, FormatT>(datetime, format, output_size, stream, mr);
}

void check_type(cudf::column_view const& datetime)
{
  CUDF_EXPECTS(datetime.type().id() == cudf::type_id::TIMESTAMP_DAYS ||
                 datetime.type().id() == cudf::type_id::TIMESTAMP_MICROSECONDS,
               "The date/time input must be either day or microsecond timestamps.");
}

void check_types(cudf::column_view const& datetime, cudf::column_view const& format)
{
  check_type(datetime);
  CUDF_EXPECTS(format.type().id() == cudf::type_id::STRING,
               "The format input must be of string type.");
}

}  // namespace

std::unique_ptr<cudf::column> truncate(cudf::column_view const& datetime,
                                       cudf::column_view const& format,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  check_types(datetime, format);
  CUDF_EXPECTS(datetime.size() == 1 || datetime.size() == format.size(),
               "The input date/time column must have exactly one row or the same number of rows as "
               "the format column.");
  auto const size = format.size();
  if (size == 0 || datetime.size() == datetime.null_count() ||
      format.size() == format.null_count()) {
    return cudf::make_fixed_width_column(
      datetime.type(), size, cudf::mask_state::ALL_NULL, stream, mr);
  }

  return truncate_dispatcher(datetime, format, size, stream, mr);
}

std::unique_ptr<cudf::column> truncate(cudf::column_view const& datetime,
                                       std::string const& format,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  check_type(datetime);

  auto const size = datetime.size();
  if (datetime.size() == 0 || datetime.size() == datetime.null_count()) {
    return cudf::make_fixed_width_column(
      datetime.type(), size, cudf::mask_state::ALL_NULL, stream, mr);
  }

  return truncate_dispatcher(datetime, format, size, stream, mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> truncate(cudf::column_view const& datetime,
                                       cudf::column_view const& format,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return detail::truncate(datetime, format, stream, mr);
}

std::unique_ptr<cudf::column> truncate(cudf::column_view const& datetime,
                                       std::string const& format,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return detail::truncate(datetime, format, stream, mr);
}

}  // namespace spark_rapids_jni
