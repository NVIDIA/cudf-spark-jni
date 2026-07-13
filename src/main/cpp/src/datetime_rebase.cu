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

#include "datetime_utils.hpp"
#include "nvtx_ranges.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>

#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace {

// Convert a date in Julian calendar to the number of days since epoch.
// Follow the implementation of `days_from_julian` from
// https://howardhinnant.github.io/date_algorithms.html
__device__ __inline__ auto days_from_julian(cuda::std::chrono::year_month_day const& ymd)
{
  auto const month = static_cast<uint32_t>(ymd.month());
  auto const day   = static_cast<uint32_t>(ymd.day());
  auto const year  = static_cast<int32_t>(ymd.year()) - (month <= 2);

  int32_t const era          = (year >= 0 ? year : year - 3) / 4;
  uint32_t const year_of_era = static_cast<uint32_t>(year - era * 4);  // [0, 3]
  uint32_t const day_of_year =
    (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;   // [0, 365]
  uint32_t const day_of_era = year_of_era * 365 + day_of_year;  // [0, 1460]
  return era * 1461 + static_cast<int32_t>(day_of_era) - 719470;
}

// Convert the given number of days since the epoch day 1970-01-01 to a local date in Proleptic
// Gregorian calendar, reinterpreting the result as in Julian calendar, then compute the number of
// days since the epoch from that Julian local date.
// This is to match with Apache Spark's `localRebaseGregorianToJulianDays` function.
std::unique_ptr<cudf::column> gregorian_to_julian_days(cudf::column_view const& input,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::TIMESTAMP_DAYS,
               "The input column type must be microsecond timestamp.",
               std::invalid_argument);

  auto output = cudf::make_timestamp_column(input.type(),
                                            input.size(),
                                            cudf::copy_bitmask(input, stream, mr),
                                            input.null_count(),
                                            stream,
                                            mr);

  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(input.size()),
    output->mutable_view().begin<cudf::timestamp_D>(),
    cuda::proclaim_return_type<cudf::timestamp_D>(
      [d_input = input.begin<cudf::timestamp_D>()] __device__(auto const idx) {
        auto constexpr julian_end = cuda::std::chrono::year_month_day{
          cuda::std::chrono::year{1582}, cuda::std::chrono::month{10}, cuda::std::chrono::day{4}};
        auto constexpr gregorian_start = cuda::std::chrono::year_month_day{
          cuda::std::chrono::year{1582}, cuda::std::chrono::month{10}, cuda::std::chrono::day{15}};

        auto const days_ts          = d_input[idx].time_since_epoch().count();
        auto const days_since_epoch = cuda::std::chrono::sys_days(cudf::duration_D{days_ts});

        // Convert the input into local date in Proleptic Gregorian calendar.
        auto const ymd = cuda::std::chrono::year_month_day(days_since_epoch);
        if (ymd > julian_end && ymd < gregorian_start) {
          // This is the same as rebasing from the local date given at `gregorian_start`.
          return cudf::timestamp_D{cudf::duration_D{-141427}};
        }

        // No change since this time.
        if (ymd >= gregorian_start) { return d_input[idx]; }

        // Reinterpret year/month/day as in Julian calendar then compute the days since epoch.
        return cudf::timestamp_D{cudf::duration_D{days_from_julian(ymd)}};
      }));

  return output;
}

// Convert a number of Julian days since epoch to local date in Julian calendar.
// Follow the implementation of `julian_from_days` from
// https://howardhinnant.github.io/date_algorithms.html
__device__ __inline__ cuda::std::chrono::year_month_day julian_from_days(int32_t days)
{
  auto const z                = days + 719470;
  int32_t const era           = (z >= 0 ? z : z - 1460) / 1461;
  uint32_t const day_of_era   = static_cast<uint32_t>(z - era * 1461);   // [0, 1460]
  uint32_t const year_of_era  = (day_of_era - day_of_era / 1460) / 365;  // [0, 3]
  int32_t const year          = static_cast<int32_t>(year_of_era) + era * 4;
  uint32_t const day_of_year  = day_of_era - 365 * year_of_era;        // [0, 365]
  uint32_t const mp           = (5 * day_of_year + 2) / 153;           // [0, 11]
  uint32_t const month        = mp + (mp < 10 ? 3 : -9);               // [1, 12]
  uint32_t const day_of_month = day_of_year - (153 * mp + 2) / 5 + 1;  // [1, 31]

  return cuda::std::chrono::year_month_day{cuda::std::chrono::year{year + (month <= 2)},
                                           cuda::std::chrono::month{month},
                                           cuda::std::chrono::day{day_of_month}};
}

// Convert the given number of days since the epoch day 1970-01-01 to a local date in Julian
// calendar, reinterpreting the result as in Proleptic Gregorian calendar, then compute the number
// of days since the epoch from that Gregorian local date. This is to match with Apache Spark's
// `localRebaseJulianToGregorianDays` function.
std::unique_ptr<cudf::column> julian_to_gregorian_days(cudf::column_view const& input,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::TIMESTAMP_DAYS,
               "The input column type must be microsecond timestamp.",
               std::invalid_argument);

  auto output = cudf::make_timestamp_column(input.type(),
                                            input.size(),
                                            cudf::copy_bitmask(input, stream, mr),
                                            input.null_count(),
                                            stream,
                                            mr);

  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(input.size()),
                    output->mutable_view().begin<cudf::timestamp_D>(),
                    cuda::proclaim_return_type<cudf::timestamp_D>(
                      [d_input = input.begin<cudf::timestamp_D>()] __device__(auto const idx) {
                        auto const days_ts = d_input[idx].time_since_epoch().count();
                        if (days_ts >= -141427) {  // Gregorian start day
                          return d_input[idx];
                        }

                        // Reinterpret year/month/day as in Gregorian calendar then compute the days
                        // since epoch.
                        auto const ymd = julian_from_days(days_ts);
                        auto const result =
                          cuda::std::chrono::local_days{ymd}.time_since_epoch().count();
                        return cudf::timestamp_D{cudf::duration_D{result}};
                      }));

  return output;
}

/**
 * @brief Struct store results of extracting time components from a timestamp.
 */
struct time_components {
  int32_t hour;
  int32_t minute;
  int32_t second;
  int32_t subsecond;
};

/**
 * @brief Specialized modulo expression that handles negative values.
 *
 * @code{.pseudo}
 * Examples:
 *     modulo(1,60)  ->  1
 *     modulo(-1,60) -> 59
 * @endcode
 */
__device__ __inline__ auto modulo_time(int64_t time, int64_t base)
{ return static_cast<int32_t>(((time % base) + base) % base); }

/**
 * @brief This function handles converting units by dividing and adjusting for negative values.
 *
 * @code{.pseudo}
 * Examples:
 *     scale(-61,60) -> -2
 *     scale(-60,60) -> -1
 *     scale(-59,60) -> -1
 *     scale( 59,60) ->  0
 *     scale( 60,60) ->  1
 *     scale( 61,60) ->  1
 * @endcode
 */
__device__ __inline__ int64_t scale_time(int64_t time, int64_t base)
{ return (time - ((time < 0) * (base - 1L))) / base; }

int64_t constexpr MICROS_PER_SECOND = 1'000'000L;

__device__ __inline__ time_components get_time_components(int64_t micros)
{
  auto const subsecond = modulo_time(micros, MICROS_PER_SECOND);

  // Convert microseconds to seconds.
  micros = micros / MICROS_PER_SECOND - ((micros < 0) && (subsecond != 0));

  auto const hour   = modulo_time(scale_time(micros, 3600), 24);
  auto const minute = modulo_time(scale_time(micros, 60), 60);
  auto const second = modulo_time(micros, 60);

  return time_components{hour, minute, second, subsecond};
}

// Convert the given number of microseconds since the epoch day 1970-01-01T00:00:00Z to a local
// date-time in Proleptic Gregorian calendar, reinterpreting the result as in Julian calendar, then
// compute the number of microseconds since the epoch from that Julian local date-time.
// This is to match with Apache Spark's `rebaseGregorianToJulianMicros` function with timezone
// fixed to UTC.
std::unique_ptr<cudf::column> gregorian_to_julian_micros(cudf::column_view const& input,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::TIMESTAMP_MICROSECONDS,
               "The input column type must be microsecond timestamp.",
               std::invalid_argument);

  auto output = cudf::make_timestamp_column(input.type(),
                                            input.size(),
                                            cudf::copy_bitmask(input, stream, mr),
                                            input.null_count(),
                                            stream,
                                            mr);

  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(input.size()),
    output->mutable_view().begin<cudf::timestamp_us>(),
    cuda::proclaim_return_type<cudf::timestamp_us>(
      [d_input = input.begin<cudf::timestamp_us>()] __device__(auto const idx) {
        // This timestamp corresponds to October 15th, 1582 UTC.
        // After this day, there is no difference in microsecond values between Gregorian
        // and Julian calendars.
        int64_t constexpr last_switch_gregorian_ts = -12219292800000000L;

        auto const micros_ts = d_input[idx].time_since_epoch().count();
        if (micros_ts >= last_switch_gregorian_ts) { return d_input[idx]; }

        // Convert the input into local date-time in Proleptic Gregorian calendar.
        auto const days_since_epoch = cuda::std::chrono::sys_days(static_cast<cudf::duration_D>(
          cuda::std::chrono::floor<cuda::std::chrono::days>(cudf::duration_us(micros_ts))));
        auto const ymd              = cuda::std::chrono::year_month_day(days_since_epoch);
        auto const timeparts        = get_time_components(micros_ts);

        auto constexpr julian_end = cuda::std::chrono::year_month_day{
          cuda::std::chrono::year{1582}, cuda::std::chrono::month{10}, cuda::std::chrono::day{4}};
        auto constexpr gregorian_start = cuda::std::chrono::year_month_day{
          cuda::std::chrono::year{1582}, cuda::std::chrono::month{10}, cuda::std::chrono::day{15}};

        // Reinterpret the local date-time as in Julian calendar and compute microseconds since
        // the epoch from that Julian local date-time.
        // If the input date is outside of both calendars, consider it as it is a local date
        // given at `gregorian_start` (-141427 Julian days since epoch).
        auto const julian_days =
          (ymd > julian_end && ymd < gregorian_start) ? -141427 : days_from_julian(ymd);
        int64_t result = (julian_days * 24L * 3600L) + (timeparts.hour * 3600L) +
                         (timeparts.minute * 60L) + timeparts.second;
        result *= MICROS_PER_SECOND;  // to microseconds
        result += timeparts.subsecond;

        return cudf::timestamp_us{cudf::duration_us{result}};
      }));

  return output;
}

// Convert the given number of microseconds since the epoch day 1970-01-01T00:00:00Z to a local
// date-time in Julian calendar, reinterpreting the result as in Proleptic Gregorian calendar, then
// compute the number of microseconds since the epoch from that Gregorian local date-time.
// This is to match with Apache Spark's `rebaseJulianToGregorianMicros` function with timezone
// fixed to UTC.
std::unique_ptr<cudf::column> julian_to_gregorian_micros(cudf::column_view const& input,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::TIMESTAMP_MICROSECONDS,
               "The input column type must be microsecond timestamp.",
               std::invalid_argument);

  auto output = cudf::make_timestamp_column(input.type(),
                                            input.size(),
                                            cudf::copy_bitmask(input, stream, mr),
                                            input.null_count(),
                                            stream,
                                            mr);

  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(input.size()),
    output->mutable_view().begin<cudf::timestamp_us>(),
    cuda::proclaim_return_type<cudf::timestamp_us>(
      [d_input = input.begin<cudf::timestamp_us>()] __device__(auto const idx) {
        // This timestamp corresponds to October 15th, 1582 UTC.
        // After this day, there is no difference in microsecond values between Gregorian
        // and Julian calendars.
        int64_t constexpr last_switch_gregorian_ts = -12219292800000000L;

        auto const micros_ts = d_input[idx].time_since_epoch().count();
        if (micros_ts >= last_switch_gregorian_ts) { return d_input[idx]; }

        // Convert the input into local date-time in Julian calendar.
        auto const days_since_epoch = cuda::std::chrono::sys_days(static_cast<cudf::duration_D>(
          cuda::std::chrono::floor<cuda::std::chrono::days>(cudf::duration_us(micros_ts))));
        auto const ymd              = julian_from_days(days_since_epoch.time_since_epoch().count());
        auto const timeparts        = get_time_components(micros_ts);

        // Reinterpret the local date-time as in Gregorian calendar and compute microseconds since
        // the epoch from that Gregorian local date-time.
        auto const gregorian_days = cuda::std::chrono::local_days(ymd).time_since_epoch().count();
        int64_t result            = (gregorian_days * 24L * 3600L) + (timeparts.hour * 3600L) +
                                    (timeparts.minute * 60L) + timeparts.second;
        result *= MICROS_PER_SECOND;  // to microseconds
        result += timeparts.subsecond;

        return cudf::timestamp_us{cudf::duration_us{result}};
      }));

  return output;
}

}  // namespace

namespace spark_rapids_jni {

std::unique_ptr<cudf::column> rebase_gregorian_to_julian(cudf::column_view const& input,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  auto const type = input.type().id();
  CUDF_EXPECTS(
    type == cudf::type_id::TIMESTAMP_DAYS || type == cudf::type_id::TIMESTAMP_MICROSECONDS,
    "The input must be either day or microsecond timestamps to rebase.");

  if (input.size() == 0) { return cudf::make_empty_column(input.type()); }
  return type == cudf::type_id::TIMESTAMP_DAYS ? gregorian_to_julian_days(input, stream, mr)
                                               : gregorian_to_julian_micros(input, stream, mr);
}

std::unique_ptr<cudf::column> rebase_julian_to_gregorian(cudf::column_view const& input,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();

  auto const type = input.type().id();
  CUDF_EXPECTS(
    type == cudf::type_id::TIMESTAMP_DAYS || type == cudf::type_id::TIMESTAMP_MICROSECONDS,
    "The input must be either day or microsecond timestamps to rebase.");

  if (input.size() == 0) { return cudf::make_empty_column(input.type()); }
  return type == cudf::type_id::TIMESTAMP_DAYS ? julian_to_gregorian_days(input, stream, mr)
                                               : julian_to_gregorian_micros(input, stream, mr);
}

}  // namespace spark_rapids_jni
