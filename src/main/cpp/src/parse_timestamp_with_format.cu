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

#include "cast_string.hpp"
#include "date_parse.cuh"
#include "datetime_utils.cuh"
#include "nvtx_ranges.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/atomic>
#include <cuda/std/bit>
#include <cuda/stream>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// This kernel mirrors the formatters Spark uses for to_timestamp / unix_timestamp /
// ParseToTimestamp: java.text.SimpleDateFormat for LEGACY, and java.time.DateTimeFormatter with a
// STRICT resolver for CORRECTED. CSV dates have a separate entry point below: they require
// strict widths, signed/extended years and date output, without expression whitespace rules.
// The JSON read path has different format and whitespace rules;
// this parser is not a drop-in replacement for it.
namespace spark_rapids_jni {

namespace {

// ---- Low-level string-parsing primitives. ------------------------------------------------------

// LEGACY SimpleDateFormat parses a numeric field as a long and narrows it with Number.intValue(),
// so the value keeps only its low 32 bits; a value beyond the long range becomes a double whose
// intValue() saturates to an invalid field value. Thus `4294969320` reads as 2024, while
// `9223372036854777832` is rejected. Reads one or more digits before `end`.
__device__ bool read_legacy_digits(date_parse::cursor<>& input, int end, int& v)
{
  int64_t acc = 0;
  int digits  = 0;
  while (input.pos < end) {
    int const c = input.peek() - '0';
    if (c < 0 || c > 9) { break; }
    if (acc > (LLONG_MAX - c) / 10) { return false; }
    acc = acc * 10 + c;
    ++digits;
    ++input.pos;
  }
  v = static_cast<int32_t>(static_cast<uint32_t>(acc));
  return digits > 0;
}

// Returns true iff the first non-[ \t] char is '\n'. Mirrors rejectLeadingNewlineThenStrip.
__device__ bool has_leading_newline(unsigned char const* p, int end)
{
  int probe = 0;
  while (probe < end && (p[probe] == ' ' || p[probe] == '\t')) {
    ++probe;
  }
  return probe < end && p[probe] == '\n';
}

// Year/month/day/hour/minute/second; defaults match cuDF strftime missing-field semantics.
struct parsed_dt {
  int year   = 1970;
  int month  = 1;
  int day    = 1;
  int hour   = 0;
  int minute = 0;
  int second = 0;
};

// ---- Token model. ------------------------------------------------------------------------------

// Token kinds used by the per-row walker. Mirrors how Spark's DateTimeFormatter / SimpleDateFormat
// internally model a pattern as a sequence of printer-parser steps: each step consumes a
// well-defined chunk of the input and either succeeds or fails the row.
enum tok_kind : uint8_t {
  TOK_DIGITS,  // a = field, b = min_digits, c = max_digits
  // STRICT DateTimeFormatter year with SignStyle.EXCEEDS_PAD.
  TOK_CSV_YEAR,         // a = reserved adjacent digits, b = min_digits, c = max_digits
  TOK_LEGACY_DIGITS,    // a = field, b = obeyCount width (0 means unbounded)
  TOK_LITERAL,          // a = literal char
  TOK_TRAIL_EOF,        // pos must equal end
  TOK_TRAIL_NON_DIGIT,  // pos == end OR p[pos] is not a digit (legacy tail rule)
};

// `parsed_dt` slot for a digit token to write into.
enum field : uint8_t { FLD_YEAR = 0, FLD_MONTH, FLD_DAY, FLD_HOUR, FLD_MINUTE, FLD_SECOND };

struct format_token {
  uint8_t kind;
  uint8_t a;
  uint8_t b;
  uint8_t c;
};

// ---- Host-side: compile a Spark-style pattern string to a token stream. -------------------------
//
// Pattern letters follow JDK SimpleDateFormat / DateTimeFormatter conventions:
//   y → year (run length determines digit width, e.g. yyyy = 4)
//   M → month (uppercase)        m → minute (lowercase, NOT month)
//   d → day                      H → hour-of-day
//   s → second
// A letter run must have length 2 for non-year fields; year is whatever its run length says.
// Width policy:
//   - CORRECTED uses exact width, including packed runs.
//   - LEGACY mirrors SimpleDateFormat's `obeyCount` rule: a field uses a raw input window of its
//     pattern width when the next field is numeric with no delimiter. Skipped space/tab consumes
//     that window, and the field fails when the window extends past the end of the input.
//     Otherwise it parses one or more digits after skipping space/tab, permitting arbitrary
//     leading zeroes and narrowing the value like Number.intValue().
//     Thus `MMyyyy` reads `12024` as month 12/year 24, while `yyyyMMdd` reads `2024101` as
//     year 2024/month 10/day 1.
//   - Every policy rejects one- and two-letter year patterns. SimpleDateFormat interprets an
//     exactly two-digit input through a moving 80-year window, and DateTimeFormatter reads `yy`
//     as a reduced year based at 2000 and `y` as a variable-width year. This kernel implements
//     neither. The cudf-spark compatibility allowlists use four-letter years.
//   - LEGACY returns null for years past the timestamp range (after 294247), where Spark CPU
//     fails the query with an ArithmeticException from millisToMicros. This is a known
//     deviation; EXCEPTION and CORRECTED match Spark.
//   - LEGACY rejects year 0. Non-lenient SimpleDateFormat parses `y` as year-of-era, whose
//     GregorianCalendar minimum is 1, so `MMyyyy` input `010` is null rather than 0000-01-01.
//   - CORRECTED `yyyy/MM/dd` keeps the existing cudf-spark compatibility contract and accepts
//     1-2 digit month/day fields. This intentionally DEVIATES from Spark CPU, whose STRICT
//     DateTimeFormatter rejects single-digit fields ("2024/5/6" is null on CPU); the GPU
//     over-accepts. Pinned by parseTimestampWithFormat_correctedSlashDateDeviation.
// Literal handling:
//   - A space matches exactly one ' '. Spark rejects 'T' as the separator for a space pattern
//     under both policies (unlike the format-less cast, which accepts 'T').
//   - LEGACY: SimpleDateFormat skips [ \t] before every numeric field, including packed fields.
// The trailing token is TOK_TRAIL_EOF for CORRECTED and TOK_TRAIL_NON_DIGIT for LEGACY.

uint8_t letter_to_field(char c)
{
  switch (c) {
    case 'y': return FLD_YEAR;
    case 'M': return FLD_MONTH;
    case 'd': return FLD_DAY;
    case 'H': return FLD_HOUR;
    case 'm': return FLD_MINUTE;
    case 's': return FLD_SECOND;
    default: throw std::invalid_argument(std::string("unsupported pattern letter: ") + c);
  }
}

enum class format_dialect {
  corrected_expression,
  legacy_expression,
  exception_expression,
  csv_date
};

format_token compile_csv_year(std::string const& format, size_t next, uint8_t width)
{
  CUDF_EXPECTS(width == 4, "CSV date year pattern must be yyyy", std::invalid_argument);
  size_t reserved = 0;
  // Reserve following fixed-width numeric fields, as DateTimeFormatter's adjacent-value parser
  // does. This handles packed formats through the same tokens as separated numeric formats.
  while (next < format.size() && std::isalpha(static_cast<unsigned char>(format[next])) &&
         format[next] != 'y') {
    auto const letter = format[next];
    do {
      ++reserved;
      ++next;
    } while (next < format.size() && format[next] == letter);
  }
  CUDF_EXPECTS(reserved <= 255, "too many adjacent date digits", std::invalid_argument);
  return {TOK_CSV_YEAR, static_cast<uint8_t>(reserved), 4, 19};
}

std::vector<format_token> compile_format(std::string const& fmt, format_dialect dialect)
{
  bool const legacy           = dialect == format_dialect::legacy_expression;
  bool const csv_date         = dialect == format_dialect::csv_date;
  bool const strict_corrected = csv_date || dialect == format_dialect::exception_expression;
  std::vector<format_token> out;
  size_t const n       = fmt.size();
  bool saw_digit_field = false;
  bool const corrected_variable_width_slash_date =
    !legacy && !strict_corrected && fmt == "yyyy/MM/dd";
  for (size_t i = 0; i < n;) {
    char const c = fmt[i];
    if (std::isalpha(static_cast<unsigned char>(c))) {
      size_t j = i;
      while (j < n && fmt[j] == c) {
        ++j;
      }
      if (j - i > 9) {
        throw std::invalid_argument(std::string("pattern letter run too long: ") + c);
      }
      // Non-year fields must have a 2-letter run. JDK uses run length 3+ to mean text forms
      // (e.g. MMM = month name) which this kernel does not implement; silently treating them
      // as N-digit fields would mask caller bugs.
      if (c != 'y' && (j - i) != 2) {
        throw std::invalid_argument(std::string("non-year pattern letter run must be length 2: ") +
                                    c);
      }
      uint8_t const run = static_cast<uint8_t>(j - i);
      if (!csv_date && c == 'y' && run <= 2) {
        throw std::invalid_argument("one- and two-letter year patterns are not supported");
      }
      if (legacy) {
        bool const abuts_next_field = j < n && std::isalpha(static_cast<unsigned char>(fmt[j]));
        out.push_back({TOK_LEGACY_DIGITS,
                       letter_to_field(c),
                       static_cast<uint8_t>(abuts_next_field ? run : 0),
                       0});
      } else if (csv_date && c == 'y') {
        out.push_back(compile_csv_year(fmt, j, run));
      } else {
        bool const variable_width = corrected_variable_width_slash_date && c != 'y';
        out.push_back({TOK_DIGITS, letter_to_field(c), variable_width ? uint8_t{1} : run, run});
      }
      saw_digit_field = true;
      i               = j;
    } else {
      // A space is an ordinary literal (matches one ' '). Reject non-ASCII: it would alias
      // UTF-8 continuation bytes against the input.
      if (static_cast<unsigned char>(c) >= 0x80) {
        throw std::invalid_argument("non-ASCII literal in pattern is not supported");
      }
      out.push_back({TOK_LITERAL, static_cast<uint8_t>(c), 0, 0});
      ++i;
    }
  }
  if (!saw_digit_field) { throw std::invalid_argument("timestamp format has no datetime fields"); }
  out.push_back({legacy ? TOK_TRAIL_NON_DIGIT : TOK_TRAIL_EOF, 0, 0, 0});
  return out;
}

// ---- Device-side: per-row walker over the compiled token stream. --------------------------------

__device__ bool read_csv_year(date_parse::cursor<>& input, format_token token, int& year)
{
  bool const negative = input.consume('-');
  bool const positive = !negative && input.consume('+');
  auto probe          = input;
  int digits          = 0;
  while (digits < token.c + token.a && probe.peek() >= '0' && probe.peek() <= '9') {
    ++digits;
    ++probe.pos;
  }
  digits -= token.a;
  if (digits < token.b || input.read_digits(digits, year) != digits) { return false; }
  // Spark converts yyyy to proleptic uuuu. EXCEEDS_PAD requires '+' beyond four positive
  // digits, forbids it for four digits, and rejects negative zero.
  if (positive ? digits <= token.b : (!negative && digits != token.b)) { return false; }
  if (negative && year == 0) { return false; }
  if (negative) { year = -year; }
  return true;
}

template <bool CheckConflicts>
__device__ bool store_field_value(int& target, int value, uint8_t mask, uint8_t& seen_fields)
{
  if constexpr (CheckConflicts) {
    if ((seen_fields & mask) != 0 && target != value) { return false; }
    seen_fields |= mask;
  }
  target = value;
  return true;
}

template <bool CheckConflicts>
__device__ bool store_field(parsed_dt& d, uint8_t field, int v, uint8_t& seen_fields)
{
  switch (field) {
    case FLD_YEAR: return store_field_value<CheckConflicts>(d.year, v, 1 << FLD_YEAR, seen_fields);
    case FLD_MONTH:
      return store_field_value<CheckConflicts>(d.month, v, 1 << FLD_MONTH, seen_fields);
    case FLD_DAY: return store_field_value<CheckConflicts>(d.day, v, 1 << FLD_DAY, seen_fields);
    case FLD_HOUR: return store_field_value<CheckConflicts>(d.hour, v, 1 << FLD_HOUR, seen_fields);
    case FLD_MINUTE:
      return store_field_value<CheckConflicts>(d.minute, v, 1 << FLD_MINUTE, seen_fields);
    case FLD_SECOND:
      return store_field_value<CheckConflicts>(d.second, v, 1 << FLD_SECOND, seen_fields);
    default: return false;
  }
}

template <bool CheckConflicts = false>
__device__ bool walk_tokens(unsigned char const* p,
                            int pos,
                            int end,
                            format_token const* __restrict__ tokens,
                            int num_tokens,
                            parsed_dt& d)
{
  date_parse::cursor<> input{p, pos, end};
  // CSV rejects conflicting repetitions; expression specializations keep last-value semantics.
  uint8_t seen_fields = 0;
  for (int i = 0; i < num_tokens; ++i) {
    auto const& t = tokens[i];
    bool ok       = true;
    switch (t.kind) {
      case TOK_CSV_YEAR: {
        int year = 0;
        ok       = read_csv_year(input, t, year) &&
             store_field<CheckConflicts>(d, FLD_YEAR, year, seen_fields);
        break;
      }
      case TOK_DIGITS: {
        int v = 0;
        ok    = input.read_digits(t.c, v) >= t.b;
        if (ok) { ok = store_field<CheckConflicts>(d, t.a, v, seen_fields); }
        break;
      }
      case TOK_LEGACY_DIGITS: {
        // SimpleDateFormat skips only ' ' and '\t' before each numeric field. It captures the
        // packed field's input limit before skipping, so skipped characters count against the
        // pattern width, and it fails the field when that limit is past the end of the input.
        // Thus " 12024" with MMyyyy has only one month digit inside the two-character window,
        // and "2 1" with yyyyMM is rejected.
        if (t.b != 0 && input.end - input.pos < t.b) {
          ok = false;
          break;
        }
        int const field_end = t.b == 0 ? input.end : input.pos + t.b;
        while (input.pos < field_end && (input.peek() == ' ' || input.peek() == '\t')) {
          ++input.pos;
        }
        int v = 0;
        ok    = read_legacy_digits(input, field_end, v);
        if (ok) { ok = store_field<CheckConflicts>(d, t.a, v, seen_fields); }
        break;
      }
      case TOK_LITERAL: ok = input.consume(t.a); break;
      case TOK_TRAIL_EOF: ok = input.eof(); break;
      case TOK_TRAIL_NON_DIGIT: ok = !(input.peek() >= '0' && input.peek() <= '9'); break;
      default: ok = false; break;
    }
    if (!ok) { return false; }
  }
  return true;
}

struct parse_with_format_fn {
  cudf::column_device_view d_strings;
  cudf::device_span<format_token const> tokens;
  bool legacy;
  cudf::device_span<format_token const> legacy_tokens;
  cudf::size_type* first_error_row;
  cudf::size_type* first_disagreement_row;
  bool* validity;
  cudf::timestamp_us* output;

  // Sets the row's null bit. The output value is left uninitialized — under cuDF's null mask
  // convention, masked-off positions are not read by downstream consumers.
  __device__ void set_invalid(cudf::size_type idx) const { validity[idx] = false; }

  __device__ bool parse(unsigned char const* p,
                        int size,
                        cudf::device_span<format_token const> parse_tokens,
                        bool parse_legacy,
                        cudf::timestamp_us& parsed) const
  {
    int pos = 0;
    int end = size;

    if (parse_legacy) {
      if (has_leading_newline(p, end)) { return false; }
    }

    parsed_dt d{};
    if (!walk_tokens(p, pos, end, parse_tokens.data(), static_cast<int>(parse_tokens.size()), d) ||
        (parse_legacy && d.year < 1) ||
        !date_time_utils::is_valid_date_for_timestamp(d.year, d.month, d.day) ||
        !date_time_utils::is_valid_time(d.hour, d.minute, d.second, /*us*/ 0)) {
      return false;
    }

    int64_t const days    = date_time_utils::to_epoch_day(d.year, d.month, d.day);
    int64_t const seconds = days * 86400L + d.hour * 3600L + d.minute * 60L + d.second;
    int64_t result_us     = 0;
    if (overflow_checker::get_timestamp_overflow(seconds, /*us*/ 0, result_us)) { return false; }
    parsed = cudf::timestamp_us{cudf::duration_us{result_us}};
    return true;
  }

  __device__ void operator()(cudf::size_type idx) const
  {
    if (d_strings.is_null(idx)) {
      set_invalid(idx);
      return;
    }
    auto const sv = d_strings.element<cudf::string_view>(idx);
    auto const* p = reinterpret_cast<unsigned char const*>(sv.data());

    cudf::timestamp_us parsed;
    if (parse(p, sv.size_bytes(), tokens, legacy, parsed)) {
      validity[idx] = true;
      output[idx]   = parsed;
      return;
    }

    set_invalid(idx);
    bool const parser_disagreement =
      !legacy_tokens.empty() && parse(p, sv.size_bytes(), legacy_tokens, true, parsed);
    if (first_error_row != nullptr) {
      auto first_error_row_ref =
        cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device>{*first_error_row};
      first_error_row_ref.fetch_min(idx, cuda::memory_order_relaxed);
    }
    if (first_disagreement_row != nullptr && parser_disagreement) {
      auto first_disagreement_row_ref =
        cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device>{*first_disagreement_row};
      first_disagreement_row_ref.fetch_min(idx, cuda::memory_order_relaxed);
    }
  }
};

struct csv_date_parser {
  cudf::device_span<format_token const> tokens;
  csv_date_parse_options options;

  __device__ bool operator()(cudf::string_view input, cudf::timestamp_D& result) const
  {
    auto const* p   = cuda::std::bit_cast<unsigned char const*>(input.data());
    auto const size = input.size_bytes();
    if (tokens.empty()) {
      if (date_parse::parse_cast(
            date_parse::cursor<>{p, 0, size}, options.trim_control_characters, result)) {
        return true;
      }
    } else {
      parsed_dt d{};
      if (walk_tokens<true>(p, 0, size, tokens.data(), static_cast<int>(tokens.size()), d) &&
          date_time_utils::is_valid_time(d.hour, d.minute, d.second, 0) &&
          date_parse::to_days(d.year, d.month, d.day, result)) {
        return true;
      }
    }
    return options.enable_fallback &&
           date_parse::parse_cast(
             date_parse::cursor<true>{p, 0, size}, options.trim_control_characters, result);
  }
};

}  // namespace

std::unique_ptr<cudf::column> parse_csv_date_strings(cudf::strings_column_view const& input,
                                                     std::optional<std::string> const& format,
                                                     csv_date_parse_options options,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  auto const host_tokens =
    format ? compile_format(*format, format_dialect::csv_date) : std::vector<format_token>{};
  if (input.size() == 0 || !format) {
    return date_parse::parse_dates(input, csv_date_parser{{}, options}, stream, mr);
  }

  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto staging = cudf::detail::make_pinned_vector_async<format_token>(host_tokens.size(), stream);
  std::copy(host_tokens.begin(), host_tokens.end(), staging.begin());
  auto tokens = cudf::detail::make_device_uvector_async(staging, stream, temp_mr);
  return date_parse::parse_dates(
    input,
    csv_date_parser{cudf::device_span<format_token const>{tokens.data(), tokens.size()}, options},
    stream,
    mr);
}

std::unique_ptr<cudf::column> parse_timestamp_strings_with_format(
  cudf::strings_column_view const& input,
  std::string const& format,
  bool legacy,
  bool exception_policy,
  bool fail_on_error,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  CUDF_EXPECTS(!(legacy && exception_policy),
               "LEGACY and EXCEPTION policies cannot both be enabled",
               std::invalid_argument);
  auto host_tokens =
    compile_format(format,
                   legacy ? format_dialect::legacy_expression
                          : (exception_policy ? format_dialect::exception_expression
                                              : format_dialect::corrected_expression));
  auto host_legacy_tokens = exception_policy
                              ? compile_format(format, format_dialect::legacy_expression)
                              : std::vector<format_token>{};
  auto const num_rows     = input.size();
  if (num_rows == 0) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_to_id<cudf::timestamp_us>()});
  }

  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto host_token_staging =
    cudf::detail::make_pinned_vector_async<format_token>(host_tokens.size(), stream);
  std::copy(host_tokens.begin(), host_tokens.end(), host_token_staging.begin());
  auto device_tokens = cudf::detail::make_device_uvector_async(host_token_staging, stream, temp_mr);

  auto host_legacy_token_staging =
    cudf::detail::make_pinned_vector_async<format_token>(host_legacy_tokens.size(), stream);
  std::copy(
    host_legacy_tokens.begin(), host_legacy_tokens.end(), host_legacy_token_staging.begin());
  auto device_legacy_tokens =
    cudf::detail::make_device_uvector_async(host_legacy_token_staging, stream, temp_mr);

  std::unique_ptr<rmm::device_scalar<cudf::size_type>> first_error_row;
  if (fail_on_error) {
    first_error_row =
      std::make_unique<rmm::device_scalar<cudf::size_type>>(num_rows, stream, temp_mr);
  }
  std::unique_ptr<rmm::device_scalar<cudf::size_type>> first_disagreement_row;
  if (exception_policy) {
    first_disagreement_row =
      std::make_unique<rmm::device_scalar<cudf::size_type>>(num_rows, stream, temp_mr);
  }

  auto const d_input = cudf::column_device_view::create(input.parent(), stream, temp_mr);
  auto result =
    cudf::make_timestamp_column(cudf::data_type{cudf::type_to_id<cudf::timestamp_us>()},
                                num_rows,
                                cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                                0,
                                stream,
                                mr);
  // Every code path in parse_with_format_fn::operator() writes validity[idx], so leaving the
  // buffer uninitialized is safe.
  auto validity = rmm::device_uvector<bool>(num_rows, stream, temp_mr);

  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, temp_mr),
    thrust::make_counting_iterator(0),
    num_rows,
    parse_with_format_fn{
      *d_input,
      cudf::device_span<format_token const>{device_tokens.data(), device_tokens.size()},
      legacy,
      cudf::device_span<format_token const>{device_legacy_tokens.data(),
                                            device_legacy_tokens.size()},
      first_error_row ? first_error_row->data() : nullptr,
      first_disagreement_row ? first_disagreement_row->data() : nullptr,
      validity.begin(),
      result->mutable_view().begin<cudf::timestamp_us>()});

  if (first_error_row || first_disagreement_row) {
    // value(stream) synchronizes the stream before returning the device value to the host.
    auto const first_invalid = first_error_row ? first_error_row->value(stream) : num_rows;
    auto const first_disagreement =
      first_disagreement_row ? first_disagreement_row->value(stream) : num_rows;
    auto const row = std::min(first_invalid, first_disagreement);
    if (row < num_rows) {
      auto const error         = cudf::get_element(input.parent(), row, stream, temp_mr);
      auto const& string_error = static_cast<cudf::string_scalar const&>(*error);
      throw cast_error(row, string_error.to_string(stream), row == first_disagreement);
    }
  }

  auto [output_bitmask, null_count] =
    cudf::bools_to_mask(cudf::device_span<bool const>(validity), stream, mr);
  if (null_count) { result->set_null_mask(std::move(*output_bitmask), null_count); }

  return result;
}

}  // namespace spark_rapids_jni
