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
#include <cudf/table/table_device_view.cuh>
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
#include <cuda/buffer>
#include <cuda/std/bit>
#include <cuda/std/limits>
#include <cuda/stream>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
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

// LEGACY parses via java.text.SimpleDateFormat, which skips only ' ' and '\t' before fields — it
// does not trimAll. So the outer trim must match: a leading control byte like '\r'/'\f'/'\v'
// rejects on CPU and must not be treated as whitespace here.
__device__ bool is_whitespace(unsigned char c) { return c == ' ' || c == '\t'; }

// Returns true iff the first non-[ \t] char is '\n'. Mirrors rejectLeadingNewlineThenStrip.
__device__ bool has_leading_newline(unsigned char const* p, int end)
{
  int probe = 0;
  while (probe < end && (p[probe] == ' ' || p[probe] == '\t')) {
    ++probe;
  }
  return probe < end && p[probe] == '\n';
}

// In-place trim: advance pos past leading whitespace, pull end back past trailing.
__device__ void trim(unsigned char const* p, int& pos, int& end)
{
  while (pos < end && is_whitespace(p[pos])) {
    ++pos;
  }
  while (pos < end && is_whitespace(p[end - 1])) {
    --end;
  }
}

// Year/month/day/hour/minute/second; defaults match cuDF strftime missing-field semantics.
struct parsed_dt {
  int year           = 1970;
  int month          = 1;
  int day            = 1;
  int hour           = 0;
  int minute         = 0;
  int second         = 0;
  int microsecond    = 0;
  int offset_seconds = 0;
};

// ---- Token model. ------------------------------------------------------------------------------

// Token kinds used by the per-row walker. Mirrors how Spark's DateTimeFormatter / SimpleDateFormat
// internally model a pattern as a sequence of printer-parser steps: each step consumes a
// well-defined chunk of the input and either succeeds or fails the row.
enum tok_kind : uint8_t {
  TOK_DIGITS,  // a = field, b = min_digits, c = max_digits
  // STRICT DateTimeFormatter year with SignStyle.EXCEEDS_PAD.
  TOK_STRICT_YEAR,      // a = reserved adjacent digits, b = min_digits, c = max_digits
  TOK_FRACTION,         // b = allow empty fraction, c = max digits (1-9), converted to microseconds
  TOK_OFFSET,           // a = X run length (1-3), b = allow optional offset seconds
  TOK_LITERAL,          // a = literal char
  TOK_SKIP_HT_WS,       // skip [ \t]* (legacy whitespace fold)
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
//   - "Packed" runs (a digit field abutting another digit field without a literal between them,
//     e.g. yyyyMMdd) get exact width — otherwise the boundary is ambiguous.
//   - Otherwise CORRECTED uses exact width and LEGACY uses [1, 2].
//   - CORRECTED `yyyy/MM/dd` keeps the existing cudf-spark compatibility contract and accepts
//     1-2 digit month/day fields. This intentionally DEVIATES from Spark CPU, whose STRICT
//     DateTimeFormatter rejects single-digit fields ("2024/5/6" is null on CPU); the GPU
//     over-accepts. Pinned by parseTimestampWithFormat_correctedSlashDateDeviation.
// Literal handling:
//   - A space matches exactly one ' '. Spark rejects 'T' as the separator for a space pattern
//     under both policies (unlike the format-less cast, which accepts 'T').
//   - LEGACY: SimpleDateFormat skips [ \t] before every numeric field, so a TOK_SKIP_HT_WS
//     precedes each non-packed digit field (subsumes the old REMOVE_WHITESPACE_FROM_MONTH_DAY).
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

enum class timestamp_parse_mode { corrected, legacy, exception, checked };

format_dialect dialect_for(timestamp_parse_mode mode)
{
  switch (mode) {
    case timestamp_parse_mode::legacy: return format_dialect::legacy_expression;
    case timestamp_parse_mode::exception: return format_dialect::exception_expression;
    default: return format_dialect::corrected_expression;
  }
}

format_token compile_strict_year(std::string const& format,
                                 size_t next,
                                 uint8_t width,
                                 timestamp_format_parse_options options)
{
  CUDF_EXPECTS(width == 4, "strict year pattern must be yyyy", std::invalid_argument);
  size_t reserved = 0;
  // Reserve following fixed-width numeric fields, as DateTimeFormatter's adjacent-value parser
  // does. This handles packed formats through the same tokens as separated numeric formats.
  // Offsets and variable-width fractions end the adjacent digit run. A single S
  // participates only on JDKs whose formatter supports fixed-width fraction adjacency.
  while (next < format.size() && std::isalpha(static_cast<unsigned char>(format[next])) &&
         format[next] != 'y' && format[next] != 'X') {
    auto const letter = format[next];
    if (letter == 'S' && (!options.adjacent_single_fraction ||
                          (next + 1 < format.size() && format[next + 1] == 'S'))) {
      break;
    }
    do {
      ++reserved;
      ++next;
    } while (next < format.size() && format[next] == letter);
  }
  CUDF_EXPECTS(reserved <= 255, "too many adjacent datetime digits", std::invalid_argument);
  return {TOK_STRICT_YEAR, static_cast<uint8_t>(reserved), 4, 19};
}

format_token compile_extended_timestamp_token(char letter,
                                              uint8_t width,
                                              timestamp_format_parse_options options)
{
  if (letter == 'S') {
    return {TOK_FRACTION, 0, static_cast<uint8_t>(options.allow_empty_fraction), width};
  }
  CUDF_EXPECTS(width <= 3, "offset pattern must be X, XX or XXX", std::invalid_argument);
  return {TOK_OFFSET, width, static_cast<uint8_t>(options.allow_offset_seconds), 0};
}

std::vector<format_token> compile_format(std::string const& fmt,
                                         format_dialect dialect,
                                         timestamp_format_parse_options options = {})
{
  bool const legacy           = dialect == format_dialect::legacy_expression;
  bool const csv_date         = dialect == format_dialect::csv_date;
  bool const strict_corrected = csv_date || dialect == format_dialect::exception_expression;
  std::vector<format_token> out;
  size_t const n       = fmt.size();
  bool saw_digit_field = false;
  bool const corrected_variable_width_slash_date =
    !legacy && !strict_corrected && fmt == "yyyy/MM/dd";
  bool const extended_timestamp = fmt.find_first_of("SX'") != std::string::npos;
  CUDF_EXPECTS(!extended_timestamp || dialect == format_dialect::corrected_expression,
               "fractional seconds, offsets and quoted literals require CORRECTED policy",
               std::invalid_argument);
  for (size_t i = 0; i < n;) {
    char const c = fmt[i];
    if (c == '\'') {
      // Restrict quoted literals to the ISO date/time separator supported by the caller.
      CUDF_EXPECTS(fmt.compare(i, 3, "'T'") == 0,
                   "only the quoted literal 'T' is supported",
                   std::invalid_argument);
      out.push_back({TOK_LITERAL, 'T', 0, 0});
      i += 3;
    } else if (std::isalpha(static_cast<unsigned char>(c))) {
      size_t j = i;
      while (j < n && fmt[j] == c) {
        ++j;
      }
      bool const packed = (i > 0 && std::isalpha(static_cast<unsigned char>(fmt[i - 1]))) ||
                          (j < n && std::isalpha(static_cast<unsigned char>(fmt[j])));
      if (j - i > 9) {
        throw std::invalid_argument(std::string("pattern letter run too long: ") + c);
      }
      if (c == 'S' || c == 'X') {
        out.push_back(compile_extended_timestamp_token(c, static_cast<uint8_t>(j - i), options));
        i = j;
        continue;
      }
      // Non-year fields must have a 2-letter run. JDK uses run length 3+ to mean text forms
      // (e.g. MMM = month name) which this kernel does not implement; silently treating them
      // as N-digit fields would mask caller bugs.
      if (c != 'y' && (j - i) != 2) {
        throw std::invalid_argument(std::string("non-year pattern letter run must be length 2: ") +
                                    c);
      }
      uint8_t const run         = static_cast<uint8_t>(j - i);
      bool const variable_width = (legacy && !packed) || corrected_variable_width_slash_date;
      uint8_t const min_d       = (c == 'y') ? run : (variable_width ? 1 : run);
      uint8_t const max_d       = run;
      // Skip [ \t] before each field, except inside a packed run which stays exact-width.
      bool const abuts_prev_field = (i > 0 && std::isalpha(static_cast<unsigned char>(fmt[i - 1])));
      if (legacy && !abuts_prev_field) { out.push_back({TOK_SKIP_HT_WS, 0, 0, 0}); }
      if ((csv_date || extended_timestamp) && c == 'y') {
        out.push_back(compile_strict_year(fmt, j, run, options));
      } else {
        out.push_back({TOK_DIGITS, letter_to_field(c), min_d, max_d});
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

__device__ bool read_strict_year(date_parse::cursor<>& input, format_token token, int& year)
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

__device__ bool read_offset(date_parse::cursor<>& input,
                            int width,
                            bool allow_seconds,
                            int& offset_seconds)
{
  if (input.consume('Z') || input.consume('z')) {
    offset_seconds = 0;
    return true;
  }
  bool const negative = input.consume('-');
  if (!negative && !input.consume('+')) { return false; }
  int hour   = 0;
  int minute = 0;
  if (input.read_digits(2, hour) != 2) { return false; }
  if (width == 3 && !input.consume(':')) { return false; }
  // X has optional minutes; XX and XXX require them. A single minute digit is
  // invalid.
  if (width != 1 || (input.peek() >= '0' && input.peek() <= '9')) {
    if (input.read_digits(2, minute) != 2) { return false; }
  }
  int second = 0;
  if (allow_seconds && width > 1 &&
      (width == 3 ? input.consume(':') : (input.peek() >= '0' && input.peek() <= '9'))) {
    if (input.read_digits(2, second) != 2) { return false; }
  }
  if (minute > 59 || second > 59 || hour > 18 || (hour == 18 && (minute != 0 || second != 0))) {
    return false;
  }
  offset_seconds = (hour * 3'600 + minute * 60 + second) * (negative ? -1 : 1);
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
  // CSV checks numeric repetitions; the new fraction and offset tokens always reject conflicts.
  uint8_t seen_fields = 0;
  int fraction_nanos  = 0;
  for (int i = 0; i < num_tokens; ++i) {
    auto const& t = tokens[i];
    bool ok       = true;
    switch (t.kind) {
      case TOK_STRICT_YEAR: {
        int year = 0;
        ok       = read_strict_year(input, t, year) &&
             store_field<CheckConflicts>(d, FLD_YEAR, year, seen_fields);
        break;
      }
      case TOK_DIGITS: {
        int v = 0;
        ok    = input.read_digits(t.c, v) >= t.b;
        if (ok) { ok = store_field<CheckConflicts>(d, t.a, v, seen_fields); }
        break;
      }
      case TOK_FRACTION: {
        int fraction = 0;
        int digits   = input.read_digits(t.c, fraction);
        // Older JDKs accept zero digits when another character follows, but still
        // reject an empty fraction at EOF. The following tokens validate that tail.
        ok = digits > 0 || (t.b && !input.eof());
        for (; digits < 9; ++digits) {
          fraction *= 10;
        }
        // Compare before truncation: distinct nanosecond values may have identical microseconds.
        ok = ok && store_field_value<true>(fraction_nanos, fraction, 1 << 6, seen_fields);
        d.microsecond = fraction_nanos / 1'000;
        break;
      }
      case TOK_OFFSET: {
        int offset = 0;
        ok         = read_offset(input, t.a, t.b, offset) &&
             store_field_value<true>(d.offset_seconds, offset, 1 << 7, seen_fields);
        break;
      }
      case TOK_LITERAL: ok = input.consume(t.a) || (t.a == 'T' && input.consume('t')); break;
      case TOK_SKIP_HT_WS:
        while (input.peek() == ' ' || input.peek() == '\t') {
          ++input.pos;
        }
        break;
      case TOK_TRAIL_EOF: ok = input.eof(); break;
      case TOK_TRAIL_NON_DIGIT: ok = !(input.peek() >= '0' && input.peek() <= '9'); break;
      default: ok = false; break;
    }
    if (!ok) { return false; }
  }
  return true;
}

struct timestamp_timezone_view {
  cudf::column_device_view const* timezone_columns;
  cudf::size_type timezone_index;
  int32_t default_offset_seconds;
  bool has_explicit_offset;

  template <bool UseRegion>
  __device__ int64_t to_utc_seconds(int64_t seconds, int32_t explicit_offset_seconds) const
  {
    if constexpr (UseRegion) {
      return convert_timestamp<cudf::timestamp_s>(
               cudf::timestamp_s{cudf::duration_s{seconds}},
               cudf::lists_column_device_view{timezone_columns[0]},
               cudf::lists_column_device_view{timezone_columns[1]},
               timezone_index,
               true)
        .time_since_epoch()
        .count();
    } else {
      return seconds - (has_explicit_offset ? explicit_offset_seconds : default_offset_seconds);
    }
  }
};

class timestamp_timezone_resolver {
 public:
  timestamp_timezone_resolver(cudf::strings_column_view const& input,
                              timestamp_parse_timezone timezone,
                              bool has_explicit_offset,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
    : view_{nullptr, timezone.index, timezone.offset_seconds, has_explicit_offset}
  {
    bool const use_region = timezone.transitions != nullptr && !has_explicit_offset;
    std::vector<cudf::column_view> views{input.parent()};
    if (use_region) {
      CUDF_EXPECTS(timezone.transitions->num_columns() == 2,
                   "timezone table must have two columns");
      CUDF_EXPECTS(timezone.transitions->column(0).type().id() == cudf::type_id::LIST &&
                     timezone.transitions->column(1).type().id() == cudf::type_id::LIST,
                   "timezone table columns must be LIST columns");
      CUDF_EXPECTS(timezone.index >= 0 && timezone.index < timezone.transitions->num_rows(),
                   "timezone index is out of range");
      views.push_back(timezone.transitions->column(0));
      views.push_back(timezone.transitions->column(1));
    }
    // A span permits the input and timezone columns to have different row counts.
    // The synchronous helper completes the single metadata upload before staging is released.
    auto [storage, columns] =
      cudf::create_column_device_views<cudf::column_device_view,
                                       cudf::host_span<cudf::column_view const>>(
        {views.data(), views.size()}, stream, mr);
    storage_ = std::move(storage);
    columns_ = columns;
    if (use_region) { view_.timezone_columns = columns_ + 1; }
  }

  timestamp_timezone_view device_view() const { return view_; }
  cudf::column_device_view const* input_device_view() const { return columns_; }

 private:
  std::unique_ptr<cuda::device_buffer<std::byte>> storage_;
  cudf::column_device_view const* columns_;
  timestamp_timezone_view view_;
};

template <bool UseRegion>
struct parse_with_format_fn {
  cudf::column_device_view const* d_strings;
  cudf::device_span<format_token const> tokens;
  bool legacy;
  cudf::device_span<format_token const> legacy_tokens;
  cudf::size_type* first_exception_row;
  bool* validity;
  cudf::timestamp_us* output;
  int32_t* arithmetic_overflow;
  timestamp_timezone_view timezone;

  // Sets the row's null bit. The output value is left uninitialized — under cuDF's null mask
  // convention, masked-off positions are not read by downstream consumers.
  __device__ void set_invalid(cudf::size_type idx) const { validity[idx] = false; }

  __device__ bool set_overflow() const
  {
    if (arithmetic_overflow != nullptr) {
      cuda::atomic_ref<int32_t, cuda::thread_scope_device>{*arithmetic_overflow}.store(
        1, cuda::memory_order_relaxed);
    }
    return false;
  }

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
      trim(p, pos, end);
      if (pos >= end) { return false; }
    }

    parsed_dt d{};
    if (!walk_tokens(p, pos, end, parse_tokens.data(), static_cast<int>(parse_tokens.size()), d) ||
        d.year < -999'999'999 || d.year > 999'999'999 ||
        !date_time_utils::is_valid_month_day(d.year, d.month, d.day) ||
        !date_time_utils::is_valid_time(d.hour, d.minute, d.second, d.microsecond)) {
      return false;
    }
    // A valid JDK date outside this coarse range cannot be rescued by any UTC offset.
    // Reject it before the timezone helper's epoch-day conversion, which uses int32 days.
    if (d.year < -300'000 || d.year > 300'000) { return set_overflow(); }

    int64_t const days    = date_time_utils::to_epoch_day(d.year, d.month, d.day);
    int64_t const seconds = timezone.to_utc_seconds<UseRegion>(
      days * 86'400L + d.hour * 3'600L + d.minute * 60L + d.second, d.offset_seconds);
    // Match Spark's checked seconds-to-microseconds multiplication before adding the fraction.
    constexpr int64_t micros_per_second = 1'000'000;
    constexpr auto min                  = cuda::std::numeric_limits<int64_t>::min();
    constexpr auto max                  = cuda::std::numeric_limits<int64_t>::max();
    constexpr auto min_seconds          = min / micros_per_second;
    if (seconds < min_seconds || seconds > max / micros_per_second ||
        (seconds == max / micros_per_second && d.microsecond > max % micros_per_second)) {
      return set_overflow();
    }
    int64_t const result_us = seconds * micros_per_second + d.microsecond;
    parsed                  = cudf::timestamp_us{cudf::duration_us{result_us}};
    return true;
  }

  __device__ void operator()(cudf::size_type idx) const
  {
    if (d_strings->is_null(idx)) {
      set_invalid(idx);
      return;
    }
    auto const sv = d_strings->element<cudf::string_view>(idx);
    auto const* p = reinterpret_cast<unsigned char const*>(sv.data());

    cudf::timestamp_us parsed;
    if (parse(p, sv.size_bytes(), tokens, legacy, parsed)) {
      validity[idx] = true;
      output[idx]   = parsed;
      return;
    }

    set_invalid(idx);
    if (first_exception_row != nullptr && parse(p, sv.size_bytes(), legacy_tokens, true, parsed)) {
      auto first_exception_row_ref =
        cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device>{*first_exception_row};
      first_exception_row_ref.fetch_min(idx, cuda::memory_order_relaxed);
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

namespace {

std::unique_ptr<cudf::column> parse_timestamp_strings_with_format_impl(
  cudf::strings_column_view const& input,
  std::string const& format,
  timestamp_parse_mode mode,
  timestamp_format_parse_options options,
  timestamp_parse_timezone timezone,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  bool const legacy           = mode == timestamp_parse_mode::legacy;
  bool const exception_policy = mode == timestamp_parse_mode::exception;
  bool const check_overflow   = mode == timestamp_parse_mode::checked;
  auto host_tokens            = compile_format(format, dialect_for(mode), options);
  auto host_legacy_tokens     = exception_policy
                                  ? compile_format(format, format_dialect::legacy_expression)
                                  : std::vector<format_token>{};
  auto const num_rows         = input.size();
  if (num_rows == 0) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_to_id<cudf::timestamp_us>()});
  }

  auto const temp_mr = cudf::get_current_device_resource_ref();
  bool const has_explicit_offset =
    std::any_of(host_tokens.begin(), host_tokens.end(), [](auto const& token) {
      return token.kind == TOK_OFFSET;
    });
  timestamp_timezone_resolver timezone_resolver{
    input, timezone, has_explicit_offset, stream, temp_mr};
  std::unique_ptr<rmm::device_scalar<int32_t>> arithmetic_overflow;
  if (check_overflow) {
    arithmetic_overflow = std::make_unique<rmm::device_scalar<int32_t>>(stream, temp_mr);
    arithmetic_overflow->set_value_to_zero_async(stream);
  }
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

  std::unique_ptr<rmm::device_scalar<cudf::size_type>> first_exception_row;
  if (exception_policy) {
    first_exception_row =
      std::make_unique<rmm::device_scalar<cudf::size_type>>(num_rows, stream, temp_mr);
  }

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

  auto launch = [&](auto region_tag) {
    thrust::for_each_n(
      rmm::exec_policy_nosync(stream, temp_mr),
      thrust::make_counting_iterator(0),
      num_rows,
      parse_with_format_fn<decltype(region_tag)::value>{
        timezone_resolver.input_device_view(),
        cudf::device_span<format_token const>{device_tokens.data(), device_tokens.size()},
        legacy,
        cudf::device_span<format_token const>{device_legacy_tokens.data(),
                                              device_legacy_tokens.size()},
        first_exception_row ? first_exception_row->data() : nullptr,
        validity.begin(),
        result->mutable_view().begin<cudf::timestamp_us>(),
        arithmetic_overflow ? arithmetic_overflow->data() : nullptr,
        timezone_resolver.device_view()});
  };
  if (timezone_resolver.device_view().timezone_columns != nullptr) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }

  if (first_exception_row) {
    // value(stream) synchronizes the stream before returning the device value to the host.
    auto const row = first_exception_row->value(stream);
    if (row < num_rows) {
      auto const error         = cudf::get_element(input.parent(), row, stream, temp_mr);
      auto const& string_error = static_cast<cudf::string_scalar const&>(*error);
      throw cast_error(row, string_error.to_string(stream));
    }
  }

  auto [output_bitmask, null_count] =
    cudf::bools_to_mask(cudf::device_span<bool const>(validity), stream, mr);
  if (null_count) { result->set_null_mask(std::move(*output_bitmask), null_count); }
  // Overflow always marks a row invalid; a fully valid batch needs no flag readback.
  if (arithmetic_overflow && null_count != 0 && arithmetic_overflow->value(stream) != 0) {
    throw timestamp_parse_overflow{};
  }

  return result;
}

}  // namespace

std::unique_ptr<cudf::column> parse_timestamp_strings_with_format(
  cudf::strings_column_view const& input,
  std::string const& format,
  bool legacy,
  bool exception_policy,
  timestamp_format_parse_options options,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  CUDF_EXPECTS(!(legacy && exception_policy),
               "LEGACY and EXCEPTION policies cannot both be enabled",
               std::invalid_argument);
  auto const mode =
    legacy ? timestamp_parse_mode::legacy
           : (exception_policy ? timestamp_parse_mode::exception : timestamp_parse_mode::corrected);
  return parse_timestamp_strings_with_format_impl(input, format, mode, options, {}, stream, mr);
}

std::unique_ptr<cudf::column> parse_timestamp_strings_with_format(
  cudf::strings_column_view const& input,
  std::string const& format,
  bool legacy,
  bool exception_policy,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  return parse_timestamp_strings_with_format(
    input, format, legacy, exception_policy, {}, stream, mr);
}

std::unique_ptr<cudf::column> parse_timestamp_strings_with_format_checked(
  cudf::strings_column_view const& input,
  std::string const& format,
  timestamp_format_parse_options options,
  timestamp_parse_timezone timezone,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return parse_timestamp_strings_with_format_impl(
    input, format, timestamp_parse_mode::checked, options, timezone, stream, mr);
}

}  // namespace spark_rapids_jni
