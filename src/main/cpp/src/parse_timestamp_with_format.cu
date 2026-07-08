/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
#include "datetime_utils.cuh"
#include "nvtx_ranges.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

// This kernel mirrors the formatters Spark uses for to_timestamp / unix_timestamp /
// ParseToTimestamp: java.text.SimpleDateFormat for LEGACY, and java.time.DateTimeFormatter with a
// STRICT resolver for CORRECTED. It is NOT a drop-in for the CSV/JSON read path, which uses
// commons-lang3 FastDateFormat and has different whitespace rules — do not reuse it there.
namespace spark_rapids_jni {

namespace {

// ---- Low-level string-parsing primitives. ------------------------------------------------------

// LEGACY parses via java.text.SimpleDateFormat, which skips only ' ' and '\t' before fields — it
// does not trimAll. So the outer trim must match: a leading control byte like '\r'/'\f'/'\v'
// rejects on CPU and must not be treated as whitespace here.
__device__ bool is_whitespace(unsigned char c) { return c == ' ' || c == '\t'; }

// Read between `min_d` and `max_d` digits greedily. Advances pos by the digits read.
// Returns false (and the partial pos advance is irrelevant since walk_tokens aborts on failure).
__device__ bool read_min_max_digits(
  unsigned char const* p, int& pos, int end, int min_d, int max_d, int& v)
{
  v          = 0;
  int digits = 0;
  while (pos < end && digits < max_d) {
    int const c = static_cast<int>(p[pos]) - '0';
    if (c < 0 || c > 9) { break; }
    v = v * 10 + c;
    ++digits;
    ++pos;
  }
  return digits >= min_d;
}

__device__ bool try_parse_char(unsigned char const* p, int& pos, int end, unsigned char c)
{
  if (pos >= end || p[pos] != c) { return false; }
  ++pos;
  return true;
}

// Skip [ \t]* — LEGACY whitespace-before-field fold (see compile_format).
__device__ void skip_ht_whitespace(unsigned char const* p, int& pos, int end)
{
  while (pos < end && (p[pos] == ' ' || p[pos] == '\t')) {
    ++pos;
  }
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
  TOK_DIGITS,           // a = field, b = min_digits, c = max_digits
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
//   - CORRECTED `yyyy/MM/dd` keeps the existing spark-rapids compatibility contract and accepts
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

std::vector<format_token> compile_format(std::string const& fmt, bool legacy)
{
  std::vector<format_token> out;
  size_t const n                                 = fmt.size();
  bool saw_digit_field                           = false;
  bool const corrected_variable_width_slash_date = !legacy && fmt == "yyyy/MM/dd";
  for (size_t i = 0; i < n;) {
    char const c = fmt[i];
    if (std::isalpha(static_cast<unsigned char>(c))) {
      size_t j = i;
      while (j < n && fmt[j] == c) {
        ++j;
      }
      bool const packed = (i > 0 && std::isalpha(static_cast<unsigned char>(fmt[i - 1]))) ||
                          (j < n && std::isalpha(static_cast<unsigned char>(fmt[j])));
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
      uint8_t const run         = static_cast<uint8_t>(j - i);
      bool const variable_width = (legacy && !packed) || corrected_variable_width_slash_date;
      uint8_t const min_d       = (c == 'y') ? run : (variable_width ? 1 : run);
      uint8_t const max_d       = run;
      // Skip [ \t] before each field, except inside a packed run which stays exact-width.
      bool const abuts_prev_field = (i > 0 && std::isalpha(static_cast<unsigned char>(fmt[i - 1])));
      if (legacy && !abuts_prev_field) { out.push_back({TOK_SKIP_HT_WS, 0, 0, 0}); }
      out.push_back({TOK_DIGITS, letter_to_field(c), min_d, max_d});
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

__device__ bool store_field(parsed_dt& d, uint8_t field, int v)
{
  switch (field) {
    case FLD_YEAR: d.year = v; return true;
    case FLD_MONTH: d.month = v; return true;
    case FLD_DAY: d.day = v; return true;
    case FLD_HOUR: d.hour = v; return true;
    case FLD_MINUTE: d.minute = v; return true;
    case FLD_SECOND: d.second = v; return true;
    default: return false;
  }
}

__device__ bool walk_tokens(unsigned char const* p,
                            int pos,
                            int end,
                            format_token const* __restrict__ tokens,
                            int num_tokens,
                            parsed_dt& d)
{
  for (int i = 0; i < num_tokens; ++i) {
    auto const& t = tokens[i];
    bool ok       = true;
    switch (t.kind) {
      case TOK_DIGITS: {
        int v = 0;
        ok    = read_min_max_digits(p, pos, end, t.b, t.c, v);
        if (ok) { ok = store_field(d, t.a, v); }
        break;
      }
      case TOK_LITERAL: ok = try_parse_char(p, pos, end, t.a); break;
      case TOK_SKIP_HT_WS: skip_ht_whitespace(p, pos, end); break;
      case TOK_TRAIL_EOF: ok = (pos == end); break;
      case TOK_TRAIL_NON_DIGIT:
        if (pos < end) {
          unsigned char const c = p[pos];
          ok                    = !(c >= '0' && c <= '9');
        }
        break;
      default: ok = false; break;
    }
    if (!ok) { return false; }
  }
  return true;
}

struct parse_with_format_fn {
  cudf::column_device_view d_strings;
  format_token const* __restrict__ tokens;
  int num_tokens;
  bool legacy;
  bool* validity;
  cudf::timestamp_us* output;

  // Sets the row's null bit. The output value is left uninitialized — under cuDF's null mask
  // convention, masked-off positions are not read by downstream consumers.
  __device__ void set_invalid(cudf::size_type idx) const { validity[idx] = false; }

  __device__ void operator()(cudf::size_type idx) const
  {
    if (d_strings.is_null(idx)) {
      set_invalid(idx);
      return;
    }
    auto const sv  = d_strings.element<cudf::string_view>(idx);
    auto const* p  = reinterpret_cast<unsigned char const*>(sv.data());
    int const size = sv.size_bytes();

    int pos = 0;
    int end = size;

    if (legacy) {
      if (has_leading_newline(p, end)) {
        set_invalid(idx);
        return;
      }
      trim(p, pos, end);
      if (pos >= end) {
        set_invalid(idx);
        return;
      }
    }

    parsed_dt d{};
    if (!walk_tokens(p, pos, end, tokens, num_tokens, d) ||
        !date_time_utils::is_valid_date_for_timestamp(d.year, d.month, d.day) ||
        !date_time_utils::is_valid_time(d.hour, d.minute, d.second, /*us*/ 0)) {
      set_invalid(idx);
      return;
    }

    int64_t const days    = date_time_utils::to_epoch_day(d.year, d.month, d.day);
    int64_t const seconds = days * 86400L + d.hour * 3600L + d.minute * 60L + d.second;
    int64_t result_us     = 0;
    if (overflow_checker::get_timestamp_overflow(seconds, /*us*/ 0, result_us)) {
      set_invalid(idx);
      return;
    }
    validity[idx] = true;
    output[idx]   = cudf::timestamp_us{cudf::duration_us{result_us}};
  }
};

}  // namespace

std::unique_ptr<cudf::column> parse_timestamp_strings_with_format(
  cudf::strings_column_view const& input,
  std::string const& format,
  bool legacy,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  auto const host_tokens = compile_format(format, legacy);
  auto const num_rows    = input.size();
  if (num_rows == 0) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_to_id<cudf::timestamp_us>()});
  }

  rmm::device_uvector<format_token> device_tokens(
    host_tokens.size(), stream, cudf::get_current_device_resource_ref());
  CUDF_CUDA_TRY(cudaMemcpyAsync(device_tokens.data(),
                                host_tokens.data(),
                                sizeof(format_token) * host_tokens.size(),
                                cudaMemcpyHostToDevice,
                                stream.value()));

  auto const d_input = cudf::column_device_view::create(input.parent(), stream);
  auto result = cudf::make_timestamp_column(cudf::data_type{cudf::type_to_id<cudf::timestamp_us>()},
                                            num_rows,
                                            rmm::device_buffer{},
                                            0,
                                            stream,
                                            mr);
  // Every code path in parse_with_format_fn::operator() writes validity[idx], so leaving the
  // buffer uninitialized is safe.
  auto validity =
    rmm::device_uvector<bool>(num_rows, stream, cudf::get_current_device_resource_ref());

  thrust::for_each_n(rmm::exec_policy_nosync(stream),
                     thrust::make_counting_iterator(0),
                     num_rows,
                     parse_with_format_fn{*d_input,
                                          device_tokens.data(),
                                          static_cast<int>(device_tokens.size()),
                                          legacy,
                                          validity.begin(),
                                          result->mutable_view().begin<cudf::timestamp_us>()});

  auto [output_bitmask, null_count] =
    cudf::bools_to_mask(cudf::device_span<bool const>(validity), stream, mr);
  if (null_count) { result->set_null_mask(std::move(*output_bitmask), null_count); }

  return result;
}

}  // namespace spark_rapids_jni
