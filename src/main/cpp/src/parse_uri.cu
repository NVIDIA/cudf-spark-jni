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

#include "exception_with_row_index_utilities.hpp"
#include "nvtx_ranges.hpp"
#include "parse_uri.hpp"

#include <cudf/copying.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/convert/convert_urls.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/transform.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/std/optional>
#include <cuda/std/utility>
#include <thrust/tabulate.h>

#include <memory>
#include <optional>

namespace spark_rapids_jni {

using namespace cudf;

namespace detail {

struct uri_parts {
  string_view scheme;
  string_view host;
  string_view authority;
  string_view path;
  string_view fragment;
  string_view query;
  string_view userinfo;
  string_view port;
  string_view opaque;
  uint32_t valid{0};
};

enum class URI_chunks : int8_t {
  PROTOCOL,
  HOST,
  AUTHORITY,
  PATH,
  FRAGMENT,
  QUERY,
  USERINFO,
  PORT,
  OPAQUE
};

enum class chunk_validity : int8_t { VALID, INVALID, FATAL };

namespace {

// Some parsing errors are fatal and some parsing errors simply mean this
// thing doesn't exist or is invalid. For example, just because 280.0.1.16 is
// not a valid IPv4 address simply means if asking for the host the host is null
// but the authority is still 280.0.1.16 and the uri is not considered invalid.
// By contrast, the URI https://[15:6:g:invalid] will not return https for the
// scheme and is considered completely invalid.

__host__ __device__ constexpr bool is_alpha(char c)
{
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

__host__ __device__ constexpr bool is_numeric(char c) { return c >= '0' && c <= '9'; }

__host__ __device__ constexpr bool is_alphanum(char c) { return is_alpha(c) || is_numeric(c); }

__host__ __device__ constexpr bool is_hex(char c)
{
  return is_numeric(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

__device__ cuda::std::pair<bool, string_view::const_iterator> skip_and_validate_special(
  string_view::const_iterator iter,
  string_view::const_iterator end,
  bool allow_invalid_escapes = false)
{
  while (iter != end) {
    auto const c         = *iter;
    auto const num_bytes = cudf::strings::detail::bytes_in_char_utf8(*iter);
    if (*iter == '%' && !allow_invalid_escapes) {
      // verify following two characters are hexadecimal
      for (int i = 0; i < 2; ++i) {
        ++iter;
        if (iter == end) { return {false, iter}; }

        if (!is_hex(*iter)) { return {false, iter}; }
      }
    } else if (num_bytes > 1) {
      // UTF8 validation means it isn't whitespace and not a control character
      // the normal validation will handle anything single byte, this checks for multiple byte
      // whitespace
      auto const c = *iter;
      // There are multi-byte looking things like extended ASCII characters that are not valid UTF8.
      // Check that here.
      if ((c & 0xC0) != 0x80) { return {false, iter}; }
      if (num_bytes > 2 && ((c & 0xC000) != 0x8000)) { return {false, iter}; }
      if (num_bytes > 3 && ((c & 0xC00000) != 0x800000)) { return {false, iter}; }

      // Validate it isn't a whitespace or control unicode character.
      if ((c >= 0xc280 && c <= 0xc2a0) || c == 0xe19a80 || (c >= 0xe28080 && c <= 0xe2808a) ||
          c == 0xe280af || c == 0xe280a8 || c == 0xe2819f || c == 0xe38080) {
        return {false, iter};
      }
    } else {
      break;
    }
    ++iter;
  }

  return {true, iter};
}

template <typename Predicate>
__device__ bool validate_chunk(string_view s, Predicate fn, bool allow_invalid_escapes = false)
{
  auto iter = s.begin();
  {
    auto [valid, iter_] = skip_and_validate_special(iter, s.end(), allow_invalid_escapes);
    iter                = cuda::std::move(iter_);
    if (!valid) { return false; }
  }
  while (iter != s.end()) {
    if (!fn(iter)) { return false; }

    iter++;
    auto [valid, iter_] = skip_and_validate_special(iter, s.end(), allow_invalid_escapes);
    iter                = cuda::std::move(iter_);
    if (!valid) { return false; }
  }
  return true;
}

bool __device__ validate_scheme(string_view scheme)
{
  // A scheme simply needs to be an alpha character followed by alphanumeric
  auto iter = scheme.begin();
  if (!is_alpha(*iter)) { return false; }
  while (++iter != scheme.end()) {
    auto const c = *iter;
    if (!is_alphanum(c) && c != '+' && c != '-' && c != '.') { return false; }
  }
  return true;
}

bool __device__ validate_ipv6(string_view s)
{
  constexpr auto max_colons{8};

  if (s.size_bytes() < 2) { return false; }

  bool found_double_colon{false};
  int open_bracket_count{0};
  int close_bracket_count{0};
  int period_count{0};
  int colon_count{0};
  int percent_count{0};
  char previous_char{0};
  int address{0};
  int address_char_count{0};
  bool address_has_hex{false};

  auto const leading_double_colon = [&]() {
    auto iter = s.begin();
    if (*iter == '[') iter++;
    return *iter++ == ':' && *iter == ':';
  }();

  for (auto iter = s.begin(); iter < s.end(); ++iter) {
    auto const c = *iter;

    switch (c) {
      case '[':
        open_bracket_count++;
        if (open_bracket_count > 1) { return false; }
        break;
      case ']':
        close_bracket_count++;
        if (close_bracket_count > 1) { return false; }
        if ((period_count > 0) && (address_has_hex || address > 255)) { return false; }
        break;
      case ':':
        colon_count++;
        if (previous_char == ':') {
          if (found_double_colon) { return false; }
          found_double_colon = true;
        }
        address            = 0;
        address_has_hex    = false;
        address_char_count = 0;
        if (colon_count > max_colons || (colon_count == max_colons && !found_double_colon)) {
          return false;
        }
        // Periods before a colon don't work, periods can be an IPv4 address after this IPv6 address
        // like [1:2:3:4:5:6:d.d.d.d]
        if (period_count > 0 || percent_count > 0) { return false; }
        break;
      case '.':
        period_count++;
        if (percent_count > 0) { return false; }
        if (period_count > 3) { return false; }
        if (address_has_hex) { return false; }
        if (address > 255) { return false; }
        if (colon_count != 6 && !found_double_colon) { return false; }
        // Special case of ::1:2:3:4:5:d.d.d.d has 7 colons - but spark says this is invalid
        // if (colon_count == max_colons && !leading_double_colon) { return false; }
        if (colon_count >= max_colons) { return false; }
        address            = 0;
        address_has_hex    = false;
        address_char_count = 0;
        break;
      case '%':
        // IPv6 can define a device to use for the routing. This is expressed as '%eth0' at the end
        // of the address.
        percent_count++;
        if (percent_count > 1) { return false; }
        if ((period_count > 0) && (address_has_hex || address > 255)) { return false; }
        address            = 0;
        address_has_hex    = false;
        address_char_count = 0;
        break;
      default:
        // after % all bets are off as the device name can be nearly anything
        if (percent_count == 0) {
          if (address_char_count > 3) { return false; }
          address_char_count++;
          address *= 10;
          if (c >= 'a' && c <= 'f') {
            address += 10;
            address += c - 'a';
            address_has_hex = true;
          } else if (c >= 'A' && c <= 'Z') {
            address += 10;
            address += c - 'A';
            address_has_hex = true;
          } else if (c >= '0' && c <= '9') {
            address += c - '0';
          } else {
            return false;
          }
        }
        break;
    }
    previous_char = c;
  }

  return true;
}

bool __device__ validate_ipv4(string_view s)
{
  // dotted quad (0-255).(0-255).(0-255).(0-255)
  int address            = 0;
  int address_char_count = 0;
  int dot_count          = 0;
  for (auto iter = s.begin(); iter < s.end(); ++iter) {
    auto const c = *iter;

    // can't lead with a .
    if ((c < '0' || c > '9') && (iter == s.begin() || c != '.')) { return false; }

    if (c == '.') {
      // verify we saw at least one character and reset values
      if (address_char_count == 0) { return false; }
      address            = 0;
      address_char_count = 0;
      dot_count++;
      continue;
    }

    address_char_count++;
    address *= 10;
    address += c - '0';

    if (address > 255) { return false; }
  }

  // can't end with a .
  if (address_char_count == 0) { return false; }

  // must be 4 portions seperated by 3 dots.
  if (dot_count != 3) { return false; }

  return true;
}

bool __device__ validate_domain_name(string_view name)
{
  // domain name can be alphanum or -.
  // slash can not be the first of last character of the domain name or around a .
  bool last_was_slash          = false;
  bool last_was_period         = false;
  bool numeric_start           = false;
  int characters_before_period = 0;
  for (auto iter = name.begin(); iter < name.end(); ++iter) {
    auto const c = *iter;
    if (!is_alphanum(c) && c != '-' && c != '.') { return false; }

    // the final section can't start with a digit
    if (last_was_period && c >= '0' && c <= '9') {
      numeric_start = true;
    } else {
      numeric_start = false;
    }

    if (c == '-') {
      if (last_was_period || iter == name.begin() || iter == --name.end()) { return false; }
      last_was_slash  = true;
      last_was_period = false;
    } else if (c == '.') {
      if (last_was_slash || last_was_period || characters_before_period == 0) { return false; }
      last_was_period          = true;
      last_was_slash           = false;
      characters_before_period = 0;
    } else {
      last_was_period = false;
      last_was_slash  = false;
      characters_before_period++;
    }
  }

  // numeric start to last part of domain isn't allowed.
  if (numeric_start) { return false; }

  return true;
}

chunk_validity __device__ validate_host(string_view host)
{
  // This can be IPv4, IPv6, or a domain name.
  if (*host.begin() == '[') {
    // If last character is a ], this is IPv6 or invalid.
    if (*(host.end() - 1) != ']') {
      // invalid
      return chunk_validity::FATAL;
    }
    if (!validate_ipv6(host)) { return chunk_validity::FATAL; }

    return chunk_validity::VALID;
  }

  // If there are more [ or ] characters this is invalid.
  // Also need to find the last .
  int last_open_bracket  = -1;
  int last_close_bracket = -1;
  int last_period        = -1;

  // The original plan on this loop was to get fancy and use a reverse iterator and exit when
  // everything was found, but the expectation is there are no brackets in this string, so we have
  // to traverse the entire thing anyway to verify that. The math is easier with a forward iterator,
  // so we're back here.
  for (auto iter = host.begin(); iter < host.end(); ++iter) {
    auto const c = *iter;
    if (c == '[') {
      last_open_bracket = iter.position();
    } else if (c == ']') {
      last_close_bracket = iter.position();
    } else if (c == '.') {
      last_period = iter.position();
    }
  }

  if (last_open_bracket >= 0 || last_close_bracket >= 0) { return chunk_validity::FATAL; }

  // If we didn't find a period or if the last character is a period or the character after the last
  // period is non numeric
  if (last_period < 0 || last_period == host.length() - 1 || host[last_period + 1] < '0' ||
      host[last_period + 1] > '9') {
    // must be domain name or it is invalid
    if (validate_domain_name(host)) { return chunk_validity::VALID; }

    // the only other option is that this is a IPv4 address
  } else if (validate_ipv4(host)) {
    return chunk_validity::VALID;
  }

  return chunk_validity::INVALID;
}

bool __device__ validate_query(string_view query)
{
  // query can be alphanum and _-!.~'()*,;:$&+=?/[]@"
  return validate_chunk(
    query, cuda::proclaim_return_type<bool>([] __device__(string_view::const_iterator iter) {
      auto const c = *iter;
      if (c != '!' && c != '"' && c != '$' && !(c >= '&' && c <= ';') && c != '=' &&
          !(c >= '?' && c <= ']' && c != '\\') && !(c >= 'a' && c <= 'z') && c != '_' && c != '~') {
        return false;
      }
      return true;
    }));
}

bool __device__ validate_authority(string_view authority, bool allow_invalid_escapes)
{
  // authority needs to be alphanum and @[]_-!.'()*,;:$&+=
  return validate_chunk(authority,
                        cuda::proclaim_return_type<bool>(
                          [allow_invalid_escapes] __device__(string_view::const_iterator iter) {
                            auto const c = *iter;
                            if (c != '!' && c != '$' && !(c >= '&' && c <= ';' && c != '/') &&
                                c != '=' && !(c >= '@' && c <= '_' && c != '^' && c != '\\') &&
                                !(c >= 'a' && c <= 'z') && c != '~' &&
                                (!allow_invalid_escapes || c != '%')) {
                              return false;
                            }
                            return true;
                          }),
                        allow_invalid_escapes);
}

bool __device__ validate_userinfo(string_view userinfo)
{
  // can't be ] or [ in here
  return validate_chunk(
    userinfo, cuda::proclaim_return_type<bool>([] __device__(string_view::const_iterator iter) {
      auto const c = *iter;
      if (c == '[' || c == ']') { return false; }
      return true;
    }));
}

bool __device__ validate_port(string_view port)
{
  // port is positive numeric >=0 according to spark...shrug
  return validate_chunk(
    port, cuda::proclaim_return_type<bool>([] __device__(string_view::const_iterator iter) {
      auto const c = *iter;
      if (c < '0' && c > '9') { return false; }
      return true;
    }));
}

bool __device__ validate_path(string_view path)
{
  // path can be alphanum and @[]_-!.~'()*?/&,;:$+=
  return validate_chunk(
    path, cuda::proclaim_return_type<bool>([] __device__(string_view::const_iterator iter) {
      auto const c = *iter;
      if (c != '!' && c != '$' && !(c >= '&' && c <= ';') && c != '=' && !(c >= '@' && c <= 'Z') &&
          c != '_' && !(c >= 'a' && c <= 'z') && c != '~') {
        return false;
      }
      return true;
    }));
}

bool __device__ validate_opaque(string_view opaque)
{
  // opaque can be alphanum and @[]_-!.~'()*?/,;:$@+=
  return validate_chunk(
    opaque, cuda::proclaim_return_type<bool>([] __device__(string_view::const_iterator iter) {
      auto const c = *iter;
      if (c != '!' && c != '$' && !(c >= '&' && c <= ';') && c != '=' &&
          !(c >= '?' && c <= ']' && c != '\\') && c != '_' && c != '~' && !(c >= 'a' && c <= 'z')) {
        return false;
      }
      return true;
    }));
}

bool __device__ validate_fragment(string_view fragment)
{
  // fragment can be alphanum and @[]_-!.~'()*?/,;:$&+=
  return validate_chunk(
    fragment, cuda::proclaim_return_type<bool>([] __device__(string_view::const_iterator iter) {
      auto const c = *iter;
      if (c != '!' && c != '$' && !(c >= '&' && c <= ';') && c != '=' &&
          !(c >= '?' && c <= ']' && c != '\\') && c != '_' && c != '~' && !(c >= 'a' && c <= 'z')) {
        return false;
      }
      return true;
    }));
}

__device__ cuda::std::pair<string_view, bool> find_query_part(string_view haystack,
                                                              string_view needle)
{
  auto const n_bytes = needle.size_bytes();
  auto h             = haystack.data();
  auto const h_end   = h + haystack.size_bytes();
  auto n             = needle.data();

  // stop matching early after it can no longer contain the string we are searching for
  while (h + n_bytes < h_end) {
    bool match_needle = true;
    for (size_type jdx = 0; jdx < n_bytes; ++jdx) {
      match_needle = (h[jdx] == n[jdx]);
      if (!match_needle) { break; }
    }

    if (match_needle && h[n_bytes] == '=') {
      // we don't care about the matched part, we want the string data after '='
      h += n_bytes + 1;

      // rest of string until end or until '&' is query match
      int match_len = 0;
      auto start    = h;
      while (h < h_end && *h != '&') {
        match_len++;
        h++;
      }

      return {{start, match_len}, true};
    }

    // not match, skip to the next param if possible, which is after a &.
    while (h + n_bytes < h_end && *h != '&') {
      h++;
    }
    h++;  // skip over the & if has, or point to h_end +1
  }

  return {{}, false};
}

uri_parts __device__ validate_uri(const char* str,
                                  int len,
                                  cuda::std::optional<column_device_view const> query_match,
                                  size_type row_idx)
{
  uri_parts ret;

  auto const original_str = str;
  // look for :/# characters.
  int col      = -1;
  int slash    = -1;
  int hash     = -1;
  int question = -1;
  for (const char* c = str;
       c - str < len && (col == -1 || slash == -1 || hash == -1 || question == -1);
       ++c) {
    switch (*c) {
      case ':':
        if (col == -1) col = c - str;
        break;
      case '/':
        if (slash == -1) slash = c - str;
        break;
      case '#':
        if (hash == -1) hash = c - str;
        break;
      case '?':
        if (question == -1) question = c - str;
        break;
      default: break;
    }
  }

  // anything after the hash is part of the fragment and ignored for this part
  if (hash >= 0) {
    ret.fragment = {str + hash + 1, len - hash - 1};
    if (!validate_fragment(ret.fragment)) {
      ret.valid = 0;
      return ret;
    }
    ret.valid |= (1 << static_cast<int>(URI_chunks::FRAGMENT));

    len = hash;

    if (col > hash) col = -1;
    if (slash > hash) slash = -1;
    if (question > hash) question = -1;
  }

  // if the first ':' is after the other tokens, this doesn't have a scheme or it is invalid
  bool const has_scheme =
    (col != -1) && ((slash == -1) || (col < slash)) && ((hash == -1) || (col < hash));
  if (has_scheme) {
    // we have a scheme up to the :
    ret.scheme = {str, col};
    if (!validate_scheme(ret.scheme)) {
      ret.valid = 0;
      return ret;
    }
    ret.valid |= (1 << static_cast<int>(URI_chunks::PROTOCOL));

    // skip over scheme
    auto const skip = col + 1;
    str += skip;
    len -= skip;
    question -= skip;
    hash -= skip;
    slash -= skip;
  }

  // no more string to parse is generally an error, unless we had no scheme
  if (len <= 0) {
    // If we had a scheme then this is entirely invalid.
    // If no scheme then URI is entirely empty or we only had a fragment
    // This is equivalent to having a path that is present but empty, so mark it ok
    ret.valid = (static_cast<int>(!has_scheme) << static_cast<int>(URI_chunks::PATH));
    return ret;
  }

  // If we have a '/' as the next character or this is still the start of the string, we have a
  // heirarchical uri. If not it is opaque.
  bool const heirarchical = str[0] == '/' || str == original_str;
  if (heirarchical) {
    // a '?' will break this into query and path/authority
    if (question >= 0) {
      ret.query = {str + question + 1, len - question - 1};
      if (!validate_query(ret.query)) {
        ret.valid = 0;
        return ret;
      }

      // Maybe limit the query data if a literal or a column is passed as a filter. This alters the
      // return from the entire query to just a specific parameter. For example, query for the URI
      // http://www.nvidia.com/page?param0=5&param1=2 is param0=5&param1=2, but if the literal is
      // passed as param0, the return would simply be 5.
      if (query_match && query_match->size() > 0) {
        auto const match_idx = row_idx % query_match->size();
        if (query_match->is_null(match_idx)) {
          ret.valid = 0;
          return ret;
        }
        auto in_match = query_match->element<string_view>(match_idx);

        auto const [query, valid] = find_query_part(ret.query, in_match);
        if (!valid) {
          ret.valid = 0;
          return ret;
        }
        ret.query = query;
      }

      ret.valid |= (1 << static_cast<int>(URI_chunks::QUERY));
    }
    auto const path_len = question >= 0 ? question : len;

    if (str[0] == '/' && str[1] == '/') {
      // If we have a '/', we have //authority/path, otherwise we have //authority with no path.
      int next_slash = -1;
      for (int i = 2; i < path_len; ++i) {
        if (str[i] == '/') {
          next_slash = i;
          break;
        }
      }
      ret.authority = {&str[2],
                       next_slash == -1 ? question < 0 ? len - 2 : question - 2 : next_slash - 2};
      if (next_slash > 0) { ret.path = {str + next_slash, path_len - next_slash}; }

      if (ret.authority.size_bytes() > 0) {
        auto ipv6_address = ret.authority.size_bytes() > 2 && *ret.authority.begin() == '[';
        if (!validate_authority(ret.authority, ipv6_address)) {
          ret.valid = 0;
          return ret;
        }
        ret.valid |= (1 << static_cast<int>(URI_chunks::AUTHORITY));

        // Inspect the authority for userinfo, host, and port
        const char* auth   = ret.authority.data();
        auto auth_size     = ret.authority.size_bytes();
        int amp            = -1;
        int closingbracket = -1;
        int last_colon     = -1;
        for (int i = 0; i < auth_size; ++i) {
          switch (auth[i]) {
            case '@':
              if (amp == -1) {
                amp = i;
                if (last_colon > 0) { last_colon = -1; }
                if (closingbracket > 0) { closingbracket = -1; }
              }
              break;
            case ':': last_colon = amp > 0 ? i - amp - 1 : i; break;
            case ']':
              if (closingbracket == -1) closingbracket = amp > 0 ? i - amp : i;
              break;
          }
        }

        if (amp > 0) {
          ret.userinfo = {auth, amp};
          if (!validate_userinfo(ret.userinfo)) {
            ret.valid = 0;
            return ret;
          }
          ret.valid |= (1 << static_cast<int>(URI_chunks::USERINFO));

          // skip over the @
          amp++;

          auth += amp;
          auth_size -= amp;
        }
        if (last_colon > 0 && last_colon > closingbracket) {
          // Found a port, attempt to parse it
          ret.port = {auth + last_colon + 1, auth_size - last_colon - 1};
          if (!validate_port(ret.port)) {
            ret.valid = 0;
            return ret;
          }
          ret.valid |= (1 << static_cast<int>(URI_chunks::PORT));
          ret.host = {auth, last_colon};
        } else {
          ret.host = {auth, auth_size};
        }
        auto host_ret = validate_host(ret.host);
        switch (host_ret) {
          case chunk_validity::FATAL: ret.valid = 0; return ret;
          case chunk_validity::INVALID: ret.host = {}; break;
          case chunk_validity::VALID: ret.valid |= (1 << static_cast<int>(URI_chunks::HOST)); break;
        }
      }
    } else {
      // path with no authority
      ret.path = {str, path_len};
    }

    if (!validate_path(ret.path)) {
      ret.valid = 0;
      return ret;
    }
    ret.valid |= (1 << static_cast<int>(URI_chunks::PATH));
  } else {
    ret.opaque = {str, len};
    if (!validate_opaque(ret.opaque)) {
      ret.valid = 0;
      return ret;
    }
    ret.valid |= (1 << static_cast<int>(URI_chunks::OPAQUE));
  }

  return ret;
}

// A URI is broken into parts or chunks. There are optional chunks and required chunks. A simple URI
// such as `https://www.nvidia.com` is easy to reason about, but it could also be written as
// `www.nvidia.com`, which is still valid. On top of that, there are characters which are allowed in
// certain chunks that are not allowed in others. There have been a multitude of methods attempted
// to get this correct, but at the end of the day, we have to validate the URI completely. This
// means even the simplest task of pulling off every character before the : still requires
// understanding how to validate an ipv6 address. This kernel was originally conceived as a two-pass
// kernel that ran the same code and either filled in offsets or filled in actual data. The problem
// is that to know what characters you need to copy, you need to have parsed the entire string as a
// 2 meg string could have `:/a` at the very end and everything up to that point is protocol or it
// could end in `.com` and now it is a hostname. To prevent the code from parsing it completely for
// length and then parsing it completely to copy the data, we will store off the offset of the
// string of question. The length is already stored in the offset column, so we then have a pointer
// and a number of bytes to copy and the second pass boils down to a series of memcpy calls.

/**
 * @brief Count the number of characters of each string after parsing the protocol.
 *
 * @param in_strings Input string column
 * @param chunk Chunk of URI to return
 * @param base_ptr Pointer to the start of the character data in the strings column
 * @param out_lengths Number of characters in each decode URL
 * @param out_offsets Offsets to the start of the chunks
 * @param out_validity Bitmask of validity data, updated in function
 */
CUDF_KERNEL void parse_uri_char_counter(column_device_view const in_strings,
                                        URI_chunks chunk,
                                        char const* const base_ptr,
                                        size_type* const out_lengths,
                                        size_type* const out_offsets,
                                        bitmask_type* out_validity,
                                        cuda::std::optional<column_device_view const> query_match)
{
  // thread per row
  auto const tid = cudf::detail::grid_1d::global_thread_id();

  for (thread_index_type tidx = tid; tidx < in_strings.size();
       tidx += cudf::detail::grid_1d::grid_stride()) {
    auto const row_idx = static_cast<size_type>(tidx);
    if (in_strings.is_null(row_idx)) {
      out_lengths[row_idx] = 0;
      continue;
    }

    auto const in_string     = in_strings.element<string_view>(row_idx);
    auto const in_chars      = in_string.data();
    auto const string_length = in_string.size_bytes();

    auto const uri = validate_uri(in_chars, string_length, query_match, row_idx);
    if ((uri.valid & (1 << static_cast<int>(chunk))) == 0) {
      out_lengths[row_idx] = 0;
      clear_bit(out_validity, row_idx);
    } else {
      // stash output offsets and lengths for next kernel to do the copy
      switch (chunk) {
        case URI_chunks::PROTOCOL:
          out_lengths[row_idx] = uri.scheme.size_bytes();
          out_offsets[row_idx] = uri.scheme.data() - base_ptr;
          break;
        case URI_chunks::HOST:
          out_lengths[row_idx] = uri.host.size_bytes();
          out_offsets[row_idx] = uri.host.data() - base_ptr;
          break;
        case URI_chunks::AUTHORITY:
          out_lengths[row_idx] = uri.authority.size_bytes();
          out_offsets[row_idx] = uri.authority.data() - base_ptr;
          break;
        case URI_chunks::PATH:
          out_lengths[row_idx] = uri.path.size_bytes();
          out_offsets[row_idx] = uri.path.data() - base_ptr;
          break;
        case URI_chunks::QUERY:
          out_lengths[row_idx] = uri.query.size_bytes();
          out_offsets[row_idx] = uri.query.data() - base_ptr;
          break;
        case URI_chunks::USERINFO:
          out_lengths[row_idx] = uri.userinfo.size_bytes();
          out_offsets[row_idx] = uri.userinfo.data() - base_ptr;
          break;
        case URI_chunks::PORT:
          out_lengths[row_idx] = uri.port.size_bytes();
          out_offsets[row_idx] = uri.port.data() - base_ptr;
          break;
        case URI_chunks::FRAGMENT:
          out_lengths[row_idx] = uri.fragment.size_bytes();
          out_offsets[row_idx] = uri.fragment.data() - base_ptr;
          break;
        case URI_chunks::OPAQUE:
          out_lengths[row_idx] = uri.opaque.size_bytes();
          out_offsets[row_idx] = uri.opaque.data() - base_ptr;
          break;
      }
    }
  }
}

/**
 * @brief Parse protocol and copy from the input string column to the output char buffer.
 *
 * @param in_strings Input string column
 * @param base_ptr Pointer to the start of the character data in the strings column
 * @param src_offsets Offset value of source strings in in_strings
 * @param offsets Offset value of each string associated with `out_chars`
 * @param out_chars Character buffer for the output string column
 */
CUDF_KERNEL void parse_uri(column_device_view const in_strings,
                           char const* const base_ptr,
                           size_type const* const src_offsets,
                           size_type const* const offsets,
                           char* const out_chars)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();

  for (thread_index_type tidx = tid; tidx < in_strings.size();
       tidx += cudf::detail::grid_1d::grid_stride()) {
    auto const row_idx = static_cast<size_type>(tidx);
    auto const len     = offsets[row_idx + 1] - offsets[row_idx];

    if (len > 0) {
      for (int i = 0; i < len; i++) {
        out_chars[offsets[row_idx] + i] = base_ptr[src_offsets[row_idx] + i];
      }
    }
  }
}

}  // namespace

std::unique_ptr<column> parse_uri(strings_column_view const& input,
                                  URI_chunks chunk,
                                  std::optional<strings_column_view const> query_match,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  size_type strings_count = input.size();
  if (strings_count == 0) { return make_empty_column(type_id::STRING); }

  constexpr size_type num_warps_per_threadblock = 4;
  constexpr size_type threadblock_size = num_warps_per_threadblock * cudf::detail::warp_size;
  auto const num_threadblocks =
    std::min(65536, cudf::util::div_rounding_up_unsafe(strings_count, num_warps_per_threadblock));

  auto offset_count    = strings_count + 1;
  auto const d_strings = column_device_view::create(input.parent(), stream);
  auto const d_matches =
    query_match ? column_device_view::create(query_match->parent(), stream)
                : std::unique_ptr<column_device_view, std::function<void(column_device_view*)>>{};

  // build offsets column
  auto offsets_column = make_numeric_column(
    data_type{type_to_id<size_type>()}, offset_count, mask_state::UNALLOCATED, stream, mr);

  // build src offsets buffer
  auto src_offsets = rmm::device_uvector<size_type>(strings_count, stream);

  // copy null mask
  rmm::device_buffer null_mask =
    input.parent().nullable()
      ? cudf::copy_bitmask(input.parent(), stream, mr)
      : cudf::create_null_mask(input.size(), mask_state::ALL_VALID, stream, mr);

  // count number of bytes in each string after parsing and store it in offsets_column
  auto offsets_view         = offsets_column->view();
  auto offsets_mutable_view = offsets_column->mutable_view();
  parse_uri_char_counter<<<num_threadblocks, threadblock_size, 0, stream.value()>>>(
    *d_strings,
    chunk,
    input.chars_begin(stream),
    offsets_mutable_view.begin<size_type>(),
    reinterpret_cast<size_type*>(src_offsets.data()),
    reinterpret_cast<bitmask_type*>(null_mask.data()),
    d_matches ? cuda::std::optional<column_device_view const>{*d_matches} : cuda::std::nullopt);

  // use scan to transform number of bytes into offsets
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         offsets_view.begin<size_type>(),
                         offsets_view.end<size_type>(),
                         offsets_mutable_view.begin<size_type>());

  // copy the total number of characters of all strings combined (last element of the offset column)
  // to the host memory
  using offsets_t = cudf::scalar_type_t<cudf::size_type>;
  auto out_chars_bytes =
    static_cast<offsets_t const&>(*cudf::get_element(offsets_view, offset_count - 1, stream))
      .value(stream);

  // create the chars buffer
  auto d_out_chars = rmm::device_buffer(out_chars_bytes, stream, mr);

  // copy the characters from the input column to the output column
  parse_uri<<<num_threadblocks, threadblock_size, 0, stream.value()>>>(
    *d_strings,
    input.chars_begin(stream),
    reinterpret_cast<size_type*>(src_offsets.data()),
    offsets_column->view().begin<size_type>(),
    static_cast<char*>(d_out_chars.data()));

  auto null_count =
    cudf::null_count(reinterpret_cast<bitmask_type*>(null_mask.data()), 0, strings_count);

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(d_out_chars),
                             null_count,
                             std::move(null_mask));
}

void validate_input_uris(strings_column_view const& input, rmm::cuda_stream_view stream)
{
  if (input.size() == 0) { return; }

  // In ANSI mode, check for completely invalid URIs
  auto const d_strings = column_device_view::create(input.parent(), stream);

  // Create boolean vector: true for valid URIs, false for completely invalid URIs
  auto validity_flags = rmm::device_uvector<bool>(input.size(), stream);

  thrust::tabulate(
    rmm::exec_policy_nosync(stream),
    validity_flags.begin(),
    validity_flags.end(),
    cuda::proclaim_return_type<bool>([input = *d_strings] __device__(cudf::size_type row_idx) {
      // Input nulls remain null
      if (input.is_null(row_idx)) { return false; }

      auto const in_string = input.element<string_view>(row_idx);
      auto const uri =
        validate_uri(in_string.data(), in_string.size_bytes(), cuda::std::nullopt, row_idx);

      return uri.valid != 0;
    }));

  auto [validation_mask, null_count] =
    cudf::bools_to_mask(cudf::device_span<bool const>(validity_flags),
                        stream,
                        rmm::mr::get_current_device_resource_ref());

  // Create validation column for throw_row_error_if_any
  auto validation_column = std::make_unique<column>(
    std::move(validity_flags),
    null_count > 0 ? std::move(*validation_mask.release()) : rmm::device_buffer{0, stream},
    null_count);
  throw_row_error_if_any(input.parent(), validation_column->view(), stream);
}

}  // namespace detail

// external API

std::unique_ptr<column> parse_uri_to_protocol(strings_column_view const& input,
                                              bool ansi_mode,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  if (ansi_mode) { detail::validate_input_uris(input, stream); }
  return detail::parse_uri(input, detail::URI_chunks::PROTOCOL, std::nullopt, stream, mr);
}

std::unique_ptr<column> parse_uri_to_host(strings_column_view const& input,
                                          bool ansi_mode,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  if (ansi_mode) { detail::validate_input_uris(input, stream); }
  return detail::parse_uri(input, detail::URI_chunks::HOST, std::nullopt, stream, mr);
}

std::unique_ptr<column> parse_uri_to_query(strings_column_view const& input,
                                           bool ansi_mode,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  if (ansi_mode) { detail::validate_input_uris(input, stream); }
  return detail::parse_uri(input, detail::URI_chunks::QUERY, std::nullopt, stream, mr);
}

std::unique_ptr<cudf::column> parse_uri_to_query(cudf::strings_column_view const& input,
                                                 std::string const& query_match,
                                                 bool ansi_mode,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  if (ansi_mode) { detail::validate_input_uris(input, stream); }

  // build string_column_view from incoming query_match string
  auto d_scalar = make_string_scalar(query_match, stream);
  auto col      = make_column_from_scalar(*d_scalar, 1);

  return detail::parse_uri(input, detail::URI_chunks::QUERY, strings_column_view(*col), stream, mr);
}

std::unique_ptr<cudf::column> parse_uri_to_query(cudf::strings_column_view const& input,
                                                 cudf::strings_column_view const& query_match,
                                                 bool ansi_mode,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  CUDF_EXPECTS(input.size() == query_match.size(), "Query column must be the same size as input!");

  if (ansi_mode) { detail::validate_input_uris(input, stream); }
  return detail::parse_uri(input, detail::URI_chunks::QUERY, query_match, stream, mr);
}

std::unique_ptr<column> parse_uri_to_path(strings_column_view const& input,
                                          bool ansi_mode,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  if (ansi_mode) { detail::validate_input_uris(input, stream); }
  return detail::parse_uri(input, detail::URI_chunks::PATH, std::nullopt, stream, mr);
}
}  // namespace spark_rapids_jni
