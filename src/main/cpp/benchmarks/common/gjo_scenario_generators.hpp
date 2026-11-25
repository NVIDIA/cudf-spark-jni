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

#pragma once

#include <cudf/utilities/error.hpp>

#include <get_json_object.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace spark_rapids_jni {
namespace gjo_scenarios {

using instruction_t = path_instruction_type;
using path_tuple    = std::tuple<instruction_t, std::string, int32_t>;
using path_list     = std::vector<path_tuple>;
using multi_paths   = std::vector<path_list>;

/// @brief Seeded lowercase ASCII letters and digits
inline std::string random_string(std::mt19937_64& rng, std::size_t len)
{
  static constexpr char chars[] = "abcdefghijklmnopqrstuvwxyz0123456789";
  std::uniform_int_distribution<std::size_t> char_dist(0, sizeof(chars) - 2);
  std::string out(len, '\0');
  std::generate(out.begin(), out.end(), [&] { return chars[char_dist(rng)]; });
  return out;
}

/// @brief Pad nonempty rows ending in `}` or `]` with a random string if framing fits
inline std::string pad_row_to(std::mt19937_64& rng, std::string content, std::size_t row_bytes)
{
  if (content.size() >= row_bytes) { return content; }
  char const closer = content.back();
  if (closer != '}' && closer != ']') { return content; }
  // Net framing: 10 bytes for an object member, 3 for an array element.
  std::size_t const fixed = closer == '}' ? 10 : 3;
  std::size_t const gap   = row_bytes - content.size();
  if (gap < fixed) { return content; }
  content.pop_back();
  content += closer == '}' ? ",\"fill\":\"" : ",\"";
  content += random_string(rng, gap - fixed);
  content += closer == '}' ? "\"}" : "\"]";
  return content;
}

// Model separate fused calls: 135 paths over nested events and 33 per explode-fed projection.
// Exclude paths read by from_json or single-path calls. The 135-path table has 13 unqueried
// prefixes and queried depths 1/2/3/5 with counts 35/85/13/2.

// Queried internal nodes return subtrees; leaves hold escaped JSON, strings up to 128 B, or null.
enum class nested_node_kind : int { internal, json, scalar };

struct nested_prefix_entry {
  std::vector<char const*> segments;
  bool queried;
  nested_node_kind kind;
};

inline multi_paths nested_build_paths(std::vector<nested_prefix_entry> const& table,
                                      std::size_t expect)
{
  multi_paths paths;
  paths.reserve(expect);
  for (auto const& e : table) {
    if (!e.queried) { continue; }
    path_list p;
    p.reserve(e.segments.size());
    for (auto* s : e.segments) {
      p.push_back(path_tuple{instruction_t::NAMED, std::string(s), -1});
    }
    paths.push_back(std::move(p));
  }
  CUDF_EXPECTS(paths.size() == expect, "Nested path count drifted");
  return paths;
}

/// @brief Rows from a lexicographic, prefix-closed table (each key opens once)
inline std::vector<std::string> nested_build_rows(std::mt19937_64& rng,
                                                  std::vector<nested_prefix_entry> const& prefixes,
                                                  std::size_t num_rows,
                                                  std::size_t row_bytes,
                                                  std::uint64_t seed)
{
  std::uint64_t const salt = seed * 0x9E3779B97F4A7C15ULL;
  std::vector<std::string> rows;
  rows.reserve(num_rows);
  for (std::size_t r = 0; r < num_rows; ++r) {
    // About 10% of scalar leaves are null, determined by seed, row and leaf.
    auto present = [&](std::size_t idx) {
      return (r * 7919u + idx * 104729u + salt) % 1000u < 900u;
    };
    std::string row;
    row.reserve(row_bytes + 64);
    row += '{';
    // One comma flag per open object, including the root.
    std::vector<bool> need_comma(1, false);
    std::size_t depth = 0;
    std::vector<char const*> prev;
    std::size_t scalar_idx = 0;
    for (auto const& e : prefixes) {
      auto const& segs   = e.segments;
      std::size_t common = 0;
      while (common < prev.size() && common < segs.size() &&
             std::strcmp(prev[common], segs[common]) == 0) {
        ++common;
      }
      while (depth > common) {
        row += '}';
        --depth;
        need_comma.pop_back();
        need_comma[depth] = true;
      }
      for (std::size_t d = common; d < segs.size(); ++d) {
        if (need_comma[depth]) { row += ','; }
        row += '"';
        row += segs[d];
        row += "\":";
        if (d + 1 < segs.size() || e.kind == nested_node_kind::internal) {
          row += '{';
          ++depth;
          need_comma.push_back(false);
        } else if (e.kind == nested_node_kind::json) {
          row += "\"{\\\"sourceRole\\\":\\\"";
          row += segs[d];
          row += "\\\",\\\"eventIdentity\\\":\\\"";
          row += std::to_string(r);
          row += "\\\",\\\"semanticContext\\\":\\\"source-role=";
          row += segs[d];
          row += "|sequence=";
          row += std::to_string((r * 7919u + salt) % 10000000u);
          row += "\\\"}\"";
          need_comma[depth] = true;
        } else if (present(scalar_idx)) {
          std::string s = "source-role=";
          s += segs[d];
          s += "|field=";
          s += segs[d];
          s += "|event=";
          s += std::to_string(1000000000000000ull + (r % 100000000ull));
          s += "|session=";
          s += std::to_string(r % 4093);
          s += "|module=";
          s += std::to_string(r % 101);
          s += "|branch-element=root";
          s += "|sequence=";
          s += std::to_string((r * 7919u + scalar_idx + salt) % 10000000u);
          s += "|tenant=US_REGION_1X|platform=ios|action=moduleView";
          if (s.size() > 128) { s.resize(128); }
          row += '"';
          row += s;
          row += '"';
          ++scalar_idx;
          need_comma[depth] = true;
        } else {
          row += "null";
          ++scalar_idx;
          need_comma[depth] = true;
        }
      }
      prev = segs;
    }
    while (depth > 0) {
      row += '}';
      --depth;
      need_comma.pop_back();
      need_comma[depth] = true;
    }
    row += '}';
    rows.push_back(pad_row_to(rng, std::move(row), row_bytes));
  }
  return rows;
}

/// @brief Shared 135-path table; rank-based keys preserve each segment's byte length
inline std::vector<nested_prefix_entry> const& nested_table_135()
{
  static std::vector<nested_prefix_entry> const table = {
    {{"aafgihrfqbh"}, true, nested_node_kind::scalar},
    {{"abjgticjjopdil"}, true, nested_node_kind::scalar},
    {{"acxavwlzbtnew"}, true, nested_node_kind::scalar},
    {{"adgyggohiw"}, false, nested_node_kind::internal},
    {{"adgyggohiw", "aeczvw"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "afoyosemty"}, true, nested_node_kind::internal},
    {{"adgyggohiw", "afoyosemty", "agmcn"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "ahgrwcmaicsywkwx"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "aiwsflnvv"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "ajotpqbegzid"}, true, nested_node_kind::internal},
    {{"adgyggohiw", "ajotpqbegzid", "aiwsflnvv"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "akkhdsirjocfda"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "altgdggczij"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "bbuksnyw"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "bciukmqn"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "bdacrxgygvbtaz"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "bepiryvpgiqzwldse"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cfgqhto"}, false, nested_node_kind::internal},
    {{"adgyggohiw", "cfgqhto", "amp"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cfgqhto", "anz"}, false, nested_node_kind::internal},
    {{"adgyggohiw", "cfgqhto", "anz", "aolcipo"}, false, nested_node_kind::internal},
    {{"adgyggohiw", "cfgqhto", "anz", "aolcipo", "apgunbkqiy"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cfgqhto", "anz", "aq"}, false, nested_node_kind::internal},
    {{"adgyggohiw", "cfgqhto", "anz", "aq", "aromoqebi"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cgiprir"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "chwfvevdskocgu"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cirvgckjn"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cjrjfbibpfsxga"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "ckbhcdxrosq"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cljqalxdzhfv"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cmevtevgkqkyq"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cnyrkvn"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "coapvewmyqubgszf"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cpgjmxaauaupdwp"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cqbzgwbgpgsmhn"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "crtunkabsyvrlbtlmzeldz"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cslikpdpeejiahlaevujlw"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "ctqmgil"}, false, nested_node_kind::internal},
    {{"adgyggohiw", "ctqmgil", "askqdgkjxrze"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cubgbjlqqvkurd"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cvzmdon"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cwljckhpi"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cxlfomqcmcuvoudhvx"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cymgcvxfu"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "ddnjlviboae"}, false, nested_node_kind::internal},
    {{"adgyggohiw", "ddnjlviboae", "bdacrxgygvbtaz"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "ddnjlviboae", "cvzmdon"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "denheuo"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dfagckqk"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dgooidteicnaljkmf"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dhuen"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "diejobaspfwpybf"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "djwrdsbtlaoqjxnelg"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dkkhsjc"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dlkuac"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dmehkcypbqhvfmkcwq"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dntkebnsdpbifnjnurnjl"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dofbhqupds"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dvxxvufmfj"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dwztmdayxfdjguwcoqdbm"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dxhbpfw"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dywhenbqjsl"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dzlnpg"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "eadl"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "ebjf"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "ecemjuqtmqrzvtt"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "edilzwjmfugbqit"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "eekmxppljdiihjvaurc"}, false, nested_node_kind::internal},
    {{"adgyggohiw", "eekmxppljdiihjvaurc", "atvhvwlwvod"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "eekmxppljdiihjvaurc", "efsohulivlhyhjqetpgy"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "eekmxppljdiihjvaurc", "egvnbetqlqcjvac"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "efsohulivlhyhjqetpgy"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "ehbdquxzhudxbz"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "eizkryrjkce"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "ejbatfwy"}, true, nested_node_kind::internal},
    {{"adgyggohiw", "ejbatfwy", "eqmzthdpc"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "ekurvzowqvdjr"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "elqevfamjescoft"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "emixojpycgbxyjayjc"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "enqeqtytcaqlldpbnzfuyr"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "eonatcvuoszghhtstt"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "epkxhnuczvtaoaqmuu"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "eqmzthdpc"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "erbempgi"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "es"}, false, nested_node_kind::internal},
    {{"adgyggohiw", "es", "au"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "etlz"}, false, nested_node_kind::internal},
    {{"adgyggohiw", "etlz", "au"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "euthhvcrykzhwbs"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "evwucjsreamy"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "ewkhjxhqk"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "exssew"}, false, nested_node_kind::internal},
    {{"adgyggohiw", "exssew", "avjnndufoekzdcfz"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "eywxefgaxdzdnwecjcjnlguxj"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "ezdnsibbcfdgbodphygrndl"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "fahscmwzttf"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "fbncnlgpsr"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "fclodvgtdhzinv"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "fdumzu"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "feewkdtctf"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "ffrueye"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "fghdxmnq"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "fhut"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "fiujuqnlu"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "fjshdnbynle"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "fkaaofpimda"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "fltsezaescgumrh"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "fmrznrhelfsgepnruxyp"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "fnthjdvzlkczbvmzbegs"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "focyhshrnurrf"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "fpy"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "fqqjghiojgxytmengrarcy"}, true, nested_node_kind::scalar},
    {{"awgcqzmrsruwpckxn"}, true, nested_node_kind::scalar},
    {{"axfqwj"}, true, nested_node_kind::scalar},
    {{"ayxejylb"}, true, nested_node_kind::scalar},
    {{"azfkvyhcz"}, true, nested_node_kind::scalar},
    {{"bacozauigouy"}, true, nested_node_kind::scalar},
    {{"bbuksnyw"}, true, nested_node_kind::scalar},
    {{"bfb"}, true, nested_node_kind::scalar},
    {{"bgtdecyq"}, true, nested_node_kind::scalar},
    {{"bhdfokv"}, true, nested_node_kind::scalar},
    {{"bigdzgi"}, true, nested_node_kind::scalar},
    {{"bjuxmny"}, true, nested_node_kind::scalar},
    {{"bkurzedoj"}, true, nested_node_kind::scalar},
    {{"blmxxiioj"}, true, nested_node_kind::scalar},
    {{"bmdvzmv"}, true, nested_node_kind::scalar},
    {{"bnil"}, true, nested_node_kind::scalar},
    {{"boeuczwhngyc"}, true, nested_node_kind::scalar},
    {{"bpxfmyy"}, true, nested_node_kind::scalar},
    {{"bqbdoxllhnq"}, true, nested_node_kind::scalar},
    {{"brxc"}, false, nested_node_kind::internal},
    {{"brxc", "bs"}, true, nested_node_kind::scalar},
    {{"btqqldwd"}, true, nested_node_kind::scalar},
    {{"bucrzjmx"}, true, nested_node_kind::scalar},
    {{"bvurpxiixb"}, true, nested_node_kind::scalar},
    {{"bwqdtl"}, true, nested_node_kind::scalar},
    {{"bxhfeswsjm"}, false, nested_node_kind::internal},
    {{"bxhfeswsjm", "byhdhihpvrqlsjxoqrvo"}, true, nested_node_kind::scalar},
    {{"bzxggjswa"}, true, nested_node_kind::scalar},
    {{"cabacvnrblvghi"}, true, nested_node_kind::scalar},
    {{"ffrueye"}, true, nested_node_kind::scalar},
    {{"frgxladvzhzrw"}, true, nested_node_kind::scalar},
    {{"fsrdxjgx"}, true, nested_node_kind::scalar},
    {{"ftrymqzf"}, true, nested_node_kind::scalar},
    {{"fufljzxhiusloo"}, true, nested_node_kind::scalar},
    {{"fvkjnxeiu"}, true, nested_node_kind::scalar},
    {{"fwlnjcd"}, true, nested_node_kind::scalar},
    {{"fxtzqbc"}, true, nested_node_kind::scalar},
  };
  return table;
}

/// @brief Shared 33-path table with the same order contract as the 135-path table
inline std::vector<nested_prefix_entry> const& nested_table_33()
{
  static std::vector<nested_prefix_entry> const table = {
    {{"adgyggohiw"}, true, nested_node_kind::internal},
    {{"adgyggohiw", "cbgwkjlpkwtskl"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "ccmndeuiiybwsh"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cdhmqxvkmgrbximedqe"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cexmusrjilxavovyen"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "cfgqhto"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "czmmvpazygthexg"}, true, nested_node_kind::json},
    {{"adgyggohiw", "daqkawrwhtpp"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dbmnhompkdcdie"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dcznfidzy"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "ddnjlviboae"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dptlqcqufusrbg"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dqjfarwxykxoemwxeyjl"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dripxcmkpxnped"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dswrdiek"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dtyfxuknrs"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "duinixkwhhuqy"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "dvxxvufmfj"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "fylmuapzzs"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "fztiihvqlst"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "gawmzevfsb"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "gbubcwrpyqrnx"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "gcqilittyb"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "gdasbvkvf"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "gezmvc"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "gflnrykzcdmnch"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "ggdfdjqhivdgk"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "ghwfljupvxcev"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "gimhlmdhjfku"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "gjcxeximrwyevtbphd"}, true, nested_node_kind::scalar},
    {{"adgyggohiw", "gkievumslcv"}, true, nested_node_kind::scalar},
    {{"bhdfokv"}, true, nested_node_kind::scalar},
    {{"glkoq"}, true, nested_node_kind::scalar},
  };
  return table;
}

/// @brief Paths `$.f0`–`$.f14` followed by `$.zz_nested.x`
inline multi_paths flat_mixed_build_paths()
{
  multi_paths paths;
  paths.reserve(16);
  for (int i = 0; i < 15; ++i) {
    paths.push_back(path_list{path_tuple{instruction_t::NAMED, "f" + std::to_string(i), -1}});
  }
  paths.push_back(path_list{path_tuple{instruction_t::NAMED, "zz_nested", -1},
                            path_tuple{instruction_t::NAMED, "x", -1}});
  return paths;
}

/// @brief Up to 15 queried fields, then `zz_nested` if members remain
inline std::vector<std::string> flat_mixed_build_rows(std::mt19937_64& rng,
                                                      std::size_t num_rows,
                                                      std::size_t member_count,
                                                      std::size_t row_bytes,
                                                      std::uint64_t seed)
{
  std::uint64_t const salt = seed * 0x9E3779B97F4A7C15ULL;
  std::vector<std::string> rows;
  rows.reserve(num_rows);
  for (std::size_t r = 0; r < num_rows; ++r) {
    std::string row = "{";
    for (std::size_t m = 0; m < member_count; ++m) {
      if (m > 0) { row += ','; }
      if (m < std::min<std::size_t>(member_count, 15)) {
        row += "\"f" + std::to_string(m) + "\":\"v" + std::to_string((r + salt) % 1000) + "\"";
      } else if (m == member_count - 1) {
        row += "\"zz_nested\":{\"x\":\"n" + std::to_string((r + salt) % 1000) + "\"}";
      } else {
        row += "\"fill_m" + std::to_string(m) + "\":\"" + random_string(rng, 12) + "\"";
      }
    }
    row += '}';
    rows.push_back(pad_row_to(rng, std::move(row), row_bytes));
  }
  return rows;
}

/// @brief 32 child paths per root, ordered by root then child
inline multi_paths flat_mixed_64_build_paths()
{
  constexpr int num_roots      = 2;
  constexpr int paths_per_root = 32;
  multi_paths paths;
  paths.reserve(num_roots * paths_per_root);
  for (int root = 0; root < num_roots; ++root) {
    for (int child = 0; child < paths_per_root; ++child) {
      paths.push_back(path_list{path_tuple{instruction_t::NAMED, "f" + std::to_string(root), -1},
                                path_tuple{instruction_t::NAMED, "c" + std::to_string(child), -1}});
    }
  }
  return paths;
}

/// @brief Single-member paths in ascending numeric key order
inline multi_paths numbered_build_paths(std::size_t count, char prefix)
{
  multi_paths paths;
  paths.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    paths.push_back(path_list{path_tuple{instruction_t::NAMED, prefix + std::to_string(i), -1}});
  }
  return paths;
}

inline multi_paths value_kinds_build_paths() { return numbered_build_paths(16, 'f'); }

/// @brief Cover the escaped-string, raw integer/boolean and subtree writers
inline std::vector<std::string> value_kinds_build_rows(std::mt19937_64& rng,
                                                       std::size_t num_rows,
                                                       std::size_t row_bytes,
                                                       std::uint64_t seed)
{
  std::uint64_t const salt = seed * 0x9E3779B97F4A7C15ULL;
  auto const escaped       = [](std::string v) {
    std::string out;
    for (char c : v) {
      if (c == '"' || c == '\\') { out += '\\'; }
      out += c;
    }
    return out;
  };
  std::vector<std::string> rows;
  rows.reserve(num_rows);
  for (std::size_t r = 0; r < num_rows; ++r) {
    std::string row = "{";
    for (int i = 0; i < 16; ++i) {
      if (i > 0) { row += ','; }
      auto const kind = (r + i) % 4;
      if (kind == 0) {
        row += "\"f" + std::to_string(i) + "\":\"" +
               escaped("q\"s" + std::to_string((r + salt) % 1000)) + "\"";
      } else if (kind == 1) {
        row += "\"f" + std::to_string(i) + "\":" + std::to_string((r + salt) % 100000);
      } else if (kind == 2) {
        row += "\"f" + std::to_string(i) + "\":";
        row += ((r + salt) % 2 == 0) ? "true" : "false";
      } else {
        row += "\"f" + std::to_string(i) + "\":[1," + std::to_string((r + salt) % 100) + "]";
      }
    }
    row += '}';
    rows.push_back(pad_row_to(rng, std::move(row), row_bytes));
  }
  return rows;
}

inline multi_paths flat_wide_124_build_paths() { return numbered_build_paths(124, 'f'); }

/// @brief Depth-two paths with round-robin roots and globally numbered children
inline multi_paths flat_wide_rooted_build_paths(std::size_t num_paths, std::size_t num_roots)
{
  CUDF_EXPECTS(num_roots > 0, "wide few-root shapes need at least one root");
  multi_paths paths;
  paths.reserve(num_paths);
  for (std::size_t p = 0; p < num_paths; ++p) {
    paths.push_back(
      path_list{path_tuple{instruction_t::NAMED, "f" + std::to_string(p % num_roots), -1},
                path_tuple{instruction_t::NAMED, "c" + std::to_string(p), -1}});
  }
  return paths;
}

/// @brief Rooted rows emit children in path order; each path names a root and a child
inline std::vector<std::string> flat_wide_rooted_build_rows(std::mt19937_64& rng,
                                                            std::size_t num_rows,
                                                            std::size_t member_count,
                                                            std::size_t num_roots,
                                                            multi_paths const& paths,
                                                            std::size_t row_bytes)
{
  CUDF_EXPECTS(member_count >= num_roots, "rows must carry every rooted path's parent member");
  // Preserve query order within each root; child names are unique by path index.
  std::vector<std::vector<std::string>> children(num_roots);
  for (auto const& path : paths) {
    for (std::size_t r = 0; r < num_roots; ++r) {
      if (std::get<1>(path[0]) == "f" + std::to_string(r)) {
        children[r].push_back(std::get<1>(path[1]));
        break;
      }
    }
  }
  // One-byte fillers limit overshoot; the 600-member and 768-path rows still exceed 10 KiB.
  std::size_t rooted_bytes = 2;
  for (std::size_t r = 0; r < num_roots; ++r) {
    rooted_bytes += 6 + 2 * std::to_string(r).size() + children[r].size() * 9;
    for (auto const& child : children[r]) {
      rooted_bytes += child.size();
    }
  }
  CUDF_EXPECTS(rooted_bytes < 2 * row_bytes,
               "rooted children exceed the row budget; widen row_bytes or trim members");
  std::vector<std::string> rows;
  rows.reserve(num_rows);
  for (std::size_t r = 0; r < num_rows; ++r) {
    std::string row = "{";
    for (std::size_t m = 0; m < member_count; ++m) {
      if (m > 0) { row += ','; }
      if (m < num_roots) {
        row += "\"f" + std::to_string(m) + "\":{";
        for (std::size_t c = 0; c < children[m].size(); ++c) {
          if (c > 0) { row += ','; }
          row += "\"" + children[m][c] + "\":\"x\"";
        }
        row += '}';
      } else {
        row += "\"fill_m" + std::to_string(m) + "\":\"x\"";
      }
    }
    row += '}';
    rows.push_back(pad_row_to(rng, std::move(row), row_bytes));
  }
  return rows;
}

/// @brief Flat rows with string values at `f0`–`f123`
inline std::vector<std::string> flat_wide_124_build_rows(std::mt19937_64& rng,
                                                         std::size_t num_rows,
                                                         std::size_t row_bytes,
                                                         std::uint64_t seed)
{
  constexpr std::size_t num_members = 124;
  std::uint64_t const salt          = seed * 0x9E3779B97F4A7C15ULL;
  std::vector<std::string> rows;
  rows.reserve(num_rows);
  for (std::size_t r = 0; r < num_rows; ++r) {
    std::string row = "{";
    for (std::size_t m = 0; m < num_members; ++m) {
      if (m > 0) { row += ','; }
      row += "\"f" + std::to_string(m) + "\":\"v" + std::to_string((r + salt) % 1000) + "\"";
    }
    row += '}';
    rows.push_back(pad_row_to(rng, std::move(row), row_bytes));
  }
  return rows;
}

/// @brief 32 children per queried root; a trailing `zz_nested` if `member_count > 2`
inline std::vector<std::string> flat_mixed_64_build_rows(std::mt19937_64& rng,
                                                         std::size_t num_rows,
                                                         std::size_t member_count,
                                                         std::size_t row_bytes,
                                                         std::uint64_t seed)
{
  constexpr std::size_t num_roots      = 2;
  constexpr std::size_t paths_per_root = 32;
  std::uint64_t const salt             = seed * 0x9E3779B97F4A7C15ULL;
  std::vector<std::string> rows;
  rows.reserve(num_rows);
  for (std::size_t r = 0; r < num_rows; ++r) {
    std::string row = "{";
    for (std::size_t m = 0; m < member_count; ++m) {
      if (m > 0) { row += ','; }
      if (m < num_roots) {
        row += "\"f" + std::to_string(m) + "\":{";
        for (std::size_t c = 0; c < paths_per_root; ++c) {
          if (c > 0) { row += ','; }
          row += "\"c" + std::to_string(c) + "\":\"v" + std::to_string((r + salt) % 1000) + "\"";
        }
        row += '}';
      } else if (m == member_count - 1) {
        row += "\"zz_nested\":{\"x\":\"n" + std::to_string((r + salt) % 1000) + "\"}";
      } else {
        row += "\"fill_m" + std::to_string(m) + "\":\"" + random_string(rng, 12) + "\"";
      }
    }
    row += '}';
    rows.push_back(pad_row_to(rng, std::move(row), row_bytes));
  }
  return rows;
}

inline multi_paths flat_600_build_paths() { return numbered_build_paths(15, 'f'); }

/// @brief Flat rows with 15 queried fields; one-byte fillers fit the 600-member budget
inline std::vector<std::string> flat_600_build_rows(std::mt19937_64& rng,
                                                    std::size_t num_rows,
                                                    std::size_t member_count,
                                                    std::size_t row_bytes,
                                                    std::uint64_t seed)
{
  std::uint64_t const salt = seed * 0x9E3779B97F4A7C15ULL;
  std::vector<std::string> rows;
  rows.reserve(num_rows);
  for (std::size_t r = 0; r < num_rows; ++r) {
    std::string row = "{";
    for (std::size_t m = 0; m < member_count; ++m) {
      if (m > 0) { row += ','; }
      if (m < 15) {
        row += "\"f" + std::to_string(m) + "\":\"v" + std::to_string((r + salt) % 1000) + "\"";
      } else {
        row += "\"fill_m" + std::to_string(m) + "\":\"x\"";
      }
    }
    row += '}';
    rows.push_back(pad_row_to(rng, std::move(row), row_bytes));
  }
  return rows;
}

/// @brief Depth-one `qN` paths; the row's sibling count is independent
inline multi_paths sibling_rich_build_paths(std::size_t num_paths)
{
  return numbered_build_paths(num_paths, 'q');
}

/// @brief Total bytes in numbered keys with one-character prefixes
inline std::size_t key_sum_for(std::size_t count)
{
  std::size_t sum = 0;
  for (std::size_t i = 0; i < count; ++i) {
    sum += 1 + std::to_string(i).size();
  }
  return sum;
}

/// @brief Recorded dense-band width in bytes, or the count itself if unlisted
inline std::size_t dense_band_row_bytes(std::size_t member_count)
{
  switch (member_count) {
    case 50: return 959;
    case 150: return 2929;
    case 169: return 2028;
    case 340: return 4080;
    case 600: return 10240;
    default: return member_count;
  }
}

/// @brief Require a nonzero count and its exact width (e.g. 150 members cannot use 10 KiB)
inline void expect_dense_band_row_bytes(std::size_t member_count, std::size_t row_bytes)
{
  CUDF_EXPECTS(row_bytes == dense_band_row_bytes(member_count) && member_count != 0,
               "member-dense row_bytes is not the declared dense band width");
}

namespace detail {

enum class dense_layout { sibling, shallow, deep };

// Spread extra value bytes over the earliest random members without adding padding keys.
inline std::vector<std::string> dense_build_rows(std::mt19937_64& rng,
                                                 std::size_t num_rows,
                                                 std::size_t member_count,
                                                 std::size_t num_paths,
                                                 std::size_t row_bytes,
                                                 dense_layout layout)
{
  auto const nested = layout != dense_layout::sibling;
  auto const deep   = layout == dense_layout::deep;
  CUDF_EXPECTS(member_count > 0,
               nested ? "dense nested-parent rows need at least one child"
                      : "sibling-rich rows need at least one member");
  if (nested) {
    CUDF_EXPECTS(num_paths <= member_count, "dense nested-parent paths exceed the child count");
  }

  std::string_view const opening = deep ? R"({"r":{"p":{)" : nested ? R"({"r":{)" : "{";
  std::string_view const closing = deep ? "}}}" : nested ? "}}" : "}";
  // Quotes and colon cost five bytes per member; commas cost one fewer than the count.
  auto const fixed =
    opening.size() + closing.size() + 6 * member_count - 1 + key_sum_for(member_count);
  auto const queried_count = std::min(member_count, num_paths);
  auto const filler_count  = member_count - queried_count;
  auto const value_budget  = row_bytes - fixed;
  auto const fixed_budget  = queried_count * 4 + filler_count;
  auto const value_error = nested ? "dense nested-parent band leaves no byte for the member values"
                                  : "sibling-rich band leaves no byte for the member values";
  // Nested rows reserve four bytes per queried child even when all children are queried.
  if (nested) { CUDF_EXPECTS(value_budget >= fixed_budget, value_error); }
  auto const fixed_queried = filler_count > 0 && value_budget >= fixed_budget;
  auto const random_budget =
    fixed_queried || nested ? value_budget - queried_count * 4 : value_budget;
  auto const random_count = fixed_queried ? filler_count : member_count;
  if (!fixed_queried) { CUDF_EXPECTS(random_budget >= random_count, value_error); }
  auto const value_len = random_budget / random_count;
  auto const remainder = random_budget % random_count;

  std::vector<std::string> rows;
  rows.reserve(num_rows);
  for (std::size_t r = 0; r < num_rows; ++r) {
    std::string row{opening};
    for (std::size_t m = 0; m < member_count; ++m) {
      if (m > 0) { row += ','; }
      auto const queried = m < num_paths;
      row += '"';
      row += queried ? (nested ? 'c' : 'q') : 'm';
      row += std::to_string(m);
      row += "\":\"";
      if (fixed_queried && queried) {
        auto const digits = std::to_string(r % 1000);
        row += "v" + std::string(3 - digits.size(), '0') + digits;
      } else {
        auto const index = fixed_queried ? m - num_paths : m;
        row += random_string(rng, value_len + (index < remainder ? 1 : 0));
      }
      row += '"';
    }
    row += closing;
    CUDF_EXPECTS(row.size() == row_bytes,
                 nested ? "dense nested-parent row width drifted from the requested band"
                        : "sibling-rich row width drifted from the requested dense band");
    rows.push_back(std::move(row));
  }
  return rows;
}

}  // namespace detail

/// @brief Exact-width `qN`/`mN` rows; keep four-byte queried values if fillers exist and fit
inline std::vector<std::string> sibling_rich_build_rows(std::mt19937_64& rng,
                                                        std::size_t num_rows,
                                                        std::size_t member_count,
                                                        std::size_t num_paths,
                                                        std::size_t row_bytes)
{
  return detail::dense_build_rows(
    rng, num_rows, member_count, num_paths, row_bytes, detail::dense_layout::sibling);
}

/// @brief Paths `$.r.cN` or `$.r.p.cN` through a dense child object
inline multi_paths nested_dense_build_paths(std::size_t num_paths, bool deep)
{
  multi_paths paths;
  paths.reserve(num_paths);
  for (std::size_t p = 0; p < num_paths; ++p) {
    path_list path;
    path.reserve(deep ? 3 : 2);
    path.emplace_back(instruction_t::NAMED, "r", -1);
    if (deep) { path.emplace_back(instruction_t::NAMED, "p", -1); }
    path.emplace_back(instruction_t::NAMED, "c" + std::to_string(p), -1);
    paths.push_back(std::move(path));
  }
  return paths;
}

/// @brief Exact-width nested `cN`/`mN` rows, with a parent wrapper only when deep
inline std::vector<std::string> nested_dense_build_rows(std::mt19937_64& rng,
                                                        std::size_t num_rows,
                                                        std::size_t member_count,
                                                        std::size_t num_paths,
                                                        std::size_t row_bytes,
                                                        bool deep)
{
  auto const layout = deep ? detail::dense_layout::deep : detail::dense_layout::shallow;
  return detail::dense_build_rows(rng, num_rows, member_count, num_paths, row_bytes, layout);
}

}  // namespace gjo_scenarios
}  // namespace spark_rapids_jni
