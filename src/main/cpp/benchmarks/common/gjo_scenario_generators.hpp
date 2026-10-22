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
#include <cstring>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace spark_rapids_jni {
namespace gjo_scenarios {

using instruction_t = path_instruction_type;
using path_tuple    = std::tuple<instruction_t, std::string, int32_t>;
using path_list     = std::vector<path_tuple>;
using multi_paths   = std::vector<path_list>;

/**
 * @brief A seed-derived random alphanumeric string of the requested length.
 */
inline std::string random_string(std::mt19937_64& rng, std::size_t len)
{
  static constexpr char chars[] = "abcdefghijklmnopqrstuvwxyz0123456789";
  std::uniform_int_distribution<std::size_t> char_dist(0, sizeof(chars) - 2);
  std::string out(len, '\0');
  std::generate(out.begin(), out.end(), [&] { return chars[char_dist(rng)]; });
  return out;
}

/**
 * @brief Pad a JSON row to the requested byte length.
 *
 * Object rows absorb filler as a trailing string member, array rows as a
 * trailing string element; filler is seed-derived and unique per row, so
 * padded rows hit the requested size and reported bytes stay accurate.
 * Scalar/malformed rows (no closing bracket) are returned unpadded.
 */
inline std::string pad_row_to(std::mt19937_64& rng, std::string content, std::size_t row_bytes)
{
  if (content.size() >= row_bytes) { return content; }
  char const closer = content.back();
  if (closer != '}' && closer != ']') { return content; }
  // Fixed framing cost: `,\"fill\":\"` + `\"}` after popping `}` = 10 for an
  // object member; `,\"` + `\"]` = 3 for an array element.
  std::size_t const fixed = closer == '}' ? 10 : 3;
  std::size_t const gap   = row_bytes - content.size();
  if (gap < fixed) { return content; }
  content.pop_back();
  content += closer == '}' ? ",\"fill\":\"" : ",\"";
  content += random_string(rng, gap - fixed);
  content += closer == '}' ? "\"}" : "\"]";
  return content;
}

// ---------------------------------------------------------------------------
// Large nested scenarios.

// Modeled after a real workload whose query plan issues several FUSED
// get_json_object calls, not one wide call:
//   - NESTED_135: 135 paths — the largest fused call, wide extraction over
//     a deeply nested event column.
//   - NESTED_33: 33 paths — the narrower per-branch extraction in the
//     explode-fed output projections.
// Paths read by other kernels (from_json, single-path calls) are not benched
// here — they are not multi-path work.
// NESTED_135 depth histogram over the queried paths: 35x1, 85x2, 13x3, 2x5
// (13 table entries are unqueried prefixes; dotted keys are real
// multi-segment paths). No wildcards, no indices.

// Prefix-entry node kind: internal = has children (a queried internal path
// returns the serialized subtree); json = leaf holding a nested JSON string;
// scalar = leaf holding a ~128B string.
enum class nested_node_kind : int { internal, json, scalar };

struct nested_prefix_entry {
  std::vector<char const*> segments;
  bool queried;  // this exact path is queried by the modeled query
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

/**
 * @brief Rows for the large nested scenarios: a prefix-closed nested object
 * tree built in table order.
 *
 * The table is lexicographically sorted, so children follow their parent and
 * each key opens exactly once. Scalar values use the ~128B shape or null
 * (10%, deterministic per row/leaf); internal nodes open nested objects;
 * `pad_row_to` absorbs the remaining row budget.
 */
inline std::vector<std::string> nested_build_rows(std::mt19937_64& rng,
                                                  std::vector<nested_prefix_entry> const& prefixes,
                                                  std::size_t num_rows,
                                                  std::size_t row_bytes,
                                                  std::uint64_t seed)
{
  // The seed changes scalar values, null placement, and sequence values.
  std::uint64_t const salt = seed * 0x9E3779B97F4A7C15ULL;
  std::vector<std::string> rows;
  rows.reserve(num_rows);
  for (std::size_t r = 0; r < num_rows; ++r) {
    // Scalar null selection is deterministic for each seed, row, and leaf.
    auto present = [&](std::size_t idx) {
      return (r * 7919u + idx * 104729u + salt) % 1000u < 900u;
    };
    std::string row;
    row.reserve(row_bytes + 64);
    row += '{';
    // need_comma[d] tracks whether the object at depth d already emitted a
    // member; '{' pushes false (first member gets no comma), '}' pops.
    std::vector<bool> need_comma(1, false);
    std::size_t depth = 0;  // == need_comma.size() - 1
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
        need_comma[depth] = true;  // the closed child was a member of this level
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
          // This leaf contains serialized JSON inside a JSON string.
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

/**
 * @brief The 135-path table, defined once and shared by the paths and rows
 * builders; both depend on its depth-first lexicographic order.
 */
// Keys are obfuscated rank-derived labels preserving each segment's byte length.
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

/**
 * @brief The 33-path table, defined once and shared by the paths and rows
 * builders (same ordering contract as the 135 table).
 */
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

}  // namespace gjo_scenarios
}  // namespace spark_rapids_jni
