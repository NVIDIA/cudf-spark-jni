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
  static std::string const chars = "abcdefghijklmnopqrstuvwxyz0123456789";
  std::uniform_int_distribution<std::size_t> char_dist(0, chars.size() - 1);
  std::string out;
  out.reserve(len);
  for (std::size_t i = 0; i < len; ++i) {
    out.push_back(chars[char_dist(rng)]);
  }
  return out;
}

/**
 * @brief Pad a JSON row to the requested byte length with structure-aware filler.
 *
 * Object rows (closing `}`) absorb filler as a trailing string member; array rows (closing
 * `]`) absorb filler as a trailing string element. Every scenario builds its
 * registry-mandated structure first, so each padded row lands on the requested size and the
 * reported total bytes stays accurate. The filler is seed-derived and unique per row. Rows
 * that end in neither bracket (scalar/malformed roots in roots-invalid) are returned
 * unpadded: row_bytes is a target there, not a contract.
 */
inline std::string pad_row_to(std::mt19937_64& rng, std::string content, std::size_t row_bytes)
{
  if (content.size() >= row_bytes) { return content; }
  char const closer = content.back();
  if (closer != '}' && closer != ']') { return content; }
  // (size-1) + prefix + filler + suffix == row_bytes, so the fixed cost is
  // 10 for an object member (`,\"fill\":\"`+`\"}` after popping `}`) and 3
  // for an array element. Below that the original row stands: row_bytes is a
  // target, not a contract, for scalar/malformed roots.
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

// Modeled after a real workload: the query plan issues SEVERAL fused
// get_json_object calls, never one wide single call. This pair models the
// two fused calls of that workload:
//   - NESTED_135: 135 distinct paths (the largest fused call), the wide
//     extraction fan-out over a deeply nested event column.
//   - NESTED_33: 33 distinct paths, the narrower per-branch extraction
//     in the explode-fed output projections.
// Paths read by other kernels on the same column (from_json single-value
// reads and second-stage single-path calls) are NOT benched here — they are
// not get_json_object multi-path work.
// Depth histogram of NESTED_135: 35x1, 85x2, 13x3, 2x5 among the 135
// QUERIED paths (three unqueried prefix entries are not paths; dotted keys
// are genuine multi-segment paths). No wildcards, no indices.

// Prefix-entry node kind: internal (has children; the row carries the
// subtree and a queried internal path returns the serialized object),
// json (leaf whose value is a nested JSON string), scalar
// (leaf, ~128B semantic string).
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
 * @brief Rows for the large nested scenarios: prefix-closed nested object
 * tree over the scenario's prefix table. The table is sorted lexicographic-
 * ally, which is depth-first contiguous: children immediately follow their
 * parent, so each member key opens exactly once (no duplicate keys). Leaf
 * members carry the semantic_string shape truncated at 128;
 * internal nodes open nested objects (queried internal paths return the
 * serialized subtree); 10% of scalar leaves absent
 * (scalar_non_null_per_1000 = 900). pad_row_to absorbs the remaining row
 * budget.
 */
inline std::vector<std::string> nested_build_rows(std::mt19937_64& rng,
                                                  std::vector<nested_prefix_entry> const& prefixes,
                                                  std::size_t num_rows,
                                                  std::size_t row_bytes,
                                                  std::uint64_t seed)
{
  // Seed-derived salt so different seeds produce different field values and
  // absence patterns; without it every seed yields identical inputs.
  std::uint64_t const salt = seed * 0x9E3779B97F4A7C15ULL;
  std::vector<std::string> rows;
  rows.reserve(num_rows);
  for (std::size_t r = 0; r < num_rows; ++r) {
    // 10% of scalar leaves absent, deterministic per (row, leaf index).
    auto present = [&](std::size_t idx) {
      return (r * 7919u + idx * 104729u + salt) % 1000u < 900u;
    };
    std::string row;
    row.reserve(row_bytes + 64);
    row += '{';
    // Per-level comma state: need_comma[d] == object at nesting depth d has
    // already emitted a member. Level 0 is the root object. A fresh '{'
    // pushes false so its first member carries no comma; closing pops.
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
          // Nested JSON string member (json_target_characters 640 shape).
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
 * @brief The 135-path table, defined ONCE and shared by the paths
 * and rows builders. Table order is depth-first lexicographic (children
 * immediately follow their parent) — both consumers depend on it.
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
 * @brief The 33-path table, defined ONCE and shared by the paths
 * and rows builders.
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
