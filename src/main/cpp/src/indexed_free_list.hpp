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

#include <rmm/mr/detail/coalescing_free_list.hpp>

#include <map>
#include <set>
#include <utility>

namespace spark_rapids_jni {
namespace detail {

// rmm's coalescing_free_list is a std::list: best-fit lookup is std::min_element (O(F))
// and ordered insert is std::find_if (O(F)) — under fragmentation every alloc/free holds
// the pool mutex for a linear scan. This replacement keeps the SAME allocation policy
// (smallest block >= requested, lowest address on ties, address-ordered coalescing on
// insert; block::is_better_fit's min_element selection is exactly (size, address) pair
// order) but drives both paths through ordered indexes: O(log F).

class indexed_free_list {
 public:
  using block_type = rmm::mr::detail::block;

  /**
   * @brief Inserts a block, coalescing with the preceding/following address neighbors
   *        exactly like coalescing_free_list::insert.
   */
  void insert(block_type const& b)
  {
    if (b.pointer() == nullptr || b.size() == 0) { return; }
    auto const next     = by_addr_.lower_bound(b.pointer());
    auto const previous = (next == by_addr_.begin()) ? by_addr_.end() : std::prev(next);
    bool const merge_prev =
      (previous != by_addr_.end()) && previous->second.is_contiguous_before(b);
    bool const merge_next = (next != by_addr_.end()) && b.is_contiguous_before(next->second);
    if (merge_prev && merge_next) {
      block_type const merged = previous->second.merge(b).merge(next->second);
      erase_indexed(next->second);
      erase_indexed(previous->second);
      put_indexed(merged);
    } else if (merge_prev) {
      block_type const merged = previous->second.merge(b);
      erase_indexed(previous->second);
      put_indexed(merged);
    } else if (merge_next) {
      block_type const merged = b.merge(next->second);
      erase_indexed(next->second);
      put_indexed(merged);
    } else {
      put_indexed(b);
    }
  }

  /**
   * @brief Best-fit lookup: the smallest block >= size, lowest address on ties.
   *        Returns an invalid (default) block when nothing fits.
   */
  block_type get_block(std::size_t size)
  {
    auto const it = by_size_.lower_bound({size, nullptr});
    if (it == by_size_.end()) { return block_type{}; }
    block_type const found = by_addr_.at(it->second);
    erase_indexed(found);
    return found;
  }

  bool is_empty() const { return by_addr_.empty(); }

 private:
  void put_indexed(block_type const& b)
  {
    by_addr_[b.pointer()] = b;
    by_size_.insert({b.size(), b.pointer()});
  }

  void erase_indexed(block_type const& b)
  {
    by_addr_.erase(b.pointer());
    by_size_.erase({b.size(), b.pointer()});
  }

  // Address-ordered: coalescing neighbor lookup.
  std::map<char*, block_type> by_addr_;
  // (size, address)-ordered: best-fit lookup with the list's exact tie-break.
  std::set<std::pair<std::size_t, char*>> by_size_;
};

}  // namespace detail
}  // namespace spark_rapids_jni
