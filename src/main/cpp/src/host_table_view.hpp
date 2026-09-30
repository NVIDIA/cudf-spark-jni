/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <vector>

namespace spark_rapids_jni {

/**
 * @brief A non-owning, immutable view of host data as a column of elements,
 * some of which may be null as indicated by a bitmask.
 *
 * Unless otherwise noted, the memory layout of the `host_column_view`'s data and
 * bitmask is expected to adhere to the Arrow Physical Memory Layout
 * Specification: https://arrow.apache.org/docs/memory_layout.html
 *
 * Because `host_column_view` is non-owning, no host memory is allocated nor freed
 * when `host_column_view` objects are created or destroyed.
 */
class host_column_view {
 private:
  cudf::data_type _type{cudf::type_id::EMPTY};
  cudf::size_type _size{};
  void const* _data{};
  cudf::bitmask_type const* _null_mask{};
  cudf::size_type _null_count{};
  std::vector<host_column_view> _children{};

 public:
  host_column_view()                                   = default;
  ~host_column_view()                                  = default;
  host_column_view(host_column_view const&)            = default;
  host_column_view(host_column_view&&)                 = default;
  host_column_view& operator=(host_column_view const&) = default;
  host_column_view& operator=(host_column_view&&)      = default;

  /**
   * @brief Construct a `host_column_view` from pointers to host memory for the
   * elements and bitmask of the column.
   */
  host_column_view(cudf::data_type type,
                   cudf::size_type size,
                   void const* data,
                   cudf::bitmask_type const* null_mask,
                   cudf::size_type null_count,
                   std::vector<host_column_view> const& children = {})
    : _type{type},
      _size{size},
      _data{data},
      _null_mask{null_mask},
      _null_count{null_count},
      _children{children}
  {
    CUDF_EXPECTS(size >= 0, "Column size cannot be negative.");
    if (type.id() == cudf::type_id::EMPTY) {
      _null_count = size;
      CUDF_EXPECTS(nullptr == data, "EMPTY column should have no data.");
      CUDF_EXPECTS(nullptr == null_mask, "EMPTY column should have no null mask.");
    } else if (cudf::is_compound(type)) {
      if (type.id() != cudf::type_id::STRING) {
        CUDF_EXPECTS(nullptr == data, "Compound (parent) columns cannot have data");
      }
    } else if (size > 0) {
      CUDF_EXPECTS(nullptr != data, "Null data pointer.");
    }
    if ((null_count > 0) and (type.id() != cudf::type_id::EMPTY)) {
      CUDF_EXPECTS(nullptr != null_mask, "Invalid null mask for non-zero null count.");
    }
    if (type.id() == cudf::type_id::EMPTY) {
      CUDF_EXPECTS(num_children() == 0, "EMPTY column cannot have children.");
    }
  }

  /**
   * @brief Returns the number of elements in the column
   *
   * @return The number of elements in the column
   */
  [[nodiscard]] cudf::size_type size() const noexcept { return _size; }

  /**
   * @brief Returns the element `data_type`
   *
   * @return The `data_type` of the elements in the column
   */
  [[nodiscard]] cudf::data_type type() const noexcept { return _type; }

  /**
   * @brief Indicates if the column can contain null elements, i.e., if it has
   * an allocated bitmask.
   *
   * @note If `null_count() > 0`, this function must always return `true`.
   *
   * @return true The bitmask is allocated
   * @return false The bitmask is not allocated
   */
  [[nodiscard]] bool nullable() const noexcept { return nullptr != _null_mask; }

  /**
   * @brief Returns the count of null elements
   *
   * @return The count of null elements
   */
  [[nodiscard]] cudf::size_type null_count() const noexcept { return _null_count; }

  /**
   * @brief Indicates if the column contains null elements,
   * i.e., `null_count() > 0`
   *
   * @return true One or more elements are null
   * @return false All elements are valid
   */
  [[nodiscard]] bool has_nulls() const { return null_count() > 0; }

  /**
   * @brief Returns raw pointer to the underlying bitmask allocation.
   *
   * @note If `null_count() == 0`, this may return `nullptr`.
   * @return Raw pointer to the bitmask
   */
  [[nodiscard]] cudf::bitmask_type const* null_mask() const noexcept { return _null_mask; }

  template <typename T>
  T const* data() const noexcept
  {
    return static_cast<T const*>(_data);
  }

  /**
   * @brief Returns the specified child
   *
   * @param child_index The index of the desired child
   * @return The requested child `column_view`
   */
  [[nodiscard]] host_column_view const& child(cudf::size_type child_index) const
  {
    return _children.at(child_index);
  }

  /**
   * @brief Returns the number of child columns.
   *
   * @return The number of child columns
   */
  [[nodiscard]] cudf::size_type num_children() const noexcept { return _children.size(); }

  /**
   * @brief Returns iterator to the beginning of the ordered sequence of child column-views.
   *
   * @return An iterator to a `host_column_view` referencing the first child column
   */
  auto child_begin() const noexcept { return _children.cbegin(); }

  /**
   * @brief Returns iterator to the end of the ordered sequence of child column-views.
   *
   * @return An iterator to a `host_column_view` one past the end of the child columns
   */
  auto child_end() const noexcept { return _children.cend(); }

  /**
   * @brief Returns the child column corresponding to the offsets of a strings column
   *
   * @note This must only be called on a strings column.
   */
  [[nodiscard]] host_column_view const& strings_offsets() const
  {
    return _children.at(cudf::strings_column_view::offsets_column_index);
  }

  /**
   * @brief Returns the child column corresponding to the offsets of a lists column
   *
   * @note This must only be called on a lists column.
   */
  [[nodiscard]] host_column_view const& lists_offsets() const
  {
    return _children.at(cudf::lists_column_view::offsets_column_index);
  }

  /**
   * @brief Returns the child column containing the data of a lists column
   *
   * @note This must only be called on a lists column.
   */
  [[nodiscard]] host_column_view const& lists_child() const
  {
    return _children.at(cudf::lists_column_view::child_column_index);
  }
};

/**
 * @brief A set of host_column_view's of the same size.
 */
class host_table_view {
 private:
  std::vector<host_column_view> _columns{};
  cudf::size_type _num_rows{};

 public:
  using iterator       = decltype(std::begin(_columns));   ///< Iterator type for the table
  using const_iterator = decltype(std::cbegin(_columns));  ///< const iterator type for the table

  host_table_view()                                  = default;
  ~host_table_view()                                 = default;
  host_table_view(host_table_view const&)            = default;
  host_table_view(host_table_view&&)                 = default;
  host_table_view& operator=(host_table_view const&) = default;
  host_table_view& operator=(host_table_view&&)      = default;

  /**
   * @brief Construct from a vector of column views
   *
   * @note Because a `std::vector` is constructible from a
   * `std::initializer_list`, this constructor also supports the following
   * usage:
   * ```
   * host_column_view c0, c1, c2;
   * ...
   * host_table_view t{{c0,c1,c2}}; // Creates a `host_table_view` from c0, c1, c2
   * ```
   *
   * @throws cudf::logic_error If all views do not have the same size
   *
   * @param cols The vector of column views to construct the table from
   */
  explicit host_table_view(std::vector<host_column_view> const& cols) : _columns{cols}
  {
    if (num_columns() > 0) {
      std::for_each(_columns.begin(), _columns.end(), [this](host_column_view const& col) {
        CUDF_EXPECTS(col.size() == _columns.front().size(), "Column size mismatch.");
      });
      _num_rows = _columns.front().size();
    } else {
      _num_rows = 0;
    }
  }

  /**
   * @brief Returns an iterator to the first view in the table.
   *
   * @return An iterator to the first host_column_view
   */
  iterator begin() noexcept { return std::begin(_columns); }

  /**
   * @brief Returns an iterator to the first view in the table.
   *
   * @return An iterator to the first host_column_view
   */
  [[nodiscard]] const_iterator begin() const noexcept { return std::begin(_columns); }

  /**
   * @brief Returns an iterator one past the last column view in the table.
   *
   * `end()` acts as a place holder. Attempting to dereference it results in
   * undefined behavior.
   *
   * @return An iterator to one past the last column view in the table
   */
  iterator end() noexcept { return std::end(_columns); }

  /**
   * @brief Returns an iterator one past the last column view in the table.
   *
   * `end()` acts as a place holder. Attempting to dereference it results in
   * undefined behavior.
   *
   * @return An iterator to one past the last column view in the table
   */
  [[nodiscard]] const_iterator end() const noexcept { return std::end(_columns); }

  /**
   * @brief Returns a reference to the view of the specified column
   *
   * @throws std::out_of_range
   * If `column_index` is out of the range [0, num_columns)
   *
   * @param column_index The index of the desired column
   * @return A reference to the desired column
   */
  [[nodiscard]] host_column_view const& column(cudf::size_type column_index) const
  {
    return _columns.at(column_index);
  }

  /**
   * @brief Returns the number of columns
   *
   * @return The number of columns
   */
  [[nodiscard]] cudf::size_type num_columns() const noexcept { return _columns.size(); }

  /**
   * @brief Returns the number of rows
   *
   * @return The number of rows
   */
  [[nodiscard]] cudf::size_type num_rows() const noexcept { return _num_rows; }
};

}  // namespace spark_rapids_jni
