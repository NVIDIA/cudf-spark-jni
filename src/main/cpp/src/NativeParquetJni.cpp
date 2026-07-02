/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.
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

#include <cwctype>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// TCompactProtocol requires some #defines to work right.
// This came from the parquet code itself...
#define SIGNED_RIGHT_SHIFT_IS  1
#define ARITHMETIC_RIGHT_SHIFT 1
#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"
#include "nvtx_ranges.hpp"

#include <generated/parquet_types.h>
#include <thrift/TApplicationException.h>
#include <thrift/protocol/TCompactProtocol.h>
#include <thrift/transport/TBufferTransports.h>

namespace rapids {
namespace jni {

/**
 * Convert a string to lower case. It uses std::tolower per character which has limitations
 * and may not produce the exact same result as the JVM does. This is probably good enough
 * for now.
 */
std::string unicode_to_lower(std::string const& input)
{
  std::mbstate_t to_wc_state = std::mbstate_t();
  const char* mbstr          = input.data();
  // get the size of the wide character result
  std::size_t wide_size   = std::mbsrtowcs(nullptr, &mbstr, 0, &to_wc_state);
  auto const invalid_size = std::numeric_limits<std::size_t>::max();
  if (wide_size == invalid_size) { throw std::invalid_argument("invalid character sequence"); }

  std::vector<wchar_t> wide(wide_size + 1);
  // Set a null so we can get a proper output size from wcstombs. This is because
  // we pass in a max length of 0, so it will only stop when it see the null character.
  wide.back() = 0;
  if (std::mbsrtowcs(wide.data(), &mbstr, wide_size, &to_wc_state) != wide_size) {
    throw std::runtime_error("error during wide char converstion");
  }
  for (auto wit = wide.begin(); wit != wide.end(); ++wit) {
    *wit = std::towlower(*wit);
  }
  // Get the multi-byte result size
  std::mbstate_t from_wc_state = std::mbstate_t();
  const wchar_t* wcstr         = wide.data();
  std::size_t mb_size          = std::wcsrtombs(nullptr, &wcstr, 0, &from_wc_state);
  if (mb_size == invalid_size) {
    throw std::invalid_argument("unsupported wide character sequence");
  }
  // We are allocating a fixed size string so we can put the data directly into it
  // instead of going through a NUL terminated char* first. The NUL fill char is
  // just because we need to pass in a fill char. The value does not matter
  // because it will be overwritten. std::string itself will insert a NUL
  // terminator on the buffer it allocates internally. We don't need to worry about it.
  std::string ret(mb_size, '\0');
  if (std::wcsrtombs(ret.data(), &wcstr, mb_size, &from_wc_state) != mb_size) {
    throw std::runtime_error("error during multibyte char converstion");
  }
  return ret;
}

/**
 * Holds a set of "maps" that are used to rewrite various parts of the parquet metadata.
 * Generally each "map" is a gather map that pulls data from an input vector to be placed in
 * an output vector.
 */
struct column_pruning_maps {
  // gather map for pulling out items from the schema
  std::vector<int> schema_map;
  // Each SchemaElement also includes the number of children in it. This allows the vector
  // to be interpreted as a tree flattened depth first. These are the new values for num
  // children after the schema is gathered.
  std::vector<int> schema_num_children;
  // There are several places where a struct is stored only for a leaf column (like a column chunk)
  // This holds the gather map for those cases.
  std::vector<int> chunk_map;
};

/**
 * Pairs Parquet FileMetaData with the file-global row index offset of each
 * row group. The offset for row group i is the cumulative row count of all
 * preceding row groups in the file.
 *
 * When row groups have been filtered by byte range, the offsets are still
 * computed from all original row groups before filtering, so they remain
 * correct even when the footer contains only a subset of the file's row groups.
 */
struct parquet_footer_with_row_group_offsets {
  std::unique_ptr<parquet::format::FileMetaData> meta;
  std::vector<int64_t> row_index_offsets;  // one per row group in meta
};

/**
 * Tags what type of node is expected in the passed down Spark schema. This
 * lets us match the Spark schema to the schema in the Parquet file. Different
 * versions of parquet had different layouts for various nested types.
 */
enum class Tag { VALUE = 0, STRUCT, LIST, MAP };

/**
 * This class will handle processing column pruning for a schema. It is written as a class because
 * of JNI we are sending the names of the columns as a depth first list, like parquet does
 * internally.
 */
class column_pruner {
 public:
  /**
   * Create pruning filter from a depth first flattened tree of names and num_children.
   * The root entry is not included in names or in num_children, but parent_num_children
   * should hold how many entries there are in it.
   */
  column_pruner(std::vector<std::string> const& names,
                std::vector<int> const& num_children,
                std::vector<Tag> const& tags,
                int const parent_num_children)
    : children(), tag(Tag::STRUCT)
  {
    add_depth_first(names, num_children, tags, parent_num_children);
  }

  explicit column_pruner(Tag const in_tag) : children(), tag(in_tag) {}

  column_pruner() : children(), tag(Tag::STRUCT) {}

  /**
   * Given a schema from a parquet file create a set of pruning maps to prune columns from the rest
   * of the footer
   */
  column_pruning_maps filter_schema(std::vector<parquet::format::SchemaElement> const& schema,
                                    bool const ignore_case) const
  {
    SRJ_FUNC_RANGE();

    // These are the outputs of the computation.
    std::vector<int> chunk_map;
    std::vector<int> schema_map;
    std::vector<int> schema_num_children;
    std::size_t current_input_schema_index = 0;
    std::size_t next_input_chunk_index     = 0;

    filter_schema(schema,
                  ignore_case,
                  current_input_schema_index,
                  next_input_chunk_index,
                  chunk_map,
                  schema_map,
                  schema_num_children);

    return column_pruning_maps{
      std::move(schema_map), std::move(schema_num_children), std::move(chunk_map)};
  }

 private:
  std::string get_name(parquet::format::SchemaElement& elem,
                       const bool normalize_case = false) const
  {
    return normalize_case ? unicode_to_lower(elem.name) : elem.name;
  }

  int get_num_children(parquet::format::SchemaElement& elem) const
  {
    return elem.__isset.num_children ? elem.num_children : 0;
  }

  void skip(std::vector<parquet::format::SchemaElement> const& schema,
            std::size_t& current_input_schema_index,
            std::size_t& next_input_chunk_index) const
  {
    // We want to skip everything referenced by the current_input_schema_index and its children.
    // But we do have to update the chunk indexes as we go.
    int num_to_skip = 1;
    while (num_to_skip > 0 && current_input_schema_index < schema.size()) {
      auto schema_item = schema[current_input_schema_index];
      bool is_leaf     = schema_item.__isset.type;
      if (is_leaf) { ++next_input_chunk_index; }

      if (schema_item.__isset.num_children) {
        num_to_skip = num_to_skip + schema_item.num_children;
      }

      --num_to_skip;
      ++current_input_schema_index;
    }
  }

  /**
   * filter_schema, but specific to Tag::STRUCT.
   */
  void filter_schema_struct(std::vector<parquet::format::SchemaElement> const& schema,
                            bool const ignore_case,
                            std::size_t& current_input_schema_index,
                            std::size_t& next_input_chunk_index,
                            std::vector<int>& chunk_map,
                            std::vector<int>& schema_map,
                            std::vector<int>& schema_num_children) const
  {
    // First verify that we found a struct, like we expected to find.
    auto struct_schema_item = schema.at(current_input_schema_index);
    bool is_leaf            = struct_schema_item.__isset.type;
    if (is_leaf) { throw std::runtime_error("Found a leaf node, but expected to find a struct"); }

    int num_children = get_num_children(struct_schema_item);

    // Save the current index (the position of this struct in the full schema).
    // We'll need this to add it to the pruned schema map if it qualifies.
    int struct_index = current_input_schema_index;

    // We add the Struct to schema_map (with a placeholder for child count).
    // But we might remove it later if no children match. This ensures parent-before-children
    // ordering in schema_map.
    schema_map.push_back(struct_index);
    schema_num_children.push_back(0);  // Placeholder for child count

    // This is necessary to update the child count accurately.
    // Since we just added a new element to schema_num_children, its position is the last index.
    int struct_map_position = schema_num_children.size() - 1;
    // Move to first child
    ++current_input_schema_index;
    int matched_children_count = 0;

    // For a STRUCT we want to look for all of the children that match the name and let each of them
    // handle updating things themselves.
    for (int child_id = 0; child_id < num_children && current_input_schema_index < schema.size();
         child_id++) {
      auto schema_item = schema[current_input_schema_index];
      std::string name = get_name(schema_item, ignore_case);
      auto found       = children.find(name);

      if (found != children.end()) {
        // Record the current size of schema_map before processing the child
        // This helps determine if the child was retained after processing.
        auto const before_child_size = schema_map.size();
        found->second.filter_schema(schema,
                                    ignore_case,
                                    current_input_schema_index,
                                    next_input_chunk_index,
                                    chunk_map,
                                    schema_map,
                                    schema_num_children);

        // If the size of schema_map has increased, it means that child was successfully
        // included
        if (schema_map.size() > before_child_size) {
          ++matched_children_count;
          // Update the child count for the struct
          schema_num_children[struct_map_position]++;
        }
      } else {
        // No match was found so skip the child.
        skip(schema, current_input_schema_index, next_input_chunk_index);
      }
    }

    // If the below condition is true, it means none of this struct's children
    // were included. We should remove it from `schema_map` and `schema_num_children`.
    if (matched_children_count == 0) {
      schema_map.pop_back();
      schema_num_children.pop_back();
    }
  }

  /**
   * filter_schema, but specific to Tag::VALUE.
   */
  void filter_schema_value(std::vector<parquet::format::SchemaElement> const& schema,
                           std::size_t& current_input_schema_index,
                           std::size_t& next_input_chunk_index,
                           std::vector<int>& chunk_map,
                           std::vector<int>& schema_map,
                           std::vector<int>& schema_num_children) const
  {
    auto schema_item = schema.at(current_input_schema_index);
    bool is_leaf     = schema_item.__isset.type;
    if (!is_leaf) { throw std::runtime_error("found a non-leaf entry when reading a leaf value"); }
    if (get_num_children(schema_item) != 0) {
      throw std::runtime_error("found an entry with children when reading a leaf value");
    }
    schema_map.push_back(current_input_schema_index);
    schema_num_children.push_back(0);
    ++current_input_schema_index;
    chunk_map.push_back(next_input_chunk_index);
    ++next_input_chunk_index;
  }

  /**
   * filter_schema, but specific to Tag::LIST.
   */
  void filter_schema_list(std::vector<parquet::format::SchemaElement> const& schema,
                          bool const ignore_case,
                          std::size_t& current_input_schema_index,
                          std::size_t& next_input_chunk_index,
                          std::vector<int>& chunk_map,
                          std::vector<int>& schema_map,
                          std::vector<int>& schema_num_children) const
  {
    // By convention with the java code the child is always called "element"...
    auto found = children.at("element");
    // A list starts out as a group element(not leaf) with a ConvertedType that is a LIST
    // Under it will be a repeated element
    auto list_schema_item = schema.at(current_input_schema_index);
    std::string list_name = list_schema_item.name;
    bool is_group         = !list_schema_item.__isset.type;

    // Rules for how to parse lists from the parquet format docs
    // 1. If the repeated field is not a group, then its type is the element type and elements are
    // required.
    // 2. If the repeated field is a group with multiple fields, then its type is the element type
    // and elements are required.
    // 3. If the repeated field is a group with one field and is named either array or uses the
    // LIST-annotated group's name
    //    with _tuple appended then the repeated type is the element type and elements are required.
    // 4. Otherwise, the repeated field's type is the element type with the repeated field's
    // repetition.
    if (!is_group) {
      if (!list_schema_item.__isset.repetition_type ||
          list_schema_item.repetition_type != parquet::format::FieldRepetitionType::REPEATED) {
        throw std::runtime_error("expected list item to be repeating");
      }
      return filter_schema_value(schema,
                                 current_input_schema_index,
                                 next_input_chunk_index,
                                 chunk_map,
                                 schema_map,
                                 schema_num_children);
    }
    auto num_list_children = get_num_children(list_schema_item);
    if (num_list_children > 1) {
      if (!list_schema_item.__isset.repetition_type ||
          list_schema_item.repetition_type != parquet::format::FieldRepetitionType::REPEATED) {
        throw std::runtime_error("expected list item to be repeating");
      }
      return found.filter_schema(schema,
                                 ignore_case,
                                 current_input_schema_index,
                                 next_input_chunk_index,
                                 chunk_map,
                                 schema_map,
                                 schema_num_children);
    }
    if (num_list_children != 1) {
      throw std::runtime_error("the structure of the outer list group is not standard");
    }

    // Now that the top level group looks good add it into the maps, and then start to look at the
    // children
    schema_map.push_back(current_input_schema_index);
    schema_num_children.push_back(1);
    ++current_input_schema_index;

    auto repeated_field_schema_item = schema.at(current_input_schema_index);
    if (!repeated_field_schema_item.__isset.repetition_type ||
        repeated_field_schema_item.repetition_type !=
          parquet::format::FieldRepetitionType::REPEATED) {
      throw std::runtime_error("the structure of the list's child is not standard (non repeating)");
    }

    bool repeated_field_is_group    = !repeated_field_schema_item.__isset.type;
    int repeated_field_num_children = get_num_children(repeated_field_schema_item);
    std::string repeated_field_name = repeated_field_schema_item.name;
    if (repeated_field_is_group && repeated_field_num_children == 1 &&
        repeated_field_name != "array" && repeated_field_name != (list_name + "_tuple")) {
      // This is the "standard" format where there are two groups and then a child under the the
      // second group that holds the data. so add in the middle repeated group to the map
      schema_map.push_back(current_input_schema_index);
      schema_num_children.push_back(1);
      ++current_input_schema_index;

      // And let the child filter itself.
      found.filter_schema(schema,
                          ignore_case,
                          current_input_schema_index,
                          next_input_chunk_index,
                          chunk_map,
                          schema_map,
                          schema_num_children);
    } else {
      // This is for an older format that is some times used where it is just two levels
      found.filter_schema(schema,
                          ignore_case,
                          current_input_schema_index,
                          next_input_chunk_index,
                          chunk_map,
                          schema_map,
                          schema_num_children);
    }
  }

  /**
   * filter_schema, but specific to Tag::MAP.
   */
  void filter_schema_map(std::vector<parquet::format::SchemaElement> const& schema,
                         bool const ignore_case,
                         std::size_t& current_input_schema_index,
                         std::size_t& next_input_chunk_index,
                         std::vector<int>& chunk_map,
                         std::vector<int>& schema_map,
                         std::vector<int>& schema_num_children) const
  {
    // By convention with the java code the children are always called "key" and "value"...
    auto key_found       = children.at("key");
    auto value_found     = children.at("value");
    auto map_schema_item = schema.at(current_input_schema_index);

    // Maps are two levels. An outer group that has a ConvertedType of MAP or MAP_KEY_VALUE
    // and then an inner group that has two fields a key (that is required) and a value, that is
    // optional.

    bool is_map_group = !map_schema_item.__isset.type;
    if (!is_map_group) {
      throw std::runtime_error("expected a map item, but found a single value");
    }
    if (!map_schema_item.__isset.converted_type ||
        (map_schema_item.converted_type != parquet::format::ConvertedType::MAP &&
         map_schema_item.converted_type != parquet::format::ConvertedType::MAP_KEY_VALUE)) {
      throw std::runtime_error("expected a map type, but it was not found.");
    }
    if (get_num_children(map_schema_item) != 1) {
      throw std::runtime_error("the structure of the outer map group is not standard");
    }

    // The outer group looks good so lets add it in.
    schema_map.push_back(current_input_schema_index);
    schema_num_children.push_back(1);
    ++current_input_schema_index;

    // Now lets look at the repeated child.
    auto repeated_field_schema_item = schema.at(current_input_schema_index);
    if (!repeated_field_schema_item.__isset.repetition_type ||
        repeated_field_schema_item.repetition_type !=
          parquet::format::FieldRepetitionType::REPEATED) {
      throw std::runtime_error("found non repeating map child");
    }

    int repeated_field_num_children = get_num_children(repeated_field_schema_item);

    if (repeated_field_num_children != 1 && repeated_field_num_children != 2) {
      throw std::runtime_error("found map with wrong number of children");
    }

    schema_map.push_back(current_input_schema_index);
    schema_num_children.push_back(repeated_field_num_children);
    ++current_input_schema_index;

    // Process the key...
    key_found.filter_schema(schema,
                            ignore_case,
                            current_input_schema_index,
                            next_input_chunk_index,
                            chunk_map,
                            schema_map,
                            schema_num_children);
    if (repeated_field_num_children == 2) {
      // Process the value...
      value_found.filter_schema(schema,
                                ignore_case,
                                current_input_schema_index,
                                next_input_chunk_index,
                                chunk_map,
                                schema_map,
                                schema_num_children);
    }
  }

  /**
   * Recursive method to parse and update the maps to filter out columns in the schema and chunks.
   * Each column_pruner is responsible to parse out from schema what it holds and skip anything
   * that does not match. chunk_map, schema_map, and schema_num_children are the final outputs.
   * current_input_schema_index and next_input_chunk_index are also outputs but are state that is
   * passed to each child and returned when it consumes something.
   */
  void filter_schema(std::vector<parquet::format::SchemaElement> const& schema,
                     bool const ignore_case,
                     std::size_t& current_input_schema_index,
                     std::size_t& next_input_chunk_index,
                     std::vector<int>& chunk_map,
                     std::vector<int>& schema_map,
                     std::vector<int>& schema_num_children) const
  {
    switch (tag) {
      case Tag::STRUCT:
        filter_schema_struct(schema,
                             ignore_case,
                             current_input_schema_index,
                             next_input_chunk_index,
                             chunk_map,
                             schema_map,
                             schema_num_children);
        break;
      case Tag::VALUE:
        filter_schema_value(schema,
                            current_input_schema_index,
                            next_input_chunk_index,
                            chunk_map,
                            schema_map,
                            schema_num_children);
        break;
      case Tag::LIST:
        filter_schema_list(schema,
                           ignore_case,
                           current_input_schema_index,
                           next_input_chunk_index,
                           chunk_map,
                           schema_map,
                           schema_num_children);
        break;
      case Tag::MAP:
        filter_schema_map(schema,
                          ignore_case,
                          current_input_schema_index,
                          next_input_chunk_index,
                          chunk_map,
                          schema_map,
                          schema_num_children);
        break;
      default:
        throw std::runtime_error(std::string("INTERNAL ERROR UNEXPECTED TAG FOUND ") +
                                 std::to_string(static_cast<int>(tag)));
    }
  }

  /**
   * Do a depth first traversal to build up column_pruner into a tree that matches the schema we
   * want to filter using.
   */
  void add_depth_first(std::vector<std::string> const& names,
                       std::vector<int> const& num_children,
                       std::vector<Tag> const& tags,
                       int parent_num_children)
  {
    SRJ_FUNC_RANGE();
    if (parent_num_children == 0) {
      // There is no point in doing more the tree is empty, and it lets us avoid some corner cases
      // in the code below
      return;
    }
    auto num = names.size();
    std::vector<column_pruner*> tree_stack;
    std::vector<int> num_children_stack;
    tree_stack.push_back(this);
    num_children_stack.push_back(parent_num_children);
    for (uint64_t i = 0; i < num; ++i) {
      auto name  = names[i];
      auto num_c = num_children[i];
      auto t     = tags[i];
      tree_stack.back()->children.try_emplace(name, t);
      if (num_c > 0) {
        tree_stack.push_back(&tree_stack.back()->children[name]);
        num_children_stack.push_back(num_c);
        continue;
      }

      // go back up the stack/tree removing children until we hit one with more children
      do {
        int parent_children_left = num_children_stack.back() - 1;
        if (parent_children_left > 0) {
          num_children_stack.back() = parent_children_left;
          break;
        }

        tree_stack.pop_back();
        num_children_stack.pop_back();
      } while (tree_stack.size() > 0);
    }

    if (tree_stack.size() != 0 || num_children_stack.size() != 0) {
      throw std::invalid_argument("DIDN'T CONSUME EVERYTHING...");
    }
  }

  std::map<std::string, column_pruner> children;
  Tag tag;
};

static bool invalid_file_offset(long start_index, long pre_start_index, long pre_compressed_size)
{
  bool invalid = false;
  // checking the first rowGroup
  if (pre_start_index == 0 && start_index != 4) {
    invalid = true;
    return invalid;
  }

  // calculate start index for other blocks
  int64_t min_start_index = pre_start_index + pre_compressed_size;
  if (start_index < min_start_index) {
    // a bad offset detected, try first column's offset
    // can not use minStartIndex in case of padding
    invalid = true;
  }

  return invalid;
}

static int64_t get_offset(parquet::format::ColumnChunk const& column_chunk)
{
  auto md        = column_chunk.meta_data;
  int64_t offset = md.data_page_offset;
  if (md.__isset.dictionary_page_offset && offset > md.dictionary_page_offset) {
    offset = md.dictionary_page_offset;
  }
  return offset;
}

/**
 * A collection of row groups paired with their file-global row index offsets.
 *
 * When produced by filter_groups, the offsets are computed from all original
 * row groups before byte-range filtering, so they remain correct even when
 * only a subset survives.
 */
struct row_groups {
  std::vector<parquet::format::RowGroup> groups;
  std::vector<int64_t> row_index_offsets;
};

static row_groups filter_groups(parquet::format::FileMetaData const& meta,
                                int64_t part_offset,
                                int64_t part_length)
{
  SRJ_FUNC_RANGE();
  // This is based off of the java parquet_mr code to find the groups in a range...
  auto num_row_groups             = meta.row_groups.size();
  int64_t pre_start_index         = 0;
  int64_t pre_compressed_size     = 0;
  bool first_column_with_metadata = true;
  if (num_row_groups > 0) {
    first_column_with_metadata = meta.row_groups[0].columns[0].__isset.meta_data;
  }

  row_groups result;
  int64_t cumulative_rows = 0;
  for (uint64_t rg_i = 0; rg_i < num_row_groups; ++rg_i) {
    parquet::format::RowGroup const& row_group = meta.row_groups[rg_i];
    int64_t total_size                         = 0;
    int64_t start_index;
    auto column_chunk = row_group.columns[0];
    if (first_column_with_metadata) {
      start_index = get_offset(column_chunk);
    } else {
      // the file_offset of first block always holds the truth, while other blocks don't :
      // see PARQUET-2078 for details
      start_index = row_group.file_offset;
      if (invalid_file_offset(start_index, pre_start_index, pre_compressed_size)) {
        // first row group's offset is always 4, else
        // use minStartIndex(imprecise in case of padding, but good enough for filtering)
        start_index = (pre_start_index == 0) ? 4 : pre_start_index + pre_compressed_size;
      }
      pre_start_index     = start_index;
      pre_compressed_size = row_group.total_compressed_size;
    }
    if (row_group.__isset.total_compressed_size) {
      total_size = row_group.total_compressed_size;
    } else {
      auto num_columns = row_group.columns.size();
      for (uint64_t cc_i = 0; cc_i < num_columns; ++cc_i) {
        parquet::format::ColumnChunk const& col = row_group.columns[cc_i];
        total_size += col.meta_data.total_compressed_size;
      }
    }

    int64_t mid_point = start_index + total_size / 2;
    if (mid_point >= part_offset && mid_point < (part_offset + part_length)) {
      result.groups.push_back(row_group);
      result.row_index_offsets.push_back(cumulative_rows);
    }
    cumulative_rows += row_group.num_rows;
  }
  return result;
}

void deserialize_parquet_footer(uint8_t* buffer, uint32_t len, parquet::format::FileMetaData* meta)
{
  using ThriftBuffer = apache::thrift::transport::TMemoryBuffer;

  SRJ_FUNC_RANGE();
// A lot of this came from the parquet source code...
// Deserialize msg bytes into c++ thrift msg using memory transport.
#if PARQUET_THRIFT_VERSION_MAJOR > 0 || PARQUET_THRIFT_VERSION_MINOR >= 14
  auto conf = std::make_shared<apache::thrift::TConfiguration>();
  conf->setMaxMessageSize(std::numeric_limits<int>::max());
  auto tmem_transport = std::make_shared<ThriftBuffer>(buffer, len, ThriftBuffer::OBSERVE, conf);
#else
  auto tmem_transport = std::make_shared<ThriftBuffer>(buffer, len);
#endif

  apache::thrift::protocol::TCompactProtocolFactoryT<ThriftBuffer> tproto_factory;
  // Protect against CPU and memory bombs
  tproto_factory.setStringSizeLimit(100 * 1000 * 1000);
  // Structs in the thrift definition are relatively large (at least 300 bytes).
  // This limits total memory to the same order of magnitude as stringSize.
  tproto_factory.setContainerSizeLimit(1000 * 1000);
  std::shared_ptr<apache::thrift::protocol::TProtocol> tproto =
    tproto_factory.getProtocol(tmem_transport);
  try {
    meta->read(tproto.get());
  } catch (std::exception& e) {
    std::stringstream ss;
    ss << "Couldn't deserialize thrift: " << e.what() << "\n";
    throw std::runtime_error(ss.str());
  }
}

void filter_columns(std::vector<parquet::format::RowGroup>& groups, std::vector<int>& chunk_filter)
{
  SRJ_FUNC_RANGE();
  for (auto group_it = groups.begin(); group_it != groups.end(); ++group_it) {
    std::vector<parquet::format::ColumnChunk> new_chunks;
    for (auto it = chunk_filter.begin(); it != chunk_filter.end(); ++it) {
      new_chunks.push_back(group_it->columns[*it]);
    }
    group_it->columns = std::move(new_chunks);
  }
}

}  // namespace jni
}  // namespace rapids

extern "C" {

JNIEXPORT long JNICALL
Java_com_nvidia_spark_rapids_jni_ParquetFooter_readAndFilter(JNIEnv* env,
                                                             jclass,
                                                             jlong buffer,
                                                             jlong buffer_length,
                                                             jlong part_offset,
                                                             jlong part_length,
                                                             jobjectArray filter_col_names,
                                                             jintArray num_children,
                                                             jintArray tags,
                                                             jint parent_num_children,
                                                             jboolean ignore_case)
{
  SRJ_FUNC_RANGE();
  JNI_TRY
  {
    auto meta    = std::make_unique<parquet::format::FileMetaData>();
    uint32_t len = static_cast<uint32_t>(buffer_length);
    // We don't support encrypted parquet...
    rapids::jni::deserialize_parquet_footer(reinterpret_cast<uint8_t*>(buffer), len, meta.get());

    // Get the filter for the columns first...
    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);
    cudf::jni::native_jintArray n_num_children(env, num_children);
    cudf::jni::native_jintArray n_tags(env, tags);

    std::vector<rapids::jni::Tag> tags(n_tags.size());
    for (int i = 0; i < n_tags.size(); i++) {
      tags[i] = static_cast<rapids::jni::Tag>(n_tags[i]);
    }

    rapids::jni::column_pruner pruner(n_filter_col_names.as_cpp_vector(),
                                      std::vector(n_num_children.begin(), n_num_children.end()),
                                      tags,
                                      parent_num_children);
    auto filter = pruner.filter_schema(meta->schema, ignore_case);

    // start by filtering the schema and the chunks
    std::size_t new_schema_size = filter.schema_map.size();
    std::vector<parquet::format::SchemaElement> new_schema(new_schema_size);
    for (std::size_t i = 0; i < new_schema_size; ++i) {
      int orig_index             = filter.schema_map[i];
      int new_num_children       = filter.schema_num_children[i];
      new_schema[i]              = meta->schema[orig_index];
      new_schema[i].num_children = new_num_children;
    }
    meta->schema = std::move(new_schema);
    if (meta->__isset.column_orders) {
      std::vector<parquet::format::ColumnOrder> new_order;
      for (auto it = filter.chunk_map.begin(); it != filter.chunk_map.end(); ++it) {
        new_order.push_back(meta->column_orders[*it]);
      }
      meta->column_orders = std::move(new_order);
    }
    // Build the footer wrapper that pairs metadata with row index offsets.
    auto footer  = std::make_unique<rapids::jni::parquet_footer_with_row_group_offsets>();
    footer->meta = std::move(meta);

    // Filter row groups by byte range and compute file-global row index offsets.
    if (part_length >= 0) {
      auto filtered = rapids::jni::filter_groups(*footer->meta, part_offset, part_length);
      footer->meta->row_groups  = std::move(filtered.groups);
      footer->row_index_offsets = std::move(filtered.row_index_offsets);
    } else {
      // No byte-range filtering — offsets are just cumulative row counts.
      int64_t cumulative = 0;
      for (auto const& rg : footer->meta->row_groups) {
        footer->row_index_offsets.push_back(cumulative);
        cumulative += rg.num_rows;
      }
    }
    rapids::jni::filter_columns(footer->meta->row_groups, filter.chunk_map);

    return cudf::jni::release_as_jlong(footer);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_close(JNIEnv* env,
                                                                            jclass,
                                                                            jlong handle)
{
  JNI_TRY
  {
    auto* ptr = reinterpret_cast<rapids::jni::parquet_footer_with_row_group_offsets*>(handle);
    delete ptr;
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_getNumRows(JNIEnv* env,
                                                                                  jclass,
                                                                                  jlong handle)
{
  JNI_TRY
  {
    auto* footer = reinterpret_cast<rapids::jni::parquet_footer_with_row_group_offsets*>(handle);
    long ret     = 0;
    for (auto it = footer->meta->row_groups.begin(); it != footer->meta->row_groups.end(); ++it) {
      ret = ret + it->num_rows;
    }
    return ret;
  }
  JNI_CATCH(env, -1);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_getNumColumns(JNIEnv* env,
                                                                                     jclass,
                                                                                     jlong handle)
{
  JNI_TRY
  {
    auto* footer = reinterpret_cast<rapids::jni::parquet_footer_with_row_group_offsets*>(handle);
    int ret      = 0;
    if (footer->meta->schema.size() > 0) {
      if (footer->meta->schema[0].__isset.num_children) {
        ret = footer->meta->schema[0].num_children;
      }
    }
    return ret;
  }
  JNI_CATCH(env, -1);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_serializeThriftFile(
  JNIEnv* env, jclass, jlong handle, jobject host_memory_allocator)
{
  SRJ_FUNC_RANGE();
  JNI_TRY
  {
    auto* footer = reinterpret_cast<rapids::jni::parquet_footer_with_row_group_offsets*>(handle);
    std::shared_ptr<apache::thrift::transport::TMemoryBuffer> transportOut(
      new apache::thrift::transport::TMemoryBuffer());
    apache::thrift::protocol::TCompactProtocolFactoryT<apache::thrift::transport::TMemoryBuffer>
      factory;
    auto protocolOut = factory.getProtocol(transportOut);
    footer->meta->write(protocolOut.get());
    uint8_t* buf_ptr;
    uint32_t buf_size;
    transportOut->getBuffer(&buf_ptr, &buf_size);

    // 12 extra is for the MAGIC thrift_footer length MAGIC
    jobject ret = cudf::jni::allocate_host_buffer(env, buf_size + 12, false, host_memory_allocator);
    uint8_t* ret_addr = reinterpret_cast<uint8_t*>(cudf::jni::get_host_buffer_address(env, ret));
    ret_addr[0]       = 'P';
    ret_addr[1]       = 'A';
    ret_addr[2]       = 'R';
    ret_addr[3]       = '1';
    std::memcpy(ret_addr + 4, buf_ptr, buf_size);
    uint8_t* after = ret_addr + buf_size + 4;
    after[0]       = static_cast<uint8_t>(0xFF & buf_size);
    after[1]       = static_cast<uint8_t>(0xFF & (buf_size >> 8));
    after[2]       = static_cast<uint8_t>(0xFF & (buf_size >> 16));
    after[3]       = static_cast<uint8_t>(0xFF & (buf_size >> 24));
    after[4]       = 'P';
    after[5]       = 'A';
    after[6]       = 'R';
    after[7]       = '1';
    return ret;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_ParquetFooter_getRowIndexOffsets(JNIEnv* env, jclass, jlong handle)
{
  JNI_TRY
  {
    auto* footer = reinterpret_cast<rapids::jni::parquet_footer_with_row_group_offsets*>(handle);
    auto const& offsets = footer->row_index_offsets;
    cudf::jni::native_jlongArray result(
      env, reinterpret_cast<jlong const*>(offsets.data()), static_cast<int>(offsets.size()));
    return result.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}
}
