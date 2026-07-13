/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include "hyper_log_log_plus_plus.hpp"
#include "hyper_log_log_plus_plus_const.hpp"
#include "hyper_log_log_plus_plus_host_udf.hpp"
#include "utilities/iterator.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>

namespace spark_rapids_jni {

namespace {

struct hllpp_groupby_udf : cudf::groupby_host_udf {
  hllpp_groupby_udf(int precision_, bool is_merge_) : precision(precision_), is_merge(is_merge_) {}

  /**
   * @brief Perform the main groupby computation for HLLPP UDF.
   */
  [[nodiscard]] std::unique_ptr<cudf::column> operator()(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const override
  {
    auto const group_values = get_grouped_values();
    if (group_values.size() == 0) { return get_empty_output(stream, mr); }
    int num_groups          = get_num_groups();
    auto const group_labels = get_group_labels();
    if (is_merge) {
      // group by intermidate result, group_values are struct of long columns
      return spark_rapids_jni::group_merge_hyper_log_log_plus_plus(
        group_values, num_groups, group_labels, precision, stream, mr);
    } else {
      return spark_rapids_jni::group_hyper_log_log_plus_plus(
        group_values, num_groups, group_labels, precision, stream, mr);
    }
  }

  /**
   * @brief Create an empty column when the input is empty.
   */
  [[nodiscard]] std::unique_ptr<cudf::column> get_empty_output(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const override
  {
    int num_registers       = 1 << precision;
    int num_long_cols       = num_registers / REGISTERS_PER_LONG + 1;
    auto const results_iter = spark_rapids_jni::util::make_counting_transform_iterator(
      0, [&](int i) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}); });
    auto children =
      std::vector<std::unique_ptr<cudf::column>>(results_iter, results_iter + num_long_cols);
    return cudf::make_structs_column(0,
                                     std::move(children),
                                     0,                     // null count
                                     rmm::device_buffer{},  // null mask
                                     stream,
                                     mr);
  }

  [[nodiscard]] bool is_equal(cudf::host_udf_base const& other) const override
  {
    auto o = dynamic_cast<hllpp_groupby_udf const*>(&other);
    return o != nullptr && o->precision == this->precision && o->is_merge == this->is_merge;
  }

  [[nodiscard]] std::size_t do_hash() const override
  { return 31 * (31 * std::hash<std::string>{}({"hllpp_groupby_udf"}) + precision) + is_merge; }

  [[nodiscard]] std::unique_ptr<cudf::host_udf_base> clone() const override
  { return std::make_unique<hllpp_groupby_udf>(precision, is_merge); }

 private:
  int precision;
  bool is_merge;
};

struct hllpp_reduct_udf : cudf::reduce_host_udf {
  hllpp_reduct_udf(int precision_, bool is_merge_) : precision(precision_), is_merge(is_merge_) {}

  /**
   * @brief Create an empty scalar when the input is empty.
   */
  std::unique_ptr<cudf::scalar> get_empty_scalar(rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr) const
  {
    int num_registers       = 1 << precision;
    int num_long_cols       = num_registers / REGISTERS_PER_LONG + 1;
    auto const results_iter = spark_rapids_jni::util::make_counting_transform_iterator(
      0, [&](int i) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}); });
    auto children =
      std::vector<std::unique_ptr<cudf::column>>(results_iter, results_iter + num_long_cols);
    return std::make_unique<cudf::struct_scalar>(
      cudf::table{std::move(children)}, true, stream, mr);
  }

  /**
   * @brief Perform the main reduce computation for HLLPP UDF.
   */
  std::unique_ptr<cudf::scalar> operator()(
    cudf::column_view const& input,
    cudf::data_type,                                           /** output_dtype is useless */
    std::optional<std::reference_wrapper<cudf::scalar const>>, /** init is useless */
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const override
  {
    if (input.size() == 0) { return get_empty_scalar(stream, mr); }
    if (is_merge) {
      // input is struct of long columns
      return spark_rapids_jni::reduce_merge_hyper_log_log_plus_plus(input, precision, stream, mr);
    } else {
      return spark_rapids_jni::reduce_hyper_log_log_plus_plus(input, precision, stream, mr);
    }
  }

  [[nodiscard]] bool is_equal(cudf::host_udf_base const& other) const override
  {
    auto o = dynamic_cast<hllpp_reduct_udf const*>(&other);
    return o != nullptr && o->precision == this->precision && o->is_merge == this->is_merge;
  }

  [[nodiscard]] std::size_t do_hash() const override
  { return 31 * (31 * std::hash<std::string>{}({"hllpp_reduct_udf"}) + precision) + is_merge; }

  [[nodiscard]] std::unique_ptr<cudf::host_udf_base> clone() const override
  { return std::make_unique<hllpp_reduct_udf>(precision, is_merge); }

 private:
  int precision;
  bool is_merge;
};

}  // namespace

cudf::host_udf_base* create_hllpp_reduction_host_udf(int precision)
{ return new hllpp_reduct_udf(precision, /*is_merge*/ false); }

cudf::host_udf_base* create_hllpp_reduction_merge_host_udf(int precision)
{ return new hllpp_reduct_udf(precision, /*is_merge*/ true); }

cudf::host_udf_base* create_hllpp_groupby_host_udf(int precision)
{ return new hllpp_groupby_udf(precision, /*is_merge*/ false); }

cudf::host_udf_base* create_hllpp_groupby_merge_host_udf(int precision)
{ return new hllpp_groupby_udf(precision, /*is_merge*/ true); }

}  // namespace spark_rapids_jni
