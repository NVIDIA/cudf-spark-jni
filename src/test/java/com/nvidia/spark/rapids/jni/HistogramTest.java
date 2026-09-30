/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static com.nvidia.spark.rapids.jni.Histogram.PercentileInterpolation.ENDPOINT_DELTA;
import static com.nvidia.spark.rapids.jni.Histogram.PercentileInterpolation.WEIGHTED_ENDPOINTS;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class HistogramTest {
  private static final HostColumnVector.ListType DOUBLE_HISTOGRAM_TYPE =
      histogramType(DType.FLOAT64);
  private static final HostColumnVector.ListType FLOAT_HISTOGRAM_TYPE =
      histogramType(DType.FLOAT32);

  private static HostColumnVector.ListType histogramType(DType valueType) {
    HostColumnVector.StructType entryType = new HostColumnVector.StructType(false,
        new HostColumnVector.BasicType(true, valueType),
        new HostColumnVector.BasicType(false, DType.INT64));
    return new HostColumnVector.ListType(true, entryType);
  }

  private static HostColumnVector.StructData entry(Object value, long frequency) {
    return new HostColumnVector.StructData(value, frequency);
  }

  @SafeVarargs
  private static ColumnVector histograms(HostColumnVector.ListType type,
                                         List<HostColumnVector.StructData>... rows) {
    return ColumnVector.fromLists(type, rows);
  }

  private static void assertDoubleColumnExactly(ColumnVector actual, double... expected) {
    try (HostColumnVector host = actual.copyToHost()) {
      assertEquals(expected.length, host.getRowCount());
      for (int i = 0; i < expected.length; i++) {
        assertEquals(Double.doubleToLongBits(expected[i]),
            Double.doubleToLongBits(host.getDouble(i)), "row " + i);
      }
    }
  }

  @Test
  void testZeroFrequency() {
    try (ColumnVector values = ColumnVector.fromInts(5, 10, 30);
         ColumnVector freqs = ColumnVector.fromLongs(1, 0, 1);
         ColumnVector histogram = Histogram.createHistogramIfValid(values, freqs, true);
         ColumnVector percentiles = Histogram.percentileFromHistogram(histogram, new double[]{1},
             false);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(5.0, null, 30.0)) {
      AssertUtils.assertColumnsAreEqual(percentiles, expected);
    }
  }

  @Test
  void testAllNulls() {
    try (ColumnVector values = ColumnVector.fromBoxedInts(null, null, null);
         ColumnVector freqs = ColumnVector.fromLongs(1, 2, 3);
         ColumnVector histogram = Histogram.createHistogramIfValid(values, freqs, true);
         ColumnVector percentiles = Histogram.percentileFromHistogram(histogram, new double[]{0.5},
             false);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(null, null, null)) {
      AssertUtils.assertColumnsAreEqual(percentiles, expected);
    }
  }

  @Test
  void testInterpolationModesAcrossNegativeInfinity() {
    List<HostColumnVector.StructData> row =
        Arrays.asList(entry(Double.NEGATIVE_INFINITY, 1L), entry(10.0, 1L));
    try (ColumnVector histogram = histograms(DOUBLE_HISTOGRAM_TYPE, row);
         ColumnVector defaultResult = Histogram.percentileFromHistogram(
             histogram, new double[]{0.2}, false);
         ColumnVector weightedResult = Histogram.percentileFromHistogram(
             histogram, new double[]{0.2}, false, WEIGHTED_ENDPOINTS);
         ColumnVector deltaResult = Histogram.percentileFromHistogram(
             histogram, new double[]{0.2}, false, ENDPOINT_DELTA)) {
      assertDoubleColumnExactly(defaultResult, Double.NEGATIVE_INFINITY);
      assertDoubleColumnExactly(weightedResult, Double.NEGATIVE_INFINITY);
      assertDoubleColumnExactly(deltaResult, Double.NaN);
    }
  }

  @Test
  void testEndpointDeltaSpecialValues() {
    List<HostColumnVector.StructData> finiteToPositiveInfinity =
        Arrays.asList(entry(10.0, 1L), entry(Double.POSITIVE_INFINITY, 1L));
    List<HostColumnVector.StructData> negativeToPositiveInfinity =
        Arrays.asList(entry(Double.NEGATIVE_INFINITY, 1L),
            entry(Double.POSITIVE_INFINITY, 1L));
    List<HostColumnVector.StructData> finiteToNan =
        Arrays.asList(entry(1.0, 1L), entry(Double.NaN, 1L));
    List<HostColumnVector.StructData> equalEndpoints =
        Arrays.asList(entry(7.0, 1L), entry(7.0, 1L));
    try (ColumnVector histogram = histograms(DOUBLE_HISTOGRAM_TYPE,
             finiteToPositiveInfinity, negativeToPositiveInfinity, finiteToNan, equalEndpoints);
         ColumnVector result = Histogram.percentileFromHistogram(
             histogram, new double[]{0.2}, false, ENDPOINT_DELTA)) {
      assertDoubleColumnExactly(result,
          Double.POSITIVE_INFINITY, Double.NaN, Double.NaN, 7.0);
    }
  }

  @Test
  void testEndpointDeltaIntegralPositions() {
    List<HostColumnVector.StructData> row =
        Arrays.asList(entry(Double.NEGATIVE_INFINITY, 1L), entry(10.0, 1L));
    try (ColumnVector histogram = histograms(DOUBLE_HISTOGRAM_TYPE, row);
         ColumnVector result = Histogram.percentileFromHistogram(
             histogram, new double[]{0.0, 1.0}, false, ENDPOINT_DELTA)) {
      assertDoubleColumnExactly(result, Double.NEGATIVE_INFINITY, 10.0);
    }
  }

  @Test
  void testEndpointDeltaFloatInput() {
    List<HostColumnVector.StructData> row =
        Arrays.asList(entry(Float.NEGATIVE_INFINITY, 1L), entry(10.0f, 1L));
    try (ColumnVector histogram = histograms(FLOAT_HISTOGRAM_TYPE, row);
         ColumnVector result = Histogram.percentileFromHistogram(
             histogram, new double[]{0.2}, false, ENDPOINT_DELTA)) {
      assertDoubleColumnExactly(result, Double.NaN);
    }
  }

  @Test
  void testWeightedEndpointsDoesNotFuseMultiplyAdd() {
    List<HostColumnVector.StructData> row =
        Arrays.asList(entry(-100.0, 1L), entry(-99.0, 1L));
    try (ColumnVector histogram = histograms(DOUBLE_HISTOGRAM_TYPE, row);
         ColumnVector result = Histogram.percentileFromHistogram(
             histogram, new double[]{0.21}, false, WEIGHTED_ENDPOINTS)) {
      // Separately rounding both products before adding differs from either possible FMA
      // by one ULP.
      assertDoubleColumnExactly(result, Double.longBitsToDouble(0xc058f28f5c28f5c2L));
    }
  }

  @Test
  void testEndpointDeltaDoesNotFuseMultiplyAdd() {
    List<HostColumnVector.StructData> row =
        Arrays.asList(entry(-100.0, 1L), entry(-99.3, 1L));
    try (ColumnVector histogram = histograms(DOUBLE_HISTOGRAM_TYPE, row);
         ColumnVector result = Histogram.percentileFromHistogram(
             histogram, new double[]{0.1}, false, ENDPOINT_DELTA)) {
      // Spark's separate multiply and add differs from fused multiply-add by one ULP.
      assertDoubleColumnExactly(result, Double.longBitsToDouble(0xc058fb851eb851ecL));
    }
  }

  @Test
  void testEndpointDeltaIsMonotonicForLargeFiniteValues() {
    HostColumnVector.StructData[] entries = new HostColumnVector.StructData[20];
    for (int i = 0; i < entries.length; i++) {
      entries[i] = entry(1e18 + i * 128.0, 1L);
    }
    List<HostColumnVector.StructData> row = Arrays.asList(entries);
    try (ColumnVector histogram = histograms(DOUBLE_HISTOGRAM_TYPE, row);
         ColumnVector weightedResult = Histogram.percentileFromHistogram(
             histogram, new double[]{0.04, 0.05}, false, WEIGHTED_ENDPOINTS);
         ColumnVector deltaResult = Histogram.percentileFromHistogram(
             histogram, new double[]{0.04, 0.05}, false, ENDPOINT_DELTA)) {
      assertDoubleColumnExactly(weightedResult, 1.00000000000000013E18, 1.0E18);
      assertDoubleColumnExactly(deltaResult,
          1.00000000000000013E18, 1.00000000000000013E18);
    }
  }
}
