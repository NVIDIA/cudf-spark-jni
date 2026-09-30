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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.NativeDepsLoader;

public class ParseURI {
  static {
    NativeDepsLoader.loadNativeDeps();
  }


  /**
   * Parse protocol for each URI from the incoming column.
   *
   * @param URIColumn The input strings column in which each row contains a URI.
   * @return A string column with protocol data extracted.
   * This overload is retained temporarily. Use {@link #parseURIProtocol(ColumnView, boolean)}
   * instead; remove this overload after all callers migrate.
   */
  public static ColumnVector parseURIProtocol(ColumnView uriColumn) {
    return parseURIProtocol(uriColumn, false);
  }

  /**
   * Parse protocol for each URI from the incoming column.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param failOnError If true, use ANSI-aware parsing.
   * @return A string column with protocol data extracted.
   */
  public static ColumnVector parseURIProtocol(ColumnView uriColumn, boolean failOnError) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    return new ColumnVector(parseProtocol(uriColumn.getNativeView(), failOnError));
  }

  /**
   * Parse host for each URI from the incoming column.
   *
   * @param URIColumn The input strings column in which each row contains a URI.
   * @return A string column with host data extracted.
   * This overload is retained temporarily. Use {@link #parseURIHost(ColumnView, boolean)} instead;
   * remove this overload after all callers migrate.
   */
  public static ColumnVector parseURIHost(ColumnView uriColumn) {
    return parseURIHost(uriColumn, false);
  }

  /**
   * Parse host for each URI from the incoming column.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param failOnError If true, use ANSI-aware parsing.
   * @return A string column with host data extracted.
   */
  public static ColumnVector parseURIHost(ColumnView uriColumn, boolean failOnError) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    return new ColumnVector(parseHost(uriColumn.getNativeView(), failOnError));
  }

  /**
   * Parse query for each URI from the incoming column.
   *
   * @param URIColumn The input strings column in which each row contains a URI.
   * @return A string column with query data extracted.
   * This overload is retained temporarily. Use {@link #parseURIQuery(ColumnView, boolean)}
   * instead; remove this overload after all callers migrate.
   */
  public static ColumnVector parseURIQuery(ColumnView uriColumn) {
    return parseURIQuery(uriColumn, false);
  }

  /**
   * Parse query for each URI from the incoming column.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param failOnError If true, throw exception on invalid URLs.
   * @return A string column with query data extracted.
   */
  public static ColumnVector parseURIQuery(ColumnView uriColumn, boolean failOnError) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    return new ColumnVector(parseQuery(uriColumn.getNativeView(), failOnError));
  }

  /**
   * Parse query and return a specific parameter for each URI from the incoming column.
   *
   * @param URIColumn The input strings column in which each row contains a URI.
   * @param String The parameter to extract from the query.
   * @return A string column with query data extracted.
   * This overload is retained temporarily. Use
   * {@link #parseURIQueryWithLiteral(ColumnView, String, boolean)} instead; remove this overload
   * after all callers migrate.
   */
  public static ColumnVector parseURIQueryWithLiteral(ColumnView uriColumn, String query) {
    return parseURIQueryWithLiteral(uriColumn, query, false);
  }

  /**
   * Parse query and return a specific parameter for each URI from the incoming column.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param query The parameter to extract from the query.
   * @param failOnError If true, use ANSI-aware parsing.
   * @return A string column with query data extracted.
   */
  public static ColumnVector parseURIQueryWithLiteral(ColumnView uriColumn, String query, boolean failOnError) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    return new ColumnVector(parseQueryWithLiteral(uriColumn.getNativeView(), query, failOnError));
  }

  /**
   * Parse query and return a specific parameter for each URI from the incoming column.
   *
   * @param URIColumn The input strings column in which each row contains a URI.
   * @param String The parameter to extract from the query.
   * @return A string column with query data extracted.
   * This overload is retained temporarily. Use
   * {@link #parseURIQueryWithColumn(ColumnView, ColumnView, boolean)} instead; remove this overload
   * after all callers migrate.
   */
  public static ColumnVector parseURIQueryWithColumn(ColumnView uriColumn, ColumnView queryColumn) {
    return parseURIQueryWithColumn(uriColumn, queryColumn, false);
  }

  /**
   * Parse query and return a specific parameter for each URI from the incoming column.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param queryColumn The parameter to extract from the query.
   * @param failOnError If true, use ANSI-aware parsing.
   * @return A string column with query data extracted.
   */
  public static ColumnVector parseURIQueryWithColumn(ColumnView uriColumn, ColumnView queryColumn, boolean failOnError) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    assert queryColumn.getType().equals(DType.STRING) : "Query type must be String";
    return new ColumnVector(parseQueryWithColumn(uriColumn.getNativeView(), queryColumn.getNativeView(), failOnError));
  }

  /**
   * Parse path for each URI from the incoming column.
   *
   * @param URIColumn The input strings column in which each row contains a URI.
   * @return A string column with the URI path extracted.
   * This overload is retained temporarily. Use {@link #parseURIPath(ColumnView, boolean)} instead;
   * remove this overload after all callers migrate.
   */
  public static ColumnVector parseURIPath(ColumnView uriColumn) {
    return parseURIPath(uriColumn, false);
  }

  /**
   * Parse path for each URI from the incoming column.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param failOnError If true, use ANSI-aware parsing.
   * @return A string column with the URI path extracted.
   */
  public static ColumnVector parseURIPath(ColumnView uriColumn, boolean failOnError) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    return new ColumnVector(parsePath(uriColumn.getNativeView(), failOnError));
  }

  private static native long parseProtocol(long inputColumnHandle, boolean ansiMode);
  private static native long parseHost(long inputColumnHandle, boolean ansiMode);
  private static native long parseQuery(long inputColumnHandle, boolean ansiMode);
  private static native long parseQueryWithLiteral(long inputColumnHandle, String query, boolean ansiMode);
  private static native long parseQueryWithColumn(long inputColumnHandle, long queryColumnHandle, boolean ansiMode);
  private static native long parsePath(long inputColumnHandle, boolean ansiMode);
}
