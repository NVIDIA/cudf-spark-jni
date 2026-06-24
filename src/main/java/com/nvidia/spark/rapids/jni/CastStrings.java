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

package com.nvidia.spark.rapids.jni;

import java.time.LocalDate;
import java.time.ZoneId;

import ai.rapids.cudf.*;

/** Utility class for casting between string columns and native type columns */
public class CastStrings {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Convert a string column to an integer column of a specified type stripping away leading and
   * trailing spaces.
   *
   * @param cv the column data to process.
   * @param ansiMode true if invalid data are errors, false if they should be nulls.
   * @param type the type of the return column.
   * @return the converted column.
   */
  public static ColumnVector toInteger(ColumnView cv, boolean ansiMode, DType type) {
    return toInteger(cv, ansiMode, true, type);
  }

  /**
   * Convert a string column to an integer column of a specified type.
   *
   * @param cv the column data to process.
   * @param ansiMode true if invalid data are errors, false if they should be nulls.
   * @param strip true if leading and trailing spaces should be ignored when parsing.
   * @param type the type of the return column.
   * @return the converted column.
   */
  public static ColumnVector toInteger(ColumnView cv, boolean ansiMode, boolean strip, DType type) {
    return new ColumnVector(toInteger(cv.getNativeView(), ansiMode, strip,
        type.getTypeId().getNativeId()));
  }

  /**
   * Convert a string column to an integer column of a specified type stripping away leading and
   * trailing whitespace.
   *
   * @param cv the column data to process.
   * @param ansiMode true if invalid data are errors, false if they should be nulls.
   * @param precision the output precision.
   * @param scale the output scale.
   * @return the converted column.
   */
  public static ColumnVector toDecimal(ColumnView cv, boolean ansiMode, int precision, int scale) {
    return toDecimal(cv, ansiMode, true, precision, scale);
  }

  /**
   * Convert a string column to an integer column of a specified type.
   *
   * @param cv the column data to process.
   * @param ansiMode true if invalid data are errors, false if they should be nulls.
   * @param strip true if leading and trailing white space should be stripped.
   * @param precision the output precision.
   * @param scale the output scale.
   * @return the converted column.
   */
  public static ColumnVector toDecimal(ColumnView cv, boolean ansiMode, boolean strip,
      int precision, int scale) {
    return new ColumnVector(toDecimal(cv.getNativeView(), ansiMode, strip, precision, scale));
  }

  /**
   * Convert a float column to a formatted string column.
   *
   * @param cv the column data to process
   * @param digits the number of digits to display after the decimal point
   * @return the converted column
   */
  public static ColumnVector fromFloatWithFormat(ColumnView cv, int digits) {
    return new ColumnVector(fromFloatWithFormat(cv.getNativeView(), digits));
  }

  /**
   * Convert a float or double column to a string column.
   *
   * @param cv the column data to process
   * @return the converted column
   */
  public static ColumnVector fromFloat(ColumnView cv) {
    return fromFloat(cv, false);
  }

  /**
   * Convert a float or double column to a string column.
   *
   * @param cv the column data to process
   * @param jsonString true if non-finite values should include JSON quotes, e.g. {@code "NaN"}.
   * @return the converted column
   */
  public static ColumnVector fromFloat(ColumnView cv, boolean jsonString) {
    return new ColumnVector(fromFloat(cv.getNativeView(), jsonString));
  }

  /**
   * Convert a decimal column to a string column.
   *
   * @param cv the column data to process
   * @return the converted column
   */
  public static ColumnVector fromDecimal(ColumnView cv) {
    return new ColumnVector(fromDecimal(cv.getNativeView()));
  }

  public static ColumnVector fromLongToBinary(ColumnView cv) {
    return new ColumnVector(fromLongToBinary(cv.getNativeView()));
  }

  /**
   * Convert a string column to a given floating-point type column.
   *
   * @param cv the column data to process.
   * @param ansiMode true if invalid data are errors, false if they should be nulls.
   * @param type the type of the return column.
   * @return the converted column.
   */
  public static ColumnVector toFloat(ColumnView cv, boolean ansiMode, DType type) {
    return new ColumnVector(toFloat(cv.getNativeView(), ansiMode, type.getTypeId().getNativeId()));
  }


  public static ColumnVector toIntegersWithBase(ColumnView cv, int base,
    boolean ansiEnabled, DType type) {
    return new ColumnVector(toIntegersWithBase(cv.getNativeView(), base, ansiEnabled,
      type.getTypeId().getNativeId()));
  }

  /**
   * Converts an integer column to a string column by converting the underlying integers to the
   * specified base.
   *
   * Note: Right now we only support base 10 and 16. The hexadecimal values will be
   * returned without leading zeros or padding at the end
   * 
   * Example:
   * input = [123, -1, 0, 27, 342718233]
   * s = fromIntegersWithBase(input, 16)
   * s is [ '7B', 'FFFFFFFF', '0', '1B', '146D7719']
   * s = fromIntegersWithBase(input, 10)
   * s is ['123', '-1', '0', '27', '342718233']
   *
   * @param cv The input integer column to be converted.
   * @param base base that we want to convert to (currently only 10/16)
   * @return a new String ColumnVector
   */
  public static ColumnVector fromIntegersWithBase(ColumnView cv, int base) {
    return new ColumnVector(fromIntegersWithBase(cv.getNativeView(), base));
  }

  /**
   * Convert a strings/binary column to its hexadecimal representation.
   * Each byte is converted to a 2-character uppercase hex string.
   * For example, "AB" (bytes 0x41, 0x42) becomes "4142".
   *
   * @param cv The input strings column
   * @return a new String ColumnVector with hex representation
   */
  public static ColumnVector bytesToHex(ColumnView cv) {
    return new ColumnVector(bytesToHex(cv.getNativeView()));
  }

  /**
   * Trims and parses strings to intermediate result.
   * This is the first phase of casting string with timezone to timestamp.
   *
   * Intermediate result is a struct column with 6 sub-columns:
   * - Parse Result type: 0 Success, 1 invalid e.g. year is 7 digits 1234567
   * - seconds part of parsed UTC timestamp, it's from yyyy-mm-dd hh:mm:ss part of input string
   * - microseconds part of parsed UTC timestamp, it's from sub-second part of input string
   * - Timezone type: 0 unspecified, 1 fixed type, 2 other type, 3 invalid
   * - Timezone offset for fixed type, only applies to fixed type
   * - Timezone index to `GpuTimeZoneDB.getTimezoneInfo` table
   *
   * Note, need to use two columns (seconds and microseconds) to represent
   * the parsed timestamp string for a corner case:
   * The parsed timestamp is overflow for int64 integer, but after
   * rebasing according to timezone, it is not overflow.
   * This is a very rare case, but we need to handle it.
   *
   * @param input The input String column contains timestamp strings
   * @param defaultTimeZoneIndex The default timezone index to `GpuTimeZoneDB`
   *   timezone info table.
   * @param defaultEpochDay Default epoch day to use if just time, it's current date.
   * @param tzNameToIndexMap Timezone name to row index of timezone info table.
   *   Refer to `GpuTimeZoneDB.getTimezoneInfo` for more details.
   * @param timezoneInfo Timezone info table contains fixed-transitions and DST rules.
   * @param sparkVersion Spark version
   * @return a struct column contains 6 columns described above.
   */
  static ColumnVector parseTimestampStrings(
      ColumnView input,
      int defaultTimeZoneIndex,
      long defaultEpochDay,
      ColumnView tzNameToIndexMap,
      Table timezoneInfo,
      Version sparkVersion) {

    return new ColumnVector(parseTimestampStringsToIntermediate(
        input.getNativeView(), defaultTimeZoneIndex,
        defaultEpochDay, tzNameToIndexMap.getNativeView(), timezoneInfo.getNativeView(),
        sparkVersion.getPlatformOrdinal(), sparkVersion.getMajor(),
        sparkVersion.getMinor(), sparkVersion.getPatch()));
  }

  private static ColumnVector convertToTimestamp(
      long originInputNullCount,
      ColumnView invalid,
      ColumnView ts_seconds,
      ColumnView ts_microseconds,
      ColumnView tzType,
      ColumnView tzOffset,
      ColumnView tzIndex,
      boolean ansi_enabled) {
    ColumnVector result = GpuTimeZoneDB.fromTimestampToUtcTimestampWithTzCv(
          invalid, ts_seconds, ts_microseconds, tzType, tzOffset, tzIndex);

    try (ColumnVector tmp = result) {
      if (ansi_enabled && result.getNullCount() > originInputNullCount) {
        // has new nulls, means has any invalid data,
        // e.g.: format is invalid, year is not supported 7 digits
        // protocol: if ansi mode and has any invalid data, return null
        return null;
      } else {
        return tmp.incRefCount();
      }
    }
  }

  /**
   * Trims and parses a string column with timezones to timestamp.
   *
   * Refer to: https://github.com/apache/spark/blob/v3.5.0/sql/api/src/main/scala/
   * org/apache/spark/sql/catalyst/util/SparkDateTimeUtils.scala#L544
   *
   * Use the default timezone if timestamp string does not contain timezone.
   *
   * Supports the following formats:
   * `[+-]yyyy[y][y]`
   * `[+-]yyyy[y][y]-m[m]`
   * `[+-]yyyy[y][y]-m[m]-d[d]`
   * `[+-]yyyy[y][y]-m[m]-d[d] `
   * `[+-]yyyy[y][y]-m[m]-d[d] [h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
   * `[+-]yyyy[y][y]-m[m]-d[d]T[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
   *
   * The max length of yyyy[y][y] is 6 digits.
   *
   * Supports the following zone id forms:
   * - Z - Zulu timezone UTC+0
   * - +|-[h]h:[m]m
   * - A short id, see
   * https://docs.oracle.com/javase/8/docs/api/java/time/ZoneId.html#SHORT_IDS
   * - An id with one of the prefixes UTC+, UTC-, GMT+, GMT-, UT+ or UT-,
   * and a suffix in the formats:
   * - +|-h[h]
   * - +|-hh[:]mm
   * - +|-hh:mm:ss
   * - +|-hhmmss
   * - Region-based zone IDs in the form `area/city`, such as `Europe/Paris`
   *
   * @param input The input String column contains timestamp strings
   * @param defaultTimeZone The default timezone to be used. If there is no timezone in string,
   *   use this default timezone
   * @param ansi_enabled Throw exception if has any invalid data and ansi_enabled is true,
   * otherwise return null
   * @param sparkVersion Spark version
   * @return a timestamp column, or null if it has any invalid data and ansi_enabled is true
   */
  public static ColumnVector toTimestamp(
      ColumnView input,
      String defaultTimeZone,
      boolean ansi_enabled,
      Version sparkVersion) {

    // 1. check default timezone is valid
    Integer defaultTimeZoneIndex = GpuTimeZoneDB.getIndexToTransitionTable(defaultTimeZone);
    if (defaultTimeZoneIndex == null) {
      throw new IllegalArgumentException("Invalid default timezone: " + defaultTimeZone);
    }
    // Get the epoch day of today in default timezone, when the input string is just
    // time, e.g.: "T00:00:00", use this epoch day
    long defaultEpochDay = LocalDate.now(ZoneId.of(defaultTimeZone)).toEpochDay();

    // 2. parse to intermediate result
    try (ColumnVector tzNameToIndexMap = GpuTimeZoneDB.getTzNameToIndexMap();
        Table timezoneInfo = GpuTimeZoneDB.getTimezoneInfo();
        ColumnVector parseResult = parseTimestampStrings(
            input, defaultTimeZoneIndex, defaultEpochDay, tzNameToIndexMap,
            timezoneInfo, sparkVersion);
        ColumnView invalid = parseResult.getChildColumnView(0);
        ColumnView tsSeconds = parseResult.getChildColumnView(1);
        ColumnView tsMicroseconds = parseResult.getChildColumnView(2);
        ColumnView tzType = parseResult.getChildColumnView(3);
        ColumnView tzOffset = parseResult.getChildColumnView(4);
        ColumnView tzIndex = parseResult.getChildColumnView(5)) {

      // 3. convert to timestamp on GPU
      return convertToTimestamp(input.getNullCount(), invalid, tsSeconds, tsMicroseconds,
          tzType, tzOffset, tzIndex, ansi_enabled);
    }
  }

  /**
   * Parse date string column to date column, first trim the input strings.
   * Refer to https://github.com/apache/spark/blob/v3.5.0/sql/api/src/main/scala/
   * org/apache/spark/sql/catalyst/util/SparkDateTimeUtils.scala#L298
   *
   * Allowed formats:
   *   `[+-]yyyy*`
   *   `[+-]yyyy*-[m]m`
   *   `[+-]yyyy*-[m]m-[d]d`
   *   `[+-]yyyy*-[m]m-[d]d `
   *   `[+-]yyyy*-[m]m-[d]d *`
   *   `[+-]yyyy*-[m]m-[d]dT*`
   *
   * @param input        the input date strings
   * @param ansiEnabled is Ansi mode enabled
   * @return date column, or null if it's ansi mode and has invalid input values.
   */
  public static ColumnVector toDate(ColumnView input, boolean ansiEnabled) {
    try (ColumnVector result = new ColumnVector(parseDateStringsToDate(input.getNativeView()))) {
      if (ansiEnabled && result.getNullCount() > input.getNullCount()) {
        // has new nulls, means has any invalid data,
        // e.g.: format is invalid, year is out of range.
        // protocol: if ansi mode and has any invalid data, return null
        return null;
      } else {
        return result.incRefCount();
      }
    }
  }

  /**
   * Parse a string column into a {@code timestamp_us} column for a Spark date/timestamp
   * format pattern. Sub-second digits are not parsed; successfully parsed rows always have
   * microsecond field zero. The pattern is compiled into a token stream on the host and
   * walked per-row on the device, mirroring how Spark's {@code DateTimeFormatter} /
   * {@code SimpleDateFormat} represent a parser internally.
   *
   * <p>Pattern letters follow JDK conventions: {@code y}, {@code M}, {@code d}, {@code H},
   * {@code m}, {@code s}. Lowercase {@code m} is minute, not month. Non-year letter runs
   * must have length 2; longer runs (e.g. {@code MMM} for month name) are rejected because
   * this kernel does not implement text forms. Unsupported letters, including lowercase
   * {@code h} for 12-hour clock, {@code S} for fractional seconds, and timezone pattern
   * letters, are rejected. Space matches space or 'T' (Spark's
   * permissive date/time separator); quoted literals ({@code 'T'}) are not supported, use
   * a space instead. Pattern literals must be ASCII. In LEGACY mode, non-year digit fields
   * accept 1 or 2 digits unless adjacent to another digit field (which forces exact width
   * for boundary disambiguation), and the trailing tail accepts EOF or any non-digit.
   * Parsed values are wall-clock UTC; timezone rebasing remains the caller's responsibility.
   *
   * @param input the input string column.
   * @param format Spark format pattern (e.g. {@code "yyyy-MM-dd HH:mm:ss"}).
   * @param legacy true for {@code LegacyTimeParserPolicy}, false for CORRECTED/EXCEPTION.
   * @return a timestamp_us column where invalid rows have nulls.
   */
  public static ColumnVector parseTimestampWithFormat(ColumnView input, String format,
      boolean legacy) {
    return new ColumnVector(parseTimestampWithFormat(input.getNativeView(), format, legacy));
  }

  private static native long toInteger(long nativeColumnView, boolean ansi_enabled, boolean strip,
      int dtype);
  private static native long toDecimal(long nativeColumnView, boolean ansi_enabled, boolean strip,
      int precision, int scale);
  private static native long toFloat(long nativeColumnView, boolean ansi_enabled, int dtype);
  private static native long fromDecimal(long nativeColumnView);
  private static native long fromFloatWithFormat(long nativeColumnView, int digits);
  private static native long fromFloat(long nativeColumnView, boolean jsonString);
  private static native long fromLongToBinary(long nativeColumnView);
  private static native long toIntegersWithBase(long nativeColumnView, int base,
    boolean ansiEnabled, int dtype);
  private static native long fromIntegersWithBase(long nativeColumnView, int base);
  private static native long bytesToHex(long nativeColumnView);

  private static native long parseTimestampStringsToIntermediate(
      long input, int defaultTimezoneIndex,
      long defaultEpochDay, long tzNameToIndexMap, long tzInfo, int platformOrdinal,
      int sparkMajor, int sparkMinor, int sparkPatch);

  private static native long parseDateStringsToDate(long input);

  private static native long parseTimestampWithFormat(long input, String format, boolean legacy);

}
