/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ai.rapids.cudf.*;

import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.DayOfWeek;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.Month;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.temporal.ChronoField;
import java.time.zone.ZoneOffsetTransition;
import java.time.zone.ZoneOffsetTransitionRule;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.List;
import java.util.Random;
import java.util.TimeZone;
import java.util.concurrent.TimeUnit;

public class GpuTimeZoneDBTest {

  private static final long microsPerMillis = TimeUnit.MILLISECONDS.toMicros(1);
  private static final long MICROS_PER_SECOND = TimeUnit.SECONDS.toMicros(1);

  private static TimeZone getTimeZoneForOrc(String timezoneId) {
    return TimeZone.getTimeZone(GpuTimeZoneDB.getZoneId(timezoneId));
  }

  private static long orc2015YearBaseOffsetUs(String timezoneId) {
    OrcTimezoneInfo info = OrcTimezoneInfo.get(timezoneId);
    if (info.transitions == null && info.dstRule == null) {
      return TimeUnit.MILLISECONDS.toMicros(info.rawOffset);
    }
    TimeZone tz = getTimeZoneForOrc(timezoneId);
    return TimeUnit.MILLISECONDS.toMicros(
        tz.getOffset(OrcTimezoneInfo.utcMillisForDate(2015, 1, 1)));
  }

  private static long applyOrcBaseOffsetOnCPU(long decodedUs, long baseOffsetUs) {
    if (baseOffsetUs == 0) {
      return decodedUs;
    }

    // ORC timezone base offsets are second-aligned. For an arbitrary microsecond offset, the
    // original nanos field cannot be reconstructed reliably, so retain the plain offset behavior.
    if (baseOffsetUs % MICROS_PER_SECOND != 0) {
      return decodedUs - baseOffsetUs;
    }

    long fractionalUs = Math.floorMod(decodedUs, MICROS_PER_SECOND);
    boolean hasBorrowableFraction = fractionalUs >= microsPerMillis;
    boolean cudfAppliedBorrow = decodedUs < 0 && hasBorrowableFraction;

    long unborrowedUs = decodedUs + (cudfAppliedBorrow ? MICROS_PER_SECOND : 0L);
    long adjustedUnborrowedUs = unborrowedUs - baseOffsetUs;
    boolean apacheAppliesBorrow = adjustedUnborrowedUs < 0 && hasBorrowableFraction;

    return adjustedUnborrowedUs - (apacheAppliesBorrow ? MICROS_PER_SECOND : 0L);
  }

  private static long[] getFutureDstBoundaryMicros(String timezoneId) {
    List<ZoneOffsetTransitionRule> rules =
        ZoneId.of(timezoneId).getRules().getTransitionRules();
    assertEquals(2, rules.size(), "expected two recurring DST rules for " + timezoneId);
    long[] microseconds = new long[rules.size() * 3];
    int index = 0;
    for (ZoneOffsetTransitionRule rule : rules) {
      ZoneOffsetTransition transition = rule.createTransition(9999);
      long transitionMillis = transition.getInstant().toEpochMilli();
      microseconds[index++] = (transitionMillis - 1) * microsPerMillis;
      microseconds[index++] = transitionMillis * microsPerMillis;
      microseconds[index++] = (transitionMillis + 1) * microsPerMillis;
    }
    return microseconds;
  }

  /**
   * Java implementation of timezone conversion to compare against the GPU
   * results.
   * Refer to https://github.com/apache/orc/blob/rel/release-1.9.1/java/core/
   * src/java/org/apache/orc/impl/SerializationUtils.java#L1440
   */
  private static ColumnVector convertOrcTimezonesOnCPU(
      long[] microseconds,
      String writeTzId,
      String readerTzId) {
    long[] results = new long[microseconds.length];
    TimeZone writeTz = getTimeZoneForOrc(writeTzId);
    TimeZone readerTz = getTimeZoneForOrc(readerTzId);
    long writer2015YearBaseOffsetUs = orc2015YearBaseOffsetUs(writeTzId);
    for (int i = 0; i < microseconds.length; ++i) {
      long adjustedUs = applyOrcBaseOffsetOnCPU(microseconds[i], writer2015YearBaseOffsetUs);
      // Floor-divide µs to ms (and floor-mod for the sub-ms remainder) so reconstruction
      // round-trips for negative timestamps with a non-zero sub-millisecond component. Truncation
      // toward zero would round such an input up by one ms; at a DST gap transition that lands on
      // the post-transition offset, producing a 1-hour off-by-one. Must match the GPU kernel's
      // floor-divide in convert_timestamp_between_timezones.
      long millis = Math.floorDiv(adjustedUs, microsPerMillis);
      long writerOffset = writeTz.getOffset(millis);
      long readerOffset = readerTz.getOffset(millis);
      long adjustedMillis = millis + writerOffset - readerOffset;
      long adjustedReader = readerTz.getOffset(adjustedMillis);
      long finalDiffs = writerOffset - adjustedReader;
      results[i] =
          (millis + finalDiffs) * microsPerMillis + Math.floorMod(adjustedUs, microsPerMillis);
    }
    return ColumnVector.timestampMicroSecondsFromLongs(results);
  }

  private static long convertOrcTimezonesOnCPU(
      long microseconds, String writerTzId, String readerTzId) {
    TimeZone writerTz = getTimeZoneForOrc(writerTzId);
    TimeZone readerTz = getTimeZoneForOrc(readerTzId);
    long adjustedUs = applyOrcBaseOffsetOnCPU(
        microseconds, orc2015YearBaseOffsetUs(writerTzId));
    long millis = Math.floorDiv(adjustedUs, microsPerMillis);
    long writerOffset = writerTz.getOffset(millis);
    long readerOffset = readerTz.getOffset(millis);
    long adjustedMillis = millis + writerOffset - readerOffset;
    long adjustedReaderOffset = readerTz.getOffset(adjustedMillis);
    return adjustedUs + (writerOffset - adjustedReaderOffset) * microsPerMillis;
  }

  private static long rebaseOrcInstantToSparkOnCPU(long orcInstant, String readerTzId) {
    TimeZone readerTz = getTimeZoneForOrc(readerTzId);
    Calendar calendar = new Calendar.Builder()
        .setCalendarType("gregory")
        .setInstant(Math.floorDiv(orcInstant, microsPerMillis))
        .setTimeZone(readerTz)
        .build();
    LocalDateTime localDateTime = LocalDateTime.of(
        calendar.get(Calendar.YEAR),
        calendar.get(Calendar.MONTH) + 1,
        1,
        calendar.get(Calendar.HOUR_OF_DAY),
        calendar.get(Calendar.MINUTE),
        calendar.get(Calendar.SECOND),
        Math.toIntExact(Math.floorMod(orcInstant, MICROS_PER_SECOND)
            * TimeUnit.MICROSECONDS.toNanos(1)))
        .with(ChronoField.ERA, calendar.get(Calendar.ERA))
        .plusDays(calendar.get(Calendar.DAY_OF_MONTH) - 1L);
    ZoneId readerZone = GpuTimeZoneDB.getZoneId(readerTzId);
    java.time.ZonedDateTime zonedDateTime = localDateTime.atZone(readerZone);
    ZoneOffsetTransition transition = readerZone.getRules().getTransition(localDateTime);
    if (transition != null && transition.isOverlap()) {
      int zoneOffset = calendar.get(Calendar.ZONE_OFFSET);
      int dstOffset = calendar.get(Calendar.DST_OFFSET);
      calendar.add(Calendar.DAY_OF_MONTH, 1);
      if (zoneOffset == calendar.get(Calendar.ZONE_OFFSET)
          && dstOffset == calendar.get(Calendar.DST_OFFSET)) {
        zonedDateTime = zonedDateTime.withLaterOffsetAtOverlap();
      } else {
        zonedDateTime = zonedDateTime.withEarlierOffsetAtOverlap();
      }
    }
    return zonedDateTime.toEpochSecond() * MICROS_PER_SECOND
        + Math.floorMod(orcInstant, MICROS_PER_SECOND);
  }

  private static long convertPhysicalOrcTimestampToSparkOnCPU(
      long microseconds, String writerTzId, String readerTzId) {
    long orcInstant = convertOrcTimezonesOnCPU(microseconds, writerTzId, readerTzId);
    return rebaseOrcInstantToSparkOnCPU(orcInstant, readerTzId);
  }

  private static long convertIntegerOrcTimestampToSparkOnCPU(
      long localMicros, String readerTzId) {
    TimeZone readerTz = getTimeZoneForOrc(readerTzId);
    long localMillis = Math.floorDiv(localMicros, microsPerMillis);
    int offsetMillis = readerTz.getOffset(localMillis - readerTz.getRawOffset());
    long orcInstant = (localMillis - offsetMillis) * microsPerMillis
        + Math.floorMod(localMicros, microsPerMillis);
    return rebaseOrcInstantToSparkOnCPU(orcInstant, readerTzId);
  }

  private static ColumnVector convertOrcFromUtcOnCPU(
      Long[] microseconds,
      String readerTzId) {
    Long[] results = new Long[microseconds.length];
    TimeZone readerTz = getTimeZoneForOrc(readerTzId);
    for (int i = 0; i < microseconds.length; ++i) {
      Long valueUs = microseconds[i];
      if (valueUs != null) {
        long valueMillis = Math.floorDiv(valueUs, microsPerMillis);
        int offsetMillis = readerTz.getOffset(valueMillis - readerTz.getRawOffset());
        results[i] = (valueMillis - offsetMillis) * microsPerMillis
            + Math.floorMod(valueUs, microsPerMillis);
      }
    }
    return ColumnVector.timestampMicroSecondsFromBoxedLongs(results);
  }

  private static Long[] getOrcFromUtcBoundaryMicros(String readerTzId) {
    long minSupportedUs = LocalDateTime.of(1, 1, 1, 0, 0)
        .toEpochSecond(ZoneOffset.UTC) * MICROS_PER_SECOND;
    long maxSupportedUs = LocalDateTime.of(9999, 12, 31, 23, 59, 59)
        .toEpochSecond(ZoneOffset.UTC) * MICROS_PER_SECOND + 999_999L;
    List<Long> values = new ArrayList<>(Arrays.asList(
        null,
        minSupportedUs,
        minSupportedUs + 1,
        -3_649_379_812_521_628L,
        -2_957_649_381_472_612L,
        -1_501L,
        -1_001L,
        -999L,
        -1L,
        0L,
        1L,
        999L,
        1_001L,
        514_952_012L,
        maxSupportedUs - 1,
        maxSupportedUs));

    OrcTimezoneInfo readerInfo = OrcTimezoneInfo.get(readerTzId);
    if (readerInfo.transitions != null) {
      for (long transitionMillis : readerInfo.transitions) {
        long localTransitionUs =
            TimeUnit.MILLISECONDS.toMicros(transitionMillis + readerInfo.rawOffset);
        values.add(localTransitionUs - 1);
        values.add(localTransitionUs);
        values.add(localTransitionUs + 1);
      }
    }

    for (ZoneOffsetTransitionRule rule :
        GpuTimeZoneDB.getZoneId(readerTzId).getRules().getTransitionRules()) {
      long transitionMillis = rule.createTransition(2099).getInstant().toEpochMilli();
      long localTransitionUs =
          TimeUnit.MILLISECONDS.toMicros(transitionMillis + readerInfo.rawOffset);
      values.add(localTransitionUs - 1);
      values.add(localTransitionUs);
      values.add(localTransitionUs + 1);
    }
    return values.toArray(new Long[0]);
  }

  @Test
  void testMidnightEndOfDayTransitionRule() {
    ZoneOffset standardOffset = ZoneOffset.ofHours(2);
    ZoneOffsetTransitionRule rule = ZoneOffsetTransitionRule.of(
        Month.MARCH,
        -1,
        DayOfWeek.THURSDAY,
        LocalTime.MIDNIGHT,
        true,
        ZoneOffsetTransitionRule.TimeDefinition.WALL,
        standardOffset,
        standardOffset,
        ZoneOffset.ofHours(3));

    assertEquals(24 * 3_600,
        GpuTimeZoneDB.getTransitionRuleTimeDiffComparedToMidnight(rule));
  }

  @Test
  void testIsSupportedTimeZone() {
    // Named zones with ZoneRules.
    assertTrue(GpuTimeZoneDB.isSupportedTimeZone("UTC"));
    assertTrue(GpuTimeZoneDB.isSupportedTimeZone("Asia/Shanghai"));

    // Unknown id.
    assertFalse(GpuTimeZoneDB.isSupportedTimeZone("Invalid/Zone"));

    // Offset-style ids: "+05:30" must be accepted; malformed offsets must be
    // rejected even when the parser throws DateTimeException rather than the
    // narrower ZoneRulesException. This is the regression the widened catch in
    // isSupportedTimeZone guards against.
    assertTrue(GpuTimeZoneDB.isSupportedTimeZone("+05:30"));
    assertFalse(GpuTimeZoneDB.isSupportedTimeZone("+25:00"));
  }

  @Test
  void testConvertOrcTimezonesRejectsInvalidId() {
    // Invalid timezone IDs must surface an exception rather than silently
    // falling back to GMT. We assert the broad RuntimeException type so this
    // stays a regression guard even if the exact wrapping is refactored later.
    GpuTimeZoneDB.cacheDatabase();
    try (ColumnVector input =
        ColumnVector.timestampMicroSecondsFromLongs(new long[] {0L})) {
      assertThrows(RuntimeException.class,
          () -> GpuTimeZoneDB.convertOrcTimezones(input, "Invalid/Zone", "UTC"));
    }
  }

  @Test
  void testConvertOrcTimezonesPreservesEmptyAndNulls() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    try (ColumnVector input =
            ColumnVector.timestampMicroSecondsFromBoxedLongs(new Long[] {});
        ColumnVector actual = GpuTimeZoneDB.convertOrcTimezones(input, "UTC", "UTC")) {
      assertColumnsAreEqual(input, actual);
    }

    try (ColumnVector input =
            ColumnVector.timestampMicroSecondsFromBoxedLongs(null, 0L, null);
        ColumnVector actual = GpuTimeZoneDB.convertOrcTimezones(input, "UTC", "UTC")) {
      assertColumnsAreEqual(input, actual);
    }
  }

  @Test
  void testConvertOrcTimezonesCorrectsIgnoredWriterTimezoneEpochBorrow() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    try (ColumnVector input =
            ColumnVector.timestampMicroSecondsFromLongs(new long[] {21_087_883_873L});
        ColumnVector expected =
            ColumnVector.timestampMicroSecondsFromLongs(new long[] {-7_713_116_127L});
        ColumnVector actual =
            GpuTimeZoneDB.convertOrcTimezones(input, "Asia/Shanghai", "Asia/Shanghai")) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testConvertOrcTimezonesFixedOffsetIds() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    long[] microseconds = {0L, -1L, 1L, -2_957_649_381_472_612L};
    String[][] cases = {
        {"UTC", "+05:30"},
        {"+05:30", "UTC"},
        {"UTC", "EST"},
        {"EST", "UTC"}
    };

    for (String[] timezones : cases) {
      try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(microseconds);
          ColumnVector expected =
              convertOrcTimezonesOnCPU(microseconds, timezones[0], timezones[1]);
          ColumnVector actual =
              GpuTimeZoneDB.convertOrcTimezones(input, timezones[0], timezones[1])) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }

  @Test
  void testConvertOrcFromUtcAllTimezones() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    List<String> timezones = Arrays.asList(
        "UTC",
        "America/New_York",
        "America/Vancouver",
        "America/Los_Angeles",
        "Europe/Paris",
        "Asia/Shanghai",
        "Australia/Sydney",
        "US/Pacific",
        "PST",
        "EST",
        "+05:30");

    for (String readerTzId : timezones) {
      Long[] values = getOrcFromUtcBoundaryMicros(readerTzId);
      Long[] padded = new Long[values.length + 2];
      padded[0] = 123L;
      System.arraycopy(values, 0, padded, 1, values.length);
      padded[padded.length - 1] = 456L;

      try (ColumnVector full = ColumnVector.timestampMicroSecondsFromBoxedLongs(padded);
          ColumnVector input = full.subVector(1, values.length + 1);
          ColumnVector expected = convertOrcFromUtcOnCPU(values, readerTzId);
          GpuTimeZoneDB.OrcTimezoneContext context =
              GpuTimeZoneDB.buildOrcTimezoneContext("UTC", readerTzId);
          ColumnVector fromContext = GpuTimeZoneDB.convertOrcFromUtc(input, context);
          ColumnVector fromTimezone = GpuTimeZoneDB.convertOrcFromUtc(input, readerTzId)) {
        assertColumnsAreEqual(expected, fromContext);
        assertColumnsAreEqual(expected, fromTimezone);
      }
    }

    try (ColumnVector empty =
            ColumnVector.timestampMicroSecondsFromBoxedLongs(new Long[] {});
        ColumnVector actual = GpuTimeZoneDB.convertOrcFromUtc(empty, "UTC")) {
      assertColumnsAreEqual(empty, actual);
    }
  }

  @Test
  void testConvertOrcFromUtcMillisecondsWithNullableSliceAndExtremeValues() {
    Long[] values = {123L, null, Long.MIN_VALUE, Long.MIN_VALUE + 1,
        -2_713_909_652_001L, -2_713_909_652_000L, -1L, 0L, 1L,
        1_600_000_000_000L, 7_258_118_400_000L, Long.MAX_VALUE - 1, Long.MAX_VALUE, 456L};
    for (String zone : new String[]{"UTC", "America/Vancouver", "America/New_York",
        "Asia/Shanghai", "Pacific/Port_Moresby"}) {
      TimeZone tz = getTimeZoneForOrc(zone);
      Long[] expected = new Long[values.length - 2];
      for (int i = 0; i < expected.length; i++) {
        Long value = values[i + 1];
        if (value != null) {
          // Java long arithmetic wraps at both the lookup and the offset subtraction.
          expected[i] = value - tz.getOffset(value - tz.getRawOffset());
        }
      }
      try (ColumnVector full = ColumnVector.timestampMilliSecondsFromBoxedLongs(values);
          CloseableArray<ColumnView> slices =
              CloseableArray.wrap(full.splitAsViews(1, values.length - 1));
          ColumnVector reference = ColumnVector.timestampMilliSecondsFromBoxedLongs(expected);
          GpuTimeZoneDB.OrcTimezoneContext context =
              GpuTimeZoneDB.buildOrcTimezoneContext(zone, zone);
          ColumnVector actual = GpuTimeZoneDB.convertOrcFromUtc(slices.get(1), context)) {
        assertColumnsAreEqual(reference, actual);
      }
    }
  }

  @Test
  void testRebaseRoundedOrcInstantWithNullableSlice() {
    GpuTimeZoneDB.cacheDatabase();
    // Include both sides of Vancouver's 1884 gap, the New York and Port Moresby overlaps,
    // and modern timestamps. These are already ORC-converted instants: no writer base or
    // negative-nanosecond borrow must be applied a second time.
    Long[] values = {123L, null, -2_713_880_852_002_000L, -2_713_880_852_001_000L,
        -2_713_880_852_000_000L, -2_713_880_851_999_000L,
        -2_717_650_900_000_000L, -2_840_176_800_000_000L,
        -1L, 0L, 1L, 1_600_000_000_123_000L, 7_258_118_400_123_000L, null, 456L};
    for (String zone : new String[]{"UTC", "America/Vancouver", "America/New_York",
        "Asia/Shanghai", "Pacific/Port_Moresby"}) {
      Long[] expected = new Long[values.length - 2];
      for (int i = 0; i < expected.length; i++) {
        if (values[i + 1] != null) {
          expected[i] = rebaseOrcInstantToSparkOnCPU(values[i + 1], zone);
        }
      }
      try (ColumnVector full = ColumnVector.timestampMicroSecondsFromBoxedLongs(values);
          CloseableArray<ColumnView> slices =
              CloseableArray.wrap(full.splitAsViews(1, values.length - 1));
          ColumnVector reference = ColumnVector.timestampMicroSecondsFromBoxedLongs(expected);
          GpuTimeZoneDB.OrcTimezoneContext context =
              GpuTimeZoneDB.buildOrcTimezoneContext(zone, zone);
          ColumnVector actual = GpuTimeZoneDB.rebaseOrcInstantToSpark(slices.get(1), context)) {
        assertColumnsAreEqual(reference, actual);
      }
    }
  }

  @Test
  void testOrcToSparkEmptyAndAllNullInputs() {
    GpuTimeZoneDB.cacheDatabase();
    for (String zone : new String[]{"UTC", "America/New_York", "Pacific/Port_Moresby"}) {
      for (Long[] values : new Long[][]{new Long[0], new Long[257]}) {
        try (ColumnVector input = ColumnVector.timestampMicroSecondsFromBoxedLongs(values);
            ColumnVector millis = ColumnVector.timestampMilliSecondsFromBoxedLongs(values);
            GpuTimeZoneDB.OrcTimezoneContext context =
                GpuTimeZoneDB.buildOrcTimezoneContext("UTC", zone);
            ColumnVector physical = GpuTimeZoneDB.convertOrcTimestampToSpark(input, context);
            ColumnVector local = GpuTimeZoneDB.convertOrcIntegerTimestampToSpark(input, context);
            ColumnVector instant = GpuTimeZoneDB.rebaseOrcInstantToSpark(input, context);
            ColumnVector convertedMillis = GpuTimeZoneDB.convertOrcFromUtc(millis, context)) {
          assertColumnsAreEqual(input, physical);
          assertColumnsAreEqual(input, local);
          assertColumnsAreEqual(input, instant);
          assertColumnsAreEqual(millis, convertedMillis);
        }
      }
    }
  }

  @Test
  void testOrcToSparkFixedOffsetInputs() {
    GpuTimeZoneDB.cacheDatabase();
    String timezoneId = "+05:30";
    long[] values = {-1_234_567L, 0L, 1_234_567L};
    long[] expectedPhysicalValues = new long[values.length];
    long[] expectedLocalValues = new long[values.length];
    for (int i = 0; i < values.length; i++) {
      expectedPhysicalValues[i] =
          convertPhysicalOrcTimestampToSparkOnCPU(values[i], timezoneId, timezoneId);
      expectedLocalValues[i] = convertIntegerOrcTimestampToSparkOnCPU(values[i], timezoneId);
    }

    try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(values);
        ColumnVector expectedPhysical =
            ColumnVector.timestampMicroSecondsFromLongs(expectedPhysicalValues);
        ColumnVector expectedLocal =
            ColumnVector.timestampMicroSecondsFromLongs(expectedLocalValues);
        GpuTimeZoneDB.OrcTimezoneContext context =
            GpuTimeZoneDB.buildOrcTimezoneContext(timezoneId, timezoneId);
        ColumnVector actualPhysical = GpuTimeZoneDB.convertOrcTimestampToSpark(input, context);
        ColumnVector actualLocal =
            GpuTimeZoneDB.convertOrcIntegerTimestampToSpark(input, context)) {
      assertEquals(Long.MIN_VALUE, context.getReaderHistoricalDifferenceEndUtcUs());
      assertEquals(Long.MIN_VALUE, context.getReaderHistoricalDifferenceEndLocalUs());
      assertColumnsAreEqual(expectedPhysical, actualPhysical);
      assertColumnsAreEqual(expectedLocal, actualLocal);
    }
  }

  @Test
  void testOrcTimezoneContextConversionFailures() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(0L)) {
      GpuTimeZoneDB.OrcTimezoneContext closed =
          GpuTimeZoneDB.buildOrcTimezoneContext("UTC", "UTC");
      closed.close();
      assertThrows(IllegalStateException.class,
          () -> GpuTimeZoneDB.convertOrcTimezones(input, closed));
      assertThrows(IllegalStateException.class,
          () -> GpuTimeZoneDB.convertOrcFromUtc(input, closed));
      assertThrows(IllegalStateException.class,
          () -> GpuTimeZoneDB.convertOrcTimestampToSpark(input, closed));
      assertThrows(IllegalStateException.class,
          () -> GpuTimeZoneDB.convertOrcIntegerTimestampToSpark(input, closed));
      assertThrows(IllegalStateException.class,
          () -> GpuTimeZoneDB.rebaseOrcInstantToSpark(input, closed));
    }

    try (ColumnVector input = ColumnVector.timestampSecondsFromLongs(0L);
        GpuTimeZoneDB.OrcTimezoneContext context =
            GpuTimeZoneDB.buildOrcTimezoneContext("UTC", "UTC")) {
      assertThrows(CudfException.class,
          () -> GpuTimeZoneDB.convertOrcTimezones(input, context));
      assertThrows(CudfException.class,
          () -> GpuTimeZoneDB.convertOrcFromUtc(input, context));
      assertThrows(CudfException.class,
          () -> GpuTimeZoneDB.convertOrcTimestampToSpark(input, context));
      assertThrows(CudfException.class,
          () -> GpuTimeZoneDB.convertOrcIntegerTimestampToSpark(input, context));
      assertThrows(CudfException.class,
          () -> GpuTimeZoneDB.rebaseOrcInstantToSpark(input, context));
    }
  }

  @Test
  void testConvertPhysicalOrcTimestampToSparkAcrossNewYorkLmtTransition() {
    GpuTimeZoneDB.cacheDatabase();
    String timezoneId = "America/New_York";
    long orcInstant = -2_717_655_100_076_025L;
    long decodedMicros = orcInstant + orc2015YearBaseOffsetUs(timezoneId);
    long expectedMicros = convertPhysicalOrcTimestampToSparkOnCPU(
        decodedMicros, timezoneId, timezoneId);
    assertEquals(-2_717_655_338_076_025L, expectedMicros);

    try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(decodedMicros);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromLongs(expectedMicros);
        GpuTimeZoneDB.OrcTimezoneContext context =
            GpuTimeZoneDB.buildOrcTimezoneContext(timezoneId, timezoneId);
        ColumnVector actual = GpuTimeZoneDB.convertOrcTimestampToSpark(input, context)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testConvertPhysicalOrcTimestampToSparkWithDifferentTimezones() {
    GpuTimeZoneDB.cacheDatabase();
    String[][] timezonePairs = {
        {"America/Los_Angeles", "America/New_York"},
        {"America/New_York", "America/Los_Angeles"},
        {"UTC", "America/New_York"},
        {"America/New_York", "Pacific/Port_Moresby"},
        // Gaza's large transition table exercises global-memory and mixed staging paths.
        {"Asia/Gaza", "America/New_York"},
        {"America/New_York", "Asia/Gaza"},
        {"UTC", "Asia/Gaza"},
        {"Asia/Gaza", "Pacific/Port_Moresby"}
    };
    String[] instants = {
        "1870-01-01T00:00:00.123456Z",
        "1883-11-18T17:00:00Z",
        "1883-11-18T17:00:00.999999Z",
        "1969-12-31T23:59:59.999999Z",
        "2020-03-08T06:59:59.999999Z",
        "2020-03-08T07:00:00Z",
        "2020-11-01T05:30:00.123456Z",
        "2020-11-01T06:30:00.123456Z"
    };
    for (String[] pair : timezonePairs) {
      String writerTimezone = pair[0];
      String readerTimezone = pair[1];
      assertFalse(getTimeZoneForOrc(writerTimezone)
          .hasSameRules(getTimeZoneForOrc(readerTimezone)));
      Long[] decodedMicros = new Long[instants.length + 1];
      Long[] expectedMicros = new Long[decodedMicros.length];
      long baseOffsetUs = orc2015YearBaseOffsetUs(writerTimezone);
      for (int i = 0; i < instants.length; i++) {
        Instant instant = Instant.parse(instants[i]);
        decodedMicros[i] = instant.getEpochSecond() * MICROS_PER_SECOND
            + instant.getNano() / 1_000L + baseOffsetUs;
        expectedMicros[i] = convertPhysicalOrcTimestampToSparkOnCPU(
            decodedMicros[i], writerTimezone, readerTimezone);
      }

      try (ColumnVector input = ColumnVector.timestampMicroSecondsFromBoxedLongs(decodedMicros);
          ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(expectedMicros);
          GpuTimeZoneDB.OrcTimezoneContext context =
              GpuTimeZoneDB.buildOrcTimezoneContext(writerTimezone, readerTimezone);
          ColumnVector actual = GpuTimeZoneDB.convertOrcTimestampToSpark(input, context)) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }

  @Test
  void testConvertPhysicalOrcTimestampToSparkWithNullableSlice() {
    GpuTimeZoneDB.cacheDatabase();
    String timezoneId = "America/New_York";
    long baseOffsetUs = orc2015YearBaseOffsetUs(timezoneId);
    long historicalMicros = -2_717_655_100_076_025L + baseOffsetUs;
    long modernMicros = Instant.parse("2020-01-01T00:00:00Z").getEpochSecond()
        * MICROS_PER_SECOND + 123_456L + baseOffsetUs;

    // Keep a native view with a nonzero offset; subVector copies the slice into a new column.
    try (ColumnVector full = ColumnVector.timestampMicroSecondsFromBoxedLongs(
            123L, null, historicalMicros, null, modernMicros, 456L);
        CloseableArray<ColumnView> slices = CloseableArray.wrap(full.splitAsViews(1, 5));
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
            null, convertPhysicalOrcTimestampToSparkOnCPU(
                historicalMicros, timezoneId, timezoneId),
            null, convertPhysicalOrcTimestampToSparkOnCPU(
                modernMicros, timezoneId, timezoneId));
        GpuTimeZoneDB.OrcTimezoneContext context =
            GpuTimeZoneDB.buildOrcTimezoneContext(timezoneId, timezoneId);
        ColumnVector actual = GpuTimeZoneDB.convertOrcTimestampToSpark(slices.get(1), context)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testConvertPhysicalOrcTimestampToSparkUsesLaterHistoricalOverlap() {
    GpuTimeZoneDB.cacheDatabase();
    String timezoneId = "Africa/Johannesburg";
    long orcInstant = Instant.parse("1892-02-07T21:45:59.999Z").getEpochSecond()
        * MICROS_PER_SECOND + 999_000L;
    long decodedMicros = orcInstant + orc2015YearBaseOffsetUs(timezoneId);
    long expectedMicros = convertPhysicalOrcTimestampToSparkOnCPU(
        decodedMicros, timezoneId, timezoneId);
    assertEquals(-2_458_172_640_001_000L, expectedMicros);

    try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(decodedMicros);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromLongs(expectedMicros);
        GpuTimeZoneDB.OrcTimezoneContext context =
            GpuTimeZoneDB.buildOrcTimezoneContext(timezoneId, timezoneId);
        ColumnVector actual = GpuTimeZoneDB.convertOrcTimestampToSpark(input, context)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testConvertOrcTimestampToSparkBeforeHistoricalGapAtSubsecondPrecision() {
    GpuTimeZoneDB.cacheDatabase();
    String timezoneId = "America/Vancouver";
    ZoneId zoneId = GpuTimeZoneDB.getZoneId(timezoneId);
    TimeZone legacyTimeZone = getTimeZoneForOrc(timezoneId);
    ZoneOffsetTransition historicalGap = null;
    // JDK distributions can bundle different tzdb versions, so derive the boundary from the
    // active rules instead of pinning Vancouver's historical transition instant.
    for (ZoneOffsetTransition transition : zoneId.getRules().getTransitions()) {
      if (transition.isGap()) {
        historicalGap = transition;
        break;
      }
    }
    if (historicalGap == null) {
      throw new AssertionError("Expected a historical Vancouver gap");
    }

    long transitionInstantUs =
        historicalGap.getInstant().getEpochSecond() * MICROS_PER_SECOND;
    long gapUs = historicalGap.getDuration().getSeconds() * MICROS_PER_SECOND;
    long localMicros = historicalGap.getDateTimeAfter()
        .toEpochSecond(ZoneOffset.UTC) * MICROS_PER_SECOND - 1L;
    long localMillis = Math.floorDiv(localMicros, microsPerMillis);
    int legacyOffsetMillis = legacyTimeZone.getOffset(
        localMillis - legacyTimeZone.getRawOffset());
    long orcInstant = (localMillis - legacyOffsetMillis) * microsPerMillis
        + Math.floorMod(localMicros, microsPerMillis);
    long decodedMicros = orcInstant + orc2015YearBaseOffsetUs(timezoneId);
    long expectedPhysicalMicros = convertPhysicalOrcTimestampToSparkOnCPU(
        decodedMicros, timezoneId, timezoneId);
    long expectedIntegerMicros =
        convertIntegerOrcTimestampToSparkOnCPU(localMicros, timezoneId);
    // DOUBLE schema evolution rounds the ORC-converted instant before Spark rebases it.
    long roundedOrcInstant = Math.floorDiv(orcInstant, microsPerMillis) * microsPerMillis;
    long expectedRoundedMicros = rebaseOrcInstantToSparkOnCPU(roundedOrcInstant, timezoneId);
    assertEquals(transitionInstantUs + gapUs - 1L, expectedPhysicalMicros);
    assertEquals(expectedPhysicalMicros, expectedIntegerMicros);
    assertEquals(transitionInstantUs + gapUs - microsPerMillis, expectedRoundedMicros);

    try (ColumnVector physicalInput = ColumnVector.timestampMicroSecondsFromLongs(decodedMicros);
        ColumnVector expectedPhysical =
            ColumnVector.timestampMicroSecondsFromLongs(expectedPhysicalMicros);
        ColumnVector integerInput = ColumnVector.timestampMicroSecondsFromLongs(localMicros);
        ColumnVector expectedInteger =
            ColumnVector.timestampMicroSecondsFromLongs(expectedIntegerMicros);
        ColumnVector roundedInput = ColumnVector.timestampMicroSecondsFromLongs(roundedOrcInstant);
        ColumnVector expectedRounded =
            ColumnVector.timestampMicroSecondsFromLongs(expectedRoundedMicros);
        GpuTimeZoneDB.OrcTimezoneContext context =
            GpuTimeZoneDB.buildOrcTimezoneContext(timezoneId, timezoneId);
        ColumnVector actualPhysical =
            GpuTimeZoneDB.convertOrcTimestampToSpark(physicalInput, context);
        ColumnVector actualInteger =
            GpuTimeZoneDB.convertOrcIntegerTimestampToSpark(integerInput, context);
        ColumnVector actualRounded = GpuTimeZoneDB.rebaseOrcInstantToSpark(roundedInput, context)) {
      assertColumnsAreEqual(expectedPhysical, actualPhysical);
      assertColumnsAreEqual(expectedInteger, actualInteger);
      assertColumnsAreEqual(expectedRounded, actualRounded);
    }
  }

  @Test
  void testConvertIntegerOrcTimestampToSparkBeforeShanghaiFirstTransition() {
    GpuTimeZoneDB.cacheDatabase();
    String timezoneId = "Asia/Shanghai";
    long localMicros = -2_208_988_800L * MICROS_PER_SECOND;
    long expectedMicros = convertIntegerOrcTimestampToSparkOnCPU(localMicros, timezoneId);

    try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(localMicros);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromLongs(expectedMicros);
        GpuTimeZoneDB.OrcTimezoneContext context =
            GpuTimeZoneDB.buildOrcTimezoneContext(timezoneId, timezoneId);
        ColumnVector actual = GpuTimeZoneDB.convertOrcIntegerTimestampToSpark(input, context)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testConvertIntegerOrcTimestampToSparkUsesOrcHistoricalOverlapOffset() {
    GpuTimeZoneDB.cacheDatabase();
    String timezoneId = "America/New_York";
    long overlapStartSeconds = -2_717_668_800L;
    long overlapEndSeconds = -2_717_668_562L;
    long[] localMicros = {
        (overlapStartSeconds - 1L) * MICROS_PER_SECOND,
        overlapStartSeconds * MICROS_PER_SECOND,
        -2_717_668_680L * MICROS_PER_SECOND,
        (overlapEndSeconds - 1L) * MICROS_PER_SECOND,
        overlapEndSeconds * MICROS_PER_SECOND
    };
    long[] expectedMicros = new long[localMicros.length];
    for (int i = 0; i < localMicros.length; i++) {
      expectedMicros[i] = convertIntegerOrcTimestampToSparkOnCPU(localMicros[i], timezoneId);
    }
    assertEquals(-2_717_650_680_000_000L, expectedMicros[2]);

    try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(localMicros);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromLongs(expectedMicros);
        GpuTimeZoneDB.OrcTimezoneContext context =
            GpuTimeZoneDB.buildOrcTimezoneContext(timezoneId, timezoneId);
        ColumnVector actual = GpuTimeZoneDB.convertOrcIntegerTimestampToSpark(input, context)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testConvertIntegerOrcTimestampToSparkWithNullableSlice() {
    GpuTimeZoneDB.cacheDatabase();
    String timezoneId = "America/New_York";
    long historicalMicros = -2_717_668_680L * MICROS_PER_SECOND;
    long modernMicros = LocalDateTime.of(2020, 1, 1, 0, 0)
        .toEpochSecond(ZoneOffset.UTC) * MICROS_PER_SECOND + 123_456L;

    // Offset one also shifts the null mask relative to the original column's validity bits.
    try (ColumnVector full = ColumnVector.timestampMicroSecondsFromBoxedLongs(
            123L, null, historicalMicros, null, modernMicros, 456L);
        CloseableArray<ColumnView> slices = CloseableArray.wrap(full.splitAsViews(1, 5));
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
            null, convertIntegerOrcTimestampToSparkOnCPU(historicalMicros, timezoneId),
            null, convertIntegerOrcTimestampToSparkOnCPU(modernMicros, timezoneId));
        GpuTimeZoneDB.OrcTimezoneContext context =
            GpuTimeZoneDB.buildOrcTimezoneContext(timezoneId, timezoneId);
        ColumnVector actual =
            GpuTimeZoneDB.convertOrcIntegerTimestampToSpark(slices.get(1), context)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testHistoricalRebaseWithoutTimeZoneTransitions() {
    GpuTimeZoneDB.cacheDatabase();
    String timezoneId = "Pacific/Port_Moresby";
    OrcTimezoneInfo timezoneInfo = OrcTimezoneInfo.get(timezoneId);
    assertTrue(timezoneInfo.transitions == null);

    long orcInstant = Instant.parse("1870-01-01T00:00:00Z").getEpochSecond()
        * MICROS_PER_SECOND;
    long decodedMicros = orcInstant + orc2015YearBaseOffsetUs(timezoneId);
    long expectedPhysicalMicros = convertPhysicalOrcTimestampToSparkOnCPU(
        decodedMicros, timezoneId, timezoneId);
    long localMicros = Instant.parse("1870-01-01T10:00:00Z").getEpochSecond()
        * MICROS_PER_SECOND;
    long expectedIntegerMicros =
        convertIntegerOrcTimestampToSparkOnCPU(localMicros, timezoneId);

    try (ColumnVector physicalInput = ColumnVector.timestampMicroSecondsFromLongs(decodedMicros);
        ColumnVector expectedPhysical =
            ColumnVector.timestampMicroSecondsFromLongs(expectedPhysicalMicros);
        ColumnVector integerInput = ColumnVector.timestampMicroSecondsFromLongs(localMicros);
        ColumnVector expectedInteger =
            ColumnVector.timestampMicroSecondsFromLongs(expectedIntegerMicros);
        GpuTimeZoneDB.OrcTimezoneContext context =
            GpuTimeZoneDB.buildOrcTimezoneContext(timezoneId, timezoneId);
        ColumnVector actualPhysical =
            GpuTimeZoneDB.convertOrcTimestampToSpark(physicalInput, context);
        ColumnVector actualInteger =
            GpuTimeZoneDB.convertOrcIntegerTimestampToSpark(integerInput, context)) {
      assertColumnsAreEqual(expectedPhysical, actualPhysical);
      assertColumnsAreEqual(expectedInteger, actualInteger);
    }
  }

  @Test
  void testReaderFirstTransitionUs() {
    String transitionTzId = "Europe/Paris";
    OrcTimezoneInfo transitionInfo = OrcTimezoneInfo.get(transitionTzId);
    assertTrue(transitionInfo.rawOffset > 0);
    try (GpuTimeZoneDB.OrcTimezoneContext context =
        GpuTimeZoneDB.buildOrcTimezoneContext("UTC", transitionTzId)) {
      assertEquals(TimeUnit.MILLISECONDS.toMicros(
              transitionInfo.transitions[0] + transitionInfo.rawOffset),
          context.getReaderFirstTransitionUs());
    }

    try (GpuTimeZoneDB.OrcTimezoneContext context =
        GpuTimeZoneDB.buildOrcTimezoneContext("UTC", "+05:30")) {
      assertEquals(Long.MIN_VALUE, context.getReaderFirstTransitionUs());
    }

    GpuTimeZoneDB.OrcTimezoneContext closed =
        GpuTimeZoneDB.buildOrcTimezoneContext("UTC", "UTC");
    closed.close();
    assertThrows(IllegalStateException.class, closed::getReaderFirstTransitionUs);
  }

  @Test
  void testHistoricalDifferenceCutoffs() {
    try (GpuTimeZoneDB.OrcTimezoneContext context =
        GpuTimeZoneDB.buildOrcTimezoneContext("UTC", "America/New_York")) {
      assertEquals(Instant.parse("1883-11-18T17:00:00Z").getEpochSecond()
              * MICROS_PER_SECOND,
          context.getReaderHistoricalDifferenceEndUtcUs());
      assertEquals(LocalDateTime.of(1883, 11, 18, 12, 3, 58)
              .toEpochSecond(ZoneOffset.UTC) * MICROS_PER_SECOND,
          context.getReaderHistoricalDifferenceEndLocalUs());
    }

    try (GpuTimeZoneDB.OrcTimezoneContext context =
        GpuTimeZoneDB.buildOrcTimezoneContext("UTC", "UTC")) {
      assertEquals(Long.MIN_VALUE, context.getReaderHistoricalDifferenceEndUtcUs());
      assertEquals(Long.MIN_VALUE, context.getReaderHistoricalDifferenceEndLocalUs());
    }
  }

  @Test
  void testConvertOrcTimezones() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    // test time range: (0001-01-01 00:00:00, 9999-12-31 23:59:59)
    long min = LocalDateTime.of(1, 1, 1, 0, 0, 0)
        .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1);
    long max = LocalDateTime.of(9999, 12, 31, 23, 59, 59)
        .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1);

    // Keep the DST matrix deterministic so failures are reproducible.
    Random rng = new Random(42L);

    List<String> timezones = Arrays.asList(
        "America/Los_Angeles",
        "America/Vancouver",
        "America/Cancun",
        "Asia/Shanghai",
        "Antarctica/DumontDUrville",
        "Etc/GMT-12",
        "CNT",
        "Australia/Sydney",
        "Asia/Tokyo");

    for (String writerTz : timezones) {
      for (String readerTz : timezones) {
        // Use 1024 as a reasonable batch size for testing timezone conversions.
        long[] microseconds = new long[1024];
        for (int i = 0; i < microseconds.length; ++i) {
          // range is years from 0001 to 9999
          microseconds[i] = min + (long) (rng.nextDouble() * (max - min));
        }

        try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(microseconds);
            // Convert on CPU
            ColumnVector expected = convertOrcTimezonesOnCPU(microseconds, writerTz, readerTz);
            // Convert on GPU
            ColumnVector actual = GpuTimeZoneDB.convertOrcTimezones(input, writerTz, readerTz)) {
          assertColumnsAreEqual(expected, actual);
        }
      }
    }
  }

  @Test
  void testConvertOrcTimezonesFutureDstRuleFallback() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    for (String timezoneId : Arrays.asList("America/Los_Angeles", "Australia/Sydney")) {
      long[] microseconds = getFutureDstBoundaryMicros(timezoneId);
      String[][] cases = {
          {timezoneId, "UTC"},
          {"UTC", timezoneId}
      };

      for (String[] timezones : cases) {
        try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(microseconds);
            ColumnVector expected =
                convertOrcTimezonesOnCPU(microseconds, timezones[0], timezones[1]);
            ColumnVector actual =
                GpuTimeZoneDB.convertOrcTimezones(input, timezones[0], timezones[1])) {
          assertColumnsAreEqual(expected, actual);
        }
      }
    }
  }

  @Test
  void testConvertOrcTimezonesAsiaGazaPairedTransitions() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    long[] microseconds = {
        LocalDateTime.of(2037, 10, 15, 0, 0)
            .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1)
    };
    String[][] cases = {
        {"Asia/Gaza", "UTC"},
        {"UTC", "Asia/Gaza"}
    };

    for (String[] timezones : cases) {
      try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(microseconds);
          ColumnVector expected =
              convertOrcTimezonesOnCPU(microseconds, timezones[0], timezones[1]);
          ColumnVector actual =
              GpuTimeZoneDB.convertOrcTimezones(input, timezones[0], timezones[1])) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }
}
