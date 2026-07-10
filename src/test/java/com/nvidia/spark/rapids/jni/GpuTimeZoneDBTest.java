/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.TimeZone;
import java.util.concurrent.TimeUnit;

public class GpuTimeZoneDBTest {

  private static final long microsPerMillis = TimeUnit.MILLISECONDS.toMicros(1);
  private static final long MICROS_PER_SECOND = TimeUnit.SECONDS.toMicros(1);

  private static TimeZone getTimeZoneForOrc(String timezoneId) {
    return TimeZone.getTimeZone(GpuTimeZoneDB.getZoneId(timezoneId).getId());
  }

  private static long orc2015YearBaseOffsetUs(String timezoneId) {
    ZoneId zoneId = GpuTimeZoneDB.getZoneId(timezoneId);
    if (zoneId.getRules().isFixedOffset()) {
      int offsetSeconds = zoneId.getRules().getOffset(Instant.EPOCH).getTotalSeconds();
      return TimeUnit.SECONDS.toMicros(offsetSeconds);
    }
    TimeZone tz = TimeZone.getTimeZone(zoneId.getId());
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
    // falling back to GMT. The DST guard at the top of convertOrcTimezones
    // calls ZoneId.of(...), so an unknown id will throw before the runtime
    // build path or the GPU kernel ever runs. We assert the broad
    // RuntimeException type so this stays a regression guard even if the
    // exact wrapping (DateTimeException vs IllegalArgumentException vs
    // IllegalStateException) is refactored later.
    GpuTimeZoneDB.cacheDatabase();
    try (ColumnVector input =
        ColumnVector.timestampMicroSecondsFromLongs(new long[] {0L})) {
      assertThrows(RuntimeException.class,
          () -> GpuTimeZoneDB.convertOrcTimezones(input, "Invalid/Zone", "UTC"));
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
  void testConvertOrcTimezones() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    // test time range: (0001-01-01 00:00:00, 9999-12-31 23:59:59)
    long min = LocalDateTime.of(1, 1, 1, 0, 0, 0)
        .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1);
    long max = LocalDateTime.of(9999, 12, 31, 23, 59, 59)
        .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1);

    // use today as the random seed so we get different values each day
    Random rng = new Random(LocalDate.now().toEpochDay());

    List<String> timezones = Arrays.asList(
        "America/Los_Angeles",
        "America/Cancun",
        "Asia/Shanghai",
        "Antarctica/DumontDUrville",
        "Etc/GMT-12",
        "CNT",
        "Australia/Sydney",
        "Asia/Tokyo");

    for (String writerTz : timezones) {
      if (GpuTimeZoneDB.isDST(writerTz)) {
        // currently do not support DST conversions
        continue;
      }
      for (String readerTz : timezones) {
        if (GpuTimeZoneDB.isDST(readerTz)) {
          // currently do not support DST conversions
          continue;
        }
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
}
