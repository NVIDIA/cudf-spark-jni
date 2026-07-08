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

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.TimeZone;
import java.util.concurrent.TimeUnit;

public class GpuTimeZoneDBTest {

  private static final long microsPerMillis = TimeUnit.MILLISECONDS.toMicros(1);

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
    TimeZone writeTz = TimeZone.getTimeZone(writeTzId);
    TimeZone readerTz = TimeZone.getTimeZone(readerTzId);
    for (int i = 0; i < microseconds.length; ++i) {
      // Floor-divide µs to ms (and floor-mod for the sub-ms remainder) so reconstruction round-trips
      // for negative timestamps with a non-zero sub-millisecond component. Truncation toward zero
      // would round such an input up by one ms; at a DST gap transition that lands on the
      // post-transition offset, producing a 1-hour off-by-one. Must match the GPU kernel's
      // floor-divide in convert_timestamp_between_timezones.
      long millis = Math.floorDiv(microseconds[i], microsPerMillis);
      long writerOffset = writeTz.getOffset(millis);
      long readerOffset = readerTz.getOffset(millis);
      long adjustedMillis = millis + writerOffset - readerOffset;
      long adjustedReader = readerTz.getOffset(adjustedMillis);
      long finalDiffs = writerOffset - adjustedReader;
      results[i] = (millis + finalDiffs) * microsPerMillis + Math.floorMod(microseconds[i], microsPerMillis);
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
