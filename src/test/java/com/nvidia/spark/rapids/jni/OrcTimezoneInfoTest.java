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

import org.junit.jupiter.api.Test;

import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.Month;
import java.time.Year;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.zone.ZoneOffsetTransition;
import java.time.zone.ZoneOffsetTransitionRule;
import java.time.zone.ZoneRules;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.TimeZone;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class OrcTimezoneInfoTest {

  /**
   * An inert historical transition for synthetic {@link ZoneRules} fixtures.
   *
   * <p>On JDK 8 (the build's {@code maven.compiler.target}), {@code
   * ZoneRules.of(base, base, emptyList, emptyList, recurringRules)} -- recurring
   * rules but no concrete historical transitions -- reports {@code
   * isFixedOffset() == true} even though DST rules are present. That makes
   * {@link OrcDstRuleExtractor#extractDstRule} short-circuit to {@code null} at
   * its {@code rules.isFixedOffset()} guard before either extraction path runs.
   * JDK 17 reports {@code false}, so the gap is invisible to a JDK 17 compile
   * check and only surfaces when the 1.8-target test suite actually runs.
   *
   * <p>Including one concrete transition flips {@code isFixedOffset()} to {@code
   * false} on every JDK. Real IANA zones always carry historical transitions,
   * so production is unaffected -- only these hand-built fixtures need it. The
   * transition is dated 1900 and changes only the wall offset (the
   * standard-offset history stays at the base offset), so it perturbs neither
   * {@code getStandardOffset} nor the post-2060 probing windows the tests rely
   * on.
   */
  private static final ZoneOffsetTransition SYNTHETIC_HISTORICAL_TRANSITION =
      ZoneOffsetTransition.of(
          LocalDateTime.of(1900, 1, 1, 0, 0), ZoneOffset.ofHours(-1), ZoneOffset.UTC);

  @Test
  void testGetFixedOffsetZone() {
    // Fixed-offset zones must return a non-null OrcTimezoneInfo with the
    // offset derived from ZoneRules (not from TimeZone.getTimeZone, which
    // would silently map "+05:30" to GMT). +05:30 == 19_800_000 ms.
    OrcTimezoneInfo info = OrcTimezoneInfo.get("+05:30");
    assertNotNull(info);
    assertEquals(19_800_000, info.initialOffset);
    assertEquals(19_800_000, info.rawOffset);
    assertNull(info.transitions);
    assertNull(info.offsets);
    assertNull(info.dstRule);
  }

  @Test
  void testGetFixedOffsetNamedZone() {
    // "UTC" is a named zone whose ZoneRules.isFixedOffset() is true. Cover
    // it explicitly so a regression that treats "UTC" as a historical zone
    // (non-null transitions) — or that silently maps it to GMT via
    // TimeZone.getTimeZone — is caught. rawOffset must be 0.
    OrcTimezoneInfo info = OrcTimezoneInfo.get("UTC");
    assertNotNull(info);
    assertEquals(0, info.initialOffset);
    assertEquals(0, info.rawOffset);
    assertNull(info.transitions);
    assertNull(info.offsets);
    assertNull(info.dstRule);
  }

  @Test
  void testGetCachesByKey() {
    // computeIfAbsent must return the same instance on the second call so
    // that other threads sharing RUNTIME_TIMEZONE_INFOS see a stable object.
    OrcTimezoneInfo a = OrcTimezoneInfo.get("Asia/Kolkata");
    OrcTimezoneInfo b = OrcTimezoneInfo.get("Asia/Kolkata");
    assertSame(a, b);
  }

  @Test
  void testGetThrowsOnInvalidId() {
    // Documented contract: invalid IDs throw IllegalArgumentException.
    // There is no silent fallback to GMT.
    assertThrows(IllegalArgumentException.class,
        () -> OrcTimezoneInfo.get("Invalid/Zone"));
  }

  @Test
  void testGetAllTimezoneIdsContract() {
    List<String> ids = OrcTimezoneInfo.getAllTimezoneIds();

    assertFalse(ids.isEmpty(), "expected at least one supported timezone id");
    assertTrue(ids.contains("UTC"), "UTC must be present");
    assertTrue(ids.contains("Asia/Shanghai"), "Asia/Shanghai must be present");

    // Sorted ascending AND distinct (strict <, not <=, also catches duplicate ids).
    for (int i = 1; i < ids.size(); i++) {
      assertTrue(ids.get(i - 1).compareTo(ids.get(i)) < 0,
          "list must be sorted and distinct: " + ids.get(i - 1) + " >= " + ids.get(i));
    }

    // Every id must be one that OrcTimezoneInfo.get can build — i.e. the lister
    // and the loader agree.
    for (String id : ids) {
      assertTrue(GpuTimeZoneDB.isSupportedTimeZone(id),
          "getAllTimezoneIds returned an id that isSupportedTimeZone rejects: " + id);
    }
  }

  @Test
  void testGetHistoricalTransitionsZone() {
    // Asia/Shanghai is a non-DST named zone with real historical transitions.
    // Verify that the runtime build path populates both arrays consistently.
    //
    // Known coverage gap: zones whose ZoneRules.getTransitions() is empty
    // but whose historical offset changed cannot exercise the scan-only
    // path in collectTimeZoneTransitionsByScanning, because
    // buildHistoricalTransitions returns EMPTY early for empty transition
    // lists. Covering it would require a synthetic zone.
    OrcTimezoneInfo info = OrcTimezoneInfo.get("Asia/Shanghai");
    assertNotNull(info);
    assertNotNull(info.transitions, "Asia/Shanghai should have historical transitions");
    assertNotNull(info.offsets);
    assertNull(info.dstRule);
    assertEquals(info.transitions.length, info.offsets.length,
        "transitions and offsets must be the same length");
    // Transitions must be strictly increasing so the GPU binary search is well-defined.
    for (int i = 1; i < info.transitions.length; i++) {
      assertTrue(info.transitions[i] > info.transitions[i - 1],
          "transitions must be strictly increasing");
    }
  }

  @Test
  void testGetDstZoneIncludesKernelMetadata() {
    String timezoneId = "America/New_York";
    OrcTimezoneInfo info = OrcTimezoneInfo.get(timezoneId);
    TimeZone tz = TimeZone.getTimeZone(GpuTimeZoneDB.getZoneId(timezoneId).getId());

    assertEquals(tz.getOffset(OrcTimezoneInfo.utcMillisForDate(1, 1, 1)), info.initialOffset);
    assertEquals(tz.getRawOffset(), info.rawOffset);
    assertNotNull(info.transitions);
    assertNotNull(info.offsets);
    assertNotNull(info.dstRule);
  }

  @Test
  void testScannerRetainsPairedTransitionsBetweenEqualEndpoints() {
    long firstTransition = 8L * 3_600_000L;
    long secondTransition = 16L * 3_600_000L;
    TimeZone pairedTransitions = new TimeZone() {
      private int rawOffset;

      @Override
      public int getOffset(long date) {
        return date >= firstTransition && date < secondTransition ? 3_600_000 : 0;
      }

      @Override
      public int getOffset(int era, int year, int month, int day,
          int dayOfWeek, int milliseconds) {
        return rawOffset;
      }

      @Override
      public void setRawOffset(int offsetMillis) {
        rawOffset = offsetMillis;
      }

      @Override
      public int getRawOffset() {
        return rawOffset;
      }

      @Override
      public boolean useDaylightTime() {
        return true;
      }

      @Override
      public boolean inDaylightTime(Date date) {
        return getOffset(date.getTime()) != rawOffset;
      }
    };

    List<Long> transitions = new ArrayList<>();
    List<Integer> offsets = new ArrayList<>();
    int finalOffset = OrcTimezoneInfo.collectTimeZoneTransitionsByScanning(
        pairedTransitions, 0L, 24L * 3_600_000L, 0, transitions, offsets);

    assertEquals(Arrays.asList(firstTransition, secondTransition), transitions);
    assertEquals(Arrays.asList(3_600_000, 0), offsets);
    assertEquals(0, finalOffset);
  }

  @Test
  void testAsiaGazaRetainsImmediateOffsetReturn() {
    OrcTimezoneInfo info = OrcTimezoneInfo.get("Asia/Gaza");
    long transition = ZoneOffsetTransition.of(
        LocalDateTime.of(2037, 10, 10, 2, 0),
        ZoneOffset.ofHours(3),
        ZoneOffset.ofHours(2)).getInstant().toEpochMilli();
    int index = Arrays.binarySearch(info.transitions, transition);

    assertTrue(index >= 0, "expected the ZoneRules candidate in the transition table");
    assertEquals(7_200_000, info.offsets[index]);
    assertEquals(transition + 1, info.transitions[index + 1]);
    assertEquals(10_800_000, info.offsets[index + 1]);
  }

  // ---- DST rule extraction ----

  @Test
  void testExtractDstRuleNorthernHemisphere() {
    // America/New_York: DST starts 2nd Sunday of March, ends 1st Sunday of November.
    // dstSavings is +1h (3_600_000 ms). startMonth=2 (March, 0-based),
    // endMonth=10 (November, 0-based). DOW_GE_DOM_MODE = 2.
    OrcDstRuleExtractor.DstRule rule = extractDstRuleFor("America/New_York");
    assertNotNull(rule, "America/New_York must have a DST rule");
    assertEquals(3_600_000, rule.dstSavings);
    assertEquals(2, rule.startMonth);
    assertEquals(10, rule.endMonth);
    assertEquals(2, rule.startMode);
    assertEquals(2, rule.endMode);
    // Day-of-week 1 == Sunday in Calendar's 1=Sun..7=Sat convention.
    assertEquals(1, rule.startDayOfWeek);
    assertEquals(1, rule.endDayOfWeek);
    // Second Sunday in March: base day 8 ("Sun >= 8"). First Sunday in November: base day 1.
    assertEquals(8, rule.startDay);
    assertEquals(1, rule.endDay);
    // Probing path encodes both transitions as STANDARD time (timeMode=1) at
    // the wall-clock instants 02:00 (DST start) and 01:00 (DST end). Lock
    // these so a regression that flips timeMode to WALL would shift the
    // computed UTC transitions by dstSavings and fail verifyDstRule silently.
    assertEquals(1, rule.startTimeMode, "DST start should be STANDARD time mode");
    assertEquals(1, rule.endTimeMode, "DST end should be STANDARD time mode");
    assertEquals(2 * 3_600_000, rule.startTime, "DST start at 02:00 standard");
    assertEquals(1 * 3_600_000, rule.endTime, "DST end at 01:00 standard");
  }

  @Test
  void testExtractDstRuleEuropeLondon() {
    // Europe/London: DST starts last Sunday of March, ends last Sunday of October.
    // Encoded as DOW_GE_DOM with base day = monthLength - 6.
    OrcDstRuleExtractor.DstRule rule = extractDstRuleFor("Europe/London");
    assertNotNull(rule, "Europe/London must have a DST rule");
    assertEquals(3_600_000, rule.dstSavings);
    assertEquals(2, rule.startMonth);
    assertEquals(9, rule.endMonth);
    assertEquals(1, rule.startDayOfWeek);
    assertEquals(1, rule.endDayOfWeek);
    // Last Sunday of March (31-day month): base 25. Last Sunday of October (31-day): base 25.
    assertEquals(25, rule.startDay);
    assertEquals(25, rule.endDay);
    // Probing path encodes both ends as DOW_GE_DOM (mode=2) on STANDARD time
    // (timeMode=1). BST flips on at 01:00 standard in March and off at 01:00
    // standard in October.
    assertEquals(2, rule.startMode);
    assertEquals(2, rule.endMode);
    assertEquals(1, rule.startTimeMode);
    assertEquals(1, rule.endTimeMode);
    assertEquals(1 * 3_600_000, rule.startTime, "DST start at 01:00 standard");
    assertEquals(1 * 3_600_000, rule.endTime, "DST end at 01:00 standard");
  }

  @Test
  void testExtractDstRuleAfterHistoricalStandardOffsetChange() {
    // America/Scoresbysund changed its standard offset after the validation reference instant.
    // TimeZone.getRawOffset() reports the current standard offset, so it cannot be compared with
    // ZoneRules at that historical instant. The two APIs' actual offsets still agree there.
    assertNotNull(extractDstRuleFor("America/Scoresbysund"));
  }

  @Test
  void testExtractDstRuleSouthernHemisphere() {
    // Australia/Sydney: DST starts 1st Sunday of October, ends 1st Sunday of April.
    // Southern hemisphere — start month numerically > end month.
    OrcDstRuleExtractor.DstRule rule = extractDstRuleFor("Australia/Sydney");
    assertNotNull(rule, "Australia/Sydney must have a DST rule");
    assertEquals(3_600_000, rule.dstSavings);
    assertEquals(9, rule.startMonth);
    assertEquals(3, rule.endMonth);
    assertTrue(rule.startMonth > rule.endMonth,
        "southern hemisphere: start month should follow end month within the calendar year");
    // 1st Sunday of October: base 1. 1st Sunday of April: base 1.
    assertEquals(1, rule.startDay);
    assertEquals(1, rule.endDay);
    assertEquals(1, rule.startDayOfWeek);
    assertEquals(1, rule.endDayOfWeek);
    assertEquals(2, rule.startMode);
    assertEquals(2, rule.endMode);
    assertEquals(1, rule.startTimeMode);
    assertEquals(1, rule.endTimeMode);
    assertEquals(2 * 3_600_000, rule.startTime, "DST start at 02:00 standard");
    assertEquals(2 * 3_600_000, rule.endTime, "DST end at 02:00 standard");
  }

  @Test
  void testExtractDstRuleNoDstReturnsNull() {
    // Asia/Shanghai had DST historically (1940s, 1986-1991) but no current rule.
    // tz.useDaylightTime() must be false → extractDstRule returns null.
    assertNull(extractDstRuleFor("Asia/Shanghai"));
  }

  @Test
  void testExtractDstRuleFixedOffsetReturnsNull() {
    // Fixed-offset zones never observe DST.
    assertNull(extractDstRuleFor("UTC"));
    assertNull(extractDstRuleFor("+05:30"));
  }

  @Test
  void testExtractDstRuleThrowsOnUnsupportedRuleCount() {
    // Synthesize a TimeZone whose getOffset is constant. The probing path
    // (extractDstRuleByProbing) observes no transitions across all anchor
    // years and returns null, so extractDstRuleFromZoneRules runs with the
    // hand-crafted ZoneRules below.
    TimeZone constantOffsetWithDstFlag = new TimeZone() {
      @Override public int getOffset(long instant) { return 0; }
      @Override public int getOffset(int era, int year, int month, int day, int dow, int ms) {
        return 0;
      }
      @Override public int getRawOffset() { return 0; }
      @Override public void setRawOffset(int offsetMillis) {}
      @Override public boolean useDaylightTime() { return true; }
      @Override public boolean inDaylightTime(Date date) { return false; }
    };
    constantOffsetWithDstFlag.setID("Synthetic/UnsupportedRuleCount");

    // ZoneRules with exactly one recurring rule. Production code rejects any
    // count outside {0, 2}, so this triggers the "Unsupported ORC DST rule
    // count" branch in extractDstRuleFromZoneRules.
    ZoneOffset baseOffset = ZoneOffset.UTC;
    ZoneOffsetTransitionRule lonelyRule = ZoneOffsetTransitionRule.of(
        Month.MARCH, 8, DayOfWeek.SUNDAY, LocalTime.of(2, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        baseOffset, baseOffset, ZoneOffset.ofHours(1));
    ZoneRules syntheticRules = ZoneRules.of(
        baseOffset, baseOffset,
        Collections.emptyList(),
        Collections.singletonList(SYNTHETIC_HISTORICAL_TRANSITION),
        Collections.singletonList(lonelyRule));

    IllegalStateException ex = assertThrows(IllegalStateException.class,
        () -> OrcDstRuleExtractor.extractDstRule(
            "Synthetic/UnsupportedRuleCount", constantOffsetWithDstFlag, syntheticRules));
    assertTrue(ex.getMessage().contains("Synthetic/UnsupportedRuleCount"),
        "exception message should name the offending zone: " + ex.getMessage());
  }

  /**
   * Resolve a zone id through the same SHORT_IDS pipeline production uses.
   *
   * <p>For fixed-offset ids like {@code "+05:30"}, {@code TimeZone.getTimeZone}
   * silently returns GMT (rawOffset=0) rather than a TimeZone with the actual
   * offset, because {@code java.util.TimeZone} does not recognise the
   * offset-format id. Mirror production's {@code rules.isFixedOffset()} guard
   * here so the test does not silently feed a GMT TimeZone into
   * {@code extractDstRule}; production's
   * {@link OrcDstRuleExtractor#extractDstRule(String, java.util.TimeZone, java.time.zone.ZoneRules)}
   * now short-circuits on {@code rules.isFixedOffset()} too, but the test
   * helper keeps its own pre-call guard so a future caller pattern that drops
   * the production guard cannot silently re-introduce the trap.
   */
  private static OrcDstRuleExtractor.DstRule extractDstRuleFor(String timezoneId) {
    ZoneId zoneId = ZoneId.of(timezoneId, ZoneId.SHORT_IDS);
    ZoneRules rules = zoneId.getRules();
    // For fixed-offset zones the UTC placeholder has rawOffset=0, which would
    // *not* match the rules' actual standard offset and so would fail
    // extractDstRule's tz-vs-rules sanity check. That's safe today because
    // extractDstRule's isFixedOffset() short-circuit at the top of the method
    // returns null before the sanity check executes. If a future change ever
    // reorders these guards, the mismatch would surface here as a confusing
    // "describe different zones" exception rather than the silent-GMT bug we
    // are guarding against.
    TimeZone tz = rules.isFixedOffset()
        ? TimeZone.getTimeZone("UTC")
        : TimeZone.getTimeZone(zoneId.getId());
    return OrcDstRuleExtractor.extractDstRule(timezoneId, tz, rules);
  }

  @Test
  void testExtractDstRuleThrowsWhenBothPathsFail() {
    // Constant-offset TimeZone — probing observes no transitions across all
    // anchor years and returns null.
    TimeZone constantOffsetWithDstFlag = new TimeZone() {
      @Override public int getOffset(long instant) { return 0; }
      @Override public int getOffset(int era, int year, int month, int day, int dow, int ms) {
        return 0;
      }
      @Override public int getRawOffset() { return 0; }
      @Override public void setRawOffset(int offsetMillis) {}
      @Override public boolean useDaylightTime() { return true; }
      @Override public boolean inDaylightTime(Date date) { return false; }
    };
    constantOffsetWithDstFlag.setID("Synthetic/NoRecurringRules");

    // A single historical transition keeps rules.isFixedOffset() == false so
    // the early guard in extractDstRule does not short-circuit; the empty
    // lastRules list makes extractDstRuleFromZoneRules return null. Both paths
    // fail and the terminal "Failed to extract" throw fires.
    ZoneOffset baseOffset = ZoneOffset.UTC;
    ZoneOffsetTransition historical = ZoneOffsetTransition.of(
        LocalDateTime.of(1900, 1, 1, 0, 0),
        ZoneOffset.ofHours(-1), baseOffset);
    ZoneRules rules = ZoneRules.of(
        baseOffset, baseOffset,
        Collections.emptyList(),
        Collections.singletonList(historical),
        Collections.emptyList());

    IllegalStateException ex = assertThrows(IllegalStateException.class,
        () -> OrcDstRuleExtractor.extractDstRule(
            "Synthetic/NoRecurringRules", constantOffsetWithDstFlag, rules));
    assertTrue(ex.getMessage().contains("Synthetic/NoRecurringRules"),
        "exception message should name the offending zone: " + ex.getMessage());
    assertTrue(ex.getMessage().contains("Failed to extract"),
        "terminal throw should mention 'Failed to extract': " + ex.getMessage());
  }

  // Helper: TimeZone whose getOffset is constant. Probing finds no transitions
  // across any anchor year and returns null, so extractDstRuleFromZoneRules is
  // invoked with the hand-crafted ZoneRules in each test below.
  private static TimeZone newConstantOffsetWithDstFlag(String id) {
    return newConstantOffsetWithDstFlag(id, 0);
  }

  private static TimeZone newConstantOffsetWithDstFlag(String id, int offset) {
    TimeZone tz = new TimeZone() {
      @Override public int getOffset(long instant) { return offset; }
      @Override public int getOffset(int era, int year, int month, int day, int dow, int ms) {
        return offset;
      }
      @Override public int getRawOffset() { return offset; }
      @Override public void setRawOffset(int offsetMillis) {}
      @Override public boolean useDaylightTime() { return true; }
      @Override public boolean inDaylightTime(Date date) { return false; }
    };
    tz.setID(id);
    return tz;
  }

  @Test
  void testExtractDstRuleThrowsOnZeroDeltaRule() {
    // Two recurring rules where the second one has offsetBefore == offsetAfter
    // (delta == 0). Triggers the "Unsupported zero-delta ORC DST rule" branch.
    TimeZone tz = newConstantOffsetWithDstFlag("Synthetic/ZeroDelta");
    ZoneOffset base = ZoneOffset.UTC;
    ZoneOffsetTransitionRule startRule = ZoneOffsetTransitionRule.of(
        Month.MARCH, 8, DayOfWeek.SUNDAY, LocalTime.of(2, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, base, ZoneOffset.ofHours(1));
    ZoneOffsetTransitionRule zeroDeltaRule = ZoneOffsetTransitionRule.of(
        Month.OCTOBER, 25, DayOfWeek.SUNDAY, LocalTime.of(1, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, base, base);  // zero delta
    ZoneRules rules = ZoneRules.of(base, base,
        Collections.emptyList(),
        Collections.singletonList(SYNTHETIC_HISTORICAL_TRANSITION),
        Arrays.asList(startRule, zeroDeltaRule));
    IllegalStateException ex = assertThrows(IllegalStateException.class,
        () -> OrcDstRuleExtractor.extractDstRule("Synthetic/ZeroDelta", tz, rules));
    assertTrue(ex.getMessage().contains("zero-delta"),
        "expected 'zero-delta' in message: " + ex.getMessage());
  }

  @Test
  void testExtractDstRuleThrowsOnBothPositiveDeltaRules() {
    // Two rules both with positive delta — endTransitionRule stays null.
    // Triggers the "Failed to identify ORC DST start/end rules" branch.
    TimeZone tz = newConstantOffsetWithDstFlag("Synthetic/BothPositive");
    ZoneOffset base = ZoneOffset.UTC;
    ZoneOffset plus1 = ZoneOffset.ofHours(1);
    ZoneOffsetTransitionRule ruleA = ZoneOffsetTransitionRule.of(
        Month.MARCH, 8, DayOfWeek.SUNDAY, LocalTime.of(2, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, base, plus1);
    ZoneOffsetTransitionRule ruleB = ZoneOffsetTransitionRule.of(
        Month.JUNE, 1, DayOfWeek.SUNDAY, LocalTime.of(2, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, base, plus1);
    ZoneRules rules = ZoneRules.of(base, base,
        Collections.emptyList(),
        Collections.singletonList(SYNTHETIC_HISTORICAL_TRANSITION),
        Arrays.asList(ruleA, ruleB));
    IllegalStateException ex = assertThrows(IllegalStateException.class,
        () -> OrcDstRuleExtractor.extractDstRule("Synthetic/BothPositive", tz, rules));
    assertTrue(ex.getMessage().contains("Failed to identify"),
        "expected 'Failed to identify' in message: " + ex.getMessage());
  }

  @Test
  void testExtractDstRuleThrowsWhenTzAndRulesDescribeDifferentZones() {
    // The sanity-check in extractDstRule compares actual offsets at the same
    // reference instant. Pair a constant-zero TimeZone with ZoneRules for
    // America/New_York (offset = -18_000_000 ms at the
    // 2024 reference instant) so the check fires before either extraction
    // path runs.
    TimeZone zeroOffsetTz = newConstantOffsetWithDstFlag("Synthetic/OffsetMismatch");
    ZoneRules newYorkRules = ZoneId.of("America/New_York").getRules();
    IllegalStateException ex = assertThrows(IllegalStateException.class,
        () -> OrcDstRuleExtractor.extractDstRule(
            "Synthetic/OffsetMismatch", zeroOffsetTz, newYorkRules));
    assertTrue(ex.getMessage().contains("describe different zones"),
        "expected 'describe different zones' in message: " + ex.getMessage());
  }

  @Test
  void testExtractDstRuleThrowsOnBothNegativeDeltaRules() {
    // Symmetric to testExtractDstRuleThrowsOnBothPositiveDeltaRules. Two rules
    // both with negative delta — startTransitionRule stays null. Pins the
    // startTransitionRule == null sub-case of the || at line 157 so a future
    // change from || to && cannot slip through.
    // These malformed rules leave the recurring period at +1h at the validation instant.
    TimeZone tz = newConstantOffsetWithDstFlag("Synthetic/BothNegative", 3_600_000);
    ZoneOffset base = ZoneOffset.UTC;
    ZoneOffset plus1 = ZoneOffset.ofHours(1);
    ZoneOffsetTransitionRule ruleA = ZoneOffsetTransitionRule.of(
        Month.MARCH, 8, DayOfWeek.SUNDAY, LocalTime.of(2, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, plus1, base);
    ZoneOffsetTransitionRule ruleB = ZoneOffsetTransitionRule.of(
        Month.NOVEMBER, 1, DayOfWeek.SUNDAY, LocalTime.of(1, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, plus1, base);
    ZoneRules rules = ZoneRules.of(base, base,
        Collections.emptyList(),
        Collections.singletonList(SYNTHETIC_HISTORICAL_TRANSITION),
        Arrays.asList(ruleA, ruleB));
    IllegalStateException ex = assertThrows(IllegalStateException.class,
        () -> OrcDstRuleExtractor.extractDstRule("Synthetic/BothNegative", tz, rules));
    assertTrue(ex.getMessage().contains("Failed to identify"),
        "expected 'Failed to identify' in message: " + ex.getMessage());
  }

  @Test
  void testExtractDstRuleThrowsOnMismatchedSavings() {
    // Start gains +1h, end loses -2h. Triggers the "Mismatched ORC DST savings"
    // branch.
    TimeZone tz = newConstantOffsetWithDstFlag("Synthetic/MismatchedSavings");
    ZoneOffset base = ZoneOffset.UTC;
    ZoneOffset plus1 = ZoneOffset.ofHours(1);
    ZoneOffset plus2 = ZoneOffset.ofHours(2);
    ZoneOffsetTransitionRule startRule = ZoneOffsetTransitionRule.of(
        Month.MARCH, 8, DayOfWeek.SUNDAY, LocalTime.of(2, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, base, plus1);
    ZoneOffsetTransitionRule endRule = ZoneOffsetTransitionRule.of(
        Month.NOVEMBER, 1, DayOfWeek.SUNDAY, LocalTime.of(2, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, plus2, base);  // -2h, but start was +1h
    ZoneRules rules = ZoneRules.of(base, base,
        Collections.emptyList(),
        Collections.singletonList(SYNTHETIC_HISTORICAL_TRANSITION),
        Arrays.asList(startRule, endRule));
    IllegalStateException ex = assertThrows(IllegalStateException.class,
        () -> OrcDstRuleExtractor.extractDstRule("Synthetic/MismatchedSavings", tz, rules));
    assertTrue(ex.getMessage().contains("Mismatched ORC DST savings"),
        "expected 'Mismatched ORC DST savings' in message: " + ex.getMessage());
  }

  @Test
  void testExtractDstRuleThrowsOnUnsupportedRuleShape() {
    // First rule has null dayOfWeek (DOM-shaped rule, fixed day-of-month).
    // Triggers the "Unsupported ORC DST transition rule shape" branch in
    // fillDstRuleFromTransitionRule.
    TimeZone tz = newConstantOffsetWithDstFlag("Synthetic/DomRule");
    ZoneOffset base = ZoneOffset.UTC;
    ZoneOffset plus1 = ZoneOffset.ofHours(1);
    ZoneOffsetTransitionRule domRule = ZoneOffsetTransitionRule.of(
        Month.MARCH, 15, null, LocalTime.of(2, 0), false,  // null dayOfWeek
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, base, plus1);
    ZoneOffsetTransitionRule endRule = ZoneOffsetTransitionRule.of(
        Month.OCTOBER, 25, DayOfWeek.SUNDAY, LocalTime.of(1, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, plus1, base);
    ZoneRules rules = ZoneRules.of(base, base,
        Collections.emptyList(),
        Collections.singletonList(SYNTHETIC_HISTORICAL_TRANSITION),
        Arrays.asList(domRule, endRule));
    IllegalStateException ex = assertThrows(IllegalStateException.class,
        () -> OrcDstRuleExtractor.extractDstRule("Synthetic/DomRule", tz, rules));
    assertTrue(ex.getMessage().contains("transition rule shape"),
        "expected 'transition rule shape' in message: " + ex.getMessage());
  }

  @Test
  void testExtractDstRuleThrowsWhenPathAVerificationFails() {
    // Path A's verification-failure branch. The TimeZone returns offset 0
    // everywhere, so probing finds no transitions and returns null. Path A
    // then parses the two valid recurring rules into a DstRule with
    // dstSavings=+1h, but verifyDstRuleAcrossReferenceYears compares
    // computeDstOffset (which predicts +1h inside the DST window) against
    // tz.getOffset (constant 0) — the mismatch causes verify to return
    // false and the "ZoneRules ORC DST rule verification failed" branch
    // fires.
    TimeZone tz = newConstantOffsetWithDstFlag("Synthetic/PathAVerifyFail");
    ZoneOffset base = ZoneOffset.UTC;
    ZoneOffset plus1 = ZoneOffset.ofHours(1);
    ZoneOffsetTransitionRule startRule = ZoneOffsetTransitionRule.of(
        Month.MARCH, 8, DayOfWeek.SUNDAY, LocalTime.of(2, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, base, plus1);
    ZoneOffsetTransitionRule endRule = ZoneOffsetTransitionRule.of(
        Month.NOVEMBER, 1, DayOfWeek.SUNDAY, LocalTime.of(1, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, plus1, base);
    ZoneRules rules = ZoneRules.of(base, base,
        Collections.emptyList(),
        Collections.singletonList(SYNTHETIC_HISTORICAL_TRANSITION),
        Arrays.asList(startRule, endRule));
    IllegalStateException ex = assertThrows(IllegalStateException.class,
        () -> OrcDstRuleExtractor.extractDstRule("Synthetic/PathAVerifyFail", tz, rules));
    assertTrue(ex.getMessage().contains("ZoneRules ORC DST rule verification failed"),
        "expected verification-failed message: " + ex.getMessage());
  }

  @Test
  void testExtractDstRuleThrowsOnNegativeDayIndicator() {
    // Negative dayOfMonthIndicator encodes a DOW_LE_DOM rule ("last <dayOfWeek>
    // on or before day"); fillDstRuleFromTransitionRule rejects it via the
    // same "Unsupported ORC DST transition rule shape" guard as null dayOfWeek
    // (the two sub-cases share one || condition; this test pins the second).
    TimeZone tz = newConstantOffsetWithDstFlag("Synthetic/NegativeIndicator");
    ZoneOffset base = ZoneOffset.UTC;
    ZoneOffset plus1 = ZoneOffset.ofHours(1);
    ZoneOffsetTransitionRule negativeIndicatorRule = ZoneOffsetTransitionRule.of(
        Month.MARCH, -1, DayOfWeek.SUNDAY, LocalTime.of(2, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, base, plus1);
    ZoneOffsetTransitionRule endRule = ZoneOffsetTransitionRule.of(
        Month.OCTOBER, 25, DayOfWeek.SUNDAY, LocalTime.of(1, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, plus1, base);
    ZoneRules rules = ZoneRules.of(base, base,
        Collections.emptyList(),
        Collections.singletonList(SYNTHETIC_HISTORICAL_TRANSITION),
        Arrays.asList(negativeIndicatorRule, endRule));
    IllegalStateException ex = assertThrows(IllegalStateException.class,
        () -> OrcDstRuleExtractor.extractDstRule("Synthetic/NegativeIndicator", tz, rules));
    assertTrue(ex.getMessage().contains("transition rule shape"),
        "expected 'transition rule shape' in message: " + ex.getMessage());
  }

  /**
   * Compute the UTC millis at midnight UTC of the {@code n}-th occurrence of {@code dow} in
   * {@code month} of {@code year}.
   *
   * @param year  calendar year
   * @param month month (1-based via {@link Month})
   * @param dow   day of week to match
   * @param n     occurrence index (1 = first, 2 = second, ...)
   * @return UTC epoch millis at midnight of the resolved date
   */
  private static long nthDayOfWeekUtcMs(int year, Month month, DayOfWeek dow, int n) {
    LocalDate firstOfMonth = LocalDate.of(year, month, 1);
    int firstDow = firstOfMonth.getDayOfWeek().getValue(); // 1..7, Mon..Sun
    int diff = dow.getValue() - firstDow;
    if (diff < 0) diff += 7;
    int day = 1 + diff + (n - 1) * 7;
    return LocalDate.of(year, month, day).toEpochDay() * 86_400_000L;
  }

  @Test
  void testExtractDstRuleFebruaryClampDoesNotThrow() {
    // dayOfMonthIndicator=29 + SUNDAY for FEBRUARY forces ruleDay (29) >
    // monthLength (28 in non-leap years). Without the Math.min(ruleDay,
    // monthLength) anchor clamp in computeRuleDay's MODE_DOW_GE_DOM branch,
    // LocalDate.of(year, FEBRUARY, 29) on a non-leap year would throw
    // DateTimeException inside verifyDstRuleAcrossReferenceYears.
    //
    // For indicator=29 the clamp makes computeRuleDay collapse to "last day
    // of February": 29 on leap years, 28 on non-leap years (and the day-of-
    // week constraint is silently dropped by the result clamp). The
    // synthetic tz must mirror that semantics or verify will fail before
    // the test can observe the clamp's effect; we therefore have the tz
    // report DST starting on the last day of February.
    //
    // Path B is forced to fail (custom getDSTSavings()==0 mismatch) so
    // Path A runs. Path A copies getDayOfMonthIndicator() verbatim into
    // rule.startDay, so the stored value is 29 -- the clamp lives in
    // computeRuleDay, not in Path A's encoding.
    TimeZone tz = new TimeZone() {
      @Override public int getOffset(long instant) {
        int year = LocalDate.ofEpochDay(Math.floorDiv(instant, 86_400_000L)).getYear();
        // Mirror computeRuleDay(MODE_DOW_GE_DOM, 29, SUNDAY, year, FEBRUARY)
        // step by step: anchor = min(29, monthLength); search forward for the
        // first SUNDAY >= anchor; clamp the result to monthLength. The outer
        // result clamp is what collapses the day-of-week constraint when the
        // anchor is already at the month boundary -- which is exactly the
        // case we want to exercise.
        int monthLength = Year.of(year).isLeap() ? 29 : 28;
        int anchorDay = Math.min(29, monthLength);
        int targetDow = LocalDate.of(year, Month.FEBRUARY, anchorDay).getDayOfWeek().getValue();
        int diff = DayOfWeek.SUNDAY.getValue() - targetDow;
        if (diff < 0) diff += 7;
        int computedDay = Math.min(anchorDay + diff, monthLength);
        long dstStart = LocalDate.of(year, Month.FEBRUARY, computedDay)
            .toEpochDay() * 86_400_000L + 2 * 3_600_000L;
        long dstEnd = nthDayOfWeekUtcMs(year, Month.NOVEMBER, DayOfWeek.SUNDAY, 1) + 1 * 3_600_000L;
        return (instant >= dstStart && instant < dstEnd) ? 2 * 3_600_000 : 0;
      }
      @Override public int getOffset(int era, int year, int month, int day, int dow, int ms) {
        return 0;
      }
      @Override public int getRawOffset() { return 0; }
      @Override public void setRawOffset(int offsetMillis) {}
      @Override public boolean useDaylightTime() { return true; }
      @Override public int getDSTSavings() { return 0; } // force Path B verify to fail
      @Override public boolean inDaylightTime(Date date) { return false; }
    };
    tz.setID("Synthetic/FebruaryClamp");

    ZoneOffset base = ZoneOffset.UTC;
    ZoneOffset plus2 = ZoneOffset.ofHours(2);
    ZoneOffsetTransitionRule startRule = ZoneOffsetTransitionRule.of(
        Month.FEBRUARY, 29, DayOfWeek.SUNDAY, LocalTime.of(2, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, base, plus2);
    ZoneOffsetTransitionRule endRule = ZoneOffsetTransitionRule.of(
        Month.NOVEMBER, 1, DayOfWeek.SUNDAY, LocalTime.of(1, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, plus2, base);
    ZoneRules rules = ZoneRules.of(base, base,
        Collections.emptyList(),
        Collections.singletonList(SYNTHETIC_HISTORICAL_TRANSITION),
        Arrays.asList(startRule, endRule));

    OrcDstRuleExtractor.DstRule rule = OrcDstRuleExtractor.extractDstRule(
        "Synthetic/FebruaryClamp", tz, rules);
    assertNotNull(rule, "Feb indicator=29 must not throw -- clamp should prevent DateTimeException");
    // Path A stores getDayOfMonthIndicator() unchanged; the clamp is applied
    // later in computeRuleDay. Surviving verifyDstRuleAcrossReferenceYears
    // round-trip is what proves the clamp prevented a DateTimeException.
    assertEquals(29, rule.startDay,
        "Path A stores indicator verbatim; clamp lives in computeRuleDay");
  }

  @Test
  void testExtractDstRuleByProbingReturnsNullOnMultipleDstOnTransitions() {
    // extractDstRuleByProbing returns null when probing finds more than one
    // offset-increasing transition in a year. All other Path-A tests reach
    // Path A by failing Path B's cross-year verify -- the early-return on
    // "second DST-on transition seen" was never exercised.
    //
    // This synthetic TimeZone spring-forwards twice a year (March 2nd Sunday
    // and June 1st Sunday) and falls back once (November 1st Sunday). The
    // ZoneRules supplies no recurring transition rules, so Path A also
    // returns null (Unsupported rule count). The terminal "Failed to extract"
    // IllegalStateException confirms probing returned null due to the
    // multiple-DST-on guard rather than some other reason.
    TimeZone tz = new TimeZone() {
      @Override public int getOffset(long instant) {
        int year = LocalDate.ofEpochDay(Math.floorDiv(instant, 86_400_000L)).getYear();
        long forward1 = nthDayOfWeekUtcMs(year, Month.MARCH, DayOfWeek.SUNDAY, 2) + 2 * 3_600_000L;
        long forward2 = nthDayOfWeekUtcMs(year, Month.JUNE, DayOfWeek.SUNDAY, 1) + 2 * 3_600_000L;
        long back = nthDayOfWeekUtcMs(year, Month.NOVEMBER, DayOfWeek.SUNDAY, 1) + 1 * 3_600_000L;
        if (instant >= forward1 && instant < forward2) return 1 * 3_600_000;
        if (instant >= forward2 && instant < back) return 2 * 3_600_000;
        return 0;
      }
      @Override public int getOffset(int era, int year, int month, int day, int dow, int ms) {
        return 0;
      }
      @Override public int getRawOffset() { return 0; }
      @Override public void setRawOffset(int offsetMillis) {}
      @Override public boolean useDaylightTime() { return true; }
      @Override public int getDSTSavings() { return 3_600_000; }
      @Override public boolean inDaylightTime(Date date) { return false; }
    };
    tz.setID("Synthetic/MultipleDstOn");
    ZoneOffset base = ZoneOffset.UTC;
    // A single historical transition is enough to make rules.isFixedOffset()
    // return false (without it the entry-point short-circuit at the top of
    // extractDstRule would return null before probing runs). The
    // transitionRules list stays empty so Path A also returns null on the
    // "empty getTransitionRules()" early-out, leaving the terminal
    // "Failed to extract" throw as the expected outcome.
    ZoneOffsetTransition historicalTransition = ZoneOffsetTransition.of(
        LocalDateTime.of(1900, 1, 1, 0, 0), ZoneOffset.ofHours(-1), base);
    ZoneRules rules = ZoneRules.of(base, base,
        Collections.emptyList(),
        Collections.singletonList(historicalTransition),
        Collections.emptyList());
    IllegalStateException ex = assertThrows(IllegalStateException.class,
        () -> OrcDstRuleExtractor.extractDstRule("Synthetic/MultipleDstOn", tz, rules));
    assertTrue(ex.getMessage().contains("Failed to extract"),
        "expected 'Failed to extract' in terminal throw: " + ex.getMessage());
  }

  @Test
  void testExtractDstRuleByProbingReturnsNullOnMultipleDstOffTransitions() {
    // extractDstRuleByProbing returns null when probing finds more than one
    // offset-decreasing transition in a year. This is the symmetric counterpart
    // of testExtractDstRuleByProbingReturnsNullOnMultipleDstOnTransitions: it
    // exercises the second-DST-off early-return in DstTransitions
    // .recordDstOffTransition, which no other test reaches.
    //
    // This synthetic TimeZone springs forward once (March 2nd Sunday) and falls
    // back twice (July 1st Sunday and November 1st Sunday). The first fall-back
    // records dstOffTransition; the second trips the "more than one DST-off"
    // guard, so recordTransition returns false and probing returns null. The
    // ZoneRules supplies no recurring transition rules, so Path A also returns
    // null (empty getTransitionRules early-out). The terminal "Failed to
    // extract" IllegalStateException confirms probing returned null due to the
    // multiple-DST-off guard rather than some other reason.
    TimeZone tz = new TimeZone() {
      @Override public int getOffset(long instant) {
        int year = LocalDate.ofEpochDay(Math.floorDiv(instant, 86_400_000L)).getYear();
        long forward = nthDayOfWeekUtcMs(year, Month.MARCH, DayOfWeek.SUNDAY, 2) + 2 * 3_600_000L;
        long back1 = nthDayOfWeekUtcMs(year, Month.JULY, DayOfWeek.SUNDAY, 1) + 1 * 3_600_000L;
        long back2 = nthDayOfWeekUtcMs(year, Month.NOVEMBER, DayOfWeek.SUNDAY, 1) + 1 * 3_600_000L;
        if (instant >= forward && instant < back1) return 2 * 3_600_000;
        if (instant >= back1 && instant < back2) return 1 * 3_600_000;
        return 0;
      }
      @Override public int getOffset(int era, int year, int month, int day, int dow, int ms) {
        return 0;
      }
      @Override public int getRawOffset() { return 0; }
      @Override public void setRawOffset(int offsetMillis) {}
      @Override public boolean useDaylightTime() { return true; }
      @Override public int getDSTSavings() { return 2 * 3_600_000; }
      @Override public boolean inDaylightTime(Date date) { return false; }
    };
    tz.setID("Synthetic/MultipleDstOff");
    ZoneOffset base = ZoneOffset.UTC;
    // A single historical transition is enough to make rules.isFixedOffset()
    // return false (without it the entry-point short-circuit at the top of
    // extractDstRule would return null before probing runs). The
    // transitionRules list stays empty so Path A also returns null on the
    // "empty getTransitionRules()" early-out, leaving the terminal
    // "Failed to extract" throw as the expected outcome.
    ZoneOffsetTransition historicalTransition = ZoneOffsetTransition.of(
        LocalDateTime.of(1900, 1, 1, 0, 0), ZoneOffset.ofHours(-1), base);
    ZoneRules rules = ZoneRules.of(base, base,
        Collections.emptyList(),
        Collections.singletonList(historicalTransition),
        Collections.emptyList());
    IllegalStateException ex = assertThrows(IllegalStateException.class,
        () -> OrcDstRuleExtractor.extractDstRule("Synthetic/MultipleDstOff", tz, rules));
    assertTrue(ex.getMessage().contains("Failed to extract"),
        "expected 'Failed to extract' in terminal throw: " + ex.getMessage());
  }

  @Test
  void testExtractDstRuleViaZoneRulesFallback() {
    // Path A (ZoneRules fallback) success scenario. The custom TimeZone
    // observes a real +2h DST window but reports getDSTSavings()==+1h.
    // Probing therefore extracts a rule with dstSavings=+1h, whose
    // computeDstOffset prediction (rawOffset + 1h) disagrees with the
    // observed +2h inside the DST window — verifyDstRule fails, so
    // extractDstRuleByProbing returns null. The synthetic ZoneRules below
    // carries the actual +2h delta; extractDstRuleFromZoneRules re-derives
    // dstSavings from the rule's offset deltas, verify passes, and the
    // returned DstRule has dstSavings=+2h (proving Path A ran).
    //
    // The end rule uses TimeDefinition.UTC to exercise
    // getTransitionRuleTimeMode's TIME_MODE_UTC branch — the only
    // production path that produces non-STANDARD timeMode in DstRule.
    TimeZone tz = new TimeZone() {
      @Override public int getOffset(long instant) {
        LocalDate date = LocalDate.ofEpochDay(Math.floorDiv(instant, 86_400_000L));
        int year = date.getYear();
        long dstStart = nthDayOfWeekUtcMs(year, Month.MARCH, DayOfWeek.SUNDAY, 2) + 2 * 3_600_000L;
        long dstEnd = nthDayOfWeekUtcMs(year, Month.NOVEMBER, DayOfWeek.SUNDAY, 1) + 2 * 3_600_000L;
        return (instant >= dstStart && instant < dstEnd) ? 2 * 3_600_000 : 0;
      }
      @Override public int getOffset(int era, int year, int month, int day, int dow, int ms) {
        return 0; // never called by extractDstRule
      }
      @Override public int getRawOffset() { return 0; }
      @Override public void setRawOffset(int offsetMillis) {}
      @Override public boolean useDaylightTime() { return true; }
      @Override public int getDSTSavings() { return 1 * 3_600_000; } // intentionally wrong
      @Override public boolean inDaylightTime(Date date) { return false; }
    };
    tz.setID("Synthetic/PathASuccess");

    ZoneOffset base = ZoneOffset.UTC;
    ZoneOffset plus2 = ZoneOffset.ofHours(2);
    ZoneOffsetTransitionRule startRule = ZoneOffsetTransitionRule.of(
        Month.MARCH, 8, DayOfWeek.SUNDAY, LocalTime.of(2, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, base, plus2);
    ZoneOffsetTransitionRule endRule = ZoneOffsetTransitionRule.of(
        Month.NOVEMBER, 1, DayOfWeek.SUNDAY, LocalTime.of(2, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.UTC,
        base, plus2, base);
    ZoneRules rules = ZoneRules.of(base, base,
        Collections.emptyList(),
        Collections.singletonList(SYNTHETIC_HISTORICAL_TRANSITION),
        Arrays.asList(startRule, endRule));

    OrcDstRuleExtractor.DstRule rule = OrcDstRuleExtractor.extractDstRule(
        "Synthetic/PathASuccess", tz, rules);
    assertNotNull(rule, "Path A must succeed");
    // dstSavings derived from rule deltas = +2h. If Path B had succeeded
    // we would see +1h instead (tz.getDSTSavings).
    assertEquals(2 * 3_600_000, rule.dstSavings,
        "Path A derives dstSavings from offset deltas, not tz.getDSTSavings()");
    assertEquals(2, rule.startMonth);
    assertEquals(8, rule.startDay);
    assertEquals(1, rule.startDayOfWeek);
    assertEquals(2, rule.startMode);
    assertEquals(1, rule.startTimeMode);  // STANDARD
    assertEquals(2 * 3_600_000, rule.startTime);
    assertEquals(10, rule.endMonth);
    assertEquals(1, rule.endDay);
    assertEquals(1, rule.endDayOfWeek);
    assertEquals(2, rule.endMode);
    assertEquals(2, rule.endTimeMode);  // UTC — covers TIME_MODE_UTC branch
    assertEquals(2 * 3_600_000, rule.endTime);
  }

  @Test
  void testExtractDstRuleMidnightEndOfDayRule() {
    // ZoneOffsetTransitionRule.of(..., midnightEndOfDay=true, ...) encodes a
    // transition at 24:00 (start of the next day). getTransitionRuleTimeMillis
    // must return 24 * 3_600_000 for this case, NOT
    // LocalTime.MIDNIGHT.toSecondOfDay() = 0. The TimeZone fires its DST start
    // at March 2nd-Sunday + 24h (i.e., the 0:00 boundary of the following
    // Monday) and reports getDSTSavings()==0 so Path B fails verify and Path A
    // runs; the extracted rule's startTime is asserted to be 24h.
    TimeZone tz = new TimeZone() {
      @Override public int getOffset(long instant) {
        int year = LocalDate.ofEpochDay(Math.floorDiv(instant, 86_400_000L)).getYear();
        long dstStart = nthDayOfWeekUtcMs(year, Month.MARCH, DayOfWeek.SUNDAY, 2) + 24 * 3_600_000L;
        long dstEnd = nthDayOfWeekUtcMs(year, Month.NOVEMBER, DayOfWeek.SUNDAY, 1) + 2 * 3_600_000L;
        return (instant >= dstStart && instant < dstEnd) ? 2 * 3_600_000 : 0;
      }
      @Override public int getOffset(int era, int year, int month, int day, int dow, int ms) {
        return 0;
      }
      @Override public int getRawOffset() { return 0; }
      @Override public void setRawOffset(int offsetMillis) {}
      @Override public boolean useDaylightTime() { return true; }
      @Override public int getDSTSavings() { return 0; } // force Path B verify to fail
      @Override public boolean inDaylightTime(Date date) { return false; }
    };
    tz.setID("Synthetic/MidnightEndOfDay");

    ZoneOffset base = ZoneOffset.UTC;
    ZoneOffset plus2 = ZoneOffset.ofHours(2);
    ZoneOffsetTransitionRule startRule = ZoneOffsetTransitionRule.of(
        Month.MARCH, 8, DayOfWeek.SUNDAY, LocalTime.MIDNIGHT, true,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, base, plus2);
    ZoneOffsetTransitionRule endRule = ZoneOffsetTransitionRule.of(
        Month.NOVEMBER, 1, DayOfWeek.SUNDAY, LocalTime.of(2, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, plus2, base);
    ZoneRules rules = ZoneRules.of(base, base,
        Collections.emptyList(),
        Collections.singletonList(SYNTHETIC_HISTORICAL_TRANSITION),
        Arrays.asList(startRule, endRule));

    OrcDstRuleExtractor.DstRule rule = OrcDstRuleExtractor.extractDstRule(
        "Synthetic/MidnightEndOfDay", tz, rules);
    assertNotNull(rule, "Path A with midnight-end-of-day start rule must succeed");
    assertEquals(24 * 3_600_000, rule.startTime,
        "isMidnightEndOfDay must produce time=24h, not LocalTime.MIDNIGHT.toSecondOfDay()=0");
  }

  @Test
  void testExtractDstRuleTimeDefinitionWallMode() {
    // Path A success with start rule encoded as TimeDefinition.WALL —
    // exercises both TIME_MODE_WALL in getTransitionRuleTimeMode and the
    // wall-time branch in computeTransitionUtcMillis. The custom TimeZone
    // observes a +2h DST window from 2nd Sun of March 02:00 UTC to 1st Sun
    // of November 01:00 UTC but reports getDSTSavings()==0 — Path B's
    // probing extracts dstSavings=0 from getDSTSavings, verifyDstRule then
    // disagrees (predicted 0, observed +2h), Path B returns null. Path A's
    // ZoneRules deltas yield dstSavings=+2h, verify passes, the WALL start
    // rule is encoded as startTimeMode=0.
    TimeZone tz = new TimeZone() {
      @Override public int getOffset(long instant) {
        int year = LocalDate.ofEpochDay(Math.floorDiv(instant, 86_400_000L)).getYear();
        long dstStart = nthDayOfWeekUtcMs(year, Month.MARCH, DayOfWeek.SUNDAY, 2) + 2 * 3_600_000L;
        long dstEnd = nthDayOfWeekUtcMs(year, Month.NOVEMBER, DayOfWeek.SUNDAY, 1) + 1 * 3_600_000L;
        return (instant >= dstStart && instant < dstEnd) ? 2 * 3_600_000 : 0;
      }
      @Override public int getOffset(int era, int year, int month, int day, int dow, int ms) {
        return 0;
      }
      @Override public int getRawOffset() { return 0; }
      @Override public void setRawOffset(int offsetMillis) {}
      @Override public boolean useDaylightTime() { return true; }
      @Override public int getDSTSavings() { return 0; } // forces Path B verify to fail
      @Override public boolean inDaylightTime(Date date) { return false; }
    };
    tz.setID("Synthetic/WallMode");

    ZoneOffset base = ZoneOffset.UTC;
    ZoneOffset plus2 = ZoneOffset.ofHours(2);
    ZoneOffsetTransitionRule startRule = ZoneOffsetTransitionRule.of(
        Month.MARCH, 8, DayOfWeek.SUNDAY, LocalTime.of(2, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.WALL,
        base, base, plus2);
    ZoneOffsetTransitionRule endRule = ZoneOffsetTransitionRule.of(
        Month.NOVEMBER, 1, DayOfWeek.SUNDAY, LocalTime.of(1, 0), false,
        ZoneOffsetTransitionRule.TimeDefinition.STANDARD,
        base, plus2, base);
    ZoneRules rules = ZoneRules.of(base, base,
        Collections.emptyList(),
        Collections.singletonList(SYNTHETIC_HISTORICAL_TRANSITION),
        Arrays.asList(startRule, endRule));

    OrcDstRuleExtractor.DstRule rule = OrcDstRuleExtractor.extractDstRule(
        "Synthetic/WallMode", tz, rules);
    assertNotNull(rule, "Path A with WALL start rule must succeed");
    assertEquals(0, rule.startTimeMode,
        "TimeDefinition.WALL must produce TIME_MODE_WALL (0)");
    assertEquals(1, rule.endTimeMode,
        "TimeDefinition.STANDARD must produce TIME_MODE_STANDARD (1)");
  }
}
