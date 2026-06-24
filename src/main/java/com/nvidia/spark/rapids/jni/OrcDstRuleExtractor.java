/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.zone.ZoneOffsetTransitionRule;
import java.time.zone.ZoneRules;
import java.util.List;
import java.util.TimeZone;

/**
 * Recovers a recurring DST rule for an IANA timezone in the shape the GPU
 * side consumes for ORC DST conversion.
 *
 * <p>Two extraction paths are supported. The probing path is tried first
 * because it captures what {@link TimeZone#getOffset(long)} actually returns
 * (which is the source of truth the GPU side must match for ORC byte
 * compatibility); the {@link ZoneRules#getTransitionRules()} path is used as
 * a fallback for zones whose recurring rule cannot be recovered from hourly
 * probes.
 *
 * <p>This class is intentionally separate from {@link OrcTimezoneInfo} — the
 * historical-transition machinery and the DST-rule extraction share no
 * read/write state, just a couple of small package-private helpers
 * ({@link OrcTimezoneInfo#utcMillisForDate} and
 * {@link OrcTimezoneInfo#binarySearchTransition}).
 */
final class OrcDstRuleExtractor {
  private OrcDstRuleExtractor() {}

  // Reference years used to cross-check the extracted DST rule against
  // java.util.TimeZone.getOffset. A near-future anchor (2060) catches
  // divergence within the typical application lifetime; the far-future
  // anchors exercise the recurring-rule fallback well past any historical
  // transition entry in tzdata.
  private static final int[] DST_RULE_VALIDATION_YEARS = {2060, 2400, 9997};

  /**
   * Recurring DST rule for a single zone, encoded in the same shape that
   * {@link java.util.SimpleTimeZone} stores internally and that the GPU side
   * consumes. {@code month} is 0-based (Calendar.JANUARY=0), {@code dayOfWeek}
   * follows Calendar's 1=Sun..7=Sat convention, and {@code dstSavings} /
   * {@code time} are in milliseconds.
   *
   * <p>The {@code *Mode} fields encode how {@code *Day}/{@code *DayOfWeek}
   * combine — see the {@code MODE_*} constants. {@code *TimeMode} selects
   * the time-of-day basis — see the {@code TIME_MODE_*} constants.
   */
  static final class DstRule {
    // Day-rule modes for {start,end}Mode — matches SimpleTimeZone's internal
    // encoding so the GPU side can consume the values directly.
    static final int MODE_DOM          = 0;
    static final int MODE_DOW_IN_MONTH = 1;
    static final int MODE_DOW_GE_DOM   = 2;
    static final int MODE_DOW_LE_DOM   = 3;

    // Time-of-day basis for {start,end}TimeMode.
    static final int TIME_MODE_WALL     = 0;
    static final int TIME_MODE_STANDARD = 1;
    static final int TIME_MODE_UTC      = 2;

    int dstSavings;
    int startMonth;
    int startDay;
    int startDayOfWeek;
    int startTime;
    int startTimeMode;
    int startMode;
    int endMonth;
    int endDay;
    int endDayOfWeek;
    int endTime;
    int endTimeMode;
    int endMode;
  }

  /**
   * Extract the recurring DST rule for a zone, or {@code null} if the zone has
   * no DST.
   *
   * @param timezoneId IANA timezone id; used only in exception messages
   * @param tz {@link TimeZone} for the zone; must describe the same zone as
   *     {@code rules}
   * @param rules {@link ZoneRules} for the zone
   * @return the recurring DST rule, or {@code null} if the zone has no DST
   *     ({@code rules.isFixedOffset()} or {@code !tz.useDaylightTime()})
   * @throws IllegalStateException if the zone reports DST but neither
   *     extraction path produces a usable rule — for example, an unsupported
   *     {@link ZoneRules#getTransitionRules()} count (not 0 and not 2), a
   *     transition rule shape outside DOW_GE_DOM, mismatched start vs. end
   *     DST savings, zero-delta savings, or cross-year verification mismatch
   *     against {@code tz.getOffset} on the anchor years
   *     {@code 2060, 2400, 9997}.
   */
  static DstRule extractDstRule(String timezoneId, TimeZone tz, ZoneRules rules) {
    // Fixed-offset zones (e.g. "+05:30") have no DST. Guard explicitly because
    // TimeZone.getTimeZone(zoneId) silently returns GMT for such ids on most
    // JVMs, which would leave `tz` describing a different zone than `rules`.
    // Mirrors the guard in OrcTimezoneInfo.buildRuntimeOrcTimezoneInfo.
    if (rules.isFixedOffset() || !tz.useDaylightTime()) {
      return null;
    }
    // Sanity-check that tz and rules describe the same zone. Both Path A and
    // Path B assume this, but the two-argument signature lets a caller pass
    // mismatched objects (the silent-GMT trap above is one way this can
    // happen). Compare standard offsets at a recent reference instant -- the
    // epoch is unsafe because some zones (e.g. Europe/London) were on a
    // different standard offset in 1970 (British Standard Time experiment).
    Instant ref = Instant.parse("2024-01-15T00:00:00Z");
    int rulesStandardOffsetMillis = rules.getStandardOffset(ref).getTotalSeconds() * 1000;
    if (tz.getRawOffset() != rulesStandardOffsetMillis) {
      throw new IllegalStateException(
          "TimeZone and ZoneRules describe different zones for timezone: " + timezoneId
              + " (tz.rawOffset=" + tz.getRawOffset()
              + ", rules.standardOffset=" + rulesStandardOffsetMillis + ")");
    }
    DstRule rule = extractDstRuleByProbing(tz);
    if (rule != null) {
      return rule;
    }
    rule = extractDstRuleFromZoneRules(timezoneId, tz, rules);
    if (rule != null) {
      return rule;
    }
    throw new IllegalStateException("Failed to extract ORC DST rule for timezone: " + timezoneId);
  }

  // ---- Path A: from ZoneRules.getTransitionRules() ----

  private static DstRule extractDstRuleFromZoneRules(String timezoneId, TimeZone tz,
      ZoneRules rules) {
    List<ZoneOffsetTransitionRule> transitionRules = rules.getTransitionRules();
    if (transitionRules.isEmpty()) {
      return null;
    }
    if (transitionRules.size() != 2) {
      throw new IllegalStateException("Unsupported ORC DST rule count for timezone: " + timezoneId);
    }

    ZoneOffsetTransitionRule startTransitionRule = null;
    ZoneOffsetTransitionRule endTransitionRule = null;
    for (ZoneOffsetTransitionRule transitionRule : transitionRules) {
      int deltaMillis = (transitionRule.getOffsetAfter().getTotalSeconds()
          - transitionRule.getOffsetBefore().getTotalSeconds()) * 1000;
      if (deltaMillis > 0) {
        startTransitionRule = transitionRule;
      } else if (deltaMillis < 0) {
        endTransitionRule = transitionRule;
      } else {
        throw new IllegalStateException("Unsupported zero-delta ORC DST rule for timezone: "
            + timezoneId);
      }
    }
    if (startTransitionRule == null || endTransitionRule == null) {
      throw new IllegalStateException("Failed to identify ORC DST start/end rules for timezone: "
          + timezoneId);
    }

    int dstSavings = (startTransitionRule.getOffsetAfter().getTotalSeconds()
        - startTransitionRule.getOffsetBefore().getTotalSeconds()) * 1000;
    int endDeltaMillis = (endTransitionRule.getOffsetBefore().getTotalSeconds()
        - endTransitionRule.getOffsetAfter().getTotalSeconds()) * 1000;
    if (dstSavings != endDeltaMillis) {
      throw new IllegalStateException("Mismatched ORC DST savings for timezone: " + timezoneId);
    }

    DstRule rule = new DstRule();
    rule.dstSavings = dstSavings;
    fillDstRuleFromTransitionRule(timezoneId, rule, startTransitionRule, true);
    fillDstRuleFromTransitionRule(timezoneId, rule, endTransitionRule, false);

    if (!verifyDstRuleAcrossReferenceYears(tz, rule)) {
      throw new IllegalStateException("ZoneRules ORC DST rule verification failed for timezone: "
          + timezoneId);
    }
    return rule;
  }

  private static void fillDstRuleFromTransitionRule(String timezoneId, DstRule rule,
      ZoneOffsetTransitionRule transitionRule, boolean isStartRule) {
    // We only accept rules shaped as "first <dayOfWeek> on or after <dayOfMonthIndicator>",
    // i.e. ZoneRules' positive-day-indicator form. A negative indicator would mean
    // "last <dayOfWeek> on or before day" (DOW_LE_DOM_MODE, mode=3); we reject those
    // here so that downstream code can assume DOW_GE_DOM_MODE (mode=2) unconditionally.
    if (transitionRule.getDayOfWeek() == null
        || transitionRule.getDayOfMonthIndicator() <= 0) {
      throw new IllegalStateException("Unsupported ORC DST transition rule shape for timezone: "
          + timezoneId);
    }

    int month = transitionRule.getMonth().getValue() - 1;
    int day = transitionRule.getDayOfMonthIndicator();
    int dayOfWeek = toCalendarDayOfWeek(transitionRule.getDayOfWeek().getValue());
    int time = getTransitionRuleTimeMillis(transitionRule);
    int timeMode = getTransitionRuleTimeMode(transitionRule);
    // Guaranteed by the precondition above (DayOfMonthIndicator > 0).
    int mode = DstRule.MODE_DOW_GE_DOM;

    if (isStartRule) {
      rule.startMonth = month;
      rule.startDay = day;
      rule.startDayOfWeek = dayOfWeek;
      rule.startTime = time;
      rule.startTimeMode = timeMode;
      rule.startMode = mode;
    } else {
      rule.endMonth = month;
      rule.endDay = day;
      rule.endDayOfWeek = dayOfWeek;
      rule.endTime = time;
      rule.endTimeMode = timeMode;
      rule.endMode = mode;
    }
  }

  private static int getTransitionRuleTimeMillis(ZoneOffsetTransitionRule transitionRule) {
    int secondOfDay = transitionRule.isMidnightEndOfDay()
        ? 24 * 3_600
        : transitionRule.getLocalTime().toSecondOfDay();
    return secondOfDay * 1000;
  }

  private static int getTransitionRuleTimeMode(ZoneOffsetTransitionRule transitionRule) {
    ZoneOffsetTransitionRule.TimeDefinition timeDef = transitionRule.getTimeDefinition();
    if (ZoneOffsetTransitionRule.TimeDefinition.UTC == timeDef) {
      return DstRule.TIME_MODE_UTC;
    } else if (ZoneOffsetTransitionRule.TimeDefinition.STANDARD == timeDef) {
      return DstRule.TIME_MODE_STANDARD;
    } else {
      return DstRule.TIME_MODE_WALL;
    }
  }

  private static int toCalendarDayOfWeek(int javaTimeDayOfWeek) {
    // java.time DayOfWeek: 1=Mon..7=Sun  ->  Calendar: 1=Sun..7=Sat
    return (javaTimeDayOfWeek % 7) + 1;
  }

  // ---- Path B: probe tz.getOffset() and recover the recurring rule ----

  private static DstRule extractDstRuleByProbing(TimeZone tz) {
    for (int refYear : DST_RULE_VALIDATION_YEARS) {
      DstRule rule = extractDstRuleByProbing(tz, refYear);
      if (rule != null && verifyDstRuleAcrossReferenceYears(tz, rule)) {
        return rule;
      }
    }
    return null;
  }

  private static DstRule extractDstRuleByProbing(TimeZone tz, int refYear) {
    DstTransitions transitions = findDstTransitionsByProbing(tz, refYear);
    if (transitions == null || !transitions.hasBothTransitions()) {
      return null;
    }

    return buildDstRuleFromProbedTransitions(tz, transitions);
  }

  private static DstTransitions findDstTransitionsByProbing(TimeZone tz, int refYear) {
    long janFirst = OrcTimezoneInfo.utcMillisForDate(refYear, 1, 1);
    long nextJanFirst = OrcTimezoneInfo.utcMillisForDate(refYear + 1, 1, 1);

    int initialOffset = tz.getOffset(janFirst - 1);
    int prevOffset = initialOffset;
    long step = 3_600_000L; // 1 hour
    DstTransitions transitions = new DstTransitions(initialOffset);

    for (long ms = janFirst; ms < nextJanFirst; ms += step) {
      int curOffset = tz.getOffset(ms);
      if (curOffset == prevOffset) {
        continue;
      }

      long exactMs = OrcTimezoneInfo.binarySearchTransition(tz, ms - step, ms);
      if (!transitions.recordTransition(prevOffset, curOffset, exactMs)) {
        return null;
      }
      prevOffset = curOffset;
      // SimpleTimeZone-style zones have exactly two transitions per year.
      // Once both are recorded and the running offset is back to the year-
      // start standard offset, the remainder of the year is a no-op tail
      // (~1-3 months for typical northern-hemisphere zones). Any later
      // change would correctly trigger the "more than one DST-on/off"
      // early-return above, so the only thing this break can skip is
      // wasted probes -- ~720-2200 calls per year on common DST zones.
      if (transitions.hasCompleteRuleAtOffset(prevOffset)) {
        break;
      }
    }

    return transitions;
  }

  private static DstRule buildDstRuleFromProbedTransitions(TimeZone tz,
      DstTransitions transitions) {
    DstRule rule = new DstRule();
    rule.dstSavings = tz.getDSTSavings();

    int[] startFields = decodeTransition(transitions.dstOnTransition, tz.getRawOffset());
    rule.startMonth = startFields[0];
    rule.startDay = startFields[1];
    rule.startDayOfWeek = startFields[2];
    rule.startTime = startFields[3];
    // decodeTransition converts to standard local time.
    rule.startTimeMode = DstRule.TIME_MODE_STANDARD;
    rule.startMode = startFields[4];

    int[] endFields = decodeTransition(transitions.dstOffTransition, tz.getRawOffset());
    rule.endMonth = endFields[0];
    rule.endDay = endFields[1];
    rule.endDayOfWeek = endFields[2];
    rule.endTime = endFields[3];
    rule.endTimeMode = DstRule.TIME_MODE_STANDARD;
    rule.endMode = endFields[4];

    return rule;
  }

  private static final class DstTransitions {
    private static final long MISSING_TRANSITION = -1;

    private final int initialOffset;
    private long dstOnTransition = MISSING_TRANSITION;
    private long dstOffTransition = MISSING_TRANSITION;

    private DstTransitions(int initialOffset) {
      this.initialOffset = initialOffset;
    }

    private boolean recordTransition(int prevOffset, int curOffset, long exactMs) {
      if (curOffset > prevOffset) {
        return recordDstOnTransition(exactMs);
      }
      return recordDstOffTransition(exactMs);
    }

    private boolean recordDstOnTransition(long exactMs) {
      // More than one DST-on transition in the same year would mean this year
      // doesn't fit a SimpleTimeZone-style two-transition rule; let the caller
      // fall back to extractDstRuleFromZoneRules.
      if (dstOnTransition != MISSING_TRANSITION) {
        return false;
      }
      dstOnTransition = exactMs;
      return true;
    }

    private boolean recordDstOffTransition(long exactMs) {
      if (dstOffTransition != MISSING_TRANSITION) {
        return false;
      }
      dstOffTransition = exactMs;
      return true;
    }

    private boolean hasCompleteRuleAtOffset(int offset) {
      return hasBothTransitions() && offset == initialOffset;
    }

    private boolean hasBothTransitions() {
      return dstOnTransition != MISSING_TRANSITION
          && dstOffTransition != MISSING_TRANSITION;
    }
  }

  /**
   * Decode a UTC transition instant into
   * {@code [month(0-11), baseDay, dayOfWeek(1-7), timeMs, mode]}.
   *
   * <p><b>Precondition:</b> {@code utcMs} is the first millisecond at which
   * the new offset applies — the convention returned by
   * {@link OrcTimezoneInfo#binarySearchTransition}. With that convention,
   * {@code utcMs + rawOffsetMs} recovers the standard-time wall clock at
   * the transition moment for both spring-forward (offset before == rawOffset)
   * and fall-back (offset after == rawOffset). If a future change shifts the
   * binary-search return convention by one ms, the decoded {@code timeMs}
   * will drift and {@code baseDay} can cross a midnight boundary.
   *
   * <p>Recurring weekday rules are encoded as DOW_GE_DOM (mode=2). The base
   * day is the earliest possible day of the matching occurrence in the month:
   * 1st=1, 2nd=8, 3rd=15, 4th=22, last={@code monthLength - 6}. This mirrors
   * encodings like "Sun >= 8" for the second Sunday in March and "Sun >= 25"
   * for the last Sunday in October.
   */
  private static int[] decodeTransition(long utcMs, int rawOffsetMs) {
    long localMs = utcMs + rawOffsetMs;
    LocalDateTime ldt = LocalDateTime.ofInstant(Instant.ofEpochMilli(localMs), ZoneOffset.UTC);

    int month = ldt.getMonthValue() - 1; // 0-based for Calendar compat
    int dayOfMonth = ldt.getDayOfMonth();
    int dayOfWeek = toCalendarDayOfWeek(ldt.getDayOfWeek().getValue());
    int timeInDay = ldt.getHour() * 3_600_000
        + ldt.getMinute() * 60_000
        + ldt.getSecond() * 1000
        + ldt.getNano() / 1_000_000;

    int monthLength = ldt.toLocalDate().lengthOfMonth();
    int dayOfWeekInMonth = (dayOfMonth - 1) / 7 + 1;
    boolean isLastOccurrence = dayOfMonth + 7 > monthLength;
    int baseDayOfMonth = isLastOccurrence
        ? monthLength - 6
        : 1 + (dayOfWeekInMonth - 1) * 7;

    return new int[]{month, baseDayOfMonth, dayOfWeek, timeInDay, DstRule.MODE_DOW_GE_DOM};
  }

  // ---- Verification: ensure the extracted rule matches tz.getOffset ----

  private static boolean verifyDstRuleAcrossReferenceYears(TimeZone tz, DstRule rule) {
    for (int refYear : DST_RULE_VALIDATION_YEARS) {
      if (!verifyDstRule(tz, rule, refYear)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Verify the extracted rule matches {@code tz.getOffset} around transition
   * boundaries and at monthly sample points for {@code refYear ± 1} (3 years).
   * DST mismatches only manifest near transitions, so dense sampling at
   * boundaries plus monthly spot checks reduces a naive full-year scan from
   * ~52K probes to ~200 per reference year.
   */
  private static boolean verifyDstRule(TimeZone tz, DstRule rule, int refYear) {
    int rawOffsetMs = tz.getRawOffset();
    for (int y = refYear - 1; y <= refYear + 1; y++) {
      long dstStart = computeTransitionUtcMillis(y, rule.startMonth, rule.startDay,
          rule.startDayOfWeek, rule.startTime, rule.startTimeMode, rule.startMode,
          rawOffsetMs, rule.dstSavings, true);
      long dstEnd = computeTransitionUtcMillis(y, rule.endMonth, rule.endDay,
          rule.endDayOfWeek, rule.endTime, rule.endTimeMode, rule.endMode,
          rawOffsetMs, rule.dstSavings, false);

      // All sample points below land within the wall-clock year y for any
      // real DST rule (boundaries ±12h, plus monthly noon-UTC anchors), so
      // skip the per-sample year derivation and re-computation of these
      // bounds inside computeDstOffset -- ~186 computeTransitionUtcMillis
      // calls per verifyDstRule otherwise. Hand the already-computed bounds
      // to the variant.
      long[] boundaries = {dstStart, dstEnd};
      for (long boundary : boundaries) {
        long from = boundary - 12 * 3_600_000L;
        long to = boundary + 12 * 3_600_000L;
        for (long ms = from; ms <= to; ms += 3_600_000L) {
          if (tz.getOffset(ms) != computeDstOffsetWithBounds(ms, rawOffsetMs, rule,
              dstStart, dstEnd)) {
            return false;
          }
        }
      }

      for (int m = 1; m <= 12; m++) {
        long ms = OrcTimezoneInfo.utcMillisForDate(y, m, 1) + 12 * 3_600_000L;
        if (tz.getOffset(ms) != computeDstOffsetWithBounds(ms, rawOffsetMs, rule,
            dstStart, dstEnd)) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * In-DST classification with bounds supplied by the caller. The only caller
   * is {@link #verifyDstRule}, which already computes {@code dstStart} and
   * {@code dstEnd} for the year being checked; passing them in avoids
   * recomputing them per sample. A general-purpose entry point that derives
   * the year from {@code utcMs} can be reintroduced in a follow-up part when
   * a non-verification caller appears.
   */
  private static int computeDstOffsetWithBounds(long utcMs, int rawOffsetMs, DstRule rule,
      long dstStart, long dstEnd) {
    boolean inDst = dstStart < dstEnd
        ? (utcMs >= dstStart && utcMs < dstEnd)
        : (utcMs >= dstStart || utcMs < dstEnd);
    return inDst ? rawOffsetMs + rule.dstSavings : rawOffsetMs;
  }

  private static long computeTransitionUtcMillis(int year, int ruleMonth, int ruleDay,
      int ruleDayOfWeek, int ruleTime, int ruleTimeMode, int ruleMode, int rawOffsetMs,
      int dstSavingsMs, boolean isStartRule) {
    int actualDay = computeRuleDay(ruleMode, ruleDay, ruleDayOfWeek, year, ruleMonth);
    long utcMs = OrcTimezoneInfo.utcMillisForDate(year, ruleMonth + 1, actualDay) + ruleTime;
    if (ruleTimeMode == DstRule.TIME_MODE_WALL) {
      // WALL time: subtract raw offset and (for end transitions) also DST savings.
      utcMs -= rawOffsetMs;
      if (!isStartRule) {
        utcMs -= dstSavingsMs;
      }
    } else if (ruleTimeMode == DstRule.TIME_MODE_STANDARD) {
      utcMs -= rawOffsetMs;
    }
    // TIME_MODE_UTC is already in UTC.
    return utcMs;
  }

  private static int computeRuleDay(int ruleMode, int ruleDay, int ruleDayOfWeek, int year,
      int month) {
    LocalDate firstOfMonth = LocalDate.of(year, month + 1, 1);
    int monthLength = firstOfMonth.lengthOfMonth();
    int firstDayOfWeek = toCalendarDayOfWeek(firstOfMonth.getDayOfWeek().getValue());

    // The current extraction paths (Path A's fillDstRuleFromTransitionRule and
    // Path B's decodeTransition) emit only MODE_DOW_GE_DOM. The other three
    // branches are kept for forward compatibility -- e.g. a Part 3+ caller that
    // constructs a DstRule directly from a serialised form. Until such a
    // caller exists they are not reachable from any test; add coverage when
    // one appears.
    switch (ruleMode) {
      case DstRule.MODE_DOW_IN_MONTH: {
        // Clamp the result into [1, monthLength] so a "Nth occurrence" that
        // overflows the month (e.g. 5th Sunday in a 28-day February) or
        // underflows (e.g. -5th occurrence in a 28-day month) collapses to a
        // valid in-month day rather than escaping with a DateTimeException
        // from utcMillisForDate. Mirrors the within-month clamp applied to
        // MODE_DOW_GE_DOM below and SimpleTimeZone's documented behaviour.
        if (ruleDay > 0) {
          int diff = ruleDayOfWeek - firstDayOfWeek;
          if (diff < 0) diff += 7;
          return Math.min(1 + diff + (ruleDay - 1) * 7, monthLength);
        } else {
          int lastDayOfWeek = toCalendarDayOfWeek(
              LocalDate.of(year, month + 1, monthLength).getDayOfWeek().getValue());
          int diff = lastDayOfWeek - ruleDayOfWeek;
          if (diff < 0) diff += 7;
          return Math.max(monthLength - diff + (ruleDay + 1) * 7, 1);
        }
      }
      case DstRule.MODE_DOW_GE_DOM: {
        // Per ZoneOffsetTransitionRule.getDayOfMonthIndicator(), the indicator
        // may exceed monthLength (e.g. Feb 29 in a non-leap year, treated as
        // Mar 1). Clamp the anchor before LocalDate.of so it never throws, and
        // clamp the result so a DOW_GE_DOM rule whose computed day overflows
        // the month produces a valid in-month day rather than escaping with a
        // DateTimeException from utcMillisForDate. This mirrors SimpleTimeZone's
        // documented within-month clamp for DOW_GE_DOM.
        int anchorDay = Math.min(ruleDay, monthLength);
        int targetDayOfWeek = toCalendarDayOfWeek(
            LocalDate.of(year, month + 1, anchorDay).getDayOfWeek().getValue());
        int diff = ruleDayOfWeek - targetDayOfWeek;
        if (diff < 0) diff += 7;
        return Math.min(anchorDay + diff, monthLength);
      }
      case DstRule.MODE_DOW_LE_DOM: {
        // Mirrors the MODE_DOW_GE_DOM clamp above: the day-of-month indicator
        // can exceed monthLength (e.g. 31 in February). Clamp the anchor
        // before LocalDate.of so it never throws, and clamp the result to a
        // valid in-month day so utcMillisForDate cannot receive day <= 0.
        int anchorDay = Math.min(ruleDay, monthLength);
        int targetDayOfWeek = toCalendarDayOfWeek(
            LocalDate.of(year, month + 1, anchorDay).getDayOfWeek().getValue());
        int diff = targetDayOfWeek - ruleDayOfWeek;
        if (diff < 0) diff += 7;
        return Math.max(anchorDay - diff, 1);
      }
      case DstRule.MODE_DOM:
      default:
        return ruleDay;
    }
  }
}
