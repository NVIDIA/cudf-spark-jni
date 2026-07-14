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

import java.time.DateTimeException;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.time.zone.ZoneOffsetTransition;
import java.time.zone.ZoneRules;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TimeZone;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Holds ORC timezone metadata generated at runtime from public java.time/java.util APIs.
 * Historical transitions come from ZoneRules, while offsets before the first transition and
 * future recurring DST behavior are derived from java.util.TimeZone so ORC rebasing matches
 * SerializationUtils.convertBetweenTimezones semantics without relying on non-public ZoneInfo APIs.
 *
 * <p><b>Runtime dependency:</b> because the metadata is generated on the fly from
 * {@link java.util.TimeZone}/{@link java.time.zone.ZoneRules}, the exact transition table is
 * determined by the JVM's bundled IANA {@code tzdata}. Different JDK distributions or
 * {@code tzdata} versions may produce slightly different historical transitions for the same
 * zone id. This is strictly more correct than the previous frozen OpenJDK-8 snapshot, but users
 * debugging cross-environment differences should first check the JVM's {@code tzdata} version.
 */
class OrcTimezoneInfo {
  public OrcTimezoneInfo(int initialOffset, int rawOffset, long[] transitions, int[] offsets,
      OrcDstRuleExtractor.DstRule dstRule) {
    this.initialOffset = initialOffset;
    this.rawOffset = rawOffset;
    this.transitions = transitions;
    this.offsets = offsets;
    this.dstRule = dstRule;
  }

  // Historical offset before the first transition, in milliseconds.
  final int initialOffset;

  // in milliseconds
  final int rawOffset;

  // in milliseconds
  final long[] transitions;

  // in milliseconds
  final int[] offsets;

  // Recurring rule used after the historical transition table, or null for no DST.
  final OrcDstRuleExtractor.DstRule dstRule;

  // Lower bound of the range ORC supports (year 0001-01-01 UTC). Computed via
  // java.time.LocalDate, which uses the proleptic Gregorian calendar, whereas
  // java.util.TimeZone.getOffset(long) internally uses a hybrid Julian/Gregorian
  // calendar with the 1582 cutover for date-field interpretations. In practice
  // this difference does not affect offset lookup (which is purely instant-based
  // for ZoneInfo), so the two calendars agree on the offset at this instant.
  private static final long MIN_SUPPORTED_ORC_UTC_MILLIS = utcMillisForDate(1, 1, 1);
  // Base probe width used by collectTimeZoneTransitionsByScanning. The scanner
  // detects a transition by sampling tz.getOffset(probe) and comparing it to
  // the running offset; a pair of transitions A->B->A whose two endpoints fall
  // inside one probe step will net to zero and slip through. 6 hours is
  // smaller than the minimum spacing between any two real transitions in the
  // current IANA tzdata (the closest pairs are DST start/end, ~hours apart on
  // separate days), so paired transitions cannot hide in a single window.
  private static final long HISTORICAL_TRANSITION_SCAN_STEP_MILLIS = 6L * 3600_000L;

  // year, month, and day are all 1-indexed, matching LocalDate.of conventions
  // (e.g. month=1 is January). This avoids the easy-to-misread mix of 0-based
  // month and 1-based day at the call site.
  //
  // Package-private so OrcDstRuleExtractor can share the same anchor.
  static long utcMillisForDate(int year, int month, int day) {
    return LocalDate.of(year, month, day).toEpochDay() * 24L * 3600_000L;
  }

  @Override
  public String toString() {
    return "OrcTimezoneInfo{" +
        "initialOffset=" + initialOffset +
        ", rawOffset=" + rawOffset +
        ", transitions=" + Arrays.toString(transitions) +
        ", offsets=" + Arrays.toString(offsets) +
        ", dstRule=" + dstRule +
        '}';
  }

  private static final ConcurrentMap<String, OrcTimezoneInfo> RUNTIME_TIMEZONE_INFOS =
      new ConcurrentHashMap<>();

  /**
   * Get timezone info for the specified timezone ID.
   * Historical transitions are generated at runtime from public JVM APIs and cached per ID.
   *
   * @param timezoneId timezone ID
   * @return timezone info
   * @throws IllegalArgumentException if {@code timezoneId} is not a valid zone ID accepted
   *     by {@link GpuTimeZoneDB#getZoneId(String)}. There is no silent fallback to GMT.
   */
  public static OrcTimezoneInfo get(String timezoneId) {
    return RUNTIME_TIMEZONE_INFOS.computeIfAbsent(
        timezoneId,
        OrcTimezoneInfo::buildRuntimeOrcTimezoneInfo);
  }

  /**
   * Build ORC timezone metadata from public java.time/java.util APIs. Invalid IDs use the same
   * validation as {@link GpuTimeZoneDB#getZoneId(String)} and fail with
   * {@link IllegalArgumentException} (no silent fallback to GMT).
   *
   * <p><b>Cost:</b> this is non-trivial — it scans every historical {@link ZoneOffsetTransition}
   * from year 1 onward. Results are cached in {@link #RUNTIME_TIMEZONE_INFOS} (see
   * {@link #get(String)}), so callers should always go through {@code get(...)} rather than
   * invoking this directly.
   */
  private static OrcTimezoneInfo buildRuntimeOrcTimezoneInfo(String timezoneId) {
    final ZoneId zoneId;
    try {
      zoneId = GpuTimeZoneDB.getZoneId(timezoneId);
    } catch (DateTimeException e) {
      throw new IllegalArgumentException("Timezone ID not found: " + timezoneId, e);
    }

    ZoneRules rules = zoneId.getRules();
    if (rules.isFixedOffset()) {
      // IDs like "+05:30" are valid ZoneIds but TimeZone.getTimeZone() silently
      // maps them to GMT (offset 0). Derive the offset from ZoneRules instead so
      // the GPU path doesn't treat them as UTC.
      int fixedOffsetMs = rules.getOffset(Instant.EPOCH).getTotalSeconds() * 1000;
      return new OrcTimezoneInfo(fixedOffsetMs, fixedOffsetMs, null, null, null);
    }
    // Use the canonical ID from the resolved ZoneId (e.g. "Asia/Kolkata" for
    // input "IST") so that TimeZone and ZoneRules always refer to the same
    // zone, regardless of how the JVM's legacy TimeZone database maps
    // 3-letter aliases. ZoneId.SHORT_IDS in getZoneId resolves "IST" to
    // "Asia/Kolkata"; TimeZone.getTimeZone("IST") may map to a different
    // zone on some JVM distributions, which would silently produce mixed
    // offset data with no exception.
    TimeZone tz = TimeZone.getTimeZone(zoneId.getId());
    int initialOffset = getInitialOffset(tz);
    OrcDstRuleExtractor.DstRule dstRule =
        OrcDstRuleExtractor.extractDstRule(timezoneId, tz, rules);
    List<ZoneOffsetTransition> transitionList = rules.getTransitions();
    HistoricalTransitions historicalTransitions = buildHistoricalTransitions(tz, transitionList);
    if (historicalTransitions.transitions == null) {
      return new OrcTimezoneInfo(initialOffset, tz.getRawOffset(), null, null, dstRule);
    }
    return new OrcTimezoneInfo(initialOffset,
        tz.getRawOffset(), historicalTransitions.transitions, historicalTransitions.offsets,
        dstRule);
  }

  /**
   * Returns the sorted list of timezone IDs that {@link #get(String)} can build —
   * the intersection of {@link TimeZone#getAvailableIDs()} and
   * {@link GpuTimeZoneDB#isSupportedTimeZone(String)}. POSIX-style entries (e.g.
   * {@code "EST5EDT"}, {@code "SystemV/AST4"}) that some JDK builds expose but
   * {@code ZoneId.of(id, ZoneId.SHORT_IDS)} rejects are filtered out.
   *
   * <p>The result is computed on every call; callers that need it repeatedly
   * should cache it themselves.
   *
   * @return sorted list of ORC-supported timezone IDs
   */
  public static List<String> getAllTimezoneIds() {
    String[] ids = TimeZone.getAvailableIDs();
    Arrays.sort(ids);
    List<String> result = new ArrayList<>(ids.length);
    for (String id : ids) {
      if (GpuTimeZoneDB.isSupportedTimeZone(id)) {
        result.add(id);
      }
    }
    return result;
  }

  private static int getInitialOffset(TimeZone tz) {
    // ORC only supports timestamps from year 0001 onward. For dates before the
    // first historical transition in that range, java.util.TimeZone can differ
    // from ZoneRules' earliest wall offset (for example, it may use the zone's
    // standard raw offset instead of an older LMT offset). Sample the beginning
    // of the supported range so the GPU matches TimeZone.getOffset().
    return tz.getOffset(MIN_SUPPORTED_ORC_UTC_MILLIS);
  }

  private static HistoricalTransitions buildHistoricalTransitions(
      TimeZone tz,
      List<ZoneOffsetTransition> transitionList) {
    if (transitionList.isEmpty()) {
      return HistoricalTransitions.EMPTY;
    }

    List<Long> transitions = new ArrayList<>();
    List<Integer> offsets = new ArrayList<>();
    long scanCursor = MIN_SUPPORTED_ORC_UTC_MILLIS;
    int currentOffset = getInitialOffset(tz);
    boolean hasPreviousCandidate = false;

    for (ZoneOffsetTransition transition : transitionList) {
      long transitionMs = transition.getInstant().toEpochMilli();
      if (transitionMs < MIN_SUPPORTED_ORC_UTC_MILLIS) {
        continue;
      }

      long beforeTransitionMs = transitionMs - 1;
      int offsetBeforeTransition = tz.getOffset(beforeTransitionMs);
      if (beforeTransitionMs >= scanCursor) {
        if (hasPreviousCandidate) {
          // Reconcile every interval between ZoneRules candidates. Endpoints with the same
          // offset do not prove that the interval is transition-free: TimeZone may contain an
          // A -> B -> A round trip that ZoneRules does not expose.
          currentOffset = collectTimeZoneTransitionsByScanning(
              tz, scanCursor, beforeTransitionMs, currentOffset, transitions, offsets);
        } else if (offsetBeforeTransition != currentOffset) {
          // Keep the year-0001-to-first-candidate path logarithmic. ZoneRules has no candidate
          // in this interval, which is typically about 1,900 years long.
          currentOffset = collectInitialTimeZoneTransitions(
              tz, scanCursor, beforeTransitionMs, currentOffset, transitions, offsets);
        }
      }

      int offsetAtTransition = tz.getOffset(transitionMs);
      if (offsetAtTransition != offsetBeforeTransition) {
        transitions.add(transitionMs);
        offsets.add(offsetAtTransition);
        currentOffset = offsetAtTransition;
      }

      // Some JDK tzdata versions expose a ZoneRules candidate that TimeZone observes for only
      // one millisecond. Sampling T+1 catches that immediate return and anchors the following
      // bounded scan with the correct running offset.
      long afterTransitionMs = transitionMs + 1;
      int offsetAfterTransition = tz.getOffset(afterTransitionMs);
      if (offsetAfterTransition != currentOffset) {
        transitions.add(afterTransitionMs);
        offsets.add(offsetAfterTransition);
        currentOffset = offsetAfterTransition;
      }
      scanCursor = afterTransitionMs;
      hasPreviousCandidate = true;
    }

    if (transitions.isEmpty()) {
      return HistoricalTransitions.EMPTY;
    }
    return new HistoricalTransitions(toLongArray(transitions), toIntArray(offsets));
  }

  static int collectTimeZoneTransitionsByScanning(
      TimeZone tz,
      long scanStartMs,
      long scanEndMs,
      int startOffset,
      List<Long> transitions,
      List<Integer> offsets) {
    long cursor = scanStartMs;
    int currentOffset = startOffset;
    while (cursor < scanEndMs) {
      long lo = cursor;
      long hi = Math.min(lo + HISTORICAL_TRANSITION_SCAN_STEP_MILLIS, scanEndMs);
      int hiOffset = tz.getOffset(hi);
      if (hiOffset == currentOffset) {
        cursor = hi;
        continue;
      }

      long exactTransition = binarySearchTransition(tz, lo, hi);
      int offsetAfterTransition = tz.getOffset(exactTransition);
      transitions.add(exactTransition);
      offsets.add(offsetAfterTransition);
      currentOffset = offsetAfterTransition;
      cursor = exactTransition;
    }
    return currentOffset;
  }

  private static int collectInitialTimeZoneTransitions(
      TimeZone tz,
      long scanStartMs,
      long scanEndMs,
      int startOffset,
      List<Long> transitions,
      List<Integer> offsets) {
    long cursor = scanStartMs;
    int currentOffset = startOffset;
    while (cursor < scanEndMs) {
      long lo = cursor;
      long step = HISTORICAL_TRANSITION_SCAN_STEP_MILLIS;
      long hi = Math.min(lo + step, scanEndMs);
      int hiOffset = tz.getOffset(hi);
      while (hiOffset == currentOffset && hi < scanEndMs) {
        lo = hi;
        step = Math.min(step * 2L, scanEndMs - hi);
        hi = lo + step;
        hiOffset = tz.getOffset(hi);
      }
      if (hiOffset == currentOffset) {
        cursor = hi;
        continue;
      }

      long exactTransition = binarySearchTransition(tz, lo, hi);
      int offsetAfterTransition = tz.getOffset(exactTransition);
      transitions.add(exactTransition);
      offsets.add(offsetAfterTransition);
      currentOffset = offsetAfterTransition;
      cursor = exactTransition;
    }
    return currentOffset;
  }

  // Package-private so OrcDstRuleExtractor can reuse the same bracketed
  // binary search.
  static long binarySearchTransition(TimeZone tz, long lo, long hi) {
    int loOffset = tz.getOffset(lo);
    while (hi - lo > 1) {
      long mid = lo + (hi - lo) / 2;
      if (tz.getOffset(mid) == loOffset) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    return hi;
  }

  private static long[] toLongArray(List<Long> values) {
    long[] result = new long[values.size()];
    for (int i = 0; i < values.size(); i++) {
      result[i] = values.get(i);
    }
    return result;
  }

  private static int[] toIntArray(List<Integer> values) {
    int[] result = new int[values.size()];
    for (int i = 0; i < values.size(); i++) {
      result[i] = values.get(i);
    }
    return result;
  }

  private static final class HistoricalTransitions {
    static final HistoricalTransitions EMPTY = new HistoricalTransitions(null, null);

    final long[] transitions;
    final int[] offsets;

    private HistoricalTransitions(long[] transitions, int[] offsets) {
      this.transitions = transitions;
      this.offsets = offsets;
    }
  }
}
