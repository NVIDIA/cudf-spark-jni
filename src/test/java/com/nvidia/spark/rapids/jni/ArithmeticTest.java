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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Scalar;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class ArithmeticTest {

  @Test
  void multiplyAnsiOffWithOverflow() {
    // Integer.MAX_VALUE * 2 = -2 in non-ANSI/non-try mode
    try (
        ColumnVector left = ColumnVector.fromInts(0, 1, Integer.MAX_VALUE);
        ColumnVector right = ColumnVector.fromInts(0, 1, 2);
        ColumnVector expected = ColumnVector.fromInts(0, 1, -2);
        ColumnVector actual = Arithmetic.multiply(left, right, /* isAnsiMode */ false,
            /* isTryMode */ false)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void multiplyAnsiOnWithOverflow() {
    // Integer.MAX_VALUE * 2 throws exception in ANSI mode
    try (
        ColumnVector left = ColumnVector.fromInts(0, 1, Integer.MAX_VALUE);
        ColumnVector right = ColumnVector.fromInts(0, 1, 2)) {
      Arithmetic.multiply(left, right, /* isAnsiMode */ true, /* isTryMode */ false);
      Assertions.fail("Expected ExceptionWithRowIndex due to overflow");
    } catch (ExceptionWithRowIndex e) {
      Assertions.assertEquals(2, e.getRowIndex());
    }
  }

  @Test
  void multiplyTryOnWithOverflow() {
    // Integer.MAX_VALUE * 2 = null in try mode
    try (
        ColumnVector left = ColumnVector.fromInts(0, 1, Integer.MAX_VALUE);
        ColumnVector right = ColumnVector.fromInts(0, 1, 2);
        ColumnVector expected = ColumnVector.fromBoxedInts(0, 1, null);
        ColumnVector actual = Arithmetic.multiply(left, right, /* isAnsiMode */ false,
            /* isTryMode */ true)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void multiplyLongScalar() {
    // Integer.MAX_VALUE * 2 = -2 in non-ANSI/non-try mode
    try (
        ColumnVector left = ColumnVector.fromLongs(0, 1, Long.MAX_VALUE);
        Scalar right = Scalar.fromLong(2);
        ColumnVector expected = ColumnVector.fromBoxedLongs(0L, 2L, null);
        ColumnVector actual = Arithmetic.multiply(left, right, /* isAnsiMode */ false,
            /* isTryMode */ true)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void multiplyScalarInt() {
    // Integer.MAX_VALUE * 2 = -2 in non-ANSI/non-try mode
    try (
        Scalar left = Scalar.fromInt(2);
        ColumnVector right = ColumnVector.fromInts(0, 1, Integer.MAX_VALUE);
        ColumnVector expected = ColumnVector.fromInts(0, 2, -2);
        ColumnVector actual = Arithmetic.multiply(left, right, /* isAnsiMode */ false,
            /* isTryMode */ false)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void multiplyScalarIntAnsi() {
    // Integer.MAX_VALUE * 2 = -2 in non-ANSI/non-try mode
    try (
        Scalar left = Scalar.fromInt(2);
        ColumnVector right = ColumnVector.fromInts(0, 1, Integer.MAX_VALUE)) {
      Arithmetic.multiply(left, right, /* isAnsiMode */ true,
          /* isTryMode */ false);
      Assertions.fail("Expected ExceptionWithRowIndex due to overflow");
    } catch (ExceptionWithRowIndex e) {
      Assertions.assertEquals(2, e.getRowIndex());
    }
  }

  @Test
  void roundFloatsHalfUp() {
    try (ColumnVector v = ColumnVector.fromBoxedFloats(1.234f, 25.66f, null, 154.9f, 2346f);
         ColumnVector result1 = Arithmetic.round(v, 0, RoundMode.HALF_UP);
         ColumnVector result2 = Arithmetic.round(v, 1, RoundMode.HALF_UP);
         ColumnVector result3 = Arithmetic.round(v, -1, RoundMode.HALF_UP);
         ColumnVector expected1 = ColumnVector.fromBoxedFloats(1f, 26f, null, 155f, 2346f);
         ColumnVector expected2 = ColumnVector.fromBoxedFloats(1.2f, 25.7f, null, 154.9f, 2346f);
         ColumnVector expected3 = ColumnVector.fromBoxedFloats(0f, 30f, null, 150f, 2350f)) {
      assertColumnsAreEqual(expected1, result1);
      assertColumnsAreEqual(expected2, result2);
      assertColumnsAreEqual(expected3, result3);
    }
  }

  @Test
  void roundFloatsHalfEven() {
    try (ColumnVector v = ColumnVector.fromBoxedFloats(1.5f, 2.5f, 1.35f, null, 1.25f, 15f, 25f);
         ColumnVector result1 = Arithmetic.round(v, RoundMode.HALF_EVEN);
         ColumnVector result2 = Arithmetic.round(v, 1, RoundMode.HALF_EVEN);
         ColumnVector result3 = Arithmetic.round(v, -1, RoundMode.HALF_EVEN);
         ColumnVector expected1 = ColumnVector.fromBoxedFloats(2f, 2f, 1f, null, 1f, 15f, 25f);
         ColumnVector expected2 = ColumnVector.fromBoxedFloats(1.5f, 2.5f, 1.4f, null, 1.2f, 15f, 25f);
         ColumnVector expected3 = ColumnVector.fromBoxedFloats(0f, 0f, 0f, null, 0f, 20f, 20f)) {
      assertColumnsAreEqual(expected1, result1);
      assertColumnsAreEqual(expected2, result2);
      assertColumnsAreEqual(expected3, result3);
    }
  }

  @Test
  void roundIntsHalfUp() {
    try (ColumnVector v = ColumnVector.fromBoxedInts(12, 135, 160, -1454, null, -1500, -140, -150);
         ColumnVector result1 = Arithmetic.round(v, 2, RoundMode.HALF_UP);
         ColumnVector result2 = Arithmetic.round(v, -2, RoundMode.HALF_UP);
         ColumnVector expected1 = ColumnVector.fromBoxedInts(12, 135, 160, -1454, null, -1500, -140, -150);
         ColumnVector expected2 = ColumnVector.fromBoxedInts(0, 100, 200, -1500, null, -1500, -100, -200)) {
      assertColumnsAreEqual(expected1, result1);
      assertColumnsAreEqual(expected2, result2);
    }
  }

  @Test
  void roundIntsHalfEven() {
    try (ColumnVector v = ColumnVector.fromBoxedInts(12, 24, 135, 160, null, 1450, 1550, -1650);
         ColumnVector result1 = Arithmetic.round(v, 2, RoundMode.HALF_EVEN);
         ColumnVector result2 = Arithmetic.round(v, -2, RoundMode.HALF_EVEN);
         ColumnVector expected1 = ColumnVector.fromBoxedInts(12, 24, 135, 160, null, 1450, 1550, -1650);
         ColumnVector expected2 = ColumnVector.fromBoxedInts(0, 0, 100, 200, null, 1400, 1600, -1600)) {
      assertColumnsAreEqual(expected1, result1);
      assertColumnsAreEqual(expected2, result2);
    }
  }

  @Test
  void roundDecimal() {
    final int dec32Scale1 = -2;
    final int resultScale1 = -3;

    final int[] DECIMAL32_1 = new int[]{14, 15, 16, 24, 25, 26} ;
    final int[] expectedHalfUp = new int[]{1, 2, 2, 2, 3, 3};
    final int[] expectedHalfEven = new int[]{1, 2, 2, 2, 2, 3};
    try (ColumnVector v = ColumnVector.decimalFromInts(-dec32Scale1, DECIMAL32_1);
         ColumnVector roundHalfUp = Arithmetic.round(v, -3, RoundMode.HALF_UP);
         ColumnVector roundHalfEven = Arithmetic.round(v, -3, RoundMode.HALF_EVEN);
         ColumnVector answerHalfUp = ColumnVector.decimalFromInts(-resultScale1, expectedHalfUp);
         ColumnVector answerHalfEven = ColumnVector.decimalFromInts(-resultScale1, expectedHalfEven)) {
      assertColumnsAreEqual(answerHalfUp, roundHalfUp);
      assertColumnsAreEqual(answerHalfEven, roundHalfEven);
    }
  }

  @Test
  void roundDefault() {
    try (ColumnVector v = ColumnVector.fromBoxedFloats(1.234f, 25.66f, null, 154.9f, 2346f);
         ColumnVector result = Arithmetic.round(v);
         ColumnVector expected = ColumnVector.fromBoxedFloats(1f, 26f, null, 155f, 2346f)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void roundWithDecimalPlaces() {
    try (ColumnVector v = ColumnVector.fromBoxedFloats(1.234f, 25.66f, null, 154.9f, 2346f);
         ColumnVector result = Arithmetic.round(v, 2);
         ColumnVector expected = ColumnVector.fromBoxedFloats(1.23f, 25.66f, null, 154.9f, 2346f)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void roundDoublesHalfUp() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles(1.234, 25.66, null, 154.9, 2346.0);
         ColumnVector result1 = Arithmetic.round(v, 0, RoundMode.HALF_UP);
         ColumnVector result2 = Arithmetic.round(v, 1, RoundMode.HALF_UP);
         ColumnVector result3 = Arithmetic.round(v, -1, RoundMode.HALF_UP);
         ColumnVector expected1 = ColumnVector.fromBoxedDoubles(1.0, 26.0, null, 155.0, 2346.0);
         ColumnVector expected2 = ColumnVector.fromBoxedDoubles(1.2, 25.7, null, 154.9, 2346.0);
         ColumnVector expected3 = ColumnVector.fromBoxedDoubles(0.0, 30.0, null, 150.0, 2350.0)) {
      assertColumnsAreEqual(expected1, result1);
      assertColumnsAreEqual(expected2, result2);
      assertColumnsAreEqual(expected3, result3);
    }
  }

  @Test
  void roundDoublesHalfEven() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles(1.5, 2.5, 1.35, null, 1.25, 15.0, 25.0);
         ColumnVector result1 = Arithmetic.round(v, RoundMode.HALF_EVEN);
         ColumnVector result2 = Arithmetic.round(v, 1, RoundMode.HALF_EVEN);
         ColumnVector result3 = Arithmetic.round(v, -1, RoundMode.HALF_EVEN);
         ColumnVector expected1 = ColumnVector.fromBoxedDoubles(2.0, 2.0, 1.0, null, 1.0, 15.0, 25.0);
         ColumnVector expected2 = ColumnVector.fromBoxedDoubles(1.5, 2.5, 1.4, null, 1.2, 15.0, 25.0);
         ColumnVector expected3 = ColumnVector.fromBoxedDoubles(0.0, 0.0, 0.0, null, 0.0, 20.0, 20.0)) {
      assertColumnsAreEqual(expected1, result1);
      assertColumnsAreEqual(expected2, result2);
      assertColumnsAreEqual(expected3, result3);
    }
  }

  // ==================== ANSI Mode Round Overflow Tests ====================

  @Test
  void roundByteAnsiNoOverflow() {
    // Values that don't overflow after rounding with scale -1
    // round(10, -1) = 10, round(50, -1) = 50, round(-50, -1) = -50
    try (ColumnVector v = ColumnVector.fromBoxedBytes((byte) 10, (byte) 50, null, (byte) -50);
         ColumnVector result = Arithmetic.round(v, -1, RoundMode.HALF_UP, true);
         ColumnVector expected = ColumnVector.fromBoxedBytes((byte) 10, (byte) 50, null, (byte) -50)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void roundByteAnsiWithOverflow() {
    // round(125, -1) = 130 which overflows Byte (max 127)
    // For byte with scale -1: threshold = (127/10)*10 + 5 = 125
    // So values >= 125 will overflow
    try (ColumnVector v = ColumnVector.fromBoxedBytes((byte) 10, (byte) 50, (byte) 125)) {
      ExceptionWithRowIndex e = Assertions.assertThrows(ExceptionWithRowIndex.class, () -> {
        Arithmetic.round(v, -1, RoundMode.HALF_UP, true);
      });
      Assertions.assertEquals(2, e.getRowIndex());
    }
  }
  
  @Test
  void roundByteAnsiNegativeOverflow() {
    // round(-125, -1) = -130 which overflows Byte (min -128)
    try (ColumnVector v = ColumnVector.fromBoxedBytes((byte) 10, (byte) -125)) {
      ExceptionWithRowIndex e = Assertions.assertThrows(ExceptionWithRowIndex.class, () -> {
        Arithmetic.round(v, -1, RoundMode.HALF_UP, true);
      });
      Assertions.assertEquals(1, e.getRowIndex());
    }
  }

  @Test
  void roundShortAnsiNoOverflow() {
    // Values that don't overflow after rounding
    try (ColumnVector v = ColumnVector.fromBoxedShorts((short) 10, (short) 200, (short) 32700)) {
      // round(32700, -2) = 32700 (OK, doesn't overflow)
      Arithmetic.round(v, -2, RoundMode.HALF_UP, true);
      // This should pass - no overflow
    }
  }

  @Test
  void roundShortAnsiWithOverflow() {
    // round(32760, -2) = 32800 which overflows Short (max 32767)
    try (ColumnVector v = ColumnVector.fromBoxedShorts((short) 10, (short) 100, (short) 32760)) {
      ExceptionWithRowIndex e = Assertions.assertThrows(ExceptionWithRowIndex.class, () -> {
        Arithmetic.round(v, -2, RoundMode.HALF_UP, true);
      });
      Assertions.assertEquals(2, e.getRowIndex());
    }
  }

  @Test
  void roundShortAnsiNegativeOverflow() {
    // round(-32760, -2) = -32800 which overflows Short (min -32768)
    try (ColumnVector v = ColumnVector.fromBoxedShorts((short) 10, (short) -32760)) {
      ExceptionWithRowIndex e = Assertions.assertThrows(ExceptionWithRowIndex.class, () -> {
        Arithmetic.round(v, -2, RoundMode.HALF_UP, true);
      });
      Assertions.assertEquals(1, e.getRowIndex());
    }
  }

  @Test
  void roundIntAnsiWithOverflow() {
    // round(2147483640, -2) = 2147483600 (OK)
    // round(2147483650, -2) = 2147483700 which would overflow Int, but can't represent in int
    // Use a value that rounds up: round(2147483645, -1) = 2147483650 (overflow)
    try (ColumnVector v = ColumnVector.fromBoxedInts(10, 100, 2147483645)) {
      ExceptionWithRowIndex e = Assertions.assertThrows(ExceptionWithRowIndex.class, () -> {
        Arithmetic.round(v, -1, RoundMode.HALF_UP, true);
      });
      Assertions.assertEquals(2, e.getRowIndex());
    }
  }

  @Test
  void roundIntAnsiNoOverflow() {
    // Values that don't overflow after rounding
    try (ColumnVector v = ColumnVector.fromBoxedInts(12, 135, 160, null, -1454);
         ColumnVector result = Arithmetic.round(v, -2, RoundMode.HALF_UP, true);
         ColumnVector expected = ColumnVector.fromBoxedInts(0, 100, 200, null, -1500)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void roundLongAnsiWithOverflow() {
    // For Long, overflow happens at scale -19 or beyond
    // round(9e18, -19) would mathematically be 1e19 which overflows Long
    // Values with leading digit > 4 when divided by 1e18 will overflow
    try (ColumnVector v = ColumnVector.fromBoxedLongs(10L, 100L, 5000000000000000001L)) {
      // 5e18 / 1e18 = 5, which is > 4, so it would round up to 1e19 (overflow)
      ExceptionWithRowIndex e = Assertions.assertThrows(ExceptionWithRowIndex.class, () -> {
        Arithmetic.round(v, -19, RoundMode.HALF_UP, true);
      });
      Assertions.assertEquals(2, e.getRowIndex());
    }
  }

  @Test
  void roundLongAnsiNoOverflow() {
    // Values with leading digit <= 4 should not overflow
    try (ColumnVector v = ColumnVector.fromBoxedLongs(10L, 100L, 4000000L);
         ColumnVector result = Arithmetic.round(v, -18, RoundMode.HALF_UP, true);
         ColumnVector expected = ColumnVector.fromBoxedLongs(0L, 0L, 0L)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void roundNonAnsiModeNoException() {
    // Non-ANSI mode should NOT throw even with overflow - it allows wrapping
    try (ColumnVector v = ColumnVector.fromBoxedShorts((short) 10, (short) 100, (short) 32760);
         ColumnVector result = Arithmetic.round(v, -2, RoundMode.HALF_UP, false)) {
      // Should not throw, overflow wraps naturally
      Assertions.assertNotNull(result);
      // Note: We don't verify the wrapped value as it's implementation-dependent
    }
  }

  @Test
  void roundPositiveDecimalPlacesNoOverflow() {
    // Positive decimal places on integral types returns input as-is (no fractional part)
    try (ColumnVector v = ColumnVector.fromBoxedInts(12, 135, null, 160);
         ColumnVector result = Arithmetic.round(v, 2, RoundMode.HALF_UP, true);
         ColumnVector expected = ColumnVector.fromBoxedInts(12, 135, null, 160)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void roundWithNulls() {
    // Null values should remain null and not cause overflow check issues
    try (ColumnVector v = ColumnVector.fromBoxedShorts(null, (short) 100, null, (short) 32760)) {
      ExceptionWithRowIndex e = Assertions.assertThrows(ExceptionWithRowIndex.class, () -> {
        Arithmetic.round(v, -2, RoundMode.HALF_UP, true);
      });
      Assertions.assertEquals(3, e.getRowIndex());
    }
  }

  @Test
  void roundHalfEvenAnsi() {
    // Test HALF_EVEN rounding mode with ANSI overflow check
    try (ColumnVector v = ColumnVector.fromBoxedShorts((short) 32750, (short) 32755);
         // 32750 rounds to 32800 with HALF_UP, but 32800 with HALF_EVEN too (5 rounds to even)
         // 32755 -> 32760 with -1 scale
         ColumnVector result = Arithmetic.round(v, -1, RoundMode.HALF_EVEN, true);
         ColumnVector expected = ColumnVector.fromBoxedShorts((short) 32750, (short) 32760)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void roundByteHalfEvenNoOverflow() {
    // Key edge case: 125 with scale -1
    // HALF_UP: 125 rounds to 130 (overflow for byte, max 127)
    // HALF_EVEN: 125 rounds to 120 (12 is even, no overflow)
    // This test verifies that HALF_EVEN correctly allows 125 without overflow
    try (ColumnVector v = ColumnVector.fromBoxedBytes((byte) 125, (byte) 115, (byte) 105);
         ColumnVector result = Arithmetic.round(v, -1, RoundMode.HALF_EVEN, true);
         // 125 -> 120 (rounds to even), 115 -> 120 (rounds to even), 105 -> 100 (rounds to even)
         ColumnVector expected = ColumnVector.fromBoxedBytes((byte) 120, (byte) 120, (byte) 100)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void roundByteHalfUpVsHalfEvenOverflow() {
    // Demonstrate the difference in overflow thresholds between HALF_UP and HALF_EVEN
    
    // HALF_UP: 125 should overflow (rounds to 130, exceeds byte max 127)
    try (ColumnVector v = ColumnVector.fromBoxedBytes((byte) 125)) {
      ExceptionWithRowIndex e = Assertions.assertThrows(ExceptionWithRowIndex.class, () -> {
        Arithmetic.round(v, -1, RoundMode.HALF_UP, true);
      });
      Assertions.assertEquals(0, e.getRowIndex());
    }

    // HALF_EVEN: 125 should NOT overflow (rounds to 120, even quotient)
    try (ColumnVector v = ColumnVector.fromBoxedBytes((byte) 125);
         ColumnVector result = Arithmetic.round(v, -1, RoundMode.HALF_EVEN, true);
         ColumnVector expected = ColumnVector.fromBoxedBytes((byte) 120)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void roundByteHalfEvenOverflowAtCorrectThreshold() {
    // For HALF_EVEN with byte and scale -1:
    // Max quotient = 127 / 10 = 12 (even)
    // Since 12 is even, values at exactly half (125) round down to 120 (no overflow)
    // But 126 rounds to 130 (overflow) because it's above the half point and rounds to 13
    try (ColumnVector v = ColumnVector.fromBoxedBytes((byte) 126)) {
      ExceptionWithRowIndex e = Assertions.assertThrows(ExceptionWithRowIndex.class, () -> {
        Arithmetic.round(v, -1, RoundMode.HALF_EVEN, true);
      });
      Assertions.assertEquals(0, e.getRowIndex());
    }
  }

  @Test
  void roundByteHalfEvenNegativeOverflow() {
    // For negative values: min = -128, min_quotient = -128 / 10 = -12
    // abs(min_quotient) = 12 (even)
    // -125 at exactly half rounds to -120 (no overflow for HALF_EVEN)
    try (ColumnVector v = ColumnVector.fromBoxedBytes((byte) -125);
         ColumnVector result = Arithmetic.round(v, -1, RoundMode.HALF_EVEN, true);
         ColumnVector expected = ColumnVector.fromBoxedBytes((byte) -120)) {
      assertColumnsAreEqual(expected, result);
    }

    // But -126 should overflow (rounds to -130)
    try (ColumnVector v = ColumnVector.fromBoxedBytes((byte) -126)) {
      ExceptionWithRowIndex e = Assertions.assertThrows(ExceptionWithRowIndex.class, () -> {
        Arithmetic.round(v, -1, RoundMode.HALF_EVEN, true);
      });
      Assertions.assertEquals(0, e.getRowIndex());
    }
  }

}
