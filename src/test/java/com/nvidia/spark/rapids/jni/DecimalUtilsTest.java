/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.Arrays;

import static ai.rapids.cudf.AssertUtils.*;
import static org.junit.jupiter.api.Assertions.*;

public class DecimalUtilsTest {
  ColumnVector makeDec128Column(String ... values) {
    BigDecimal[] decVals = new BigDecimal[values.length];
    for (int i = 0; i < values.length; i++) {
      if (values[i] != null) {
        decVals[i] = new BigDecimal(values[i]);
      }
    }
    try (ColumnVector small = ColumnVector.fromDecimals(decVals)) {
      return small.castTo(DType.create(DType.DTypeEnum.DECIMAL128, small.getType().getScale()));
    }
  }

  @Test
  void simplePosMultiplyOneByZero() {
    try (ColumnVector lhs =
             makeDec128Column("1.0", "10.0", "1000000000000000000000000000000000000.0");
         ColumnVector rhs =
             makeDec128Column("1",   "1",    "1");
         ColumnVector expectedBasic =
             makeDec128Column("1.0", "10.0", "1000000000000000000000000000000000000.0");
         ColumnVector expectedOverflow =
             ColumnVector.fromBooleans(false, false, false);
         Table found = DecimalUtils.multiply128(lhs, rhs, -1)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void simplePosMultiplyOneByOne() {
    try (ColumnVector lhs =
             makeDec128Column("1.0", "3.7");
         ColumnVector rhs =
             makeDec128Column("1.0", "1.5");
         ColumnVector expectedBasic =
             makeDec128Column("1.0", "5.6");
         ColumnVector expectedOverflow =
             ColumnVector.fromBooleans(false, false);
         Table found = DecimalUtils.multiply128(lhs, rhs, -1)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void simplePosMultiplyZeroByNegOne() {
    try (ColumnVector lhs =
             makeDec128Column("1");
         ColumnVector rhs =
             makeDec128Column("1e1");
         ColumnVector expectedBasic =
             makeDec128Column("10.0");
         ColumnVector expectedOverflow =
             ColumnVector.fromBooleans(false);
         Table found = DecimalUtils.multiply128(lhs, rhs, -1)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void multiply128WithoutInterimCast() {
    try (ColumnVector lhs = makeDec128Column("-8533444864753048107770677711.1312637916");
         ColumnVector rhs = makeDec128Column("-12.0000000000");
         ColumnVector expectedBasic = makeDec128Column("102401338377036577293248132533.575165");
         ColumnVector expectedOverflow = ColumnVector.fromBooleans(false);
         Table found = DecimalUtils.multiply128(lhs, rhs, -6, false)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void largePosMultiplyTenByTen() {
    try (ColumnVector lhs =
             makeDec128Column("577694940161436285811555447.3103121126");
         ColumnVector rhs =
             makeDec128Column("100.0000000000");
         ColumnVector expectedBasic =
             makeDec128Column("57769494016143628581155544731.031211");
         ColumnVector expectedOverflow =
             ColumnVector.fromBooleans(false);
         Table found = DecimalUtils.multiply128(lhs, rhs, -6)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void overflowMult() {
    try (ColumnVector lhs =
             makeDec128Column("577694938495380589068894346.7625198736");
         ColumnVector rhs =
             makeDec128Column("-1258508260891400005608241690.1564700995");
         ColumnVector expectedOverflow =
             ColumnVector.fromBooleans(true);
         Table found = DecimalUtils.multiply128(lhs, rhs, -6)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
    }

  }

  @Test
  void simpleNegMultiplyOneByZero() {
    try (ColumnVector lhs =
             makeDec128Column("1.0",  "-1.0", "10.0");
         ColumnVector rhs =
             makeDec128Column("-1",   "1",    "-1");
         ColumnVector expectedBasic =
             makeDec128Column("-1.0", "-1.0", "-10.0");
         ColumnVector expectedOverflow =
             ColumnVector.fromBooleans(false, false, false);
         Table found = DecimalUtils.multiply128(lhs, rhs, -1)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void simpleNegMultiplyOneByOne() {
    try (ColumnVector lhs =
             makeDec128Column("1.0",  "-1.0", "3.7");
         ColumnVector rhs =
             makeDec128Column("-1.0", "-1.0", "-1.5");
         ColumnVector expectedBasic =
             makeDec128Column("-1.0",  "1.0", "-5.6");
         ColumnVector expectedOverflow =
             ColumnVector.fromBooleans(false, false, false);
         Table found = DecimalUtils.multiply128(lhs, rhs, -1)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void simpleNegMultiplyTenByTenSparkCompat() {
    // many of the numbers listed here are *NOT* what BigDecimal would 
    // normally spit out. Spark has a bug https://issues.apache.org/jira/browse/SPARK-40129
    // which causes some rounding to be off, so these come directly from
    // Spark. It should be simple to fix this issue by deleting code, or bypassing the
    // first divide step when/if Spark fixes it.
    try (ColumnVector lhs =
             makeDec128Column("3358377338823096511784947656.4650294583",
                 "7161021785186010157110137546.5940777916",
                 "9173594185998001607642838421.5479932913");
         ColumnVector rhs =
             makeDec128Column("-12.0000000000",
                 "-12.0000000000",
                 "-12.0000000000");
         ColumnVector expectedBasic =
             makeDec128Column("-40300528065877158141419371877.580354",
                 "-85932261422232121885321650559.128933",
                 "-110083130231976019291714061058.575920");
         ColumnVector expectedOverflow =
             ColumnVector.fromBooleans(false, false, false);
         Table found = DecimalUtils.multiply128(lhs, rhs, -6)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void simplePosDivOneByZero() {
    try (ColumnVector lhs =
             makeDec128Column("1.0", "10.0", "1.0", "1000000000000000000000000000000000000.0");
         ColumnVector rhs =
             makeDec128Column("1",   "2",    "0",   "5");
         ColumnVector expectedBasic =
             makeDec128Column("1.0", "5.0",  "0",   "200000000000000000000000000000000000.0");
         ColumnVector expectedOverflow =
             ColumnVector.fromBooleans(false, false, true, false);
         Table found = DecimalUtils.divide128(lhs, rhs, -1)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void intDivide() {
    try (ColumnVector lhs =
             makeDec128Column("3396191716868766147341919609.06", "-6893798181986328848375556144.67");
         ColumnVector rhs =
             makeDec128Column("7317548469.64", "98565515088.44");
         ColumnVector expectedBasic = ColumnVector.fromLongs(464116053478747633L, -69941278912819784L);
         ColumnVector expectedOverflow = ColumnVector.fromBooleans(false, false);
         Table found = DecimalUtils.integerDivide128(lhs, rhs)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void intDivideNotOverflow() {
    // Spark doesn't report this as an overflow even though this has obviously overflowed.
    // In case of integral divide, it still bases whether a column has overflown by checking the
    // 128-bit value and not the returned 64-bit value
    try (ColumnVector lhs =
             makeDec128Column("451635271134476686911387864.48", "5313675970270560086329837153.18");
         ColumnVector rhs =
             makeDec128Column("-961.110", "181.958");
         ColumnVector expectedBasic = ColumnVector.fromLongs(2284624887606872042L, -2928582767902049472L);
         ColumnVector expectedOverflow = ColumnVector.fromBooleans(false, false);
         Table found = DecimalUtils.integerDivide128(lhs, rhs)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void intDivideOverflow() {
    try (ColumnVector lhs =
             makeDec128Column("-999999999999999999999999999999999999.99", "999999999999999999999999999999999999.99");
         ColumnVector rhs = makeDec128Column("0", "0");
         ColumnVector expectedOverflow = ColumnVector.fromBooleans(true, true);
         Table found = DecimalUtils.integerDivide128(lhs, rhs)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
    }
  }

  @Test
  void remainder1() {
    try (ColumnVector lhs = 
            makeDec128Column("2775750723350045263458396405825339066", "2775750723350045263458396405825339066", "-2775750723350045263458396405825339066", "-2775750723350045263458396405825339066");
        ColumnVector rhs = 
            makeDec128Column("-4890990637589340307512622401149178814.1", "4890990637589340307512622401149178814.1", "-4890990637589340307512622401149178814.1", "4890990637589340307512622401149178814.1");
        ColumnVector expected =
            makeDec128Column("2775750723350045263458396405825339066.0", "2775750723350045263458396405825339066.0", "-2775750723350045263458396405825339066.0", "-2775750723350045263458396405825339066.0");
        ColumnVector expectedOverflow = ColumnVector.fromBooleans(false, false, false, false);
        Table found = DecimalUtils.remainder128(lhs, rhs, -1)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expected, found.getColumn(1));
    }
  }

  @Test
  void remainder2() {
    try (ColumnVector lhs = 
            makeDec128Column("-80968577325845461854951721352418610.13", "-80968577325845461854951721352418610.13", "-66686472768705331734321352506496901.71");
        ColumnVector rhs = 
            makeDec128Column("6749200345857154099505910298895800952.1", "-6749200345857154099505910298895800952.1", "-43880265997097383351377368851255372.5");
        ColumnVector expected =
            makeDec128Column("-80968577325845461854951721352418610.13", "-80968577325845461854951721352418610.13", "-22806206771607948382943983655241529.21");
        ColumnVector expectedOverflow = ColumnVector.fromBooleans(false, false, false);
        Table found = DecimalUtils.remainder128(lhs, rhs, -2)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expected, found.getColumn(1));
    }
  }

  @Test
  void remainder7() {
    try (ColumnVector lhs = 
            makeDec128Column("5776949384953805890688943467625198736");
        ColumnVector rhs = 
            makeDec128Column("-67337920196996830.354487679299");
        ColumnVector expected =
            makeDec128Column("16310460742282291.8108019");
        ColumnVector expectedOverflow = ColumnVector.fromBooleans(false);
        Table found = DecimalUtils.remainder128(lhs, rhs, -7)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expected, found.getColumn(1));
    }
  }

  @Test
  void remainder10() {
    try (ColumnVector lhs = 
            makeDec128Column("5776949384953805890688943467625198736");
        ColumnVector rhs = 
            makeDec128Column("-6733792019699683035.4487679299");
        ColumnVector expected =
            makeDec128Column("3585222007130884413.9709383255");
        ColumnVector expectedOverflow = ColumnVector.fromBooleans(false);
        Table found = DecimalUtils.remainder128(lhs, rhs, -10)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expected, found.getColumn(1));
    }
  }

  @Test
  void simplePosDivOneByOne() {
    try (ColumnVector lhs =
             makeDec128Column("1.0", "3.7", "99.9");
         ColumnVector rhs =
             makeDec128Column("1.0", "1.5", "4.5");
         ColumnVector expectedBasic =
             makeDec128Column("1.0", "2.5", "22.2");
         ColumnVector expectedOverflow =
             ColumnVector.fromBooleans(false, false, false);
         Table found = DecimalUtils.divide128(lhs, rhs, -1)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void simpleNegDivOneByOne() {
    try (ColumnVector lhs =
             makeDec128Column("1.0", "-3.7", "-99.9");
         ColumnVector rhs =
             makeDec128Column("-1.0", "1.5", "-4.5");
         ColumnVector expectedBasic =
             makeDec128Column("-1.0", "-2.5", "22.2");
         ColumnVector expectedOverflow =
             ColumnVector.fromBooleans(false, false, false);
         Table found = DecimalUtils.divide128(lhs, rhs, -1)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void divComplex() {
    try (ColumnVector lhs =
             makeDec128Column("100000000000000000000000000000000");
         ColumnVector rhs =
             makeDec128Column("3.0000000000000000000000000000000000000");
         ColumnVector expectedBasic =
             makeDec128Column("33333333333333333333333333333333.333333");
         ColumnVector expectedOverflow =
             ColumnVector.fromBooleans(false);
         Table found = DecimalUtils.divide128(lhs, rhs, -6)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void div17() {
    try (ColumnVector lhs =
             makeDec128Column("1454.48287885760884146",
                 "3655.54438423288356646");
         ColumnVector rhs =
             makeDec128Column("100.00000000000000000",
                 "100.00000000000000000");
         ColumnVector expectedBasic =
             makeDec128Column("14.54482878857608841",
                 "36.55544384232883566");
         ColumnVector expectedOverflow =
             ColumnVector.fromBooleans(false, false);
         Table found = DecimalUtils.divide128(lhs, rhs, -17)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void div17WithPosScale() {
    try (ColumnVector lhs =
             makeDec128Column("1454.48287885760884146");
         ColumnVector rhs =
             makeDec128Column("1e2");
         ColumnVector expectedBasic =
             makeDec128Column("14.54482878857608841");
         ColumnVector expectedOverflow =
             ColumnVector.fromBooleans(false);
         Table found = DecimalUtils.divide128(lhs, rhs, -17)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void div21WithPosScale() {
    try (ColumnVector lhs =
             makeDec128Column("5776949401614362.858115554473103121126");
         ColumnVector rhs =
             makeDec128Column("1e2");
         ColumnVector expectedBasic =
             makeDec128Column("57769494016143.628581");
         ColumnVector expectedOverflow =
             ColumnVector.fromBooleans(false);
         Table found = DecimalUtils.divide128(lhs, rhs, -6)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void div21() {
    try (ColumnVector lhs =
             makeDec128Column("60250054953505368.439892586764888491018",
                 "91910085134512953.335347579448489062875",
                 "51312633107598808.869351260608653423886");
         ColumnVector rhs =
             makeDec128Column("97982875273794447.385070145919990343867",
                 "94478503341597285.814104936062234698349",
                 "92266075543848323.800466593082956765923");
         ColumnVector expectedBasic =
             makeDec128Column("0.614904",
                 "0.972815",
                 "0.556138");
         ColumnVector expectedOverflow =
             ColumnVector.fromBooleans(false, false, false);
         Table found = DecimalUtils.divide128(lhs, rhs, -6)) {
      assertColumnsAreEqual(expectedOverflow, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void addPrecision38ScaleNeg10WithOverflow() {
    try (
        ColumnVector lhs = makeDec128Column("9191008513307131620269245301.1615457290",
            "-9191008513307131620269245301.1615457290");
        ColumnVector rhs = makeDec128Column("9447850332473678680446404122.5624623187",
            "-9447850332473678680446404122.5624623187");
        ColumnVector expectedOverflow = ColumnVector.fromBooleans(true, true);
        Table result = DecimalUtils.add128(lhs, rhs, -10)) {
      assertColumnsAreEqual(expectedOverflow, result.getColumn(0));
    }
  }

  @Test
  void addPrecision38ScaleNeg10() {
    try (
        ColumnVector lhs = makeDec128Column("9191008513307131620269245301.1615457290",
            "-9191008513307131620269245301.1615457290",
            "577694938495380589068894346.7625198736",
            "-7949989536398283250841565918.6123449781",
            "-569260079419403643627836417.1451349695",
            "4268696962649098725873162852.3422176564",
            "948521076935839001259204571.1574829065",
            "-9299778357834801251892834048.0026057082",
            "8127384240098008972235509102.7063990819",
            "-1012433127481465711031073593.0625063701",
            "-3008128675386495592846447084.0906874636");
        ColumnVector rhs = makeDec128Column("9447850332473678680446404122.5624623187",
            "-9447850332473678680446404122.5624623187",
            "-1258508260891400005608241690.1564700995",
            "0E-10",
            "4506903505351346531188531230.8104179784",
            "8289592062844478064245294937.3714242072",
            "475827447078875704758652459.0564660621",
            "960510811873374359477931158.7077642783",
            "7213672086663445017824298126.4525607205",
            "2346189245818456940830953479.5847958897",
            "449885491907950809374133839.5150485453");
        ColumnVector expected = makeDec128Column("18638858845780810300715649423.724008048",
            "-18638858845780810300715649423.724008048",
            "-680813322396019416539347343.393950226",
            "-7949989536398283250841565918.612344978",
            "3937643425931942887560694813.665283009",
            "12558289025493576790118457789.713641864",
            "1424348524014714706017857030.213948969",
            "-8339267545961426892414902889.294841430",
            "15341056326761453990059807229.158959802",
            "1333756118336991229799879886.522289520",
            "-2558243183478544783472313244.575638918");
        ColumnVector expectedOverflow = ColumnVector.fromBooleans(false, false, false, false, false,
            false, false, false, false, false, false);
        Table result = DecimalUtils.add128(lhs, rhs, -9)) {
      assertColumnsAreEqual(expectedOverflow, result.getColumn(0));
      assertColumnsAreEqual(expected, result.getColumn(1));
    }
  }

  @Test
  void addPrecision38Scale5() {
    try (
        ColumnVector lhs = makeDec128Column(
            "4.2701861951571908374098848594277520E+39",
            "-9.51477182371612065851896242097995638E+40",
            "-2.0167866914929483784509827485383359E+39",
            "3.09186385410128070998385426348594484E+40",
            "7.1672663199631946247197119155144713E+39",
            "-9.32396355260007858810554960112006290E+40",
            "8.24190234828859904475261796305602287E+40",
            "6.10646349654220618869425418121505315E+40",
            "-5.4790787707639406411507823776332565E+39",
            null);
        ColumnVector rhs = makeDec128Column(
            "-7.4015414116488076297669800353634627E+39",
            "8.26223612055178995785348949126553327E+40",
            "3.27796298399180383738215644697505864E+40",
            "6.23318861108302118457923491160201752E+40",
            "1.2868445730284429449720988121912717E+39",
            "-9.89573762074541324330058371364880604E+40",
            "1.83583924726137822744760302018523424E+40",
            "5.39262612260712860406222466457256229E+40",
            "-1.0688816822936864401341690563696501E+39",
            "-1.0688816822936864401341690563696501E+39");
        ColumnVector expected = makeDec128Column(
            "-3.1313552164916167923570951759357107E+39",
            "-1.25253570316433070066547292971442311E+40",
            "3.07628431484250899953705817212122505E+40",
            "9.32505246518430189456308917508796236E+40",
            "8.4541108929916375696918107277057430E+39",
            "-1.921970117334549183140613331476886894E+41",
            "1.007774159554997727220022098324125711E+41",
            "1.149908961914933479275647884578761544E+41",
            "-6.5479604530576270812849514340029066E+39",
            null);
        ColumnVector expectedOverflow = ColumnVector.fromBoxedBooleans(false, false, false, false,
            false, false, false, false, false, null);
        Table result = DecimalUtils.add128(lhs, rhs, 5)) {
      assertColumnsAreEqual(expectedOverflow, result.getColumn(0));
      assertColumnsAreEqual(expected, result.getColumn(1));
    }
  }

  @Test
  void addDifferentScales() {
    try (
        ColumnVector lhs = makeDec128Column(
            "9191008513307131620269245301.1615457290",
            "-9191008513307131620269245301.1615457290",
            "577694938495380589068894346.7625198736",
            "-7949989536398283250841565918.6123449781",
            "-569260079419403643627836417.1451349695",
            "4268696962649098725873162852.3422176564",
            "948521076935839001259204571.1574829065",
            "-9299778357834801251892834048.0026057082",
            "8127384240098008972235509102.7063990819",
            "-1012433127481465711031073593.0625063701");
        ColumnVector rhs = makeDec128Column(
            "451635271134476686911387864.48",
            "-9037370400215680718822505020.06",
            "-200173438757934601210092407.67",
            "3022290197578200820919308997.64",
            "388221337108432989001879408.73",
            "-9119163961520067341639997328.82",
            "7732813484881363300406806463.83",
            "5941454871287785414686091453.79",
            "-357209139972312354271434821.33",
            "-857448828702886587693936536.21");

        ColumnVector expected = makeDec128Column(
            "9642643784441608307180633165.641545729",
            "-18228378913522812339091750321.221545729",
            "377521499737445987858801939.092519874",
            "-4927699338820082429922256920.972344978",
            "-181038742310970654625957008.415134970",
            "-4850466998870968615766834476.477782344",
            "8681334561817202301666011034.987482907",
            "-3358323486547015837206742594.212605708",
            "7770175100125696617964074281.376399082",
            "-1869881956184352298725010129.272506370");
        ColumnVector expectedOverflow = ColumnVector.fromBoxedBooleans(false, false, false, false,
            false, false, false, false, false, false);
        Table result = DecimalUtils.add128(lhs, rhs, -9)) {
      assertColumnsAreEqual(expectedOverflow, result.getColumn(0));
      assertColumnsAreEqual(expected, result.getColumn(1));
    }
  }

  @Test
  void mulTestOverflow() {
    try (
        ColumnVector lhs = makeDec128Column("50000000000000000000000000000000000000");
        ColumnVector rhs = makeDec128Column("2");
        ColumnVector expectedOverflow = ColumnVector.fromBooleans(true);
        Table result = DecimalUtils.multiply128(lhs, rhs, 0)) {
      assertColumnsAreEqual(expectedOverflow, result.getColumn(0));
    }
  }

  @Test
  void addTestOverflow() {
    try (
        ColumnVector lhs = makeDec128Column("99999999999999999999999999999999999999");
        ColumnVector rhs = makeDec128Column("1");
        ColumnVector expectedOverflow = ColumnVector.fromBooleans(true);
        Table result = DecimalUtils.add128(lhs, rhs, 0)) {
      assertColumnsAreEqual(expectedOverflow, result.getColumn(0));
    }
  }

  @Test
  void subTestOverflow() {
    try (
        ColumnVector lhs = makeDec128Column("-99999999999999999999999999999999999999");
        ColumnVector rhs = makeDec128Column("1");
        ColumnVector expectedOverflow = ColumnVector.fromBooleans(true);
        Table result = DecimalUtils.subtract128(lhs, rhs, 0)) {
      assertColumnsAreEqual(expectedOverflow, result.getColumn(0));
    }
  }

  @Test
  void subDifferentScales() {
    try (
        ColumnVector lhs = makeDec128Column(
            "9191008513307131620269245301.1615457290",
            "-9191008513307131620269245301.1615457290",
            "577694938495380589068894346.7625198736",
            "-7949989536398283250841565918.6123449781",
            "-569260079419403643627836417.1451349695",
            "4268696962649098725873162852.3422176564",
            "948521076935839001259204571.1574829065",
            "-9299778357834801251892834048.0026057082",
            "8127384240098008972235509102.7063990819",
            "-1012433127481465711031073593.0625063701");
        ColumnVector rhs = makeDec128Column(
            "451635271134476686911387864.48",
            "-9037370400215680718822505020.06",
            "-200173438757934601210092407.67",
            "3022290197578200820919308997.64",
            "388221337108432989001879408.73",
            "-9119163961520067341639997328.82",
            "7732813484881363300406806463.83",
            "5941454871287785414686091453.79",
            "-357209139972312354271434821.33",
            "-857448828702886587693936536.21");

        ColumnVector expected = makeDec128Column(
            "8739373242172654933357857436.681545729",
            "-153638113091450901446740281.101545729",
            "777868377253315190278986754.432519874",
            "-10972279733976484071760874916.252344978",
            "-957481416527836632629715825.875134970",
            "13387860924169166067513160181.162217656",
            "-6784292407945524299147601892.672517094",
            "-15241233229122586666578925501.792605708",
            "8484593380070321326506943924.036399082",
            "-154984298778579123337137056.852506370");
        ColumnVector expectedOverflow = ColumnVector.fromBoxedBooleans(false, false, false, false,
            false, false, false, false, false, false);
        Table result = DecimalUtils.subtract128(lhs, rhs, -9)) {
      assertColumnsAreEqual(expectedOverflow, result.getColumn(0));
      assertColumnsAreEqual(expected, result.getColumn(1));
    }
  }

  @Test
  void subDifferentLargeScales() {
    try (
        ColumnVector lhs = makeDec128Column("0.964808037102759");
        ColumnVector rhs = makeDec128Column("1");
        ColumnVector expected = makeDec128Column("-0.0351919628972410000000000");
        ColumnVector expectedOverflow = ColumnVector.fromBooleans(false);
        Table result = DecimalUtils.subtract128(lhs, rhs, -25)) {
      assertColumnsAreEqual(expectedOverflow, result.getColumn(0));
      assertColumnsAreEqual(expected, result.getColumn(1));
    }
  }

  @Test
  void floatingPointToDecimalTest() {
    try (
        ColumnVector input1 = ColumnVector.fromDoubles(3527.61953125);
        ColumnVector input2 = ColumnVector.fromDoubles(9.95);
        ColumnVector input3 = ColumnVector.fromDoubles(10.3);
        ColumnVector input4 = ColumnVector.fromDoubles(-10000000.0, -100000.0, 1.0, 100.0, 1000.0);
        ColumnVector input5 = ColumnVector.fromDoubles(-10000000.0, 1.0, Double.NaN, -2.0, Double.NEGATIVE_INFINITY);

        ColumnVector expected1 = ColumnVector.decimalFromLongs(-7, 35276195313L);
        ColumnVector expected2 = ColumnVector.decimalFromInts(-1, 100);
        ColumnVector expected3 = ColumnVector.decimalFromBigInt(-1, new BigInteger("103"));
        ColumnVector expected4 = ColumnVector.decimalFromBoxedInts(-1, null, null, 10, 1000, null);
        ColumnVector expected5 = ColumnVector.decimalFromBoxedLongs(-1, null, 10L, null, -20L, null)
    ) {
      DecimalUtils.CastFloatToDecimalResult output1 = DecimalUtils.floatingPointToDecimal(input1, DType.create(DType.DTypeEnum.DECIMAL64, -7), 12);
      DecimalUtils.CastFloatToDecimalResult output2 = DecimalUtils.floatingPointToDecimal(input2, DType.create(DType.DTypeEnum.DECIMAL32, -1), 3);
      DecimalUtils.CastFloatToDecimalResult output3 = DecimalUtils.floatingPointToDecimal(input3, DType.create(DType.DTypeEnum.DECIMAL128, -1), 18);
      DecimalUtils.CastFloatToDecimalResult output4 = DecimalUtils.floatingPointToDecimal(input4, DType.create(DType.DTypeEnum.DECIMAL32, -1), 4);
      DecimalUtils.CastFloatToDecimalResult output5 = DecimalUtils.floatingPointToDecimal(input5, DType.create(DType.DTypeEnum.DECIMAL64, -1), 4);

      try {
        assert (output1.failureRowId < 0);
        assert (output2.failureRowId < 0);
        assert (output3.failureRowId < 0);
        assert (output4.failureRowId >= 0);
        assert (output5.failureRowId >= 0);

        assertColumnsAreEqual(expected1, output1.result);
        assertColumnsAreEqual(expected2, output2.result);
        assertColumnsAreEqual(expected3, output3.result);
        assertColumnsAreEqual(expected4, output4.result);
        assertColumnsAreEqual(expected5, output5.result);
      } finally {
        output1.result.close();
        output2.result.close();
        output3.result.close();
        output4.result.close();
        output5.result.close();
      }
    }
  }

  private void assertFloatingConversion(boolean floatInput, DType target, int precision,
      Double[] values, BigInteger[] coefficients, boolean hasFailure) {
    Float[] floats = new Float[values.length];
    for (int i = 0; i < values.length; i++) {
      floats[i] = values[i] == null ? null : values[i].floatValue();
    }
    try (ColumnVector input = floatInput ? ColumnVector.fromBoxedFloats(floats) :
             ColumnVector.fromBoxedDoubles(values);
         ColumnVector wideExpected = ColumnVector.decimalFromBigInt(target.getScale(), coefficients);
         ColumnVector expected = wideExpected.castTo(target)) {
      DecimalUtils.CastFloatToDecimalResult converted =
          DecimalUtils.floatingPointToDecimal(input, target, precision);
      try (ColumnVector result = converted.result) {
        assertColumnsAreEqual(expected, result);
        if (hasFailure) {
          assertTrue(converted.failureRowId >= 0 && converted.failureRowId < values.length);
          int failed = (int) converted.failureRowId;
          assertNull(coefficients[failed]);
          assertNotNull(values[failed]);
          assertTrue(Double.isFinite(values[failed]));
        } else {
          assertTrue(converted.failureRowId < 0);
        }
      }
    }
  }

  @Test
  void floatingPointStorageOverflow() {
    DType[] targets = {
        DType.create(DType.DTypeEnum.DECIMAL32, -2),
        DType.create(DType.DTypeEnum.DECIMAL32, -9),
        DType.create(DType.DTypeEnum.DECIMAL64, -18),
        DType.create(DType.DTypeEnum.DECIMAL128, -36)};
    int[] precisions = {9, 9, 18, 38};
    double[] overflowing = {43000000.0, 4.35, 18.5, 340.2823669209385};
    for (boolean floatInput : new boolean[]{false, true}) {
      for (int i = 0; i < targets.length; i++) {
        assertFloatingConversion(floatInput, targets[i], precisions[i],
            new Double[]{0.0, -0.0, overflowing[i], -overflowing[i],
                floatInput ? (double) Float.MAX_VALUE : Double.MAX_VALUE, null,
                Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.MIN_VALUE},
            new BigInteger[]{BigInteger.ZERO, BigInteger.ZERO, null, null, null,
                null, null, null, null, BigInteger.ZERO}, true);
      }
    }
  }

  @Test
  void floatingPointRoundedMagnitudeOverflow() {
    DType[] targets = {
        DType.create(DType.DTypeEnum.DECIMAL32, -2),
        DType.create(DType.DTypeEnum.DECIMAL64, -2),
        DType.create(DType.DTypeEnum.DECIMAL128, -2),
        DType.create(DType.DTypeEnum.DECIMAL128, -2),
        DType.create(DType.DTypeEnum.DECIMAL128, 0)};
    int[] precisions = {9, 18, 38, 38, 38};
    // The last two coefficients also exceed 2^127, so they must be checked unsigned.
    double[] overflowing = {1.5e7, 1.5e16, 1.5e36, 1.8e36, 1.99e38};
    for (boolean floatInput : new boolean[]{false, true}) {
      for (int i = 0; i < targets.length; i++) {
        double actualInput = floatInput ? (double) (float) overflowing[i] : overflowing[i];
        double scaledMagnitude = actualInput * Math.pow(10.0, -targets[i].getScale());
        double precisionBound = Math.pow(10.0, precisions[i]);
        // Qualify the actual source value: it must pass the loose envelope and reach the
        // exact magnitude check, including after rounding the input to float precision.
        assertTrue(scaledMagnitude > precisionBound && scaledMagnitude < 2 * precisionBound);
        BigInteger unit = BigInteger.TEN.pow(-targets[i].getScale());
        assertFloatingConversion(floatInput, targets[i], precisions[i],
            new Double[]{overflowing[i], -overflowing[i], 1.0, -1.0, 0.0, null},
            new BigInteger[]{null, null, unit, unit.negate(), BigInteger.ZERO, null}, true);
      }
    }
  }

  @Test
  void floatingPointNegativeScaleAndRoundingCarry() {
    for (boolean floatInput : new boolean[]{false, true}) {
      // 2^39 is exactly representable in either input type. Its extra rounding digit
      // does not fit DECIMAL32 even though the final rounded coefficient does.
      assertFloatingConversion(floatInput, DType.create(DType.DTypeEnum.DECIMAL32, 3), 9,
          new Double[]{549755813888.0, -549755813888.0, 0.0},
          new BigInteger[]{BigInteger.valueOf(549755814), BigInteger.valueOf(-549755814),
              BigInteger.ZERO}, false);
      assertFloatingConversion(floatInput, DType.create(DType.DTypeEnum.DECIMAL32, -2), 3,
          new Double[]{9.99, -9.99, 9.999, -9.999},
          new BigInteger[]{BigInteger.valueOf(999), BigInteger.valueOf(-999), null, null}, true);
    }
    BigInteger coefficient = new BigInteger("50000000000000000000000000000000000000");
    assertFloatingConversion(false, DType.create(DType.DTypeEnum.DECIMAL128, 2), 38,
        new Double[]{5e39, -5e39},
        new BigInteger[]{coefficient, coefficient.negate()}, false);
    assertFloatingConversion(false, DType.create(DType.DTypeEnum.DECIMAL32, 3), 9,
        new Double[]{9e11, -9e11},
        new BigInteger[]{BigInteger.valueOf(900000000), BigInteger.valueOf(-900000000)}, false);
  }

  @Test
  void floatingPointSpecialValuesDoNotReportOverflow() {
    for (boolean floatInput : new boolean[]{false, true}) {
      for (int scale : new int[]{-38, 0, 38}) {
        assertFloatingConversion(floatInput, DType.create(DType.DTypeEnum.DECIMAL128, scale), 38,
            new Double[]{null, Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
                0.0, -0.0},
            new BigInteger[]{null, null, null, null, BigInteger.ZERO, BigInteger.ZERO}, false);
      }
    }
  }

  @Test
  void floatingPointRejectsUnsupportedScaleAndPrecision() {
    try (ColumnVector input = ColumnVector.fromDoubles(0.0)) {
      // Construct the type first so the assertion exercises native validation, rather than
      // DType's separate restriction on negative scales.
      for (int scale : new int[]{39, 50, 60, Integer.MIN_VALUE, Integer.MAX_VALUE}) {
        DType target = DType.create(DType.DTypeEnum.DECIMAL128, scale);
        CudfException error = assertThrows(CudfException.class, () -> {
          DecimalUtils.CastFloatToDecimalResult converted = DecimalUtils.floatingPointToDecimal(
              input, target, 38);
          try (ColumnVector ignored = converted.result) {
            // Close an unexpected result if the native precondition is not enforced.
          }
        });
        assertTrue(error.getMessage().contains("supports scales from -38 to 38"));
      }
      for (int precision : new int[]{0, 10}) {
        CudfException error = assertThrows(CudfException.class, () -> {
          DecimalUtils.CastFloatToDecimalResult converted = DecimalUtils.floatingPointToDecimal(
              input, DType.create(DType.DTypeEnum.DECIMAL32, 0), precision);
          try (ColumnVector ignored = converted.result) {
            // Close an unexpected result if the native precondition is not enforced.
          }
        });
        assertTrue(error.getMessage().contains("Precision must be between 1"));
      }
    }
  }

  private static ColumnVector makeDec128ColumnFromBigDecimals(BigDecimal[] values, int cudfScale) {
    BigInteger[] unscaled = new BigInteger[values.length];
    for (int i = 0; i < values.length; i++) {
      unscaled[i] = values[i] == null ? null : values[i].unscaledValue();
    }
    return ColumnVector.decimalFromBigInt(cudfScale, unscaled);
  }

  @Test
  void addSubLargeBatchAllRescaleExponents() {
    int numRows = 600;
    boolean[] noOverflow = new boolean[numRows];
    BigDecimal[] aVals = new BigDecimal[numRows];
    BigDecimal[] bVals = new BigDecimal[numRows];
    for (int i = 0; i < numRows; i++) {
      aVals[i] = BigDecimal.valueOf(i - 300);
      bVals[i] = BigDecimal.valueOf(i % 13, 30);
    }
    try (ColumnVector lhs = makeDec128ColumnFromBigDecimals(aVals, 0);
         ColumnVector rhs = makeDec128ColumnFromBigDecimals(bVals, -30);
         ColumnVector expectedOverflow = ColumnVector.fromBooleans(noOverflow)) {
      for (int drop = 1; drop <= 30; drop++) {
        int outBigDecimalScale = 30 - drop;
        int outCudfScale = -outBigDecimalScale;
        BigDecimal[] expectedAddVals = new BigDecimal[numRows];
        BigDecimal[] expectedSubVals = new BigDecimal[numRows];
        for (int i = 0; i < numRows; i++) {
          expectedAddVals[i] = aVals[i].add(bVals[i])
              .setScale(outBigDecimalScale, RoundingMode.HALF_UP);
          expectedSubVals[i] = aVals[i].subtract(bVals[i])
              .setScale(outBigDecimalScale, RoundingMode.HALF_UP);
        }
        try (ColumnVector expectedAdd = makeDec128ColumnFromBigDecimals(expectedAddVals, outCudfScale);
             ColumnVector expectedSub = makeDec128ColumnFromBigDecimals(expectedSubVals, outCudfScale);
             Table addRes = DecimalUtils.add128(lhs, rhs, outCudfScale);
             Table subRes = DecimalUtils.subtract128(lhs, rhs, outCudfScale)) {
          assertColumnsAreEqual(expectedOverflow, addRes.getColumn(0), "add overflow drop=" + drop);
          assertColumnsAreEqual(expectedAdd, addRes.getColumn(1), "add result drop=" + drop);
          assertColumnsAreEqual(expectedOverflow, subRes.getColumn(0), "sub overflow drop=" + drop);
          assertColumnsAreEqual(expectedSub, subRes.getColumn(1), "sub result drop=" + drop);
        }
      }
    }
  }

  @Test
  void addSubLargeBatchHighRaiseExponents() {
    int numRows = 600;
    boolean[] noOverflow = new boolean[numRows];
    int[] raises = {38, 50, 68, 73, 75};
    BigDecimal[] bVals = new BigDecimal[numRows];
    Arrays.fill(bVals, BigDecimal.ZERO.setScale(38, RoundingMode.UNNECESSARY));
    try (ColumnVector rhs = makeDec128ColumnFromBigDecimals(bVals, -38);
         ColumnVector expectedOverflow = ColumnVector.fromBooleans(noOverflow)) {
      for (int raise : raises) {
        int lhsCudfScale = raise - 38;
        BigInteger[] aUnscaled = new BigInteger[numRows];
        BigDecimal[] expectedAddVals = new BigDecimal[numRows];
        BigDecimal[] expectedSubVals = new BigDecimal[numRows];
        for (int i = 0; i < numRows; i++) {
          long v = (i % 7) - 3;
          aUnscaled[i] = BigInteger.valueOf(v);
          BigDecimal expected = BigDecimal.valueOf(v).scaleByPowerOfTen(lhsCudfScale)
              .setScale(0, RoundingMode.UNNECESSARY);
          expectedAddVals[i] = expected;
          expectedSubVals[i] = expected;
        }
        try (ColumnVector lhs = ColumnVector.decimalFromBigInt(lhsCudfScale, aUnscaled);
             ColumnVector expectedAdd = makeDec128ColumnFromBigDecimals(expectedAddVals, 0);
             ColumnVector expectedSub = makeDec128ColumnFromBigDecimals(expectedSubVals, 0);
             Table addRes = DecimalUtils.add128(lhs, rhs, 0);
             Table subRes = DecimalUtils.subtract128(lhs, rhs, 0);
             Table swappedAddRes = DecimalUtils.add128(rhs, lhs, 0)) {
          assertColumnsAreEqual(expectedOverflow, addRes.getColumn(0), "add overflow raise=" + raise);
          assertColumnsAreEqual(expectedAdd, addRes.getColumn(1), "add result raise=" + raise);
          assertColumnsAreEqual(expectedOverflow, subRes.getColumn(0), "sub overflow raise=" + raise);
          assertColumnsAreEqual(expectedSub, subRes.getColumn(1), "sub result raise=" + raise);
          assertColumnsAreEqual(expectedOverflow, swappedAddRes.getColumn(0),
              "swapped add overflow raise=" + raise);
          assertColumnsAreEqual(expectedAdd, swappedAddRes.getColumn(1),
              "swapped add result raise=" + raise);
        }
      }
    }
  }

  @Test
  void addSubScaleDiffBoundary() {
    BigInteger maxDec38 = BigInteger.TEN.pow(38).subtract(BigInteger.ONE);
    // scaleDiff == 76 (38 - (-38)) is the max supported exponent in pow_ten_table
    try (ColumnVector lhs = ColumnVector.decimalFromBigInt(38, BigInteger.ONE);
         ColumnVector rhsNeg = ColumnVector.decimalFromBigInt(-38, maxDec38.negate());
         ColumnVector rhsPos = ColumnVector.decimalFromBigInt(-38, maxDec38);
         ColumnVector expectedOverflow = ColumnVector.fromBooleans(false);
         ColumnVector expectedResult = ColumnVector.decimalFromBigInt(0, maxDec38);
         Table addRes = DecimalUtils.add128(lhs, rhsNeg, 0);
         Table subRes = DecimalUtils.subtract128(lhs, rhsPos, 0)) {
      assertColumnsAreEqual(expectedOverflow, addRes.getColumn(0), "add scaleDiff=76 overflow");
      assertColumnsAreEqual(expectedResult, addRes.getColumn(1), "add scaleDiff=76 result");
      assertColumnsAreEqual(expectedOverflow, subRes.getColumn(0), "sub scaleDiff=76 overflow");
      assertColumnsAreEqual(expectedResult, subRes.getColumn(1), "sub scaleDiff=76 result");
    }

    // scaleDiff == 77 (39 - (-38)) and int-overflowing scale diff must be rejected
    try (ColumnVector scale39 = ColumnVector.decimalFromBigInt(39, BigInteger.ZERO);
         ColumnVector scaleNeg38 = ColumnVector.decimalFromBigInt(-38, BigInteger.ZERO);
         ColumnVector scaleMax = ColumnVector.decimalFromBigInt(Integer.MAX_VALUE, BigInteger.ZERO);
         ColumnVector scaleNeg1 = ColumnVector.decimalFromBigInt(-1, BigInteger.ZERO)) {
      assertThrows(IllegalArgumentException.class, () -> {
        try (Table ignored = DecimalUtils.add128(scale39, scaleNeg38, 0)) {}
      });
      assertThrows(IllegalArgumentException.class, () -> {
        try (Table ignored = DecimalUtils.add128(scaleNeg38, scale39, 0)) {}
      });
      assertThrows(IllegalArgumentException.class, () -> {
        try (Table ignored = DecimalUtils.subtract128(scale39, scaleNeg38, 0)) {}
      });
      assertThrows(IllegalArgumentException.class, () -> {
        try (Table ignored = DecimalUtils.subtract128(scaleNeg38, scale39, 0)) {}
      });
      assertThrows(IllegalArgumentException.class, () -> {
        try (Table ignored = DecimalUtils.add128(scaleMax, scaleNeg1, 0)) {}
      });
      assertThrows(IllegalArgumentException.class, () -> {
        try (Table ignored = DecimalUtils.subtract128(scaleMax, scaleNeg1, 0)) {}
      });
    }
  }

  @Test
  void intDivideLargeBatchAllScaleDiffs() {
    int numRows = 600;
    boolean[] noOverflow = new boolean[numRows];
    BigDecimal[] aVals = new BigDecimal[numRows];
    for (int i = 0; i < numRows; i++) {
      aVals[i] = BigDecimal.valueOf((i - 300) * 13L, 1);
    }
    try (ColumnVector lhs = makeDec128ColumnFromBigDecimals(aVals, -1);
         ColumnVector expectedOverflow = ColumnVector.fromBooleans(noOverflow)) {
      for (int shift = 1; shift <= 30; shift++) {
        int bScale = 1 + shift;
        BigDecimal[] bVals = new BigDecimal[numRows];
        long[] expectedQuot = new long[numRows];
        for (int i = 0; i < numRows; i++) {
          BigDecimal denom = BigDecimal.valueOf(((i % 7) + 1) * 25L, 2)
              .setScale(bScale, RoundingMode.UNNECESSARY);
          bVals[i] = denom;
          expectedQuot[i] = aVals[i].divideToIntegralValue(denom).longValueExact();
        }
        try (ColumnVector rhs = makeDec128ColumnFromBigDecimals(bVals, -bScale);
             ColumnVector expected = ColumnVector.fromLongs(expectedQuot);
             Table res = DecimalUtils.integerDivide128(lhs, rhs)) {
          assertColumnsAreEqual(expectedOverflow, res.getColumn(0), "intDiv overflow shift=" + shift);
          assertColumnsAreEqual(expected, res.getColumn(1), "intDiv result shift=" + shift);
        }
      }
    }

    // Also exercise n_shift_exp > 0 (lhs scale smaller than rhs scale)
    int[] posShifts = {11, 14, 16, 18};
    BigDecimal[] aFineVals = new BigDecimal[numRows];
    for (int i = 0; i < numRows; i++) {
      aFineVals[i] = BigDecimal.valueOf((i - 300) * 13L).scaleByPowerOfTen(-1)
          .setScale(20, RoundingMode.UNNECESSARY);
    }
    try (ColumnVector lhs = makeDec128ColumnFromBigDecimals(aFineVals, -20);
         ColumnVector expectedOverflow = ColumnVector.fromBooleans(noOverflow)) {
      for (int shift : posShifts) {
        int bScale = 20 - shift;
        BigDecimal[] bVals = new BigDecimal[numRows];
        long[] expectedQuot = new long[numRows];
        for (int i = 0; i < numRows; i++) {
          BigDecimal denom = BigDecimal.valueOf(((i % 7) + 1) * 25L, 2)
              .setScale(bScale, RoundingMode.UNNECESSARY);
          bVals[i] = denom;
          expectedQuot[i] = aFineVals[i].divideToIntegralValue(denom).longValueExact();
        }
        try (ColumnVector rhs = makeDec128ColumnFromBigDecimals(bVals, -bScale);
             ColumnVector expected = ColumnVector.fromLongs(expectedQuot);
             Table res = DecimalUtils.integerDivide128(lhs, rhs)) {
          assertColumnsAreEqual(expectedOverflow, res.getColumn(0),
              "intDiv posShift overflow shift=" + shift);
          assertColumnsAreEqual(expected, res.getColumn(1),
              "intDiv posShift result shift=" + shift);
        }
      }
    }

    // Also exercise n_shift_exp < -38 (remaining_exp = 11, 14, 16, 18)
    BigDecimal[] bScale38Vals = new BigDecimal[numRows];
    for (int i = 0; i < numRows; i++) {
      bScale38Vals[i] = BigDecimal.valueOf(((i % 3) + 1) * 25L, 2)
          .setScale(38, RoundingMode.UNNECESSARY);
    }
    try (ColumnVector rhs = makeDec128ColumnFromBigDecimals(bScale38Vals, -38);
         ColumnVector expectedOverflow = ColumnVector.fromBooleans(noOverflow)) {
      for (int remExp : posShifts) {
        BigInteger[] aUnscaled = new BigInteger[numRows];
        long[] expectedQuot = new long[numRows];
        for (int i = 0; i < numRows; i++) {
          long v = (i % 3) - 1;
          aUnscaled[i] = BigInteger.valueOf(v);
          BigDecimal num = BigDecimal.valueOf(v).scaleByPowerOfTen(remExp);
          expectedQuot[i] = num.divideToIntegralValue(bScale38Vals[i]).longValueExact();
        }
        try (ColumnVector lhs = ColumnVector.decimalFromBigInt(remExp, aUnscaled);
             ColumnVector expected = ColumnVector.fromLongs(expectedQuot);
             Table res = DecimalUtils.integerDivide128(lhs, rhs)) {
          assertColumnsAreEqual(expectedOverflow, res.getColumn(0),
              "intDiv longShift overflow remExp=" + remExp);
          assertColumnsAreEqual(expected, res.getColumn(1),
              "intDiv longShift result remExp=" + remExp);
        }
      }
    }
  }

  @Test
  void divideMultiplyRemainderLargeBatchLargeScaleDiffs() {
    int numRows = 600;
    boolean[] noOverflow = new boolean[numRows];
    int[] shifts = {11, 14, 16, 18, 25};
    try (ColumnVector expectedOverflow = ColumnVector.fromBooleans(noOverflow)) {
      for (int shift : shifts) {
        // Choose output/rhs scales so each op passes `shift` directly to pow_ten:
        //   divide128:    -n_shift_exp = divOutScale - 1 + divBScale = shift
        //   multiply128:  exponent     = (1 + bScale) - mulOutScale  = shift
        //   remainder128: -n_shift_exp = bScale - 1                  = shift
        int bScale = 1 + shift;
        int divOutScale = 6;
        int divBScale = shift - (divOutScale - 1);
        int mulOutScale = 2;
        BigDecimal[] aVals = new BigDecimal[numRows];
        BigDecimal[] bDivVals = new BigDecimal[numRows];
        BigDecimal[] bVals = new BigDecimal[numRows];
        BigDecimal[] expDiv = new BigDecimal[numRows];
        BigDecimal[] expMul = new BigDecimal[numRows];
        BigDecimal[] expRem = new BigDecimal[numRows];
        for (int i = 0; i < numRows; i++) {
          BigDecimal a = BigDecimal.valueOf((i - 300) * 13L, 1);
          BigDecimal bBase = BigDecimal.valueOf(((i % 7) + 1) * 25L, 2);
          BigDecimal bDiv = bBase.setScale(divBScale, RoundingMode.UNNECESSARY);
          BigDecimal b = bBase.setScale(bScale, RoundingMode.UNNECESSARY);
          aVals[i] = a;
          bDivVals[i] = bDiv;
          bVals[i] = b;
          expDiv[i] = a.divide(bDiv, divOutScale, RoundingMode.HALF_UP);
          expMul[i] = a.multiply(b).setScale(mulOutScale, RoundingMode.HALF_UP);
          expRem[i] = a.remainder(b).setScale(bScale, RoundingMode.UNNECESSARY);
        }
        try (ColumnVector lhs = makeDec128ColumnFromBigDecimals(aVals, -1);
             ColumnVector rhsDiv = makeDec128ColumnFromBigDecimals(bDivVals, -divBScale);
             ColumnVector rhs = makeDec128ColumnFromBigDecimals(bVals, -bScale);
             ColumnVector expDivCol = makeDec128ColumnFromBigDecimals(expDiv, -divOutScale);
             ColumnVector expMulCol = makeDec128ColumnFromBigDecimals(expMul, -mulOutScale);
             ColumnVector expRemCol = makeDec128ColumnFromBigDecimals(expRem, -bScale);
             Table divRes = DecimalUtils.divide128(lhs, rhsDiv, -divOutScale);
             Table mulRes = DecimalUtils.multiply128(lhs, rhs, -mulOutScale, false);
             Table remRes = DecimalUtils.remainder128(lhs, rhs, -bScale)) {
          assertColumnsAreEqual(expectedOverflow, divRes.getColumn(0), "div overflow shift=" + shift);
          assertColumnsAreEqual(expDivCol, divRes.getColumn(1), "div result shift=" + shift);
          assertColumnsAreEqual(expectedOverflow, mulRes.getColumn(0), "mul overflow shift=" + shift);
          assertColumnsAreEqual(expMulCol, mulRes.getColumn(1), "mul result shift=" + shift);
          assertColumnsAreEqual(expectedOverflow, remRes.getColumn(0), "rem overflow shift=" + shift);
          assertColumnsAreEqual(expRemCol, remRes.getColumn(1), "rem result shift=" + shift);
        }
      }

      // Also exercise multiply128 scale-up (exponent < 0) and precision10 via interimCast
      BigDecimal[] smallA = new BigDecimal[numRows];
      BigDecimal[] smallB = new BigDecimal[numRows];
      BigDecimal[] expMulUp = new BigDecimal[numRows];
      BigDecimal[] bigA = new BigDecimal[numRows];
      BigDecimal[] bigB = new BigDecimal[numRows];
      BigDecimal[] expBigMul = new BigDecimal[numRows];
      BigInteger pow10_25 = BigInteger.TEN.pow(25);
      BigInteger pow10_10 = BigInteger.TEN.pow(10);
      for (int i = 0; i < numRows; i++) {
        smallA[i] = BigDecimal.valueOf(i - 300);
        smallB[i] = BigDecimal.valueOf((i % 9) - 4);
        expMulUp[i] = smallA[i].multiply(smallB[i]).setScale(16, RoundingMode.UNNECESSARY);

        // 54-digit product -> first_div_precision = 16 (pow_ten(16)), then drop 13 digits to scale 6
        BigInteger aUnscaled = BigInteger.valueOf(1_000_000_000L + i)
            .multiply(pow10_25)
            .add(BigInteger.valueOf((i % 97) + 1));
        BigInteger bUnscaled = BigInteger.valueOf(1_000_000_000L + (i % 17))
            .multiply(pow10_10)
            .add(BigInteger.valueOf((i % 13) + 1));
        bigA[i] = new BigDecimal(aUnscaled, 20);
        bigB[i] = new BigDecimal(bUnscaled, 15);
        expBigMul[i] = bigA[i].multiply(bigB[i])
            .setScale(19, RoundingMode.HALF_UP)
            .setScale(6, RoundingMode.HALF_UP);
      }
      try (ColumnVector lhsSmall = makeDec128ColumnFromBigDecimals(smallA, 0);
           ColumnVector rhsSmall = makeDec128ColumnFromBigDecimals(smallB, 0);
           ColumnVector expMulUpCol = makeDec128ColumnFromBigDecimals(expMulUp, -16);
           Table mulUpRes = DecimalUtils.multiply128(lhsSmall, rhsSmall, -16, false);
           ColumnVector lhsBig = makeDec128ColumnFromBigDecimals(bigA, -20);
           ColumnVector rhsBig = makeDec128ColumnFromBigDecimals(bigB, -15);
           ColumnVector expBigMulCol = makeDec128ColumnFromBigDecimals(expBigMul, -6);
           Table bigMulRes = DecimalUtils.multiply128(lhsBig, rhsBig, -6, true)) {
        assertColumnsAreEqual(expectedOverflow, mulUpRes.getColumn(0), "mulUp overflow");
        assertColumnsAreEqual(expMulUpCol, mulUpRes.getColumn(1), "mulUp result");
        assertColumnsAreEqual(expectedOverflow, bigMulRes.getColumn(0), "bigMul interimCast overflow");
        assertColumnsAreEqual(expBigMulCol, bigMulRes.getColumn(1), "bigMul interimCast result");
      }
    }
  }
}
