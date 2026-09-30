/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
import ai.rapids.cudf.DType;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.math.BigInteger;

import static ai.rapids.cudf.AssertUtils.*;

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
}
