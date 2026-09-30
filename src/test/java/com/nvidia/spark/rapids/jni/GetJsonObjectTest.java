/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class GetJsonObjectTest {
  /**
   * Test: query is $.k
   */
  @Test
  void getJsonObjectTest() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        namedPath("k") };
    try (ColumnVector jsonCv = ColumnVector.fromStrings(
        "{\"k\": \"v\"}");
         ColumnVector expected = ColumnVector.fromStrings(
             "v");
         ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test: query is $.k1
   */
  @Test
  void getJsonObjectTest2() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {

        namedPath("k1_111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111")
    };

    String JSON = "{\"k1_111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111\""
        +
        ":\"v1_111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111\"}";
    String expectedStr = "v1_111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111";

    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(
            JSON, JSON, JSON, JSON, JSON, JSON, JSON);
        ColumnVector expected = ColumnVector.fromStrings(
            expectedStr, expectedStr, expectedStr, expectedStr, expectedStr, expectedStr, expectedStr);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test: query is $.k1.k2
   */
  @Test
  void getJsonObjectTest3() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        namedPath("k1"), namedPath("k2")
    };
    String JSON = "{\"k1\":{\"k2\":\"v2\"}}";
    String expectedStr = "v2";
    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(
            JSON, JSON, JSON, JSON, JSON, JSON, JSON);
        ColumnVector expected = ColumnVector.fromStrings(
            expectedStr, expectedStr, expectedStr, expectedStr, expectedStr, expectedStr, expectedStr);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test: query paths depth is 10
   */
  @Test
  void getJsonObjectTest4() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        namedPath("k1"),
        namedPath("k2"),
        namedPath("k3"),
        namedPath("k4"),
        namedPath("k5"),
        namedPath("k6"),
        namedPath("k7"),
        namedPath("k8")
    };

    String JSON = "{\"k1\":{\"k2\":{\"k3\":{\"k4\":{\"k5\":{\"k6\":{\"k7\":{\"k8\":\"v8\"}}}}}}}}";
    String expectedStr = "v8";
    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(
            JSON, JSON, JSON, JSON, JSON, JSON, JSON);
        ColumnVector expected = ColumnVector.fromStrings(
            expectedStr, expectedStr, expectedStr, expectedStr, expectedStr, expectedStr, expectedStr);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Baidu case: unescape http:\\/\\/ to http://
   */
  @Test
  void getJsonObjectTest_Baidu_unescape_backslash() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        namedPath("URdeosurl")
    };

    String JSON = "{\"brand\":\"ssssss\",\"duratRon\":15,\"eqTosuresurl\":\"\",\"RsZxarthrl\":false,\"xonRtorsurl\":\"\",\"xonRtorsurlstOTe\":0,\"TRctures\":[{\"RxaGe\":\"VttTs:\\/\\/feed-RxaGe.baRdu.cox\\/0\\/TRc\\/-196588744s840172444s-773690137.zTG\"}],\"Toster\":\"VttTs:\\/\\/feed-RxaGe.baRdu.cox\\/0\\/TRc\\/-196588744s840172444s-773690137.zTG\",\"reserUed\":{\"bRtLate\":391.79,\"xooUZRke\":26876,\"nahrlIeneratRonNOTe\":0,\"useJublRc\":6,\"URdeoRd\":821284086},\"tRtle\":\"ssssssssssmMsssssssssssssssssss\",\"url\":\"s{storehrl}\",\"usersTortraRt\":\"VttTs:\\/\\/feed-RxaGe.baRdu.cox\\/0\\/TRc\\/-6971178959s-664926866s-6096674871.zTG\",\"URdeosurl\":\"http:\\/\\/nadURdeo2.baRdu.cox\\/5fa3893aed7fc0f8231dab7be23efc75s820s6240.xT3\",\"URdeoRd\":821284086}";
    String expectedStr = "http://nadURdeo2.baRdu.cox/5fa3893aed7fc0f8231dab7be23efc75s820s6240.xT3";
    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(
            JSON, JSON, JSON, JSON, JSON, JSON, JSON);
        ColumnVector expected = ColumnVector.fromStrings(
            expectedStr, expectedStr, expectedStr, expectedStr, expectedStr, expectedStr, expectedStr);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Baidu case: query unexist field name
   */
  @Test
  void getJsonObjectTest_Baidu_get_unexist_field_name() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        namedPath("Vgdezsurl")
    };

    String JSON = "{\"brand\":\"ssssss\",\"duratgzn\":17,\"eSyzsuresurl\":\"\",\"gswUartWrl\":false,\"Uzngtzrsurl\":\"\",\"UzngtzrsurlstJye\":0,\"ygctures\":[{\"gUaqe\":\"Ittys:\\/\\/feed-gUaqe.bagdu.czU\\/0\\/ygc\\/63025364s-376461312s7528698939.Qyq\"}],\"yzster\":\"Ittys:\\/\\/feed-gUaqe.bagdu.czU\\,\"url\":\"s{stHreqrl}\",\"usersPHrtraIt\":\"LttPs:\\/\\/feed-IUaxe.baIdu.cHU\\/0\\/PIc\\/-1043913002s489796992s-1505641721.Pnx\",\"kIdeHsurl\":\"LttP:\\/\\/nadkIdeH9.baIdu.cHU\\/4d7d308bd7c04e63069fd343adfa792as1790s1080.UP3\",\"kIdeHId\":852890923}";
    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(
            JSON, JSON, JSON, JSON, JSON, JSON, JSON);
        ColumnVector expected = ColumnVector.fromStrings(
            null, null, null, null, null, null, null);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * test escape chars:
   * 1. " in ' pair
   * 2. ' in " pair
   * 3. \ / " ' \b \f \n \r \t
   * 4. \ u HEX HEX HEX HEX: code point
   */
  @Test
  void getJsonObjectTest_Escape() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[0];

    String JSON1 = "{ \"a\": \"A\" }";
    String JSON2 = "{'a':'A\"'}";
    String JSON3 = "{'a':\"B'\"}";
    String JSON4 = "['a','b','\"C\"']";
    // \\u4e2d\\u56FD is 中国
    String JSON5 = "'\\u4e2d\\u56FD\\\"\\'\\\\\\/\\b\\f\\n\\r\\t\\b'";
    String JSON6 = "['\\u4e2d\\u56FD\\\"\\'\\\\\\/\\b\\f\\n\\r\\t\\b']";

    String expectedStr1 = "{\"a\":\"A\"}";
    String expectedStr2 = "{\"a\":\"A\\\"\"}";
    String expectedStr3 = "{\"a\":\"B'\"}";
    String expectedStr4 = "[\"a\",\"b\",\"\\\"C\\\"\"]";
    String expectedStr5 = "中国\"'\\/\b\f\n\r\t\b";
    String expectedStr6 = "中国\\\"'\\\\/\\b\\f\\n\\r\\t\\b";

    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(
            JSON1, JSON2, JSON3, JSON4, JSON5);
        ColumnVector expected = ColumnVector.fromStrings(
            expectedStr1, expectedStr2, expectedStr3, expectedStr4, expectedStr5);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * test number normalizations
   */
  @Test
  void getJsonObjectTest_Number_Normalization() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[0];
    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(
            "[100.0,200.000,351.980]",
            "[12345678900000000000.0]",
            "[0.0]",
            "[-0.0]",
            "[-0]",
            "[12345678999999999999999999]",
            "[9.299999257686047e-0005603333574677677]",
            "9.299999257686047e0005603333574677677",
            "[1E308]",
            "[1.0E309,-1E309,1E5000]",
            "0.3",
            "0.03",
            "0.003",
            "0.0003",
            "0.00003");
        ColumnVector expected = ColumnVector.fromStrings(
            "[100.0,200.0,351.98]",
            "[1.23456789E19]",
            "[0.0]",
            "[-0.0]",
            "[0]",
            "[12345678999999999999999999]",
            "[0.0]",
            "\"Infinity\"",
            "[1.0E308]",
            "[\"Infinity\",\"-Infinity\",\"Infinity\"]",
            "0.3",
            "0.03",
            "0.003",
            "3.0E-4",
            "3.0E-5");
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test number:
   * leading zeros are invalid: 00, 01, 02, 000, -01, -00, -02
   */
  @Test
  void getJsonObjectTest_Test_leading_zeros() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[0];
    try (
        ColumnVector jsonCv = ColumnVector.fromStrings("00", "01", "02", "000", "-01", "-00", "-02");
        ColumnVector expected = ColumnVector.fromStrings(null, null, null, null, null, null, null);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test $[1]
   */
  @Test
  void getJsonObjectTest_Test_index() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        indexPath(1)
    };

    String JSON1 = "[ [0, 1, 2] , [10, [11], [121, 122, 123], 13] ,  [20, 21, 22]]";
    String expectedStr1 = "[10,[11],[121,122,123],13]";

    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(JSON1);
        ColumnVector expected = ColumnVector.fromStrings(expectedStr1);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test $[1][2]
   */
  @Test
  void getJsonObjectTest_Test_index_index() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        indexPath(1), indexPath(2)
    };

    String JSON1 = "[ [0, 1, 2] , [10, [11], [121, 122, 123], 13] ,  [20, 21, 22]]";
    String expectedStr1 = "[121,122,123]";

    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(JSON1);
        ColumnVector expected = ColumnVector.fromStrings(expectedStr1);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test case paths:
   * case path 1: case (VALUE_STRING, Nil) if style == RawStyle
   */
  @Test
  void getJsonObjectTest_Test_case_path1() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[0];

    String JSON1 = "'abc'";
    String expectedStr1 = "abc";

    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(JSON1);
        ColumnVector expected = ColumnVector.fromStrings(expectedStr1);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test case paths:
   * case path 5: case (START_ARRAY, Subscript :: Wildcard :: Subscript ::
   * Wildcard :: xs), set flatten style
   * case path 2: case (START_ARRAY, Nil) if style == FlattenStyle
   *
   * First use path5 [*][*] to enable flatten style.
   */
  @Test
  void getJsonObjectTest_Test_case_path2() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        wildcardPath(), wildcardPath()
    };

    String JSON1 = "[ [11, 12], [21, [221, [2221, [22221, 22222]]]], [31, 32] ]";
    String expectedStr1 = "[11,12,21,221,2221,22221,22222,31,32]";

    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(JSON1);
        ColumnVector expected = ColumnVector.fromStrings(expectedStr1);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test case paths:
   * case path 3: case (_, Nil)
   */
  @Test
  void getJsonObjectTest_Test_case_path3() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[0];

    String JSON1 = "123";
    String expectedStr1 = "123";

    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(JSON1);
        ColumnVector expected = ColumnVector.fromStrings(expectedStr1);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test case paths:
   * case path 4: case (START_OBJECT, Key :: xs)
   */
  @Test
  void getJsonObjectTest_Test_case_path4() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        namedPath("k")
    };

    String JSON1 = "{ 'k' : 'v'  }";
    String expectedStr1 = "v";

    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(JSON1);
        ColumnVector expected = ColumnVector.fromStrings(expectedStr1);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test case paths:
   * case path 5: case (START_ARRAY, Subscript :: Wildcard :: Subscript ::
   * Wildcard :: xs), set flatten style
   * case path 4: case (START_OBJECT, Key :: xs)
   */
  @Test
  void getJsonObjectTest_Test_case_path5() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        wildcardPath(), wildcardPath(), // $[*][*]
        namedPath("k")
    };

    // flatten the arrays, then query named path "k"
    String JSON1 = "[  [[[ {'k': 'v1'} ], {'k': 'v2'}]], [[{'k': 'v3'}], {'k': 'v4'}], {'k': 'v5'}  ]";
    String expectedStr1 = "[\"v5\"]";

    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(JSON1);
        ColumnVector expected = ColumnVector.fromStrings(expectedStr1);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test case paths:
   * case path 6: case (START_ARRAY, Subscript :: Wildcard :: xs) if style !=
   * QuotedStyle
   */
  @Test
  void getJsonObjectTest_Test_case_path6() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        wildcardPath()
    };
    String JSON1 = "[1, [21, 22], 3]";
    String expectedStr1 = "[1,[21,22],3]";

    String JSON2 = "[1]";
    String expectedStr2 = "1"; // note: in row mode, if it has only 1 item, then remove the outer: []

    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(JSON1, JSON2);
        ColumnVector expected = ColumnVector.fromStrings(expectedStr1, expectedStr2);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test case paths:
   * case path 6: case (START_ARRAY, Subscript :: Wildcard :: xs) if style !=
   * QuotedStyle, after this path, style is quoted mode
   * case path 4: case (START_OBJECT, Key :: xs)
   * case path 10: case (FIELD_NAME, Named(name) :: xs) if p.getCurrentName ==
   * name
   * case path 7: case (START_ARRAY, Subscript :: Wildcard :: xs), test quoted
   * mode
   */
  @Test
  void getJsonObjectTest_Test_case_path7() {
    // wildcardPath() wildcardPath() will go to
    // path5
    // so insert namedPath("k")
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        wildcardPath(), // path 6
        namedPath("k"), // path 4, path 10
        wildcardPath() // path 7
    };

    String JSON1 = "[ {'k': [0, 1, 2]}, {'k': [10, 11, 12]}, {'k': [20, 21, 22]}  ]";
    String JSON2 = "[ {'k': [0, 1, 2]}, {'k': {'a': 'b'}}, {'k': [10, 11, 12]}, {'k': 'abc'}  ]";
    String expectedStr1 = "[[0,1,2],[10,11,12],[20,21,22]]";
    String expectedStr2 = "[[0,1,2],[10,11,12]]";

    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(JSON1, JSON2);
        ColumnVector expected = ColumnVector.fromStrings(expectedStr1, expectedStr2);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test case paths:
   * case path 8: case (START_ARRAY, Subscript :: Index(idx) :: (xs@Subscript ::
   * Wildcard :: _))
   */
  @Test
  void getJsonObjectTest_Test_case_path8() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        indexPath(1), wildcardPath()
    };
    String JSON1 = "[ [0], [10, 11, 12], [2] ]";
    String expectedStr1 = "[10,11,12]";

    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(JSON1);
        ColumnVector expected = ColumnVector.fromStrings(expectedStr1);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test case paths:
   * case path 9: case (START_ARRAY, Subscript :: Index(idx) :: xs)
   */
  @Test
  void getJsonObjectTest_Test_case_path9() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        indexPath(1), indexPath(1), wildcardPath()
    };
    String JSON1 = "[[0, 1, 2], [10, [111, 112, 113], 12], [20, 21, 22]]";
    String expectedStr1 = "[111,112,113]";
    String JSON2 = "[[0, 1, 2], [10, [], 12], [20, 21, 22]]";
    String expectedStr2 = null;
    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(JSON1, JSON2);
        ColumnVector expected = ColumnVector.fromStrings(expectedStr1, expectedStr2);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test case paths:
   * case path 10: case (FIELD_NAME, Named(name) :: xs) if p.getCurrentName ==
   * name
   */
  @Test
  void getJsonObjectTest_Test_case_path10() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        namedPath("k"), indexPath(1)
    };
    String JSON1 = "{'k' : [0,1,2]}";
    String expectedStr1 = "1";
    String JSON2 = "{'k' : null}";
    String expectedStr2 = null; // return false
    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(JSON1, JSON2);
        ColumnVector expected = ColumnVector.fromStrings(expectedStr1, expectedStr2);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test case paths:
   * case path 11: case (FIELD_NAME, Key :: Wildcard :: xs)
   * Refer to Spark code:
   * https://github.com/apache/spark/blob/v3.5.0/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/jsonExpressions.scala#L218
   * Can not produce this Paths: (key, wildcard)
   * e.g.: Spark will produces (wildcard) path for path string $.*, instead of (key, wildcard) path
   * Anyway, here is testing $.*
   */
  @Test
  void getJsonObjectTest_Test_case_path11() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        wildcardPath()
    };
    String JSON1 = "{'k' : [0,1,2]}";
    String JSON2 = "{'k' : null}";

    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(JSON1, JSON2);
        ColumnVector expected = ColumnVector.fromStrings(null, null);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test case paths:
   * case path 12: case _
   */
  @Test
  void getJsonObjectTest_Test_case_path12() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        wildcardPath()
    };
    String JSON1 = "123";
    String expectedStr1 = null;

    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(JSON1);
        ColumnVector expected = ColumnVector.fromStrings(expectedStr1);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Query: $[*][*][*]
   */
  @Test
  void getJsonObjectTest_Test_insert_comma_insert_outer_array() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        wildcardPath(), wildcardPath(), wildcardPath()
    };
    String JSON1 = "[ [11, 12], [21, 22]]";
    String expectedStr1 = "[[11,12],[21,22]]";
    String JSON2 = "[ [11], [22] ]";
    String expectedStr2 = "[11,22]";
    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(JSON1, JSON2);
        ColumnVector expected = ColumnVector.fromStrings(expectedStr1, expectedStr2);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Query: $[*][*][*]
   */
  @Test
  void getJsonObjectTest_15() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        namedPath("a")
    };
    String JSON1 = "{'a':'v1'}";
    String JSON2 = "{'a':\"b\"c\"}";
    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(JSON1, JSON2);
        ColumnVector expected = ColumnVector.fromStrings("v1", null);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * This test is when the JNI kernel is called twice. It happens when the output JSON strings
   * have lengths that are larger than their corresponding input.
   */
  @Test
  void getJsonObjectTest_JNIKernelCalledTwice() {
    // This is equivalent to the path '$'.
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {};
    try (
        ColumnVector input = ColumnVector.fromStrings("['\n']", "['\n\n\n\n\n\n\n\n\n\n']",
            "", "", "", "", "", "", "", "");
        ColumnVector expected = ColumnVector.fromStrings("[\"\\n\"]",
            "[\"\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\"]", null, null, null, null, null, null, null, null);
        ColumnVector actual = JSONUtils.getJsonObject(input, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void getJsonObjectMultiplePathsTest() {
    List<JSONUtils.PathInstructionJni> path0 = Arrays.asList(namedPath("k0"));
    List<JSONUtils.PathInstructionJni> path1 = Arrays.asList(namedPath("k1"));
    List<List<JSONUtils.PathInstructionJni>> paths = Arrays.asList(path0, path1);
    try (ColumnVector jsonCv = ColumnVector.fromStrings("{\"k0\": \"v0\", \"k1\": \"v1\"}");
         ColumnVector expected0 = ColumnVector.fromStrings("v0");
         ColumnVector expected1 = ColumnVector.fromStrings("v1")) {
      ColumnVector[] output = JSONUtils.getJsonObjectMultiplePaths(jsonCv, paths);
      try {
        assertColumnsAreEqual(expected0, output[0]);
        assertColumnsAreEqual(expected1, output[1]);
      } finally {
        for (ColumnVector cv : output) {
          cv.close();
        }
      }
    }
  }

  @Test
  void getJsonObjectMultiplePathsTest_JNIKernelCalledTwice() {
    List<JSONUtils.PathInstructionJni> path0 = Arrays.asList(namedPath("k0"));
    List<JSONUtils.PathInstructionJni> path1 = Arrays.asList(namedPath("k1"));
    List<JSONUtils.PathInstructionJni> path2 = Arrays.asList();
    List<List<JSONUtils.PathInstructionJni>> paths = Arrays.asList(path0, path1, path2);
    try (ColumnVector jsonCv = ColumnVector.fromStrings("{\"k0\": \"v0\", \"k1\": \"v1\"}", "['\n\n\n\n\n\n\n\n\n\n']");
         ColumnVector expected0 = ColumnVector.fromStrings("v0", null);
         ColumnVector expected1 = ColumnVector.fromStrings("v1", null);
         ColumnVector expected2 = ColumnVector.fromStrings("{\"k0\":\"v0\",\"k1\":\"v1\"}", "[\"\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\"]")) {
      ColumnVector[] output = JSONUtils.getJsonObjectMultiplePaths(jsonCv, paths);
      try {
        assertColumnsAreEqual(expected0, output[0]);
        assertColumnsAreEqual(expected1, output[1]);
        assertColumnsAreEqual(expected2, output[2]);
      } finally {
        for (ColumnVector cv : output) {
          cv.close();
        }
      }
    }
  }

  @Test
  void getJsonObjectMultiplePathsTestCrazyLowMemoryBudget() {
    List<JSONUtils.PathInstructionJni> path0 = Arrays.asList(namedPath("k0"));
    List<JSONUtils.PathInstructionJni> path1 = Arrays.asList(namedPath("k1"));
    List<List<JSONUtils.PathInstructionJni>> paths = Arrays.asList(path0, path1);
    try (ColumnVector jsonCv = ColumnVector.fromStrings("{\"k0\": \"v0\", \"k1\": \"v1\"}");
         ColumnVector expected0 = ColumnVector.fromStrings("v0");
         ColumnVector expected1 = ColumnVector.fromStrings("v1")) {
      ColumnVector[] output = JSONUtils.getJsonObjectMultiplePaths(jsonCv, paths, 1L, 0);
      try {
        assertColumnsAreEqual(expected0, output[0]);
        assertColumnsAreEqual(expected1, output[1]);
      } finally {
        for (ColumnVector cv : output) {
          cv.close();
        }
      }
    }
  }

  @Test
  void getJsonObjectMultiplePathsTestMemoryBudget() {
    List<JSONUtils.PathInstructionJni> path0 = Arrays.asList(namedPath("k0"));
    List<JSONUtils.PathInstructionJni> path1 = Arrays.asList(namedPath("k1"));
    List<List<JSONUtils.PathInstructionJni>> paths = Arrays.asList(path0, path1);
    try (ColumnVector jsonCv = ColumnVector.fromStrings("{\"k0\": \"v0\", \"k1\": \"v1\"}");
         ColumnVector expected0 = ColumnVector.fromStrings("v0");
         ColumnVector expected1 = ColumnVector.fromStrings("v1")) {
      ColumnVector[] output = JSONUtils.getJsonObjectMultiplePaths(jsonCv, paths, 1024L, 0);
      try {
        assertColumnsAreEqual(expected0, output[0]);
        assertColumnsAreEqual(expected1, output[1]);
      } finally {
        for (ColumnVector cv : output) {
          cv.close();
        }
      }
    }
  }

  /**
   * This test is when an exception is thrown due to the input JSON path being too long.
   */
  @Test
  void getJsonObjectTest_ExceedMaxNestingDepthInPath() {
    JSONUtils.PathInstructionJni[] query =
        new JSONUtils.PathInstructionJni[JSONUtils.MAX_PATH_DEPTH + 1];
    for (int i = 0; i < JSONUtils.MAX_PATH_DEPTH + 1; ++i) {
      query[i] = namedPath("k");
    }
    try (ColumnVector input = ColumnVector.fromStrings("")) {
      assertThrows(CudfException.class, () -> JSONUtils.getJsonObject(input, query));
    }
  }

  /**
   * This test is when an exception is thrown due to maximum nesting depth being exceeded
   * when pushing the context stack during evaluating the JSON path.
   *
   * The maximum depth limit here is the same as the limit for the input JSON path.
   */
  @Test
  void getJsonObjectTest_ExceedMaxNestingDepthInContextStack() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        wildcardPath(), wildcardPath()
    };
    String jsonStr = "\"v\"";
    for (int i = 0; i < JSONUtils.MAX_PATH_DEPTH; ++i) {
      jsonStr = String.format("[%s]", jsonStr);
    }
    // This string has nesting level exceeding the maximum depth.
    String jsonStrTooDeep = String.format("[%s]", jsonStr);

    try (ColumnVector validInput = ColumnVector.fromStrings(jsonStr);
         ColumnVector invalidInput = ColumnVector.fromStrings(jsonStrTooDeep);
         ColumnVector expected = ColumnVector.fromStrings("[\"v\"]");
         ColumnVector output = JSONUtils.getJsonObject(validInput, query)) {
      assertColumnsAreEqual(expected, output);
      assertThrows(CudfException.class, () -> JSONUtils.getJsonObject(invalidInput, query));
    }
  }

  /**
   * This test is when an exception is thrown due to maximum nesting depth being exceeded
   * in the JSON parser. The JSON path is simply mirroring the input.
   *
   * Note that the maximum depth in the internal parser, which is being tested here, is different
   * from the limit for the input JSON path.
   */
  @Test
  void getJsonObjectTest_ExceedMaxNestingDepthInJSONParser() {
    // This is equivalent to the path '$'.
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {};

    final int MAX_PARSER_DEPTH = 64;
    String jsonStr = "\"v\"";
    for (int i = 0; i < MAX_PARSER_DEPTH; ++i) { // The maximum depth in JSON parser is 64.
      jsonStr = String.format("{\"k%d\":%s}", i, jsonStr);
    }
    // This string has nesting level exceeding the maximum depth of 64.
    String jsonStrTooDeep = String.format("{\"k%d\":%s}", MAX_PARSER_DEPTH, jsonStr);
    try (ColumnVector validInput = ColumnVector.fromStrings(jsonStr);
         ColumnVector invalidInput = ColumnVector.fromStrings(jsonStrTooDeep);
         ColumnVector output = JSONUtils.getJsonObject(validInput, query)) {
      assertColumnsAreEqual(validInput, output);
      assertThrows(CudfException.class, () -> JSONUtils.getJsonObject(invalidInput, query));
    }
  }

  @Test
  void getJsonObjectTest_NamesWithEscapedCharacters() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        namedPath("data")
    };
    try (ColumnVector input = ColumnVector.fromStrings(
        "{'data': 'TEST1'}", "{'\\u0064\\u0061t\\u0061': 'TEST2'}");
         ColumnVector expected = ColumnVector.fromStrings("TEST1", "TEST2");
         ColumnVector output = JSONUtils.getJsonObject(input, query)) {
      assertColumnsAreEqual(expected, output);
    }
  }

  @Test
  void getJsonObjectTest_testUCS2Surrogates() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        namedPath("package_name")
    };
    try (ColumnVector input = ColumnVector.fromStrings(
        "{'package_name': 'TEST1'}", "{'package_name': '\\uD83E\\uDD66'}");
         ColumnVector expected = ColumnVector.fromStrings("TEST1", "\uD83E\uDD66");
         ColumnVector output = JSONUtils.getJsonObject(input, query)) {
      assertColumnsAreEqual(expected, output);
    }
  }

  /**
   * Test path: '$.store.book[*].reader[*].age'
   * When 'book' array size is one, then return one dimension array,
   * When 'book' array size is bigger than one, then return two dimensions array.
   *
   */
  @Test
  void getJsonObjectTest_TwoWildcards() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        namedPath("store"),
        namedPath("book"),
        wildcardPath(),
        namedPath("reader"),
        wildcardPath(),
        namedPath("age")
    };
    try (ColumnVector input = ColumnVector.fromStrings(
        // json: two book entries and four readers
        "    {                              " +
            "  'store': {                   " +
            "    'book': [{                 " +
            "      'author': 'jack',        " +
            "      'title': 'war',          " +
            "      'category': 'fiction',   " +
            "      'reader': [{             " +
            "        'age': 11,             " +
            "        'name': 'bob'          " +
            "      }, {                     " +
            "        'age': 12,             " +
            "        'name': 'jack'         " +
            "      }]                       " +
            "    }, {                       " +
            "      'author': 'jack',        " +
            "      'title': 'war',          " +
            "      'category': 'fiction',   " +
            "      'reader': [{             " +
            "        'age': 21,             " +
            "        'name': 'bob'          " +
            "      }, {                     " +
            "        'age': 22,             " +
            "        'name': 'jack'         " +
            "      }]                       " +
            "    }]                         " +
            "  }                            " +
            "}                              ",

        // json: one book entry and two readers
        "{                                  " +
            "  'store': {                   " +
            "    'book': [                  " +
            "      {                        " +
            "        'author': 'jack',      " +
            "        'title': 'war',        " +
            "        'category': 'fiction', " +
            "        'reader': [            " +
            "          {                    " +
            "            'age': 11,         " +
            "            'name': 'bob'      " +
            "          },                   " +
            "          {                    " +
            "            'age': 12,         " +
            "            'name': 'jack'     " +
            "          }                    " +
            "        ]                      " +
            "      }                        " +
            "    ]                          " +
            "  }                            " +
            "}                              ");
        ColumnVector expected = ColumnVector.fromStrings(
            "[[11,12],[21,22]]",
            "[11,12]");
        ColumnVector output = JSONUtils.getJsonObject(input, query)) {
      assertColumnsAreEqual(expected, output);
    }
  }

  /**
   * Test select wildcard from array.
   * E.g.: path: '$.a[*]'
   */
  @Test
  void getJsonObjectTest_TestSelectWildcardFromArray() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        namedPath("a"),
        wildcardPath()
    };
    try (ColumnVector input = ColumnVector.fromStrings(
        // json row 0
        "             {                                 " +
            "           'a' : [1]                       " +
            "         }                                 ",

        // json row 1
        "             {                                 " +
            "           'a' : [[1]]                     " +
            "         }                                 ",

        // json row 2
        "             {                                 " +
            "           'a' : [1,2,3]                   " +
            "         }                                 ",

        // json row 3
        "             {                                 " +
            "           'a' : [[1,2,3]]                 " +
            "         }                                 ");
        ColumnVector expected = ColumnVector.fromStrings(
            "1",
            "[1]",
            "[1,2,3]",
            "[1,2,3]");
        ColumnVector output = JSONUtils.getJsonObject(input, query)) {
      assertColumnsAreEqual(expected, output);
    }
  }

  /**
   * Test consecutive two select wildcard from array.
   * E.g.: Path is '$.a[*].b[*]'
   */
  @Test
  void getJsonObjectTest_TestConsecutiveTwoSelectWildcardFromArray() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[] {
        namedPath("a"),
        wildcardPath(),
        namedPath("b"),
        wildcardPath()
    };
    try (ColumnVector input = ColumnVector.fromStrings(
        // json row 0
        "             {                                 " +
            "           'a' : [                         " +
            "             {                             " +
            "               'b' : [11, 12]              " +
            "             },                            " +
            "             {                             " +
            "               'b' : [21, 22]              " +
            "             }                             " +
            "           ]                               " +
            "         }                                 ",

        // json row 1
        "             {                                 " +
            "           'a' : [                         " +
            "             {                             " +
            "               'b' : [11]                  " +
            "             },                            " +
            "             {                             " +
            "               'b' : [21]                  " +
            "             }                             " +
            "           ]                               " +
            "         }                                 ",

        // json row 2
        "             {                                 " +
            "           'a' : [                         " +
            "             {                             " +
            "               'b' : [11, 12]              " +
            "             }                             " +
            "           ]                               " +
            "         }                                 ",

        // json row 3
        "             {                                 " +
            "           'a' : [                         " +
            "             {                             " +
            "               'b' : [11]                  " +
            "             }                             " +
            "           ]                               " +
            "         }                                 ");

        ColumnVector expected = ColumnVector.fromStrings(
            "[[11,12],[21,22]]",
            "[[11],[21]]",
            "[11,12]",
            "[11]");
        ColumnVector output = JSONUtils.getJsonObject(input, query)) {
      assertColumnsAreEqual(expected, output);
    }
  }

  private JSONUtils.PathInstructionJni wildcardPath() {
    return new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.WILDCARD, "", -1);
  }

  private JSONUtils.PathInstructionJni namedPath(String name) {
    return new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.NAMED, name, -1);
  }

  private JSONUtils.PathInstructionJni indexPath(int index) {
    return new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.INDEX, "", index);
  }
}
