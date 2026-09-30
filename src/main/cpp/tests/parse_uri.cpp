/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <exception_with_row_index.hpp>
#include <parse_uri.hpp>

struct ParseURIProtocolTests : public cudf::test::BaseFixture {};
struct ParseURIHostTests : public cudf::test::BaseFixture {};
struct ParseURIQueryTests : public cudf::test::BaseFixture {};
struct ParseURIPathTests : public cudf::test::BaseFixture {};

enum class test_types {
  SIMPLE,
  SPARK_EDGES,
  IPv6,
  IPv4,
  UTF8,
  QUERY,
};

namespace {
cudf::test::strings_column_wrapper get_test_data(test_types t)
{
  switch (t) {
    case test_types::SIMPLE:
      return cudf::test::strings_column_wrapper({
        "https://www.nvidia.com/s/uri?param1=2",
        "http://www.nvidia.com",
        "file://path/to/a/cool/file",
        "smb://network/path/to/file",
        "http:/www.nvidia.com",
        "file:path/to/a/cool/file",
        "/network/path/to/file",
        "nvidia.com",
        "www.nvidia.com/s/uri",
      });

    case test_types::SPARK_EDGES:
      return cudf::test::strings_column_wrapper(
        {"https://nvidia.com/https&#://nvidia.com",
         "https://http://www.nvidia.com",
         "filesystemmagicthing://bob.yaml",
         "nvidia.com:8080",
         "http://thisisinvalid.data/due/to-the_character%s/inside*the#url`~",
         "file:/absolute/path",
         "//www.nvidia.com",
         "#bob",
         "#this%doesnt#make//sense://to/me",
         "HTTP:&bob",
         "/absolute/path",
         "http://%77%77%77.%4EV%49%44%49%41.com",
         "https:://broken.url",
         "https://www.nvidia.com/q/This%20is%20a%20query",
         "https://www.nvidia.com/\x93path/path/to/file",
         "http://?",
         "http://??",
         "http://\?\?/",
         "http://#",
         "http://user:pass@host/file;param?query;p2",
         "http://[1:2:3:4:5:6:7::]",
         "http://[::2:3:4:5:6:7:8]",
         "http://[fe80::7:8%eth0]",
         "http://[fe80::7:8%1]",
         "http://foo.bar/abc/\\\\\\http://foo.bar/abc.gif\\\\\\",
         "www.nvidia.com:8100/servlet/impc.DisplayCredits?primekey_in=2000041100:05:14115240636",
         "https://nvidia.com/2Ru15Ss ",
         "http://www.nvidia.com/plugins//##",
         "www.nvidia.com:81/Free.fr/L7D9qw9X4S-aC0&amp;D4X0/Panels&amp;solutionId=0X54a/"
         "cCdyncharset=UTF-8&amp;t=01wx58Tab&amp;ps=solution/"
         "ccmd=_help&amp;locale0X1&amp;countrycode=MA/",
         "http://www.nvidia.com/tags.php?%2F88\323\351\300ึณวน\331\315\370%2F",
         "http://www.nvidia.com//wp-admin/includes/index.html#9389#123",
         "http://www.nvidia.com/"
         "object.php?object=ะก-\320%9Fะฑ-ะฟ-ะก\321%82\321%80ะตะป\321%8Cะฝะฐ-\321%83ะป-\320%"
         "97ะฐะฒะพะด\321%81ะบะฐ\321%8F.html&sid=5",
         "http://www.nvidia.com/picshow.asp?id=106&mnid=5080&classname=\271\253ืฐฦช",
         "http://-.~_!$&'()*+,;=:%40:80%2f::::::@nvidia.com:443",
         "http://userid:password@example.com:8080/",
         "http://.www.nvidia.com./",
         "http://www.nvidia..com/"});
    case test_types::IPv6:
      return cudf::test::strings_column_wrapper({
        "https://[fe80::]",
        "https://[2001:0db8:85a3:0000:0000:8a2e:0370:7334]",
        "https://[2001:0DB8:85A3:0000:0000:8A2E:0370:7334]",
        "https://[2001:db8::1:0]",
        "http://[2001:db8::2:1]",
        "https://[::1]",
        "https://[2001:db8:85a3:8d3:1319:8a2e:370:7348]:443",
        "https://[2001:db8:3333:4444:5555:6666:1.2.3.4]/path/to/file",
        "https://[2001:db8:3333:4444:5555:6666:7777:8888:1.2.3.4]/path/to/file",
        "https://[::db8:3333:4444:5555:6666:1.2.3.4]/path/to/file]",  // this is valid, but spark
                                                                      // doesn't think so
      });
    case test_types::IPv4:
      return cudf::test::strings_column_wrapper({
        "https://192.168.1.100/",
        "https://192.168.1.100:8443/",
        "https://192.168.1.100.5/",
        "https://192.168.1/",
        "https://280.100.1.1/",
        "https://182.168..100/path/to/file",
      });
    case test_types::UTF8:
      return cudf::test::strings_column_wrapper({
        "https://nvidia.com/%4EV%49%44%49%41",
        "http://%77%77%77.%4EV%49%44%49%41.com",
        "http://✪↩d⁚f„⁈.ws/123",
        "https:// /path/to/file",
      });
    case test_types::QUERY:
      return cudf::test::strings_column_wrapper(
        {"https://www.nvidia.com/path?param0=1&param2=3&param4=5",
         "https:// /?params=5&cloth=0&metal=1&param0=param3",
         "https://[2001:db8::2:1]:443/parms/in/the/uri?a=b&param0=true",
         "https://[::1]/?invalid=param&f„⁈.=7&param0=3",
         "https://[::1]/?invalid=param&param0=f„⁈&~.=!@&^",
         "userinfo@www.nvidia.com/path?query=1&param0=5#Ref",
         "https://www.nvidia.com/path?brokenparam0=1&fakeparam0=5&param0=true",
         "http://nvidia.com?CBA=CBA&C=C"});
    default: CUDF_FAIL("Test type unsupported!"); return cudf::test::strings_column_wrapper();
  }
}
}  // namespace

TEST_F(ParseURIProtocolTests, Simple)
{
  auto const col    = get_test_data(test_types::SIMPLE);
  auto const result = spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper const expected(
    {"https", "http", "file", "smb", "http", "file", "", "", ""}, {1, 1, 1, 1, 1, 1, 0, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(ParseURIProtocolTests, SparkEdges)
{
  auto const col    = get_test_data(test_types::SPARK_EDGES);
  auto const result = spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper const expected(
    {"https",
     "https",
     "filesystemmagicthing",
     "nvidia.com",
     "",
     "file",
     "",
     "",
     "",
     "HTTP",
     "",
     "http",
     "https",
     "https",
     "",
     "http",
     "http",
     "http",
     "http",
     "http",
     "http",
     "http",
     "http",
     "http",
     "",
     "www.nvidia.com",
     "",
     "",
     "www.nvidia.com",
     "",
     "",
     "",
     "",
     "http",
     "http",
     "http",
     "http"},
    {1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(ParseURIProtocolTests, IP6)
{
  auto const col    = get_test_data(test_types::IPv6);
  auto const result = spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper const expected(
    {"https", "https", "https", "https", "http", "https", "https", "https", "", ""},
    {1, 1, 1, 1, 1, 1, 1, 1, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(ParseURIProtocolTests, IP4)
{
  auto const col    = get_test_data(test_types::IPv4);
  auto const result = spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper const expected(
    {"https", "https", "https", "https", "https", "https"});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(ParseURIProtocolTests, UTF8)
{
  auto const col    = get_test_data(test_types::UTF8);
  auto const result = spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper const expected({"https", "http", "http", ""}, {1, 1, 1, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(ParseURIHostTests, Simple)
{
  auto const col    = get_test_data(test_types::SIMPLE);
  auto const result = spark_rapids_jni::parse_uri_to_host(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper const expected(
    {"www.nvidia.com", "www.nvidia.com", "path", "network", "", "", "", "", ""},
    {1, 1, 1, 1, 0, 0, 0, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(ParseURIHostTests, SparkEdges)
{
  auto const col    = get_test_data(test_types::SPARK_EDGES);
  auto const result = spark_rapids_jni::parse_uri_to_host(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper const expected(
    {"nvidia.com",
     "http",
     "bob.yaml",
     "",
     "",
     "",
     "www.nvidia.com",
     "",
     "",
     "",
     "",
     "",
     "",
     "www.nvidia.com",
     "",
     "",
     "",
     "",
     "",
     "host",
     "[1:2:3:4:5:6:7::]",
     "[::2:3:4:5:6:7:8]",
     "[fe80::7:8%eth0]",
     "[fe80::7:8%1]",
     "",
     "",
     "",
     "",
     "",
     "",
     "",
     "",
     "",
     "nvidia.com",
     "example.com",
     "",
     ""},
    {1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(ParseURIHostTests, IP6)
{
  auto const col    = get_test_data(test_types::IPv6);
  auto const result = spark_rapids_jni::parse_uri_to_host(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper const expected({"[fe80::]",
                                                     "[2001:0db8:85a3:0000:0000:8a2e:0370:7334]",
                                                     "[2001:0DB8:85A3:0000:0000:8A2E:0370:7334]",
                                                     "[2001:db8::1:0]",
                                                     "[2001:db8::2:1]",
                                                     "[::1]",
                                                     "[2001:db8:85a3:8d3:1319:8a2e:370:7348]",
                                                     "[2001:db8:3333:4444:5555:6666:1.2.3.4]",
                                                     "",
                                                     ""},
                                                    {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(ParseURIHostTests, IP4)
{
  auto const col    = get_test_data(test_types::IPv4);
  auto const result = spark_rapids_jni::parse_uri_to_host(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper const expected(
    {"192.168.1.100", "192.168.1.100", "", "", "", ""}, {1, 1, 0, 0, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(ParseURIHostTests, UTF8)
{
  auto const col    = get_test_data(test_types::UTF8);
  auto const result = spark_rapids_jni::parse_uri_to_host(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper const expected({"nvidia.com", "", "", ""}, {1, 0, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(ParseURIQueryTests, Simple)
{
  auto const col    = get_test_data(test_types::SIMPLE);
  auto const result = spark_rapids_jni::parse_uri_to_query(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper const expected({"param1=2", "", "", "", "", "", "", "", ""},
                                                    {1, 0, 0, 0, 0, 0, 0, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(ParseURIQueryTests, SparkEdges)
{
  auto const col    = get_test_data(test_types::SPARK_EDGES);
  auto const result = spark_rapids_jni::parse_uri_to_query(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper const expected(
    {"",  "",   "", "",         "", "", "", "", "", "", "", "", "", "", "",
     "",  // empty
     "?", "?/", "", "query;p2", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(ParseURIQueryTests, Queries)
{
  auto const col = get_test_data(test_types::QUERY);

  {
    auto const result = spark_rapids_jni::parse_uri_to_query(cudf::strings_column_view{col});

    cudf::test::strings_column_wrapper const expected({"param0=1&param2=3&param4=5",
                                                       "",
                                                       "a=b&param0=true",
                                                       "invalid=param&f„⁈.=7&param0=3",
                                                       "",
                                                       "query=1&param0=5",
                                                       "brokenparam0=1&fakeparam0=5&param0=true",
                                                       "CBA=CBA&C=C"},
                                                      {1, 0, 1, 1, 0, 1, 1, 1});

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
  }
  {
    auto const result =
      spark_rapids_jni::parse_uri_to_query(cudf::strings_column_view{col}, std::string("param0"));
    cudf::test::strings_column_wrapper const expected({"1", "", "true", "3", "", "5", "true", ""},
                                                      {1, 0, 1, 1, 0, 1, 1, 0});

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
  }
  {
    auto const result =
      spark_rapids_jni::parse_uri_to_query(cudf::strings_column_view{col}, std::string("C"));
    cudf::test::strings_column_wrapper const expected({"", "", "", "", "", "", "", "C"},
                                                      {0, 0, 0, 0, 0, 0, 0, 1});

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
  }
  {
    cudf::test::strings_column_wrapper const query(
      {"param0", "q", "a", "invalid", "test", "query", "fakeparam0", "C"});
    cudf::test::strings_column_wrapper const expected({"1", "", "b", "param", "", "1", "5", "C"},
                                                      {1, 0, 1, 1, 0, 1, 1, 1});

    auto const result = spark_rapids_jni::parse_uri_to_query(cudf::strings_column_view{col},
                                                             cudf::strings_column_view{query});

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
  }
}

TEST_F(ParseURIPathTests, Simple)
{
  auto const col    = get_test_data(test_types::SIMPLE);
  auto const result = spark_rapids_jni::parse_uri_to_path(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper const expected({"/s/uri",
                                                     "",
                                                     "/to/a/cool/file",
                                                     "/path/to/file",
                                                     "/www.nvidia.com",
                                                     "",
                                                     "/network/path/to/file",
                                                     "nvidia.com",
                                                     "www.nvidia.com/s/uri"},
                                                    {1, 1, 1, 1, 1, 0, 1, 1, 1});

  cudf::test::print(result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(ParseURIPathTests, SparkEdges)
{
  auto const col    = get_test_data(test_types::SPARK_EDGES);
  auto const result = spark_rapids_jni::parse_uri_to_path(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper const expected(
    {"/https&",
     "//www.nvidia.com",
     "",
     "",
     "",
     "/absolute/path",
     "",
     "",
     "",
     "",
     "/absolute/path",
     "",
     "",
     "/q/This%20is%20a%20query",
     "",
     "",
     "",
     "",
     "",
     "/file;param",
     "",
     "",
     "",
     "",
     "",
     "",
     "",
     "",
     "",
     "",
     "",
     "",
     "",
     "",
     "/",
     "/",
     "/"},
    {1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1});

  cudf::test::print(result->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(ParseURIPathTests, IP6)
{
  auto const col    = get_test_data(test_types::IPv6);
  auto const result = spark_rapids_jni::parse_uri_to_path(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper const expected(
    {"", "", "", "", "", "", "", "/path/to/file", "", ""}, {1, 1, 1, 1, 1, 1, 1, 1, 0, 0});

  cudf::test::print(result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(ParseURIPathTests, IP4)
{
  auto const col    = get_test_data(test_types::IPv4);
  auto const result = spark_rapids_jni::parse_uri_to_path(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper const expected({"/", "/", "/", "/", "/", "/path/to/file"});

  cudf::test::print(result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(ParseURIPathTests, UTF8)
{
  auto const col    = get_test_data(test_types::UTF8);
  auto const result = spark_rapids_jni::parse_uri_to_path(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper const expected({"/%4EV%49%44%49%41", "", "/123", ""},
                                                    {1, 1, 1, 0});

  cudf::test::print(result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

// ANSI mode tests
struct ParseURIAnsiTests : public cudf::test::BaseFixture {};

TEST_F(ParseURIAnsiTests, BasicAnsiMode)
{
  // Test valid URLs
  cudf::test::strings_column_wrapper const valid_col(
    {"https://www.nvidia.com/path", "ftp://example.com"});

  // ANSI mode with valid URLs should work normally
  auto const result_valid =
    spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{valid_col}, true);
  cudf::test::strings_column_wrapper const expected_valid({"https", "ftp"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_valid, result_valid->view());

  // Non-ANSI mode should also work
  auto const result_non_ansi =
    spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{valid_col}, false);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_valid, result_non_ansi->view());

  // Test invalid URLs
  cudf::test::strings_column_wrapper const invalid_col(
    {"https://www.nvidia.com/path", "://completely-malformed"});

  // ANSI mode with invalid URLs should throw exception
  EXPECT_THROW(
    spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{invalid_col}, true),
    spark_rapids_jni::exception_with_row_index);

  // Non-ANSI mode should work without exception
  auto const result_invalid_non_ansi =
    spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{invalid_col}, false);
  cudf::test::strings_column_wrapper const expected_invalid({"https", ""}, {1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_invalid, result_invalid_non_ansi->view());
}

TEST_F(ParseURIAnsiTests, MissingQueryParameterAnsiMode)
{
  cudf::test::strings_column_wrapper input{
    "https://secure.payment.com/process?amount=100&currency=USD",  // has 'amount'
    "http://analytics.site.com/track?event=click&user=456",        // no 'amount'
    "ftp://backup.server.com/files/data.csv"                       // no query at all
  };

  // Non-ANSI mode should return NULLs for missing parameters
  {
    auto results =
      spark_rapids_jni::parse_uri_to_query(cudf::strings_column_view(input), std::string("amount"));
    cudf::test::strings_column_wrapper expected({"100", "", ""}, {1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *results);
  }

  // ANSI mode should ALSO return NULLs for missing parameters, not throw exceptions
  {
    auto results = spark_rapids_jni::parse_uri_to_query(
      cudf::strings_column_view(input), std::string("amount"), true);
    cudf::test::strings_column_wrapper expected({"100", "", ""}, {1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *results);
  }

  {
    auto results = spark_rapids_jni::parse_uri_to_query(cudf::strings_column_view(input),
                                                        std::string("nonexistent"));
    cudf::test::strings_column_wrapper expected({"", "", ""}, {0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *results);
  }

  // ANSI mode should still throw exception for invalid URLs
  cudf::test::strings_column_wrapper invalid_input{"://completely-malformed"};
  EXPECT_THROW(spark_rapids_jni::parse_uri_to_query(
                 cudf::strings_column_view(invalid_input), std::string("param"), true),
               spark_rapids_jni::exception_with_row_index);
}
