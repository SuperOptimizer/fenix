// test_config.cpp — minimal TOML config reader.
#define FENIX_TEST_MAIN
#include "core/config.hpp"
#include "core/test.hpp"

#include <cmath>

using namespace fenix;

TEST(config_parses_sections_and_types) {
    const char* toml = R"(
# fenix recipe
name = "unroll-demo"
threads = 8

[winding]
pitch = 7.5
enabled = true
sigmas = [1.0, 2.0, 4.0]

[render]
samp = 4
mode = "mean"   # trailing comment
)";
    auto c = Config::parse(toml);
    REQUIRE(c.has_value());
    CHECK(c->get_string("name") == std::optional<std::string>("unroll-demo"));
    CHECK(c->get_int("threads") == std::optional<s64>(8));
    CHECK(std::abs(*c->get_float("winding.pitch") - 7.5) < 1e-9);
    CHECK(c->get_bool("winding.enabled") == std::optional<bool>(true));
    CHECK(c->get_string("render.mode") == std::optional<std::string>("mean"));
    CHECK(c->get_int("render.samp") == std::optional<s64>(4));

    auto arr = c->get_array("winding.sigmas");
    REQUIRE(arr.size() == 3);
    CHECK(arr[2] == "4.0");
}

TEST(config_missing_keys_are_nullopt) {
    auto c = Config::parse("a = 1\n");
    REQUIRE(c.has_value());
    CHECK(!c->get_int("nope").has_value());
    CHECK(!c->get_string("also.nope").has_value());
    CHECK(c->has("a"));
}

TEST(config_bad_line_errors) {
    auto c = Config::parse("[sec]\nthis line has no equals\n");
    REQUIRE(!c.has_value());
    CHECK(c.error().code == Errc::decode_error);
}
