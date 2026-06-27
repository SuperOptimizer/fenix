// test_core.cpp — sample test (per-test-file binary). Define FENIX_TEST_MAIN for main().
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"

using namespace fenix;

TEST(error_carries_code_and_message) {
    Expected<int> e = err(Errc::not_found, "nope");
    REQUIRE(!e.has_value());
    CHECK(e.error().code == Errc::not_found);
    CHECK(e.error().message == "nope");
}

TEST(stage_registry_roundtrip) {
    int before = static_cast<int>(registry().size());
    register_stage(Stage{"__test_stage", "help", [](std::span<const std::string_view>,
                                                    Context&) -> Expected<int> {
                             return 0;
                         }});
    CHECK(static_cast<int>(registry().size()) == before + 1);
    REQUIRE(find_stage("__test_stage") != nullptr);
    CHECK(find_stage("does_not_exist") == nullptr);
}

TEST(version_is_set) { CHECK(!version.empty()); }
