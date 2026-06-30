// test_archive.cpp — .fxvol create/write/close/open/read roundtrip + coverage tri-state.
#define FENIX_TEST_MAIN
#include "codec/archive.hpp"
#include "core/core.hpp"
#include "core/test.hpp"

#include <cmath>
#include <filesystem>
#include <vector>

using namespace fenix;
using namespace fenix::codec;

static std::vector<f32> structured_chunk() {
    const s64 s = fxvol_chunk_side;
    std::vector<f32> b(static_cast<usize>(s * s * s));
    Pcg32 rng{5};
    const Index3 st{s * s, s, 1};
    for (s64 z = 0; z < s; ++z)
        for (s64 y = 0; y < s; ++y)
            for (s64 x = 0; x < s; ++x)
                b[static_cast<usize>(z * st.z + y * st.y + x * st.x)] =
                    40.0f + 80.0f * std::exp(-0.1f * static_cast<f32>((x - 30) * (x - 30))) +
                    rng.next_f32();
    return b;
}

TEST(archive_roundtrip_and_coverage) {
    auto tmp = std::filesystem::temp_directory_path() / "fenix_test.fxvol";
    const std::string path = tmp.string();
    std::filesystem::remove(path);

    Extent3 dims{64, 128, 64};  // chunk extent: z=1, y=2, x=1
    auto data = structured_chunk();

    {
        auto a = VolumeArchive::create(path, dims, {.q = 4.0f});
        REQUIRE(a.has_value());
        REQUIRE(a->write_chunk({0, 0, 0}, data).has_value());
        // second chunk left all-zero -> recorded ZERO
        std::vector<f32> air(data.size(), 0.0f);
        REQUIRE(a->write_chunk({0, 1, 0}, air).has_value());
        REQUIRE(a->close().has_value());
    }

    auto a = VolumeArchive::open(path);
    REQUIRE(a.has_value());
    CHECK(a->dims() == dims);
    CHECK(a->chunk_extent() == ChunkCoord{1, 2, 1});
    CHECK(a->coverage({0, 0, 0}) == Coverage::Real);
    CHECK(a->coverage({0, 1, 0}) == Coverage::Zero);
    CHECK(a->coverage({0, 5, 0}) == Coverage::Absent);

    auto got = a->read_chunk({0, 0, 0});
    REQUIRE(got.has_value());
    REQUIRE(got->size() == data.size());
    f32 max_err = 0;
    for (usize i = 0; i < data.size(); ++i) max_err = std::max(max_err, std::abs((*got)[i] - data[i]));
    CHECK(max_err < 20.0f);  // q=4 lossy reconstruction

    auto air = a->read_chunk({0, 1, 0});
    REQUIRE(air.has_value());
    CHECK((*air)[123] == 0.0f);  // ZERO chunk -> zeros
    auto absent = a->read_chunk({0, 5, 0});
    REQUIRE(absent.has_value());
    CHECK((*absent)[0] == 0.0f);  // ABSENT -> fill

    std::filesystem::remove(path);
}
