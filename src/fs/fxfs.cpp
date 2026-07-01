// src/fs/fxfs.cpp — fxfs: a FUSE filesystem that exposes a .fxvol as a flat, uncompressed ZYX array.
//
// Mount a .fxvol and read it as if it were one giant raw volume: `…/volume.raw` is a regular file of size
// Z·Y·X·elemsize whose bytes are the C-order (x-fastest) ZYX voxels. A read (or an mmap page fault routed
// through FUSE) decodes the covering 64³ tile(s) on demand via the archive's decoded-16³-chunk SIEVE cache
// (codec/block_cache.hpp) — no manual decode/cache plumbing in the consumer. The kernel page cache then
// caches the decoded pages. See ADR 0007 (FUSE over an in-kernel module; lossy-writeback policy).
//
// v1 is READ-ONLY (the safe, lossless, common case for inference/visualization/analysis). Writable mounts
// (recompress-on-flush via a dirty-tile overlay) are the next step per ADR 0007. The archive store is
// dtype-agnostic, so the element type is carried in via --dtype (default u8 — these CT scrolls are u8).
//
// Firewalled behind -DFENIX_FS=ON (links libfuse3); not part of the header-only single-TU build.
#define FUSE_USE_VERSION 31
#include <fuse.h>

#include "codec/archive.hpp"
#include "core/core.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {
using namespace fenix;

// One mounted archive (single global — fxfs mounts exactly one .fxvol per process).
struct State {
    codec::VolumeArchive ar;
    Extent3 dims{};   // LOD0 dims (ZYX)
    u64 esz = 1;      // element size in bytes (u8 = 1)
    u64 raw_size = 0;  // Z·Y·X·esz — the size of volume.raw
    std::string meta;  // meta.toml contents
};
State* g = nullptr;

constexpr const char* kRaw = "/volume.raw";
constexpr const char* kMeta = "/meta.toml";

int fx_getattr(const char* path, struct stat* st, struct fuse_file_info*) {
    std::memset(st, 0, sizeof(*st));
    if (std::strcmp(path, "/") == 0) {
        st->st_mode = S_IFDIR | 0555;
        st->st_nlink = 2;
        return 0;
    }
    if (std::strcmp(path, kRaw) == 0) {
        st->st_mode = S_IFREG | 0444;
        st->st_nlink = 1;
        st->st_size = static_cast<off_t>(g->raw_size);
        return 0;
    }
    if (std::strcmp(path, kMeta) == 0) {
        st->st_mode = S_IFREG | 0444;
        st->st_nlink = 1;
        st->st_size = static_cast<off_t>(g->meta.size());
        return 0;
    }
    return -ENOENT;
}

int fx_readdir(const char* path, void* buf, fuse_fill_dir_t filler, off_t, struct fuse_file_info*,
               enum fuse_readdir_flags) {
    if (std::strcmp(path, "/") != 0) return -ENOENT;
    filler(buf, ".", nullptr, 0, static_cast<fuse_fill_dir_flags>(0));
    filler(buf, "..", nullptr, 0, static_cast<fuse_fill_dir_flags>(0));
    filler(buf, "volume.raw", nullptr, 0, static_cast<fuse_fill_dir_flags>(0));
    filler(buf, "meta.toml", nullptr, 0, static_cast<fuse_fill_dir_flags>(0));
    return 0;
}

int fx_open(const char* path, struct fuse_file_info* fi) {
    if (std::strcmp(path, kRaw) != 0 && std::strcmp(path, kMeta) != 0) return -ENOENT;
    if ((fi->flags & O_ACCMODE) != O_RDONLY) return -EACCES;  // read-only mount (ADR 0007 v1)
    return 0;
}

int fx_read(const char* path, char* buf, size_t size, off_t offset, struct fuse_file_info*) {
    if (std::strcmp(path, kMeta) == 0) {
        if (static_cast<u64>(offset) >= g->meta.size()) return 0;
        const size_t n = std::min(size, g->meta.size() - static_cast<size_t>(offset));
        std::memcpy(buf, g->meta.data() + offset, n);
        return static_cast<int>(n);
    }
    if (std::strcmp(path, kRaw) != 0) return -ENOENT;
    if (static_cast<u64>(offset) >= g->raw_size) return 0;
    const u64 end = std::min<u64>(g->raw_size, static_cast<u64>(offset) + size);
    const u64 n = end - static_cast<u64>(offset);

    // esz==1 (u8): a byte index IS a voxel linear index. Walk the requested span; for each voxel decode its
    // (z,y,x) from the C-order linear index and read it from the covering decoded 16³ block. The "current
    // block" is cached across the loop so a contiguous x-run (16 voxels) costs one block16 call, and the
    // SIEVE cache amortizes the 64³ tile decode across all 64 of its 16³ blocks. A decode/fetch error is
    // EIO — never silently served as air.
    const s64 Y = g->dims.y, X = g->dims.x;
    for (u64 i = 0; i < n; ++i) {
        const u64 lin = static_cast<u64>(offset) + i;
        const s64 x = static_cast<s64>(lin % static_cast<u64>(X));
        const s64 y = static_cast<s64>((lin / static_cast<u64>(X)) % static_cast<u64>(Y));
        const s64 z = static_cast<s64>(lin / (static_cast<u64>(X) * static_cast<u64>(Y)));
        // sample_f32 widens the cached NATIVE-dtype voxel ephemerally; the block cache amortizes the 64³ tile
        // decode across its 64 blocks (SIEVE, byte-budgeted). A decode/fetch error is EIO, never silent air.
        auto v = g->ar.sample_f32(0, z, y, x);
        if (!v) return -EIO;
        const s32 q = static_cast<s32>(std::lround(*v));
        buf[i] = static_cast<char>(static_cast<u8>(q < 0 ? 0 : q > 255 ? 255 : q));
    }
    return static_cast<int>(n);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <archive.fxvol> <mountpoint> [fuse opts]\n"
                     "  mounts the .fxvol read-only; read <mountpoint>/volume.raw as a flat ZYX u8 array.\n",
                     argv[0]);
        return 2;
    }
    const std::string apath = argv[1];
    auto ar = fenix::codec::VolumeArchive::open(apath, false);
    if (!ar) {
        std::fprintf(stderr, "fxfs: cannot open %s: %s\n", apath.c_str(), ar.error().message.c_str());
        return 1;
    }
    static State st{std::move(*ar), {}, 1, 0, {}};
    st.dims = st.ar.dims();
    st.ar.reserve_cache(1ull << 30, 16);  // 1 GiB decoded-16³-block cache (bounded; the rest stays on disk)
    st.esz = 1;                            // u8 archives (dtype-agnostic store; these scrolls are u8)
    st.raw_size = static_cast<fenix::u64>(st.dims.z) * static_cast<fenix::u64>(st.dims.y) *
                  static_cast<fenix::u64>(st.dims.x) * st.esz;
    char mbuf[512];
    std::snprintf(mbuf, sizeof mbuf,
                  "[fxvol]\ndims_zyx = [%lld, %lld, %lld]\ndtype = \"u8\"\nq = %.2f\nlods = %u\n"
                  "raw_bytes = %llu\n",
                  static_cast<long long>(st.dims.z), static_cast<long long>(st.dims.y),
                  static_cast<long long>(st.dims.x), static_cast<double>(st.ar.params().q), st.ar.nlods(),
                  static_cast<unsigned long long>(st.raw_size));
    st.meta = mbuf;
    g = &st;

    // Hand fuse_main argv[0] + the mountpoint + any trailing fuse options (drop our archive-path arg).
    std::vector<char*> fa;
    fa.push_back(argv[0]);
    for (int i = 2; i < argc; ++i) fa.push_back(argv[i]);

    static struct fuse_operations ops {};
    ops.getattr = fx_getattr;
    ops.readdir = fx_readdir;
    ops.open = fx_open;
    ops.read = fx_read;
    return fuse_main(static_cast<int>(fa.size()), fa.data(), &ops, nullptr);
}
