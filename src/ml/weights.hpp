// ml/weights.hpp — reader for the hand-rolled `.fxweights` flat file produced by
// tools/ml-export/convert_weights.py. Layout (little-endian):
//   "FXWT" | u32 version | u32 count | u32 reserved
//   count × { u16 name_len | name | u8 dtype | u8 ndim | u64 nbytes | u64 data_off | i64 shape[ndim] }
//   data blobs at their (64-aligned) offsets.
// dtype codes: 0=f32 1=f16 2=bf16 3=f64 4=i64 5=i32 6=u8 7=i16 8=bool.
//
// Only included under FENIX_ML (it returns torch tensors). We mmap the file and wrap each blob
// with torch::from_blob, then clone onto the target device so the mapping can be released.
#pragma once

#include "core/core.hpp"
#include "ml/torch_env.hpp"

#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

namespace fenix::ml {

using WeightMap = std::unordered_map<std::string, torch::Tensor>;

namespace detail {
inline torch::Dtype fxw_dtype(std::uint8_t code) {
    switch (code) {
        case 0: return torch::kFloat32;
        case 1: return torch::kFloat16;
        case 2: return torch::kBFloat16;
        case 3: return torch::kFloat64;
        case 4: return torch::kInt64;
        case 5: return torch::kInt32;
        case 6: return torch::kUInt8;
        case 7: return torch::kInt16;
        case 8: return torch::kBool;
        default: return torch::kFloat32;
    }
}
}  // namespace detail

// Load all tensors from a .fxweights file onto `device` (CPU clones, then .to(device)).
inline Expected<WeightMap> load_fxweights(const std::string& path,
                                          torch::Device device = torch::kCPU) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return fenix::err(Errc::io_error, "fxweights: cannot open " + path);
    struct stat st {};
    if (::fstat(fd, &st) != 0) { ::close(fd); return fenix::err(Errc::io_error, "fxweights: fstat failed"); }
    const auto sz = static_cast<std::size_t>(st.st_size);
    void* base = ::mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) return fenix::err(Errc::io_error, "fxweights: mmap failed");

    const auto* p = static_cast<const std::uint8_t*>(base);
    auto rd = [&](std::size_t off, auto& out) { std::memcpy(&out, p + off, sizeof(out)); };

    WeightMap out;
    auto fail = [&](const std::string& m) { ::munmap(base, sz); return fenix::err(Errc::decode_error, m); };

    if (sz < 16 || std::memcmp(p, "FXWT", 4) != 0) return fail("fxweights: bad magic");
    std::uint32_t version = 0, count = 0;
    rd(4, version); rd(8, count);
    if (version != 1) return fail("fxweights: unsupported version");

    std::size_t cur = 16;
    out.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        std::uint16_t nlen = 0; rd(cur, nlen); cur += 2;
        std::string name(reinterpret_cast<const char*>(p + cur), nlen); cur += nlen;
        std::uint8_t dtype = 0, ndim = 0; rd(cur, dtype); rd(cur + 1, ndim); cur += 2;
        std::uint64_t nbytes = 0, data_off = 0; rd(cur, nbytes); rd(cur + 8, data_off); cur += 16;
        std::vector<std::int64_t> shape(ndim);
        for (std::uint8_t d = 0; d < ndim; ++d) { rd(cur, shape[d]); cur += 8; }
        if (data_off + nbytes > sz) return fail("fxweights: blob out of range for " + name);
        auto opts = torch::TensorOptions().dtype(detail::fxw_dtype(dtype));
        // from_blob over the mmap, then clone to own the memory before munmap, then to(device).
        torch::Tensor t = torch::from_blob(const_cast<std::uint8_t*>(p) + data_off, shape, opts).clone();
        out.emplace(std::move(name), t.to(device));
    }
    ::munmap(base, sz);
    return out;
}

// Copy weights into a module by exact parameter name. Returns #matched; sets `missing` to the
// names the module expects but the file lacks (a non-empty list means the arch/file disagree).
inline int load_into(torch::nn::Module& mod, const WeightMap& w,
                     std::vector<std::string>* missing = nullptr) {
    torch::NoGradGuard ng;
    int matched = 0;
    for (auto& kv : mod.named_parameters()) {
        auto it = w.find(kv.key());
        if (it == w.end()) { if (missing) missing->push_back(kv.key()); continue; }
        kv.value().copy_(it->second.to(kv.value().dtype()).reshape(kv.value().sizes()));
        ++matched;
    }
    for (auto& kv : mod.named_buffers()) {
        auto it = w.find(kv.key());
        if (it != w.end()) kv.value().copy_(it->second.to(kv.value().dtype()).reshape(kv.value().sizes()));
    }
    return matched;
}

}  // namespace fenix::ml
