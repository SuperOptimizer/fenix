// io/sigv4.hpp — AWS Signature Version 4 request signing (first-party; no OpenSSL —
// own SHA-256/HMAC). Scope: what fenix needs — signed GETs against private S3 buckets
// (and later PUT for CAS uploads). Credentials from the environment
// (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN); region parsed from
// the virtual-hosted host (s3.<region>.amazonaws.com, default us-east-1). The signer is
// a PURE function of (method, host, path, query, timestamp, creds) so the AWS
// documented test vector pins it without any network.
#pragma once

#include "core/core.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::io::sigv4 {

// ---- SHA-256 (FIPS 180-4), byte-oriented, single-shot -------------------------------
namespace detail {

inline constexpr std::array<u32, 64> kK = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline u32 rotr(u32 x, int n) { return (x >> n) | (x << (32 - n)); }

inline std::array<u8, 32> sha256(std::span<const u8> msg) {
    std::array<u32, 8> h = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    // pad: msg || 0x80 || zeros || u64 bitlen, to a 64-byte multiple
    std::vector<u8> m(msg.begin(), msg.end());
    const u64 bitlen = static_cast<u64>(m.size()) * 8;
    m.push_back(0x80);
    while (m.size() % 64 != 56) m.push_back(0);
    for (int i = 7; i >= 0; --i) m.push_back(static_cast<u8>(bitlen >> (8 * i)));

    for (usize off = 0; off < m.size(); off += 64) {
        u32 w[64];
        for (int t = 0; t < 16; ++t)
            w[t] = (static_cast<u32>(m[off + 4 * static_cast<usize>(t)]) << 24) |
                   (static_cast<u32>(m[off + 4 * static_cast<usize>(t) + 1]) << 16) |
                   (static_cast<u32>(m[off + 4 * static_cast<usize>(t) + 2]) << 8) |
                   static_cast<u32>(m[off + 4 * static_cast<usize>(t) + 3]);
        for (int t = 16; t < 64; ++t) {
            const u32 s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
            const u32 s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16] + s0 + w[t - 7] + s1;
        }
        u32 a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int t = 0; t < 64; ++t) {
            const u32 S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const u32 ch = (e & f) ^ (~e & g);
            const u32 t1 = hh + S1 + ch + kK[static_cast<usize>(t)] + w[t];
            const u32 S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const u32 maj = (a & b) ^ (a & c) ^ (b & c);
            const u32 t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    std::array<u8, 32> out{};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j) out[static_cast<usize>(4 * i + j)] = static_cast<u8>(h[static_cast<usize>(i)] >> (8 * (3 - j)));
    return out;
}

inline std::array<u8, 32> hmac_sha256(std::span<const u8> key, std::span<const u8> msg) {
    std::array<u8, 64> k{};
    if (key.size() > 64) {
        const auto kh = sha256(key);
        std::memcpy(k.data(), kh.data(), 32);
    } else {
        std::memcpy(k.data(), key.data(), key.size());
    }
    std::vector<u8> inner;
    inner.reserve(64 + msg.size());
    for (int i = 0; i < 64; ++i) inner.push_back(static_cast<u8>(k[static_cast<usize>(i)] ^ 0x36));
    inner.insert(inner.end(), msg.begin(), msg.end());
    const auto ih = sha256(inner);
    std::vector<u8> outer;
    outer.reserve(64 + 32);
    for (int i = 0; i < 64; ++i) outer.push_back(static_cast<u8>(k[static_cast<usize>(i)] ^ 0x5c));
    outer.insert(outer.end(), ih.begin(), ih.end());
    return sha256(outer);
}

inline std::string hex(std::span<const u8> b) {
    static constexpr char kH[] = "0123456789abcdef";
    std::string s;
    s.reserve(b.size() * 2);
    for (u8 c : b) {
        s.push_back(kH[c >> 4]);
        s.push_back(kH[c & 15]);
    }
    return s;
}

inline std::span<const u8> bytes(std::string_view s) {
    return {reinterpret_cast<const u8*>(s.data()), s.size()};
}

}  // namespace detail

struct Credentials {
    std::string access_key, secret_key, session_token;  // token optional
    [[nodiscard]] bool valid() const { return !access_key.empty() && !secret_key.empty(); }
};

inline Credentials env_credentials() {
    Credentials c;
    if (const char* a = std::getenv("AWS_ACCESS_KEY_ID")) c.access_key = a;
    if (const char* s = std::getenv("AWS_SECRET_ACCESS_KEY")) c.secret_key = s;
    if (const char* t = std::getenv("AWS_SESSION_TOKEN")) c.session_token = t;
    return c;
}

// region from a virtual-hosted S3 host: bucket.s3.<region>.amazonaws.com; plain
// s3.amazonaws.com (or unknown) -> us-east-1
inline std::string region_from_host(std::string_view host) {
    const auto s3 = host.find(".s3.");
    if (s3 == std::string_view::npos) return "us-east-1";
    const auto rest = host.substr(s3 + 4);
    const auto dot = rest.find('.');
    if (dot == std::string_view::npos || rest.substr(0, dot) == "amazonaws") return "us-east-1";
    return std::string(rest.substr(0, dot));
}

// The headers a signed GET must send, given an already-encoded path+query.
// amz_date: "YYYYMMDDTHHMMSSZ". Payload is empty (GET): hash = SHA256("").
struct SignedHeaders {
    std::string authorization, amz_date, amz_token, amz_content_sha;
};

inline SignedHeaders sign_get(std::string_view host, std::string_view path, std::string_view query,
                              std::string_view amz_date, const Credentials& cred,
                              std::string_view service = "s3", std::string_view region_override = {}) {
    using namespace detail;
    static constexpr std::string_view kEmptySha =
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const std::string region = region_override.empty() ? region_from_host(host) : std::string(region_override);
    const std::string date(amz_date.substr(0, 8));

    std::string canon_headers = "host:" + std::string(host) + "\n" + "x-amz-content-sha256:" +
                                std::string(kEmptySha) + "\n" + "x-amz-date:" + std::string(amz_date) + "\n";
    std::string signed_list = "host;x-amz-content-sha256;x-amz-date";
    if (!cred.session_token.empty()) {
        canon_headers += "x-amz-security-token:" + cred.session_token + "\n";
        signed_list += ";x-amz-security-token";
    }
    const std::string canonical = "GET\n" + std::string(path.empty() ? "/" : path) + "\n" + std::string(query) +
                                  "\n" + canon_headers + "\n" + signed_list + "\n" + std::string(kEmptySha);
    const std::string scope = date + "/" + region + "/" + std::string(service) + "/aws4_request";
    const std::string to_sign = "AWS4-HMAC-SHA256\n" + std::string(amz_date) + "\n" + scope + "\n" +
                                hex(sha256(bytes(canonical)));
    const std::string k0 = "AWS4" + cred.secret_key;
    auto kd = hmac_sha256(bytes(k0), bytes(date));
    kd = hmac_sha256(kd, bytes(region));
    kd = hmac_sha256(kd, bytes(service));
    kd = hmac_sha256(kd, bytes(std::string_view("aws4_request")));
    const std::string sig = hex(hmac_sha256(kd, bytes(to_sign)));

    SignedHeaders out;
    out.amz_date = std::string(amz_date);
    out.amz_content_sha = std::string(kEmptySha);
    out.amz_token = cred.session_token;
    out.authorization = "AWS4-HMAC-SHA256 Credential=" + cred.access_key + "/" + scope + ", SignedHeaders=" +
                        signed_list + ", Signature=" + sig;
    return out;
}

}  // namespace fenix::io::sigv4
