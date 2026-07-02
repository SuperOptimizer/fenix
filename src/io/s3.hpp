// io/s3.hpp — minimal S3/HTTP GET client over libcurl. A fresh C++ rewrite of the design in
// SuperOptimizer/libs3 (studied, not copied): thread-local reused easy handle (keeps the
// connection pool + TLS session warm across requests on a thread), a low-speed *stall*
// watchdog instead of a hard transfer timeout (so a slow-but-progressing big GET still
// finishes), exponential backoff with jitter, and the key invariant from io/CLAUDE.md:
//   404 -> absent (air), distinct from a hard fetch failure (retry then error, NEVER silent air).
// Read-only + anonymous (open-access buckets); SigV4/auth can layer on later. Guarded by
// FENIX_HAVE_CURL so the core still builds if libcurl is absent.
#pragma once

#include "core/core.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#ifdef FENIX_HAVE_CURL
#include <cctype>
#include <chrono>
#include <curl/curl.h>
#include <mutex>
#include <thread>
#endif

namespace fenix::io {

inline bool is_remote(std::string_view s) {
    return s.starts_with("http://") || s.starts_with("https://") || s.starts_with("s3://");
}

// s3://bucket/key -> https://bucket.s3.amazonaws.com/key ; http(s):// passes through unchanged.
inline std::string to_https(std::string_view url) {
    if (url.starts_with("s3://")) {
        const std::string_view rest = url.substr(5);
        const auto slash = rest.find('/');
        const std::string_view bucket = rest.substr(0, slash);
        const std::string_view key = slash == std::string_view::npos ? std::string_view{} : rest.substr(slash + 1);
        return "https://" + std::string(bucket) + ".s3.amazonaws.com/" + std::string(key);
    }
    return std::string(url);
}

struct HttpConfig {
    long connect_timeout_s = 10;
    long transfer_stall_s = 30;  // abort only if throughput stays < 1 KB/s this long
    int max_retries = 4;
};

// GET a URL. Returns the bytes on 2xx, std::nullopt on 404 (absent), or a hard Error after
// exhausting retries (transport failure / 5xx / 403 / 429). A transient failure NEVER returns
// nullopt — absent and fetch-failed are kept distinct (io/CLAUDE.md).
inline Expected<std::optional<std::vector<u8>>> http_get(const std::string& url,
                                                         const HttpConfig& cfg = {});

#ifdef FENIX_HAVE_CURL
namespace detail {
inline void curl_global_once() {
    static std::once_flag f;
    std::call_once(f, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}
inline std::size_t write_vec(char* ptr, std::size_t sz, std::size_t nm, void* ud) {
    const std::size_t n = sz * nm;
    auto* v = static_cast<std::vector<u8>*>(ud);
    v->insert(v->end(), reinterpret_cast<u8*>(ptr), reinterpret_cast<u8*>(ptr) + n);
    return n;
}
// Reserve the body vector from Content-Length before the first write callback (one alloc instead of
// log2(N) doubling reallocs for a multi-MB chunk).
inline std::size_t header_reserve(char* ptr, std::size_t sz, std::size_t nm, void* ud) {
    const std::size_t n = sz * nm;
    std::string_view line(ptr, n);
    constexpr std::string_view kCL = "content-length:";
    if (line.size() > kCL.size()) {
        std::string lower(line.substr(0, kCL.size()));
        for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (lower == kCL) {
            std::string_view val = line.substr(kCL.size());
            long len = 0;
            for (char c : val) { if (c >= '0' && c <= '9') len = len * 10 + (c - '0'); else if (len) break; }
            if (len > 0 && len < (1L << 30)) static_cast<std::vector<u8>*>(ud)->reserve(static_cast<std::size_t>(len));
        }
    }
    return n;
}
// Thread-local reused handle (connection + TLS reuse). Lives for the thread's lifetime. Static options
// (the ones that never change per request) are set ONCE on first use — curl_easy_reset would wipe them
// every call, so we do NOT reset; only the per-request URL + write targets change in http_get.
inline CURL* thread_handle() {
    thread_local CURL* h = [] {
        CURL* c = curl_easy_init();
        if (c) {
            curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);  // thread-safe timeouts
            curl_easy_setopt(c, CURLOPT_TCP_KEEPALIVE, 1L);
            curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(c, CURLOPT_MAXREDIRS, 10L);
            curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, write_vec);
            curl_easy_setopt(c, CURLOPT_HEADERFUNCTION, header_reserve);
            curl_easy_setopt(c, CURLOPT_USERAGENT, "fenix/io");
            curl_easy_setopt(c, CURLOPT_BUFFERSIZE, 262144L);  // 256 KB recv buffer (fewer syscalls on MB chunks)
            curl_easy_setopt(c, CURLOPT_ACCEPT_ENCODING, "");  // advertise identity/gzip; zarr raw is uncompressed
        }
        return c;
    }();
    return h;
}
inline void backoff(int attempt) {
    const long ms = 200L * (1L << attempt) + (attempt * 53) % 101;  // exp backoff + small jitter
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}
}  // namespace detail

inline Expected<std::optional<std::vector<u8>>> http_get(const std::string& url, const HttpConfig& cfg) {
    detail::curl_global_once();
    const std::string resolved = to_https(url);
    CURL* h = detail::thread_handle();
    if (!h) return err(Errc::io_error, "curl init failed");

    // Per-request options only — the static ones (keepalive, write fn, buffer size, ...) are set once in
    // thread_handle() and MUST survive across requests for connection/TLS reuse, so we do NOT reset here.
    curl_easy_setopt(h, CURLOPT_URL, resolved.c_str());
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, cfg.connect_timeout_s);
    curl_easy_setopt(h, CURLOPT_LOW_SPEED_LIMIT, 1024L);                // 1 KB/s ...
    curl_easy_setopt(h, CURLOPT_LOW_SPEED_TIME, cfg.transfer_stall_s);  // ... for this long => stalled

    for (int attempt = 0; attempt <= cfg.max_retries; ++attempt) {
        std::vector<u8> body;
        curl_easy_setopt(h, CURLOPT_WRITEDATA, &body);
        curl_easy_setopt(h, CURLOPT_HEADERDATA, &body);

        const CURLcode rc = curl_easy_perform(h);
        if (rc == CURLE_OK) {
            long status = 0;
            curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
            if (status == 200 || status == 206) return std::optional<std::vector<u8>>(std::move(body));
            if (status == 404) return std::optional<std::vector<u8>>(std::nullopt);  // absent (air)
            const bool retryable = status >= 500 || status == 403 || status == 429;
            if (retryable && attempt < cfg.max_retries) { detail::backoff(attempt); continue; }
            return err(Errc::fetch_failed, "GET " + resolved + " -> HTTP " + std::to_string(status));
        }
        if (attempt < cfg.max_retries) { detail::backoff(attempt); continue; }
        return err(Errc::fetch_failed, "GET " + resolved + " curl: " + curl_easy_strerror(rc));
    }
    return err(Errc::fetch_failed, "GET " + resolved + " exhausted retries");
}
#else
inline Expected<std::optional<std::vector<u8>>> http_get(const std::string&, const HttpConfig&) {
    return err(Errc::unsupported, "fenix built without libcurl (set FENIX_HAVE_CURL)");
}
#endif

}  // namespace fenix::io
