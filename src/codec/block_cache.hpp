// codec/block_cache.hpp — a byte-budgeted, sharded SIEVE cache of DECODED 16³ chunks (one std::vector<f32>
// per 16³ DCT block). The 64³ tile is the atomic DECODE unit (its 64 blocks share rANS tables), but the
// CACHE granularity is 16³: a tile decode populates all 64 of its 16³ chunks, so the decode amortizes
// across the tile while eviction stays 16³-fine. shared_ptr values give natural pinning — an evicted chunk
// stays alive as long as a reader still holds its Ref (no manual refcount/pin). SIEVE (NSDI'24): simpler
// than LRU, scan-resistant, low-contention (a hit only sets a bit; the "hand" advances only on eviction).
// Sharded by key to cut lock contention under parallel_for. See ADR 0006 + docs/design/fxvol-v4-layout.md §7.
#pragma once

#include "core/types.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace fenix::codec {

class BlockCache {
public:
    using Block = std::vector<f32>;
    using Ref = std::shared_ptr<const Block>;

    explicit BlockCache(u64 budget_bytes, int shards = 16) : shards_(static_cast<usize>(shards < 1 ? 1 : shards)) {
        const u64 per = budget_bytes / shards_.size();
        for (auto& s : shards_) s.budget = per ? per : 1;
    }

    // Look up a decoded 16³ chunk by key. Hit ⇒ pin (shared_ptr copy) + mark visited; miss ⇒ nullptr.
    Ref get(u64 key) {
        Shard& s = shard_(key);
        std::lock_guard<std::mutex> lk(s.mu);
        auto it = s.map.find(key);
        if (it == s.map.end()) {
            misses_.fetch_add(1, std::memory_order_relaxed);
            return nullptr;
        }
        it->second->visited = true;
        hits_.fetch_add(1, std::memory_order_relaxed);
        return it->second->val;
    }

    // Insert a decoded 16³ chunk, evicting (SIEVE) until within the shard byte budget. No-op if present.
    void put(u64 key, Ref val) {
        if (!val) return;
        const u64 b = static_cast<u64>(val->size()) * sizeof(f32);
        Shard& s = shard_(key);
        std::lock_guard<std::mutex> lk(s.mu);
        if (s.map.contains(key)) return;
        while (s.bytes + b > s.budget && s.tail) evict_(s);
        Node* n = new Node{key, std::move(val), b, false, nullptr, nullptr};
        push_head_(s, n);
        s.map.emplace(key, n);
        s.bytes += b;
        ++s.count;
    }

    [[nodiscard]] u64 hits() const { return hits_.load(std::memory_order_relaxed); }
    [[nodiscard]] u64 misses() const { return misses_.load(std::memory_order_relaxed); }
    [[nodiscard]] u64 bytes() const {
        u64 t = 0;
        for (auto& s : shards_) { std::lock_guard<std::mutex> lk(s.mu); t += s.bytes; }
        return t;
    }

private:
    struct Node {
        u64 key;
        Ref val;
        u64 bytes;
        bool visited;
        Node* older;  // toward the tail (oldest)
        Node* newer;  // toward the head (newest)
    };
    struct Shard {
        mutable std::mutex mu;  // mutable so bytes() (const) can lock
        std::unordered_map<u64, Node*> map;
        Node* head = nullptr;  // newest (newer == null)
        Node* tail = nullptr;  // oldest (older == null)
        Node* hand = nullptr;  // SIEVE eviction cursor, walks oldest→newest
        u64 bytes = 0;
        u64 budget = 0;
        usize count = 0;
        ~Shard() { for (auto& [k, n] : map) delete n; }
    };

    Shard& shard_(u64 key) { return shards_[(key ^ (key >> 33)) % shards_.size()]; }

    static void push_head_(Shard& s, Node* n) {
        n->newer = nullptr;
        n->older = s.head;
        if (s.head) s.head->newer = n; else s.tail = n;
        s.head = n;
    }
    static void unlink_(Shard& s, Node* n) {
        if (n->older) n->older->newer = n->newer; else s.tail = n->newer;  // n was the tail
        if (n->newer) n->newer->older = n->older; else s.head = n->older;  // n was the head
    }
    // SIEVE: from the hand (oldest→newest), skip+clear visited objects; evict the first unvisited one.
    static void evict_(Shard& s) {
        Node* h = s.hand ? s.hand : s.tail;
        for (usize g = 0; g < 2 * s.count + 2 && h; ++g) {
            if (h->visited) { h->visited = false; h = h->newer ? h->newer : s.tail; } else break;
        }
        if (!h) h = s.tail;
        if (!h) return;
        s.hand = h->newer;  // next eviction resumes toward the head (null ⇒ wrap to tail)
        unlink_(s, h);
        s.map.erase(h->key);
        s.bytes -= h->bytes;
        --s.count;
        delete h;
    }

    std::vector<Shard> shards_;
    std::atomic<u64> hits_{0};
    std::atomic<u64> misses_{0};
};

}  // namespace fenix::codec
