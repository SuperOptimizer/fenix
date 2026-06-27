// core/arena.hpp — a bump/region allocator for hot loops + per-thread scratch. Avoids
// per-iteration malloc churn (a recurring predecessor hotspot). Trivially-destructible
// payloads only (no destructors are run); reset reclaims everything at once.
#pragma once

#include "core/types.hpp"

#include <bit>
#include <cstdlib>
#include <memory>
#include <new>

namespace fenix {

class Arena {
public:
    explicit Arena(usize capacity_bytes)
        : cap_(capacity_bytes),
          base_(static_cast<u8*>(::operator new(capacity_bytes, std::align_val_t(64)))) {}

    ~Arena() { ::operator delete(base_, std::align_val_t(64)); }

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // Allocate raw aligned bytes; returns nullptr if the arena is exhausted.
    [[nodiscard]] void* alloc(usize bytes, usize align = 16) {
        usize p = (offset_ + (align - 1)) & ~(align - 1);
        if (p + bytes > cap_) return nullptr;
        offset_ = p + bytes;
        return base_ + p;
    }

    // Typed array (uninitialized). Trivially-copyable T only.
    template <class T>
        requires std::is_trivially_copyable_v<T>
    [[nodiscard]] T* alloc_n(usize n) {
        return static_cast<T*>(alloc(n * sizeof(T), alignof(T) < 16 ? 16 : alignof(T)));
    }

    void reset() { offset_ = 0; }
    [[nodiscard]] usize used() const { return offset_; }
    [[nodiscard]] usize capacity() const { return cap_; }

    // RAII marker: restores the bump offset when it leaves scope (nested scratch regions).
    class Scope {
    public:
        explicit Scope(Arena& a) : arena_(a), mark_(a.offset_) {}
        ~Scope() { arena_.offset_ = mark_; }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

    private:
        Arena& arena_;
        usize mark_;
    };

private:
    usize cap_ = 0;
    usize offset_ = 0;
    u8* base_ = nullptr;
};

}  // namespace fenix
