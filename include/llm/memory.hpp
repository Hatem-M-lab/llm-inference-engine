// include/llm/memory.hpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#if defined(_WIN32)
  #include <malloc.h>   // _aligned_malloc / _aligned_free
#endif
#include "llm/common.hpp"

namespace llm {

inline bool is_power_of_two(size_t x) { return x != 0 && (x & (x - 1)) == 0; }

// Allocate `nbytes` with the given (power-of-two) alignment. Pair with aligned_free.
// (Why 64 bytes by default? See Challenge 1.2 -- cache lines and SIMD.)
inline void* aligned_malloc(size_t nbytes, size_t alignment = 64) {
    LLM_ASSERT(is_power_of_two(alignment), "alignment must be a power of two");
#if defined(_WIN32)
    return _aligned_malloc(nbytes ? nbytes : 1, alignment);
#else
    // NOTE: std::aligned_alloc (C++17) requires size be a multiple of alignment,
    // so we round up. (It is also not provided by MSVC's runtime, which is why
    // Windows uses _aligned_malloc above.)
    const size_t rounded = (nbytes + alignment - 1) & ~(alignment - 1);
    return std::aligned_alloc(alignment, rounded ? rounded : alignment);
#endif
}

inline void aligned_free(void* p) noexcept {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}


// A bump / arena allocator for transient activations.
// Allocate once; carve out scratch with allocate(); reclaim it all with reset().
class Arena {
public:
    explicit Arena(size_t capacity_bytes, size_t alignment = 64)
        : alignment_(alignment), capacity_(capacity_bytes) {
        LLM_ASSERT(is_power_of_two(alignment), "alignment must be a power of two");
        base_ = static_cast<uint8_t*>(aligned_malloc(capacity_bytes, alignment));
        LLM_CHECK(base_ != nullptr, "Arena: backing allocation failed");
    }
    ~Arena() { aligned_free(base_); }
    Arena(const Arena&)            = delete;
    Arena& operator=(const Arena&) = delete;

    void* allocate(size_t nbytes) {
        const size_t aligned_off = (offset_ + alignment_ - 1) & ~(alignment_ - 1);
        LLM_CHECK(aligned_off + nbytes <= capacity_, "Arena: out of memory");
        void* p  = base_ + aligned_off;
        offset_  = aligned_off + nbytes;
        if (offset_ > high_water_) high_water_ = offset_;
        return p;
    }

    void   reset()      noexcept { offset_ = 0; }       // keep high_water_
    size_t used()       const noexcept { return offset_; }
    size_t high_water() const noexcept { return high_water_; }
    size_t capacity()   const noexcept { return capacity_; }

private:
    uint8_t* base_       = nullptr;
    size_t   alignment_  = 64;
    size_t   capacity_   = 0;
    size_t   offset_     = 0;
    size_t   high_water_ = 0;
};

}  // namespace llm
