#include "test_framework.hpp"
#include "llm/profile.hpp"
#include <vector>

// tests/test_profile.cpp
#include "test_framework.hpp"
#include "llm/profile.hpp"
#include <thread>
#include <chrono>
using namespace llm;

TEST(scoped_timer_measures_elapsed) {
    Profiler p;
    { PROFILE(p, "sleep"); std::this_thread::sleep_for(std::chrono::milliseconds(20)); }
    CHECK(p.totals["sleep"] >= 0.015);            // at least ~15 ms recorded
    CHECK(p.counts["sleep"] == 1);
}

TEST(decode_matmul_is_memory_bound) {
    const double ratio = 30.0;                    // a representative CPU FLOP:byte ridge point
    // decode: one token (M=1) times a 4096x4096 weight
    double di = matmul_intensity(/*M=*/1, /*N=*/4096, /*K=*/4096);
    CHECK(is_memory_bound(di * (4.0*(1*4096 + 4096*4096 + 1*4096)),
                          4.0*(1*4096 + 4096*4096 + 1*4096), ratio));
    CHECK(di < ratio);                            // intensity ~ 0.5, well below the ridge
}

TEST(prefill_matmul_is_compute_bound) {
    const double ratio = 30.0;
    double pi = matmul_intensity(/*M=*/512, /*N=*/4096, /*K=*/4096);   // 512-token prefill
    CHECK(pi > ratio);                            // many tokens per weight read -> compute-bound
}
