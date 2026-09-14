// include/llm/profile.hpp
#pragma once
#include <string>
#include <unordered_map>
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>

namespace llm {

struct Profiler {
    std::unordered_map<std::string, double> totals;   // section -> seconds
    std::unordered_map<std::string, int>    counts;
    void add(const std::string& name, double secs) { totals[name] += secs; counts[name] += 1; }
    void report() const {
        std::vector<std::pair<std::string,double>> rows(totals.begin(), totals.end());
        std::sort(rows.begin(), rows.end(), [](auto& a, auto& b){ return a.second > b.second; });
        for (auto& r : rows)
            std::printf("%-24s %8.3f ms  (x%d)\n", r.first.c_str(), r.second*1e3, counts.at(r.first));
    }
};

struct ScopedTimer {
    Profiler& p; std::string name;
    std::chrono::high_resolution_clock::time_point t0;
    ScopedTimer(Profiler& prof, std::string n)
        : p(prof), name(std::move(n)), t0(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() {
        auto t1 = std::chrono::high_resolution_clock::now();
        p.add(name, std::chrono::duration<double>(t1 - t0).count());
    }
};

#define LLM_CONCAT_(a, b) a##b
#define LLM_CONCAT(a, b)  LLM_CONCAT_(a, b)
#define PROFILE(prof, name) llm::ScopedTimer LLM_CONCAT(_timer_, __LINE__)((prof), (name))

// arithmetic intensity (FLOPs per byte) of an [M,K] x [K,N] matmul, fp32
inline double matmul_intensity(int M, int N, int K) {
    const double flops = 2.0 * M * N * K;
    const double bytes = 4.0 * ((double)M*K + (double)K*N + (double)M*N);
    return flops / bytes;
}

inline bool is_memory_bound(double flops, double bytes, double machine_flop_per_byte) {
    return (flops / bytes) < machine_flop_per_byte;
}

}  // namespace llm
