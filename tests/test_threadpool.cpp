#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "llm/threadpool.hpp"
#include <vector>
#include <atomic>

// tests/test_threadpool.cpp
// (reuses randt from the model tests)
#include "test_framework.hpp"
#include "llm/threadpool.hpp"
#include "llm/tensor.hpp"
#include <atomic>
#include <vector>
#include <chrono>
using namespace llm;

TEST(parallel_for_covers_range_exactly_once) {
    ThreadPool pool(4);
    const int n = 1000;
    std::vector<std::atomic<int>> hits(n);
    for (auto& h : hits) h.store(0);
    pool.parallel_for(n, [&](int b, int e) { for (int i = b; i < e; ++i) hits[i].fetch_add(1); });
    for (int i = 0; i < n; ++i) CHECK(hits[i].load() == 1);     // every index once, none twice
}

TEST(threaded_linear_matches_serial) {
    const int M = 3, in = 64, out = 128;
    Tensor x = randt({M, in}, 1), w = randt({out, in}, 2);
    Tensor ys = Tensor::empty({M, out}, DType::F32);
    Tensor yp = Tensor::empty({M, out}, DType::F32);
    set_num_threads(1); linear_mt(ys, x, w);                   // serial
    set_num_threads(4); linear_mt(yp, x, w);                   // threaded
    set_num_threads(1);
    for (int i = 0; i < M * out; ++i)
        CHECK_CLOSE(ys.data_ptr<float>()[i], yp.data_ptr<float>()[i], 1e-5);
}

TEST(threaded_matmul_is_faster) {                              // on a multicore CPU
    const int M = 256, in = 1024, out = 1024;
    Tensor x = randt({M, in}, 1), w = randt({out, in}, 2);
    Tensor y = Tensor::empty({M, out}, DType::F32);
    auto bench = [&](int threads) {
        set_num_threads(threads);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < 5; ++r) linear_mt(y, x, w);
        return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    };
    double serial = bench(1), threaded = bench(8);
    set_num_threads(1);
    CHECK(threaded < serial);                                  // parallelism pays off
}
