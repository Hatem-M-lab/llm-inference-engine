// include/llm/gemm.hpp
#pragma once
#include <chrono>
#include <cstdint>
#include "llm/common.hpp"   // [fix] LLM_ASSERT used by matmul_simd

namespace llm {

// C[M,N] = A[M,K] * B[K,N], all row-major float32. C is overwritten.
void matmul_naive(float* C, const float* A, const float* B, int M, int N, int K);

// Throughput of a matmul callable, in GFLOP/s. fn() must perform one M-N-K multiply.
template <typename F>
double bench_gflops(F&& fn, int M, int N, int K, int iters = 10) {
    fn();  // warm up: page in memory, settle caches
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) fn();
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const double flop = 2.0 * M * N * K * iters;
    return flop / (secs * 1e9);
}


// gemm.hpp
void matmul_ikj(float* C, const float* A, const float* B, int M, int N, int K);


// gemm.hpp
void matmul_blocked(float* C, const float* A, const float* B, int M, int N, int K);


// gemm.hpp
void matmul_parallel(float* C, const float* A, const float* B,
                     int M, int N, int K, int num_threads = 0);


void matmul_simd(float* C, const float* A, const float* B, int M, int N, int K);

}  // namespace llm
