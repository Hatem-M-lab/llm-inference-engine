// tests/test_gemm.cpp
#include "test_framework.hpp"
#include "llm/gemm.hpp"
#include <vector>
using namespace llm;

TEST(matmul_naive_hand_computed) {
    // [2x3] * [3x2] = [2x2]
    float A[6] = {1, 2, 3,   4, 5, 6};            // 2x3
    float B[6] = {7, 8,   9, 10,   11, 12};       // 3x2
    float C[4];
    matmul_naive(C, A, B, /*M=*/2, /*N=*/2, /*K=*/3);
    CHECK_CLOSE(C[0],  58.0f, 1e-4);   // 1*7 + 2*9 + 3*11
    CHECK_CLOSE(C[1],  64.0f, 1e-4);   // 1*8 + 2*10 + 3*12
    CHECK_CLOSE(C[2], 139.0f, 1e-4);   // 4*7 + 5*9 + 6*11
    CHECK_CLOSE(C[3], 154.0f, 1e-4);   // 4*8 + 5*10 + 6*12
}

TEST(matmul_naive_identity) {
    const int n = 4;
    std::vector<float> A(n * n), I(n * n, 0.0f), C(n * n);
    for (int i = 0; i < n * n; ++i) A[i] = float(i + 1);
    for (int i = 0; i < n; ++i) I[i * n + i] = 1.0f;        // identity
    matmul_naive(C.data(), A.data(), I.data(), n, n, n);    // A * I == A
    for (int i = 0; i < n * n; ++i) CHECK_CLOSE(C[i], A[i], 1e-4);
}

// add to tests/test_gemm.cpp
#include "llm/gemm.hpp"
#include <vector>
#include <cstdint>

// A tiny deterministic RNG so tests are reproducible.
static void fill_pseudo(std::vector<float>& v, uint32_t seed) {
    uint32_t s = seed;
    for (float& x : v) { s = s * 1664525u + 1013904223u; x = (s >> 8) * (1.0f / 16777216.0f) - 0.5f; }
}

// Reusable: does `fn` match the naive oracle on random M-N-K inputs?
template <typename F>
static void check_against_naive(F&& fn, int M, int N, int K) {
    std::vector<float> A(M * K), B(K * N), C_ref(M * N), C_got(M * N);
    fill_pseudo(A, 1); fill_pseudo(B, 2);
    matmul_naive(C_ref.data(), A.data(), B.data(), M, N, K);
    fn(C_got.data(), A.data(), B.data(), M, N, K);
    float worst = 0.0f;
    for (int i = 0; i < M * N; ++i) worst = std::max(worst, std::fabs(C_got[i] - C_ref[i]));
    CHECK(worst < 1e-2f);     // reordered summation: small fp differences expected
}

TEST(matmul_ikj_matches_naive) {
    check_against_naive([](float* C, const float* A, const float* B, int M, int N, int K)
                        { matmul_ikj(C, A, B, M, N, K); }, 64, 48, 80);
    check_against_naive([](float* C, const float* A, const float* B, int M, int N, int K)
                        { matmul_ikj(C, A, B, M, N, K); }, 128, 128, 128);
}

// add to tests/test_gemm.cpp  (reuses check_against_naive from 3.2)

TEST(matmul_blocked_matches_naive) {
    auto fn = [](float* C, const float* A, const float* B, int M, int N, int K)
              { matmul_blocked(C, A, B, M, N, K); };
    check_against_naive(fn, 128, 128, 128);
    check_against_naive(fn, 100, 130, 70);     // not multiples of the block size
}

// add to tests/test_gemm.cpp  (reuses check_against_naive from 3.2)

TEST(matmul_parallel_matches_naive) {
    auto fn = [](float* C, const float* A, const float* B, int M, int N, int K)
              { matmul_parallel(C, A, B, M, N, K, /*threads=*/4); };
    check_against_naive(fn, 200, 256, 128);
    check_against_naive(fn, 257, 129, 64);    // ragged sizes + uneven row split
}

TEST(matmul_parallel_single_thread_equals_blocked) {
    const int M = 96, N = 80, K = 64;
    std::vector<float> A(M * K), B(K * N), C1(M * N), C2(M * N);
    fill_pseudo(A, 3); fill_pseudo(B, 4);
    matmul_blocked(C1.data(), A.data(), B.data(), M, N, K);
    matmul_parallel(C2.data(), A.data(), B.data(), M, N, K, 1);
    for (int i = 0; i < M * N; ++i) CHECK_CLOSE(C2[i], C1[i], 1e-4);  // same work, 1 thread
}

// add to tests/test_gemm.cpp  (reuses check_against_naive, fill_pseudo)

TEST(boss_matmul_simd_matches_naive) {
    auto fn = [](float* C, const float* A, const float* B, int M, int N, int K)
              { matmul_simd(C, A, B, M, N, K); };
    check_against_naive(fn, 64,  64,  64);
    check_against_naive(fn, 128, 256, 96);     // M, N multiples of 8; K arbitrary
    check_against_naive(fn, 512, 512, 512);
}

// The measured bar: run as a benchmark (timing asserts are machine-dependent,
// so we print the ratio and assert only a conservative lower bound).
TEST(boss_matmul_simd_is_faster) {
    const int M = 512, N = 512, K = 512;
    std::vector<float> A(M * K), B(K * N), C(M * N);
    fill_pseudo(A, 5); fill_pseudo(B, 6);
    const double gf_blocked = bench_gflops(
        [&]{ matmul_blocked(C.data(), A.data(), B.data(), M, N, K); }, M, N, K);
    const double gf_simd = bench_gflops(
        [&]{ matmul_simd(C.data(), A.data(), B.data(), M, N, K); }, M, N, K);
    std::printf("blocked: %.1f GFLOP/s   simd: %.1f GFLOP/s   speedup: %.2fx\n",
                gf_blocked, gf_simd, gf_simd / gf_blocked);
    CHECK(gf_simd >= 1.5 * gf_blocked);        // the boss bar
}
