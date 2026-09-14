// src/gemm.cpp
#include "llm/gemm.hpp"
#include <cstring>
#include <algorithm>
#include <thread>
#include <vector>
#include <immintrin.h>

namespace llm {

void matmul_naive(float* C, const float* A, const float* B, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[(int64_t)i * K + k] * B[(int64_t)k * N + j];
            C[(int64_t)i * N + j] = acc;
        }
    }
}


// gemm.cpp

void matmul_ikj(float* C, const float* A, const float* B, int M, int N, int K) {
    std::memset(C, 0, sizeof(float) * (size_t)M * N);     // ikj accumulates
    for (int i = 0; i < M; ++i) {
        float* crow = C + (int64_t)i * N;
        for (int k = 0; k < K; ++k) {
            const float  a    = A[(int64_t)i * K + k];     // scalar, hoisted
            const float* brow = B + (int64_t)k * N;        // contiguous row of B
            for (int j = 0; j < N; ++j)
                crow[j] += a * brow[j];                    // contiguous stream
        }
    }
}


// gemm.cpp

void matmul_blocked(float* C, const float* A, const float* B, int M, int N, int K) {
    constexpr int MC = 64, NC = 64, KC = 64;            // tile sizes (tune these)
    std::memset(C, 0, sizeof(float) * (size_t)M * N);

    for (int ic = 0; ic < M; ic += MC) {
        const int iend = std::min(ic + MC, M);
        for (int kc = 0; kc < K; kc += KC) {
            const int kend = std::min(kc + KC, K);
            for (int jc = 0; jc < N; jc += NC) {
                const int jend = std::min(jc + NC, N);
                // ikj over the tile; the B slab [kc,kend)x[jc,jend) stays hot
                for (int i = ic; i < iend; ++i) {
                    float* crow = C + (int64_t)i * N;
                    for (int k = kc; k < kend; ++k) {
                        const float  a    = A[(int64_t)i * K + k];
                        const float* brow = B + (int64_t)k * N;
                        for (int j = jc; j < jend; ++j)
                            crow[j] += a * brow[j];
                    }
                }
            }
        }
    }
}


// gemm.cpp

// Blocked ikj over output rows [row0, row1). Reused by blocked + parallel.
static void gemm_rows(float* C, const float* A, const float* B,
                      int M, int N, int K, int row0, int row1) {
    constexpr int NC = 64, KC = 64;
    std::memset(C + (int64_t)row0 * N, 0,
                sizeof(float) * (size_t)(row1 - row0) * N);   // this band only
    for (int kc = 0; kc < K; kc += KC) {
        const int kend = std::min(kc + KC, K);
        for (int jc = 0; jc < N; jc += NC) {
            const int jend = std::min(jc + NC, N);
            for (int i = row0; i < row1; ++i) {
                float* crow = C + (int64_t)i * N;
                for (int k = kc; k < kend; ++k) {
                    const float  a    = A[(int64_t)i * K + k];
                    const float* brow = B + (int64_t)k * N;
                    for (int j = jc; j < jend; ++j) crow[j] += a * brow[j];
                }
            }
        }
    }
}

void matmul_parallel(float* C, const float* A, const float* B,
                     int M, int N, int K, int num_threads) {
    if (num_threads <= 0) {
        num_threads = (int)std::thread::hardware_concurrency();
        if (num_threads <= 0) num_threads = 1;
    }
    num_threads = std::min(num_threads, M);               // at most one band per row
    const int rows_per = (M + num_threads - 1) / num_threads;

    std::vector<std::thread> pool;
    pool.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        const int r0 = t * rows_per;
        const int r1 = std::min(r0 + rows_per, M);
        if (r0 >= r1) break;
        pool.emplace_back(gemm_rows, C, A, B, M, N, K, r0, r1);
    }
    for (std::thread& th : pool) th.join();
}


// gemm.cpp
#if defined(__AVX2__)
#endif

static constexpr int MR = 8, NR = 8;

// Pack an MR x kl strip of A (rows ic.., depth kc..) as Apack[k*MR + i].
static void pack_A(const float* A, int K, int ic, int kc, int kl, float* Apack) {
    for (int k = 0; k < kl; ++k)
        for (int i = 0; i < MR; ++i)
            Apack[k * MR + i] = A[(int64_t)(ic + i) * K + (kc + k)];
}

// Pack a kl x NR tile of B (depth kc.., cols jc..) as Bpack[k*NR + j].
static void pack_B(const float* B, int N, int kc, int jc, int kl, float* Bpack) {
    for (int k = 0; k < kl; ++k)
        for (int j = 0; j < NR; ++j)
            Bpack[k * NR + j] = B[(int64_t)(kc + k) * N + (jc + j)];
}

// C_tile[8][8] += sum_k Apack[k*8 + i] * Bpack[k*8 + j].   Accumulates into C.
static void micro_kernel(const float* Apack, const float* Bpack, int kl,
                         float* C, int N) {
#if defined(__AVX2__)
    __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps(), c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps(), c7 = _mm256_setzero_ps();
    for (int k = 0; k < kl; ++k) {
        const __m256 b = _mm256_loadu_ps(Bpack + k * NR);
        const float* a = Apack + k * MR;
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(a + 0), b, c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(a + 1), b, c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(a + 2), b, c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(a + 3), b, c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(a + 4), b, c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(a + 5), b, c5);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(a + 6), b, c6);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(a + 7), b, c7);
    }
    #define ADD_ROW(r, cc) \
        _mm256_storeu_ps(C + (r) * N, _mm256_add_ps(_mm256_loadu_ps(C + (r) * N), cc))
    ADD_ROW(0, c0); ADD_ROW(1, c1); ADD_ROW(2, c2); ADD_ROW(3, c3);
    ADD_ROW(4, c4); ADD_ROW(5, c5); ADD_ROW(6, c6); ADD_ROW(7, c7);
    #undef ADD_ROW
#else
    for (int i = 0; i < MR; ++i)
        for (int k = 0; k < kl; ++k) {
            const float a = Apack[k * MR + i];
            for (int j = 0; j < NR; ++j)
                C[(int64_t)i * N + j] += a * Bpack[k * NR + j];
        }
#endif
}

void matmul_simd(float* C, const float* A, const float* B, int M, int N, int K) {
    LLM_ASSERT(M % MR == 0 && N % NR == 0,
               "matmul_simd demo requires M, N multiples of 8 (see Going Further)");
    constexpr int KC = 256;                              // K-slab for packing/reuse
    std::memset(C, 0, sizeof(float) * (size_t)M * N);

    std::vector<float> Bpack((size_t)KC * N);            // whole B panel for a K-slab
    std::vector<float> Apack((size_t)MR * KC);           // one row-strip of A

    for (int kc = 0; kc < K; kc += KC) {
        const int kl = std::min(KC, K - kc);
        for (int j = 0; j < N; j += NR)                  // pack B panel once per slab
            pack_B(B, N, kc, j, kl, Bpack.data() + (size_t)j * kl);
        for (int i = 0; i < M; i += MR) {
            pack_A(A, K, i, kc, kl, Apack.data());       // pack A strip once per row-tile
            for (int j = 0; j < N; j += NR)
                micro_kernel(Apack.data(), Bpack.data() + (size_t)j * kl, kl,
                             C + (int64_t)i * N + j, N);
        }
    }
}

}  // namespace llm
