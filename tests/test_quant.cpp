#include "test_framework.hpp"
#include "llm/quant.hpp"
#include "llm/gemm.hpp"

// tests/test_quant.cpp
#include "test_framework.hpp"
#include "llm/quant.hpp"
#include <vector>
#include <cmath>
using namespace llm;

TEST(q8_roundtrip_error_within_half_step) {
    const int64_t n = 64;                              // 2 blocks
    std::vector<float> w(n);
    uint32_t s = 1;
    for (auto& x : w) { s = s*1664525u+1013904223u; x = ((s>>9)*(1.0f/8388608.0f)-0.5f) * 3.0f; }
    auto blocks = quantize_q8_0(w.data(), n);
    std::vector<float> dq(n); dequantize_q8_0(blocks, dq.data());
    for (size_t b = 0; b < blocks.size(); ++b) {
        const float d = blocks[b].scale;
        for (int i = 0; i < QK; ++i)
            CHECK(std::fabs(w[b*QK + i] - dq[b*QK + i]) <= d * 0.5f + 1e-6f);
    }
}

TEST(q8_preserves_block_absmax) {
    std::vector<float> w(QK, 0.1f);
    w[5] = -2.0f; w[20] = 1.5f;                        // absmax = 2.0 at index 5
    auto blocks = quantize_q8_0(w.data(), QK);
    std::vector<float> dq(QK); dequantize_q8_0(blocks, dq.data());
    CHECK_CLOSE(blocks[0].scale, 2.0f / 127.0f, 1e-7);
    CHECK_CLOSE(dq[5], -2.0f, 1e-4);                   // largest magnitude is exact (q = -127)
}

// add to tests/test_quant.cpp
#include <cstdint>

static float rms_rel(const float* a, const float* b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) { double e = a[i]-b[i]; num += e*e; den += (double)b[i]*b[i]; }
    return (float)std::sqrt(num / (den + 1e-12));
}
static void float_linear(float* Y, const float* X, const float* W, int S, int N, int K) {
    for (int i = 0; i < S; ++i) for (int o = 0; o < N; ++o) {
        float a = 0; for (int k = 0; k < K; ++k) a += X[i*K+k] * W[o*K+k];
        Y[i*N+o] = a;
    }
}

TEST(q4_roundtrip_error_within_one_step) {
    const int64_t n = 64;
    std::vector<float> w(n);
    uint32_t s = 9; for (auto& x : w) { s = s*1664525u+1013904223u; x = ((s>>9)*(1.0f/8388608.0f)-0.5f)*2.0f; }
    auto blocks = quantize_q4_0(w.data(), n);
    std::vector<float> dq(n); dequantize_q4_0(blocks, dq.data());
    for (size_t b = 0; b < blocks.size(); ++b) {
        const float d = blocks[b].scale;
        for (int i = 0; i < QK; ++i)
            CHECK(std::fabs(w[b*QK + i] - dq[b*QK + i]) <= d + 1e-6f);  // 4-bit on [-8,7]: worst case one step
    }
}

TEST(quantized_linear_q8_close_to_float) {
    const int S = 2, N = 4, K = 64;                    // K multiple of QK
    std::vector<float> w(N*K), x(S*K);
    uint32_t s = 3;
    for (auto& z : w) { s = s*1664525u+1013904223u; z = (s>>9)*(1.0f/8388608.0f)-0.5f; }
    for (auto& z : x) { s = s*1664525u+1013904223u; z = ((s>>9)*(1.0f/8388608.0f)-0.5f)*0.8f; }
    std::vector<float> yf(S*N); float_linear(yf.data(), x.data(), w.data(), S, N, K);
    auto qw = quantize_matrix_q8_0(w.data(), N, K);
    std::vector<float> yq(S*N); linear_q8_0(yq.data(), x.data(), qw, S, N, K);
    CHECK(rms_rel(yq.data(), yf.data(), S*N) < 0.02f);   // Q8 near-lossless
}

TEST(quantized_linear_q4_close_to_float) {
    const int S = 2, N = 4, K = 64;
    std::vector<float> w(N*K), x(S*K);
    uint32_t s = 5;
    for (auto& z : w) { s = s*1664525u+1013904223u; z = (s>>9)*(1.0f/8388608.0f)-0.5f; }
    for (auto& z : x) { s = s*1664525u+1013904223u; z = ((s>>9)*(1.0f/8388608.0f)-0.5f)*0.8f; }
    std::vector<float> yf(S*N); float_linear(yf.data(), x.data(), w.data(), S, N, K);
    auto qw = quantize_matrix_q4_0(w.data(), N, K);
    std::vector<float> yq(S*N); linear_q4_0(yq.data(), x.data(), qw, S, N, K);
    CHECK(rms_rel(yq.data(), yf.data(), S*N) < 0.12f);   // Q4: small but real error
}

// add to tests/test_quant.cpp  (reuses rms_rel, float_linear)

TEST(boss_quantization_compression_ratio) {
    const int N = 64, K = 128;
    std::vector<float> w(N * K);
    uint32_t s = 1; for (auto& z : w) { s = s*1664525u+1013904223u; z = (s>>9)*(1.0f/8388608.0f)-0.5f; }
    auto q8 = quantize_matrix_q8_0(w.data(), N, K);
    auto q4 = quantize_matrix_q4_0(w.data(), N, K);
    const double fb  = (double)N * K * sizeof(float);
    const double q8b = (double)q8.size() * sizeof(Q8Block);
    const double q4b = (double)q4.size() * sizeof(Q4Block);
    CHECK(fb / q8b >= 3.5);                        // ~3.56x
    CHECK(fb / q4b >= 6.0);                        // ~6.4x
}

TEST(boss_quantized_projection_preserves_prediction) {
    const int vocab = 48, hidden = 64;             // hidden is a multiple of QK
    std::vector<float> w(vocab * hidden), x(hidden);
    uint32_t s = 11;
    for (auto& z : w) { s = s*1664525u+1013904223u; z = (s>>9)*(1.0f/8388608.0f)-0.5f; }
    for (auto& z : x) { s = s*1664525u+1013904223u; z = (s>>9)*(1.0f/8388608.0f)-0.5f; }
    const int winner = 7;                          // make row 7 the clear best match
    for (int k = 0; k < hidden; ++k) w[winner * hidden + k] = x[k] * 2.0f;

    std::vector<float> lf(vocab); float_linear(lf.data(), x.data(), w.data(), 1, vocab, hidden);
    auto q8 = quantize_matrix_q8_0(w.data(), vocab, hidden);
    auto q4 = quantize_matrix_q4_0(w.data(), vocab, hidden);
    std::vector<float> l8(vocab), l4(vocab);
    linear_q8_0(l8.data(), x.data(), q8, 1, vocab, hidden);
    linear_q4_0(l4.data(), x.data(), q4, 1, vocab, hidden);

    auto argmax = [&](const std::vector<float>& l) {
        int a = 0; for (int o = 1; o < vocab; ++o) if (l[o] > l[a]) a = o; return a;
    };
    CHECK(argmax(lf) == winner);                   // float ranks the winner first
    CHECK(argmax(l8) == winner);                   // Q8 preserves the prediction
    CHECK(argmax(l4) == winner);                   // Q4 preserves it too
    CHECK(rms_rel(l8.data(), lf.data(), vocab) < 0.01f);
    CHECK(rms_rel(l4.data(), lf.data(), vocab) < 0.12f);
}
