#pragma once
// Shared test helpers reused across unit test files.
// (In the book these are introduced per-challenge and "reused"; a real repo
//  collects them here so multiple test translation units can share them.)
#include "test_framework.hpp"
#include "llm/tensor.hpp"
#include "llm/model.hpp"
#include "llm/gemm.hpp"
#include <initializer_list>
#include <vector>
#include <cstdint>
#include <cmath>

namespace llm {

inline Tensor randt(std::initializer_list<int64_t> shape, uint32_t seed) {
    Tensor t = Tensor::empty(shape, DType::F32);
    uint32_t s = seed; float* p = t.data_ptr<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        s = s * 1664525u + 1013904223u;
        p[i] = (s >> 9) * (1.0f / 8388608.0f) - 0.5f;     // ~ [-0.5, 0.5)
    }
    return t;
}

inline LayerWeights attn_weights(const ModelConfig& c, uint32_t seed) {
    LayerWeights w;
    const int qd = c.n_heads * c.head_dim, kvd = c.n_kv_heads * c.head_dim;
    w.wq = randt({qd,  c.hidden_size}, seed + 1);
    w.wk = randt({kvd, c.hidden_size}, seed + 2);
    w.wv = randt({kvd, c.hidden_size}, seed + 3);
    w.wo = randt({c.hidden_size, qd},  seed + 4);
    return w;
}

inline LayerWeights full_weights(const ModelConfig& c, uint32_t seed) {
    LayerWeights w = attn_weights(c, seed);
    w.attn_norm = randt({c.hidden_size}, seed + 10);
    w.ffn_norm  = randt({c.hidden_size}, seed + 11);
    w.w_gate = randt({c.intermediate_size, c.hidden_size}, seed + 12);
    w.w_up   = randt({c.intermediate_size, c.hidden_size}, seed + 13);
    w.w_down = randt({c.hidden_size, c.intermediate_size}, seed + 14);
    return w;
}

inline ModelWeights tiny_model(const ModelConfig& c, uint32_t seed) {
    ModelWeights m;
    m.token_embedding = randt({c.vocab_size, c.hidden_size}, seed);
    m.final_norm      = randt({c.hidden_size}, seed + 100);
    m.lm_head         = randt({c.vocab_size, c.hidden_size}, seed + 200);
    for (int l = 0; l < c.n_layers; ++l)
        m.layers.push_back(full_weights(c, seed + 1000u * (l + 1)));
    return m;
}

inline ModelConfig tiny_cfg() {
    ModelConfig c; c.vocab_size = 32; c.hidden_size = 8; c.n_layers = 2;
    c.n_heads = 2; c.n_kv_heads = 1; c.head_dim = 4; c.intermediate_size = 16;
    return c;
}

inline void fill_pseudo(std::vector<float>& v, uint32_t seed) {
    uint32_t s = seed;
    for (float& x : v) { s = s * 1664525u + 1013904223u; x = (s >> 8) * (1.0f / 16777216.0f) - 0.5f; }
}

template <typename F>
inline void check_against_naive(F&& fn, int M, int N, int K) {
    std::vector<float> A(M * K), B(K * N), C_ref(M * N), C_got(M * N);
    fill_pseudo(A, 1); fill_pseudo(B, 2);
    matmul_naive(C_ref.data(), A.data(), B.data(), M, N, K);
    fn(C_got.data(), A.data(), B.data(), M, N, K);
    float worst = 0.0f;
    for (int i = 0; i < M * N; ++i) worst = std::max(worst, std::fabs(C_got[i] - C_ref[i]));
    CHECK(worst < 1e-2f);     // reordered summation: small fp differences expected
}

inline float rms_rel(const float* a, const float* b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) { double e = a[i]-b[i]; num += e*e; den += (double)b[i]*b[i]; }
    return (float)std::sqrt(num / (den + 1e-12));
}

inline void float_linear(float* Y, const float* X, const float* W, int S, int N, int K) {
    for (int i = 0; i < S; ++i) for (int o = 0; o < N; ++o) {
        float a = 0; for (int k = 0; k < K; ++k) a += X[i*K+k] * W[o*K+k];
        Y[i*N+o] = a;
    }
}

}  // namespace llm
