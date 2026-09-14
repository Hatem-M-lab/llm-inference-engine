// include/llm/quant.hpp
#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

namespace llm {

constexpr int QK = 32;                 // block size

struct Q8Block {
    float  scale;                      // d  (GGUF stores this as f16)
    int8_t q[QK];
};

inline std::vector<Q8Block> quantize_q8_0(const float* w, int64_t n) {
    const int64_t nb = n / QK;
    std::vector<Q8Block> out(nb);
    for (int64_t b = 0; b < nb; ++b) {
        const float* blk = w + b * QK;
        float amax = 0.0f;
        for (int i = 0; i < QK; ++i) amax = std::max(amax, std::fabs(blk[i]));
        const float d  = amax / 127.0f;
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        out[b].scale = d;
        for (int i = 0; i < QK; ++i) {
            int q = static_cast<int>(std::lround(blk[i] * id));
            q = std::max(-127, std::min(127, q));
            out[b].q[i] = static_cast<int8_t>(q);
        }
    }
    return out;
}

inline void dequantize_q8_0(const std::vector<Q8Block>& blocks, float* out) {
    for (size_t b = 0; b < blocks.size(); ++b) {
        const float d = blocks[b].scale;
        for (int i = 0; i < QK; ++i) out[b * QK + i] = blocks[b].q[i] * d;
    }
}


// quant.hpp
struct Q4Block {
    float   scale;                     // d
    uint8_t q[QK / 2];                 // byte t: weight t (low nibble), weight t+16 (high nibble)
};

inline std::vector<Q4Block> quantize_q4_0(const float* w, int64_t n) {
    const int64_t nb = n / QK;
    std::vector<Q4Block> out(nb);
    for (int64_t b = 0; b < nb; ++b) {
        const float* blk = w + b * QK;
        float amax = 0.0f;
        for (int i = 0; i < QK; ++i) amax = std::max(amax, std::fabs(blk[i]));
        const float d  = amax / 8.0f;
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        out[b].scale = d;
        for (int i = 0; i < QK / 2; ++i) {
            int q0 = static_cast<int>(std::lround(blk[i]          * id)) + 8;
            int q1 = static_cast<int>(std::lround(blk[i + QK / 2] * id)) + 8;
            q0 = std::max(0, std::min(15, q0));
            q1 = std::max(0, std::min(15, q1));
            out[b].q[i] = static_cast<uint8_t>(q0 | (q1 << 4));
        }
    }
    return out;
}

inline void dequantize_q4_0(const std::vector<Q4Block>& blocks, float* out) {
    for (size_t b = 0; b < blocks.size(); ++b) {
        const float d = blocks[b].scale;
        for (int i = 0; i < QK / 2; ++i) {
            const uint8_t byte = blocks[b].q[i];
            out[b * QK + i]          = (static_cast<int>(byte & 0x0F) - 8) * d;
            out[b * QK + i + QK / 2] = (static_cast<int>(byte >> 4)   - 8) * d;
        }
    }
}


// quant.hpp  -- row-wise matrix quantizers and quantized linears
template <class QuantFn>
inline auto quantize_matrix(const float* w, int N, int K, QuantFn qfn) {
    using Block = typename decltype(qfn(w, (int64_t)K))::value_type;
    std::vector<Block> out; out.reserve((size_t)N * (K / QK));
    for (int o = 0; o < N; ++o) {
        auto row = qfn(w + (int64_t)o * K, K);
        out.insert(out.end(), row.begin(), row.end());
    }
    return out;
}
inline std::vector<Q4Block> quantize_matrix_q4_0(const float* w, int N, int K) {
    return quantize_matrix(w, N, K, [](const float* p, int64_t n){ return quantize_q4_0(p, n); });
}
inline std::vector<Q8Block> quantize_matrix_q8_0(const float* w, int N, int K) {
    return quantize_matrix(w, N, K, [](const float* p, int64_t n){ return quantize_q8_0(p, n); });
}

inline void linear_q8_0(float* Y, const float* X, const std::vector<Q8Block>& qw,
                        int S, int N, int K) {
    const int bpr = K / QK;
    for (int i = 0; i < S; ++i) {
        const float* xi = X + (int64_t)i * K;
        for (int o = 0; o < N; ++o) {
            const Q8Block* row = qw.data() + (int64_t)o * bpr;
            float acc = 0.0f;
            for (int b = 0; b < bpr; ++b) {
                const float d = row[b].scale; const float* xk = xi + b * QK;
                for (int t = 0; t < QK; ++t) acc += xk[t] * (row[b].q[t] * d);
            }
            Y[(int64_t)i * N + o] = acc;
        }
    }
}

inline void linear_q4_0(float* Y, const float* X, const std::vector<Q4Block>& qw,
                        int S, int N, int K) {
    const int bpr = K / QK;
    for (int i = 0; i < S; ++i) {
        const float* xi = X + (int64_t)i * K;
        for (int o = 0; o < N; ++o) {
            const Q4Block* row = qw.data() + (int64_t)o * bpr;
            float acc = 0.0f;
            for (int b = 0; b < bpr; ++b) {
                const float d = row[b].scale; const float* xk = xi + b * QK;
                for (int t = 0; t < QK / 2; ++t) {
                    const uint8_t byte = row[b].q[t];
                    acc += xk[t]          * ((static_cast<int>(byte & 0x0F) - 8) * d);
                    acc += xk[t + QK / 2] * ((static_cast<int>(byte >> 4)   - 8) * d);
                }
            }
            Y[(int64_t)i * N + o] = acc;
        }
    }
}

}  // namespace llm
