// include/llm/ops.hpp
#pragma once
#include <cstdint>
#include "llm/tensor.hpp"
#include <cmath>

namespace llm {

// ---- Elementwise (contiguous, float32) ----
void add(Tensor& out, const Tensor& a, const Tensor& b);        // out = a + b
void mul(Tensor& out, const Tensor& a, const Tensor& b);        // out = a * b
void scale(Tensor& out, const Tensor& in, float s);             // out = s * in
void add_inplace(Tensor& a, const Tensor& b);                   // a += b (residual)

// Map a scalar callable over a contiguous tensor: out[i] = f(in[i]).
template <typename F>
inline void apply_unary(Tensor& out, const Tensor& in, F f) {
    LLM_ASSERT(out.is_contiguous() && in.is_contiguous(),
               "apply_unary: tensors must be contiguous");
    LLM_ASSERT(out.numel() == in.numel(), "apply_unary: size mismatch");
    float* o = out.data_ptr<float>();
    const float* x = in.data_ptr<float>();
    const int64_t n = in.numel();
    for (int64_t i = 0; i < n; ++i) o[i] = f(x[i]);
}


// ---- Scalar activations (inline; the compiler folds these into loops) ----
inline float sigmoidf(float x) {
    // Branch-stable: each path only ever exponentiates a non-positive number.
    if (x >= 0.0f) { const float z = std::exp(-x); return 1.0f / (1.0f + z); }
    else           { const float z = std::exp( x); return z / (1.0f + z); }
}

inline float siluf(float x) { return x * sigmoidf(x); }

inline float geluf(float x) {                         // tanh approximation
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(kSqrt2OverPi * (x + 0.044715f * x3)));
}

inline float gelu_exactf(float x) {                   // exact, via erf
    constexpr float kInvSqrt2 = 0.7071067811865476f;
    return 0.5f * x * (1.0f + std::erf(x * kInvSqrt2));
}

// ---- Tensor activation wrappers ----
inline void silu(Tensor& out, const Tensor& in) {
    apply_unary(out, in, [](float v) { return siluf(v); });
}
inline void gelu(Tensor& out, const Tensor& in) {
    apply_unary(out, in, [](float v) { return geluf(v); });
}

// SwiGLU's elementwise step: out = SiLU(gate) * up  (one fused pass).
void swiglu(Tensor& out, const Tensor& gate, const Tensor& up);


// Row-wise RMS normalization over the last dimension, scaled by `weight` [D].
void rmsnorm(Tensor& out, const Tensor& in, const Tensor& weight, float eps = 1e-5f);


// Numerically-stable softmax over the last dimension, row by row.
void softmax_lastdim(Tensor& out, const Tensor& in);

}  // namespace llm
