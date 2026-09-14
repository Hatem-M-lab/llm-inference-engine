// src/ops.cpp
#include "llm/ops.hpp"
#include <cmath>
#include <algorithm>

namespace llm {

static void require_same(const Tensor& a, const Tensor& b) {
    LLM_ASSERT(a.is_contiguous() && b.is_contiguous(), "elementwise: need contiguous");
    LLM_ASSERT(a.numel() == b.numel(), "elementwise: size mismatch");
}

void add(Tensor& out, const Tensor& a, const Tensor& b) {
    require_same(a, b); require_same(out, a);
    float* o = out.data_ptr<float>();
    const float* x = a.data_ptr<float>();
    const float* y = b.data_ptr<float>();
    const int64_t n = a.numel();
    for (int64_t i = 0; i < n; ++i) o[i] = x[i] + y[i];
}

void mul(Tensor& out, const Tensor& a, const Tensor& b) {
    require_same(a, b); require_same(out, a);
    float* o = out.data_ptr<float>();
    const float* x = a.data_ptr<float>();
    const float* y = b.data_ptr<float>();
    const int64_t n = a.numel();
    for (int64_t i = 0; i < n; ++i) o[i] = x[i] * y[i];
}

void scale(Tensor& out, const Tensor& in, float s) {
    require_same(out, in);
    float* o = out.data_ptr<float>();
    const float* x = in.data_ptr<float>();
    const int64_t n = in.numel();
    for (int64_t i = 0; i < n; ++i) o[i] = s * x[i];
}

void add_inplace(Tensor& a, const Tensor& b) {
    require_same(a, b);
    float* x = a.data_ptr<float>();
    const float* y = b.data_ptr<float>();
    const int64_t n = a.numel();
    for (int64_t i = 0; i < n; ++i) x[i] += y[i];
}


void swiglu(Tensor& out, const Tensor& gate, const Tensor& up) {
    require_same(gate, up); require_same(out, gate);
    float* o = out.data_ptr<float>();
    const float* g = gate.data_ptr<float>();
    const float* u = up.data_ptr<float>();
    const int64_t n = gate.numel();
    for (int64_t i = 0; i < n; ++i) o[i] = siluf(g[i]) * u[i];
}


void rmsnorm(Tensor& out, const Tensor& in, const Tensor& weight, float eps) {
    LLM_ASSERT(in.is_contiguous() && out.is_contiguous() && weight.is_contiguous(),
               "rmsnorm: tensors must be contiguous");
    LLM_ASSERT(in.numel() == out.numel(), "rmsnorm: in/out size mismatch");
    const int64_t D = in.shape().back();
    LLM_ASSERT(weight.numel() == D, "rmsnorm: weight length must equal last dim");
    const int64_t rows = in.numel() / D;

    const float* x = in.data_ptr<float>();
    const float* w = weight.data_ptr<float>();
    float*       o = out.data_ptr<float>();

    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * D;
        float*       orow = o + r * D;

        float ss = 0.0f;                                  // accumulate in float32
        for (int64_t j = 0; j < D; ++j) ss += xr[j] * xr[j];

        const float inv_rms = 1.0f / std::sqrt(ss / static_cast<float>(D) + eps);
        for (int64_t j = 0; j < D; ++j) orow[j] = xr[j] * inv_rms * w[j];
    }
}


void softmax_lastdim(Tensor& out, const Tensor& in) {
    LLM_ASSERT(in.is_contiguous() && out.is_contiguous(),
               "softmax: tensors must be contiguous");
    LLM_ASSERT(in.numel() == out.numel(), "softmax: size mismatch");
    const int64_t D = in.shape().back();
    const int64_t rows = in.numel() / D;

    const float* x = in.data_ptr<float>();
    float*       o = out.data_ptr<float>();

    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = x + r * D;
        float*       orow = o + r * D;

        float m = xr[0];                                   // 1) row max
        for (int64_t j = 1; j < D; ++j) m = std::max(m, xr[j]);

        float sum = 0.0f;                                  // 2) shifted exp + sum
        for (int64_t j = 0; j < D; ++j) {
            const float e = std::exp(xr[j] - m);           // exponent <= 0: no overflow
            orow[j] = e;
            sum += e;
        }

        const float inv = 1.0f / sum;                      // 3) normalize
        for (int64_t j = 0; j < D; ++j) orow[j] *= inv;
    }
}

}  // namespace llm
